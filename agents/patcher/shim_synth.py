"""Phase-2 downshift — pipeline-authored leaf-shim synthesis (deliverable 2).

The Phase-2 double→float downshift needs the ``ql::`` **non-template** leaf overloads
(``kAbs``/``Imag``/``Real``/``Sign``/``Max``/``Min``/``Htheta`` on the reference scalar and
its complex container) to exist at the *target* precision.  The generic templates
(``kAbs<T>``/``kLog``/``kSqrt``/``kConj``/``kPow``/``iszero``/``Constants<T>`` and every
``kokkosUtils.h`` helper) already instantiate at the target automatically — only the
hand-written non-template overloads are missing (they are bound to the reference scalar by
signature, so the compiler will not re-instantiate them).

This module SYNTHESIZES those missing siblings, per-integral-TU, as a generated header
(``kokkosMaths_<precision>_shim.hpp``) layered on top of the untouched double reference
header (``kokkosMaths.h``).  It works because the target precision has **library-native**
support for every leaf body: a leaf is either a library pass-through (``Kokkos::abs(x)``),
a member/identity access (``x.real()`` / ``x``), or a scalar ternary/arith — all valid at
the target scalar/complex type.  (Precisions *without* library-native leaves — dd, ff —
cannot be served this way; see PHASE_2_SHIM_SYNTHESIS_DESIGN §8 and STOP #EEE, which is why
``ff`` requires a static wrapper header + an ff-native complex container, not this shim.)

**Structural, not name-mapped (feedback_no_placeholder_patterns / STOP #SS).**  The leaf
inventory is *extracted* from the reference header — the namespace-scoped, non-template
function definitions whose signature names the reference scalar token — never a baked-in
leaf list.  If the reference header gains, drops, or edits a non-template leaf, the shim
tracks it (and the sha256 inventory stamp invalidates the cached shim).  The generator
takes the reference/target *scalar tokens* as opaque parameters and rewrites one for the
other; it never branches on a precision name.  ``float``/``double``/``Kokkos``/``ql`` appear
only as the caller-supplied tokens, comments, and the generated code — not as control flow.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_SHA_TAG = "@shim-inventory-sha256:"


class ShimSynthError(RuntimeError):
    """A shim-synthesis precondition failed (unparseable reference header, etc.)."""


@dataclass(frozen=True)
class LeafRecord:
    """One non-template reference-precision leaf extracted from the reference header.

    ``name`` is the function name, ``signature`` the whitespace-normalized head
    (``<ret> <name>(<params>)``), and ``source`` the whitespace-normalized full definition
    (head + body).  The inventory sha is computed over ``(name, source)`` so a change to
    *either* a signature or a body invalidates a cached shim.
    """

    name: str
    signature: str
    source: str


# --------------------------------------------------------------------------- #
# reference-header parsing (structural)
# --------------------------------------------------------------------------- #

def _strip_comments(text: str) -> str:
    """Remove ``/*...*/`` and ``//...`` comments (so classification never trips on a
    comment that happens to contain ``template``/``struct``/the scalar token)."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def _namespace_body(text: str, namespace: str) -> str:
    """The text inside ``namespace <namespace> { ... }`` (brace-matched).

    Isolates the free-function scope so struct members (``Constants<T>``) — which live one
    brace deeper — and anything outside the namespace are excluded structurally, not by
    name.  ``namespace`` is a caller token (``ql`` for qcdloop), never assumed.
    """
    m = re.search(rf"\bnamespace\s+{re.escape(namespace)}\b", text)
    if not m:
        raise ShimSynthError(f"reference header has no `namespace {namespace}`")
    brace = text.find("{", m.end())
    if brace < 0:
        raise ShimSynthError(f"`namespace {namespace}` has no opening brace")
    depth = 0
    for i in range(brace, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1:i]
    raise ShimSynthError(f"`namespace {namespace}` is unbalanced")


def _split_top_level(body: str) -> list[str]:
    """Split a namespace body into its top-level units (one free function / struct / decl).

    A unit is a run of text that either closes a ``{...}`` block back to depth 0 (a
    definition with a body) or ends in a ``;`` at depth 0 (a plain declaration such as the
    ``using complex = ...;`` alias).  Depth is tracked by brace matching, so a struct's
    members never leak out as separate units.
    """
    units: list[str] = []
    buf: list[str] = []
    depth = 0
    for c in body:
        if c == "{":
            depth += 1
            buf.append(c)
        elif c == "}":
            depth -= 1
            buf.append(c)
            if depth == 0:
                unit = "".join(buf).strip()
                if unit:
                    units.append(unit)
                buf = []
        elif c == ";" and depth == 0:
            unit = "".join(buf).strip()
            if unit:
                units.append(unit + ";")
            buf = []
        else:
            buf.append(c)
    tail = "".join(buf).strip()
    if tail:
        units.append(tail)
    return units


_HEAD_RE = re.compile(
    r"^(?P<ret>[\w:<>\s\*&]+?)\s+(?P<name>\w+)\s*\((?P<params>.*)\)\s*$", re.S)


def extract_inventory(reference_header_text: str, *, reference_scalar: str,
                      namespace: str = "ql") -> list[LeafRecord]:
    """Extract the non-template reference-precision leaf overloads from the header.

    A unit qualifies iff it is a namespace-scoped **function definition** (has a body) that
    is **not** a template (its head has no ``template`` keyword), **not** a struct/class,
    and whose head names ``reference_scalar`` as a whole word (so it is an overload bound to
    the reference precision — e.g. ``double`` or ``Kokkos::complex<double>``).  Returned in
    **source order** (order of appearance in the reference header) so a sibling that calls
    another leaf (``Sign``/``Max``/``Min`` call ``kAbs``) sees the callee's sibling declared
    first, exactly as the reference header orders them.  The inventory sha is order-stable
    regardless (:func:`inventory_sha256` sorts internally).
    """
    body = _strip_comments(_namespace_body(reference_header_text, namespace))
    scalar_re = re.compile(rf"\b{re.escape(reference_scalar)}\b")
    leaves: list[LeafRecord] = []
    for unit in _split_top_level(body):
        if "{" not in unit or "KOKKOS_INLINE_FUNCTION" not in unit:
            continue                                   # not a function definition
        head = unit[:unit.index("{")]
        if "template" in head or "struct" in head or "class" in head:
            continue                                   # template / type def — auto-instantiates
        sig_head = head.replace("KOKKOS_INLINE_FUNCTION", " ").strip()
        if not scalar_re.search(sig_head):
            continue                                   # not a reference-precision overload
        m = _HEAD_RE.match(sig_head)
        if not m:
            continue                                   # not a plain function signature
        name = m.group("name")
        signature = re.sub(r"\s+", " ", sig_head).strip()
        source = re.sub(r"\s+", " ", unit).strip()
        leaves.append(LeafRecord(name=name, signature=signature, source=source))
    return leaves   # source order (callee-before-caller preserved from the reference header)


def inventory_sha256(inventory: list[LeafRecord]) -> str:
    """Stable sha256 over the extracted inventory ``(name, source)`` pairs.

    Keys the shim's regeneration: recomputed from the current reference header on each emit;
    if it differs from the shim's embedded stamp (or the shim is absent) the shim is
    regenerated, else the cached shim is reused.  Captures both signature and body changes.
    Order-independent (sorted internally) so a pure reordering of the reference leaves does
    not needlessly invalidate a byte-equivalent shim.
    """
    h = hashlib.sha256()
    for leaf in sorted(inventory, key=lambda l: (l.name, l.signature)):
        h.update(leaf.name.encode())
        h.update(b"\0")
        h.update(leaf.source.encode())
        h.update(b"\0")
    return h.hexdigest()


def read_embedded_sha(shim_text: str) -> str | None:
    """The inventory sha stamped in a generated shim's first-line comment, or ``None``."""
    m = re.search(rf"{re.escape(_SHA_TAG)}\s*([0-9a-f]+)", shim_text)
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# sibling rendering (precision-parameterized token rewrite)
# --------------------------------------------------------------------------- #

def _render_sibling(leaf: LeafRecord, *, reference_scalar: str,
                    target_scalar: str) -> str:
    """Render ``leaf``'s target-precision sibling by rewriting the reference scalar token.

    A whole-word substitution ``reference_scalar -> target_scalar`` over the whole
    definition rewrites the signature **and** the body in one pass: it turns
    ``Kokkos::complex<double>`` into ``Kokkos::complex<float>`` (the token sits inside the
    angle brackets), ``double(0)`` into ``float(0)``, and leaves library calls
    (``Kokkos::abs(x)``), member accesses (``x.real()``) and unrelated types (``int`` for
    ``Sign``) untouched.  The result is a strictly target-typed overload — which is what
    keeps it ODR-safe alongside the double reference (STOP #DDD): it does not shadow the
    double overload, it sits beside it.
    """
    sub = re.compile(rf"\b{re.escape(reference_scalar)}\b")
    rewritten = sub.sub(target_scalar, leaf.source)
    return "    " + rewritten


def render_shim(reference_header_text: str, *, reference_scalar: str,
                target_scalar: str, reference_name: str = "kokkosMaths.h",
                precision_label: str = "", namespace: str = "ql") -> str:
    """Render the full ``kokkosMaths_<precision>_shim.hpp`` for one target precision.

    Extracts the reference leaf inventory (structural), stamps its sha256, and emits one
    target-precision sibling per leaf inside ``namespace <namespace>``.  ``reference_scalar``
    / ``target_scalar`` are opaque tokens (``"double"`` / ``"float"`` for Phase-2); the
    generator never branches on their value — a future library-native precision selects the
    same path with its own tokens.
    """
    inventory = extract_inventory(reference_header_text,
                                  reference_scalar=reference_scalar, namespace=namespace)
    if not inventory:
        raise ShimSynthError(
            f"no non-template `{reference_scalar}` leaf overloads found in the reference "
            f"header — nothing to synthesize (is `namespace {namespace}` / the scalar "
            f"token correct?)")
    sha = inventory_sha256(inventory)
    label = precision_label or target_scalar
    siblings = "\n\n".join(
        _render_sibling(leaf, reference_scalar=reference_scalar,
                        target_scalar=target_scalar)
        for leaf in inventory)
    return (
        f"// {_SHA_TAG} {sha}  reference={reference_name} precision={label}\n"
        f"//\n"
        f"// Pipeline-synthesized {label} leaf siblings for the non-template {namespace}:: "
        f"overloads\n"
        f"// in {reference_name}, binding to library-native {label} instantiations "
        f"(Phase-2 downshift).\n"
        f"// Generated into the CLONED per-integral TU only — the snapshot reference header\n"
        f"// is pristine (STOP #Z).  Do NOT hand-edit: regenerate via "
        f"agents.patcher.shim_synth\n"
        f"// (the sha above invalidates this shim when the reference leaf set changes).\n"
        f"\n"
        f"#pragma once\n"
        f"\n"
        f"namespace {namespace} {{\n"
        f"\n"
        f"{siblings}\n"
        f"\n"
        f"}}  // namespace {namespace}\n")
