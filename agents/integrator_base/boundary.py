"""Regional boundary-patch synthesis (deterministic) — the regional analogue of
:mod:`agents.integrator_base.c8`.

The regional ff/dd integrators split their work in two:

* the **shim** (LLM-generated) provides the extended-precision *types, operators and
  named constants* the region needs, referencing the vendored ``Kokkos::Experimental::FloatFloat``
  / ``Kokkos::Experimental::DoubleDouble`` headers; and
* the **boundary patch** (this module, *deterministic*) rewrites the region's own
  source so the promoted arithmetic is actually wired in: it promotes the region's
  reads to the extended scalar on entry, keeps the region's computed locals in the
  extended scalar, and demotes them back to the caller's precision on exit.

Keeping the boundary patch deterministic (design §P4 "LLM should NOT generate the
boundary patch directly") means it replays cleanly, shrinks the LLM error surface,
and gives the Patcher clean retry semantics — a re-roll re-generates only the shim;
the patch machinery is fixed.

Transform (applied to the inclusive 1-based line range ``[line_start, line_end]``):

1. ``#include "<shim>"`` inserted once after the file's ``#pragma once``.
2. **Reads → extended (entry).** For each read ``r`` (from the characterizer's
   ``region_local_vars``): ``<scalar> r__ff = <scalar>(r);`` before the region, and
   every whole-word ``r`` inside the region is renamed to ``r__ff``.  (Rule R1.)
3. **Region internals stay extended (dataflow).** A statement-level local
   declaration ``<T> w = <rhs>`` whose ``<rhs>`` consumes a value already promoted
   to the extended scalar (a promoted read, or an earlier promoted local — the
   promotion chains through the region) is retyped ``<scalar> w__ext = <rhs>``
   (Rule R2) and every whole-word ``w`` inside the region is renamed to ``w__ext``.
   A write ``w`` declared *before* the region and re-assigned inside it (reported in
   ``writes`` by Fix C) is seeded with ``<scalar> w__ext = <scalar>(w);`` at entry
   and likewise renamed.
4. **Writes → caller (exit).** After the region each write is demoted back under
   its original name via two-limb reconstruction: a region-local decl to its
   *original declared type* ``<T> w = static_cast<T>(w__ext.hi) + static_cast<T>
   (w__ext.lo);``; a pre-declared write assigned back ``w = static_cast<caller>
   (w__ext.hi) + …``.  Two-limb reconstruction is the extended types' own
   conversion-out idiom (neither ``FloatFloat`` nor ``DoubleDouble`` defines ``operator
   double``).

**Why dataflow, not a fixed caller type.** Real HPC kernels declare their locals
through template aliases (qcdloop's ``TMass`` / ``TScale``, both ``double`` at the
vanilla instantiation), not the literal spelling the Patcher passes as
``caller_type``.  Detecting which locals to promote by *what their RHS consumes*
— rather than by matching a fixed type token — makes the transform work on
templated source, and demoting to the local's own declared type keeps downstream
code (which reads that local under its original type) well-formed.  Fix-C's
``writes`` still feeds the pre-declared (Case B) writes; ``caller_type`` is the
demotion target only for those.

**Assumptions (kernel subset, documented).** A single contiguous region; a promoted
local is a statement-level ``<type> <name> = <init>`` (one name per statement, not
``double a, b;``, and not a split ``double r; r = …;``).  Discrete-typed locals
(``int`` / ``bool`` / ``unsigned`` / …) are never retyped (Rule 1 — integers stay
in integer land).  The rewrite is comment/string/char-literal-aware and whole-word.
"""

from __future__ import annotations

import difflib

# Identifier alphabet (same as region_scan / the P3a lexer).
_IDENT_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")

_READ_SUFFIX = "__ff"
_WRITE_SUFFIX = "__ext"

# Discrete/integer type tokens never retyped to the extended scalar (Rule 1 —
# keep integer computations in integer land).
_INT_TYPES = frozenset({
    "int", "bool", "char", "short", "long", "unsigned", "signed", "size_t",
    "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t", "ptrdiff_t",
})

# Tokens that can lead a `<tok> <ident> =` triple without it being a declaration.
_NON_TYPE_LEADERS = frozenset({
    "return", "if", "else", "for", "while", "do", "switch", "case", "sizeof",
    "new", "delete", "throw", "co_return", "co_await", "co_yield",
})

# Storage-class / cv qualifiers that may lead a declaration before its type token
# (skipped by the bare/multi-declarator scanner so ``const TMass Y, S;`` is found).
_DECL_QUALIFIERS = frozenset({
    "const", "constexpr", "static", "volatile", "mutable", "register",
    "inline", "thread_local", "extern",
})


class _Tok:
    """A code token with its char span in the scanned text."""

    __slots__ = ("text", "start", "end")

    def __init__(self, text: str, start: int, end: int):
        self.text = text
        self.start = start
        self.end = end


def _tokenize(text: str) -> list[_Tok]:
    """Identifier + single-char punctuation tokens, with spans, skipping comments
    and string / char literals (same lexical state machine as
    :func:`agents.shared.region_scan._tokenize`, but span-carrying)."""
    toks: list[_Tok] = []
    i, n = 0, len(text)
    state = "code"

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line_comment"; i += 2; continue
            if ch == "/" and nxt == "*":
                state = "block_comment"; i += 2; continue
            if ch == '"':
                state = "string"; i += 1; continue
            if ch == "'":
                state = "char"; i += 1; continue
            if ch in _IDENT_CHARS:
                j = i
                while j < n and text[j] in _IDENT_CHARS:
                    j += 1
                toks.append(_Tok(text[i:j], i, j))
                i = j
                continue
            if not ch.isspace():
                toks.append(_Tok(ch, i, i + 1))
            i += 1
            continue

        # inside a comment / literal: consume until it closes.
        if state == "line_comment":
            if ch == "\n":
                state = "code"
            i += 1
        elif state == "block_comment":
            if ch == "*" and nxt == "/":
                state = "code"; i += 2
            else:
                i += 1
        elif state == "string":
            if ch == "\\" and nxt:
                i += 2
            elif ch == '"':
                state = "code"; i += 1
            else:
                i += 1
        elif state == "char":
            if ch == "\\" and nxt:
                i += 2
            elif ch == "'":
                state = "code"; i += 1
            else:
                i += 1

    return toks


def _is_ident(text: str) -> bool:
    return bool(text) and text[0] in _IDENT_CHARS and not text[0].isdigit()


class _Decl:
    """A statement-level local declaration ``<type> <name> = <init> ;``."""

    __slots__ = ("name", "type_idx", "type_text", "rhs_idents")

    def __init__(self, name: str, type_idx: int, type_text: str, rhs_idents: set[str]):
        self.name = name          # declared identifier
        self.type_idx = type_idx  # token index of the type to retype
        self.type_text = type_text
        self.rhs_idents = rhs_idents  # identifiers in the initializer expression


def _scan_decls(toks: list[_Tok]) -> list[_Decl]:
    """Find statement-level ``<type> <name> = <init>`` declarations (source order).

    A declaration is two consecutive identifiers followed by ``=`` at paren depth 0
    (so function parameters and loop/condition bindings inside ``(...)`` are
    excluded), where the leading identifier is a type (not a control keyword) and
    the name is not immediately a call ``name(``.  Integer-typed locals are skipped
    by the caller (Rule 1).  The RHS identifier set is collected up to the next
    top-level ``;`` for the dataflow check.
    """
    decls: list[_Decl] = []
    depth = 0
    n = len(toks)
    for i in range(n):
        t = toks[i]
        if t.text == "(":
            depth += 1
            continue
        if t.text == ")":
            depth = max(0, depth - 1)
            continue
        if depth != 0 or not _is_ident(t.text) or t.text in _NON_TYPE_LEADERS:
            continue
        if i + 2 >= n or not _is_ident(toks[i + 1].text) or toks[i + 2].text != "=":
            continue
        # collect RHS identifiers until the next top-level ';'
        rhs: set[str] = set()
        d = 0
        j = i + 3
        while j < n:
            tj = toks[j].text
            if tj in "([{":
                d += 1
            elif tj in ")]}":
                d -= 1
            elif tj == ";" and d == 0:
                break
            elif _is_ident(tj):
                rhs.add(tj)
            j += 1
        decls.append(_Decl(toks[i + 1].text, i, t.text, rhs))
    return decls


class _BareDecl:
    """A statement-level declaration recovered by :func:`_scan_bare_decls`.

    Unlike :class:`_Decl` (which requires the ``<type> <name> =`` init triple), this
    also captures bare (``TMass Y;``) and bare multi-declarator (``TMass Y, S, A;``)
    forms.  ``names`` are all declarators sharing the leading type token; ``type_idx``
    is the token index of the leading (core) type token, ``type_text`` its spelling.
    """

    __slots__ = ("names", "type_idx", "type_text")

    def __init__(self, names: list[str], type_idx: int, type_text: str):
        self.names = names
        self.type_idx = type_idx
        self.type_text = type_text


def _scan_bare_decls(toks: list[_Tok]) -> list[_BareDecl]:
    """Find statement-level declarations, including bare / multi-declarator forms.

    The existing :func:`_scan_decls` only recognizes ``<type> <name> =`` (an
    initialized single declarator) and so misses ``TMass Y, S, A;`` (bare,
    multi-declarator) — the exact shape a chain *carrier* takes at its declaration
    (design §7/§8).  This scanner splits the region into statements at **paren depth
    0** (so a ``;`` inside a ``(...)`` — a for-header, a call — does not split, and a
    ``;`` inside a function body still does; same depth discipline as
    :func:`_scan_decls`).  Braces ``{`` / ``}`` are themselves statement separators.
    For each statement of the form ``<quals>* <type> <name> [, <name>]* ;`` (each
    declarator optionally ``= init`` or ``[extent]``) it records every declared name
    under the shared leading type.  Constructor-init / call shapes (``name(...)``)
    and member accesses (``a.b``, ``a::b``) are rejected, mirroring
    :func:`agents.patcher.chain_promote._local_decls`.
    """
    out: list[_BareDecl] = []
    n = len(toks)
    stmt_start = 0
    paren = 0
    for i in range(n + 1):
        tx = toks[i].text if i < n else ";"
        if tx == "(":
            paren += 1
            continue
        if tx == ")":
            paren = max(0, paren - 1)
            continue
        if paren != 0:
            continue
        if tx == ";" or tx in "{}":
            rec = _parse_bare_decl_stmt(toks, stmt_start, i)
            if rec is not None:
                out.append(rec)
            stmt_start = i + 1
    return out


def _parse_bare_decl_stmt(toks: list[_Tok], lo: int, hi: int) -> _BareDecl | None:
    """Parse ``toks[lo:hi]`` (one ``;``-terminated statement) as a declaration.

    Returns a :class:`_BareDecl` when the statement is ``<quals>* <type> <name>
    [, <name>]*`` (each declarator optionally ``= init`` or ``[extent]``), else
    ``None``.  Source-only, no type resolution — a leading control keyword or a
    ``name(`` shape bails.
    """
    k = lo
    # skip leading storage-class / cv qualifiers
    while k < hi and toks[k].text in _DECL_QUALIFIERS:
        k += 1
    # a declaration must be preceded by a statement boundary, not a member access
    # (``a.b`` / ``a::b`` / ``a->b``; the tokenizer emits ``::``/``->`` as two
    # single-char tokens, so a trailing ``:`` / ``>`` also signals member access).
    if k > 0 and toks[k - 1].text in (".", ":", ">"):
        return None
    if k >= hi or not _is_ident(toks[k].text) or toks[k].text in _NON_TYPE_LEADERS:
        return None
    type_idx, type_text = k, toks[k].text
    j = k + 1
    # skip a namespace-qualified type ``a::b::c`` (the boundary tokenizer emits
    # ``::`` as two ``:`` punctuation tokens).  The retype target is the last
    # segment's ident.
    while j + 2 < hi and toks[j].text == ":" and toks[j + 1].text == ":" \
            and _is_ident(toks[j + 2].text):
        type_idx, type_text = j + 2, toks[j + 2].text
        j += 3
    # skip a template argument list ``<...>``
    if j < hi and toks[j].text == "<":
        d = 0
        while j < hi:
            if toks[j].text == "<":
                d += 1
            elif toks[j].text == ">":
                d -= 1
            j += 1
            if d <= 0:
                break
    while j < hi and toks[j].text in ("&", "*"):
        j += 1
    while j < hi and toks[j].text in _DECL_QUALIFIERS:
        j += 1
    # declarators: name [= init] [ [extent] ] (, name …)*
    names: list[str] = []
    while j < hi:
        if not _is_ident(toks[j].text):
            return None
        nm = toks[j].text
        nxt = toks[j + 1].text if j + 1 < hi else ";"
        if nxt == "(":
            return None                       # call / constructor-init / function decl
        if nxt not in (",", ";", "=", "["):
            return None                       # not a plain declarator (``a . b`` etc.)
        names.append(nm)
        j += 1
        if nxt in ("=", "["):                 # skip initializer / array extent
            d = 0
            while j < hi:
                tj = toks[j].text
                if tj in "([{":
                    d += 1
                elif tj in ")]}":
                    d -= 1
                elif tj == "<":               # template-id in init (``= ql::f<T,U>(…)``)
                    d += 1
                elif tj == ">":
                    d -= 1
                elif tj == "," and d == 0:
                    break
                j += 1
        if j < hi and toks[j].text == ",":
            j += 1
            continue
        break
    if not names:
        return None
    return _BareDecl(names, type_idx, type_text)


class _Promotion:
    """The region's promotion partition (reads / pre-declared writes / promoted
    local decls / full promoted-name set) — the shared dataflow both the boundary
    patch and the Gap-A qualified-call lint key off.

    ``carrier_writes`` (design §8) are the chain-carrier names actually written in
    the region — the boundary layer counts each as a *landing* so the no-op guard
    does not fire when the only region effect is a carrier write.
    """

    __slots__ = ("toks", "ident_texts", "pure_reads", "caseB", "decl_writes",
                 "names", "carrier_writes")

    def __init__(self, toks, ident_texts, pure_reads, caseB, decl_writes, names,
                 carrier_writes=frozenset()):
        self.toks = toks
        self.ident_texts = ident_texts
        self.pure_reads = pure_reads
        self.caseB = caseB
        self.decl_writes = decl_writes
        self.names = names
        self.carrier_writes = carrier_writes


def _compute_promotion(region_text: str, reads: list[str], writes: list[str],
                       closure_names: frozenset[str] = frozenset(),
                       element_bases: frozenset[str] = frozenset()) -> _Promotion:
    """Partition a region's identifiers into the extended-scalar promotion sets.

    Reads promote unconditionally (Rule R1); a region-local decl promotes iff its
    initializer consumes an already-promoted value, chaining to a fixpoint (Rule
    R2); integer/bool locals never promote (Rule 1).  Factored out of
    :func:`synthesize_boundary_patch` so the lint can reuse the exact same
    dataflow the patch will apply.

    ``closure_names`` (design §8) are chain-carrier variables whose *declaration*
    is widened to the extended type elsewhere (the emission layer, §7).  A carrier
    is neither a read-only input nor a truncating sink at this region's boundary, so
    it is **excluded** from ``pure_reads`` / ``caseB`` / ``decl_writes`` and **seeded
    into the ``promoted`` dataflow set** (its extended value flows through region
    locals that consume it).  It is *not* renamed and *not* given a ``r__``/``w__``
    boundary alias — the widened decl already carries the extended type end-to-end.
    Carrier names actually written in this region are recorded in ``carrier_writes``
    so the no-op guard treats a carrier write as a landing.

    ``element_bases`` (region-core element promotion, 2026-07-28) are fixed-size
    complex-aggregate names whose *element occurrences* ``base[k]`` are promoted at
    the read site (never the array decl — the d1 whole-array failure mode is avoided
    by construction).  A base is seeded into the dataflow **only** so a region-local
    decl-init consuming an element (``TOutput xs = cxs[0];``) chains to promotion via
    Rule R2; the base name itself is deliberately kept OUT of ``pure_reads`` /
    ``promoted`` / ``names`` (it is never renamed or aliased, and must not trip the
    Gap-A qualified-call lint that reads :func:`compute_promoted_names`).
    """
    toks = _tokenize(region_text)
    ident_texts = {t.text for t in toks if _is_ident(t.text)}

    all_decls = _scan_decls(toks)
    decl_names = {d.name for d in all_decls}

    # Case B: pre-declared writes (Fix C) re-assigned in the region.  Carrier names
    # are excluded (§8): their decl is widened elsewhere, so they are neither seeded
    # nor demoted at this boundary.
    caseB = [w for w in _dedupe(writes)
             if w in ident_texts and w not in decl_names and w not in closure_names]
    # Reads: promoted unconditionally on entry (a name that is also a region-local
    # decl is fundamentally a write — exclude it).  Carriers are excluded (§8).
    pure_reads = [r for r in _dedupe(reads)
                  if r in ident_texts and r not in decl_names and r not in caseB
                  and r not in closure_names]

    # Carriers actually referenced-as-written in this region (a landing for §8's
    # no-op guard).  A carrier's decl is outside the region, so it appears here as a
    # re-assignment reported in ``writes``.
    carrier_writes = frozenset(w for w in _dedupe(writes)
                               if w in closure_names and w in ident_texts)

    # Seed carriers into the promoted set so a region-local decl whose RHS consumes a
    # carrier chains to promotion (Rule R2), even though the carrier itself is never
    # renamed/aliased.
    promoted: set[str] = set(pure_reads) | set(caseB) | set(closure_names)
    # ``seed`` drives the dataflow reachability check; ``element_bases`` participate in
    # it (a decl-init consuming ``base[k]`` chains) but are kept OUT of ``promoted`` so
    # the base name is never renamed/aliased/lint-flagged (region-core element promo).
    seed: set[str] = set(promoted) | set(element_bases)
    decl_writes: list[_Decl] = []
    changed = True
    while changed:
        changed = False
        for d in all_decls:
            if d.name in seed or d.type_text in _INT_TYPES \
                    or d.name in closure_names:
                continue
            if d.rhs_idents & seed:
                seed.add(d.name)
                promoted.add(d.name)
                decl_writes.append(d)
                changed = True

    return _Promotion(toks, ident_texts, pure_reads, caseB, decl_writes, promoted,
                      carrier_writes)


def compute_promoted_names(region_text: str, reads: list[str], writes: list[str]) -> set[str]:
    """Public: the set of region identifiers promoted to the extended scalar.

    Used by the Gap-A qualified-call bridge lint to decide whether a namespace-
    qualified math call is invoked on a promoted (extended-typed) argument.
    """
    return set(_compute_promotion(region_text, reads, writes).names)


def _apply_spans(text: str, edits: list[tuple[int, int, str]]) -> str:
    """Apply (start, end, replacement) span edits to ``text`` (non-overlapping)."""
    out = []
    pos = 0
    for start, end, repl in sorted(edits):
        out.append(text[pos:start])
        out.append(repl)
        pos = end
    out.append(text[pos:])
    return "".join(out)


# The limb members of the two-limb extended scalars (``DoubleDouble`` / ``FloatFloat``),
# most-significant first.  Named here so the two-limb spelling has one definition.
TWO_LIMB_MEMBERS: tuple[str, ...] = ("hi", "lo")


def narrow_extended_scalar(expr: str, caller_type: str,
                           limbs: tuple[str, ...] = TWO_LIMB_MEMBERS) -> str:
    """Reconstruct a caller-precision scalar from a multi-limb extended scalar ``expr``.

    THE single source of truth for the extended → caller reconstruction idiom (STOP #TT:
    no one-off boundary narrowing).  The extended scalar types define no ``operator
    double``, so the boundary rebuilds the value by summing its limbs:

        ("hi", "lo")               -> static_cast<T>(e.hi) + static_cast<T>(e.lo)
        ("f0", "f1", "f2", "f3")   -> static_cast<T>(e.f0) + ... + static_cast<T>(e.f3)
        ()                         -> static_cast<T>(e)          (native single scalar)

    ``limbs`` is the type's own limb members in MOST-SIGNIFICANT-FIRST order, and the sum
    is emitted in that order.  That ordering is what the two-limb profiles have always
    emitted, so dd/ff drivers stay byte-identical; for a four-limb ``QuadFloat`` it costs
    at most a sub-ulp difference against summing the tail first (the limbs are
    non-overlapping and decreasing, so every ordering lands within one ulp of the
    correctly-rounded caller-precision value — immaterial against a ~31-digit oracle).

    **Every limb is summed — no limb is dropped.** Truncating a ``QuadFloat`` to ``f0``
    alone would silently deliver ~7 digits where the type carries ~29.

    ``expr`` is any well-formed sub-expression naming the extended value (a name, a
    ``name.real()``, a driver printer's ``v``)."""
    if not limbs:
        return f"static_cast<{caller_type}>({expr})"
    return " + ".join(f"static_cast<{caller_type}>({expr}.{m})" for m in limbs)


def narrow_two_limb_scalar(expr: str, caller_type: str, two_limb: bool = True) -> str:
    """Two-limb (or native) spelling of :func:`narrow_extended_scalar`.

    The boolean-flavoured entry point the regional / fan-out / chain boundary machinery
    passes its ``two_limb`` flag to.  That flag is genuinely binary on those paths: they
    serve ``DoubleDouble`` / ``FloatFloat`` (``.hi``/``.lo``) or a native ``float``, and
    no four-limb type reaches them.  A caller that knows its type's limb members — the
    Phase-1 whole-TU flip printer, which also serves ``QuadFloat`` — should call
    :func:`narrow_extended_scalar` with an explicit ``limbs`` instead."""
    return narrow_extended_scalar(expr, caller_type,
                                  TWO_LIMB_MEMBERS if two_limb else ())


def _demote_expr(name: str, caller_type: str, two_limb: bool = True) -> str:
    """Demote a promoted *scalar* region write back to the caller precision.

    For a two-limb extended scalar (``FloatFloat`` / ``DoubleDouble``) this is two-limb
    reconstruction — ``static_cast<T>(w.hi) + static_cast<T>(w.lo)`` — the extended
    types' own conversion-out idiom (no ``operator double`` exists).  For a *native*
    single-limb scalar (plain ``float``, which has no ``.hi``/``.lo`` members) it is
    a plain cast — ``static_cast<T>(w)`` — so a float demotion compiles.  Delegates the
    reconstruction to :func:`narrow_two_limb_scalar` (shared with the Phase-1 flip
    boundary — STOP #TT).
    """
    return narrow_two_limb_scalar(name + _WRITE_SUFFIX, caller_type, two_limb)


def _demote_complex_expr(name: str, target_type: str, caller_type: str,
                         two_limb: bool = True) -> str:
    """Demote a promoted *complex-container* write back to the caller complex type.

    Phase 2d: a region operand whose type is a complex container
    (``Kokkos::complex<double>``, aliased ``TOutput``) promotes to the extended
    *container* (``FloatFloatComplex`` / ``DoubleDoubleComplex`` / ``std::complex<float>``), not the
    scalar.  On exit each component is reconstructed to the caller real precision and
    the caller complex value is rebuilt via ``target_type(re, im)``:

    * two-limb container (``FloatFloatComplex`` / ``DoubleDoubleComplex``, components carry ``.hi``/``.lo``)
      → ``T(static_cast<C>(w.re.hi)+static_cast<C>(w.re.lo),
             static_cast<C>(w.im.hi)+static_cast<C>(w.im.lo))``;
    * native container (``std::complex<float>``, components are plain ``float``) →
      ``T(static_cast<C>(w.real()), static_cast<C>(w.imag()))``.

    ``target_type`` is the write's own declared complex spelling (e.g. ``TOutput`` or
    ``Kokkos::complex<double>``) — assignable from the reconstructed value.
    """
    ext = name + _WRITE_SUFFIX
    if not two_limb:
        return (f"{target_type}(static_cast<{caller_type}>({ext}.real()), "
                f"static_cast<{caller_type}>({ext}.imag()))")
    re_ = narrow_two_limb_scalar(f"{ext}.re", caller_type, two_limb)
    im_ = narrow_two_limb_scalar(f"{ext}.im", caller_type, two_limb)
    return f"{target_type}({re_}, {im_})"


def _promote_complex_entry(name: str, src_expr: str, complex_type: str,
                           scalar_type: str) -> str:
    """Entry cast promoting a complex read/write ``src_expr`` to ``complex_type``.

    Each component is wrapped in the extended *scalar* first
    (``FloatFloatComplex(FloatFloat(z.real()), FloatFloat(z.imag()))``) so the value keeps full caller
    precision: a bare ``FloatFloatComplex(double, double)`` would bind ``FloatFloatComplex(float,
    float)`` and silently narrow the entry to single precision.  ``.real()``/``.imag()``
    are defined on every complex spelling in play (Kokkos/std/vendored).
    """
    return (f"{complex_type}({scalar_type}({src_expr}.real()), "
            f"{scalar_type}({src_expr}.imag()))")


def _complex_cast_indices(toks: list["_Tok"], complex_tokens: frozenset[str],
                          promoted_names: set[str]) -> set[int]:
    """Token indices of a functional cast ``T(<...promoted...>)`` to rewrite to the
    extended complex type (Phase 2d).

    A cast whose type name ``T`` is a complex spelling (``T`` in ``complex_tokens``)
    and whose balanced ``(...)`` argument references a promoted (extended-typed) name
    must build the extended *container*, not the caller's complex — ``TOutput(si*ta)``
    → ``FloatFloatComplex(si__ff*ta__ff)`` (``FloatFloatComplex`` has a ctor from ``FloatFloat``; the
    caller's ``Kokkos::complex<double>`` does not).  A cast with no promoted operand is
    left alone (still a plain caller-precision value).  A ``T`` in template-argument
    position (``Constants<T>``) is followed by ``>`` / ``::``, never ``(``, so it is
    never matched here.
    """
    idx: set[int] = set()
    n = len(toks)
    for i in range(n - 1):
        if toks[i].text not in complex_tokens or toks[i + 1].text != "(":
            continue
        depth = 0
        j = i + 1
        has_promoted = False
        while j < n:
            tj = toks[j].text
            if tj == "(":
                depth += 1
            elif tj == ")":
                depth -= 1
                if depth == 0:
                    break
            elif tj in promoted_names:
                has_promoted = True
            j += 1
        if has_promoted:
            idx.add(i)
    return idx


def _match_bracket(toks: list["_Tok"], open_idx: int) -> int | None:
    """Index of the ``]`` matching the ``[`` at ``open_idx`` (nested brackets ok)."""
    depth = 0
    for k in range(open_idx, len(toks)):
        tx = toks[k].text
        if tx == "[":
            depth += 1
        elif tx == "]":
            depth -= 1
            if depth == 0:
                return k
    return None


def _match_paren(toks: list["_Tok"], open_idx: int) -> int | None:
    """Index of the ``)`` matching the ``(`` at ``open_idx`` (nested parens ok)."""
    depth = 0
    for k in range(open_idx, len(toks)):
        tx = toks[k].text
        if tx == "(":
            depth += 1
        elif tx == ")":
            depth -= 1
            if depth == 0:
                return k
    return None


def _within_any(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    """True if ``[start, end)`` lies within any span in ``spans``."""
    return any(s <= start and end <= e for s, e in spans)


def widen_carrier_assign_line(line: str, carriers: frozenset[str],
                              complex_type: str, scalar_type: str) -> str | None:
    """Widen a plain assignment ``C = RHS;`` to a widened complex *carrier* ``C``.

    Region-core element promotion (2026-07-28), deliverable (c) — *receiving-local
    widen*.  A closure carrier's decl is widened to the chain's dd complex type by the
    ``ClosureDecl`` path, but a carrier written on a NON-region line (a sibling branch
    the pipeline did not select as a landing region — e.g. B14 ``fac = TOutput(-xs /
    (m2*m4*ta));`` at B2m.h:398) keeps its caller-precision RHS, which no longer assigns
    to the now-dd carrier.  Widen the RHS so it matches the carrier decl.

    Two RHS shapes, both build-exercised by the region transform:

    * a functional complex cast ``T( … )`` (``T`` a complex spelling) whose ``(…)`` is
      the whole RHS → rewrite the leading cast token to ``complex_type`` (``DoubleDoubleComplex(-xs
      / …)``, which binds ``DoubleDoubleComplex(double)``);
    * any other complex value ``v`` → component reconstruction
      ``complex_type(scalar_type((v).real()), scalar_type((v).imag()))``.

    Returns the rewritten line, or ``None`` when the line is not such an assignment
    (a decl, a compound assign, a multi-line statement, an already-dd RHS, or an RHS that
    already references a carrier — i.e. already dd-producing).  Never fires when
    ``carriers`` is empty, so a variant with no widened complex carrier is untouched.
    """
    if not carriers:
        return None
    toks = _tokenize(line)
    if len(toks) < 4:
        return None
    if toks[0].text not in carriers:
        return None
    # plain assignment: ``C = …`` (not ``==``/``+=`` and not a decl ``T C = …``).
    if toks[1].text != "=" or toks[2].text == "=":
        return None
    semi = _stmt_end(toks, 2)
    if semi is None:                        # multi-line statement — out of scope
        return None
    rhs_start = toks[2].start
    rhs_end = toks[semi].start
    rhs = line[rhs_start:rhs_end].strip()
    if not rhs:
        return None
    if _looks_dd(rhs, scalar_type, complex_type):
        return None                         # RHS already produces a dd value
    if any(t.text in carriers for t in toks[2:semi]):
        return None                         # RHS reads another carrier → already dd
    # Shape 1: the RHS is exactly a functional complex cast ``T( … )``.
    if (_is_ident(toks[2].text) and toks[2].text != complex_type
            and toks[3].text == "("):
        close = _match_paren(toks, 3)
        if close is not None and close == semi - 1:
            return line[:toks[2].start] + complex_type + line[toks[2].end:]
    # Shape 2: reconstruct the caller complex value into the dd container.
    wrapped = (f"{complex_type}({scalar_type}(({rhs}).real()), "
               f"{scalar_type}(({rhs}).imag()))")
    return line[:rhs_start] + wrapped + line[rhs_end:]


def demote_exit_carriers_line(line: str, carriers: frozenset[str],
                              caller_complex: str, caller_type: str,
                              two_limb: bool) -> str | None:
    """Demote widened-carrier *reads* on a designed-exit store line to caller precision.

    Region-core element promotion (2026-07-28), deliverable (b) — *Shape-1 designed-exit
    narrowing*.  A designed exit (``res(i,k) = fac`` kernel-output store, an out-param
    store, a benign extract) is a carried dd value's intended projection back to caller
    precision; the closure already exempts these lines from the write-truncation gate but
    the store itself is still emitted verbatim, so a now-dd carrier no longer assigns to
    the caller-precision sink (B14 ``res(i,1) = fac`` / ``res(i,0) = fac * wlogtmu``).
    Reconstruct the caller complex value at each carrier read so the projection lands.

    Only READ occurrences of a name in ``carriers`` are demoted (a store *to* the carrier,
    or a member/scope-qualified use, is skipped); a caller-precision co-operand
    (``wlogtmu``) is left untouched, so ``fac * wlogtmu`` becomes ``<demote(fac)> *
    wlogtmu`` — a caller-precision product assignable to the caller sink.  Returns the
    rewritten line, or ``None`` when no carrier read occurs (no-op).
    """
    if not carriers:
        return None
    toks = _tokenize(line)
    n = len(toks)
    edits: list[tuple[int, int, str]] = []
    for i, t in enumerate(toks):
        if t.text not in carriers:
            continue
        prev = toks[i - 1].text if i > 0 else ";"
        if prev in (".", ">", ":"):          # ``.c`` / ``->c`` / ``::c`` member/scope
            continue
        nxt = toks[i + 1].text if i + 1 < n else ";"
        nxt2 = toks[i + 2].text if i + 2 < n else ";"
        if nxt == "=" and nxt2 != "=":       # store target, not a read
            continue
        demoted = _demote_complex_value(t.text, caller_complex, caller_type, two_limb)
        edits.append((t.start, t.end, demoted))
    if not edits:
        return None
    return _apply_spans(line, edits)


def _element_read_edits(toks: list["_Tok"], region_text: str,
                        element_bases: frozenset[str], complex_type: str,
                        scalar_type: str) -> tuple[list[tuple[int, int, str]],
                                                   list[tuple[int, int]]]:
    """Span edits wrapping each ``base[...]`` READ of a fixed-size complex aggregate.

    An occurrence ``base[k]`` (``base`` in ``element_bases``) that is not the LHS of a
    bare store ``base[k] = …`` (nor a compound assign ``base[k] += …``) is a read: it
    is wrapped in :func:`_promote_complex_entry` so the element enters the promoted
    arithmetic at ``complex_type`` (full caller precision preserved component-wise).
    Returns ``(edits, spans)`` — ``spans`` are the wrapped ``base[...]`` char ranges so
    the caller can suppress overlapping inner edits.
    """
    edits: list[tuple[int, int, str]] = []
    spans: list[tuple[int, int]] = []
    n = len(toks)
    for i in range(n - 1):
        if toks[i].text not in element_bases or toks[i + 1].text != "[":
            continue
        # A member/qualified access ``x.base`` / ``x::base`` / ``x->base`` is a
        # different entity than the aggregate decl — skip it.
        prev = toks[i - 1].text if i > 0 else ";"
        if prev in (".", "::", "->"):
            continue
        close = _match_bracket(toks, i + 1)
        if close is None:
            continue
        after = toks[close + 1].text if close + 1 < n else ";"
        after2 = toks[close + 2].text if close + 2 < n else ";"
        # LHS of a store: bare ``=`` (not ``==``) or a compound assign ``+=``/``*=``/…
        if after == "=" and after2 != "=":
            continue
        if after in ("+", "-", "*", "/") and after2 == "=":
            continue
        start, end = toks[i].start, toks[close].end
        src = region_text[start:end]
        edits.append((start, end,
                      _promote_complex_entry(toks[i].text, src, complex_type,
                                             scalar_type)))
        spans.append((start, end))
    return edits, spans


def _looks_dd(rhs: str, scalar_type: str, complex_type: str | None) -> bool:
    """Heuristic: does ``rhs`` carry a promoted (extended) value?

    True when it names the extended scalar/container spelling or a boundary read/write
    alias — the only ways a promoted value reaches an element store in this transform.
    Keeps store demotion from firing on a plain caller-precision assignment.
    """
    return (scalar_type in rhs
            or (complex_type is not None and complex_type in rhs)
            or _READ_SUFFIX in rhs or _WRITE_SUFFIX in rhs)


def _stmt_end(toks: list["_Tok"], k: int) -> int | None:
    """Index of the depth-0 ``;`` at or after token ``k`` (statement terminator)."""
    depth = 0
    for j in range(k, len(toks)):
        tx = toks[j].text
        if tx in ("(", "[", "{"):
            depth += 1
        elif tx in (")", "]", "}"):
            depth -= 1
        elif tx == ";" and depth == 0:
            return j
    return None


def _demote_complex_value(expr: str, target_type: str, caller_type: str,
                          two_limb: bool) -> str:
    """Demote an arbitrary promoted-complex ``expr`` to ``target_type`` (caller precision).

    The store analogue of :func:`_demote_complex_expr`, but operating on a raw sub-
    expression (an element-store RHS) rather than a ``name+suffix`` alias.
    """
    e = f"({expr})"
    if not two_limb:
        return (f"{target_type}(static_cast<{caller_type}>({e}.real()), "
                f"static_cast<{caller_type}>({e}.imag()))")
    re_ = narrow_two_limb_scalar(f"{e}.re", caller_type, two_limb)
    im_ = narrow_two_limb_scalar(f"{e}.im", caller_type, two_limb)
    return f"{target_type}({re_}, {im_})"


def _demote_element_stores(text: str, ebases: dict, caller_type: str,
                           two_limb: bool, scalar_type: str,
                           complex_type: str | None) -> str:
    """Demote ``base[k] = <dd expr>;`` element stores back to the caller complex type.

    Re-tokenizes the already-promoted region text, finds each bare store into a
    fixed-size complex aggregate element, and — only when the RHS :func:`_looks_dd` —
    reconstructs the caller complex value so the aggregate stays at caller precision
    (its declaration is never retyped).  Compound assigns and reads are left alone.
    """
    toks = _tokenize(text)
    n = len(toks)
    edits: list[tuple[int, int, str]] = []
    for i in range(n - 1):
        if toks[i].text not in ebases or toks[i + 1].text != "[":
            continue
        prev = toks[i - 1].text if i > 0 else ";"
        if prev in (".", "::", "->"):
            continue
        close = _match_bracket(toks, i + 1)
        if close is None:
            continue
        after = toks[close + 1].text if close + 1 < n else ";"
        after2 = toks[close + 2].text if close + 2 < n else ";"
        if not (after == "=" and after2 != "="):
            continue                       # only a bare store is an element write
        eq = close + 1
        semi = _stmt_end(toks, eq + 1)
        if semi is None:
            continue
        rhs = text[toks[eq + 1].start:toks[semi].start].strip()
        if not _looks_dd(rhs, scalar_type, complex_type):
            continue
        target = ebases[toks[i].text]
        new_rhs = _demote_complex_value(rhs, target, caller_type, two_limb)
        edits.append((toks[eq + 1].start, toks[semi].start, new_rhs))
    return _apply_spans(text, edits)


def promote_region_block(
    region_text: str,
    reads: list[str],
    writes: list[str],
    scalar_type: str,
    caller_type: str = "double",
    two_limb: bool = True,
    *,
    complex_type: str | None = None,
    complex_tokens=frozenset(),
    complex_names=frozenset(),
    caller_complex: str | None = None,
    closure_names=frozenset(),
    element_bases=frozenset(),
) -> tuple[list[str], bool]:
    """Promote a region's source to ``scalar_type``; return ``(block_lines, promoted)``.

    ``block_lines`` is the region rewritten as entry casts (reads/pre-declared
    writes → extended) + the retyped/renamed region body + exit demotions (writes →
    caller precision), one source line per element, ready to splice in place of the
    original region.  ``promoted`` is ``False`` when nothing in the region promotes
    (no reads, no pre-declared writes, no dataflow-reached local decls) — in that
    case ``block_lines`` is the region verbatim.

    Phase 2d — **complex-container promotion.**  A promoted operand whose type is a
    complex container promotes to the extended *complex* type ``complex_type``
    (``FloatFloatComplex`` / ``DoubleDoubleComplex`` / ``std::complex<float>``) instead of the scalar,
    fixing the dominant Phase-2c ``llm_gen_failed`` class (``FloatFloat(complex)`` /
    ``complex(FloatFloat)`` etc.).  An operand is treated as complex when the caller flags
    its name in ``complex_names`` (reads / pre-declared writes, classified from the
    enclosing function's decls + the app's template-parameter binding) or when a
    region-local decl's declared type token is in ``complex_tokens`` (e.g. ``TOutput``,
    the literal ``complex``).  Complex reads promote via
    :func:`_promote_complex_entry`, complex writes demote via
    :func:`_demote_complex_expr` (component-wise reconstruction), and a functional cast
    ``TOutput(<promoted>)`` in the body is rewritten to the extended container ctor
    (:func:`_complex_cast_indices`).  ``caller_complex`` is the concrete caller complex
    spelling (``Kokkos::complex<double>``) a pre-declared complex write demotes back to.
    When ``complex_type`` is ``None`` this is exactly the pre-2d scalar-only transform.

    This is the single definition of the regional promotion transform: both the
    *diff-producing* :func:`synthesize_boundary_patch` (Phase-1 include-site
    boundary patch) and the *text-producing* Phase-2a fan-out (which splices the
    promoted block into a copied function variant) build on it, so the two realizations
    of "promote this region" stay bit-identical.
    """
    closure_names = frozenset(closure_names)
    # Normalize element bases to ``{base name: element-type spelling}`` (region-core
    # element promotion).  A plain iterable of names maps each to the caller complex
    # spelling (the store-demotion target).  Only used when complex promotion is on.
    if isinstance(element_bases, dict):
        ebases = dict(element_bases)
    else:
        ebases = {b: (caller_complex or caller_type) for b in element_bases}
    prom = _compute_promotion(region_text, reads, writes, closure_names,
                              frozenset(ebases))
    toks = prom.toks
    pure_reads = prom.pure_reads
    caseB = prom.caseB
    decl_writes = prom.decl_writes

    # No promotable *local write* (region-local decl or pre-declared write) means the
    # promotion has nowhere to land in the region body: entry-promoted reads flow
    # straight into a sink the transform does not retype — a subscripted aggregate
    # store (``res(i,k) = …`` / ``res(i,k) /= …``), a call, or a bare expression.
    #
    # For an UPCAST (ff/dd, ``two_limb``) that shape is inert or unconvertible: the
    # widened value either fails to convert (``complex<double> /= FloatFloat``) or is
    # silently truncated back to the caller precision on store — no observable effect.
    # Report it honestly as a no-op (→ Patcher ``promotion_no_op``), Phase 2d.
    #
    # For a DOWNCAST (native ``float``, ``two_limb`` false) the same shape is NOT a
    # no-op *when there is a read to demote*: demoting a read to ``float`` and feeding
    # it into the (double) sink loses precision that propagates into the stored value —
    # a genuine, discriminating measurement (boxGPU.h:140-142 ``res(i,k) /= scalefac2``,
    # de≈5.8e-8 ≫ baseline; regressed to promotion_no_op in the first 2d-A cut, restored
    # here).  A downcast with *no* promotable read (``T c = T(k);`` — the sole operand is
    # an int index) is still an empty payload and no-ops.
    #
    # So the guard fires when the promotion cannot land AND either the rung is an upcast
    # or there is nothing to promote; only a downcast-with-reads falls through.
    #
    # A carrier write is a genuine landing (§8): the region writes a value into a
    # decl the emission layer has widened to the extended type, so the widened value
    # persists past this region — the promotion is NOT a no-op even when the region
    # declares/re-assigns nothing else.
    if (not decl_writes and not caseB and not prom.carrier_writes
            and (two_limb or not pure_reads)):
        return region_text.split("\n"), False

    complex_tokens = frozenset(complex_tokens)
    complex_names = frozenset(complex_names)
    use_complex = complex_type is not None

    # Which promoted operands are complex containers (else real scalars).
    complex_set: set[str] = set()
    if use_complex:
        complex_set |= {r for r in pure_reads if r in complex_names}
        complex_set |= {w for w in caseB if w in complex_names}
        complex_set |= {d.name for d in decl_writes if d.type_text in complex_tokens}

    rename_map = {r: r + _READ_SUFFIX for r in pure_reads}
    rename_map.update({w: w + _WRITE_SUFFIX for w in caseB})
    rename_map.update({d.name: d.name + _WRITE_SUFFIX for d in decl_writes})

    # Per-decl retype target: the complex container for a complex-typed local, else
    # the extended scalar.
    retype_target: dict[int, str] = {}
    for d in decl_writes:
        retype_target[d.type_idx] = (complex_type if (use_complex and
                                     d.type_text in complex_tokens) else scalar_type)

    cast_idx: set[int] = set()
    if use_complex and complex_tokens:
        cast_idx = _complex_cast_indices(toks, complex_tokens, set(rename_map))

    edits: list[tuple[int, int, str]] = []
    for i, t in enumerate(toks):
        if i in retype_target:
            edits.append((t.start, t.end, retype_target[i]))
        elif i in cast_idx:
            edits.append((t.start, t.end, complex_type))
        elif t.text in rename_map:
            edits.append((t.start, t.end, rename_map[t.text]))

    # Region-core element promotion: wrap each ``base[k]`` READ occurrence of a
    # fixed-size complex aggregate in an entry cast to ``complex_type`` so a promoted
    # dd operand no longer multiplies a caller-precision ``Kokkos::complex<double>``
    # element (the STOP #CC ``complex<DoubleDoubleComplex>`` form).  The array declaration is
    # left untouched — no whole-array retype, so the d1 failure mode cannot recur.
    if use_complex and ebases:
        elem_edits, elem_spans = _element_read_edits(
            toks, region_text, frozenset(ebases), complex_type, scalar_type)
        # A whole-``base[k]``-span replacement subsumes any inner-token edit; drop
        # main edits that fall inside a wrapped element span (defensive — inner index
        # tokens are not promoted names in practice).
        edits = [e for e in edits if not _within_any(e[0], e[1], elem_spans)]
        edits += elem_edits
    new_region_text = _apply_spans(region_text, edits)

    # Demote element STORES on exit: a bare ``base[k] = <dd expr>;`` into a caller-
    # precision aggregate must reconstruct the caller complex value (design row 3).
    # Guarded by ``_looks_dd`` so only genuine dd RHSs are rewritten (no in-scope
    # integral exercises this today; correctness is proven by unit test, not the
    # sweep — avoids a dead-code false positive).
    if use_complex and ebases:
        new_region_text = _demote_element_stores(
            new_region_text, ebases, caller_type, two_limb, scalar_type, complex_type)

    region_lines = region_text.split("\n")
    indent = _leading_ws(region_lines[0]) if region_lines else ""
    entry: list[str] = []
    for r in pure_reads:
        if r in complex_set:
            rhs = _promote_complex_entry(r, r, complex_type, scalar_type)
            entry.append(f"{indent}{complex_type} {r}{_READ_SUFFIX} = {rhs};"
                         f"  // Rule R1: promote complex region read to {complex_type}")
        else:
            entry.append(f"{indent}{scalar_type} {r}{_READ_SUFFIX} = {scalar_type}({r});"
                         f"  // Rule R1: promote region read to {scalar_type}")
    for w in caseB:
        if w in complex_set:
            rhs = _promote_complex_entry(w, w, complex_type, scalar_type)
            entry.append(f"{indent}{complex_type} {w}{_WRITE_SUFFIX} = {rhs};"
                         f"  // Rule R1: seed pre-declared complex write in {complex_type}")
        else:
            entry.append(f"{indent}{scalar_type} {w}{_WRITE_SUFFIX} = {scalar_type}({w});"
                         f"  // Rule R1: seed pre-declared write in {scalar_type}")

    exit_lines: list[str] = []
    for d in decl_writes:   # region-local decl → declare the alias at its own type
        if d.name in complex_set:
            expr = _demote_complex_expr(d.name, d.type_text, caller_type, two_limb)
        else:
            expr = _demote_expr(d.name, d.type_text, two_limb)
        exit_lines.append(
            f"{indent}{d.type_text} {d.name} = {expr};"
            f"  // Rule R1: demote region write to {d.type_text}")
    for w in caseB:         # pre-declared write → assign back at the caller precision
        if w in complex_set:
            tgt = caller_complex or caller_type
            expr = _demote_complex_expr(w, tgt, caller_type, two_limb)
            exit_lines.append(f"{indent}{w} = {expr};"
                              f"  // Rule R1: demote complex region write to {tgt}")
        else:
            exit_lines.append(f"{indent}{w} = {_demote_expr(w, caller_type, two_limb)};"
                              f"  // Rule R1: demote region write to {caller_type}")

    return entry + new_region_text.split("\n") + exit_lines, True


def write_truncation_inert(
    region_text: str,
    reads: list[str],
    writes: list[str],
    two_limb: bool,
    *,
    caller_type: str = "double",
    complex_tokens=frozenset(),
    caller_complex: str | None = None,
    closure_names=frozenset(),
) -> bool:
    """Phase 2d-B — provably-inert *write-boundary truncation* detector.

    Returns ``True`` when an UPCAST promotion is provably numerically inert because
    every value it widens is truncated back to caller precision at the region
    boundary — there is no persistent *wider* sink for the extended value to survive
    in.  The Patcher turns a ``True`` here into a terminal ``write_truncation`` status
    (no build), the upcast analogue of the empty-payload ``promotion_no_op``.

    Fires iff ALL of:

    * ``two_limb`` — an UPCAST (ff / dd).  A native ``float`` DOWNCAST is *never*
      flagged: truncating a read to a narrower ``float`` at an aggregate/store sink
      genuinely loses precision (``boxGPU.h:140-142``, de≈5.8e-8) — the exact
      regression the 2d-A guard fix restored.  Same ``two_limb`` discipline.
    * the promotion has something to widen (a read / pre-declared write / promoted
      local decl) — else it is the *empty payload* ``promotion_no_op`` class, not
      write-truncation.
    * every promoted region-local decl lands at a **recognized** caller-precision
      type — the literal ``caller_type``, the ``caller_complex`` spelling, or a
      ``complex_tokens`` entry (which resolves to the caller complex, e.g. ``TOutput``
      → ``Kokkos::complex<double>``).  A decl at an *unrecognized* template type
      (``TScale`` / ``TMass``) is treated as a possibly-wider persistent sink and the
      region is left to honest build+measure — conservative, so ``boxGPU.h:139``
      (``const TScale scalefac2 = scalefac * scalefac;``) stays a real measurement.
    * there is at least one **provable** caller-precision landing — a pre-declared
      (Case-B) write (always demoted to ``caller_type`` on exit) or a decl at a
      recognized caller type.  A bare ``return`` / expression with no store is not
      flagged (nothing provably truncates; an extended multi-op reduction rounded
      once at the return could discriminate — honest build+measure).

    Deterministic, source-only, upstream of any build.  Mirrors
    :func:`promote_region_block`'s no-op guard but for the *landed-but-truncated*
    case rather than the *nothing-promotes* case; the two are mutually exclusive by
    construction (this needs a landing, that fires only when none exists).

    ``closure_names`` (design §8) are chain carriers whose decl is widened to the
    extended type by the emission layer.  They are excluded from the ``caseB`` /
    ``decl_writes`` sets this gate inspects (the exclusion happens inside
    :func:`_compute_promotion`), so a region whose only "truncating" writes are
    now-widened carriers is no longer read as inert — the extended value survives in
    the widened carrier decl.  The gate reasoning for every non-carrier write is
    unchanged; the change is strictly additive.
    """
    if not two_limb:
        return False

    closure_names = frozenset(closure_names)
    prom = _compute_promotion(region_text, reads, writes, closure_names)
    if (not prom.pure_reads and not prom.caseB and not prom.decl_writes
            and not prom.carrier_writes):
        return False  # empty payload → promotion_no_op territory, not truncation

    recognized = {caller_type}
    if caller_complex:
        recognized.add(caller_complex)
    recognized |= set(complex_tokens)

    # A region-local decl at an unrecognized (possibly-wider) type is a persistent
    # sink the extended value could survive in → not provably inert; bail.
    if any(d.type_text not in recognized for d in prom.decl_writes):
        return False

    # Need ≥1 landing that provably truncates back to caller precision: a Case-B
    # write (always demoted to caller_type) or a recognized-caller decl.  After the
    # guard above every decl is recognized, so a decl is a provable landing.
    return bool(prom.caseB) or bool(prom.decl_writes)


def synthesize_boundary_patch(
    *,
    rel_file: str,
    file_text: str,
    line_start: int,
    line_end: int,
    reads: list[str],
    writes: list[str],
    scalar_type: str,
    caller_type: str = "double",
    shim_include: str | None = None,
    two_limb: bool = True,
) -> str | None:
    """Synthesize the region's boundary patch as a ``git apply -p1`` unified diff.

    Parameters mirror the regional integrator contract: ``rel_file`` is the
    repo-relative path (drives the ``a/``,``b/`` diff labels), ``file_text`` the
    full original file, ``[line_start, line_end]`` the inclusive 1-based region,
    ``reads`` the characterizer's ``region_local_vars``, ``writes`` the Fix-C write
    set, ``scalar_type`` the extended C++ spelling (e.g. ``Kokkos::Experimental::FloatFloat``),
    ``caller_type`` the precision to demote back to, and ``shim_include`` the shim
    header basename to ``#include``.  ``two_limb`` selects the write-demotion idiom:
    two-limb reconstruction for an extended scalar (default), or a plain cast for a
    native single-limb ``float``.  Returns the diff, or ``None`` if the region needs
    no boundary edit (no reads, no writes, no include).
    """
    original_lines = file_text.split("\n")
    if line_start < 1 or line_end > len(original_lines) or line_start > line_end:
        return None

    region_lines = original_lines[line_start - 1:line_end]
    region_text = "\n".join(region_lines)

    # Dataflow promotion partition (reads / pre-declared writes / promoted local
    # decls) via the shared transform — same block the Phase-2a fan-out splices into
    # a variant copy.  ``promoted`` is False when nothing in the region promotes.
    new_block, promoted = promote_region_block(
        region_text, reads, writes, scalar_type, caller_type, two_limb)

    if not promoted and not shim_include:
        return None

    patched_lines = original_lines[:line_start - 1] + new_block + original_lines[line_end:]

    # -- shim include after #pragma once ---------------------------------------
    if shim_include:
        patched_lines = _insert_shim_include(patched_lines, shim_include)

    patched_text = "\n".join(patched_lines)
    if file_text.endswith("\n") and not patched_text.endswith("\n"):
        patched_text += "\n"

    if patched_text == file_text:
        return None

    diff = difflib.unified_diff(
        file_text.splitlines(keepends=True),
        patched_text.splitlines(keepends=True),
        fromfile=f"a/{rel_file}", tofile=f"b/{rel_file}",
    )
    combined = "".join(diff)
    return combined if combined.strip() else None


def insert_shim_include(lines: list[str], shim_include: str) -> list[str]:
    """Public wrapper over :func:`_insert_shim_include` (used by the Phase-2a fan-out
    to give a variant-bearing file its shim ``#include``, idempotently)."""
    return _insert_shim_include(lines, shim_include)


class BoundaryError(RuntimeError):
    """A deterministic boundary rewrite that cannot be realized safely.

    Raised by :func:`widen_return_type_line` when the ``orig_type`` token is not
    present at the given return line (and the line is not already widened) — a silent
    no-op there would emit a variant whose *signature* says one type while its body
    returns another, i.e. a wrong-typed variant at runtime.  The design's rule-(c)
    safety argument (CLOSURE_SCOPED_CHAINS_DESIGN.md §7) depends on this rewrite being
    deterministic and complete, so an unrecognized shape hard-fails rather than
    papering over."""


def widen_return_type_line(
    source: str, *, return_line: int, orig_type: str, dd_type: str,
    function_name: str,
) -> str:
    """Widen the return-type token of a function *declaration* in ``source``.

    The mirror of :func:`widen_decl_type_line` for a callable's **return type**
    (CLOSURE_SCOPED_CHAINS_DESIGN.md §7, rule (c)).  Where the decl-widen rewrites a
    local's declaration so an interior chain write lands in a dd carrier, this
    rewrites a per-integral variant's return type so a dd value carried out via
    ``return`` survives the return instead of rounding to caller precision (B10's
    ``:707`` severance).  It edits the variant *clone* only; the shared original is
    never passed here (Appendix invariant).

    ``return_line`` is the 1-based line (within ``source``) where the return-type
    token *starts* — the signature line, e.g. ``KOKKOS_INLINE_FUNCTION TOutput
    Li2omx2(...)``, NOT the ``return`` statement.  ``orig_type`` is the leading
    return-type token as source spells it (``TOutput``); ``dd_type`` the widened
    replacement (``DoubleDoubleComplex`` / ``Kokkos::Experimental::DoubleDouble``); ``function_name`` the
    variant name for diagnostics only (the rewrite operates on the source position,
    so the source may still carry the *original* function name at this point).

    Handled shapes (all via the shared :func:`_tokenize` scanner — no new parsing
    strategy):

    * **template return types** — ``TOutput``, ``std::complex<TScale>``,
      ``typename std::conditional<...>::type``.  The return type is every token
      before the function-name identifier that precedes the parameter-list ``(``; the
      ``orig_type`` base token within it is swapped, so a qualified / templated type
      keeps its qualifiers and template arguments byte-for-byte.
    * **multi-line return types** — a long template return whose leading token is on
      ``return_line`` and whose function name falls on the next line: the forward
      scan spans lines until the parameter-list ``(`` and rewrites the token on
      whichever line it lands.
    * **const / reference-qualified returns** — ``const TOutput&``: ``const`` and
      ``&`` are preserved; only the ``orig_type`` base token is rewritten.
    * **static / inline / template / macro keywords before the return type**
      (``KOKKOS_INLINE_FUNCTION``, ``static``, ``inline``): preserved — they are part
      of the pre-name token region and are never the ``orig_type`` base token.

    Idempotent: re-applying an already-widened return (``orig_type`` gone, ``dd_type``
    present in the return-type region) returns ``source`` unchanged rather than
    raising.  Raises :class:`BoundaryError` when ``orig_type`` is neither found nor
    already-widened (a coordinate / type mismatch, or an unhandled shape such as an
    ``auto`` trailing-return ``-> decltype(...)`` / SFINAE ``enable_if`` return).
    """
    lines = source.split("\n")
    n = len(lines)
    if return_line < 1 or return_line > n:
        raise BoundaryError(
            f"widen_return_type_line: return_line {return_line} out of range "
            f"[1, {n}] for {function_name!r}")

    # Gather tokens from return_line forward until the parameter-list '(' at
    # angle/bracket depth 0 — the opener that follows the function name.  A few
    # lines of span covers a multi-line template return type.
    span: list[tuple[int, _Tok]] = []      # (line_index_0based, token)
    paren_open: int | None = None          # index into ``span`` of the param '('
    depth = 0
    li = return_line - 1
    last_li = min(n - 1, li + 6)
    while li <= last_li and paren_open is None:
        for t in _tokenize(lines[li]):
            span.append((li, t))
            if t.text in ("<", "["):
                depth += 1
            elif t.text in (">", "]"):
                depth = max(0, depth - 1)
            elif t.text == "(" and depth == 0:
                paren_open = len(span) - 1
                break
        li += 1
    if paren_open is None:
        raise BoundaryError(
            f"widen_return_type_line: no parameter-list '(' found within reach of "
            f"return line {return_line} for {function_name!r}; not a function "
            f"declaration in the recognized form")

    # function name = last identifier immediately before the param '('; the
    # return-type region is every token before it (qualifiers + type expression).
    fn_idx: int | None = None
    for k in range(paren_open - 1, -1, -1):
        if _is_ident(span[k][1].text):
            fn_idx = k
            break
    if fn_idx is None:
        raise BoundaryError(
            f"widen_return_type_line: no function-name identifier before '(' at "
            f"return line {return_line} for {function_name!r}")
    rt = span[:fn_idx]

    # locate the orig_type base-type token (first occurrence in the return-type
    # region).  A ``::``-qualified type's last segment is a legitimate rewrite target
    # (``std::complex`` → orig_type ``complex``; mirrors :func:`widen_decl_type_line`,
    # which navigates ``::`` to the last segment), so a leading ``:`` is NOT a
    # disqualifier — only a value member access ``a.b`` is.
    target: tuple[int, _Tok] | None = None
    for k, (lidx, t) in enumerate(rt):
        if t.text != orig_type:
            continue
        prev = rt[k - 1][1].text if k > 0 else ""
        if prev == ".":
            continue
        target = (lidx, t)
        break

    if target is None:
        # idempotent re-application: the return type is already widened → no-op.
        dd_tail = dd_type.split("::")[-1].split("<")[0]
        if any(t.text == dd_tail for _l, t in rt):
            return source
        rt_text = " ".join(t.text for _l, t in rt)
        raise BoundaryError(
            f"widen_return_type_line: orig_type {orig_type!r} not found in the "
            f"return type of {function_name!r} at line {return_line} (return-type "
            f"tokens: {rt_text!r}) and it is not already widened to {dd_type!r} — "
            f"refusing to edit; a silent no-op would emit a wrong-typed variant")

    lidx, tok = target
    lines[lidx] = lines[lidx][:tok.start] + dd_type + lines[lidx][tok.end:]
    return "\n".join(lines)


def widen_decl_type_line(line: str, orig_type: str, dd_type: str) -> str | None:
    """Rewrite the leading (core) type token of a bare declaration on ``line``.

    A **carrier** declaration (Blocker A, design §7) is a bare / bare
    multi-declarator statement (``TMass Y, S, A;``) whose type must be widened so
    the interior chain writes land in a dd carrier instead of truncating.  This
    reuses the Subtask-3 bare-decl scanner (:func:`_scan_bare_decls` /
    :func:`_parse_bare_decl_stmt`) to locate the leading type token and swaps its
    text for ``dd_type`` — widening every same-type sibling of a multi-declarator
    in one edit (§2 conservative policy).

    Returns the rewritten line, or ``None`` when the line does not parse as a bare
    declaration whose leading type is ``orig_type`` (caller treats ``None`` as "no
    change" — an idempotent re-render of an already-widened decl, or a
    coordinate/type mismatch, leaves the line verbatim).  Uses the token span so
    only the type token is replaced; qualifiers, ``const``, siblings, and trailing
    initializers are preserved byte-for-byte.
    """
    toks = _tokenize(line)
    for rec in _scan_bare_decls(toks):
        if rec.type_text != orig_type:
            continue
        tok = toks[rec.type_idx]
        return line[:tok.start] + dd_type + line[tok.end:]
    return None


def _insert_shim_include(lines: list[str], shim_include: str) -> list[str]:
    """Insert ``#include "<shim>"`` into a target header (idempotent).

    Placement (design fix, 2026-07-19): the shim *specializes* templates
    (``Constants<...>``, the extended-precision ops) that the header's own app
    ``#include``s declare.  If the shim lands before those includes, the compiler
    sees the specialization before the primary template — ``"Constants is not a
    class template"``.  So the shim goes **after every #include in the header
    preamble** (system or app — after the last one is trivially after all app
    ones) and **before the first code/decl**.  When the header has no includes we
    fall back to the top of the file, preserving ``#pragma once`` / include-guard
    semantics (insert *after* them, never before).
    """
    include_line = f'#include "{shim_include}"'
    if any(line.strip() == include_line for line in lines):
        return lines
    at = _shim_insert_index(lines)
    return lines[:at] + [include_line] + lines[at:]


def _shim_insert_index(lines: list[str]) -> int:
    """Index at which to splice the shim include (comment-aware preamble scan).

    Priority within the leading preamble (everything before the first code/decl
    line): after the last ``#include`` > after an include-guard ``#define`` >
    after ``#pragma once`` > top of file.  Block/line comments (e.g. a license
    header) are skipped so a copyright banner never looks like code and truncates
    the scan before the include block.
    """
    last_include = None
    guard_define = None
    pragma_once = None
    pending_ifndef: str | None = None
    in_block_comment = False

    for idx, raw in enumerate(lines):
        line = raw

        # -- consume an open block comment (may close mid-line) --
        if in_block_comment:
            close = line.find("*/")
            if close == -1:
                continue                      # whole line still inside the comment
            in_block_comment = False
            line = line[close + 2:]           # scan whatever trails the close

        stripped = line.strip()
        if stripped == "":
            continue                          # blank / comment-only remainder
        if stripped.startswith("//"):
            continue                          # line comment
        if stripped.startswith("/*"):
            # one-line /* ... */ → comment; otherwise open a block comment.
            if "*/" not in stripped[2:]:
                in_block_comment = True
            continue

        if stripped.startswith("#"):
            directive = stripped[1:].lstrip()
            if directive.startswith("include"):
                last_include = idx
                pending_ifndef = None
            elif directive.startswith("pragma") and directive.split()[1:2] == ["once"]:
                pragma_once = idx
                pending_ifndef = None
            elif directive.startswith("ifndef"):
                toks = directive.split()
                pending_ifndef = toks[1] if len(toks) > 1 else None
            elif directive.startswith("define"):
                toks = directive.split()
                if pending_ifndef is not None and len(toks) > 1 and toks[1] == pending_ifndef:
                    guard_define = idx        # classic include-guard #define
                pending_ifndef = None
            else:
                # any other directive breaks the classic #ifndef/#define adjacency
                pending_ifndef = None
            continue

        # first genuine code/decl line → the preamble is over.
        break

    if last_include is not None:
        return last_include + 1
    if guard_define is not None:
        return guard_define + 1
    if pragma_once is not None:
        return pragma_once + 1
    return 0


def _leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _dedupe(items: list[str]) -> list[str]:
    """Order-preserving de-duplication."""
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out
