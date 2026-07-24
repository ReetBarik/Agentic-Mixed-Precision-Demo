"""Shared engine for the *regional* integrators (ff / dd).

``ff_integrator.integrate_region`` and ``dd_integrator.integrate_region`` are
structural twins: given a region descriptor ``(file, line_start, line_end,
variables)`` and a working-tree SHA, both (1) read the region source at that SHA,
(2) recover its write set, (3) ask an LLM for an extended-precision **shim** (the
types / operators / named constants the region needs), and (4) synthesize a
deterministic **boundary patch** that promotes the region's reads to the extended
scalar on entry and demotes its writes back on exit.  They differ only in their
ruleset (system prompt), their concrete C++ scalar/complex spellings
(``quad::ffun::ffloat`` vs ``quad::ddfun::ddouble``), and a per-type note in the
user turn (DD needs hex-encoded constant tables).  That common flow lives here;
the two ``agent.py`` modules are thin wrappers that supply a :class:`RegionalSpec`.

The design (§P4) keeps the boundary patch deterministic and out of the LLM's hands
— the LLM produces only the shim; :mod:`agents.integrator_base.boundary` turns
``(reads, writes, scalar_type, caller_type)`` into the diff.  A re-roll (Patcher's
N=3 retry, ``attempt`` varying the user turn) regenerates only the shim.

Modeled on :func:`agents.tracked_integrator.agent.integrate` (SOURCE_HASH cache,
streaming generation, rule-justified output); the regional differences are the
region-scoped inputs and the boundary patch as a first-class output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from agents.integrator_base import boundary, cache, llm, shim_merge
from agents.integrator_base.region import RegionIntegrationResult
from agents.shared import constant_derive as cderive
from agents.shared.region_scan import RegionScanError, extract_region_writes

# Max output tokens for a regional shim.  A regional shim is far smaller than the
# whole-app tracked shim (a single region's types/constants), but keep generous
# headroom so a shim is never truncated mid-file.
_MAX_OUTPUT_TOKENS = 8000

# A regional shim's include set is CLOSED (system-prompt rule C1): the vendored
# extended-precision headers, and nothing app-source.  The dominant Patcher
# failure in the 2026-07-18 shakedown (run 20260718_194556_67dbcf37) was DD shims
# hallucinating app-source includes — ``#include "ql/constants.h"``,
# ``<qcdloop/types.h>``, ``<Kokkos_Macros.hpp>`` — that are not on the shim's
# include path, so every such build died with ``fatal error: <path>: No such file
# or directory`` before the shim was ever honestly tested (P6a "dd_untested").
# The prompt is the primary fix; this lint is the deterministic safety net.
#
# Standard-library headers are harmless (always on the include path) and a minimal
# shim should not need them, but allowing them avoids rejecting an otherwise
# buildable shim — a false reject would burn a retry.  App-source headers are the
# actual failure mode and are always rejected.
_STDLIB_HEADERS = frozenset({
    "cstdint", "cstddef", "cstdlib", "cmath", "cfloat", "climits", "cassert",
    "complex", "limits", "type_traits", "utility", "array", "tuple", "algorithm",
})

# ``#include <foo>`` / ``#include "foo"`` as a real preprocessor directive (first
# non-whitespace token is ``#include``, so a ``//``-commented line never matches).
_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]')

# --------------------------------------------------------------------------- #
# Gap A — namespace-qualified math bridge
# --------------------------------------------------------------------------- #
# A namespace-qualified call ``Ns::fn(x)`` skips ADL: name lookup only searches
# ``Ns`` (and its enclosing scopes), NOT the vendored ``quad::ffun`` / ``quad::ddfun``
# namespace where the shim's ADL overloads live.  So when a *promoted* (extended-
# typed) value flows into ``Ns::fn(...)`` and ``Ns`` is not the vendored namespace,
# the shim must inject a bridging overload into ``Ns`` (or a using-declaration) or
# the call falls back to the primary, which tries to narrow the extended value to a
# built-in float and dies with ``cannot convert 'quad::ddfun::ddouble' to 'const
# double'`` (the 2026-07-18 B0m.h:69 symptom).

# Root namespaces the shim already reaches via ADL — never need a bridge.
_VENDORED_NS_ROOTS = frozenset({"quad"})

# Standard C++ <cmath>/<complex> free-function names.  Framework-agnostic: this is
# the math library vocabulary, NOT a list of target frameworks — a qualified call
# ``AnyNs::sqrt(promoted)`` needs a bridge regardless of which framework AnyNs is.
# App-specific math wrappers (qcdloop's kAbs/kSqrt, etc.) are deliberately NOT here
# — they are handled by the prompt + build gate, so the lint never guesses at names
# it cannot know are math.
_MATH_FN_NAMES = frozenset({
    "abs", "fabs", "sqrt", "cbrt", "exp", "exp2", "expm1",
    "log", "log2", "log10", "log1p", "pow", "hypot",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sincos",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "erf", "erfc", "tgamma", "lgamma",
    "floor", "ceil", "round", "trunc", "nearbyint", "rint",
    "fmod", "remainder", "fmin", "fmax", "fdim", "copysign", "nextafter",
    "conj", "real", "imag", "norm", "arg", "polar", "proj",
})

# ``Root::...::fn(`` — the full qualifier chain in group 1, the called function in
# group 2.  A ``<`` after the name (``Ns::Type<...>``) is not matched (no ``(``),
# so class-template accessors like ``Constants<T>::pi()`` are excluded here.
_QUALIFIED_CALL_RE = re.compile(
    r'(?<![\w:])((?:[A-Za-z_]\w*\s*::\s*)+)([A-Za-z_]\w*)\s*\(')


def _balanced_args(text: str, open_paren: int) -> str:
    """Return the argument text between ``text[open_paren] == '('`` and its match."""
    depth = 0
    i = open_paren
    n = len(text)
    while i < n:
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren + 1:i]
        i += 1
    return text[open_paren + 1:]


def _contains_promoted(arg_text: str, promoted: frozenset[str]) -> bool:
    """True if ``arg_text`` references a promoted identifier (whole-word)."""
    if not promoted:
        return False
    for m in re.finditer(r'[A-Za-z_]\w*', arg_text):
        if m.group(0) in promoted:
            return True
    return False


def find_qualified_math_calls(region_text: str, promoted: frozenset[str]):
    """Namespace-qualified math calls on promoted args that need a bridge (Gap A).

    Returns a de-duplicated list of ``(qualifier_root, fn, full_qualifier)`` for
    each ``Root::...::fn(<args with a promoted operand>)`` where ``fn`` is a
    standard math function and ``Root`` is not the vendored namespace.  These are
    exactly the calls whose extended-typed argument would otherwise be narrowed to
    a built-in float by the primary overload.
    """
    found: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for m in _QUALIFIED_CALL_RE.finditer(region_text):
        chain = re.sub(r"\s+", "", m.group(1))       # e.g. "cuda::std::"
        fn = m.group(2)
        root = chain.split("::", 1)[0]
        if fn not in _MATH_FN_NAMES or root in _VENDORED_NS_ROOTS:
            continue
        args = _balanced_args(region_text, m.end() - 1)
        if not _contains_promoted(args, promoted):
            continue
        key = (root, fn)
        if key in seen:
            continue
        seen.add(key)
        found.append((root, fn, chain.rstrip(":")))
    return found


def _shim_bridges_qualifier(shim_body: str, root: str, fn: str) -> bool:
    """True if the shim provides a bridge reachable from ``root::fn`` calls.

    Accepts either injection into the qualifier namespace (``namespace root { …
    fn … }``) or a using-declaration path (``using namespace root;`` /
    ``using ns::fn;``) — the two remedies the prompt sanctions ((a) and (b)).
    """
    if re.search(r'\busing\s+namespace\s+' + re.escape(root) + r'\b', shim_body):
        return True
    if re.search(r'\busing\b[^;\n]*::\s*' + re.escape(fn) + r'\b', shim_body):
        return True
    # namespace <root> injected (possibly nested) AND fn defined somewhere in shim
    if re.search(r'\bnamespace\s+' + re.escape(root) + r'\b', shim_body) and \
            re.search(r'\b' + re.escape(fn) + r'\s*\(', shim_body):
        return True
    return False


def _lint_qualified_bridges(region_text: str, shim_body: str,
                            promoted: frozenset[str]) -> str | None:
    """Reject a shim that omits a bridge for a namespace-qualified math call.

    Deterministic safety net for Gap A (mirrors the C1 include lint): if the
    region invokes ``Ns::fn(promoted)`` on a standard math function and the shim
    injects no bridge into ``Ns`` (and no using-declaration for it), the call will
    narrow the extended value to a float and break the build — so treat it as a
    retryable misgeneration.  Returns ``None`` when every such call is bridged.
    """
    missing: list[str] = []
    for root, fn, chain in find_qualified_math_calls(region_text, promoted):
        if not _shim_bridges_qualifier(shim_body, root, fn):
            missing.append(f"{chain}::{fn}")
    if not missing:
        return None
    return (
        "C3 bridge lint: region makes namespace-qualified math call(s) "
        f"{missing} on promoted (extended-typed) operands, but the shim injects no "
        "bridging overload into that namespace (and no using-declaration). A "
        "qualified call skips ADL, so the vendored quad:: overloads are not found "
        "and the extended value is narrowed to a built-in float (hard build "
        "failure). Emit an overload in the qualifier namespace forwarding to the "
        "vendored op (C3). Treating as a retryable misgeneration."
    )


# --------------------------------------------------------------------------- #
# Gap B — source-derivable constants
# --------------------------------------------------------------------------- #
# R3's original cascade stopped at "vendored factory or memorized hex pair"; a
# named constant with neither fell through to the Rule R4 #error even when its own
# source definition made it trivially derivable (``_ieps50 = TScale(1e-50)``).  We
# resolve such constants deterministically (agents.shared.constant_derive) and hand
# the model the ready-made ``make_dd(...)`` / ``make_ff(...)`` value so it never
# guesses bits or bails to R4.

# A name read as a member/namespace accessor that leads a call or template-id:
# ``::name(`` / ``::template name<…>``.  We capture the name and let the source
# walk decide whether it is actually a constant (a type/method name that resolves
# to no literal definition is silently dropped) — trying to skip the template-arg
# list in the regex mis-spans nested ``<…>::…<…>`` and captures the wrong token.
_ACCESSOR_CONST_RE = re.compile(r'::\s*(?:template\s+)?([A-Za-z_]\w*)\s*(?=[<(])')
# A macro / ALL-CAPS constant read bare (``M_PI``, ``TWO_PI``, ``MY_TINY`` if caps).
_MACRO_CONST_RE = re.compile(r'(?<![\w:.])(M_[A-Z0-9_]+|[A-Z][A-Z0-9_]{2,})\b')

# Header extensions worth scanning for a constant's definition, and scan caps
# (deterministic + bounded — a numeric kernel's constants live in a handful of
# headers, never hundreds).
_SOURCE_EXTS = (".h", ".hpp", ".hh", ".hxx", ".cuh", ".inl", ".ipp")
_MAX_SOURCE_FILES = 400
_MAX_SOURCE_BYTES = 512 * 1024


def _find_constant_candidates(region_text: str) -> list[str]:
    """Identifier names in the region that may denote a named constant (Gap B).

    Union of accessor-style reads (``Constants<T>::_ieps50<…>()``) and bare
    macro/ALL-CAPS reads.  Math free functions handled by the Gap-A bridge are
    excluded; the real filter is downstream — only names that actually resolve to
    a derivable source definition are surfaced.
    """
    names: list[str] = []
    seen: set[str] = set()
    for rx in (_ACCESSOR_CONST_RE, _MACRO_CONST_RE):
        for m in rx.finditer(region_text):
            nm = m.group(1)
            if nm in _MATH_FN_NAMES or nm in seen:
                continue
            seen.add(nm)
            names.append(nm)
    return names


def _gather_constant_sources(repo_path: str | None, region_src: str) -> list[str]:
    """Source texts to search for a constant's definition: the region file first,
    then scan-reachable repo headers (bounded)."""
    sources = [region_src]
    if not repo_path:
        return sources
    root = Path(repo_path)
    if not root.is_dir():
        return sources
    count = 0
    for p in sorted(root.rglob("*")):
        if count >= _MAX_SOURCE_FILES:
            break
        if p.suffix.lower() not in _SOURCE_EXTS or not p.is_file():
            continue
        try:
            if p.stat().st_size > _MAX_SOURCE_BYTES:
                continue
            sources.append(p.read_text(encoding="utf-8", errors="ignore"))
            count += 1
        except OSError:
            continue
    return sources


def derive_region_constants(region_text: str, sources: list[str], scalar: str,
                            complex_type: str | None = None):
    """Resolve + derive every source-derivable constant the region reads (Gap B).

    ``scalar`` is ``"dd"`` or ``"ff"``; ``complex_type`` is the concrete C++ complex
    spelling (``quad::ddfun::ddcomplex`` / ``quad::ffun::ffcomplex``) used to
    assemble a *complex-container* constant into a complete value.  Each entry is a
    dict with the constant ``name``, the resolved source ``rhs``, and one of:

    * ``expr`` — a full ready-made value (a scalar constant derived whole, OR a
      complex-container constant assembled as ``<complex_type>(<re>, <im>)``);
    * ``literals`` — per-literal derivations for a composite RHS the model must
      assemble itself (only when the whole value could not be derived).

    A complex container (``_ieps50 = TOutput{_zero(), 1e-50}``, an *imaginary* iε
    regulator) is derived whole so the model can no longer collapse it to a real
    scalar (the residual dd_untested cause after Wave 1 — see
    :func:`agents.shared.constant_derive.derive_complex_from_rhs`).
    """
    out: list[dict] = []
    for name in _find_constant_candidates(region_text):
        rhs = cderive.resolve_constant_rhs(name, sources)
        if rhs is None:
            continue
        whole = cderive.derive_from_rhs(name, rhs, scalar)
        if whole is not None:
            out.append({"name": name, "rhs": rhs, "expr": whole.expr,
                        "how": whole.how, "literals": []})
            continue
        # Complex-container constant (imaginary iε regulator etc.): derive BOTH
        # limbs and assemble the full complex value with the vendored complex type.
        if complex_type:
            cx = cderive.derive_complex_from_rhs(name, rhs, scalar, sources)
            if cx is not None:
                out.append({"name": name, "rhs": rhs,
                            "expr": f"{complex_type}({cx.real}, {cx.imag})",
                            "how": "complex", "literals": []})
                continue
        lits = cderive.derive_literals_in(rhs, scalar)
        if lits:
            out.append({"name": name, "rhs": rhs, "expr": None, "how": "composite",
                        "literals": [{"lit": d.name, "expr": d.expr} for d in lits]})
    return out


@dataclass
class RegionalSpec:
    """The per-integrator surface the shared engine needs (ff vs dd)."""

    system_prompt: str            # the LLM ruleset (bytes feed SOURCE_HASH)
    cpp_scalar: str               # e.g. "quad::ffun::ffloat"
    cpp_complex: str              # e.g. "quad::ffun::ffcomplex"
    vendored_headers: list[str]   # e.g. ["ff_math.hpp", "ff_complex.hpp"]
    shim_prefix: str              # "ff" | "dd" — filename tag
    constant_note: str = ""       # extra user-turn guidance (DD hex constants)
    # Include-set allowlist for the C1 lint.  ``None`` -> derived from
    # ``vendored_headers`` ∪ the standard-library set; set explicitly to override.
    allowed_includes: list[str] | None = None
    # --- target-kind knobs -------------------------------------------------- #
    # A two-limb extended scalar ({hi, lo}: ffloat / ddouble) demotes writes back
    # to the caller via two-limb reconstruction; a *native* single-limb scalar
    # (plain ``float``) has no ``.hi``/``.lo`` and demotes with a plain cast.
    two_limb: bool = True
    # Run the Gap-A namespace-qualified bridge lint.  Only extended (non-native)
    # scalars can narrow through a qualified call; a native ``float`` needs no
    # bridge (a ``float`` argument binds a ``double`` overload by widening).
    emit_bridges: bool = True
    # Run the Gap-B two-limb constant derivation.  A native ``float`` carries no
    # sub-limb precision, so a source literal is just its float literal — the app's
    # own ``Constants<float>`` (visible at the include site) already supplies them.
    derive_constants: bool = True


def run_integrate_region(
    spec: RegionalSpec,
    *,
    file: str,
    line_start: int,
    line_end: int,
    variables: list[str],
    working_tree: str,
    scalar_type: str,
    caller_type: str = "double",
    direction: str = "in",
    out_dir: Path,
    attempt: int = 0,
    repo_path: str | None = None,
    cfg=None,
    llm_fn=None,
) -> RegionIntegrationResult:
    """Generate a regional ff/dd shim + boundary patch for one region.

    ``scalar_type`` is the short tag the Patcher passes (``"ffloat"`` /
    ``"ddouble"``); the concrete C++ spelling used in the shim and boundary patch
    comes from ``spec.cpp_scalar``.  ``llm_fn(system, user, attempt) -> str`` is a
    test seam — when ``None`` the real streaming call is used (built from ``cfg``,
    defaulting to :class:`~agents.config.PipelineConfig`).  Never raises past the
    seam: any failure is returned as an ``llm_failed`` result so the Patcher's
    bounded retry (P4) can re-roll.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve the region source at the pinned SHA.
    try:
        src = _git_show(repo_path, working_tree, file)
    except Exception as exc:  # noqa: BLE001
        return RegionIntegrationResult.failed(f"could not read {file}@{working_tree}: {exc}")

    lines = src.split("\n")
    if line_start < 1 or line_end > len(lines) or line_start > line_end:
        return RegionIntegrationResult.failed(
            f"region {file}:{line_start}-{line_end} out of range (file has {len(lines)} lines)")
    region_src = "\n".join(lines[line_start - 1:line_end])

    # 2. Region writes (Fix C).  Keyed on the caller type; for a vanilla region the
    #    template-keyed scan finds nothing and the boundary module recovers the
    #    write set from its own decl scan.  A scan error is non-fatal.
    try:
        writes = extract_region_writes(file, line_start, line_end, working_tree,
                                       tracked_type=caller_type)
    except RegionScanError:
        writes = []

    # 3. SOURCE_HASH cache key (region ⊕ ruleset ⊕ scalar ⊕ writes).
    cache_key = cache.compute_region_hash(region_src, spec.system_prompt,
                                          spec.cpp_scalar, writes)
    # Region-scoped shim filename: the line range makes the name unique per
    # region so two regions in the SAME file with byte-identical source (which
    # produce the same content cache_key) get DISTINCT shim files instead of
    # colliding on one name.  Without the ``L{start}_{end}`` scope the later
    # region cache-hits the earlier region's shim, installs a byte-identical
    # copy, nets no tree change, and the Patcher's commit fails "nothing to
    # commit" — historically escalated to a fatal run abort.  It also stops a
    # later region from silently overwriting an earlier region's accepted shim.
    # The full cache_key still gates the content (via ``_is_cache_hit``), so a
    # re-run of the SAME region still reuses its shim.
    shim_name = (f"{Path(file).stem}_{spec.shim_prefix}"
                 f"_L{line_start}_{line_end}_{cache_key[:8]}.h")
    shim_out = out_dir / shim_name

    # 4. Cache hit (only on the first attempt — a retry must re-roll the shim).
    if attempt == 0 and _is_cache_hit(shim_out, cache_key):
        shim_text = shim_out.read_text(encoding="utf-8")
        canonical_name = _install_canonical(spec, repo_path, shim_text)
        patch = _boundary(spec, file, src, line_start, line_end, variables, writes,
                          caller_type, canonical_name, repo_path)
        return RegionIntegrationResult(status="ok", shim_paths=[str(shim_out)],
                                       boundary_patch=patch, llm_tokens=0)

    # 4a. Deterministic region analysis feeding the user turn (Gaps A + B).
    #     - promoted-name set (shared with the boundary patch dataflow) tells us
    #       which qualified math calls land on an extended-typed operand (Gap A);
    #     - source-derivable constants are resolved + pre-derived so the model gets
    #       ready-made make_dd/make_ff values instead of hitting Rule R4 (Gap B).
    promoted = frozenset(boundary.compute_promoted_names(region_src, list(variables),
                                                         list(writes)))
    qualified_calls = find_qualified_math_calls(region_src, promoted) if spec.emit_bridges else []
    if spec.derive_constants:
        sources = _gather_constant_sources(repo_path, src)
        derived_constants = derive_region_constants(region_src, sources,
                                                    spec.shim_prefix, spec.cpp_complex)
    else:
        derived_constants = []

    # 5. Generate the shim (LLM).
    user_msg = _build_user_message(spec, file, region_src, variables, writes,
                                   line_start, line_end, caller_type,
                                   qualified_calls=qualified_calls,
                                   derived_constants=derived_constants)
    if attempt > 0:
        user_msg += f"\n// regeneration attempt {attempt}\n"

    try:
        if llm_fn is not None:
            shim_body = llm_fn(spec.system_prompt, user_msg, attempt)
            tokens = 0
        else:
            from agents.config import PipelineConfig
            shim_body, tokens = llm.stream_shim(
                spec.system_prompt, user_msg, cfg or PipelineConfig(), _MAX_OUTPUT_TOKENS)
    except Exception as exc:  # noqa: BLE001
        return RegionIntegrationResult.failed(f"LLM generation failed: {exc}")

    if not shim_body or not shim_body.strip():
        return RegionIntegrationResult.failed("LLM returned empty shim")

    # 5a. Deterministic include-set lint (C1 safety net).  A shim that pulls an
    #     app-source header cannot build — reject it as a misgen so the Patcher's
    #     N=3 retry re-rolls, exactly as for any other bad generation.  This does
    #     NOT count against the Strategy transition budget (a failed gen never
    #     produces an accepted transition).
    bad_include = _lint_include_set(shim_body, _allowed_include_set(spec))
    if bad_include is not None:
        return RegionIntegrationResult.failed(bad_include, llm_tokens=tokens)

    # 5b. Deterministic namespace-qualified bridge lint (Gap A / C3 safety net).
    #     A qualified math call on a promoted operand with no bridge overload in
    #     the shim narrows the extended value to a float and breaks the build —
    #     reject as a retryable misgen so the Patcher re-rolls (same semantics as
    #     the C1 lint; never counts against the Strategy transition budget).
    if spec.emit_bridges:
        bad_bridge = _lint_qualified_bridges(region_src, shim_body, promoted)
        if bad_bridge is not None:
            return RegionIntegrationResult.failed(bad_bridge, llm_tokens=tokens)

    # 5c. Deterministic complex anti-pattern lint (Phase 2d, bucket (a) insurance).
    #     A shim must never wrap the extended scalar in a std/Kokkos complex; the
    #     vendored ffcomplex/ddcomplex is the container.  Retryable misgen.
    bad_complex = _lint_complex_antipattern(shim_body, spec.cpp_scalar)
    if bad_complex is not None:
        return RegionIntegrationResult.failed(bad_complex, llm_tokens=tokens)

    # 6. Stamp the SOURCE_HASH and persist the per-region artifact (out_dir copy —
    #    the forensic/cache record of THIS region's generated shim), then merge it
    #    into the canonical per-family shim installed in the tree (Wave-3 dedup).
    shim_text = cache.apply_source_hash(shim_body, cache_key)
    shim_out.write_text(shim_text, encoding="utf-8")
    canonical_name = _install_canonical(spec, repo_path, shim_text)

    # 7. Deterministic boundary patch — includes the canonical merged shim, so
    #    every region of this family shares ONE definition of each TU-global symbol.
    patch = _boundary(spec, file, src, line_start, line_end, variables, writes,
                      caller_type, canonical_name, repo_path)

    return RegionIntegrationResult(status="ok", shim_paths=[str(shim_out)],
                                   boundary_patch=patch, llm_tokens=tokens)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _allowed_include_set(spec: RegionalSpec) -> frozenset[str]:
    """Header names a regional shim may `#include` (C1): vendored ∪ stdlib.

    ``spec.allowed_includes`` overrides the vendored set when given; the
    standard-library headers are always permitted (harmless, always on path).
    """
    base = spec.allowed_includes if spec.allowed_includes is not None else spec.vendored_headers
    return frozenset(base) | _STDLIB_HEADERS


def _lint_include_set(shim_body: str, allowed: frozenset[str]) -> str | None:
    """Return an error message if the shim `#include`s anything outside ``allowed``.

    The C1 rule keeps the shim's include set closed to the vendored extended-
    precision headers (plus harmless stdlib headers).  Any other `#include` — an
    app-source path like ``ql/constants.h`` / ``qcdloop/types.h`` / ``Kokkos_*.hpp``
    — is not on the shim's include path and guarantees a hard build failure, so we
    treat it as a misgeneration.  Returns ``None`` when the include set is clean.
    """
    forbidden: list[str] = []
    for line in shim_body.splitlines():
        m = _INCLUDE_RE.match(line)
        if not m:
            continue
        header = m.group(1).strip()
        if header not in allowed:
            forbidden.append(header)
    if not forbidden:
        return None
    return (
        "C1 include lint: shim includes forbidden app-source header(s) "
        f"{forbidden} — a regional shim's include set is closed to the vendored "
        f"headers {sorted(allowed - _STDLIB_HEADERS)} (+ stdlib). App-source "
        "headers are not on the shim include path and break the build; the boundary "
        "patch owns caller-side wiring. Treating as a retryable misgeneration."
    )


def _lint_complex_antipattern(shim_body: str, cpp_scalar: str) -> str | None:
    """Reject a shim that instantiates ``std``/``Kokkos::complex`` on an *extended*
    scalar (Phase 2d, bucket (a) regression insurance).

    ``Kokkos::complex<T>`` / ``std::complex<T>`` require a cv-unqualified built-in
    floating-point ``T``; a two-limb class scalar (``ffloat`` / ``ddouble``) trips the
    ``static_assert`` (``complex can only be instantiated for a cv-unqualified floating
    point type``) or finds no ctor.  The vendored ``ffcomplex`` / ``ddcomplex`` standalone
    types are the correct complex container (SPEC Rule 3).  A *native* ``float`` is
    exempt — ``Kokkos::complex<float>`` is a legal instantiation and is in fact the
    float rung's own complex spelling — so the guard fires only for the extended
    two-limb scalars.  Returns ``None`` when the shim is clean.  This class was 0/30 in
    the Phase-2c runs (the LLM is Rule-3 compliant); the lint keeps it that way.
    """
    core = cpp_scalar.rsplit("::", 1)[-1]
    if core not in ("ffloat", "ddouble"):
        return None
    pat = re.compile(r"\bcomplex\s*<\s*[^>]*\b" + re.escape(core) + r"\b")
    hits = [m.group(0) for m in pat.finditer(shim_body)]
    if not hits:
        return None
    return (
        f"Rule 3 complex anti-pattern: shim instantiates a std/Kokkos complex on the "
        f"extended scalar {hits} — such a complex requires a cv-unqualified built-in "
        f"floating-point element and rejects the two-limb class scalar ``{core}`` "
        f"(static_assert / no ctor). Use the vendored complex container instead of "
        f"``complex<{core}>``. Treating as a retryable misgeneration."
    )


def _boundary(spec, file, src, line_start, line_end, variables, writes,
              caller_type, shim_name, repo_path) -> str | None:
    return boundary.synthesize_boundary_patch(
        rel_file=_rel_file(file, repo_path),
        file_text=src,
        line_start=line_start, line_end=line_end,
        reads=list(variables), writes=list(writes),
        scalar_type=spec.cpp_scalar, caller_type=caller_type,
        shim_include=shim_name, two_limb=spec.two_limb,
    )


def _is_cache_hit(shim_out: Path, cache_key: str) -> bool:
    if not shim_out.exists():
        return False
    return cache.extract_source_hash(shim_out.read_text(encoding="utf-8")) == cache_key


def canonical_shim_name(spec: RegionalSpec) -> str:
    """Filename of the single per-family canonical shim (Wave-3 dedup).

    All regions of a family (``dd`` / ``ff`` / ``float``) merge their generated
    TU-global symbols into this one header, so the translation unit sees exactly
    one definition of each ``Constants<T>`` specialization / ``ql::`` helper.
    """
    return f"ql_shim_{spec.shim_prefix}.h"


def _install_canonical(spec: RegionalSpec, repo_path: str | None,
                       shim_body: str) -> str:
    """Merge ``shim_body`` into the canonical per-family shim in the tree.

    Reads the current canonical shim (if the Patcher already committed a sibling
    region's shim for this family), merges the new region's symbols in — class
    specializations accumulate members, free functions/namespaces dedup by
    signature (keep-first) — and writes the merged canonical back.  Returns the
    canonical shim's basename (what the boundary patch ``#include``s).  When
    ``repo_path`` is unset (no candidate tree) this is a no-op returning the name.
    """
    canonical_name = canonical_shim_name(spec)
    if not repo_path:
        return canonical_name
    canonical_path = Path(repo_path) / canonical_name
    existing = canonical_path.read_text(encoding="utf-8") if canonical_path.exists() else None
    merged = shim_merge.merge_into_canonical(existing, shim_body)
    canonical_path.write_text(merged, encoding="utf-8")
    return canonical_name


def _rel_file(file: str, repo_path: str | None) -> str:
    p = Path(file)
    if not p.is_absolute():
        return p.as_posix()
    if repo_path:
        try:
            return p.resolve().relative_to(Path(repo_path).resolve()).as_posix()
        except ValueError:
            pass
    return p.name


def _git_show(repo_path: str | None, sha: str, file: str) -> str:
    """Return ``file`` content at ``sha`` via ``git show`` (mirrors region_scan)."""
    import subprocess

    p = Path(file)
    anchor = Path(repo_path) if repo_path else (p.parent if p.is_absolute() else Path.cwd())
    root = subprocess.run(["git", "-C", str(anchor), "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if p.is_absolute():
        rel = p.resolve().relative_to(Path(root).resolve()).as_posix()
    else:
        rel = p.as_posix()
    r = subprocess.run(["git", "-C", root, "show", f"{sha}:{rel}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    return r.stdout


def _build_user_message(spec: RegionalSpec, file: str, region_src: str,
                        variables: list[str], writes: list[str],
                        line_start: int, line_end: int, caller_type: str,
                        qualified_calls=None, derived_constants=None) -> str:
    """Assemble the user turn: region source + read/write sets + scalar + API.

    ``qualified_calls`` (Gap A) and ``derived_constants`` (Gap B) are deterministic
    region-analysis hints: the former lists namespace-qualified math calls that
    need a bridge overload, the latter hands the model ready-made make_dd/make_ff
    values for source-derivable constants so they never reach Rule R4.
    """
    parts: list[str] = []
    parts.append(
        f"Promote the code region below to the extended scalar type "
        f"`{spec.cpp_scalar}` (and `{spec.cpp_complex}` for complex values). Emit a "
        f"C++ shim header that supplies the named-constant wrappers and any missing "
        f"operators/overloads this region needs so it compiles once its reads are "
        f"promoted and its locals are the extended type. The caller precision is "
        f"`{caller_type}`.\n"
    )
    parts.append(f"## Region — `{file}:{line_start}-{line_end}`\n```cpp\n{region_src}\n```\n")

    reads = ", ".join(variables) if variables else "(none)"
    written = ", ".join(writes) if writes else "(none reported; boundary handles region-local decls)"
    parts.append(
        f"## Boundary variables\n"
        f"- Reads (promoted to `{spec.cpp_scalar}` on entry — handled by the "
        f"deterministic boundary patch, NOT by your shim): {reads}\n"
        f"- Writes (demoted back to `{caller_type}` on exit — also boundary-handled): "
        f"{written}\n"
        f"Do NOT emit the promote/demote casts yourself; the boundary patch inserts "
        f"them. Your shim provides only the types/operators/constants the promoted "
        f"region references.\n"
    )

    parts.append(
        f"## Extended-precision API (vendored; call these, do NOT redefine the type)\n"
        f"The scalar `{spec.cpp_scalar}` and complex `{spec.cpp_complex}` are already "
        f"defined in: {', '.join(spec.vendored_headers)}. Your shim must `#include` "
        f"them (angle-bracket, they are on the include path).\n"
    )
    if spec.constant_note:
        parts.append(spec.constant_note + "\n")

    bridge_note = _format_qualified_calls(spec, qualified_calls)
    if bridge_note:
        parts.append(bridge_note)
    const_note = _format_derived_constants(spec, derived_constants)
    if const_note:
        parts.append(const_note)

    parts.append(
        f"## Output\n"
        f"Emit ONLY the complete shim header contents — no prose, no markdown fences. "
        f"Start with `#pragma once`, then a `// SOURCE_HASH: PENDING` line (the caller "
        f"stamps the real hash). `#include` the vendored headers above. Every "
        f"generated overload, specialization, or constant wrapper must carry a "
        f"comment naming the rule (Rule N / C-N / R-N) that justified it. If you "
        f"cannot classify something, emit the Rule R4 `#error` escape hatch rather "
        f"than guessing."
    )
    return "\n".join(parts)


def _format_qualified_calls(spec: RegionalSpec, qualified_calls) -> str:
    """Gap-A hint: namespace-qualified math calls that need a bridge overload."""
    if not qualified_calls:
        return ""
    vendored_ns = spec.cpp_scalar.rsplit("::", 1)[0]   # e.g. quad::ddfun
    lines = [
        "## Namespace-qualified calls needing a bridge (C3)",
        "These calls in the region are **namespace-qualified**, so they skip ADL "
        f"and will NOT find your `{vendored_ns}` overloads. Each is invoked on a "
        "promoted (extended-typed) operand, so the primary overload would narrow "
        "the extended value to a built-in float (hard build failure). For EACH, "
        "inject a bridging overload into that namespace forwarding to the vendored "
        "op (preferred), or a using-declaration if injecting there is forbidden:",
    ]
    for root, fn, chain in qualified_calls:
        lines.append(
            f"- `{chain}::{fn}(...)` → e.g. "
            f"`namespace {root} {{ ... {fn}({spec.cpp_scalar} x) {{ "
            f"return {vendored_ns}::{fn}(x); }} }}` (name the originating call in a comment)."
        )
    return "\n".join(lines) + "\n"


def _format_derived_constants(spec: RegionalSpec, derived_constants) -> str:
    """Gap-B hint: source-derivable constants pre-derived to make_dd/make_ff.

    Surfaces the resolved source RHS and the exact bit-pair value so the model
    uses it verbatim instead of guessing hex or bailing to Rule R4.
    """
    if not derived_constants:
        return ""
    factory = "make_dd" if spec.shim_prefix == "dd" else "make_ff"
    lines = [
        "## Source-derivable constants (Rule R3, step 3)",
        "The constants below were resolved from their source definitions and "
        f"pre-derived to exact `{factory}(...)` bit pairs. Use these VERBATIM — do "
        "NOT guess hex and do NOT emit Rule R4 for them. A source `double`/`float` "
        "literal carries only that precision, so its faithful extended value has a "
        "zero low word; this is correct, not a truncation.",
    ]
    for c in derived_constants:
        if c.get("expr") and c.get("how") == "complex":
            lines.append(
                f"- `{c['name']}` (source RHS `{c['rhs']}`) is a COMPLEX container "
                f"(e.g. an imaginary iε regulator `0 + im·i`) → return the FULL "
                f"complex value `{c['expr']}` VERBATIM. Your wrapper MUST return the "
                f"complex type and preserve BOTH the real and imaginary parts — do "
                f"NOT collapse it to a real scalar (that drops the imaginary axis)."
            )
        elif c.get("expr"):
            lines.append(
                f"- `{c['name']}` (source RHS `{c['rhs']}`, {c['how']}) → `{c['expr']}`"
            )
        elif c.get("literals"):
            lits = "; ".join(f"`{l['lit']}` → `{l['expr']}`" for l in c["literals"])
            lines.append(
                f"- `{c['name']}` (source RHS `{c['rhs']}`) is composite; assemble it "
                f"from these derived literals: {lits}"
            )
    return "\n".join(lines) + "\n"
