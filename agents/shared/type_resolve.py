"""App template-parameter → concrete-type resolution (Phase 2d).

The regional boundary transform must know which of a region's promoted operands are
**complex** (``Kokkos::complex<double>``) vs real **scalars** (``double``): a complex
operand promotes to the extended *container* (``ffcomplex`` / ``ddcomplex``), while a
real scalar promotes to the extended *scalar* (``ffloat`` / ``ddouble``).  Promoting a
complex operand to the scalar type is the dominant Phase-2c ``llm_gen_failed`` cause —
``ffloat(complex)`` / ``complex(ffloat)`` / ``complex /= ffloat`` never compile.

In a templated kernel the operand types are **template parameters** (``TOutput`` /
``TMass`` / ``TScale``) whose concrete binding lives ONLY at the driver's instantiation
site — ``run_app<Kokkos::complex<double>, double, double, ...>`` — never in the
(uninstantiated) headers.  Neither a token scan nor libclang can recover it from a
region alone: an uninstantiated primary template collapses ``TOutput`` to
``type-parameter-0-0``.  So we recover the binding the only way it survives — by
**source scan of the app**, pairing a template *definition*'s parameter names with a
concrete *instantiation*'s argument types.

Deterministic, source-only, runtime-free — the same discipline as
:mod:`agents.shared.constant_derive` and :mod:`agents.shared.region_scan`.  One more
scan layer: given the app source roots (the tree headers ∪ the driver that
instantiates the entry template), return ``{param_name: concrete_type}`` and the
subset of parameter names that denote a complex container.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

# Source extensions worth scanning for template defs / instantiations, and bounds
# (mirrors constant_derive — a kernel's driver + headers are a handful of files).
_SOURCE_EXTS = (".h", ".hpp", ".hh", ".hxx", ".cuh", ".inl", ".ipp", ".cpp", ".cc", ".cxx")
_MAX_SOURCE_FILES = 600
_MAX_SOURCE_BYTES = 1024 * 1024

# Aggregate / container core type spellings (last ``::`` segment) — a value of one of
# these is not a scalar the boundary can wrap; it is never promoted.  Matches the
# region_scan aggregate set plus common std containers.
_AGGREGATE_CORES = frozenset({"View", "Array", "vector", "array", "span", "tuple", "pair"})

# A ``template < ... >`` head followed by the thing it templates.  We capture the
# parameter clause; the def name is the first identifier after the ``>`` that leads a
# ``(`` (function) or ``<`` (its own explicit args) or ``{`` (struct/class body) — see
# ``_template_defs``.
_TEMPLATE_HEAD_RE = re.compile(r"template\s*<")
_PARAM_RE = re.compile(r"(?:class|typename)\s+([A-Za-z_]\w*)")


class TypeResolveError(RuntimeError):
    """Raised only for programmer errors (bad roots argument); scan failures are
    non-fatal and yield an empty binding so the caller degrades to scalar-only."""


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def classify_concrete_type(spelling: str) -> str:
    """Classify a concrete C++ type spelling as ``complex`` / ``aggregate`` / ``scalar``.

    Keyed on the *core* type name — the last ``::`` segment before any ``<`` — so
    ``Kokkos::complex<double>`` and ``std::complex<float>`` both classify ``complex``;
    ``Kokkos::View<...>`` / ``Kokkos::Array<...>`` classify ``aggregate``; a pointer or
    C-array is ``aggregate``; ``double`` / ``float`` / a bare identifier is ``scalar``.
    """
    s = spelling.strip()
    if not s:
        return "scalar"
    if "*" in s or "[" in s:                    # pointer / C-array → not a scalar
        return "aggregate"
    head = s.split("<", 1)[0]                    # drop template-arg list
    head = head.replace("&", " ")               # references are not aggregates
    core = head.rsplit("::", 1)[-1].strip()     # last namespace segment
    parts = core.split()                        # drop leading cv-qualifiers ("const X")
    core = parts[-1] if parts else core
    if core == "complex":
        return "complex"
    if core in _AGGREGATE_CORES:
        return "aggregate"
    return "scalar"


def resolve_bindings(app_roots, caller_type: str = "double") -> dict[str, str]:
    """Map each entry-template parameter name to its concrete instantiation type.

    Scans ``app_roots`` for template definitions and concrete instantiations of the
    same name, zips parameter names against concrete arguments, and — when a template
    has several instantiations (a vanilla + a dd driver, say) — selects the one whose
    non-complex/non-aggregate arguments equal ``caller_type`` (the baseline build the
    pass measures against, e.g. ``double``).  Returns ``{param: concrete_type}`` merged
    across every resolvable template; ``{}`` when nothing resolves (caller then
    degrades to scalar-only promotion, i.e. the pre-2d behavior).
    """
    roots = _norm_roots(app_roots)
    return dict(_resolve_bindings_cached(roots, caller_type))


def complex_param_names(bindings: dict[str, str]) -> frozenset[str]:
    """Parameter names in ``bindings`` whose concrete type is a complex container."""
    return frozenset(p for p, c in bindings.items()
                     if classify_concrete_type(c) == "complex")


def complex_type_tokens(bindings: dict[str, str]) -> frozenset[str]:
    """Type-name TOKENS that denote a complex type in a region's scope.

    The complex-bound template parameter names (``TOutput``) plus the literal complex
    core spelling ``complex`` — so the boundary transform recognizes both a
    template-typed decl ``const TOutput fac`` and an explicit ``Kokkos::complex<...>``
    or a functional cast ``TOutput(...)`` as complex.
    """
    return complex_param_names(bindings) | frozenset({"complex"})


# --------------------------------------------------------------------------- #
# scanning internals
# --------------------------------------------------------------------------- #

def _norm_roots(app_roots) -> tuple[str, ...]:
    if app_roots is None:
        return ()
    if isinstance(app_roots, (str, Path)):
        app_roots = [app_roots]
    return tuple(sorted(str(Path(r).resolve()) for r in app_roots if r))


@lru_cache(maxsize=64)
def _resolve_bindings_cached(roots: tuple[str, ...], caller_type: str) -> tuple[tuple[str, str], ...]:
    texts = _gather_texts(roots)
    defs: dict[str, list[str]] = {}
    for _name, text in texts:
        for dname, params in _template_defs(text).items():
            # keep the first (or the richest) param list seen for a name
            if dname not in defs or len(params) > len(defs[dname]):
                defs[dname] = params

    binding: dict[str, str] = {}
    for dname, params in defs.items():
        insts = []
        for _name, text in texts:
            insts.extend(_instantiations(text, dname, len(params)))
        if not insts:
            continue
        chosen = _select_instantiation(insts, caller_type)
        if chosen is None:
            continue
        for p, a in zip(params, chosen):
            # first binding wins unless the existing one is uninformative
            if p not in binding:
                binding[p] = a
    return tuple(sorted(binding.items()))


def _select_instantiation(insts: list[list[str]], caller_type: str) -> list[str] | None:
    """Pick the instantiation matching the baseline build the pass measures against.

    The vanilla build spells its numeric args as the caller type (``double``,
    ``Kokkos::complex<double>``); a sibling dd build spells them ``ddouble`` /
    ``ddcomplex``.  Score each instantiation by how many arguments name ``caller_type``
    (as a bare scalar or the real component of a ``complex<caller_type>``) and take the
    max — vanilla scores on its ``double`` args, dd scores zero.  Non-numeric class args
    (a ``Printer`` policy type) are ignored.  Ties (and single-build apps) keep the
    first instantiation.
    """
    def caller_hits(args: list[str]) -> int:
        n = 0
        for a in args:
            kind = classify_concrete_type(a)
            if kind == "scalar" and _norm_scalar(a) == caller_type:
                n += 1
            elif kind == "complex" and _complex_real_component(a) == caller_type:
                n += 1
        return n

    if not insts:
        return None
    return max(insts, key=caller_hits)


def _complex_real_component(spelling: str) -> str:
    """Real component type of a ``complex<T>`` spelling (``Kokkos::complex<double>`` →
    ``double``); ``""`` when it has no template argument."""
    inner, _ = _balanced_angle(spelling, spelling.find("<")) if "<" in spelling else (None, 0)
    if not inner:
        return ""
    return inner.split(",", 1)[0].strip().rsplit("::", 1)[-1]


def _norm_scalar(spelling: str) -> str:
    return spelling.strip().rsplit("::", 1)[-1].split("<", 1)[0].strip()


def _gather_texts(roots: tuple[str, ...]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    count = 0
    for root in roots:
        rp = Path(root)
        if not rp.exists():
            continue
        paths = [rp] if rp.is_file() else sorted(rp.rglob("*"))
        for p in paths:
            if count >= _MAX_SOURCE_FILES:
                break
            if not p.is_file() or p.suffix.lower() not in _SOURCE_EXTS:
                continue
            try:
                if p.stat().st_size > _MAX_SOURCE_BYTES:
                    continue
                out.append((p.name, p.read_text(encoding="utf-8", errors="ignore")))
                count += 1
            except OSError:
                continue
    return out


def _template_defs(text: str) -> dict[str, list[str]]:
    """``{def_name: [type-param names]}`` for ``template<...> ... name`` defs in text.

    Only type parameters (``class`` / ``typename``) are captured, in order; a non-type
    (value) template parameter is skipped so positional zip against concrete *type*
    args stays aligned for the common all-type-parameter kernels.  A def whose head has
    no type parameters, or no identifiable name, is ignored.
    """
    out: dict[str, list[str]] = {}
    for m in _TEMPLATE_HEAD_RE.finditer(text):
        clause, after = _balanced_angle(text, m.end() - 1)
        if clause is None:
            continue
        params = _PARAM_RE.findall(clause)
        if not params:
            continue
        name = _def_name_after(text, after)
        if name:
            out.setdefault(name, params)
    return out


def _def_name_after(text: str, pos: int) -> str | None:
    """First identifier after a template head that leads a ``(``/``<``/``{`` — the
    templated function or class name.  Skips return-type / specifier tokens."""
    tail = text[pos:pos + 400]
    # the def name is the identifier immediately preceding the first '(' (function)
    # or '{'/':' (class) — take the last identifier before that delimiter.
    m = re.search(r"([A-Za-z_]\w*)\s*(?:\(|<|\{)", tail)
    # prefer a function form `name(` — find the identifier right before the first '('
    mp = re.search(r"([A-Za-z_]\w*)\s*\(", tail)
    if mp:
        return mp.group(1)
    return m.group(1) if m else None


def _instantiations(text: str, name: str, arity: int) -> list[list[str]]:
    """Concrete argument lists for ``name< ... >`` occurrences (top-level args)."""
    res: list[list[str]] = []
    # ``(?<![\w])`` allows a namespace qualifier (``ql_app::run_app<...>`` — the char
    # before ``run_app`` is ``:``, not a word char) while still rejecting ``xrun_app``.
    for m in re.finditer(r"(?<![\w])" + re.escape(name) + r"\s*<", text):
        args = _split_angle_args(text, m.end() - 1)
        if args and any(_looks_concrete(a) for a in args):
            res.append(args)
    return res


def _looks_concrete(arg: str) -> bool:
    """A concrete type arg has a namespace/template/builtin shape, not a bare 1-char
    template placeholder like ``T`` (heuristic: contains ``::`` or ``<`` or is a known
    builtin or is multi-word)."""
    a = arg.strip()
    if "::" in a or "<" in a or " " in a:
        return True
    return _norm_scalar(a) in ("double", "float", "int", "long", "bool", "char")


def _balanced_angle(text: str, open_idx: int) -> tuple[str | None, int]:
    """Given ``text[open_idx] == '<'`` return ``(inner_text, index_past_close)``."""
    if open_idx >= len(text) or text[open_idx] != "<":
        return None, open_idx
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        c = text[i]
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:i], i + 1
        i += 1
    return None, open_idx


def _split_angle_args(text: str, open_idx: int) -> list[str]:
    """Top-level comma-split of a ``< ... >`` argument list at ``text[open_idx]``."""
    inner, _ = _balanced_angle(text, open_idx)
    if inner is None:
        return []
    args: list[str] = []
    depth = 0
    cur = ""
    for c in inner:
        if c in "<([{":
            depth += 1
            cur += c
        elif c in ">)]}":
            depth -= 1
            cur += c
        elif c == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += c
    if cur.strip():
        args.append(cur.strip())
    return args
