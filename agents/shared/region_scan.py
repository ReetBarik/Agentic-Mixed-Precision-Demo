"""Region-local *write* extraction (Fix C) — the source-scan companion to the
characterizer's region-local *reads*.

Motivation
----------
The characterizer report's ``region_local_vars`` is a region's *reads* (the named
source vars its ops consume as leaf operands).  The ff/dd boundary patch also
needs the region's *writes* — the tracked-typed locals the region *computes*,
which downstream code reads and which the boundary patch must demote to the
caller's precision on region exit.  As flagged in HANDOFF.md ("Where journal data
was insufficient"), the write set is **not recoverable from the tracked-datatype
journal**: ``LogRecord`` has no LHS field, ``track()`` emits no record, and
``prov_vars`` entries carry no scope.

So we recover writes the only way they survive: from **source**.  This module is
a deterministic, source-only, runtime-free scanner.  It mirrors the Patcher's
P3a strategy exactly:

* **Preferred backend: libclang** (imported lazily).  Precise: it resolves
  ``tracked_type<T>`` VAR_DECLs and overloaded ``operator=`` re-assignments off
  the real AST.
* **Fallback: a comment/string/char-literal-aware keyword-token lexer** for when
  the libclang bindings are absent (as they were on the P3a cluster image).  Same
  corruption-safety guarantee for the constrained subset of C++ that appears in
  numerical kernels — it does not attempt full C++.

A third, discovered-in-practice trigger for the fallback: libclang bindings can
be *present yet unable to resolve the tracked type* when a single file is parsed
without its include context (the type's header is missing).  clang then
mis-recovers ``Tracked<double> a = …`` as ``int a`` and reports zero writes.  We
guard against that: an *empty* libclang result over a region whose text still
contains ``tracked_type<`` is treated as a resolution failure and handed to the
include-free lexer.  See HANDOFF.md for the full note.

Semantics
---------
The return value is a **write set**: each written local appears **once**, in
source-textual order of its first write within the region.  (A ``Tracked<double>
a = …;`` followed by a later ``a = …;`` yields ``["a"]``, not ``["a", "a"]``.)
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

# Identifier characters (same alphabet the Patcher's P3a lexer uses).
_IDENT_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")

# Assignment operators (plain + compound).  ``==`` is deliberately excluded so a
# comparison is never mistaken for a write.
_ASSIGN_OPS = {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}

# Multi-char operator tokens the fallback lexer must emit whole, so ``=`` is never
# confused with ``==`` / ``<=`` etc. (longest-match order matters).
_MULTI_OPS = ("<<=", ">>=", "==", "!=", "<=", ">=", "&&", "||", "::",
              "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<", ">>")

# Discrete/integer type spellings whose locals are never promoted to an extended
# float scalar (Rule 1 — an ``int`` index or count must stay integral; promoting
# ``i`` in ``res(i,0)`` would corrupt the subscript).  Same alphabet the boundary
# transform's ``_INT_TYPES`` keys off.
_INT_TYPE_TOKENS = frozenset({
    "int", "bool", "char", "short", "long", "unsigned", "signed", "size_t",
    "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t", "ptrdiff_t", "void",
})

# Container / aggregate core type spellings whose locals are never a *scalar* read
# (a ``Kokkos::View`` / ``Kokkos::Array`` is an output view or coefficient buffer,
# not a value the boundary transform can wrap in ``ext(x)``).  Matched on the
# last ``::`` segment of the core type, so ``Kokkos::View`` matches ``View``.
_AGGREGATE_TYPE_TOKENS = frozenset({"View", "Array"})

# Type qualifiers / storage specifiers stripped when reducing a declaration's type
# tokens to its core type spelling.
_TYPE_QUALIFIERS = frozenset({"const", "volatile", "constexpr", "static",
                              "inline", "register", "mutable", "typename"})


class RegionScanError(RuntimeError):
    """Raised when the region source cannot be read at the requested SHA."""


def extract_region_writes(
    file: str,
    line_start: int,
    line_end: int,
    working_tree: str,            # SHA; resolves via ``git show <sha>:<file>``
    tracked_type: str = "Tracked",   # parameterized for future scalar types
) -> list[str]:
    """Region-local write set: LHS names of ``tracked_type<T>`` assignments in
    the inclusive 1-based line range ``[line_start, line_end]``.

    Deterministic, source-only, no runtime dependency.  Prefers libclang; falls
    back to a keyword-token lexer when the bindings are absent or cannot resolve
    the tracked type.
    """
    src = _read_at_sha(file, working_tree)
    names = _extract_libclang(src, file, line_start, line_end, tracked_type)
    if names is not None:
        return names
    return _extract_fallback(src, line_start, line_end, tracked_type)


# --------------------------------------------------------------------------- #
# region *read* derivation (Phase 2c — source-derived promotion payload)
# --------------------------------------------------------------------------- #

def region_reads_from_function(func_source: str, func_line_start: int,
                               region_start: int, region_end: int) -> list[str]:
    """Scalar-typed reads a region consumes, derived from its enclosing function.

    Phase 2c fix for the *empty promotion payload*: the characterizer emits
    ``region_local_vars=[]`` (or provenance-indexed junk) for qcdloop's template
    regions, so :func:`agents.integrator_base.boundary.promote_region_block` had no
    reads to retype and returned the region verbatim (a bit-identical no-op).  This
    recovers a usable reads set the only way it survives for a template region:
    from **source**.

    The reads set is the region's *scalar value reads* — identifiers appearing in
    ``[region_start, region_end]`` (absolute 1-based file lines) that are declared
    as function-scope **params or locals of a scalar (floating / template) type**.
    It is deliberately conservative: it excludes

    * integer indices/counts (``const int i`` — promoting ``i`` would corrupt a
      ``res(i,0)`` subscript) and other discrete-typed locals;
    * aggregate containers (``Kokkos::View`` / ``Kokkos::Array`` — a view/buffer is
      not an ``ext(x)``-wrappable scalar);
    * type names, namespace qualifiers, member fields, and call targets (these are
      never declared *names*, so they never enter the scalar-name universe);

    while including the floating/template locals the promotion actually needs
    (``si``, ``ta``, ``fac``, ``lnrat_*``, ``mu2``, ``scalefac2``, …).

    ``func_source`` is the enclosing function's source text (the lines
    ``func_line_start..func_end`` of the file); ``func_line_start`` is that first
    line's 1-based file number, used to map the absolute region range onto
    ``func_source``.  Deterministic, source-only, no runtime or type-resolution
    dependency — a pure token scan (reads, unlike writes, need no AST type
    resolution, and libclang cannot resolve a template region's types without its
    include context anyway; see the module docstring).

    A read that turns out to be region-*local* (declared inside the region) or a
    pre-declared *write* is harmlessly re-classified by ``_compute_promotion``'s
    dataflow — over-inclusion here costs nothing, so the scan errs toward the
    superset of scalar names and lets the boundary transform partition them.
    """
    toks = _tokenize(func_source)                 # ``.line`` is 1-based in func_source
    universe = _scalar_name_universe(toks)
    rs = region_start - func_line_start + 1
    re_ = region_end - func_line_start + 1

    # Phase 2d (d1): names used as an array subscript BASE anywhere in the region
    # (``name[``) are aggregates/pointers, not promotable scalars — promoting one
    # yields ``ffloat[int]`` / ``operator[](ffloat,int)`` build failures (the
    # xpi_in-style Kokkos::Array reads).  Exclude them from the derived reads.
    subscripted = _subscripted_names(toks, rs, re_)

    reads: list[str] = []
    seen: set[str] = set()
    n = len(toks)
    for idx, t in enumerate(toks):
        if not (rs <= t.line <= re_):
            continue
        if not _is_ident_tok(t.text) or t.text not in universe or t.text in seen:
            continue
        if t.text in subscripted:
            continue
        # A bare ``name =`` at this position is a pure (non-compound) write target,
        # not a read; ``_compute_promotion`` seeds it from Fix-C's write set.  A
        # compound ``name +=`` reads too, so only plain ``=`` is excluded (the
        # tokenizer emits ``+=`` etc. whole, so this never mistakes one for ``=``).
        nxt = toks[idx + 1].text if idx + 1 < n else ""
        if nxt == "=":
            continue
        seen.add(t.text)
        reads.append(t.text)
    return reads


def _subscripted_names(toks: list["_Tok"], rs: int, re_: int) -> set[str]:
    """Identifiers used as an array-subscript base (``name [``) within the region."""
    out: set[str] = set()
    n = len(toks)
    for i in range(n - 1):
        t = toks[i]
        if rs <= t.line <= re_ and _is_ident_tok(t.text) and toks[i + 1].text == "[":
            out.add(t.text)
    return out


def region_complex_read_names(func_source: str, complex_tokens) -> set[str]:
    """Names (params or body locals) declared with a complex-container type.

    Phase 2d: the boundary transform promotes a complex-typed read to the extended
    *complex* container (``ffcomplex`` / ``ddcomplex``) rather than the scalar.  A read
    is complex when its core declared type token is in ``complex_tokens`` — the
    complex-bound template-parameter names (``TOutput``) plus the literal ``complex``
    (from :func:`agents.shared.type_resolve.complex_type_tokens`).  Pure token scan
    over the enclosing function, consistent with :func:`region_reads_from_function`.
    """
    complex_tokens = set(complex_tokens)
    return {n for n, core in name_core_types(func_source).items()
            if core in complex_tokens}


def name_core_types(func_source: str) -> dict[str, str]:
    """Map each param / body-local name to its *core* declared type token.

    Core = the outermost type name (last ``::`` segment, before any ``<`` / ``&`` /
    qualifier) — so ``const TOutput fac`` → ``fac: TOutput`` and
    ``Kokkos::complex<double> z = …`` → ``z: complex``.  Mirrors the universe scan in
    :func:`_scalar_name_universe`; used to classify a read as complex vs scalar."""
    toks = _tokenize(func_source)
    out: dict[str, str] = {}
    out.update(_param_name_core_types(toks))
    out.update(_local_name_core_types(toks))
    return out


def _param_name_core_types(toks: list["_Tok"]) -> dict[str, str]:
    """{param name: core type token} for the function's parameter list."""
    n = len(toks)
    open_idx = next((k for k in range(n) if toks[k].text == "("), None)
    if open_idx is None:
        return {}
    close_idx = _match_paren(toks, open_idx)
    if close_idx is None:
        return {}
    out: dict[str, str] = {}
    depth = angle = 0
    clause: list[_Tok] = []

    def flush(cl: list["_Tok"]) -> None:
        nm = _param_name(cl)
        if nm:
            core = _core_type_name([t.text for t in cl if t.text != nm])
            if core:
                out[nm] = core

    for k in range(open_idx + 1, close_idx):
        tx = toks[k].text
        if tx in "([{":
            depth += 1; clause.append(toks[k]); continue
        if tx in ")]}":
            depth -= 1; clause.append(toks[k]); continue
        if tx == "<":
            angle += 1; clause.append(toks[k]); continue
        if tx == ">":
            angle = max(0, angle - 1); clause.append(toks[k]); continue
        if tx == ">>":
            angle = max(0, angle - 2); clause.append(toks[k]); continue
        if tx == "," and depth == 0 and angle == 0:
            flush(clause); clause = []; continue
        clause.append(toks[k])
    flush(clause)
    return out


def _local_name_core_types(toks: list["_Tok"]) -> dict[str, str]:
    """{local name: core type token} for ``<type> <name> = <init>`` body decls."""
    out: dict[str, str] = {}
    n = len(toks)
    for i in range(n - 2):
        type_tok, name_tok, eq_tok = toks[i], toks[i + 1], toks[i + 2]
        if not (_is_ident_tok(type_tok.text) and _is_ident_tok(name_tok.text)
                and eq_tok.text == "="):
            continue
        prev = toks[i - 1].text if i >= 1 else ""
        if prev in (".", "::", "->"):
            continue
        core = _core_type_name([type_tok.text])
        if core:
            out.setdefault(name_tok.text, core)
    return out


def _core_type_name(type_toks: list[str]) -> str:
    """Core (outermost) type name of a declaration's type tokens, or ``""``.

    The leading ``ident(::ident)*`` run after skipping qualifiers, taken up to the
    first ``<`` / ``&`` / trailing qualifier, reduced to its last ``::`` segment —
    ``Kokkos::complex<double>`` → ``complex``, ``const TOutput`` → ``TOutput``.  A
    pointer/C-array type yields ``""`` (never a scalar/complex operand)."""
    if any(tx in ("*", "[", "]") for tx in type_toks):
        return ""
    name_parts: list[str] = []
    started = False
    for tx in type_toks:
        if tx in _TYPE_QUALIFIERS:
            if started:
                break
            continue
        if tx == "::":
            if started:
                name_parts.append(tx)
            continue
        if _is_ident_tok(tx):
            name_parts.append(tx)
            started = True
            continue
        break
    if not name_parts:
        return ""
    return "".join(name_parts).rsplit("::", 1)[-1]


def _scalar_name_universe(toks: list["_Tok"]) -> set[str]:
    """Names declared (as params or body locals) with a scalar (float/template) type.

    Params are the identifiers in the function's parameter list; body locals are the
    ``<type> <name> = <init>`` statement-level declarations.  Both are classified by
    their core type spelling (:func:`_core_type_is_scalar`)."""
    names: set[str] = set()
    names |= _param_scalar_names(toks)
    names |= _local_decl_scalar_names(toks)
    return names


def _param_scalar_names(toks: list["_Tok"]) -> set[str]:
    """Scalar-typed parameter names from the function's parameter list.

    The parameter list is the first ``(...)`` in ``toks`` (a template ``<...>`` or a
    bare ``KOKKOS_INLINE_FUNCTION`` macro carries no ``(``, so the first ``(`` opens
    the params).  Each top-level ``,``-separated clause's *name* is its last
    identifier (skipping a trailing ``[...]``); the tokens before the name are its
    type."""
    n = len(toks)
    open_idx = next((k for k in range(n) if toks[k].text == "("), None)
    if open_idx is None:
        return set()
    close_idx = _match_paren(toks, open_idx)
    if close_idx is None:
        return set()

    names: set[str] = set()
    depth = 0          # () [] {} nesting
    angle = 0          # <> template-argument nesting (params are type contexts, so
                       # ``<`` opens a template arg list, never a comparison)
    clause: list[_Tok] = []

    def flush(cl: list["_Tok"]) -> None:
        if _param_is_scalar(cl):
            nm = _param_name(cl)
            if nm:
                names.add(nm)

    for k in range(open_idx + 1, close_idx):
        tx = toks[k].text
        if tx in "([{":
            depth += 1; clause.append(toks[k]); continue
        if tx in ")]}":
            depth -= 1; clause.append(toks[k]); continue
        if tx == "<":
            angle += 1; clause.append(toks[k]); continue
        if tx == ">":
            angle = max(0, angle - 1); clause.append(toks[k]); continue
        if tx == ">>":
            angle = max(0, angle - 2); clause.append(toks[k]); continue
        if tx == "," and depth == 0 and angle == 0:
            flush(clause); clause = []; continue
        clause.append(toks[k])
    flush(clause)
    return names


def _param_name(clause: list["_Tok"]) -> str | None:
    """The parameter name: the last identifier before any trailing ``[...]``."""
    end = len(clause)
    # drop a trailing array extent ``[...]`` (e.g. ``T (&a)[4]``) if present
    while end > 0 and clause[end - 1].text == "]":
        depth = 0
        j = end - 1
        while j >= 0:
            if clause[j].text == "]":
                depth += 1
            elif clause[j].text == "[":
                depth -= 1
                if depth == 0:
                    break
            j -= 1
        end = j
    for j in range(end - 1, -1, -1):
        if _is_ident_tok(clause[j].text):
            return clause[j].text
    return None


def _param_is_scalar(clause: list["_Tok"]) -> bool:
    """A parameter is a scalar read source iff its *type* is scalar (non-int,
    non-aggregate, non-pointer)."""
    name = _param_name(clause)
    type_toks = [t.text for t in clause if t.text != name]
    return _core_type_is_scalar(type_toks)


def _local_decl_scalar_names(toks: list["_Tok"]) -> set[str]:
    """Scalar-typed body-local names from ``<type> <name> = <init>`` declarations.

    Mirrors the boundary transform's decl detection (two adjacent identifiers then
    ``=``) but keyed on the *type token* (the identifier before the name) being a
    scalar spelling — so ``const TMass si = …`` adds ``si`` while ``int massive =
    0`` and ``Kokkos::Array<…> xpi = …`` are skipped."""
    names: set[str] = set()
    n = len(toks)
    for i in range(n - 2):
        type_tok, name_tok, eq_tok = toks[i], toks[i + 1], toks[i + 2]
        if not (_is_ident_tok(type_tok.text) and _is_ident_tok(name_tok.text)
                and eq_tok.text == "="):
            continue
        # ``a = b`` re-assignment (``name_tok`` is the RHS lead, not a decl name):
        # a decl needs the type token to be a type, so a preceding ``.``/``::``/``(``
        # (member/qualified/call context) disqualifies it.
        prev = toks[i - 1].text if i >= 1 else ""
        if prev in (".", "::") or prev == "->":
            continue
        if _core_type_is_scalar([type_tok.text]):
            names.add(name_tok.text)
    return names


def _core_type_is_scalar(type_toks: list[str]) -> bool:
    """True if a declaration's type tokens denote a scalar (float/template) type.

    The *core* type is the leading ``ident(::ident)*`` run (after skipping leading
    qualifiers), taken up to the first ``<`` / ``&`` / trailing qualifier — i.e. the
    **outermost** type name, not an inner template argument, so
    ``Kokkos::Array<...TMass...>`` resolves to ``Array`` (aggregate), not ``TMass``.
    The last ``::`` segment of the core is matched against the integer and aggregate
    blocklists.  A pointer (``*``) or C-array (``[``) anywhere makes it non-scalar.
    What remains — ``double`` / ``float`` or a template type parameter (``TMass`` /
    ``TOutput`` / ``T``) — is scalar."""
    if any(tx in ("*", "[", "]") for tx in type_toks):
        return False                           # pointer / C-array → not a scalar
    name_parts: list[str] = []
    started = False
    for tx in type_toks:
        if tx in _TYPE_QUALIFIERS:
            if started:
                break                          # trailing qualifier ends the name
            continue
        if tx == "::":
            if started:
                name_parts.append(tx)
            continue
        if _is_ident_tok(tx):
            name_parts.append(tx)
            started = True
            continue
        break                                  # '<', '&', ',', '>', … end the name
    if not name_parts:
        return False
    core = "".join(name_parts).rsplit("::", 1)[-1]
    if core in _INT_TYPE_TOKENS or core in _AGGREGATE_TYPE_TOKENS:
        return False
    return True


def _match_paren(toks: list["_Tok"], open_idx: int) -> int | None:
    """Index of the ``)`` matching the ``(`` at ``open_idx`` (or ``None``)."""
    depth = 0
    for k in range(open_idx, len(toks)):
        if toks[k].text == "(":
            depth += 1
        elif toks[k].text == ")":
            depth -= 1
            if depth == 0:
                return k
    return None


# --------------------------------------------------------------------------- #
# git source resolution
# --------------------------------------------------------------------------- #

def _git(cwd: str | Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(cwd), *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RegionScanError(
            f"git {' '.join(args)} failed in {cwd}:\n{r.stderr.strip()}")
    return r.stdout


def _read_at_sha(file: str, sha: str) -> str:
    """Return the content of ``file`` at commit ``sha`` via ``git show``.

    An absolute ``file`` is anchored at its own directory and made repo-relative;
    a relative ``file`` is taken as repo-relative and anchored at the cwd (which
    must be inside the repo).
    """
    p = Path(file)
    if p.is_absolute():
        anchor = p.parent
        root = _git(anchor, "rev-parse", "--show-toplevel").strip()
        rel = p.resolve().relative_to(Path(root).resolve()).as_posix()
    else:
        root = _git(Path.cwd(), "rev-parse", "--show-toplevel").strip()
        rel = p.as_posix()
    return _git(root, "show", f"{sha}:{rel}")


# --------------------------------------------------------------------------- #
# libclang backend (preferred)
# --------------------------------------------------------------------------- #

def _import_clang():
    """Import the libclang bindings.

    Isolated so tests can monkeypatch it to raise ``ImportError`` and exercise
    the fallback path deterministically.
    """
    import clang.cindex as cindex   # noqa: PLC0415 (lazy by design)
    return cindex


def _extract_libclang(src: str, file: str, line_start: int, line_end: int,
                      tracked_type: str) -> list[str] | None:
    """libclang extraction, or ``None`` when the caller should fall back.

    Returns ``None`` if the bindings are absent, the shared library cannot be
    loaded, or the parse could not resolve ``tracked_type`` (empty result over a
    region whose text still contains ``tracked_type<``).
    """
    try:
        cindex = _import_clang()
    except ImportError:
        return None
    try:
        index = cindex.Index.create()
    except Exception:   # LibclangError: bindings present, shared lib missing.
        return None

    name = Path(file).name
    tu = index.parse(name, args=["-x", "c++", "-std=c++17"],
                     unsaved_files=[(name, src)])

    writes: list[tuple[int, int, str]] = []   # (line, column, varname)

    def visit(cursor) -> None:
        for ch in cursor.get_children():
            loc = ch.location
            if loc.file and loc.file.name == name and line_start <= loc.line <= line_end:
                nm = _libclang_write_name(ch, tracked_type, cindex)
                if nm is not None:
                    writes.append((loc.line, loc.column, nm))
            visit(ch)

    visit(tu.cursor)
    names = _dedupe_source_order(writes)

    if not names and _region_has_tracked_text(src, line_start, line_end, tracked_type):
        # libclang parsed but resolved nothing where the text clearly declares the
        # tracked type -> unresolved include context.  Let the lexer try.
        return None
    return names


def _libclang_write_name(cursor, tracked_type: str, cindex) -> str | None:
    """Name written by ``cursor`` if it is a tracked-typed decl/assignment."""
    kind = cursor.kind
    if kind == cindex.CursorKind.VAR_DECL:
        if _is_tracked(cursor.type.spelling, tracked_type):
            return cursor.spelling
        return None
    if kind == cindex.CursorKind.CALL_EXPR and _is_assign_operator(cursor.spelling):
        # Overloaded ``operator=`` / ``operator+=`` on a class-based tracked type.
        # The assigned object is the first child.
        kids = list(cursor.get_children())
        if kids:
            return _ref_name_if_tracked(kids[0], tracked_type, cindex)
    return None


def _ref_name_if_tracked(cursor, tracked_type: str, cindex) -> str | None:
    """Unwrap ``UNEXPOSED_EXPR`` wrappers; return the ref name if tracked-typed."""
    while cursor.kind == cindex.CursorKind.UNEXPOSED_EXPR:
        kids = list(cursor.get_children())
        if not kids:
            return None
        cursor = kids[0]
    if cursor.kind == cindex.CursorKind.DECL_REF_EXPR and \
            _is_tracked(cursor.type.spelling, tracked_type):
        return cursor.spelling
    return None


def _is_assign_operator(spelling: str) -> bool:
    """True for ``operator=`` / ``operator+=`` etc., but not ``operator==``."""
    if not spelling.startswith("operator"):
        return False
    return spelling[len("operator"):] in _ASSIGN_OPS


def _is_tracked(spelling: str, tracked_type: str) -> bool:
    """True if a type spelling denotes ``tracked_type<...>`` (cv/ref/namespace
    tolerant): ``const ql::Tracked<double>&`` -> matches ``Tracked``."""
    s = spelling.strip()
    for q in ("const ", "volatile "):
        while s.startswith(q):
            s = s[len(q):].lstrip()
    s = s.rstrip(" &*")
    if "<" not in s:
        return False
    head = s.split("<", 1)[0].strip()
    head = head.rsplit("::", 1)[-1]
    return head == tracked_type


# --------------------------------------------------------------------------- #
# keyword-lexer fallback (no libclang / unresolved type)
# --------------------------------------------------------------------------- #

def _region_has_tracked_text(src: str, line_start: int, line_end: int,
                             tracked_type: str) -> bool:
    """Cheap textual probe: does the region contain ``tracked_type<`` at all?"""
    pat = re.compile(re.escape(tracked_type) + r"\s*<")
    lines = src.splitlines()
    for ln in range(line_start, line_end + 1):
        if 1 <= ln <= len(lines) and pat.search(lines[ln - 1]):
            return True
    return False


class _Tok:
    __slots__ = ("text", "line")

    def __init__(self, text: str, line: int):
        self.text = text
        self.line = line


def _tokenize(text: str) -> list[_Tok]:
    """Code tokens (identifiers + operators) with line numbers, skipping comments
    and string / char literals — the same lexical state machine as the Patcher's
    P3a rewriter (``agents/patcher/edits.py``)."""
    toks: list[_Tok] = []
    i, n = 0, len(text)
    line = 1
    state = "code"   # code | line_comment | block_comment | string | char

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
            if ch == "\n":
                line += 1; i += 1; continue
            if ch in _IDENT_CHARS:
                j = i
                while j < n and text[j] in _IDENT_CHARS:
                    j += 1
                toks.append(_Tok(text[i:j], line))
                i = j
                continue
            if ch.isspace():
                i += 1; continue
            # multi-char operators (longest match first), else single char.
            for op in _MULTI_OPS:
                if text.startswith(op, i):
                    toks.append(_Tok(op, line)); i += len(op); break
            else:
                toks.append(_Tok(ch, line)); i += 1
            continue

        # inside a comment / literal: consume until it closes, tracking lines.
        if ch == "\n":
            line += 1
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


def _extract_fallback(src: str, line_start: int, line_end: int,
                      tracked_type: str) -> list[str]:
    """Include-free textual write extraction over the constrained kernel subset.

    Recognizes two write forms whose LHS type is ``tracked_type<T>``:

    * declaration/init ``tracked_type < ... > name`` (the name after the balanced
      template close), skipping a following ``(`` — a function declaration is not
      a local write; and
    * re-assignment ``name <assign-op> ...`` where ``name`` is a tracked local
      seen declared earlier in the file.

    The tracked-name universe is built from tracked-typed declarations across the
    whole file (a forward pass, so a decl above the region enables detection of an
    in-region re-assignment); only writes whose line falls in the region are
    emitted.
    """
    toks = _tokenize(src)
    tracked_vars: set[str] = set()
    writes: list[tuple[int, int, str]] = []   # (line, tok_index, name)
    n = len(toks)
    i = 0

    while i < n:
        t = toks[i]

        # -- declaration:  tracked_type < ... > name --
        if t.text == tracked_type and i + 1 < n and toks[i + 1].text == "<":
            k = _skip_template(toks, i + 1)   # index of the token after the '>'
            if k is not None and k < n and _is_ident_tok(toks[k].text):
                name_tok = toks[k]
                following = toks[k + 1].text if k + 1 < n else ""
                if following != "(":          # exclude function declarations
                    tracked_vars.add(name_tok.text)
                    if line_start <= name_tok.line <= line_end:
                        writes.append((name_tok.line, k, name_tok.text))
                i = k + 1
                continue

        # -- re-assignment:  name <assign-op> ... --
        if _is_ident_tok(t.text) and t.text in tracked_vars:
            nxt = toks[i + 1].text if i + 1 < n else ""
            if nxt in _ASSIGN_OPS and line_start <= t.line <= line_end:
                writes.append((t.line, i, t.text))

        i += 1

    return _dedupe_source_order(writes)


def _skip_template(toks: list[_Tok], open_idx: int) -> int | None:
    """Given the index of a ``<`` opening a template argument list, return the
    index just past the matching ``>`` (``>>`` closes two levels)."""
    depth = 0
    k = open_idx
    n = len(toks)
    while k < n:
        tx = toks[k].text
        if tx == "<":
            depth += 1
        elif tx == ">":
            depth -= 1
            if depth == 0:
                return k + 1
        elif tx == ">>":
            depth -= 2
            if depth <= 0:
                return k + 1
        k += 1
    return None


def _is_ident_tok(text: str) -> bool:
    return bool(text) and text[0] in _IDENT_CHARS and not text[0].isdigit()


# --------------------------------------------------------------------------- #
# shared
# --------------------------------------------------------------------------- #

def _dedupe_source_order(writes: list[tuple[int, int, str]]) -> list[str]:
    """Set semantics with source-textual order: sort by (line, position), then
    keep each name's first occurrence."""
    seen: set[str] = set()
    ordered: list[str] = []
    for _line, _pos, name in sorted(writes, key=lambda w: (w[0], w[1])):
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered
