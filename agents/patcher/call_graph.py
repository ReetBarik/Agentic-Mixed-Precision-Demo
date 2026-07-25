"""Phase 2a — static call-graph builder for the Patcher fan-out.

Given an integral's entry-point function, a source tree, and the app's include
paths, build the static call graph rooted at that entry point.  The fan-out
(:mod:`agents.patcher.fanout`) uses it to (1) find the function that *contains* a
region intent, and (2) enumerate every caller-path from the entry point down to
that function, so it can generate one variant per path and cascade the renames
upward.  One graph is built per per-integral pass and cached in memory for the
pass's duration (~21 libclang invocations for a full 21-integral run).

Hybrid extraction — why not pure AST
------------------------------------
qcdloop is template-heavy, and its dispatch is a chain of **dependent** template
calls: ``ql::BO`` calls ``ql::B1m<TOutput,TMass,TScale>(...)``, which is a
type-dependent name in an *uninstantiated* primary template.  libclang does **not**
build ``CALL_EXPR`` nodes for such calls (the compound-statement bodies come back
empty), so a pure-AST edge walk silently misses the most important edges — exactly
the "silently missing call edges" failure the design warns about.

So we split responsibilities:

* **libclang owns the authoritative facts** it *does* recover reliably even with a
  broken include context — most function/template **definitions** and their source
  **extents** (verified: ``BO`` @ boxGPU.h:69-143, ``B0m``/``B1``/... @ box/B0m.h).
  These drive region→function resolution.
* **A comment/string-aware token scan over each function's body text** recovers the
  call **edges**: an identifier that (a) names a function in the definition universe
  and (b) is immediately followed by ``(`` or ``<`` (a call or an explicit
  template-id call) is an edge.  This sidesteps the dependent-call AST gap entirely
  and is the same "recover it from source, deterministically" strategy Fix C uses
  for region writes.
* **The same token scan also recovers template-function DEFINITIONS + extents.**
  In a broken-include parse libclang not only misses dependent *calls* — it also
  drops some ``template<...>`` function definitions outright and truncates or
  mislabels others (observed in the template-heavy ``kokkosUtils.h``: ``ddilog`` /
  ``kfn`` dropped entirely; ``Lnrat`` / ``Li2omx2`` truncated / marked non-template).
  A region intent landing inside such a definition would resolve to "no enclosing
  function" and the fan-out / chain-promote would fail.  So a brace-matched scan for
  ``template<...>`` function heads (:func:`_scan_template_funcdefs`) supplies the
  extents libclang drops and *supersedes* the truncated ones it returns for the same
  name+file.  The scan is strictly additive/corrective: genuine non-template defs
  libclang already resolves are left untouched.

If libclang comes up empty on the entry point (bindings missing, shared lib
unloadable, or the parse resolved zero definitions) we raise :class:`CallGraphError`
rather than returning a graph with silently-missing edges — a fail-loud contract the
fan-out relies on.
"""

from __future__ import annotations

import bisect
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# Cursor kinds that are function *definitions* we treat as call-graph nodes.
_FUNC_KIND_NAMES = ("FUNCTION_DECL", "FUNCTION_TEMPLATE", "CXX_METHOD",
                    "FUNCTION_TEMPLATE_SPECIALIZATION")

# Header extensions scanned when auto-detecting the entry-point translation unit.
_HEADER_EXTS = (".h", ".hpp", ".hh", ".hxx", ".cuh", ".inl", ".ipp")

_IDENT_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


class CallGraphError(RuntimeError):
    """The call graph could not be built (fail-loud; never a silent empty graph)."""


@dataclass(frozen=True)
class FuncDef:
    """One function/template *definition* with its source extent (1-based, inclusive)."""

    name: str
    file: str            # absolute path
    line_start: int
    line_end: int
    is_template: bool

    @property
    def basename(self) -> str:
        return Path(self.file).name

    def contains(self, line: int) -> bool:
        return self.line_start <= line <= self.line_end

    @property
    def span(self) -> int:
        return self.line_end - self.line_start


@dataclass
class CallGraph:
    """The static call graph rooted at ``root``.

    ``defs`` maps a function name to all its definitions (overloads / multiple
    template signatures share a name); ``edges`` / ``reverse`` are name→name
    adjacency (forward: caller→callees; reverse: callee→callers) for downward
    enumeration and upward rename cascade.
    """

    root: str
    tu_file: str
    defs: dict[str, list[FuncDef]] = field(default_factory=dict)
    edges: dict[str, set[str]] = field(default_factory=dict)
    reverse: dict[str, set[str]] = field(default_factory=dict)

    # -- lookups ------------------------------------------------------------

    def has(self, name: str) -> bool:
        return name in self.defs

    def callees_of(self, name: str) -> set[str]:
        return set(self.edges.get(name, set()))

    def callers_of(self, name: str) -> set[str]:
        return set(self.reverse.get(name, set()))

    def enclosing_function(self, file: str, line: int) -> FuncDef | None:
        """The definition whose extent contains ``(file, line)`` (innermost wins).

        ``file`` may be a bare basename (characterization region keys are bare —
        ``B0m.h``) or a path; matching is by basename, with the smallest
        containing extent chosen so a helper nested in a larger def resolves to the
        helper.  Returns ``None`` when no definition contains the line.
        """
        want = Path(file).name
        best: FuncDef | None = None
        for fds in self.defs.values():
            for fd in fds:
                if fd.basename == want and fd.contains(line):
                    if best is None or fd.span < best.span:
                        best = fd
        return best

    # -- traversal ----------------------------------------------------------

    def enumerate_paths(self, target: str, *, max_depth: int = 16,
                        max_paths: int = 1024) -> tuple[list[list[str]], bool]:
        """All simple caller-paths from ``root`` to ``target`` (root-first).

        Returns ``(paths, truncated)``.  ``paths`` is a list of node-name lists,
        each starting at ``root`` and ending at ``target``; a ``target == root``
        request yields the single trivial path ``[root]``.  ``truncated`` is True
        if the ``max_paths`` cap was hit (the fan-out logs this — a silent cap
        would understate over-generation).  Cycles are impossible in a static call
        graph of this code, but the simple-path DFS (no repeated node) is
        cycle-safe regardless.
        """
        if target not in self.defs:
            return [], False
        if target == self.root:
            return [[self.root]], False

        paths: list[list[str]] = []
        truncated = False

        def dfs(node: str, trail: list[str]) -> None:
            nonlocal truncated
            if len(paths) >= max_paths:
                truncated = True
                return
            if len(trail) > max_depth:
                return
            for callee in sorted(self.edges.get(node, ())):
                if callee in trail:            # simple-path: skip a revisit
                    continue
                if callee == target:
                    paths.append(trail + [callee])
                    if len(paths) >= max_paths:
                        truncated = True
                        return
                else:
                    dfs(callee, trail + [callee])

        dfs(self.root, [self.root])
        return paths, truncated


# --------------------------------------------------------------------------- #
# builder
# --------------------------------------------------------------------------- #

def _import_clang():
    """Lazy libclang import (isolated so tests can force the missing-bindings path)."""
    import clang.cindex as cindex   # noqa: PLC0415
    return cindex


@lru_cache(maxsize=1)
def _builtin_include_args() -> tuple[str, ...]:
    """Best-effort clang builtin-header include (silences a spurious ``stddef.h``
    diagnostic).  Never fatal: if no ``clang`` is on PATH we return no args and the
    parse proceeds — the missing builtin does not affect definition/extent recovery.
    """
    try:
        r = subprocess.run(["clang", "-print-resource-dir"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ()
    rd = r.stdout.strip()
    inc = Path(rd) / "include" if rd else None
    return (f"-isystem{inc}",) if inc and inc.is_dir() else ()


def _detect_tu_file(root: str, tree: Path) -> Path:
    """Pick the translation unit to parse when the caller gives none.

    Heuristic: among tree headers that textually *define* ``root`` (``root`` followed
    by ``(`` on a non-``#`` line), prefer the one with the most ``#include``
    directives — the "meta-header" that transitively pulls in the callees (qcdloop's
    ``boxGPU.h``).  Deterministic (ties broken by path) so a re-run is byte-stable.
    """
    candidates: list[tuple[int, str]] = []
    for p in sorted(tree.rglob("*")):
        if p.suffix.lower() not in _HEADER_EXTS or not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        defines_root = any(
            root in ln and f"{root}(" in ln.replace(" ", "") and not ln.lstrip().startswith("#")
            for ln in text.splitlines())
        # a looser definition probe: `root` as a whole word followed by `(` anywhere
        if not defines_root:
            defines_root = _defines_symbol(text, root)
        if defines_root:
            n_inc = sum(1 for ln in text.splitlines() if ln.lstrip().startswith("#include"))
            candidates.append((n_inc, str(p)))
    if not candidates:
        raise CallGraphError(
            f"no header under {tree} defines entry point {root!r}; pass tu_file explicitly")
    candidates.sort(key=lambda c: (-c[0], c[1]))
    return Path(candidates[0][1])


def _defines_symbol(text: str, name: str) -> bool:
    """Whole-word ``name`` immediately followed by ``(`` somewhere in ``text``."""
    toks = _scan_idents(text)
    return any(t == name and nxt == "(" for t, nxt in toks)


def build_call_graph(
    root: str,
    tree_path: str | Path,
    *,
    tu_file: str | Path | None = None,
    include_paths: list[str] | None = None,
    extra_args: list[str] | None = None,
) -> CallGraph:
    """Build the static call graph rooted at ``root`` over the tree at ``tree_path``.

    ``tu_file`` is the header to parse as the translation unit (defaults to a
    detected meta-header that defines ``root`` — see :func:`_detect_tu_file`).
    ``include_paths`` are extra ``-I`` dirs (the tree root and its ``box/`` subdir
    are always added); ``extra_args`` are appended verbatim to the clang args.

    Raises :class:`CallGraphError` if the bindings are absent, the shared lib
    cannot load, the entry point is not found as a definition, or the parse
    resolved zero definitions — never returns a silently-empty graph.
    """
    tree = Path(tree_path).resolve()
    if not tree.is_dir():
        raise CallGraphError(f"tree_path is not a directory: {tree}")

    tu = Path(tu_file).resolve() if tu_file else _detect_tu_file(root, tree)
    if not tu.is_file():
        raise CallGraphError(f"translation-unit file not found: {tu}")

    try:
        cindex = _import_clang()
    except ImportError as exc:
        raise CallGraphError(
            f"libclang bindings unavailable ({exc}); cannot build call graph") from exc
    try:
        index = cindex.Index.create()
    except Exception as exc:   # LibclangError: bindings present, shared lib missing
        raise CallGraphError(f"libclang shared library could not load: {exc}") from exc

    inc_dirs = [str(tree), str(tree / "box")]
    for extra in include_paths or []:
        inc_dirs.append(str(extra))
    args = ["-x", "c++", "-std=c++17", "-ferror-limit=0"]
    args += [f"-I{d}" for d in inc_dirs]
    args += list(_builtin_include_args())
    args += list(extra_args or [])

    try:
        translation_unit = index.parse(str(tu), args=args)
    except Exception as exc:   # noqa: BLE001 - any parse crash is a build failure
        raise CallGraphError(f"libclang failed to parse {tu}: {exc}") from exc

    defs = _collect_defs(translation_unit, cindex, tree)
    _merge_template_funcdefs(defs, tree)
    if not defs:
        # A real qcdloop header parses to >1000 defs; zero means a broken parse
        # (bad include context) — fail loud rather than emit an edgeless graph.
        raise CallGraphError(
            f"libclang resolved zero function definitions in {tu} — likely a broken "
            f"include context (args={args!r}); refusing to return an empty call graph")
    if root not in defs:
        raise CallGraphError(
            f"entry point {root!r} not found as a definition in {tu} "
            f"({len(defs)} defs found); check the entry-point name / tu_file")

    edges, reverse = _extract_edges(defs)
    return CallGraph(root=root, tu_file=str(tu), defs=defs, edges=edges, reverse=reverse)


# --------------------------------------------------------------------------- #
# definition collection (libclang) + edge extraction (token scan)
# --------------------------------------------------------------------------- #

def _collect_defs(tu, cindex, tree: Path) -> dict[str, list[FuncDef]]:
    """All function/template/method *definitions* under ``tree`` (name → defs)."""
    func_kinds = {getattr(cindex.CursorKind, k)
                  for k in _FUNC_KIND_NAMES if hasattr(cindex.CursorKind, k)}
    template_kind = getattr(cindex.CursorKind, "FUNCTION_TEMPLATE", None)
    tree_str = str(tree)
    out: dict[str, list[FuncDef]] = {}
    seen: set[tuple[str, str, int]] = set()

    def visit(cursor) -> None:
        for ch in cursor.get_children():
            if ch.kind in func_kinds and ch.is_definition():
                loc = ch.location
                if loc.file:
                    fpath = str(Path(loc.file.name).resolve())
                    if fpath.startswith(tree_str):
                        key = (ch.spelling, fpath, ch.extent.start.line)
                        if ch.spelling and key not in seen:
                            seen.add(key)
                            out.setdefault(ch.spelling, []).append(FuncDef(
                                name=ch.spelling, file=fpath,
                                line_start=ch.extent.start.line,
                                line_end=ch.extent.end.line,
                                is_template=(ch.kind == template_kind)))
            visit(ch)

    visit(tu.cursor)
    return out


# --------------------------------------------------------------------------- #
# template-function extent recovery (token/brace scan — symmetric to edges)
# --------------------------------------------------------------------------- #

# First word after a ``template<...>`` clause that means "this is NOT a function
# template" — a class/alias/nested-template/variable template.  We skip these but
# keep scanning INTO their bodies (so member function templates are still found).
_NON_FUNC_TEMPLATE_HEADS = {
    "class", "struct", "union", "enum", "using", "concept", "template",
}


def _mask_noncode(text: str) -> str:
    """Blank every comment/string/char-literal char with a space (offsets + newlines
    preserved) so structural brace/angle/paren matching sees only code.

    Same lexical discipline as :func:`_scan_idents` / the boundary scanner: a brace
    or angle bracket inside a comment or string never perturbs the match.
    """
    out = list(text)
    i, n = 0, len(text)
    state = "code"
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                out[i] = out[i + 1] = " "; state = "line_comment"; i += 2; continue
            if ch == "/" and nxt == "*":
                out[i] = out[i + 1] = " "; state = "block_comment"; i += 2; continue
            if ch == '"':
                out[i] = " "; state = "string"; i += 1; continue
            if ch == "'":
                out[i] = " "; state = "char"; i += 1; continue
            i += 1
        elif state == "line_comment":
            if ch == "\n":
                state = "code"
            else:
                out[i] = " "
            i += 1
        elif state == "block_comment":
            if ch == "*" and nxt == "/":
                out[i] = out[i + 1] = " "; state = "code"; i += 2
            else:
                if ch != "\n":
                    out[i] = " "
                i += 1
        elif state == "string":
            if ch == "\\" and nxt:
                out[i] = out[i + 1] = " "; i += 2
            elif ch == '"':
                out[i] = " "; state = "code"; i += 1
            else:
                if ch != "\n":
                    out[i] = " "
                i += 1
        elif state == "char":
            if ch == "\\" and nxt:
                out[i] = out[i + 1] = " "; i += 2
            elif ch == "'":
                out[i] = " "; state = "code"; i += 1
            else:
                if ch != "\n":
                    out[i] = " "
                i += 1
    return "".join(out)


def _match_delim(s: str, k: int, opench: str, closech: str) -> int:
    """Index of the ``closech`` matching the ``opench`` at ``s[k]``; -1 if unbalanced.

    Char-level so ``>>`` closing two ``<`` levels (C++ nested template-ids) matches
    correctly.  ``s`` must be the masked (code-only) text.
    """
    depth = 0
    n = len(s)
    while k < n:
        c = s[k]
        if c == opench:
            depth += 1
        elif c == closech:
            depth -= 1
            if depth == 0:
                return k
        k += 1
    return -1


def _parse_one_template(masked: str, tpl_start: int, after_kw: int, n: int,
                        line_of) -> tuple[tuple[str, int, int] | None, int]:
    """Parse one ``template<...>`` head at ``tpl_start``.

    Returns ``((name, line_start, line_end), continue_off)`` for a function-template
    *definition*, or ``(None, continue_off)`` when the head is a declaration, a
    class/alias/variable template, or malformed.  ``continue_off`` is where the outer
    scan resumes (past the body for a def; just past the keyword otherwise, so member
    templates inside a skipped class template are still discovered).
    """
    # 1. the template parameter clause '<...>'
    k = after_kw
    while k < n and masked[k].isspace():
        k += 1
    if k >= n or masked[k] != "<":
        return None, after_kw
    clause_end = _match_delim(masked, k, "<", ">")
    if clause_end < 0:
        return None, after_kw
    p = clause_end + 1

    # 2. walk the declarator up to the parameter-list '(' at paren-depth 0,
    #    remembering the identifier immediately before it (the function name).
    last_ident: str | None = None
    first_word = True
    while p < n:
        c = masked[p]
        if c.isspace():
            p += 1
            continue
        if c in _IDENT_CHARS and not c.isdigit():
            j = p
            while j < n and masked[j] in _IDENT_CHARS:
                j += 1
            word = masked[p:j]
            if first_word and word in _NON_FUNC_TEMPLATE_HEADS:
                return None, after_kw          # not a function template
            first_word = False
            last_ident = word
            p = j
            continue
        if c == "<":                            # a return-type template-id: skip it
            close = _match_delim(masked, p, "<", ">")
            if close < 0:
                return None, after_kw
            p = close + 1
            continue
        if c == "(":                            # parameter list opens
            break
        if c in ";{=":                          # decl / variable template / no body
            return None, p + 1
        p += 1
    if p >= n or masked[p] != "(" or not last_ident:
        return None, after_kw

    # 3. skip the parameter list, then find the body '{' (definition) or ';' (decl).
    rparen = _match_delim(masked, p, "(", ")")
    if rparen < 0:
        return None, after_kw
    q = rparen + 1
    while q < n and masked[q] not in "{;":
        q += 1
    if q >= n or masked[q] == ";":              # declaration, not a definition
        return None, (q + 1 if q < n else after_kw)
    close_brace = _match_delim(masked, q, "{", "}")
    if close_brace < 0:
        return None, after_kw
    rec = (last_ident, line_of(tpl_start), line_of(close_brace))
    return rec, close_brace + 1


def _scan_template_funcdefs(text: str) -> list[tuple[str, int, int]]:
    """Recover ``template<...>`` function *definitions* as ``(name, ls, le)`` (1-based,
    inclusive) via a masked brace scan — the extents libclang drops or truncates in a
    broken-include parse.  Non-template defs are never emitted (strictly a template
    supplement).
    """
    masked = _mask_noncode(text)
    n = len(masked)
    line_starts = [0]
    for i, ch in enumerate(masked):
        if ch == "\n":
            line_starts.append(i + 1)

    def line_of(off: int) -> int:
        return bisect.bisect_right(line_starts, off)

    out: list[tuple[str, int, int]] = []
    i = 0
    while i < n:
        ch = masked[i]
        if ch in _IDENT_CHARS and not ch.isdigit():
            j = i
            while j < n and masked[j] in _IDENT_CHARS:
                j += 1
            if masked[i:j] == "template":
                rec, cont = _parse_one_template(masked, i, j, n, line_of)
                if rec is not None:
                    out.append(rec)
                i = cont if cont > j else j
                continue
            i = j
            continue
        i += 1
    return out


def _install_def(defs: dict[str, list[FuncDef]], fd: FuncDef) -> None:
    """Add ``fd``, superseding any same-name+same-file def whose extent OVERLAPS it.

    An overlapping libclang entry is the truncated / mislabeled one the scan corrects;
    non-overlapping siblings (other overloads) are preserved.  Idempotent: a re-scan
    of the same extent replaces its own prior entry rather than duplicating it.
    """
    lst = defs.setdefault(fd.name, [])
    kept = [ex for ex in lst
            if not (ex.file == fd.file
                    and not (ex.line_end < fd.line_start or ex.line_start > fd.line_end))]
    kept.append(fd)
    defs[fd.name] = kept


def _merge_template_funcdefs(defs: dict[str, list[FuncDef]], tree: Path) -> None:
    """Supplement ``defs`` with template-function extents recovered by the brace scan.

    Scans only files libclang already produced ≥1 def for (the TU's include closure)
    — bounded, and avoids inventing defs for functions outside the parsed unit.
    Mutates ``defs`` in place.
    """
    files = sorted({fd.file for fds in defs.values() for fd in fds})
    for fpath in files:
        try:
            text = Path(fpath).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name, ls, le in _scan_template_funcdefs(text):
            _install_def(defs, FuncDef(name=name, file=fpath,
                                       line_start=ls, line_end=le, is_template=True))


def _extract_edges(defs: dict[str, list[FuncDef]]
                   ) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Call edges by a comment/string-aware token scan over each def's body text.

    An identifier that names a function in the universe (``defs``) and is
    immediately followed by ``(`` (a call) or ``<`` (an explicit template-id call)
    is an edge caller→callee.  Self-edges are dropped (recursion is not a rename
    concern here).  Reads each source file once.
    """
    universe = set(defs)
    edges: dict[str, set[str]] = {}
    reverse: dict[str, set[str]] = {}
    file_lines: dict[str, list[str]] = {}

    for name, fds in defs.items():
        callees: set[str] = set()
        for fd in fds:
            lines = file_lines.get(fd.file)
            if lines is None:
                try:
                    lines = Path(fd.file).read_text(encoding="utf-8", errors="ignore").split("\n")
                except OSError:
                    lines = []
                file_lines[fd.file] = lines
            body = "\n".join(lines[fd.line_start - 1:fd.line_end])
            for ident, nxt in _scan_idents(body):
                if ident != name and ident in universe and nxt in ("(", "<"):
                    callees.add(ident)
        if callees:
            edges[name] = callees
            for c in callees:
                reverse.setdefault(c, set()).add(name)
    return edges, reverse


def _scan_idents(text: str) -> list[tuple[str, str]]:
    """Identifier tokens with the next significant char, skipping comments/literals.

    Returns ``[(identifier, next_non_space_char), ...]``.  Same lexical state
    machine as :mod:`agents.integrator_base.boundary` / region_scan (line/block
    comments, string and char literals are skipped) so a commented-out or
    stringified call is never mistaken for an edge.
    """
    out: list[tuple[str, str]] = []
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
            if ch in _IDENT_CHARS and not ch.isdigit():
                j = i
                while j < n and text[j] in _IDENT_CHARS:
                    j += 1
                ident = text[i:j]
                # next significant (non-space) char after the identifier
                k = j
                while k < n and text[k].isspace():
                    k += 1
                out.append((ident, text[k] if k < n else ""))
                i = j
                continue
            if ch in _IDENT_CHARS:            # a number lead — consume it
                j = i
                while j < n and text[j] in _IDENT_CHARS:
                    j += 1
                i = j
                continue
            i += 1
            continue
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
    return out
