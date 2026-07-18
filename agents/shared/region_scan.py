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
