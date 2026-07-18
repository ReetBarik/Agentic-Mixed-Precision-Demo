"""Regional boundary-patch synthesis (deterministic) — the regional analogue of
:mod:`agents.integrator_base.c8`.

The regional ff/dd integrators split their work in two:

* the **shim** (LLM-generated) provides the extended-precision *types, operators and
  named constants* the region needs, referencing the vendored ``quad::ffun::ffloat``
  / ``quad::ddfun::ddouble`` headers; and
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
   conversion-out idiom (neither ``ffloat`` nor ``ddouble`` defines ``operator
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


def _demote_expr(name: str, caller_type: str) -> str:
    """Two-limb reconstruction back to the caller precision (the extended types'
    own conversion-out idiom; no ``operator double`` exists)."""
    ext = name + _WRITE_SUFFIX
    return (f"static_cast<{caller_type}>({ext}.hi) + "
            f"static_cast<{caller_type}>({ext}.lo)")


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
) -> str | None:
    """Synthesize the region's boundary patch as a ``git apply -p1`` unified diff.

    Parameters mirror the regional integrator contract: ``rel_file`` is the
    repo-relative path (drives the ``a/``,``b/`` diff labels), ``file_text`` the
    full original file, ``[line_start, line_end]`` the inclusive 1-based region,
    ``reads`` the characterizer's ``region_local_vars``, ``writes`` the Fix-C write
    set, ``scalar_type`` the extended C++ spelling (e.g. ``quad::ffun::ffloat``),
    ``caller_type`` the precision to demote back to, and ``shim_include`` the shim
    header basename to ``#include``.  Returns the diff, or ``None`` if the region
    needs no boundary edit (no reads, no writes, no include).
    """
    original_lines = file_text.split("\n")
    if line_start < 1 or line_end > len(original_lines) or line_start > line_end:
        return None

    region_lines = original_lines[line_start - 1:line_end]
    region_text = "\n".join(region_lines)
    toks = _tokenize(region_text)
    ident_texts = {t.text for t in toks if _is_ident(t.text)}

    all_decls = _scan_decls(toks)
    decl_names = {d.name for d in all_decls}

    # Case B: pre-declared writes (Fix C) re-assigned in the region — declared
    # above it, so not among the region's own decls.
    caseB = [w for w in _dedupe(writes) if w in ident_texts and w not in decl_names]

    # Reads: promoted unconditionally on entry (Rule R1).  A name that is both a
    # region-local decl and a "read" is fundamentally a write — exclude it.
    pure_reads = [r for r in _dedupe(reads)
                  if r in ident_texts and r not in decl_names and r not in caseB]

    # Dataflow: a region-local decl is promoted iff its initializer consumes a value
    # already promoted (a read, a Case-B write, or an earlier promoted local).  The
    # promotion chains through the region in source order to a fixpoint.  Discrete
    # (integer/bool) locals are never promoted (Rule 1).
    promoted: set[str] = set(pure_reads) | set(caseB)
    decl_writes: list[_Decl] = []
    changed = True
    while changed:
        changed = False
        for d in all_decls:
            if d.name in promoted or d.type_text in _INT_TYPES:
                continue
            if d.rhs_idents & promoted:
                promoted.add(d.name)
                decl_writes.append(d)
                changed = True

    if not pure_reads and not decl_writes and not caseB and not shim_include:
        return None

    rename_map = {r: r + _READ_SUFFIX for r in pure_reads}
    rename_map.update({w: w + _WRITE_SUFFIX for w in caseB})
    rename_map.update({d.name: d.name + _WRITE_SUFFIX for d in decl_writes})
    retype_idx = {d.type_idx for d in decl_writes}

    # -- rewrite region tokens (retype promoted decls + rename reads/writes) -----
    edits: list[tuple[int, int, str]] = []
    for i, t in enumerate(toks):
        if i in retype_idx:
            edits.append((t.start, t.end, scalar_type))
        elif t.text in rename_map:
            edits.append((t.start, t.end, rename_map[t.text]))
    new_region_text = _apply_spans(region_text, edits)

    # -- entry / exit lines (match region indentation) --------------------------
    indent = _leading_ws(region_lines[0]) if region_lines else ""
    entry: list[str] = []
    for r in pure_reads:
        entry.append(f"{indent}{scalar_type} {r}{_READ_SUFFIX} = {scalar_type}({r});"
                     f"  // Rule R1: promote region read to {scalar_type}")
    for w in caseB:
        entry.append(f"{indent}{scalar_type} {w}{_WRITE_SUFFIX} = {scalar_type}({w});"
                     f"  // Rule R1: seed pre-declared write in {scalar_type}")

    exit_lines: list[str] = []
    for d in decl_writes:   # region-local decl → declare the alias at its own type
        exit_lines.append(
            f"{indent}{d.type_text} {d.name} = {_demote_expr(d.name, d.type_text)};"
            f"  // Rule R1: demote region write to {d.type_text}")
    for w in caseB:         # pre-declared write → assign back at the caller type
        exit_lines.append(f"{indent}{w} = {_demote_expr(w, caller_type)};"
                          f"  // Rule R1: demote region write to {caller_type}")

    new_block = entry + new_region_text.split("\n") + exit_lines
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


def _insert_shim_include(lines: list[str], shim_include: str) -> list[str]:
    """Insert ``#include "<shim>"`` after the first ``#pragma once`` (idempotent)."""
    include_line = f'#include "{shim_include}"'
    if any(line.strip() == include_line for line in lines):
        return lines
    for idx, line in enumerate(lines):
        if line.strip() == "#pragma once":
            return lines[:idx + 1] + [include_line] + lines[idx + 1:]
    # no pragma once — prepend at the very top
    return [include_line] + lines


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
