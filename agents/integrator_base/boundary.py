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
   ``region_local_vars``) that is not itself a region-local write:
   ``<scalar> r__ff = <scalar>(r);`` before the region, and every whole-word ``r``
   inside the region is renamed to ``r__ff``.  (Rule R1.)
3. **Region internals stay extended.** A region-local declaration
   ``<caller> w = …`` is retyped to ``<scalar> w__ext = …`` (Rule R2); a write
   ``w`` declared *before* the region (re-assigned inside it) is seeded with
   ``<scalar> w__ext = <scalar>(w);`` at entry.  Every whole-word ``w`` inside the
   region is renamed to ``w__ext``.
4. **Writes → caller (exit).** After the region, each write is demoted back under
   its original name: ``w = static_cast<caller>(w__ext.hi) + static_cast<caller>
   (w__ext.lo);`` (a declaration ``<caller> w = …`` when ``w`` was region-local).
   The two-limb reconstruction is the extended types' own conversion-out idiom
   (neither ``ffloat`` nor ``ddouble`` defines ``operator double``).

**Write-set sources.** Region-local declarations (``<caller> w``) are recovered by
this module's own comment/string-aware scan — the reliable path for a vanilla
(``double``/``float``) region, where the Fix-C libclang/lexer scanner keys on
``tracked_type<…>`` template syntax and finds nothing.  The ``writes`` argument
(from :func:`agents.shared.region_scan.extract_region_writes`) is consumed for
writes *declared before* the region (and for already-extended regions in
``ff-to-dd`` composites); the two sources are unioned.

**Assumptions (kernel subset, documented).** A single contiguous region; one
declaration per statement (``<caller> name = …``, not ``double a, b;``); plain
value locals (no ``const`` / reference / pointer qualifiers on the retyped decl).
The rewrite is comment/string/char-literal-aware and whole-word — it never edits a
token inside a comment or string, nor a substring of a longer identifier.
"""

from __future__ import annotations

import difflib

# Identifier alphabet (same as region_scan / the P3a lexer).
_IDENT_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")

_READ_SUFFIX = "__ff"
_WRITE_SUFFIX = "__ext"


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


def _scan_local_decls(toks: list[_Tok], caller_type: str) -> tuple[list[str], set[int]]:
    """Find region-local declarations ``<caller_type> <name>`` (not function decls).

    Returns ``(names_in_source_order, retype_token_indices)`` — the declared names
    and the token indices of the ``caller_type`` keyword to retype to the scalar.
    Two forms are excluded: a ``caller_type`` whose declared name is immediately
    followed by ``(`` is a function declaration/definition; and a ``caller_type``
    appearing inside parentheses (paren depth > 0) is a function parameter or a
    loop/condition binding, not a statement-level local write.
    """
    names: list[str] = []
    seen: set[str] = set()
    retype_idx: set[int] = set()
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
        if t.text != caller_type or depth != 0:
            continue
        if i + 1 >= n or not _is_ident(toks[i + 1].text):
            continue
        name = toks[i + 1].text
        following = toks[i + 2].text if i + 2 < n else ""
        if following == "(":            # function decl, not a local write
            continue
        retype_idx.add(i)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names, retype_idx


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

    # -- classify writes --------------------------------------------------------
    decl_writes, retype_idx = _scan_local_decls(toks, caller_type)   # Case A
    decl_set = set(decl_writes)
    # Case B: passed writes that appear in the region but are NOT declared here
    # (re-assigned locals declared above the region / already-extended locals).
    caseB = [w for w in _dedupe(writes) if w in ident_texts and w not in decl_set]
    write_names = decl_writes + caseB
    write_set = set(write_names)

    pure_reads = [r for r in _dedupe(reads) if r in ident_texts and r not in write_set]

    if not pure_reads and not write_names and not shim_include:
        return None

    rename_map = {r: r + _READ_SUFFIX for r in pure_reads}
    rename_map.update({w: w + _WRITE_SUFFIX for w in write_names})

    # -- rewrite region tokens (retype decls + rename reads/writes) -------------
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
    for w in decl_writes:   # Case A: region-local decl → declare the caller alias
        exit_lines.append(f"{indent}{caller_type} {w} = {_demote_expr(w, caller_type)};"
                          f"  // Rule R1: demote region write to {caller_type}")
    for w in caseB:         # Case B: assign back to the pre-declared caller var
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
