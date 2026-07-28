"""Preprocessor-active line resolution for the call-graph fan-out.

Some entry points and dispatch functions have *multiple* definitions in the tree
where only ONE survives preprocessing under the app's build defines.  qcdloop's box
dispatch is the motivating case: each group header (``B0m.h``, ``B1m.h``, …) carries
a **pruned** ``BO`` guarded by ``#ifndef QCDLOOP_BOX_FULL_DISPATCH``, and the
meta-header (``boxGPU.h``) ``#define``s that macro *before* including the group
headers, so all the pruned copies are preprocessor-dead and only the meta-header's
full-dispatch ``BO`` reaches the compiler.  A fan-out that reroutes the entry point
must land on the *live* definition, not an arbitrary one — picking a dead copy
silently no-ops the whole promotion (STOP #A).

Why a source-text walk (not libclang)
-------------------------------------
libclang's Python bindings do not expose per-cursor preprocessor-branch-active
information (no ``isPreprocessorBranchActive`` equivalent on ``Cursor``), and on the
template-heavy, broken-include parses qcdloop needs (see :mod:`agents.patcher.call_graph`)
even the C API's branch info is unreliable.  So — matching the hybrid discipline
call_graph.py already established (libclang for definitions/extents it recovers
reliably, a deterministic source-text scan for what it drops) — preprocessor liveness
is recovered by a **conditional-directive-aware include-chain walk** over the same
translation unit, tracking ``#define`` / ``#undef`` state and ``#if`` / ``#ifdef`` /
``#ifndef`` / ``#elif`` / ``#else`` / ``#endif`` nesting.  A line is *active* iff every
enclosing conditional branch it sits under is taken.

This is intentionally a *structural* recognizer: it reads whatever guard macros the
source itself uses and evaluates ``defined(...)`` / ``!defined(...)`` / bare-macro /
``0`` / ``1`` conditions against the accumulated define set.  It invents no macros and
consults no app-specific identifiers.  An ``#if`` expression it cannot evaluate
(arithmetic, ``&&`` / ``||`` compounds) is treated conservatively as **taken** — it
over-includes rather than silently drops a definition, so the fan-out's tie-break
(``≥2`` live candidates → fail loud) catches an unresolved case instead of guessing.
"""

from __future__ import annotations

import re
from pathlib import Path

# ``#include "x"`` / ``#include <x>`` — first group is the delimiter (quote vs angle).
_INCLUDE_RE = re.compile(r'#\s*include\s*([<"])([^>"]+)[>"]')
# A single ``defined X`` / ``defined(X)`` / ``!defined(X)`` condition — the only
# ``#if`` forms we evaluate; anything else is conservatively "taken".
_DEFINED_RE = re.compile(r'^(!)?\s*defined\s*\(?\s*(\w+)\s*\)?$')
_IDENT_RE = re.compile(r'^\w+$')


def defines_from_args(extra_args) -> set[str]:
    """Extract ``-D`` macro names from a clang/gcc-style argument list.

    Recognizes ``-DNAME``, ``-DNAME=value`` and the split ``-D NAME`` forms; the
    value (if any) is discarded — the walk only tracks definedness, which is all the
    guard conditions in play need.  Anything not a ``-D`` flag is ignored.
    """
    out: set[str] = set()
    args = list(extra_args or [])
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-D" and i + 1 < len(args):
            name = args[i + 1].split("=", 1)[0].strip()
            if name:
                out.add(name)
            i += 2
            continue
        if a.startswith("-D") and len(a) > 2:
            name = a[2:].split("=", 1)[0].strip()
            if name:
                out.add(name)
        i += 1
    return out


def _resolve_include(current: Path, target: str, angle: bool,
                     include_dirs: list[Path]) -> Path | None:
    """Resolve an ``#include`` target to a file on disk (``None`` if not found).

    A quoted include is searched relative to the including file first (the C rule),
    then the ``-I`` dirs; an angled include searches only the ``-I`` dirs.  Missing
    includes (system headers outside the tree) resolve to ``None`` and are skipped —
    the walk cares only about definitions *inside* the tree.
    """
    candidates: list[Path] = []
    if not angle:
        candidates.append(current.parent / target)
    candidates.extend(d / target for d in include_dirs)
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def _eval_if(expr: str, defines: set[str]) -> bool:
    """Evaluate a ``#if`` controlling expression against ``defines`` (conservative).

    Handles the forms the tree actually uses — ``defined X`` / ``defined(X)`` /
    ``!defined(X)``, a bare macro name (truthy iff defined), and the literals ``0`` /
    ``1``.  Any richer expression (arithmetic, ``&&`` / ``||``) is treated as **taken**
    so a definition inside it is never silently dropped; an unexpected over-inclusion
    surfaces later as a ≥2-live-candidate tie-break (fail loud), not a wrong pick.
    """
    e = expr.strip()
    if e == "0":
        return False
    if e == "1":
        return True
    m = _DEFINED_RE.match(e)
    if m:
        negate, name = m.group(1), m.group(2)
        present = name in defines
        return (not present) if negate else present
    if _IDENT_RE.match(e):
        return e in defines
    return True  # unevaluable -> conservatively active


def compute_active_lines(tu_file: str | Path, include_dirs, defines=()) -> dict[str, set[int]]:
    """Map each reachable file to the set of its *preprocessor-active* line numbers.

    Walks the include closure of ``tu_file`` (a translation-unit header) under the
    initial ``defines``, resolving ``#include`` against ``include_dirs`` and honouring
    ``#define`` / ``#undef`` and conditional nesting.  Returns ``{abs_path: {active
    line numbers, 1-based}}``; a file that is included but whose every definition sits
    under an inactive branch still appears (with a smaller active set), while a file
    never reached does not appear at all.

    Each file is processed **once** (an include guard / ``#pragma once`` idempotence
    that matches every header in play), on its first *active* inclusion — so a line's
    activeness reflects the define state at the point the file is first pulled in, the
    same state the compiler sees for a ``#pragma once`` header.
    """
    inc_dirs = [Path(d) for d in include_dirs]
    active: dict[str, set[int]] = {}
    visited: set[str] = set()
    defs = set(defines)

    def process(path: Path) -> None:
        key = str(path)
        if key in visited:
            return
        visited.add(key)
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
        except OSError:
            return
        aset = active.setdefault(key, set())
        # Each stack entry: (branch_active, any_branch_taken_in_this_group).  A line is
        # active iff every entry's branch_active is True.
        stack: list[tuple[bool, bool]] = []

        def enclosing_active() -> bool:
            return all(b for b, _ in stack)

        def parent_active() -> bool:
            return all(b for b, _ in stack[:-1]) if len(stack) > 1 else True

        for idx, raw in enumerate(lines, 1):
            s = raw.strip()
            if s.startswith("#"):
                body = s[1:].lstrip()
                if body.startswith("ifndef"):
                    name = _first_token(body[len("ifndef"):])
                    on = enclosing_active() and (name not in defs)
                    stack.append((on, on))
                elif body.startswith("ifdef"):
                    name = _first_token(body[len("ifdef"):])
                    on = enclosing_active() and (name in defs)
                    stack.append((on, on))
                elif body.startswith("if") and not body.startswith("ifn"):
                    # ``#if <expr>`` (``#ifdef`` / ``#ifndef`` handled above)
                    on = enclosing_active() and _eval_if(body[len("if"):], defs)
                    stack.append((on, on))
                elif body.startswith("elif"):
                    if stack:
                        _, taken = stack[-1]
                        on = (not taken) and parent_active() and _eval_if(body[len("elif"):], defs)
                        stack[-1] = (on, taken or on)
                elif body.startswith("else"):
                    if stack:
                        _, taken = stack[-1]
                        on = (not taken) and parent_active()
                        stack[-1] = (on, taken or on)
                elif body.startswith("endif"):
                    if stack:
                        stack.pop()
                elif body.startswith("define") and enclosing_active():
                    name = _first_token(body[len("define"):]).split("(", 1)[0]
                    if name:
                        defs.add(name)
                elif body.startswith("undef") and enclosing_active():
                    name = _first_token(body[len("undef"):])
                    defs.discard(name)
                elif body.startswith("include") and enclosing_active():
                    m = _INCLUDE_RE.match(s)
                    if m:
                        tgt = _resolve_include(path, m.group(2), m.group(1) == "<", inc_dirs)
                        if tgt is not None:
                            process(tgt)
                continue
            if enclosing_active():
                aset.add(idx)

    process(Path(tu_file).resolve())
    return active


def _first_token(text: str) -> str:
    """First whitespace-delimited token of ``text`` (empty string if none)."""
    parts = text.split()
    return parts[0] if parts else ""
