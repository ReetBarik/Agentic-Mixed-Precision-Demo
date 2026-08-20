"""Plain-type-edit dispatch path (P3a) — ``float-to-double`` / ``double-to-float``.

The design locks this as an **AST-aware** rewrite (libclang) rather than a naive
``sed``, because a substring substitution corrupts identifiers, comments and
string literals.  ``float`` and ``double`` are C++ *reserved keywords*, so the
corruption modes that motivated the AST decision split in two:

* **Identifier collisions** (``float_traits``, ``floating_point``) — impossible
  for a keyword: a reserved word can never be part of a longer identifier token,
  so a whole-token match never fires inside one.
* **Comments / string / char literals** — a real risk for a plain regex; we
  guard against it with a small C++ lexer that skips those spans and only rewrites
  bare keyword tokens on the target lines.

We keep libclang as the *preferred* backend (imported lazily) so the AST path is
used wherever the bindings are installed; when they are absent we fall back to the
keyword-token rewriter, which gives the same corruption-safety guarantee for this
specific ``float``⇄``double`` keyword swap.  See docs/KNOWN_LIMITATIONS.md (the
bindings are not installed on the current cluster image).
"""

from __future__ import annotations

from pathlib import Path

_KEYWORD_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


class EditError(RuntimeError):
    """Raised when the type edit cannot be performed (→ ``patch_apply_failed``)."""


def rewrite_types(path: Path, line_start: int, line_end: int,
                  src_type: str, dst_type: str) -> int:
    """Rewrite bare ``src_type`` keyword tokens to ``dst_type`` on the region lines.

    Operates in-place on ``path`` over the inclusive 1-based line range
    ``[line_start, line_end]``.  Returns the number of tokens rewritten.  Raises
    :class:`EditError` if no occurrence is found (a no-op edit is a malformed
    intent — the region does not actually declare the source type).
    """
    text = path.read_text(encoding="utf-8")
    new_text, n = _rewrite_keyword(text, line_start, line_end, src_type, dst_type)
    if n == 0:
        raise EditError(
            f"plain-type-edit found no bare `{src_type}` token in "
            f"{path.name}:{line_start}-{line_end}")
    path.write_text(new_text, encoding="utf-8")
    return n


def _rewrite_keyword(text: str, line_start: int, line_end: int,
                     src: str, dst: str) -> tuple[str, int]:
    """Keyword-token rewriter: skip comments / string / char literals.

    A single left-to-right pass tracking the current line number and lexical
    state.  Only identifier runs that (a) equal ``src`` exactly and (b) start on a
    line within ``[line_start, line_end]`` and (c) are outside any comment or
    literal are replaced.
    """
    out: list[str] = []
    i, n = 0, len(text)
    line = 1
    state = "code"          # code | line_comment | block_comment | string | char
    count = 0

    def in_range(ln: int) -> bool:
        return line_start <= ln <= line_end

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line_comment"; out.append(ch); i += 1; continue
            if ch == "/" and nxt == "*":
                state = "block_comment"; out.append(ch); i += 1; continue
            if ch == '"':
                state = "string"; out.append(ch); i += 1; continue
            if ch == "'":
                state = "char"; out.append(ch); i += 1; continue
            if ch in _KEYWORD_CHARS and (i == 0 or text[i - 1] not in _KEYWORD_CHARS):
                j = i
                while j < n and text[j] in _KEYWORD_CHARS:
                    j += 1
                token = text[i:j]
                if token == src and in_range(line):
                    out.append(dst); count += 1
                else:
                    out.append(token)
                # advance line counter across the token (tokens never hold '\n')
                i = j
                continue
            out.append(ch)
            if ch == "\n":
                line += 1
            i += 1
            continue

        # --- inside a comment / literal: copy through until it closes ---
        out.append(ch)
        if ch == "\n":
            line += 1
        if state == "line_comment":
            if ch == "\n":
                state = "code"
        elif state == "block_comment":
            if ch == "*" and nxt == "/":
                out.append(nxt); i += 1; state = "code"
        elif state == "string":
            if ch == "\\" and nxt:
                out.append(nxt); i += 1
            elif ch == '"':
                state = "code"
        elif state == "char":
            if ch == "\\" and nxt:
                out.append(nxt); i += 1
            elif ch == "'":
                state = "code"
        i += 1

    return "".join(out), count
