"""Merge TU-global symbols across regional shims into one canonical per-family shim.

Every regional shim the LLM emits defines *full* translation-unit-global symbols:
a `template<> struct ql::Constants<T> { ... };` specialization and, often,
free-function overloads in `namespace ql` (`ql::kAbs`, `ql::Max`, `ql::Real`,
`ql::Lnrat`, `ql::iszero`, …) plus occasional bridges into other namespaces. C++
allows exactly **one** definition of each such symbol per translation unit. All
box headers compile into a single TU, so when the Patcher installs a *second*
shim that also defines `Constants<DoubleDouble>` (or re-defines a `ql::` helper with
the same signature) the build dies with `redefinition of 'struct
ql::Constants<...>'`. That is the WAVE3 residual: 72 of 79 `llm_gen_failed` events
(see ``runs/qcdloop/WAVE3_CHARACTERIZATION_2026-07-21.md``).

The fix (Wave-3 design Q1=B / Q2=2a): rather than install a fresh per-region shim
into the tree, **merge** each region's generated symbols into ONE canonical
per-family shim (``ql_shim_dd.h`` / ``ql_shim_ff.h`` / ``ql_shim_float.h``). Class
specializations accumulate members (unioned by member signature); free functions
and namespace blocks are deduped by signature. `#pragma once` then guarantees a
single definition per TU. The merge is **keep-first**: the already-committed (and
already-validated) definition wins on a duplicate, and only genuinely-new symbols
are appended.

This is deliberately NOT a full C++ parser — it is a brace-aware chunker tuned to
the shim format the integrator prompts produce (see the shakedown/WAVE runs). It
is symbol-agnostic: nothing here mentions ``Constants`` by name, so the same code
dedups any TU-global symbol a shim emits at namespace or specialization scope.
The build gate remains the ground-truth backstop — a merge that produced invalid
C++ would fail the build and be retried exactly like any other misgeneration.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"][^>"]+[>"]')
_PRAGMA_ONCE_RE = re.compile(r'^\s*#\s*pragma\s+once\b')
_SOURCE_HASH_RE = re.compile(r'^\s*//\s*SOURCE_HASH:')
_OTHER_DIRECTIVE_RE = re.compile(r'^\s*#')

_NS_OPEN_RE = re.compile(r'^namespace\s+([A-Za-z_]\w*)\s*\{')
_FWD_RE = re.compile(r'^(?:template\s*<[^>]*>\s*)?(?:struct|class)\s+([A-Za-z_]\w*)\s*;')
_SPEC_RE = re.compile(r'^template\s*<\s*>\s*(?:struct|class)\s+([A-Za-z_]\w*)\s*<')


# --------------------------------------------------------------------------- #
# normalization
# --------------------------------------------------------------------------- #

def _norm_type(s: str) -> str:
    """Normalize a type spelling so ``::Kokkos::Experimental::DoubleDouble`` and
    ``Kokkos::Experimental::DoubleDouble`` (and whitespace variants) compare equal."""
    s = re.sub(r'\s+', '', s.strip())
    while s.startswith('::'):
        s = s[2:]
    return s


def _norm_ws(s: str) -> str:
    """Collapse all runs of whitespace to a single space and strip."""
    return re.sub(r'\s+', ' ', s.strip())


def _split_commas_top(s: str) -> list[str]:
    """Split ``s`` on commas that are at bracket-depth 0 (``()<>[]{}``)."""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for c in s:
        if c in '(<[{':
            depth += 1
        elif c in ')>]}':
            depth -= 1
        if c == ',' and depth == 0:
            parts.append(''.join(cur))
            cur = []
        else:
            cur.append(c)
    if cur:
        parts.append(''.join(cur))
    return parts


def _strip_param_names(argstr: str) -> str:
    """Best-effort: drop parameter *names* and defaults, keep normalized types.

    So ``Max(DoubleDouble a, DoubleDouble b)`` and ``Max(DoubleDouble x, DoubleDouble y)`` key alike,
    while ``kAbs(const DoubleDouble&)`` (unnamed) still normalizes stably.
    """
    out: list[str] = []
    for p in _split_commas_top(argstr):
        p = p.split('=', 1)[0].strip()
        if not p:
            continue
        m = re.match(r'^(.*?)([A-Za-z_]\w*)\s*$', p, re.S)
        if m and m.group(1).strip():
            out.append(_norm_type(m.group(1)))   # head is the type, tail was the name
        else:
            out.append(_norm_type(p))
    return ','.join(out)


# --------------------------------------------------------------------------- #
# brace-aware chunking
# --------------------------------------------------------------------------- #

def _split_top_level(body: str) -> list[str]:
    """Split ``body`` (no preprocessor lines) into top-level C++ items.

    An item ends at a ``;`` at brace-depth 0 (a forward decl) or at the ``}`` that
    closes a brace-depth-0 block (a namespace / struct / function), including any
    trailing ``;`` and same-line ``//`` comment.  Leading comments stay attached
    to the item that follows them.
    """
    items: list[str] = []
    cur: list[str] = []
    depth = 0
    i = 0
    n = len(body)
    while i < n:
        c = body[i]
        if c == '/' and i + 1 < n and body[i + 1] == '/':
            j = body.find('\n', i)
            j = n if j == -1 else j
            cur.append(body[i:j])
            i = j
            continue
        if c == '/' and i + 1 < n and body[i + 1] == '*':
            j = body.find('*/', i + 2)
            j = n if j == -1 else j + 2
            cur.append(body[i:j])
            i = j
            continue
        if c in '"\'':
            j = i + 1
            while j < n:
                if body[j] == '\\':
                    j += 2
                    continue
                if body[j] == c:
                    j += 1
                    break
                j += 1
            cur.append(body[i:j])
            i = j
            continue
        cur.append(c)
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                j = i + 1
                while j < n and body[j] in ' \t':
                    cur.append(body[j])
                    j += 1
                if j < n and body[j] == ';':
                    cur.append(';')
                    j += 1
                while j < n and body[j] in ' \t':
                    cur.append(body[j])
                    j += 1
                if j + 1 < n and body[j] == '/' and body[j + 1] == '/':
                    k = body.find('\n', j)
                    k = n if k == -1 else k
                    cur.append(body[j:k])
                    j = k
                items.append(''.join(cur).strip())
                cur = []
                i = j
                continue
        elif c == ';' and depth == 0:
            items.append(''.join(cur).strip())
            cur = []
        i += 1
    tail = ''.join(cur).strip()
    if tail:
        items.append(tail)
    return [it for it in items if it]


def _first_top_brace(s: str) -> int | None:
    """Index of the first ``{`` at angle-bracket depth 0 (a block opener)."""
    ang = 0
    for i, c in enumerate(s):
        if c == '<':
            ang += 1
        elif c == '>':
            if ang:
                ang -= 1
        elif c == '{' and ang == 0:
            return i
    return None


def _balanced(s: str, open_idx: int, opener: str = '(', closer: str = ')') -> str:
    """Text between ``s[open_idx] == opener`` and its matching ``closer``."""
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == opener:
            depth += 1
        elif s[i] == closer:
            depth -= 1
            if depth == 0:
                return s[open_idx + 1:i]
    return s[open_idx + 1:]


def _spec_args(s: str) -> str:
    """Template arguments of a specialization ``struct NAME< ARGS > {``."""
    m = re.search(r'(?:struct|class)\s+[A-Za-z_]\w*\s*<', s)
    if not m:
        return ''
    lt = s.index('<', m.end() - 1)
    return _balanced(s, lt, '<', '>')


def _fn_key(s: str):
    """Key a free-function definition as ``('fn', name, arg-types)`` or None."""
    brace = _first_top_brace(s)
    sig = s[:brace] if brace is not None else s
    matches = list(re.finditer(r'(operator\s*[^\s(]+|[A-Za-z_]\w*)\s*\(', sig))
    if not matches:
        return None
    m = matches[-1]
    paren = sig.index('(', m.start())
    params = _balanced(s, paren)
    return ('fn', re.sub(r'\s+', '', m.group(1)), _strip_param_names(params))


# --------------------------------------------------------------------------- #
# IR
# --------------------------------------------------------------------------- #

@dataclass
class Spec:
    """A ``template<> struct NAME<ARGS> { ... };`` specialization."""
    header: str                                   # up to and incl. the opening '{'
    footer: str                                   # closing '};' (+ any trailing comment)
    member_order: list = field(default_factory=list)      # member keys, in order
    members: dict = field(default_factory=dict)           # key -> member text


@dataclass
class Scope:
    """A namespace body (or the file top level): ordered, keyed entries.

    ``order`` is a list of ``(kind, key)`` where kind is ``'ns'`` / ``'spec'`` /
    ``'leaf'``; the payload dicts hold the values.
    """
    order: list = field(default_factory=list)
    namespaces: dict = field(default_factory=dict)        # name -> Scope
    specs: dict = field(default_factory=dict)             # key -> Spec
    leaves: dict = field(default_factory=dict)            # key -> text


def _parse_scope(body: str) -> Scope:
    scope = Scope()
    for item in _split_top_level(body):
        _add_item(scope, item)
    return scope


def _add_item(scope: Scope, item: str) -> None:
    stripped = item.lstrip()
    # leading comment(s) may precede the construct — find the code start
    code = _strip_leading_comments(stripped)

    m = _NS_OPEN_RE.match(code)
    if m:
        name = m.group(1)
        open_idx = item.index('{')
        inner = _balanced(item, open_idx, '{', '}')
        child = scope.namespaces.get(name)
        if child is None:
            child = Scope()
            scope.namespaces[name] = child
            scope.order.append(('ns', name))
        for sub in _split_top_level(inner):
            _add_item(child, sub)
        return

    m = _SPEC_RE.match(code)
    if m:
        name = m.group(1)
        args = _norm_type(_spec_args(item))
        key = ('spec', name, args)
        open_idx = _first_top_brace(item)
        if open_idx is not None:
            header = item[:open_idx + 1]
            members_body = _balanced(item, open_idx, '{', '}')
            close = _matching_close(item, open_idx)
            footer = item[close:] if close is not None else '};'
            spec = scope.specs.get(key)
            if spec is None:
                spec = Spec(header=header.rstrip(), footer=footer.strip())
                scope.specs[key] = spec
                scope.order.append(('spec', key))
            _merge_members(spec, members_body)
            return

    if _FWD_RE.match(code) and '{' not in code:
        fm = _FWD_RE.match(code)
        key = ('fwd', fm.group(1))
        if key not in scope.leaves:
            scope.leaves[key] = item.strip()
            scope.order.append(('leaf', key))
        return

    fk = _fn_key(code)
    if fk is not None:
        if fk not in scope.leaves:
            scope.leaves[fk] = item.strip()
            scope.order.append(('leaf', fk))
        return

    key = ('raw', _norm_ws(code))
    if key not in scope.leaves:
        scope.leaves[key] = item.strip()
        scope.order.append(('leaf', key))


def _merge_members(spec: Spec, members_body: str) -> None:
    for member in _split_top_level(members_body):
        code = _strip_leading_comments(member.lstrip())
        fk = _fn_key(code)
        if fk is not None:
            key = ('mem', fk[1], fk[2])
        else:
            key = ('memraw', _norm_ws(code))
        if key not in spec.members:
            spec.members[key] = member.strip()
            spec.member_order.append(key)


def _strip_leading_comments(s: str) -> str:
    """Return ``s`` with any leading ``//`` / ``/* */`` comment lines removed."""
    i = 0
    n = len(s)
    while i < n:
        while i < n and s[i] in ' \t\r\n':
            i += 1
        if s.startswith('//', i):
            j = s.find('\n', i)
            i = n if j == -1 else j + 1
        elif s.startswith('/*', i):
            j = s.find('*/', i + 2)
            i = n if j == -1 else j + 2
        else:
            break
    return s[i:]


def _matching_close(s: str, open_idx: int) -> int | None:
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    return None


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

@dataclass
class ShimIR:
    includes: list = field(default_factory=list)
    directives: list = field(default_factory=list)
    top: Scope = field(default_factory=Scope)


def parse_shim(text: str) -> ShimIR:
    """Parse a shim into includes + directives + a top-level :class:`Scope`."""
    ir = ShimIR()
    body_lines: list[str] = []
    for line in text.splitlines():
        if _PRAGMA_ONCE_RE.match(line) or _SOURCE_HASH_RE.match(line):
            continue
        if _INCLUDE_RE.match(line):
            hdr = line.strip()
            if hdr not in ir.includes:
                ir.includes.append(hdr)
            continue
        if _OTHER_DIRECTIVE_RE.match(line):
            d = line.strip()
            if d not in ir.directives:
                ir.directives.append(d)
            continue
        body_lines.append(line)
    ir.top = _parse_scope("\n".join(body_lines))
    return ir


def _merge_scope(dst: Scope, src: Scope) -> None:
    for kind, key in src.order:
        if kind == 'ns':
            child = dst.namespaces.get(key)
            if child is None:
                child = Scope()
                dst.namespaces[key] = child
                dst.order.append(('ns', key))
            _merge_scope(child, src.namespaces[key])
        elif kind == 'spec':
            existing = dst.specs.get(key)
            incoming = src.specs[key]
            if existing is None:
                dst.specs[key] = incoming
                dst.order.append(('spec', key))
            else:
                for mkey in incoming.member_order:
                    if mkey not in existing.members:
                        existing.members[mkey] = incoming.members[mkey]
                        existing.member_order.append(mkey)
        else:  # leaf — keep-first
            if key not in dst.leaves:
                dst.leaves[key] = src.leaves[key]
                dst.order.append(('leaf', key))


def merge_ir(existing: ShimIR, new: ShimIR) -> ShimIR:
    """Merge ``new`` into ``existing`` (keep-first on duplicates)."""
    for inc in new.includes:
        if inc not in existing.includes:
            existing.includes.append(inc)
    for d in new.directives:
        if d not in existing.directives:
            existing.directives.append(d)
    _merge_scope(existing.top, new.top)
    return existing


def _render_spec(spec: Spec, indent: str) -> str:
    inner = indent + "    "
    body = ("\n\n").join(_reindent(spec.members[k], inner) for k in spec.member_order)
    return f"{_reindent(spec.header, indent)}\n{body}\n{indent}{spec.footer.strip()}"


def _render_scope(scope: Scope, indent: str) -> str:
    out: list[str] = []
    # Forward declarations must precede any specialization of the same template,
    # so emit all forward-decl leaves first (in insertion order), then everything
    # else in order.  A ``template<> struct Constants<X>`` that appeared before its
    # ``template<class T> struct Constants;`` in a later-merged shim would
    # otherwise specialize an undeclared primary.
    def _is_fwd(entry):
        kind, key = entry
        return kind == 'leaf' and isinstance(key, tuple) and key and key[0] == 'fwd'

    for kind, key in [e for e in scope.order if _is_fwd(e)]:
        out.append(_reindent(scope.leaves[key], indent))
    for kind, key in [e for e in scope.order if not _is_fwd(e)]:
        if kind == 'ns':
            inner = _render_scope(scope.namespaces[key], indent + "    ")
            out.append(f"{indent}namespace {key} {{\n{inner}\n{indent}}} // namespace {key}")
        elif kind == 'spec':
            out.append(_render_spec(scope.specs[key], indent))
        else:
            out.append(_reindent(scope.leaves[key], indent))
    return ("\n\n").join(out)


def _reindent(text: str, indent: str) -> str:
    """Re-indent a (possibly multi-line) item so its first line sits at ``indent``.

    ``_split_top_level`` strips each item, so the first line has no leading
    whitespace; the continuation lines keep their original indentation.  We dedent
    the continuation lines by their common leading whitespace and re-indent the
    whole item to ``indent``, preserving relative structure."""
    lines = text.split('\n')
    if len(lines) == 1:
        return indent + lines[0].strip()
    rest = [ln for ln in lines[1:] if ln.strip()]
    common = min((len(ln) - len(ln.lstrip()) for ln in rest), default=0)
    out = [indent + lines[0].strip()]
    for ln in lines[1:]:
        out.append(indent + ln[common:] if ln.strip() else '')
    return '\n'.join(out)


def render_shim(ir: ShimIR) -> str:
    parts: list[str] = ["#pragma once"]
    body = _render_scope(ir.top, "")
    digest = hashlib.sha256(
        ("\n".join(ir.includes) + "\n" + body).encode("utf-8")).hexdigest()
    parts.append(f"// SOURCE_HASH: {digest}")
    parts.append("// Canonical merged regional shim — one definition per TU-global")
    parts.append("// symbol (Wave-3 dedup, agents/integrator_base/shim_merge.py).")
    parts.extend(ir.includes)
    parts.extend(ir.directives)
    parts.append("")
    parts.append(body)
    return "\n".join(parts) + "\n"


def dedup_inline(text: str) -> str:
    """Collapse a redundant ``inline`` specifier adjacent to ``KOKKOS_INLINE_FUNCTION``.

    ``KOKKOS_INLINE_FUNCTION`` already expands to ``inline`` (host) / ``__device__
    __host__ inline`` (CUDA), so an LLM shim that writes ``KOKKOS_INLINE_FUNCTION
    inline T f(...)`` (the Phase-2c kokkosUtils.h:703 artifact) yields ``inline inline``
    after macro expansion → ``duplicate 'inline'`` (a hard build failure).  This
    deterministic sanitizer drops the redundant specifier (either order) and collapses
    a plain ``inline inline`` run, so the otherwise-correct shim builds."""
    text = re.sub(r"\bKOKKOS_INLINE_FUNCTION\s+inline\b", "KOKKOS_INLINE_FUNCTION", text)
    text = re.sub(r"\binline\s+KOKKOS_INLINE_FUNCTION\b", "KOKKOS_INLINE_FUNCTION", text)
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\binline\s+inline\b", "inline", text)
    return text


def merge_into_canonical(existing_text: str | None, new_shim_body: str) -> str:
    """Merge ``new_shim_body`` into the canonical shim ``existing_text``.

    ``existing_text`` is the current canonical shim (``None`` / empty for the first
    lander).  Returns the rendered merged canonical shim text.  Both inputs are passed
    through :func:`dedup_inline` first so a redundant ``inline`` specifier never
    survives into the merged TU (Phase 2d d2 fix).
    """
    new_ir = parse_shim(dedup_inline(new_shim_body))
    if existing_text and existing_text.strip():
        base = parse_shim(dedup_inline(existing_text))
    else:
        base = ShimIR()
    merged = merge_ir(base, new_ir)
    return render_shim(merged)
