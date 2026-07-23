"""Phase 2a — Patcher call-graph fan-out (regions only).

When Strategy accepts an intent to modify a *region* of function ``f`` at precision
``P``, the fan-out realizes it as **per-caller-path function variants** instead of a
type-specialization shim.  For each caller-path ``entry -> h -> g -> f`` from the
integral's entry point down to ``f``:

* ``f`` is copied to a variant ``f_g_h_<integral>`` with the region promoted to ``P``
  inline (via :func:`agents.integrator_base.boundary.promote_region_block`);
* each function above ``f`` on the path is copied to a variant whose body calls the
  child's variant (``g -> g_h_<integral>`` calling ``f_g_h_<integral>``; ``h ->
  h_<integral>`` calling ``g_h_<integral>``);
* the **entry point is never renamed** — its call site lives in the shared, read-only
  driver — so only its *body* is edited in place to call the first-level variant.

Why this fixes the numerical no-op (Blocker #1)
-----------------------------------------------
The retired shim path specialized ``Constants<T>`` for the target precision, but the
binary only ever instantiated the app at plain ``double`` and never referenced the
specialization, so it was dead code and coefficients were bit-identical to baseline.
A variant ``f_g_h_B1`` is a *new function referenced by name* at the (renamed) call
sites up to the entry point — the compiler must compile it, and its precision is baked
into the body — so the change is real and provable (the build gate ``nm``-checks that
every declared variant symbol is present; see :mod:`agents.patcher.gates`).

Degenerate case — region in the entry point itself
--------------------------------------------------
qcdloop's signal-recovery lines (``boxGPU.h:99-101``) live in the entry point ``BO``.
There is nothing above ``BO`` to rename and ``BO`` cannot be renamed (shared driver
call site), so the region is promoted **in place** in ``BO``'s (already-referenced)
body.  This still fixes Blocker #1 — the promoted arithmetic is genuinely compiled —
but produces no new symbol, so the symbol-presence gate has nothing extra to assert
for those lines.

Statelessness across intents
----------------------------
Strategy drives the Patcher one intent at a time, committing between intents; the
fan-out is therefore **stateless across intents** — all variant state is persisted in
the source itself, in a per-file ``AMP-FANOUT`` block whose manifest comment carries
the authoritative :class:`VariantSpec` list.  On each intent the block is parsed,
merged with the new intent's specs, and re-rendered deterministically from the
originals, so retries and shared-prefix intents converge (the same intermediate
variant accumulates every downstream reroute + region promotion) rather than
duplicating.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from agents.integrator_base import boundary
from agents.patcher.call_graph import CallGraph, FuncDef
from agents.patcher.variant_naming import assert_no_collisions, variant_names_for_path

# Per-file generated-variant block markers.  The manifest comment on the first line
# after BEGIN is the source of truth; the rendered definitions below it are derived
# deterministically from it, so a merge re-parses the manifest, not the C++.
_BLOCK_BEGIN = "// ===== AMP-FANOUT-BEGIN (generated variants; edit the pipeline, not here) ====="
_BLOCK_END = "// ===== AMP-FANOUT-END ====="
_MANIFEST_PREFIX = "// AMP-FANOUT-MANIFEST: "

_IDENT_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


class FanoutError(RuntimeError):
    """A fan-out that cannot be realized (region not in a function, target
    unreachable from the entry point, or a variant-name collision)."""


@dataclass
class FanoutSettings:
    """Per-pass fan-out configuration injected into the Patcher.

    ``entry_point`` is the integral's call-graph root (qcdloop: ``BO``); ``integral``
    is the naming suffix for this pass's variants (``B1``).  ``include_paths`` /
    ``extra_args`` are passed to the libclang call-graph build; ``max_paths`` caps
    path enumeration to a shared helper (over-generation is bounded, and a hit is
    logged, not silent).  ``enabled`` lets a caller wire the settings but keep the
    classic regional path (used by tests / A/B runs).
    """

    entry_point: str
    integral: str
    include_paths: list[str] = field(default_factory=list)
    extra_args: list[str] = field(default_factory=list)
    max_paths: int = 1024
    enabled: bool = True


# One call graph per (tree, entry point) — built once and reused for the pass's
# lifetime (one libclang invocation per integral).  Keyed by the resolved tree path
# so concurrent per-integral passes (distinct trees) never collide; a process is one
# pass under run_all_integrals' ProcessPoolExecutor, so this is pass-scoped.
_GRAPH_CACHE: dict[tuple[str, str], CallGraph] = {}


def graph_for_pass(settings: "FanoutSettings", tree_root: str | Path) -> CallGraph:
    """Return the cached call graph for this pass (building it on first use).

    Built on the clean tree at first use (before any fan-out edit), so original
    function extents stay valid for the pass — the fan-out only ever appends variant
    blocks *after* the originals and edits the entry-point body in place (no line
    shift), so cached coordinates remain correct.
    """
    from agents.patcher.call_graph import build_call_graph
    key = (str(Path(tree_root).resolve()), settings.entry_point)
    g = _GRAPH_CACHE.get(key)
    if g is None:
        g = build_call_graph(settings.entry_point, tree_root,
                             include_paths=settings.include_paths,
                             extra_args=settings.extra_args)
        _GRAPH_CACHE[key] = g
    return g


def clear_graph_cache() -> None:
    """Drop all cached call graphs (tests; a new pass on a reused tree path)."""
    _GRAPH_CACHE.clear()


@dataclass
class Promote:
    """One region promotion baked into a variant (region coords in ORIGINAL/file
    line numbers, 1-based inclusive)."""

    region_start: int
    region_end: int
    reads: list[str]
    writes: list[str]
    scalar_type: str
    two_limb: bool
    caller_type: str = "double"


@dataclass
class VariantSpec:
    """The deterministic recipe for one variant function.

    A variant is rebuilt from scratch on every merge by copying the original
    function's source (``orig_start..orig_end`` in ``file``), applying every
    :class:`Promote` (region → extended scalar) and every reroute (call to a child
    function → the child's variant name), and renaming the definition (and any
    self-calls) ``orig_name -> variant_name``.  Being a pure function of this spec
    makes fan-out idempotent across retries and mergeable across intents.
    """

    variant_name: str
    orig_name: str
    file: str
    orig_start: int
    orig_end: int
    promotes: list[Promote] = field(default_factory=list)
    reroutes: dict[str, str] = field(default_factory=dict)
    shim_includes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "VariantSpec":
        promotes = [Promote(**p) for p in d.get("promotes", [])]
        return cls(
            variant_name=d["variant_name"], orig_name=d["orig_name"],
            file=d["file"], orig_start=d["orig_start"], orig_end=d["orig_end"],
            promotes=promotes, reroutes=dict(d.get("reroutes", {})),
            shim_includes=list(d.get("shim_includes", [])))

    def merge(self, other: "VariantSpec") -> None:
        """Fold ``other`` (same variant) into this spec: union reroutes / shim
        includes and append any region promotion not already present."""
        self.reroutes.update(other.reroutes)
        for inc in other.shim_includes:
            if inc not in self.shim_includes:
                self.shim_includes.append(inc)
        have = {(p.region_start, p.region_end) for p in self.promotes}
        for p in other.promotes:
            if (p.region_start, p.region_end) not in have:
                self.promotes.append(p)


@dataclass
class FanoutResult:
    """What one :func:`fan_out_region` call produced."""

    declared_variants: list[str]        # variant symbols the build must contain
    files_touched: list[str]
    root_edited: bool = False           # entry-point body edited in place
    in_place_region: bool = False       # region was IN the entry point (no new symbol)
    paths_enumerated: int = 0
    truncated: bool = False             # path enumeration hit its cap


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def fan_out_region(
    *,
    file: str,
    line_start: int,
    line_end: int,
    reads: list[str],
    writes: list[str],
    integral: str,
    graph: CallGraph,
    tree_root: str | Path,
    scalar_type: str,
    two_limb: bool,
    shim_include: str | None,
    caller_type: str = "double",
    max_paths: int = 1024,
) -> FanoutResult:
    """Realize a region intent as per-caller-path function variants.

    ``file`` / ``line_start`` / ``line_end`` locate the region (file may be a bare
    basename — resolved against the tree); ``reads`` are the region's promoted reads
    (characterizer ``region_local_vars``), ``writes`` the Fix-C write set.
    ``scalar_type`` / ``two_limb`` / ``shim_include`` describe the target precision
    (from the ff/dd/float integrator).  Mutates the tree under ``tree_root`` in place
    and returns a :class:`FanoutResult`.  Raises :class:`FanoutError` when the region
    is not inside a known function or its function is unreachable from the entry point.
    """
    tree = Path(tree_root).resolve()
    fd = graph.enclosing_function(file, line_start)
    if fd is None:
        raise FanoutError(
            f"region {file}:{line_start}-{line_end} is not inside any known function "
            f"(call graph rooted at {graph.root!r}); cannot fan out")

    # --- degenerate: region is IN the entry point -> promote in place ---------
    if fd.name == graph.root:
        touched = _promote_in_place(tree, fd, line_start, line_end, reads, writes,
                                    scalar_type, two_limb, caller_type, shim_include)
        return FanoutResult(declared_variants=[], files_touched=touched,
                            root_edited=True, in_place_region=True,
                            paths_enumerated=1)

    paths, truncated = graph.enumerate_paths(fd.name, max_paths=max_paths)
    if not paths:
        raise FanoutError(
            f"function {fd.name!r} (containing region {file}:{line_start}-{line_end}) "
            f"is unreachable from entry point {graph.root!r}; the call graph is "
            f"incomplete or the region belongs to another integral")

    # --- accumulate variant specs per path (+ root reroutes) -----------------
    # per-file: variant_name -> VariantSpec (this call's contributions)
    new_specs: dict[str, dict[str, VariantSpec]] = {}
    root_reroutes: dict[str, str] = {}
    name_maps: list[dict[str, str]] = []

    for path in paths:
        names = variant_names_for_path(path, integral)
        name_maps.append(names)
        root_reroutes[path[1]] = names[path[1]]     # entry-point body -> first variant
        for idx in range(1, len(path)):
            func = path[idx]
            vname = names[func]
            child = path[idx + 1] if idx + 1 < len(path) else None
            cur_fd = fd if func == fd.name else _pick_def(graph, func, must_call=child)
            file_key = cur_fd.file
            spec = new_specs.setdefault(file_key, {}).get(vname)
            if spec is None:
                spec = VariantSpec(variant_name=vname, orig_name=func,
                                   file=cur_fd.file, orig_start=cur_fd.line_start,
                                   orig_end=cur_fd.line_end)
                new_specs[file_key][vname] = spec
            if child is not None:
                spec.reroutes[child] = names[child]
            if func == fd.name:
                spec.promotes.append(Promote(
                    region_start=line_start, region_end=line_end,
                    reads=list(reads), writes=list(writes),
                    scalar_type=scalar_type, two_limb=two_limb, caller_type=caller_type))
                if shim_include and shim_include not in spec.shim_includes:
                    spec.shim_includes.append(shim_include)

    assert_no_collisions(name_maps)

    # --- apply: merge specs into each file's fan-out block --------------------
    touched: set[str] = set()
    declared: list[str] = []
    for file_key, specs in new_specs.items():
        _merge_into_file(Path(file_key), specs)
        touched.add(file_key)
        declared.extend(specs.keys())

    # --- edit the entry point's body in place (reroute to first-level variant) --
    root_file = _resolve_root_file(graph)
    root_fd = _pick_def(graph, graph.root)
    if _reroute_in_function(Path(root_file), root_fd, root_reroutes):
        touched.add(root_file)

    return FanoutResult(declared_variants=sorted(set(declared)),
                        files_touched=sorted(touched), root_edited=True,
                        paths_enumerated=len(paths), truncated=truncated)


# --------------------------------------------------------------------------- #
# variant rendering
# --------------------------------------------------------------------------- #

def _original_text(fd_file: str, start: int, end: int) -> list[str]:
    """Original source lines of a function (1-based inclusive), read from the tree.

    Originals are never edited in place (the fan-out block is spliced *after* every
    original, before the namespace close), so their line coordinates from the call
    graph stay valid throughout a pass.
    """
    lines = Path(fd_file).read_text(encoding="utf-8", errors="replace").split("\n")
    return lines[start - 1:end]


def render_variant(spec: VariantSpec) -> str:
    """Render a variant definition's C++ text from its spec (deterministic).

    Copies the original function, applies every region promotion (descending by
    line so earlier edits do not shift later coordinates), reroutes calls to child
    variants, and renames the definition + self-calls ``orig_name -> variant_name``.
    """
    lines = _original_text(spec.file, spec.orig_start, spec.orig_end)
    for p in sorted(spec.promotes, key=lambda q: q.region_start, reverse=True):
        local_s = p.region_start - spec.orig_start
        local_e = p.region_end - spec.orig_start
        if local_s < 0 or local_e >= len(lines) or local_s > local_e:
            raise FanoutError(
                f"promote region {p.region_start}-{p.region_end} out of range for "
                f"{spec.orig_name} [{spec.orig_start}-{spec.orig_end}]")
        region_text = "\n".join(lines[local_s:local_e + 1])
        block, _ = boundary.promote_region_block(
            region_text, p.reads, p.writes, p.scalar_type, p.caller_type, p.two_limb)
        lines = lines[:local_s] + block + lines[local_e + 1:]

    text = "\n".join(lines)
    rename_map = {spec.orig_name: spec.variant_name}
    rename_map.update(spec.reroutes)
    return _rewrite_calls(text, rename_map)


def _render_block(specs: dict[str, VariantSpec]) -> list[str]:
    """Render the full per-file fan-out block (manifest comment + variant defs)."""
    ordered = [specs[k] for k in sorted(specs)]
    manifest = {"variants": [s.to_json() for s in ordered]}
    out = [_BLOCK_BEGIN, _MANIFEST_PREFIX + json.dumps(manifest, sort_keys=True), ""]
    for s in ordered:
        out.append(f"// --- variant {s.variant_name} (of {s.orig_name}) ---")
        out.append(render_variant(s))
        out.append("")
    out.append(_BLOCK_END)
    return out


def _merge_into_file(path: Path, new_specs: dict[str, VariantSpec]) -> None:
    """Merge ``new_specs`` into ``path``'s fan-out block and rewrite the block.

    Parses any existing block's manifest (the authoritative spec list), folds in the
    new specs (accumulating reroutes + promotions on shared variants), re-renders the
    block, and splices it before the file's namespace close.  Also ensures every
    referenced shim header is ``#include``d once.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")

    existing, lines = _extract_block(lines)
    merged: dict[str, VariantSpec] = dict(existing)
    for vname, spec in new_specs.items():
        if vname in merged:
            merged[vname].merge(spec)
        else:
            merged[vname] = spec

    # collect shim includes across all specs and ensure they are present
    includes: list[str] = []
    for s in merged.values():
        for inc in s.shim_includes:
            if inc not in includes:
                includes.append(inc)
    for inc in includes:
        lines = boundary.insert_shim_include(lines, inc)

    block = _render_block(merged)
    at = _namespace_close_index(lines)
    lines = lines[:at] + block + lines[at:]
    path.write_text("\n".join(lines), encoding="utf-8")


def _extract_block(lines: list[str]) -> tuple[dict[str, VariantSpec], list[str]]:
    """Pull the existing fan-out block out of ``lines``.

    Returns ``(specs, lines_without_block)`` — ``specs`` parsed from the manifest
    comment (empty if no block), ``lines_without_block`` the file with the block
    removed so it can be re-rendered and re-spliced.
    """
    begin = end = None
    for i, ln in enumerate(lines):
        if ln.strip() == _BLOCK_BEGIN.strip():
            begin = i
        elif ln.strip() == _BLOCK_END.strip():
            end = i
            break
    if begin is None or end is None or end < begin:
        return {}, lines
    specs: dict[str, VariantSpec] = {}
    for ln in lines[begin:end + 1]:
        s = ln.strip()
        if s.startswith(_MANIFEST_PREFIX.strip()):
            payload = s[len(_MANIFEST_PREFIX.strip()):].strip()
            try:
                manifest = json.loads(payload)
            except json.JSONDecodeError:
                manifest = {"variants": []}
            for d in manifest.get("variants", []):
                spec = VariantSpec.from_json(d)
                specs[spec.variant_name] = spec
            break
    remaining = lines[:begin] + lines[end + 1:]
    # drop a single blank line left where the block was, if any, to avoid growth
    if begin < len(remaining) and remaining[begin - 1: begin] == [""] and \
            begin >= 1 and remaining[begin:begin + 1] == [""]:
        remaining = remaining[:begin] + remaining[begin + 1:]
    return specs, remaining


def _namespace_close_index(lines: list[str]) -> int:
    """Index at which to splice the fan-out block: before the last line that closes
    the outermost namespace (a bare ``}`` — qcdloop headers close ``namespace ql``
    at EOF).  Falls back to EOF when no such line is found.
    """
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "}":
            return i
    return len(lines)


# --------------------------------------------------------------------------- #
# in-place edits (entry point)
# --------------------------------------------------------------------------- #

def _promote_in_place(tree: Path, fd: FuncDef, line_start: int, line_end: int,
                      reads: list[str], writes: list[str], scalar_type: str,
                      two_limb: bool, caller_type: str,
                      shim_include: str | None) -> list[str]:
    """Promote a region that lives *in the entry point* in place (no rename)."""
    path = Path(fd.file)
    lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    region_text = "\n".join(lines[line_start - 1:line_end])
    block, promoted = boundary.promote_region_block(
        region_text, reads, writes, scalar_type, caller_type, two_limb)
    if promoted:
        lines = lines[:line_start - 1] + block + lines[line_end:]
    if shim_include:
        lines = boundary.insert_shim_include(lines, shim_include)
    path.write_text("\n".join(lines), encoding="utf-8")
    return [fd.file]


def _reroute_in_function(path: Path, fd: FuncDef, reroutes: dict[str, str]) -> bool:
    """Rename calls ``child -> child_variant`` inside ``fd``'s body, in place.

    Idempotent: a call already pointing at the variant is left alone.  Returns True
    if the file changed.  Only tokens *inside* ``fd``'s extent are rewritten so an
    identically-named call elsewhere in the file is untouched.
    """
    active = {k: v for k, v in reroutes.items() if k != v}
    if not active:
        return False
    lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    head = lines[:fd.line_start - 1]
    body = "\n".join(lines[fd.line_start - 1:fd.line_end])
    tail = lines[fd.line_end:]
    new_body = _rewrite_calls(body, active)
    if new_body == body:
        return False
    merged = head + new_body.split("\n") + tail
    path.write_text("\n".join(merged), encoding="utf-8")
    return True


# --------------------------------------------------------------------------- #
# call-site token rewriter (comment/string aware)
# --------------------------------------------------------------------------- #

def _rewrite_calls(text: str, rename_map: dict[str, str]) -> str:
    """Rename whole-word identifiers in ``rename_map`` that lead a call or explicit
    template-id call (immediately followed by ``(`` or ``<``), skipping comments and
    string/char literals.  Renames the definition name too (its ``name(`` param
    list) and any self-calls, which is what we want for a copied variant.
    """
    if not rename_map:
        return text
    out: list[str] = []
    i, n = 0, len(text)
    state = "code"
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                out.append(text[i:i + 2]); i += 2; state = "line_comment"; continue
            if ch == "/" and nxt == "*":
                out.append(text[i:i + 2]); i += 2; state = "block_comment"; continue
            if ch == '"':
                out.append(ch); i += 1; state = "string"; continue
            if ch == "'":
                out.append(ch); i += 1; state = "char"; continue
            if ch in _IDENT_CHARS and not ch.isdigit():
                j = i
                while j < n and text[j] in _IDENT_CHARS:
                    j += 1
                ident = text[i:j]
                k = j
                while k < n and text[k].isspace():
                    k += 1
                follow = text[k] if k < n else ""
                if ident in rename_map and follow in ("(", "<"):
                    out.append(rename_map[ident])
                else:
                    out.append(ident)
                i = j
                continue
            out.append(ch); i += 1
            continue
        # inside comment / literal — copy verbatim until it closes
        out.append(ch)
        if state == "line_comment":
            if ch == "\n":
                state = "code"
            i += 1
        elif state == "block_comment":
            if ch == "*" and nxt == "/":
                out.append(nxt); i += 2; state = "code"
            else:
                i += 1
        elif state == "string":
            if ch == "\\" and nxt:
                out.append(nxt); i += 2
            elif ch == '"':
                state = "code"; i += 1
            else:
                i += 1
        elif state == "char":
            if ch == "\\" and nxt:
                out.append(nxt); i += 2
            elif ch == "'":
                state = "code"; i += 1
            else:
                i += 1
    return "".join(out)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _pick_def(graph: CallGraph, name: str, must_call: str | None = None) -> FuncDef:
    """Pick the definition of ``name`` (the one calling ``must_call`` when given).

    qcdloop's path functions are single-definition; ``must_call`` disambiguates the
    rare overload case by choosing the definition whose body actually invokes the
    child on the path (so a variant reroutes the right call).
    """
    fds = graph.defs.get(name)
    if not fds:
        raise FanoutError(f"no definition for {name!r} in the call graph")
    if must_call is None or len(fds) == 1:
        return fds[0]
    for fd in fds:
        body = "\n".join(_original_text(fd.file, fd.line_start, fd.line_end))
        for ident, follow in _idents_with_follow(body):
            if ident == must_call and follow in ("(", "<"):
                return fd
    return fds[0]


def _idents_with_follow(text: str):
    """(identifier, next-significant-char) pairs skipping comments/literals — a thin
    reuse of the rewriter's lexer for the overload probe."""
    pairs: list[tuple[str, str]] = []
    i, n = 0, len(text)
    state = "code"
    while i < n:
        ch = text[i]; nxt = text[i + 1] if i + 1 < n else ""
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
                k = j
                while k < n and text[k].isspace():
                    k += 1
                pairs.append((text[i:j], text[k] if k < n else ""))
                i = j; continue
            i += 1; continue
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
    return pairs


def _resolve_root_file(graph: CallGraph) -> str:
    """File that defines the entry point (for the in-place body reroute)."""
    fds = graph.defs.get(graph.root)
    if not fds:
        raise FanoutError(f"entry point {graph.root!r} has no definition")
    return fds[0].file
