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

signal_class filter — skip precision rungs on cancellation regions (Phase 2e)
----------------------------------------------------------------------------
Some regions cannot be rescued by *any* precision rung: a
``cancellation_cascade`` (chained near-equal subtractions) or a
``local_cancellation`` (``|a-b|→0``) loses its leading digits to catastrophic
cancellation, and widening the intermediates (float→ff→dd) does not restore them —
the loss is algorithmic, not representational.  Enumerating float/ff/dd on such a
region wastes one LLM shim generation + one build per rung, only to come back
measured-INERT.

The Patcher therefore consults each region's characterizer ``signal_class``
(supplied per pass via :attr:`FanoutSettings.signal_class_by_region`, keyed by the
region's ``file:line`` location) BEFORE any generation.  When the region is a
cascade / local-cancellation class and the intent is a *precision transition*
(:data:`agents.strategy.models.TRANSITION_KINDS` — not a reformulate kind), the
dispatch short-circuits to the terminal ``awaiting_algorithmic_rewrite`` status
(see :mod:`agents.patcher.dispatch`) — no LLM call, no build.  The correctness
walk's ``double→dd`` attempt is the only precision rung it enumerates for these
regions, so the short-circuit yields exactly one such cell per region (the walk
then settles as ``dd_untested`` and stops).

This is only the *filter* (skip cleanly).  The actual fix — firing the algorithmic
rewrites — is a separate, larger phase: Strategy already models the rewrite
catalog in :func:`agents.strategy.walk._rewrites_for` (``reformulate-kahan`` for
cascade, the identity catalog for local cancellation), but those intents are not
yet plumbed into the fan-out.  Wiring them there is the follow-up this status
flags as the backlog.

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
from agents.shared import region_scan

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


# Signal classes whose regions cannot be rescued by any precision rung — the loss is
# algorithmic (chained/near-equal cancellation), so widening intermediates is inert.
# Kept as a local set (avoids a dispatch→strategy import for one enum) but mirrors
# agents.strategy.models.SIGNAL_CANCELLATION_CASCADE / SIGNAL_LOCAL_CANCELLATION.
_ALGORITHMIC_REWRITE_SIGNAL_CLASSES = frozenset({
    "cancellation_cascade", "local_cancellation",
})


def awaits_algorithmic_rewrite(signal_class: str | None) -> bool:
    """True iff ``signal_class`` marks a region a precision rung cannot fix.

    Such a region is skipped by the fan-out (no rung enumeration, no build/LLM) and
    flagged ``awaiting_algorithmic_rewrite`` — it needs a Kahan/identity reformulate,
    not a wider type.  See the module docstring's *signal_class filter* section."""
    return signal_class in _ALGORITHMIC_REWRITE_SIGNAL_CLASSES


def signal_class_map(regions) -> dict[str, str]:
    """``{region_id: signal_class}`` from a characterization report's region records.

    ``regions`` is the report's per-integral ``regions`` mapping (``{region_id ->
    record}``, region_id already a ``file:line`` / ``file:start-end`` string that
    matches ``RegionTarget.location``).  Empty keys and records without a
    ``signal_class`` are dropped.  Handed to :attr:`FanoutSettings.signal_class_by_region`
    so the Patcher can consult a region's class without threading it through every
    intent."""
    out: dict[str, str] = {}
    if isinstance(regions, dict):
        items = regions.items()
    else:  # tolerate a list of records that carry their own id
        items = ((r.get("region_id") or r.get("location"), r) for r in (regions or []))
    for rid, rec in items:
        if not rid or not isinstance(rec, dict):
            continue
        sc = rec.get("signal_class")
        if sc:
            out[rid] = sc
    return out


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
    # Phase 2d: app source roots (tree headers ∪ the driver that instantiates the
    # entry template) scanned to resolve which region operands are complex containers
    # vs real scalars (agents.shared.type_resolve).  Empty → complex-container
    # promotion is disabled and the transform degrades to the pre-2d scalar-only path.
    app_source_roots: list[str] = field(default_factory=list)
    # Phase 2e signal_class filter: {region_id -> signal_class} from the report (built
    # via signal_class_map).  A precision-rung intent on a cascade / local-cancellation
    # region is short-circuited to awaiting_algorithmic_rewrite (no LLM, no build).
    # Empty → the filter is inert (fail-open; pre-2e behavior for passes that don't
    # supply the map).
    signal_class_by_region: dict = field(default_factory=dict)


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
    # Phase 2d complex-container promotion (all default to the pre-2d scalar-only
    # behavior so an old manifest re-renders identically).
    complex_type: str | None = None
    complex_tokens: list[str] = field(default_factory=list)
    complex_names: list[str] = field(default_factory=list)
    caller_complex: str | None = None
    # Blocker A (design §8): chain-carrier names whose declaration the emission layer
    # widens to the extended type (ClosureDecl below).  At THIS region's boundary a
    # carrier is neither a read-only input nor a truncating sink, so promote_region_block
    # must not seed / alias / demote it — the widened decl carries the extended value
    # end-to-end.  Defaults empty so a pre-Blocker-A manifest re-renders identically.
    closure_names: list[str] = field(default_factory=list)


@dataclass
class ClosureDecl:
    """One carrier declaration widened in a variant (Blocker A, design §7).

    A **carrier** is a variable declared OUTSIDE a chain's line set but written by
    one chain link and read by another; its declaration lives at caller precision,
    so the widened (dd) value written by the interior link truncates back at the
    decl.  Widening the type token on the decl line to the chain's internal dd type
    closes that truncation.

    Coordinates are in ORIGINAL/file line numbers (1-based), scoped to the
    variant's ``file``.  ``orig_type`` is the leading (core) type token as the
    boundary bare-decl scanner spells it (last namespace segment of a qualified
    type); ``dd_type`` the extended replacement (``quad::ddfun::ddouble`` /
    ``ddcomplex``).  Rewriting the leading type token widens every same-type
    sibling of a multi-declarator (``TMass Y, S, A;``) per the §2 conservative
    policy.  ``name`` is the carrier that motivated the record (a sibling may ride
    along); kept for forensics only.
    """

    decl_line: int
    orig_type: str
    dd_type: str
    name: str | None = None


@dataclass
class VariantSpec:
    """The deterministic recipe for one variant function.

    A variant is rebuilt from scratch on every merge by copying the original
    function's source (``orig_start..orig_end`` in ``file``), applying every
    :class:`Promote` (region → extended scalar), every :class:`ClosureDecl`
    (carrier decl line → widened type token) and every reroute (call to a child
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
    # Blocker A (design §7): carrier declarations widened to the chain's dd type at
    # emission time.  Defaults empty so a pre-Blocker-A manifest re-renders
    # identically; populated by the chain coordinator in Subtask 5.
    closure_decls: list[ClosureDecl] = field(default_factory=list)

    def to_json(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "VariantSpec":
        promotes = [Promote(**p) for p in d.get("promotes", [])]
        closure_decls = [ClosureDecl(**c) for c in d.get("closure_decls", [])]
        return cls(
            variant_name=d["variant_name"], orig_name=d["orig_name"],
            file=d["file"], orig_start=d["orig_start"], orig_end=d["orig_end"],
            promotes=promotes, reroutes=dict(d.get("reroutes", {})),
            shim_includes=list(d.get("shim_includes", [])),
            closure_decls=closure_decls)

    def merge(self, other: "VariantSpec") -> None:
        """Fold ``other`` (same variant) into this spec: union reroutes / shim
        includes and append any region promotion or carrier decl-widen not already
        present."""
        self.reroutes.update(other.reroutes)
        for inc in other.shim_includes:
            if inc not in self.shim_includes:
                self.shim_includes.append(inc)
        have = {(p.region_start, p.region_end) for p in self.promotes}
        for p in other.promotes:
            if (p.region_start, p.region_end) not in have:
                self.promotes.append(p)
        have_cd = {(c.decl_line, c.orig_type, c.dd_type) for c in self.closure_decls}
        for c in other.closure_decls:
            if (c.decl_line, c.orig_type, c.dd_type) not in have_cd:
                self.closure_decls.append(c)
                have_cd.add((c.decl_line, c.orig_type, c.dd_type))


@dataclass
class FanoutResult:
    """What one :func:`fan_out_region` call produced."""

    declared_variants: list[str]        # variant symbols the build must contain
    files_touched: list[str]
    root_edited: bool = False           # entry-point body edited in place
    in_place_region: bool = False       # region was IN the entry point (no new symbol)
    paths_enumerated: int = 0
    truncated: bool = False             # path enumeration hit its cap
    # Phase 2c promotion_no_op gate: True iff the intent's region promotion actually
    # retyped something (``promote_region_block`` returned ``promoted=True``).  When
    # False the variant/in-place body is byte-identical to the original at the region
    # — an empty promotion payload — which the Patcher turns into a terminal
    # ``promotion_no_op`` failure instead of a silent inert (bit-identical) candidate.
    promotion_applied: bool = False
    # Phase 2d-B write-boundary-truncation gate: True iff the promotion retyped the body
    # but every landing truncates back to caller precision (no wider persistent sink) —
    # an UPCAST that is numerically inert.  The Patcher turns it into a terminal
    # ``write_truncation`` (no build), the upcast analogue of ``promotion_no_op``.
    write_truncation: bool = False
    # The reads set actually used (source-derived when the intent carried none) —
    # surfaced for forensics / tests.
    reads_used: list[str] = field(default_factory=list)


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
    complex_type: str | None = None,
    app_source_roots=(),
) -> FanoutResult:
    """Realize a region intent as per-caller-path function variants.

    ``file`` / ``line_start`` / ``line_end`` locate the region (file may be a bare
    basename — resolved against the tree); ``reads`` are the region's promoted reads
    (characterizer ``region_local_vars``), ``writes`` the Fix-C write set.
    ``scalar_type`` / ``two_limb`` / ``shim_include`` describe the target precision
    (from the ff/dd/float integrator).  ``complex_type`` (+ ``app_source_roots`` for
    the template-parameter binding) enables the Phase-2d complex-container promotion —
    a region operand whose type resolves to a complex container promotes to
    ``complex_type`` (``ffcomplex`` / ``ddcomplex`` / ``Kokkos::complex<float>``)
    instead of the scalar.  Mutates the tree under ``tree_root`` in place and returns a
    :class:`FanoutResult`.  Raises :class:`FanoutError` when the region is not inside a
    known function or its function is unreachable from the entry point.
    """
    tree = Path(tree_root).resolve()
    fd = graph.enclosing_function(file, line_start)
    if fd is None:
        raise FanoutError(
            f"region {file}:{line_start}-{line_end} is not inside any known function "
            f"(call graph rooted at {graph.root!r}); cannot fan out")

    # Enclosing-function source: used to source-derive the promotion reads (Phase 2c)
    # and to classify which of them are complex containers (Phase 2d).
    func_src = "\n".join(_original_text(fd.file, fd.line_start, fd.line_end))

    # Phase 2c: source-derive the promotion reads when the intent carried none
    # (qcdloop's template regions report ``region_local_vars=[]``).
    reads = list(reads)
    if not reads:
        reads = region_scan.region_reads_from_function(
            func_src, fd.line_start, line_start, line_end)

    # Phase 2d: resolve the app's template-parameter binding (TOutput → complex, …)
    # and classify which reads are complex containers.
    complex_tokens, caller_complex = _resolve_complex_binding(
        app_source_roots, caller_type, complex_type)
    complex_names = _complex_reads(func_src, reads, complex_tokens) if complex_type else []
    ckw = dict(complex_type=complex_type, complex_tokens=list(complex_tokens),
               complex_names=complex_names, caller_complex=caller_complex)

    # Compute the promotion payload up front (independent of the variant/in-place
    # rendering path): ``promoted`` is False when the region retypes nothing — an
    # empty payload the caller turns into a terminal ``promotion_no_op``.
    region_text = "\n".join(_original_text(fd.file, line_start, line_end))
    _, promotion_applied = boundary.promote_region_block(
        region_text, reads, writes, scalar_type, caller_type, two_limb,
        complex_type=complex_type, complex_tokens=frozenset(complex_tokens),
        complex_names=frozenset(complex_names), caller_complex=caller_complex)

    # Phase 2d-B: an upcast whose promotion lands only in caller-precision stores is
    # numerically inert (truncated at the boundary).  Detect it here, upstream of the
    # variant emission + build, so the caller can terminal-fail (write_truncation)
    # instead of paying a build for a candidate whose delta == baseline.
    write_truncation = promotion_applied and boundary.write_truncation_inert(
        region_text, reads, writes, two_limb, caller_type=caller_type,
        complex_tokens=frozenset(complex_tokens), caller_complex=caller_complex)

    # --- degenerate: region is IN the entry point -> promote in place ---------
    if fd.name == graph.root:
        touched = _promote_in_place(tree, fd, line_start, line_end, reads, writes,
                                    scalar_type, two_limb, caller_type, shim_include,
                                    ckw)
        return FanoutResult(declared_variants=[], files_touched=touched,
                            root_edited=True, in_place_region=True,
                            paths_enumerated=1, promotion_applied=promotion_applied,
                            write_truncation=write_truncation,
                            reads_used=list(reads))

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
    _accumulate_region_specs(
        paths=paths, fd=fd, line_start=line_start, line_end=line_end,
        reads=reads, writes=writes, integral=integral, graph=graph,
        scalar_type=scalar_type, two_limb=two_limb, shim_include=shim_include,
        caller_type=caller_type, ckw=ckw,
        new_specs=new_specs, root_reroutes=root_reroutes, name_maps=name_maps)
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
                        paths_enumerated=len(paths), truncated=truncated,
                        promotion_applied=promotion_applied,
                        write_truncation=write_truncation, reads_used=list(reads))


def _accumulate_region_specs(
    *, paths, fd, line_start: int, line_end: int, reads, writes, integral: str,
    graph: CallGraph, scalar_type: str, two_limb: bool, shim_include: str | None,
    caller_type: str, ckw: dict,
    new_specs: dict[str, dict[str, "VariantSpec"]],
    root_reroutes: dict[str, str], name_maps: list[dict[str, str]],
    closure_names=(),
) -> None:
    """Accumulate the variant specs + root reroutes for ONE region's caller paths.

    Extracted from :func:`fan_out_region` (behavior-preserving) so the Phase-2f
    chain coordinator (:mod:`agents.patcher.chain_promote`) can call it once per
    chain region into SHARED accumulators before a single ``_merge_into_file`` per
    file.  Because ``variant_names_for_path`` is a pure function of ``(path,
    integral)``, two regions reaching the same ancestor produce the same variant
    name, so their :class:`Promote`s / reroutes merge onto one :class:`VariantSpec`
    via ``setdefault`` — that is what makes multi-region chain promotion coherent.

    ``paths`` are the already-enumerated caller paths for ``fd`` (root-first);
    ``ckw`` is the complex-container kwargs dict (``complex_type`` / ``complex_tokens``
    / ``complex_names`` / ``caller_complex``).  Mutates ``new_specs`` /
    ``root_reroutes`` / ``name_maps`` in place.
    """
    complex_type = ckw.get("complex_type")
    complex_tokens = ckw.get("complex_tokens", [])
    complex_names = ckw.get("complex_names", [])
    caller_complex = ckw.get("caller_complex")
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
                    scalar_type=scalar_type, two_limb=two_limb, caller_type=caller_type,
                    complex_type=complex_type, complex_tokens=list(complex_tokens),
                    complex_names=list(complex_names), caller_complex=caller_complex,
                    closure_names=list(closure_names)))
                if shim_include and shim_include not in spec.shim_includes:
                    spec.shim_includes.append(shim_include)


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

    Copies the original function, applies every region promotion and every carrier
    decl-widen (descending by line so earlier edits do not shift later
    coordinates), reroutes calls to child variants, and renames the definition +
    self-calls ``orig_name -> variant_name``.

    Region promotions (:class:`Promote`, a multi-line block replacement) and
    carrier decl-widens (:class:`ClosureDecl`, a single-line type-token rewrite,
    Blocker A §7) share ONE descending-line-order pass.  Sorting both by their
    starting line, highest first, guarantees each edit's file coordinates are still
    valid when it runs regardless of the length delta an earlier (lower-line) edit
    would introduce — a carrier decl above a promoted region is rewritten after the
    region has already been replaced, so its ``decl_line`` never shifts.
    """
    lines = _original_text(spec.file, spec.orig_start, spec.orig_end)
    # Descending by start line; both edit kinds carry a file-absolute start.  Promote
    # is tagged 0 and ClosureDecl 1 so that, in the degenerate case of a decl line
    # equal to a region start (should not occur — a widened carrier is by definition
    # outside the chain line set), the region replacement runs first deterministically.
    edits = ([(p.region_start, 0, p) for p in spec.promotes]
             + [(c.decl_line, 1, c) for c in spec.closure_decls])
    for start, kind, e in sorted(edits, key=lambda t: (t[0], t[1]), reverse=True):
        if kind == 0:
            p = e
            local_s = p.region_start - spec.orig_start
            local_e = p.region_end - spec.orig_start
            if local_s < 0 or local_e >= len(lines) or local_s > local_e:
                raise FanoutError(
                    f"promote region {p.region_start}-{p.region_end} out of range for "
                    f"{spec.orig_name} [{spec.orig_start}-{spec.orig_end}]")
            region_text = "\n".join(lines[local_s:local_e + 1])
            block, _ = boundary.promote_region_block(
                region_text, p.reads, p.writes, p.scalar_type, p.caller_type,
                p.two_limb, complex_type=p.complex_type,
                complex_tokens=frozenset(p.complex_tokens),
                complex_names=frozenset(p.complex_names),
                caller_complex=p.caller_complex,
                closure_names=frozenset(p.closure_names))
            lines = lines[:local_s] + block + lines[local_e + 1:]
        else:
            c = e
            local = c.decl_line - spec.orig_start
            if local < 0 or local >= len(lines):
                raise FanoutError(
                    f"carrier decl line {c.decl_line} out of range for "
                    f"{spec.orig_name} [{spec.orig_start}-{spec.orig_end}]")
            widened = boundary.widen_decl_type_line(lines[local], c.orig_type,
                                                    c.dd_type)
            if widened is not None:
                lines[local] = widened

    text = "\n".join(lines)
    rename_map = {spec.orig_name: spec.variant_name}
    rename_map.update(spec.reroutes)
    return _rewrite_calls(text, rename_map)


def _topo_order(specs: dict[str, VariantSpec]) -> list[VariantSpec]:
    """Order variants so a callee variant precedes every caller variant.

    A variant ``A`` that reroutes a call to variant ``B`` (``B`` in ``A.reroutes``)
    references ``B`` by its *qualified* name (``ql::B<...>``); qualified name lookup
    inside a template body is NOT deferred to instantiation, so ``B`` must be
    *defined earlier* in the namespace.  Callee variants are always deeper on the
    path than their callers, so this DFS post-order (dependencies first) is a valid
    topological emission order; ties broken by name for determinism.
    """
    names = set(specs)
    deps = {a: sorted(v for v in specs[a].reroutes.values() if v in names)
            for a in names}
    ordered: list[str] = []
    visited: set[str] = set()

    def visit(a: str) -> None:
        if a in visited:
            return
        visited.add(a)
        for b in deps[a]:
            visit(b)
        ordered.append(a)

    for a in sorted(names):
        visit(a)
    return [specs[a] for a in ordered]


def _render_block(specs: dict[str, VariantSpec]) -> list[str]:
    """Render the full per-file fan-out block (manifest comment + variant defs)."""
    ordered = _topo_order(specs)
    # manifest key order stays name-sorted (stable spec record), independent of the
    # emission order (which is topological so callee variants are defined first).
    manifest = {"variants": [specs[k].to_json() for k in sorted(specs)]}
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
                      shim_include: str | None, ckw: dict | None = None) -> list[str]:
    """Promote a region that lives *in the entry point* in place (no rename)."""
    path = Path(fd.file)
    lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    region_text = "\n".join(lines[line_start - 1:line_end])
    ckw = ckw or {}
    block, promoted = boundary.promote_region_block(
        region_text, reads, writes, scalar_type, caller_type, two_limb,
        complex_type=ckw.get("complex_type"),
        complex_tokens=frozenset(ckw.get("complex_tokens", [])),
        complex_names=frozenset(ckw.get("complex_names", [])),
        caller_complex=ckw.get("caller_complex"),
        closure_names=frozenset(ckw.get("closure_names", [])))
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


def _resolve_complex_binding(app_source_roots, caller_type: str,
                             complex_type: str | None):
    """Return ``(complex_tokens, caller_complex)`` for the app (Phase 2d).

    ``complex_tokens`` are the type-name tokens that denote a complex container in a
    region's scope (the complex-bound template-parameter names ∪ the literal
    ``complex``); ``caller_complex`` is the concrete caller complex spelling
    (``Kokkos::complex<double>``) a pre-declared complex write demotes back to.
    Returns ``(frozenset(), None)`` when complex promotion is disabled (no
    ``complex_type``) or the binding cannot be resolved (no ``app_source_roots``), so
    the transform degrades to the pre-2d scalar-only path.
    """
    if not complex_type or not app_source_roots:
        return frozenset(), None
    from agents.shared import type_resolve as tr
    bindings = tr.resolve_bindings(app_source_roots, caller_type)
    tokens = tr.complex_type_tokens(bindings)
    caller_complex = next(
        (c for c in bindings.values() if tr.classify_concrete_type(c) == "complex"),
        None)
    return tokens, caller_complex


def _complex_reads(func_src: str, reads: list[str], complex_tokens) -> list[str]:
    """Subset of ``reads`` whose declared type (in ``func_src``) is a complex container."""
    if not complex_tokens:
        return []
    complex_names = region_scan.region_complex_read_names(func_src, complex_tokens)
    return [r for r in reads if r in complex_names]


def _resolve_root_file(graph: CallGraph) -> str:
    """File that defines the entry point (for the in-place body reroute)."""
    fds = graph.defs.get(graph.root)
    if not fds:
        raise FanoutError(f"entry point {graph.root!r} has no definition")
    return fds[0].file
