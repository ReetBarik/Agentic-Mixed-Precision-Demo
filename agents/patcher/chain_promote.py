"""Phase 2f — coordinated multi-region, multi-file dd promotion of a whole chain.

Where :func:`agents.patcher.fanout.fan_out_region` promotes ONE region per call,
this module promotes an entire **cancellation-cascade chain** — a set of
``(file, line)`` regions spanning several functions — to double-double as a single
coordinated envelope.  The chain-internal type contract is dd; the chain *boundary*
contract is caller precision (reads widened at entry, writes truncated at exit) —
the deterministic realization of the chain integrator's C9 rule.

Mechanism (all reused from the region fan-out; the manifest is the source of truth):

* For each chain region, resolve its enclosing function, derive reads/writes, and
  compute the per-region promotion via ``boundary.promote_region_block``.
* Regions in the entry point are promoted in place; the rest accumulate variant
  specs via the shared :func:`agents.patcher.fanout._accumulate_region_specs` into
  ONE ``new_specs`` / ``root_reroutes`` set across the whole chain.  Because variant
  names are a pure function of ``(path, integral)``, two chain regions reaching the
  same ancestor land on the SAME :class:`VariantSpec` — the caller variant carries
  both its own region promotion AND the reroute into the callee variant, so the
  chain-internal calls widen bottom-up (``_topo_order`` emits callee-before-caller).
* One ``_merge_into_file`` per touched file writes the AMP-FANOUT block.
* Variants are integral-scoped (``ddilog_B12`` vs ``ddilog_B14``); the ORIGINAL
  shared helper is never edited — a correctness requirement (Item 7 §3: promoting
  the original ``ddilog`` would re-measure benign B8/B9 floors).

Chain-scope gates (the envelope, not individual links):

* ``chain_promotion_no_op`` — fires iff NO region in the chain promotes anything.
  A single empty-payload link is NOT gated (its neighbours may promote, and its
  intra-chain writes stay wide for the next link to read).
* ``chain_write_truncation`` — fires iff the chain's OUTERMOST region (shallowest
  enclosing function on the call graph — the last landing before the value returns
  to the shared driver) truncates every widened write back to caller precision with
  no wider persistent sink.  Intra-chain widened writes are EXEMPT (consumed by the
  next link) — this is exactly the reasoning 2d-B's per-region gate would get wrong
  at chain scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agents.integrator_base import boundary
from agents.patcher.call_graph import CallGraph
from agents.patcher.fanout import (
    FanoutError, Promote, VariantSpec,
    _accumulate_region_specs, _complex_reads, _merge_into_file, _original_text,
    _pick_def, _promote_in_place, _reroute_in_function, _resolve_complex_binding,
    _resolve_root_file,
)
from agents.patcher.variant_naming import assert_no_collisions
from agents.shared import region_scan


@dataclass
class ChainManifest:
    """The input to :func:`chain_promote`: one cascade chain to widen to dd.

    ``lines`` are the chain's ``(file, line_start, line_end)`` regions (1-based
    inclusive), spanning one or more functions.  ``entry_point`` is the integral's
    call-graph root (matches the graph's ``root``); ``integral`` is the variant
    naming suffix (per-integral — ``B12``).  ``reads_by_region`` / ``writes_by_region``
    optionally override the source-derived operand sets, keyed by the ``(file,
    line_start, line_end)`` tuple; when absent they are derived from source
    (``region_scan``), exactly as the single-region fan-out does for reads.
    """

    chain_id: str
    integral: str
    entry_point: str
    lines: list[tuple[str, int, int]]
    reads_by_region: dict[tuple[str, int, int], list[str]] = field(default_factory=dict)
    writes_by_region: dict[tuple[str, int, int], list[str]] = field(default_factory=dict)


@dataclass
class ChainFanoutResult:
    """What one :func:`chain_promote` call produced (chain analogue of FanoutResult)."""

    declared_variants: list[str]
    files_touched: list[str]
    root_edited: bool = False
    in_place_regions: int = 0
    paths_enumerated: int = 0
    truncated: bool = False
    # Chain-scope 2c: True iff ANY region in the chain retyped something.  False =>
    # the whole chain's promotable payload is empty -> terminal promotion_no_op.
    promotion_applied: bool = False
    # Chain-scope 2d-B: True iff the OUTERMOST region widens but truncates every
    # landing back to caller precision with no wider persistent sink -> the chain's
    # output boundary is numerically inert -> terminal write_truncation.
    write_truncation: bool = False
    # Entry-point body reroutes applied (child -> variant), for silent-bypass telemetry.
    reroutes: dict[str, str] = field(default_factory=dict)
    # Per-region reads actually used (source-derived when not supplied), keyed by region.
    reads_used: dict[tuple[str, int, int], list[str]] = field(default_factory=dict)


def chain_promotion_no_op(per_region_promoted: list[bool]) -> bool:
    """Chain-scope 2c gate: fires iff NO region in the chain promotes anything.

    A single empty-payload link is NOT gated — the chain is widened as an envelope,
    so its neighbours may promote and its intra-chain writes stay wide for the next
    link.  Only a chain whose ENTIRE payload is empty is a true no-op.
    """
    return not any(per_region_promoted)


def chain_write_truncation(*, outermost_region_text: str, outermost_reads: list[str],
                           outermost_writes: list[str], two_limb: bool,
                           caller_type: str = "double",
                           complex_tokens=frozenset(), caller_complex=None) -> bool:
    """Chain-scope 2d-B gate: applied to the chain's OUTERMOST region only.

    Reuses the per-region :func:`agents.integrator_base.boundary.write_truncation_inert`
    on the last landing before the value returns to the shared driver.  Intra-chain
    regions are deliberately NOT checked — their widened writes are consumed by the
    next link (a wider persistent sink exists), which is exactly the condition that
    flips the per-region gate's reasoning at chain scope.
    """
    return boundary.write_truncation_inert(
        outermost_region_text, outermost_reads, outermost_writes, two_limb,
        caller_type=caller_type, complex_tokens=complex_tokens,
        caller_complex=caller_complex)


def chain_promote(*, manifest: ChainManifest, graph: CallGraph,
                  tree_root: str | Path, scalar_type: str, two_limb: bool,
                  shim_include: str | None, caller_type: str = "double",
                  complex_type: str | None = None, max_paths: int = 1024,
                  app_source_roots=()) -> ChainFanoutResult:
    """Widen a whole cascade chain to ``scalar_type`` (dd) as one coordinated envelope.

    Mutates the tree under ``tree_root`` in place (per-file AMP-FANOUT blocks + the
    entry-point body reroute) and returns a :class:`ChainFanoutResult` carrying the
    chain-scope 2c/2d gate verdicts.  Raises :class:`FanoutError` if a chain region
    is not inside a known function or its function is unreachable from the entry
    point.
    """
    tree = Path(tree_root).resolve()
    complex_tokens, caller_complex = _resolve_complex_binding(
        app_source_roots, caller_type, complex_type)

    new_specs: dict[str, dict[str, VariantSpec]] = {}
    root_reroutes: dict[str, str] = {}
    name_maps: list[dict[str, str]] = []

    per_region_promoted: list[bool] = []
    region_meta: list[dict] = []       # for outermost-by-depth gate computation
    reads_used: dict[tuple[str, int, int], list[str]] = {}
    touched: set[str] = set()
    in_place_regions = 0
    paths_enumerated = 0
    truncated = False

    for (file, ls, le) in manifest.lines:
        fd = graph.enclosing_function(file, ls)
        if fd is None:
            raise FanoutError(
                f"chain {manifest.chain_id}: region {file}:{ls}-{le} is not inside any "
                f"known function (call graph rooted at {graph.root!r}); cannot promote")
        rkey = (file, ls, le)

        func_src = "\n".join(_original_text(fd.file, fd.line_start, fd.line_end))
        region_text = "\n".join(_original_text(fd.file, ls, le))

        # Derive reads/writes from source when the manifest carries none (qcdloop's
        # template regions report empty operand sets — same recovery the fan-out uses).
        reads = list(manifest.reads_by_region.get(rkey, []))
        if not reads:
            reads = region_scan.region_reads_from_function(func_src, fd.line_start, ls, le)
        writes = list(manifest.writes_by_region.get(rkey, []))
        if not writes:
            writes = region_scan.region_writes_from_source(region_text)
        reads_used[rkey] = list(reads)

        complex_names = _complex_reads(func_src, reads, complex_tokens) if complex_type else []
        ckw = dict(complex_type=complex_type, complex_tokens=list(complex_tokens),
                   complex_names=complex_names, caller_complex=caller_complex)

        _, promoted = boundary.promote_region_block(
            region_text, reads, writes, scalar_type, caller_type, two_limb,
            complex_type=complex_type, complex_tokens=frozenset(complex_tokens),
            complex_names=frozenset(complex_names), caller_complex=caller_complex)
        per_region_promoted.append(promoted)

        if fd.name == graph.root:
            # Region lives in the entry point: promote in place (no new symbol).
            _promote_in_place(tree, fd, ls, le, reads, writes, scalar_type,
                              two_limb, caller_type, shim_include, ckw)
            in_place_regions += 1
            touched.add(fd.file)
            depth = 0
        else:
            paths, tr = graph.enumerate_paths(fd.name, max_paths=max_paths)
            if not paths:
                raise FanoutError(
                    f"chain {manifest.chain_id}: function {fd.name!r} (containing region "
                    f"{file}:{ls}-{le}) is unreachable from entry point {graph.root!r}")
            truncated = truncated or tr
            paths_enumerated += len(paths)
            _accumulate_region_specs(
                paths=paths, fd=fd, line_start=ls, line_end=le,
                reads=reads, writes=writes, integral=manifest.integral, graph=graph,
                scalar_type=scalar_type, two_limb=two_limb, shim_include=shim_include,
                caller_type=caller_type, ckw=ckw,
                new_specs=new_specs, root_reroutes=root_reroutes, name_maps=name_maps)
            depth = min(len(p) for p in paths) - 1     # hops from root (shallower = outer)

        region_meta.append(dict(depth=depth, region_text=region_text,
                                reads=reads, writes=writes, promoted=promoted))

    assert_no_collisions(name_maps)

    # --- one merge per touched file (the whole chain's specs for that file) ----
    declared: list[str] = []
    for file_key, specs in new_specs.items():
        _merge_into_file(Path(file_key), specs)
        touched.add(file_key)
        declared.extend(specs.keys())

    # --- reroute the entry-point body once (union of every region's root reroute) --
    root_edited = in_place_regions > 0
    if root_reroutes:
        root_file = _resolve_root_file(graph)
        root_fd = _pick_def(graph, graph.root)
        if _reroute_in_function(Path(root_file), root_fd, root_reroutes):
            touched.add(root_file)
        root_edited = True

    # --- chain-scope gates ----------------------------------------------------
    promotion_applied = any(per_region_promoted)
    write_truncation = False
    if promotion_applied and region_meta:
        outer = min(region_meta, key=lambda m: m["depth"])
        write_truncation = chain_write_truncation(
            outermost_region_text=outer["region_text"],
            outermost_reads=outer["reads"], outermost_writes=outer["writes"],
            two_limb=two_limb, caller_type=caller_type,
            complex_tokens=frozenset(complex_tokens), caller_complex=caller_complex)

    return ChainFanoutResult(
        declared_variants=sorted(set(declared)),
        files_touched=sorted(touched), root_edited=root_edited,
        in_place_regions=in_place_regions, paths_enumerated=paths_enumerated,
        truncated=truncated, promotion_applied=promotion_applied,
        write_truncation=write_truncation, reroutes=dict(root_reroutes),
        reads_used=reads_used)
