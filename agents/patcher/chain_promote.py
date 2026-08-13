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
* ``chain_write_truncation`` — fires iff any INTERIOR region of the chain truncates
  every widened write back to caller precision with no wider persistent sink.  The
  chain's OUTERMOST region(s) — those at the shallowest depth on the call graph, the
  last landing before the value returns to the shared driver — are EXEMPT: their
  store to the caller-precision output IS the chain's *designed* exit boundary (the
  final dd result rounded down to caller precision after the chain has done its work
  at dd), not evidence of inertness.  An interior write that truncates back to caller
  precision, by contrast, injects double roundoff *between* chain links and genuinely
  breaks the chain — that is what this gate catches.  Applying 2d-B's per-region gate
  to the outermost region is exactly the reasoning it would get wrong at chain scope:
  the per-region detector reads the exit-boundary rounding as inertness, when at chain
  scope it is the intended output handoff (Reet 2026-07-25, Fix 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agents.integrator_base import boundary
from agents.patcher.call_graph import CallGraph
from agents.patcher.clonable_leaf import clonable_leaf, is_dd_boundary
from agents.patcher.fanout import (
    ClosureDecl, FanoutError, Promote, ReturnWiden, VariantSpec,
    _accumulate_region_specs, _complex_reads, _merge_into_file, _merge_return_widen,
    _original_text, _pick_def, _promote_in_place, _reroute_in_function,
    _resolve_complex_binding, _resolve_root_file,
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
    # Chain-scope 2d-B: True iff some INTERIOR region widens but truncates every
    # landing back to caller precision with no wider persistent sink -> an intra-chain
    # write injecting double roundoff between links -> terminal write_truncation.  The
    # chain's OUTERMOST region is EXEMPT (its truncation is the designed exit boundary).
    write_truncation: bool = False
    # Entry-point body reroutes applied (child -> variant), for silent-bypass telemetry.
    reroutes: dict[str, str] = field(default_factory=dict)
    # Per-region reads actually used (source-derived when not supplied), keyed by region.
    reads_used: dict[tuple[str, int, int], list[str]] = field(default_factory=dict)
    # Blocker A carrier-closure terminals (BLOCKER_A_CARRIER_DESIGN.md §9).  Set BEFORE
    # any tree mutation when a strict carrier's decl cannot be widened, so the chain is
    # abandoned (no variants emitted) rather than emitted with a truncating interior
    # seam that the 2d-B gate would then reject.  ``carrier_detail`` names the offending
    # carrier(s) for the terminal Gen message.
    chain_carrier_unwidenable: bool = False
    chain_carrier_external: bool = False
    # CLOSURE_SCOPED_CHAINS_DESIGN §2.4 / §3.2 iii — set BEFORE any tree mutation when a
    # *destination escape* (a rule-(b) write to shared state, or a non-benign extract)
    # materially severs a carried value's dd flow to a designed exit.  Like the two
    # carrier terminals above, the chain is abandoned (no variants emitted) rather than
    # emitted with a truncating seam the interior gate would then reject.
    chain_closure_escapes: bool = False
    carrier_detail: str = ""
    # The closure names actually threaded into the promotion (forensics/tests).
    closure_names: list[str] = field(default_factory=list)
    # --- rule (d): leaf-callee promotion (LEAF_CALLEE_PROMOTION_DESIGN.md §2.8) ----
    # Set BEFORE any tree mutation when rule (d) would grow F past the §2.8 circuit
    # breaker (>8 frames or recursion depth >3): abandon the chain cleanly (graceful
    # degradation), ``oversized_detail`` naming the offending leaf.
    chain_closure_oversized: bool = False
    oversized_detail: str = ""
    # Rule-(d) discovery result (recorded, NOT emitted — L3 materialises the clones):
    # ``leaf_reroutes`` maps each promoted leaf's ORIGINAL name to its per-integral
    # clone name (``Lnrat`` -> ``Lnrat_B10``); ``leaf_frames`` is the ordered leaf-name
    # list added to F; ``leaf_escapes`` is ``(name, reason)`` per leaf the predicate
    # refused (a narrowed ``chain_closure_escapes`` frontier, §4.2).
    leaf_reroutes: dict = field(default_factory=dict)
    leaf_frames: list = field(default_factory=list)
    leaf_escapes: list = field(default_factory=list)


def chain_promotion_no_op(per_region_promoted: list[bool]) -> bool:
    """Chain-scope 2c gate: fires iff NO region in the chain promotes anything.

    A single empty-payload link is NOT gated — the chain is widened as an envelope,
    so its neighbours may promote and its intra-chain writes stay wide for the next
    link.  Only a chain whose ENTIRE payload is empty is a true no-op.
    """
    return not any(per_region_promoted)


def _designed_exit_kind(kind: str) -> bool:
    """Whether a designed-exit ``kind`` exempts its landing line from the interior
    write-truncation gate (CLOSURE_SCOPED_CHAINS_DESIGN.md §3.2).

    * ``kernel_output`` / ``out_param`` (clause i) and ``extract`` (clause iii, already
      filtered to *benign* extracts by the closure) — a carried value leaving the chain
      function set at caller precision at its designed landing: exempt.
    * ``return_widened`` (clause ii) — a return in a frame whose variant return type the
      closure WIDENED under rule (c): the value carries dd across the return, no
      truncation, so the landing is exempt.  A plain ``return`` (the frame's return type
      was NOT widened — rule (c) refused or did not reach it) still truncates to caller
      precision and is NOT exempt, so the gate keeps rejecting it (§3.3: B10 with rule
      (c) disabled must still reject at :707).
    """
    return kind in ("kernel_output", "out_param", "extract", "return_widened")


def chain_write_truncation(region_meta: list[dict], *, two_limb: bool,
                           caller_type: str = "double",
                           complex_tokens=frozenset(), caller_complex=None,
                           closure_names=frozenset(), designed_exits=()) -> bool:
    """Chain-scope 2d-B gate, reformulated to the §3.2 designed-exit predicate.

    ``region_meta`` is the per-region list built by :func:`chain_promote`, each entry
    carrying ``depth`` (hops from the entry point — shallower = outer), ``span``
    ``(file, ls, le)``, ``region_text``, ``reads`` and ``writes``.  The gate fires iff a
    carried value is truncated to caller precision at a landing that is NOT a designed
    exit; a region is skipped when its landing is designed:

    * **Non-regression net (outermost depth):** the chain's shallowest region(s) store
      to the caller-precision output — the chain's designed exit boundary — so they are
      skipped exactly as before (Reet 2026-07-25, Fix 1).
    * **Designed-exit region exemption (§3.2):** an interior region every one of whose
      lines is a designed-exit landing (``designed_exits`` filtered by
      :func:`_designed_exit_kind`) is the value's intended projection to caller precision
      and is skipped.  Conservative — a region with any non-designed line is still
      checked, so a genuine interior severance cannot slip through (§7 falsification).

    Every remaining INTERIOR region is checked with the per-region
    :func:`agents.integrator_base.boundary.write_truncation_inert` (closure names excluded
    from its truncating-sink scan, so a widened carrier is not misread as inertness); the
    gate fires iff ANY such region trips it.

    Returns ``False`` for a single-region chain (nothing is interior) — a lone region is
    its own designed exit boundary; the outermost exemption covers it.
    """
    if not region_meta:
        return False
    designed_lines = {(f, l) for (f, l, k, _cv, _d) in designed_exits
                      if _designed_exit_kind(k)}
    outermost_depth = min(m["depth"] for m in region_meta)
    for m in region_meta:
        if m["depth"] == outermost_depth:
            continue  # non-regression net: outermost store is the designed exit boundary
        f, ls, le = m["span"]
        if designed_lines and all((f, L) in designed_lines
                                  for L in range(ls, le + 1)):
            continue  # designed-exit region — every line is an intended caller-precision landing
        if boundary.write_truncation_inert(
                m["region_text"], m["reads"], m["writes"], two_limb,
                caller_type=caller_type, complex_tokens=complex_tokens,
                caller_complex=caller_complex, closure_names=closure_names):
            return True
    return False


# --------------------------------------------------------------------------- #
# Blocker A — carrier closure (BLOCKER_A_CARRIER_DESIGN.md §2, §4)
#
# A *carrier* is a variable declared OUTSIDE a chain's line set, written by one
# interior chain line and read by another — the value it carries crosses a
# chain-line boundary at caller precision, so the interior write truncates the
# widened (dd) value back to double and the 2d-B :func:`chain_write_truncation`
# gate correctly rejects the patch.  The fix (Subtasks 3-7) widens the carrier's
# DECLARATION alongside the line bodies.  This module owns only the *analysis*:
# given a chain's line set, classify its carriers as widenable / unwidenable /
# external.  It computes nothing that mutates the tree.
# --------------------------------------------------------------------------- #


@dataclass
class CarrierClosure:
    """The value closure of one chain (CLOSURE_SCOPED_CHAINS_DESIGN.md §2).

    This object carries BOTH views of the closure:

    **The Fix-A-equivalent compat subset** (rule (a) restricted to reads on another
    *chain* line — the Blocker-A strict-carrier test).  As of Subtask 1b these fields
    are RETAINED for the ``unwidenable_reasons`` / ``external_reasons`` terminals and for
    forensics/tests only; production emission now threads the enlarged ``closure_names``
    / ``closure_decl_widens`` (below) end-to-end.

    * ``widenable`` — ``(file, decl_line, name, dd_type)`` per name whose declaration
      the emission layer widens to the chain's internal dd type (carriers + their
      same-line multi-declarator siblings, §2).
    * ``unwidenable_reasons`` — ``(name, reason)`` per strict carrier whose decl is a
      function parameter (terminal ``chain_carrier_unwidenable``).
    * ``external_reasons`` — ``(name, reason)`` per strict carrier whose decl is a
      global / class member / output container (terminal ``chain_carrier_external``).
    * ``decl_widens`` — one record PER widened decl LINE, ``(file, decl_line,
      orig_type, dd_type, representative_name)`` (widening the leading type token
      widens every same-type sibling in one edit, §2).

    **The enlarged value closure** (rules (a) generalised to reads on *any* line in
    the frame, plus (b) forward-flow — CLOSURE_SCOPED_CHAINS_DESIGN.md §2.3).  Not yet
    consumed; Subtask 1b wires it into the interior gate.

    * ``closure_widenable`` / ``closure_decl_widens`` — the enlarged analogues of
      ``widenable`` / ``decl_widens`` (rule (a) local carriers + rule (b) locals).
    * ``designed_exits`` — ``(file, line, kind, carried_values, detail)`` rule (b)
      forward-flow that terminates at the chain's designed landing at caller precision:
      ``kind`` is ``"return"`` | ``"out_param"`` | ``"kernel_output"``, and
      ``carried_values`` is the ``frozenset[str]`` of carried values that land there
      (the gate's benign-extract procedure keys on this, §3.2).  These do NOT
      cross-frame propagate — that is rule (c), Subtask 2a/2b.
    * ``source_escapes`` — ``(name, reason)`` per carried value flowing INTO a callee
      not in the chain function set as an argument (``ga34* -> ql::Real``, §2.4).
      Purely diagnostic: a source escape does NOT block the escaping name from the
      closure (``ga34*`` stays widenable), and does NOT on its own fire the terminal.
    * ``destination_escapes`` — ``(name, reason)`` per rule-(b) WRITE that lands in
      shared state (global / class member / output container) not a per-integral kernel
      output, AND per rule-(b) local write whose producing source-line escaped.  A
      destination escape BLOCKS that name from widening and, when it materially blocks
      dd flow to a designed exit, fires ``chain_closure_escapes``.
    * ``blocking_escapes`` — the subset of ``destination_escapes`` that materially
      severs a carried value not reaching any designed exit (§2.4); non-empty drives the
      terminal.
    * ``escape_reasons`` — compat union of ``source_escapes`` + ``destination_escapes``
      (Subtask 1a shape); retained for existing readers, superseded by the split.
    * ``return_widens`` — rule (c) variant-return-type widenings as
      :class:`~agents.patcher.fanout.ReturnWiden` records.  Subtask 2a upgrades this
      from a ``frozenset`` to a ``list[ReturnWiden]`` (a proper collection the
      emission layer attaches to variants) but leaves it EMPTY — rule (c)'s algorithm
      (:func:`_expand_value_closure` cross-frame propagation) populates it in Subtask 2b.
    """

    widenable: list[tuple[str, int, str, str]] = field(default_factory=list)
    unwidenable_reasons: list[tuple[str, str]] = field(default_factory=list)
    external_reasons: list[tuple[str, str]] = field(default_factory=list)
    decl_widens: list[tuple[str, int, str, str, str]] = field(default_factory=list)
    # --- enlarged value closure (rules (a)+(b), §2.3) ---------------------------
    closure_widenable: list[tuple[str, int, str, str]] = field(default_factory=list)
    closure_decl_widens: list[tuple[str, int, str, str, str]] = field(default_factory=list)
    designed_exits: list[tuple[str, int, str, frozenset, str]] = field(default_factory=list)
    source_escapes: list[tuple[str, str]] = field(default_factory=list)
    destination_escapes: list[tuple[str, str]] = field(default_factory=list)
    # The subset of ``destination_escapes`` that MATERIALLY blocks dd flow: a carried
    # value severed here that does not reach ANY designed exit elsewhere (§2.4).  Non-empty
    # => the ``chain_closure_escapes`` terminal fires; a destination escape whose severed
    # values all still reach a designed exit is recorded but does NOT block.
    blocking_escapes: list[tuple[str, str]] = field(default_factory=list)
    # Subtask 2a: a proper collection of ReturnWiden records (was a frozenset in 1a).
    # Stays EMPTY here; rule (c)'s cross-frame propagation (Subtask 2b) populates it.
    return_widens: list = field(default_factory=list)
    # Carriers whose DECL lies ON a chain line (so the body transform owns widening the
    # decl — they get NO ``closure_decl_widens`` record) but which are still carried
    # values the boundary transform must keep WIDE at their producing/consuming chain
    # regions (Li2omx2's ``lnarg``/``lnomarg`` @702/703, read by the dd cancellation
    # @704).  Threaded into ``closure_names`` so ``promote_region_block`` does not demote
    # them at their decl-init landing.
    closure_body_names: set = field(default_factory=set)
    # --- rule (d): leaf-callee promotion (LEAF_CALLEE_PROMOTION_DESIGN.md §1.1) ---
    # A leaf callee ``g`` (called from a chain line, no chain line in its own body) that
    # the ``clonable_leaf`` predicate accepts is added to F as a new *cloned* frame.
    # ``leaf_reroutes`` maps the ORIGINAL callee name to its per-integral clone name
    # (``Lnrat`` -> ``Lnrat_B10``); L3 materialises the clone body + reroutes the call
    # sites — this subtask only RECORDS the reroute (design §8 non-goal: no emission).
    # ``leaf_frames`` is the ordered list of original leaf names pulled into F.
    leaf_reroutes: dict = field(default_factory=dict)
    leaf_frames: list = field(default_factory=list)
    # A leaf on a chain line the predicate REFUSED (not clonable): a narrowed
    # ``chain_closure_escapes`` frontier (§4.2) — ``(name, reason)`` per refused leaf.
    leaf_escapes: list = field(default_factory=list)
    # §2.8 circuit breaker: rule (d) would grow F past 8 frames or recurse past depth 3.
    chain_closure_oversized: bool = False
    oversized_detail: str = ""

    @property
    def escape_reasons(self) -> list[tuple[str, str]]:
        """Compat union of ``source_escapes`` + ``destination_escapes`` (Subtask 1a
        shape).  Retained for existing readers; new code should consult the split."""
        return sorted(set(self.source_escapes) | set(self.destination_escapes))

    @property
    def carrier_names(self) -> set[str]:
        """Fix-A compat view: names the emission layer widens today (carriers +
        siblings).  A filter of the enlarged closure to the strict-carrier subset —
        every current consumer keeps receiving exactly this (Subtask 1b retires it)."""
        return {name for _f, _l, name, _t in self.widenable}

    @property
    def closure_names(self) -> set[str]:
        """Every name in the ENLARGED value closure (rules (a) generalised + (b)),
        plus chain-line decl-init carriers whose decl the body transform owns."""
        return ({name for _f, _l, name, _t in self.closure_widenable}
                | set(self.closure_body_names))


@dataclass
class LeafPromotionContext:
    """Everything rule (d) (leaf-callee promotion) needs — supplied ONLY when a
    caller opts leaf promotion in (LEAF_CALLEE_PROMOTION_DESIGN.md §1.1, L2).

    When this is ``None`` (the default for every pre-L2 caller, including
    :func:`chain_promote`'s production path), rule (d) is inert and the closure is
    byte-identical to before — the STOP #B guarantee.  When supplied,
    :func:`_expand_value_closure` runs the rule-(d) frame-discovery fixed point,
    growing ``F`` with cloned leaf frames before rules (a)/(b)/(c) run over the
    enlarged set.  L2 records the discovery (``leaf_reroutes`` / ``leaf_frames`` /
    ``leaf_escapes`` on the :class:`CarrierClosure`); L3 owns emission.

    * ``graph`` — the :class:`~agents.patcher.call_graph.CallGraph`, for resolving a
      leaf's :class:`~agents.patcher.call_graph.FuncDef` extent + clause-(3) overload
      count.
    * ``surface`` — the L1′ :class:`~agents.integrator_base.shallow_wrapper.VendoredSurface`.
    * ``is_class1_synthesizable`` — the pure L1′ manifest query (clause 2(ii)).
    * ``source_instantiates_at_dd`` — ``name -> bool`` (clause 2(iii)).
    * ``resolve_primary_body`` — ``name -> str | None`` primary-body lookup, for the
      predicate's clause-(2) classification and recursion.
    * ``integral`` — the per-integral clone suffix (``B10`` -> ``Lnrat_B10``).
    * ``source_provides_dd`` — optional ``name -> bool`` (Resolution A, L3-resume):
      does the analysed source ALREADY define a concrete dd overload for this Class-1
      wrapper?  When it does, the wrapper is a dd *boundary* (clause 2(v)) read from
      source, not a synth dep — the pipeline emits no ODR-colliding overlay (STOP #K).
      ``None`` (the default) disables clause (v): pre-enrichment behaviour, byte-identical.
    * ``max_frames`` / ``max_depth`` — the §2.8 circuit-breaker thresholds.
    """

    graph: object
    surface: object
    is_class1_synthesizable: object
    source_instantiates_at_dd: object
    resolve_primary_body: object
    integral: str
    source_provides_dd: object = None
    max_frames: int = 8
    max_depth: int = 3


@dataclass
class _DeclStmt:
    """One statement-level local declaration recovered from a function body.

    ``names`` are all declarators sharing the leading type token (so a bare
    multi-declarator ``TMass Y, S, A;`` yields one ``_DeclStmt`` with three names);
    widening rewrites that one type token, widening every sibling (§2).
    """

    decl_line: int          # 1-based absolute file line of the leading type token
    core_type: str          # outermost type-name token (``TMass`` / ``TOutput``)
    names: list[str]


def compute_value_closure(
    *, manifest: ChainManifest, graph: CallGraph, scalar_type: str,
    complex_type: str | None = None, complex_tokens=frozenset(),
    max_paths: int = 1024, leaf_ctx: "LeafPromotionContext | None" = None,
) -> CarrierClosure:
    """Compute a chain's value closure (CLOSURE_SCOPED_CHAINS_DESIGN.md §2).

    Pure, source-only analysis over ``manifest.lines`` — no tree mutation, no new
    source-analysis machinery (reuses :mod:`agents.shared.region_scan` +
    :class:`~agents.patcher.call_graph.CallGraph`).  Produces two views on one
    :class:`CarrierClosure` (see its docstring):

    * the **Fix-A compat subset** (``widenable`` / ``unwidenable_reasons`` /
      ``external_reasons`` / ``decl_widens``): the Blocker-A strict-carrier test — a
      name (1) written by an interior chain line, (2) read by *another chain line*,
      (3) declared outside the chain line set, (4) not a write target of the
      outermost region.  Unchanged from Blocker A so every current consumer of this
      object is byte-identical this subtask;

    * the **enlarged value closure** (``closure_widenable`` / ``closure_decl_widens``
      / ``designed_exits`` / ``escape_reasons`` / ``return_widens``): the least fixed
      point of rules (a) [intra-frame carrier, generalised to a read on *any* line in
      the frame] and (b) [forward flow to a local / out-param / return / kernel
      output], with ``chain_closure_escapes`` refusals at the frontier (§2.3, §2.4).
      Rule (c) (cross-frame return propagation) is Subtask 2a/2b — ``return_widens``
      stays empty here.

    Raises :class:`~agents.patcher.fanout.FanoutError` if a chain region is not
    inside any known function or its function is unreachable from the entry point —
    the same fail-loud contract as :func:`chain_promote`.
    """
    # -- 1/2. per-line read/write/depth, derived from source (region_scan) -------
    per_line: list[dict] = []
    func_src: dict[tuple[str, int, int], str] = {}   # (file, ls, le) -> source text
    func_line_start: dict[tuple[str, int, int], int] = {}
    chain_lineset: dict[str, set[int]] = {}          # file -> {chain line numbers}

    for (file, ls, le) in manifest.lines:
        fd = graph.enclosing_function(file, ls)
        if fd is None:
            raise FanoutError(
                f"carrier closure for chain {manifest.chain_id}: region "
                f"{file}:{ls}-{le} is not inside any known function "
                f"(call graph rooted at {graph.root!r})")
        fkey = (fd.file, fd.line_start, fd.line_end)
        if fkey not in func_src:
            func_src[fkey] = "\n".join(_original_text(fd.file, fd.line_start, fd.line_end))
            func_line_start[fkey] = fd.line_start
        region_text = "\n".join(_original_text(fd.file, ls, le))
        for ln in range(ls, le + 1):
            chain_lineset.setdefault(fd.file, set()).add(ln)
        per_line.append(dict(
            file=fd.file, line=ls, fd=fd, fkey=fkey,
            depth=_region_depth(graph, fd, manifest.chain_id, max_paths),
            reads=_names_read_in_region(region_text),
            writes=set(region_scan.region_writes_from_source(region_text))))

    if not per_line:
        return CarrierClosure()

    outermost_depth = min(p["depth"] for p in per_line)

    # -- 3. candidate carriers: written on an interior line AND read across lines -
    interior_writes: dict[str, set[int]] = {}    # name -> interior chain lines writing it
    all_writes: dict[str, set[int]] = {}         # name -> every chain line writing it
    all_reads: dict[str, set[int]] = {}          # name -> every chain line reading it
    outer_write_targets: set[str] = set()        # names written by the outermost region
    name_funcs: dict[str, set[tuple[str, int, int]]] = {}  # name -> chain funcs touching it

    for p in per_line:
        interior = p["depth"] != outermost_depth
        for w in p["writes"]:
            all_writes.setdefault(w, set()).add(p["line"])
            name_funcs.setdefault(w, set()).add(p["fkey"])
            if interior:
                interior_writes.setdefault(w, set()).add(p["line"])
            else:
                outer_write_targets.add(w)
        for r in p["reads"]:
            all_reads.setdefault(r, set()).add(p["line"])
            name_funcs.setdefault(r, set()).add(p["fkey"])

    out = CarrierClosure()
    widen_groups: dict[tuple[str, int], _DeclStmt] = {}   # (file, decl_line) -> stmt

    for name in sorted(interior_writes):
        wlines = interior_writes[name]                    # cond 1: interior write ✓
        rlines = all_reads.get(name, set())
        # cond 2: read by a chain line so the value crosses a chain-line boundary —
        # a strict write→read on different lines, or a same-line read-write plus at
        # least one other touching line (the OR clause of §2).
        touched = rlines | all_writes.get(name, set())
        if not (rlines and len(touched) >= 2):
            continue
        # cond 4: a write target of the outermost region is the designed exit
        # boundary (demoted on purpose) — excluded from the closure entirely (§5).
        if name in outer_write_targets:
            continue

        # a carrier local lives in ONE function; a name visible across >1 chain
        # function is not a plain local → global / class member / output container.
        funcs = name_funcs.get(name, set())
        if len(funcs) > 1:
            out.external_reasons.append((
                name, f"declared outside a single chain function (touched by "
                f"{len(funcs)} chain functions) — global / class member / output "
                f"container; v1 refuses to widen shared state"))
            continue

        fkey = next(iter(funcs))
        decls = _local_decls(func_src[fkey], func_line_start[fkey])
        param_cores = region_scan._param_name_core_types(
            region_scan._tokenize(func_src[fkey]))

        stmt = decls.get(name)
        if stmt is not None:
            # cond 3: a decl INSIDE the chain line set is already widened by the body
            # transform — out of scope (not returned).
            if stmt.decl_line in chain_lineset.get(fkey[0], set()):
                continue
            widen_groups.setdefault((fkey[0], stmt.decl_line), stmt)
        elif name in param_cores:
            out.unwidenable_reasons.append((
                name, f"declared as a parameter of the chain function at "
                f"{fkey[0]}:{fkey[1]} — v1 refuses to rewrite function signatures"))
        else:
            out.external_reasons.append((
                name, f"no local or parameter declaration found in the enclosing "
                f"chain function — global / class member / output container; v1 "
                f"refuses to widen shared state"))

    # -- 6/7. expand each widened decl line to ALL its declarators (siblings, §2) --
    for (dfile, dline), stmt in widen_groups.items():
        dd_type = _carrier_dd_type(stmt.core_type, scalar_type,
                                   complex_type, complex_tokens)
        for nm in stmt.names:
            out.widenable.append((dfile, dline, nm, dd_type))
        # one decl-widen record per decl LINE — widening the leading type token widens
        # every same-type sibling in a single edit (§2), so the emission layer needs
        # only the line + the orig→dd type map, not one edit per name.
        out.decl_widens.append((dfile, dline, stmt.core_type, dd_type, stmt.names[0]))

    out.widenable.sort(key=lambda w: (w[0], w[1], w[2]))
    out.decl_widens.sort(key=lambda w: (w[0], w[1]))
    out.unwidenable_reasons.sort()
    out.external_reasons.sort()

    # -- enlarged value closure (rules (a)+(b) fixed point, §2.3) ----------------
    # Computed alongside the Fix-A subset above and stored in the closure_* fields;
    # no current consumer reads these, so the compat behaviour is untouched.
    frames = {p["fkey"]: p["fd"] for p in per_line}
    frame_names = {p["fd"].name for p in per_line}
    _expand_value_closure(
        out, frames=frames, frame_names=frame_names, func_src=func_src,
        func_line_start=func_line_start, chain_lineset=chain_lineset,
        scalar_type=scalar_type, complex_type=complex_type,
        complex_tokens=complex_tokens, leaf_ctx=leaf_ctx)
    return out


def _region_depth(graph: CallGraph, fd, chain_id: str, max_paths: int) -> int:
    """Hops from the entry point to ``fd`` (shallower = outer), mirroring the depth
    :func:`chain_promote` records in ``region_meta``.  A region in the entry point is
    depth 0; any other function's depth is ``min(len(path)) - 1`` over the caller
    paths.  Raises :class:`FanoutError` for a function unreachable from the root."""
    if fd.name == graph.root:
        return 0
    paths, _ = graph.enumerate_paths(fd.name, max_paths=max_paths)
    if not paths:
        raise FanoutError(
            f"carrier closure for chain {chain_id}: function {fd.name!r} is "
            f"unreachable from entry point {graph.root!r}")
    return min(len(p) for p in paths) - 1


def _is_already_dd(core_type: str, scalar_type: str,
                   complex_type: str | None) -> bool:
    """True when ``core_type`` already denotes a dd type — so widening it is a no-op.

    Guards the double-widen (Shape 3, ``Kokkos::complex<Kokkos::Experimental::DoubleDoubleComplex>``):
    a carrier whose source token is *already* the dd scalar/complex spelling (or any
    ``Kokkos::Experimental::`` type, or a complex container wrapping one) must NOT be widened
    again.  Matched on the core token AND its last ``::`` segment so an
    already-qualified ``Kokkos::Experimental::DoubleDoubleComplex`` and a bare ``DoubleDoubleComplex`` both hit.
    """
    if not core_type:
        return False
    last = core_type.split("::")[-1]
    dd_spellings = {scalar_type, scalar_type.split("::")[-1]}
    if complex_type:
        dd_spellings |= {complex_type, complex_type.split("::")[-1]}
    if core_type in dd_spellings or last in dd_spellings:
        return True
    # Any vendored dd type (Kokkos::Experimental::*) is already extended precision.
    return core_type.startswith("Kokkos::Experimental::") or "Kokkos::Experimental::" in core_type


def _carrier_dd_type(core_type: str, scalar_type: str,
                     complex_type: str | None, complex_tokens) -> str:
    """The chain-internal dd type a carrier decl widens to: the complex container
    when the carrier's core type is a complex-bound token, else the scalar dd type
    (§7 — ``Kokkos::Experimental::DoubleDouble`` / ``DoubleDoubleComplex``).

    Idempotence guard (Shape 3): if the core type is ALREADY a dd type, return it
    unchanged — re-wrapping an already-``DoubleDoubleComplex`` operand in the complex container
    is what produces the nonsensical ``Kokkos::complex<Kokkos::Experimental::DoubleDoubleComplex>``."""
    if _is_already_dd(core_type, scalar_type, complex_type):
        return core_type
    if complex_type and core_type in complex_tokens:
        return complex_type
    return scalar_type


def _names_read_in_region(region_text: str) -> set[str]:
    """Identifiers *read* in a region — a raw token scan (not filtered to any
    scalar-name universe, so a carrier whose declared type is unrecognized, e.g.
    ``TMass``, is still detected).

    Excludes pure ``name =`` write targets (a compound ``name +=`` reads and the
    tokenizer emits ``+=`` whole, so it is kept), call targets (``name (``), and
    member / namespace-qualified references (``a.b`` / ``a->b`` / ``ql::x``).
    Over-inclusion is harmless: a name only becomes a carrier candidate if it is
    ALSO written on an interior chain line, so a stray type/constant read never
    promotes to a false carrier.
    """
    toks = region_scan._tokenize(region_text)
    n = len(toks)
    reads: set[str] = set()
    for i, t in enumerate(toks):
        if not region_scan._is_ident_tok(t.text):
            continue
        prev = toks[i - 1].text if i > 0 else ""
        nxt = toks[i + 1].text if i + 1 < n else ""
        if prev in (".", "->", "::"):
            continue                       # member / qualified reference lead
        if nxt in ("=", "(", "::"):
            continue                       # write target, call, or qualifier lead
        reads.add(t.text)
    return reads


def _local_decls(func_src: str, func_line_start: int) -> dict[str, _DeclStmt]:
    """Map each statement-level local name to its :class:`_DeclStmt`.

    Recognizes init (``T a = …``), bare (``T a;``) and bare multi-declarator
    (``TMass Y, S, A;``) forms — the last of which the boundary transform's
    ``_scan_decls`` (which requires ``<type> <name> =``) misses, so a dedicated
    scanner is needed (design §7).  A pure token scan over the enclosing function,
    consistent with :mod:`agents.shared.region_scan`; source-only, no AST type
    resolution (libclang cannot resolve a template region's types without its
    include context — it mislabels ``ddilog`` itself as a ``VAR_DECL``).
    """
    toks = region_scan._tokenize(func_src)
    out: dict[str, _DeclStmt] = {}
    stmt: list = []
    paren = 0

    def handle(stmt: list) -> None:
        if not stmt:
            return
        # skip leading storage/cv qualifiers
        k = 0
        while k < len(stmt) and stmt[k].text in region_scan._TYPE_QUALIFIERS:
            k += 1
        if k >= len(stmt) or not region_scan._is_ident_tok(stmt[k].text):
            return
        # tokenizer lines are 1-based within func_src; map to the absolute file line.
        type_line = func_line_start + stmt[k].line - 1
        # core type: leading ident, then (:: ident)* — reduced to its last segment
        j = k
        parts = [stmt[j].text]
        j += 1
        while j + 1 < len(stmt) and stmt[j].text == "::" \
                and region_scan._is_ident_tok(stmt[j + 1].text):
            parts.append(stmt[j + 1].text)
            j += 2
        core = parts[-1]
        # skip a template argument list ``<...>`` on the type
        if j < len(stmt) and stmt[j].text == "<":
            d = 0
            while j < len(stmt):
                tx = stmt[j].text
                if tx == "<":
                    d += 1
                elif tx == ">":
                    d -= 1
                elif tx == ">>":
                    d -= 2
                j += 1
                if d <= 0:
                    break
        while j < len(stmt) and stmt[j].text in ("&", "*"):
            j += 1
        while j < len(stmt) and stmt[j].text in region_scan._TYPE_QUALIFIERS:
            j += 1
        # declarators: name [= init] [ [extent] ] (, name …)* — bail on any shape
        # that is not a plain variable declaration (``name(`` = call/ctor/func decl).
        names: list[str] = []
        while j < len(stmt):
            if not region_scan._is_ident_tok(stmt[j].text):
                break
            nm = stmt[j].text
            nxt = stmt[j + 1].text if j + 1 < len(stmt) else ";"
            if nxt == "(":
                return                     # call / constructor-init / function decl
            if nxt not in (",", ";", "=", "["):
                return                     # not a plain declarator (e.g. ``a . b``)
            names.append(nm)
            j += 1
            if nxt in ("=", "["):          # skip initializer / array extent
                d = 0
                while j < len(stmt):
                    tx = stmt[j].text
                    if tx in "([{":
                        d += 1
                    elif tx in ")]}":
                        d -= 1
                    elif tx == "<":        # template-id in init (``= ql::f<T,U>(…)``)
                        d += 1
                    elif tx == ">":
                        d -= 1
                    elif tx == ">>":
                        d -= 2
                    elif tx == "," and d == 0:
                        break
                    j += 1
            if j < len(stmt) and stmt[j].text == ",":
                j += 1
                continue
            break
        if not names or len(parts) < 1:
            return
        rec = _DeclStmt(decl_line=type_line, core_type=core, names=names)
        for nm in names:
            out.setdefault(nm, rec)

    for t in toks:
        tx = t.text
        if tx == "(":
            paren += 1
            stmt.append(t)
            continue
        if tx == ")":
            paren = max(0, paren - 1)
            stmt.append(t)
            continue
        if tx in "{}":
            handle(stmt)
            stmt = []
            continue
        if tx == ";" and paren == 0:
            handle(stmt)
            stmt = []
            continue
        stmt.append(t)
    return out


# --------------------------------------------------------------------------- #
# Closure-scoped chains — enlarged value closure (CLOSURE_SCOPED_CHAINS_DESIGN.md §2)
#
# The Fix-A carrier analysis above is the strict special case (rule (a) restricted to
# reads on another *chain* line).  These helpers compute the enlarged closure: the
# least fixed point of rule (a) [intra-frame carrier, read on ANY frame line] and rule
# (b) [forward flow to a local / out-param / return / kernel output], with
# ``chain_closure_escapes`` refusals at the frontier.  Rule (c) (cross-frame return
# propagation) is Subtask 2a/2b and is NOT computed here.
# --------------------------------------------------------------------------- #


def _scan_calls(line_text: str) -> list[tuple[str, str, set[str]]]:
    """Calls on one source line: ``(qualified_callee, last_segment, arg_idents)``.

    ``last_segment`` is the callee's final name token (``Real`` for ``ql::Real``),
    matched against the chain function set and the type-token cast set by rule (b).
    ``arg_idents`` is every identifier read inside the call's parentheses (any depth),
    excluding member / qualified-reference leads and nested call targets — enough to
    decide whether a carried value flows into the callee.
    """
    toks = region_scan._tokenize(line_text)
    n = len(toks)
    calls: list[tuple[str, str, set[str]]] = []
    i = 0
    while i < n:
        t = toks[i]
        # A call is ``ident (`` OR a template-id ``ident < template-args > (`` — the
        # latter is qcdloop's dominant form (``ql::Li2omx2<TOutput,...>(...)``).  Find
        # the ``(`` that opens the argument list, skipping a balanced ``<...>`` if the
        # ident is immediately followed by one (over-detection of a stray comparison is
        # harmless: a call is only acted on when ``last`` is in the chain function set).
        paren_i: int | None = None
        if region_scan._is_ident_tok(t.text) and i + 1 < n:
            if toks[i + 1].text == "(":
                paren_i = i + 1
            elif toks[i + 1].text == "<":
                d = 0
                j2 = i + 1
                while j2 < n:
                    tx2 = toks[j2].text
                    if tx2 == "<":
                        d += 1
                    elif tx2 == ">":
                        d -= 1
                    elif tx2 == ">>":
                        d -= 2
                    j2 += 1
                    if d <= 0:
                        break
                if j2 < n and toks[j2].text == "(":
                    paren_i = j2
        if paren_i is not None:
            prefix: list[str] = []          # qualify: (ident ::)* ident
            k = i - 1
            while k - 1 >= 0 and toks[k].text == "::" \
                    and region_scan._is_ident_tok(toks[k - 1].text):
                prefix.insert(0, toks[k - 1].text)
                k -= 2
            last = t.text
            qual = "::".join(prefix + [last]) if prefix else last
            depth = 0
            args: set[str] = set()
            j = paren_i
            while j < n:
                tx = toks[j].text
                if tx == "(":
                    depth += 1
                elif tx == ")":
                    depth -= 1
                    if depth == 0:
                        break
                elif region_scan._is_ident_tok(tx):
                    prev = toks[j - 1].text
                    nxt = toks[j + 1].text if j + 1 < n else ""
                    # exclude member/qualified leads, and call targets — either a plain
                    # ``ident(`` or a template-id ``ident<`` (the nested-call target is
                    # recorded as its own call by the token-by-token advance below).
                    if prev not in (".", "->", "::") and nxt not in ("(", "::", "<"):
                        args.add(tx)
                j += 1
            calls.append((qual, last, args))
            # advance by ONE token (not past the whole call): a nested call inside the
            # argument list (``TOutput(... ql::ddilog<...>(arg) ...)``) is a distinct
            # chain-internal edge rule (c) needs, so it must be scanned in turn.
            i += 1
            continue
        i += 1
    return calls


def _decl_init_writes(line_text: str) -> set[str]:
    """Names WRITTEN by a statement-level declaration-with-initializer on one line.

    ``region_scan.region_writes_from_source`` deliberately EXCLUDES decl-init targets
    (``const TOutput lnarg = …``) — a decl is a *landing* the boundary transform's
    ``_compute_promotion`` classifies separately, not a Case-B pre-declared write.  But
    for the value-closure seed a decl-init on a chain line whose value is read by another
    chain line IS a produced carrier (Li2omx2's ``lnarg``/``lnomarg`` @702/703 feed the
    dd cancellation @704; B10's ``dilog4 = ql::Li2omx2<…>()`` receives a rule-(c) dd
    return).  This recovers exactly those LHS names: the declarator identifier(s) of a
    statement-leading ``<type> <name> [= …]`` at paren depth 0 — reusing the same
    :func:`_local_decls` scanner so the ``<>`` / multi-declarator handling matches.
    A bare ``<type> <name>;`` with no initializer produces no write (nothing computed).
    """
    toks = region_scan._tokenize(line_text)
    out: set[str] = set()
    n = len(toks)
    # decl-init only: the line must contain a single-'=' assignment (an initializer),
    # not a bare declaration; _local_decls already parses the declarator list.
    has_init = any(
        t.text == "=" and (i + 1 >= n or toks[i + 1].text != "=")
        and (i == 0 or toks[i - 1].text not in ("!", "<", ">", "+", "-", "*", "/", "%",
                                                "&", "|", "^", "="))
        for i, t in enumerate(toks))
    if not has_init:
        return out
    for nm, st in _local_decls(line_text, 1).items():
        out.add(nm)
    return out


def _scan_container_stores(line_text: str) -> list[tuple[str, set[str]]]:
    """Indexed / functor stores on one line: ``(container_name, rhs_idents)``.

    Matches a statement-leading ``name(...) = …`` or ``name[...] = …`` (the qcdloop
    kernel-output form ``res(i,k) = …``) — an assignment, not a call statement or a
    ``==`` comparison.  ``rhs_idents`` are the identifiers read on the right-hand side.
    """
    toks = region_scan._tokenize(line_text)
    n = len(toks)
    stores: list[tuple[str, set[str]]] = []
    i = 0
    while i < n:
        t = toks[i]
        if region_scan._is_ident_tok(t.text) and i + 1 < n \
                and toks[i + 1].text in ("(", "["):
            prev = toks[i - 1].text if i > 0 else ""
            if prev not in ("", ";", "{", "}"):
                i += 1
                continue                    # not statement-leading -> a nested call/cast
            depth = 0
            j = i + 1
            while j < n:
                tx = toks[j].text
                if tx in ("(", "["):
                    depth += 1
                elif tx in (")", "]"):
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            a = j + 1
            if a < n and toks[a].text == "=" \
                    and (a + 1 >= n or toks[a + 1].text != "="):
                rhs: set[str] = set()
                m = a + 1
                while m < n:
                    tx = toks[m].text
                    if region_scan._is_ident_tok(tx):
                        p = toks[m - 1].text
                        nx = toks[m + 1].text if m + 1 < n else ""
                        if p not in (".", "->", "::") and nx not in ("(", "::"):
                            rhs.add(tx)
                    m += 1
                stores.append((t.text, rhs))
            i = j + 1
            continue
        i += 1
    return stores


def _has_binary_addsub(line_text: str) -> bool:
    """True iff ``line_text`` contains a *binary* ``+`` / ``-`` (a cancellation-prone
    op), as opposed to only unary sign or ``+=`` compound tokens.

    A ``+``/``-`` is binary when its preceding token is a value terminator — an
    identifier, a closing ``)`` / ``]``, or a numeric literal.  This is the decidable
    proxy the benign-extract procedure (§3.2 clause iii) uses for "the producing chain
    line performs a subtraction/addition": the near-equal difference that produces a
    carried value (B13's ``ga34m = TOutput(...) - root``) is exactly this shape.
    """
    toks = region_scan._tokenize(line_text)
    for i, t in enumerate(toks):
        if t.text not in ("+", "-"):
            continue
        prev = toks[i - 1].text if i > 0 else ""
        if prev and (region_scan._is_ident_tok(prev) or prev in (")", "]")
                     or prev[:1].isdigit()):
            return True
    return False


def _return_type_signature(func_src: str, func_line_start: int):
    """Locate a function's return-type token for rule (c)'s :class:`ReturnWiden`.

    Returns ``(sig_line, orig_type)`` where ``sig_line`` is the 1-based ABSOLUTE file
    line of the function-name identifier that precedes the parameter-list ``(`` (the
    signature line — the same coordinate :func:`boundary.widen_return_type_line`
    expects), and ``orig_type`` is the leading core return-type token as source spells
    it (its last ``::`` segment, mirroring :func:`_local_decls`).  Returns ``None`` if
    the return type cannot be recovered (an ``auto`` trailing-return / SFINAE shape) —
    rule (c) then declines to widen that frame's return (the gate keeps it truncating).

    Source-only token scan over the enclosing function; skips the leading ``template
    <...>`` clause, storage / macro qualifiers, and any ``::``-qualified / templated
    type, so it recovers ``TMass`` from ``KOKKOS_INLINE_FUNCTION TMass ddilog(...)``
    and ``TOutput`` from ``KOKKOS_INLINE_FUNCTION TOutput Li2omx2(...)``.
    """
    toks = region_scan._tokenize(func_src)
    n = len(toks)
    i = 0
    # skip a leading ``template < ... >`` clause
    if i < n and toks[i].text == "template":
        i += 1
        if i < n and toks[i].text == "<":
            d = 0
            while i < n:
                tx = toks[i].text
                if tx == "<":
                    d += 1
                elif tx == ">":
                    d -= 1
                elif tx == ">>":
                    d -= 2
                i += 1
                if d <= 0:
                    break
    # Find the parameter-list ``(`` at angle/bracket depth 0 — the opener that follows
    # the function name — then the function name is the ident immediately before it and
    # the return type is the last core token before the function name.  This mirrors
    # boundary.widen_return_type_line so the recovered coordinate is exactly what it
    # rewrites, and avoids fragile macro-vs-type heuristics (a single-letter type ``T``
    # is not distinguishable from a macro by casing).
    paren = None
    depth = 0
    k = i
    while k < n:
        tx = toks[k].text
        if tx in ("<", "["):
            depth += 1
        elif tx in (">", "]"):
            depth = max(0, depth - 1)
        elif tx == ">>":
            depth = max(0, depth - 2)
        elif tx == "(" and depth == 0:
            paren = k
            break
        k += 1
    if paren is None:
        return None
    # function-name ident = last ident before the param '('
    fn = None
    for k in range(paren - 1, i - 1, -1):
        if region_scan._is_ident_tok(toks[k].text):
            fn = k
            break
    if fn is None or fn == i:
        return None                          # no return-type token before the name
    # the return-type region is every token in [i, fn); its core token is the LAST
    # identifier immediately before the function name (walking back over ``&`` / ``*`` /
    # ``const`` and any trailing ``>`` of a template-arg list), reduced to its last
    # ``::`` segment.  Using the LAST core token (not the first) skips leading attribute
    # macros (``KOKKOS_INLINE_FUNCTION``) / storage keywords, which are never the base
    # return type — mirroring boundary.widen_return_type_line's rewrite target.
    r = fn - 1
    while r > i and toks[r].text in ("&", "*", "const"):
        r -= 1
    # if the token before the name closes a template-arg list (``std::vector<T> f``),
    # the base type is the ident that OPENED it; walk back to that ident.
    if r > i and toks[r].text in (">", ">>"):
        d = 2 if toks[r].text == ">>" else 1
        r -= 1
        while r > i and d > 0:
            if toks[r].text == "<":
                d -= 1
            elif toks[r].text == ">":
                d += 1
            elif toks[r].text == ">>":
                d += 2
            r -= 1
        while r > i and toks[r].text not in ("::",) \
                and not region_scan._is_ident_tok(toks[r].text):
            r -= 1
    if r < i or not region_scan._is_ident_tok(toks[r].text):
        return None
    core = toks[r].text
    sig_line = func_line_start + toks[fn].line - 1
    return sig_line, core


def _expand_value_closure(out: CarrierClosure, *, frames, frame_names, func_src,
                          func_line_start, chain_lineset, scalar_type,
                          complex_type, complex_tokens,
                          leaf_ctx: "LeafPromotionContext | None" = None) -> None:
    """Populate ``out``'s enlarged-closure fields (rules (a)+(b)+(c) fixed point, §2.3).

    ``frames`` maps each chain frame key ``(file, ls, le)`` to its function decl;
    ``frame_names`` is the set of chain function names ``F`` (a call whose target is
    in ``F`` is a chain-internal edge — rule (c) territory, not an escape).

    When ``leaf_ctx`` is supplied (L2 opt-in), rule (d) leaf-callee promotion runs
    AFTER the a/b/c fixed point (:func:`_discover_leaf_frames`), recording clone
    reroutes / refusals / the circuit breaker on ``out``.  With ``leaf_ctx=None`` the
    a/b/c result is byte-identical to before (STOP #B).
    """
    # type tokens that, used as a call target, denote a cast rather than an escape
    type_tokens: set[str] = set(complex_tokens) | {scalar_type}
    decls_by_frame: dict = {}
    param_by_frame: dict = {}
    for fkey in frames:
        d = _local_decls(func_src[fkey], func_line_start[fkey])
        decls_by_frame[fkey] = d
        for st in d.values():
            type_tokens.add(st.core_type)
        param_by_frame[fkey] = region_scan._param_name_core_types(
            region_scan._tokenize(func_src[fkey]))

    # -- per-frame, per-line scan (reads / writes / calls / stores / return) -----
    line_reads: dict = {}
    line_writes: dict = {}
    line_calls: dict = {}
    line_stores: dict = {}
    line_return: dict = {}
    line_addsub: dict = {}                   # fkey -> {line: has a binary +/-}
    line_decl_inits: dict = {}               # fkey -> {line: decl-init LHS names}
    frame_writes: dict = {}                 # fkey -> every name written in the frame
    chain_lines_in_frame: dict = {}
    for fkey in frames:
        ls0 = func_line_start[fkey]
        le0 = fkey[2]
        src_lines = func_src[fkey].split("\n")
        lr, lw, lc, lst, lret, ladd = {}, {}, {}, {}, {}, {}
        fw: set[str] = set()
        di: dict = {}                        # L -> decl-init LHS names on that line
        for L in range(ls0, le0 + 1):
            idx = L - ls0
            txt = src_lines[idx] if 0 <= idx < len(src_lines) else ""
            writes = set(region_scan.region_writes_from_source(txt))
            lr[L] = _names_read_in_region(txt)
            lw[L] = writes
            di[L] = _decl_init_writes(txt)
            lc[L] = _scan_calls(txt)
            lst[L] = _scan_container_stores(txt)
            lret[L] = any(tk.text == "return" for tk in region_scan._tokenize(txt))
            ladd[L] = _has_binary_addsub(txt)
            fw |= writes
        line_reads[fkey] = lr
        line_writes[fkey] = lw
        line_calls[fkey] = lc
        line_stores[fkey] = lst
        line_return[fkey] = lret
        line_addsub[fkey] = ladd
        line_decl_inits[fkey] = di
        frame_writes[fkey] = fw
        cl = chain_lineset.get(fkey[0], set())
        chain_lines_in_frame[fkey] = {L for L in cl if ls0 <= L <= le0}

    # -- seed (§2.2): names WRITTEN on a chain line -------------------------------
    # A value merely READ on a chain line is an input (a constant / param, or a value
    # whose dd-ness would require rule (c) — e.g. B10's dilog4/dilog5 from Li2omx2's
    # return); under rules (a),(b) alone it is not yet carried, so the seed is the
    # names the chain writes on its own lines (the a,b-only fixed point stops exactly
    # where §2.7 says it must — at Li2omx2's return — without rule (c)).
    W: dict = {}
    for fkey in frames:
        seed: set[str] = set()
        for L in chain_lines_in_frame[fkey]:
            seed |= line_writes[fkey].get(L, set())
        W[fkey] = seed

    widen_groups: dict = {}                 # (file, decl_line) -> _DeclStmt
    # Designed exits accumulate carried values across rounds; keyed by
    # (file, line, kind, target) -> set of carried values landing there (§3.2 / A.2).
    designed: dict = {}
    # A.1 — the escape record split.  ``source_escapes`` (carried value flowing INTO a
    # non-F callee argument) is purely diagnostic and does NOT block the source from the
    # closure.  ``dest_escapes`` (rule-(b) writes to shared state, and non-benign
    # extract destinations, decided after the fixed point) block their name from
    # widening and drive the ``chain_closure_escapes`` terminal.
    source_escapes: dict = {}                # name -> reason
    dest_escapes: dict = {}                  # name -> reason
    # Per destination escape, the carried values it severs — used after the fixed point
    # to decide which escapes MATERIALLY block (a severed value reaching no designed exit).
    dest_escape_carried: dict = {}           # dest name -> set of carried values severed
    # Deferred extract landings: a rule-(b) local write whose producing line passed a
    # carried value into a non-F callee (``x34* = ql::Real(ga34*)``).  Its benign
    # (designed exit) vs severance (destination escape) classification needs the FINAL
    # carried set + widened siblings, so it is decided after the fixed point (§3.2 iii).
    extract_cands: dict = {}                 # (file, line, dest) -> {carried, callee}

    # -- rule (c) accumulators (CLOSURE_SCOPED_CHAINS_DESIGN.md §2.3) --------------
    # ``return_widen_frames`` keys each callee frame whose return the closure widens by
    # its frame key; the value is (sig_line, orig_type) from _return_type_signature.
    # Recorded at FRAME level (the callee's ORIGINAL name), then attached to every
    # per-caller-path variant at emission time (chain_promote._attach_return_widens),
    # because a single callee (ddilog) fans out to many per-path variants and the
    # closure has no path/graph context to enumerate them here (STOP #5 resolution).
    return_widen_frames: dict = {}           # callee fkey -> (sig_line, orig_type)
    # Frame name -> its frame key, for the caller->callee edge lookup below.  A name
    # visible in exactly one chain frame; a name in more than one is not a plain callee
    # frame (rule (c) does not fire on it — it would already be an external/global).
    fkey_by_name: dict = {}
    for fk in frames:
        fkey_by_name.setdefault(frames[fk].name, set()).add(fk)

    def _mark_designed(dfile, dline, kind, target, carried) -> bool:
        k = (dfile, dline, kind, target)
        prev = designed.get(k)
        if prev is None:
            designed[k] = set(carried)
            return True
        if not carried <= prev:
            prev |= carried
            return True
        return False

    def _receiving_locals_at_calls(caller_fk, callee_name: str):
        """Local names in ``caller_fk`` that BIND the result of a call to
        ``callee_name`` (a chain-internal edge), with the line the binding is on.

        Two binding forms in qcdloop's chains:
          * decl-init  ``const TOutput dilog4 = ql::Li2omx2<...>(...)``  (B10:236) —
            ``region_writes_from_source`` reports no write on a decl-init, so the
            receiving local is recovered from the frame's decl table; the call target
            is the only call on the line.
          * plain assign ``Li2omx2 = ... ql::ddilog<...>(...) ...``     (kokkos:704) —
            the LHS is a normal write.  The call may be nested inside a cast; the
            binding local is every name written on the line.

        Yields ``(recv_local, line)``.  A call whose result is consumed as a bare
        sub-expression with no binding (fed straight into another call / a return
        expression) yields nothing here — that value has no receiving *local* to widen
        in ``caller_fk``; if it is itself returned, the return is handled by rule (b)
        marking + a further rule (c) hop from ``caller_fk``.
        """
        cf_decls = decls_by_frame[caller_fk]
        for L in range(func_line_start[caller_fk], caller_fk[2] + 1):
            calls_on_L = line_calls[caller_fk].get(L, [])
            if not any(last == callee_name for _q, last, _a in calls_on_L):
                continue
            # plain-assign writes (LHS locals declared in this frame)
            for w in line_writes[caller_fk].get(L, set()):
                if cf_decls.get(w) is not None:
                    yield (w, L)
            # decl-init binding: no write reported, but the line declares a local whose
            # decl line IS this line (``const TOutput dilog4 = ...``).
            for nm, st in cf_decls.items():
                if st.decl_line == L and nm not in line_writes[caller_fk].get(L, set()):
                    yield (nm, L)

    def _apply_rule_c(callee_fk) -> bool:
        """Fire rule (c) for a callee frame ``callee_fk`` that returns a carried value.

        Returns True iff it changed the accumulators (a new return-widen recorded or a
        receiving local newly carried).  Honors the §8 boundary: a callee whose return
        also requires a widened INWARD parameter is refused (STOP #4 territory) — but
        v1's rule (c) never widens a parameter, so a callee needing dd IN is already
        caught by the param terminal upstream; here we only widen the callee's return
        type (outward) and re-seed the caller's receiving local.
        """
        callee_name = frames[callee_fk].name
        ch = False
        for caller_fk in frames:
            if caller_fk == callee_fk:
                continue
            for (recv, L) in _receiving_locals_at_calls(caller_fk, callee_name):
                # record the callee's return-type widen once (frame-level; STOP #5)
                if callee_fk not in return_widen_frames:
                    sig = _return_type_signature(
                        func_src[callee_fk], func_line_start[callee_fk])
                    if sig is not None:
                        return_widen_frames[callee_fk] = sig
                        ch = True
                # re-seed the caller's receiving local -> rule (a) widens it next round
                if recv not in W[caller_fk]:
                    W[caller_fk].add(recv)
                    ch = True
                # A decl-init binding (``const TOutput dilog4 = ql::Li2omx2<...>()``)
                # is not seen as a write by region_writes_from_source, so rule (a)'s
                # ``written_on_carried`` test would miss it and never widen the decl —
                # leaving the dd return truncated straight back into the caller-precision
                # local (rule (c) point 2 defeated).  Register the binding line as a
                # write of the receiving local so rule (a) recognizes and widens its
                # decl (idempotent — a set add).
                lw_L = line_writes[caller_fk].setdefault(L, set())
                if recv not in lw_L:
                    lw_L.add(recv)
                    frame_writes[caller_fk].add(recv)
                    ch = True
        return ch

    MAX_ROUNDS = 64
    changed = True
    rounds = 0
    while changed and rounds < MAX_ROUNDS:
        changed = False
        rounds += 1
        for fkey in frames:
            file = fkey[0]
            decls = decls_by_frame[fkey]
            params = param_by_frame[fkey]
            cif = chain_lines_in_frame[fkey]
            frame_lines = range(func_line_start[fkey], fkey[2] + 1)
            wf = W[fkey]
            # carried lines: chain lines + any line that writes a carried value
            carried_lines = set(cif)
            for L in frame_lines:
                if line_writes[fkey].get(L, set()) & wf:
                    carried_lines.add(L)

            # ---- rule (a): intra-frame carrier decl widening -------------------
            for v in sorted(wf):
                stmt = decls.get(v)
                if stmt is None:
                    continue                # param / global: not a local decl to widen
                if stmt.decl_line in cif:
                    continue                # decl inside chain set -> body transform owns it
                written_on_carried = any(
                    v in line_writes[fkey].get(L, set()) for L in carried_lines)
                read_lines = [L for L in frame_lines
                              if v in line_reads[fkey].get(L, set())]
                if not (written_on_carried and read_lines):
                    continue
                key = (file, stmt.decl_line)
                if key not in widen_groups:
                    widen_groups[key] = stmt
                    changed = True

            # ---- rule (b): forward flow ----------------------------------------
            for L in frame_lines:
                carried_here = line_reads[fkey].get(L, set()) & wf
                if not carried_here:
                    continue
                # source escape: a carried value passed into a callee NOT in F (not a
                # cast).  Diagnostic only — does NOT block the source (A.1).
                escaped_here: set = set()
                escape_callee: dict = {}
                for (qual, last, args) in line_calls[fkey].get(L, []):
                    if last in frame_names or last in type_tokens:
                        continue            # chain-internal edge (rule c) / a type cast
                    esc = args & wf
                    for nm in sorted(esc):
                        escaped_here.add(nm)
                        escape_callee.setdefault(nm, qual)
                        if nm not in source_escapes:
                            source_escapes[nm] = (
                                f"carried value {nm!r} enters callee {qual!r} (not "
                                f"in the chain function set) as an argument at "
                                f"{Path(file).name}:{L} — v1 does not widen its "
                                f"signature")
                            changed = True
                # rule (b) -> out-param / local write
                for w in line_writes[fkey].get(L, set()):
                    if w in params:
                        # carried_values = {w} (A.2): the out-param IS the landing.
                        if _mark_designed(file, L, "out_param", w, {w}):
                            changed = True
                        continue
                    if decls.get(w) is None:
                        # neither local nor param -> global / class member / shared
                        # output container (§2.4): a destination escape (blocks).
                        if w not in dest_escapes:
                            dest_escapes[w] = (
                                f"rule (b) writes shared global/member {w!r} at "
                                f"{Path(file).name}:{L} — v1 does not widen shared state")
                            changed = True
                        dest_escape_carried.setdefault(w, set()).update(carried_here)
                        continue
                    if escaped_here:
                        # producer escaped into a non-F callee: an EXTRACT to caller
                        # precision.  Defer benign (designed exit) vs severance
                        # (destination escape) to after the fixed point (§3.2 iii).
                        key = (file, L, w)
                        if key not in extract_cands:
                            callee = escape_callee[sorted(escape_callee)[0]]
                            extract_cands[key] = dict(
                                carried=set(carried_here), callee=callee)
                            changed = True
                        continue
                    if w not in wf:
                        wf.add(w)
                        changed = True
                # rule (b) -> indexed kernel-output store (res(i,k)) -> designed exit
                for (container, rhs) in line_stores[fkey].get(L, []):
                    rc = rhs & wf
                    if rc and _mark_designed(file, L, "kernel_output", container, rc):
                        changed = True
                # rule (b) -> return -> designed exit
                if line_return[fkey].get(L):
                    if _mark_designed(file, L, "return", "", set(carried_here)):
                        changed = True
                    # ---- rule (c): cross-frame return propagation --------------
                    # This frame g returns a carried value.  If another chain frame f
                    # calls g on a chain-internal edge (f -> g) and binds the result
                    # into a receiving local, then (1) g's variant return type widens
                    # (recorded here, attached per-path at emission), and (2) the
                    # receiving local re-enters rule (a) in f (added to W[f]); the
                    # fixed point then climbs the DAG (§2.3 rule c).
                    if _apply_rule_c(fkey):
                        changed = True

    # -- expand each widened decl line to all its declarators (siblings, §2) ------
    widened_names: set[str] = set()
    for (dfile, dline), stmt in widen_groups.items():
        dd_type = _carrier_dd_type(stmt.core_type, scalar_type,
                                   complex_type, complex_tokens)
        for nm in stmt.names:
            out.closure_widenable.append((dfile, dline, nm, dd_type))
            widened_names.add(nm)
        out.closure_decl_widens.append(
            (dfile, dline, stmt.core_type, dd_type, stmt.names[0]))

    # -- body-owned chain-line decl-init carriers (closure_body_names) ------------
    # A decl-init on a CHAIN line (``const TOutput lnarg = …`` @702) whose value is READ
    # on ANOTHER chain line in the same frame (the dd cancellation @704) is a carrier the
    # body transform already widens (its decl is inside the chain set, so rule (a) leaves
    # it to promote_region_block).  But the boundary transform demotes the decl-init
    # landing to caller precision UNLESS the name is in closure_names — so record it here.
    # Strict chain-scope (write AND read both on chain lines) so a frame's non-chain
    # locals are untouched: this leaves every a/b/c fixed-point result — and thus B13's
    # baseline — byte-identical, adding only the missing "keep wide" signal.
    for fk in frames:
        cif = chain_lines_in_frame[fk]
        di = line_decl_inits[fk]
        for wl in cif:
            for nm in di.get(wl, set()):
                read_elsewhere = any(
                    nm in line_reads[fk].get(rl, set())
                    for rl in cif if rl != wl)
                if read_elsewhere:
                    out.closure_body_names.add(nm)

    # -- classify deferred extract landings (§3.2 clause iii, benign-extract) ------
    # Both checks are decidable from the closure result alone (no source re-scan):
    #   (1) >=1 carried value landing at the extract has a producing chain line that
    #       performs a binary +/- (a cancellation) with a carried/widened operand — the
    #       dd cancellation residual is already resolved before the projection; and
    #   (2) the extract's destination does not feed another carrier downstream in the
    #       frame (no line reads it and writes a carried value).
    # Benign  -> a designed exit (kind "extract"); the projection to caller precision
    #            loses nothing double could keep (B13's x34*).
    # Else    -> a destination escape (conservative reject): the residual is not
    #            provably resolved, so the extract is a genuine severance (§3.3).
    def _producing_line_cancels(v: str) -> bool:
        for fk in frames:
            wfk = W[fk]
            operands = wfk | widened_names
            for L in chain_lines_in_frame[fk]:
                if (v in line_writes[fk].get(L, set())
                        and line_addsub[fk].get(L)
                        and (line_reads[fk].get(L, set()) & operands)):
                    return True
        return False

    def _dest_feeds_carrier(dest: str, efile: str) -> bool:
        for fk in frames:
            if fk[0] != efile:
                continue
            wfk = W[fk]
            for L in range(func_line_start[fk], fk[2] + 1):
                if (dest in line_reads[fk].get(L, set())
                        and (line_writes[fk].get(L, set()) & wfk)):
                    return True
        return False

    for (efile, eline, dest) in sorted(extract_cands):
        info = extract_cands[(efile, eline, dest)]
        carried = info["carried"]
        check1 = any(_producing_line_cancels(v) for v in carried)
        check2 = not _dest_feeds_carrier(dest, efile)
        if check1 and check2:
            _mark_designed(efile, eline, "extract", dest, carried)
        else:
            if dest not in dest_escapes:
                dest_escapes[dest] = (
                    f"rule (b) local {dest!r} at {Path(efile).name}:{eline} receives a "
                    f"carried value truncated by extract {info['callee']!r} to caller "
                    f"precision whose cancellation residual is not provably resolved — "
                    f"conservative chain_closure_escapes (design §3.2 clause iii)")
            dest_escape_carried.setdefault(dest, set()).update(carried)

    # -- rule (c): emit ReturnWiden records (frame-level; §2.3, §7) ---------------
    # One record per callee frame whose return the closure widened, naming the callee's
    # ORIGINAL function name and its signature return-type line.  chain_promote's
    # _attach_return_widens binds each to every per-caller-path variant of that function
    # whose original extent contains the signature line (STOP #5 resolution — the
    # closure has no path context to enumerate variant names here).  Emitted only if the
    # return-type token is recoverable AND is not already the dd type (a ``void`` /
    # already-dd return contributes nothing).
    for callee_fk, (sig_line, orig_type) in sorted(return_widen_frames.items()):
        dd_type = _carrier_dd_type(orig_type, scalar_type,
                                   complex_type, complex_tokens)
        if orig_type == dd_type:
            continue
        out.return_widens.append(ReturnWiden(
            return_line=sig_line, orig_type=orig_type, dd_type=dd_type,
            function_name=frames[callee_fk].name))
    out.return_widens.sort(key=lambda r: (r.function_name, r.return_line))

    out.closure_widenable.sort(key=lambda w: (w[0], w[1], w[2]))
    out.closure_decl_widens.sort(key=lambda w: (w[0], w[1]))
    # File-lines whose enclosing frame's return type the closure WIDENED under rule (c):
    # a return landing there carries dd across, no truncation, so it is a designed exit
    # (§3.2 clause ii).  A return in a frame whose type was NOT widened still truncates,
    # so it stays a plain ``return`` the interior gate checks (§3.3 correctness — B10
    # with rule (c) disabled must still reject at :707).
    widened_return_frames = set(return_widen_frames)

    def _return_is_widened(dfile: str, dline: int) -> bool:
        for fk in widened_return_frames:
            if fk[0] == dfile and func_line_start[fk] <= dline <= fk[2]:
                return True
        return False

    # designed_exits 5-tuple (A.2): (file, line, kind, carried_values, detail); sort by
    # a key that avoids ordering the frozenset (unorderable across sets).
    exits: list = []
    for (dfile, dline, kind, target), carried in designed.items():
        if kind == "return" and _return_is_widened(dfile, dline):
            kind = "return_widened"     # clause (ii): rule-(c) widened -> exempt
        detail = ",".join(sorted(carried)) if kind.startswith("return") else target
        exits.append((dfile, dline, kind, frozenset(carried), detail))
    out.designed_exits = sorted(exits, key=lambda d: (d[0], d[1], d[2], d[4]))
    out.source_escapes = sorted(source_escapes.items())
    out.destination_escapes = sorted(dest_escapes.items())

    # A destination escape MATERIALLY blocks (§2.4) iff it severs a carried value that
    # reaches NO designed exit anywhere else — the value's dd flow is truly lost.  A
    # severance whose values all still land at a designed exit is recorded but does not
    # fire the terminal (the chain still delivers its dd result to a designed landing).
    reaches_exit: set = set()
    for carried in designed.values():
        reaches_exit |= carried
    blocking: dict = {}
    for dest, reason in dest_escapes.items():
        if dest_escape_carried.get(dest, set()) - reaches_exit:
            blocking[dest] = reason
    out.blocking_escapes = sorted(blocking.items())

    # -- rule (d): leaf-callee promotion (LEAF_CALLEE_PROMOTION_DESIGN.md §1.1) -----
    # Runs only under an explicit opt-in (L2); inert otherwise so the a/b/c result
    # above is byte-identical (STOP #B).  Frame-discovery ONLY — it records clone
    # reroutes / refusals / the §2.8 circuit breaker on ``out``; L3 owns emission.
    if leaf_ctx is not None:
        _discover_leaf_frames(
            out, leaf_ctx=leaf_ctx, frames=frames, frame_names=frame_names,
            line_calls=line_calls, line_stores=line_stores,
            chain_lines_in_frame=chain_lines_in_frame,
            type_tokens=type_tokens, scalar_type=scalar_type)


def _discover_leaf_frames(out: CarrierClosure, *, leaf_ctx, frames, frame_names,
                          line_calls, line_stores, chain_lines_in_frame, type_tokens,
                          scalar_type) -> None:
    """Rule (d): grow ``F`` with clonable leaf callees (design §1.1, §2.6–2.8).

    A leaf callee is a call ``ql::g<…>(…)`` on a chain line whose target ``g`` is
    NOT already a chain frame (``g ∉ F``), not a functional cast, and not a
    dd-boundary symbol — the frontier today's a/b/c fixed point refuses as a
    ``chain_closure_escapes``.  For each such ``g`` this runs the ``clonable_leaf``
    predicate; if it accepts, ``g`` (and every transitive clonable dep it names) is
    recorded as a new frame with a per-integral clone name ``g_<integral>`` and a
    reroute (``leaf_reroutes[g] = g_<integral>``); if it refuses, ``g`` is a narrowed
    ``leaf_escapes`` frontier.  The §2.8 circuit breaker aborts (``chain_closure_
    oversized``) if F would exceed ``max_frames`` frames or a leaf recurses past
    ``max_depth``.  Records only — no tree mutation, no emission (L3).
    """
    graph = leaf_ctx.graph
    resolve = leaf_ctx.resolve_primary_body

    def _boundary(qual: str, last: str) -> bool:
        return is_dd_boundary(
            qual, last, surface=leaf_ctx.surface,
            source_instantiates_at_dd=leaf_ctx.source_instantiates_at_dd,
            is_class1_synthesizable=leaf_ctx.is_class1_synthesizable,
            resolve_primary_body=resolve,
            source_provides_dd=leaf_ctx.source_provides_dd)

    # Candidate leaf callees: every call ON a chain line whose target is (1) not already
    # a chain frame, (2) not a functional cast, (3) not an indexed kernel-output store
    # (``res(i,0) = …`` — a store, not a callee), and (4) not a dd-boundary symbol (a
    # vendored op / Class-1 wrapper / source constant the pipeline synthesizes or reads
    # rather than clones, §2.6).  Deterministic first-appearance order so discovery is
    # reproducible; the SAME boundary classifier :func:`clonable_leaf` clause (2) uses,
    # so the two never disagree on what is a leaf.
    candidates: list[tuple[str, str]] = []
    seen_cand: set[str] = set()
    for fkey in frames:
        cif = sorted(chain_lines_in_frame.get(fkey, set()))
        for L in cif:
            store_names = {c for c, _rhs in line_stores[fkey].get(L, [])}
            for (qual, last, _args) in line_calls[fkey].get(L, []):
                if last in frame_names or last in type_tokens:
                    continue                 # chain-internal edge / a functional cast
                if last in store_names:
                    continue                 # indexed kernel-output store, not a callee
                if last in seen_cand:
                    continue
                seen_cand.add(last)
                if _boundary(qual, last):
                    continue                 # vendored / Class-1 / source — not a leaf
                candidates.append((qual, last))

    # F starts at the existing chain frame count; each accepted leaf (and its clonable
    # deps) adds one.  The circuit breaker (§2.8) trips on frame count or recursion
    # depth — the predicate's own depth cap surfaces the latter as a refusal, which we
    # promote to ``chain_closure_oversized`` when the reason names the depth cap.
    n_frames = len(frames)
    reroutes: dict = {}
    frame_list: list = []
    escapes: list = []

    def _record_clone(name: str) -> None:
        nonlocal n_frames
        if name in reroutes:
            return
        reroutes[name] = f"{name}_{leaf_ctx.integral}"
        frame_list.append(name)
        n_frames += 1

    for (qual, g) in candidates:
        if n_frames >= leaf_ctx.max_frames:
            out.chain_closure_oversized = True
            out.oversized_detail = (
                f"rule (d) would grow F past {leaf_ctx.max_frames} frames "
                f"(§2.8 circuit breaker) at leaf {g!r}")
            break
        body = resolve(g)
        res = clonable_leaf(
            qual, body, None,
            call_graph=graph, surface=leaf_ctx.surface,
            is_class1_synthesizable=leaf_ctx.is_class1_synthesizable,
            source_instantiates_at_dd=leaf_ctx.source_instantiates_at_dd,
            resolve_primary_body=resolve, scalar_type=scalar_type,
            source_provides_dd=leaf_ctx.source_provides_dd,
            frame_names=frame_names, type_tokens=type_tokens,
            binds_shared_param=None,
            depth=1, max_depth=leaf_ctx.max_depth)
        if res.ok:
            # count this leaf + its transitive clonable deps against the breaker
            new_names = [g] + [d for d in res.transitive_deps if d not in reroutes]
            if n_frames + len(new_names) > leaf_ctx.max_frames:
                out.chain_closure_oversized = True
                out.oversized_detail = (
                    f"rule (d) would grow F past {leaf_ctx.max_frames} frames "
                    f"(§2.8 circuit breaker) admitting leaf {g!r} + "
                    f"{len(res.transitive_deps)} transitive dep(s)")
                break
            _record_clone(g)
            for dep in res.transitive_deps:
                _record_clone(dep)
        elif "§2.8 circuit breaker" in res.reason:
            out.chain_closure_oversized = True
            out.oversized_detail = res.reason
            break
        else:
            escapes.append((g, res.reason))

    out.leaf_reroutes = dict(sorted(reroutes.items()))
    out.leaf_frames = sorted(frame_list)
    out.leaf_escapes = sorted(escapes)


def _attach_return_widens(return_widens: list,
                          new_specs: dict[str, dict[str, VariantSpec]]) -> None:
    """Attach each rule-(c) :class:`ReturnWiden` to the variant(s) it names (§7).

    Rule (c) records a :class:`ReturnWiden` at FRAME level: ``function_name`` is the
    callee's ORIGINAL function name (``Li2omx2``, ``ddilog``) and ``return_line`` its
    signature return-type line, because the closure analysis has no caller-path context
    to enumerate the per-path variant names (a single callee like ``ddilog`` fans out to
    17 per-path variants — STOP #5 resolution).  A widened-return callee is copied to
    one variant PER caller path, and the widened return rides on EVERY such variant, so
    this binds each record to every spec whose ``orig_name == rw.function_name`` AND
    whose original extent ``[orig_start, orig_end]`` contains ``rw.return_line`` (the
    same line-containment attach the closure-decl widens use).  Multiple records for one
    variant combine via :func:`_merge_return_widen` (equal → dedup, differ →
    :class:`FanoutError`).  A record naming a function NO emitted spec clones raises
    :class:`FanoutError` — a real wiring bug (the closure demanded a return-widen on a
    function that produced no variant; STOP #5), caught here rather than papered over.
    """
    if not return_widens:
        return
    specs_by_orig: dict[str, list[VariantSpec]] = {}
    for specs in new_specs.values():
        for spec in specs.values():
            specs_by_orig.setdefault(spec.orig_name, []).append(spec)
    for rw in return_widens:
        candidates = specs_by_orig.get(rw.function_name, [])
        targets = [s for s in candidates
                   if s.orig_start <= rw.return_line <= s.orig_end]
        if not targets:
            raise FanoutError(
                f"return_widen names function {rw.function_name!r} but no emitted "
                f"variant clones it at return line {rw.return_line} (return "
                f"{rw.orig_type}->{rw.dd_type}); the closure demanded a return-widen on "
                f"a function no variant clones — a rule-(c) wiring bug (STOP #5)")
        for spec in targets:
            spec.return_widen = _merge_return_widen(
                spec.return_widen, rw, spec.variant_name)


def _select_leaf_overload(graph: CallGraph, name: str, arg_cores: set[str]):
    """Pick the overload of leaf ``name`` the chain actually CALLS (design §4a).

    ``Lnrat`` has two overloads — ``(TOutput,TOutput)`` control-flow (:126) and
    ``(TScale,TScale)`` shallow (:138) — and the B10 chain reaches only the LATTER:
    both ``Li2omx2``'s ``Lnrat(v,x)`` (``v,x`` ``TScale``) and ``B10``'s own direct
    ``Lnrat(si,mu2)`` etc. (``TScale`` / ``TMass`` locals) resolve to :138, because at
    instantiation ``TScale == TMass == DoubleDouble`` exact-matches ``(TScale,TScale)`` while
    ``(TOutput,TOutput) == (DoubleDoubleComplex,DoubleDoubleComplex)`` would need a conversion.
    ``resolve_primary_body`` / ``_pick_def`` both return ``defs[0]`` (the :126 one),
    which is WRONG: cloning it and rerouting the ``TScale``-arg calls to it would corrupt
    the leaf body.  ``_pick_def``'s ``must_call`` cannot disambiguate either (both
    overloads bottom out in the same boundary set).

    Select structurally by the call-site argument core types (unioned over every chain
    call of ``name`` — one clone serves them all, so they must agree): the reachable
    overload is the one whose parameter core types are all DRAWN FROM the argument cores
    (``{TScale} ⊆ {TScale,TMass}`` for :138; ``{TOutput} ⊄`` for :126).  Ties break to the
    most-specific (largest) parameter-core set.  A single-definition leaf skips matching
    (the common case).  Raises :class:`FanoutError` when no overload's parameters are
    covered, or the best match is ambiguous — e.g. two chain call sites that genuinely
    need different overloads — refusing to guess a leaf body rather than emit a
    wrong-typed clone (STOP).  Framework-agnostic: no identifier is hard-coded; the choice
    is a pure function of the graph's overload signatures and the call-site arg types.
    """
    fds = graph.defs.get(name) or []
    if not fds:
        raise FanoutError(f"leaf {name!r} has no definition in the call graph")
    if len(fds) == 1:
        return fds[0]
    scored: list[tuple[int, object]] = []
    for fd in fds:
        body = "\n".join(_original_text(fd.file, fd.line_start, fd.line_end))
        pcores = set(region_scan._param_name_core_types(
            region_scan._tokenize(body)).values())
        if pcores and pcores <= arg_cores:
            scored.append((len(pcores), fd))
    if not scored:
        raise FanoutError(
            f"leaf {name!r}: no overload of {len(fds)} whose parameter core types are "
            f"covered by the call-site argument cores {sorted(arg_cores)}; refusing to "
            f"guess a leaf body (STOP)")
    best = max(s for s, _ in scored)
    winners = [fd for s, fd in scored if s == best]
    if len(winners) != 1:
        raise FanoutError(
            f"leaf {name!r}: {len(winners)} overloads tie at the most-specific match for "
            f"call-site argument cores {sorted(arg_cores)} — two call sites likely need "
            f"different overloads; refusing to guess a leaf body (STOP)")
    return winners[0]


def _materialize_leaf_variants(
        closure: CarrierClosure, *, manifest: ChainManifest, graph: CallGraph,
        scalar_type: str, complex_type: str | None, complex_tokens,
        shim_include: str | None, new_specs: dict[str, dict[str, VariantSpec]],
        root_reroutes: dict[str, str]) -> list[str]:
    """Rule (d) L3: emit each discovered leaf clone + wire its caller reroute (§4).

    For every ``(g, g_clone)`` in ``closure.leaf_reroutes`` (``Lnrat`` -> ``Lnrat_B10``):

    1. locate the chain frames that CALL ``g`` and their call-site argument core types,
       and select the overload of ``g`` those calls resolve to
       (:func:`_select_leaf_overload` — the ``TScale`` ``Lnrat``, not the ``TOutput`` one);
    2. build a :class:`VariantSpec` that is a verbatim rename of that overload (``promotes=
       []`` — the body is not retyped, §1.1; its dd-ness rides on the return widen + the
       template instantiation), carrying a :class:`ReturnWiden` widening the return to the
       chain-consistent dd type.  That dd type is derived by :func:`_carrier_dd_type` with
       the SAME ``complex_tokens`` the closure used, so the clone returns EXACTLY the type
       the caller's receiving local was widened to (``prod`` -> ``DoubleDouble`` for B10) — the
       identical derivation rule (c) uses for ``Li2omx2``/``ddilog`` (line ~1422);
    3. add the caller-side reroute ``g -> g_clone`` to every variant of a calling frame
       (and, if the entry point itself calls ``g``, to ``root_reroutes``), so
       ``ql::Lnrat<…>(v,x)`` rewrites to ``ql::Lnrat_B10<…>(v,x)`` (design §4b).

    A clone that itself calls another discovered leaf reroutes that call too; a
    self-recursive leaf is handled by ``render_variant``'s ``orig_name -> variant_name``
    rename (design §3.2 — vacuous for B10, ``Lnrat`` is a sink, §2.7).  Inert when
    ``leaf_reroutes`` is empty (no ``leaf_ctx`` -> byte-identical, STOP #B).  Returns the
    emitted clone names.
    """
    emitted: list[str] = []
    if not closure.leaf_reroutes:
        return emitted

    # Chain frames = the enclosing functions of the chain lines (unique by extent).  A
    # leaf's callers are a subset of these (a leaf is called ON a chain line, §2.6).
    chain_fds: dict[tuple[str, int], object] = {}
    for (file, ls, le) in manifest.lines:
        fd = graph.enclosing_function(file, ls)
        if fd is not None:
            chain_fds[(fd.file, fd.line_start)] = fd

    def _calls_in(body: str):
        for ln in body.split("\n"):
            for call in _scan_calls(ln):
                yield call

    for g, g_clone in sorted(closure.leaf_reroutes.items()):
        # (1) callers of g among the chain frames + the call-site arg core types.
        caller_names: set[str] = set()
        arg_cores: set[str] = set()
        for fd in chain_fds.values():
            body = "\n".join(_original_text(fd.file, fd.line_start, fd.line_end))
            nct = region_scan.name_core_types(body)
            calls_g = False
            for (_qual, last, args) in _calls_in(body):
                if last == g:
                    calls_g = True
                    arg_cores |= {nct[a] for a in args if a in nct}
            if calls_g:
                caller_names.add(fd.name)

        leaf_fd = _select_leaf_overload(graph, g, arg_cores)

        # (2) the clone spec: verbatim body (promotes=[]) + return widen.
        leaf_body = "\n".join(
            _original_text(leaf_fd.file, leaf_fd.line_start, leaf_fd.line_end))
        spec = VariantSpec(variant_name=g_clone, orig_name=g, file=leaf_fd.file,
                           orig_start=leaf_fd.line_start, orig_end=leaf_fd.line_end)
        sig = _return_type_signature(leaf_body, leaf_fd.line_start)
        if sig is None:
            raise FanoutError(
                f"leaf {g!r}: return type unrecoverable (auto / SFINAE); cannot widen "
                f"the clone's return — it would truncate dd back to caller precision (STOP)")
        sig_line, orig_type = sig
        dd_type = _carrier_dd_type(orig_type, scalar_type, complex_type, complex_tokens)
        if orig_type != dd_type:
            spec.return_widen = ReturnWiden(
                return_line=sig_line, orig_type=orig_type, dd_type=dd_type,
                function_name=g_clone)
        # a clone that calls ANOTHER discovered leaf reroutes that call too (sink for B10).
        for g2, g2_clone in closure.leaf_reroutes.items():
            if g2 != g and any(last == g2 for (_q, last, _a) in _calls_in(leaf_body)):
                spec.reroutes[g2] = g2_clone
        if shim_include and shim_include not in spec.shim_includes:
            spec.shim_includes.append(shim_include)

        bucket = new_specs.setdefault(leaf_fd.file, {})
        if g_clone in bucket:
            raise FanoutError(
                f"leaf clone {g_clone!r} already emitted for this chain — a duplicate "
                f"leaf materialisation (wiring bug)")
        bucket[g_clone] = spec
        emitted.append(g_clone)

        # (3) caller-side reroute: g -> g_clone in every variant of a calling frame; the
        # in-place entry point is rerouted via root_reroutes when it is itself a caller.
        for specs in new_specs.values():
            for s in specs.values():
                if s.orig_name in caller_names:
                    s.reroutes[g] = g_clone
        if graph.root in caller_names:
            root_reroutes[g] = g_clone

    return emitted


def chain_promote(*, manifest: ChainManifest, graph: CallGraph,
                  tree_root: str | Path, scalar_type: str, two_limb: bool,
                  shim_include: str | None, caller_type: str = "double",
                  complex_type: str | None = None, max_paths: int = 1024,
                  app_source_roots=(),
                  leaf_ctx: "LeafPromotionContext | None" = None) -> ChainFanoutResult:
    """Widen a whole cascade chain to ``scalar_type`` (dd) as one coordinated envelope.

    Mutates the tree under ``tree_root`` in place (per-file AMP-FANOUT blocks + the
    entry-point body reroute) and returns a :class:`ChainFanoutResult` carrying the
    chain-scope 2c/2d gate verdicts.  Raises :class:`FanoutError` if a chain region
    is not inside a known function or its function is unreachable from the entry
    point.

    ``leaf_ctx`` (L2 opt-in) enables rule (d) leaf-callee promotion: the closure runs
    the frame-discovery fixed point, and the §2.8 circuit breaker is a terminal
    (``chain_closure_oversized``) computed BEFORE any tree mutation.  With
    ``leaf_ctx=None`` the pass is byte-identical to the pre-L2 behaviour (STOP #B).
    """
    tree = Path(tree_root).resolve()
    complex_tokens, caller_complex = _resolve_complex_binding(
        app_source_roots, caller_type, complex_type)

    # --- Blocker A: carrier closure (BLOCKER_A_CARRIER_DESIGN.md §4) -----------
    # Classify the chain's carriers BEFORE any tree mutation.  A strict carrier whose
    # decl cannot be widened (a function parameter, or shared state) is a terminal:
    # emitting the chain would leave a truncating interior seam the 2d-B gate then
    # rejects, so abandon the chain cleanly with the diagnostic status instead.  A
    # widenable carrier's name is threaded into every region's boundary transform so it
    # is neither demoted at a region exit nor read as inert, and its declaration is
    # widened in the variant (VariantSpec.closure_decls, applied per file below).
    closure = compute_value_closure(
        manifest=manifest, graph=graph, scalar_type=scalar_type,
        complex_type=complex_type, complex_tokens=complex_tokens, max_paths=max_paths,
        leaf_ctx=leaf_ctx)
    # Rule (d) §2.8 circuit breaker — abandon cleanly before any mutation (graceful
    # degradation, not a scope choice; for Group A this never fires — B10 frontier is
    # depth-1, 3 frames).
    if closure.chain_closure_oversized:
        return ChainFanoutResult(
            declared_variants=[], files_touched=[],
            chain_closure_oversized=True, oversized_detail=closure.oversized_detail)
    if closure.unwidenable_reasons:
        names = ", ".join(f"{n} ({r})" for n, r in closure.unwidenable_reasons)
        return ChainFanoutResult(
            declared_variants=[], files_touched=[],
            chain_carrier_unwidenable=True, carrier_detail=names)
    if closure.external_reasons:
        names = ", ".join(f"{n} ({r})" for n, r in closure.external_reasons)
        return ChainFanoutResult(
            declared_variants=[], files_touched=[],
            chain_carrier_external=True, carrier_detail=names)
    # CLOSURE_SCOPED_CHAINS_DESIGN §2.4 — a destination escape that materially severs a
    # carried value's dd flow to a designed exit: abandon the chain cleanly rather than
    # emit a truncating seam the interior gate would then reject.
    if closure.blocking_escapes:
        names = ", ".join(f"{n} ({r})" for n, r in closure.blocking_escapes)
        return ChainFanoutResult(
            declared_variants=[], files_touched=[],
            chain_closure_escapes=True, carrier_detail=names)
    closure_names = closure.closure_names

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
        # Region-core element promotion: fixed-size complex-aggregate bases whose element
        # reads must be wrapped (d1 leaves them at caller precision → STOP #CC otherwise).
        element_bases = (region_scan.region_element_bases(func_src, complex_tokens)
                         if complex_type else {})
        ckw = dict(complex_type=complex_type, complex_tokens=list(complex_tokens),
                   complex_names=complex_names, caller_complex=caller_complex,
                   closure_names=list(closure_names), element_bases=element_bases)

        _, promoted = boundary.promote_region_block(
            region_text, reads, writes, scalar_type, caller_type, two_limb,
            complex_type=complex_type, complex_tokens=frozenset(complex_tokens),
            complex_names=frozenset(complex_names), caller_complex=caller_complex,
            closure_names=frozenset(closure_names), element_bases=element_bases)
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
                new_specs=new_specs, root_reroutes=root_reroutes, name_maps=name_maps,
                closure_names=closure_names)
            depth = min(len(p) for p in paths) - 1     # hops from root (shallower = outer)

        region_meta.append(dict(depth=depth, span=(fd.file, ls, le),
                                region_text=region_text,
                                reads=reads, writes=writes, promoted=promoted))

    assert_no_collisions(name_maps)

    # --- closure: attach decl-widens to the variants that own them --------------
    # Each decl-widen record is (file, decl_line, orig_type, dd_type, name).  A closure
    # local lives in exactly one interior (non-entry) chain function (its rule-(a)/(b)
    # writes are interior so the function is never the in-place entry point), which the
    # fan-out copied to one variant PER caller path — the widened decl must ride on
    # EVERY such variant (the decl line is inside each copy's original extent).  The
    # emission pass (render_variant) rewrites the leading type token, widening the whole
    # multi-declarator in one edit (§2).
    for (dfile, dline, orig_type, dd_type, name) in closure.closure_decl_widens:
        for spec in new_specs.get(dfile, {}).values():
            if spec.orig_start <= dline <= spec.orig_end:
                cd = ClosureDecl(decl_line=dline, orig_type=orig_type,
                                 dd_type=dd_type, name=name)
                key = (cd.decl_line, cd.orig_type, cd.dd_type)
                if key not in {(c.decl_line, c.orig_type, c.dd_type)
                               for c in spec.closure_decls}:
                    spec.closure_decls.append(cd)

    # --- element promotion: attach complex-carrier names + designed exits ---------
    # Region-core element promotion (2026-07-28) carrier body reconciliation (deliverables
    # b + c).  A widened *complex* carrier (``closure_widenable`` whose ``dd_type`` is the
    # complex container) rides every variant whose extent holds its decl line; a
    # truncating designed exit (kernel_output / out_param / benign extract) rides every
    # variant whose extent holds its store line.  render_variant then widens a sibling
    # assignment to the carrier and demotes the carrier reads at those exits.  Gated on
    # ``complex_type`` so a scalar-only chain attaches nothing and stays byte-identical.
    if complex_type:
        for (dfile, dline, nm, dd_type) in closure.closure_widenable:
            if dd_type != complex_type:
                continue
            for spec in new_specs.get(dfile, {}).values():
                if spec.orig_start <= dline <= spec.orig_end:
                    if nm not in spec.closure_complex_names:
                        spec.closure_complex_names.append(nm)
        for (efile, eline, kind, _carried, _detail) in closure.designed_exits:
            for spec in new_specs.get(efile, {}).values():
                if spec.orig_start <= eline <= spec.orig_end:
                    rec = (eline, kind)
                    if rec not in spec.designed_exits:
                        spec.designed_exits.append(rec)

    # --- closure: attach return-type widens to the variants that own them --------
    # Rule (c) (Subtask 2b) records, per callee frame whose return the closure widens,
    # a ReturnWiden naming the variant, its signature line, and the orig->dd type.  Like
    # a closure local, a widened-return callee is copied to one variant PER caller path,
    # so the widen rides on EVERY variant of that function.  This is the wiring 2b will
    # exercise; here ``closure.return_widens`` is empty (rule (c) not yet implemented),
    # so this loop is a no-op.  Conflicts (two records widening one variant to different
    # types) are caught by VariantSpec.merge/_merge_return_widen; a record naming a
    # variant no emitted spec clones is a wiring bug (2b would create it) — fail loud.
    _attach_return_widens(closure.return_widens, new_specs)

    # --- rule (d) L3: materialise the discovered leaf-callee clones ---------------
    # ``closure.leaf_reroutes`` (rule (d), Subtask L2) names each leaf the chain reaches
    # that the design promotes by clone-and-rename (``Lnrat -> Lnrat_B10``) rather than by
    # in-place widen — a distinct symbol is what breaks the STOP #K forwarding-overload
    # self-recursion.  Emit each clone as its own VariantSpec (verbatim body + a
    # chain-consistent return widen) and wire the caller-side reroute onto every calling
    # frame's variants (and the entry point).  Empty with ``leaf_ctx=None`` (STOP #B), so
    # this is byte-identical to the pre-L3 pass unless rule (d) opted in.
    _materialize_leaf_variants(
        closure, manifest=manifest, graph=graph, scalar_type=scalar_type,
        complex_type=complex_type, complex_tokens=complex_tokens,
        shim_include=shim_include, new_specs=new_specs, root_reroutes=root_reroutes)

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
    # 2d-B fires only on INTERIOR regions (an intra-chain truncation breaks the chain);
    # the outermost region's truncation is the chain's designed exit boundary (Fix 1).
    promotion_applied = any(per_region_promoted)
    write_truncation = False
    if promotion_applied and region_meta:
        write_truncation = chain_write_truncation(
            region_meta, two_limb=two_limb, caller_type=caller_type,
            complex_tokens=frozenset(complex_tokens), caller_complex=caller_complex,
            closure_names=frozenset(closure_names),
            designed_exits=closure.designed_exits)

    return ChainFanoutResult(
        declared_variants=sorted(set(declared)),
        files_touched=sorted(touched), root_edited=root_edited,
        in_place_regions=in_place_regions, paths_enumerated=paths_enumerated,
        truncated=truncated, promotion_applied=promotion_applied,
        write_truncation=write_truncation, reroutes=dict(root_reroutes),
        reads_used=reads_used, closure_names=sorted(closure_names),
        leaf_reroutes=dict(closure.leaf_reroutes),
        leaf_frames=list(closure.leaf_frames),
        leaf_escapes=list(closure.leaf_escapes))
