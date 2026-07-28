"""Rule (d) — leaf-callee frame-discovery (Subtask L2, deliverables b/c/d).

Tests :func:`agents.patcher.chain_promote._discover_leaf_frames` via
:func:`compute_value_closure` / :func:`chain_promote` with a
:class:`~agents.patcher.chain_promote.LeafPromotionContext` (the L2 opt-in).  Two
layers:

* **STOP #B guard** — with ``leaf_ctx=None`` (every pre-L2 caller), the closure's
  a/b/c result is byte-identical, so rule (d) is provably inert unless opted in;
* **real B10/B13/B14 chains** (committed ``runs/qcdloop_headers_full`` tree) — rule
  (d) discovers exactly ``Lnrat`` on B10 (the design's depth-1 single-sink frontier,
  §2.7), records the per-integral clone reroute ``Lnrat -> Lnrat_B10`` WITHOUT
  emitting it (L3's job), and the §2.8 circuit breaker degrades gracefully.

The predicate itself is unit-tested in ``tests/patcher/test_clonable_leaf.py``; here
we pin its integration into the closure fixed point against the real tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.patcher import fanout
from agents.patcher.chain_promote import (
    ChainManifest, LeafPromotionContext, compute_value_closure,
)
from agents.integrator_base.shallow_wrapper import (
    _find_primary_defs, is_class1_synthesizable, surface_from_spelling,
)
from tests.patcher.fanout.conftest import requires_libclang, requires_qcdloop_full

_QCDLOOP_FULL = Path(__file__).resolve().parents[3] / "runs" / "qcdloop_headers_full"

# Real cascade chains (byte-identical to the committed tree; mirrors
# test_carrier_closure.py's _REAL_CHAINS so a source shift fails loudly here too).
_REAL_CHAINS = {
    "B10": ["B1m.h:227", "B1m.h:240", "B1m.h:241",
            "kokkosUtils.h:174", "kokkosUtils.h:177", "kokkosUtils.h:199",
            "kokkosUtils.h:212", "kokkosUtils.h:702", "kokkosUtils.h:703",
            "kokkosUtils.h:704"],
    "B13": ["B2m.h:300", "B2m.h:301", "B2m.h:305", "B2m.h:306", "B2m.h:355",
            "B2m.h:533", "kokkosUtils.h:212", "kokkosUtils.h:702"],
    "B14": ["B2m.h:401", "B2m.h:578", "kokkosUtils.h:1208"],
}

# Constants<T> accessors + template helpers the source instantiates at dd (the double
# primary at T=ddouble and the enriched dd source, §2.3).  A structural set (a leading
# ``_`` accessor / a known template helper) — the test's stand-in for the pipeline's
# real source-instantiation query, kept deliberately explicit for the trail.
_SOURCE_DD = frozenset(
    "_ipio2 _half _pi2o6 _pi _one _two _zero _four _three _C _num_C _ieps _reps "
    "_eps _neglig _qlonshellcutoff iszero kPow".split())


def _sources():
    return [p.read_text() for p in sorted(_QCDLOOP_FULL.glob("*.h"))]


def _make_ctx(graph, integral, *, max_frames=8, max_depth=3):
    srcs = _sources()

    def resolve(name):
        defs = _find_primary_defs(name, srcs)
        return defs[0] if defs else None

    def src_dd(name):
        return name in _SOURCE_DD

    surface = surface_from_spelling("quad::ddfun::ddouble", "quad::ddfun::ddcomplex")
    return LeafPromotionContext(
        graph=graph, surface=surface,
        is_class1_synthesizable=is_class1_synthesizable,
        source_instantiates_at_dd=src_dd, resolve_primary_body=resolve,
        integral=integral, max_frames=max_frames, max_depth=max_depth)


def _closure(graph, integral, *, leaf_ctx=None):
    lines = [(c.split(":")[0], int(c.split(":")[1]), int(c.split(":")[1]))
             for c in _REAL_CHAINS[integral]]
    man = ChainManifest(chain_id=f"cascade_{integral}", integral=integral,
                        entry_point="BO", lines=lines)
    return compute_value_closure(
        manifest=man, graph=graph, scalar_type="quad::ddfun::ddouble",
        complex_type="quad::ddfun::ddcomplex", leaf_ctx=leaf_ctx)


# --------------------------------------------------------------------------- #
# STOP #B — rule (d) is inert without an opt-in
# --------------------------------------------------------------------------- #

@requires_libclang
@requires_qcdloop_full
def test_no_leaf_ctx_is_byte_identical(qcdloop_full_graph):
    # The a/b/c closure result must be UNCHANGED whether or not rule (d) later runs.
    # Compare every enlarged-closure field with leaf_ctx=None vs a context that only
    # ADDS leaf_* fields; the pre-existing fields must match exactly (STOP #B).
    base = _closure(qcdloop_full_graph, "B10", leaf_ctx=None)
    ctx = _make_ctx(qcdloop_full_graph, "B10")
    withd = _closure(qcdloop_full_graph, "B10", leaf_ctx=ctx)
    assert withd.closure_widenable == base.closure_widenable
    assert withd.closure_decl_widens == base.closure_decl_widens
    assert withd.designed_exits == base.designed_exits
    assert withd.source_escapes == base.source_escapes
    assert withd.destination_escapes == base.destination_escapes
    assert withd.blocking_escapes == base.blocking_escapes
    assert [(r.function_name, r.return_line) for r in withd.return_widens] == \
           [(r.function_name, r.return_line) for r in base.return_widens]
    assert withd.closure_body_names == base.closure_body_names
    # and the base closure carries NO leaf-promotion records at all.
    assert base.leaf_frames == [] and base.leaf_reroutes == {}
    assert base.leaf_escapes == [] and base.chain_closure_oversized is False


# --------------------------------------------------------------------------- #
# real B10 — the headline case: rule (d) discovers exactly Lnrat (§2.7)
# --------------------------------------------------------------------------- #

@requires_libclang
@requires_qcdloop_full
def test_real_b10_discovers_lnrat_only(qcdloop_full_graph):
    ctx = _make_ctx(qcdloop_full_graph, "B10")
    cc = _closure(qcdloop_full_graph, "B10", leaf_ctx=ctx)
    # §2.7: the B10 rule-(d) frontier is depth-1 and closed — Lnrat is the single
    # clonable sink (ddilog/Li2omx2 are already chain frames for B10).
    assert cc.leaf_frames == ["Lnrat"]
    assert cc.leaf_escapes == []
    assert cc.chain_closure_oversized is False


@requires_libclang
@requires_qcdloop_full
def test_real_b10_records_per_integral_clone_reroute(qcdloop_full_graph):
    # Deliverable (d): RECORD the reroute (Lnrat -> Lnrat_B10), do NOT emit the clone
    # body (that is L3).  The clone name is per-integral so B10/B13 never collide.
    ctx = _make_ctx(qcdloop_full_graph, "B10")
    cc = _closure(qcdloop_full_graph, "B10", leaf_ctx=ctx)
    assert cc.leaf_reroutes == {"Lnrat": "Lnrat_B10"}


@requires_libclang
@requires_qcdloop_full
def test_real_b13_discovers_lnrat_distinct_clone(qcdloop_full_graph):
    # B13's chain also reaches Li2omx2's Lnrat call; the clone name is B13-scoped, so
    # Lnrat_B13 != Lnrat_B10 (per-integral clones, §3.3 — no cross-integral coupling).
    ctx = _make_ctx(qcdloop_full_graph, "B13")
    cc = _closure(qcdloop_full_graph, "B13", leaf_ctx=ctx)
    assert cc.leaf_frames == ["Lnrat"]
    assert cc.leaf_reroutes == {"Lnrat": "Lnrat_B13"}


@requires_libclang
@requires_qcdloop_full
def test_real_b14_discovers_no_leaf(qcdloop_full_graph):
    # B14 is dd-sufficient (Item 7); its dominant chain names no clonable leaf, so rule
    # (d) adds nothing (design §7 item 2 / §1.3 — Group A needs rule (d) for B10 only).
    ctx = _make_ctx(qcdloop_full_graph, "B14")
    cc = _closure(qcdloop_full_graph, "B14", leaf_ctx=ctx)
    assert cc.leaf_frames == []
    assert cc.leaf_reroutes == {}
    assert cc.chain_closure_oversized is False


# --------------------------------------------------------------------------- #
# §2.8 circuit breaker
# --------------------------------------------------------------------------- #

@requires_libclang
@requires_qcdloop_full
def test_real_b10_circuit_breaker_on_frame_cap(qcdloop_full_graph):
    # B10 has 3 chain frames (B10/ddilog/Li2omx2); with max_frames=3, admitting Lnrat
    # would make 4 -> the §2.8 breaker trips, no clone recorded (graceful degradation).
    ctx = _make_ctx(qcdloop_full_graph, "B10", max_frames=3)
    cc = _closure(qcdloop_full_graph, "B10", leaf_ctx=ctx)
    assert cc.chain_closure_oversized is True
    assert "§2.8 circuit breaker" in cc.oversized_detail
    assert cc.leaf_frames == []             # nothing admitted past the breaker


@requires_libclang
@requires_qcdloop_full
def test_real_b10_default_frame_cap_does_not_trip(qcdloop_full_graph):
    # For Group A the breaker never fires at the default 8-frame threshold (§2.8).
    ctx = _make_ctx(qcdloop_full_graph, "B10")     # default max_frames=8
    cc = _closure(qcdloop_full_graph, "B10", leaf_ctx=ctx)
    assert cc.chain_closure_oversized is False
    assert cc.leaf_frames == ["Lnrat"]


# --------------------------------------------------------------------------- #
# chain_promote end-to-end — the terminal + result surfacing (no tree mutation)
# --------------------------------------------------------------------------- #

@requires_libclang
@requires_qcdloop_full
def test_chain_promote_oversized_terminal(tmp_path, qcdloop_full_graph):
    # The §2.8 breaker is a chain_promote terminal computed BEFORE any tree mutation:
    # it abandons the chain cleanly (no variants) with chain_closure_oversized set.
    import shutil
    from agents.patcher.chain_promote import chain_promote
    for p in _QCDLOOP_FULL.glob("*.h"):
        shutil.copy(p, tmp_path / p.name)
    lines = [(c.split(":")[0], int(c.split(":")[1]), int(c.split(":")[1]))
             for c in _REAL_CHAINS["B10"]]
    man = ChainManifest(chain_id="cascade_B10", integral="B10",
                        entry_point="BO", lines=lines)
    ctx = _make_ctx(qcdloop_full_graph, "B10", max_frames=3)
    res = chain_promote(
        manifest=man, graph=qcdloop_full_graph, tree_root=tmp_path,
        scalar_type="quad::ddfun::ddouble", two_limb=True, shim_include=None,
        complex_type="quad::ddfun::ddcomplex", leaf_ctx=ctx)
    assert res.chain_closure_oversized is True
    assert res.declared_variants == []
    assert res.files_touched == []
    assert "§2.8 circuit breaker" in res.oversized_detail


# --------------------------------------------------------------------------- #
# L3 emission — _materialize_leaf_variants renders the clone + reroutes callers
# --------------------------------------------------------------------------- #

def _emit_over_tmp(tmp_path, integral, *, leaf_ctx_integral=None, complex_type=None):
    """Run ``chain_promote`` over a COMPLETE working copy of the vendored tree.

    The call graph must be built over the SAME tree the run mutates: ``chain_promote``
    writes to the graph's ``FuncDef.file`` paths, so building over the pristine snapshot
    while writing elsewhere would either corrupt the snapshot or miss the writes.  Copies
    the whole tree (incl. ``box/``) into ``tmp_path`` and roots the graph there.  Returns
    ``(res, tree)``; with ``leaf_ctx_integral=None`` no rule-(d) opt-in is passed (STOP
    #B — the pass is byte-identical to the pre-L3 behavior).
    """
    import shutil
    from agents.patcher import fanout
    from agents.patcher.call_graph import build_call_graph
    from agents.patcher.chain_promote import chain_promote

    tree = tmp_path / "tree"
    shutil.copytree(_QCDLOOP_FULL, tree)
    fanout.clear_graph_cache()
    graph = build_call_graph("BO", tree, tu_file=tree / "boxGPU.h")

    leaf_ctx = None
    if leaf_ctx_integral is not None:
        srcs = [p.read_text() for p in sorted(tree.glob("**/*.h"))]

        def resolve(name):
            defs = _find_primary_defs(name, srcs)
            return defs[0] if defs else None

        surface = surface_from_spelling(
            "quad::ddfun::ddouble", "quad::ddfun::ddcomplex")
        leaf_ctx = LeafPromotionContext(
            graph=graph, surface=surface,
            is_class1_synthesizable=is_class1_synthesizable,
            source_instantiates_at_dd=lambda n: n in _SOURCE_DD,
            resolve_primary_body=resolve, integral=leaf_ctx_integral)

    lines = [(c.split(":")[0], int(c.split(":")[1]), int(c.split(":")[1]))
             for c in _REAL_CHAINS[integral]]
    man = ChainManifest(chain_id=f"cascade_{integral}", integral=integral,
                        entry_point="BO", lines=lines)
    res = chain_promote(
        manifest=man, graph=graph, tree_root=tree,
        scalar_type="quad::ddfun::ddouble", two_limb=True, shim_include=None,
        complex_type=complex_type, leaf_ctx=leaf_ctx)
    return res, tree


@requires_libclang
@requires_qcdloop_full
def test_l3_emits_lnrat_clone_and_reroutes_callers(tmp_path):
    # The headline L3 deliverable: with the rule-(d) opt-in, chain_promote MATERIALISES
    # the discovered leaf clone (Lnrat_B10) as its own variant and reroutes every chain
    # caller's Lnrat call to it.
    res, tree = _emit_over_tmp(tmp_path, "B10", leaf_ctx_integral="B10",
                               complex_type="quad::ddfun::ddcomplex")
    assert res.chain_closure_oversized is False
    assert res.leaf_reroutes == {"Lnrat": "Lnrat_B10"}
    assert "Lnrat_B10" in res.declared_variants

    txt = (tree / "kokkosUtils.h").read_text()
    # (1) the clone is emitted as the @138 SHALLOW overload (TScale,TScale params) —
    #     never the @126 control-flow (TOutput,TOutput) one.
    assert "Lnrat_B10(TScale const& x, TScale const& y)" in txt
    # (2) the clone body is verbatim (still calls the source-provided dd boundaries) and
    #     never names itself or ql::Lnrat (no self-recursion — STOP #K premise).
    clone = txt[txt.index("variant Lnrat_B10"):]
    clone = clone[:clone.index("// --- variant", 1)]
    assert "ql::kLog(ql::kAbs(x / y))" in clone
    assert "Lnrat_B10" not in clone.split("Lnrat_B10(", 1)[1]   # body has no self-call
    # (3) every Li2omx2 chain call is rerouted to the clone; the ORIGINAL Lnrat and its
    #     original callers are untouched (renames live only on the emitted variants).
    assert "ql::Lnrat_B10<TOutput, TMass, TScale>(v, x)" in txt


@requires_libclang
@requires_qcdloop_full
def test_l3_no_leaf_ctx_emits_no_clone(tmp_path):
    # STOP #B at the emission layer: without the opt-in, _materialize_leaf_variants is
    # inert — no Lnrat_B10 anywhere, and leaf_reroutes stays empty.
    res, tree = _emit_over_tmp(tmp_path, "B10", leaf_ctx_integral=None,
                               complex_type="quad::ddfun::ddcomplex")
    assert res.leaf_reroutes == {}
    assert "Lnrat_B10" not in res.declared_variants
    assert "Lnrat_B10" not in (tree / "kokkosUtils.h").read_text()


@requires_libclang
@requires_qcdloop_full
def test_l3_leaves_vendored_snapshot_pristine(tmp_path):
    # The security invariant: all mutation is confined to the tmp working copy; the
    # committed vendored snapshot is byte-for-byte unchanged by an emission run.
    before = {p.name: p.read_bytes()
              for p in sorted(_QCDLOOP_FULL.glob("**/*.h"))}
    _emit_over_tmp(tmp_path, "B10", leaf_ctx_integral="B10",
                   complex_type="quad::ddfun::ddcomplex")
    after = {p.name: p.read_bytes()
             for p in sorted(_QCDLOOP_FULL.glob("**/*.h"))}
    assert before == after
