"""Ranking function — the two class-driven queues (design: "Ranking function")."""

import json

from agents.strategy.ranking import (
    build_correctness_queue, build_queues, build_speedup_queue, error_threshold,
    flop_weighted_score, load_flop_weights,
)
from tests.strategy.conftest import make_region

# Minimal flop-weight table (shape of ratio_multipliers.json): log ≫ add in ff.
_WEIGHTS = {"native_double": {"add": 1, "mul": 1, "div": 1, "log": 20},
            "native_float": {"add": 1, "mul": 1, "div": 1, "log": 20},
            "ff": {"add": 11, "mul": 32, "div": 42, "log": 2350}}

TOL = 10.0
ABOVE = 1e-8   # > 1e-10 threshold → needs correctness
BELOW = 1e-12  # < 1e-10 threshold → fine


def test_error_threshold():
    assert error_threshold(10.0) == 1e-10


def test_tier_order_is_fixed():
    # one region per class, all above the error threshold where relevant
    lc = make_region("B14", "B2m.h", 401, "local_cancellation", max_cond=1e16, max_rel_err=ABOVE)
    cc = make_region("B12", "B2m.h", 355, "cancellation_cascade", max_cond=1e6, max_rel_err=ABOVE)
    lnr = make_region("B10", "B0m.h", 230, "log_near_root", max_cond=1e3, max_rel_err=ABOVE)
    st = make_region("B1", "B0m.h", 116, "stable", max_cond=10.0, max_rel_err=ABOVE)

    q = build_correctness_queue([st, lnr, cc, lc], TOL)
    assert [r.signal_class for r in q] == [
        "local_cancellation", "cancellation_cascade", "log_near_root", "stable"]


def test_local_cancellation_always_tier1_even_at_low_error():
    # local_cancellation is not gated on max_rel_err — cond > 1e15 IS the class.
    lc = make_region("B14", "B2m.h", 401, "local_cancellation", max_cond=1e16, max_rel_err=BELOW)
    q = build_correctness_queue([lc], TOL)
    assert [r.signal_class for r in q] == ["local_cancellation"]


def test_cascade_below_threshold_excluded():
    cc = make_region("B12", "B2m.h", 355, "cancellation_cascade", max_rel_err=BELOW)
    assert build_correctness_queue([cc], TOL) == []


def test_stable_within_tolerance_not_in_correctness():
    st = make_region("B1", "B0m.h", 116, "stable", max_rel_err=BELOW)
    assert build_correctness_queue([st], TOL) == []


def test_stable_surprising_error_is_tier4():
    st = make_region("B1", "B0m.h", 116, "stable", max_rel_err=ABOVE)
    q = build_correctness_queue([st], TOL)
    assert len(q) == 1 and q[0].signal_class == "stable"


def test_intratier_sorted_by_cond_desc():
    a = make_region("B1", "f.h", 1, "local_cancellation", max_cond=1e16)
    b = make_region("B1", "f.h", 2, "local_cancellation", max_cond=9e17)
    c = make_region("B1", "f.h", 3, "local_cancellation", max_cond=5e15)
    q = build_correctness_queue([a, b, c], TOL)
    assert [r.max_cond for r in q] == [9e17, 1e16, 5e15]


def test_speedup_ranked_by_op_count_desc():
    small = make_region("B1", "f.h", 1, "stable", pred_float=BELOW, op_count=3)
    big = make_region("B1", "f.h", 2, "stable", pred_float=BELOW, op_count=99)
    mid = make_region("B1", "f.h", 3, "stable", pred_float=BELOW, op_count=20)
    q = build_speedup_queue([small, big, mid], TOL)
    assert [r.op_count for r in q] == [99, 20, 3]


def test_speedup_excluded_when_not_even_ff_safe():
    # pred_ff defaults to pred_float here (1e-7 > 1e-10 at tol=10): can't meet
    # tolerance even in ff → not demotable at all → excluded.
    unsafe = make_region("B1", "f.h", 1, "stable", pred_float=1e-7, op_count=50)
    assert build_speedup_queue([unsafe], TOL) == []


def test_speedup_admits_ff_only_safe_region():
    # float-unsafe (1e-7 > 1e-10) but ff-safe (1e-12 <= 1e-10): admitted for a
    # double->ff demotion, which the strict float gate used to drop entirely.
    ff_only = make_region("B1", "f.h", 1, "stable", pred_float=1e-7, pred_ff=1e-12,
                          op_count=50)
    q = build_speedup_queue([ff_only], TOL)
    assert [r.target.location for r in q] == ["f.h:1"]


def test_speedup_ff_gate_subsumes_float_at_low_tolerance():
    # At tolerance 6 (thr 1e-6) a float-safe region is still admitted (ff <= float
    # <= thr); the walk can demote it all the way to float.
    float_safe = make_region("B1", "f.h", 1, "stable", pred_float=1e-8, pred_ff=1e-13,
                             op_count=50)
    assert len(build_speedup_queue([float_safe], 6.0)) == 1


def test_speedup_excludes_correctness_regions():
    # a stable region above the error bar AND float-safe: goes to correctness tier4,
    # must NOT also appear in speedup.
    dual = make_region("B1", "f.h", 1, "stable", max_rel_err=ABOVE, pred_float=BELOW, op_count=50)
    corr, spd = build_queues([dual], TOL)
    assert len(corr) == 1
    assert spd == []


def test_speedup_nonstable_excluded():
    lc = make_region("B1", "f.h", 1, "local_cancellation", pred_float=BELOW, op_count=50)
    assert build_speedup_queue([lc], TOL) == []


# ---------------------------------------------------------------------------
# Wave-3 WI3: flop-weighted speedup ordering
# ---------------------------------------------------------------------------

def test_flop_weight_reorders_over_raw_op_count():
    # equal op_count (10) but different mix: log-heavy vs add-heavy.  Raw op_count
    # ties → location; flop-weight (ff col: log=2350 ≫ add=11) puts log-heavy first.
    add_heavy = make_region("B1", "a.h", 1, "stable", pred_ff=BELOW,
                            op_count=10, ops={"add": 10})
    log_heavy = make_region("B1", "z.h", 9, "stable", pred_ff=BELOW,
                            op_count=10, ops={"log": 9, "add": 1})
    # raw op_count order: tie broken by location → a.h:1 before z.h:9
    raw = build_speedup_queue([log_heavy, add_heavy], TOL)
    assert [r.target.location for r in raw] == ["a.h:1", "z.h:9"]
    # flop-weighted: log-heavy dominates despite the later location
    weighted = build_speedup_queue([add_heavy, log_heavy], TOL, flop_weights=_WEIGHTS)
    assert [r.target.location for r in weighted] == ["z.h:9", "a.h:1"]


def test_flop_weighted_score_uses_target_column():
    r = make_region("B1", "f.h", 1, "stable", ops={"log": 2, "add": 3})
    # ff column: 2*2350 + 3*11 = 4733
    assert flop_weighted_score(r, _WEIGHTS, "ff") == 4733
    # float column (native_float): 2*20 + 3*1 = 43
    assert flop_weighted_score(r, _WEIGHTS, "float") == 43
    # unknown op defaults to weight 1
    r2 = make_region("B1", "f.h", 2, "stable", ops={"weirdop": 4})
    assert flop_weighted_score(r2, _WEIGHTS, "ff") == 4


def test_build_speedup_falls_back_to_op_count_without_weights():
    small = make_region("B1", "f.h", 1, "stable", pred_ff=BELOW, op_count=3)
    big = make_region("B1", "f.h", 2, "stable", pred_ff=BELOW, op_count=99)
    q = build_speedup_queue([small, big], TOL, flop_weights=None)
    assert [r.op_count for r in q] == [99, 3]


def test_load_flop_weights_missing_returns_none(tmp_path):
    assert load_flop_weights(tmp_path / "nope.json") is None
    assert load_flop_weights(None) is None


def test_load_flop_weights_reads_table(tmp_path):
    p = tmp_path / "ratio.json"
    p.write_text(json.dumps(_WEIGHTS))
    w = load_flop_weights(p)
    assert w["ff"]["log"] == 2350
