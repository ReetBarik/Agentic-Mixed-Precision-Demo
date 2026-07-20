"""Retry-walk state machine (design: "Retry policy")."""

import pytest

from agents.strategy.walk import RetryWalk
from tests.strategy.conftest import make_region


def drive(mode, signal_class, outcomes, baseline="double"):
    """Run a walk against a scripted list of (accepted, genuine_reject) outcomes.

    Returns (proposed_intents, WalkResult).
    """
    rec = make_region("B1", "f.h", 10, signal_class, max_cond=1e16)
    walk = RetryWalk(rec, mode, tolerance=10.0, baseline=baseline)
    proposed = []
    i = 0
    while True:
        intent = walk.propose(f"iter_{i}")
        if intent is None:
            break
        proposed.append(intent)
        acc, gen = outcomes[i]
        walk.resolve(accepted=acc, genuine_reject=gen)
        i += 1
    return proposed, walk.result()


# ---- correctness ladder ----

def test_correctness_dd_clears_immediately():
    proposed, res = drive("correctness", "stable", [(True, False)])
    assert [p.kind for p in proposed] == ["double-to-dd"]
    assert res.status == "cleared" and res.final_precision == "dd"


def test_correctness_ff_baseline_walks_up():
    # from ff baseline: ff-to-double (reject) then ff-to-dd (clears)
    proposed, res = drive("correctness", "stable",
                          [(False, True), (True, False)], baseline="ff")
    assert [p.kind for p in proposed] == ["ff-to-double", "ff-to-dd"]
    assert res.final_precision == "dd" and res.status == "cleared"


def test_correctness_current_precision_from_baseline():
    proposed, _ = drive("correctness", "stable",
                        [(False, True), (False, True)], baseline="ff")
    # both promotions transition FROM the ff baseline (correctness never keeps a
    # rejected rung), so current_precision stays ff until DD is retained.
    assert proposed[0].current_precision == "ff"
    assert proposed[1].current_precision == "ff"


# ---- DD ceiling detection ----

def test_cascade_ceiling_tries_kahan():
    proposed, res = drive("correctness", "cancellation_cascade",
                          [(False, True), (False, True)])
    assert [p.kind for p in proposed] == ["double-to-dd", "reformulate-kahan"]
    assert res.status == "dd_ceiling" and res.ceiling_kind == "dd_ceiling"
    assert res.attempted_rewrites == ["kahan"] and res.final_precision == "dd"


def test_local_cancellation_ceiling_walks_identity_catalog():
    proposed, res = drive("correctness", "local_cancellation",
                          [(False, True)] + [(False, True)] * 4)
    kinds = [p.kind for p in proposed]
    assert kinds[0] == "double-to-dd"
    assert all(k == "reformulate-identity" for k in kinds[1:])
    assert res.attempted_rewrites == ["log1p", "expm1", "hypot", "1-cos->2sin2"]
    assert res.status == "dd_ceiling"


def test_rewrite_clears_layers_on_dd():
    # dd reject → identity log1p reject → expm1 clears; DD retained under rewrite.
    proposed, res = drive("correctness", "local_cancellation",
                          [(False, True), (False, True), (True, False)])
    assert res.status == "cleared" and res.rewrite_accepted is True
    assert res.final_precision == "dd"            # rewrite layered on DD (Q2)
    assert proposed[-1].current_precision == "dd"
    assert res.attempted_rewrites == ["log1p", "expm1"]


def test_log_near_root_no_rewrite_straight_to_ceiling():
    proposed, res = drive("correctness", "log_near_root", [(False, True)])
    assert [p.kind for p in proposed] == ["double-to-dd"]
    assert res.status == "dd_ceiling" and res.attempted_rewrites == []


# ---- dd_untested vs dd_ceiling (P6a) ----

def test_dd_untested_on_patcher_failure():
    # Patcher failed at the DD rung → not a genuine reject → dd_untested.
    proposed, res = drive("correctness", "cancellation_cascade", [(False, False)])
    assert [p.kind for p in proposed] == ["double-to-dd"]
    assert res.status == "dd_untested" and res.ceiling_kind == "dd_untested"
    assert res.final_precision == "double"   # DD not installed
    assert res.attempted_rewrites == []      # no rewrite attempted


def test_dd_untested_never_tries_rewrite():
    # even for a rewrite-eligible class, an untested DD short-circuits.
    _, res = drive("correctness", "local_cancellation", [(False, False)])
    assert res.status == "dd_untested"


# ---- speedup ladder ----

def test_speedup_walks_down_and_backs_off():
    # double-to-ff accepts, ff-to-float rejects → settle at ff.
    proposed, res = drive("speedup", "stable", [(True, False), (False, False)])
    assert [p.kind for p in proposed] == ["double-to-ff", "ff-to-float"]
    assert res.status == "settled" and res.final_precision == "ff"


def test_speedup_first_reject_stops_at_baseline():
    proposed, res = drive("speedup", "stable", [(False, False)])
    assert [p.kind for p in proposed] == ["double-to-ff"]
    assert res.final_precision == "double"


def test_speedup_all_the_way_to_float():
    proposed, res = drive("speedup", "stable", [(True, False), (True, False)])
    assert [p.kind for p in proposed] == ["double-to-ff", "ff-to-float"]
    assert res.final_precision == "float"


# ---- speedup required_by floor (cascade-chain overlap) ----

def test_speedup_floor_blocks_demotion_below_floor():
    # a promoted chain requires this line at dd → speedup must not demote at all
    rec = make_region("B1", "f.h", 10, "stable", pred_float=1e-30)
    walk = RetryWalk(rec, "speedup", tolerance=10.0, baseline="double", floor="dd")
    intent = walk.propose("iter_0")
    assert intent is None                       # double-to-ff would drop below dd
    assert walk.result().status == "settled"
    assert walk.result().final_precision == "double"


def test_speedup_floor_allows_demotion_down_to_floor():
    # floor ff: double may demote to ff (accepted) but not below to float
    rec = make_region("B1", "f.h", 10, "stable", pred_float=1e-30)
    walk = RetryWalk(rec, "speedup", tolerance=10.0, baseline="double", floor="ff")
    first = walk.propose("iter_0")
    assert first.kind == "double-to-ff"
    walk.resolve(accepted=True)                 # demoted to ff
    assert walk.propose("iter_1") is None        # ff-to-float would breach the floor
    res = walk.result()
    assert res.status == "settled" and res.final_precision == "ff"


def test_speedup_no_floor_demotes_to_float():
    # sanity: without a floor the same region walks all the way to float
    rec = make_region("B1", "f.h", 10, "stable", pred_float=1e-30)
    walk = RetryWalk(rec, "speedup", tolerance=10.0, baseline="double")
    i = 0
    while (intent := walk.propose(f"iter_{i}")) is not None:
        walk.resolve(accepted=True)
        i += 1
    assert walk.result().final_precision == "float"


# ---- template-typed float rung: Wave-2 regional path (was CALIBRATION.md §Bug 4) ----

def test_speedup_template_region_tries_double_to_float_directly():
    # Wave 2: a template-typed region (float_via=VIA_REGIONAL) now attempts
    # `double-to-float` DIRECTLY via the regional float integrator — the FIRST
    # thing proposed — instead of settling at ff (the Wave-1 gate).  Accepting it
    # settles at float (the cheapest passing rung).  The intent is tagged
    # via="regional" so the Patcher routes it to the float integrator.
    from agents.strategy.models import VIA_REGIONAL
    rec = make_region("B1", "f.h", 10, "stable", pred_float=1e-30)
    walk = RetryWalk(rec, "speedup", tolerance=10.0, baseline="double",
                     float_via=VIA_REGIONAL)
    intent = walk.propose("iter_0")
    assert intent.kind == "double-to-float"
    assert intent.via == VIA_REGIONAL
    walk.resolve(accepted=True)                 # float accepted → cheapest, stop
    assert walk.propose("iter_1") is None
    res = walk.result()
    assert res.status == "settled" and res.final_precision == "float"


def test_speedup_template_region_falls_back_to_ff_when_float_rejects():
    # Wave 2: if `double-to-float` rejects (float too lossy), the regional walk
    # falls back to `double-to-ff` (preserving the demotion Wave 1 already won).
    from agents.strategy.models import VIA_REGIONAL
    rec = make_region("B1", "f.h", 10, "stable", pred_float=1e-30)
    walk = RetryWalk(rec, "speedup", tolerance=10.0, baseline="double",
                     float_via=VIA_REGIONAL)
    kinds = []
    intent = walk.propose("iter_0")
    kinds.append(intent.kind)
    walk.resolve(accepted=False)                # float rejected → try ff
    intent = walk.propose("iter_1")
    kinds.append(intent.kind)
    assert intent.via == VIA_REGIONAL
    walk.resolve(accepted=True)                 # ff accepted → cheapest passing, stop
    assert walk.propose("iter_2") is None
    res = walk.result()
    assert kinds == ["double-to-float", "double-to-ff"]
    assert res.status == "settled" and res.final_precision == "ff"


def test_speedup_template_region_regional_floor_blocks_float():
    # The cascade-chain floor still binds the regional path: floor=ff means float
    # is skipped and only ff is tried.
    from agents.strategy.models import VIA_REGIONAL
    rec = make_region("B1", "f.h", 10, "stable", pred_float=1e-30)
    walk = RetryWalk(rec, "speedup", tolerance=10.0, baseline="double",
                     floor="ff", float_via=VIA_REGIONAL)
    intent = walk.propose("iter_0")
    assert intent.kind == "double-to-ff"        # float target below floor → skipped
    walk.resolve(accepted=False)
    assert walk.propose("iter_1") is None
    assert walk.result().final_precision == "double"


def test_speedup_non_template_region_still_reaches_float_via_plain():
    # Control: float_via=VIA_PLAIN (default) keeps the historical single-step
    # ladder for non-template code (double→ff→ff-to-float via plain edit).
    from agents.strategy.models import VIA_PLAIN
    rec = make_region("B1", "f.h", 10, "stable", pred_float=1e-30)
    walk = RetryWalk(rec, "speedup", tolerance=10.0, baseline="double",
                     float_via=VIA_PLAIN)
    kinds = []
    while (intent := walk.propose("iter")) is not None:
        kinds.append(intent.kind)
        assert intent.via == VIA_PLAIN
        walk.resolve(accepted=True)
    assert kinds == ["double-to-ff", "ff-to-float"]
    assert walk.result().final_precision == "float"


# ---- Wave-3 WI1/WI2: float-rung guard (float_ok=False) ----

def test_speedup_plain_float_ok_false_settles_at_ff():
    # plain ladder with the guard on: double->ff accepts, then float is NOT
    # attempted (guard) → settle at ff after a single accepted step.
    rec = make_region("B1", "f.h", 10, "stable", pred_float=1e-30)
    walk = RetryWalk(rec, "speedup", tolerance=10.0, baseline="double", float_ok=False)
    first = walk.propose("i0")
    assert first.kind == "double-to-ff"
    walk.resolve(accepted=True)
    assert walk.propose("i1") is None            # ff->float guarded off
    res = walk.result()
    assert res.status == "settled" and res.final_precision == "ff"


def test_speedup_regional_float_ok_false_skips_float_target():
    # regional plan drops the float rung entirely: only double->ff is proposed.
    from agents.strategy.models import VIA_REGIONAL
    rec = make_region("B1", "f.h", 10, "stable", pred_float=1e-30)
    walk = RetryWalk(rec, "speedup", tolerance=10.0, baseline="double",
                     float_via=VIA_REGIONAL, float_ok=False)
    first = walk.propose("i0")
    assert first.kind == "double-to-ff"
    walk.resolve(accepted=False)
    assert walk.propose("i1") is None            # no float fallback attempted
    assert walk.result().final_precision == "double"


def test_speedup_float_ok_true_still_reaches_float():
    # control: guard off (default True) preserves the Wave-2 reach-to-float path.
    rec = make_region("B1", "f.h", 10, "stable", pred_float=1e-30)
    walk = RetryWalk(rec, "speedup", tolerance=10.0, baseline="double", float_ok=True)
    kinds = []
    while (intent := walk.propose("i")) is not None:
        kinds.append(intent.kind)
        walk.resolve(accepted=True)
    assert kinds == ["double-to-ff", "ff-to-float"]
    assert walk.result().final_precision == "float"


# ---- float-baseline edge (float-to-dd unsupported) ----

def test_float_baseline_exhausts_without_dd():
    proposed, res = drive("correctness", "stable",
                          [(False, True), (False, True)], baseline="float")
    assert [p.kind for p in proposed] == ["float-to-ff", "float-to-double"]
    assert res.status == "exhausted" and res.final_precision == "float"


# ---- protocol guards ----

def test_propose_before_resolve_raises():
    rec = make_region("B1", "f.h", 10, "stable")
    walk = RetryWalk(rec, "correctness", 10.0)
    walk.propose("iter_0")
    with pytest.raises(RuntimeError):
        walk.propose("iter_1")


def test_result_before_termination_raises():
    rec = make_region("B1", "f.h", 10, "stable")
    walk = RetryWalk(rec, "correctness", 10.0)
    with pytest.raises(RuntimeError):
        walk.result()
