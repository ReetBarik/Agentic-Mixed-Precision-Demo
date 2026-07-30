"""Whole-TU-only Strategy walk (``strategy_mode="tu_only"``).

Drives the mechanical walk with a *scripted* ``tu_measure_fn`` (no builds, no LLM):
a mock report with three integrals exercising each terminal state —

* ``B_ok``   — double already clears the bar  -> tu_no_flip_needed, routes double,
               then downshifts to ff (still clears) -> routes ff.
* ``B_dd``   — double below the bar, dd rescues -> tu_accepted (correctness),
               float/ff too narrow -> stays dd.
* ``B_ff``   — double below the bar, dd not attempted needed? dd fails to clear,
               float too narrow, ff clears -> routes ff.

Asserts the per-integral routing, the gate verdicts, and that neither the Patcher
nor the Validator callable is ever consulted in tu_only mode.
"""

import json

from agents.config import StrategyBudget, StrategyConfig
from agents.strategy import agent as strategy_agent
from agents.strategy.tu_walk import (
    TU_ACCEPTED, TU_BUILD_FAILED, TU_NO_FLIP_NEEDED, TU_REJECTED_BELOW_TOL,
    float_is_candidate,
)


# --------------------------------------------------------------------------
# mock report
# --------------------------------------------------------------------------

def _region(pred_float):
    return {"signal_class": "stable", "max_cond": 10.0, "max_rel_err": 1e-16,
            "predicted_rel_err_if_float": pred_float,
            "predicted_rel_err_if_ff": pred_float,
            "prov_vars": ["v"], "region_local_vars": ["v"], "ops": {"mul": 1},
            "n": 100, "non_localizable": False, "value_range_ok_for_float": True}


def write_report(tmp_path, integrals):
    """integrals: {name: pred_float} -> one localizable region each."""
    data = {"kind": "stability_report", "schema_version": 1, "samples_seen": {},
            "integrals": {
                name: {"class_counts": {}, "cascade_chains": [],
                       "regions": {f"{name}m.h:10": _region(pf)}}
                for name, pf in integrals.items()}}
    p = tmp_path / "report.json"
    p.write_text(json.dumps(data))
    return str(p)


# --------------------------------------------------------------------------
# scripted tu_measure_fn: table[(integral, target)] -> measure dict
# --------------------------------------------------------------------------

def make_measure_fn(table, *, calls=None):
    def tu_measure_fn(integral, target):
        if calls is not None:
            calls.append((integral, target))
        return table[(integral, target)]
    return tu_measure_fn


def _base(d):
    return {"built": True, "baseline_digits": d}


def _cand(base, cand, *, built=True):
    return {"built": built, "baseline_digits": base, "candidate_digits": cand,
            "log_tail": "" if built else "compile error"}


def run_tu(tmp_path, report, measure_fn, *, tolerance=7.0, promote_fn=None):
    cfg = StrategyConfig(tolerance=tolerance, runs_root=tmp_path / "runs",
                         strategy_mode="tu_only",
                         budget=StrategyBudget(max_iters=10**6))
    state = {
        "characterization_report_path": report,
        "strategy_repo_path": None, "strategy_starting_sha": None,
        "patcher_fn": None, "validator_fn": None,
        "tu_measure_fn": measure_fn, "tu_promote_fn": promote_fn,
        "strategy_config": cfg,
    }
    delta = strategy_agent.run(state)
    result = delta["strategy_result"]
    report_json = json.loads(
        (tmp_path / "runs" / "strategy").glob("*/report.json").__next__().read_text())
    return result, report_json


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def test_three_integral_routing(tmp_path):
    # B_ok: double clears (12 >= 7); ff also clears -> ff.
    # B_dd: double below (5 < 7); dd rescues (15.9); float/ff too narrow -> dd.
    # B_ff: double below (5 < 7); dd fails to clear (below bar); float narrow, ff clears -> ff.
    report = write_report(tmp_path, {
        "B_ok": 1e-2,   # float-unsafe (pred > 1e-7) — float pruned
        "B_dd": 1e-2,
        "B_ff": 1e-2,
    })
    table = {
        # baseline
        ("B_ok", "baseline"): _base(12.0),
        ("B_dd", "baseline"): _base(5.0),
        ("B_ff", "baseline"): _base(5.0),
        # correctness dd (B_ok skipped: baseline already clears)
        ("B_dd", "dd"): _cand(5.0, 15.9),     # clears 7 -> dd
        # dd=6.0 below bar but a strict lift (+1) -> UPSHIFT accepts (best-effort);
        # the speedup ff below overrides it since ff clears the bar.
        ("B_ff", "dd"): _cand(5.0, 6.0),
        # speedup ff (float pruned for all — pred_float=1e-2)
        ("B_ok", "ff"): _cand(12.0, 9.0),     # clears 7 -> ff
        ("B_dd", "ff"): _cand(5.0, 4.0),      # below bar -> reject, stays dd
        ("B_ff", "ff"): _cand(5.0, 8.0),      # clears 7 -> ff
    }
    calls = []
    result, rep = run_tu(tmp_path, report, make_measure_fn(table, calls=calls))

    routing = result["tu_routing"]
    assert routing == {"B_ok": "ff", "B_dd": "dd", "B_ff": "ff"}
    assert result["precision_distribution"]["ff"] == 2
    assert result["precision_distribution"]["dd"] == 1
    assert result["precision_distribution"]["double"] == 0

    # float never attempted (pred_float=1e-2 > error_threshold(7)=1e-7).
    assert not any(t == "float" for _, t in calls)
    # B_ok skipped dd (baseline already clears) — no dd call for it.
    assert ("B_ok", "dd") not in calls
    # B_dd, once routed dd and its ff rejected, still recorded the ff attempt.
    assert ("B_dd", "ff") in calls
    # B_ff routed ff, so its ff was attempted after dd rejected.
    assert ("B_ff", "dd") in calls and ("B_ff", "ff") in calls


def test_statuses_recorded(tmp_path):
    report = write_report(tmp_path, {"B_ok": 1e-2, "B_dd": 1e-2, "B_ff": 1e-2})
    table = {
        ("B_ok", "baseline"): _base(12.0),
        ("B_dd", "baseline"): _base(5.0),
        ("B_ff", "baseline"): _base(5.0),
        ("B_dd", "dd"): _cand(5.0, 15.9),
        ("B_ff", "dd"): _cand(5.0, None, built=False),   # build fail
        ("B_ok", "ff"): _cand(12.0, 9.0),
        ("B_dd", "ff"): _cand(5.0, 4.0),
        ("B_ff", "ff"): _cand(5.0, 8.0),
    }
    _, rep = run_tu(tmp_path, report, make_measure_fn(table))

    rows = {r["integral"]: r for r in rep["tu_rows"]}
    # B_ok correctness candidate is no_flip_needed (double clears).
    dd_ok = [c for c in rows["B_ok"]["candidates"] if c["target"] == "dd"]
    assert dd_ok and dd_ok[0]["status"] == TU_NO_FLIP_NEEDED
    # B_dd dd is accepted.
    dd_dd = [c for c in rows["B_dd"]["candidates"] if c["target"] == "dd"][0]
    assert dd_dd["status"] == TU_ACCEPTED
    # B_ff dd is a build failure.
    dd_ff = [c for c in rows["B_ff"]["candidates"] if c["target"] == "dd"][0]
    assert dd_ff["status"] == TU_BUILD_FAILED
    # B_dd ff is rejected below tolerance.
    ff_dd = [c for c in rows["B_dd"]["candidates"] if c["target"] == "ff"][0]
    assert ff_dd["status"] == TU_REJECTED_BELOW_TOL


def test_float_attempted_when_signal_plausible(tmp_path):
    # pred_float below error_threshold(7)=1e-7 -> float IS attempted.
    report = write_report(tmp_path, {"B_x": 1e-9})
    table = {
        ("B_x", "baseline"): _base(5.0),
        ("B_x", "dd"): _cand(5.0, 15.9),
        ("B_x", "float"): _cand(5.0, 8.0),   # clears 7 -> float wins, ff not tried
        ("B_x", "ff"): _cand(5.0, 9.0),
    }
    calls = []
    result, _ = run_tu(tmp_path, report, make_measure_fn(table, calls=calls))
    assert result["tu_routing"]["B_x"] == "float"
    assert ("B_x", "float") in calls
    # float cleared -> ff not attempted (first clearing precision wins).
    assert ("B_x", "ff") not in calls


def test_promote_fn_called_for_non_double_routes(tmp_path):
    report = write_report(tmp_path, {"B_dd": 1e-2, "B_dbl": 1e-2})
    table = {
        ("B_dd", "baseline"): _base(5.0),
        ("B_dbl", "baseline"): _base(5.0),
        ("B_dd", "dd"): _cand(5.0, 15.9),     # -> dd
        # dd with NO lift (candidate <= baseline) and below bar -> reject, stays double.
        ("B_dbl", "dd"): _cand(5.0, 5.0),
        ("B_dd", "ff"): _cand(5.0, 4.0),
        ("B_dbl", "ff"): _cand(5.0, 4.0),     # below bar -> stays double
    }
    promoted = []
    run_tu(tmp_path, report, make_measure_fn(table),
           promote_fn=lambda i, p: promoted.append((i, p)))
    assert promoted == [("B_dd", "dd")]       # only the non-double route promoted


def test_missing_measure_fn_fails_loud(tmp_path):
    report = write_report(tmp_path, {"B_x": 1e-2})
    cfg = StrategyConfig(tolerance=7.0, runs_root=tmp_path / "runs",
                         strategy_mode="tu_only")
    state = {"characterization_report_path": report,
             "tu_measure_fn": None, "strategy_config": cfg,
             "patcher_fn": None, "validator_fn": None}
    try:
        strategy_agent.run(state)
    except ValueError as exc:
        assert "tu_measure_fn" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing tu_measure_fn")


def test_float_is_candidate_gate():
    # pred below threshold -> candidate; above -> pruned; None/kill-switch -> open.
    assert float_is_candidate(1e-9, 7.0)
    assert not float_is_candidate(1e-2, 7.0)
    assert float_is_candidate(None, 7.0)
    assert float_is_candidate(1e-2, 7.0, report_prunes=False)
