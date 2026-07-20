"""Wave-3 report-field prunes end-to-end (WI1 range guard, WI2 pred-float gate,
WI3 flop-weighted ordering) driven through the real Strategy agent with a scripted
Patcher/Validator (no git), asserting both the walk behavior AND the never-silent
``speedup_summary`` telemetry.

The Validator accepts every demotion (double->ff, ff->float, double->float) so the
ONLY thing that stops the walk short of float is a Wave-3 gate — making the gate's
effect observable purely from which kinds the Patcher was asked to build.
"""

import json
from pathlib import Path

from agents.config import StrategyBudget, StrategyConfig
from agents.strategy import agent as strategy_agent


def _stable(*, pf, pff, ops, range_ok=None, rel=1e-16):
    r = {"signal_class": "stable", "max_cond": 10.0, "max_rel_err": rel,
         "predicted_rel_err_if_float": pf, "predicted_rel_err_if_ff": pff,
         "prov_vars": ["v"], "ops": ops, "n": 100, "non_localizable": False}
    if range_ok is not None:
        r["value_range_ok_for_float"] = range_ok
    return r


def _write(tmp_path, regions):
    """regions: {"file:line": region_dict} under a single integral."""
    report = {"kind": "stability_report", "schema_version": 1, "no_id_records": 0,
              "samples_seen": {}, "integrals": {"A": {"regions": regions,
                                                      "cascade_chains": []}}}
    p = tmp_path / "report.json"
    p.write_text(json.dumps(report))
    return str(p)


def _make_patcher():
    calls = {"n": 0, "kinds": []}

    def patcher(intent, ctx):
        calls["n"] += 1
        calls["kinds"].append(intent["kind"])
        t = intent["target"]
        sha = f"{intent['kind']}|{t['file']}:{t['line_start']}|{calls['n']}"
        return {"status": "ok", "candidate_sha": sha, "parent_sha": "p",
                "artifacts": {}, "error": None, "llm_tokens": 1}

    patcher.calls = calls
    return patcher


def _accepting_validator():
    """Accept every demotion so only a Wave-3 gate can halt the walk before float."""
    def validator(sha, ctx):
        return {"verdict": "accept", "candidate": {"min_precise_digits": 12.0},
                "current": {"min_precise_digits": 5.0}}
    return validator


def _run(tmp_path, report, *, tolerance=10.0, report_prunes=True,
         ratio_path="__default__"):
    kw = {}
    if ratio_path != "__default__":
        kw["ratio_multipliers_path"] = ratio_path
    cfg = StrategyConfig(
        tolerance=tolerance, runs_root=tmp_path / "runs", report_prunes=report_prunes,
        budget=StrategyBudget(max_iters=10**6, max_iters_correctness=10**5,
                              max_iters_speedup=10**5, max_wall_clock_sec=600,
                              max_llm_tokens=10**12),
        diminishing_returns_k=10**6, **kw)
    patcher = _make_patcher()
    state = {"characterization_report_path": report,
             "strategy_repo_path": None, "strategy_starting_sha": None,
             "strategy_config": cfg, "patcher_fn": patcher,
             "validator_fn": _accepting_validator()}
    delta = strategy_agent.run(state)
    rep = json.loads(Path(delta["strategy_result"]["report_json_path"]).read_text())
    return rep, patcher.calls["kinds"]


# ---------------------------------------------------------------------------
# WI1 — value_range_ok_for_float float-rung guard
# ---------------------------------------------------------------------------

def test_wi1_range_unsafe_stops_at_ff(tmp_path):
    # float-safe by the error model, but range-unsafe → float NOT attempted; ff is.
    report = _write(tmp_path, {"f.h:1": _stable(pf=1e-30, pff=1e-30, ops={"mul": 5},
                                                range_ok=False)})
    rep, kinds = _run(tmp_path, report)
    assert "double-to-ff" in kinds
    assert not any(k.endswith("-to-float") for k in kinds)
    ss = rep["speedup_summary"]
    assert ss["regions_skipped_range_unsafe"] == 1
    assert ss["regions_skipped_pred_float"] == 0
    assert rep["precision_distribution"]["ff"] == 1


def test_wi1_range_ok_true_attempts_float(tmp_path):
    report = _write(tmp_path, {"f.h:1": _stable(pf=1e-30, pff=1e-30, ops={"mul": 5},
                                                range_ok=True)})
    rep, kinds = _run(tmp_path, report)
    assert any(k.endswith("-to-float") for k in kinds)
    assert rep["speedup_summary"]["regions_skipped_range_unsafe"] == 0
    assert rep["precision_distribution"]["float"] == 1


def test_wi1_fail_open_when_field_missing(tmp_path):
    # no value_range_ok_for_float → default True → float attempted, no skip counted.
    report = _write(tmp_path, {"f.h:1": _stable(pf=1e-30, pff=1e-30, ops={"mul": 5})})
    rep, kinds = _run(tmp_path, report)
    assert any(k.endswith("-to-float") for k in kinds)
    assert rep["speedup_summary"]["regions_skipped_range_unsafe"] == 0


# ---------------------------------------------------------------------------
# WI2 — predicted_rel_err_if_float float-step gate
# ---------------------------------------------------------------------------

def test_wi2_pred_float_above_thr_skips_float(tmp_path):
    # pred_ff safe (admitted to speedup q) but pred_float > 1e-10 → float skipped.
    report = _write(tmp_path, {"f.h:1": _stable(pf=1e-7, pff=1e-30, ops={"mul": 5},
                                                range_ok=True)})
    rep, kinds = _run(tmp_path, report)
    assert "double-to-ff" in kinds
    assert not any(k.endswith("-to-float") for k in kinds)
    ss = rep["speedup_summary"]
    assert ss["regions_skipped_pred_float"] == 1
    assert ss["regions_skipped_range_unsafe"] == 0


def test_wi2_pred_float_below_thr_attempts_float(tmp_path):
    report = _write(tmp_path, {"f.h:1": _stable(pf=1e-12, pff=1e-30, ops={"mul": 5},
                                                range_ok=True)})
    rep, kinds = _run(tmp_path, report)
    assert any(k.endswith("-to-float") for k in kinds)
    assert rep["speedup_summary"]["regions_skipped_pred_float"] == 0


def test_wi1_precedence_over_wi2(tmp_path):
    # range-unsafe AND pred_float-unsafe → counted as range (WI1 checked first).
    report = _write(tmp_path, {"f.h:1": _stable(pf=1e-7, pff=1e-30, ops={"mul": 5},
                                                range_ok=False)})
    rep, _ = _run(tmp_path, report)
    ss = rep["speedup_summary"]
    assert ss["regions_skipped_range_unsafe"] == 1
    assert ss["regions_skipped_pred_float"] == 0


# ---------------------------------------------------------------------------
# WI3 — flop-weight availability telemetry
# ---------------------------------------------------------------------------

def test_wi3_flop_weighted_true_with_default_table(tmp_path):
    # the committed runs/qcdloop/ratio_multipliers.json is the default source.
    report = _write(tmp_path, {"f.h:1": _stable(pf=1e-30, pff=1e-30, ops={"mul": 5})})
    rep, _ = _run(tmp_path, report)
    assert rep["speedup_summary"]["speedup_queue_flop_weighted"] is True


def test_wi3_flop_weighted_false_when_table_missing(tmp_path):
    report = _write(tmp_path, {"f.h:1": _stable(pf=1e-30, pff=1e-30, ops={"mul": 5})})
    rep, _ = _run(tmp_path, report, ratio_path=str(tmp_path / "absent.json"))
    assert rep["speedup_summary"]["speedup_queue_flop_weighted"] is False


# ---------------------------------------------------------------------------
# kill-switch — report_prunes=False disables all three
# ---------------------------------------------------------------------------

def test_kill_switch_disables_all_prunes(tmp_path):
    # range-unsafe region: with prunes OFF, float is attempted anyway and no skip
    # is counted; flop-weighting is off too.
    report = _write(tmp_path, {"f.h:1": _stable(pf=1e-30, pff=1e-30, ops={"mul": 5},
                                                range_ok=False)})
    rep, kinds = _run(tmp_path, report, report_prunes=False)
    assert any(k.endswith("-to-float") for k in kinds)
    ss = rep["speedup_summary"]
    assert ss["report_prunes_enabled"] is False
    assert ss["regions_skipped_range_unsafe"] == 0
    assert ss["speedup_queue_flop_weighted"] is False
