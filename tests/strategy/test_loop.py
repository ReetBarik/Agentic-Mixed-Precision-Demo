"""Main-loop integration — stopping conditions and P6 statuses end-to-end.

These drive ``strategy_agent.run`` with programmable mocked Patcher/Validator and
no git repo (git side effects are covered by test_e2e), focusing on the loop's
control flow: budget/diminishing-returns/commit-failed stops, the timeout
retry-once, dd_untested via a Patcher failure at the DD rung, and the
strategy-bug accounting for patch_apply_failed.
"""

import json
from pathlib import Path

from agents.config import StrategyBudget, StrategyConfig
from agents.strategy import agent as strategy_agent


def write_report(tmp_path, regions):
    """regions: list of (integral, file, line, signal_class, max_rel_err, pred_float)."""
    integrals = {}
    for i, (integ, file, line, sig, rel, pf) in enumerate(regions):
        integrals.setdefault(integ, {"class_counts": {}, "regions": {}})
        integrals[integ]["regions"][f"{file}:{line}"] = {
            "signal_class": sig, "max_cond": 1e16, "max_rel_err": rel,
            "predicted_rel_err_if_float": pf, "prov_vars": ["v"],
            "ops": {"sub": 2}, "n": 100, "non_localizable": False,
        }
    report = {"kind": "stability_report", "schema_version": 1, "no_id_records": 0,
              "samples_seen": {}, "integrals": integrals}
    p = tmp_path / "report.json"
    p.write_text(json.dumps(report))
    return str(p)


def run_agent(tmp_path, report, patcher, validator, **cfg_kw):
    cfg = StrategyConfig(
        tolerance=10.0, runs_root=tmp_path / "runs", strategy_mode="region",
        budget=StrategyBudget(
            max_iters=cfg_kw.pop("max_iters", 10**7),
            max_wall_clock_sec=cfg_kw.pop("max_wall", 600),
            max_llm_tokens=cfg_kw.pop("max_tokens", 10**12)),
        diminishing_returns_k=cfg_kw.pop("k", 10**7))
    state = {
        "characterization_report_path": report,
        "strategy_repo_path": None, "strategy_starting_sha": None,
        "strategy_config": cfg, "patcher_fn": patcher, "validator_fn": validator,
    }
    delta = strategy_agent.run(state)
    rep = json.loads(Path(delta["strategy_result"]["report_json_path"]).read_text())
    return delta["strategy_result"], rep


def ok_patcher(status_seq=None):
    """Patcher returning 'ok' (or a scripted status sequence)."""
    calls = {"n": 0}
    def patcher(intent, ctx):
        calls["n"] += 1
        status = "ok"
        if status_seq is not None:
            status = status_seq[min(calls["n"] - 1, len(status_seq) - 1)]
        return {"status": status, "candidate_sha": f"sha{calls['n']}",
                "parent_sha": "p", "artifacts": {}, "error": None}
    patcher.calls = calls
    return patcher


def const_validator(verdict):
    def validator(sha, ctx):
        return {"verdict": verdict, "candidate": {"min_precise_digits": 7.2},
                "current": {"min_precise_digits": 5.0}}
    return validator


# --------------------------------------------------------------------------

def test_commit_failed_aborts_internal_error(tmp_path):
    report = write_report(tmp_path, [("A", "a.h", 5, "log_near_root", 1e-3, 1e-2)])
    res, rep = run_agent(tmp_path, report, ok_patcher(["commit_failed"]),
                         const_validator("accept"))
    assert res["status"] == "internal_error"
    assert rep["status"] == "internal_error"
    tags = [json.loads(l)["log_tag"]
            for l in Path(rep["iteration_log_path"]).read_text().splitlines()]
    assert tags == ["fatal"]


def test_budget_max_iters_correctness_only(tmp_path):
    # Two-phase walk: max_iters=2 splits 70/30 → correctness cap 1, speedup cap 1.
    # A correctness-only workload soft-exhausts phase 1 at its cap and (phase 2
    # empty) ends "success" (overall status = phase-2's status), NOT a run-level
    # budget_exhausted — phase-1 hitting its cap is a soft hand-off.
    regions = [(f"I{i}", "a.h", 10 + i, "log_near_root", 1e-3, 1e-2) for i in range(5)]
    report = write_report(tmp_path, regions)
    res, rep = run_agent(tmp_path, report, ok_patcher(), const_validator("reject"),
                         max_iters=2)
    assert res["status"] == "success"
    assert rep["budget_iters_used"] == 1                    # correctness cap = 1
    assert rep["phase_summary"]["correctness"]["iterations"] == 1
    assert rep["phase_summary"]["correctness"]["iter_cap"] == 1
    assert rep["phase_summary"]["speedup"]["iterations"] == 0


def test_diminishing_returns_partial(tmp_path):
    regions = [(f"I{i}", "a.h", 10 + i, "log_near_root", 1e-3, 1e-2) for i in range(5)]
    report = write_report(tmp_path, regions)
    res, rep = run_agent(tmp_path, report, ok_patcher(), const_validator("reject"), k=2)
    assert res["status"] == "partial"


def test_dd_untested_via_build_failed(tmp_path):
    report = write_report(tmp_path, [("A", "a.h", 5, "log_near_root", 1e-3, 1e-2)])
    res, rep = run_agent(tmp_path, report, ok_patcher(["build_failed"]),
                         const_validator("accept"))
    cs = rep["correctness_summary"]
    assert cs["regions_dd_untested"] == 1 and cs["regions_at_dd_ceiling"] == 0
    ceil = cs["ceiling_regions"][0]
    assert ceil["ceiling_kind"] == "dd_untested" and ceil["final_min_digits"] is None
    # build_failed counts against the budget (Bucket A)
    assert rep["budget_iters_used"] == 1


def test_timeout_retries_once_then_ok(tmp_path):
    report = write_report(tmp_path, [("A", "a.h", 5, "log_near_root", 1e-3, 1e-2)])
    patcher = ok_patcher(["timeout", "ok"])
    res, rep = run_agent(tmp_path, report, patcher, const_validator("accept"))
    # same intent tried twice (timeout → retry → ok), then validated + cleared
    assert patcher.calls["n"] == 2
    assert rep["precision_distribution"]["dd"] == 1
    assert rep["status"] == "success"


def test_timeout_twice_folds_to_reject(tmp_path):
    report = write_report(tmp_path, [("A", "a.h", 5, "log_near_root", 1e-3, 1e-2)])
    res, rep = run_agent(tmp_path, report, ok_patcher(["timeout", "timeout"]),
                         const_validator("accept"))
    # second timeout folds to build_failed-equivalent → DD never tested → untested
    assert rep["correctness_summary"]["regions_dd_untested"] == 1


def test_patch_apply_failed_is_strategy_bug_and_free(tmp_path):
    report = write_report(tmp_path, [("A", "a.h", 5, "log_near_root", 1e-3, 1e-2)])
    res, rep = run_agent(tmp_path, report, ok_patcher(["patch_apply_failed"]),
                         const_validator("accept"))
    rec = json.loads(Path(rep["iteration_log_path"]).read_text().splitlines()[0])
    assert rec["strategy_bug"] is True and rec["log_tag"] == "strategy_bug"
    assert rep["budget_iters_used"] == 0        # strategy bug doesn't count vs budget


def test_clean_success_all_accept(tmp_path):
    report = write_report(tmp_path, [
        ("A", "a.h", 5, "local_cancellation", 1e-3, 1e-2),
        ("B", "b.h", 5, "stable", 1e-16, 1e-2),   # correctness-clean, float-unsafe → no queue
    ])
    res, rep = run_agent(tmp_path, report, ok_patcher(), const_validator("accept"))
    assert res["status"] == "success"
    # A cleared at dd; B untouched at double
    assert rep["precision_distribution"] == {"float": 0, "ff": 0, "double": 1, "dd": 1}
