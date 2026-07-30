"""Two-phase walk — correctness phase then speedup phase, split iteration budget.

Mirrors the mocked-loop pattern of ``test_loop`` (no git; scripted Patcher /
Validator) but drives the phase machinery: the per-phase iteration caps, the
phase-1→phase-2 spillover, the phase-2 skip of regions promoted to dd in phase 1,
and the empty-speedup-queue no-op.  The Patcher encodes the intent kind + target
into the candidate SHA so the Validator can return a kind-dependent verdict
without a real repo.
"""

import json
from pathlib import Path

from agents.config import StrategyBudget, StrategyConfig
from agents.strategy import agent as strategy_agent


# --------------------------------------------------------------------------
# report builder
# --------------------------------------------------------------------------

def _stable(pf, pff, rel, ops):
    return {"signal_class": "stable", "max_cond": 10.0, "max_rel_err": rel,
            "predicted_rel_err_if_float": pf, "predicted_rel_err_if_ff": pff,
            "prov_vars": ["v"], "ops": ops, "n": 100, "non_localizable": False}


def _correctness_region(pf, pff, rel, ops):
    # Phase 2c: correctness regions must carry a NON-stable signal class (stable
    # regions no longer enter the correctness queue).  ``log_near_root`` is gated on
    # ``max_rel_err > thr`` and (like the former stable-tier-4) has no rewrite step —
    # it walks straight to the dd ceiling, so the scripted-walk expectations hold.
    r = _stable(pf, pff, rel, ops)
    r["signal_class"] = "log_near_root"
    return r


def write_report(tmp_path, *, correctness=(), speedup=(), chains=()):
    """Build a stability report.

    * ``correctness`` — ``(integral, file, line)`` ``log_near_root`` regions with
      high rel-err (correctness tier 3) that are ff-*unsafe* (never in the speedup q).
    * ``speedup``     — ``(integral, file, line, op_count)`` stable, ff-safe AND
      float-safe (``pred_float`` under the tol=10 bar so the Wave-3 WI2 float gate
      does not pre-empt the ``ff->float`` step), low-rel-err regions (speedup queue
      only).  Their float rung is exercised via the Validator, not the walk gate.
    * ``chains``      — ``(integral, chain_id, [(file, line), ...])`` cascade
      chains (correctness tier 2).
    """
    integrals: dict = {}

    def _slot(integ):
        return integrals.setdefault(
            integ, {"class_counts": {}, "regions": {}, "cascade_chains": []})

    for integ, file, line in correctness:
        _slot(integ)["regions"][f"{file}:{line}"] = _correctness_region(1e-2, 1e-2, 1e-3, {"sub": 2})
    for integ, file, line, opc in speedup:
        _slot(integ)["regions"][f"{file}:{line}"] = _stable(1e-12, 1e-12, 1e-16, {"mul": opc})
    for integ, cid, spans in chains:
        _slot(integ)["cascade_chains"].append({
            "kind": "cascade_chain", "chain_id": cid,
            "chain": [{"file": f, "line_start": l, "line_end": l} for f, l in spans],
            "signal_class": "cancellation_cascade", "non_localizable": False,
            "max_cond": 1e6, "max_rel_err": 1e-3,
            "predicted_rel_err_if_float": 1e-2, "predicted_rel_err_if_ff": 1e-2,
            "ops": {"sub": 2}, "n": 2, "region_local_vars": ["v"]})

    report = {"kind": "stability_report", "schema_version": 1, "no_id_records": 0,
              "samples_seen": {}, "integrals": integrals}
    p = tmp_path / "report.json"
    p.write_text(json.dumps(report))
    return str(p)


# --------------------------------------------------------------------------
# scripted Patcher / Validator (kind encoded in the SHA)
# --------------------------------------------------------------------------

def make_patcher():
    calls = {"n": 0}

    def patcher(intent, ctx):
        calls["n"] += 1
        t = intent["target"]
        sha = f"{intent['kind']}|{t['file']}:{t['line_start']}|{calls['n']}"
        return {"status": "ok", "candidate_sha": sha, "parent_sha": "p",
                "artifacts": {}, "error": None, "llm_tokens": 10}

    patcher.calls = calls
    return patcher


def make_validator():
    """double->dd rejects (dd ceiling); double->ff accepts, ff->float rejects
    (speedup settles at ff)."""
    def validator(sha, ctx):
        kind = sha.split("|")[0]
        if kind == "double-to-dd":
            verdict, digits = "reject", 7.2
        elif kind == "double-to-ff":
            verdict, digits = "accept", 12.0
        elif kind == "ff-to-float":
            verdict, digits = "reject", 6.0
        else:
            verdict, digits = "reject", 5.0
        return {"verdict": verdict, "candidate": {"min_precise_digits": digits},
                "current": {"min_precise_digits": 5.0}}
    return validator


def run_agent(tmp_path, report, *, tolerance=10.0, **budget_kw):
    cfg = StrategyConfig(
        tolerance=tolerance, runs_root=tmp_path / "runs", strategy_mode="region",
        budget=StrategyBudget(
            max_iters=budget_kw.pop("max_iters", 10**7),
            max_iters_correctness=budget_kw.pop("max_iters_correctness", None),
            max_iters_speedup=budget_kw.pop("max_iters_speedup", None),
            max_wall_clock_sec=600, max_llm_tokens=10**12),
        diminishing_returns_k=budget_kw.pop("k", 10**7))
    state = {
        "characterization_report_path": report,
        "strategy_repo_path": None, "strategy_starting_sha": None,
        "strategy_config": cfg,
        "patcher_fn": make_patcher(), "validator_fn": make_validator(),
    }
    delta = strategy_agent.run(state)
    rep = json.loads(Path(delta["strategy_result"]["report_json_path"]).read_text())
    itlog = [json.loads(l)
             for l in Path(rep["iteration_log_path"]).read_text().splitlines()]
    return delta["strategy_result"], rep, itlog


def _phases(itlog):
    return [e["phase"] for e in itlog]


# --------------------------------------------------------------------------
# (a) phase 1 consumes its full budget; phase 2 still runs the reserved chunk
# --------------------------------------------------------------------------

def test_phase1_full_budget_phase2_still_runs(tmp_path):
    report = write_report(
        tmp_path,
        correctness=[("A", "a.h", 1), ("B", "b.h", 1), ("C", "c.h", 1)],
        speedup=[("D", "d.h", 1, 50), ("E", "e.h", 1, 40)])
    res, rep, itlog = run_agent(tmp_path, report,
                                max_iters_correctness=1, max_iters_speedup=10)

    # phase 1 stopped at its cap (1 correctness iteration); phase 2 still ran.
    assert _phases(itlog).count("correctness") == 1
    assert _phases(itlog).count("speedup") == 4          # 2 regions x (ff accept + float reject)
    assert rep["phase_summary"]["correctness"]["iterations"] == 1
    assert rep["phase_summary"]["speedup"]["iterations"] == 4
    # both speedup regions settled at ff; overall status = phase-2's (success)
    assert res["status"] == "success"
    assert rep["precision_distribution"]["ff"] == 2
    # every iteration carries a phase tag
    assert all(e["phase"] in ("correctness", "speedup") for e in itlog)


# --------------------------------------------------------------------------
# (b) phase 1 finishes early; unused budget spills into phase 2
# --------------------------------------------------------------------------

def test_phase1_early_finish_spills_into_phase2(tmp_path):
    report = write_report(
        tmp_path,
        correctness=[("A", "a.h", 1)],
        speedup=[("D", "d.h", 1, 50), ("E", "e.h", 1, 40)])
    # speedup cap is only 1, but phase 1 uses 1 of its 10 → 9 spill → effective 10.
    res, rep, itlog = run_agent(tmp_path, report,
                                max_iters_correctness=10, max_iters_speedup=1)

    assert rep["phase_summary"]["correctness"]["iterations"] == 1
    # without spillover the speedup phase would have stopped after 1 iteration;
    # the 9 spilled iterations let both regions run to completion (4 iterations).
    assert rep["phase_summary"]["speedup"]["iter_cap"] == 10     # 1 + 9 spill
    assert rep["phase_summary"]["speedup"]["iterations"] == 4
    assert res["status"] == "success"
    assert rep["precision_distribution"]["ff"] == 2


def test_speedup_cap_binds_without_spillover(tmp_path):
    # Control for (b): phase 1 eats its whole cap → no spill → speedup cap of 1
    # bites and the run ends budget_exhausted after a single speedup iteration.
    report = write_report(
        tmp_path,
        correctness=[("A", "a.h", 1)],
        speedup=[("D", "d.h", 1, 50), ("E", "e.h", 1, 40)])
    res, rep, itlog = run_agent(tmp_path, report,
                                max_iters_correctness=1, max_iters_speedup=1)
    assert rep["phase_summary"]["speedup"]["iter_cap"] == 1      # no spill
    assert rep["phase_summary"]["speedup"]["iterations"] == 1
    assert res["status"] == "budget_exhausted"                  # phase-2 cap = hard stop


# --------------------------------------------------------------------------
# (c) phase 2 skips a region promoted to dd in phase 1 (via a cascade chain)
# --------------------------------------------------------------------------

def test_phase2_skips_dd_promoted_region(tmp_path):
    # Ov.h:5 is BOTH a cascade-chain line (promoted to dd in phase 1) AND a stable
    # speedup candidate; the speedup phase must skip it.  Free.h:1 demotes normally.
    report = write_report(
        tmp_path,
        chains=[("X", "cascade_X_1", [("Ov.h", 5)])],
        speedup=[("X", "Ov.h", 5, 30), ("Y", "Free.h", 1, 20)])
    # this validator must accept the chain's double-to-dd promotion
    res, rep, itlog = run_agent(tmp_path, report)

    speedup_targets = {(e["target"]["file"], e["target"]["line_start"])
                       for e in itlog if e["phase"] == "speedup"}
    assert ("Ov.h", 5) not in speedup_targets          # skipped (dd-promoted)
    assert ("Free.h", 1) in speedup_targets            # free region worked
    assert rep["phase_summary"]["speedup"]["skipped_dd_promoted"] >= 1

    prec = {(a["file"], a["line_start"]): a["precision"]
            for a in rep["precision_assignment"]}
    assert prec[("Ov.h", 5)] == "dd"                   # chain floor, unchanged
    assert prec[("Free.h", 1)] == "ff"                 # demoted in phase 2


# --------------------------------------------------------------------------
# (d) tolerance high enough that the speedup queue is empty → phase 2 no-ops
# --------------------------------------------------------------------------

def test_empty_speedup_queue_phase2_noops(tmp_path):
    # At tolerance 15 (thr 1e-15) the ff-safe-at-1e-12 region no longer clears, so
    # the speedup queue is empty; phase 2 runs zero iterations and the run still
    # ends success (phase-2's status), with the correctness region worked.
    report = write_report(
        tmp_path,
        correctness=[("A", "a.h", 1)],
        speedup=[("D", "d.h", 1, 50)])
    res, rep, itlog = run_agent(tmp_path, report, tolerance=15.0)

    assert _phases(itlog).count("speedup") == 0
    assert rep["phase_summary"]["speedup"]["iterations"] == 0
    assert res["status"] == "success"
    # correctness region was worked (promoted to dd ceiling)
    assert _phases(itlog).count("correctness") == 1
    assert rep["precision_distribution"]["dd"] == 1
