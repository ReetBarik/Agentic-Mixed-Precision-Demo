"""Cascade-chain representative dedup (WI1 / CALIBRATION.md §Bug 2).

Chains are walked on their representative line (``lines[0]``); many chains sharing
a representative must NOT each re-drive that line.  These tests use the mocked
loop (no git; scripted Patcher/Validator) and assert the driver walks once while
the group's other members are recorded ``chain_dedup_skipped`` yet still get the
promoted precision distributed across their own lines.
"""

import json
from pathlib import Path

from agents.config import StrategyBudget, StrategyConfig
from agents.strategy import agent as strategy_agent


# --------------------------------------------------------------------------
# report builder — cascade chains only
# --------------------------------------------------------------------------

def write_report(tmp_path, chains):
    """chains: list of (integral, chain_id, [(file, line), ...])."""
    integrals: dict = {}
    for integ, cid, spans in chains:
        slot = integrals.setdefault(
            integ, {"class_counts": {}, "regions": {}, "cascade_chains": []})
        slot["cascade_chains"].append({
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


def make_patcher():
    calls = {"n": 0, "targets": []}

    def patcher(intent, ctx):
        calls["n"] += 1
        t = intent["target"]
        calls["targets"].append((t["file"], t["line_start"]))
        sha = f"{intent['kind']}|{t['file']}:{t['line_start']}|{calls['n']}"
        return {"status": "ok", "candidate_sha": sha, "parent_sha": "p",
                "artifacts": {}, "error": None, "llm_tokens": 10}

    patcher.calls = calls
    return patcher


def accept_dd_validator():
    """double-to-dd accepts (chain clears at dd in a single walk iteration)."""
    def validator(sha, ctx):
        kind = sha.split("|")[0]
        verdict = "accept" if kind == "double-to-dd" else "reject"
        return {"verdict": verdict, "candidate": {"min_precise_digits": 12.0},
                "current": {"min_precise_digits": 5.0}}
    return validator


def run_agent(tmp_path, report):
    cfg = StrategyConfig(
        tolerance=7.0, runs_root=tmp_path / "runs",
        budget=StrategyBudget(max_iters=10**7, max_wall_clock_sec=600,
                              max_llm_tokens=10**12),
        diminishing_returns_k=10**7)
    patcher = make_patcher()
    state = {
        "characterization_report_path": report,
        "strategy_repo_path": None, "strategy_starting_sha": None,
        "strategy_config": cfg,
        "patcher_fn": patcher, "validator_fn": accept_dd_validator(),
    }
    delta = strategy_agent.run(state)
    rep = json.loads(Path(delta["strategy_result"]["report_json_path"]).read_text())
    itlog = [json.loads(l)
             for l in Path(rep["iteration_log_path"]).read_text().splitlines()]
    return patcher, rep, itlog


# --------------------------------------------------------------------------

def test_three_chains_share_rep_one_walk_two_skipped(tmp_path):
    # 5 chains: c1/c2/c3 share representative s.h:5; c4, c5 have distinct reps.
    report = write_report(tmp_path, [
        ("A", "c1", [("s.h", 5), ("s.h", 6)]),
        ("A", "c2", [("s.h", 5), ("s.h", 7)]),
        ("A", "c3", [("s.h", 5), ("s.h", 8)]),
        ("B", "c4", [("t.h", 5)]),
        ("C", "c5", [("u.h", 5)]),
    ])
    patcher, rep, itlog = run_agent(tmp_path, report)

    # exactly 3 walks fired (one per representative group), each cleared in a
    # single double-to-dd accept — NOT 5.
    assert len(itlog) == 3
    walked_targets = [(e["target"]["file"], e["target"]["line_start"]) for e in itlog]
    assert walked_targets.count(("s.h", 5)) == 1        # shared rep walked once
    assert ("t.h", 5) in walked_targets and ("u.h", 5) in walked_targets

    # the two group siblings are recorded chain_dedup_skipped
    assert rep["correctness_summary"]["regions_chain_dedup_skipped"] == 2

    # every chain's precision floor is still distributed: the shared rep line lists
    # all three chain_ids in required_by; each unique tail line is floored at dd.
    prec = {(a["file"], a["line_start"]): a for a in rep["precision_assignment"]}
    assert prec[("s.h", 5)]["precision"] == "dd"
    assert prec[("s.h", 5)]["required_by"] == ["c1", "c2", "c3"]
    for tail, cid in [(("s.h", 6), "c1"), (("s.h", 7), "c2"), (("s.h", 8), "c3")]:
        assert prec[tail]["precision"] == "dd"
        assert prec[tail]["required_by"] == [cid]


def test_rep_already_at_target_skips_walk(tmp_path):
    # c1 floors s.h:6 (its tail) at dd; c2's representative IS s.h:6, so the
    # already-at-target guard skips c2's walk entirely.
    report = write_report(tmp_path, [
        ("A", "c1", [("s.h", 5), ("s.h", 6)]),
        ("A", "c2", [("s.h", 6), ("s.h", 9)]),
    ])
    patcher, rep, itlog = run_agent(tmp_path, report)

    walked = [(e["target"]["file"], e["target"]["line_start"]) for e in itlog]
    assert walked == [("s.h", 5)]                       # only c1 drove a walk
    assert ("s.h", 6) not in walked                     # c2 skipped (already dd)
    assert rep["correctness_summary"]["regions_chain_dedup_skipped"] == 1
    # c2's own tail line still floored at dd via the ledger
    prec = {(a["file"], a["line_start"]): a for a in rep["precision_assignment"]}
    assert prec[("s.h", 9)]["precision"] == "dd"
    assert prec[("s.h", 9)]["required_by"] == ["c2"]
