"""End-to-end: a 3-region report driven through a real branch with mocked
Patcher/Validator.  Exercises the correctness-then-speedup loop, DD-ceiling
retention, per-patch commits, reject-reset, and the output artifacts.

Region script (tolerance = 6):
  * A.h:5  local_cancellation  → tier 1 → DD reject + all identities reject → dd_ceiling
  * B.h:5  cancellation_cascade → tier 2 → clears at DD
  * C.h:5  stable, float-safe   → speedup → double→ff accept, ff→float reject → settle ff
"""

import json
import subprocess
from pathlib import Path

import pytest

from agents.config import StrategyConfig, StrategyBudget
from agents.strategy import agent as strategy_agent


# --------------------------------------------------------------------------
# fixtures: a real git repo + a tiny characterization report
# --------------------------------------------------------------------------

def _git(repo, *args, **kw):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True, **kw)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "tree"
    (root / "headers").mkdir(parents=True)
    for name in ("A.h", "B.h", "C.h"):
        (root / "headers" / name).write_text("\n".join(f"line {i}" for i in range(1, 11)) + "\n")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t.t")
    _git(root, "config", "user.name", "t")
    _git(root, "config", "commit.gpgsign", "false")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    start = _git(root, "rev-parse", "HEAD").stdout.strip()
    return root, start


@pytest.fixture
def report_path(tmp_path):
    def region(sig, cond, rel_err, pred_float, ops):
        return {"signal_class": sig, "max_cond": cond, "max_rel_err": rel_err,
                "predicted_rel_err_if_float": pred_float, "prov_vars": ["v"],
                "ops": ops, "n": 100, "non_localizable": False}
    report = {
        "kind": "stability_report", "schema_version": 1, "no_id_records": 0,
        "samples_seen": {}, "integrals": {
            "IA": {"class_counts": {}, "regions": {
                "headers/A.h:5": region("local_cancellation", 1e16, 1e-3, 1e-2, {"sub": 3})}},
            "IB": {"class_counts": {}, "regions": {
                "headers/B.h:5": region("cancellation_cascade", 1e6, 1e-3, 1e-2, {"sub": 5})}},
            "IC": {"class_counts": {}, "regions": {
                "headers/C.h:5": region("stable", 10.0, 1e-16, 1e-8, {"mul": 7})}},
        }}
    p = tmp_path / "report.json"
    p.write_text(json.dumps(report))
    return p


# --------------------------------------------------------------------------
# mocked Patcher / Validator
# --------------------------------------------------------------------------

def make_patcher(repo_root):
    """Always ok; makes a real commit that appends a marker to the region file."""
    def patcher(intent, ctx):
        f = repo_root / intent["target"]["file"]
        with open(f, "a") as fh:
            fh.write(f"// {intent['kind']} {ctx['iter_id']} {intent['rationale_id']}\n")
        _git(repo_root, "add", "-A")
        msg = (f"[{intent['rationale_id']}] {intent['kind']} "
               f"{intent['target']['file']}:{intent['target']['line_start']}")
        _git(repo_root, "commit", "-q", "-m", msg)
        sha = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
        return {"status": "ok", "candidate_sha": sha, "parent_sha": ctx["parent_sha"],
                "artifacts": {}, "error": None, "llm_tokens": 100}
    return patcher


def make_validator(repo_root):
    """Reads the candidate commit message to decide — exercises the SHA contract."""
    def validator(candidate_sha, ctx):
        msg = _git(repo_root, "log", "-1", "--format=%s", candidate_sha).stdout.strip()
        # "[iter_N] <kind> <file>:<line>"
        parts = msg.split()
        kind, loc = parts[1], parts[2]
        if "A.h" in loc:
            verdict, digits = "reject", 7.2         # never clears → dd_ceiling
        elif "B.h" in loc:
            verdict = "accept" if kind == "double-to-dd" else "reject"
            digits = 15.0
        elif "C.h" in loc:
            verdict = "accept" if kind == "double-to-ff" else "reject"
            digits = 9.0
        else:
            verdict, digits = "reject", 0.0
        return {"verdict": verdict, "candidate": {"min_precise_digits": digits},
                "current": {"min_precise_digits": 5.0}}
    return validator


# --------------------------------------------------------------------------
# the test
# --------------------------------------------------------------------------

def test_e2e_three_region_run(repo, report_path, tmp_path):
    root, start = repo
    state = {
        "characterization_report_path": str(report_path),
        "strategy_repo_path": str(root),
        "strategy_starting_sha": start,
        "strategy_config": StrategyConfig(
            tolerance=6.0, runs_root=tmp_path / "runs",
            budget=StrategyBudget(max_iters=500, max_wall_clock_sec=600, max_llm_tokens=10**9),
            diminishing_returns_k=20),
        "patcher_fn": make_patcher(root),
        "validator_fn": make_validator(root),
    }

    delta = strategy_agent.run(state)
    res = delta["strategy_result"]

    # -- state delta shape (Q5) --
    assert set(res) == {"status", "run_id", "final_branch", "report_json_path",
                        "report_md_path", "cumulative_diff_path"}
    assert res["status"] == "success"
    assert res["final_branch"] == f"strategy/{res['run_id']}"

    # -- report.json shape --
    report = json.loads(Path(res["report_json_path"]).read_text())
    assert report["status"] == "success"
    assert report["tolerance"] == 6.0
    assert report["precision_distribution"] == {"float": 0, "ff": 1, "double": 0, "dd": 2}

    # precision assignments: A=dd (ceiling-retained), B=dd (accept), C=ff (speedup)
    prec = {a["file"]: a["precision"] for a in report["precision_assignment"]}
    assert prec == {"headers/A.h": "dd", "headers/B.h": "dd", "headers/C.h": "ff"}

    # -- ceiling regions (top billing) --
    cs = report["correctness_summary"]
    assert cs["regions_at_dd_ceiling"] == 1
    assert cs["regions_dd_untested"] == 0
    ceil = cs["ceiling_regions"][0]
    assert "A.h:5" in ceil["location"]
    assert ceil["ceiling_kind"] == "dd_ceiling"
    assert ceil["final_min_digits"] == 7.2
    assert ceil["attempted_rewrites"] == ["log1p", "expm1", "hypot", "1-cos->2sin2"]
    assert report["algorithmic_rewrites"] == []   # none of A's rewrites cleared

    # -- iteration log: 5 (A) + 1 (B) + 2 (C) = 8 records, all present --
    lines = Path(report["iteration_log_path"]).read_text().splitlines()
    assert len(lines) == 8 == report["iterations"]
    tags = [json.loads(l)["log_tag"] for l in lines]
    assert tags.count("") == 8  # all Patcher-ok here; verdict-driven, no failure tags

    # -- branch state: 3 kept commits on top of start (rejects were reset away) --
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    assert branch == res["final_branch"]
    count = _git(root, "rev-list", "--count", f"{start}..HEAD").stdout.strip()
    assert count == "3"

    # -- final.diff + markdown exist and are non-trivial --
    diff = Path(res["cumulative_diff_path"]).read_text()
    assert "A.h" in diff and "B.h" in diff and "C.h" in diff
    md = Path(res["report_md_path"]).read_text()
    assert "Ceiling regions" in md and "dd_ceiling" in md
