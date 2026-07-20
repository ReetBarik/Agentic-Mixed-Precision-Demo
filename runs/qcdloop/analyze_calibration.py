#!/usr/bin/env python3
"""Compute per-phase calibration metrics from a Strategy run + its char report.

Reads a Strategy run dir (``report.json`` + ``iterations.jsonl``) and the
characterization report it walked, and prints the numbers CALIBRATION.md needs:
per-phase iters/accepts/rejects/dd_untested/llm_gen_failed, iters-per-accept and
iters-per-region-attempted, the phase-2 skip rate (dd-promoted regions dropped
from the speedup candidate set), the speedup-phase kind breakdown (plain-edit vs
LLM path), and an inferred terminal status per phase.

Everything here is derived, not measured live, so it is safe to re-run against
any completed Strategy run.

Usage:
    python runs/qcdloop/analyze_calibration.py \
        --run-dir runs/qcdloop/strategy/<run_id> \
        --report  runs/qcdloop/report_10k.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from agents.patcher.dispatch import dispatch_path  # noqa: E402
from agents.strategy.characterization import load_regions  # noqa: E402
from agents.strategy.dispatch import DISPATCH  # noqa: E402
from agents.strategy.ranking import build_queues  # noqa: E402

# statuses that count against the iteration budget (strategy/dispatch.py)
BUDGET_STATUSES = {s for s, e in DISPATCH.items() if e.counts_budget}
# Bucket-A rejects that count budget (compile/runtime/timeout — not llm/bug/fatal)
BUCKET_A = {s for s, e in DISPATCH.items() if e.is_reject and e.counts_budget}


def _target_key(row: dict) -> tuple:
    t = row.get("target", {})
    return (t.get("file"), t.get("line_start"), t.get("line_end"), row.get("kind"))


def _region_key(row: dict) -> tuple:
    t = row.get("target", {})
    return (t.get("file"), t.get("line_start"), t.get("line_end"))


def phase_metrics(rows: list[dict], phase: str) -> dict:
    pr = [r for r in rows if r.get("phase") == phase]
    budget_iters = sum(1 for r in pr if r["patcher_status"] in BUDGET_STATUSES)
    accepts = sum(1 for r in pr if r.get("accepted"))
    genuine_rejects = sum(1 for r in pr
                          if r["patcher_status"] == "ok" and r.get("validator_verdict") == "reject")
    bucket_a = sum(1 for r in pr if r["patcher_status"] in BUCKET_A)
    llm_gen_failed = sum(1 for r in pr if r["patcher_status"] == "llm_gen_failed")
    empty_candidate = sum(1 for r in pr if r["patcher_status"] == "empty_candidate")
    strategy_bug = sum(1 for r in pr if r["patcher_status"] == "patch_apply_failed")
    commit_failed = sum(1 for r in pr if r["patcher_status"] == "commit_failed")
    regions = {_region_key(r) for r in pr}
    by_status = Counter(r["patcher_status"] for r in pr)

    # kind breakdown with dispatch-path classification (plain-edit vs LLM path)
    kinds = defaultdict(lambda: {"iters": 0, "accepts": 0, "budget_iters": 0})
    for r in pr:
        k = r["kind"]
        kinds[k]["iters"] += 1
        if r.get("accepted"):
            kinds[k]["accepts"] += 1
        if r["patcher_status"] in BUDGET_STATUSES:
            kinds[k]["budget_iters"] += 1

    return {
        "phase": phase,
        "total_iterations": len(pr),
        "budget_iters": budget_iters,
        "accepts": accepts,
        "genuine_rejects": genuine_rejects,
        "bucket_a_rejects": bucket_a,
        "llm_gen_failed": llm_gen_failed,
        "empty_candidate": empty_candidate,
        "commit_failed": commit_failed,
        "strategy_bug": strategy_bug,
        "distinct_regions_attempted": len(regions),
        "by_patcher_status": dict(by_status),
        "iters_per_accept": (budget_iters / accepts) if accepts else None,
        "iters_per_region": (budget_iters / len(regions)) if regions else None,
        "kinds": {k: dict(v) for k, v in sorted(kinds.items())},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    report = json.loads((run_dir / "report.json").read_text())
    rows = [json.loads(l) for l in (run_dir / "iterations.jsonl").read_text().splitlines() if l.strip()]

    tol = report["tolerance"]
    regions, _meta = load_regions(args.report)
    corr_q, speedup_q = build_queues(regions, tol)

    corr = phase_metrics(rows, "correctness")
    spd = phase_metrics(rows, "speedup")

    ps = report.get("phase_summary", {})
    cap_c = ps.get("correctness", {}).get("iter_cap")
    cap_s = ps.get("speedup", {}).get("iter_cap")
    skipped = ps.get("speedup", {}).get("skipped_dd_promoted", 0)

    # terminal-status inference per phase
    def term(phase, budget_iters, cap):
        if cap is not None and budget_iters >= cap:
            return f"budget_exhausted (hit cap {cap})"
        return f"completed/drained (used {budget_iters}/{cap})"

    out = {
        "run_id": report["run_id"],
        "overall_status": report["status"],
        "tolerance": tol,
        "duration_sec": report["duration_sec"],
        "total_iterations": report["iterations"],
        "budget_iters_used": report["budget_iters_used"],
        "llm_tokens_used": report["llm_tokens_used"],
        "precision_distribution": report["precision_distribution"],
        "queues": {
            "correctness_queue_len": len(corr_q),
            "speedup_queue_len": len(speedup_q),
            "n_chains_ranked_note": "chains ranked separately inside correctness phase",
        },
        "phase_1_correctness": {
            **corr,
            "iter_cap": cap_c,
            "inferred_terminal": term("correctness", corr["budget_iters"], cap_c),
        },
        "phase_2_speedup": {
            **spd,
            "iter_cap_effective": cap_s,
            "skipped_dd_promoted": skipped,
            "speedup_skip_rate": (skipped / len(speedup_q)) if speedup_q else None,
            "inferred_terminal": term("speedup", spd["budget_iters"], cap_s),
        },
        "correctness_summary": report["correctness_summary"],
        # Wave-3 report-prune telemetry (WI1/WI2/WI3) — echoed verbatim so the
        # validation re-run reports the per-prune bite without extra wiring.
        "speedup_summary": report.get("speedup_summary", {
            "note": "pre-Wave-3 run: no speedup_summary in report.json"}),
        "shakedown_baseline": {
            "accepts": 8, "iterations": 19, "accept_rate": 8 / 19,
            "dd_untested": 11, "dd_untested_rate": 11 / 19,
            "note": "run 20260718_194556_67dbcf37, tol=8, single-budget (pre-a29477d)",
        },
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
