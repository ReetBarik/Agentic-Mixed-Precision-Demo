#!/usr/bin/env python3
"""Post-run pass-criteria summary for the Wave-3 + tail-battery 10k validation.

Reads a Strategy run dir (``report.json`` + ``iterations.jsonl``) and prints the
numbers PIPELINE_v1 needs, grouped as the reframed pass criteria:

  * terminal status + demotion count (precision_distribution)
  * Wave-3 prune telemetry (WI1 regions_skipped_range_unsafe, WI2
    regions_flagged_pred_float, WI3 speedup_queue_flop_weighted) from speedup_summary
  * tail-battery telemetry: batteries run, hash mismatches, samples tested, and a
    verdict_reason breakdown separating *correctness regressions* (insufficient_fix
    / tail-driven regression on a non-dd rung) from *expected* genuine dd-ceiling
    rejects.

Usage:
    python runs/qcdloop/analyze_tail_run.py --run-dir runs/qcdloop/strategy/<id>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    report = json.loads((run_dir / "report.json").read_text())
    rows = [json.loads(l) for l in (run_dir / "iterations.jsonl").read_text().splitlines()
            if l.strip()]

    # --- tail telemetry ---
    tail_rows = [r for r in rows if isinstance(r.get("tail"), dict)]
    batteries = [r for r in tail_rows if r["tail"].get("batteries_run", 0) > 0]
    hash_mismatches = sum(r["tail"].get("hash_mismatches", 0) for r in tail_rows)
    max_samples = max((r["tail"].get("samples_tested", 0) for r in tail_rows), default=0)
    offsets = max((r["tail"].get("offsets", 0) for r in tail_rows), default=0)

    # --- verdict breakdown ---
    reasons = Counter(r.get("verdict_reason") for r in rows if r.get("verdict_reason"))
    # a "correctness regression" (must be zero for good candidates) = a non-dd rung
    # that rejected for insufficient_fix or regression.  Genuine dd-ceiling rejects
    # at the double-to-dd rung are EXPECTED (physics ceiling), not regressions.
    def is_dd_rung(r):
        return str(r.get("kind", "")).endswith("-to-dd")
    reject_rows = [r for r in rows if r.get("validator_verdict") == "reject"]
    dd_ceiling_rejects = [r for r in reject_rows if is_dd_rung(r)]
    nondd_regressions = [r for r in reject_rows
                         if not is_dd_rung(r)
                         and r.get("verdict_reason") in ("regression", "insufficient_fix")]
    # tail-driven: a regression where the tail cand min is the binding minimum
    tail_driven = []
    for r in reject_rows:
        t = r.get("tail") or {}
        tc = t.get("cand_min_precise_digits")
        rc = r.get("candidate_min_precise_digits")
        if (r.get("verdict_reason") == "regression" and tc is not None
                and rc is not None and tc < rc):
            tail_driven.append(r)

    strat_bugs = [r for r in rows if r.get("strategy_bug")]
    commit_failed = [r for r in rows if r.get("patcher_status") == "commit_failed"]

    ss = report.get("speedup_summary", {})
    out = {
        "run_id": report.get("run_id"),
        "status": report.get("status"),
        "duration_sec": report.get("duration_sec"),
        "duration_min": (round(report["duration_sec"] / 60, 1)
                         if report.get("duration_sec") else None),
        "iterations": report.get("iterations"),
        "precision_distribution": report.get("precision_distribution"),
        "wave3_prune_telemetry": {
            "regions_skipped_range_unsafe": ss.get("regions_skipped_range_unsafe"),
            "regions_flagged_pred_float": ss.get("regions_flagged_pred_float"),
            "speedup_queue_flop_weighted": ss.get("speedup_queue_flop_weighted"),
            "report_prunes_enabled": ss.get("report_prunes_enabled"),
        },
        "tail_telemetry": {
            "validations_with_tail_battery": len(batteries),
            "tail_hash_mismatches": hash_mismatches,
            "tail_offsets_dispatched": offsets,
            "max_tail_samples_tested": max_samples,
        },
        "verdict_reasons": dict(reasons),
        "correctness_regressions_nondd": len(nondd_regressions),
        "tail_driven_regression_rejects": len(tail_driven),
        "expected_dd_ceiling_rejects": len(dd_ceiling_rejects),
        "strategy_bugs": len(strat_bugs),
        "commit_failed": len(commit_failed),
    }
    # surface the actual offending rows so failures are never silent
    if nondd_regressions:
        out["nondd_regression_samples"] = [
            {"iter": r["iter_id"], "kind": r["kind"], "reason": r.get("verdict_reason"),
             "target": r.get("target")}
            for r in nondd_regressions[:20]]
    if tail_driven:
        out["tail_driven_samples"] = [
            {"iter": r["iter_id"], "kind": r["kind"], "tail": r.get("tail")}
            for r in tail_driven[:20]]

    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
