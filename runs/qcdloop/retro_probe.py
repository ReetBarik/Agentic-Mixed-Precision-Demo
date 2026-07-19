#!/usr/bin/env python3
"""Float retro probe — re-validate the Wave-1+2 10k run's float-accepted regions
at tighter tolerances to measure float's precision headroom.

Cheap replay: NO walk, NO LLM, NO fresh characterization.  For each region we
reconstruct the exact cumulative candidate patch the walk validated
(``git diff <starting_sha>..<candidate_sha>`` in the run's headers repo) and
call the *same* Validator (:func:`agents.validator.validate.validate`) with the
*same* base_state / snapshot the walk used.

Key structural fact exploited for cost control (validate() source, ``_decide``):
the tolerance is a **pure threshold** applied at the very end to a single
``candidate.min_precise_digits`` — it never touches the builds or the
precise-digit scoring.  So one validate() call per region yields the region's
``cand_min`` (and ``curr_min``); survival at every tolerance is then
``_decide(cand_min, curr_min, max_regression, floor=tol)``.  This is exactly the
verdict the walk would have produced at that tolerance, and it collapses the
naive 86*4 = 344 builds to 86 while making monotonicity structural rather than
merely observed (a downward threshold on a fixed number cannot be non-monotone).

Determinism cross-check: each replayed ``cand_min`` is compared against the
``candidate_min_precise_digits`` recorded in iterations.jsonl.

Source run: 20260719_185132_033cbe69  (langgraph-agents @ 3fd5ad3, CALIBRATION_v2)
Headers repo (holds the candidate commits): ~/amp_strategy_headers_repo
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agents.validator.validate import validate, _decide  # noqa: E402

RUN_ID = "20260719_185132_033cbe69"
RUN_DIR = REPO / "runs" / "qcdloop" / "strategy" / RUN_ID
ITERATIONS = RUN_DIR / "iterations.jsonl"
HEADERS_REPO = Path.home() / "amp_strategy_headers_repo"
# base commit of the headers repo for THIS run ("base: qcdloop_headers_full snapshot")
STARTING_SHA = "93f577258bf2d54e4ebba00fc282938e6d5ab6ba"

# Faithful replay of the walk's config (run_strategy_10k.sh + run_strategy_e2e.py).
BASE_STATE = {
    "vanilla_headers": str(REPO / "runs" / "qcdloop_headers_full"),
    "dd_source_repo": str(Path.home() / "qcdloop"),
    "dd_ref": "ddfun_enabled",
    "accepted_patches": [],
    "kokkos_root": str(Path.home() / "kokkos-install"),
}
SNAPSHOT = {"seed": 12345, "sample_count": 1000}
MAX_REGRESSION = 0.5          # validate() default (regression guard, tol-independent)
TOLERANCES = [8, 9, 10, 11]
EXPECTED_COUNT = 86           # CALIBRATION_v2: "lines -> float (final) = 86"
DIGIT_TOL = 1e-4             # cross-check tolerance (min_precise_digits rounded to 4dp)

OUT_JSON = REPO / "runs" / "qcdloop" / "float_retro_probe_results.json"


def _float_accepted_regions() -> list[dict]:
    regs = []
    for line in ITERATIONS.read_text().splitlines():
        d = json.loads(line)
        if d.get("kind") == "double-to-float" and d.get("accepted") is True:
            regs.append(d)
    return regs


def _cumulative_patch(candidate_sha: str) -> str:
    r = subprocess.run(
        ["git", "-C", str(HEADERS_REPO), "diff", f"{STARTING_SHA}..{candidate_sha}"],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"git diff {STARTING_SHA[:8]}..{candidate_sha[:8]} failed:\n{r.stderr}")
    return r.stdout


def main() -> int:
    regs = _float_accepted_regions()
    print(f"[probe] float-accepted regions in {ITERATIONS.name}: {len(regs)}",
          flush=True)
    if len(regs) != EXPECTED_COUNT:
        print(f"RETRO_PROBE_DISCREPANCY: expected {EXPECTED_COUNT} float-accepted "
              f"regions (CALIBRATION_v2) but iterations.jsonl has {len(regs)}. "
              f"Refusing to guess which set to probe.", flush=True)
        return 2

    # sanity: all candidate commits present in the headers repo
    for r in regs:
        sha = r["candidate_sha"]
        t = subprocess.run(["git", "-C", str(HEADERS_REPO), "cat-file", "-t", sha],
                           capture_output=True, text=True).stdout.strip()
        if t != "commit":
            print(f"RETRO_PROBE_BLOCKED: candidate_sha {sha} not a commit in "
                  f"{HEADERS_REPO} (got {t!r}).", flush=True)
            return 2

    results = []
    t_start = time.monotonic()
    for i, r in enumerate(regs):
        tgt = r["target"]
        sha = r["candidate_sha"]
        patch = _cumulative_patch(sha)
        t0 = time.monotonic()
        # tolerance here is irrelevant to cand_min/curr_min; we threshold ourselves.
        v = validate(BASE_STATE, patch, tolerance=float(max(TOLERANCES)),
                     snapshot=SNAPSHOT, persist=False)
        wall = time.monotonic() - t0
        cand_min = v["candidate"]["min_precise_digits"]
        curr_min = v["current"]["min_precise_digits"]
        hot = v["candidate"].get("hotspot") or {}
        recorded = r.get("candidate_min_precise_digits")

        # survival at each tol == the real verdict at that tol (pure threshold)
        survive = {}
        for tol in TOLERANCES:
            verdict, reason = _decide(cand_min, curr_min, MAX_REGRESSION,
                                      floor=float(tol))
            survive[tol] = {"verdict": verdict, "reason": reason,
                            "accept": verdict == "accept"}

        # determinism cross-check vs the value the walk recorded
        xcheck_ok = (recorded is not None
                     and abs(cand_min - float(recorded)) <= DIGIT_TOL)

        row = {
            "iter_id": r["iter_id"],
            "file": tgt["file"],
            "line": tgt["line_start"],
            "line_end": tgt["line_end"],
            "candidate_sha": sha,
            "cand_min_precise_digits": cand_min,
            "curr_min_precise_digits": curr_min,
            "delta": round(cand_min - curr_min, 6),
            "recorded_cand_min": recorded,
            "xcheck_ok": xcheck_ok,
            "hotspot_integral": hot.get("integral"),
            "hotspot_component": hot.get("component"),
            "hotspot_precise_digits": hot.get("precise_digits"),
            "survive": survive,
            "wall_seconds": round(wall, 2),
        }
        results.append(row)
        flag = "" if xcheck_ok else "  <-- XCHECK MISMATCH"
        print(f"[{i+1:2d}/{len(regs)}] {tgt['file']}:{tgt['line_start']:<4d} "
              f"cand={cand_min:.4f} curr={curr_min:.4f} d={row['delta']:+.4f} "
              f"hot={hot.get('integral')}/{hot.get('component')} "
              f"({wall:.1f}s){flag}", flush=True)

    # ---- survival table (survivors out of N at each tol) ----
    n = len(results)
    survival = {tol: sum(1 for x in results if x["survive"][tol]["accept"])
                for tol in TOLERANCES}
    survival[7] = n  # all were accepted at tol=7 by construction (the source walk)

    # ---- monotonicity check (should be structural; verify anyway) ----
    mono_violations = []
    for x in results:
        acc = [(tol, x["survive"][tol]["accept"]) for tol in TOLERANCES]
        for (t_lo, a_lo), (t_hi, a_hi) in zip(acc, acc[1:]):
            if a_hi and not a_lo:  # passes stricter tol but fails looser one
                mono_violations.append({"region": f"{x['file']}:{x['line']}",
                                        "fail_tol": t_lo, "pass_tol": t_hi})

    xcheck_fail = [x for x in results if not x["xcheck_ok"]]

    summary = {
        "run_id": RUN_ID,
        "n_regions": n,
        "snapshot": SNAPSHOT,
        "max_regression": MAX_REGRESSION,
        "tolerances": TOLERANCES,
        "survival": survival,
        "xcheck_failures": len(xcheck_fail),
        "monotonicity_violations": mono_violations,
        "total_wall_seconds": round(time.monotonic() - t_start, 1),
    }

    OUT_JSON.write_text(json.dumps({"summary": summary, "regions": results},
                                   indent=2))
    print("\n=== SURVIVAL TABLE ===", flush=True)
    print(f"  tol=7 (source): {n}/{n}", flush=True)
    for tol in TOLERANCES:
        print(f"  tol={tol}: {survival[tol]}/{n}", flush=True)
    print(f"xcheck failures: {len(xcheck_fail)} / {n}", flush=True)
    print(f"monotonicity violations: {len(mono_violations)}", flush=True)
    print(f"results written: {OUT_JSON}", flush=True)
    print(f"total wall: {summary['total_wall_seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
