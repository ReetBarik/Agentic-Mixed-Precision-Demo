#!/usr/bin/env python3
"""
Greedy search: start from all-double (pristine tree), repeatedly add one float
local (patch) while compare vs baseline still passes --min-digits.

Patches must stack (minimal context; separate lines per local where needed).

Apply order follows mutation_candidates.locals[].id order in targets.json.

Usage:
  ./scripts/mutation_combo_greedy.py [--driver ddilog] [--min-digits 10] [--batch 10] [--seed 123]

Exit 0 if final set non-empty or greedy completes; 1 on error.
"""

import argparse
import json
import os
import re
import sys
import time

from mutation_ops import apply_py_path, apply_stack, revert_stack, run_capture
from targets_lib import (
    experiments_generated_dir,
    mutation_local_ids_in_order,
    mutation_patch_path,
    repo_root,
    require_driver,
)


def run(cmd, cwd, env):
    return run_capture(cmd, cwd, env)


def parse_min_digits(text):
    m = re.search(r"min_precise_digits=([0-9.eE+-]+)", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def main():
    ap = argparse.ArgumentParser(description="greedy float combo for mutation patches")
    ap.add_argument(
        "--driver",
        default="ddilog",
        help="targets.json drivers[].id (default: ddilog)",
    )
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--min-digits", type=float, default=10.0)
    ap.add_argument(
        "--tie-break",
        choices=["margin", "first"],
        default="margin",
        help="among passing candidates: pick highest min_precise_digits, or first in sort order",
    )
    args = ap.parse_args()

    root = repo_root()
    driver = require_driver(root, args.driver)
    apply_py = apply_py_path(root)
    run_exp = os.path.join(root, "scripts", "run_experiment.sh")

    env = os.environ.copy()
    env["AGENTIC_MIXED_PRECISION_DEMO_ROOT"] = root

    candidates = [
        i
        for i in mutation_local_ids_in_order(driver)
        if os.path.isfile(mutation_patch_path(root, driver, i))
    ]
    if not candidates:
        pd = (driver.get("mutation_candidates") or {}).get("patches_directory", "")
        print(
            "error: no patches under {} for driver {}".format(pd, args.driver),
            file=sys.stderr,
        )
        sys.exit(2)

    accepted = []
    remaining = set(candidates)
    trace = []

    while remaining:
        best_cand = None
        best_score = -1.0
        best_out = ""
        best_rc = 1

        try_list = sorted(remaining)
        for cand in try_list:
            trial = set(accepted) | {cand}
            ok, order, err = apply_stack(
                root, apply_py, args.driver, driver, trial, env
            )
            if not ok:
                trace.append(
                    {
                        "trial": sorted(trial),
                        "error": "apply_failed",
                        "detail": err[:500],
                    }
                )
                continue

            ts = time.strftime("%Y%m%d_%H%M%S")
            out_csv = os.path.join(
                experiments_generated_dir(root, args.driver),
                "{}_combo_try_{}_{}.csv".format(
                    args.driver,
                    "_".join(sorted(trial)),
                    ts,
                ),
            )

            rc, out = run(
                [
                    "bash",
                    run_exp,
                    "--driver",
                    args.driver,
                    "-o",
                    out_csv,
                    "--batch",
                    str(args.batch),
                    "--seed",
                    str(args.seed),
                    "--min-digits",
                    str(args.min_digits),
                ],
                cwd=root,
                env=env,
            )
            revert_stack(root, apply_py, args.driver, order, env)

            md = parse_min_digits(out)
            trace.append(
                {
                    "trial": sorted(trial),
                    "compare_exit": rc,
                    "min_precise_digits": md,
                    "csv": out_csv,
                }
            )

            if rc != 0:
                continue

            score = md if md is not None else 0.0
            if args.tie_break == "first":
                best_cand = cand
                best_score = score
                best_out = out
                best_rc = rc
                break
            if score > best_score:
                best_score = score
                best_cand = cand
                best_out = out
                best_rc = rc

        if best_cand is None:
            break

        accepted.append(best_cand)
        remaining.remove(best_cand)
        print(
            json.dumps(
                {
                    "round": len(accepted),
                    "added": best_cand,
                    "accepted_set": sorted(accepted),
                    "min_precise_digits": best_score,
                }
            )
        )

    result = {
        "greedy": "done",
        "driver": args.driver,
        "accepted_float_locals": sorted(accepted),
        "count": len(accepted),
        "min_digits_threshold": args.min_digits,
        "trace_tail": trace[-20:],
    }
    out_path = os.path.join(
        experiments_generated_dir(root, args.driver),
        "combo_greedy_{}.json".format(time.strftime("%Y%m%d_%H%M%S")),
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"result_file": out_path, **result}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
