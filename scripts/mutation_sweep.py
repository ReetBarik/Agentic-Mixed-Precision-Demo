#!/usr/bin/env python3
"""
For each mutation id that has a patch file under mutation_candidates.patches_directory,
apply patch, run compile + driver + compare_results (via run_experiment.sh), then revert.

Usage:
  ./scripts/mutation_sweep.py --driver ddilog --id A
  ./scripts/mutation_sweep.py --driver ddilog --all

Requires: bash, patch, targets.json, scripts/run_experiment.sh
"""

import argparse
import json
import os
import sys

from mutation_trial import trial_one_mutation
from targets_lib import mutation_patch_path, repo_root, require_driver


def main():
    ap = argparse.ArgumentParser(description="mutation sweep (apply → experiment → revert)")
    ap.add_argument(
        "--driver",
        default="ddilog",
        help="targets.json drivers[].id (default: ddilog)",
    )
    ap.add_argument("--id", action="append", dest="ids", help="mutation id (repeatable)")
    ap.add_argument(
        "--all",
        action="store_true",
        help="all ids in targets.json that have a .patch file",
    )
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--min-digits", type=float, default=10.0)
    ap.add_argument(
        "--no-build",
        action="store_true",
        help="pass through to run_experiment.sh (skip compile)",
    )
    args = ap.parse_args()

    root = repo_root()
    driver = require_driver(root, args.driver)

    if args.all:
        candidates = []
        for loc in (driver.get("mutation_candidates") or {}).get("locals", []):
            i = loc.get("id")
            if i:
                candidates.append(i)
        ids = [i for i in candidates if os.path.isfile(mutation_patch_path(root, driver, i))]
    elif args.ids:
        ids = args.ids
    else:
        ap.error("pass --id <name> or --all")

    results = []
    for mid in ids:
        pfile = mutation_patch_path(root, driver, mid)
        if not os.path.isfile(pfile):
            print("skip {} (no patch file {})".format(mid, pfile), file=sys.stderr)
            continue

        r = trial_one_mutation(
            args.driver,
            driver,
            mid,
            args.batch,
            args.seed,
            args.min_digits,
            no_build=args.no_build,
        )
        results.append(r)

        if r["apply_exit"] != 0:
            print(json.dumps({"id": mid, "error": "apply_failed", "exit": r["apply_exit"]}))
            continue

        rc = r.get("compare_exit")
        if r.get("revert_exit") not in (None, 0):
            print(
                json.dumps(
                    {
                        "id": mid,
                        "warning": "revert_failed",
                        "exit": r["revert_exit"],
                    }
                ),
                file=sys.stderr,
            )

        print(
            json.dumps(
                {
                    "id": mid,
                    "compare_exit": rc,
                    "csv": r.get("csv"),
                    "pass": rc == 0,
                }
            )
        )

    if any(r.get("apply_exit", 0) != 0 for r in results):
        sys.exit(2)
    if any(
        r.get("compare_exit", 0) != 0
        for r in results
        if r.get("apply_exit") == 0 and r.get("compare_exit") is not None
    ):
        sys.exit(1)


if __name__ == "__main__":
    main()
