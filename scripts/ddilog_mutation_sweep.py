#!/usr/bin/env python3
"""
For each mutation id that has patches/ddilog/<id>.patch, apply patch, run
compile + ddilog_driver + compare_results (via run_ddilog_experiment.sh), then revert.

Usage:
  ./scripts/ddilog_mutation_sweep.py --id A
  ./scripts/ddilog_mutation_sweep.py --all

Requires: bash, patch, targets.json, scripts/run_ddilog_experiment.sh
"""

import argparse
import json
import os
import subprocess
import sys
import time


def repo_root():
    env = os.environ.get("AGENTIC_MIXED_PRECISION_DEMO_ROOT")
    if env:
        return env
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_ddilog_mutation_ids(root):
    path = os.path.join(root, "targets.json")
    with open(path) as f:
        data = json.load(f)
    for d in data.get("drivers", []):
        if d.get("id") == "ddilog":
            mc = d.get("mutation_candidates") or {}
            out = []
            for loc in mc.get("locals", []):
                i = loc.get("id")
                if i:
                    out.append(i)
            return out
    return []


def patch_path(root, mid):
    return os.path.join(root, "patches", "ddilog", mid + ".patch")


def run(cmd, cwd, env=None):
    e = os.environ.copy()
    if env:
        e.update(env)
    p = subprocess.run(cmd, cwd=cwd, env=e)
    return p.returncode


def main():
    ap = argparse.ArgumentParser(description="ddilog mutation sweep")
    ap.add_argument("--id", action="append", dest="ids", help="mutation id (repeatable)")
    ap.add_argument("--all", action="store_true", help="all ids in targets.json that have a .patch file")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--min-digits", type=float, default=10.0)
    args = ap.parse_args()

    root = repo_root()
    apply_sh = os.path.join(root, "scripts", "apply_ddilog_patch.sh")
    run_exp = os.path.join(root, "scripts", "run_ddilog_experiment.sh")

    if args.all:
        candidates = load_ddilog_mutation_ids(root)
        ids = [i for i in candidates if os.path.isfile(patch_path(root, i))]
    elif args.ids:
        ids = args.ids
    else:
        ap.error("pass --id <name> or --all")

    env = os.environ.copy()
    env["AGENTIC_MIXED_PRECISION_DEMO_ROOT"] = root

    results = []
    for mid in ids:
        pfile = patch_path(root, mid)
        if not os.path.isfile(pfile):
            print("skip {} (no patch file {})".format(mid, pfile), file=sys.stderr)
            continue

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_csv = os.path.join(
            root,
            "experiments",
            "ddilog",
            "generated",
            "ddilog_mutation_{}_{}_{}_{}.csv".format(mid, args.batch, args.seed, ts),
        )

        rc_apply = run(["bash", apply_sh, "apply", mid], cwd=root, env=env)
        if rc_apply != 0:
            results.append({"id": mid, "phase": "apply", "exit": rc_apply})
            print(json.dumps({"id": mid, "error": "apply_failed", "exit": rc_apply}))
            continue

        try:
            rc = run(
                [
                    "bash",
                    run_exp,
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
            results.append({"id": mid, "compare_exit": rc, "csv": out_csv})
            print(
                json.dumps(
                    {
                        "id": mid,
                        "compare_exit": rc,
                        "csv": out_csv,
                        "pass": rc == 0,
                    }
                )
            )
        finally:
            rc_rev = run(["bash", apply_sh, "revert", mid], cwd=root, env=env)
            if rc_rev != 0:
                print(
                    json.dumps(
                        {"id": mid, "warning": "revert_failed", "exit": rc_rev}
                    ),
                    file=sys.stderr,
                )

    if any(r.get("phase") == "apply" for r in results):
        sys.exit(2)
    if any(r.get("compare_exit", 0) != 0 for r in results if "compare_exit" in r):
        sys.exit(1)


if __name__ == "__main__":
    main()
