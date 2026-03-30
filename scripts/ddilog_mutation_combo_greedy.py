#!/usr/bin/env python3
"""
Greedy search: start from all-double (pristine tree), repeatedly add one float
local (patch) while compare vs baseline still passes --min-digits.

Patches must stack (use patches/ddilog/*.patch with minimal context; Y/S/A are
separate lines in kokkosUtils.h).

Apply order is top-to-bottom in the function: T, Y, S, A, H, ALFA, B1, B2, B0.

Usage:
  ./scripts/ddilog_mutation_combo_greedy.py [--min-digits 10] [--batch 10] [--seed 123]

Exit 0 if final set non-empty or greedy completes; 1 on error.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

# Must match dependency-free apply order (file order in ddilog)
PATCH_ORDER = ["T", "Y", "S", "A", "H", "ALFA", "B1", "B2", "B0"]


def repo_root():
    env = os.environ.get("AGENTIC_MIXED_PRECISION_DEMO_ROOT")
    if env:
        return env
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_ids(root):
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


def ordered_subset(ids_set):
    return [i for i in PATCH_ORDER if i in ids_set]


def run(cmd, cwd, env):
    p = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    return p.returncode, p.stdout or ""


def parse_min_digits(text):
    m = re.search(r"min_precise_digits=([0-9.eE+-]+)", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def apply_stack(root, apply_sh, ids_set, env):
    order = ordered_subset(set(ids_set))
    for mid in order:
        rc, out = run(["bash", apply_sh, "apply", mid], cwd=root, env=env)
        if rc != 0:
            for m2 in reversed(order[: order.index(mid)]):
                run(["bash", apply_sh, "revert", m2], cwd=root, env=env)
            return False, order[: order.index(mid)], out
    return True, order, ""


def revert_stack(root, apply_sh, order, env):
    for mid in reversed(order):
        run(["bash", apply_sh, "revert", mid], cwd=root, env=env)


def main():
    ap = argparse.ArgumentParser(description="greedy float combo for ddilog")
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
    apply_sh = os.path.join(root, "scripts", "apply_ddilog_patch.sh")
    run_exp = os.path.join(root, "scripts", "run_ddilog_experiment.sh")

    env = os.environ.copy()
    env["AGENTIC_MIXED_PRECISION_DEMO_ROOT"] = root

    candidates = [i for i in load_ids(root) if os.path.isfile(patch_path(root, i))]
    if not candidates:
        print("error: no patches in patches/ddilog/", file=sys.stderr)
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
            ok, order, err = apply_stack(root, apply_sh, trial, env)
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
                root,
                "experiments",
                "ddilog",
                "generated",
                "ddilog_combo_try_{}_{}.csv".format(
                    "_".join(sorted(trial)), ts
                ),
            )

            rc, out = run(
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
            revert_stack(root, apply_sh, order, env)

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
        "accepted_float_locals": sorted(accepted),
        "count": len(accepted),
        "min_digits_threshold": args.min_digits,
        "trace_tail": trace[-20:],
    }
    out_path = os.path.join(
        root,
        "experiments",
        "ddilog",
        "generated",
        "combo_greedy_{}.json".format(time.strftime("%Y%m%d_%H%M%S")),
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"result_file": out_path, **result}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
