#!/usr/bin/env python3
"""Verify a proposed patch in an isolated git worktree."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cmd, cwd, env=None):
    p = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    return p.returncode, p.stdout or ""


def _safe_json_path(root, driver):
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(root, "experiments", driver, "generated")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    return os.path.join(out_dir, "verify_report_{0}.json".format(ts))


def main():
    ap = argparse.ArgumentParser(description="Apply and verify a patch in a temp worktree")
    ap.add_argument("--patch-file", required=True, help="Path to unified diff patch")
    ap.add_argument("--driver", default="ddilog", help="targets.json driver id")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--min-digits", type=float, default=10.0)
    ap.add_argument(
        "--no-build",
        action="store_true",
        help="Pass through to scripts/run_experiment.sh",
    )
    ap.add_argument(
        "--keep-worktree",
        action="store_true",
        help="Do not remove temporary worktree (for debugging)",
    )
    ap.add_argument(
        "--report",
        default=None,
        help="Optional report output path (json). Default: experiments/<driver>/generated/verify_report_<ts>.json",
    )
    args = ap.parse_args()

    root = _repo_root()
    patch_file = os.path.abspath(args.patch_file)
    if not os.path.isfile(patch_file):
        print("error: patch file not found: {0}".format(patch_file), file=sys.stderr)
        return 2

    base_tmp = tempfile.mkdtemp(prefix="llm-patch-verify-")
    worktree = os.path.join(base_tmp, "wt")
    report = {
        "driver": args.driver,
        "patch_file": patch_file,
        "worktree": worktree,
        "batch": args.batch,
        "seed": args.seed,
        "min_digits": args.min_digits,
        "apply_exit": None,
        "experiment_exit": None,
        "worktree_add_exit": None,
        "worktree_remove_exit": None,
        "pass": False,
        "logs": {},
    }

    # Create detached worktree from current HEAD.
    rc_add, out_add = _run(["git", "worktree", "add", "--detach", worktree], cwd=root)
    report["worktree_add_exit"] = rc_add
    report["logs"]["worktree_add"] = out_add[-4000:]
    if rc_add != 0:
        out_path = args.report or _safe_json_path(root, args.driver)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(out_path)
        return 2

    try:
        with open(patch_file, encoding="utf-8", errors="replace") as pf:
            p_apply = subprocess.run(
                ["patch", "-p1", "--forward"],
                cwd=worktree,
                stdin=pf,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
        report["apply_exit"] = p_apply.returncode
        report["logs"]["patch_apply"] = (p_apply.stdout or "")[-8000:]
        if p_apply.returncode != 0:
            out_path = args.report or _safe_json_path(root, args.driver)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(out_path)
            return 2

        run_script = os.path.join(worktree, "scripts", "run_experiment.sh")
        out_csv = os.path.join(
            worktree,
            "experiments",
            args.driver,
            "generated",
            "verify_run_{0}_{1}_{2}.csv".format(args.batch, args.seed, time.strftime("%Y%m%d_%H%M%S")),
        )
        cmd = [
            "bash",
            run_script,
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
        ]
        if args.no_build:
            cmd.append("--no-build")

        env = os.environ.copy()
        env["AGENTIC_MIXED_PRECISION_DEMO_ROOT"] = worktree
        rc_exp, out_exp = _run(cmd, cwd=worktree, env=env)
        report["experiment_exit"] = rc_exp
        report["pass"] = rc_exp == 0
        report["candidate_csv"] = out_csv
        report["logs"]["experiment"] = out_exp[-20000:]

        out_path = args.report or _safe_json_path(root, args.driver)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(out_path)
        return 0 if rc_exp == 0 else 1
    finally:
        if not args.keep_worktree:
            rc_rm, out_rm = _run(["git", "worktree", "remove", "--force", worktree], cwd=root)
            report["worktree_remove_exit"] = rc_rm
            report["logs"]["worktree_remove"] = out_rm[-4000:]
            shutil.rmtree(base_tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

