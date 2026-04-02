#!/usr/bin/env python3
"""Verify a proposed patch in an isolated git worktree."""

import argparse
import json
import os
import glob
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


def _extract_report_file(stdout_text):
    try:
        obj = json.loads(stdout_text)
        return obj.get("report_file")
    except Exception:
        pass
    marker = '"report_file":'
    if marker not in stdout_text:
        return None
    i = stdout_text.rfind(marker)
    snippet = stdout_text[i:]
    q1 = snippet.find('"', len(marker))
    q2 = snippet.find('"', q1 + 1)
    if q1 >= 0 and q2 > q1:
        return snippet[q1 + 1 : q2]
    return None


def _latest_onboard_report(worktree, target_id):
    pattern = os.path.join(
        worktree, "experiments", target_id, "generated", "onboarding_apply_*.json"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return files[-1]


def _run_onboard_apply(worktree, spec_file, target_id, batch, seed, min_digits, regen_baseline):
    cmd = [
        sys.executable,
        "-m",
        "llm_agent.onboard_target",
        "--spec-file",
        spec_file,
        "--target-id",
        target_id,
        "--generator",
        "deterministic",
        "--apply",
        "--allow-existing",
        "--batch",
        str(batch),
        "--seed",
        str(seed),
        "--min-digits",
        str(min_digits),
    ]
    if regen_baseline:
        cmd.append("--regen-baseline")
    rc, out = _run(cmd, cwd=worktree)
    report_file = _extract_report_file(out) or _latest_onboard_report(worktree, target_id)
    summary = None
    if report_file and os.path.isfile(report_file):
        try:
            with open(report_file, encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = None
    return rc, out, report_file, summary


def main():
    ap = argparse.ArgumentParser(description="Apply and verify a patch in a temp worktree")
    ap.add_argument("--patch-file", required=True, help="Path to unified diff patch")
    ap.add_argument("--driver", default="ddilog", help="targets.json driver id")
    ap.add_argument("--spec-file", default="", help="Optional spec JSON for spec-mode verify")
    ap.add_argument("--target-id", default="", help="Target id in spec file")
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
        "spec_file": args.spec_file or None,
        "target_id": args.target_id or None,
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
        if args.spec_file:
            target_id = args.target_id or args.driver
            spec_path = args.spec_file
            if not os.path.isabs(spec_path):
                spec_path = os.path.join(root, spec_path)
            if not os.path.isfile(spec_path):
                report["apply_exit"] = 2
                report["logs"]["spec"] = "missing spec file: {0}".format(spec_path)
                out_path = args.report or _safe_json_path(root, target_id)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                print(out_path)
                return 2
            if spec_path.startswith(root + os.sep):
                spec_rel = os.path.relpath(spec_path, root)
                spec_worktree = os.path.join(worktree, spec_rel)
            else:
                spec_worktree = spec_path

            # Build baseline on clean worktree before patch apply.
            rc_base, out_base, rpt_base, sum_base = _run_onboard_apply(
                worktree=worktree,
                spec_file=spec_worktree,
                target_id=target_id,
                batch=args.batch,
                seed=args.seed,
                min_digits=args.min_digits,
                regen_baseline=True,
            )
            report["logs"]["onboard_baseline"] = out_base[-12000:]
            report["onboard_baseline_report"] = rpt_base
            if rc_base != 0:
                report["apply_exit"] = 2
                out_path = args.report or _safe_json_path(root, target_id)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                print(out_path)
                return 2

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
            out_id = (args.target_id or args.driver) if args.spec_file else args.driver
            out_path = args.report or _safe_json_path(root, out_id)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(out_path)
            return 2

        if args.spec_file:
            target_id = args.target_id or args.driver
            rc_run, out_run, rpt_run, sum_run = _run_onboard_apply(
                worktree=worktree,
                spec_file=spec_worktree,
                target_id=target_id,
                batch=args.batch,
                seed=args.seed,
                min_digits=args.min_digits,
                regen_baseline=False,
            )
            report["experiment_exit"] = (0 if rc_run == 0 else 2)
            report["pass"] = rc_run == 0
            report["candidate_csv"] = None
            report["onboard_apply_report"] = rpt_run
            report["logs"]["experiment"] = out_run[-20000:]
            if sum_run:
                steps = {s.get("step"): s for s in sum_run.get("steps", [])}
                run_step = steps.get("run_candidate") or {}
                run_artifacts = run_step.get("artifacts") or {}
                report["candidate_csv"] = run_artifacts.get("run_csv")
                cmp_step = steps.get("smoke_compare") or {}
                cmp_details = cmp_step.get("details") or {}
                tail = cmp_details.get("log_tail") or ""
                if tail:
                    report["logs"]["experiment"] = (
                        (report["logs"].get("experiment", "") or "")
                        + "\n"
                        + tail
                    )[-20000:]

            out_path = args.report or _safe_json_path(root, target_id)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(out_path)
            return 0 if report["pass"] else 1

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

