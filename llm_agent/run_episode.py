#!/usr/bin/env python3
"""One-shot LLM episode: propose patch, verify patch, print verdict JSON."""

import argparse
import json
import os
import subprocess
import sys
import time


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cmd, cwd):
    p = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    return p.returncode, p.stdout or ""


def _last_nonempty_line(text):
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    return lines[-1] if lines else ""


def main():
    ap = argparse.ArgumentParser(description="Run one propose+verify LLM patch episode")
    ap.add_argument("--driver", default="ddilog")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--user", default=None)
    ap.add_argument("--model", default="claudeopus46")
    ap.add_argument("--fallback-model", default="gpt4turbo")
    ap.add_argument("--min-digits", type=float, default=10.0)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--keep-worktree", action="store_true")
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Optional output dir for patch/report (default experiments/<driver>/generated)",
    )
    args = ap.parse_args()

    root = _repo_root()
    out_dir = args.output_dir or os.path.join(root, "experiments", args.driver, "generated")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    patch_path = os.path.join(out_dir, "episode_patch_{0}.patch".format(ts))
    report_path = os.path.join(out_dir, "episode_verify_{0}.json".format(ts))

    propose_cmd = [
        sys.executable,
        "-m",
        "llm_agent.patch_proposer",
        "--driver",
        args.driver,
        "--min-digits",
        str(args.min_digits),
        "--model",
        args.model,
        "--fallback-model",
        args.fallback_model,
        "--output",
        patch_path,
    ]
    if args.base_url:
        propose_cmd += ["--base-url", args.base_url]
    if args.user:
        propose_cmd += ["--user", args.user]

    rc_prop, out_prop = _run(propose_cmd, cwd=root)
    resolved_patch = patch_path
    if rc_prop == 0:
        maybe = _last_nonempty_line(out_prop)
        if maybe and os.path.isfile(maybe):
            resolved_patch = maybe

    result = {
        "driver": args.driver,
        "propose_exit": rc_prop,
        "verify_exit": None,
        "pass": False,
        "patch_file": resolved_patch,
        "verify_report": report_path,
        "logs": {
            "propose": out_prop[-20000:],
            "verify": "",
        },
    }

    if rc_prop != 0:
        print(json.dumps(result, indent=2))
        return 2

    verify_cmd = [
        sys.executable,
        "-m",
        "llm_agent.patch_verify",
        "--patch-file",
        resolved_patch,
        "--driver",
        args.driver,
        "--batch",
        str(args.batch),
        "--seed",
        str(args.seed),
        "--min-digits",
        str(args.min_digits),
        "--report",
        report_path,
    ]
    if args.no_build:
        verify_cmd.append("--no-build")
    if args.keep_worktree:
        verify_cmd.append("--keep-worktree")

    rc_ver, out_ver = _run(verify_cmd, cwd=root)
    result["verify_exit"] = rc_ver
    result["pass"] = rc_ver == 0
    result["logs"]["verify"] = out_ver[-20000:]

    # If verifier wrote report, attach a condensed view.
    if os.path.isfile(report_path):
        try:
            with open(report_path, encoding="utf-8") as f:
                vr = json.load(f)
            result["verify_summary"] = {
                "apply_exit": vr.get("apply_exit"),
                "experiment_exit": vr.get("experiment_exit"),
                "pass": vr.get("pass"),
                "candidate_csv": vr.get("candidate_csv"),
            }
        except Exception:
            pass

    print(json.dumps(result, indent=2))
    if rc_ver == 0:
        return 0
    if rc_ver == 1:
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())

