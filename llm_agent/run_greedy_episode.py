#!/usr/bin/env python3
"""Greedy LLM controller: propose single-local edits, accumulate accepted set."""

import argparse
import json
import os
import re
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


def _parse_min_precise_digits(text):
    m = re.search(r"min_precise_digits=([0-9.eE+-]+)", text or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _load_driver(root, driver_id):
    with open(os.path.join(root, "targets.json"), encoding="utf-8") as f:
        data = json.load(f)
    for d in data.get("drivers", []):
        if d.get("id") == driver_id:
            return d
    return None


def _load_spec_target(root, spec_file, target_id):
    path = spec_file if os.path.isabs(spec_file) else os.path.join(root, spec_file)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and isinstance(payload.get("targets"), list):
        for t in payload.get("targets", []):
            if t.get("id") == target_id:
                return t
    if isinstance(payload, dict) and payload.get("id") == target_id:
        return payload
    return None


def _candidate_symbols(driver):
    mc = driver.get("mutation_candidates") or {}
    out = []
    for loc in mc.get("locals", []):
        sym = loc.get("symbol")
        if sym:
            out.append(sym)
    return out


def _candidate_symbols_from_spec(target):
    out = []
    for sym in (target.get("locals_for_downcast", []) or []):
        if isinstance(sym, str) and sym.strip():
            out.append(sym.strip())
    return out


def _impl_rel(driver):
    mc = driver.get("mutation_candidates") or {}
    return mc.get("implementation_relative") or "src/kokkosUtils.h"


def _impl_rel_from_spec(target):
    return target.get("header_path") or "src/kokkosUtils.h"


def _exec_rel(driver, driver_id):
    if not driver:
        return "build/{0}_driver".format(driver_id)
    return driver.get("executable_relative") or "build/{0}_driver".format(driver_id)


def _validate_guided_patch(patch_path, impl_rel, focus_var, known_symbols, accepted_vars):
    if not os.path.isfile(patch_path):
        return False, "policy_reject: patch file missing"
    touched_files = []
    line_changes = 0
    focus_seen = False
    touched_symbols = set()
    with open(patch_path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("+++ b/"):
                touched_files.append(line[len("+++ b/") :])
                continue
            if not line:
                continue
            if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
                continue
            if line[0] not in ("+", "-"):
                continue
            line_changes += 1
            body = line[1:]
            if re.search(r"\b{0}\b".format(re.escape(focus_var)), body):
                focus_seen = True
            for sym in known_symbols:
                if re.search(r"\b{0}\b".format(re.escape(sym)), body):
                    touched_symbols.add(sym)
    uniq_files = sorted(set(touched_files))
    if len(uniq_files) != 1 or uniq_files[0] != impl_rel:
        return False, "policy_reject: patch must touch only {0}".format(impl_rel)
    if line_changes > 16:
        return False, "policy_reject: patch too large for greedy guided mode"
    if not focus_seen:
        return False, "policy_reject: focus variable {0} not found in changes".format(focus_var)
    allowed = set([focus_var]) | set(accepted_vars or [])
    illegal = sorted([s for s in touched_symbols if s not in allowed])
    if illegal:
        return False, "policy_reject: touched non-allowed semantic vars ({0}); allowed set is ({1})".format(
            ", ".join(illegal), ", ".join(sorted(allowed))
        )
    return True, None


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


def _verify_with_stack(
    root,
    driver_id,
    patch_stack,
    batch,
    seed,
    min_digits,
    no_build,
    spec_file="",
    target_id="",
):
    base_tmp = tempfile.mkdtemp(prefix="llm-greedy-verify-")
    worktree = os.path.join(base_tmp, "wt")
    out = {
        "apply_exit": None,
        "experiment_exit": None,
        "pass": False,
        "candidate_csv": None,
        "worktree": worktree,
        "logs": {},
    }
    rc_add, out_add = _run(["git", "worktree", "add", "--detach", worktree], cwd=root)
    out["logs"]["worktree_add"] = out_add[-4000:]
    if rc_add != 0:
        out["apply_exit"] = 2
        shutil.rmtree(base_tmp, ignore_errors=True)
        return 2, out
    try:
        # Worktree may include a copied build cache pointing to a different source path.
        # Remove it so compile.sh/cmake can reconfigure cleanly for this worktree.
        build_dir = os.path.join(worktree, "build")
        if os.path.isdir(build_dir):
            try:
                shutil.rmtree(build_dir)
            except Exception:
                pass
        os.makedirs(build_dir, exist_ok=True)

        if spec_file:
            spec_abs = spec_file if os.path.isabs(spec_file) else os.path.join(root, spec_file)
            if spec_abs.startswith(root + os.sep):
                spec_rel = os.path.relpath(spec_abs, root)
                spec_in_worktree = os.path.join(worktree, spec_rel)
            else:
                spec_in_worktree = spec_abs
            resolved_target = target_id or driver_id
            rc_base, out_base, rpt_base, _sum_base = _run_onboard_apply(
                worktree=worktree,
                spec_file=spec_in_worktree,
                target_id=resolved_target,
                batch=batch,
                seed=seed,
                min_digits=min_digits,
                regen_baseline=True,
            )
            out["logs"]["onboard_baseline"] = out_base[-12000:]
            out["onboard_baseline_report"] = rpt_base
            if rc_base != 0:
                out["apply_exit"] = 2
                return 2, out

        for pth in patch_stack:
            with open(pth, encoding="utf-8", errors="replace") as pf:
                p = subprocess.run(
                    ["patch", "-p1", "--forward"],
                    cwd=worktree,
                    stdin=pf,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                )
            out["logs"]["patch_apply"] = (out["logs"].get("patch_apply", "") + (p.stdout or ""))[-12000:]
            if p.returncode != 0:
                out["apply_exit"] = p.returncode
                return 2, out

        if spec_file:
            resolved_target = target_id or driver_id
            rc_run, out_run, rpt_run, sum_run = _run_onboard_apply(
                worktree=worktree,
                spec_file=spec_in_worktree,
                target_id=resolved_target,
                batch=batch,
                seed=seed,
                min_digits=min_digits,
                regen_baseline=False,
            )
            out["onboard_apply_report"] = rpt_run
            out["logs"]["experiment"] = out_run[-20000:]
            out["experiment_exit"] = (0 if rc_run == 0 else 2)
            out["pass"] = rc_run == 0
            if sum_run:
                steps = {s.get("step"): s for s in sum_run.get("steps", [])}
                run_step = steps.get("run_candidate") or {}
                run_artifacts = run_step.get("artifacts") or {}
                out["candidate_csv"] = run_artifacts.get("run_csv")
                cmp_step = steps.get("smoke_compare") or {}
                cmp_details = cmp_step.get("details") or {}
                tail = cmp_details.get("log_tail") or ""
                if tail:
                    out["logs"]["experiment"] = (
                        (out["logs"].get("experiment", "") or "") + "\n" + tail
                    )[-20000:]
            return (0 if out["pass"] else 1), out

        # Compile inside the temp worktree first, then run experiment with --no-build.
        compile_script = os.path.join(worktree, "scripts", "compile.sh")
        rc_build, out_build = _run(["bash", compile_script, worktree], cwd=worktree)
        out["logs"]["build"] = out_build[-20000:]
        driver = _load_driver(root, driver_id)
        driver_rel = _exec_rel(driver, driver_id)
        driver_abs = os.path.join(worktree, driver_rel)
        if rc_build != 0 or (not os.path.isfile(driver_abs)):
            out["experiment_exit"] = 2
            if not os.path.isfile(driver_abs):
                out["logs"]["build"] = (
                    (out["logs"].get("build", "") or "")
                    + "\nerror: built driver missing after compile: {0}\n".format(driver_abs)
                )[-20000:]
            return 2, out

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_csv = os.path.join(
            worktree,
            "experiments",
            driver_id,
            "generated",
            "greedy_verify_{0}_{1}_{2}.csv".format(batch, seed, ts),
        )
        run_script = os.path.join(worktree, "scripts", "run_experiment.sh")
        cmd = [
            "bash",
            run_script,
            "--driver",
            driver_id,
            "-o",
            out_csv,
            "--batch",
            str(batch),
            "--seed",
            str(seed),
            "--min-digits",
            str(min_digits),
        ]
        if no_build:
            cmd.append("--no-build")
        else:
            cmd.append("--no-build")
        env = os.environ.copy()
        env["AGENTIC_MIXED_PRECISION_DEMO_ROOT"] = worktree
        rc_exp, out_exp = _run(cmd, cwd=worktree, env=env)
        out["experiment_exit"] = rc_exp
        out["pass"] = rc_exp == 0
        out["candidate_csv"] = out_csv
        out["logs"]["experiment"] = out_exp[-20000:]
        return (0 if rc_exp == 0 else (1 if rc_exp == 1 else 2)), out
    finally:
        _run(["git", "worktree", "remove", "--force", worktree], cwd=root)
        shutil.rmtree(base_tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="LLM greedy accumulation controller")
    ap.add_argument("--driver", default="ddilog")
    ap.add_argument("--spec-file", default="", help="Optional spec JSON for spec-mode greedy runs")
    ap.add_argument("--target-id", default="", help="Target id in spec file (defaults to --driver)")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--user", default=None)
    ap.add_argument("--model", default="claudeopus46")
    ap.add_argument("--fallback-model", default="gpt4turbo")
    ap.add_argument("--min-digits", type=float, default=10.0)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-iterations-per-candidate", type=int, default=2)
    ap.add_argument(
        "--max-propose-retries",
        type=int,
        default=2,
        help="Retries for proposer format/transport failures per variable (does not consume candidate iterations)",
    )
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--focus-vars", default="", help="Optional comma-separated subset/order")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    root = _repo_root()
    run_id = args.target_id or args.driver
    if args.spec_file:
        try:
            spec_target = _load_spec_target(root, args.spec_file, run_id)
        except Exception:
            spec_target = None
        if not spec_target:
            print("error: unknown spec target id {0}".format(run_id), file=sys.stderr)
            return 2
        all_symbols = _candidate_symbols_from_spec(spec_target)
        impl_rel = _impl_rel_from_spec(spec_target)
    else:
        driver = _load_driver(root, args.driver)
        if not driver:
            print("error: unknown driver id {0}".format(args.driver), file=sys.stderr)
            return 2
        all_symbols = _candidate_symbols(driver)
        impl_rel = _impl_rel(driver)

    if args.focus_vars.strip():
        requested = [x.strip() for x in args.focus_vars.split(",") if x.strip()]
        symbols = [s for s in requested if s in set(all_symbols)]
    else:
        symbols = list(all_symbols)

    out_dir = args.output_dir or os.path.join(root, "experiments", run_id, "generated")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    run_ts = time.strftime("%Y%m%d_%H%M%S")

    accepted_vars = []
    accepted_patch_paths = []
    rejected_vars = []
    trace = []

    for sym in symbols:
        feedback = (
            "Current accepted greedy set: {0}. Propose only incremental change for {1}.".format(
                ", ".join(accepted_vars) if accepted_vars else "(empty)", sym
            )
        )
        accepted_this = False
        semantic_it = 1
        propose_failures = 0
        while semantic_it <= max(1, args.max_iterations_per_candidate):
            patch_path = os.path.join(
                out_dir,
                "greedy_patch_{0}_{1}_{2}.patch".format(run_ts, sym, semantic_it),
            )
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
                "--focus-vars",
                sym,
                "--feedback",
                feedback,
                "--output",
                patch_path,
            ]
            if args.spec_file:
                propose_cmd += ["--spec-file", args.spec_file, "--target-id", run_id]
            if args.base_url:
                propose_cmd += ["--base-url", args.base_url]
            if args.user:
                propose_cmd += ["--user", args.user]

            rc_prop, out_prop = _run(propose_cmd, cwd=root)
            attempt = {
                "var": sym,
                "iteration": semantic_it,
                "propose_exit": rc_prop,
                "patch_file": patch_path,
                "policy_reject": None,
                "verify_exit": None,
                "pass": False,
                "min_precise_digits": None,
                "logs": {"propose": out_prop[-6000:], "verify": ""},
            }
            if rc_prop != 0:
                trace.append(attempt)
                propose_failures += 1
                feedback = (
                    "Proposal formatting failed (attempt {0}/{1}); return strict JSON only for {2}.".format(
                        propose_failures, max(1, args.max_propose_retries), sym
                    )
                )
                if propose_failures >= max(1, args.max_propose_retries):
                    break
                continue

            ok, reject = _validate_guided_patch(
                patch_path, impl_rel, sym, all_symbols, accepted_vars
            )
            if not ok:
                attempt["verify_exit"] = 2
                attempt["policy_reject"] = reject
                trace.append(attempt)
                feedback = "{0}. Retry with declaration-local change for {1} only.".format(reject, sym)
                semantic_it += 1
                continue

            rc_ver, ver = _verify_with_stack(
                root=root,
                driver_id=args.driver,
                patch_stack=accepted_patch_paths + [patch_path],
                batch=args.batch,
                seed=args.seed,
                min_digits=args.min_digits,
                no_build=args.no_build,
                spec_file=args.spec_file,
                target_id=run_id,
            )
            attempt["verify_exit"] = rc_ver
            attempt["pass"] = rc_ver == 0
            attempt["verify_summary"] = {
                "apply_exit": ver.get("apply_exit"),
                "experiment_exit": ver.get("experiment_exit"),
                "pass": ver.get("pass"),
                "candidate_csv": ver.get("candidate_csv"),
            }
            attempt["logs"]["verify"] = ((ver.get("logs") or {}).get("experiment") or "")[-6000:]
            attempt["min_precise_digits"] = _parse_min_precise_digits(
                ((ver.get("logs") or {}).get("experiment") or "")
            )
            trace.append(attempt)

            if rc_ver == 0:
                accepted_vars.append(sym)
                accepted_patch_paths.append(patch_path)
                accepted_this = True
                break

            feedback = (
                "Failed threshold {0} for {1}. "
                "Previous min_precise_digits={2}. "
                "Keep focused declaration-only patch.".format(
                    args.min_digits, sym, attempt["min_precise_digits"]
                )
            )
            semantic_it += 1
        if not accepted_this:
            rejected_vars.append(sym)

    result = {
        "driver": args.driver,
        "spec_file": args.spec_file or None,
        "target_id": args.target_id or None,
        "mode": "llm_greedy_accumulate",
        "min_digits_threshold": args.min_digits,
        "accepted_vars": accepted_vars,
        "rejected_vars": rejected_vars,
        "accepted_patch_paths": accepted_patch_paths,
        "count_accepted": len(accepted_vars),
        "count_rejected": len(rejected_vars),
        "trace": trace,
    }
    out_json = os.path.join(out_dir, "greedy_episode_{0}.json".format(run_ts))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"result_file": out_json, **result}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

