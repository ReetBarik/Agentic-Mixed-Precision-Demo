#!/usr/bin/env python3
"""One-shot LLM episode: propose patch, verify patch, print verdict JSON."""

import argparse
import json
import os
import re
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


def _parse_min_precise_digits(text):
    m = re.search(r"min_precise_digits=([0-9.eE+-]+)", text or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _load_targets_locals(root, driver_id):
    try:
        with open(os.path.join(root, "targets.json"), encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    for d in data.get("drivers", []):
        if d.get("id") == driver_id:
            mc = d.get("mutation_candidates") or {}
            out = []
            for loc in mc.get("locals", []):
                sym = loc.get("symbol")
                if sym:
                    out.append(sym)
            return out
    return []


def _load_spec_target(root, spec_file, target_id):
    path = spec_file if os.path.isabs(spec_file) else os.path.join(root, spec_file)
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("targets"), list):
        for t in payload.get("targets", []):
            if t.get("id") == target_id:
                return t
    if isinstance(payload, dict) and payload.get("id") == target_id:
        return payload
    return None


def _load_spec_locals(root, spec_file, target_id):
    tgt = _load_spec_target(root, spec_file, target_id)
    if not tgt:
        return []
    out = []
    for sym in tgt.get("locals_for_downcast", []) or []:
        if isinstance(sym, str) and sym.strip():
            out.append(sym.strip())
    return out


def _load_impl_rel(root, driver_id):
    try:
        with open(os.path.join(root, "targets.json"), encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return "src/kokkosUtils.h"
    for d in data.get("drivers", []):
        if d.get("id") == driver_id:
            mc = d.get("mutation_candidates") or {}
            return mc.get("implementation_relative") or "src/kokkosUtils.h"
    return "src/kokkosUtils.h"


def _load_impl_rel_from_spec(root, spec_file, target_id):
    tgt = _load_spec_target(root, spec_file, target_id)
    if not tgt:
        return "src/kokkosUtils.h"
    return tgt.get("header_path") or "src/kokkosUtils.h"


def _validate_guided_patch(patch_path, impl_rel, focus_var, known_symbols):
    """Policy checks for guided mode before expensive verify."""
    if not os.path.isfile(patch_path):
        return False, "policy_reject: patch file missing"

    touched_files = []
    added_removed = 0
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
            lead = line[0]
            if lead not in ("+", "-"):
                continue
            added_removed += 1
            body = line[1:]
            if focus_var and re.search(r"\b{0}\b".format(re.escape(focus_var)), body):
                focus_seen = True
            for sym in known_symbols:
                if re.search(r"\b{0}\b".format(re.escape(sym)), body):
                    touched_symbols.add(sym)

    # Exactly one file, and it must be the implementation file.
    uniq_files = sorted(set(touched_files))
    if len(uniq_files) != 1 or uniq_files[0] != impl_rel:
        return (
            False,
            "policy_reject: patch must touch only {0}, got {1}".format(
                impl_rel, uniq_files
            ),
        )

    # Guided mode: keep patch small.
    if added_removed > 16:
        return (
            False,
            "policy_reject: patch too large for guided mode ({0} +/- lines)".format(
                added_removed
            ),
        )

    # Focus variable should appear in the changed lines.
    if focus_var and not focus_seen:
        return (
            False,
            "policy_reject: focus variable {0} not present in changed lines".format(
                focus_var
            ),
        )

    # Disallow touching many semantic locals in guided mode.
    if focus_var and len(touched_symbols) > 2:
        return (
            False,
            "policy_reject: touched too many semantic variables ({0})".format(
                ", ".join(sorted(touched_symbols))
            ),
        )

    return True, None


def _summarize_patch(patch_path):
    if not os.path.isfile(patch_path):
        return {
            "explanation": "Proposal generated, but patch file could not be read.",
            "details": [],
        }

    file_path = None
    adds = 0
    dels = 0
    details = []
    seen = set()
    semantic_targets = []
    cast_added = False

    # Detect patterns like:
    # + const float H_f = ...
    # + const TMass H = TMass(H_f);
    # and direct replacements:
    # - const TMass H = ...
    # + const float H = ...
    float_tmp_decl_re = re.compile(
        r"^\+\s*(?:const\s+)?float\s+([A-Za-z_][A-Za-z0-9_]*)\b"
    )
    tdecl_old_re = re.compile(
        r"^-\s*(?:const\s+)?TMass\s+([A-Za-z_][A-Za-z0-9_]*)\b"
    )
    fdecl_new_re = re.compile(
        r"^\+\s*(?:const\s+)?float\s+([A-Za-z_][A-Za-z0-9_]*)\b"
    )
    cast_back_re = re.compile(
        r"^\+\s*(?:const\s+)?TMass\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*TMass\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*;"
    )
    float_tmp_names = set()
    old_tmass_decl_names = set()

    with open(patch_path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("+++ b/"):
                file_path = line[len("+++ b/") :]
                continue

            if not line:
                continue

            lead = line[0]
            if lead == "+" and not line.startswith("+++"):
                adds += 1
                content = line[1:].strip()
                tmp_decl = float_tmp_decl_re.match(line)
                if tmp_decl:
                    float_tmp_names.add(tmp_decl.group(1))
                cast_back = cast_back_re.match(line)
                if cast_back:
                    semantic_name = cast_back.group(1)
                    tmp_name = cast_back.group(2)
                    if tmp_name in float_tmp_names and semantic_name not in semantic_targets:
                        semantic_targets.append(semantic_name)
                if "static_cast<float>" in content:
                    cast_added = True
            elif lead == "-":
                dels += 1
                old_decl = tdecl_old_re.match(line)
                if old_decl:
                    old_tmass_decl_names.add(old_decl.group(1))

    # Direct TMass -> float declaration replacements.
    with open(patch_path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            if not line.startswith("+") or line.startswith("+++"):
                continue
            new_decl = fdecl_new_re.match(line)
            if not new_decl:
                continue
            var_name = new_decl.group(1)
            if var_name in old_tmass_decl_names and var_name not in semantic_targets:
                semantic_targets.append(var_name)

    if semantic_targets:
        target_text = ", ".join(semantic_targets)
        msg = "Change datatype computation pathway to float for variables {0} in {1}".format(
            target_text, file_path or "target source"
        )
        details.append(msg)
        explanation = "Proposal: " + msg
    elif cast_added:
        msg = "Introduce float casts in {0}".format(file_path or "target source")
        details.append(msg)
        explanation = "Proposal: " + msg
    else:
        target = file_path or "target source"
        explanation = (
            "Proposal: patch updates {0} ({1} additions, {2} removals).".format(
                target, adds, dels
            )
        )
    return {"explanation": explanation, "details": details}


def main():
    ap = argparse.ArgumentParser(description="Run one propose+verify LLM patch episode")
    ap.add_argument("--driver", default="ddilog")
    ap.add_argument("--spec-file", default="", help="Optional spec JSON for spec-mode episodes")
    ap.add_argument("--target-id", default="", help="Target id in spec file (defaults to --driver)")
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
        "--max-iterations",
        type=int,
        default=1,
        help="Retry loop count for LLM propose->verify (default 1)",
    )
    ap.add_argument(
        "--guided-search",
        action="store_true",
        help="Try one semantic variable focus at a time from targets.json locals",
    )
    ap.add_argument(
        "--focus-vars",
        default="",
        help="Optional comma-separated focus vars; overrides auto locals in guided mode",
    )
    ap.add_argument(
        "--hybrid-curated-first",
        action="store_true",
        help="Run mutation_combo_greedy first and persist its result before LLM retries",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Optional output dir for patch/report (default experiments/<driver>/generated)",
    )
    args = ap.parse_args()

    root = _repo_root()
    episode_id = args.target_id or args.driver
    out_dir = args.output_dir or os.path.join(root, "experiments", episode_id, "generated")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    result = {
        "driver": args.driver,
        "spec_file": args.spec_file or None,
        "target_id": (args.target_id or None),
        "propose_exit": None,
        "verify_exit": None,
        "pass": False,
        "patch_file": None,
        "proposal": None,
        "verify_report": None,
        "best_so_far": None,
        "attempts": [],
        "logs": {
            "propose": "",
            "verify": "",
        },
    }
    focus_vars = [x.strip() for x in args.focus_vars.split(",") if x.strip()]
    if args.spec_file:
        if args.guided_search and not focus_vars:
            focus_vars = _load_spec_locals(root, args.spec_file, episode_id)
        known_symbols = _load_spec_locals(root, args.spec_file, episode_id)
        impl_rel = _load_impl_rel_from_spec(root, args.spec_file, episode_id)
    else:
        if args.guided_search and not focus_vars:
            focus_vars = _load_targets_locals(root, args.driver)
        known_symbols = _load_targets_locals(root, args.driver)
        impl_rel = _load_impl_rel(root, args.driver)
    if not focus_vars:
        focus_vars = [""]

    # Hybrid curated-first baseline run.
    if args.hybrid_curated_first:
        hybrid_cmd = [
            sys.executable,
            os.path.join(root, "scripts", "mutation_combo_greedy.py"),
            "--driver",
            args.driver,
            "--batch",
            str(args.batch),
            "--seed",
            str(args.seed),
            "--min-digits",
            str(args.min_digits),
        ]
        rc_h, out_h = _run(hybrid_cmd, cwd=root)
        result["hybrid_curated_first"] = {
            "exit": rc_h,
            "log_tail": out_h[-8000:],
        }

    best_score = -1e30
    best_attempt = None
    feedback = ""
    stop_reason = "exhausted_iterations"
    attempt_idx = 0

    for fvar in focus_vars:
        for _ in range(max(1, args.max_iterations)):
            attempt_idx += 1
            attempt_ts = time.strftime("%Y%m%d_%H%M%S")
            focus_tag = fvar if fvar else "any"
            patch_path = os.path.join(
                out_dir, "episode_patch_{0}_{1}_{2}.patch".format(ts, attempt_idx, focus_tag)
            )
            report_path = os.path.join(
                out_dir, "episode_verify_{0}_{1}_{2}.json".format(ts, attempt_idx, focus_tag)
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
                "--output",
                patch_path,
            ]
            if args.spec_file:
                propose_cmd += ["--spec-file", args.spec_file, "--target-id", episode_id]
            if args.base_url:
                propose_cmd += ["--base-url", args.base_url]
            if args.user:
                propose_cmd += ["--user", args.user]
            if fvar:
                propose_cmd += ["--focus-vars", fvar]
            if feedback:
                propose_cmd += ["--feedback", feedback]

            rc_prop, out_prop = _run(propose_cmd, cwd=root)
            resolved_patch = patch_path
            if rc_prop == 0:
                maybe = _last_nonempty_line(out_prop)
                if maybe and os.path.isfile(maybe):
                    resolved_patch = maybe

            attempt = {
                "iteration": attempt_idx,
                "focus_var": fvar or None,
                "propose_exit": rc_prop,
                "patch_file": resolved_patch,
                "proposal": _summarize_patch(resolved_patch),
                "verify_exit": None,
                "pass": False,
                "policy_reject": None,
                "verify_report": report_path,
                "min_precise_digits": None,
                "logs": {"propose": out_prop[-8000:], "verify": ""},
            }
            if rc_prop != 0:
                result["attempts"].append(attempt)
                feedback = "Patch proposal failed. Propose a smaller deterministic patch."
                continue

            if args.guided_search and fvar:
                ok, reject_reason = _validate_guided_patch(
                    resolved_patch,
                    impl_rel,
                    fvar,
                    known_symbols,
                )
                if not ok:
                    attempt["verify_exit"] = 2
                    attempt["policy_reject"] = reject_reason
                    attempt["verify_summary"] = {
                        "apply_exit": None,
                        "experiment_exit": None,
                        "pass": False,
                        "candidate_csv": None,
                    }
                    result["attempts"].append(attempt)
                    feedback = (
                        "{0}. Keep only focused variable {1} with minimal declaration-local edits."
                    ).format(reject_reason, fvar)
                    continue

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
            if args.spec_file:
                verify_cmd += ["--spec-file", args.spec_file, "--target-id", episode_id]
            if args.no_build:
                verify_cmd.append("--no-build")
            if args.keep_worktree:
                verify_cmd.append("--keep-worktree")

            rc_ver, out_ver = _run(verify_cmd, cwd=root)
            attempt["verify_exit"] = rc_ver
            attempt["pass"] = rc_ver == 0
            attempt["logs"]["verify"] = out_ver[-8000:]

            if os.path.isfile(report_path):
                try:
                    with open(report_path, encoding="utf-8") as f:
                        vr = json.load(f)
                    exp_log = ((vr.get("logs") or {}).get("experiment") or "")
                    score = _parse_min_precise_digits(exp_log)
                    attempt["min_precise_digits"] = score
                    attempt["verify_summary"] = {
                        "apply_exit": vr.get("apply_exit"),
                        "experiment_exit": vr.get("experiment_exit"),
                        "pass": vr.get("pass"),
                        "candidate_csv": vr.get("candidate_csv"),
                    }
                except Exception:
                    score = None
            else:
                score = None

            result["attempts"].append(attempt)

            score_for_rank = score if score is not None else (-1e20 if rc_ver != 0 else 0.0)
            if score_for_rank > best_score:
                best_score = score_for_rank
                best_attempt = attempt

            if rc_ver == 0:
                stop_reason = "passed_threshold"
                break

            feedback = (
                "Previous patch failed threshold min_digits>={0}. "
                "Try a smaller or different float change. "
                "Latest summary: {1}".format(
                    args.min_digits, attempt["proposal"]["explanation"]
                )
            )
        if stop_reason == "passed_threshold":
            break

    if best_attempt:
        result["propose_exit"] = best_attempt.get("propose_exit")
        result["verify_exit"] = best_attempt.get("verify_exit")
        result["pass"] = bool(best_attempt.get("pass"))
        result["patch_file"] = best_attempt.get("patch_file")
        result["proposal"] = best_attempt.get("proposal")
        result["verify_report"] = best_attempt.get("verify_report")
        result["best_so_far"] = {
            "iteration": best_attempt.get("iteration"),
            "focus_var": best_attempt.get("focus_var"),
            "min_precise_digits": best_attempt.get("min_precise_digits"),
            "pass": best_attempt.get("pass"),
        }
        result["verify_summary"] = best_attempt.get("verify_summary")
        result["logs"]["propose"] = best_attempt.get("logs", {}).get("propose", "")
        result["logs"]["verify"] = best_attempt.get("logs", {}).get("verify", "")
    result["stop_reason"] = stop_reason

    print(json.dumps(result, indent=2))
    if result["pass"]:
        return 0
    if result.get("verify_exit") == 1:
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())

