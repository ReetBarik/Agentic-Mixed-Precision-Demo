#!/usr/bin/env python3
"""Ask Argo for a unified diff proposal for a target driver."""

import argparse
import json
import os
import sys
import difflib

from llm_agent.argo_client import ArgoClient


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_targets(root):
    with open(os.path.join(root, "targets.json"), encoding="utf-8") as f:
        return json.load(f)


def _driver_entry(targets, driver_id):
    for d in targets.get("drivers", []):
        if d.get("id") == driver_id:
            return d
    return None


def _extract_patch_text(text):
    # Prefer explicit unified diff blocks if present.
    start = text.find("--- ")
    idx = text.find("\n+++ ")
    if start != -1 and idx != -1 and idx > start:
        return text[start:].strip() + "\n"
    # Accept full text fallback; caller can inspect/save manually.
    return text.strip() + "\n"


def _extract_json_object(text):
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(s[start : end + 1])
    except Exception:
        return None


def _validate_guided_json_object(obj):
    if not isinstance(obj, dict):
        return None
    if set(obj.keys()) != {"old_line", "new_line"}:
        return None
    old_line = obj.get("old_line")
    new_line = obj.get("new_line")
    if not isinstance(old_line, str) or not isinstance(new_line, str):
        return None
    if not old_line.strip() or not new_line.strip():
        return None
    return {"old_line": old_line, "new_line": new_line}


def _repair_guided_json(client, original_messages, raw_text):
    repair_messages = list(original_messages)
    repair_messages.append({"role": "assistant", "content": raw_text})
    repair_messages.append(
        {
            "role": "user",
            "content": (
                "Rewrite your previous response as valid JSON only.\n"
                "Return exactly one JSON object with exactly these keys:\n"
                '{"old_line": "...", "new_line": "..."}\n'
                "Do not include markdown or any extra text."
            ),
        }
    )
    try:
        resp = client.chat(
            messages=repair_messages,
            response_format={"type": "json_object"},
        )
    except Exception:
        return None
    text = client.extract_text(resp)
    return _validate_guided_json_object(_extract_json_object(text))


def _build_unified_diff_single_replace(rel_path, original_text, old_line, new_line):
    old_line = old_line.rstrip("\n")
    new_line = new_line.rstrip("\n")
    lines = original_text.splitlines()
    idx = -1
    for i, line in enumerate(lines):
        if line == old_line:
            idx = i
            break
    if idx < 0:
        return None, "old_line_not_found"
    new_lines = list(lines)
    new_lines[idx] = new_line
    diff = list(
        difflib.unified_diff(
            lines,
            new_lines,
            fromfile="a/" + rel_path,
            tofile="b/" + rel_path,
            n=0,
            lineterm="",
        )
    )
    return "\n".join(diff) + "\n", None


def _read_file(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _build_messages(args, driver, impl_rel, impl_src):
    mc = driver.get("mutation_candidates") or {}
    locals_list = [x.get("symbol") for x in mc.get("locals", []) if x.get("symbol")]
    patches_avail = mc.get("patches_available") or []
    input_domain = driver.get("input_domain") or {}

    strict_greedy_mode = bool(args.focus_vars.strip())
    if strict_greedy_mode:
        system_prompt = (
            "You are a precise C/C++ code transformation assistant operating in GREEDY STEP mode.\n"
            "Return ONLY JSON (no markdown, no explanation).\n"
            "JSON schema: {\"old_line\": \"exact existing line\", \"new_line\": \"replacement line\"}.\n"
            "Modify ONLY the implementation file and ONLY the focused semantic variable(s).\n"
            "Produce minimal edits similar to curated single-local mutation patches.\n"
            "In guided mode, prefer declaration-line-only edits for the focused variable(s).\n"
            "Avoid expression-wide rewrites or broad cast insertion outside focused declarations.\n"
            "Do not refactor, rename unrelated symbols, or change function signatures.\n"
        )
    else:
        system_prompt = (
            "You are a precise C/C++ code transformation assistant.\n"
            "Return ONLY a unified diff patch with headers and hunks.\n"
            "Do not include markdown fences or explanation.\n"
            "Patch must apply from repo root with: patch -p1 --forward < file.patch\n"
            "Use file paths exactly as a/{path} and b/{path}.\n"
            "Keep edits minimal and targeted to local variable precision changes.\n"
        )

    user_prompt = (
        "Task: Propose one patch for driver '{driver_id}'.\n"
        "Goal: increase float usage in the target function while trying to preserve precision.\n"
        "Numerical threshold to respect later: min precise digits >= {min_digits}.\n"
        "Output format: {output_mode}\n\n"
        "Driver summary: {summary}\n"
        "Input domain: {input_domain}\n"
        "Implementation file: {impl_rel}\n"
        "Known local symbols (if relevant): {locals_list}\n"
        "Existing patch ids: {patches_avail}\n\n"
        "Constraints:\n"
        "1) Touch only the implementation file.\n"
        "2) Keep patch small and deterministic.\n"
        "3) Do not change function signatures or unrelated logic.\n"
        "4) Prefer downcasting selected locals to float or float-temporary operations.\n"
        "5) Keep patch style close to curated mutation patches: single-local and minimal.\n\n"
        "{greedy_block}"
        "{guided_block}"
        "{feedback_block}"
        "Implementation file content follows:\n"
        "===== BEGIN FILE =====\n"
        "{impl_src}\n"
        "===== END FILE =====\n"
    ).format(
        driver_id=args.driver,
        min_digits=args.min_digits,
        summary=driver.get("summary", ""),
        input_domain=json.dumps(input_domain, indent=2),
        output_mode=(
            "JSON object with old_line/new_line for deterministic diff generation"
            if strict_greedy_mode
            else "unified diff ONLY"
        ),
        impl_rel=impl_rel,
        locals_list=", ".join(locals_list) if locals_list else "(none listed)",
        patches_avail=", ".join(patches_avail) if patches_avail else "(none listed)",
        greedy_block=(
            "GREEDY STEP MODE:\n"
            "- Focus semantic variable(s): {0}\n"
            "- Propose exactly one candidate mutation step for this focus.\n"
            "- Do not modify other semantic variables from the known list.\n"
            "- Restrict edits to declaration line(s) for the focused variable(s) whenever possible.\n"
            "- Do NOT introduce broad static_cast<float> rewrites across unrelated expressions.\n"
            "- If declaration-only edit is impossible, keep any extra edit to a minimum and explain via patch content only.\n\n".format(
                args.focus_vars
            )
            if strict_greedy_mode
            else ""
        ),
        guided_block=(
            "Guidance: prioritize changes around these variables only: {0}\n\n".format(
                args.focus_vars
            )
            if args.focus_vars
            else ""
        ),
        feedback_block=(
            "Previous attempt feedback (use to revise):\n{0}\n\n".format(args.feedback)
            if args.feedback
            else ""
        ),
        impl_src=impl_src,
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def main():
    ap = argparse.ArgumentParser(description="Generate a unified diff proposal using Argo")
    ap.add_argument("--driver", default="ddilog", help="targets.json drivers[].id")
    ap.add_argument("--min-digits", type=float, default=10.0)
    ap.add_argument("--base-url", default=None, help="Argo proxy base URL")
    ap.add_argument("--user", default=None, help="Argo username (default ARGO_USERNAME)")
    ap.add_argument("--model", default="claudeopus46")
    ap.add_argument("--fallback-model", default="gpt4turbo")
    ap.add_argument(
        "--focus-vars",
        default="",
        help="Comma-separated semantic vars to focus on (guided search helper)",
    )
    ap.add_argument(
        "--feedback",
        default="",
        help="Optional prior failure feedback text to inform the next proposal",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Optional patch output file path; default: experiments/<driver>/generated/proposed_patch_<ts>.patch",
    )
    ap.add_argument(
        "--raw-response",
        action="store_true",
        help="Print raw model text instead of extracted patch",
    )
    args = ap.parse_args()

    root = _repo_root()
    targets = _load_targets(root)
    driver = _driver_entry(targets, args.driver)
    if not driver:
        print("error: unknown driver id {!r}".format(args.driver), file=sys.stderr)
        return 2

    mc = driver.get("mutation_candidates") or {}
    impl_rel = mc.get("implementation_relative")
    if not impl_rel:
        print("error: driver has no mutation_candidates.implementation_relative", file=sys.stderr)
        return 2
    impl_abs = os.path.join(root, impl_rel)
    if not os.path.isfile(impl_abs):
        print("error: implementation file not found: {}".format(impl_abs), file=sys.stderr)
        return 2

    impl_src = _read_file(impl_abs)
    messages = _build_messages(args, driver, impl_rel, impl_src)

    client = ArgoClient(
        base_url=args.base_url,
        user=args.user,
        model=args.model,
        fallback_model=args.fallback_model,
    )
    strict_greedy_mode = bool(args.focus_vars.strip())
    response_format = {"type": "json_object"} if strict_greedy_mode else None
    resp = client.chat(messages=messages, response_format=response_format)
    text = client.extract_text(resp)
    if strict_greedy_mode and not args.raw_response:
        # Guided mode must obey exact JSON schema; repair once if needed.
        obj = _validate_guided_json_object(_extract_json_object(text))
        if obj is None:
            obj = _repair_guided_json(client, messages, text)
        if obj is None:
            print(
                "error: guided mode expected JSON with old_line/new_line; got:\n{0}".format(
                    text[:600]
                ),
                file=sys.stderr,
            )
            return 2
        out_text, err = _build_unified_diff_single_replace(
            impl_rel, impl_src, obj["old_line"], obj["new_line"]
        )
        if err:
            print(
                "error: could not build patch from guided JSON ({0})".format(err),
                file=sys.stderr,
            )
            return 2
    else:
        out_text = text if args.raw_response else _extract_patch_text(text)

    if args.output:
        out_path = args.output
    else:
        ts = __import__("time").strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(root, "experiments", args.driver, "generated")
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, "proposed_patch_{}.patch".format(ts))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out_text)

    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

