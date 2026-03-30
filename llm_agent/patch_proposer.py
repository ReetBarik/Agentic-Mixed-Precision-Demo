#!/usr/bin/env python3
"""Ask Argo for a unified diff proposal for a target driver."""

import argparse
import json
import os
import re
import sys

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


def _read_file(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _build_messages(args, driver, impl_rel, impl_src):
    mc = driver.get("mutation_candidates") or {}
    locals_list = [x.get("symbol") for x in mc.get("locals", []) if x.get("symbol")]
    patches_avail = mc.get("patches_available") or []
    input_domain = driver.get("input_domain") or {}

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
        "Output format: unified diff ONLY.\n\n"
        "Driver summary: {summary}\n"
        "Input domain: {input_domain}\n"
        "Implementation file: {impl_rel}\n"
        "Known local symbols (if relevant): {locals_list}\n"
        "Existing patch ids: {patches_avail}\n\n"
        "Constraints:\n"
        "1) Touch only the implementation file unless absolutely necessary.\n"
        "2) Keep patch small and deterministic.\n"
        "3) Do not change function signatures or unrelated logic.\n"
        "4) Prefer downcasting selected locals to float or float-temporary operations.\n\n"
        "Implementation file content follows:\n"
        "===== BEGIN FILE =====\n"
        "{impl_src}\n"
        "===== END FILE =====\n"
    ).format(
        driver_id=args.driver,
        min_digits=args.min_digits,
        summary=driver.get("summary", ""),
        input_domain=json.dumps(input_domain, indent=2),
        impl_rel=impl_rel,
        locals_list=", ".join(locals_list) if locals_list else "(none listed)",
        patches_avail=", ".join(patches_avail) if patches_avail else "(none listed)",
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
    resp = client.chat(messages=messages)
    text = client.extract_text(resp)
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

