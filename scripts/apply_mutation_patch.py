#!/usr/bin/env python3
"""
Apply or revert a unified diff listed under mutation_candidates.patches_directory
for a driver in targets.json (paths like a/src/kokkosUtils.h; patch -p1 from repo root).

Usage:
  ./scripts/apply_mutation_patch.py apply <driver_id> <patch_id>
  ./scripts/apply_mutation_patch.py revert <driver_id> <patch_id>
"""

import argparse
import os
import subprocess
import sys

from targets_lib import mutation_patch_path, repo_root, require_driver


def main():
    ap = argparse.ArgumentParser(description="apply or revert a mutation patch from targets.json")
    ap.add_argument("action", choices=["apply", "revert"])
    ap.add_argument("driver_id", help="targets.json drivers[].id (e.g. ddilog)")
    ap.add_argument("patch_id", help="mutation id (e.g. T); looks for <patches_directory>/<id>.patch")
    args = ap.parse_args()

    root = repo_root()
    driver = require_driver(root, args.driver_id)
    path = mutation_patch_path(root, driver, args.patch_id)
    if not os.path.isfile(path):
        print("error: patch not found: {}".format(path), file=sys.stderr)
        return 2

    if args.action == "apply":
        cmd = ["patch", "-p1", "--forward"]
    else:
        cmd = ["patch", "-p1", "-R"]

    with open(path, encoding="utf-8", errors="replace") as f:
        p = subprocess.run(cmd, cwd=root, stdin=f, universal_newlines=True)
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())
