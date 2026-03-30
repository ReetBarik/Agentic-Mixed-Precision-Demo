"""Shared apply/revert stacks for mutation patches (used by combo_greedy)."""

import os
import subprocess
import sys

from targets_lib import ordered_subset


def run_capture(cmd, cwd, env):
    p = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    return p.returncode, p.stdout or ""


def apply_stack(root, apply_py, driver_id, driver, ids_set, env):
    """Apply patches in targets.json locals order. On failure, revert partial stack."""
    order = ordered_subset(driver, set(ids_set))
    for mid in order:
        rc, out = run_capture(
            [sys.executable, apply_py, "apply", driver_id, mid], cwd=root, env=env
        )
        if rc != 0:
            for m2 in reversed(order[: order.index(mid)]):
                subprocess.run(
                    [sys.executable, apply_py, "revert", driver_id, m2],
                    cwd=root,
                    env=env,
                )
            return False, order[: order.index(mid)], out
    return True, order, ""


def revert_stack(root, apply_py, driver_id, order, env):
    for mid in reversed(order):
        subprocess.run(
            [sys.executable, apply_py, "revert", driver_id, mid], cwd=root, env=env
        )


def apply_py_path(root):
    return os.path.join(root, "scripts", "apply_mutation_patch.py")
