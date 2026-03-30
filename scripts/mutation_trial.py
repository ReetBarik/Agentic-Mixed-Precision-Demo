"""Single-local mutation trial: apply → run_experiment → revert (shared by mutation_sweep)."""

import os
import subprocess
import sys
import time

from targets_lib import experiments_generated_dir, mutation_patch_path, repo_root


def trial_one_mutation(
    driver_id,
    driver,
    mid,
    batch,
    seed,
    min_digits,
    no_build=False,
):
    """
    Apply one patch, run compare experiment, always revert.

    Returns dict: id, apply_exit, compare_exit, csv, revert_exit, patch_path
    """
    root = repo_root()
    apply_py = os.path.join(root, "scripts", "apply_mutation_patch.py")
    run_exp = os.path.join(root, "scripts", "run_experiment.sh")

    env = os.environ.copy()
    env["AGENTIC_MIXED_PRECISION_DEMO_ROOT"] = root

    pfile = mutation_patch_path(root, driver, mid)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(
        experiments_generated_dir(root, driver_id),
        "{}_mutation_{}_{}_{}_{}.csv".format(driver_id, mid, batch, seed, ts),
    )

    def run_cmd(cmd):
        p = subprocess.run(cmd, cwd=root, env=env)
        return p.returncode

    rc_apply = run_cmd([sys.executable, apply_py, "apply", driver_id, mid])
    if rc_apply != 0:
        return {
            "id": mid,
            "apply_exit": rc_apply,
            "compare_exit": None,
            "csv": out_csv,
            "revert_exit": None,
            "patch_path": pfile,
        }

    rc_cmp = None
    try:
        cmd = [
            "bash",
            run_exp,
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
        rc_cmp = run_cmd(cmd)
    finally:
        rc_rev = run_cmd([sys.executable, apply_py, "revert", driver_id, mid])

    return {
        "id": mid,
        "apply_exit": 0,
        "compare_exit": rc_cmp,
        "csv": out_csv,
        "revert_exit": rc_rev,
        "patch_path": pfile,
    }
