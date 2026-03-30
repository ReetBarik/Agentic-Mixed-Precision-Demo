"""
Shared helpers for reading targets.json (drivers, mutation_candidates, paths).
Used by mutation scripts and orchestration; keep JSON as the single catalog.
"""

import json
import os


def repo_root():
    env = os.environ.get("AGENTIC_MIXED_PRECISION_DEMO_ROOT")
    if env:
        return env
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_targets(root=None):
    root = root or repo_root()
    path = os.path.join(root, "targets.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_driver(data, driver_id):
    for d in data.get("drivers", []):
        if d.get("id") == driver_id:
            return d
    return None


def require_driver(root, driver_id):
    data = load_targets(root)
    d = get_driver(data, driver_id)
    if not d:
        raise SystemExit("error: unknown driver id {!r} in targets.json".format(driver_id))
    return d


def driver_executable_path(root, driver):
    rel = driver.get("executable_relative") or ""
    if not rel:
        raise SystemExit("error: driver missing executable_relative in targets.json")
    return os.path.join(root, rel)


def baseline_csv_path(root, driver_id, batch, seed):
    return os.path.join(
        root, "baselines", driver_id, "{}_baseline_{}_{}.csv".format(driver_id, batch, seed)
    )


def experiments_generated_dir(root, driver_id):
    return os.path.join(root, "experiments", driver_id, "generated")


def mutation_patches_directory(root, driver):
    mc = driver.get("mutation_candidates") or {}
    rel = mc.get("patches_directory")
    if not rel:
        raise SystemExit(
            "error: driver has no mutation_candidates.patches_directory in targets.json"
        )
    return os.path.join(root, rel)


def mutation_patch_path(root, driver, patch_id):
    return os.path.join(mutation_patches_directory(root, driver), patch_id + ".patch")


def mutation_local_ids_in_order(driver):
    mc = driver.get("mutation_candidates") or {}
    out = []
    for loc in mc.get("locals", []):
        i = loc.get("id")
        if i:
            out.append(i)
    return out


def ordered_subset(driver, ids_set):
    order = mutation_local_ids_in_order(driver)
    return [i for i in order if i in ids_set]
