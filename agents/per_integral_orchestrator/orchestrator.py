"""``run_per_integral_pass`` — one isolated Strategy pass for a single integral.

Orchestration only (Phase 1).  Given a consolidated report and a *base* headers
git repo, this:

1. filters the report to ``integral_name`` (:func:`filter_report`),
2. fresh-clones the base repo into an isolated per-pass tree,
3. optionally runs a fail-fast vanilla build gate on the fresh clone (so a broken
   clone fails immediately instead of masquerading as a Patcher failure),
4. runs the caller-supplied ``pipeline_fn`` (the real Strategy/Patcher/Validator
   wiring; a fake in tests) against the filtered report + cloned tree,
5. writes a manifest (:func:`build_manifest`).

Isolation contract
------------------
Every mutable artifact of a pass lives under its own ``out_dir``:

* ``out_dir/report_<integral>.json`` — the filtered report,
* ``out_dir/tree_<integral>/``       — the cloned headers tree (the ONLY thing the
  Patcher mutates; ``git clone`` preserves the base commit SHA so
  ``starting_sha`` is valid in the clone),
* ``out_dir/manifest_<integral>.json`` — the manifest,
* and the ``pipeline_fn`` must direct Strategy's ``runs_root`` (Patcher build/logs/
  shims) under ``out_dir`` too.

Because every pass gets a distinct ``out_dir``, concurrent passes never share a
mutable path.  The cmake ``-S`` source dir (``runs/qcdloop/app``) and the vanilla
baseline headers are read-only inputs (cmake writes only to its per-pass ``-B``
dir; the Validator copies the baseline into its own scratch), so they are safe to
share across parallel passes.  The app drivers emit no ``journal.jsonl``, so the
characterization driver's hardcoded-cwd journal is a non-issue here.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

from agents.per_integral_orchestrator.filter_report import filter_report
from agents.per_integral_orchestrator.manifest import build_manifest

# pipeline_fn(filtered_report_path, tree_path, out_dir) -> strategy_result dict
PipelineFn = Callable[[Path, Path, Path], dict]
# build_gate_fn(tree_path) -> None; raises on failure
BuildGateFn = Callable[[Path], None]


def _git_clone(base_repo_path: Path, dest: Path) -> None:
    """Fresh ``git clone`` of ``base_repo_path`` into ``dest`` (preserves SHAs).

    Uses ``--local`` so the clone is a fast hardlink copy of a local repo; the
    working tree is fully materialized (no ``--bare``) because the Patcher edits
    files at the tree root.  Persists a git identity so later Patcher commits
    succeed regardless of global config (mirrors ``run_strategy_e2e``).
    """
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["git", "clone", "--local", "--quiet",
         str(Path(base_repo_path).resolve()), str(dest)],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"git clone {base_repo_path} -> {dest} failed:\n{r.stderr}")
    for k, v in (("user.name", "amp-per-integral"), ("user.email", "amp@local")):
        subprocess.run(["git", "-C", str(dest), "config", k, v],
                       capture_output=True, text=True, check=False)


def run_per_integral_pass(
    integral_name: str,
    report_path: str | Path,
    base_repo_path: str | Path,
    out_dir: str | Path,
    *,
    pipeline_fn: PipelineFn,
    clone_fn: Callable[[Path, Path], None] = _git_clone,
    build_gate_fn: Optional[BuildGateFn] = None,
) -> dict:
    """Run one isolated per-integral Strategy pass; return the manifest dict.

    ``pipeline_fn`` receives ``(filtered_report_path, tree_path, out_dir)`` and
    must run Strategy (driving Patcher + Validator against ``tree_path``, writing
    its ``runs_root`` under ``out_dir``) and return the ``strategy_result`` bundle
    (``report_json_path`` / ``cumulative_diff_path`` / ``status`` / ...).  It is
    injected so this core is unit-testable without the LLM pipeline.

    ``build_gate_fn`` (optional) is a fail-fast vanilla build of the fresh clone,
    run before ``pipeline_fn``; a raise aborts the pass with a clear error.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filtered_report = out_dir / f"report_{integral_name}.json"
    filter_meta = filter_report(report_path, integral_name, filtered_report)

    tree = out_dir / f"tree_{integral_name}"
    clone_fn(Path(base_repo_path), tree)

    if build_gate_fn is not None:
        build_gate_fn(tree)

    t0 = time.monotonic()
    strategy_result = pipeline_fn(filtered_report, tree, out_dir)
    wall_sec = time.monotonic() - t0

    if not isinstance(strategy_result, dict):
        raise TypeError(
            f"pipeline_fn must return the strategy_result dict, got "
            f"{type(strategy_result).__name__}")

    manifest = build_manifest(
        integral_name, strategy_result, filter_meta,
        timing={"wall_sec": round(wall_sec, 3)}, tree_path=str(tree))

    manifest_path = out_dir / f"manifest_{integral_name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest["_manifest_path"] = str(manifest_path)
    return manifest
