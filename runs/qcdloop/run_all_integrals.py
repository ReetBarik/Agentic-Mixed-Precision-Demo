#!/usr/bin/env python3
"""Per-integral fan-out driver — Phase 1 of the caller-scoped pipeline.

Runs the existing Strategy -> Patcher -> Validator pipeline **once per integral**
against a report filtered to that integral, each in a fully isolated output tree,
and records a manifest + sizing summary per pass.  This is the qcdloop-specific
wiring layer around the generic ``agents.per_integral_orchestrator`` core; it
reuses ``run_strategy_e2e.py``'s Patcher/Validator/Strategy assembly verbatim.

Isolation (see agents/per_integral_orchestrator/orchestrator.py): every pass owns
``--out-dir/<integral>/`` — its filtered report, cloned headers tree, Strategy
``runs_root`` (so the Patcher build/logs/shims land per-pass), and manifest.  The
cmake ``-S`` app dir and the vanilla baseline headers are read-only inputs shared
across passes; the Validator builds in its own tempdir.  Concurrent passes
(``--workers N``) therefore never share a mutable path — each worker rebuilds the
pipeline from picklable config inside its own process (ProcessPoolExecutor, like
run_chunked.py), because the Patcher/Validator closures are not picklable.

Usage (under the venv + module env, proxy up):
    python runs/qcdloop/run_all_integrals.py \
        --report runs/qcdloop/report_5k.json \
        --integrals B1 --sample-count 5000 --tolerance 10 \
        --out-dir runs/qcdloop/per_integral_out
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from agents.per_integral_orchestrator import run_per_integral_pass  # noqa: E402
from runs.qcdloop.run_strategy_e2e import _build_headers_repo, _git  # noqa: E402

APP_CMAKE_DIR = HERE / "app"


# ---------------------------------------------------------------------------
# Worker — runs in its own process (ProcessPoolExecutor) or inline (workers=1).
# Everything it receives is a plain picklable dict; it rebuilds the (unpicklable)
# Patcher/Validator/Strategy pipeline itself.
# ---------------------------------------------------------------------------
def _run_one(task: dict) -> dict:
    """Run a single per-integral pass; return a small result record."""
    # imports live here so the module imports cheaply in the parent and each
    # worker process picks up the heavy deps independently.
    from agents.config import PipelineConfig, StrategyBudget, StrategyConfig
    from agents.patcher.agent import make_patcher_fn
    from agents.strategy import agent as strategy_agent
    from agents.validator import runner as _runner
    from agents.validator import tail as _tail
    from agents.validator.agent import make_validator_fn

    from agents.validator import scorer as _scorer

    integral = task["integral"]
    out_dir = Path(task["out_dir"])
    starting_sha = task["starting_sha"]
    kokkos_root = task["kokkos_root"]
    # Phase 2b: the scorer appends one measured (region_id, rung) cell per validated
    # candidate here during the pass; the un-measured (patcher_failed / build_failed /
    # wire_failed) cells are folded in from the iteration log after the pass.
    scored_manifest = out_dir / f"scored_{integral}.jsonl"

    def build_gate(tree: Path) -> None:
        """Fail-fast: the fresh clone must build vanilla before any Patcher work."""
        _runner.build_driver(tree, "vanilla", out_dir / "vanilla_gate_build",
                             Path(kokkos_root))

    def pipeline_fn(filtered_report: Path, tree: Path, pass_out: Path) -> dict:
        snapshot = {"seed": task["seed"], "sample_count": task["sample_count"]}
        budget = StrategyBudget(
            max_iters=task["max_iters"],
            max_iters_correctness=task["max_iters_correctness"],
            max_iters_speedup=task["max_iters_speedup"],
            max_wall_clock_sec=task["max_wall_hours"] * 3600.0,
        )
        strategy_config = StrategyConfig(
            tolerance=task["tolerance"],
            budget=budget,
            snapshot=snapshot,
            runs_root=pass_out,          # -> pass_out/strategy/<run_id>/ (isolated)
            **({"diminishing_returns_k": task["dr_k"]}
               if task["dr_k"] is not None else {}),
        )
        build_config = {"app_cmake_dir": str(APP_CMAKE_DIR),
                        "kokkos_root": kokkos_root}
        # Phase 2a: enable call-graph fan-out for regional intents (variants instead
        # of dead type-specialization shims).  The call graph is rooted at the
        # integral entry point and built once per pass against the cloned tree.
        fanout = None
        if task.get("fanout"):
            from agents.patcher.fanout import FanoutSettings, clear_graph_cache
            clear_graph_cache()          # fresh graph per pass (per-process reuse only)
            fanout = FanoutSettings(
                entry_point=task["entry_point"], integral=integral,
                max_paths=task.get("fanout_max_paths", 1024),
                # Phase 2d: the driver dir carries the entry-template instantiation
                # (run_app<Kokkos::complex<double>, double, double, ...>) that binds
                # TOutput→complex / TMass,TScale→double, so the boundary transform can
                # promote complex operands to the extended complex container.
                app_source_roots=[str(HERE / "src")])
        patcher_fn = make_patcher_fn(build_config=build_config,
                                     config=PipelineConfig(), fanout=fanout)
        tail_samples = _tail.load_tail_samples(filtered_report)
        base_state = {
            "vanilla_headers": task["vanilla_headers"],
            "dd_source_repo": task["dd_repo"],
            "dd_ref": task["dd_ref"],
            "accepted_patches": [],
            "kokkos_root": kokkos_root,
            "tail_samples": tail_samples,
        }
        validator_fn = make_validator_fn(
            base_state, starting_sha, str(tree), tolerance=task["tolerance"],
            scorer_manifest_path=str(scored_manifest), iteration_id=0,
            baseline_spec=_scorer.qcdloop_baseline_spec())
        state = {
            "characterization_report_path": str(filtered_report),
            "strategy_repo_path": str(tree),
            "strategy_starting_sha": starting_sha,
            "patcher_fn": patcher_fn,
            "validator_fn": validator_fn,
            "strategy_config": strategy_config,
        }
        delta = strategy_agent.run(state)
        return delta["strategy_result"]

    t0 = time.monotonic()
    try:
        manifest = run_per_integral_pass(
            integral, task["report_path"], task["base_repo_path"], out_dir,
            pipeline_fn=pipeline_fn, build_gate_fn=build_gate)
    except Exception as exc:  # noqa: BLE001 - surface any worker failure to main
        return {"integral": integral, "ok": False,
                "err": repr(exc), "wall_sec": time.monotonic() - t0}

    # Phase 2b: assemble the full scorer manifest (measured cells + the un-measured
    # codegen/build/wire cells recovered from the iteration log) and record its path.
    scorer_manifest_path = out_dir / f"manifest_scorer_{integral}.jsonl"
    iter_log = (manifest.get("artifacts") or {}).get("iteration_log_path")
    scorer_rows = _scorer.assemble_manifest(
        scored_manifest, iter_log, scorer_manifest_path)
    _annotate_manifest(manifest, scorer_manifest_path, scorer_rows)

    disk_bytes = _dir_size(out_dir)
    return {
        "integral": integral,
        "ok": True,
        "status": manifest.get("status"),
        "counts": manifest.get("counts"),
        "iterations": manifest.get("iterations"),
        "wall_sec": manifest.get("timing", {}).get("wall_sec"),
        "disk_bytes": disk_bytes,
        "manifest_path": manifest.get("_manifest_path"),
        "scorer_manifest_path": str(scorer_manifest_path),
        "scorer_cells": _scorer_cell_summary(scorer_rows),
    }


def _annotate_manifest(manifest: dict, scorer_manifest_path: Path,
                       scorer_rows: list) -> None:
    """Record the scorer-manifest path + cell tally into the pass manifest JSON."""
    manifest["scorer_manifest_path"] = str(scorer_manifest_path)
    manifest["scorer_cells"] = _scorer_cell_summary(scorer_rows)
    mpath = manifest.get("_manifest_path")
    if mpath and Path(mpath).is_file():
        persisted = {k: v for k, v in manifest.items() if k != "_manifest_path"}
        Path(mpath).write_text(json.dumps(persisted, indent=2))


def _scorer_cell_summary(scorer_rows: list) -> dict:
    """Small status tally over the assembled scorer rows (for the run summary)."""
    from agents.validator.scorer import STATUS_MEASURED
    by_status: dict[str, int] = {}
    measured_deltas = 0
    for r in scorer_rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        if r["status"] == STATUS_MEASURED and r.get("delta_effective") is not None:
            measured_deltas += 1
    return {"total": len(scorer_rows), "by_status": by_status,
            "measured_with_delta": measured_deltas}


def _dir_size(path: Path) -> int:
    total = 0
    for p in Path(path).rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _integral_names(report_path: Path) -> list[str]:
    data = json.loads(report_path.read_text())
    return sorted(data.get("integrals", {}))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default=str(HERE / "report_5k.json"))
    ap.add_argument("--integrals", nargs="*", default=None,
                    help="Integral names to run (default: all in the report).")
    ap.add_argument("--out-dir", default=str(HERE / "per_integral_out"))
    ap.add_argument("--workers", type=int, default=1,
                    help="Concurrent per-integral passes (default 1 = sequential).")
    ap.add_argument("--sample-count", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tolerance", type=float, default=10.0)
    ap.add_argument("--base-repo", default=str(Path.home() / "amp_per_integral_base_repo"))
    ap.add_argument("--dd-repo", default=str(Path.home() / "qcdloop"))
    ap.add_argument("--dd-ref", default="ddfun_enabled")
    ap.add_argument("--kokkos-root", default=str(Path.home() / "kokkos-install"))
    ap.add_argument("--max-iters", type=int, default=8)
    ap.add_argument("--max-iters-correctness", type=int, default=None)
    ap.add_argument("--max-iters-speedup", type=int, default=None)
    ap.add_argument("--max-wall-hours", type=float, default=4.0)
    ap.add_argument("--dr-k", type=int, default=None)
    ap.add_argument("--fanout", action="store_true",
                    help="Phase 2a: realize regional intents as per-caller-path "
                         "function variants (call-graph fan-out) instead of shims.")
    ap.add_argument("--entry-point", default="BO",
                    help="Call-graph root for fan-out (qcdloop integral entry point).")
    ap.add_argument("--fanout-max-paths", type=int, default=1024,
                    help="Cap on caller-paths enumerated per intent (over-generation "
                         "bound; a hit is logged, not silent).")
    ap.add_argument("--clean", action="store_true",
                    help="Remove --out-dir before running.")
    args = ap.parse_args(argv)

    report = Path(args.report).resolve()
    if not report.is_file():
        raise SystemExit(f"report not found: {report}")
    integrals = args.integrals or _integral_names(report)

    out_root = Path(args.out_dir).resolve()
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Validator baseline must be pristine == the base headers repo's base commit.
    vanilla_headers = REPO / "runs" / "qcdloop_headers_full"
    dirty = _git(REPO, "status", "--porcelain", str(vanilla_headers))
    if dirty:
        raise SystemExit(
            f"main tree {vanilla_headers} is dirty; refusing to run "
            f"(validator baseline must equal the base-repo base):\n{dirty}")

    # Build the flat base headers repo ONCE; all passes clone it (common
    # starting_sha, since git clone preserves the base commit SHA).
    base_repo = Path(args.base_repo).resolve()
    starting_sha = _build_headers_repo(base_repo, vanilla_headers)

    print("=== per-integral fan-out config ===", flush=True)
    print(f"  report        : {report}", flush=True)
    print(f"  integrals     : {integrals} ({len(integrals)})", flush=True)
    print(f"  out_dir       : {out_root}", flush=True)
    print(f"  workers       : {args.workers}", flush=True)
    print(f"  base_repo     : {base_repo}", flush=True)
    print(f"  starting_sha  : {starting_sha}", flush=True)
    print(f"  vanilla       : {vanilla_headers}", flush=True)
    print(f"  dd_repo@ref   : {args.dd_repo}@{args.dd_ref}", flush=True)
    print(f"  tolerance     : {args.tolerance}", flush=True)
    print(f"  fanout        : {args.fanout} (entry={args.entry_point}, "
          f"max_paths={args.fanout_max_paths})", flush=True)
    print("===================================", flush=True)

    tasks = [{
        "integral": integral,
        "report_path": str(report),
        "base_repo_path": str(base_repo),
        "starting_sha": starting_sha,
        "out_dir": str(out_root / integral),
        "vanilla_headers": str(vanilla_headers),
        "dd_repo": args.dd_repo,
        "dd_ref": args.dd_ref,
        "kokkos_root": args.kokkos_root,
        "tolerance": args.tolerance,
        "sample_count": args.sample_count,
        "seed": args.seed,
        "max_iters": args.max_iters,
        "max_iters_correctness": args.max_iters_correctness,
        "max_iters_speedup": args.max_iters_speedup,
        "max_wall_hours": args.max_wall_hours,
        "dr_k": args.dr_k,
        "fanout": args.fanout,
        "entry_point": args.entry_point,
        "fanout_max_paths": args.fanout_max_paths,
    } for integral in integrals]

    t0 = time.monotonic()
    results: list[dict] = []
    if args.workers <= 1:
        for t in tasks:
            print(f"-> pass {t['integral']} ...", flush=True)
            res = _run_one(t)
            _print_pass(res)
            results.append(res)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run_one, t): t for t in tasks}
            for fut in as_completed(futs):
                res = fut.result()
                _print_pass(res)
                results.append(res)

    total_wall = time.monotonic() - t0
    summary = _sizing_summary(results, total_wall, args.workers, len(integrals))
    summary_path = out_root / "sizing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    _print_summary(summary, summary_path)
    return 0 if all(r.get("ok") for r in results) else 1


def _print_pass(res: dict) -> None:
    if not res.get("ok"):
        print(f"  {res['integral']}: FAILED ({res.get('err','')[:300]})", flush=True)
        return
    c = res.get("counts") or {}
    sc = res.get("scorer_cells") or {}
    print(f"  {res['integral']}: {res.get('status')} "
          f"accepts={c.get('accepted')} rejects={c.get('rejected')} "
          f"iters={res.get('iterations')} wall={res.get('wall_sec')}s "
          f"disk={(res.get('disk_bytes') or 0)/1e6:.1f}MB", flush=True)
    if sc:
        print(f"      scorer cells: {sc.get('total')} "
              f"({sc.get('by_status')}); manifest={res.get('scorer_manifest_path')}",
              flush=True)


def _sizing_summary(results, total_wall, workers, n_total) -> dict:
    ok = [r for r in results if r.get("ok")]
    walls = [r["wall_sec"] for r in ok if r.get("wall_sec")]
    disks = [r["disk_bytes"] for r in ok if r.get("disk_bytes")]
    avg_wall = sum(walls) / len(walls) if walls else 0.0
    avg_disk = sum(disks) / len(disks) if disks else 0.0
    return {
        "passes_ok": len(ok),
        "passes_failed": len(results) - len(ok),
        "total_wall_sec": round(total_wall, 1),
        "workers": workers,
        "avg_wall_sec_per_pass": round(avg_wall, 1),
        "avg_disk_bytes_per_pass": int(avg_disk),
        # Extrapolation to the full 21-integral run.
        "extrapolation_21": {
            "sequential_hours": round(avg_wall * 21 / 3600.0, 2),
            "parallel_4_workers_hours": round(avg_wall * 21 / 4 / 3600.0, 2),
            "parallel_8_workers_hours": round(avg_wall * 21 / 8 / 3600.0, 2),
            "disk_all_21_gb": round(avg_disk * 21 / 1e9, 2),
        },
        "per_pass": results,
    }


def _print_summary(summary: dict, path: Path) -> None:
    print("\n=== sizing summary ===", flush=True)
    print(f"  passes ok/failed : {summary['passes_ok']}/{summary['passes_failed']}",
          flush=True)
    print(f"  total wall       : {summary['total_wall_sec']}s "
          f"(workers={summary['workers']})", flush=True)
    print(f"  avg wall/pass    : {summary['avg_wall_sec_per_pass']}s", flush=True)
    print(f"  avg disk/pass    : {summary['avg_disk_bytes_per_pass']/1e6:.1f}MB",
          flush=True)
    ex = summary["extrapolation_21"]
    print(f"  21 seq           : ~{ex['sequential_hours']}h", flush=True)
    print(f"  21 @4 / @8       : ~{ex['parallel_4_workers_hours']}h / "
          f"~{ex['parallel_8_workers_hours']}h", flush=True)
    print(f"  21 disk          : ~{ex['disk_all_21_gb']}GB", flush=True)
    print(f"  wrote {path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
