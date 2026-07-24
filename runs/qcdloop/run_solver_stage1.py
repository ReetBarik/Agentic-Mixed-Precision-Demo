#!/usr/bin/env python3
"""Phase 2e Stage 1 — greedy mixed-precision solver on a SINGLE integral (B12).

Consumes the per-integral fan-out's assembled scorer manifest
(``manifest_scorer_<I>.jsonl``), ranks its measured DISCRIM ``(region, rung)``
cells float<ff<dd, and greedily layers each onto an accumulated source tree — build
+ whole-app validate + accept/revert — under a p100 >= 6.0 precise-digits gate
(``agents.solver``).  The accumulated tree at the end is the optimized source.

STOP: Stage 1 is B12-only by design (PLAN 2e §Scope).  Do NOT point this at all 21
integrals — that is Stage 2, gated on Reet reviewing Stage 1 output.

Run under the venv + module env with the proxy up, detached so it survives:
    tmux new-session -d -s solver1 \
        "runs/qcdloop/run_solver_stage1.py > runs/qcdloop/run_solver_stage1.log 2>&1"

Usage:
    python runs/qcdloop/run_solver_stage1.py \
        --integral B12 \
        --manifest runs/qcdloop/per_integral_out_2e_measure/B12/manifest_scorer_B12.jsonl \
        --report   runs/qcdloop/report_5k.json \
        --out-dir  runs/qcdloop/solver_stage1_B12
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from agents.config import PipelineConfig                              # noqa: E402
from agents.patcher.agent import make_patcher_fn                      # noqa: E402
from agents.patcher.fanout import FanoutSettings, clear_graph_cache   # noqa: E402
from agents.per_integral_orchestrator.filter_report import filter_report  # noqa: E402
from agents.per_integral_orchestrator.orchestrator import _git_clone  # noqa: E402
from agents.solver import (ApplyResult, ValidateResult, build_queue,  # noqa: E402
                           load_manifest_rows, solve)
from agents.solver.intent import intent_from_candidate, region_variables  # noqa: E402
from agents.solver.report import write_report                         # noqa: E402
from agents.strategy.agent import _new_run_id                         # noqa: E402
from agents.strategy.gitops import GitRepo                            # noqa: E402
from agents.validator import runner as _runner                        # noqa: E402
from agents.validator import scorer as _scorer                         # noqa: E402
from agents.validator import tail as _tail                            # noqa: E402
from agents.validator.agent import make_validator_fn                  # noqa: E402
from runs.qcdloop.run_strategy_e2e import _build_headers_repo, _git   # noqa: E402

APP_CMAKE_DIR = HERE / "app"
GATE = 6.0  # LOCKED (Reet 2026-07-24): p100 >= 6 precise decimal digits.


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--integral", default="B12")
    ap.add_argument("--manifest", default=str(
        HERE / "per_integral_out_2e_measure" / "B12" / "manifest_scorer_B12.jsonl"))
    ap.add_argument("--report", default=str(HERE / "report_5k.json"))
    ap.add_argument("--out-dir", default=str(HERE / "solver_stage1_B12"))
    ap.add_argument("--base-repo",
                    default=str(Path.home() / "amp_solver_stage1_base_repo"))
    ap.add_argument("--dd-repo", default=str(Path.home() / "qcdloop"))
    ap.add_argument("--dd-ref", default="ddfun_enabled")
    ap.add_argument("--kokkos-root", default=str(Path.home() / "kokkos-install"))
    ap.add_argument("--sample-count", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--entry-point", default="BO")
    ap.add_argument("--gate", type=float, default=GATE)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args(argv)

    integral = args.integral
    if integral != "B12":
        print(f"[solver] WARNING Stage 1 is B12-only by design; got {integral!r}. "
              f"Proceeding, but this is outside the reviewed scope.", file=sys.stderr)
    if args.gate != GATE:
        print(f"[solver] gate overridden to {args.gate} (locked default {GATE}).",
              file=sys.stderr)

    out_dir = Path(args.out_dir).resolve()
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = Path(args.report).resolve()
    manifest = Path(args.manifest).resolve()
    for p in (report, manifest):
        if not p.is_file():
            raise SystemExit(f"missing input: {p}")

    # -- pristine-baseline guard (same as run_all_integrals) --
    vanilla_headers = REPO / "runs" / "qcdloop_headers_full"
    dirty = _git(REPO, "status", "--porcelain", str(vanilla_headers))
    if dirty:
        raise SystemExit(f"main tree {vanilla_headers} is dirty; refusing to run:\n{dirty}")

    # -- base repo + isolated tree clone (starting_sha survives the clone) --
    base_repo = Path(args.base_repo).resolve()
    starting_sha = _build_headers_repo(base_repo, vanilla_headers)
    tree = out_dir / f"tree_{integral}"
    if tree.exists():
        shutil.rmtree(tree)
    _git_clone(base_repo, tree)

    # -- fail-fast: fresh clone must build vanilla before any solver work --
    _runner.build_driver(tree, "vanilla", out_dir / "vanilla_gate_build",
                         Path(args.kokkos_root))

    # -- filtered report (region-local vars for intent reconstruction + tail) --
    filtered = out_dir / f"report_{integral}.json"
    filter_report(report, integral, filtered)
    report_regions = json.loads(filtered.read_text())["integrals"][integral]["regions"]

    # -- queue from the measured manifest --
    rows = load_manifest_rows(str(manifest))
    qb = build_queue(rows)
    all_region_ids = {r["region_id"] for r in rows}

    print("=== solver Stage 1 config ===", flush=True)
    print(f"  integral      : {integral}", flush=True)
    print(f"  manifest      : {manifest}", flush=True)
    print(f"  tree          : {tree}", flush=True)
    print(f"  starting_sha  : {starting_sha}", flush=True)
    print(f"  gate          : p100 >= {args.gate} precise digits", flush=True)
    print(f"  queue         : {len(qb.queue)} DISCRIM candidates "
          f"({len(qb.inert)} inert excluded, {len(qb.non_measured)} non-measured)",
          flush=True)
    for c in qb.queue:
        print(f"      [{c.rank}] {c.region_id:20s} {c.rung:5s} "
              f"de={c.delta_effective:.3e} base={c.baseline_delta_effective:.3e}",
          flush=True)
    print("=============================", flush=True)

    # -- pipeline wiring (mirrors run_all_integrals._run_one) --
    run_id = _new_run_id(str(starting_sha) + str(manifest))
    branch = f"solver/{run_id}"
    run_dir = out_dir / "patcher_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    clear_graph_cache()
    fanout = FanoutSettings(entry_point=args.entry_point, integral=integral,
                            app_source_roots=[str(HERE / "src")])
    build_config = {"app_cmake_dir": str(APP_CMAKE_DIR),
                    "kokkos_root": args.kokkos_root}
    patcher_fn = make_patcher_fn(build_config=build_config,
                                 config=PipelineConfig(), fanout=fanout)
    snapshot = {"seed": args.seed, "sample_count": args.sample_count}
    base_state = {
        "vanilla_headers": str(vanilla_headers),
        "dd_source_repo": args.dd_repo, "dd_ref": args.dd_ref,
        "accepted_patches": [], "kokkos_root": args.kokkos_root,
        "tail_samples": _tail.load_tail_samples(filtered),
    }
    validator_fn = make_validator_fn(
        base_state, starting_sha, str(tree), tolerance=args.gate,
        baseline_spec=_scorer.qcdloop_baseline_spec())

    repo = GitRepo(tree)
    repo.create_branch(branch, starting_sha)

    _iter = {"i": 0}

    def apply_fn(cand, parent):
        i = _iter["i"]; _iter["i"] += 1
        clear_graph_cache()  # graph must reflect the accumulated tree
        variables = region_variables(report_regions, cand.region_id)
        intent = intent_from_candidate(cand, variables=variables,
                                       rationale_id=f"solver-{i:03d}")
        ctx = {"run_id": run_id, "branch": branch, "repo_path": str(tree),
               "parent_sha": parent, "run_dir": str(run_dir), "iter_id": i}
        t0 = time.monotonic()
        resp = patcher_fn(intent.to_patcher(), ctx)
        arts = resp.get("artifacts") or {}
        return ApplyResult(
            ok=(resp.get("status") == "ok"),
            candidate_sha=resp.get("candidate_sha"),
            patcher_status=resp.get("status"),
            gate_binary=arts.get("gate_binary"),
            gate_tree_hash=arts.get("gate_tree_hash"),
            error=resp.get("error"),
            wall_sec=round(time.monotonic() - t0, 1))

    _diag = {"baseline_hotspot": None}

    def validate_fn(candidate_sha, gate_binary, gate_tree_hash):
        ctx = {"run_id": run_id, "branch": branch, "repo_path": str(tree),
               "tolerance": args.gate, "snapshot": snapshot,
               "iter_id": _iter["i"], "gate_binary": gate_binary,
               "gate_tree_hash": gate_tree_hash}
        t0 = time.monotonic()
        v = validator_fn(candidate_sha, ctx)
        if _diag["baseline_hotspot"] is None:
            _diag["baseline_hotspot"] = (v.get("current") or {}).get("hotspot")
        return ValidateResult(
            cand_min=(v.get("candidate") or {}).get("min_precise_digits"),
            curr_min=(v.get("current") or {}).get("min_precise_digits"),
            combined_cand_min=v.get("cand_min_precise_digits"),
            verdict=v.get("verdict"), verdict_reason=v.get("verdict_reason"),
            wall_sec=round(time.monotonic() - t0, 1))

    def revert_fn(parent):
        repo.reset_hard(parent)

    def head_fn():
        return repo.head()

    def on_event(o):
        print(f"  [{o.candidate.rung:5s}] {o.candidate.region_id:20s} -> "
              f"{o.outcome:24s} min {o.min_before}->{o.min_after} "
              f"({o.reason}) [{o.wall_sec}s]", flush=True)

    t_solve = time.monotonic()
    res = solve(qb.queue, apply_fn=apply_fn, validate_fn=validate_fn,
                revert_fn=revert_fn, head_fn=head_fn, gate=args.gate,
                all_region_ids=all_region_ids, on_event=on_event)
    solve_wall = round(time.monotonic() - t_solve, 1)

    # -- persist artifacts: cumulative diff, merged tree pointer, JSON, report --
    diff_path = out_dir / "final.diff"
    repo.write_cumulative_diff(starting_sha, diff_path)
    result_json = out_dir / "solver_result.json"
    _write_result_json(result_json, res, qb, integral, str(tree), str(diff_path),
                       solve_wall, starting_sha, args)
    per_integral_floor = _per_integral_floor(args.seed, args.sample_count)
    md_path = out_dir / f"SOLVER_STAGE1_{integral}.md"
    write_report(md_path, res, qb, integral=integral, tree_path=str(tree),
                 diff_path=str(diff_path), manifest_path=str(manifest),
                 report_regions=report_regions, gate=args.gate,
                 solve_wall_sec=solve_wall, snapshot=snapshot,
                 per_integral_floor=per_integral_floor,
                 baseline_hotspot=_diag["baseline_hotspot"])

    print("\n=== solver Stage 1 summary ===", flush=True)
    print(f"  baseline p100 : {res.baseline_min}", flush=True)
    print(f"  final p100    : {res.final_min}", flush=True)
    print(f"  accepts       : {len(res.accepted)}  rejects: {len(res.rejected)}",
          flush=True)
    print(f"  precision dist: {res.precision_distribution()}", flush=True)
    print(f"  stopped       : {res.stopped or '(queue exhausted)'}", flush=True)
    print(f"  merged tree   : {tree}", flush=True)
    print(f"  wrote         : {md_path}", flush=True)
    print(f"  wrote         : {result_json}", flush=True)
    return 0


def _per_integral_floor(seed: int, sample_count: int) -> dict | None:
    """Per-integral worst-case min_precise_digits from the vanilla baseline scoring.

    Reads the Validator's persisted ``current_precise_digits.jsonl`` (the vanilla
    tree scoring, cached per DD-tree-hash + snapshot).  Returns
    ``{integral: worst_p100}`` or None if no matching scoring file is found.  Used
    only for the report's blocking-finding table; a missing file degrades the
    report gracefully (table omitted).
    """
    from agents.validator.validate import _VALIDATOR_ROOT
    root = Path(_VALIDATOR_ROOT)
    if not root.exists():
        return None
    cands = sorted(root.glob(f"*_seed{seed}_n{sample_count}/current_precise_digits.jsonl"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        return None
    floor: dict[str, float] = {}
    with open(cands[0]) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            integ = r.get("integral")
            for d in r.get("digits", []):
                if d is None:
                    continue
                if integ not in floor or d < floor[integ]:
                    floor[integ] = d
    return floor or None


def _write_result_json(path, res, qb, integral, tree, diff, wall, starting_sha,
                       args) -> None:
    payload = {
        "integral": integral, "gate": args.gate, "starting_sha": starting_sha,
        "tree_path": tree, "diff_path": diff, "final_head": res.final_head,
        "baseline_min_precise_digits": res.baseline_min,
        "final_min_precise_digits": res.final_min,
        "stopped": res.stopped, "stop_detail": res.stop_detail,
        "solve_wall_sec": wall,
        "precision_distribution": res.precision_distribution(),
        "queue_size": len(qb.queue), "inert_excluded": len(qb.inert),
        "non_measured": len(qb.non_measured),
        "region_final": res.region_final,
        "outcomes": [{
            "region_id": o.candidate.region_id, "rung": o.candidate.rung,
            "outcome": o.outcome, "min_before": o.min_before,
            "min_after": o.min_after, "combined_min_after": o.combined_min_after,
            "reason": o.reason, "patcher_status": o.patcher_status,
            "validator_verdict": o.validator_verdict,
            "candidate_sha": o.candidate_sha, "wall_sec": o.wall_sec,
        } for o in res.outcomes],
    }
    Path(path).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
