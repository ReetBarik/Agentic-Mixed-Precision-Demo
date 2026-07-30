#!/usr/bin/env python3
"""First real end-to-end Strategy run on a characterization report.

Skips the characterize graph node (Q5 seam): assembles a PipelineState with a
fixed report path + real Patcher/Validator adapters and calls
``agents.strategy.agent.run`` directly.

Isolation model
---------------
The Patcher mutates its ``repo_root`` (creates ``strategy/<run_id>``, commits
header edits, resets rejected candidates) and the regional integrator assumes a
*flat* tree whose root IS ``QL_HEADERS`` (it drops generated shims at the tree
root so a basename ``#include`` resolves).  We therefore give it a dedicated git
repo that is a COPY of ``runs/qcdloop_headers_full`` with ``boxGPU.h`` at the
root (``git init`` + one base commit).  The Validator instead needs a *pristine*
``vanilla_headers`` at ``starting_sha`` for its "current baseline" build, so it
keeps pointing at the main tree's ``runs/qcdloop_headers_full`` (clean, and
byte-identical to the headers repo's base commit).  The cumulative candidate diff
is taken in the headers repo (``git diff starting_sha..candidate_sha``, paths
relative to the headers root) and applied onto the pristine copy.

Usage (under the venv + module env):
    python runs/qcdloop/run_strategy_e2e.py \
        --report runs/qcdloop/report_smoke.json \
        --sample-count 1000 --tolerance 8
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from agents.config import PipelineConfig, StrategyBudget, StrategyConfig  # noqa: E402
from agents.patcher.agent import make_patcher_fn  # noqa: E402
from agents.strategy import agent as strategy_agent  # noqa: E402
from agents.validator.agent import make_validator_fn  # noqa: E402
from agents.validator import tail as _tail  # noqa: E402


def _git(repo: Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args],
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed in {repo}:\n{r.stderr}")
    return r.stdout.strip()


def _build_headers_repo(dest: Path, headers_src: Path) -> str:
    """Copy ``headers_src`` into a fresh git repo rooted at ``dest``; return HEAD.

    The tree root is the headers root (``boxGPU.h`` at top), so it doubles as
    ``QL_HEADERS`` — the layout the regional integrator + build gate assume.
    """
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(headers_src, dest)
    _git(dest, "init", "-q")
    # persistent identity so later Patcher commits (gitops.commit_all) succeed
    # regardless of global git config
    _git(dest, "config", "user.name", "amp-strategy")
    _git(dest, "config", "user.email", "amp@local")
    _git(dest, "add", "-A")
    _git(dest, "commit", "-q", "-m", "base: qcdloop_headers_full snapshot")
    return _git(dest, "rev-parse", "HEAD")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default=str(HERE / "report_smoke.json"))
    ap.add_argument("--sample-count", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tolerance", type=float, default=8.0)
    ap.add_argument("--headers-repo", default=str(Path.home() / "amp_strategy_headers_repo"))
    ap.add_argument("--dd-repo", default=str(Path.home() / "qcdloop"))
    ap.add_argument("--dd-ref", default="ddfun_enabled")
    ap.add_argument("--kokkos-root", default=str(Path.home() / "kokkos-install"))
    ap.add_argument("--max-iters", type=int, default=8,
                    help="Total StrategyBudget cap (smoke default 8). Used only to "
                         "derive the 70/30 phase split when --max-iters-correctness "
                         "/ --max-iters-speedup are not both given.")
    ap.add_argument("--max-iters-correctness", type=int, default=None,
                    help="Phase-1 (correctness) counting-iteration cap. Overrides the "
                         "70%% split of --max-iters. Unused phase-1 budget spills "
                         "forward into phase 2.")
    ap.add_argument("--max-iters-speedup", type=int, default=None,
                    help="Phase-2 (speedup) counting-iteration cap. Overrides the "
                         "30%% split of --max-iters. Phase-1 spill is added on top.")
    ap.add_argument("--max-wall-hours", type=float, default=4.0,
                    help="Wall-clock ceiling (safety net; iters should bind first).")
    ap.add_argument("--dr-k", type=int, default=None,
                    help="diminishing_returns_k: stop 'partial' after K consecutive "
                         "non-accepts. Default (None) uses StrategyConfig's 20. Raise "
                         "it when the cascade-chain phase's repeated per-representative "
                         "llm_gen_failed (which don't consume budget) would trip the "
                         "streak before the correctness budget cap binds.")
    ap.add_argument("--strategy-mode", default="tu_only",
                    choices=["tu_only", "region"],
                    help="tu_only (default): whole-TU-only mechanical walk (no LLM); "
                         "region: historical per-region Patcher-LLM walk.")
    ap.add_argument("--tu-out-dir", default=None,
                    help="tu_only: build/measure scratch dir (default: "
                         "<report-parent>/tu_e2e_out).")
    args = ap.parse_args(argv)

    report = Path(args.report).resolve()
    if not report.is_file():
        raise SystemExit(f"report not found: {report}")

    vanilla_headers = REPO / "runs" / "qcdloop_headers_full"
    # the validator's "current baseline" must be pristine == the headers repo base
    dirty = _git(REPO, "status", "--porcelain", str(vanilla_headers))
    if dirty:
        raise SystemExit(
            f"main tree {vanilla_headers} is dirty; refusing to run "
            f"(validator baseline must equal the headers-repo base):\n{dirty}")

    # dedicated flat headers repo for the Patcher (root == QL_HEADERS)
    repo = Path(args.headers_repo).resolve()
    starting_sha = _build_headers_repo(repo, vanilla_headers)

    snapshot = {"seed": args.seed, "sample_count": args.sample_count}

    budget = StrategyBudget(
        max_iters=args.max_iters,
        max_iters_correctness=args.max_iters_correctness,
        max_iters_speedup=args.max_iters_speedup,
        max_wall_clock_sec=args.max_wall_hours * 3600.0,
    )
    strategy_config = StrategyConfig(
        tolerance=args.tolerance,
        budget=budget,
        snapshot=snapshot,
        strategy_mode=args.strategy_mode,
        runs_root=HERE,                 # runs/qcdloop/strategy/<run_id>/
        **({"diminishing_returns_k": args.dr_k} if args.dr_k is not None else {}),
    )

    # tu_only: mechanical whole-TU walk driven by the injected L-measure provider.
    # No Patcher LLM / Validator callable are consulted (guarded off in the agent);
    # they stay None so an accidental region-walk invocation fails loud.
    if args.strategy_mode == "tu_only":
        from runs.qcdloop.tu_provider import make_tu_measure_fn  # noqa: E402
        tu_out = Path(args.tu_out_dir) if args.tu_out_dir else (
            report.parent / "tu_e2e_out")
        tu_measure_fn = make_tu_measure_fn(
            out_dir=tu_out, kokkos_root=args.kokkos_root,
            dd_repo=args.dd_repo, dd_ref=args.dd_ref,
            sample_count=args.sample_count)
        state = {
            "characterization_report_path": str(report),
            "strategy_repo_path": None,        # no working-tree mutation (route only)
            "strategy_starting_sha": None,
            "patcher_fn": None,
            "validator_fn": None,
            "tu_measure_fn": tu_measure_fn,
            "strategy_config": strategy_config,
        }
    else:
        build_config = {
            "app_cmake_dir": str(HERE / "app"),
            "kokkos_root": args.kokkos_root,
        }
        patcher_fn = make_patcher_fn(build_config=build_config, config=PipelineConfig())

        # Tail battery: if the report carries per-integral tail_samples (from
        # emit_tail_offsets.py), thread them into base_state so the Validator re-tests
        # the adversarial offsets on every candidate.  Absent → fail-open (random-only).
        tail_samples = _tail.load_tail_samples(report)
        tail_offsets = _tail.all_offsets(tail_samples) if tail_samples else []

        base_state = {
            "vanilla_headers": str(vanilla_headers),
            "dd_source_repo": args.dd_repo,
            "dd_ref": args.dd_ref,
            "accepted_patches": [],
            "kokkos_root": args.kokkos_root,
            "tail_samples": tail_samples,
        }
        validator_fn = make_validator_fn(
            base_state, starting_sha, str(repo), tolerance=args.tolerance)

        state = {
            "characterization_report_path": str(report),
            "strategy_repo_path": str(repo),
            "strategy_starting_sha": starting_sha,
            "patcher_fn": patcher_fn,
            "validator_fn": validator_fn,
            "strategy_config": strategy_config,
        }

    print("=== Strategy e2e config ===", flush=True)
    print(f"  strategy_mode   : {args.strategy_mode}", flush=True)
    print(f"  report          : {report}", flush=True)
    print(f"  starting_sha    : {starting_sha}", flush=True)
    print(f"  headers_repo    : {repo}", flush=True)
    print(f"  vanilla_headers : {vanilla_headers}", flush=True)
    print(f"  dd_repo@ref     : {args.dd_repo}@{args.dd_ref}", flush=True)
    print(f"  kokkos_root     : {args.kokkos_root}", flush=True)
    print(f"  tolerance       : {args.tolerance}", flush=True)
    print(f"  snapshot        : {snapshot}", flush=True)
    if args.strategy_mode != "tu_only":
        if tail_samples:
            print(f"  tail battery    : {len(tail_samples)} integrals, "
                  f"{len(tail_offsets)} distinct offsets (always-on)", flush=True)
        else:
            print("  tail battery    : none in report (fail-open, random-only)",
                  flush=True)
    cap_c, cap_s = budget.phase_caps()
    print(f"  budget          : max_iters={budget.max_iters} "
          f"wall={args.max_wall_hours}h tokens={budget.max_llm_tokens}", flush=True)
    print(f"  phase caps      : correctness={cap_c} speedup={cap_s} "
          f"(speedup gets phase-1 spill on top)", flush=True)
    print(f"  dr_k            : {strategy_config.diminishing_returns_k}", flush=True)
    print(f"  model           : {PipelineConfig().model}", flush=True)
    print(f"  base_url        : {PipelineConfig().base_url}", flush=True)
    print("===========================", flush=True)

    delta = strategy_agent.run(state)

    print("\n=== Strategy result delta ===", flush=True)
    for k, v in delta.items():
        print(f"  {k}: {v}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
