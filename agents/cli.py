"""Entry point: python -m agents.cli characterize ..."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from agents.config import PipelineConfig
from agents.orchestrator import build_graph
from agents.state import PipelineState


def main() -> None:
    parser = argparse.ArgumentParser(prog="agents.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    char = sub.add_parser("characterize", help="Run the characterizer pipeline on a kernel.")
    char.add_argument("--kernel", required=True, nargs="+",
                      help="Path(s) to kernel source file(s).")
    char.add_argument("--kernel-name", required=True,
                      help="Name of the kernel function.")
    char.add_argument("--ranges-yaml", required=True,
                      help="YAML file with input ranges (see fixtures/input_ranges/).")
    char.add_argument("--samples", type=int, default=512,
                      help="Number of samples (default: 512).")
    char.add_argument("--out", default="runs/out",
                      help="Output directory (default: runs/out).")
    char.add_argument("--model", default=None,
                      help="Override the Argo model name.")
    char.add_argument("--flag-threshold", type=float, default=1e8,
                      help="Condition number threshold for flagging ops (default: 1e8).")
    char.add_argument("--strategy-override", choices=["interop", "opaque", "inline"],
                      default=None,
                      help="Force a single interop strategy for all non-templatable calls.")
    char.add_argument("--max-driver-attempts", type=int, default=5,
                      help="Max LLM driver attempts incl. the first; retries on compile "
                           "failure feed the build error back to the LLM (default: 5).")
    char.add_argument("--tracked-root", default=None,
                      help="Path to the Tracked library checkout (default: third_party/tracked).")
    char.add_argument("--kokkos-root", default=None,
                      help="Path to Kokkos install (required for kokkos-serial kernels).")

    args = parser.parse_args()

    if args.command == "characterize":
        _run_characterize(args)


def _run_characterize(args: argparse.Namespace) -> None:
    ranges_path = Path(args.ranges_yaml)
    with ranges_path.open(encoding="utf-8") as f:
        ranges_doc = yaml.safe_load(f)
    input_ranges: dict[str, tuple[float, float]] = {
        k: tuple(v) for k, v in ranges_doc.get("ranges", {}).items()
    }

    cfg_kwargs: dict = {
        "sample_count": args.samples,
        "flag_threshold": args.flag_threshold,
        "strategy_override": args.strategy_override,
        "max_driver_attempts": args.max_driver_attempts,
    }
    if args.tracked_root:
        cfg_kwargs["tracked_root"] = Path(args.tracked_root).resolve()
    if args.kokkos_root:
        cfg_kwargs["kokkos_root"] = Path(args.kokkos_root).resolve()
    if args.model:
        cfg_kwargs["model"] = args.model

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_kwargs["out_dir"] = out_dir

    cfg = PipelineConfig(**cfg_kwargs)

    # Characterize-only: the strategy node needs injected callables + a fixed
    # report (driven by runs/qcdloop/run_strategy_e2e.py, not by this CLI).
    graph = build_graph(through_strategy=False)

    initial_state: PipelineState = {
        "source_files": [str(Path(k).resolve()) for k in args.kernel],
        "kernel_name": args.kernel_name,
        "input_ranges": input_ranges,
        "build_instructions": "",
        "whole_app_driver": None,
        "config": cfg,
        "sensitivity_profiles": [],
        "symbolic_hints": [],
        "instrumentation_specs": [],
        "journal_paths": [],
        "strategy_queue": [],
        "current_patch": None,
        "validation_result": None,
        "accepted_patches": [],
        "rejected_patches": [],
        "iteration": 0,
        "errors": [],
    }

    final_state = graph.invoke(initial_state)

    if final_state.get("errors"):
        print("[cli] Pipeline completed with errors:", file=sys.stderr)
        for err in final_state["errors"]:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    profiles = final_state.get("sensitivity_profiles", [])
    if profiles:
        profile = profiles[-1]
        print(f"[cli] Profile for {args.kernel_name}:")
        print(f"  samples_run    : {profile.samples_run}")
        print(f"  ops found      : {len(profile.per_op)}")
        print(f"  opaque_coverage: {profile.opaque_coverage:.0%}")
        if profile.top_hotspots:
            top = profile.top_hotspots[0]
            print(f"  top hotspot    : {top.op} @ {top.location} (cond={top.max_cond:.2e})")
        for note in profile.notes:
            print(f"  note: {note}")
        print(f"[cli] Outputs written to {out_dir}/")
    else:
        print("[cli] No profiles produced.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
