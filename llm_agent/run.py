"""Main CLI entry point for the agentic mixed-precision optimization workflow.

Usage:
    python3.12 -m llm_agent.run \\
        --file path/to/header.h \\
        --function function_name \\
        [--skills downcast] \\
        [--min-digits 10] \\
        [--batch 10] \\
        [--seed 123] \\
        [--max-iterations 3] \\
        [--max-driver-retries 5] \\
        [--base-url http://127.0.0.1:8083/argoapi/] \\
        [--output-dir experiments/generated/]
"""

import argparse
import json
import os
import sys

from llm_agent import config
from llm_agent.graphs.orchestrator import build_orchestrator
from llm_agent.state import OptimizationState


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic mixed-precision optimization for C++ functions."
    )
    parser.add_argument(
        "--file",
        required=True,
        metavar="PATH",
        help="Repo-relative path to the C++ header/source containing the target function.",
    )
    parser.add_argument(
        "--function",
        required=True,
        metavar="NAME",
        help="Name of the function to analyze and optimize.",
    )
    parser.add_argument(
        "--skills",
        nargs="+",
        default=["downcast"],
        metavar="SKILL",
        help="Optimization skills to apply (default: downcast).",
    )
    parser.add_argument(
        "--min-digits",
        type=float,
        default=config.DEFAULT_MIN_DIGITS,
        metavar="N",
        help="Minimum precise decimal digits required vs double baseline (default: %(default)s).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=config.DEFAULT_BATCH,
        metavar="N",
        help="Number of random input samples per run (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.DEFAULT_SEED,
        metavar="N",
        help="RNG seed for reproducible inputs (default: %(default)s).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=config.MAX_ITERATIONS_PER_VAR,
        metavar="N",
        help="Max LLM proposal attempts per variable (default: %(default)s).",
    )
    parser.add_argument(
        "--max-driver-retries",
        type=int,
        default=5,
        metavar="N",
        help="Max compile-fix iterations for the driver skill (default: %(default)s).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        metavar="URL",
        help="Override the Anthropic proxy base URL (default: $ANTHROPIC_BASE_URL or config).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Directory for generated CSVs and summaries (default: <root>/experiments).",
    )
    parser.add_argument(
        "--root",
        default=None,
        metavar="DIR",
        help="Repository root (default: inferred from this file's location).",
    )

    args = parser.parse_args()

    root = args.root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    initial_state = OptimizationState(
        file_path=args.file,
        function_name=args.function,
        root=root,
        min_digits=args.min_digits,
        batch=args.batch,
        seed=args.seed,
        max_iterations=args.max_iterations,
        max_driver_retries=args.max_driver_retries,
        skills=args.skills,
        base_url=args.base_url,
        output_dir=args.output_dir,
        signature=None,
        baseline_csv=None,
        skill_results={},
        error=None,
    )

    orchestrator = build_orchestrator()
    final_state = orchestrator.invoke(initial_state)

    summary = {
        "function":      final_state.get("function_name"),
        "file":          final_state.get("file_path"),
        "framework":     (final_state.get("signature") or {}).get("framework"),
        "baseline_csv":  final_state.get("baseline_csv"),
        "error":         final_state.get("error"),
        "skill_results": final_state.get("skill_results") or {},
    }

    print(json.dumps(summary, indent=2))

    if final_state.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
