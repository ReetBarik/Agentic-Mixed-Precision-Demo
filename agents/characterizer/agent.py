"""Characterizer agent — top-level LangGraph node.

Internal flow:
    spec_build → driver_gen → build_run → log_parse → symbolic_overlay → emit profile
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agents.characterizer import driver_gen, log_parser, symbolic_overlay
from agents.characterizer.spec import InstrumentationSpec
from agents.build_run import agent as build_run_agent
from agents.state import PipelineState


def run(state: PipelineState) -> dict:
    cfg = state["config"]
    kernel_name = state["kernel_name"]
    source_files = state["source_files"]
    input_ranges = state["input_ranges"]

    try:
        # Step 1: build instrumentation spec from source
        spec = _spec_build(source_files, kernel_name, input_ranges, cfg)

        # Step 2: LLM generates the micro-driver
        driver_result = driver_gen.generate(spec, cfg)

        # Step 3: build and run via deterministic subprocess wrapper
        run_result = build_run_agent.build_and_run(
            driver_source=driver_result.driver_source,
            framework=spec.framework,
            cfg=cfg,
            work_dir=cfg.out_dir,
        )

        updates: dict = {
            "instrumentation_specs": [spec],
        }

        if run_result.returncode != 0:
            msg = (
                f"build/run failed for {kernel_name}: "
                f"{run_result.stderr[:2000]}"
            )
            print(f"[characterizer] {msg}", file=sys.stderr)
            return {**updates, "errors": [msg]}

        updates["journal_paths"] = [str(run_result.journal_path)]

        # Step 4: parse JSONL → SensitivityProfile
        profile = log_parser.parse(
            journal_path=run_result.journal_path,
            kernel_name=kernel_name,
            flag_threshold=cfg.flag_threshold,
            top_n=cfg.top_n_hotspots,
            sample_count=spec.sample_count,
            work_dir=cfg.out_dir,
        )

        # Step 5: symbolic overlay (best-effort, never gates pipeline)
        hints = []
        try:
            hints = symbolic_overlay.analyze(source_files, kernel_name, cfg)
        except Exception as exc:
            print(f"[characterizer] symbolic_overlay failed (non-fatal): {exc}", file=sys.stderr)

        # Step 6: emit
        _emit(run_result.work_dir, driver_result, profile, hints)

        return {
            **updates,
            "sensitivity_profiles": [profile],
            "symbolic_hints": hints,
        }

    except Exception as exc:
        msg = f"characterizer raised for {kernel_name}: {exc}"
        print(f"[characterizer] {msg}", file=sys.stderr)
        return {"errors": [msg]}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _spec_build(
    source_files: list[str],
    kernel_name: str,
    input_ranges: dict[str, tuple[float, float]],
    cfg,
) -> InstrumentationSpec:
    """Parse kernel signature and build an InstrumentationSpec.

    Uses regex for v1 (libclang in v2 when signatures get hairier).
    """
    import re

    # Read all source files and search for the kernel
    source_text = ""
    for path in source_files:
        source_text += Path(path).read_text(encoding="utf-8") + "\n"

    # Extract the function signature — handles template + non-template, value return + void
    sig_pattern = re.compile(
        r"""
        (?:template\s*<[^>]*>[\s\n]*)? # optional template<...>
        [\w:\s*&<>,]+?                  # return type
        \b""" + re.escape(kernel_name) + r"""\s*
        \(([^)]*)\)                     # parameter list
        """,
        re.VERBOSE | re.DOTALL,
    )
    m = sig_pattern.search(source_text)
    if not m:
        raise ValueError(f"Could not find signature for {kernel_name!r} in source files")

    raw_signature = m.group(0).strip()
    param_text = m.group(1).strip()

    # Parse parameter list into (name, type) pairs
    parameter_types: list[tuple[str, str]] = []
    if param_text:
        for raw_param in param_text.split(","):
            raw_param = raw_param.strip()
            if not raw_param:
                continue
            # Strip default values
            raw_param = raw_param.split("=")[0].strip()
            # Last token is the name
            tokens = raw_param.split()
            if len(tokens) >= 2:
                name = tokens[-1].lstrip("*&")
                type_str = " ".join(tokens[:-1])
                parameter_types.append((name, type_str))
            else:
                parameter_types.append((raw_param, raw_param))

    # Detect framework from source text.
    # Match Kokkos_Core.hpp, Kokkos::, and KOKKOS_ macros.
    framework = "plain-cpp"
    if re.search(r"Kokkos_Core\.hpp|Kokkos::|KOKKOS_", source_text):
        framework = "kokkos-serial"

    # Detect user-side dispatchers (function calls that look like kXxx)
    dispatcher_pattern = re.compile(r"\b(k[A-Z]\w+)\s*\(")
    detected_dispatchers = list(set(dispatcher_pattern.findall(source_text)))

    # Per-parameter template instantiation: params that appear as arguments to
    # Imag()/Real() calls are complex; all others are real scalars.
    complex_param_names: set[str] = set()
    for m in re.finditer(r'\b(?:Imag|Real)\s*\(\s*(\w+)', source_text):
        complex_param_names.add(m.group(1))

    template_instantiation: dict[str, str] = {}
    for name, type_str in parameter_types:
        if "template" in type_str.lower() or type_str.startswith("T"):
            if name in complex_param_names:
                template_instantiation[type_str] = "tracked::Complex<double>"
            else:
                template_instantiation[type_str] = "tracked::Tracked<double>"

    return InstrumentationSpec(
        kernel_name=kernel_name,
        kernel_signature=raw_signature,
        parameter_types=parameter_types,
        input_ranges=input_ranges,
        template_instantiation=template_instantiation,
        sample_count=cfg.sample_count,
        framework=framework,
        source_files=source_files,
        detected_dispatchers=detected_dispatchers,
    )


def _emit(work_dir: Path, driver_result, profile, hints: list) -> None:
    """Write output artifacts to work_dir."""
    import dataclasses

    work_dir.mkdir(parents=True, exist_ok=True)

    decisions_path = work_dir / "interop_decisions.json"
    decisions_path.write_text(
        json.dumps(
            [dataclasses.asdict(d) if hasattr(d, "__dataclass_fields__") else d.__dict__
             for d in driver_result.interop_decisions],
            indent=2,
        ),
        encoding="utf-8",
    )

    profile_path = work_dir / "sensitivity_profile.json"
    profile_path.write_text(
        json.dumps(_profile_to_dict(profile), indent=2),
        encoding="utf-8",
    )

    hints_path = work_dir / "symbolic_hints.json"
    hints_path.write_text(
        json.dumps([_hint_to_dict(h) for h in hints], indent=2),
        encoding="utf-8",
    )


def _profile_to_dict(p) -> dict:
    import dataclasses
    d = dataclasses.asdict(p)
    # set[str] isn't JSON-serialisable
    for rec in d.get("per_op", []):
        rec["provenance_union"] = list(rec.get("provenance_union", []))
    for rec in d.get("per_line", {}).values():
        rec["provenance_union"] = list(rec.get("provenance_union", []))
    if "top_hotspots" in d:
        for rec in d["top_hotspots"]:
            rec["provenance_union"] = list(rec.get("provenance_union", []))
    return d


def _hint_to_dict(h) -> dict:
    import dataclasses
    return dataclasses.asdict(h)
