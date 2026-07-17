"""LangGraph TypedDict state for the mixed-precision pipeline."""

from __future__ import annotations

from operator import add
from typing import Annotated, Callable, TypedDict

from agents.characterizer.profile import SensitivityProfile, SymbolicHint
from agents.characterizer.spec import InstrumentationSpec
from agents.config import PipelineConfig, StrategyConfig


class PipelineState(TypedDict):
    # ---- Inputs (set once at pipeline entry) ----
    source_files: list[str]
    kernel_name: str
    input_ranges: dict[str, tuple[float, float]]
    build_instructions: str
    whole_app_driver: str | None
    config: PipelineConfig

    # ---- Characterizer outputs (Annotated[..., add] for fan-out safety) ----
    sensitivity_profiles: Annotated[list[SensitivityProfile], add]
    symbolic_hints: Annotated[list[SymbolicHint], add]
    instrumentation_specs: Annotated[list[InstrumentationSpec], add]
    journal_paths: Annotated[list[str], add]

    # ---- Strategy / patcher / validator (stubbed in v1) ----
    strategy_queue: list           # plain — single writer in v1
    current_patch: dict | None     # plain — single writer in v1
    validation_result: dict | None # plain — single writer in v1
    accepted_patches: Annotated[list[dict], add]
    rejected_patches: Annotated[list[dict], add]

    # ---- Strategy inputs (fields Strategy reads; see docs/strategy_patcher_design.md) ----
    # Strategy owns the full remediation loop internally (Q5): it drives Patcher +
    # Validator as callables rather than as separate graph nodes.
    characterization_report_path: str | None   # fixed report (stability_report JSON)
    strategy_repo_path: str | None             # git working tree Strategy branches from
    strategy_starting_sha: str | None          # caller-supplied base SHA
    strategy_config: StrategyConfig | None     # tolerance, budget, K, N, snapshot, runs_root
    # Patcher(intent: dict, ctx: dict) -> P2 response dict.
    patcher_fn: Callable[[dict, dict], dict] | None
    # Validator(candidate_sha: str, ctx: dict) -> verdict dict.
    validator_fn: Callable[[str, dict], dict] | None

    # ---- Strategy output (thin pointer bundle; fat artifacts live on disk) ----
    strategy_result: dict | None

    # ---- Bookkeeping ----
    iteration: int
    errors: Annotated[list[str], add]
