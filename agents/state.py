"""LangGraph TypedDict state for the mixed-precision pipeline."""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from agents.characterizer.profile import SensitivityProfile, SymbolicHint
from agents.characterizer.spec import InstrumentationSpec
from agents.config import PipelineConfig


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

    # ---- Bookkeeping ----
    iteration: int
    errors: Annotated[list[str], add]
