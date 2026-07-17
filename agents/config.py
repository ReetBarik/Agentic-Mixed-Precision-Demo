"""Pipeline configuration — environment defaults + dataclass for per-run overrides."""

import os
from dataclasses import dataclass, field
from pathlib import Path

PROXY_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8083/argoapi/")
AUTH_TOKEN = (
    os.environ.get("ANTHROPIC_AUTH_TOKEN")
    or os.environ.get("ARGO_USERNAME", "")
)
# Argo model name — check available models via `run-argo.sh` if this needs updating
DEFAULT_MODEL = os.environ.get("ARGO_MODEL", "claudeopus47")


@dataclass
class StrategyBudget:
    """Hard caps on a single Strategy run. Any cap exceeded → status budget_exhausted."""

    max_iters: int = 500                     # counting iterations (see StrategyConfig notes)
    max_wall_clock_sec: float = 6 * 3600.0   # 6h wall-clock ceiling
    max_llm_tokens: int = 20_000_000         # cumulative LLM tokens reported by Patcher


@dataclass
class StrategyConfig:
    """Knobs Strategy reads from PipelineState.strategy_config.

    ``tolerance`` is the minimum precise-digit bar (design default 10).
    ``diminishing_returns_k`` — if the last K counting iterations all fail to
    accept, declare the run stuck (status ``partial``).
    ``recharacterize_after_n`` — locked as fixed-report-only; set to a value we
    won't hit in practice so re-characterization never triggers.
    ``snapshot`` — seed + sample_count handed to the Validator callable.
    ``runs_root`` — parent of ``strategy/<run_id>/`` artifact dirs; defaults to
    ``<repo>/runs/qcdloop`` when None.
    """

    tolerance: float = 10.0
    budget: StrategyBudget = field(default_factory=StrategyBudget)
    diminishing_returns_k: int = 20
    recharacterize_after_n: int = 10**9      # effectively disabled (fixed-report-only)
    snapshot: dict = field(default_factory=lambda: {"seed": 12345, "sample_count": 100000})
    runs_root: Path | None = None


@dataclass
class PipelineConfig:
    model: str = DEFAULT_MODEL
    flag_threshold: float = 1e8      # cond > this → OpRecord.flagged = True
    sample_count: int = 512
    top_n_hotspots: int = 10
    # Force a single interop strategy for all non-templatable calls: "interop" | "opaque" | "inline"
    strategy_override: str | None = None
    max_driver_attempts: int = 5      # total LLM driver attempts incl. the first
    retry_stderr_chars: int = 3000    # truncation budget for fed-back build stderr
    tracked_root: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "third_party" / "tracked"
    )
    kokkos_root: Path | None = None   # path to Kokkos install (optional)
    out_dir: Path | None = None       # where to write output artifacts (cli sets this)
    cxx_standard: int = 17
    base_url: str = field(default_factory=lambda: PROXY_BASE_URL)
    auth_token: str = field(default_factory=lambda: AUTH_TOKEN)
