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
    """Hard caps on a single Strategy run. Any cap exceeded → status budget_exhausted.

    ``max_iters`` is the *total* counting-iteration budget, split across the
    two-phase walk (correctness then speedup). ``max_iters_correctness`` /
    ``max_iters_speedup`` override the split per phase: when both are ``None`` the
    total is split 70/30; when only one is set the other defaults proportionally
    (0.7 / 0.3 of ``max_iters``). Unused correctness budget spills forward into the
    speedup phase at the phase boundary; unused speedup budget never spills back
    (correctness is already terminated). ``max_wall_clock_sec`` / ``max_llm_tokens``
    are global ceilings across both phases.
    """

    max_iters: int = 500                     # total counting-iteration budget
    max_iters_correctness: int | None = None  # phase-1 cap (default: 0.7 * max_iters)
    max_iters_speedup: int | None = None      # phase-2 cap (default: 0.3 * max_iters)
    max_wall_clock_sec: float = 6 * 3600.0   # 6h wall-clock ceiling (global)
    max_llm_tokens: int = 20_000_000         # cumulative LLM tokens reported by Patcher

    def phase_caps(self) -> tuple[int, int]:
        """Resolve ``(correctness_cap, speedup_cap)`` from the knobs.

        Explicit per-phase knobs are honored; a ``None`` knob defaults to its 70/30
        proportion of ``max_iters``. When *neither* is set the speedup cap takes the
        exact remainder so the two sum to ``max_iters`` (no rounding drift).
        """
        c, s = self.max_iters_correctness, self.max_iters_speedup
        if c is None and s is None:
            c = round(0.7 * self.max_iters)
            s = self.max_iters - c
        elif c is None:
            c = round(0.7 * self.max_iters)
        elif s is None:
            s = round(0.3 * self.max_iters)
        return max(0, int(c)), max(0, int(s))


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
    # 60 (was 20): the cascade-chain phase produces long non-accept streaks of
    # llm_gen_failed which don't consume budget but DO bump the DR counter, so 20
    # tripped `partial` before the correctness budget could bind even after the
    # chain-representative dedup (see runs/qcdloop/CALIBRATION.md §50k recommendation).
    diminishing_returns_k: int = 60
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
