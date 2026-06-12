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
class PipelineConfig:
    model: str = DEFAULT_MODEL
    flag_threshold: float = 1e8      # cond > this → OpRecord.flagged = True
    sample_count: int = 512
    top_n_hotspots: int = 10
    # Force a single interop strategy for all non-templatable calls: "interop" | "opaque" | "inline"
    strategy_override: str | None = None
    tracked_root: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "third_party" / "tracked"
    )
    kokkos_root: Path | None = None   # path to Kokkos install (optional)
    out_dir: Path | None = None       # where to write output artifacts (cli sets this)
    cxx_standard: int = 17
    base_url: str = field(default_factory=lambda: PROXY_BASE_URL)
    auth_token: str = field(default_factory=lambda: AUTH_TOKEN)
