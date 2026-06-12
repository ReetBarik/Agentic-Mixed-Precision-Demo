"""Data classes for characterizer output."""

from dataclasses import dataclass, field


@dataclass
class OpRecord:
    op: str                       # "add", "sub", "mul", "opaque", etc.
    location: str                 # "file:fn:line" or "" if not captured
    max_cond: float
    max_rel_err: float
    sample_count: int
    provenance_union: set[str]
    flagged: bool                 # max_cond > config.flag_threshold


@dataclass
class SensitivityProfile:
    kernel: str
    samples_run: int
    per_op: list[OpRecord] = field(default_factory=list)          # sorted by max_cond desc
    per_line: dict[str, OpRecord] = field(default_factory=dict)   # rolled up by source location
    per_variable: dict[str, float] = field(default_factory=dict)  # var → max cond it appeared in
    top_hotspots: list[OpRecord] = field(default_factory=list)    # top-N by max_cond
    opaque_coverage: float = 0.0  # fraction of records that are opaque
    notes: list[str] = field(default_factory=list)


@dataclass
class SymbolicHint:
    idiom: str                    # "log_sum_exp_naive", "naive_variance", ...
    location: str                 # source range "file:fn:start-end"
    severity: str                 # "low" | "medium" | "high"
    suggested_rewrite: str        # short prose
