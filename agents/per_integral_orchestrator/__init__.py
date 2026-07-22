"""Per-integral pass orchestrator (Phase 1 of the caller-scoped pipeline).

Runs the existing Strategy -> Patcher -> Validator pipeline **once per integral**
against a report filtered to that integral, in a fully isolated output tree.  This
recovers the per-integral precision signal that Strategy's ``_merge_by_line``
worst-cases away (see ``runs/qcdloop/PHASE_B_PROBE_2026-07-22.md``: 41.1% of shared
source lines would get a cheaper precision under per-integral routing).

Scope (Phase 1): pure orchestration + report filtering + tree isolation.  Strategy,
Patcher, Validator, the reducer and the characterizer are untouched; there is no
call-graph fan-out, no caller-scoped shim naming, no rename cascade, and no combine
step.  Cascade chains stay on the merged whole-app path (the probe found them 100%
uniformly ``dd`` — zero per-integral payoff); per-integral routing applies to regions
only.
"""

from agents.per_integral_orchestrator.filter_report import filter_report
from agents.per_integral_orchestrator.manifest import build_manifest
from agents.per_integral_orchestrator.orchestrator import run_per_integral_pass

__all__ = ["filter_report", "build_manifest", "run_per_integral_pass"]
