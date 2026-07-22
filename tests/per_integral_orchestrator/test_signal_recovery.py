"""Signal-recovery smoke test — consumes the Phase B probe finding.

The Phase B probe (runs/qcdloop/PHASE_B_PROBE_2026-07-22.md) found that
``boxGPU.h:99-101`` merge to ``dd`` under Strategy's whole-app ``_merge_by_line``,
but B1 alone would reach ``float`` there.  This test proves the per-integral
*filter* actually delivers that finer signal: after filtering report_5k.json to
B1, Strategy's own ranking (``build_queues`` via ``phase_b_probe.region_decisions``)
emits ``float`` intent on those lines, where the whole-app merge emits ``dd``.

Deterministic (no Patcher/Validator/LLM): it measures the upstream intent target,
which is exactly the layer the merge destroys and the layer routing recovers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.config import StrategyConfig
from agents.per_integral_orchestrator import filter_report
from agents.strategy.characterization import load_regions
from agents.strategy.ranking import load_flop_weights

_REPO = Path(__file__).resolve().parents[2]
_REPORT = _REPO / "runs" / "qcdloop" / "report_5k.json"

pytestmark = pytest.mark.skipif(
    not _REPORT.is_file(), reason="report_5k.json not present")


def _region_decisions(records, tol, flop_weights):
    # import lazily so the module imports even if runs/ isn't a package on the path
    from runs.qcdloop.phase_b_probe import region_decisions
    return region_decisions(records, tol, flop_weights)


def test_b1_recovers_float_on_boxgpu_99_101(tmp_path):
    cfg = StrategyConfig()
    tol = float(cfg.tolerance)
    fw = load_flop_weights(_REPO / "runs" / "qcdloop" / "ratio_multipliers.json")

    # whole-app merged decision (today's Strategy path)
    merged, _ = load_regions(_REPORT, merge=True)
    merged_dec = _region_decisions(merged, tol, fw)

    # per-integral decision via the orchestrator's filter
    b1_report = tmp_path / "report_B1.json"
    filter_report(_REPORT, "B1", b1_report)
    b1_regions, _ = load_regions(b1_report, merge=True)
    b1_dec = _region_decisions(b1_regions, tol, fw)

    ladder = {"float": 0, "ff": 1, "double": 2, "dd": 3}

    # the probe's top-2 wasted-headroom lines: merged=dd, B1=float
    for line in (100, 101):
        key = ("boxGPU.h", line, line)
        assert key in merged_dec, f"{key} missing from merged decisions"
        assert key in b1_dec, f"{key} missing from B1 decisions"
        assert merged_dec[key] == "dd", (
            f"expected merged {key} == dd, got {merged_dec[key]}")
        assert b1_dec[key] == "float", (
            f"expected B1 {key} == float (routing recovers signal), "
            f"got {b1_dec[key]}")
        assert ladder[b1_dec[key]] < ladder[merged_dec[key]]

    # and B1 recovers a cheaper rung on multiple lines overall (probe: 12)
    cheaper = [k for k, p in b1_dec.items()
               if k in merged_dec and ladder[p] < ladder[merged_dec[k]]]
    assert len(cheaper) >= 5, f"expected several recovered lines, got {cheaper}"
