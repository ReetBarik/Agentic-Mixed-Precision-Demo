"""Deliverable 3 — per-integral dispatch layer (RES-stream selector/merger).

The merge logic is exercised with a stubbed ``run_and_aggregate`` (via monkeypatch) so
no built binaries are needed; the harness exercises the real binaries end to end.
"""

from __future__ import annotations

from array import array
from pathlib import Path

import pytest

from agents.patcher import flip_dispatch as fd
from agents.patcher.flip_dispatch import (
    BinarySource, DispatchPlan, dispatch_and_aggregate)
from agents.patcher.precision_flip import TargetPrecision


def _coeff(tag: float, total: int = 1):
    # Distinguishable (hi, lo) arrays so a merge can be traced to its source.
    from agents.validator.coeffs import N_COMPONENTS
    n = total * N_COMPONENTS
    return (array("d", [tag] * n), array("d", [tag * 10] * n))


def _stub_runner(monkeypatch, table: dict[str, dict[str, tuple]]):
    """Map binary path -> {integral: coeff}; patch run_and_aggregate to serve it."""
    def fake(binary, total, *, chunk=0, workers=1):
        return dict(table[str(binary)])
    monkeypatch.setattr(fd, "run_and_aggregate", fake)


# --------------------------------------------------------------------------- #
# plan validation
# --------------------------------------------------------------------------- #

def test_plan_rejects_double_claimed_integral():
    base = BinarySource(Path("/v"), "double", ("B1",), "vanilla")
    f1 = BinarySource(Path("/a"), TargetPrecision.DD, ("B10",), "B1m")
    f2 = BinarySource(Path("/b"), TargetPrecision.DD, ("B10",), "B2m")
    with pytest.raises(ValueError):
        DispatchPlan(baseline=base, flips=(f1, f2))


def test_plan_promoted_and_source_for():
    base = BinarySource(Path("/v"), "double", ("B1", "B12"), "vanilla")
    f1 = BinarySource(Path("/a"), TargetPrecision.DD, ("B10",), "B1m")
    plan = DispatchPlan(baseline=base, flips=(f1,))
    assert plan.promoted == {"B10"}
    assert plan.source_for("B10").label == "B1m"
    assert plan.source_for("B1").label == "vanilla"   # falls to baseline


# --------------------------------------------------------------------------- #
# merge semantics
# --------------------------------------------------------------------------- #

def test_promoted_integral_taken_from_flip_others_from_baseline(monkeypatch):
    vB1, vB10 = _coeff(1.0), _coeff(2.0)     # baseline emits both (B10 zero-ish at double)
    dB10 = _coeff(9.0)                        # dd binary's honest B10
    _stub_runner(monkeypatch, {
        "/v": {"B1": vB1, "B10": vB10},
        "/a": {"B10": dB10},
    })
    plan = DispatchPlan(
        baseline=BinarySource(Path("/v"), "double", ("B1", "B10"), "vanilla"),
        flips=(BinarySource(Path("/a"), TargetPrecision.DD, ("B10",), "B1m"),))
    res = dispatch_and_aggregate(plan, total=1)
    # B1 from baseline, B10 from the dd flip.
    assert res.coeffs["B1"] is vB1
    assert res.coeffs["B10"] is dB10
    assert res.provenance == {"B1": "double", "B10": "dd"}


def test_missing_claimed_integral_fails_loud(monkeypatch):
    _stub_runner(monkeypatch, {
        "/v": {"B1": _coeff(1.0)},
        "/a": {"B99": _coeff(9.0)},           # dd binary does NOT emit the claimed B10
    })
    plan = DispatchPlan(
        baseline=BinarySource(Path("/v"), "double", ("B1",), "vanilla"),
        flips=(BinarySource(Path("/a"), TargetPrecision.DD, ("B10",), "B1m"),))
    with pytest.raises(ValueError, match="expected to emit"):
        dispatch_and_aggregate(plan, total=1)


def test_no_flips_is_pure_baseline(monkeypatch):
    _stub_runner(monkeypatch, {"/v": {"B1": _coeff(1.0), "B2": _coeff(2.0)}})
    plan = DispatchPlan(baseline=BinarySource(Path("/v"), "double", ("B1", "B2"), "vanilla"))
    res = dispatch_and_aggregate(plan, total=1)
    assert set(res.coeffs) == {"B1", "B2"}
    assert set(res.provenance.values()) == {"double"}


def test_multiple_flip_groups_merge(monkeypatch):
    _stub_runner(monkeypatch, {
        "/v": {"B1": _coeff(1.0), "B10": _coeff(0.0), "B12": _coeff(0.0)},
        "/a": {"B10": _coeff(9.0)},
        "/b": {"B12": _coeff(8.0)},
    })
    plan = DispatchPlan(
        baseline=BinarySource(Path("/v"), "double", ("B1", "B10", "B12"), "vanilla"),
        flips=(BinarySource(Path("/a"), TargetPrecision.DD, ("B10",), "B1m"),
               BinarySource(Path("/b"), TargetPrecision.DD, ("B12",), "B2m")))
    res = dispatch_and_aggregate(plan, total=1)
    assert res.provenance == {"B1": "double", "B10": "dd", "B12": "dd"}
