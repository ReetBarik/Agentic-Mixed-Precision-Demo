"""Filter fidelity + chain integrity + existing-Strategy compatibility."""

from __future__ import annotations

import json

import pytest

from agents.per_integral_orchestrator import filter_report
from agents.strategy.characterization import load_chains, load_regions
from tests.per_integral_orchestrator.conftest import _region, make_report_dict


def _load(path):
    return json.loads(path.read_text())


def test_filter_keeps_only_target_integral(synth_report, tmp_path):
    report_path, _ = synth_report
    out = tmp_path / "report_B1.json"
    meta = filter_report(report_path, "B1", out)

    doc = _load(out)
    assert list(doc["integrals"]) == ["B1"]           # no other integral present
    assert set(doc["samples_seen"]) == {"B1"}         # samples_seen narrowed
    assert doc["schema_version"] == 2
    assert meta == {"integral": "B1", "n_regions": 2, "n_chains": 0,
                    "schema_version": 2}

    # no leakage: every retained region is tagged B1
    for region in doc["integrals"]["B1"]["regions"].values():
        assert region["integral"] == "B1"


def test_filter_counts_match_per_integral(synth_report, tmp_path):
    report_path, _ = synth_report
    assert filter_report(report_path, "B4", tmp_path / "b4.json")["n_regions"] == 3
    assert filter_report(report_path, "B4", tmp_path / "b4.json")["n_chains"] == 1
    assert filter_report(report_path, "X9", tmp_path / "x9.json")["n_regions"] == 1


def test_filter_unknown_integral_raises(synth_report, tmp_path):
    report_path, _ = synth_report
    with pytest.raises(KeyError):
        filter_report(report_path, "NOPE", tmp_path / "nope.json")


def test_filter_fidelity_violation_raises(tmp_path):
    # a region under B1 mis-tagged as B4 must fail loudly (no silent leak)
    doc = make_report_dict({"B1": {"regions": {
        "boxGPU.h:99": _region("B1"),
        "boxGPU.h:100": _region("B1", integral_tag="B4"),
    }}})
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="fidelity"):
        filter_report(p, "B1", tmp_path / "out.json")


def test_filter_preserves_chain_whole(synth_report, tmp_path):
    report_path, _ = synth_report
    out = tmp_path / "b4.json"
    filter_report(report_path, "B4", out)

    chains, meta = load_chains(out)
    assert meta["n_chains"] == 1
    (chain,) = chains
    assert chain.integral == "B4"
    assert chain.chain_id == "B4::c0"
    # the two-line span survived intact (not truncated to one line)
    assert len(chain.lines) == 2
    assert [(ln.file, ln.line_start, ln.line_end) for ln in chain.lines] == [
        ("box/B4m.h", 10, 12), ("box/B4m.h", 20, 22)]


def test_filtered_report_loads_via_strategy_loaders(synth_report, tmp_path):
    """Filtered single-integral report loads cleanly via the real loaders."""
    report_path, _ = synth_report
    out = tmp_path / "b4.json"
    filter_report(report_path, "B4", out)

    regions, meta = load_regions(out, merge=True)
    assert meta["schema_version"] == 2
    assert {r.integral for r in regions} == {"B4"}   # only B4 records
    assert len(regions) == 3
    chains, cmeta = load_chains(out)
    assert cmeta["n_chains"] == 1
