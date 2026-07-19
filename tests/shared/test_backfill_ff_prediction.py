"""Unit tests for the predicted_rel_err_if_ff backfill utility."""

import json

import pytest

from agents.shared import backfill_ff_prediction as bf
from agents.shared.stability_reducer import U_FF, U_FLOAT


def _report(regions=None, variables=None, chains=None):
    idata = {"class_counts": {}, "regions": regions or {}}
    if variables is not None:
        idata["variables"] = variables
    if chains is not None:
        idata["cascade_chains"] = chains
    return {"kind": "stability_report", "schema_version": 1, "samples_seen": {},
            "integrals": {"A": idata}}


def test_derives_ff_from_sensitivity_exactly():
    reg = {"predicted_rel_err_if_float": U_FLOAT * 1e12, "max_sensitivity": 1e12}
    assert bf._backfill_entry(reg) is True
    assert reg["predicted_rel_err_if_ff"] == pytest.approx(U_FF * 1e12)


def test_derives_ff_from_float_when_no_sensitivity():
    reg = {"predicted_rel_err_if_float": 1e-8}      # no max_sensitivity
    assert bf._backfill_entry(reg) is True
    assert reg["predicted_rel_err_if_ff"] == pytest.approx(1e-8 * (U_FF / U_FLOAT))


def test_idempotent_leaves_existing_ff_untouched():
    reg = {"predicted_rel_err_if_float": 1e-8, "predicted_rel_err_if_ff": 4.2e-14}
    assert bf._backfill_entry(reg) is False
    assert reg["predicted_rel_err_if_ff"] == 4.2e-14


def test_skips_entry_with_no_float_prediction():
    reg = {"max_cond": 1e6}                          # nothing to derive from
    assert bf._backfill_entry(reg) is False
    assert "predicted_rel_err_if_ff" not in reg


def test_backfill_report_covers_regions_variables_and_chains():
    report = _report(
        regions={"f.h:10": {"predicted_rel_err_if_float": 1e-8, "max_sensitivity": 1e0}},
        variables={"v0": {"predicted_rel_err_if_float": 2e-8, "max_sensitivity": 2e0}},
        chains=[{"chain_id": "c1", "predicted_rel_err_if_float": 3e-2,
                 "max_sensitivity": 5e5}],
    )
    updated = bf.backfill_report(report)
    assert updated == 3
    a = report["integrals"]["A"]
    assert a["regions"]["f.h:10"]["predicted_rel_err_if_ff"] == pytest.approx(U_FF * 1e0)
    assert a["variables"]["v0"]["predicted_rel_err_if_ff"] == pytest.approx(U_FF * 2e0)
    assert a["cascade_chains"][0]["predicted_rel_err_if_ff"] == pytest.approx(U_FF * 5e5)


def test_chains_as_dict_are_supported():
    report = _report(chains={"c1": {"chain_id": "c1",
                                    "predicted_rel_err_if_float": 1e-2}})
    # cascade_chains keyed by chain_id (pre-finalize shape)
    report["integrals"]["A"]["cascade_chains"] = {
        "c1": {"chain_id": "c1", "predicted_rel_err_if_float": 1e-2}}
    assert bf.backfill_report(report) == 1


def test_backfill_file_writes_and_is_idempotent(tmp_path):
    p = tmp_path / "report.json"
    p.write_text(json.dumps(_report(
        regions={"f.h:10": {"predicted_rel_err_if_float": 1e-8, "max_sensitivity": 1e0}})))
    assert bf.backfill_file(p) == 1
    reloaded = json.loads(p.read_text())
    assert (reloaded["integrals"]["A"]["regions"]["f.h:10"]["predicted_rel_err_if_ff"]
            == pytest.approx(U_FF * 1e0))
    # second run is a no-op
    assert bf.backfill_file(p) == 0


def test_dry_run_does_not_rewrite(tmp_path):
    p = tmp_path / "report.json"
    original = json.dumps(_report(
        regions={"f.h:10": {"predicted_rel_err_if_float": 1e-8}}))
    p.write_text(original)
    assert bf.backfill_file(p, dry_run=True) == 1
    assert p.read_text() == original     # untouched on disk
