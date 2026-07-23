"""Tests for the Phase-2b scorer (agents/validator/scorer.py).

Covers the four new-test areas from the handoff:

* **scorer contract** — a known-patched build (synthetic coeff arrays) produces the
  expected delta cells (random + adversarial split, effective = max, scope filter).
* **baseline cache correctness** — the ``baseline_id`` / ``battery_version`` cache
  key changes iff app source + baseline_spec + battery change (and only then).
* **manifest schema round-trip** — write -> read -> row is loss-free (incl.
  forward-compatible unknown fields).
* **status enum coverage** — every enum value is reachable; validation rejects an
  unknown status; the Patcher-status -> cell-status map is exhaustive.

Plus region_id canonicalization stability, over-generation collapse, and manifest
assembly from the iteration log.
"""

from __future__ import annotations

import json
from array import array

import pytest

from agents.validator import scorer as sc
from agents.validator.coeffs import N_COMPONENTS


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _arrays(n_samples: int, fill: float = 1.0):
    """A ``(hi, lo)`` pair of length ``n_samples*6``, hi=fill, lo=0."""
    total = n_samples * N_COMPONENTS
    return (array("d", [fill] * total), array("d", [0.0] * total))


def _set(pair, sample: int, component: int, value: float) -> None:
    hi, _lo = pair
    hi[sample * N_COMPONENTS + component] = value


# ---------------------------------------------------------------------------
# region_id + rung canonicalization (stability)
# ---------------------------------------------------------------------------

def test_region_id_single_and_multi_line():
    assert sc.canonical_region_id("B0m.h", 126) == "B0m.h:126"
    assert sc.canonical_region_id("B0m.h", 126, 126) == "B0m.h:126"
    assert sc.canonical_region_id("B0m.h", 126, 130) == "B0m.h:126-130"


def test_region_id_matches_region_target_location():
    """region_id must be byte-identical to the Strategy RegionTarget.location it is
    plumbed from — that identity is the whole cross-run stability guarantee."""
    from agents.strategy.models import RegionTarget
    single = RegionTarget("B0m.h", 126, 126)
    multi = RegionTarget("B0m.h", 126, 130)
    assert single.location == sc.canonical_region_id("B0m.h", 126, 126)
    assert multi.location == sc.canonical_region_id("B0m.h", 126, 130)


def test_rung_from_kind():
    assert sc.rung_from_kind("double-to-dd") == "dd"
    assert sc.rung_from_kind("double-to-float") == "float"
    assert sc.rung_from_kind("ff-to-double") == "double"
    assert sc.rung_from_kind("reformulate-kahan") == "reformulate-kahan"
    assert sc.rung_from_kind("reformulate-identity") == "reformulate-identity"


# ---------------------------------------------------------------------------
# scorer contract — deltas from known arrays
# ---------------------------------------------------------------------------

def test_delta_over_arrays_p100_max():
    ref = _arrays(2)
    cand = _arrays(2)
    _set(cand, 0, 0, 1.0 + 1e-14)   # tiny error, sample 0
    _set(cand, 1, 0, 1.0 + 1e-9)    # larger error, sample 1 -> the p100
    d = sc.delta_over_arrays({"B1": cand}, {"B1": ref}, ["B1"])
    assert d == pytest.approx(1e-9, rel=1e-3)


def test_delta_scope_filters_other_integrals():
    ref = _arrays(1)
    cand_b1 = _arrays(1)
    _set(cand_b1, 0, 0, 1.0 + 1e-14)
    cand_b2 = _arrays(1)
    _set(cand_b2, 0, 0, 1.0 + 1e-3)   # B2 much worse — must NOT leak into a B1 cell
    d = sc.delta_over_arrays(
        {"B1": cand_b1, "B2": cand_b2}, {"B1": ref, "B2": ref}, ["B1"])
    assert d == pytest.approx(1e-14, rel=1e-2)


def test_delta_none_scope_is_whole_app_identity():
    ref = _arrays(1)
    cand_b1 = _arrays(1)
    _set(cand_b1, 0, 0, 1.0 + 1e-14)
    cand_b2 = _arrays(1)
    _set(cand_b2, 0, 0, 1.0 + 1e-3)
    # empty scope -> reduce over all integrals (the default identity), so the worst
    # integral (B2) sets the delta — the pre-2b whole-app behavior.
    d = sc.delta_over_arrays(
        {"B1": cand_b1, "B2": cand_b2}, {"B1": ref, "B2": ref}, None)
    assert d == pytest.approx(1e-3, rel=1e-2)


def test_delta_analytic_zero_excluded():
    """A DD reference at the per-sample noise floor is an analytic zero and must
    contribute 0 (relative metric undefined), never a spurious huge rel-err."""
    ref = _arrays(1)
    # component 1 is an analytic zero relative to the scale (comp 0 = 1.0)
    ref[0][1] = 1e-30
    cand = _arrays(1)
    cand[0][1] = 1e-20   # nonzero roundoff around the zero
    d = sc.delta_over_arrays({"B1": cand}, {"B1": ref}, ["B1"])
    assert d == 0.0


def test_delta_over_tail_and_effective_max():
    ref = _arrays(1)
    cand = _arrays(1)
    _set(cand, 0, 0, 1.0 + 1e-12)                 # random slice -> 1e-12
    dd_tail = {"B1": {5: [(1.0, 0.0)] * N_COMPONENTS}}
    cand_tail = {"B1": {5: [(1.0 + 1e-7, 0.0)] + [(1.0, 0.0)] * (N_COMPONENTS - 1)}}
    row = sc.score_cell(
        region_id="B0m.h:126", rung="dd", iteration_id=0,
        candidate_coeffs={"B1": cand}, dd_ref_coeffs={"B1": ref},
        integrals_scope=["B1"], candidate_tail=cand_tail, dd_ref_tail=dd_tail)
    assert row.delta_random == pytest.approx(1e-12, rel=1e-2)
    assert row.delta_adversarial == pytest.approx(1e-7, rel=1e-2)
    # effective = max(adversarial, random)
    assert row.delta_effective == row.delta_adversarial


def test_adversarial_null_when_tail_empty():
    ref = _arrays(1)
    cand = _arrays(1)
    _set(cand, 0, 0, 1.0 + 1e-12)
    row = sc.score_cell(
        region_id="B0m.h:126", rung="dd", iteration_id=0,
        candidate_coeffs={"B1": cand}, dd_ref_coeffs={"B1": ref},
        integrals_scope=["B1"])
    assert row.delta_adversarial is None
    assert row.delta_effective == row.delta_random


def test_baseline_delta_exposes_inertness():
    """When the candidate output equals the unpatched baseline, delta_effective ==
    baseline_delta_effective — the manifest's one-field inertness signal."""
    ref = _arrays(1)
    inert = _arrays(1)
    _set(inert, 0, 0, 1.0 + 1e-13)   # candidate == baseline (both this value)
    baseline = _arrays(1)
    _set(baseline, 0, 0, 1.0 + 1e-13)
    row = sc.score_cell(
        region_id="B0m.h:126", rung="dd", iteration_id=0,
        candidate_coeffs={"B1": inert}, dd_ref_coeffs={"B1": ref},
        integrals_scope=["B1"], baseline_coeffs={"B1": baseline})
    assert row.delta_effective == row.baseline_delta_effective   # inert patch

    # a genuinely improving candidate: delta below baseline_delta
    better = _arrays(1)
    _set(better, 0, 0, 1.0 + 1e-16)
    row2 = sc.score_cell(
        region_id="B0m.h:126", rung="dd", iteration_id=0,
        candidate_coeffs={"B1": better}, dd_ref_coeffs={"B1": ref},
        integrals_scope=["B1"], baseline_coeffs={"B1": baseline})
    assert row2.delta_effective < row2.baseline_delta_effective


def test_baseline_delta_null_when_omitted():
    ref = _arrays(1)
    cand = _arrays(1)
    _set(cand, 0, 0, 1.0 + 1e-12)
    row = sc.score_cell(
        region_id="B0m.h:126", rung="dd", iteration_id=0,
        candidate_coeffs={"B1": cand}, dd_ref_coeffs={"B1": ref},
        integrals_scope=["B1"])
    assert row.baseline_delta_effective is None


def test_effective_delta_helper():
    assert sc.effective_delta(1e-7, 1e-12) == 1e-7
    assert sc.effective_delta(None, 1e-12) == 1e-12
    assert sc.effective_delta(1e-7, None) == 1e-7
    assert sc.effective_delta(None, None) is None


# ---------------------------------------------------------------------------
# baseline cache correctness — key changes iff (spec, app source, battery) change
# ---------------------------------------------------------------------------

def test_baseline_id_stable_and_sensitive():
    spec_dd = sc.qcdloop_baseline_spec()
    spec_double = sc.default_baseline_spec()
    a = sc.baseline_id(spec_dd, "appsrc", "ddsrc")
    # identical inputs -> identical id (cache HIT)
    assert a == sc.baseline_id(spec_dd, "appsrc", "ddsrc")
    # different spec -> different id (cache MISS)
    assert a != sc.baseline_id(spec_double, "appsrc", "ddsrc")
    # different app source -> different id
    assert a != sc.baseline_id(spec_dd, "appsrc2", "ddsrc")
    # different DD oracle source -> different id
    assert a != sc.baseline_id(spec_dd, "appsrc", "ddsrc2")


def test_battery_version_stable_and_sensitive():
    b1 = sc.snapshot_battery_spec({"seed": 12345, "sample_count": 5000})
    b1b = sc.snapshot_battery_spec({"seed": 12345, "sample_count": 5000})
    b2 = sc.snapshot_battery_spec({"seed": 12345, "sample_count": 10000})
    b3 = sc.snapshot_battery_spec({"seed": 999, "sample_count": 5000})
    b4 = sc.snapshot_battery_spec({"seed": 12345, "sample_count": 5000},
                                  adversarial_offsets=[7, 9])
    assert b1["version"] == b1b["version"]        # identical -> HIT
    assert b1["version"] != b2["version"]         # count change -> MISS
    assert b1["version"] != b3["version"]         # seed change -> MISS
    assert b1["version"] != b4["version"]         # adversarial change -> MISS
    assert b1["adversarial"] == [] and b4["adversarial"]  # 2b stub vs populated


# ---------------------------------------------------------------------------
# manifest schema round-trip
# ---------------------------------------------------------------------------

def test_manifest_round_trip(tmp_path):
    rows = [
        sc.ManifestRow(region_id="B0m.h:126", rung="dd", iteration_id=0,
                       status=sc.STATUS_MEASURED, delta_random=1e-14,
                       baseline_id="b", battery_version="v", intent_id=8,
                       integrals_scope=["B1"]),
        sc.ManifestRow(region_id="B0m.h:126", rung="float", iteration_id=0,
                       status=sc.STATUS_PATCHER_FAILED, intent_id=6,
                       patcher_metadata={"patcher_status": "llm_gen_failed"}),
    ]
    path = tmp_path / "m.jsonl"
    sc.write_rows(path, rows)
    back = sc.read_rows(path)
    assert len(back) == 2
    assert back[0]["region_id"] == "B0m.h:126"
    assert back[0]["delta_effective"] == 1e-14
    assert back[1]["status"] == sc.STATUS_PATCHER_FAILED
    # to_dict -> from_dict -> to_dict is idempotent
    r0 = sc.ManifestRow.from_dict(back[0])
    assert r0.to_dict() == back[0]


def test_manifest_preserves_unknown_forward_fields(tmp_path):
    path = tmp_path / "m.jsonl"
    raw = {"region_id": "X:1", "rung": "dd", "iteration_id": 0,
           "status": sc.STATUS_MEASURED, "future_field": 42}
    path.write_text(json.dumps(raw) + "\n")
    rows = sc.read_rows(path)
    row = sc.ManifestRow.from_dict(rows[0])
    # unknown field is preserved (not dropped) under patcher_metadata._extra
    assert row.patcher_metadata["_extra"]["future_field"] == 42


def test_append_row_accumulates(tmp_path):
    path = tmp_path / "m.jsonl"
    sc.append_row(path, sc.ManifestRow(region_id="A:1", rung="dd", iteration_id=0,
                                       status=sc.STATUS_MEASURED))
    sc.append_row(path, sc.ManifestRow(region_id="A:2", rung="ff", iteration_id=0,
                                       status=sc.STATUS_MEASURED))
    assert len(sc.read_rows(path)) == 2


# ---------------------------------------------------------------------------
# status enum coverage
# ---------------------------------------------------------------------------

def test_all_statuses_constructable():
    for status in sc.STATUSES:
        row = sc.ManifestRow(region_id="A:1", rung="dd", iteration_id=0,
                             status=status)
        assert row.status == status


def test_unknown_status_rejected():
    with pytest.raises(ValueError):
        sc.ManifestRow(region_id="A:1", rung="dd", iteration_id=0, status="bogus")
    with pytest.raises(ValueError):
        sc.validate_row({"region_id": "A:1", "rung": "dd", "iteration_id": 0,
                         "status": "bogus"})


def test_empty_region_id_rejected():
    with pytest.raises(ValueError):
        sc.ManifestRow(region_id="", rung="dd", iteration_id=0,
                       status=sc.STATUS_MEASURED)


def test_cell_status_map_covers_fanout_modes():
    assert sc.cell_status_for("ok") == sc.STATUS_MEASURED
    assert sc.cell_status_for("llm_gen_failed") == sc.STATUS_PATCHER_FAILED
    assert sc.cell_status_for("build_failed") == sc.STATUS_BUILD_FAILED
    assert sc.cell_status_for("call_graph_build_failed") == sc.STATUS_BUILD_FAILED
    assert sc.cell_status_for("ok", "silent_bypass") == sc.STATUS_WIRE_FAILED
    assert sc.cell_status_for("ok", "variant_name_collision") == sc.STATUS_WIRE_FAILED
    # unknown -> conservative patcher_failed
    assert sc.cell_status_for("something_new") == sc.STATUS_PATCHER_FAILED
    assert sc.cell_status_for(None) == sc.STATUS_PATCHER_FAILED


# ---------------------------------------------------------------------------
# collapse (fan-out over-generation) + iteration-log assembly
# ---------------------------------------------------------------------------

def test_collapse_min_delta_per_key():
    rows = [
        {"region_id": "A:1", "rung": "dd", "status": "measured",
         "delta_effective": 1e-9},
        {"region_id": "A:1", "rung": "dd", "status": "measured",
         "delta_effective": 1e-14},   # better -> wins the collapse
        {"region_id": "A:1", "rung": "float", "status": "measured",
         "delta_effective": 1e-6},
    ]
    best = sc.collapse_min_delta(rows)
    assert best[("A:1", "dd")]["delta_effective"] == 1e-14
    assert best[("A:1", "float")]["delta_effective"] == 1e-6


def test_collapse_measured_beats_non_measured():
    rows = [
        {"region_id": "A:1", "rung": "float", "status": "patcher_failed",
         "delta_effective": None},
        {"region_id": "A:1", "rung": "float", "status": "measured",
         "delta_effective": 1e-6},
    ]
    best = sc.collapse_min_delta(rows)
    assert best[("A:1", "float")]["status"] == "measured"


def test_rows_from_iteration_log_classifies():
    iters = [
        {"target": {"file": "B0m.h", "line_start": 126, "line_end": 126},
         "kind": "double-to-dd", "patcher_status": "ok", "accepted": False,
         "validator_verdict": "reject", "iter_id": 8},          # measured elsewhere
        {"target": {"file": "B0m.h", "line_start": 126, "line_end": 126},
         "kind": "double-to-float", "patcher_status": "llm_gen_failed",
         "accepted": False, "iter_id": 6},                       # patcher_failed
        {"target": {"file": "B0m.h", "line_start": 200, "line_end": 200},
         "kind": "double-to-float", "patcher_status": "build_failed",
         "accepted": False, "iter_id": 7},                       # build_failed
        {"target": {"file": "B0m.h", "line_start": 300, "line_end": 300},
         "kind": "double-to-ff", "patcher_status": "ok", "accepted": True,
         "iter_id": 9},                                          # accepted -> skip
    ]
    rows = sc.rows_from_iteration_log(iters, iteration_id=0)
    by = {(r.region_id, r.rung): r for r in rows}
    # ok (dd-ceiling retain) is skipped here (its measured cell exists)
    assert ("B0m.h:126", "dd") not in by
    assert by[("B0m.h:126", "float")].status == sc.STATUS_PATCHER_FAILED
    assert by[("B0m.h:200", "float")].status == sc.STATUS_BUILD_FAILED
    # accepted skipped
    assert ("B0m.h:300", "ff") not in by


def test_rows_from_iteration_log_respects_measured_keys():
    iters = [
        {"target": {"file": "B0m.h", "line_start": 126, "line_end": 126},
         "kind": "double-to-float", "patcher_status": "llm_gen_failed",
         "accepted": False, "iter_id": 6},
    ]
    rows = sc.rows_from_iteration_log(
        iters, measured_keys={("B0m.h:126", "float")})
    assert rows == []


def test_assemble_manifest_merges(tmp_path):
    scored = tmp_path / "scored.jsonl"
    sc.write_rows(scored, [
        sc.ManifestRow(region_id="B0m.h:126", rung="dd", iteration_id=0,
                       status=sc.STATUS_MEASURED, delta_random=1e-14,
                       baseline_id="bid", battery_version="bver",
                       integrals_scope=["B1"], intent_id=8),
    ])
    iter_log = tmp_path / "iterations.jsonl"
    iter_log.write_text("\n".join(json.dumps(r) for r in [
        {"target": {"file": "B0m.h", "line_start": 126, "line_end": 126},
         "kind": "double-to-dd", "patcher_status": "ok", "accepted": False,
         "validator_verdict": "reject", "iter_id": 8},
        {"target": {"file": "B0m.h", "line_start": 126, "line_end": 126},
         "kind": "double-to-float", "patcher_status": "llm_gen_failed",
         "accepted": False, "iter_id": 6},
    ]) + "\n")
    out = tmp_path / "manifest_scorer.jsonl"
    merged = sc.assemble_manifest(scored, iter_log, out, iteration_id=0)
    assert out.is_file()
    keys = {(r["region_id"], r["rung"]) for r in merged}
    assert ("B0m.h:126", "dd") in keys       # measured
    assert ("B0m.h:126", "float") in keys     # patcher_failed folded in
    failed = next(r for r in merged if r["rung"] == "float")
    assert failed["status"] == sc.STATUS_PATCHER_FAILED
    # failure cell inherits the same cache key as the measured cells
    assert failed["baseline_id"] == "bid"
    assert failed["battery_version"] == "bver"
