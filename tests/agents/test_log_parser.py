"""Unit tests for agents.characterizer.log_parser — no LLM, no build."""

import json
import tempfile
from pathlib import Path

import pytest

from agents.characterizer.log_parser import parse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _tmp_journal(records: list[dict]) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    path = Path(tmp.name)
    tmp.close()
    _write_jsonl(records, path)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_empty_journal():
    path = _tmp_journal([])
    profile = parse(path, kernel_name="k", flag_threshold=1e8)
    assert profile.kernel == "k"
    assert profile.samples_run == 0
    assert profile.per_op == []
    assert profile.per_line == {}
    assert profile.per_variable == {}
    assert profile.top_hotspots == []
    assert profile.opaque_coverage == 0.0
    assert profile.notes == []


def test_single_op():
    records = [{"op": "add", "loc": "k.cpp:fn:5", "cond": 2.5, "rel_err": 1e-10, "prov": ["x"]}]
    path = _tmp_journal(records)
    profile = parse(path, kernel_name="k", flag_threshold=1e8)
    assert len(profile.per_op) == 1
    rec = profile.per_op[0]
    assert rec.op == "add"
    assert rec.location == "k.cpp:fn:5"
    assert rec.max_cond == pytest.approx(2.5)
    assert rec.max_rel_err == pytest.approx(1e-10)
    assert rec.sample_count == 1
    assert rec.provenance_union == {"x"}
    assert not rec.flagged


def test_same_location_rollup():
    """Multiple records with the same (op, loc) should be merged into one OpRecord."""
    records = [
        {"op": "sub", "loc": "f.cpp:g:10", "cond": 1e6,  "rel_err": 1e-8, "prov": ["a"]},
        {"op": "sub", "loc": "f.cpp:g:10", "cond": 1e9,  "rel_err": 1e-6, "prov": ["b"]},
        {"op": "sub", "loc": "f.cpp:g:10", "cond": 5e7,  "rel_err": 1e-9, "prov": ["a", "c"]},
    ]
    path = _tmp_journal(records)
    profile = parse(path, kernel_name="k", flag_threshold=1e8)
    assert len(profile.per_op) == 1
    rec = profile.per_op[0]
    assert rec.max_cond == pytest.approx(1e9)
    assert rec.max_rel_err == pytest.approx(1e-6)
    assert rec.sample_count == 3
    assert rec.provenance_union == {"a", "b", "c"}
    assert rec.flagged    # 1e9 > 1e8 threshold


def test_multiple_ops_sorted_by_cond():
    """per_op must be sorted by max_cond descending."""
    records = [
        {"op": "add", "loc": "f.cpp:g:1", "cond": 10.0,  "rel_err": 0.0, "prov": []},
        {"op": "sub", "loc": "f.cpp:g:2", "cond": 1e10,  "rel_err": 0.0, "prov": []},
        {"op": "mul", "loc": "f.cpp:g:3", "cond": 100.0, "rel_err": 0.0, "prov": []},
    ]
    path = _tmp_journal(records)
    profile = parse(path, kernel_name="k", flag_threshold=1e8)
    assert [r.op for r in profile.per_op] == ["sub", "mul", "add"]


def test_sample_count_aggregation():
    """sample_count must equal the number of individual records for a key."""
    records = [{"op": "add", "loc": "f.cpp:g:1", "cond": float(i), "rel_err": 0.0, "prov": []}
               for i in range(1, 7)]
    path = _tmp_journal(records)
    profile = parse(path, kernel_name="k", flag_threshold=1e8)
    assert profile.per_op[0].sample_count == 6
    assert profile.per_op[0].max_cond == 6.0


def test_heavy_opaque_coverage_note():
    """When >50 % of records are opaque, a note must be added."""
    records = [
        {"op": "opaque", "loc": "f.cpp:g:1", "cond": 1.0, "rel_err": 0.0, "prov": ["Kokkos::log"]},
        {"op": "opaque", "loc": "f.cpp:g:2", "cond": 1.0, "rel_err": 0.0, "prov": []},
        {"op": "add",    "loc": "f.cpp:g:3", "cond": 2.0, "rel_err": 0.0, "prov": []},
    ]
    path = _tmp_journal(records)
    profile = parse(path, kernel_name="k", flag_threshold=1e8)
    assert profile.opaque_coverage == pytest.approx(2 / 3)
    assert any("opaque" in note.lower() for note in profile.notes)


def test_opaque_coverage_below_threshold_no_note():
    records = [
        {"op": "opaque", "loc": "f.cpp:g:1", "cond": 1.0, "rel_err": 0.0, "prov": []},
        {"op": "add",    "loc": "f.cpp:g:2", "cond": 2.0, "rel_err": 0.0, "prov": []},
        {"op": "mul",    "loc": "f.cpp:g:3", "cond": 3.0, "rel_err": 0.0, "prov": []},
    ]
    path = _tmp_journal(records)
    profile = parse(path, kernel_name="k", flag_threshold=1e8)
    assert profile.opaque_coverage == pytest.approx(1 / 3)
    assert profile.notes == []


def test_provenance_union_and_per_variable():
    """per_variable should map each variable to the max cond it appeared in."""
    records = [
        {"op": "add", "loc": "f.cpp:g:1", "cond": 5.0,   "rel_err": 0.0, "prov": ["x", "y"]},
        {"op": "sub", "loc": "f.cpp:g:2", "cond": 100.0,  "rel_err": 0.0, "prov": ["y", "z"]},
    ]
    path = _tmp_journal(records)
    profile = parse(path, kernel_name="k", flag_threshold=1e8)
    assert profile.per_variable["x"] == pytest.approx(5.0)
    assert profile.per_variable["y"] == pytest.approx(100.0)   # max across two ops
    assert profile.per_variable["z"] == pytest.approx(100.0)


def test_per_line_keeps_worst():
    """per_line should retain the worst (highest cond) record per location."""
    records = [
        {"op": "add", "loc": "f.cpp:g:5", "cond": 10.0,  "rel_err": 0.0, "prov": []},
        {"op": "sub", "loc": "f.cpp:g:5", "cond": 1e9,   "rel_err": 0.0, "prov": []},
    ]
    path = _tmp_journal(records)
    profile = parse(path, kernel_name="k", flag_threshold=1e8)
    assert "f.cpp:g:5" in profile.per_line
    assert profile.per_line["f.cpp:g:5"].max_cond == pytest.approx(1e9)


def test_top_hotspots_respects_n():
    records = [{"op": "add", "loc": f"f.cpp:g:{i}", "cond": float(i), "rel_err": 0.0, "prov": []}
               for i in range(1, 16)]
    path = _tmp_journal(records)
    profile = parse(path, kernel_name="k", flag_threshold=1e8, top_n=5)
    assert len(profile.top_hotspots) == 5
    assert profile.top_hotspots[0].max_cond == 15.0   # sorted desc


def test_flagged_threshold():
    records = [
        {"op": "sub", "loc": "f.cpp:g:1", "cond": 1e8 - 1, "rel_err": 0.0, "prov": []},
        {"op": "sub", "loc": "f.cpp:g:2", "cond": 1e8,     "rel_err": 0.0, "prov": []},
        {"op": "sub", "loc": "f.cpp:g:3", "cond": 1e8 + 1, "rel_err": 0.0, "prov": []},
    ]
    path = _tmp_journal(records)
    profile = parse(path, kernel_name="k", flag_threshold=1e8)
    flagged = {r.location: r.flagged for r in profile.per_op}
    assert not flagged["f.cpp:g:1"]
    assert not flagged["f.cpp:g:2"]  # strictly greater than threshold
    assert flagged["f.cpp:g:3"]


def test_missing_loc_field_defaults_to_empty_string():
    records = [{"op": "add", "cond": 3.0, "rel_err": 0.0, "prov": ["a"]}]
    path = _tmp_journal(records)
    profile = parse(path, kernel_name="k", flag_threshold=1e8)
    assert profile.per_op[0].location == ""
    # Records with empty location are NOT added to per_line
    assert "" not in profile.per_line


def test_alternate_field_names():
    """Parser should accept 'location' as an alias for 'loc', 'provenance' for 'prov'."""
    records = [{"op": "mul", "location": "f.cpp:g:7", "cond": 4.0, "rel_err": 0.0, "provenance": ["p"]}]
    path = _tmp_journal(records)
    profile = parse(path, kernel_name="k", flag_threshold=1e8)
    assert profile.per_op[0].location == "f.cpp:g:7"
    assert profile.per_op[0].provenance_union == {"p"}
