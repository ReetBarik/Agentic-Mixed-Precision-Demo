"""Tail-sample battery: determinism check, hard-reject scoring, and fail-open.

These exercise the pure-Python tail logic (no Kokkos / driver builds) by feeding
synthetic candidate/DD coeff structures and monkeypatching the two seams that
would otherwise shell out to a compiled driver (``tail.determinism_hash`` and
``validate._dd_tail_coeffs``).
"""

import struct

import pytest

import importlib

from agents.validator import tail
# package re-exports the validate() fn, shadowing the submodule -> use import_module
validate = importlib.import_module("agents.validator.validate")
from agents.validator.coeffs import COMPONENT_LABELS
from agents.validator.precise_digits import MAX_DIGITS_F


def _pair(x: float):
    """A (hi, lo) component pair for a plain double (lo == 0), like vanilla."""
    return (x, 0.0)


def _round_trip(x: float) -> float:
    """double -> IEEE bits -> double (identity; keeps values representable)."""
    return struct.unpack("<d", struct.pack("<d", x))[0]


# ---------------------------------------------------------------------------
# Determinism check
# ---------------------------------------------------------------------------

def test_determinism_mismatch_raises_loudly(monkeypatch):
    # The candidate binary's regenerated hash disagrees with the report's frozen
    # hash for B12 -> hard, loud failure; never a silent fall-back.
    monkeypatch.setattr(tail, "determinism_hash",
                        lambda binary, n=tail.DETERMINISM_N:
                        {"B12": "sha256:actualBADhash", "B1": "sha256:ok"})
    expected = {"B12": "sha256:frozenGOODhash", "B1": "sha256:ok"}
    with pytest.raises(tail.DeterminismMismatch) as ei:
        tail.verify_determinism("fake_bin", expected, ["B12", "B1"])
    assert ei.value.integral == "B12"
    assert "DETERMINISM_MISMATCH: B12" in str(ei.value)


def test_determinism_match_passes(monkeypatch):
    monkeypatch.setattr(tail, "determinism_hash",
                        lambda binary, n=tail.DETERMINISM_N:
                        {"B12": "sha256:h", "B1": "sha256:h2"})
    # no exception
    tail.verify_determinism("fake_bin", {"B12": "sha256:h", "B1": "sha256:h2"},
                            ["B12", "B1"])


def test_determinism_missing_expected_is_skipped(monkeypatch):
    # An integral absent from the report's expected map is the caller's fail-open
    # responsibility; verify_determinism must not raise on it.
    monkeypatch.setattr(tail, "determinism_hash",
                        lambda binary, n=tail.DETERMINISM_N: {"B1": "sha256:x"})
    tail.verify_determinism("fake_bin", {"B1": "sha256:x"}, ["B1", "B99"])


# ---------------------------------------------------------------------------
# offset helpers
# ---------------------------------------------------------------------------

def test_offset_union_dedups_across_criteria():
    ts = {
        "max_rel_err":   [{"offset": 5}, {"offset": 42}],
        "max_cond":      [{"offset": 42}, {"offset": 7}],
        "max_abs_value": [{"offset": 100}],
        "min_abs_value": [{"offset": 5}],
    }
    assert tail.integral_offsets(ts) == [5, 7, 42, 100]
    assert tail.all_offsets({"B12": ts, "B1": {"max_rel_err": [{"offset": 1}]}}) \
        == [1, 5, 7, 42, 100]


# ---------------------------------------------------------------------------
# Tail scoring — hard reject
# ---------------------------------------------------------------------------

def test_score_tail_flags_failing_offset():
    # DD reference ~1.0 on coeff0.real; candidate is off by 1e-3 at offset 42
    # (3 digits) but exact at offset 5 -> tail min ~3 digits, hotspot at 42.
    tail_spec = {"B12": {"max_rel_err": [{"offset": 5}, {"offset": 42}]}}
    good = [_pair(1.0)] + [_pair(0.0)] * 5
    dd = {"B12": {5: good, 42: [_pair(1.0)] + [_pair(0.0)] * 5}}
    cand = {
        "B12": {
            5: good,                                   # exact -> MAX_DIGITS
            42: [_pair(1.0 + 1e-3)] + [_pair(0.0)] * 5,  # rel err 1e-3 -> ~3 digits
        }
    }
    stats = validate._score_tail(cand, dd, tail_spec)
    assert stats["tail_samples_tested"] == 2
    assert stats["integrals_covered"] == ["B12"]
    assert 2.9 < stats["tail_min_precise_digits"] < 3.1
    assert stats["tail_hotspot"]["offset"] == 42
    assert stats["tail_hotspot"]["component"] == COMPONENT_LABELS[0]


def test_tail_failure_is_hard_reject(monkeypatch):
    # A candidate that would pass the random battery (curr/cand min ~12) but fails
    # a tail offset (3 digits) must be rejected once the tail min is folded in.
    tail_spec = {"B12": {"max_rel_err": [{"offset": 42}]}}
    dd = {"B12": {42: [_pair(1.0)] + [_pair(0.0)] * 5}}
    cand = {"B12": {42: [_pair(1.0 + 1e-3)] + [_pair(0.0)] * 5}}
    monkeypatch.setattr(validate, "_dd_tail_coeffs",
                        lambda *a, **k: dd)
    stats = validate._tail_battery(tail_spec, cand, [42],
                                   "repo", "ref", "kok", "ddhash")
    assert stats["tail_batteries_run"] == 1
    assert stats["tail_samples_tested"] == 1
    tail_min = stats["tail_min_precise_digits"]
    assert 2.9 < tail_min < 3.1
    # random battery said 12 digits; combined min drops to ~3 -> reject at tol 7.
    combined = min(12.0, tail_min)
    verdict, reason = validate._decide(combined, 12.0, 0.5, floor=7.0)
    assert verdict == "reject"


def test_passing_tail_does_not_lower_min(monkeypatch):
    # Precision-preserving candidate: exact on the tail offset -> tail min at the
    # cap, combined min unchanged, verdict accepts.
    tail_spec = {"B12": {"max_rel_err": [{"offset": 42}]}}
    good = [_pair(1.0)] + [_pair(0.0)] * 5
    monkeypatch.setattr(validate, "_dd_tail_coeffs",
                        lambda *a, **k: {"B12": {42: good}})
    stats = validate._tail_battery(tail_spec, {"B12": {42: good}}, [42],
                                   "repo", "ref", "kok", "ddhash")
    assert stats["tail_min_precise_digits"] == pytest.approx(round(MAX_DIGITS_F, 4))
    combined = min(9.0, stats["tail_min_precise_digits"])
    verdict, reason = validate._decide(combined, 9.0, 0.5, floor=7.0)
    assert verdict == "accept"


# ---------------------------------------------------------------------------
# Fail-open
# ---------------------------------------------------------------------------

def test_fail_open_no_tail_spec():
    stats = validate._tail_battery({}, None, [], "repo", "ref", "kok", "ddhash")
    assert stats["tail_batteries_run"] == 0
    assert stats["tail_min_precise_digits"] is None
    assert stats["tail_samples_tested"] == 0


def test_fail_open_missing_integral_warns_once(monkeypatch, capsys):
    # Report lists tail_samples for B12 and B99, but the driver only produced
    # coeffs for B12 (B99 e.g. absent) -> B99 skipped with a one-time warning,
    # B12 still scored.
    tail_spec = {
        "B12": {"max_rel_err": [{"offset": 1}]},
        "B99": {"max_rel_err": [{"offset": 1}]},
    }
    good = [_pair(1.0)] + [_pair(0.0)] * 5
    monkeypatch.setattr(validate, "_dd_tail_coeffs",
                        lambda *a, **k: {"B12": {1: good}})
    # reset the module-level warn set for a clean assertion
    validate._TAIL_WARNED.clear()
    stats = validate._tail_battery(tail_spec, {"B12": {1: good}}, [1],
                                   "repo", "ref", "kok", "ddhash")
    assert stats["tail_batteries_run"] == 1
    assert "B99" in stats["integrals_skipped"]
    out = capsys.readouterr().out
    assert out.count("skipping B99") == 1
    # a second run does not re-warn for B99
    validate._tail_battery(tail_spec, {"B12": {1: good}}, [1],
                           "repo", "ref", "kok", "ddhash")
    out2 = capsys.readouterr().out
    assert "skipping B99" not in out2
