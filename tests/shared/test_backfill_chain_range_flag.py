"""Unit tests for the cascade-chain value_range_ok_for_float backfill (WI1)."""

from agents.shared import backfill_chain_range_flag as bf
from agents.shared.stability_reducer import (
    FLT_MAX, FLT_MIN_NORMAL, chain_range_ok_for_float,
)


def _chain(spans):
    return {"kind": "cascade_chain",
            "chain": [{"file": f, "line_start": a, "line_end": b}
                      for (f, a, b) in spans]}


def _regions(**kv):
    # kv: "file:line" -> value_range_ok_for_float bool
    return {k: {"value_range_ok_for_float": v} for k, v in kv.items()}


def test_chain_safe_when_all_contributor_lines_safe():
    ch = _chain([("A.h", 10, 10), ("B.h", 20, 20)])
    regs = {"A.h:10": {"value_range_ok_for_float": True},
            "B.h:20": {"value_range_ok_for_float": True}}
    assert chain_range_ok_for_float(ch, regs) is True


def test_chain_unsafe_when_any_contributor_line_unsafe():
    ch = _chain([("A.h", 10, 10), ("B.h", 20, 20)])
    regs = {"A.h:10": {"value_range_ok_for_float": True},
            "B.h:20": {"value_range_ok_for_float": False}}
    assert chain_range_ok_for_float(ch, regs) is False


def test_missing_region_defaults_safe_fail_open():
    ch = _chain([("A.h", 10, 10), ("Z.h", 99, 99)])   # Z.h:99 not classified
    regs = {"A.h:10": {"value_range_ok_for_float": True}}
    assert chain_range_ok_for_float(ch, regs) is True


def test_multiline_span_checks_every_line():
    ch = _chain([("A.h", 10, 12)])                     # lines 10,11,12
    regs = {"A.h:10": {"value_range_ok_for_float": True},
            "A.h:11": {"value_range_ok_for_float": False},  # middle line unsafe
            "A.h:12": {"value_range_ok_for_float": True}}
    assert chain_range_ok_for_float(ch, regs) is False


def test_derives_from_abs_val_range_end_to_end():
    # A region whose measured |val| dips below FLT_MIN_NORMAL is range-unsafe, and
    # a chain touching that line inherits the unsafe flag.
    from agents.shared.stability_reducer import _range_ok_for_float
    subnormal = {"abs_val_min": FLT_MIN_NORMAL / 10, "abs_val_max": 1.0}
    assert _range_ok_for_float(subnormal) is False
    huge = {"abs_val_min": 1.0, "abs_val_max": FLT_MAX * 10}
    assert _range_ok_for_float(huge) is False


def test_backfill_stamps_all_chains_and_counts_unsafe():
    report = {"integrals": {"A": {
        "regions": {"A.h:10": {"value_range_ok_for_float": False},
                    "A.h:20": {"value_range_ok_for_float": True}},
        "cascade_chains": [
            _chain([("A.h", 10, 10)]),      # -> unsafe
            _chain([("A.h", 20, 20)]),      # -> safe
        ],
    }}}
    counts = bf.backfill(report)
    assert counts == {"A": 2}
    chains = report["integrals"]["A"]["cascade_chains"]
    assert chains[0]["value_range_ok_for_float"] is False
    assert chains[1]["value_range_ok_for_float"] is True


def test_backfill_idempotent():
    report = {"integrals": {"A": {
        "regions": {"A.h:10": {"value_range_ok_for_float": False}},
        "cascade_chains": [_chain([("A.h", 10, 10)])],
    }}}
    bf.backfill(report)
    first = report["integrals"]["A"]["cascade_chains"][0]["value_range_ok_for_float"]
    bf.backfill(report)
    second = report["integrals"]["A"]["cascade_chains"][0]["value_range_ok_for_float"]
    assert first == second is False
