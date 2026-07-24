"""Unit tests — solver candidate queue construction + rank order (Phase 2e)."""

from agents.solver.queue import Candidate, RUNG_RANK, build_queue


def _row(region_id, rung, de, bde, status="measured", kind=None, intent="speedup",
         via="regional"):
    return {
        "region_id": region_id, "rung": rung, "status": status,
        "delta_effective": de, "baseline_delta_effective": bde,
        "patcher_metadata": {"kind": kind or f"double-to-{rung}",
                             "intent": intent, "via": via},
        "intent_id": 0,
    }


# --- DISCRIM / INERT filtering -------------------------------------------------
def test_measured_discrim_enters_queue():
    rows = [_row("A.h:10", "float", de=1e-7, bde=1e-13)]
    qb = build_queue(rows)
    assert [c.region_id for c in qb.queue] == ["A.h:10"]
    assert qb.inert == []


def test_measured_inert_excluded_but_recorded():
    # delta_effective == baseline -> byte-identical whole-app output -> no-op
    rows = [_row("A.h:10", "float", de=2.5e-4, bde=2.5e-4)]
    qb = build_queue(rows)
    assert qb.queue == []
    assert [c.region_id for c in qb.inert] == ["A.h:10"]


def test_non_measured_rows_excluded_and_recorded():
    rows = [_row("A.h:10", "dd", de=None, bde=None, status="patcher_failed"),
            _row("B.h:20", "float", de=None, bde=None, status="write_truncation")]
    qb = build_queue(rows)
    assert qb.queue == []
    assert len(qb.non_measured) == 2


def test_region_with_only_inert_rung_stays_double():
    rows = [_row("A.h:10", "float", de=2.5e-4, bde=2.5e-4),   # INERT
            _row("A.h:10", "ff", de=2.5e-4, bde=2.5e-4)]       # INERT
    qb = build_queue(rows)
    assert qb.queue == []
    assert "A.h:10" not in qb.regions_in_queue


# --- rank order ---------------------------------------------------------------
def test_rank_order_float_ff_dd():
    rows = [_row("A.h:10", "dd", de=1e-30, bde=1e-13),
            _row("A.h:10", "ff", de=1e-14, bde=1e-13),
            _row("A.h:10", "float", de=1e-7, bde=1e-13)]
    qb = build_queue(rows)
    assert [c.rung for c in qb.queue] == ["float", "ff", "dd"]


def test_rank_tiebreak_region_id_ascending_within_rung():
    rows = [_row("Z.h:1", "float", de=1e-7, bde=1e-13),
            _row("A.h:9", "float", de=1e-7, bde=1e-13),
            _row("A.h:100", "float", de=1e-7, bde=1e-13)]
    qb = build_queue(rows)
    # string sort: "A.h:100" < "A.h:9" < "Z.h:1"
    assert [c.region_id for c in qb.queue] == ["A.h:100", "A.h:9", "Z.h:1"]


def test_dedup_region_rung_keeps_most_improving():
    rows = [_row("A.h:10", "float", de=9e-7, bde=1e-13),
            _row("A.h:10", "float", de=2e-7, bde=1e-13)]   # smaller delta = better
    qb = build_queue(rows)
    assert len(qb.queue) == 1
    assert qb.queue[0].delta_effective == 2e-7


def test_unknown_rung_ignored():
    rows = [_row("A.h:10", "quad", de=1e-7, bde=1e-13)]
    qb = build_queue(rows)
    assert qb.queue == []
    assert len(qb.non_measured) == 1


def test_candidate_metadata_carried_through():
    rows = [_row("A.h:10", "float", de=1e-7, bde=1e-13,
                 kind="double-to-float", intent="speedup", via="regional")]
    c = build_queue(rows).queue[0]
    assert (c.kind, c.intent, c.via) == ("double-to-float", "speedup", "regional")
    assert c.is_discrim is True
    assert c.rank == RUNG_RANK["float"]


# --- Phase 2f: chain_dd tier --------------------------------------------------
def _chain_row(chain_id, predicted_lift, de=1e-30, bde=1e-4, lines=None):
    return {
        "region_id": chain_id, "rung": "chain_dd", "status": "measured",
        "delta_effective": de, "baseline_delta_effective": bde,
        "patcher_metadata": {"kind": "double-to-dd", "intent": "correctness",
                             "via": "chain"},
        "chain_lines": lines or [["f.h", 10, 10]],
        "predicted_lift": predicted_lift, "intent_id": 0,
    }


def test_chain_dd_is_tier1_ranked_by_lift_then_single_region():
    rows = [_row("A.h:10", "float", de=1e-7, bde=1e-13),
            _chain_row("cascade_small", 5.0),
            _chain_row("cascade_big", 18.0)]
    qb = build_queue(rows)
    # chains first (tier-1) by predicted_lift desc, then the single-region float (tier-2)
    assert [c.region_id for c in qb.queue] == ["cascade_big", "cascade_small", "A.h:10"]
    assert qb.queue[0].is_chain and qb.queue[0].rung == "chain_dd"


def test_chain_dd_carries_lift_and_lines():
    rows = [_chain_row("c", 12.5, lines=[["f.h", 5, 5], ["g.h", 9, 9]])]
    c = build_queue(rows).queue[0]
    assert c.is_chain and c.predicted_lift == 12.5
    assert c.chain_lines == (("f.h", 5, 5), ("g.h", 9, 9))


def test_chain_dd_inert_still_excluded():
    # a chain whose whole-app delta == baseline is a no-op -> excluded like any INERT
    rows = [_chain_row("c", 10.0, de=2e-4, bde=2e-4)]
    qb = build_queue(rows)
    assert qb.queue == [] and [c.region_id for c in qb.inert] == ["c"]
