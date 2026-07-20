"""Report loading + per-line merge (agents.strategy.characterization)."""

import json

from agents.strategy.characterization import load_chains, load_regions


def _report(tmp_path, integrals):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"kind": "stability_report", "schema_version": 1,
                             "no_id_records": 0, "samples_seen": {},
                             "integrals": integrals}))
    return str(p)


def _region(sig, cond=1.0, rel=1e-16, pf=1e-16, prov=("v",), ops=None, pff=None):
    r = {"signal_class": sig, "max_cond": cond, "max_rel_err": rel,
         "predicted_rel_err_if_float": pf, "prov_vars": list(prov),
         "ops": ops or {"mul": 1}, "n": 100, "non_localizable": False}
    if pff is not None:
        r["predicted_rel_err_if_ff"] = pff
    return r


def test_non_localizable_skipped(tmp_path):
    integrals = {"A": {"regions": {
        "f.h:10": _region("stable"),
        "": _region("cancellation_cascade"),           # empty key → skipped
    }}}
    regs, meta = load_regions(_report(tmp_path, integrals))
    assert meta["non_localizable_skipped"] == 1
    assert len(regs) == 1 and regs[0].target.location == "f.h:10"


def test_non_localizable_flag_skipped(tmp_path):
    integrals = {"A": {"regions": {
        "f.h:10": {**_region("stable"), "non_localizable": True},
    }}}
    regs, meta = load_regions(_report(tmp_path, integrals))
    assert regs == [] and meta["non_localizable_skipped"] == 1


def test_merge_same_line_across_integrals(tmp_path):
    # same source line in two integrals: stable (low cond) + local_cancellation (high)
    integrals = {
        "A": {"regions": {"f.h:10": _region("stable", cond=10.0, rel=1e-12,
                                            pf=1e-9, prov=("a",), ops={"mul": 2})}},
        "B": {"regions": {"f.h:10": _region("local_cancellation", cond=1e16, rel=1e-3,
                                            pf=1e-2, prov=("b",), ops={"sub": 5})}},
    }
    regs, meta = load_regions(_report(tmp_path, integrals))
    assert meta["raw_regions"] == 2 and meta["n_regions"] == 1
    r = regs[0]
    assert r.signal_class == "local_cancellation"     # most severe wins
    assert r.max_cond == 1e16                          # worst-case
    assert r.max_rel_err == 1e-3
    assert r.predicted_rel_err_if_float == 1e-2        # unsafe if unsafe anywhere
    assert r.predicted_rel_err_if_ff == 1e-2           # ff falls back to float, worst-case
    assert r.integral == "B"                           # highest-cond representative
    assert set(r.integrals) == {"A", "B"}
    assert r.target.variables == ["a", "b"]            # union, order-preserving


def test_predicted_ff_loaded_and_merged_worst_case(tmp_path):
    # a report carrying a distinct ff signal: loaded verbatim, merged worst-case.
    integrals = {
        "A": {"regions": {"f.h:10": _region("stable", cond=10.0, pf=1e-8, pff=1e-13,
                                            prov=("a",))}},
        "B": {"regions": {"f.h:10": _region("stable", cond=20.0, pf=1e-7, pff=1e-11,
                                            prov=("b",))}},
    }
    regs, _ = load_regions(_report(tmp_path, integrals))
    r = regs[0]
    assert r.predicted_rel_err_if_ff == 1e-11          # max across integrals
    assert r.predicted_rel_err_if_float == 1e-7


def test_predicted_ff_falls_back_to_float_when_absent(tmp_path):
    # legacy report (report_1k/100k): no ff field → conservative float fallback.
    integrals = {"A": {"regions": {"f.h:10": _region("stable", pf=3e-9)}}}
    regs, _ = load_regions(_report(tmp_path, integrals))
    assert regs[0].predicted_rel_err_if_ff == 3e-9


def test_no_merge_keeps_per_integral(tmp_path):
    integrals = {
        "A": {"regions": {"f.h:10": _region("stable")}},
        "B": {"regions": {"f.h:10": _region("stable")}},
    }
    regs, meta = load_regions(_report(tmp_path, integrals), merge=False)
    assert len(regs) == 2 and meta["merged"] is False


def test_op_count_from_ops_sum(tmp_path):
    integrals = {"A": {"regions": {"f.h:10": _region("stable", ops={"mul": 3, "add": 4})}}}
    regs, _ = load_regions(_report(tmp_path, integrals))
    assert regs[0].op_count == 7


# ---------------------------------------------------------------------------
# Wave-3 WI1: value_range_ok_for_float  +  WI3: per-op mix
# ---------------------------------------------------------------------------

def test_value_range_ok_loaded(tmp_path):
    integrals = {"A": {"regions": {
        "ok.h:1": {**_region("stable"), "value_range_ok_for_float": True},
        "no.h:2": {**_region("stable"), "value_range_ok_for_float": False},
    }}}
    regs, _ = load_regions(_report(tmp_path, integrals))
    by = {r.target.location: r.value_range_ok_for_float for r in regs}
    assert by == {"ok.h:1": True, "no.h:2": False}


def test_value_range_fails_open_true_when_absent(tmp_path):
    # legacy report with no flag → default True (do not gate), warns once.
    integrals = {"A": {"regions": {"f.h:10": _region("stable")}}}
    regs, _ = load_regions(_report(tmp_path, integrals))
    assert regs[0].value_range_ok_for_float is True


def test_value_range_merge_unsafe_if_unsafe_anywhere(tmp_path):
    # same line: safe in A, unsafe in B → merged worst-case = unsafe (False).
    integrals = {
        "A": {"regions": {"f.h:10": {**_region("stable"),
                                     "value_range_ok_for_float": True}}},
        "B": {"regions": {"f.h:10": {**_region("stable"),
                                     "value_range_ok_for_float": False}}},
    }
    regs, meta = load_regions(_report(tmp_path, integrals))
    assert meta["n_regions"] == 1
    assert regs[0].value_range_ok_for_float is False


def test_ops_mix_loaded_and_merged_worst_case(tmp_path):
    # per-op mix carried; merge takes element-wise max across integrals.
    integrals = {
        "A": {"regions": {"f.h:10": _region("stable", ops={"mul": 2, "log": 1})}},
        "B": {"regions": {"f.h:10": _region("stable", ops={"mul": 5, "div": 3})}},
    }
    regs, _ = load_regions(_report(tmp_path, integrals))
    assert regs[0].ops == {"mul": 5, "log": 1, "div": 3}


def test_chain_carries_range_flag_and_ops(tmp_path):
    ch = _chain("cascade_IX_a_1", [("B2m.h", 355)], ops={"log": 2, "sub": 4})
    ch["value_range_ok_for_float"] = False
    integrals = {"IX": {"regions": {}, "cascade_chains": [ch]}}
    chains, _ = load_chains(_report(tmp_path, integrals))
    c = chains[0]
    assert c.value_range_ok_for_float is False
    assert c.ops == {"log": 2, "sub": 4}
    # walk_record propagates both
    wr = c.walk_record()
    assert wr.value_range_ok_for_float is False and wr.ops == {"log": 2, "sub": 4}


# ---------------------------------------------------------------------------
# cascade chains
# ---------------------------------------------------------------------------

def _chain(chain_id, spans, cond=1e6, rel=1e-3, ops=None, lv=("v",)):
    return {"kind": "cascade_chain", "chain_id": chain_id,
            "chain": [{"file": f, "line_start": l, "line_end": l} for f, l in spans],
            "signal_class": "cancellation_cascade", "non_localizable": False,
            "max_cond": cond, "max_rel_err": rel, "predicted_rel_err_if_float": 1e-2,
            "ops": ops or {"sub": 2}, "n": 2, "region_local_vars": list(lv)}


def test_load_chains_builds_multiline_records(tmp_path):
    integrals = {"IX": {"regions": {}, "cascade_chains": [
        _chain("cascade_IX_a_1", [("B2m.h", 355), ("B0m.h", 230)])]}}
    chains, meta = load_chains(_report(tmp_path, integrals))
    assert meta["n_chains"] == 1
    c = chains[0]
    assert c.chain_id == "cascade_IX_a_1"
    assert c.signal_class == "cancellation_cascade"
    assert [ (t.file, t.line_start, t.line_end) for t in c.lines ] == [
        ("B2m.h", 355, 355), ("B0m.h", 230, 230)]
    assert c.op_count == 2
    # ff falls back to the chain's float prediction and propagates to walk_record
    assert c.predicted_rel_err_if_ff == 1e-2
    assert c.walk_record().predicted_rel_err_if_ff == 1e-2
    # walk_record's representative target is the first chain line
    assert c.walk_record().target.location == "B2m.h:355"


def test_load_chains_empty_when_absent(tmp_path):
    integrals = {"A": {"regions": {"f.h:10": _region("stable")}}}
    chains, meta = load_chains(_report(tmp_path, integrals))
    assert chains == [] and meta["n_chains"] == 0
