"""Report loading + per-line merge (agents.strategy.characterization)."""

import json

from agents.strategy.characterization import load_regions


def _report(tmp_path, integrals):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"kind": "stability_report", "schema_version": 1,
                             "no_id_records": 0, "samples_seen": {},
                             "integrals": integrals}))
    return str(p)


def _region(sig, cond=1.0, rel=1e-16, pf=1e-16, prov=("v",), ops=None):
    return {"signal_class": sig, "max_cond": cond, "max_rel_err": rel,
            "predicted_rel_err_if_float": pf, "prov_vars": list(prov),
            "ops": ops or {"mul": 1}, "n": 100, "non_localizable": False}


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
    assert r.integral == "B"                           # highest-cond representative
    assert set(r.integrals) == {"A", "B"}
    assert r.target.variables == ["a", "b"]            # union, order-preserving


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
