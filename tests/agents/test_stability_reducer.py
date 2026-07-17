"""Unit tests for the characterizer stability reducer.

Synthetic v0.3 journals (records carry ``id``/``in``/``prov_vars`` and the
``@integral=<name>/sample=<i>`` scope suffix) exercise:

* the forward-cone amplification pass — the cascade counterexample where an
  early, locally-benign node is correctly flagged downcast-unsafe via ``amp``
  (the thing a per-line rollup structurally cannot see);
* stable mechanistic classification, and the policy-neutral prediction
  (``predicted_rel_err_if_float``) a caller applies its own margin to;
* the float value-range guard (underflow) as a measured fact, not a verdict;
* gate-(a) atan2-saturation filtering (excluded from max_cond, counted apart);
* scope-aware grouping across integrals/samples;
* shard-merge == reduce-of-concatenation (associativity);
* prov_vars / legacy-prov handling (the schema-drift fix).
"""

from __future__ import annotations

import json

import pytest

from agents.shared import stability_reducer as sr


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def rec(op, at, rid, ins, val, cond, rel_err, prov_vars=None, prov_consts=None,
        prov=None):
    r = {"op": op, "at": at, "id": rid, "in": list(ins),
         "val": val, "cond": cond, "rel_err": rel_err}
    if prov_vars is not None:
        r["prov_vars"] = prov_vars
    if prov_consts is not None:
        r["prov_consts"] = prov_consts
    if prov is not None:
        r["prov"] = prov
    return r


def write_journal(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def opid(op, file, line, n, scope):
    return f"{op}@{file}:{line}#{n}@{scope}"


# ---------------------------------------------------------------------------
# scope parsing
# ---------------------------------------------------------------------------

def test_scope_str_and_parse():
    rid = opid("sub", "box/B2m.h", 88, 3, "integral=B15/sample=42")
    assert sr._scope_str(rid) == "integral=B15/sample=42"
    assert sr._parse_scope(sr._scope_str(rid)) == {"integral": "B15", "sample": "42"}
    assert sr._scope_str("a") == ""            # bare source id
    assert sr._scope_str("mul@f:1#2") == ""    # unscoped op id


def test_prov_vars_schema_drift():
    v03 = rec("add", "f:1", "add@f:1#1@integral=X/sample=0", ["a", "b"], 1.0, 1, 1e-16,
              prov_vars=["a", "b"], prov_consts=["PI"])
    assert sr._prov_vars(v03) == ["a", "b"]          # not empty on v0.3 (the fix)
    legacy = rec("add", "f:1", None, ["a"], 1.0, 1, 1e-16, prov=["a", "b"])
    assert sr._prov_vars(legacy) == ["a", "b"]


# ---------------------------------------------------------------------------
# forward cone — the cascade counterexample
# ---------------------------------------------------------------------------

def _cascade_records():
    scope = "integral=CASC/sample=0"
    a = opid("add", "k.cpp", 1, 1, scope)   # early, locally benign
    b = opid("sub", "k.cpp", 2, 1, scope)   # downstream cancellation sink
    return [
        rec("add", "k.cpp:cscf:1", a, ["e", "f"], 2.0, 1.0, 1e-16,
            prov_vars=["e", "f"]),
        rec("sub", "k.cpp:cscf:2", b, [a, "g"], 1e-11, 1e12, 1e-4,
            prov_vars=["e", "f", "g"]),
    ]


def test_amp_pass_propagates_downstream_conditioning():
    nodes, amp, node_sens, source_sens, source_ids = sr._analyze_sample(
        _cascade_records(), sr.ReducerConfig())
    a = opid("add", "k.cpp", 1, 1, "integral=CASC/sample=0")
    b = opid("sub", "k.cpp", 2, 1, "integral=CASC/sample=0")
    assert amp[b] == 1.0                       # output sink
    assert amp[a] == pytest.approx(1e12)       # inherits the sink's cond
    assert node_sens[a] == pytest.approx(1e12)
    assert source_ids == {"e", "f", "g"}
    # the early source variables inherit the danger too
    assert source_sens["e"] == pytest.approx(1e12)


def test_early_benign_node_high_amp(tmp_path):
    j = write_journal(tmp_path / "casc.jsonl", _cascade_records())
    report = sr.report_from_journals([j])
    regions = report["integrals"]["CASC"]["regions"]
    early = regions["k.cpp:cscf:1"]

    # Locally the early node looks perfectly stable (mechanistic class)...
    assert early["max_cond"] == pytest.approx(1.0)
    assert early["max_rel_err"] == pytest.approx(1e-16)
    assert early["signal_class"] == "stable"
    # ...but the forward cone exposes the danger as a MEASUREMENT: huge amp, and a
    # large predicted float error. Strategy applies its own margin to this number.
    assert early["max_amp"] == pytest.approx(1e12)
    assert early["predicted_rel_err_if_float"] == pytest.approx(sr.U_FLOAT * 1e12)
    # ff prediction is the same measurement at ff's tighter unit roundoff — the
    # Strategy speedup queue compares this to its margin to queue double->ff.
    assert early["predicted_rel_err_if_ff"] == pytest.approx(sr.U_FF * 1e12)
    # the reducer emits no direction/verdict — that is Strategy's job
    assert "direction" not in early
    assert "downcast_safe" not in early

    # the sink is mechanistically flagged (high cond), still no direction
    assert regions["k.cpp:cscf:2"]["signal_class"] == "log_near_root"


# ---------------------------------------------------------------------------
# stable class / policy-neutral prediction / value-range guard
# ---------------------------------------------------------------------------

def _stable_records():
    scope = "integral=STAB/sample=0"
    s = opid("mul", "s.cpp", 1, 1, scope)
    v = opid("mul", "s.cpp", 2, 1, scope)
    return [
        rec("mul", "s.cpp:f:1", s, ["p", "q"], 1.0, 1.0, 1e-16, prov_vars=["p", "q"]),
        rec("mul", "s.cpp:f:2", v, ["p", "q"], 1e-40, 1.0, 1e-16, prov_vars=["p", "q"]),
    ]


def test_stable_region_emits_measurements_not_verdict(tmp_path):
    j = write_journal(tmp_path / "stab.jsonl", _stable_records())
    report = sr.report_from_journals([j])
    stable = report["integrals"]["STAB"]["regions"]["s.cpp:f:1"]
    assert stable["signal_class"] == "stable"
    assert stable["value_range_ok_for_float"] is True
    # cond*amp == 1 here, so the predicted float error is just float's own u
    assert stable["predicted_rel_err_if_float"] == pytest.approx(sr.U_FLOAT)
    # ...and the ff prediction is ff's own u (~1.4e-14), which clears a 10-digit
    # margin where float cannot — this is exactly what unblocks the ff speedup queue
    assert stable["predicted_rel_err_if_ff"] == pytest.approx(sr.U_FF)
    assert stable["predicted_rel_err_if_ff"] < 1e-10 < stable["predicted_rel_err_if_float"]
    # policy-neutral: no direction/verdict — Strategy owns that
    assert "downcast_safe" not in stable
    assert "direction" not in stable


def test_prediction_lets_caller_apply_any_margin(tmp_path):
    # One (expensive) characterization run serves any acceptance policy: the
    # reducer emits predicted_rel_err_if_float and the caller compares to ITS
    # margin. The digit-budget dependence lives entirely on the caller side.
    j = write_journal(tmp_path / "stab.jsonl", _stable_records())
    report = sr.report_from_journals([j])
    pred = report["integrals"]["STAB"]["regions"]["s.cpp:f:1"]["predicted_rel_err_if_float"]
    assert pred < 1e-6      # within a 6-digit acceptance margin
    # float's own u (~6e-8) exceeds a 1e-10 margin, so 10 digits is unreachable
    assert pred > 1e-10


def test_value_range_guard_flags_underflow(tmp_path):
    j = write_journal(tmp_path / "stab.jsonl", _stable_records())
    report = sr.report_from_journals([j])
    underflow = report["integrals"]["STAB"]["regions"]["s.cpp:f:2"]
    # locally as "stable" as s.cpp:f:1, but its value underflows float's range —
    # a measured fact the caller weighs, not a verdict the reducer makes
    assert underflow["signal_class"] == "stable"
    assert underflow["value_range_ok_for_float"] is False
    assert "downcast_safe" not in underflow


# ---------------------------------------------------------------------------
# gate-(a) atan2 saturation
# ---------------------------------------------------------------------------

def test_gate_a_filtering(tmp_path):
    scope = "integral=GATE/sample=0"
    g = opid("atan2", "g.cpp", 1, 1, scope)       # pure gate-a location
    n1 = opid("atan2", "g.cpp", 2, 1, scope)      # gate-a at a mixed location
    n2 = opid("mul", "g.cpp", 2, 2, scope)        # normal op, same location
    records = [
        rec("atan2", "g.cpp:f:1", g, ["z"], 0.5, sr.ATAN2_SATURATION, 1e-16,
            prov_vars=["z"]),
        rec("atan2", "g.cpp:f:2", n1, ["z"], 0.5, sr.ATAN2_SATURATION, 1e-16,
            prov_vars=["z"]),
        rec("mul", "g.cpp:f:2", n2, ["z", "z"], 0.25, 5.0, 1e-15, prov_vars=["z"]),
    ]
    j = write_journal(tmp_path / "gate.jsonl", records)
    report = sr.report_from_journals([j], sr.ReducerConfig())
    regions = report["integrals"]["GATE"]["regions"]

    pure = regions["g.cpp:f:1"]
    assert pure["max_cond"] == 0.0                  # saturation excluded
    assert pure["gate_a_count"] == 1
    assert pure["signal_class"] == "atan2_saturation"

    mixed = regions["g.cpp:f:2"]
    assert mixed["gate_a_count"] == 1
    assert mixed["max_cond"] == pytest.approx(5.0)  # normal op survives, cap doesn't


# ---------------------------------------------------------------------------
# scope-aware grouping
# ---------------------------------------------------------------------------

def test_scope_grouping_separates_integrals(tmp_path):
    records = [
        rec("add", "f:1", opid("add", "f", 1, 1, "integral=B1/sample=0"),
            ["a", "b"], 1.0, 1.0, 1e-16, prov_vars=["a", "b"]),
        rec("mul", "f:2", opid("mul", "f", 2, 1, "integral=B2/sample=0"),
            ["a", "b"], 1.0, 1.0, 1e-16, prov_vars=["a", "b"]),
        rec("mul", "f:2", opid("mul", "f", 2, 1, "integral=B2/sample=1"),
            ["a", "b"], 1.0, 1.0, 1e-16, prov_vars=["a", "b"]),
    ]
    j = write_journal(tmp_path / "multi.jsonl", records)
    shard = sr.reduce_journal(j)
    assert set(shard["integrals"]) == {"B1", "B2"}
    assert shard["samples_seen"] == {"B1": 1, "B2": 2}


# ---------------------------------------------------------------------------
# line-scope code regions (operator ops carry the line= scope tag)
# ---------------------------------------------------------------------------

def test_line_scope_regions_and_cross_boundary_amp(tmp_path):
    base = "integral=LINE/sample=0"
    line = "integral=LINE/sample=0/line=acc.h:10"
    # A is computed on an earlier line (base scope); B is the accumulation op
    # inside the wrapped res-write line and consumes A across the scope boundary.
    a = opid("mul", "?", 0, 1, base)                  # note: at="" (operator op)
    b = f"sub@?#1@{line}"                             # operator op, line-scoped
    c = opid("add", "?", 0, 2, base)                  # another base-scope op
    records = [
        rec("mul", "", a, ["e", "f"], 2.0, 1.0, 1e-16, prov_vars=["e", "f"]),
        rec("sub", "", b, [a, "g"], 1e-11, 1e12, 1e-4, prov_vars=["e", "f", "g"]),
        rec("add", "", c, ["p", "q"], 3.0, 1.0, 1e-16, prov_vars=["p", "q"]),
    ]
    j = write_journal(tmp_path / "line.jsonl", records)

    shard = sr.reduce_journal(j)
    assert shard["samples_seen"] == {"LINE": 1}       # one sample despite scope churn
    regions = shard["integrals"]["LINE"]["regions"]
    # the wrapped accumulation is its own code region, keyed by the line label
    assert "acc.h:10" in regions
    assert regions["acc.h:10"]["ops"] == {"sub": 1}
    assert regions["acc.h:10"]["max_cond"] == pytest.approx(1e12)
    # the forward cone crosses the line boundary: A (base scope) inherits the
    # sink's conditioning even though A and B are in different scope strings.
    assert regions[""]["max_amp"] == pytest.approx(1e12)

    report = sr.report_from_journals([j])
    acc = report["integrals"]["LINE"]["regions"]["acc.h:10"]
    # the accumulation sink is mechanistically flagged (high cond) — measured
    # only, no remediation direction
    assert acc["signal_class"] == "log_near_root"
    assert "direction" not in acc


def test_line_scope_value_must_be_basename_no_slash(tmp_path):
    """Pin the hard constraint on the ``line=`` value: it MUST be a basename
    (``B2m.h:84``), never a path (``box/B2m.h:84``).

    ``current_scope_suffix`` joins the scope stack with ``/`` and ``_parse_scope``
    splits on ``/``, so a ``/`` inside a ``line=`` value is read as a scope
    boundary: ``line=box/B2m.h:84`` parses as ``{"line": "box"}`` plus a stray
    ``B2m.h:84`` part with no ``=`` (discarded).  The region then collapses to the
    directory name ``box`` — every box header would alias together, destroying
    per-line attribution.  The per-statement injector emits basenames; this test
    fails loudly if that invariant is ever broken (in the injector or by a change
    to the scope separator).
    """
    # Unit level: _region_key on the two id shapes.
    slashed = f"sub@?#1@integral=B2/sample=0/line=box/B2m.h:84"
    basename = f"sub@?#1@integral=B2/sample=0/line=B2m.h:84"
    assert sr._region_key({"id": slashed}) == "box"          # truncated — WRONG
    assert sr._region_key({"id": basename}) == "B2m.h:84"    # correct

    # Reduce level: a slashed line scope buckets under the bare directory, so two
    # distinct source lines in different headers would alias; basenames don't.
    j_bad = write_journal(tmp_path / "slash.jsonl", [
        rec("sub", "", slashed, ["e", "f"], 1.0, 1e6, 1e-9, prov_vars=["e", "f"]),
    ])
    regions_bad = sr.reduce_journal(j_bad)["integrals"]["B2"]["regions"]
    assert "box" in regions_bad and "B2m.h:84" not in regions_bad

    j_ok = write_journal(tmp_path / "base.jsonl", [
        rec("sub", "", basename, ["e", "f"], 1.0, 1e6, 1e-9, prov_vars=["e", "f"]),
    ])
    regions_ok = sr.reduce_journal(j_ok)["integrals"]["B2"]["regions"]
    assert "B2m.h:84" in regions_ok and "box" not in regions_ok


# ---------------------------------------------------------------------------
# merge associativity
# ---------------------------------------------------------------------------

def test_merge_equals_reduce_of_concatenation(tmp_path):
    s0 = "integral=MRG/sample=0"
    s1 = "integral=MRG/sample=1"
    r0 = [rec("sub", "m.cpp:f:1", opid("sub", "m.cpp", 1, 1, s0),
              ["a", "b"], 3.0, 10.0, 1e-12, prov_vars=["a", "b"])]
    r1 = [rec("sub", "m.cpp:f:1", opid("sub", "m.cpp", 1, 1, s1),
              ["a", "b"], 3.0, 1000.0, 1e-6, prov_vars=["a", "b"])]

    j0 = write_journal(tmp_path / "s0.jsonl", r0)
    j1 = write_journal(tmp_path / "s1.jsonl", r1)
    jcat = write_journal(tmp_path / "cat.jsonl", r0 + r1)

    merged = sr.merge_reports([sr.reduce_journal(j0), sr.reduce_journal(j1)])
    direct = sr.reduce_journal(jcat)

    assert merged["samples_seen"] == direct["samples_seen"] == {"MRG": 2}
    assert merged["integrals"] == direct["integrals"]

    region = merged["integrals"]["MRG"]["regions"]["m.cpp:f:1"]
    assert region["n"] == 2
    assert region["max_cond"] == pytest.approx(1000.0)
    assert region["rel_err_hist"]["total"] == 2


def test_merge_combines_same_integral_across_shards(tmp_path):
    # the real sharding case: same integral, disjoint sample ranges
    r0 = [rec("mul", "x:1", opid("mul", "x", 1, 1, "integral=B7/sample=0"),
              ["a", "b"], 1.0, 2.0, 1e-15, prov_vars=["a", "b"])]
    r1 = [rec("mul", "x:1", opid("mul", "x", 1, 1, "integral=B7/sample=1"),
              ["a", "b"], 1.0, 9.0, 1e-15, prov_vars=["a", "b"])]
    j0 = write_journal(tmp_path / "a.jsonl", r0)
    j1 = write_journal(tmp_path / "b.jsonl", r1)
    merged = sr.merge_reports([sr.reduce_journal(j0), sr.reduce_journal(j1)])
    assert merged["samples_seen"] == {"B7": 2}
    assert merged["integrals"]["B7"]["regions"]["x:1"]["max_cond"] == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# provenance survives into the report
# ---------------------------------------------------------------------------

def test_prov_vars_populated_in_report(tmp_path):
    j = write_journal(tmp_path / "casc.jsonl", _cascade_records())
    report = sr.report_from_journals([j])
    early = report["integrals"]["CASC"]["regions"]["k.cpp:cscf:1"]
    assert early["prov_vars"] == ["e", "f"]


# ---------------------------------------------------------------------------
# region-local variables (reads in-scope) — the tight peer of prov_vars
# ---------------------------------------------------------------------------

def test_region_local_vars_are_direct_reads_not_transitive_union(tmp_path):
    j = write_journal(tmp_path / "casc.jsonl", _cascade_records())
    report = sr.report_from_journals([j])
    regions = report["integrals"]["CASC"]["regions"]

    # early node reads both leaf source vars directly → both are region-local
    early = regions["k.cpp:cscf:1"]
    assert early["region_local_vars"] == ["e", "f"]

    # the sink reads (produced-node a, leaf g): only g is a direct source-var read.
    # prov_vars is the TRANSITIVE union [e, f, g] (e, f flow in via a) — the wrong
    # input for a regional promotion; region_local_vars is the tight subset [g].
    sink = regions["k.cpp:cscf:2"]
    assert sink["prov_vars"] == ["e", "f", "g"]
    assert sink["region_local_vars"] == ["g"]
    assert set(sink["region_local_vars"]) <= set(sink["prov_vars"])


def test_region_local_vars_merge_is_union(tmp_path):
    # same region across two samples reading disjoint leaf vars → union on merge
    r0 = [rec("sub", "m.cpp:f:1", opid("sub", "m.cpp", 1, 1, "integral=U/sample=0"),
              ["a", "b"], 1.0, 10.0, 1e-12, prov_vars=["a", "b"])]
    r1 = [rec("sub", "m.cpp:f:1", opid("sub", "m.cpp", 1, 1, "integral=U/sample=1"),
              ["a", "c"], 1.0, 10.0, 1e-12, prov_vars=["a", "c"])]
    j0 = write_journal(tmp_path / "u0.jsonl", r0)
    j1 = write_journal(tmp_path / "u1.jsonl", r1)
    report = sr.finalize_report(
        sr.merge_reports([sr.reduce_journal(j0), sr.reduce_journal(j1)]))
    region = report["integrals"]["U"]["regions"]["m.cpp:f:1"]
    assert region["region_local_vars"] == ["a", "b", "c"]


def test_region_local_vars_excludes_consts_and_literals(tmp_path):
    # a leaf operand that is a named const (prov_consts) or literal is NOT a
    # source var, so it must not appear in region_local_vars.
    scope = "integral=K/sample=0"
    m = opid("mul", "s.cpp", 1, 1, scope)
    records = [
        rec("mul", "s.cpp:f:1", m, ["p", "PI", "_lit@1"], 1.0, 1.0, 1e-16,
            prov_vars=["p"], prov_consts=["PI"]),
    ]
    j = write_journal(tmp_path / "k.jsonl", records)
    report = sr.report_from_journals([j])
    region = report["integrals"]["K"]["regions"]["s.cpp:f:1"]
    assert region["region_local_vars"] == ["p"]        # PI, _lit@1 excluded
