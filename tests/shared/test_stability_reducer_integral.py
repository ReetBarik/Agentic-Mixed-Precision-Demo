"""Phase-A schema-v2 tests: the additive per-record ``integral`` tag.

The stability report was ALREADY per-integral (top-level ``integrals[<name>]``
bucketing).  Schema v2 only stamps that parent-bucket name onto every emitted
*record* — region records and cascade-chain records — so a downstream consumer
can key on ``(line, integral)`` without re-deriving it from the enclosing
bucket.  This is a pure data-layer decoration: no structural change, no signal
change (a shared source line still produces one record per integral, exactly as
before — it just now says which integral it is).

These tests pin, on hand-built journals:

* every region record carries ``integral`` == its parent bucket name, on all
  three emit paths (shard / merge / finalize);
* a source line shared by N integrals yields N records, one per bucket, each
  self-describing its integral (unchanged count, added tag);
* every cascade-chain record carries ``integral`` consistent with its
  ``chain_id`` (``cascade_<integral>_...``);
* ``schema_version`` is bumped to 2.

The pre-existing behavioural suite (`tests/agents/test_stability_reducer.py`)
runs unchanged — this change is additive.
"""

from __future__ import annotations

import json

import pytest

from agents.shared import stability_reducer as sr


# ---------------------------------------------------------------------------
# helpers (mirror tests/agents/test_stability_reducer.py)
# ---------------------------------------------------------------------------

def _rec(op, rid, ins, val, cond, rel_err, prov_vars=None):
    r = {"op": op, "at": "", "id": rid, "in": list(ins),
         "val": val, "cond": cond, "rel_err": rel_err}
    if prov_vars is not None:
        r["prov_vars"] = prov_vars
    return r


def _write_journal(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _plain_region_records(integral: str) -> list[dict]:
    """One well-conditioned-ish region at ``shared.h:10`` under ``integral``.

    A lone add/sub sink with rel_err below the cascade floor: it is a plain
    region (no cascade victim), so the integral tag under test is exercised on
    the ordinary region path, not the chain path.
    """
    base = f"integral={integral}/sample=0"
    line = base + "/line=shared.h:10"
    return [
        _rec("sub", f"sub@?#1@{line}", ["a", "b"], 1.0, 1e6, 1e-9,
             prov_vars=["a", "b"]),
    ]


def _cascade_records(integral: str) -> list[dict]:
    """A two-contributor cascade under ``integral`` (one victim => one chain).

    Structure mirrors the behavioural suite's cascade fixture: two near-cancelling
    subs on distinct source lines feed a low-cond, high-rel_err accumulation sink.
    """
    base = f"integral={integral}/sample=0"
    L1 = base + "/line=B2m.h:355"
    L2 = base + "/line=B0m.h:230"
    x = f"mul@?:0#1@{base}"
    y = f"mul@?:0#2@{base}"
    c1 = f"sub@?#1@{L1}"
    c2 = f"sub@?#2@{L2}"
    v = f"add@?:0#3@{base}"
    return [
        _rec("mul", x, ["e", "f"], 1.0000001, 1.0, 1e-16, prov_vars=["e", "f"]),
        _rec("mul", y, ["g", "h"], 1.0, 1.0, 1e-16, prov_vars=["g", "h"]),
        _rec("sub", c1, [x, y], 1e-7, 2.0e7, 1e-16),                 # val-ratio cancel
        _rec("sub", c2, ["p", "q"], 1e-7, 100.0, 1e-16, prov_vars=["p", "q"]),
        _rec("add", v, [c1, c2], 2e-7, 1.0, 1e-4),                   # victim (DAG sink)
    ]


# ---------------------------------------------------------------------------
# region records carry the integral tag on every emit path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("integrals", [
    ["B1"],                     # one line, one integral
    ["B1", "B4"],               # one line, two integrals -> two tagged records
    ["B1", "B4", "BIN0"],       # ...and generally N -> N
])
def test_shared_line_yields_one_tagged_record_per_integral(tmp_path, integrals):
    records: list[dict] = []
    for integ in integrals:
        records += _plain_region_records(integ)
    j = _write_journal(tmp_path / "regions.jsonl", records)

    shard = sr.reduce_journal(str(j))
    assert shard["schema_version"] == 2

    # the same source line appears once PER integral bucket (unchanged count),
    # and each record self-describes its integral (the added tag).
    assert set(shard["integrals"]) == set(integrals)
    seen_tags = []
    for name, idata in shard["integrals"].items():
        reg = idata["regions"]["shared.h:10"]
        assert reg["integral"] == name          # matches its parent bucket
        seen_tags.append(reg["integral"])
    assert sorted(seen_tags) == sorted(integrals)


@pytest.mark.parametrize("integrals", [["B1"], ["B1", "B4"]])
def test_integral_tag_survives_merge_and_finalize(tmp_path, integrals):
    records: list[dict] = []
    for integ in integrals:
        records += _plain_region_records(integ)
    j = _write_journal(tmp_path / "regions.jsonl", records)

    # merge path (merge_reports) — every merged region keeps its bucket tag
    merged = sr.merge_reports([sr.reduce_journal(str(j))])
    assert merged["schema_version"] == 2
    for name, idata in merged["integrals"].items():
        assert idata["regions"]["shared.h:10"]["integral"] == name

    # finalize path (the report Strategy reads) — tag present on classified region
    report = sr.report_from_journals([str(j)])
    assert report["schema_version"] == 2
    for name, idata in report["integrals"].items():
        assert idata["regions"]["shared.h:10"]["integral"] == name
        # the ranked projection inherits it too (it spreads the region dict)
        for row in idata["top_regions_by_rel_err"]:
            assert row["integral"] == name


def test_merge_equals_reduce_of_concatenation_with_tag(tmp_path):
    """The v2 tag must not break merge==reduce-of-concatenation associativity."""
    r0 = _plain_region_records("MRG")
    r1 = [dict(r, id=r["id"].replace("sample=0", "sample=1")) for r in r0]
    j0 = _write_journal(tmp_path / "s0.jsonl", r0)
    j1 = _write_journal(tmp_path / "s1.jsonl", r1)
    jcat = _write_journal(tmp_path / "cat.jsonl", r0 + r1)

    merged = sr.merge_reports([sr.reduce_journal(str(j0)), sr.reduce_journal(str(j1))])
    direct = sr.reduce_journal(str(jcat))
    assert merged["integrals"] == direct["integrals"]     # tag included, still equal
    assert merged["integrals"]["MRG"]["regions"]["shared.h:10"]["integral"] == "MRG"


# ---------------------------------------------------------------------------
# cascade-chain records carry the integral tag, consistent with chain_id
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("integrals", [["CASC"], ["B1", "B4"]])
def test_cascade_chain_carries_integral_consistent_with_chain_id(tmp_path, integrals):
    records: list[dict] = []
    for integ in integrals:
        records += _cascade_records(integ)
    j = _write_journal(tmp_path / "casc.jsonl", records)

    report = sr.report_from_journals([str(j)])
    assert report["schema_version"] == 2

    for name in integrals:
        chains = report["integrals"][name]["cascade_chains"]
        assert len(chains) == 1                             # one victim -> one chain
        chain = chains[0]
        assert chain["integral"] == name                    # explicit tag == bucket
        # tag is consistent with the (independently derived) chain_id encoding
        assert chain["chain_id"].startswith(f"cascade_{name}_")


def test_cascade_chain_tag_present_in_shard(tmp_path):
    """The tag is on the chain from the reduce (shard) stage, not just finalize."""
    j = _write_journal(tmp_path / "casc.jsonl", _cascade_records("CASC"))
    shard = sr.reduce_journal(str(j))
    chains = shard["integrals"]["CASC"]["cascade_chains"]
    assert len(chains) == 1
    (chain,) = chains.values()
    assert chain["integral"] == "CASC"
    assert chain["chain_id"].startswith("cascade_CASC_")
