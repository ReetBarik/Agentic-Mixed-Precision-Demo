"""Parity tests for the parallel shard merge (``agents.shared.fast_merge``).

``fast_merge.merge_shard_files`` must be semantically identical to the reference
``finalize_report(merge_reports(<shards>))`` slow path.  These tests pin the
fields the fast path historically dropped: ``cascade_chains`` (Strategy's
correctness tier 2) and ``region_local_vars`` (the ff/dd_integrator input).
"""

from __future__ import annotations

import json

from agents.shared import fast_merge
from agents.shared import stability_reducer as sr

from tests.agents.test_stability_reducer import (
    _cascade_chain_records,
    rec,
    opid,
    write_journal,
)


def _two_shard_paths(tmp_path):
    """Two disjoint-sample shards (each a reduced journal), written to disk.

    Sample 0 and sample 1 each carry their own cascade victim (distinct chain
    ids) plus a shared well-conditioned region, so the merge must both union the
    cascade chains and combine the ordinary region across shards.
    """
    r0, _ = _cascade_chain_records()
    r1 = []
    for rr in r0:
        rr = dict(rr)
        rr["id"] = rr["id"].replace("sample=0", "sample=1")
        rr["in"] = [i.replace("sample=0", "sample=1") for i in rr["in"]]
        r1.append(rr)

    j0 = write_journal(tmp_path / "s0.jsonl", r0)
    j1 = write_journal(tmp_path / "s1.jsonl", r1)

    shard0 = sr.reduce_journal(j0)
    shard1 = sr.reduce_journal(j1)
    p0 = tmp_path / "shard_0.json"
    p1 = tmp_path / "shard_1.json"
    sr._write_json(shard0, p0)
    sr._write_json(shard1, p1)
    return [str(p0), str(p1)], [shard0, shard1]


def test_fast_merge_preserves_cascade_chains(tmp_path):
    paths, shards = _two_shard_paths(tmp_path)

    out_path = tmp_path / "fast.json"
    fast_merge.merge_shard_files(paths, str(out_path))
    fast = json.loads(out_path.read_text())

    slow = sr.finalize_report(sr.merge_reports(shards))

    fast_chains = fast["integrals"]["CASC"]["cascade_chains"]
    slow_chains = slow["integrals"]["CASC"]["cascade_chains"]

    # two disjoint samples -> two distinct chains, carried through the fast merge
    assert len(fast_chains) == 2
    assert [c["chain_id"] for c in fast_chains] == [c["chain_id"] for c in slow_chains]
    assert fast_chains == slow_chains        # full structural parity with slow path


def test_fast_merge_regions_match_slow_path(tmp_path):
    # region-level parity, including region_local_vars (the ff/dd_integrator
    # input the fast path must not drop).  The cond-fallback contributor reads
    # two leaf source vars, so its region carries a non-empty region_local_vars.
    paths, shards = _two_shard_paths(tmp_path)

    out_path = tmp_path / "fast.json"
    fast_merge.merge_shard_files(paths, str(out_path))
    fast = json.loads(out_path.read_text())

    slow = sr.finalize_report(sr.merge_reports(shards))

    fast_regions = fast["integrals"]["CASC"]["regions"]
    slow_regions = slow["integrals"]["CASC"]["regions"]
    assert fast_regions == slow_regions      # full region parity, incl. region_local_vars

    # and region_local_vars actually made it through (not just "both empty")
    all_rlv = [v for r in fast_regions.values() for v in r["region_local_vars"]]
    assert all_rlv, "expected at least one region_local_vars entry to survive the merge"
