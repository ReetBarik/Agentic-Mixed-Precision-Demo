#!/usr/bin/env python3
"""Cheap subsampling / right-sizing study for the qcdloop characterizer.

NOT a new characterization and NO LLM calls: a pure reducer *replay* over the
already-frozen per-chunk shard reports in ``/tmp/qcdloop_shards`` (the 200
``shard_<offset>.json`` files, each = 500 samples/integral, that produced
``report_100k.json`` before being deleted from the repo tree).

The finalized ``report_100k.json`` itself is NOT sample-sliceable -- its
per-region aggregates are collapsed across all 100k samples.  The shards ARE:
a stability_shard_report merges cleanly (associative), so merging the first K
shards == the report for the first K*500 samples per integral (bit-exact, since
the driver's --sample-offset chunking makes chunk [i*500,(i+1)*500) identical to
those samples in one [0,100000) run).  So N maps to a shard prefix:

    N = 5k -> 10 shards, 10k -> 20, 25k -> 50, 50k -> 100, 100k -> 200

We fold shards in offset order ONCE (single pass, each shard read once) and
snapshot the region state at each N boundary.  Only the *region* aggregates are
merged -- prov_vars and the (huge) per-sample ``variables`` blob are dropped up
front: they drive the 13.7 GB report size and merge wall/RAM but are irrelevant
to the region/tier saturation question this study answers.  Region merge +
classification reuse the reducer's own ``_merge_region`` / ``_classify_region``,
so tiers are identical to what ``finalize_report`` would emit.

Emits a JSON blob on stdout (consumed by the caller for the markdown table).
"""
from __future__ import annotations

import copy
import glob
import json
import os
import re
import resource
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from agents.shared import stability_reducer as sr  # noqa: E402

try:
    import orjson

    def _loads(b):
        return orjson.loads(b)
except ImportError:  # pragma: no cover
    def _loads(b):
        return json.loads(b)

SHARD_DIR = "/tmp/qcdloop_shards"
CHUNK = 500
N_LIST = [5_000, 10_000, 25_000, 50_000, 100_000]
CFG = sr.ReducerConfig()

# Tiers, in the study's reporting order.  gate-(b) local cancellation ==
# "local_cancellation"; log-near-root == "log_near_root"; the accumulated-error
# cascade tier == "cancellation_cascade" (expected empty here -- shards predate
# cascade post-processing).  "stable" is the residual no-signal class; the
# atan2 saturation cap is reported separately.
NONSTABLE = ("log_near_root", "local_cancellation", "cancellation_cascade")


def _peak_rss_gb() -> float:
    # ru_maxrss is KiB on Linux; process-cumulative max (monotonic).
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _shard_paths() -> list[str]:
    paths = sorted(glob.glob(f"{SHARD_DIR}/shard_*.json"))
    offs = [int(re.search(r"shard_(\d+)\.json", p).group(1)) for p in paths]
    expected = list(range(0, 100_000, CHUNK))
    if offs != expected:
        raise SystemExit(f"shard set is not the contiguous [0,100000) prefix: "
                         f"have {len(offs)} offsets, missing "
                         f"{sorted(set(expected) - set(offs))[:10]}")
    return paths


def _merge_shard_regions(acc: dict, shard: dict) -> None:
    """Fold one shard's REGION aggregates into ``acc`` (per (integral, loc)).

    Region-only: drop prov_vars before merging so the fold stays O(#regions),
    and skip the ``variables`` dict entirely.  Everything ``_signal_class`` /
    ``_classify_region`` reads (ops, n, max_cond, gate_a_count, max_rel_err,
    rel_err_hist, max_sensitivity, max_amp, abs_val_*) is preserved exactly.
    """
    for integral, idata in shard.get("integrals", {}).items():
        dst = acc.setdefault(integral, {})
        for loc, reg in idata.get("regions", {}).items():
            reg["prov_vars"] = ()          # cheap merge; not used by classifier
            reg["region_local_vars"] = ()
            sr._merge_region(dst.setdefault(loc, sr._new_region_json()), reg)


def _snapshot_tiers(acc: dict) -> dict:
    """Classify the current accumulator into per-integral region tiers.

    Returns per-integral {loc: signal_class}, plus cascade_chain count (always 0
    for these pre-cascade shards; carried so the caller can report the skip).
    """
    out = {}
    for integral, regions in acc.items():
        out[integral] = {loc: sr._signal_class(reg, CFG)[0]
                         for loc, reg in regions.items()}
    return out


def _region_keys(tiers: dict, only=None) -> set:
    """Set of (integral, loc) region keys, optionally filtered to a tier set."""
    keys = set()
    for integral, locs in tiers.items():
        for loc, cls in locs.items():
            if only is None or cls in only:
                keys.add((integral, loc))
    return keys


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main() -> int:
    paths = _shard_paths()
    checkpoints = {n: n // CHUNK for n in N_LIST}   # N -> #shards
    want_at = {k: n for n, k in checkpoints.items()}

    acc: dict = {}
    t0 = time.monotonic()
    snaps: dict[int, dict] = {}
    per_n_stats: dict[int, dict] = {}

    for i, p in enumerate(paths, start=1):
        _merge_shard_regions(acc, _loads(Path(p).read_bytes()))
        if i in want_at:
            n = want_at[i]
            tiers = _snapshot_tiers(copy.deepcopy(acc))
            snaps[n] = tiers
            # aggregate tier counts across integrals
            tier_counts: dict[str, int] = {}
            per_integral_region_count: dict[str, int] = {}
            for integral, locs in tiers.items():
                per_integral_region_count[integral] = len(locs)
                for cls in locs.values():
                    tier_counts[cls] = tier_counts.get(cls, 0) + 1
            per_n_stats[n] = {
                "shards": i,
                "wall_s_cumulative": round(time.monotonic() - t0, 2),
                "peak_rss_gb": round(_peak_rss_gb(), 3),
                "total_regions": sum(per_integral_region_count.values()),
                "distinct_locations": len({loc for locs in tiers.values()
                                           for loc in locs}),
                "tier_counts": tier_counts,
                "per_integral_region_count": per_integral_region_count,
            }

    # Jaccard between successive N -- full region-key set and non-stable-only.
    jaccard = []
    for a, b in zip(N_LIST, N_LIST[1:]):
        ka_all, kb_all = _region_keys(snaps[a]), _region_keys(snaps[b])
        ka_ns, kb_ns = _region_keys(snaps[a], NONSTABLE), _region_keys(snaps[b], NONSTABLE)
        # tier churn: regions present in BOTH whose class changed a->b
        churn = 0
        common = ka_all & kb_all
        for integral, loc in common:
            if snaps[a][integral][loc] != snaps[b][integral][loc]:
                churn += 1
        jaccard.append({
            "from": a, "to": b,
            "jaccard_all_regions": round(_jaccard(ka_all, kb_all), 4),
            "jaccard_nonstable": round(_jaccard(ka_ns, kb_ns), 4),
            "n_nonstable_from": len(ka_ns), "n_nonstable_to": len(kb_ns),
            "tier_churn_regions": churn,
        })

    print(json.dumps({
        "n_list": N_LIST,
        "per_n": per_n_stats,
        "jaccard_successive": jaccard,
        "cascade_note": "shards carry no cascade_chains (pre-cascade "
                        "post-processing) -> cascade tier not measurable here",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
