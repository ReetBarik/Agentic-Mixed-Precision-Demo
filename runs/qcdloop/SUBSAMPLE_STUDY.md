# Right-sizing N for characterization — subsampling study (2026-07-18)

**Method (cheap, no new characterization, no LLM).** The finalized
`report_100k.json` is *not* sample-sliceable (its per-region aggregates are
collapsed across all 100k samples). But the 200 per-chunk **shard reports** that
built it survived in `/tmp/qcdloop_shards/` (`shard_<offset>.json`, each = 500
samples/integral). A `stability_shard_report` merges associatively, so **merging
the first K shards == the report for the first K×500 samples per integral**, and
this is bit-exact with a single `[0,K×500)` run (the driver's `--sample-offset`
chunking guarantees it). N therefore maps to a shard prefix: 5k→10, 10k→20,
25k→50, 50k→100, 100k→200. We fold the shards once in offset order, snapshotting
region state at each N (`runs/qcdloop/subsample_study.py`). Only **region**
aggregates are merged (reusing the reducer's own `_merge_region` /
`_classify_region`); `prov_vars` and the huge per-sample `variables` blob are
dropped — they drive report size/merge cost but are irrelevant to region/tier
saturation.

> No `--sample-limit` flag was needed on the reducer: the reducer consumes
> journals (all deleted), and shard-prefix merging *is* the sample slicing the
> map/merge natively supports.

**Validation.** The replay's @100k tier counts equal `report_100k.json`'s
aggregate `class_counts` exactly: `stable 1168 / log_near_root 84 /
cancellation_cascade 80 / local_cancellation 35` (and B1 = 35 regions all
stable, matching the frozen report). The region replay is faithful.

## Region count + tiers + Jaccard vs N

| N | shards | distinct regions¹ | distinct locations | stable | log_near_root | local_cancel (gate-b) | cancel_cascade² | **non-stable** | J_all vs prev | J_nonstable vs prev | tier churn³ |
|---|--------|-------------------|--------------------|--------|---------------|-----------------------|-----------------|----------------|---------------|---------------------|-------------|
| 5k   |  10 | 1335 | 500 | 1189 | 37 | 27 | 82 | **146** | — | — | — |
| 10k  |  20 | 1335 | 500 | 1192 | 39 | 29 | 75 | **143** | 1.000 | 0.940 | 11 |
| 25k  |  50 | 1363 | 510 | 1192 | 57 | 33 | 81 | **171** | 0.979 | 0.794 | 32 |
| 50k  | 100 | 1364 | 510 | 1173 | 74 | 35 | 82 | **191** | 0.999 | 0.866 | 28 |
| 100k | 200 | 1367 | 510 | 1168 | 84 | 35 | 80 | **199** | 0.998 | 0.902 | 19 |

¹ distinct `(integral, location)` pairs — the true per-integral hotspot count.
² **region-level** `cancellation_cascade` signal class (high rel_err + low
per-op cond). This is *not* the localized `cascade_chains` tier — the frozen
shards predate cascade post-processing and carry **zero** `cascade_chains`, so
the localized cascade tier (step 4) is **not measurable here and is skipped**.
³ regions present at both N whose signal class flips between them.

## Cost per N

Compute cost is linear in N; the replay-merge cost is a separate (tiny) axis.

| N | chunks | compute (core-hr)⁴ | wall @W=16⁴ | shards on disk | replay-merge wall⁵ | replay RSS⁵ |
|---|--------|--------------------|-------------|----------------|--------------------|-------------|
| 5k   |  10 | ~1.0  | ~6 min  | 4.4 GB | 20.8 s | 1.97 GB |
| 10k  |  20 | ~2.0  | ~12 min | 8.8 GB | 40.2 s | 1.97 GB |
| 25k  |  50 | ~5.1  | ~25 min | 22 GB  | 97.9 s | 1.97 GB |
| 50k  | 100 | ~10.1 | ~44 min | 44 GB  | 193 s  | 1.97 GB |
| 100k | 200 | ~20.3 | ~81 min | 88 GB  | 386 s  | 1.97 GB |

⁴ Derived from the only preserved timing (`run_chunked_100k.log`, a 59-of-200
resume run): ~365 s single-worker per 500-sample chunk, 4 waves of 16 in 1496 s.
Full 200-chunk run ≈ **20 core-hr / ~75–80 min wall @ W=16**. **Peak journal on
disk is width-bound (~82 GB = 16×5.12 GB) and independent of N** — only the kept
shards (0.44 GB/chunk) and final report (13.7 GB @100k) scale with N. The first
141 chunks' log was not preserved; these are estimates from the resume cadence.
⁵ This study's region-only fold: RSS is flat (region state is tiny; ~1 shard in
flight); wall is just orjson parsing 0.44 GB×(N/500). The N-dependent RAM/size of
the *real* pipeline lives entirely in the dropped `variables`/`prov_vars` blob,
which `fast_merge` already keeps bounded via per-integral partitioning.

## Verdict

**Two saturation curves, two knees.**

- **Region *coverage* (which code regions exist) saturates almost immediately.**
  `J_all ≈ 1.0` throughout; 5k already holds 1335/1367 = **97.6%** of all regions
  and 500/510 locations. The only genuine new-location discovery happens between
  10k and 25k (`J_all` 0.979, +10 locations). Past 25k, no new code regions
  appear. If all you need is the region *map*, **10k is plenty**.

- **The *remediation set* (non-stable regions + correct tier) needs more and is
  still moving at 100k.** Non-stable regions grow 146→143→171→191→199, and 19
  regions still flip tier in the final 50k→100k doubling (`J_nonstable` 0.90).
  Decomposing: `local_cancellation` (gate-b) **saturates by ~50k** (27→29→33→35→35);
  the region-level cascade signal is flat/noisy ~80 from 5k; but **`log_near_root`
  never plateaus** — 37→39→57→74→84, still +10 (+13.5%) in the last doubling.
  Those are exactly the elevated-conditioning (cond 1e6–1e15) regions that rare
  bad input draws keep surfacing, and they are precisely what upcast remediation
  targets.

**Recommendation.**
- **Dev / pipeline-iteration loop → 10k.** Full region map, ~72% of the
  non-stable set, ~2 core-hr, ~12 min. Good enough to exercise Strategy end-to-end;
  *not* a basis for final remediation decisions.
- **Real characterization Strategy acts on → 50k.** The knee for the actionable
  set: `local_cancellation` and cascade tiers saturated, coverage complete,
  `log_near_root` at 74/84 = 88% of its 100k value. Going 50k→100k costs 2×
  (~10 more core-hr, ~40 min, +44 GB shards) to gain **10** more `log_near_root`
  regions and settle 19 tier flips — real but sharply diminishing.
- **100k only if the `log_near_root` tail is mission-critical**, since that tier
  has not plateaued. But brute-forcing N past 100k is the wrong lever: the tail
  is rare-input-driven, so an **importance-sampled follow-up over the
  high-condition input subspace** would surface it far cheaper than doubling N
  again.

**Bottom line: the coverage knee is ~10k, the actionable-tier knee is ~50k.
Default the next full run to 50k; reserve 100k (or targeted importance sampling)
for chasing the `log_near_root` tail.**
