# Phase B probe — per-integral intent divergence

_Read-only measurement, report_5k.json, generated 2026-07-22. No pipeline change; replays Strategy's `load_regions`/`load_chains`/`build_queues` verbatim._

## Verdict

**WELL MOTIVATED — 41.1% of shared lines disagree (>25%). Worst-casing throws away real per-integral signal.**

A *precision decision* here is the **intent target** Strategy emits from the report (cheapest rung the walk may aim for) — not the Validator-settled final precision (no Patcher/Validator runs). Merging happens upstream of the walk, so intent divergence is exactly the signal Phase B routing would recover.

## Inputs

- report schema_version: **2** (v2 = per-record `integral` tag, Phase A commit 5b6b82c)
- tolerance: **10.0** precise digits → rel-err bar `1e-10`
- ladder (cheap→dear): `float < ff < double < dd`
- non-localizable region entries skipped: 103

## Region count: merged vs unmerged

| view | records | note |
|---|---|---|
| merged (`merge=True`, today) | 480 | one worst-case region per source line |
| unmerged (`merge=False`) | 1232 | one region per (integral, line) |
| **ratio** | **2.57×** | avg integrals compiling a shared line |

## Same-line precision disagreement

Restricted to source lines that appear in ≥2 integrals (the only lines the merge can lose signal on). A line *disagrees* when ≥2 of its integrals would emit different intent targets.

| metric | count | fraction of shared |
|---|---|---|
| shared lines (≥2 integrals) | 107 | 100% |
| **agree** (all integrals same precision) | 63 | 58.9% |
| **disagree** (≥2 precisions) | 44 | 41.1% |

## Disagreement magnitude distribution

Among disagreeing lines, the set of distinct intent targets across integrals:

| precision span | lines |
|---|---|
| {float, double} | 30 |
| {float, double, dd} | 6 |
| {double, dd} | 6 |
| {float, dd} | 1 |
| {float, ff} | 1 |

## Top-20 wasted-headroom lines

The direct "what would routing buy" table: lines the merge forces to a dear precision that N−1 integrals would escape. `wasted` = Σ ladder-steps of over-precision across integral instances; `cheaper` = integrals that would get a lower rung under routing; `forcers` = integrals pinning the worst case.

| # | line | merged | per-integral | wasted | cheaper | forcer(s) |
|---|---|---|---|---|---|---|
| 1 | `boxGPU.h:101` | dd | float×5, double×8, dd×8 | 23 | 13/21 | B10, B13, B5, BIN0, BIN1, BIN2, BIN3,… |
| 2 | `boxGPU.h:99` | dd | float×5, double×8, dd×8 | 23 | 13/21 | B10, B13, B5, BIN0, BIN1, BIN2, BIN3,… |
| 3 | `boxGPU.h:100` | dd | float×3, double×11, dd×7 | 20 | 14/21 | B10, B13, B2, B4, B5, B7, B8 |
| 4 | `boxGPU.h:104` | double | float×8, double×13 | 16 | 8/21 | B1, B10, B12, B13, B15, B16, B2, B3, … |
| 5 | `kokkosUtils.h:140` | dd | float×1, double×8, dd×7 | 11 | 9/16 | B11, B12, B13, B2, B3, B4, B5 |
| 6 | `B0m.h:405` | double | float×5, double×1 | 10 | 5/6 | BIN0 |
| 7 | `B1m.h:283` | double | float×5, double×1 | 10 | 5/6 | BIN1 |
| 8 | `kokkosUtils.h:689` | dd | float×2, double×3, dd×2 | 9 | 5/7 | B12, B13 |
| 9 | `kokkosUtils.h:690` | dd | float×2, double×3, dd×2 | 9 | 5/7 | B12, B13 |
| 10 | `kokkosUtils.h:156` | dd | double×8, dd×2 | 8 | 8/10 | B12, B13 |
| 11 | `kokkosUtils.h:181` | dd | double×8, dd×2 | 8 | 8/10 | B12, B13 |
| 12 | `B0m.h:408` | double | float×3, double×3 | 6 | 3/6 | B3, B5, BIN0 |
| 13 | `B0m.h:410` | double | float×3, double×3 | 6 | 3/6 | B4, B5, BIN0 |
| 14 | `B1m.h:286` | double | float×3, double×3 | 6 | 3/6 | B10, B9, BIN1 |
| 15 | `B2m.h:529` | double | float×3, double×1 | 6 | 3/4 | BIN2 |
| 16 | `boxGPU.h:93` | double | float×3, double×18 | 6 | 3/21 | B10, B11, B12, B13, B15, B16, B2, B3,… |
| 17 | `boxGPU.h:94` | double | float×3, double×18 | 6 | 3/21 | B10, B11, B12, B13, B15, B16, B2, B3,… |
| 18 | `boxGPU.h:95` | double | float×3, double×18 | 6 | 3/21 | B10, B11, B12, B13, B15, B16, B2, B3,… |
| 19 | `boxGPU.h:96` | double | float×3, double×18 | 6 | 3/21 | B10, B11, B12, B13, B15, B16, B2, B3,… |
| 20 | `boxGPU.h:127` | dd | float×2, dd×1 | 6 | 2/3 | BIN0 |

## Adversarial: is the disagreement a lone-outlier artifact?

If each worst-case were pinned by a single dominant integral (e.g. one BIN cascade always demanding dd), routing would still buy the N−1 others their cheaper rung — so a high sole-forcer count *strengthens* the payoff case; it does not weaken it. What would weaken it: the worst-case being demanded by *most* integrals (then routing helps few).

- disagreeing lines pinned by a **single** integral at the worst rung: **15/44** (34.1%) → these are clean N−1 wins.

Worst-case *forcer* concentration (how often each integral pins a shared line's worst case) — a flat spread means no single integral explains the divergence:

| integral | lines it forces |
|---|---|
| B13 | 21 |
| B12 | 20 |
| B5 | 15 |
| B10 | 14 |
| B2 | 13 |
| B3 | 13 |
| BIN0 | 12 |
| B9 | 12 |
| B4 | 11 |
| B8 | 11 |
| BIN1 | 10 |
| BIN2 | 9 |

- merge-result ≠ worst-case-of-parts on **0** shared lines (sanity: `_merge_by_line` should equal the per-integral max; non-zero here flags where re-running the tier logic on worst-cased *signals* lands on a different rung than the max of independent decisions).

## Cascade-chain analogue

264095 chains loaded (not merged by Strategy; deduped by representative line at walk time, worst-case distributed). A chain is *worked* → floors its lines toward `dd` iff `max_rel_err > thr` (`_rank_chains` eligibility); else its lines stay `double`.

> **Degenerate decision.** 264095/264095 chains (100.0%) are worked — every cascade chain is ill-conditioned enough to demand `dd`. The chain decision therefore carries **no per-integral variation** to recover: worked vs not-worked is the only axis, and it is uniformly "worked". Per-integral routing on chains has structurally **zero** payoff on this codebase — not because integrals happen to agree on a nuanced split, but because there is no split to make. This is the correct null result to compare against the region finding below.

**Case (a) — same `chain_id` across integrals, differing decision:**

| metric | count | fraction |
|---|---|---|
| chain_ids in ≥2 integrals | 0 | 100% |
| … that disagree (worked in one, not another) | 0 | 0.0% |

_(0 shared chain_ids ⇒ `chain_id`s are namespaced per integral and never recur across them; the cross-integral question lives entirely in case (b).)_

**Case (b) — a source line covered by chains in ≥2 integrals, differing per-integral chain-precision:**

| metric | count | fraction |
|---|---|---|
| lines covered by chains in ≥2 integrals | 27 | 100% |
| … that disagree (dd in one integral, double in another) | 0 | 0.0% |

## Method & faithfulness

- Decisions come from Strategy's own `build_queues` (correctness + speedup queues) plus the WI1 float-rung gate (`value_range_ok_for_float`), mirroring `agent._float_rung_ok`. No tier predicate is re-implemented.
- Per-integral decisions run `build_queues` **inside each integral** so the speedup exclude-by-key logic is clean (line keys are unique within one integral). This is the faithful "if this integral were alone" counterfactual that routing would realize.
- `float_via` (plain vs regional) affects only the *path* to a rung, never the cheapest reachable target, so the repo source probe is not consulted.
- Intent target ≠ Validator-settled precision: the walk may settle dearer if the Validator rejects a demotion. This probe bounds the *upstream* signal the merge destroys before the walk ever runs — the ceiling on routing payoff.

