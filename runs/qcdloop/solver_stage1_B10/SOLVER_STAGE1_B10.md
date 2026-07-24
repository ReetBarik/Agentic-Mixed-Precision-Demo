# Solver Stage — B10 (greedy, regression-relative gate, 0.5-digit margin)

Phase 2e: a pipeline stage that **writes an optimized source tree**.  Greedy sequential-layering over the fan-out's measured DISCRIM `(region, rung)` cells, ranked float<ff<dd, each applied on top of the accumulated tree, kept iff the whole-app p100 does not regress more than the margin below the double baseline.

* **Integral:** B10
* **Manifest:** `/home/rbarik/Agentic-Mixed-Precision-Demo/runs/qcdloop/per_integral_out_stage2/B10/manifest_scorer_B10.jsonl`
* **Gate:** regression-relative — accept iff candidate `min_precise_digits >= baseline_min_precise_digits - 0.5` (the same 0.5-digit regression guard `validate()` bundles; FP128 whole-app oracle).  Replaces the Stage-1 absolute `p100 >= 6` (Reet 2026-07-24).
* **Snapshot:** seed=12345, sample_count=5000
* **Merged tree:** `/home/rbarik/Agentic-Mixed-Precision-Demo/runs/qcdloop/solver_stage1_B10/tree_B10` (HEAD `5946f055fca2`)
* **Cumulative diff:** `/home/rbarik/Agentic-Mixed-Precision-Demo/runs/qcdloop/solver_stage1_B10/final.diff`
* **Solve wall:** 153.1s

## Headline

Queue exhausted normally. Baseline whole-app p100 = **3.6906**; accept threshold = **3.1906** (baseline − 0.5); final = **3.6906** digits.  5 accepted, 0 rejected, 0 apply-failed, 1 skipped (region already resolved).

## Candidate queue (rank order)

6 DISCRIM competitors; 2 measured-INERT excluded (byte-identical whole-app output → no speedup); 18 never reached `measured` (2c/2d terminal gates + signal_class `awaiting_algorithmic_rewrite`).

| # | region | rung | why ranked here | Δ (region) | baseline Δ |
|---|--------|------|-----------------|-----------|-----------|
| 0 | `B1m.h:227` | float | cheapest rung (try first) | 5.899e-05 | 1.324e-10 |
| 1 | `B1m.h:236` | float | cheapest rung (try first) | 3.292e-03 | 1.324e-10 |
| 2 | `B1m.h:237` | float | cheapest rung (try first) | 1.046e-02 | 1.324e-10 |
| 3 | `boxGPU.h:139` | float | cheapest rung (try first) | 1.444e-07 | 1.324e-10 |
| 4 | `boxGPU.h:141` | float | cheapest rung (try first) | 5.765e-08 | 1.324e-10 |
| 5 | `boxGPU.h:139` | ff | float-float; tried when float rejected | 1.324e-10 | 1.324e-10 |

> Intra-rung tiebreak = region_id ascending (deterministic; the measurement layer gives no principled cross-region order within a rung — flop-weighting is a v2 refinement, see handoff).

## Per-candidate outcomes

| region | rung | outcome | p100 before | p100 after | validator | wall | reason |
|--------|------|---------|------------|-----------|-----------|------|--------|
| `B1m.h:227` | float | **accepted** | 3.6906 | 3.6906 | reject | 22.0s | p100 3.6906 >= baseline 3.6906 - margin 0.5 = 3.1906 |
| `B1m.h:236` | float | **accepted** | 3.6906 | 3.6906 | reject | 19.1s | p100 3.6906 >= baseline 3.6906 - margin 0.5 = 3.1906 |
| `B1m.h:237` | float | **accepted** | 3.6906 | 3.6906 | reject | 71.1s | p100 3.6906 >= baseline 3.6906 - margin 0.5 = 3.1906 |
| `boxGPU.h:139` | float | **accepted** | 3.6906 | 3.6906 | reject | 19.9s | p100 3.6906 >= baseline 3.6906 - margin 0.5 = 3.1906 |
| `boxGPU.h:141` | float | **accepted** | 3.6906 | 3.6906 | reject | 20.9s | p100 3.6906 >= baseline 3.6906 - margin 0.5 = 3.1906 |
| `boxGPU.h:139` | ff | **skipped_region_resolved** | 3.6906 | — | — | 0.0s | region already at 'float' (cheaper rung accepted) |

## Final precision distribution (B10 regions)

| precision | region count |
|-----------|-------------|
| float | 5 |
| ff | 0 |
| dd | 0 |
| double | 12 |

**Final whole-app min_precise_digits (p100): 3.6906** (baseline 3.6906, threshold 3.1906).

Regions that moved off double:
* `B1m.h:227` → **float**
* `B1m.h:236` → **float**
* `B1m.h:237` → **float**
* `boxGPU.h:139` → **float**
* `boxGPU.h:141` → **float**

## Regions where float was proposed but rejected

None — every float candidate that reached the gate held it (or no float candidate was rejected).

## Measured-INERT cells excluded from the queue

Byte-identical whole-app output (`delta_effective == baseline_delta_effective`): the promotion was a numerical no-op, so it carries no speedup and is left at double.  These are the residue the 2c/2d `promotion_no_op` / `write_truncation` gates could not prove statically pre-build.

| region | rung | Δ (== baseline) |
|--------|------|----------------|
| `kokkosUtils.h:212` | dd | 1.324e-10 |
| `boxGPU.h:142` | float | 1.324e-10 |

## What the greedy assumption may leave on the table

* No candidate was rejected — the greedy walk accepted every DISCRIM competitor it could apply, so there is no solo/joint gap to record for this integral.
* Greedy first-accept-per-region takes the **cheapest** rung that holds, never re-examining whether a more conservative rung would have freed a sibling region to demote further.
* **2 measured-INERT cells** were excluded as no-ops. The 2c/2d investigation established these are structural no-ops, not hidden wins.

## Reet review

### Solver-design judgment calls

1. **Regression-relative gate (0.5-digit margin vs baseline).** Accept iff `cand_min >= baseline_min - 0.5`. The baseline is measured once on the unpatched tree at solve start and is the reference for the whole run. This is the same 0.5-digit figure `validate()` uses as its regression guard (`DEFAULT_MAX_REGRESSION`), reused not re-invented. It replaces the Stage-1 absolute `p100 >= 6`, which was unsatisfiable when the target integral is the whole-app global-min hotspot (B12 = 3.69).
2. **Queue = measured DISCRIM only; measured-INERT excluded.** An INERT cell would trivially 'accept' (byte-identical → no regression) and lock the region at a no-op rung ahead of a genuinely cheaper DISCRIM rung.
3. **Intra-rung tiebreak = region_id ascending.** Deterministic but arbitrary; flop-weighting (WI3 table exists) is the obvious v2 upgrade.
4. **`current_precision='double'` for every intent.** Sound only under first-accept-per-region (a region is patched at most once, from double).

### Cost

* B10 solve: **153.1s** wall; 5 of 6 queued candidates actually built+validated (~31s/candidate incl. Patcher fan-out build + whole-app validate; the rest were skipped/short-circuited).

### Measurement-layer notes

* No apply-failures: every measured DISCRIM cell re-generated cleanly on the accumulated tree.
* **2 measured-INERT cells** slipped past the 2c/2d static gates (build + measure but produce byte-identical output).
* Baseline whole-app p100 = 3.6906 (accept threshold 3.1906). Confirm this matches expectation for the target integral; a large mismatch would indicate snapshot/oracle drift.
