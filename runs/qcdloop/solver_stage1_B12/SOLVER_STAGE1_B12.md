# Solver Stage 1 — B12 (greedy, p100 ≥ 6)

Phase 2e Stage 1: the first pipeline stage that **writes an optimized source tree**.  Greedy sequential-layering over the fan-out's measured DISCRIM `(region, rung)` cells, ranked float<ff<dd, each applied on top of the accumulated tree, kept iff the whole-app p100 precise-digits gate holds.

* **Integral:** B12 (Stage-1 scope — B12 only)
* **Manifest:** `/home/rbarik/Agentic-Mixed-Precision-Demo/runs/qcdloop/per_integral_out_2e_measure/B12/manifest_scorer_B12.jsonl`
* **Gate:** p100 = min_precise_digits across the random battery ≥ **6** (FP128 whole-app oracle). Locked; not the PLAN default p99≥10 (per FLOAT_RETRO_PROBE.md the aggregate whole-app floor is ~8.84 digits at BIN1). **See the blocking finding — for the B12 pass the whole-app floor is B12's own 3.69, below the gate.**
* **Snapshot:** seed=12345, sample_count=5000
* **Merged tree:** `/home/rbarik/Agentic-Mixed-Precision-Demo/runs/qcdloop/solver_stage1_B12/tree_B12` (HEAD `6eaf4a72cf24`)
* **Cumulative diff:** `/home/rbarik/Agentic-Mixed-Precision-Demo/runs/qcdloop/solver_stage1_B12/final.diff`
* **Solve wall:** 25.5s

## Headline

**STOPPED — `stopped_gate_unimplementable`.** baseline whole-app min_precise_digits=3.6906 < gate=6.0; no candidate can satisfy p100>=gate when the unpatched tree already fails it

## ⛔ Stage-1 blocking finding — the p100 gate is unsatisfiable for B12

The solver stopped on the **baseline**: the unpatched whole-app min_precise_digits (p100) is **3.6906 < gate 6**, so *no* candidate can pass — the floor is set before the solver touches anything.  This is the explicit STOP-and-flag case (PLAN 2e §Gate: "do not silently pick a tolerance other than 6.0; if the gate is structurally unimplementable, STOP and flag").  The solver did **not** retune the gate.

**Hotspot:** integral `B12`, sample 3868, component `coeff0.imag` — precise digits 3.6906 (rel-err ≈ 2.04e-04 vs the FP128 oracle).

**Per-integral double-precision floor (vanilla whole-app, this battery).** Only the *target* integral is the global-min hotspot:

| integral | worst-case p100 | < gate? |
|----------|----------------|:-------:|
| `B12` ← target | 3.6906 | **yes** |
| `B16` | 6.5693 | no |
| `BIN1` | 8.0947 | no |
| `B13` | 8.5777 | no |
| `BIN3` | 8.9081 | no |
| `BIN2` | 9.1220 | no |
| `B11` | 9.4601 | no |
| `BIN4` | 9.5803 | no |
| `B10` | 9.8781 | no |
| `B8` | 10.1387 | no |
| `B4` | 10.2498 | no |
| `B9` | 11.5301 | no |
| `B5` | 11.5853 | no |
| `B7` | 11.6247 | no |
| `B6` | 11.7706 | no |
| `B3` | 11.9853 | no |
| `B15` | 12.0087 | no |
| `B2` | 12.1421 | no |
| `B1` | 12.1585 | no |
| `BIN0` | 12.1904 | no |
| `B14` | 13.1855 | no |

1 of 21 integrals sit below the gate at double precision — and the target `B12` is the worst.

### Why this is genuine, not an artifact

The Validator scores each component against a **per-sample `ref_scale`** (the largest |DD coeff| in that sample) with an analytic-zero band (`effectively_zero` → capped, counted in `zeroed_components`).  A near-zero-reference component would therefore report ~0 digits *or be banded out* — not a moderate 3.69.  3.69 digits = a real 2.04e-4 relative error against the sample's characteristic magnitude: a **genuine double-precision catastrophic-cancellation floor** intrinsic to B12's algorithm at that sample, not a scoring artifact.

### Why no candidate can lift it

Every measured DISCRIM candidate (`B2m.h:188 float`, `boxGPU.h:139 float`, `boxGPU.h:139 ff`) leaves the hotspot component untouched — the first candidate (`B2m.h:188 float`) built + validated cleanly but produced p100 3.6906 = baseline (Δ ≈ 0 on the global min).  The dd upgrades that *could* add precision are exactly the measured-INERT cells (delta == baseline) — they do not touch the cancellation either.  So the floor is invariant under the entire catalog the fan-out measured for B12.

### The measurement-layer gap this exposes (the point of Stage 1)

The **whole-app global-min gate is the wrong instrument for a per-integral solver whose target integral is itself the global-min hotspot.**  FLOAT_RETRO_PROBE.md already recommended a *per-component, float-touched* instrument over the global-min gate; this run makes it concrete.  Options for Reet (the solver deliberately picks none — locked at 6.0):

1. **Regression-relative gate** — accept iff the candidate does not *worsen* the whole-app min beyond a small margin vs the double baseline (this is exactly `validate()`'s built-in 0.5-digit regression guard). Under it, B12 float candidates that leave the 3.69 floor untouched (Δ≈0) would pass — the solver would accept float where it is harmless and land real speedup, which is the actual intent for an ill-conditioned integral.  This is the smallest change and the most defensible.
2. **Per-target-integral absolute gate** — score p100 over the target integral's components only, against a floor calibrated to *its* achievable precision (e.g. its dd-oracle self-consistency), not the whole-app 6.0.
3. **Hotspot mask** — exclude the provably-cancellation components from the absolute floor (they are workload physics ceilings that bind double itself), keeping 6.0 on the rest.

**Recommendation:** option 1 (regression-relative) for Stage 2 — it preserves an absolute-safety intuition while not penalizing the solver for an ill-conditioning it cannot fix.  Needs your sign-off; it changes the locked gate semantics.

## Candidate queue (rank order)

3 DISCRIM competitors; 5 measured-INERT excluded (byte-identical whole-app output → no speedup); 19 never reached `measured` (2c/2d terminal gates).

| # | region | rung | why ranked here | Δ (region) | baseline Δ |
|---|--------|------|-----------------|-----------|-----------|
| 0 | `B2m.h:188` | float | cheapest rung (try first) | 2.039e-04 | 2.039e-04 |
| 1 | `boxGPU.h:139` | float | cheapest rung (try first) | 2.039e-04 | 2.039e-04 |
| 2 | `boxGPU.h:139` | ff | float-float; tried when float rejected | 2.039e-04 | 2.039e-04 |

> Intra-rung tiebreak = region_id ascending (deterministic; the measurement layer gives no principled cross-region order within a rung — flop-weighting is a v2 refinement, see handoff).

## Per-candidate outcomes

| region | rung | outcome | p100 before | p100 after | validator | wall | reason |
|--------|------|---------|------------|-----------|-----------|------|--------|
| `B2m.h:188` | float | **stopped_gate_unimplementable** | 3.6906 | 3.6906 | reject | 25.5s | baseline whole-app min_precise_digits=3.6906 < gate=6.0; no candidate can satisfy p100>=gate when the unpatched tree already fails it |

## Final precision distribution (B12 regions)

| precision | region count |
|-----------|-------------|
| float | 0 |
| ff | 0 |
| dd | 0 |
| double | 18 |

**Final whole-app min_precise_digits (p100): 3.6906** (baseline 3.6906).

## Regions where float was proposed but rejected

None — every float candidate that reached the gate held it (or no float candidate was rejected).

## Measured-INERT cells excluded from the queue

Byte-identical whole-app output (`delta_effective == baseline_delta_effective`): the promotion was a numerical no-op, so it carries no speedup and is left at double.  These are the residue the 2c/2d `promotion_no_op` / `write_truncation` gates could not prove statically pre-build.

| region | rung | Δ (== baseline) |
|--------|------|----------------|
| `kokkosUtils.h:212` | dd | 2.039e-04 |
| `boxGPU.h:141` | float | 2.039e-04 |
| `boxGPU.h:142` | float | 2.039e-04 |
| `B2m.h:193` | float | 2.039e-04 |
| `B2m.h:240` | float | 2.039e-04 |

## What the greedy assumption may leave on the table

* N/A — the walk never got past the baseline (the gate is unsatisfiable before any candidate could be judged). The solo-vs-joint question only becomes meaningful once the gate admits at least the baseline; see the blocking finding.

## Reet review before Stage 2

### Solver-design judgment calls

1. **Gate on the raw p100 metric, not the Validator's accept verdict.** `validate()`'s own verdict bundles a 0.5-digit *regression* guard vs the ~8.84-digit baseline, which would reject any candidate that legitimately spends precision down toward 6. The solver reads `candidate.min_precise_digits` and applies `≥ 6.0` itself. If you want the regression guard *too*, that is a one-line change — but it changes the locked semantics.
2. **Queue = measured DISCRIM only; measured-INERT excluded.** An INERT cell would trivially 'accept' (byte-identical → gate holds) and lock the region at a no-op rung ahead of a genuinely cheaper DISCRIM rung. Excluding them is both task-consistent ('no measured DISCRIM rung → stay double') and avoids that pathology.
3. **Intra-rung tiebreak = region_id ascending.** Deterministic but arbitrary; the measurement layer offers no principled cross-region order within a rung. Flop-weighting (WI3 table already exists) is the obvious v2 upgrade.
4. **`current_precision='double'` for every intent.** Sound only under first-accept-per-region (a region is patched at most once, from double). If v2 ever re-demotes an already-demoted region, this must thread the accumulated precision.

### Stage 2 (all 21 integrals) cost estimate

* B12 solve: **25.5s** wall; 1 of 3 queued candidates actually built+validated (~26s/candidate incl. Patcher fan-out build + whole-app validate; the rest were skipped/short-circuited — the run STOPPED on the baseline).
* Each candidate = 1 Patcher fan-out (LLM gen + build) + 1 whole-app validate (build-fused). API cost is dominated by the fan-out LLM calls (float/ff/dd shim generation).
* B12 queue (3) is small; other integrals with richer DISCRIM sets (e.g. B1 had 8 measured DISCRIM cells) will have larger queues. A rough upper bound for 21 integrals: if the mean queue is ~2× B12's and per-candidate wall holds, order-of-magnitude **0.9–1.8h sequential**, less with per-integral workers (the passes are independent, like run_all_integrals `--workers`).
* Refine this after Stage 2 review: the honest number needs the real per-integral queue sizes, which only the manifests give.

### Measurement-layer gaps the solver exposed

* No apply-failures: every measured DISCRIM cell re-generated cleanly on the accumulated tree.
* **5 measured-INERT cells** slipped past the 2c/2d static gates (they build + measure but produce byte-identical output). Not wrong, but each wasted one build in the measure pass; a tighter static gate would save that. Enumerated above.
* **The gate itself is the headline gap.** The baseline whole-app p100 for the B12 pass is 3.6906 — below the 6.0 gate — because the target integral is the whole-app global-min hotspot. This is *not* a snapshot/oracle drift (the FLOAT_RETRO ~8.84 figure is the aggregate run's BIN1 floor; this battery's global min is the target's own 3.69 sample). See the blocking finding for the gate-instrument options.

**STOP: do not run Stage 2 (all 21 integrals) until this is reviewed** — especially the gate-instrument decision, which Stage 2 depends on.
