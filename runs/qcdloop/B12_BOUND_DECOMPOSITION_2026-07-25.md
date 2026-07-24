# Item 6-revised — B12 COMPUTED vs ALGORITHMIC via tracked-bound decomposition

**Date:** 2026-07-24 · **Branch:** langgraph-agents · head at start `85b9eb7`
**Mode:** read-only over existing 5k shards; no runs, no builds, no LLM, no >dd reference.

## Verdict: **COMPUTED cancellation**

> **COMPUTED — at B12's floor the tracked first-order bound is *tight at the
> cascade-chain level*: the dilog→combination chain (`kokkosUtils.h:212/702` →
> `B2m.h:206/207/241`) has `max_sensitivity = 1.03e22`, so
> `predicted_rel_err_if_double = 1.15e6` against a measured chain rel-err of
> `1.62e7` — within ~1 order (tightness 0.07). The observed catastrophic
> cancellation is therefore machine-ε roundoff amplified by the chain, not an
> analytic property invisible to precision. Since cond and amp are
> precision-invariant, whole-chain dd drops the amplified error by
> `U_double/U_dd = 2^53 ≈ 15.95` orders, lifting B12's floor from **3.69 digits
> to ~19–20 digits** (dd's ~31.9-digit precision minus the ~12.3-digit
> cancellation loss). Tier-B (call-graph dd fan-out over this *bounded* chain)
> has demonstrable value and is the honest post-Stage-2 phase.**

The sanity leg passes: all 7 measured-INERT single-region dd cells have
`predicted_rel_err_if_dd ≤ their measured baseline delta`, so the bound is not
systematically loose on this codebase — and B12 `kokkosUtils.h:212` shows exactly
*why single-region dd is inert while whole-chain dd works* (below).

---

## 1. Method

Reuses `agents/shared/stability_reducer.py`'s `merge_reports` + `finalize_report`
(the `merge` CLI path) verbatim; the one-off driver
`runs/qcdloop/b12_bound_decomposition.py` only (a) trims each of the 10 shards to
the integrals of interest before merging, and (b) adds the two rungs the shard
schema does not ship. The tracked-bound framework (reducer module header):

* `max_sensitivity = cond × amp` — the first-order forward-cone amplification of a
  machine-ε roundoff injected at a region/chain to the observable output. `cond`
  and `amp` are properties of the **math function**, invariant to working
  precision; only the injected unit-roundoff `U` changes with the rung.
* `predicted_rel_err_if_<rung> = U_<rung> × max_sensitivity`.

| rung | U | value |
|------|-----|-------|
| float | 2⁻²⁴ | 5.96e-8 |
| ff | 2⁻⁴⁶ | 1.42e-14 |
| double | 2⁻⁵³ | 1.11e-16 |
| dd | 2⁻¹⁰⁶ | 1.23e-32 |

(The prompt cited `U_ff=2⁻⁴⁴`; the reducer's own `U_FF = 2⁻⁴⁶` empirical ff floor
is used as source of truth. `U_double=2⁻⁵³` and `U_dd=2⁻¹⁰⁶` as prompted.)

**Discriminator.** `predicted_if_dd = predicted_if_double × 2⁻⁵³` *mechanically*,
so "is `predicted_if_dd ≪ measured`?" is trivially true and not the test. The real
test is **whether the first-order model explains the measured error** — i.e. is
`predicted_if_double ≈ measured`? If tight → the error IS roundoff amplification →
dd (which changes `U`) recovers it → COMPUTED. If `predicted_if_double ≪ measured`
→ the region's own cone does not capture the error and the bound cannot quantify
the dd lift at that scope.

**Data:** `/vast/projects/pepper_hep/qcdloop_shards_5k/` — 10 shards, 5000 samples
(500 × 10), written 2026-07-22. B12 present with 72 regions + 936 cascade chains;
per-region `max_sensitivity` intact. Full numbers: `b12_bound_decomposition.json`.

---

## 2. The hotspot region `B2m.h:241` — region-level bound is LOOSE (expected)

`res(i,0) = … − 2·dilog1 − dilog2 − dilog3` (Item 5's floor region), merged over 5k:

| field | value |
|-------|-------|
| signal_class | `log_near_root` |
| max_cond | 2.42e9 |
| max_amp | 4.71e9 |
| max_sensitivity | 4.71e9 |
| **measured max_rel_err** | **1.62e7** |
| predicted_rel_err_if_float | 2.81e2 |
| predicted_rel_err_if_ff | 6.69e-5 |
| predicted_rel_err_if_double | 5.23e-7 |
| predicted_rel_err_if_dd | 5.80e-23 |
| **tightness (pred_double/measured)** | **3.2e-14** |

At the **region** level the bound under-predicts by ~14 orders. This is *not* an
algorithmic signal — it is the documented limitation of a per-region forward cone:
the error at `res(i,0)` is **injected upstream** (in the double dilog evaluations)
and amplified *into* line 241 by the cancellation; region 241's *own* `cond × amp`
only measures amplification *from* 241 onward, so it cannot see the upstream
injection. Reading the region bound alone would wrongly suggest "loose ⇒
algorithmic." The correct instrument is the cascade chain, which spans the
injection site and the cancellation together.

> Measured `rel_err` values here exceed 1 because the tracked shadow error is
> normalized by a result that passes through/near zero under cancellation — it is
> an *amplification* indicator, not a bounded accuracy. Predicted values share the
> same normalization (`U × sens`), so the **ratio** (tightness) is a valid
> apples-to-apples comparison independent of that scale.

---

## 3. The cascade chain — bound is TIGHT (the decisive result)

The reducer's `cascade_chain` objects localize the accumulated cancellation the
per-region bound misses. B12's top chains by measured rel-err (distinct line-sets):

| measured | max_sensitivity | pred_double | pred_dd | tightness | n | chain lines |
|---------:|----------------:|------------:|--------:|----------:|--:|-------------|
| **1.62e7** | **1.03e22** | **1.15e6** | **1.27e-10** | **0.07** | 6 | `B2m.h:206,207,241` + `kokkosUtils.h:212,702` |
| 9.53e6 | 6.09e21 | 6.76e5 | 7.51e-11 | 0.07 | 11 | + `B2m.h:533` + `kokkosUtils.h:174,177,703` |
| 3.08e2 | 2.78e9 | 3.08e-7 | 3.42e-23 | ~1e-9 | 11 | `B2m.h:241,534` + `kokkosUtils.h:231,935,936,937` |
| 1.10e1 | 7.02e15 | 7.79e-1 | 8.65e-17 | 0.07 | 8 | `B2m.h:205,208,216,241,533` + `kokkosUtils.h:212` |
| 3.58e0 | 2.29e15 | 2.54e-1 | 8.65e-17 | 0.07 | 9 | `B2m.h:206,207,241` + `kokkosUtils.h:199,212` |

The dominant chain is **exactly the dilog → combination cascade** Item 5
identified: the Chebyshev dilog (`kokkosUtils.h:212/702`) feeding the near-equal
subtraction at `res(i,0)` (`B2m.h:206/207/241`). At **chain** scope the first-order
bound is **tight**: `predicted_if_double = 1.15e6` vs measured `1.62e7`, tightness
**0.07** — the model recovers the measured amplification to within ~1 order (and on
the conservative-low side). The mechanism is confirmed **finite-precision roundoff
amplified by cancellation = COMPUTED**.

Whole-chain dd: `predicted_if_dd = 1.27e-10 = predicted_if_double × 2⁻⁵³`, a
**15.95-order** reduction of the amplified error.

**Translating to the whole-app floor** (scale-normalized, validator's metric): double
retains 3.69 of ~16 digits → cancellation loss ≈ **12.3 digits**. dd carries ~31.9
working digits; minus the same 12.3-digit loss → **≈ 19.6 digits** (rel-err ~2.5e-20).
So whole-chain dd lifts B12's floor from **3.69 → ~19–20 digits**. (The chain bound
under-predicts measured by ~14×, so a conservative read is ~18–20 digits.)

---

## 4. Top B12 regions by raw `max_sensitivity` — why the chain object is needed

| location | sensitivity | measured rel_err | pred_double | tightness | class |
|----------|------------:|-----------------:|------------:|----------:|-------|
| B2m.h:534 | 1.05e22 | 1.05e-14 | 1.16e6 | 1.1e20 | stable |
| boxGPU.h:79 | 1.05e22 | 2.22e-16 | 1.16e6 | 5.2e21 | stable |
| boxGPU.h:91/92/95 | 1.05e22 | 3.33e-16 | 1.16e6 | 3.5e21 | stable |

Raw region-sensitivity ranking is dominated by **`stable`, well-conditioned regions
with long forward cones** whose *measured* error is ~machine-ε (1e-14–1e-16). Here
`predicted_double ≫ measured` (tightness ~1e20–1e21): the amp bound is
**conservatively over-flagging** (as its docstring warns — it ignores max-gating).
These are *not* floor drivers. This is precisely why the cascade-chain localization
exists and is the right instrument: it filters to actual cancellation victims
(§3), where the bound is tight, rather than long-cone stable regions where it is
loose-high.

---

## 5. Sanity check — 7 measured-INERT dd cells (Item 5)

For a single-region dd promotion measured byte-identical (INERT), the region's
`predicted_rel_err_if_dd` should sit at/below the integral's achievable floor
(baseline delta) — else the bound would be over-claiming a lift dd did not deliver.

| integral | region | class | sensitivity | pred_dd | baseline dd Δ | pred_dd ≤ Δ |
|----------|--------|-------|------------:|--------:|--------------:|:-----------:|
| B10 | kokkosUtils.h:212 | log_near_root | 5.89e14 | 7.26e-18 | 1.324e-10 | ✅ |
| B8  | kokkosUtils.h:212 | log_near_root | 1.76e11 | 2.17e-21 | 7.266e-11 | ✅ |
| B8  | kokkosUtils.h:206 | stable | 9.20e11 | 1.13e-20 | 7.266e-11 | ✅ |
| B8  | kokkosUtils.h:199 | stable | 9.21e11 | 1.14e-20 | 7.266e-11 | ✅ |
| B8  | kokkosUtils.h:174 | stable | 2.81e10 | 3.46e-22 | 7.266e-11 | ✅ |
| B9  | kokkosUtils.h:212 | log_near_root | 1.23e10 | 1.52e-22 | 2.950e-12 | ✅ |
| B12 | kokkosUtils.h:212 | log_near_root | 2.24e10 | 2.77e-22 | 2.039e-4  | ✅ |

All 7 pass — the bound is reliable on this codebase (necessary check for trusting
§3). The B12 row is the key corroboration of the COMPUTED story **and** of why
single-region dd is inert:

* `kokkosUtils.h:212` (the dilog) has `predicted_if_dd = 2.77e-22`, ~18 orders
  **below** B12's floor `2.039e-4`. So the dilog region's *own* cone contribution
  is already far sub-floor — widening only 212 cannot move the floor, exactly the
  measured INERT result (byte-identical).
* Yet the **chain** that *contains* 212 (§3) has `max_sensitivity = 1.03e22` and
  drives the whole floor. The floor is a **chain property**: widening one link
  leaves the other links' double roundoff to be amplified by the same cancellation.
  Single-region dd is inert ⇔ whole-chain dd is required — and (§3) whole-chain dd
  works. This is the crisp reconciliation of Item 5's INERT measurements with a
  COMPUTED verdict.

---

## 6. Implication for Tier-B (post-Stage-2)

The floor-driving subgraph is **bounded and localized**, not a sprawling call
graph — the two dominant chains cover:

* dilog (Chebyshev) branches: `kokkosUtils.h:174,177,199,212,702,703`
* B2m combination lines: `B2m.h:205,206,207,208,216,241,533,534`

A Tier-B call-graph dd fan-out would need to widen the **dilog evaluation +
its B2m combination as one coordinated multi-file shim** (arguments → `ql::ddilog`
→ the `res(i,0)` subtraction), all in dd, so no double roundoff is left upstream of
the cancellation. That is a well-scoped next phase — a bounded coordinated
promotion, materially smaller than a whole-app dd build.

---

## 7. Caveats (honest bounds on this diagnostic)

* **Bound-derived, not measured.** The ~19–20-digit dd floor is an extrapolation of
  a first-order worst-path model, corroborated by the tight chain-level fit (0.07)
  and the 7-cell sanity leg — not a dd-vs-quad measurement (that remains the
  Item-6-original feature, still unbuilt).
* **`rel_err` normalization.** Chain/region `max_rel_err` blows up past 1 near the
  cancellation zero-crossing; tightness is used as a scale-invariant *ratio*, which
  is the sound comparison. The digit-lift translation uses the validator's
  scale-normalized floor (3.69) independently.
* **Region-level looseness is real** and would mislead on its own; the verdict
  rests on the **chain-level** fit, which is the instrument the reducer already
  produces for accumulated cancellation.

---

## 8. Reproducibility

```bash
.venv/bin/python runs/qcdloop/b12_bound_decomposition.py
# reads /vast/projects/pepper_hep/qcdloop_shards_5k/shard_*.json (~50s),
# reuses stability_reducer.merge_reports + finalize_report,
# writes runs/qcdloop/b12_bound_decomposition.json + console tables.
```

Baseline floor (3.6906 digits, sample 3868, coeff0.imag) from
`SOLVER_STAGE1_DD_PROBE.md` (Item 3); INERT baseline deltas from
`DD_TRIAGE_2026-07-25.md` (Item 5).

---

## 9. Recommendation

**Verdict COMPUTED.** Launch **Stage 2 speedup-only now** (Item 5's Path 1 remains
correct for the single-region measurement layer), and adopt **Tier-B (call-graph dd
fan-out over the dilog→combination chain, §6) as the honest post-Stage-2 phase** to
recover B12's floor. Do not launch Stage 2 until Reet reviews this verdict.

Read-only: no solver/fan-out/gate/manifest changes; no Stage 2 launched.
