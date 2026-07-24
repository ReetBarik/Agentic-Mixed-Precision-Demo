# Item 7 — bound decomposition across all 21 integrals

**Date:** 2026-07-25 · **Branch:** langgraph-agents · head at start `2bf8723`
**Mode:** read-only over the existing 5k shards; no runs, no builds, no LLM, no
>dd reference. Reuses `stability_reducer.merge_reports` + `finalize_report`
verbatim; loops the Item-6-revised decomposition arithmetic over all 21 integrals.

## Headline

> Across the 21-integral suite the bound decomposition partitions cleanly into
> **10 STABLE_ALREADY** (no lift needed), **11 COMPUTED** (cancellation is
> roundoff amplification with a *tight* chain bound → a precision lever exists),
> and **0 ALGORITHMIC / 0 INCONCLUSIVE**. Every cascade floor in qcdloop is
> roundoff-amplified cancellation — there is **no purely-analytic floor** for
> which precision is irrelevant. But the 11 COMPUTED integrals split by whether
> **dd is enough**: `dd` fully recovers **4** of them (B12, B13, B14, B10 →
> ~19–26 digits) and is **insufficient** for **7** (B15, B16, BIN0–BIN4), whose
> cancellation loss (24–47 digits) exceeds dd's ~32-digit budget — those need
> beyond-dd (quad or algorithmic rewrite), out of Tier-B's dd scope. The
> floor-driving chains are **bounded** (3–10 lines, exactly two files each:
> one `kokkosUtils.h` special function + one `Bnm.h` combination) and cluster
> into **three shared-helper families** — most sharply, `ltspence`/`cspence`
> (`kokkosUtils.h:550/608`) drives **all five BIN integrals at once**, so a
> Tier-B dd promotion of a shared helper is inherently cross-integral.

**Load-bearing caveat.** Only 5 integrals (B1/B8/B9/B10/B12) have measured solver
floors; the other 16 floors are **derived** from the shard dominant-chain rel-err
via a model calibrated on those 5 points. For B13/B14 the derivation is in
calibration range; for **B15/B16/BIN0–4 it is extrapolated ~2–5× beyond the
calibration edge** and is low-confidence in the exact digit count (the *mechanism*
verdict is robust; the digit floor is not). See §5 and §7.

---

## 1. Method

Identical framework to `B12_BOUND_DECOMPOSITION_2026-07-25.md` (Item 6-revised):

* `max_sensitivity = cond × amp` — first-order forward-cone amplification of a
  machine-ε roundoff to an observable output. `cond`/`amp` are properties of the
  **math function**, precision-invariant; only the injected unit-roundoff `U`
  changes per rung. `predicted_rel_err_if_<rung> = U_<rung> × max_sensitivity`.

  | rung | U | value |
  |------|-----|-------|
  | float | 2⁻²⁴ | 5.96e-8 |
  | ff | 2⁻⁴⁶ | 1.42e-14 |
  | double | 2⁻⁵³ | 1.11e-16 |
  | dd | 2⁻¹⁰⁶ | 1.23e-32 |

* **Discriminator (chain tightness).** `tightness = predicted_if_double / measured`
  at the integral's **dominant cascade chain** (the `cancellation_cascade` chain
  with the largest measured rel-err). Within ~2 orders of 1 (`1e-3 ≤ t ≤ 1e1`)
  ⇒ the first-order model *explains* the measured error ⇒ error is roundoff
  amplified by cancellation ⇒ **COMPUTED** (dd, which changes `U` by 2⁻⁵³,
  recovers it). `t ≪ 1e-3` at chain scope ⇒ no chain cone captures the error ⇒
  **ALGORITHMIC**.

* **Floor.** 5 integrals have measured solver floors (used verbatim). The other 16
  are derived from the dominant-chain rel-err `l = log10(chain_re)`:

  ```
  loss(l) = 6.0 + 1.05·(l − 1)         for l > 1     (digits lost to cancellation)
  floor   = 15.95 − loss                              (double ≈ 15.95 digits)
  floor   = min(11.5, 10.0 + 0.4·(1−l))for l ≤ 1     (benign ceiling regime)
  ```

  Calibration on the 5 measured points (measured in parens): B12 `l=7.21`→3.43
  (3.69), B10 `l=1.29`→9.70 (9.88), B8 `l=−2.22`→11.3 (10.14), B9 `l=−3.37`→11.5
  (11.53), B1 (no chain)→ n/a (12.16). Agreement is within ~0.3 digits for
  B9/B10/B12 and ~1.2 for B8 — good on the *in-range* cases (§5). The model
  **saturates**: it floors the double result at 0 but reports the *uncapped*
  `loss` so dd-sufficiency (`31.9 − loss ≥ 10`?) is still visible.

* **Predicted dd floor** = `31.9 − loss` (dd's ~31.9 working digits minus the same
  cancellation loss) = `floor_double + 15.95` capped, and may go **negative** when
  `loss > 31.9` (dd insufficient).

**Data:** `/vast/projects/pepper_hep/qcdloop_shards_5k/` — 10 shards, 5000 samples
(500×10) per integral, written 2026-07-22. All 21 present. Raw numbers:
`bound_decomposition_all_21.json`.

---

## 2. Per-integral table (21 rows)

`domRE` = dominant-chain measured rel-err; `tight` = tightness; `loss` = derived
cancellation digit-loss; `dd→` = predicted whole-chain-dd floor; `dd✓` = dd
recovers to ≥10 digits. Floor **source**: `M`=measured solver run,
`D`=derived-in-range, `D!`=derived-**extrapolated** (low confidence),
`Dn`=derived-no-chain.

| integral | floor | src | verdict | domRE | tight | loss | dd→ | dd✓ | dominant chain (helper + combination) |
|----------|------:|:---:|---------|------:|------:|-----:|----:|:---:|----------------------------------------|
| **B12** | **3.69** | M | **COMPUTED** | 1.6e7 | 0.071 | 12.3 | 19.6 | ✅ | `ddilog`(212,702) → `B2m.h:206,207,241` |
| **B14** | 5.21 | D | **COMPUTED** | 3.3e5 | 0.199 | 10.7 | 21.2 | ✅ | `kfn`(1208) → `B2m.h:401,405,578` |
| **B13** | 8.62 | D | **COMPUTED** | 1.9e2 | 0.071 | 7.3 | 24.6 | ✅ | `ddilog`(212,702) → `B2m.h:300–355,533` |
| **B10** | 9.88 | M | **COMPUTED** | 2.0e1 | 0.003 | 6.1 | 25.8 | ✅ | `ddilog`(174–212,702–704) → `B1m.h:227,240,241` |
| **B16** | ~0 | D! | **COMPUTED** | 6.7e17 | 0.183 | 23.7 | 8.2 | ❌ | `cLi2omx2/omx3`(666,744…),`kfn`(1198,1208) → `B3m.h:177,183,230` |
| **BIN3**| ~0 | D! | **COMPUTED** | 1.9e23 | 0.045 | 29.4 | 2.5 | ❌ | `cspence/ltspence`(550,608),`li2series`(253) → `B3m.h:76,78,109` |
| **B15** | ~0 | D! | **COMPUTED** | 8.2e23 | 0.183 | 30.1 | 1.8 | ❌ | `cLi2omx2`(666,673),`kfn`(1198) → `B2m.h:492,496,578` |
| **BIN4**| ~0 | D! | **COMPUTED** | 5.4e25 | 0.038 | 32.0 | −0.1 | ❌ | `cspence/ltspence`(550,608),`li2series`(253) → `B4m.h:119,195,198,233` |
| **BIN0**| ~0 | D! | **COMPUTED** | 4.0e35 | 0.067 | 42.3 | −10.4 | ❌ | `cspence/ltspence`(550,608) → `B0m.h:68,88` |
| **BIN2**| ~0 | D! | **COMPUTED** | 8.5e38 | 0.067 | 45.8 | −13.9 | ❌ | `cspence/ltspence`(550,608),`li2series`(249,253) → `B2m.h:64,65,84` |
| **BIN1**| ~0 | D! | **COMPUTED** | 1.1e40 | 0.067 | 47.0 | −15.1 | ❌ | `cspence/ltspence`(550,608),`li2series`(253) → `B1m.h:62,63,79` |
| B9 | 11.53 | M | STABLE_ALREADY | 4.2e-4 | 0.003 | 4.4 | 27.5 | – | benign dilog cancellation |
| B8 | 10.14 | M | STABLE_ALREADY | 6.1e-3 | 0.003 | 5.8 | 26.1 | – | benign dilog cancellation |
| B2 | 11.5 | D | STABLE_ALREADY | 7.4e-5 | 0.11 | 4.5 | 27.5 | – | shallow chain |
| B4 | 11.5 | D | STABLE_ALREADY | 4.7e-6 | ~0 | 4.5 | 27.5 | – | shallow chain (tiny rel-err) |
| B3 | 10.32 | D | STABLE_ALREADY | 1.6e0 | 0.091 | 5.6 | 26.3 | – | shallow chain |
| B5 | 10.39 | D | STABLE_ALREADY | 1.1e0 | 0.091 | 5.6 | 26.3 | – | shallow chain |
| B1 | 12.16 | M | STABLE_ALREADY | – | – | 3.8 | 28.1 | – | **no cascade chains** |
| B6 | ~13.2 | Dn | STABLE_ALREADY | – | – | – | – | – | **no cascade chains** |
| B11 | ~10.1 | Dn | STABLE_ALREADY | – | – | – | – | – | **no cascade chains** |
| B7 | ~9.7 | Dn | STABLE_ALREADY | – | – | – | – | – | **no cascade chains** |

(Rows: COMPUTED first, ranked by predicted dd lift; then STABLE_ALREADY.)

---

## 3. Cross-integral chain-topology inventory (the Tier-B design input)

Every dominant floor-driving chain has the **same two-file shape**: a special
function in `src/kokkosUtils.h` feeding a near-equal add/sub combination in the
integral's `B{n}m.h` coefficient file. Chains are **bounded** — 3–10 distinct
source lines across exactly those two files; `n_contributors` ranges 4–33 but the
*line-set* stays small. **No chain sprawls across many files.** They cluster into
**three shared-helper families**:

### Family 1 — real dilog (`ddilog` + Li2 ratio wrappers)
`kokkosUtils.h:174,177,199,206,212` (`ddilog`, Chebyshev) + `702,703,704`
(`Li2omrat`/`Li2omx2` ratio wrappers).
* **Integrals:** B10, B12, B13 (combination in `B1m.h` / `B2m.h`). Also the
  *benign* B8/B9 hit the same dilog branches but at sub-floor amplitude.
* **dd:** sufficient (loss 6–12 → dd floor 20–26).
* **Shared lines:** `212`,`702` (dilog + wrapper) common to B10/B12/B13; `703`
  to B10/B12; `704` to B10/B13.

### Family 2 — complex Spence (`ltspence` / `cspence` / `li2series`)
`kokkosUtils.h:550` (`ltspence`) + `608` (`cspence`) + `249,253` (`li2series`).
* **Integrals:** **all five BIN0–BIN4** (combination in `B0m.h`…`B4m.h`).
* **dd:** **insufficient** (loss 29–47 → dd floor 2.5 down to −15).
* **Shared lines:** `550` **and** `608` common to **all five** BIN integrals;
  `253` to BIN1–4; `249` to BIN1/BIN2. This is the **tightest cross-integral
  coupling in the suite.**

### Family 3 — complex Li₂ of transformed argument (`cLi2omx2` / `cLi2omx3` / `kfn`)
`kokkosUtils.h:666` (`cLi2omx2`) + `744,746,752,754` (`cLi2omx3`) + `1198,1208`
(`kfn`) + `673`.
* **Integrals:** B14 (combination `B2m.h`), B15 (`B2m.h`), B16 (`B3m.h`).
* **dd:** mixed — B14 sufficient (loss 10.7), B15/B16 insufficient (loss 24–30).
* **Shared lines:** `1208` common to B14/B16; `666`,`1198` to B15/B16.

### Tier-B sub-questions, answered

1. **Same subgraph across integrals, or structurally different shapes?**
   **Three** structural shapes (Families 1–3), not one and not 21. Within a
   family the shape is *the same pattern* (special-func → same helper lines →
   `Bnm.h` subtraction); across families the helper is a different special
   function (real dilog vs complex Spence vs complex Li₂-of-transform).

2. **Bounded or sprawling chains?** Uniformly **bounded**: 3–10 lines over two
   files. The largest line-sets are B10/B13 (~8 helper lines of the dilog
   Chebyshev branch set) and B16 (7 helper lines across `cLi2omx2/omx3/kfn`).
   None crosses a third file. This matches B12's ~14-line/2-file chain and means
   a Tier-B shim per integral is a *small coordinated multi-file patch*, not a
   call-graph-wide rewrite.

3. **Would Tier-B have to promote a shared helper that then affects other
   integrals?** **Yes — this is the dominant coupling and the central 2f
   question.** Promoting `kokkosUtils.h:550/608` (`ltspence`/`cspence`) to dd
   touches **BIN0, BIN1, BIN2, BIN3, BIN4 simultaneously**; promoting
   `212/702` (`ddilog`+wrapper) touches **B10, B12, B13** (and perturbs the
   benign B8/B9 that share the branch). A helper promotion is therefore **not
   integral-local**: Tier-B/2f must either (a) promote the shared helper once and
   accept it lands on every sharing integral, or (b) template/clone the helper
   per call-path so promotions stay integral-scoped. The measured benign
   integrals (B8/B9) that share Family-1 lines are the reason (a) is not free —
   a dd promotion of the shared dilog would re-measure B8/B9 too (expected inert,
   but must be verified, not assumed).

---

## 4. Prioritized Tier-B target list

Ranked by predicted dd floor lift, split by whether **dd alone** clears the
≥10-digit bar. **Tier-B (dd fan-out) should scope to Group A**; Group B is
flagged as *beyond-dd* — dd narrows but does not close the floor, so committing
Tier-B to them would under-deliver.

### Group A — dd-sufficient (Tier-B dd fan-out fully recovers) — **4 integrals**

| rank | integral | floor | dd floor | lift | helper to widen | confidence |
|:----:|----------|------:|---------:|-----:|-----------------|------------|
| 1 | **B12** | 3.69 | 19.6 | **+15.9** | `ddilog`+wrapper `212,702` → `B2m.h:206,207,241` | **measured** |
| 2 | **B14** | 5.21 | 21.2 | **+16.0** | `kfn` `1208` → `B2m.h:401,405,578` | derived-in-range |
| 3 | **B13** | 8.62 | 24.6 | **+16.0** | `ddilog`+wrapper `212,702` → `B2m.h:300–355,533` | derived-in-range |
| 4 | **B10** | 9.88 | 25.8 | **+16.0** | `ddilog` branch `174–212,702–704` → `B1m.h:227,240,241` | **measured** (¹) |

¹ **B10 caveat:** its *single-region* dd was **measured INERT** (`physics_ceiling`,
DD_TRIAGE Item 5) — widening one link cannot move it. The tight chain bound says
*whole-chain* dd would (Item 6's single-region-inert ⇔ whole-chain-needed
reconciliation), but that is **untested**. B10's floor (9.88) is also already
close to the benign bar, so its lift *priority* is low even though the mechanism
qualifies.

### Group B — dd-insufficient / beyond-dd (loss > dd budget) — **7 integrals**

| integral | floor | dd floor | loss | helper family | note |
|----------|------:|---------:|-----:|---------------|------|
| **B16** | ~0 | 8.2 | 23.7 | Family 3 (`cLi2omx2/omx3`,`kfn`) | dd → ~8 digits (partial); still < 10 |
| **BIN3**| ~0 | 2.5 | 29.4 | Family 2 (`cspence/ltspence`) | dd barely helps |
| **B15** | ~0 | 1.8 | 30.1 | Family 3 (`cLi2omx2`,`kfn`) | dd barely helps |
| **BIN4**| ~0 | −0.1 | 32.0 | Family 2 | dd exhausted |
| **BIN0**| ~0 | −10.4 | 42.3 | Family 2 | **quad/rewrite territory** |
| **BIN2**| ~0 | −13.9 | 45.8 | Family 2 | **quad/rewrite territory** |
| **BIN1**| ~0 | −15.1 | 47.0 | Family 2 | **quad/rewrite territory** |

**Count:** Tier-B (dd) would move the floor on **11 integrals** total, but only
**fully clear the ≥10-digit bar on 4** (Group A). The remaining 7 (Group B) need
beyond-dd; **all Group-B floors are extrapolated (low-confidence)** and should be
**measured before any beyond-dd decision** (§7).

---

## 5. Shard-derived vs measured floor — sanity check

The derivation model reproduces the 5 measured floors well **inside its
calibration range**; there is **no large disagreement** among measured integrals:

| integral | measured | model-from-chain | Δ | note |
|----------|---------:|-----------------:|--:|------|
| B12 | 3.69 | 3.43 | −0.26 | in range (`l=7.21`) |
| B10 | 9.88 | 9.70 | −0.18 | in range (`l=1.29`) |
| B9  | 11.53 | 11.5 | −0.03 | benign regime |
| B8  | 10.14 | 11.3 | +1.16 | benign regime (model optimistic ~1 digit) |
| B1  | 12.16 | n/a | – | no chains |

**The one conceptual disagreement worth flagging to Reet.** Prior work treats
**B12 as the whole-app global-min hotspot (3.69 digits)** — but that ranking was
established over the *measured* set (≈B1–B12). The shard indicators show
**BIN0–4, B15, B16 have dominant-chain amplification 10–33 orders larger than
B12's** (chain rel-err up to 1.1e40 vs B12's 1.6e7). *If* those chains reach
scored coefficients, several of these unmeasured integrals would be **worse
hotspots than B12**, and "B12 is the global min" would be false for the full
21-integral suite.

**Counter-caution (do not over-read this).** A large chain rel-err does **not**
guarantee a catastrophic whole-app floor: **B10's** dominant chain blows up to
19.6 (≫1) yet its measured floor is a benign 9.88 (`physics_ceiling`, dd-inert) —
the cancellation victim's contribution to the scored coefficient is attenuated.
So the extreme BIN/B15/B16 numbers establish **severe cancellation is present and
warrants measurement**, *not* that their floors are confirmed catastrophic. The
honest statement: **the global-min assumption is untested for the 9 never-measured
integrals, and the shard data is consistent with it being wrong.**

---

## 6. What this means for Tier-B (decomposition facts only — no design)

* The lever exists and is **bounded**: 4 integrals (B12/B13/B14/B10) are clean dd
  wins over 2-file, ≤10-line chains in **two** helper families (real-dilog,
  `kfn`).
* The **shared-helper coupling is real and concentrated**: `ltspence`/`cspence`
  (all BIN) and `ddilog` (B10/B12/B13). Any dd promotion of a shared helper is a
  multi-integral event by construction.
* **dd is not universally sufficient**: the deepest-cancellation family
  (complex-Spence, all BIN) loses 29–47 digits — past dd's ~32-digit budget —
  so a dd-only Tier-B **cannot** close BIN0–4/B15. Scoping Tier-B to Group A
  avoids overfitting the mechanism to floors it cannot reach.

*(Per the task, Tier-B design is deliberately out of scope here. The above are
decomposition observations, not a mechanism proposal.)*

---

## 7. Caveats / honest bounds

* **16 of 21 floors are derived, not measured.** B13/B14 are in calibration
  range; **B15/B16/BIN0–4 are extrapolated 2–5× beyond the calibration edge**
  (`l` up to 40 vs calibrated max 7.2). Their **digit floors and dd-sufficiency
  signs are low-confidence**; the *mechanism* verdict (tight bound → COMPUTED) is
  robust because tightness is a scale-invariant ratio, but "dd insufficient by
  −15 digits" for BIN1 is a model extrapolation, not a measurement.
* **`rel_err` normalization** blows past 1 near cancellation zero-crossings (an
  amplification indicator, not bounded accuracy); tightness (a ratio) is the
  sound scale-invariant comparison, and the floor derivation is deliberately
  coarse.
* **Reaches-a-scored-output is assumed, not proven.** `amp` is computed to the
  per-sample DAG sinks; whether a given chain's sink is a *scored* `res(i,*)`
  coefficient is not verifiable from shards alone (the whole-app journal is
  empty). B10 shows a chain can amplify hugely yet land benignly on the scored
  output. This is the single largest reason to **measure** Group-A/B floors
  before committing Tier-B scope.
* **5k samples** is fine for chain-topology and dominant-sensitivity
  identification (this task); it is not a rare-event floor measurement. Per the
  prompt, no re-characterization at higher sample count is proposed — but a
  **measured solver floor** (double-vs-dd, the existing scorer) on the 9
  never-measured integrals is the obvious cheap follow-up and is *not*
  re-characterization.

---

## 8. Reproducibility

```bash
.venv/bin/python runs/qcdloop/bound_decomposition_all_21.py
# reads /vast/projects/pepper_hep/qcdloop_shards_5k/shard_*.json (~50s load + merge),
# caches the merged report to runs/qcdloop/.merged_all_21.cache.json,
# reuses stability_reducer.merge_reports + finalize_report,
# writes runs/qcdloop/bound_decomposition_all_21.json + console tables.
```

Measured floors from the Phase-2b per-integral scorer manifests
(`per_integral_out_stage2/{B8,B9,B10}`, `per_integral_out_2e_measure/{B1,B12}`,
max `baseline_delta_effective` → digits) and, for B12, the whole-app min
(`SOLVER_STAGE1_DD_PROBE.md`). Special-function names read from
`src/kokkosUtils.h`. INERT/`physics_ceiling` context from
`DD_TRIAGE_2026-07-25.md`. Framework from `B12_BOUND_DECOMPOSITION_2026-07-25.md`.

## Constraints honored

Read-only. No solver/fan-out/gate/manifest changes. No Stage 2 launched. No LLM,
no builds, no runs. `merge_reports`/`finalize_report` reused verbatim;
`b12_bound_decomposition.py` used as the arithmetic template. No Tier-B design
proposed. Malformed/missing data: none — all 21 integrals present with
well-formed chains; extrapolated floors flagged rather than fabricated.
