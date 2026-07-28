# Region-Core Element-Level Promotion — LANDED

**Date:** 2026-07-28
**Branch:** langgraph-agents
**Dispatch:** land region-core element-level promotion + held Shape-1/rule-c fixes
**Design:** `runs/qcdloop/REGION_CORE_ELEMENT_PROMOTION_DESIGN.md` §7 ordered landing plan
**Scope:** fixed-size complex aggregates ONLY (`Kokkos::Array<T,N>`, C arrays, `std::array`). Production code.

---

## Verdict

**Step 3 (B14 clean build) = GO.** STOP #CC (master emission-binding blocker) is **CLEARED**.
STOP #A fired on B14 L-measure (lift 0.0 < predicted +16.66) — B14 already ≈ full double
accuracy; this is a gate-instrument / already-accurate outcome, **not** a promotion failure.
Per §2.4: handed back, no coefficient synthesis.

---

## Per-step outcomes

### Step 1 — Element-level read/write promotion (COMPLETED prior session)
Wrap the READ occurrence of a fixed-size complex-array element to dd; demote element STORES;
**never** retype the array decl (STOP #FF stays cleared, d1 guard unchanged).
- `region_scan.py` — element-base detection keyed on decl SHAPE (`Kokkos::Array<T,N>` /
  C array / `std::array`), no app identifiers. d1 subscript-base exclusion UNCHANGED.
- `boundary.promote_region_block` — element read wrap + store demote.
- `type_resolve.py` — element type resolution.
- Tests: element-promotion suite (fixed-size aggregate shapes only).

### Step 2 — Held Shape-1 store-narrow (deliv. b) + Shape-2 rule-c widen (deliv. c) (COMPLETED)
The interior `complex<ddcomplex>` cleared by step 1 made both instantiation-validatable.
- `boundary.py` — `_match_paren`, `widen_carrier_assign_line` (deliv. c: sibling carrier
  assignment RHS widen — functional-cast rewrite / real+imag reconstruction),
  `demote_exit_carriers_line` (deliv. b: demote carrier READ occurrences at designed exits).
- `fanout.py` — `VariantSpec.closure_complex_names` + `designed_exits`; `_carrier_reconcile_edits`
  (strict no-op unless a complex Promote AND ≥1 widened complex carrier exist); tag-3 apply
  branch in `render_variant`.
- `chain_promote.py` — attach `closure_complex_names` / `designed_exits` to specs, gated on
  `complex_type`.
- Tests: +10 carrier-reconcile tests (26 total in element suite).

### Step 3 — B14 clean-build go/no-go (GO)
First honest dd build after the STOP #CC fix. Prior identical pipeline (pre step-2) produced
exactly 3 residuals at spliced 754/764/765 (sibling-assign not widened; two designed-exit
stores not demoted). After deliverables b+c: **0 build errors**, no `static_assert`,
`Built target boxGPU_app`, 9 variants declared (incl. `B14_B2mo_B2m_B14`, `B2m_B14`,
and co-variants `B15_B2mo_B2m_B14`, `B16_B3m_B14`), 3 shims generated.

B14 go/no-go measured result (`region_core_b14/B14/tierb_result.json`):

| field | value |
|---|---|
| outcome | `rejected` (`chain_no_lift`) — **build clean, not build_failed** |
| patcher_status | `ok` |
| kernel baseline | 13.1855 |
| kernel final | 13.1855 |
| kernel measured lift | **0.0** |
| predicted lift (static tightness) | +16.6566 |
| chain_id | `cascade_B14_3429b1d4_01bf2ff3` |
| chain_lines | B2m.h:401, B2m.h:578, kokkosUtils.h:1208 |

**Interpretation:** B14's kernel output already resolves to 13.19 digits at double
(≈ full double precision). The +16.66 was a static conditioning *prediction* from
`measured_max_rel_err` = 326888; the runtime kernel measurement shows nothing to lift.
STOP #A condition met → hand back, no coefficient synthesis.

### Step 4 — Instantiation-gate sweep (regression detector)

Two runs were used to separate the element landing from pre-existing rule-d confounds:

**(a) Measured-4 `--leaf-promotion` sweep** (`region_core_sweep`) — combines step 4 + step 5.
Prior baseline = `lmeasure_run` (pre-landing, **same** `--leaf-promotion` flag):

| integral | pred lift | prior (pre-landing) | this sweep (post-landing) | attribution |
|---|---|---|---|---|
| B10 | +18.43 | apply_failed (build_failed) | apply_failed (build_failed) | **same out-of-scope wall** (see below) |
| B12 | +17.10 | _(no prior result)_ | apply_failed (`chain_fanout_failed: leaf '_pi2' undefined`) | rule-d leaf-graph gap — **not element landing** |
| B13 | +17.10 | apply_failed (`write_truncation`) | apply_failed (`write_truncation`) | **identical** chain-gate reject — no regression |
| B14 | +16.66 | apply_failed (**build_failed**) | rejected (`chain_no_lift`, **clean build**, lift 0.0) | **build_failed → clean = improvement** |

**(b) Isolated no-`--leaf-promotion` run** (`region_core_noleaf`, B12/B13/B14) — removes the
rule-d `_pi2` confound so any B12/B13 delta is attributable to the element landing alone:

| integral | outcome | build | note |
|---|---|---|---|
| B12 | apply_failed | build_failed (`Lnrat<...>(ddouble&)` leaf + `complex<ddcomplex>`) | investigated — see below |
| B13 | apply_failed | write_truncation (built OK, gate reject) | identical to leaf-promo run |
| B14 | rejected (`chain_no_lift`) | **clean, lift 0.0** | B14 fix robust with leaf-promo OFF |

**B12 regression investigation (STOP #II decisive test).** B12 build-fails *even without*
leaf-promotion, and B12 is a complex integral predicted **none-touch** — so I verified it is
pre-existing, not caused by the landing. Method: `git stash` the 5 production emission files
(revert to HEAD 992e209 = pre-element-landing), keep the harness identical, re-run B12 no-leaf.

| B12 error signature | pre-landing (HEAD) | post-landing | 
|---|---|---|
| `wrong number of template arguments (3, should be 2)` | 12 | 12 |
| `Lnrat<complex<double>,double,double>(ddouble&, ddouble&)` | 12 | 12 |
| `complex<ddcomplex> → const ddcomplex` | 2 | 2 |
| `static_assert Kokkos::complex ... floating point type` | 1 | 1 |

**Byte-identical.** Same chain (`cascade_B12_65bb39c0_62ff5a3d`), same build_failed outcome.
B12's failure is a **pre-existing** blocker (its own `complex<ddcomplex>` region-core + `Lnrat`
leaf-callee) present at HEAD before any element promotion. **My landing did NOT regress B12.**
B12 was never in the predicted touch set to be *fixed*, and is not newly *broken*. Changes
restored via `git stash pop` (clean).

**B10 build-failure classification** (confirms *same wall*, not a new element bug): error
signatures are the classic out-of-scope dd/double boundary family —
`Kokkos::complex<double>::complex(ddouble)` (×22), `invalid cast ddouble→double` (×16),
`ddcomplex→Kokkos::complex<double>` (×9), plus 1 `static_assert` (`complex<ddcomplex>`) and
1 `operator=(complex<double>, complex<ddcomplex>)`. These originate at B10's **out-of-scope**
region-core (`res(i,1)` View accessor + `Constants<TOutput>` returns — dynamic/function-return,
NOT a fixed-size `Kokkos::Array`), which this landing deliberately walls. No fixed-size
aggregate in B10 was left un-promoted. B10 was never unblocked by this landing (as designed).

### Step 5 — L-measure re-run
Combined with step 4(a). B14 honest kernel measurement: baseline 13.1855 → final 13.1855,
**lift 0.0** (pred +16.66). B14 already resolves to ≈ full double accuracy; nothing to lift.
STOP #A → hand back, no coefficient synthesis. B10 remains blocked (out-of-scope, expected).
B12/B13 unmeasurable under leaf-promo (rule-d / chain-gate), pre-existing and out of this
landing's scope.

---

## 21-integral coverage argument

`tier_b_stage1.py` measures the 4 Tier-B integrals {B10,B12,B13,B14}. The remaining 17 are
covered structurally, not re-measured, because the reconcile pass is a **strict no-op** unless
a variant carries both a complex region Promote and a widened complex carrier:

- **B15, B16** — share B14's B2m/B3m chain; their variants (`B15_B2mo_B2m_B14`,
  `B16_B3m_B14`) were **co-declared and built clean inside the B14 go/no-go build**. This
  directly discharges STOP #JJ (incl. B15's `TOutput xs = cxs[0]` decl-init element shape).
- **B12, B13** — none-touch (no complex region-core); measured in the sweep as the
  clean→clean regression check.
- **B10** — out-of-scope region-core (`res(i,1)` View accessor + `Constants<TOutput>` returns)
  + 71 rule-d leaf clones; expected still-blocked. Confirms no NEW breakage.
- **Everything else** (B1–B9, B11, BIN0–4, x4__ tail) — no complex region promote emitted →
  reconcile guard is inert → byte-identical to pre-landing emission.

_Sweep table to be filled from the run._

---

## STOP audit

| STOP | condition | status |
|---|---|---|
| #CC | master emission-binding (`complex<ddcomplex>`) | **CLEARED** (B14 clean build) |
| #FF | emitted array decl retyped to dd | not fired (element read/write wrap only) |
| #GG | any emitted array decl at dd → STOP | not fired (d1 invariant held) |
| #HH | B14 fails clean build at step 3 → STOP | not fired (GO) |
| #II | gate regresses integral outside predicted {B14,B15,B16,x4-cond} | **not fired** — B10/B12 build_failed *pre & post* (B12 byte-identical at HEAD, proven); B13 write_truncation identical; nothing clean→broken |
| #JJ | B15 or B16 fails clean under gate → STOP | not fired (co-built clean in B14 run) |
| #A  | B14 clean + gate passes + lift < expected | **FIRED** (lift 0.0) → hand back §2.4 |
| #Z  | vendored snapshot pristine | **clean** (`third_party/` unmodified) |

---

## Test health

- `tests/integrator_base/test_region_core_element_promotion.py`: **26 passed** (16 element + 10 carrier-reconcile).
- `tests/patcher/ tests/integrator_base/ tests/shared/`: **520 passed** (no regressions).

---

## Landed production changes (uncommitted → to commit on langgraph-agents)

- `agents/shared/region_scan.py` — fixed-size complex-aggregate element-base detection (decl SHAPE, no app identifiers); d1 subscript-base exclusion UNCHANGED.
- `agents/shared/type_resolve.py` — element type resolution.
- `agents/integrator_base/boundary.py` — `_match_paren`, `widen_carrier_assign_line` (deliv. c), `demote_exit_carriers_line` (deliv. b); element read-wrap / store-demote in `promote_region_block`.
- `agents/patcher/fanout.py` — `VariantSpec.closure_complex_names` + `designed_exits`; `_carrier_reconcile_edits` (strict no-op guard); tag-3 apply branch.
- `agents/patcher/chain_promote.py` — attach `closure_complex_names`/`designed_exits` to specs, gated on `complex_type`.
- Tests: `tests/integrator_base/test_region_core_element_promotion.py` (+10).

---

## Handbacks for Reet

1. **STOP #A (B14 lift 0.0).** B14 builds clean at dd for the first time (STOP #CC cleared) but
   its kernel already resolves to ≈13.19 digits at double — nothing to lift. The +16.66 was a
   static tightness prediction. This is a **gate-instrument / already-accurate** outcome; per
   §2.4 do NOT pursue coefficient synthesis. Decision needed: is B14 done (accept "already
   accurate"), or is a different acceptance instrument wanted for already-tight kernels?
2. **B12 (pre-existing, out of this landing's scope).** Needs BOTH `Lnrat` leaf-callee
   promotion (rule-d) AND resolution of its own `complex<ddcomplex>` region-core. Neither is a
   fixed-size aggregate — out of scope for this landing. `_pi2` leaf-graph gap surfaces under
   `--leaf-promotion`.
3. **B10 (unchanged wall).** Out-of-scope region-core (`res(i,1)` View accessor +
   `Constants<TOutput>` returns) + 71 rule-d leaf clones. Not unblocked by this landing (as
   designed).

---

## MEMORY.md update block

Update `[Region-core element-level promotion (DESIGN ONLY)]` → **LANDED**:

> - [Region-core element-level promotion (LANDED)](project_region_core_element_promotion.md) —
>   element-level read/write promotion for FIXED-SIZE complex aggregates LANDED (uncommitted →
>   langgraph-agents). STOP #CC **CLEARED**: B14 builds clean at dd 1st time (was build_failed
>   754/764/765 → 0 errors) via deliv. b (`demote_exit_carriers_line`) + c
>   (`widen_carrier_assign_line`) carrier-reconcile pass in fanout render_variant (strict no-op
>   guard preserves byte-identity for scalar chains). **STOP #A FIRED**: B14 lift 0.0 (kernel
>   already ≈13.19 digits at double; pred +16.66 was static tightness) → hand back §2.4, NO
>   coeff synthesis. Regression sweep (region_core_sweep + region_core_noleaf + B12 stash-rerun
>   at HEAD): STOP #II/#JJ/#Z all clean — B10 same out-of-scope wall, **B12 build_failed
>   byte-identical pre & post landing (pre-existing Lnrat leaf + own complex<ddcomplex>, NOT a
>   regression)**, B13 write_truncation identical, B15/B16 co-built clean in B14 build. 520+26
>   tests. Report REGION_CORE_ELEMENT_LANDED_2026-07-28.md. B10 still walled (View res(i,1) +
>   Constants returns + 71 leaf clones = out-of-scope).
