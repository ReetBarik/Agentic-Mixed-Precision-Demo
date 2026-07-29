# Phase-2 Float Downshift — LANDED (pipeline-authored shim synthesis) — 2026-07-29

Implementation dispatch following the shim-synthesis scoping report
(`PHASE_2_SHIM_SYNTHESIS_DESIGN_2026-07-29.md`). This is the **float-only** landing: the
mandatory ff-container feasibility probe (deliverable 1) fired **STOP #EEE**, so Phase-2
lands double→float only; ff stays enrichment-required and is handed back.

## 0. Executive verdict

| item | outcome |
|---|---|
| **STOP #EEE** (ff container) | **FIRED** — `Kokkos::complex<quad::ffun::ffloat>` fails Kokkos_Complex.hpp:35 `static_assert(is_floating_point_v<RealType>)` (ffloat is a custom struct). Scope reduced to **float-only**; FF profile `available=False`, signature handed back. |
| Shim generator (`agents/patcher/shim_synth.py`) | **LANDED** — structural leaf-inventory extractor + precision-parameterized sibling renderer + sha256 invalidation. Extracts the **13** non-template `ql::` leaves from `kokkosMaths.h` (no baked-in leaf list). |
| FLOAT profile | **ENABLED** — `shim_synthesis=True, maths_reference_header="kokkosMaths.h", available=True`. FF stays `available=False`. |
| `to_d` float arm | **LANDED** — `runs/qcdloop/src/boxGPU_app_recipes.hpp:193` (1-line `is_same_v<...,float>` arm; two-limb `.hi` preserved for dd/ff). |
| Downshift routing (`route_downshift`) | **LANDED** — raw-double + parametric → first available of `(FLOAT, FF)`; dd candidates never downshifted (STOP #ZZ). |
| Acceptance gate `lift_direction` | **LANDED** — `LiftDirection.DOWNSHIFT`: accept iff `lift >= -margin` (precision-preserving); `UPSHIFT` unchanged (`lift > margin`). |
| **L-measure (B1–B9, B11)** | **RAN** — all 10 build clean at float (instantiation gate 10/10, no STOP #BBB/#CCC/#DDD), run, genuine float compute; **all 10 REJECT** on precision loss (lift −7.6 … −9.8). **0/10 accepted.** |
| Regression | **CLEAN** — 964 pass / 1 flaky live-LLM test (passes on re-run, untouched by Phase-2); 495 deterministic patcher+integrator_base green; acc1482 7/7; DD render byte-identical; snapshot + third_party pristine. |

**Bottom line:** the float downshift **mechanism** is fully landed, buildable, and honest —
every one of the 10 raw-double integrals compiles + runs at float via zero-enrichment shim
synthesis. But float's ~7-digit budget **cannot preserve** the accuracy these box integrals
need (baseline 9–13 digits vs the dd reference), so the precision-preserving downshift gate
**correctly rejects all 10** back to raw double. This is exactly the scoping report's §6
prediction caveat: the mechanism is viable for all 10; *which accept* is an L-measure
question, and the answer is **none** — float is too narrow for these integrals.

**Final 21-integral precision assignment (unchanged from Phase-1):** 11 dd candidates stay
dd; B1–B9, B11 stay **raw double** (float rejected). No integral moves to float.

---

## 1. STOP #EEE — the ff-container probe (deliverable 1, done FIRST)

Per the dispatch's load-bearing ordering constraint, the empirical ff feasibility probe ran
before any code landed. A `/tmp` clone compiled a TU instantiating
`Kokkos::complex<quad::ffun::ffloat>`:

```
Kokkos_Complex.hpp:35:22: error: static assertion failed: Kokkos::complex can only be
  instantiated for a cv-unqualified floating point type
   35 |   static_assert(std::is_floating_point_v<RealType> && ...
note: 'std::is_floating_point_v<quad::ffun::ffloat>' evaluates to false
```

Definitive: `ffloat` is a custom two-limb struct (`float hi, lo`), not a built-in FP type,
so the `Kokkos::complex` container rejects it at compile time. There is **no** container-level
workaround. Combined with the scoping report's §8 finding (no library-native
`Kokkos::abs<ffcomplex>` to bind a shim to), **ff is fundamentally unavailable to path-(b)
shim synthesis**. Per the dispatch verdict gate, scope reduced to float-only.

**Hand-back for a future ff-container investigation:** ff downshift requires BOTH (a) a
static `kokkosMaths_ff.h` wrapper layering `ql::` leaves on the `quad::ffun` primitives
(the original STOP #XX enrichment ask), AND (b) an ff-native complex container
(`quad::ffun::ffcomplex`, NOT `Kokkos::complex<ffloat>`). The FF `PrecisionProfile` already
carries `cpp_output="ql::ffun::ffcomplex"` for when that work is authorized; it stays
`available=False` until both prerequisites exist.

---

## 2. What landed (deliverables 2–6)

### 2.1 `agents/patcher/shim_synth.py` (deliverable 2) — NEW

The pipeline-authored leaf-shim generator. **Structural, precision-parameterized, sha-keyed.**

- **`extract_inventory(reference_header_text, *, reference_scalar, namespace="ql")`** — parses
  the namespace-scoped, **non-template** function definitions whose head names the reference
  scalar token (brace-matched namespace isolation, comment-stripped, `if constexpr`-safe
  classification). Returns the **13** §1.1 leaves in **source order** (so `kAbs` precedes its
  callers `Sign`/`Max`/`Min`/`Htheta`). Excludes the generic templates
  (`kAbs<T>`/`kLog`/`kSqrt`/`kConj`/`kPow`/`iszero`), `Constants<T>` struct members, and the
  `using complex` alias — all of which auto-instantiate at float. **No baked-in leaf list**
  (feedback_no_placeholder_patterns).
- **`render_shim(...)`** — emits one target-precision sibling per leaf by a whole-word
  `reference_scalar → target_scalar` token rewrite over each definition (rewrites the
  signature *and* the body in one pass: `Kokkos::complex<double>`→`Kokkos::complex<float>`,
  `double(0)`→`float(0)`; leaves `Kokkos::abs`, `x.real()`, `int` return types untouched).
  Output signatures are strictly target-typed → ODR-safe beside the double reference
  (STOP #DDD). **Never branches on a precision name** (STOP #SS): a synthetic `myhalf`
  precision selects the same path with its own tokens (unit-tested).
- **`inventory_sha256` / `read_embedded_sha`** — the shim's first line carries
  `// @shim-inventory-sha256: <hex> reference=kokkosMaths.h precision=float`. The sha is
  order-independent (sorted internally) over `(name, source)` pairs, so a leaf signature or
  body change invalidates a cached shim while a pure reorder does not (§3.4).

### 2.2 `agents/patcher/tu_emit.py` (deliverable 3)

- `PrecisionProfile` gains `shim_synthesis`, `maths_reference_header`, `reference_scalar`
  (defaults keep dd/quad profiles byte-identical).
- FLOAT profile → `shim_synthesis=True, maths_reference_header="kokkosMaths.h",
  reference_scalar="double", available=True`. FF stays `available=False` (STOP #EEE, commented).
- `render_wrapper` gains a shim-synthesis branch: a two-line wrapper (`#include
  "kokkosMaths.h"` then `#include "kokkosMaths_float_shim.hpp"`) instead of the `#define`
  ladder. The static-header ladder (dd/quad/double) is unchanged.
- `emit_flip_tu` gains `_emit_shim`: for a shim-synthesis profile it reads the clone's
  reference header, renders the shim, and writes `kokkosMaths_<precision>_shim.hpp` into the
  clone (reusing `_refuse_snapshot` — STOP #Z), sha-keyed to skip a no-op rewrite. Returns
  `FlipTU.shim_path`. `render_group_driver` is unchanged.

### 2.3 `runs/qcdloop/src/boxGPU_app_recipes.hpp:193` (deliverable 4)

`to_d` gains the middle `float` arm (`static_cast<double>(v)`); the two-limb `.hi` arm now
covers dd/ff only. Pipeline-owned app source (not the snapshot); the one file legitimately
modified by this landing.

### 2.4 `agents/patcher/precision_flip.py` (deliverable 5)

`route_downshift(integral, *, dd_candidate, graph, target_frames, available_targets,
preference=DOWNSHIFT_PREFERENCE)`. A raw-double, parametric integral routes to the first
target in `(FLOAT, FF)` that is in `available_targets` (passed in from the profile table so
this module never imports it and never hard-codes which precisions are live). A dd candidate
routes to raw double (STOP #ZZ); a non-parametric subtree or an empty available set stays
double.

### 2.5 `agents/patcher/flip_gate.py` (deliverable 6)

`LiftDirection` enum + `direction` param on `evaluate`/`evaluate_all`. `UPSHIFT` (default,
Phase-1) keeps `lift > margin`; `DOWNSHIFT` (Phase-2) is `lift >= -margin` — a precision-
preserving float (lift 0.0) is accepted, a genuine loss is rejected. Same predicate shape,
direction-selected threshold; no per-integral special cases.

---

## 3. L-measure results (deliverable 7)

`runs/qcdloop/phase2_lmeasure.py` — clone snapshot (STOP #Z), build vanilla baseline, build
dd oracle from `ddfun_enabled` via git archive (reference only), build per-group **float**
flip TUs from the clone alone, measure per-integral min precise-digits (baseline & candidate
both vs dd-ref over 2000 samples), apply the downshift gate.
`runs/qcdloop/phase2_lmeasure_out/phase2_lmeasure.json`:

| integral | group | built | base digits | float digits | lift | verdict |
|---|---|---|---|---|---|---|
| B1  | B0m | ✅ | 12.49 | 2.66 | −9.83 | reject |
| B2  | B0m | ✅ | 12.92 | 4.06 | −8.87 | reject |
| B3  | B0m | ✅ | 11.99 | 3.53 | −8.45 | reject |
| B4  | B0m | ✅ | 11.37 | 3.79 | −7.58 | reject |
| B5  | B0m | ✅ | 12.71 | 4.23 | −8.48 | reject |
| B6  | B1m | ✅ | 12.27 | 3.33 | −8.94 | reject |
| B7  | B1m | ✅ | 11.62 | 3.62 | −8.01 | reject |
| B8  | B1m | ✅ | 10.77 | 2.64 | −8.13 | reject |
| B9  | B1m | ✅ | 11.67 | 2.67 | −9.01 | reject |
| B11 | B2m | ✅ | 9.46  | 0.48 | −8.98 | reject |

`flip_build_failed: []` — **the instantiation gate passed for every group** (B0m/B1m/B2m all
compile clean at float via the synthesized shim). **0/10 accepted.**

**Interpretation.** The rejections are *not* a mechanism failure — every integral builds,
runs, and does genuine float compute (verified: RES mantissa tails are float-truncated). They
reject because IEEE float carries ~7 decimal digits and these box integrals need 9–13 (vs the
dd reference), so float loses 7.6–9.8 digits. The downshift gate's job is to catch exactly
this and keep the integral at raw double, which it did — uniformly, with no special-casing.
Float downshift is **viable but not accepted** for the qcdloop box family: the workload is
too ill-conditioned for single precision. A future half/bfloat precision would reject the
same way; a precision *between* float and double with library-native support could be routed
by the same machinery (set `shim_synthesis=True` + its tokens) with no code change.

---

## 4. Regression preservation (deliverable 8)

- **Full suite:** 964 passed, 1 failed → the single failure is
  `tests/dd_integrator/test_regional.py::test_real_llm_ieps50_derived_not_r4`, a
  `@pytest.mark.llm` **live-LLM** test that misgenerated `complex<ddouble>` (caught correctly
  by the anti-pattern guard). **Re-ran in isolation → passed.** Nondeterministic live-LLM
  flake, in a path Phase-2 does not touch (`git status` shows no changes under
  `dd_integrator`/`integrator_base`).
- **Deterministic patcher + integrator_base:** 495 passed (`-m "not llm"`).
- **New tests:** `test_shim_synth.py` (11), plus additions to `test_tu_emit.py`,
  `test_precision_flip.py`, `test_flip_gate.py` — 65 patcher tests green.
- **acc1482 boundary:** `test_flip_boundary.py` 7/7.
- **DD Phase-1 render byte-identical:** dd wrapper + driver contain no shim leakage, no new
  `#define` (asserted directly). All 11 Phase-1 dd accepts are unaffected (STOP #ZZ).
- **Snapshot + third_party pristine** (`git status --porcelain runs/qcdloop_headers_full
  third_party` empty before and after the run). Only `boxGPU_app_recipes.hpp` (deliverable 4)
  is modified.

---

## 5. STOP audit

| STOP | definition | state |
|---|---|---|
| **#EEE** | `Kokkos::complex<ffloat>` won't compile | **FIRED** → float-only landing; ff handed back |
| **#SS** | shim generator branches on a precision name | **not fired** — token-parameterized; `myhalf` selects the same path (test) |
| **#BBB** | a non-template leaf needs hand-authored math | **not fired** — all 13 leaves library-pass-through / member / scalar-ternary |
| **#CCC** | float leaf fails to compile (no library-native instantiation) | **not fired** — all 10 float TUs build clean |
| **#DDD** | shim ODR-collides / shadows the double overloads | **not fired** — strictly float-typed signatures; double TU + shim coexist |
| **#ZZ** | Phase-2 regresses a Phase-1 dd accept | **not fired** — dd candidates never downshifted; dd render byte-identical |
| **#Z** | snapshot pristine | **clean** — all generation into clones; snapshot + third_party untouched |
| **#A** | a lift is measured against dead code | **not fired** — genuine float compute confirmed (float-truncated RES) |

---

## 6. Artifacts

- Production: `agents/patcher/shim_synth.py` (new), `agents/patcher/tu_emit.py`,
  `agents/patcher/precision_flip.py`, `agents/patcher/flip_gate.py`,
  `runs/qcdloop/src/boxGPU_app_recipes.hpp`.
- Harness: `runs/qcdloop/phase2_lmeasure.py`; results
  `runs/qcdloop/phase2_lmeasure_out/phase2_lmeasure.json` + `phase2_lmeasure.log`.
- Tests: `tests/patcher/test_shim_synth.py` (new) + additions to `test_tu_emit.py`,
  `test_precision_flip.py`, `test_flip_gate.py`.
- Reports: this file; prior `PHASE_2_SHIM_SYNTHESIS_DESIGN_2026-07-29.md`,
  `PHASE_2_STOP_XX_2026-07-29.md`.

---

## 7. Hand-backs for Reet

1. **ff downshift** stays enrichment-required (STOP #EEE): needs a static `kokkosMaths_ff.h`
   **and** an ff-native `quad::ffun::ffcomplex` container (not `Kokkos::complex<ffloat>`).
   Separate future dispatch.
2. **Float rejected for all 10 box integrals** — not a defect, a workload fact (single
   precision too narrow). If a wider-than-float, narrower-than-double, library-native
   precision is ever of interest, the machinery routes it with a table change only
   (`shim_synthesis=True` + tokens); no new code.
3. **No source enrichment** was needed or performed for float, as scoped.
