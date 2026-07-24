# Item 6 — B12 whole-integral dd diagnostic: STOP (reuse path not clean)

**Date:** 2026-07-24
**Branch:** langgraph-agents · head at start `9c66541` (Item 5 dd triage)
**Scope:** diagnostic probe only (no solver/fan-out/gate/shim/manifest changes)
**Outcome:** 🛑 **STOP — the probe as specified is not executable by reuse.** The
question ("does whole-chain dd lift B12's floor?") requires a precision reference
*strictly above* double-double, and no such reference exists in the current
host-only pipeline. Building one is a bounded feature, not a minutes-scale probe.
Handing the scope call back to Reet per the Item 6 constraint:

> "If this turns into a design exercise rather than a probe, STOP and report —
> that means the reuse path isn't as clean as MEMORY.md suggests and Reet needs
> to make the scope call."

---

## 1. The question Item 6 asks

Item 5 established that B12's floor comes from a dilog-chain cancellation at
`res(i,0) = … − 2·dilog1 − dilog2 − dilog3` (single-region dd can't span it; the
dilogs are computed in double upstream). Item 6 wants to decide between two
hypotheses by **measuring** B12's `min_precise_digits` under **whole-chain dd**:

* **COMPUTED cancellation** — dd on the whole chain lifts the floor from ~3.69 to
  ~15–18 digits → a future Tier-B (call-graph dd fan-out) has demonstrable value.
* **ALGORITHMIC cancellation** — dd leaves the floor at ~3.69 → no precision-only
  lever; arc ceiling confirmed at Stage 2 speedup-only.

Deciding this is a **measurement**, not a prediction.

## 2. Why it cannot be measured by reuse — the precision-ladder blocker

Accuracy is distance from truth. To measure the accuracy of a **dd-computed**
result, the reference must be **strictly more precise than dd**. The pipeline's
precision ladder has exactly two rungs, and **dd is the top rung** — it *is* the
oracle:

| Rung | `QL_MODE` | Driver | Tree | Role |
|------|-----------|--------|------|------|
| double (~16 dig) | `vanilla` | `runs/qcdloop/src/boxGPU_vanilla.cpp` | `qcdloop_headers_full` | candidate/baseline |
| double-double (~31 dig) | `dd` | `runs/qcdloop/src/boxGPU_dd.cpp` | `qcdloop@ddfun_enabled` | **ground-truth reference** |

`agents/validator/validate.py` scores `precise_digits = −log10(|cand − dd_ref|/|dd_ref|)`.
The reference is **dd**, not FP128.

Consequence for a whole-app dd candidate:

* The Item 6 "whole-integral dd build" **is** exactly `_build_dd_binary` /
  `_run_dd` — the same build that produces the oracle. Scoring it against the dd
  oracle is **dd-vs-dd on identical code = bit-identical = 31.9 (the cap)**. This
  is precisely the documented smoke result "DD-vs-DD self min = 31.9092"
  (`project_validator_architecture`). It measures plumbing, not the cancellation
  floor — it tells us nothing about the 3.69 lift.
* There is **no FP128 / quad / MPFR reference** anywhere in the pipeline
  (`grep` for `float128|quadmath|mpfr|qd_real` over `agents/`, `third_party/`,
  `runs/qcdloop/src/` → nothing usable; `boxGPU_dd.cpp` explicitly avoids
  libquadmath, reconstructing the dd reference as `Decimal(hi)+Decimal(lo)`).

**The task contract's premise is mistaken.** It says *"Score against the FP128
reference the way validate() does."* `validate()` does **not** score against
FP128 — its top tier is dd. The mental model "the dd oracle is itself a whole-app
dd build" (true) was conflated with "we can measure a dd candidate against it"
(false — circular). There is no FP128 reference to reuse.

## 3. Why the one measurement we *can* reuse does not discriminate

The only quantity obtainable from existing tiers is `|dd − double|/|dd|` — i.e.
double's own relative error, ≈ the **baseline** we already have:

* **B12 baseline (double vs dd), from Item 3 / DD_PROBE:** whole-app
  `min_precise_digits = 3.6906`, hotspot **sample 3868, component `coeff0.imag`**.

This does **not** separate the two hypotheses:

* COMPUTED cancellation and ALGORITHMIC cancellation produce the **same**
  `|dd − double|` signature (both just reflect that double is wrong by ~2e-4 rel).
* `|dd − double|` says double disagrees with dd; it says nothing about whether dd
  itself is **correct**. Only a reference above dd can reveal how many digits the
  dd result actually retains. So the discriminating measurement is fundamentally
  unavailable at the double/dd ladder.

## 4. A quad facility exists — but off-pipeline and GPU-only

`~/qcdloop` has a `quad_enabled` branch (`83c89a3`) that reintroduces quad math
(`kokkosMaths_quad.h`, `quad_complex.hpp`, `quad_math.hpp`). It is **not** the
`ddfun_enabled` tree the oracle uses, and it is **CUDA-only**:

```cpp
// quad_enabled:src/qcdloop/kokkosMaths_wrapper.h
#ifdef USE_QUAD_COMPLEX
#ifdef KOKKOS_ENABLE_CUDA
#include "kokkosMaths_quad.h"
#else
#error "USE_QUAD_COMPLEX requires KOKKOS_ENABLE_CUDA to be defined"
#endif
```

The entire Validator/oracle pipeline is **host-only Serial** (locked with Reet
2026-07-17). So the quad reference, as it exists today, **cannot build** in the
pipeline — it hard-`#error`s without CUDA. It is a real Tier-3 candidate, but not
a reusable one.

## 5. What answering Item 6 would actually take (the design exercise)

To get a genuine dd-vs-Tier3 measurement of B12's floor:

1. Add `QL_MODE=quad` to `runs/qcdloop/app/CMakeLists.txt` (currently
   `vanilla | dd` only).
2. Write `runs/qcdloop/src/boxGPU_quad.cpp` + a `QuadPrinter` (analogue of
   `boxGPU_dd.cpp`'s `DDPrinter`) emitting quad values as hex.
3. Supply a **host-capable** quad backend — either port the CUDA-only
   `kokkosMaths_quad.h` to host `__float128`/libquadmath, or stand up a CUDA
   build (which breaks the bit-identical host-Serial input invariant). Both are
   new code + toolchain decisions.
4. Add a Python quad-hex → `Decimal` reconstruction path (the current parser only
   knows double hex and dd `hi|lo`).
5. Add a scoring mode that references a **dd candidate** against the **quad**
   reference (today's `_score`/`precise_digits_fast` implicitly assume the
   candidate is double and the reference is dd).

This is new codegen + new pipeline wiring + a numerics/toolchain call — hours-plus
and design decisions, squarely a **feature**, not the minutes-scale
reuse-the-oracle probe Item 6 scoped. Per the constraints ("Do not design new shim
generation logic. This is a probe, not a feature.") I stopped rather than build it.

## 6. Verdict

**No COMPUTED/ALGORITHMIC verdict is issued — it cannot be measured with existing
infrastructure, and issuing one would be a prediction dressed as a measurement.**

The honest status:

* B12's double baseline floor is **3.6906** (sample 3868, `coeff0.imag`) — known.
* Whole-app dd scored against the dd oracle is **31.9 (trivial self-consistency)**
  — carries no floor-lift information.
* The discriminating measurement requires a **> dd** reference that the host-only
  pipeline does not have and cannot build by reuse.

## 7. Scope call for Reet

Item 6's stated purpose was to decide the shape of post-Stage-2 work. Given the
blocker, the options are:

* **(A) Launch Stage 2 speedup-only now** (Path 1 from Item 5). Item 5 already
  established single-region dd is structurally empty and the floor needs
  whole-chain dd *or* algorithmic rewrite (scoped out). Ship the 21 trees, design
  Phase 2f merge, and **document the dd column as untested** — the same honest
  posture Item 5 recommended. The COMPUTED-vs-ALGORITHMIC distinction is not
  required to ship speedup-only.
* **(B) Fund a Tier-3 (quad) reference feature** (§5) *first*, then run this
  probe against it to get a real verdict, then decide Tier-B vs arc-ceiling. This
  is a separate, bounded feature session — not Item 6's minutes budget.

Physical intuition (a subtraction of near-equal dilogs computed in finite
precision) *suggests* COMPUTED cancellation and thus that Tier-B would help — but
that is a hypothesis, and Item 6 explicitly wanted the measurement. Recommend
**(A)** now and defer **(B)** unless a real dd column is needed to justify Tier-B.

No Stage 2 launched, no code changed, no gate/fan-out/manifest touched (read-only,
per Item 6).

## 8. Method / reproducibility

Read-only investigation:

```bash
# Pipeline has only vanilla|dd; dd is the oracle:
sed -n '1,50p' runs/qcdloop/app/CMakeLists.txt          # QL_MODE = vanilla | dd
sed -n '1,30p'  runs/qcdloop/src/boxGPU_dd.cpp           # dd oracle, no libquadmath
grep -n 'precise_digits =' agents/validator/precise_digits.py   # ref = dd

# No FP128/quad/mpfr reference in the pipeline:
grep -rliE 'float128|quadmath|mpfr|qd_real|__float128' agents/ third_party/ runs/qcdloop/src/

# A quad facility exists only on an off-pipeline, CUDA-only branch:
git -C ~/qcdloop branch -a | grep quad          # quad_enabled
git -C ~/qcdloop show quad_enabled:src/qcdloop/kokkosMaths_wrapper.h | sed -n '11,19p'
```

Baseline figures cited from `runs/qcdloop/SOLVER_STAGE1_DD_PROBE.md` (Item 3) and
`runs/qcdloop/DD_TRIAGE_2026-07-25.md` (Item 5); DD-vs-DD self = 31.9092 from
`project_validator_architecture` memory.
