# Phase-2 Endpoint-Lock Speedup — STOP #XX (header enrichment required) — 2026-07-29

Dispatch: "Phase-2 endpoint-lock speedup + Phase-1 cleanups." Part A (Phase-1 report +
MEMORY reframe under Reet's GPU-ceiling acceptance criterion) **landed**. Part B (Phase-2
downshift implementation) is **blocked at the header-enablement gate** — the required
precision maths headers are not vendored — so per the dispatch's verdict gate this report
lands **Part A only** and hands Part B back for source enrichment (STOP #XX).

## 0. Executive verdict

| item | outcome |
|---|---|
| Part A1 — report reframe (§0/§3.2/§4/§5) | **LANDED** — `PHASE_1_LANDED_2026-07-28.md` reframed to GPU-ceiling acceptance; all 11 candidates accepted; STOP #WW retained as future-validation note |
| Part A2 — MEMORY reframe | **LANDED** — `project_phase1_landing.md` + `MEMORY.md` index updated |
| Part B — header/profile enablement | **🛑 STOP #XX** — **both** ff and float precision maths headers are absent from the vendored snapshot AND `ddfun_enabled` AND every branch |
| Generator precision-parameterization | **CONFIRMED clean (no STOP #YY)** — `tu_emit.py` renders a correct float driver once the profile is enabled in-memory; the blocker is missing headers, not a dd-hardcoded generator |
| STOP #YY | **not fired** — Decision 2 discipline honored end-to-end |
| STOP #ZZ | **not applicable** — no Phase-2 routing ran; Phase-1 dd candidates untouched |
| STOP #Z | **clean** — no writes under `runs/qcdloop_headers_full/`; probe used a `/tmp` clone, now removed |

**Bottom line:** Phase 2's *mechanism* is ready — the wrapper/driver generator, the dispatch
layer, and the acceptance gate are all already precision-parameterized from Phase 1
(Decision 2 / STOP #SS discipline), exactly as the dispatch predicted. The only thing
missing is the **precision maths headers themselves**, which are human-authored source
enrichment (analogous to the dd primitives at `e3d2e45` / the Phase-1 `third_party/`
enrichment), NOT a pipeline mutation. Phase 2 cannot proceed for either precision until the
headers land.

---

## 1. What Phase-2 needs (the endpoint-lock speedup direction)

Per the dispatch's own correction, the meaningful Phase-2 direction is **DOWN**:

- For each integral currently at **raw double** (B1–B9, B11 = 10 integrals): attempt a
  **float** downshift. Accept iff float **builds AND lift >= 0.0** vs the raw-double baseline
  (precision preserved; float on GPU is a strict speedup by construction).
- Phase-1 **dd candidates** (the 11 accepts) are **not touched** by default — a dd→ff/float
  downshift is mathematically doomed (ff/float carry less precision than dd, so measured
  lift is negative). Deferred unless characterization ever suggests an over-precisioned dd
  integral (edge case).

So the immediate Phase-2 work is the **double → float** downshift, which needs a **float**
precision maths header. An **ff** header would only be needed for a future dd→ff experiment
(not the default Phase-2 flow), but the dispatch asked the enablement question for both.

---

## 2. Header determination (the STOP #XX evidence)

### 2.1 What is vendored today

| header | location | status |
|---|---|---|
| `kokkosMaths.h` | `runs/qcdloop_headers_full/` | present — **double** maths (`Constants<T>` + `double`/`complex<double>` leaf overloads) |
| `kokkosMaths_dd.h` | `runs/qcdloop_headers_full/` | present — **dd** maths (`ql::ddfun::` `Constants` + ddouble/ddcomplex leaf overloads) |
| `kokkosMaths_wrapper.h` | `runs/qcdloop_headers_full/` | present — snapshot version branches only `USE_QUAD_COMPLEX` vs double |
| `kokkosMaths_ff.h` | — | **ABSENT** everywhere (snapshot, `ddfun_enabled`, all local + remote branches) |
| `kokkosMaths_float.h` | — | **ABSENT** (a float analog does not exist as a separate header) |

Low-level primitives that exist but are **not** the maths-wrapper layer:

| primitive header | namespace | note |
|---|---|---|
| `third_party/include/dd_math.hpp`, `dd_complex.hpp` | `quad::ddfun` | dd scalar/complex arithmetic; the dd maths header aliases `ql::ddfun = quad::ddfun` on top of these |
| `third_party/include/ff_math.hpp`, `ff_complex.hpp` | `quad::ffun` | ff scalar (`ffloat`) / complex (`ffcomplex`) arithmetic — **primitives only**; there is **no `kokkosMaths_ff.h` wrapper** that specializes `Constants<T>` and the `ql::` leaf overloads (`kAbs`/`kLog`/`Real`/`Imag`/`Sign`/`Max`/`Min`/`Htheta`) at ff precision |

The dd case is the template: `kokkosMaths_dd.h` is the *wrapper layer* that sits on top of
the `quad::ddfun` primitives and provides the `Constants<T>` specialization plus the
precision-specific `ql::` leaf overloads the box integrals call. **ff has the primitives but
not this wrapper layer; float has neither a wrapper nor float-typed leaf overloads.**

### 2.2 float compile probe (empirical, decisive)

The FLOAT profile in `tu_emit.PROFILES` is designed to reuse the existing double
`kokkosMaths.h` (`define_macro=None`, `maths_header="kokkosMaths.h"`,
`cpp_output="Kokkos::complex<float>"`, `cpp_scalar="float"`). Whether that actually
instantiates at float is an **empirical** question, because `kokkosMaths.h`'s leaf functions
are **`double`-hardcoded non-template overloads**, not templates:

```
kokkosMaths.h:292  double kAbs(Kokkos::complex<double> const& x)
kokkosMaths.h:319  double Imag(double const& x)
kokkosMaths.h:331  double Real(Kokkos::complex<double> const& x)
kokkosMaths.h:339  Kokkos::complex<double> Sign(Kokkos::complex<double> const& x)
kokkosMaths.h:344  double Max(double const& a, double const& b)      (+ Min, Htheta ...)
```

I ran the existing precision-parameterized generator with the FLOAT profile enabled
**in-memory only** (no source edit), emitted a per-group float driver for `box/B1m.h` into a
`/tmp` clone, and compiled it against the snapshot clone + Kokkos. **Result: build fails**
with two independent errors:

1. **Leaf-overload gap (needs a header change):**
   ```
   box/B1m.h:73  error: no match for 'operator>' (operands 'Kokkos::complex<float>' and 'Kokkos::complex<float>')
                 if (ql::kAbs(x4[0]) > ql::kAbs(x4[1]))
   ```
   `ql::kAbs(Kokkos::complex<float>)` finds no `double kAbs(Kokkos::complex<double>)` overload
   (float→complex<double> is not a viable conversion for the const-ref overload as ranked),
   so it binds the generic `template<T> T kAbs(T)` which returns a **`complex<float>`** rather
   than a real — and comparing two complexes has no `operator>`. Every `Real`/`Imag`/`Sign`/
   `Max`/`Min`/`Htheta` call on a float/`complex<float>` has the same defect. This is the
   float analog of exactly why `kokkosMaths_dd.h` exists: dd needed a wrapper layer with
   dd-typed leaf overloads, and float needs the same.

2. **Recipe input helper (secondary):**
   ```
   boxGPU_app_recipes.hpp:195  error: request for member 'hi' in 'v', which is of non-class type 'float'
                               else return v.hi;
   ```
   `to_d` handles only `double` (identity) and two-limb `.hi`. A float path needs a
   `std::is_same_v<..., float>` arm (a small recipe extension — pipeline-owned, not a
   snapshot header). This alone would not block; error (1) is the hard STOP.

**Conclusion for float:** the double `kokkosMaths.h` **cannot** be instantiated at float as-is
— the determination the dispatch asked for. Float needs a `kokkosMaths_float.h` analog:
a float-specialized `Constants<T>` (float literals / a suitable Chebyshev+Bernoulli table)
and float-typed leaf overloads (`kAbs`/`Real`/`Imag`/`Sign`/`Max`/`Min`/`Htheta` for `float`
and `Kokkos::complex<float>`). That is human-authored source enrichment.

### 2.3 ff determination

`kokkosMaths_ff.h` is absent entirely. The `quad::ffun` primitives exist, but the
maths-wrapper layer (ff-specialized `Constants<T>` + ff/`ffcomplex` leaf overloads) that the
box integrals call does not. This is exactly the shape of the dd enrichment
(`kokkosMaths_dd.h` on top of `quad::ddfun`), just never authored for ff. → enrichment
needed.

### 2.4 Not STOP #YY — the generator is genuinely precision-parameterized

STOP #YY would fire if the generator turned out to be secretly dd-hardcoded (i.e. Phase-1's
Decision 2 discipline was not honored end-to-end). **It did not.** With the FLOAT profile
enabled in-memory, `render_group_driver(..., TargetPrecision.FLOAT)` produced a correct,
well-formed float driver by a pure table lookup — the right `run_app<Kokkos::complex<float>,
float, float, FloatPrinter>` instantiation, the right single-limb `FloatPrinter` (a plain
`static_cast<double>` narrow), no `#define` arm (float is the default `kokkosMaths.h` arm),
and the `#include <Kokkos_Complex.hpp>` the single-limb path needs. The *only* reason it
does not build is the missing float leaf overloads in the (unmodifiable) snapshot header —
a header gap, not a generator gap. **Decision 2 / STOP #SS held end-to-end.** No code path
change is needed in the generator for ff/float — only profile enablement, which is gated on
the headers.

---

## 3. The enrichment ask (hand-back for Reet)

Phase 2 needs a **human-authored** precision maths header, analogous to the dd enrichment at
`e3d2e45` and the Phase-1 `third_party/` primitive completion — **NOT** a pipeline mutation,
and **NOT** a write to `runs/qcdloop_headers_full/` (STOP #Z).

**Primary (unblocks the real Phase-2 double→float downshift):**
- **`kokkosMaths_float.h`** — a float analog of `kokkosMaths_dd.h`:
  - `ql::Constants<T>` specialized/usable at float (the Chebyshev `_C` / Bernoulli `_B`
    tables and the `_pi`/`_eps*`/`_one`/… constants at float precision; float literals or a
    float-narrowed table),
  - float-typed leaf overloads: `kAbs`, `kLog`, `kSqrt`, `kConj`, `Real`, `Imag`, `Sign`,
    `Max`, `Min`, `Htheta` for `float` and `Kokkos::complex<float>` (the set that
    `kokkosMaths_dd.h` provides for ddouble/ddcomplex).
  - Whether this is a new standalone header or a `USE_FLOAT_COMPLEX` arm added to a
    human-authored wrapper is Reet's call; the generator will pick it up via a one-line
    `PROFILES[FLOAT]` change (header name + `available=True` + macro, if any).
  - The recipe `to_d` will need a `float` arm (pipeline-owned; I can land that once the
    header exists).

**Secondary (only for a future dd→ff experiment, not the default Phase-2 flow):**
- **`kokkosMaths_ff.h`** — an ff analog of `kokkosMaths_dd.h` on top of the existing
  `quad::ffun` primitives (`third_party/include/ff_math.hpp`, `ff_complex.hpp`):
  ff-specialized `Constants<T>` + ff/`ffcomplex` leaf overloads. Deferred — ff downshift from
  dd is mathematically doomed and float downshift from double is the Phase-2 win.

Per the dispatch's "If only float is available (ff missing): proceed with float-only Phase 2"
branch — **neither is available today**, so Phase 2 proceeds with **neither** until at least
`kokkosMaths_float.h` lands. Recommended order: land `kokkosMaths_float.h`, re-run this
dispatch's Part B for a **float-only Phase 2** on B1–B9/B11, defer ff to a follow-up.

---

## 4. What I did NOT do (dispatch bans honored)

- Did **not** modify `runs/qcdloop_headers_full/` (STOP #Z) — the float probe used a `/tmp`
  clone, now deleted; snapshot `git status` clean.
- Did **not** author a float or ff maths header myself — that is the human enrichment ask,
  not a pipeline output (and inventing a `Constants<T>` table would be coefficient synthesis,
  §3.4 ban).
- Did **not** flip `PROFILES[FF].available` / `PROFILES[FLOAT].available` to `True` — leaving
  them `False` keeps the fail-loud guard honest (selecting an unavailable precision raises,
  never silently degrades). Enabling them before their headers exist would be the STOP #SS
  trap in reverse.
- Did **not** touch any Phase-1 dd candidate (STOP #ZZ not applicable — no routing ran).
- Did **not** use `ddfun_enabled` as a build input (it remains Validator oracle only).
- Did **not** re-litigate Phase-1 acceptance under quad/analytic framing (Part A honors the
  GPU-ceiling criterion as final).

---

## 5. STOP audit

| STOP | state |
|---|---|
| **#XX** (required precision header absent) | **FIRED** — both `kokkosMaths_float.h` and `kokkosMaths_ff.h` absent; Phase-2 downshift blocked pending human enrichment |
| **#YY** (generator secretly dd-hardcoded) | **not fired** — generator renders correct float driver by table lookup; Decision 2 honored end-to-end |
| **#ZZ** (Phase-2 regresses a Phase-1 accept) | **not applicable** — no Phase-2 routing executed; dd candidates untouched |
| **#Z** (snapshot pristine) | **clean** — no snapshot writes; probe clone in `/tmp`, removed |
| **#SS** (precision-parameterized, not dd-hardcoded) | **held** — `available` flags left `False`, fail-loud guard intact |

---

## 6. Regression preservation

- All Phase-1 accepts (11 dd candidates) remain at dd — untouched.
- No source changed except the two Part-A report reframes and the two memory files; the
  Phase-1 machinery (`agents/patcher/{precision_flip,tu_emit,flip_dispatch,flip_gate}.py`,
  `narrow_two_limb_scalar`) is byte-identical.
- `PROFILES` table byte-identical (ff/float still `available=False`).
- acc1482 26 tests + Phase-1 48 tests + full patcher/integrator/shared suites unaffected
  (no code touched).

---

## 7. MEMORY.md update block

Applied (see `project_phase1_landing.md` + `MEMORY.md`):
- Phase-1 accepts all 11 candidates under Reet's GPU-ceiling criterion (FINAL); dd is the
  GPU precision ceiling; build-clean AND lift > 0.0 vs raw double is the definitive Phase-1
  rule; do not re-litigate under quad/analytic.
- Realizable Phase-1 lift capped at ~15.9 (double output floor) by design; +3.77 for B10 is
  correct, not a shortfall vs +18.43 (which assumed a dd-output contract).
- STOP #WW's vs-dd circularity reclassified as a future validation concern for workload
  characterization, not a Phase-1 gate.
- Phase-2 float/ff downshift blocked on STOP #XX: `kokkosMaths_ff.h` + a float-instantiable
  maths header both absent; generator clean (no STOP #YY); handed back for header enrichment.

---

## 8. Verdict

Per the dispatch's verdict gate: **"Header enrichment needed → STOP #XX, land Part A
cleanups only, hand back for enrichment."** That is this landing:

- **Part A (A1 + A2) landed.**
- **Part B blocked at STOP #XX** — hand back for `kokkosMaths_float.h` (primary) and
  `kokkosMaths_ff.h` (secondary) enrichment.
- On `kokkosMaths_float.h` landing, re-run this dispatch's Part B for a **float-only Phase 2**
  (double→float downshift on B1–B9, B11), enabling `PROFILES[FLOAT]` and adding the recipe
  `to_d` float arm. ff and Phase 3 stay queued as separate future dispatches.

---

## 9. Artifacts

- Reframed report: `runs/qcdloop/PHASE_1_LANDED_2026-07-28.md` (§0/§3.2/§4/§5).
- This report: `runs/qcdloop/PHASE_2_STOP_XX_2026-07-29.md`.
- Memory: `project_phase1_landing.md`, `MEMORY.md`.
- Machinery examined (unchanged): `agents/patcher/{precision_flip,tu_emit,flip_dispatch,flip_gate}.py`.
- float probe: in-memory profile enable + `/tmp` clone compile (transient, removed; snapshot
  pristine).
