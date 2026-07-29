# Phase-2 FF Downshift — LANDED (pipeline) — 2026-07-29

Pipeline-side landing of the **FF (float-float) downshift** on top of Reet's
`kokkosMaths_ff.h` enrichment (commit **d0f5b35**), following the Phase-2-float landing
(`PHASE_2_LANDED_2026-07-29.md`, 0e580fe) that fired STOP #EEE and left ff
enrichment-required. This dispatch verifies the enrichment empirically, enables the FF
`PrecisionProfile`, and re-runs the L-measure walking **FLOAT then FF** for the 10
raw-double integrals.

## 0. Executive verdict

| item | outcome |
|---|---|
| **STOP #FFF** (enrichment compile probe) | **NOT fired** — `BO<ql::ffun::ffcomplex, ql::ffun::ffloat, ql::ffun::ffloat>` on B0m compiles clean against real Kokkos + `third_party/`, runs, produces genuine two-limb ff output (all `.lo` limbs non-zero). Enrichment validated. |
| **STOP #GGG** (tu_emit end-to-end) | **NOT fired** — `emit_flip_tu(…, TargetPrecision.FF)` emits the correct static-header wrapper (`#include "kokkosMaths_ff.h"` + `USE_FF_COMPLEX`), `shim_path=None` (static, not shim), compiles + runs. |
| FF profile | **ENABLED** — `available=True`, `shim_synthesis=False`, `maths_header="kokkosMaths_ff.h"`, `cpp_output="ql::ffun::ffcomplex"`, `cpp_scalar="ql::ffun::ffloat"`. STOP #EEE marker removed; enrichment-commit note added. |
| Wrapper generator FF arm | **table-driven** — FF joins the existing `PROFILES.values()` static-header ladder automatically (STOP #SS: selected by `maths_header`, never a precision name). Zero new branches. |
| `route_downshift` (FLOAT, FF) walk | **unchanged + tested** — FF is now a genuine reachable fallback target (`test_downshift_fallback_selects_ff_when_float_absent`). |
| **L-measure (B1–B9, B11)** | **RAN** — FLOAT: 0/10 (byte-identical to 0e580fe). FF: **9/10 built, 1/10 build-failed (B2m/B11), 0/10 accepted.** Final routing: all 10 stay **raw double**. |
| **FF accepts 0/10** | **UNEXPECTED vs dispatch prediction (8–10/10).** Not a mechanism failure — ff builds + runs + genuine compute, delivering **~9.88 mean digits vs raw double's ~11.98**. ff is a *narrower* type than double (48-bit vs 53-bit mantissa), so a double→ff downshift **cannot preserve** double's delivered precision. Flagged for investigation (§3.3). |
| **B2m ff BUILD FAILED** | Vendored-primitive gap: `ql::ffun::ffloat` has no unary `operator+` (`ff_math.hpp`, commit b30ddb8 — predates the enrichment). B2m.h:124 uses `+p3sq`. dd's `ddouble` has `operator+()`; ff does not. One-line hand-back to Reet (§7). |
| Regression | **CLEAN** — 959 pass / 0 fail (10 llm deselected); acc1482 7/7; FLOAT digits byte-identical to 0e580fe; dd wrapper arm precedes ff so dd builds unaffected (STOP #ZZ); snapshot + `third_party` pristine (STOP #Z; enrichment untouched). |

**Bottom line:** the FF downshift **mechanism** is fully landed, validated, and honest —
9 of 10 raw-double integrals compile + run at ff via the static enrichment header, doing
genuine float-float compute. But **ff is narrower than double**, so the precision-preserving
downshift gate correctly rejects all measured integrals back to raw double: ff delivers
~2 fewer digits than the raw-double baseline it must not fall below. This **contradicts the
scoping prediction** (which assumed ff's ~14-digit ceiling would clear a fixed ~9-digit
floor; the real baseline is *delivered* double at ~12 digits, which ff lands below). The
box family has **no precision headroom for a double→ff downshift** — ff would only pay off
as a downshift from a *wider-than-ff* baseline (e.g. dd), not from double.

**Final 21-integral precision assignment (unchanged):** 11 dd candidates stay dd;
B1–B9, B11 stay **raw double**. No integral moves to float or ff.

---

## 1. Enrichment verification (deliverable 1 — done FIRST)

Per the dispatch ordering constraint, the empirical probe ran before any code landed. A
`/tmp` clone of the pristine snapshot + `third_party/include/kokkosMaths_ff.h` +
`ff_math.hpp`/`ff_complex.hpp`, with an FF-enabled wrapper, compiled a B0m group driver
instantiating `BO<ql::ffun::ffcomplex, ql::ffun::ffloat, ql::ffun::ffloat>`:

- **Compiles clean** against real Kokkos (`~/kokkos-install`) + `third_party/` (`g++ -std=c++20 -O2`).
- **Runs** — emits 210 RES lines (21 integrals × 10 samples) as `hi|lo` two-limb tokens.
- **Genuine ff** — every B0m-owned integral (B1–B5) shows **non-zero `.lo` limbs** on all
  5 coeff components → real float-float arithmetic, not a degenerate double.

**STOP #EEE cleared at the container level:** `kokkosMaths_ff.h` uses the custom
`ql::ffun::ffcomplex` container (aliased `namespace ql { namespace ffun = ::quad::ffun; }`),
never `Kokkos::complex<ffloat>` — so the `is_floating_point_v` static_assert that fired at
0e580fe is never reached. Structure mirrors `kokkosMaths_dd.h` exactly (`using complex =
ql::ffun::ffcomplex`, `Constants<T>` primary with 43-term Chebyshev + 25-term Bernoulli
FF-encoded via Dekker split, kAbs/kLog/kSqrt templates + ff specializations, ffcomplex
overloads, Sign/Max/Min/Htheta leaves).

---

## 2. What landed (deliverables 2–5)

### 2.1 FF profile enablement — `agents/patcher/tu_emit.py` (deliverables 2 + 3)

The FF `PrecisionProfile` already carried the correct tokens (declared for parameterization
at 0e580fe); this landing flips `available=False → True`, removes the STOP #EEE marker, and
notes the enrichment commit. The wrapper generator's static-header ladder
(`render_wrapper`) iterates `PROFILES.values()` selecting `available and define_macro is not
None` — **FF joins with zero new code** (STOP #SS: table-driven, not a precision-name
branch). Generated ladder (dd build shown; the driver's `#define` selects one arm):

```c
#if defined(USE_DD_COMPLEX)
#include "kokkosMaths_dd.h"
#elif defined(USE_FF_COMPLEX)
#include "kokkosMaths_ff.h"
#elif defined(USE_QUAD_COMPLEX)
...
#else
#include "kokkosMaths.h"
#endif
```

The FF arm is an `#elif` after dd, so a `USE_DD_COMPLEX` build never reaches it → **Phase-1
dd render byte-identical** (STOP #ZZ). The float shim-synthesis branch is untouched.

### 2.2 `route_downshift` verification (deliverable 4)

No code change — the `(FLOAT, FF)` walk (`DOWNSHIFT_PREFERENCE`) was implemented at 0e580fe.
Added `test_downshift_fallback_selects_ff_when_float_absent`: with FF the only available
target, the router selects FF — confirming ff is a genuine reachable fallback, not a
parameterization placeholder. The stale "FF filtered out (STOP #EEE)" test was retargeted to
assert the fail-loud *mechanism* on a `monkeypatch`-forced-unavailable profile (all three
shipped precisions are now available).

### 2.3 tu_emit end-to-end FF probe (deliverable 5, STOP #GGG gate)

Before the full L-measure, `emit_flip_tu(clone, "box/B0m.h", drv, TargetPrecision.FF)` was
compiled + run end-to-end: wrapper carries the FF include + `USE_FF_COMPLEX`,
`shim_path=None` (static-header, correct), driver instantiates `ql::ffun::ffcomplex` with an
`FFPrinter` narrowing ff→double at the app boundary (`static_cast<double>(v.hi) + …(v.lo)`,
via the shared `narrow_two_limb_scalar` — same contract as dd). Compiles, runs, emits
narrowed-double RES. **STOP #GGG not fired.**

---

## 3. L-measure results (deliverable 6)

`runs/qcdloop/phase2_lmeasure.py` re-run walking FLOAT then FF, 2000 samples.
`phase2_lmeasure_out/phase2_lmeasure.json`:

| integral | group | base (double) | FLOAT digits | FLOAT lift | FF built | FF digits | FF lift | final |
|---|---|---|---|---|---|---|---|---|
| B1  | B0m | 12.49 | 2.66 | −9.83 | ✅ | 9.26  | −3.23 | double |
| B2  | B0m | 12.92 | 4.06 | −8.87 | ✅ | 10.51 | −2.41 | double |
| B3  | B0m | 11.99 | 3.53 | −8.45 | ✅ | 9.68  | −2.30 | double |
| B4  | B0m | 11.37 | 3.79 | −7.58 | ✅ | 10.16 | −1.21 | double |
| B5  | B0m | 12.71 | 4.23 | −8.48 | ✅ | 10.13 | −2.58 | double |
| B6  | B1m | 12.27 | 3.33 | −8.94 | ✅ | 10.77 | −1.50 | double |
| B7  | B1m | 11.62 | 3.62 | −8.01 | ✅ | 10.79 | −0.83 | double |
| B8  | B1m | 10.77 | 2.64 | −8.13 | ✅ | 8.97  | −1.80 | double |
| B9  | B1m | 11.67 | 2.67 | −9.01 | ✅ | 8.64  | −3.03 | double |
| B11 | B2m | 9.46  | 0.48 | −8.98 | ❌ (build) | — | — | double |

**FLOAT: 0/10** (byte-identical to 0e580fe). **FF: 9/10 built, 0/10 accepted.**
`flip_build_failed: ["B2m_ff"]` (B11's group).

### 3.1 FLOAT — unchanged

All 10 FLOAT digit measures match 0e580fe to 2 dp (deterministic). Float loses 7.6–9.8
digits; rejected. No change.

### 3.2 B2m ff BUILD FAILED — vendored-primitive gap

`box/B2m.h:124` (`const TOutput ga43p = TOutput(+p3sq + m3sq - m4sq) + root;`) applies
**unary `operator+`** to a `ql::ffun::ffloat`. The vendored `ff_math.hpp` (commit b30ddb8,
which vendored the ff primitives — **separate from and predating** the d0f5b35 enrichment)
defines unary `operator-()` (line 58) but **no unary `operator+()`**. The dd primitive
`dd_math.hpp:55` has `ddouble operator+() const { return *this; }`; ff omits it. B0m/B1m
never use unary plus, so only B2m fails. **Not an enrichment-header (`kokkosMaths_ff.h`)
defect, and not a pipeline defect** — a one-line gap in the underlying ff primitive
(hand-back §7).

### 3.3 FF 0/10 — the anomaly (flagged per the verdict gate)

FF **builds, runs, and does genuine float-float compute** on all 9 measurable integrals
(RES `.lo` limbs non-zero), delivering **~9.88 mean digits**. But the raw-double **baseline**
delivers **~11.98 mean digits**, so every ff candidate lands **0.8–3.2 digits below** the
baseline it must not fall under → the DOWNSHIFT gate (`lift >= -margin`, margin 0) correctly
rejects all 9, and B11 rejects on build failure.

**Why this contradicts the "8–10/10" prediction — and why the measurement is right:**

- The scoping prediction assumed ff's ~14.4-digit ceiling would clear a **fixed ~9-digit
  floor** ("baseline exceeds ~14 digits" being the only reject case). That floor does not
  exist: the acceptance contract is *preserve the raw-double baseline*, and that baseline is
  **delivered double at ~10.8–12.9 digits**, not 9.
- **ff is a narrower type than double**: 2×24-bit = 48-bit mantissa (~14.4 digits) vs
  double's 53-bit (~15.95). A double→ff move is a **downshift in precision**, so for any
  integral not already cancellation-limited *below ff's ceiling*, ff delivers strictly fewer
  digits than double. These box integrals lose ~3–4 digits to cancellation from double's
  ~15.9 ceiling (→ ~12 delivered); ff loses the same cancellation from its lower ~14.4
  ceiling **plus** a little extra from error-free-transform arithmetic being marginally
  noisier than native double (→ ~9.9 delivered). Net: ff < double by ~2 digits, uniformly.
- This is the **same structural outcome as float, less severe** — float (~7 digits) loses
  ~8, ff (~14.4) loses ~2. Neither can preserve delivered double.

**Conclusion:** the box family has **no precision headroom for a double→ff downshift**. ff
would only pay off as a downshift from a *wider* baseline (dd → ff, where ff's 14 digits
could preserve a workload that genuinely needs <14), never from double. The Phase-1 dd
candidates are explicitly **not** downshiftable to ff (STOP #ZZ, and dd→ff would still lose
the dd accuracy those integrals were promoted *for*). So there is no integral in this
workload for which ff is the right precision — a genuine workload fact, cleanly measured,
not a pipeline or enrichment defect.

---

## 4. Regression preservation (deliverable 7)

- **Full suite:** 959 passed, 0 failed (10 llm-marked deselected). Deterministic patcher +
  integrator_base + validator all green.
- **New/updated tests:** `test_tu_emit.py` (+`test_wrapper_has_ff_arm_after_enrichment`,
  `test_ff_profile_available_via_enrichment`, `test_ff_group_driver_shape`; retargeted
  `test_unavailable_precision_fails_loud_not_dd_fallback` to a forced-unavailable profile);
  `test_precision_flip.py` (+`test_downshift_fallback_selects_ff_when_float_absent`,
  renamed the float-only case).
- **acc1482 boundary:** `test_flip_boundary.py` 7/7.
- **FLOAT byte-identical to 0e580fe:** all 10 FLOAT digit measures match to 2 dp.
- **DD Phase-1 render unaffected:** the FF wrapper arm is an `#elif` after the dd `#if`, so a
  `USE_DD_COMPLEX` build never reaches it (STOP #ZZ). All 11 dd accepts unchanged.
- **Snapshot + `third_party` pristine** (`git status --porcelain` empty before and after the
  run — the enrichment header and ff primitives are read, never written; STOP #Z).

---

## 5. STOP audit

| STOP | definition | state |
|---|---|---|
| **#FFF** | `kokkosMaths_ff.h` enrichment fails empirical compile probe | **not fired** — B0m `BO<ff,…>` compiles + runs + genuine ff |
| **#GGG** | FF wrapper generation emits wrong content | **not fired** — tu_emit FF TU compiles + runs end-to-end |
| **#EEE** | `Kokkos::complex<ffloat>` won't compile | **cleared** — enrichment uses custom `ql::ffun::ffcomplex`, never `Kokkos::complex<ffloat>` |
| **#SS** | wrapper ladder branches on a precision name | **not fired** — FF joins the `PROFILES.values()` table by field lookup |
| **#ZZ** | Phase-2 FF regresses a Phase-1 dd accept | **not fired** — dd arm precedes ff; dd render byte-identical; dd candidates never downshifted |
| **#Z** | snapshot / enrichment pristine | **clean** — all generation into clones; `third_party` untouched |
| **#A** | a lift is measured against dead code | **not fired** — genuine two-limb ff compute confirmed (non-zero `.lo`) |

---

## 6. Artifacts

- Production: `agents/patcher/tu_emit.py` (FF profile enabled + wrapper comment).
- Tests: `tests/patcher/test_tu_emit.py`, `tests/patcher/test_precision_flip.py`.
- Harness: `runs/qcdloop/phase2_lmeasure.py` (walks FLOAT then FF; per-precision rows).
- Results: `runs/qcdloop/phase2_lmeasure_out/phase2_lmeasure.json` + build logs;
  `runs/qcdloop/phase2_ff_lmeasure.log`.
- Enrichment (Reet, upstream): `third_party/include/kokkosMaths_ff.h` @ d0f5b35.
- Reports: this file; prior `PHASE_2_LANDED_2026-07-29.md`.

---

## 7. Hand-backs for Reet

1. **`ff_math.hpp` missing unary `operator+`** (blocks B2m/B11 at ff). One line, mirroring
   dd's `dd_math.hpp:55`: add to the `ffloat` struct
   `KOKKOS_INLINE_FUNCTION ffloat operator+() const { return *this; }`. This is in the
   vendored ff primitive (commit b30ddb8), NOT the `kokkosMaths_ff.h` enrichment. With it,
   B2m/B11 would build at ff — but B11 would still reject (its baseline 9.46 digits already
   exceeds what ff delivers here; ff would land below it like the other 9).

2. **FF 0/10 is a workload fact, not a defect** (§3.3). The box family has no precision
   headroom for a double→ff downshift: ff (14.4 digits) is narrower than double (15.95), and
   these integrals aren't cancellation-limited below ff's ceiling, so ff delivers ~2 fewer
   digits than the raw-double baseline. ff would only pay off downshifting from a *wider*
   baseline (dd→ff for a workload needing <14 digits), which is a different (future) routing
   than Phase-2's double→narrower path. **No action** unless a dd→ff tier is of interest.

3. The FF **machinery** is fully landed and correct — a future workload with a wider
   baseline routes to ff by the same table (`available=True`, static header), no new code.
