# Phase-2 FF BIN Diagnosis — STOP #III verify + STOP #HHH root-cause — 2026-07-29

Pulls the `ff_math.hpp` int-ctor enrichment (**4f21245**), re-measures the four BIN targets
that stayed on double, and root-causes the BIN1/BIN2 ff-runtime-nan (STOP #HHH). **Diagnose
only — no fix pushed from this side.**

## 0. Executive verdict

| item | outcome |
|---|---|
| Source pull | **4f21245** fast-forwarded; `third_party/` pristine after pull (6 lines added to `ff_math.hpp`, int ctor). |
| STOP #III (BIN4 / B4m) | **RETIRED** — B4m ff now **builds** (int→ffloat ambiguity cleared). BIN4 then hits the runtime nan (same as BIN1/BIN2). |
| STOP #III (BIN3 / B3m) | **PERSISTS with a NEW, different error** — not `int→ffloat` (that's fixed) but the **reverse** `ffloat→int`: `B3m.h:68-70` assign an `ffloat` expression to `int ir12/ir14/ir24`. `ff_math.hpp` has no `operator int()`; `dd_math.hpp:56` does. New hand-back (§2). |
| STOP #HHH (BIN1/BIN2/BIN4 nan) | **DIAGNOSED — Category 1 (vendored ff-primitive bug).** Fault site: `ff_complex.hpp` `abs()` (line 107) and `sqrt()` (line 118) compute `re²+im²` **without hypot-style scaling**. For \|operand\| > √FLT_MAX ≈ **1.84e19**, the float square overflows to +Inf → nan. Hand-back to Reet (§3.5). |
| Final routing | **UNCHANGED** — dd=4, ff=13, double=4 (BIN1–4). No BIN target reaches ff. |
| Regression | No pipeline code moved (diagnosis ran in `/tmp`); `third_party` + snapshot pristine; flip_gate 18/18. |

**Bottom line.** The int-ctor fix did its job for **B4m** (BIN4 now builds), but BIN4 then
lands in the *same* nan as BIN1/BIN2, so it still routes to double. **B3m** fails on a
*different* vendored gap (the reverse conversion, `ffloat→int`). And the nan itself is a
genuine **ff-primitive overflow bug**: complex `abs`/`sqrt` square the operand with no
scaling, so any intermediate above √FLT_MAX overflows float — which dd never hits because
double's range is ~1e154. All four BIN integrals stay on double; two clean hand-backs to Reet.

---

## 1. STOP #III verification — did the int-ctor fix work?

Re-ran `phase2_lmeasure.py --tolerance 7.0 --targets BIN1,BIN2,BIN3,BIN4`
(`runs/qcdloop/phase2_bin_ff_out/`). Build + digit state:

| integral | group | ff build (prev → now) | ff_digits | route |
|---|---|---|---|---|
| BIN1 | B1m | built → built | **nan** | double |
| BIN2 | B2m | built → built | **nan** | double |
| BIN3 | B3m | **build-fail → build-fail (new error)** | — | double |
| BIN4 | B4m | **build-fail → BUILDS** ✅ | **nan** | double |

**BIN4 (B4m):** the int-ctor fix (4f21245) cleared the `int→ffloat` ambiguity that blocked
B4m. It now compiles and runs — landing in the shared nan (§3), so it routes to double.
STOP #III **retires for B4m/BIN4**.

`accepted at float: 0/4 · accepted at ff: 0/4 · final downshifted: 0/4`.

---

## 2. STOP #III persists for BIN3/B3m — a *different* vendored gap

B3m ff build fails with a **new** error (not the one 4f21245 targeted):

```
box/B3m.h:68:65: error: cannot convert 'quad::ffun::ffloat' to 'int' in assignment
box/B3m.h:69:65: error: cannot convert 'quad::ffun::ffloat' to 'int' in assignment
box/B3m.h:70:65: error: cannot convert 'quad::ffun::ffloat' to 'int' in assignment
```

`B3m.h:68-70` (the BIN3 body):

```cpp
int ir12 = 0, ir14 = 0, ir24 = 0;
...
if (ql::Real(k12) < -..._two()) ir12 = ql::Constants<TScale>::_ten() * ql::Sign(...);   // ffloat -> int
```

This is the **reverse** conversion from the one 4f21245 fixed:
- 4f21245 added `ffloat(int)` — an **int → ffloat** *constructor*. ✅
- B3m needs **ffloat → int** — a *conversion operator*, `operator int() const`.

`dd_math.hpp:56` has exactly this (`KOKKOS_INLINE_FUNCTION operator int() const { return (int)hi; }`),
which is why B3m/BIN3 compiles at dd but not ff. `ff_math.hpp` has no such operator.

**Per the task constraint (different compiler error → report exact error and stop, no
guess-fix), this is a clean hand-back:**

> **Hand-back (STOP #III residual, BIN3):** add to `ffloat` in `ff_math.hpp` a scalar→int
> conversion mirroring `dd_math.hpp:56`:
> `KOKKOS_INLINE_FUNCTION operator int() const { return (int)((double)hi + (double)lo); }`
> (route via the two-limb sum so ints near float's 2^24 boundary don't truncate on the hi
> limb alone). With it, B3m/BIN3 would build — and then, like the others, hit the §3 nan and
> route to double anyway. The fix retires the *build* STOP; it does not change routing.

---

## 3. STOP #HHH root-cause — BIN1/BIN2/BIN4 ff = nan

### 3.1 Category: **1 (vendored ff-primitive bug).**

Not workload-arithmetic (cat 2) and not an enrichment-coefficient error (cat 3): the nan is a
**float exponent-range overflow** inside the vendored complex `abs`/`sqrt` primitives, which
square their operand with no protective scaling.

### 3.2 The nan is input-dependent and *shared* across all three integrals

| integral | nan / total | fraction |
|---|---|---|
| BIN1 | 1048 / 2000 | 52.4% |
| BIN2 | 1048 / 2000 | 52.4% |
| BIN4 | 1048 / 2000 | 52.4% |

Not just identical counts — the **nan sample-index sets are bit-identical** across BIN1, BIN2,
BIN4 (0-line diff over the first 200 samples). Three *different* box formulas nan on *exactly*
the same inputs ⇒ the fault is in a **shared primitive**, driven by the input magnitude regime
(`bin_fill` draws momenta in [100, 1e6]; the derived `k`-ratios and discriminant scale into
the 1e19+ range for ~half the samples).

### 3.3 Fault site — sample-0 cascade trace (instrumented real BIN1 body, ff)

Instrumented the actual `BIN1` body (not a reimplementation) with per-stage `Kokkos::printf`,
built at ff, dispatched via `--sample-list`. Sample 0 = nan, sample 2 = finite:

```
# SAMPLE 0 (nan):
TRACE |a|=2.12e+08 |b|=5.13e+09 |c|=4.11e+09 |discarg|=-nan discarg_re=2.983217e+19 |disc|=-nan
                                              ^^^^^^^^^^^^^^ abs/sqrt of a FINITE 2.98e19 -> nan
TRACE |x4[0]|=-nan |x4[1]|=-nan ...          # everything downstream of disc is nan
RES,BIN1,0 -> nan

# SAMPLE 2 (finite):
TRACE |a|=4.71e+08 |b|=3.16e+07 |c|=2.57e+09 |discarg|=4.837094e+18 discarg_re=-4.837094e+18 |disc|=2.199340e+09
                                              ^^^^^^^^^^^^^^ 4.84e18 < threshold -> fine
RES,BIN1,2 -> finite (0x3db2...)
```

The discriminant argument `discarg = b² − 4ac` is a **finite** complex number
(`Real ≈ 2.98e19` at sample 0), but `|discarg|` (its `abs`) and `disc = sqrt(discarg)` come
back **nan**. The fault op is the complex `abs`/`sqrt`, not the arithmetic that built discarg.

### 3.4 Why abs/sqrt overflow — the primitive

`third_party/include/ff_complex.hpp`:

```cpp
KOKKOS_INLINE_FUNCTION ffloat abs(ffcomplex z) {          // line 106-108
    return sqrt(ffadd(ffmul(z.re, z.re), ffmul(z.im, z.im)));   // re² + im², NO scaling
}
KOKKOS_INLINE_FUNCTION ffcomplex sqrt(ffcomplex z) {     // line 116-118
    ...
    ffloat r = sqrt(ffadd(ffmul(z.re, z.re), ffmul(z.im, z.im)));  // same naive re²+im²
```

`ffmul(z.re, z.re)` multiplies the hi limbs as `float`. `float`'s max is `FLT_MAX = 3.40e38`,
so any operand with `|re| > √FLT_MAX ≈ 1.844e19` overflows to `+Inf`; then `sqrt(Inf ± …)`
and the subsequent `Inf − Inf` / `Inf / Inf` yield **nan**. Confirmed numerically:

```
sqrt(FLT_MAX) = 1.844674e+19          # the exact overflow threshold
discarg_re = 2.983e19  -> (float)²  = inf   (sample 0, nan)   2.98e19 > 1.84e19
discarg_re = 4.837e18  -> (float)²  = 2.34e37 (finite)        4.84e18 < 1.84e19  (sample 2)
```

### 3.5 Why dd survives — range, not algorithm

`dd_complex.hpp:111,123` uses the **identical** naive `re²+im²` form (dd did not implement a
smarter hypot). It survives purely because `double`'s `√DBL_MAX ≈ 1.34e154` sits far above
these ~1e19 magnitudes. So this is a genuine **ff-range bug exposed by the narrower float
exponent**, not an algorithm dd got right and ff got wrong.

**Hand-back (STOP #HHH, category 1):** the fix is a **hypot-style scaled magnitude** in
`ff_complex.hpp` `abs` and `sqrt` — factor out the larger limb before squaring:

```cpp
// abs: instead of sqrt(re*re + im*im), scale by the max component
ffloat ax = abs(z.re), ay = abs(z.im);
ffloat mx = (ax.hi >= ay.hi) ? ax : ay;
if (mx.hi == 0.0f) return ffloat(0.0f);
ffloat rx = ffdiv(ax, mx), ry = ffdiv(ay, mx);
return ffmul(mx, sqrt(ffadd(ffmul(rx,rx), ffmul(ry,ry))));   // mx * sqrt((ax/mx)²+(ay/mx)²)
```

and the analogous scaling inside complex `sqrt`. This lifts the overflow ceiling from
√FLT_MAX (1.8e19) to ~FLT_MAX (3.4e38), covering the BIN kinematic regime. This is a
`third_party/` primitive change → **Reet's call**, not pushed here.

**Note:** even with the abs/sqrt fix, BIN1/BIN2/BIN4 ff would deliver at best ~8–10 digits
(their dd-baseline sits at 8.8–9.6 with heavy cancellation); whether that clears the 7.0 bar
is a separate measurement to re-run *after* the primitive fix lands. This diagnosis does not
pre-judge the routing — it only explains the nan.

---

## 4. Final routing table (unchanged)

| tier | integrals | note |
|---|---|---|
| **dd** (4) | B14, B15, B16, BIN0 | unchanged |
| **ff** (13) | B1–B9, B11, B10, B12, B13 | unchanged |
| **double** (4) | BIN1, BIN2, BIN3, BIN4 | BIN4 now *builds* at ff but nans; BIN3 still build-fails; all four stay double |

No routing change: the int-ctor fix made BIN4 *build* but the nan keeps it on double, and BIN3
still can't build. Routing moves only if the §3.5 abs/sqrt fix **and** the §2 `operator int()`
fix both land and the re-measured ff digits then clear 7.0.

---

## 5. New / updated STOPs

- **STOP #III** — *partially retired.* Cleared for **B4m/BIN4** (int-ctor fix worked).
  **Persists for B3m/BIN3** with a *different* signature: `ffloat→int` conversion (needs
  `operator int()` in `ff_math.hpp`, §2). Documented, not fixed here.
- **STOP #HHH** — *diagnosed, category 1.* Root cause = unscaled `re²+im²` in `ff_complex.hpp`
  `abs`/`sqrt` overflowing float above √FLT_MAX ≈ 1.84e19 (§3). Hand-back spec in §3.5. Not
  fixed here (diagnose-only round; `third_party/` change is Reet's).

No brand-new STOPs beyond these two hand-backs.

---

## 6. Regression gates

- `third_party/` + `runs/qcdloop_headers_full/` (snapshot): **pristine** (git porcelain clean).
  All diagnostic instrumentation ran on `/tmp` copies.
- No pipeline code moved → full suite unchanged from the prior green run (971 pass / 1 env-only
  GCP-IAM fail). flip_gate anchor re-run: **18/18**.
- `_min_digits` p100 (min-over-samples/components) reduction untouched — the nan diagnosis is
  *why* p100 collapses to nan (one nan sample pins the min).

## 7. Artifacts

- Re-run data: `runs/qcdloop/phase2_bin_ff_out/phase2_lmeasure.json`,
  `runs/qcdloop/phase2_bin_ff_out/flip_build_B3m_ff.log` (BIN3 build error).
- Diagnosis instrumentation (ephemeral, `/tmp/bin_diag2/`): instrumented `box/B1m.h` +
  `drv_ff.cpp`; sample-0/2 cascade traces reproduced in §3.3.
