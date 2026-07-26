# Falsification probe — leaf-callee clone-and-promote (LEAF_CALLEE_PROMOTION_DESIGN §7)

Subtask-5-style single-TU probe against the **real** headers + **real** Kokkos.
Where Subtask 5 tested a *forwarding overload* (and found it self-recurses), this
probe tests the DESIGN's alternative: a **clone-and-rename promoted frame**
`Lnrat_B10` whose body computes in dd and never names `ql::Lnrat`.

```sh
module use /soft/modulefiles && module load gcc/13.3.0
P=runs/qcdloop/tier_b_stage2_leaf_promotion/probe_evidence
# Build A — vendored-only surface (mirrors what the pipeline has):
g++ -std=c++20 -w -Isrc -Ithird_party/include -I~/kokkos-install/include \
    $P/probe_clone.cpp -L~/kokkos-install/lib64 -lkokkoscore -lkokkoscontainers -ldl -o /tmp/A
# Build B — + hand-written dd support-surface overlay (-DWITH_OVERLAY):
g++ -std=c++20 -w -DWITH_OVERLAY -Isrc -Ithird_party/include -I~/kokkos-install/include \
    $P/probe_clone.cpp -L~/kokkos-install/lib64 -lkokkoscore -lkokkoscontainers -ldl -o /tmp/B && /tmp/B
```

## Results

| build | surface | compile | runtime |
|-------|---------|---------|---------|
| **A** | vendored-only (`third_party/include` = `quad::ddfun` math + `dd_math`/`dd_complex`), same as pipeline | **FAIL** (5 errors) | — |
| **B** | A **+** overlay: `ql::kAbs/kLog/Real/Imag/Sign(dd…)` + `Constants<ddouble>` | **OK** | **runs, exit 0 — NO segfault** on `(v=1.5, x=2.5)`, the exact inputs that segfaulted the Subtask-5 forwarding overload |

Build-A distinct errors (the missing support surface, verbatim):

```
error: no matching function for call to 'abs(const quad::ddfun::ddouble&)'   # ql::kAbs -> Kokkos::abs, no dd overload
error: no matching function for call to 'log(const quad::ddfun::ddouble&)'   # ql::kLog -> Kokkos::log, no dd overload
error: no matching function for call to 'Sign(quad::ddfun::ddouble)'         # ql::Sign, double/complex<double> only
error: no type named 'type' in 'struct std::enable_if<false, double>'        # Constants<ddouble> primary mis-instantiates
```

Build-B run output:

```
Lnrat_B10 dd re.hi = -0.51082562376599072   double re = -0.51082562376599072   |diff| = 0.000e+00
```

## What this proves / disproves

1. **Rename discipline WORKS (Q1 confirmed).** `Lnrat_B10`'s body names only
   `ql::kLog/kAbs/Sign/Constants` and vendored dd ops — never `Lnrat_B10` or
   `ql::Lnrat`. There is no self-call, so the Subtask-5 recursion pit
   (`ddouble,ddouble` args re-selecting the same overload) **cannot arise**. The
   clone builds and runs to completion on the segfault inputs. The design's core
   escape from Subtask 5 is empirically sound.

2. **The blocker is the SUPPORT SURFACE, not the clone (category (d)).** Build A
   pins it: promoting `Lnrat`'s body to dd requires `ql::kAbs/kLog/Real/Imag/Sign`
   and `Constants<TScale>` at `TScale = ddouble` — none of which the pipeline's
   vendored surface provides. `quad::ddfun` supplies `abs/log/...` but `ql::kAbs`
   wraps `Kokkos::abs` (no dd overload) and `ql::Sign/Real/Imag` are hard overloads
   on `double`/`Kokkos::complex<double>` only (`src/kokkosMaths.h:312-334`). This
   surface **does** exist upstream — `qcdloop@ddfun_enabled:src/qcdloop/`
   `kokkosMaths_dd.h` provides every one of them in `ql::ddfun` — but it is **not
   vendored** into `third_party/include`. That un-vendored helper layer is the
   category-(d) work the design must fund before any leaf clone can compile.

3. **Correctness on this branch is trivial-exact** because `Lnrat`'s `TScale`
   overload is a straight-line `log|·| − i·π/2·(sign−sign)` with no cancellation;
   dd and double agree to all double digits (`|diff| = 0`). The probe therefore
   confirms *compilability + termination*, not a precision lift — the lift is
   B10's `Li2omx2`/cancellation story, measured only at a full e2e re-run.

The overlay in `probe_clone.cpp` (`#ifdef WITH_OVERLAY`, ~30 LOC) is a hand-written
stand-in for the un-vendored surface; its size and content are the design's §3
support-surface bill of materials in miniature.
