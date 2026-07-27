# Falsification probes — leaf-callee clone-and-promote (LEAF_CALLEE_PROMOTION_DESIGN §7)

Subtask-5-style single-TU probes against the **real** headers + **real** Kokkos.
Where Subtask 5 tested a *forwarding overload* (and found it self-recurses), these
probes test the DESIGN's alternative: a **clone-and-rename promoted frame** `Lnrat_B10`
whose body computes in dd and never names `ql::Lnrat`.

**v2 (2026-07-27) added P2/P3/P4** to enforce the architectural constraint that the
pipeline SYNTHESIZES qcdloop-specific dd support (Class 1) or uses SOURCE-resident data
(Class 2) — it does **not** vendor a hand-ported `dd_ql_support.hpp` (v1's §3.4, rejected).

```sh
module use /soft/modulefiles && module load gcc/13.3.0
P=runs/qcdloop/tier_b_stage2_leaf_promotion/probe_evidence
KI=~/kokkos-install
# P1 (v1) — clone vs forwarding, hand overlay:
g++ -std=c++20 -w            -Isrc -Ithird_party/include -I$KI/include $P/probe_clone.cpp       -L$KI/lib64 -lkokkoscore -lkokkoscontainers -ldl -o /tmp/A     # FAIL (5 err)
g++ -std=c++20 -w -DWITH_OVERLAY -Isrc -Ithird_party/include -I$KI/include $P/probe_clone.cpp   -L$KI/lib64 -lkokkoscore -lkokkoscontainers -ldl -o /tmp/B && /tmp/B
# P2 (v2) — overlay is what the PIPELINE would SYNTHESIZE (Class-1 wrappers, no Constants):
g++ -std=c++20 -w -DWITH_SYNTH -Isrc -Ithird_party/include -I$KI/include $P/probe_clone_synth.cpp -L$KI/lib64 -lkokkoscore -lkokkoscontainers -ldl -o /tmp/Bs && /tmp/Bs
# P3 (v2) — Class-2 source instantiation (Constants<ddouble> from source primary):
g++ -std=c++20 -w -Isrc -Ithird_party/include -I$KI/include $P/probe_constants_dd.cpp -L$KI/lib64 -lkokkoscore -lkokkoscontainers -ldl -o /tmp/pC && /tmp/pC
# P4 (v2) — Option-B lift ceiling (pure dd Clenshaw, no Kokkos):
g++ -std=c++17 -O2 $P/probe_optionB_ceiling.cpp -o /tmp/ceil && /tmp/ceil
```

## Results

| probe | question | result |
|-------|----------|--------|
| **P1-A** | vendored-only surface (pipeline today) compiles clone? | **FAIL** (5 errors) |
| **P1-B** | + v1 hand overlay | **OK**, runs exit 0, no segfault on `(1.5,2.5)` |
| **P2-A_synth** | vendored-only | **FAIL** (5 errors) |
| **P2-B_synth** | + **Class-1 synthesized overlay only** (mechanical wrappers, NO hand-written Constants) | **OK**, `\|diff\|=0.000e+00` vs double |
| **P3** | does source `Constants<ddouble>` instantiate at dd? | **YES** — `num_C=19`, `_C` promotes to dd, `sum_C.hi=0.8224670334241132` |
| **P4** | how much lift does Option-B (19-coeff dd) buy? | roundoff removed ~1e-16→~1e-32; **truncation floor ~1e-16 dd can't reduce** |

P1/P2 build-A distinct errors (the Class-1 gap, verbatim):

```
error: no matching function for call to 'abs(const quad::ddfun::ddouble&)'   # ql::kAbs -> Kokkos::abs, no dd overload
error: no matching function for call to 'log(const quad::ddfun::ddouble&)'   # ql::kLog -> Kokkos::log, no dd overload
error: no matching function for call to 'Sign(quad::ddfun::ddouble)'         # ql::Sign, double/complex<double> only
error: no type named 'type' in 'struct std::enable_if<false, double>'        # from ql::kLog -> Kokkos::log (NOT Constants)
```

P4 output:

```
max |dd19 - double| over battery       = 1.110e-16   (roundoff dd buys back)
dd recurrence residual |lo/hi| @Y=0.55  = 1.037e-18   (dd carries ~18 extra digits)
19-term truncation floor ~ |C[18]|      = 1.000e-16   (dd CANNOT reduce — needs 43 coeffs)
```

## What the v2 probes prove / disprove

1. **Rename discipline WORKS (P1/P2).** `Lnrat_B10`'s body names only `ql::kLog/kAbs/Sign/
   Real/Imag/Constants` + vendored dd ops — never `Lnrat_B10`/`ql::Lnrat`. No self-call, so
   the Subtask-5 recursion pit cannot arise. Builds + runs on the segfault inputs under both
   overlays.

2. **The support surface is PIPELINE-SYNTHESIZABLE, not vendor-only (P2, the v2 correction).**
   Build B_synth clears the entire Class-1 gap with **mechanical overloads** — each a
   namespace redirect (`Kokkos::abs`→`quad::ddfun::abs`) or member accessor (`.real()`) or
   scalar-expr re-emit — that the extended Gap-A machinery (`regional.py`) can generate from
   each wrapper's own one-line primary + the vendored `quad::ddfun` surface. **No hand-written
   `Constants`, no vendored `dd_ql_support.hpp`.**

3. **Class-2 coefficient table is SOURCE-RESIDENT for Option B (P3).** The source
   `Constants<T>` primary instantiates at `T=ddouble` directly (19 coeffs promote to
   `make_dd(bits,0)`). So Option B needs zero synthesis. P3 also pins that the build-A
   `enable_if` error comes from `ql::kLog`→`Kokkos::log` (Class-1), **not** the coefficient
   table — v1's implication that `Constants<ddouble>` was a compile blocker was wrong.

4. **Option-B lift is BOUNDED, not promised (P4).** dd removes the Clenshaw recurrence
   roundoff (~1e-16 → ~1e-32) but not the 19-term series truncation (~1e-16, a coeff
   property). Since the two `dilog4−dilog5` calls share (mostly) correlated truncation but
   uncorrelated roundoff, dd recovers the part that survives the difference → design predicts
   **B10 lift +8…+16**, falsifier = measured lift < +8 (→ fund Option A / STOP #O).

5. **The probes do NOT prove a lift.** `Lnrat`'s TScale branch has no cancellation → dd==double
   (`|diff|=0`). The probes confirm *compilability + termination + a bounded ceiling*; the
   lift is B10's cancellation story, measured only at a full e2e run.

## Files

* `probe_clone.cpp` / `build_A.err` — P1 (v1, retained).
* `probe_clone_synth.cpp` / `build_A_synth.err` / `build_B_synth.err` — P2 (v2, synthesized overlay).
* `probe_constants_dd.cpp` / `probe_constants_dd.out` — P3 (v2, Class-2 source instantiation).
* `probe_optionB_ceiling.cpp` / `probe_optionB_ceiling.out` — P4 (v2, Option-B ceiling).
