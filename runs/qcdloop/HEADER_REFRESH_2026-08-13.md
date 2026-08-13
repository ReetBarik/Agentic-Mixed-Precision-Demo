# Header refresh — DD/FF vendored headers → kokkos-EP@5ae2f80, + rename to `Kokkos::Experimental`

**Date:** 2026-08-13 · **Branch:** `langgraph-agents` · **Commits:** `6742dd8` (T1) → `7acefd6` (T4) + T6

## Upstream source

| field | value |
|---|---|
| repo | `https://github.com/ReetBarik/kokkos-extended-precision-demo` |
| branch | `main` |
| sha | `5ae2f800162ef4486c121e21301ff0e81615a45c` |
| date | 2026-08-09 20:26:05 -0500 |
| files | `dd_math.hpp`, `dd_complex.hpp`, `ff_math.hpp`, `ff_complex.hpp` |

`qf_math.hpp` / `qf_complex.hpp` exist upstream and were deliberately **not** vendored.
Provenance and the full local-patch inventory live in `third_party/include/UPSTREAM.sha`.

## Verdict

**Routing is bit-identical to the `c6f1f20` baseline, and not one of the 42
(integral, target) `candidate_digits` pairs moved.** The refresh is numerically
inert on this workload.

That is a stronger null than expected — the plan anticipated FF drift on B9/B10/B11.
Because "no change" is also what a silently-ineffective refresh looks like, the null
was checked three independent ways before being accepted (see below).

No build failures, no all-integrals drift, no p100 collapse. Nothing meets the STOP criteria.

## tu_routing diff

Baseline: `runs/qcdloop/PHASE_2_TU_E2E_REFSCALE_2026-07-30_report.json` (`c6f1f20`, tol 7.0).
New: `runs/qcdloop/strategy/20260813_053248_3c0ffa83/report.json` (tol 7.0, 5000 samples,
seed 12345) — the authoritative run, with the DD oracle sourced from
`third_party/include` (see below). An earlier run with the oracle still on the fork's
copies (`20260813_051335_3e3fe3bc`) gives identical numbers.

**Every integral: unchanged.** `ff=14 · dd=3 · double=4`, same per-integral assignment.

```
B1..B11,B13,B14 -> ff     B12,B15,B16 -> dd     BIN0,1,2,4 -> double     BIN3 -> ff
```

> **Tolerance note.** The plan specified `--tolerance 10`, but the baseline was
> measured at `7.0`. Running at 10 changes routing for reasons unrelated to the
> refresh, so the comparison above is at **7.0** — matching the baseline. The
> `--tolerance 10` run was also executed (`20260813_051103_b09967e7`) and gives
> `ff=4 · double=8 · dd=9`. That spread is the higher bar, not the headers, and
> must not be read as a refresh signal.

## candidate_digits diff

No cell moved by more than 5e-7. A single value means old == new.

| integral | baseline digits | dd candidate | ff candidate | route |
|---|---|---|---|---|
| B1 | 11.8081 | — | 9.2642 | `ff` |
| B2 | 12.1421 | — | 10.0454 | `ff` |
| B3 | 12.2715 | — | 9.5024 | `ff` |
| B4 | 10.2498 | — | 8.4233 | `ff` |
| B5 | 11.5853 | — | 9.0449 | `ff` |
| B6 | 12.2693 | — | 10.1049 | `ff` |
| B7 | 11.6264 | — | 10.1825 | `ff` |
| B8 | 10.1387 | — | 8.5928 | `ff` |
| B9 | 11.5301 | — | 8.6415 | `ff` |
| B10 | 10.0927 | — | 7.8914 | `ff` |
| B11 | 9.4597 | — | 7.7693 | `ff` |
| B12 | 3.6906 | 14.3311 | 2.4065 | `dd` |
| B13 | 8.5777 | — | 7.2692 | `ff` |
| B14 | 13.0380 | — | 10.8214 | `ff` |
| B15 | 0.0000 | 0.0000 | 0.0000 | `dd` |
| B16 | 6.5636 | 12.5508 | 5.0989 | `dd` |
| BIN0 | 0.0000 | 0.0000 | 0.0000 | `double` |
| BIN1 | 8.0683 | — | 0.0000 | `double` |
| BIN2 | 9.3825 | — | 0.0000 | `double` |
| BIN3 | 9.1947 | — | 7.4874 | `ff` |
| BIN4 | 9.0380 | — | 0.0000 | `double` |

`—` = no dd candidate produced (`tu_no_flip_needed`: raw double already clears the bar).
The `0.0000` cells on B15 / BIN0-2 / BIN4 are the previously-documented analytic-zero
metric artifact, unchanged by this refresh.

## Why the null is real

A refresh that never reached the compiler would produce exactly this table, so:

1. **The candidate builds provably use the refreshed headers.** The headers repo's
   `kokkosMaths_dd.h` now opens a real `namespace ql::ddfun` over
   `Kokkos::Experimental`. The pre-refresh one aliased `= ::quad::ddfun`, which cannot
   compile once T4 deleted the compat shim. The dd flip built anyway (B12 dd =
   14.3311), so the new alias and the refreshed `third_party/include` headers are both
   in the build.

2. **Direct numeric comparison, old headers vs new.** The same program compiled twice —
   once against the `c6f1f20` headers, once against the refreshed ones — over `log`,
   `exp`, `sqrt` and complex `abs` for both DD and FF across 40 inputs:
   **320 of 320 values bitwise identical.** Sampling covers those four functions on
   positive reals; it does not exclude drift in untested routines or edge ranges.

3. **Normalised source diff.** After mapping every rename (`ddadd`→`add`,
   `ffmulf`→`multiply_scalar`, `ddnint`→`round_to_nearest_int`, `make_dd`→`from_bits`, …)
   and stripping comments/whitespace, the complex headers reduce to namespace lines plus
   upstream's **new ADL forwarding overloads in `namespace Kokkos`** — pure additions.
   Upstream's change is an API reorganisation, not a numerics change, for everything
   this workload touches.

## The DD oracle now tracks `third_party/include` (premise corrected, then fixed)

The plan stated that refreshing `third_party/include/` refreshes the DD oracle in one
shot. **It did not.** `runs/qcdloop/app/CMakeLists.txt` orders includes `QL_HEADERS`
*before* `_vendored_include`, and the oracle builds with
`QL_HEADERS=~/qcdloop/src/qcdloop@ddfun_enabled`, which ships its **own**
`dd_math.hpp`, `dd_complex.hpp`, `ff_math.hpp`, `ff_complex.hpp` under
`namespace ql::ddfun`. Those shadowed the vendored copies, so the first refresh moved
the candidate builds but left the oracle on the fork's frozen headers.

Include order alone could not fix it: `kokkosMaths_dd.h` reaches the primitives with a
**quoted** `#include "dd_math.hpp"`, and a quoted include searches the includer's own
directory before any `-I` path. Reordering `_vendored_include` first changes nothing.

**Fix — `agents/validator/runner.stage_dd_headers()`.** After the tree is
`git archive`d into a throwaway staging dir (never `~/qcdloop` itself), it:

1. deletes the four shadowing primitives, so the quoted include falls through to
   `-I third_party/include`; then
2. injects the `ql::ddfun` alias namespace over `Kokkos::Experimental` — necessary
   because the fork authors `ql::ddfun` natively *inside* its own `dd_math.hpp`, so
   removing that file removes the namespace with it.

Injected rather than overwriting `kokkosMaths_dd.h` wholesale, so future fork-side
changes (new Chebyshev tables, tolerances) survive. Idempotent, and it raises rather
than guessing if the fork's include layout changes. Wired into all four stagers:
`validator/validate.py`, `tu_provider.py`, `phase1_lmeasure.py`, `phase2_lmeasure.py`.

| build | `QL_HEADERS` | resolves dd/ff primitives from |
|---|---|---|
| DD oracle | staged `~/qcdloop/src/qcdloop` (primitives stripped) | **`third_party/include`** |
| vanilla + candidate flips | headers repo (no dd/ff primitives) | `third_party/include` |

**The switch is numerically inert.** Old oracle binary vs new, 5000 samples:
**105,000 / 105,000 output lines bitwise identical.** That is the good outcome — the
oracle keeps the exact ground truth the routing baseline was measured against, while
now actually tracking the vendored headers, so the next refresh cannot silently miss
it. (It holds because `~/qcdloop`'s `dd_complex.hpp` already carried the hypot `abs`,
so both sources agree.)

## Local patch set

Upstream carries **none** of the Agentic-local patches. A verbatim copy alone does not
compile (upstream removed `operator int()`, and never had the unary `+`, the
`FloatFloat(int)` ctor, or the scalar comparisons) and would revert the hypot `abs`
that the routing baseline is measured against. All were re-applied on top, each marked
in-file `// LOCAL PATCH (<sha>): … — not upstream`.

| sha | file(s) | what |
|---|---|---|
| `44c1ec4` | `dd_math`, `dd_complex` | unary `operator+()`, `operator int()`, 6 scalar-double comparisons; unary `operator+()` on `DoubleDoubleComplex` |
| `3ab4aa6` | `ff_math` | unary `operator+()` |
| `4f21245` | `ff_math` | explicit `FloatFloat(int)` ctor |
| `45197be` | `ff_math`, `dd_complex`, `ff_complex` | `FloatFloat operator int()`; hypot-style overflow-safe complex `abs()` + complex `sqrt()` reusing it |
| `f12d8bf` | `ff_math` | 6 scalar-float comparisons |

Persistent delta vs pristine upstream: `dd_math +20/-0`, `dd_complex +33/-2`,
`ff_math +26/-0`, `ff_complex +28/-2`.

> **One was missed in T2 and caught during T6 verification.** The normalised source
> diff surfaced `DoubleDoubleComplex operator+()` as present pre-refresh, absent
> upstream, and absent from my re-application — a sixth patch, not five. It was
> restored before the final run. It is an identity operator, so it cannot affect any
> number in the table above, but `shim_normalise.py` explicitly handles unary `+` on an
> extended operand, so its absence would have been a latent build failure. The lesson:
> the T2 inventory was built from `git log` over the vendored paths, which attributed
> the patch to a commit whose *headline* was about something else. A normalised
> old-vs-new diff is the reliable instrument and belongs in T2 next time.

## Verification summary

| check | result |
|---|---|
| `ql::` surface compiles (`kokkosMaths_dd.h`, `kokkosMaths_ff.h`) | pass — both alias namespaces, all 68 `make_dd`/`make_ff` sites, `Constants<T>` tables |
| local patches compile | pass — probe exercises every re-applied block |
| old `quad::` spelling rejected post-T4 | pass — `'quad' has not been declared` |
| numeric equivalence, old vs new headers | 320/320 values bitwise identical |
| DD oracle: fork-sourced vs `third_party`-sourced | 105,000/105,000 output lines bitwise identical |
| pytest (patcher/strategy/validator/dd_/ff_/chain) | **581 passed, 0 failed** |
| tu_only e2e, tol 7.0 | routing identical, 0/42 digit pairs drifted |
| tu_only e2e re-run with oracle on `third_party/include` | routing identical, 0/42 drifted |

### T5 straggler grep

530 raw hits; every one accounted for:

| category | count | disposition |
|---|---|---|
| `runs/archive/` | 68 | frozen |
| `tier_b_stage2_*` | 194 | frozen |
| `*.log`, `*.csv` | 75 | frozen |
| dated reports | 51 | frozen — a 2026-07-29 report must not claim an API that postdates it |
| `ql::ddfun::make_dd(` / `ql::ffun::make_ff(` | 137 | **by design** — the alias surface, kept verbatim |
| `instantiation_gate.py` `_LEGACY` constants | 3 | **by design** — parses archived build logs |
| rename-script docstring | 2 | **by design** |

Genuine stragglers: **zero.** No T5 fix commit was needed.

One narrow exception to the `tier_b_stage2` freeze was taken:
`probe_evidence/probe_constants_dd43.cpp` is *compiled by a live test*, so the freeze's
"not built by CI" rationale does not cover it. Its two `quad::ddfun` tokens were
retargeted at the `ql::ddfun` surface it actually exercises. It is the only such file.

## Known issues, not introduced here

* **Real-LLM Rule-3 misgeneration flake.** `tests/dd_integrator` real-LLM tests
  intermittently emit `complex<DoubleDouble>` instead of the vendored container.
  Measured over 5 runs × 6 tests per tree: baseline `c6f1f20` **3/30**, refreshed
  **4/30**, with 3 of 5 runs affected on both. Pre-existing; the rename does not
  measurably change the rate, and an earlier apparent increase was small-sample noise.
  The final full run happened to be clean at 581/581.

## Follow-ups

1. ~~Decide whether the DD oracle should track `third_party/include`~~ — **done.**
   `stage_dd_headers()` repoints it; verified bitwise-identical oracle output.
2. Consider upstreaming the six local patches so a future refresh is genuinely verbatim.
   The hypot `abs` in particular fixes a real overflow (`sqrt(FLT_MAX) ≈ 1.84e19`) that
   upstream still carries.
3. QF (`qf_math.hpp`, `qf_complex.hpp`, LADDER, `TargetPrecision.QF`) remains out of
   scope and unstarted.
