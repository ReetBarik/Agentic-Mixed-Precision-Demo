# Tier-B Stage-1 — chain-scoped dd promotion (2026-07-25)

Phase 2f coordinated whole-chain double-double promotion on the 4 measured Tier-B integrals. v1 promotes the dominant COMPUTED cascade chain per integral (one coordinated envelope).

- gate: positive lift >= 0.5 digits vs accumulated-min (chain_dd); tolerance 6.0 (reporting-only)
- seed 12345, sample_count 5000, entry BO

## Per-integral outcome (kernel-scoped gate)

The gate now scores each chain against ITS integral's own p100 floor (kernel-scope, Reet 2026-07-25), not the whole-app min pinned by the worst kernel (B12's hotspot). Whole-app columns are kept for cross-kernel visibility.

| I | kernel baseline | kernel final | kernel lift | predicted lift | app baseline | app final | outcome | chain | lines |
|---|---|---|---|---|---|---|---|---|---|
| B10 | — | — | — | +18.43 | — | — | apply_failed | cascade_B10_612f1391_494252c4 | 10 |
| B12 | — | — | — | +17.10 | — | — | apply_failed | cascade_B12_65bb39c0_62ff5a3d | 5 |
| B13 | — | — | — | +17.10 | — | — | apply_failed | cascade_B13_79fc5b8f_f080f240 | 8 |
| B14 | — | — | — | +16.66 | — | — | apply_failed | cascade_B14_3429b1d4_01bf2ff3 | 3 |

## Predicted vs measured lift (kernel-scoped)

- **B10** (cascade_B10_612f1391_494252c4): predicted +18.43, kernel-measured — (— -> —), whole-app lift —, tightness 0.003331756565344427, patcher_status=write_truncation, declared_dd=False
    - lines: B1m.h:227, B1m.h:240, B1m.h:241, kokkosUtils.h:174, kokkosUtils.h:177, kokkosUtils.h:199, kokkosUtils.h:212, kokkosUtils.h:702, kokkosUtils.h:703, kokkosUtils.h:704
- **B12** (cascade_B12_65bb39c0_62ff5a3d): predicted +17.10, kernel-measured — (— -> —), whole-app lift —, tightness 0.07075303644353668, patcher_status=llm_gen_failed, declared_dd=False
    - lines: B2m.h:206, B2m.h:207, B2m.h:241, kokkosUtils.h:212, kokkosUtils.h:702
- **B13** (cascade_B13_79fc5b8f_f080f240): predicted +17.10, kernel-measured — (— -> —), whole-app lift —, tightness 0.07080121254580928, patcher_status=write_truncation, declared_dd=False
    - lines: B2m.h:300, B2m.h:301, B2m.h:305, B2m.h:306, B2m.h:355, B2m.h:533, kokkosUtils.h:212, kokkosUtils.h:702
- **B14** (cascade_B14_3429b1d4_01bf2ff3): predicted +16.66, kernel-measured — (— -> —), whole-app lift —, tightness 0.19860180300800165, patcher_status=write_truncation, declared_dd=False
    - lines: B2m.h:401, B2m.h:578, kokkosUtils.h:1208

## What the two fixes changed vs the prior run (87be92f)

Both fixes did exactly what they were designed to do — they moved every integral off its
*prior* terminal state onto a *new, more-informative* one. None of the four reached the
acceptance gate this time, but for reasons that are now diagnostic rather than spurious:

| I | prior run (87be92f) | this run (afd334c) | interpretation |
|---|---|---|---|
| B10 | chain-scope 2d-B false-positive (OUTERMOST region) | **interior** write_truncation | Fix 1 cleared the false-positive; a genuine interior region now trips |
| B12 | chain-scope 2d-B false-positive (OUTERMOST region) | **llm_gen_failed** (build fail) | Fix 1 cleared the false-positive; chain now reaches the gen rung and fails there |
| B13 | transient Argo timeout | **interior** write_truncation | Fix 3: timeout cleared; real outcome surfaced |
| B14 | whole-app `chain_no_lift` (B12-pinned global min) | **interior** write_truncation | Fix 2 in place, but chain is gated by Fix 1 *upstream* of the acceptance gate |

Fix 2 (kernel-scope gate) is wired and unit-tested but was **not exercised end-to-end** this
run: every chain terminated at the Patcher (apply_failed) before a candidate reached the
solver's acceptance gate, so no `kernel_baseline`/`kernel_final`/`kernel_lift` was ever
measured (all `—` above). The kernel-scope path only runs once a chain builds and validates.

## Root causes (two open blockers for Reet — flag-and-stop)

**Blocker A — interior write_truncation is a chain-EMISSION completeness limit, not a precision result (B10/B13/B14).**
The interior gate fires because `chain_promote` widens only the chain's listed region
*lines*, not the carrier *declarations* that thread values between links. Concretely on B14:

- `B2m.h:401` writes `fac`, declared `TOutput fac;` at `B2m.h:396` — OUTSIDE the promoted
  line set, so it stays caller-precision → the interior write demotes (Case-B landing).
- `B2m.h:578` writes `Y[1][3]=Y[3][1]`, a `Kokkos::Array<...,TMass,...>` parameter — a
  caller-precision carrier the chain never widens.
- `kokkosUtils.h:1208` writes `res[0]`, a `TOutput` array — same class.

So the gate's *local* verdict is arguably correct for the patch **as emitted** (this exact
region really does round back to double), but the *conclusion* it forces — "the chain is
inert / breaks" — is wrong: the chain would carry precision if the shared carriers
(`fac`, `Y[][]`, `res[]`) were promoted to dd along with the region lines. This is the
chain-scope analogue of the outermost-region issue Fix 1 fixed: the per-region 2d-B
detector cannot see a *cross-link* sink because the sink is a declaration the chain-emission
step leaves at double. **The fix belongs in chain emission (widen carrier decls that are
written by one interior link and read by another), NOT in the gate** — do not weaken the
interior gate, which correctly rejects the currently-emitted (truncating) patch. Suggested
Stage-2 work item: `chain_promote` should collect writes across all interior links and
promote the enclosing declaration of any carrier that is both written and read within the
chain envelope. Until then B10/B13/B14 are correctly *not accepted* (the emitted patch is
genuinely lossy), just for an emission reason, not a numerical one.

**Blocker B — B12 chain hits a Patcher/LLM gen-robustness failure (`llm_gen_failed`), unrelated to precision.**
Build error (`B2m.h:768-771`): the LLM re-declared already-promoted locals
(`redeclaration of 'quad::ddfun::ddouble p3sq__ff'`, `m3sq__ff`, `m4sq__ff`) and emitted a
malformed unary `+` (`no match for 'operator+' (operand type is 'ddouble')`, 1 arg to a
2-arg operator). Pure code-generation defect in the dd rung on this chain — same class as
the residual gen gaps tracked in the 10k waves, not a gate or precision issue. Fix 1 is
what let B12 reach this rung at all (previously masked by the outermost false-positive).

## Notes
- Kernel-scope gate (Reet 2026-07-25): each chain gated against its own integral's p100 floor, not the whole-app min (which B12's hotspot pins). Wired + unit-tested; not exercised e2e this run (all chains failed at the Patcher upstream of the gate).
- Chain-scope 2d-B (Fix 1): the gate now fires only on INTERIOR chain regions; the outermost region's exit-truncation is the designed output boundary and is exempt (was false-positiving B10/B12 pre-build). Confirmed working — B10/B12 moved off the outermost false-positive onto genuine interior/gen outcomes.
- STOP after Stage-1 for review; Group B / all-21 not run.
- v1 = dominant chain per integral; multi-chain union deferred to Stage-2.
- Two open blockers above (A: carrier-decl promotion in chain emission; B: B12 dd-rung gen robustness) are for Reet to triage before Stage-2 / all-21.
