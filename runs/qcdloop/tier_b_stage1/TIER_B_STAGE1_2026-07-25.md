# Tier-B Stage-1 — chain-scoped dd promotion (2026-07-25)

Phase 2f coordinated whole-chain double-double promotion on the Tier-B integrals.
v1 promotes the dominant COMPUTED cascade chain per integral (one coordinated
envelope). This document records the **Blocker A carrier-widening re-run**
(Subtask 5, @e1971d0) on **B10 / B13 / B14** (B12 **skipped** — its separate
gen defect, Blocker B, is out of scope for this handback).

- gate: positive lift >= 0.5 digits vs accumulated-min (chain_dd); tolerance 6.0 (reporting-only)
- seed 12345, sample_count 5000, entry BO, kernel-scope gate
- run log: `runs/qcdloop/tierb_stage1_subtask5_rerun.log`

## Per-integral outcome (kernel-scoped gate)

| I | outcome | patcher_status | solve_wall | predicted lift | chain | lines |
|---|---|---|---|---|---|---|
| B10 | apply_failed | write_truncation (**new, deeper seam**) | 286.9s | +18.43 | cascade_B10_612f1391_494252c4 | 10 |
| B13 | apply_failed | write_truncation (confirmed non-carrier) | 76.6s | +17.10 | cascade_B13_79fc5b8f_f080f240 | 8 |
| B14 | apply_failed | write_truncation (confirmed non-carrier) | 117.7s | +16.66 | cascade_B14_3429b1d4_01bf2ff3 | 3 |

No integral reached the acceptance gate — but per the design success criterion
(BLOCKER_A_CARRIER_DESIGN.md §12, *"NOT all 3 accept"*), each now reaches a
**real, well-diagnosed terminal state**. The carrier fix removed the *spurious*
`write_truncation` rejection wherever a strict carrier existed; the residual
`write_truncation` verdicts are now genuine emission-completeness results, and
one of them (B10) is a **newly-surfaced, deeper** seam the carrier fix exposed.

## What the carrier fix changed — the headline result (B10)

**The carrier fix works.** B10's chain carries the blocker's worked example:
`ddilog`'s `TMass Y, S, A;` at `kokkosUtils.h:157`, where `Y` (write :174 / read
:199) and `A` (write :177 / read :212) are strict carriers. The re-run:

- **Carrier closure fired correctly**: `carrier_names = {Y, S, A}` (the whole
  multi-declarator, §2), `chain_carrier_unwidenable=False`,
  `chain_carrier_external=False`.
- **The decl was widened in the emitted variant** and the original left intact
  (direct re-emission check on a fresh clone):
  - ORIGINAL `ddilog`: `TMass Y, S, A;` — untouched.
  - VARIANT `ddilog_*_B10`: `quad::ddfun::ddouble Y, S, A;` — widened.
- **The Y/A interior seam no longer trips.** Per-region 2d-B with
  `carrier_names={Y,S,A}` reports `kokkosUtils.h:174` (writes Y) and `:177`
  (writes A) both **not inert** — the exact spurious rejections removed.
- **Behavioral proof**: the chain moved from `105.8s` (prior run, terminated at
  the Y/A pre-build gate) to `286.9s` — it now generates all 13 dd shims and
  runs the full emission before a *different* region trips. The carrier fix
  advanced the chain past the seam it was built to clear.

### The new, deeper seam B10 now trips (`kokkosUtils.h:704`, `Li2omx2`)

With Y/S/A widened, the chain-scope interior gate now fires on a **distinct**
region one level up the call graph:

```
kokkosUtils.h:688  TOutput Li2omx2(TScale v, w, x, y) {
kokkosUtils.h:691      TOutput prod, Li2omx2;          // <-- decl OUTSIDE chain lines
...
kokkosUtils.h:704      Li2omx2 = -TOutput(... - ql::ddilog<...>(arg2)) + lnarg*lnomarg - ...;  // chain line: WRITES Li2omx2
kokkosUtils.h:707      return Li2omx2;                 // NON-chain line: reads Li2omx2
```

This is the **design §3 `fac` tension, one level deeper**: `Li2omx2` (the
function's own return accumulator, declared at :691 outside the chain line set)
is written on a chain line (:704) but read only at the **non-chain** `return`
(:707) — it fails carrier **condition 2** (not read by *another chain line*), so
Fix A correctly leaves it alone. Its fate is then decided by the existing 2d-B
machinery. Because `Li2omx2` sits at **interior** depth 3 (its callee `ddilog`
is depth 4), it is NOT covered by the outermost-region exemption, so its
caller-precision store (`Li2omx2` is `TOutput`, a recognized caller-complex
type) reads as a genuine truncating landing → interior `write_truncation`.

**Interpretation**: this is not a carrier bug and not a spurious gate — the
emitted patch really does round the dd `ddilog` result back to `TOutput` at
`:704` before the value leaves `Li2omx2` at `:707`. The chain, as currently
scoped, does not include the `return Li2omx2;` line, so `:704`'s write has no
wider persistent sink. This is a **chain-SCOPING** limit distinct from Blocker A:
the dominant-chain selector stopped the chain line set at `:704` (the write) and
did not extend it to `:707` (the return), so the function's own return value is
truncated at an interior seam.

## B13 / B14 — confirmed non-carrier write_truncation (as the closure predicted)

Both B13 and B14 have **no widenable carriers** (`carrier_names = frozenset()`,
no unwidenable/external), matching the committed closure unit tests
(`test_real_b13_has_no_carriers`, `test_real_b14_has_no_carriers`). Their
`write_truncation` is therefore **confirmed genuine**, not a carrier artifact —
we have now *proven* there is no carrier to widen:

- **B13** (`B2m.h`): the interior writes `ga34pm1/ga34m/ga43pm1/ga43m` at
  :300/:301/:305/:306 are declared at :282–283 (outside the chain lines) but read
  only at the **non-chain** lines :310–:317 via `x34* = ql::Real(ga34*)` — an
  extract-to-`double`. They are written on a chain line but not read by another
  chain line → not carriers (condition 2). The store truncates → genuine
  interior `write_truncation`.
- **B14** (`B2m.h`): `fac` at :401 — the design's canonical §3 example: declared
  `TOutput fac;` at :396 (outside the chain lines), written on chain line :401,
  read only at the non-chain output stores `res(i,1)=fac/...` / `res(i,0)=fac*...`.
  Not a carrier → genuine interior `write_truncation`.

## Summary of failure classes after Blocker A

| I | prior run (afd334c) | this run (e1971d0) | what the carrier fix proved |
|---|---|---|---|
| B10 | interior write_truncation @ Y/A carrier (:174/:177) | interior write_truncation @ `Li2omx2` return (:704), **all shims generated, 105s→287s** | carrier fix WORKS (Y/S/A widened, seam cleared); a deeper **chain-scoping** seam surfaced |
| B13 | interior write_truncation | interior write_truncation @ `ga34*` (:300/:301/:305/:306) | **confirmed non-carrier** (closure = ∅); genuine truncation, not artifact |
| B14 | interior write_truncation | interior write_truncation @ `fac` (:401) | **confirmed non-carrier** (design §3); genuine truncation, not artifact |

## Two follow-ups for Reet (flag-and-stop; STOP after Stage-1)

**Follow-up 1 — B10 chain-SCOPING (new): the dominant-chain selector truncates a
function's own return.** The residual B10 seam is `Li2omx2`'s return accumulator
written at `:704` but with the chain ending there rather than at `:707` (`return
Li2omx2;`). This is NOT a carrier (correctly left alone by Fix A) and NOT a gate
bug (the store genuinely truncates). It is a **chain line-set completeness**
question: should the dominant-chain selector, when a chain line writes a
function-local that the function then `return`s, extend the chain to cover that
return so the value stays dd until the function's own boundary? That would let
the outermost-exemption apply to the return (the designed exit of `Li2omx2`),
exactly as it does for the top-level driver store. This is a Stage-2 selector /
chain-definition refinement, orthogonal to Blocker A. Do **not** weaken the
interior gate — it correctly rejects the currently-emitted (truncating) patch.

**Follow-up 2 — B13/B14 are genuinely dd-insufficient AS CURRENTLY SCOPED.**
Their cancellation carriers (`ga34*`, `fac`) are read only at non-chain
extract/store sites, so a chain that promotes only the listed lines truncates at
the write regardless of carrier-widening. Same root shape as Follow-up 1 (chain
scope stops before the value's real consumer), but here the consumer is an output
store (`ql::Real(...)` / `res(i,k)=...`), not a `return`. If Stage-2 extends the
chain scope to the consumer, the outermost-exemption would cover it; otherwise
these remain correctly not-accepted.

## Notes
- **Carrier fix validated**: closure fires on B10 (Y/S/A → dd), decl widened in
  the emitted variant, original untouched, Y/A seam cleared, 671 tests green
  (incl. new e2e integration test asserting body-promotion + carrier-decl-widen
  and that the interior gate stays silent with carrier awareness).
- **No gate weakened.** Every `write_truncation` this run is a true
  round-back-to-double in the emitted patch; the fix only removed the *spurious*
  ones (carrier writes now land in a widened dd decl).
- **B12 skipped** (Blocker B, separate gen defect — out of scope).
- Kernel-scope path (Reet 2026-07-25) still **not exercised e2e**: every chain
  terminated at the Patcher (apply_failed) upstream of the solver acceptance
  gate, so no `kernel_baseline`/`final`/`lift` was measured (all —).
- STOP after Stage-1 for Reet review; Group B / all-21 not run. v1 = dominant
  chain per integral; multi-chain union deferred to Stage-2.
