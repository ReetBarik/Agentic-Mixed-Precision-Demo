# Tier-B Stage-2 — closure Subtask 3: π-family catalog + deterministic shim normaliser (2026-07-26)

Deterministic gen-robustness sweep (CLOSURE_SCOPED_CHAINS_DESIGN.md §6.1, §8 point 1):
the π-family constant catalog extension + the post-generation shim normaliser that
together clear the two orthogonal Patcher shim-gen defects 2b diagnosed as B10's
residual blocker (`_pi2o6` #error + `T__ff` redeclaration). No gate/closure/design
changes — rule (c) stays exactly as landed in 2b.

- gate: positive lift ≥ 0.5 digits vs accumulated-min (kernel-scope); tolerance 6.0 (reporting-only)
- seed 12345, sample_count 5000, entry BO
- run dir: `runs/qcdloop/tier_b_stage2_subtask3/`
- offline: **751 tests pass** (constant_derive +26, shim_normalise +20)

## Per-integral outcome (kernel-scoped gate)

| I | outcome | patcher_status | kernel lift | predicted | 2b baseline | Δ vs 2b |
|---|---|---|---|---|---|---|
| B10 | apply_failed | llm_gen_failed | — | +18.43 | apply_failed (llm_gen_failed) | **2b's 18/18 build errors CLEARED; new orthogonal `Lnrat` #error (LLM non-determinism)** |
| B12 | rejected | **ok** | +0.00 (3.6906→3.6906) | +17.10 | (not run in 2b) | **chain BUILDS + EXECUTES e2e (Blocker B cleared); chain_no_lift = hotspot not on dominant chain** |
| B13 | apply_failed | write_truncation | — | +17.10 | write_truncation | **byte-identical** (no regression) |
| B14 | rejected | ok (chain_no_lift) | +0.00 (13.1855→13.1855) | +16.66 | rejected/chain_no_lift | **byte-identical** (no regression) |

## Headline: the two Subtask-3 target defects are CLEARED — measured against 2b's own build log

2b's report named B10's residual blocker as two orthogonal Patcher shim-gen defects.
The build-error census, 2b vs Subtask-3, on B10's failing build:

| build error class | 2b count | Subtask-3 count | cleared by |
|---|---|---|---|
| `_pi2o6 requires manual classification` (#error) | 1 | **0** | Step 1 (π-family catalog) |
| `redeclaration of 'ddouble T__ff'` | 17 | **0** | Step 2 N1 (redeclaration drop) |
| `ql::Lnrat requires manual classification` (#error) | 0 | 1 | — (NEW, orthogonal; see STOP #F) |

**18 of 18 of 2b's B10 build errors are gone.** Concretely:

- **Step 1 works end-to-end.** B10's L177 shim now emits
  `_pi2o6 → quad::ddfun::make_dd(0x3ffa51a6625307d3ULL, 0x3c81873d8912200cULL)` —
  the bit-exact double-double split of π²/6 the new catalog computes from the
  canonical π at prec=80. The chain integrator consumed the pre-derived constant
  (Rule R3 step 2/3), never reaching the R4 #error. STOP #E did not fire: the wire-up
  `derive_from_rhs → derive_region_constants → chain-integrator prompt` delivered the
  value (`_pi() * _pio6<TOutput, TMass, TScale>()` resolved to `catalog:pi_squared_over_6`).
- **Step 2 N1 works end-to-end.** The 17 `T__ff` redeclaration errors that dominated
  2b's build log are absent from every Subtask-3 attempt: the normaliser demoted each
  same-scope repeat declaration to a plain assignment, keeping the first declaration.

## B12: the chain now BUILDS and EXECUTES end-to-end (Blocker B cleared)

B12 is the strongest positive evidence in this Subtask. Its dominant chain
(`cascade_B12_65bb39c0_62ff5a3d`: B2m.h:206/207/241, kokkosUtils.h:212/702) shares the
**same** Blocker-B defect class as B10 (it includes kokkosUtils.h:702, the `Lnrat`
region), and 2b's report explicitly rated B12 as dying at `llm_gen_failed`
(re-declared promoted locals, malformed unary `operator+`) *upstream of any scoping
question* (design §8 point 1). Under Subtask 3:

```
attempt 0: gen_failed  (llm_gen_failed)
attempt 1: build_failed
attempt 2: ok           ← chain built, no #error / no redeclaration in any log
```

B12's chain **built cleanly and executed** — the first time a B12 dd chain has
compiled through the whole envelope. It then measured `chain_no_lift`
(kernel 3.6906 → 3.6906, lift 0.0). That is **not** a gen defect and **not** a
Subtask-3 failure: B12's kernel floor is the genuine cancellation hotspot
`coeff0.imag` at sample 3868 (precise_digits 3.6906 — the exact floor
[[project_blocker_a_carrier]] / Item-7 record), and v1 promotes only the *dominant*
chain, which does not touch that coefficient's cancellation. Moving B12's floor needs
a chain that covers coeff0.imag — a chain-selection scope question, orthogonal to the
catalog/normaliser and to rule (c). B12's Blocker-B gen defect, the thing this Subtask
targeted, is cleared.

## B10: a NEW, orthogonal shim-gen defect — `ql::Lnrat` helper classification (STOP #F)

With `_pi2o6` and `T__ff` gone, B10's build now fails on a **single, different** error
across all 3 attempts:

```
ql_shim_dd.h:7: error: #error "DD Chain Integrator:
  ql::Lnrat<ddouble, TMass, TScale> requires manual classification"
  (emitted by shim kokkosUtils_dd_L702 — the `const TOutput lnarg = -ql::Lnrat(...)` region)
```

This is the SAME Patcher gen-robustness class (`_ieps50` family / Blocker B), NOT a
closure/gate/catalog/normaliser defect, and it is demonstrably **LLM
non-determinism**, not a capability gap:

- In **2b**, the L702 shim (identical SOURCE_HASH silo `f3412da7`) generated a
  **working** ddouble `Lnrat` overload: `return ::ql::Lnrat<quad::ddfun::ddouble,
  TMass, TScale>(a, b);`. The pipeline *can* produce this shim.
- In **Subtask 3**, all 3 B10 attempts instead emitted the LLM's own Rule-R4 escape
  hatch (`#error … requires manual classification`) rather than the forwarding
  overload. The L702 region reads no π-family constant, so the Step-1 catalog change
  did not alter its prompt — this is pure generation variance.
- **B12 confirms recoverability**: B12's chain hit the same `Lnrat` region and
  recovered on attempt 2 (built `ok`). B10 simply lost all three draws against the
  same defect.

The normaliser is a *source* transform and correctly does **not** strip a `#error`
(that would leave an undefined `Lnrat<ddouble>` symbol — a worse, silent failure).
Fixing the `Lnrat` #error is a chain-integrator **generation-robustness** change, which
is out of this Subtask's declared scope (STOP #F: report + propose, ask Reet before
implementing). Proposal below.

## STOP-condition audit (§ STOP-and-report)

- **STOP #A (measurement falsification)** — did NOT fire. B10 never reached measurement
  (build failed upstream at the orthogonal `Lnrat` shim gen), so "builds cleanly but
  measures chain_no_lift / lift < +8" cannot apply to B10. The closure-scoped design's
  core hypothesis is neither confirmed nor falsified by a B10 headline measurement.
  (B12 built + measured chain_no_lift, but for a chain-selection reason — its dominant
  chain does not cover the coeff0.imag hotspot — not a rule-(c) falsification.)
- **STOP #B (accept ↔ reject flip)** — did NOT fire. B13 identical
  (`apply_failed`/`write_truncation`, final.diff **byte-identical** to 2b); B14 identical
  (`rejected`/`chain_no_lift`, 13.1855 → 13.1855, final.diff **byte-identical** to 2b).
  No currently-correct rejection became a false accept; no accepting chain regressed.
- **STOP #C (catalog derivation wrong value)** — did NOT fire. Every π-family entry is
  bit-exact vs an INDEPENDENT high-precision π string (test
  `test_pi_family_dd_bit_exact_vs_independent_reference`); π²/6 = 1.6449340668482264.
  The emitted B10 shim carries `make_dd(0x3ffa51a6625307d3, 0x3c81873d8912200c)`.
- **STOP #D (normaliser breaks build)** — did NOT fire. The normaliser is semantically
  null on clean input (idempotence + clean-input tests), best-effort in dispatch (never
  fails the pass), and did not break any previously-succeeding build (B13/B14
  byte-identical; B12 built successfully with the normaliser live).
- **STOP #E (catalog fires but shim still #errors)** — did NOT fire. `_pi2o6` resolved
  and the shim emitted the derived `make_dd` — the classification path works. The
  remaining #error is a DIFFERENT symbol (`Lnrat`), not a `_pi2o6` wire-up break.
- **STOP #F (new gen-defect class)** — **FIRED.** B10 build fails on `ql::Lnrat`
  helper classification, a defect not covered by N1/N2/N3. Reported here, targeted fix
  proposed below, NOT implemented (normaliser scope not expanded).
- **STOP #G (inward-parameter demand)** — did NOT fire. The `Lnrat` shim path forwards
  to the primary template with the extended output type (an *outward* return widen, the
  same shape 2b generated successfully); it did not demand inward parameter widening on
  a shared helper.

## Verdict

Subtask 3's two declared targets are **done and proven**: the π-family catalog clears
`_pi2o6` (bit-exact π²/6, resolved end-to-end into B10's shim), and the deterministic
normaliser clears the `T__ff` redeclaration class (17 → 0 in B10's build, 0 regressions
on B13/B14). Together they removed **18 of 18** of 2b's B10 build errors and let
**B12's dd chain build and execute for the first time**.

The Group-A *measurement* B10 was meant to headline is blocked by a NEW, orthogonal
shim-gen defect (`ql::Lnrat` helper #error), which is (a) the same Blocker-B /
`_ieps50` gen-robustness class this Subtask does not expand into, and (b) demonstrably
LLM non-determinism (2b generated the working `Lnrat` overload on the identical shim
silo; B12 recovered it on attempt 2 this run). **STOP #F fired.**

**No falsification of the closure-scoped design** (STOP #A did not fire); **no
regression** (STOP #B did not fire; B13/B14 byte-identical). The headline first Group-A
measured lift is deferred one gen-robustness fix — proposed below, gated on Reet.

## Proposed targeted fix (STOP #F — NOT implemented; awaiting Reet)

The `Lnrat` shim is a **namespace-qualified math-bridge helper** whose ddouble overload
forwards to the primary `ql::Lnrat` template with the extended output type — a
mechanical, deterministic transform (exactly what 2b's shim did by hand). Three options,
smallest-blast-radius first:

1. **Deterministic retry-on-#error (cheapest, no new capability).** When a chain shim's
   only build error is the integrator's own `#error "… requires manual classification"`,
   force a fresh regeneration attempt (the escape hatch is inherently non-deterministic;
   B12 shows a retry lands the working overload). This is a Patcher retry-policy tweak,
   not a normaliser change, and directly matches the observed variance. Recommended
   first move.
2. **Deterministic forwarding-overload emitter for qualified math-bridge helpers.**
   Extend the deterministic side (the Gap-A namespace-qualified bridge machinery) to
   emit the `Lnrat<ext, …>` → `::ql::Lnrat<ext, …>` forwarding overload directly, the
   same way the bridge already handles `kLog`/`log`. This removes the LLM from the
   helper-forwarding decision entirely (turns a probabilistic #error into a
   deterministic overload). Larger change; the right long-term fix.
3. **Prompt hardening (weakest).** Strengthen the chain-integrator R4-discipline for a
   qualified call into the target namespace whose primary template is visible — forbid
   the #error escape when a forwarding overload is constructible. Mitigates but does not
   eliminate the variance.

Recommend **(1) now** to unblock B10's measurement cheaply, **(2) as the durable fix**.
Both are Patcher gen-robustness work, tracked separately from the closure track, exactly
as 2b scoped it.

## After success (NOT dispatched here)

Group B measurement-only (Phase 2f, needs Reet); B13 chain-selector refinement
(narrow Li2omx2 scope, separate followup); B12 multi-chain / hotspot-covering chain
selection (would move the 3.6906 floor); cross-integral merge (Reet's call).

STOP-after-Stage-2 holds; Group B / all-21 not run.
