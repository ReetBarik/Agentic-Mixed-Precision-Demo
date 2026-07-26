# Tier-B Stage-2 — closure Subtask 2b: rule (c) cross-frame return propagation (2026-07-26)

Closure-scoped chain promotion **rule (c)** (CLOSURE_SCOPED_CHAINS_DESIGN.md §2.3, §6.3 Stage 2):
cross-frame return-type widening + designed-exit clause (ii) + the chain-line decl-init carrier
completion the two exposed. B10 is the headline (the integral rule c was designed for); B13
piggybacks; B14 is a non-regression check. B12 skipped (Subtask 3).

- gate: positive lift ≥ 0.5 digits vs accumulated-min (kernel-scope); tolerance 6.0 (reporting-only)
- seed 12345, sample_count 5000, entry BO
- run dir: `runs/qcdloop/tier_b_stage2_subtask2b/`

## Per-integral outcome (kernel-scoped gate)

| I | outcome | patcher_status | kernel lift | predicted | 1b baseline | Δ vs 1b |
|---|---|---|---|---|---|---|
| B10 | apply_failed | **llm_gen_failed** (build_failed) | — | +18.43 | write_truncation | **write_truncation CLEARED → advanced to shim-build** |
| B13 | apply_failed | write_truncation | — | +17.10 | write_truncation | unchanged (no regression) |
| B14 | rejected | ok (chain_no_lift) | +0.00 (13.1855→13.1855) | +16.66 | rejected/chain_no_lift | unchanged (no regression) |

## The headline: B10's interior write-truncation gate is CLEARED by rule (c)

Three runs, same chain, isolate the effect:

| stage | B10 patcher_status |
|---|---|
| Subtask 1b (rules a/b, no rule c) | `write_truncation` |
| Subtask 2b, rule (c) only (no decl-init carrier completion) | `write_truncation` |
| **Subtask 2b, rule (c) + `closure_body_names`** | **`llm_gen_failed` (past the gate)** |

Rule (c) fires on BOTH B10 chain-internal return edges exactly as §2.7's worked trace predicts:
- `ddilog` (kokkosUtils.h) return type `TMass` → `quad::ddfun::ddouble` (recorded ReturnWiden,
  sig line 149), consumed inside `Li2omx2` at :698/:704;
- `Li2omx2` return type `TOutput` → `ddcomplex` (ReturnWiden, sig line 688), consumed in `B10`
  at B1m.h:{235,236,237}; receiving locals `dilog3/dilog4/dilog5` re-enter rule (a) and widen;
- the cancellation `res(i,0) = dilog4 - dilog5 …` (B1m.h:241) now executes at dd and its store to
  the `res(i,k)` container is the designed exit (kernel_output). Both callee returns are
  `return_widened` designed exits (clause ii), so the interior gate no longer fires on them.

The write-truncation that remained after rule (c) alone was a SEPARATE, newly-exposed seam: Li2omx2's
own internal locals `lnarg`/`lnomarg` (kokkosUtils.h:702/703) are **decl-init on chain lines**
(`const TOutput lnarg = …`) whose value is read by the dd cancellation at :704. Because
`region_writes_from_source` deliberately excludes decl-init targets (a decl is a *landing*, not a
Case-B write), the closure never marked them as carriers and the boundary transform demoted them back
to `TOutput`, injecting double roundoff into the dd accumulation. Rule (c) EXPOSED this by making
Li2omx2's body promote at all (before 2b, Li2omx2 never widened). The fix (`CarrierClosure.
closure_body_names`) recognizes a decl-init on a chain line read by another chain line as a body-owned
carrier and threads it into `closure_names` so `promote_region_block` keeps it wide — no
`closure_decl_widens` record (the body transform already owns the in-line decl edit). With that, B10's
gate is silent and the chain emits fully.

## Why B10 still `apply_failed`: an orthogonal Patcher shim-gen defect (NOT rule c)

B10 now reaches the LLM chain-integrator shim build and fails there after 3 attempts:

```
ql_shim_dd.h:7: error: #error "DD Chain Integrator: ql::Constants<ddouble>::_pi2o6 requires manual classification"
  (emitted by shim kokkosUtils_dd_L177 — ddilog's body region)
kokkosUtils.h:1250+: error: redeclaration of 'quad::ddfun::ddouble T__ff'  (kfn shim, duplicate promotion)
```

Both are chain-integrator **LLM generation** defects, not closure/gate/emission defects:
- `_pi2o6` (π²/6) is a source-derivable constant the LLM could not classify — the exact
  `_ieps50`-class Patcher gen-robustness gap [[project_blocker_a_carrier]] / the WAVE-3 prompt work
  scopes as "Patcher gen-robustness, not precision";
- the `T__ff` redeclaration is a shim double-promotion collision in an unrelated helper region.

Neither touches rule (c), the designed-exit gate, or the variant-emission machinery — those all
succeeded (the chain passed every closure gate and rendered its variants; the build is downstream).

## STOP-condition audit (§7)

- **STOP #1 (design falsification)** — did NOT occur. B10 never reached measurement (the build failed
  upstream at LLM shim gen), so "emits but measures chain_no_lift" cannot apply. The cross-frame-return
  hypothesis is neither confirmed nor falsified by measurement yet; it IS confirmed structurally (the
  gate that encodes the truncation correction is now silent for B10 and fires for a plain, non-widened
  return — see §3.3 tests).
- **STOP #2 (still write_truncation after 2b)** — did NOT occur for B10 (cleared) or B14. B13 remains
  `write_truncation`, but that is its **1b baseline unchanged**, not a 2b composition regression: B13's
  dominant chain scopes only Li2omx2's :702 (not :704), so its `lnarg` is written on a chain line but
  read on a NON-chain line — outside the strict chain-scoped `closure_body_names` rule by design. Rule
  (c) does not fire for B13 (`return_widens=[]`): its Li2omx2 return is consumed at B2m.h:355, but the
  chain's Li2omx2 frame is scoped to a single line, so the receiving-local re-closure has nothing to
  bind. This is a **chain-selection scope** limitation, orthogonal to rule (c).
- **STOP #3 (accept↔reject flip)** — did NOT occur. B14 identical (rejected/chain_no_lift,
  13.1855→13.1855); B13 identical (write_truncation); no currently-correct rejection became a false
  accept. The `closure_body_names` addition was verified to leave B13's and B14's closures
  byte-identical to their 1b results (strict chain-scoped predicate; no seed/rule-(b)/rule-(c) change).
- **STOP #4 (inward-parameter widening on a shared helper)** — did NOT occur as a crash/false escape.
  B13's Li2omx2 receiving the `x34*` extracts (which would need dd *inward*) is exactly the §8 boundary;
  rule (c) correctly does NOT fire there (no return widen recorded), so no inward-param widen is
  attempted. B13 stays at its honest write_truncation rather than emitting an unsound inward widen.
- **STOP #5 (ReturnWiden names no emitted variant)** — did NOT occur. Frame-level ReturnWiden
  (function_name = original name, e.g. `Li2omx2`/`ddilog`; return_line = signature line) binds via
  `_attach_return_widens` by `orig_name` + line-containment to every per-caller-path variant; the B10
  emission produced 48 variants with the return widens attached, no FanoutError.

## Verdict

Rule (c) **works**: B10's interior write-truncation gate — the seam that blocked B10 through Blocker A,
Subtask 1b, and rule-(c)-alone — is cleared, and the chain emits its full dd envelope (ddilog and
Li2omx2 returning dd, dilog4/dilog5 widened, cancellation at dd, res store the designed exit). B10's
residual blocker is a pre-existing, orthogonal Patcher LLM shim-gen defect (`_pi2o6` classification +
a `T__ff` redeclaration), the same class WAVE-3 / the `_ieps50` work already scopes out of the precision
track. B13 and B14 are unchanged from their 1b baselines (no regression; B13's limitation is chain
scope, not rule c; B14 is dd-sufficient and correctly rejected).

**No STOP condition fired.** Recommend: (1) accept rule (c) as landed (the gate correction is proven and
non-regressing), (2) the `_pi2o6`/`T__ff` chain-integrator gen defects are the next blocker for a B10
*measurement*, tracked with the existing Patcher gen-robustness work, not the closure track, (3) B13's
narrow Li2omx2 chain scope (a selector refinement) and B12 (Subtask 3) remain out of scope here.

STOP-after-Stage-2 holds; Group B / all-21 not run.
