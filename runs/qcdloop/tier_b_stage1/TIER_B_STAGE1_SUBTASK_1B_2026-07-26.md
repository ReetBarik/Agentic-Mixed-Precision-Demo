# Tier-B Stage-1 — Closure Subtask 1b e2e (B13/B14) — 2026-07-26

Closure-scoped chains, **Subtask 1b of 5** (design: `runs/qcdloop/CLOSURE_SCOPED_CHAINS_DESIGN.md`).
This run re-measures the two Stage-1 integrals in scope for 1b — **B13** and **B14** —
after the designed-exit gate reformulation. B10 (needs rule (c)) and B12 (out of
scope) were deliberately skipped.

- gate: chain positive-lift ≥ 0.5 digits vs kernel-scoped accumulated-min (`chain_dd`)
- seed 12345, sample_count 5000, entry BO, tolerance 6.0 (reporting-only)
- run artifacts: `runs/qcdloop/tier_b_stage1_subtask1b/` (dedicated out-dir; canonical
  `tier_b_stage1/` B10/B12 artifacts left untouched)
- driver: `runs/qcdloop/tier_b_stage1_subtask1b.sh`, log
  `runs/qcdloop/tierb_stage1_subtask1b_run.log`

## Result summary

| I  | prev (2026-07-25)        | now                         | patcher_status | kernel base→final (lift) | verdict |
|----|--------------------------|-----------------------------|----------------|--------------------------|---------|
| B13 | apply_failed @ write_truncation | apply_failed @ write_truncation | `write_truncation` | — (never measured)       | **rule-(c) seam — STOP #3** |
| B14 | apply_failed @ write_truncation | **rejected (chain_no_lift)**    | **`ok`**       | 13.1855 → 13.1855 (+0.00) | **cleared; correctly no-lift** |

## B14 — designed-exit gate WORKS (write_truncation → measured)

B14's dominant COMPUTED chain (`cascade_B14_3429b1d4_01bf2ff3`, lines
B2m.h:401, B2m.h:578, kokkosUtils.h:1208) previously died at the chain-scope
`write_truncation` terminal — the pre-1b gate false-positived the chain's designed
output store as an interior truncation.

Under Subtask 1b the **kernel_output** landing (`res(i,k)` store) is recognized as a
designed exit and exempted, so the gate no longer fires: the chain **built, wired, and
was measured**. `patcher_status` advanced from `write_truncation` to **`ok`**, and the
Validator produced B14's own kernel floor: **p100 = 13.1855 digits at baseline, 13.1855
after** — lift **+0.00**.

The solver then **correctly rejected** the chain as `chain_no_lift` (13.1855 < 13.1855 +
0.5). This is the intended, honest outcome, not a regression:

- B14 is a **dd-sufficient** integral whose kernel is already ~13 digits precise at
  double. There is nothing for a dd promotion to lift — the dominant chain's predicted
  +16.66-digit lift was **spurious** (the chain is not this integral's floor driver).
- §7 falsification check: the pre-1b verdict was a *rejection* (apply_failed) and the
  post-1b verdict is also a *rejection* (chain_no_lift). The gate did **not** flip a
  currently-correct rejection into a false accept. ✅

Net for B14: the write_truncation false-terminal is **cleared**; the chain is now
measured and rejected for the right reason.

## B13 — genuine cross-frame return seam → rule (c), out of scope (STOP #3)

B13's dominant COMPUTED chain (`cascade_B13_79fc5b8f_f080f240`, lines
B2m.h:300/301/305/306/355/533, kokkosUtils.h:212/702) **still** ends at
`apply_failed @ write_truncation`. Direct closure diagnosis (reproduced read-only on a
fresh clone) shows this is **not** a benign-extract miss and **not** a false positive —
it is the **Li2omx2 cross-frame return seam that also blocks B10**, which requires
**rule (c) / return-type widening (Subtask 2a)** — explicitly out of scope for 1b.

Diagnosis (closure of B13's dominant chain):

```
closure_names       : ga34m ga34mm1 ga34p ga34pm1 ga43m ga43mm1 ga43p ga43pm1 root
designed_exits      : B2m.h:310 extract carried=[ga34pm1] detail='x34pm1'
                      B2m.h:311 extract carried=[ga34m]   detail='x34m'
                      B2m.h:315 extract carried=[ga43pm1] detail='x43pm1'
                      B2m.h:316 extract carried=[ga43m]   detail='x43m'
source_escapes      : ga34m/ga34pm1/ga43m/ga43pm1 -> ql::Real (diagnostic; non-blocking)
destination_escapes : []          # no shared-state / non-benign extract severance
blocking_escapes    : []          # chain_closure_escapes does NOT fire
interior gate trips : kokkosUtils.h:702  (reads v,x,w,y)  -> TRIPS write_truncation_inert
```

The tripping interior region is `kokkosUtils.h:702`:

```cpp
Li2omx2 = TOutput(ql::Constants<TScale>::template _pi2o6<...>()
                  - ql::ddilog<TOutput, TMass, TScale>(arg)) - prod;
...
return Li2omx2;                     // kokkosUtils.h:707
```

`Li2omx2` is a **helper-frame local** (`TOutput` = double) computed inside the chain and
returned to its caller via `return Li2omx2`. Widening the chain body computes the value
at dd, but the value must cross the frame boundary via the function **return**, which
stays at caller precision (double) → the dd residual is truncated at the return. The
designed-exit predicate treats `kind == "return"` as **not-yet-designed** (clause (ii),
`_designed_exit_kind` returns `False` with `# TODO(subtask-2a)`), so the gate **correctly**
rejects the chain: resolving it needs the variant's return type widened —
**rule (c), Subtask 2a/2b**.

Two things this run **confirms are correct in Subtask 1b**:

1. **benign-extract detection works.** All four `ga34*/ga43*` values that flow through
   `ql::Real(...)` extracts in B2m.h are recognized as **designed exits** (kind
   `extract`), so `destination_escapes` and `blocking_escapes` are empty and the
   `chain_closure_escapes` terminal does **not** fire spuriously. STOP #2 (benign-extract
   ambiguous on B13) did **not** occur.
2. **source vs destination escape split is right.** The `ql::Real` argument flows are
   recorded as *source* escapes (diagnostic, non-blocking — `ga34*` stay in
   `closure_names`), not destination escapes, so they neither block widening nor fire the
   terminal. A.1 exposed no broken 1a assumption (STOP #4 did not occur).

## STOP status

- **STOP #3 triggered** for B13: it still `apply_failed @ write_truncation`. But this is
  a **clean, expected stop** — B13's dominant chain shares B10's Li2omx2 cross-frame
  return seam, which the design deferred to **rule (c) / Subtask 2a**. The gate is
  rejecting correctly; no falsification, no gate gap.
- **STOP #1 / §7 (false accept)** did **not** occur — B14's and B13's verdicts both
  remained rejections.
- **STOP #2** (benign-extract ambiguity) did **not** occur — B13's extracts classify
  cleanly as designed.
- **STOP #4** (A.1/A.2 broke a 1a assumption) did **not** occur.

## Conclusion / recommendation

Subtask 1b lands as intended and is **verified end-to-end**:

- The designed-exit reformulation converts the write_truncation false-terminal into a
  real measurement wherever the chain's landing is a designed exit (B14 kernel_output).
- The escape split + benign-extract procedure classify B13's real flows correctly with
  zero false `chain_closure_escapes`.
- B13's residual `write_truncation` is now **precisely diagnosed** as the cross-frame
  Li2omx2 return seam — the same rule-(c) dependency as B10. It is **not** addressable
  in 1b by construction.

**Recommendation (for Reet):** Subtask 1b is complete and correct. B13 and B10 both wait
on **rule (c) / return-type widening (Subtask 2a)**; the two integrals share the Li2omx2
helper return, so a single rule-(c) implementation should unblock both. B14 is
dd-sufficient (already ~13 digits) — its dominant chain is not a floor driver and needs
no fix. **STOP after Stage-1 for review** before Subtask 2a.
