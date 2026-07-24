# Solver Stage — log_near_root dd probe (Phase 2e Stage-2 prep, Item 3)

**Verdict: 🛑 STOP before Item 4 (all-21). The dd-acceptance path of the solver is
unvalidated — the fan-out produces zero dd DISCRIM candidates on any probed
integral, so the solver never has a dd promotion to accept and whole-app
`min_precise_digits` never moves.** This is the exact STOP condition the handback
named for Item 3 ("if the solver doesn't accept any dd promotion, or accepts but
min doesn't move — STOP and investigate; that would mean the fan-out isn't
generating dd candidates on the relevant regions — a coverage gap upstream of the
solver").

The regression-relative gate (Item 1) is, separately, **validated end-to-end** —
see §Item 1 below. The blocker is confined to the correctness/dd dimension.

---

## What was run

* **Solver probe (Item 3 proper):** `run_solver_stage1.py --integral B10` under the
  regression-relative gate. Auto-report: `solver_stage1_B10/SOLVER_STAGE1_B10.md`.
* **Confirmation measurement passes** (to test whether the dd gap is B10-specific
  or systemic): B8, B9 measured at 5k with `--fanout`, plus the pre-existing B1 and
  B12 2e manifests. Five integrals total, all carrying `log_near_root` hotspots.
* seed=12345, sample_count=5000, tolerance=10, entry=BO.

## The finding — zero dd DISCRIM anywhere (systemic)

A dd cell is a **DISCRIM** (a candidate the solver can rank/accept) only if it was
`measured` **and** its delta differs from the baseline delta. Across all five
integrals:

| integral | dd cells | measured | INERT (no-op) | **DISCRIM** | patcher_failed (P6a) |
|----------|:--------:|:--------:|:-------------:|:-----------:|:--------------------:|
| B1       | 0 | 0 | 0 | **0** | 0 |
| B12      | 8 | 1 | 1 | **0** | 7 |
| B10      | 9 | 1 | 1 | **0** | 8 |
| B8       | 9 | 4 | 4 | **0** | 5 |
| B9       | 5 | 1 | 1 | **0** | 4 |
| **total**| **31** | **7** | **7** | **0** | **24** |

**Zero dd DISCRIM in 31 dd cells across five integrals.** The solver's dd branch is
therefore untestable on the current measurement layer — not because the solver is
wrong, but because nothing upstream feeds it a dd candidate that changes output.

### Two distinct upstream causes

1. **P6a generation gap — 24/31 dd cells (`patcher_failed`).** The Patcher fails to
   generate a valid double→dd shim on these regions (log_near_root / cascade lines).
   These never reach `measured`, so the solver never sees them. This is the same
   dd-generation-robustness gap flagged in WAVE-1+2 (`_ieps50` dd_untested) and in
   the B12 Stage-1 finding.

2. **Genuine numerical no-op — 7/31 dd cells (`measured`, INERT).** These *did*
   generate, build, and run cleanly (`patcher_status=ok`, `failure_mode=ok`,
   `via=plain`) and still produced **byte-identical** whole-app output
   (`delta == baseline_delta`). They passed both the 2c `promotion_no_op` and 2d
   `write_truncation` static gates, so they are **not** empty-payload or
   write-truncation artifacts — widening these regions to dd genuinely does not
   recover the lost digits. B10's floor is the clean example: `kokkosUtils.h:212`
   (log_near_root) is *both* the dd_ceiling region (final min 3.6906) *and* a
   measured-INERT dd cell — promoting exactly the region that holds the floor does
   not lift it, because the catastrophic cancellation has already happened in double
   upstream of that line. This is a physics ceiling, not a solver or gate defect.

Net: on every probed integral the dd rung yields **either** a generation failure
**or** a proven no-op. There is no integral where "double→dd on the hotspot lifts
whole-app min" is demonstrable today, so the solver's core value proposition on the
correctness dimension cannot be shown, and Stage 2 would produce 21 trees whose dd
columns are uniformly empty.

## Item 1 — regression-relative gate: VALIDATED (independent of the STOP)

The B10 solver run exercised the new gate cleanly. Baseline whole-app p100 = 3.6906
(< the old absolute 6.0 → the old gate would have STOPPED on the baseline, as it did
for B12). Under the regression-relative gate (accept iff `cand_min >= baseline − 0.5
= 3.1906`):

* **5 float demotions accepted** — `B1m.h:227`, `B1m.h:236`, `B1m.h:237`,
  `boxGPU.h:139`, `boxGPU.h:141` — each holds the floor (min stays 3.6906). Real
  speedup on an ill-conditioned integral the absolute gate could never have helped.
* `validator_verdict` on each accepted cell is `reject` (its bundled absolute floor),
  correctly **overridden** by the solver's own regression-relative decision — exactly
  the design intent and the B12 report's option-1 prediction.
* final p100 = baseline = 3.6906 (unchanged — no regression, no dd lift).

So Item 1 works as specified. The speedup path (float/ff DISCRIM) is healthy across
all five integrals (8/3/6/4/7 float+ff DISCRIM cells respectively).

## Item 2 — signal_class filter: unit-correct, but no e2e awaiting cells (routing note)

The filter (`agents/patcher/dispatch.py::_awaiting_rewrite`, keyed on
`FanoutSettings.signal_class_by_region`) is unit-tested and fires correctly **when a
cascade/local region is enumerated as a fan-out cell**. But across all five probed
integrals, **0 `awaiting_algorithmic_rewrite` cells were emitted**. Root cause: the
per-integral pipeline never routes `cancellation_cascade` / `local_cancellation`
regions through the fan-out `generate()` where the filter lives — they are handled
(and cheaply dedup-skipped) by the Strategy chain-walk, whose targets are chain ids
(`cascade_B10_…`) not the `file:line` region ids the filter's map is keyed on. E.g.
B10's three cascade line regions (`B1m.h:248`, `boxGPU.h:140`, `kokkosUtils.h:698`)
appear in the correctness summary as chain-based `dd_untested` ceilings, never as
fan-out cells.

Consequence: no build/LLM budget is wasted on cascades (the chain-walk already
short-circuits them via dedup), so the filter's *practical* value on this pipeline is
semantic (clean `awaiting_algorithmic_rewrite` labelling) rather than budgetary. To
actually emit awaiting cells end-to-end, the filter would need to also sit at the
chain-walk / candidate-selection layer, not only inside fan-out `generate()`. This is
a scoping observation for Reet — Item 2's stated acceptance (unit-tested filter, no
builds/LLM on cascade/local) is met; the e2e awaiting-cell emission was not part of
that acceptance and is blocked by this routing split.

## Decision required before Item 4 (Reet)

Item 4 (all-21 Stage 2) is gated on Items 1–3 landing clean. Item 3 did **not** —
the dd path is unvalidated. Options:

1. **Close the P6a dd-generation gap first**, then re-probe one log_near_root
   integral for a real dd DISCRIM, then run Stage 2. Highest confidence; addresses
   24/31 of the gap. This is Patcher gen-robustness work, upstream of the solver.
2. **Accept that dd is a genuine ceiling on these hotspots** (the 7/31 no-ops argue
   the measurable dd cells land on cancellations dd cannot fix) and **re-scope Stage 2
   as a speedup/float study** — run all 21 for the validated float/ff demotion wins,
   explicitly documenting dd columns as "untested (no dd DISCRIM in the measurement
   layer)". Fast; delivers the 21 trees; leaves correctness-via-dd for a later pass
   once the rewrite path (Item 2's `awaiting_algorithmic_rewrite` → kahan/identity) is
   wired.
3. **Investigate the 7 INERT dd cells deeper** — confirm case-by-case that each is a
   true physics ceiling vs a subtle promotion-coverage miss (the promotion widened a
   line adjacent to, not spanning, the cancellation). If any are coverage misses,
   that reopens the dd path without full P6a work.

**Recommendation:** option 1 or 3 before a correctness-bearing Stage 2; option 2 only
if you want the float/speedup trees now and accept dd stays open. The
`run_solver_stage2.py` orchestrator is written and ready either way — one launch once
you decide.

**Do not run Item 4 until this is reviewed.** No gate substitution, no rewrite wiring,
no cross-integral merge were done. The dd gap is upstream of the solver and would make
all 21 correctness columns empty in the same way.

## Artifacts

* `solver_stage1_B10/SOLVER_STAGE1_B10.md` — B10 solver auto-report (mechanics; 5
  float accepts, gate detail).
* `solver_stage1_B10/solver_result.json`, `.../tree_B10`, `.../final.diff` — merged
  B10 tree + cumulative diff.
* `per_integral_out_stage2/{B10,B8,B9}/manifest_scorer_*.jsonl` — measurement
  manifests (dd/float/ff cells).
* `per_integral_out_2e_measure/{B1,B12}/manifest_scorer_*.jsonl` — pre-existing 2e
  manifests reused for the systemic check.
