# Tier-B Stage-1 — chain-scoped dd promotion (2026-07-25)

Phase 2f coordinated whole-chain double-double promotion on the 4 measured Tier-B
integrals (B10/B12/B13/B14). v1 promotes the dominant COMPUTED cascade chain per
integral (one coordinated envelope). **STOP after Stage-1 — Reet reviews before Group B.**

- gate: positive lift >= 0.5 digits vs accumulated-min (chain_dd); tolerance 6.0 (reporting-only)
- seed 12345, sample_count 5000, entry BO
- starting_sha (headers repo): `6f21981a1402`
- call_graph template-extent fix: `41f0391` (this landed BETWEEN the two runs)

## TL;DR

The `call_graph.py` template-extent fix **fully cleared the original blocker**. In
the pre-fix run all 4 integrals fast-failed identically at ~35–52 s with
`chain region kokkosUtils.h:<N> not inside any known function` — the coordinated
chain-promote never ran. Post-fix, **all 4 chains resolve, splice, and run the full
coordinated promote** (51–649 s of real work), and each reaches a *distinct,
semantically meaningful* terminal state. No integral fails on graph resolution any
more.

But **zero chains were accepted**, for three different reasons — none of which is the
graph bug, and two of which are downstream issues I was scoped to flag-and-stop on
rather than fix:

| I | pre-fix | post-fix outcome | wall | what it means |
|---|---|---|---|---|
| B10 | apply_failed (graph) | `write_truncation` (chain-scope 2d-B gate) | 280.5 s | chain built; gate fired BEFORE validate — **suspected false positive** |
| B12 | apply_failed (graph) | `write_truncation` (chain-scope 2d-B gate) | 158.5 s | same as B10 |
| B13 | apply_failed (graph) | `llm_gen_failed` (API timeout) | 649.1 s | **transient infra**, not a real result — re-runnable |
| B14 | apply_failed (graph) | `rejected [chain_no_lift]` | 51.3 s | **first real measurement**: built + validated; lift 0.00 |

## The one real precision measurement: B14

B14's chain is the only one that built cleanly (`patcher_status=ok`), was widened to
dd, and got a full whole-app validate:

- **baseline p100 = 3.6906 → final p100 = 3.6906, measured lift = +0.00** (predicted +16.66)
- rejected under the +0.5 lift gate (`3.6906 < 3.6906 + 0.5`)
- **`baseline_hotspot` = B12 / sample 3868 / `coeff0.imag` / precise_digits 3.6906**

That hotspot is the crux. The whole-app p100 is pinned by **B12's** cancellation
floor — a *different integral* that B14's chain does not touch. So B14's dd promotion
literally cannot move the number the gate reads, regardless of how much it helps B14's
own coefficients. This is the **whole-app-gate-instrument problem** (documented for the
greedy solver in `project_phase_2e_solver_stage1`) resurfacing at chain scope: a
per-integral chain measured against a whole-app global min can only ever score a lift
if it happens to own that global min. **Predicted-vs-measured for B14 is therefore not
a real disagreement** — the +16.66 Item-7 prediction is about B14's *own* floor; the
whole-app instrument never measured it.

## The chain-scope 2d-B gate (B10, B12) — suspected false positive

Both B10 and B12 tripped `chain_write_truncation` and were denied a build+validate.
The gate (`agents/patcher/chain_promote.py::chain_write_truncation`) applies the
per-region 2d-B detector `boundary.write_truncation_inert` to the chain's **outermost**
region (shallowest on the call graph — the last landing before the value returns to
the driver). It fires when that region's writes land back at `double` with "no wider
persistent sink."

The outermost regions are exactly the integral's **output stores**:

- B10 outermost = `B1m.h:240/241` → `res(i,1) = wlog2mu + wlog4mu - wlogsmu - wlogtmu;`
  and `res(i,0) = dilog4 - dilog5 - 2*dilog1 + 2*dilog2 + 2*dilog3 + …`
- B12 outermost = `B2m.h:241` → `res(i,0) = -pi2o12 + 2*wlogsmu*wlogtmu - wlog4mu*wlog4mu + …`

`res(i,0)` **is** the persistent sink — it's the coefficient array the shared driver
reads. `res(i,0) = dilog4 - dilog5 - 2*dilog1 + …` is the catastrophic-cancellation
line; widening the dilog terms to dd so the cancellation happens in dd *before* the
single final round to `double` is the entire intended benefit of the chain. The
per-region detector, which has no notion of "this store is the chain's output
boundary," reads the final demotion-to-double as inert truncation and kills it
pre-build.

This is precisely the failure mode `chain_promote.py`'s own module docstring warns
about ("this is exactly the reasoning 2d-B's per-region gate would get wrong at chain
scope") — but the current `chain_write_truncation` still delegates to that per-region
detector for the outermost region, so the warning's own case slips through. B14 passed
the gate only because its outermost region (`B2m.h:401`, a local `TOutput fac`) is not
an output store.

**I did not change this** — it lives in `chain_promote.py`/`boundary.py`, and per the
task's "flag and stop" boundary (do not extend into fanout/dispatch/chain_promote),
reacting to it is a separate change for Reet to authorize. Flagging it here.

## B13 — transient, re-runnable

`llm_gen_failed` after 3 attempts on chain region `B2m.h:305`: the underlying error is
`api_error` / "Request timed out or interrupted" from the Argo proxy, not a
generation-logic failure. B13's 8-region chain is the largest dd shim-generation load
of the four; a re-run (or a smaller per-attempt timeout budget) should get past it.
Not a real precision result.

## Per-integral detail

| I | chain | lines | tightness | measured max_rel_err | predicted lift | measured lift | patcher_status | outcome |
|---|---|---|---|---|---|---|---|---|
| B10 | cascade_B10_612f1391_494252c4 | 10 | 3.33e-03 | 1.96e+01 | +18.43 | — (no build) | write_truncation | apply_failed |
| B12 | cascade_B12_65bb39c0_62ff5a3d | 5 | 7.08e-02 | 1.62e+07 | +17.10 | — (no build) | write_truncation | apply_failed |
| B13 | cascade_B13_79fc5b8f_f080f240 | 8 | 7.08e-02 | 1.85e+02 | +17.10 | — (infra) | llm_gen_failed | apply_failed |
| B14 | cascade_B14_3429b1d4_01bf2ff3 | 3 | 1.99e-01 | 3.27e+05 | +16.66 | **+0.00** | ok | rejected (chain_no_lift) |

### B10 — `cascade_B10_612f1391_494252c4`
- lines: B1m.h:227, B1m.h:240, B1m.h:241, kokkosUtils.h:174, kokkosUtils.h:177,
  kokkosUtils.h:199, kokkosUtils.h:212, kokkosUtils.h:702, kokkosUtils.h:703,
  kokkosUtils.h:704 (spans B1m + `ddilog` + `Li2omx2`, all now resolved)
- outermost region: `B1m.h:240/241` (output stores `res(i,1)`, `res(i,0)`)

### B12 — `cascade_B12_65bb39c0_62ff5a3d`
- lines: B2m.h:206, B2m.h:207, B2m.h:241, kokkosUtils.h:212, kokkosUtils.h:702
- outermost region: `B2m.h:241` (output store `res(i,0)`)

### B13 — `cascade_B13_79fc5b8f_f080f240`
- lines: B2m.h:300/301/305/306/355/533, kokkosUtils.h:212, kokkosUtils.h:702
- transient API timeout on B2m.h:305 shim gen

### B14 — `cascade_B14_3429b1d4_01bf2ff3`
- lines: B2m.h:401, B2m.h:578, kokkosUtils.h:1208 (`kfn`, now resolved)
- built + validated; whole-app floor pinned by B12 hotspot → lift 0.00

## What the graph fix proved (and did not)

- **Proved:** the template-extent recovery works on the real header — `ddilog`,
  `Li2omx2` (was truncated + mislabeled non-template), and `kfn` all resolve; every
  Tier-B chain line lands inside its enclosing function; chains splice and (for B14)
  build + validate end-to-end. defs 20→44 on standalone kokkosUtils.h (== 42 template
  heads + 2 printDoubleBits overloads, exact). Full suite 623 green.
- **Did NOT prove:** that dd chain promotion lifts these integrals' floors. Only B14
  reached a measurement, and the whole-app instrument couldn't see B14's own floor
  (pinned by B12). B10/B12 were gated pre-build; B13 timed out.

## Recommended next steps (for Reet — none taken)

1. **Chain-scope 2d-B gate fix** (blocks B10/B12, likely B13 too). Make
   `chain_write_truncation` exempt the chain's designated OUTPUT boundary (the
   outermost region's store into the driver-visible coefficient array) instead of
   delegating to the per-region detector that treats it as inert. This is the
   module-docstring's own stated intent; the delegation is the bug. Separate change,
   inside chain_promote.py — flag-and-stop per task scope.
2. **Gate instrument for chain acceptance** (blocks B14, and any integral not owning
   the whole-app global min). The +0.5-lift-vs-whole-app-p100 gate cannot score a
   per-integral chain whose target floor is not the global min. Same regression-relative
   / per-integral-floor instrument decision already open for the greedy solver
   (`project_phase_2e_solver_stage1`) applies here.
3. **Re-run B13** once (1) lands, to convert the transient timeout into a real result.

## Notes
- STOP after Stage-1 for review; Group B / all-21 not run.
- v1 = dominant chain per integral; multi-chain union deferred to Stage-2.
- Pre-fix (all-apply_failed) report preserved at `/tmp/TIER_B_prefix_apply_failed.md`.
