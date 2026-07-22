# Per-integral pass orchestrator — Phase 1 landing + B1 sizing

_2026-07-22. Phase 1 of the caller-scoped pipeline: run the existing
Strategy→Patcher→Validator pipeline once per integral against a filtered report in
an isolated tree. No fan-out, no caller-scoped naming, no combine — regions only;
chains stay on the merged whole-app path (Phase B: chains are 100% uniformly dd)._

## What shipped

- `agents/per_integral_orchestrator/` — generic core: `filter_report` (per-integral
  report slice with a Phase-A `integral`-tag fidelity guard; chains kept whole),
  `run_per_integral_pass` (filter → fresh-clone tree → fail-fast vanilla gate →
  injected `pipeline_fn` → manifest), `build_manifest` (accepts/rejects/settled
  precision/modified files from the fat Strategy report + iteration log + diff).
- `runs/qcdloop/run_all_integrals.py` — qcdloop fan-out driver reusing
  `run_strategy_e2e` wiring; sequential or `--workers N` (ProcessPoolExecutor);
  emits `sizing_summary.json`.
- `tests/per_integral_orchestrator/` — 10 tests (filter fidelity, chain integrity,
  tree isolation, Strategy-loader compat, manifest completeness, signal-recovery
  smoke). Adjacent suites unchanged (strategy/patcher/validator: 187 pass).

## Isolation (verified live, single-integral run)

Every mutable artifact lives under `per_integral_out/B1/`: filtered report, cloned
`tree_B1/`, Strategy `runs_root` (→ Patcher build/logs/shims), Validator tempdir,
manifest. **Base repo pristine after the run** (same SHA `65a2655`, clean status).
The three flagged contention points resolve as designed: `app_cmake_dir` is a
read-only cmake `-S` source (each build writes its own `-B`); `runs_root` is
per-pass; the app drivers emit no `journal.jsonl`. The design is correct for the
Phase-2 parallel 21-run, not just this single pass.

## B1 real-data run (report_5k.json, tol=10, seed=12345, sample_count=5000)

| metric | value |
|---|---|
| status | success (hit `dr_k=40` diminishing-returns) |
| wall (strategy pass) | 794 s (~13.2 min); total incl. base-repo build + gate 817 s |
| iterations | 40 (correctness 2, speedup 38) |
| accepts / rejects | **0 / 40** |
| settled precision dist | double×32, dd×2 (2 dd-ceiling retains) |
| disk / pass | 48 MB — dominated by the 27 MB filtered-report slice (build dirs ~5 MB) |

### Sizing extrapolation to 21 integrals (flat, from B1)

- sequential ≈ **4.6 h**; @4 workers ≈ 1.2 h; @8 workers ≈ 0.6 h
- disk ≈ **~0.6 GB** (compact filtered reports dominate; ~27 MB × 21)

**Caveats on the extrapolation:** B1 has 0 chains and hit the `dr_k` cap at 40
iters. (1) The DD-oracle build + coeff computation is a one-time cost amortized via
the shared atomic coeff cache — later passes skip it, so B1's 794 s is an *upper*
per-pass estimate. (2) The BIN* integrals carry 30k–59k chains (all dd) and more
correctness iterations, so they will run longer than B1 — the flat avg×21
understates the tail. A truer number needs one BIN* pass measured.

## Key finding — signal recovered upstream, blocked at the (still whole-app) Validator

The per-integral filter **works exactly as the Phase B probe predicted**: the walk
*attempted* `double-to-float` on `boxGPU.h:99/100/101` — the probe's top wasted-
headroom lines. Under the whole-app merge these sit in the *correctness* queue
(merged to dd) and would never be attempted as a speedup; per-integral routing put
them in B1's speedup queue and the walk aimed for float. Signal recovery is real and
demonstrated end-to-end (and deterministically, by `test_signal_recovery`).

But **every candidate settled back to double** (`insufficient_fix`,
`cand_min_precise_digits = 3.69` — *constant* across all 32 speedup attempts). Two
compounding, expected Phase-1 causes:

1. **Numerical no-op (the known Phase-1 bug).** Per-region shims are dead code in the
   double build — callers aren't rerouted (no fan-out yet), so the candidate is
   numerically identical to vanilla every time. The constant 3.69 is the signature.
2. **The Validator is still whole-app / global.** It builds the full `boxGPU_app`
   and takes the global min-precise-digits across all 21 integrals, pinned at 3.69
   by B1's own inherent cancellation — below the 8-digit floor gate. So even a
   *real* per-integral demotion would be rejected until validation is also
   per-integral.

Both are precisely what the next work items address: **call-graph fan-out**
(routes callers to the per-integral shim, kills the no-op) and per-integral
validation (a floor scoped to the integral, not the global min). Phase 1's job —
prove the filtering + isolation pattern and surface these two blockers concretely —
is done.

## Deferred (next work items)

Call-graph-aware fan-out (`g_f1_B3` variants, regions only) → rename cascade →
per-integral validation scope → combine step (21 trees → one app) → dedup at
combine. Chains stay merged throughout.

---

# Addendum — BIN4 tail-sizing pass (2026-07-22)

_A second single-integral pass on a chain-heavy integral to establish the tail of
the 21-integral sizing extrapolation. **BIN4 (38,605 chains)**, not BIN2: the probe
"~32k mid-range BIN2" was wrong — the actual chain counts are BIN2=59,001 (max),
BIN1=56,723, BIN4=38,605, BIN0=32,302, BIN3=31,630. BIN4 is the true mid-of-pack._

## BIN4 run (report_5k, tol=10, seed=12345, 5k samples, run `20260722_230945`)

| metric | B1 (ref) | BIN4 |
|---|---|---|
| chains in report | 0 | 38,605 |
| status | success | **partial** (dr_k=40 tripped) |
| wall (strategy pass) | 794 s | **1432 s (~23.9 min)** |
| iterations | 40 (corr 2, spd 38) | 40 (**corr 40, spd 0**) |
| accepts / rejects | 0 / 40 | 0 / 40 |
| settled dist | double×32, dd×2 | double×82, **dd×14** |
| regions_at_threshold (chain-floored lines) | 32 | **38,673** |
| regions_dd_untested / unresolved | 0 / 0 | 13 / 1 |
| disk / pass | 48 MB | **80.6 MB** |
| filtered-report slice | 27 MB | **75 MB** |

## Where BIN4's cost goes

- **All 40 iterations were correctness** — chains flood the correctness (dd-promotion)
  queue and BIN4 tripped `dr_k=40` before draining it; **it never reached the speedup
  phase.** B1 spent 2 iters in correctness and 38 in speedup. This is the chain-driven
  divergence the pass was launched to measure.
- **Per-iter cost:** B1 19.9 s/iter (float/ff speedup), BIN4 35.8 s/iter — but only
  **14/40** BIN4 iters reached the Validator (`patcher_status=ok`, dd candidate built +
  run); the other 26 failed upstream and were cheap. The 14 dd-validate iters carry the
  weight.

## Reject-cause breakdown (demonstrates the Phase-2 manifest gap concretely)

BIN4's 40 "rejects" are **three distinct causes** — unlike B1, where all 40 were the
same `insufficient_fix`:

| patcher_status | count | meaning |
|---|---|---|
| `ok` (→ validator) | 14 | dd candidate built + run → all `insufficient_fix`, `cand_min_precise_digits = 3.691` |
| `llm_gen_failed` | 12 | LLM could not generate the `reformulate-identity` rewrite (R3 cascade) |
| `empty_candidate` | 14 | Patcher produced no candidate (non-fatal empty, dd promotions) |

The manifest's coarse `verdict` field collapses all three into `"reject"`. **This is
exactly the Phase-2 followup flagged during Phase 1** (see below): fine for B1, lossy
for BIN4. `patcher_status` + `verdict_reason` are in `iterations.jsonl` and just need
threading into `_decision_from_iter`.

Note also: the `cand_min_precise_digits = 3.691` on BIN4's validated candidates is the
**same global floor** as B1 — confirming the Validator is still whole-app (global min
across all 21 integrals, pinned by B1's cancellation), so BIN4 candidates fail the same
floor. Blocker #2 (whole-app validation) reconfirmed on a second integral.

## DD-oracle cache — warm, no invalidation

BIN4 reused B1's cached oracle (`dd_2229ec4…_seed12345_n5000.pkl`) and vanilla baseline
(`current_…_n5000.pkl`); **no oracle rebuild.** Per-iter cost is candidate build+run
only. Cache behaves as designed (atomic `.tmp`+replace, content-keyed) — no
invalidation bug on the second pass. Base repo pristine after the run.

## Revised 21-integral extrapolation

**Key correction: wall is `dr_k`-bounded, not chain-count-bounded.** Both B1 and BIN4
hit the 40-iter `dr_k` cap, so wall does **not** scale linearly with chain count —
scaling BIN* by chains (32k→59k) would overestimate. The right model is per-family,
per-iter cost × the `dr_k` cap:

- **16 B-family** (few/no chains, speedup-dominated) ≈ **794 s** each
- **5 BIN\*** (chain-heavy, correctness-dominated) ≈ **1432 s** each

| scheduling | wall | vs naive flat |
|---|---|---|
| sequential | 16·794 + 5·1432 = **~5.5 h** | flat-from-B1 4.6 h (under), flat-from-BIN4 8.4 h (over) |
| @4 workers | **~1.4 h** (LPT makespan; BIN* are the long poles) | — |
| @8 workers | **~0.85 h** (bounded below by one 1432 s task + load balance) | — |

**Caveat:** these are `dr_k=40` *partial*-pass walls for the BIN* family (BIN4 never
ran speedup). A full drain (correctness + speedup) needs a higher `dr_k` for chain-heavy
integrals — MEMORY already notes raising `dr_k` when the cascade phase's repeated
`llm_gen_failed`/`empty_candidate` trip the streak (here 26/40 non-validator failures
tripped it). A full-pass 21-run would be longer; this sizing is for the current config.

**Disk, by contrast, *is* chain-driven** (the filtered-report slice: BIN4 75 MB ≈
1.9 KB/chain). All-21 retained ≈ **~1.2 GB** (16·48 MB + BIN* report slices scaled by
chain count); naive 80.6 MB×21 = 1.7 GB overestimates. Report slices are ephemeral
(gitignored) and can be deleted post-pass to cut peak disk.

## Phase-2 followup recorded (not fixed in Phase 1)

`build_manifest`'s decision row keeps only a coarse `verdict`; it drops `patcher_status`
(build_failed / llm_gen_failed / empty_candidate / timeout) and `verdict_reason`
(insufficient_fix / regressed / tolerance). BIN4 (3 distinct causes) makes the loss
concrete. Phase 2's fan-out will produce genuinely mixed causes → add both fields to
`agents/per_integral_orchestrator/manifest.py:_decision_from_iter`. Not blocking.
