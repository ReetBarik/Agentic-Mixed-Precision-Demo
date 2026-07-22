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
