# Known limitations

Current, verified-against-code limitations of the pipeline. Each entry names
the owning code so this list stays checkable; when an entry is fixed, delete
it here.

## Strategy / report loading

- **No streaming report loader.** `agents/strategy/characterization.py`
  (`load_regions`, `load_chains`) each `json.loads` the whole characterization
  report, independently — the file is parsed twice. Fine at n=1000; a
  full-scale (100k, ~13.7 GB) report will OOM.
- **`max_iters` under-counts.** `llm_gen_failed` iterations do not consume
  budget (`agents/strategy/dispatch.py`, `counts_budget=False` — and its
  `log_tag` is the mislabeled `llm_capacity`), and there is no hard cap on
  total loop iterations. Partially mitigated by `diminishing_returns_k=60`
  (`agents/config.py`), raised precisely because chain phases produce long
  `llm_gen_failed` streaks.
- **`precision_distribution` vs `precision_assignment` are not reconciled.**
  The former counts one entry per region (`region_final`); the latter appends
  per accepted transition plus once per chain-claimed line
  (`agents/strategy/agent.py`). Different denominators — expect off-by-ones.
- **Chain intents drive on the representative line only.** Real multi-line
  chain intents for the Patcher are deferred; `ChainRecord.walk_record()`
  returns a single-target record on `lines[0]`
  (`agents/strategy/characterization.py`).
- **The cascade victim is the DAG sink only** (consumer migration not
  modeled), and region variables fall back to the loose `prov_vars` union on
  reports predating `region_local_vars` —
  `agents/strategy/characterization.py`.
- **No qf integrator.** The region walk cannot demote/promote to quad-float;
  qf is reachable only via the whole-TU flip
  (`agents/strategy/models.py::REGION_REALIZABLE`,
  `agents/patcher/tu_emit.py`).

## Validator

- **`validate()` requires `accepted_patches == []`.** Scoring a candidate
  stacked on prior accepted patches needs the deferred master→ddfun line map
  (`agents/validator/validate.py` raises `NotImplementedError`). This limits
  true sequential layering on the region path; the solver
  (`agents/solver/`) layers on the accumulated tree instead.
- **Scorer 2b stub.** `delta_adversarial` is `None` when the adversarial
  slice is empty; `snapshot_battery_spec` is shaped for future work
  (`agents/validator/scorer.py`).
- **Tail coverage is bounded by the characterization distribution.** Failure
  modes on inputs outside it are caught by no battery (see
  `runs/qcdloop/PIPELINE_v1.md`).

## Patcher / integrators

- **Whole-app `integrate()` is unimplemented for ff and float** (regional
  path only; `agents/ff_integrator/agent.py`,
  `agents/float_integrator/agent.py` raise `NotImplementedError`). The
  chain integrator's whole-app scope raises by design (a chain is
  intrinsically regional). The dd integrator's whole-app path is a bounded
  stub that verifies and returns the hand-written `kokkosMaths_dd.h` triple.
- **Two open accept-rate levers** (flagged for the next large run):
  1. The R4 escape-hatch detector on un-vendored DD constants is
     diagnostic-only — it never changes dispatch
     (`agents/patcher/agent.py`, `agents/patcher/instantiation_gate.py`);
     `agents/shared/constant_derive.py` narrows the exposure by pre-deriving
     the common constants but does not close it.
  2. The Kokkos math overload gap: a promoted `DoubleDouble` flowing into
     `Kokkos::fabs` / `Kokkos_Complex.hpp` internals with no DD overload;
     `agents/integrator_base/shallow_wrapper.py` rewrites
     `Kokkos::abs → Kokkos::Experimental::abs` as the workaround.
- **`dispatch.is_retryable_misgen` returns `True` unconditionally** — retry
  everything, pending failure-mode data (`agents/patcher/dispatch.py`).
- **libclang is the preferred backend but not always available.** The
  exercised path on the cluster image is the comment/string-aware token-lexer
  fallback (`agents/patcher/edits.py`, `agents/shared/region_scan.py`). The
  tracked journal has no LHS/assignment-target field, which is why region
  *write*-sets are recovered by source scan at all
  (`agents/shared/stability_reducer.py`, `agents/shared/region_scan.py`).

## Characterizer

- **`TRACKED_HERE` call-site forwarding is unavailable for kernels calling
  `std::` math directly.** The `log_sum_exp` fixture only works via a 2-arg
  overload injected into `namespace std`, which no real user kernel would
  write; for un-editable kernels, ops stay shim-attributed and recall against
  kernel-named hints misses (see `agents/shared/RECALL_NOTES.md`, which also
  carries the open decision on that fixture).
- **Not characterized:** the `fndd` / `ddilog` leaf dependencies.
- **Missing tests:** no characterizer end-to-end test, no prompt snapshot
  tests (`tests/agents/` covers the parser, retry loop, and reducer).

## Build/run

- **The build/run agent is a deterministic subprocess wrapper** with a
  hardcoded HPC module chain (overridable via `PIPELINE_MODULE_LIST` /
  `PIPELINE_MODULE_USE_PATH`); LLM-driven framework detection / env
  extraction does not exist (`agents/build_run/agent.py`). Whole-app
  build+run lives in `agents/validator/runner.py`, not in the build/run
  agent.

## third_party

- **`LICENSES/` texts are not vendored** for the DD/FF (DHB) and QF
  (LBNL-BSD) lineages referenced by `third_party/include/README.md`.
