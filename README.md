# Agentic Mixed-Precision Demo

An LLM-assisted multi-agent pipeline that finds safe mixed-precision assignments
for C++ scientific computing code. Given instrumented kernels and per-argument
input ranges, it measures per-operation numerical sensitivity with the
`Tracked<T>` library, reduces the telemetry into a per-code-region stability
report, then walks a precision ladder per target — promoting unstable regions to
extended precision and demoting stable ones to cheaper precision — validating
every step against a double-double (DD) oracle.

Everything in this README describes code that exists on this branch. Current
gaps and deferred work live in [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md).

---

## The precision ladder

`agents/strategy/models.py` defines the cost-ordered ladder:

| rung | representation | ~digits | exponent range | mechanism |
|---|---|---|---|---|
| `float` | 1×FP32 | ~7 | FP32 | plain type edit, or regional LLM shim (`float_integrator`) |
| `ff` | 2×FP32 (float-float) | ~14 | **FP32** | regional LLM shim (`ff_integrator`) or whole-TU flip |
| `double` | 1×FP64 | ~15–16 | FP64 | baseline |
| `qf` | 4×FP32 (quad-float) | ~29 | **FP32** | **whole-TU flip only** (no regional qf integrator) |
| `dd` | 2×FP64 (double-double) | ~30–31 | FP64 | regional LLM shim (`dd_integrator`, `chain_integrator`) or whole-TU flip |

Two ladder facts the code enforces:

- `FP32_FAMILY = {float, ff, qf}` share FP32's exponent range. The
  characterizer's `value_range_ok_for_float` flag is a **hard range guard** on
  the whole family in both walk directions (a wider mantissa does not widen
  range).
- `REGION_REALIZABLE = {float, ff, double, dd}` — the region walk has no qf
  integrator; qf is reachable only via the whole-TU precision flip
  (`agents/patcher/tu_emit.py` + `third_party/include/kokkosMaths_qf.h`).

The extended-precision types themselves (`Kokkos::Experimental::DoubleDouble` /
`FloatFloat` / `QuadFloat` + complex containers + `ql::`-surface enrichment
headers) are vendored in `third_party/include/` (provenance: `UPSTREAM.sha`).

---

## Pipeline

```mermaid
flowchart TB
    IN[/"Kernels + input ranges<br/>+ build env"/] --> CH

    subgraph CH[Characterizer]
        direction TB
        C1["spec build (regex, pure Python)"] --> C2["micro-driver generation (LLM)"]
        C2 --> C3["build + run (deterministic)"]
        C3 -. configure/build fail .-> C2
        C3 --> C4["journal parse -> sensitivity profile"]
        C4 --> C5["symbolic overlay (LLM, optional,<br/>never gates)"]
    end

    CH --> RED["stability_reducer (shard map)<br/>+ fast_merge (reduce)"]
    RED --> REP[/"stability report (schema v2):<br/>regions, cascade chains, signal class,<br/>predicted_rel_err_if_*, value_range_ok_for_float"/]
    REP -. emit_tail_offsets.py .-> REP

    REP --> STRAT

    subgraph STRAT["Strategy (mechanical — no LLM)"]
        direction TB
        S0{strategy_mode}
        S0 -->|tu_only 'default'| TU["per-integral whole-TU walk:<br/>upshift (qf, dd), downshift (float, ff)<br/>via injected tu_measure_fn"]
        S0 -->|region| RW["two-phase region walk:<br/>correctness queue then speedup queue,<br/>chains deduped by representative line"]
        RW --> P["Patcher (LLM integrators +<br/>deterministic boundary/gates)"]
        P --> V["Validator: 3 builds vs DD oracle,<br/>random battery + regression-relative<br/>tail battery"]
        V --> RW
    end

    STRAT --> OUT[/"report.json + report.md<br/>+ per-iteration log + cumulative diff"/]
```

The LangGraph graph itself is deliberately thin — `agents/orchestrator.py`
wires `characterize → strategy → END` over the shared `PipelineState`
TypedDict (`agents/state.py`). The Patcher and Validator are **not** graph
nodes: Strategy owns the remediation loop and drives them as injected
callables (`patcher_fn` / `validator_fn`; for `tu_only` mode,
`tu_measure_fn` / `tu_promote_fn`). Design of record:
[`docs/strategy_patcher_design.md`](docs/strategy_patcher_design.md).

---

## Components

### Characterizer (`agents/characterizer/`)

1. **Spec build** — regex-based signature parse, parameter role
   classification, framework detection (plain C++ / Kokkos serial).
2. **Driver generation (LLM)** — writes a self-contained micro-driver that
   includes the kernel, wraps inputs with `tracked::track()`, samples the
   user-provided ranges, and flushes a JSONL journal. Conventions live in
   `prompts/driver_gen.txt`.
3. **Build + run** — via `agents/build_run/` (deterministic subprocess
   wrapper). Configure/build failures feed stderr back to the generator and
   retry up to `--max-driver-attempts`; runtime failures never retry.
4. **Journal parse** — `log_parser.py` aggregates per-op / per-line /
   per-variable condition numbers into `sensitivity_profile.json`.
5. **Symbolic overlay (LLM, optional)** — flags known unstable idioms as
   `symbolic_hints.json`; best-effort, never gates.

For whole-app characterization the tracked journal is reduced in-process by
`agents/shared/stability_reducer.py` (mergeable per-region stability report,
`schema_version: 2`: signal class, forward-cone `max_sensitivity`,
`predicted_rel_err_if_{float,ff}`, `value_range_ok_for_float`, cascade
chains) and merged by `agents/shared/fast_merge.py`.
`runs/qcdloop/emit_tail_offsets.py` augments a report with per-integral
adversarial `tail_samples` guarded by an input determinism hash.

### Strategy (`agents/strategy/`)

Mechanical — no LLM calls. Two modes (`StrategyConfig.strategy_mode`):

- **`tu_only`** (default): per integral, measure the baseline, then **upshift**
  through `(qf, dd)` if below tolerance (cheapest-sufficient, first accept
  wins) and **downshift** through `(float, ff)` if comfortably above,
  gated by the FP32-family range guard and `predicted_rel_err_if_float`.
  Measurement is injected (`tu_measure_fn`; qcdloop provider:
  `runs/qcdloop/tu_provider.py`).
- **`region`**: two-phase walk over ranked queues from the stability report —
  a correctness queue (local cancellation, then cascade chains, then
  log-near-root) walked **up** the ladder, then a speedup queue (stable
  regions with `predicted_rel_err_if_ff` under threshold, flop-weight-ordered
  via `ratio_multipliers.json`) walked **down**. Budget is split
  correctness/speedup (70/30 by default, unused correctness iterations spill
  forward). Cascade chains are grouped and walked once per representative
  line; accepted precision distributes to all chain lines via a
  `required_by` ledger.

Report-mining prunes (WI1 hard range gate, WI2 pred-float telemetry, WI3
flop-weighted ordering) are on by default; `STRATEGY_DISABLE_REPORT_PRUNES=1`
is the single kill switch.

### Patcher (`agents/patcher/`)

Translates one remediation intent into one committed candidate; it runs
builds as gates but renders no verdicts. Five dispatch paths
(`dispatch.py`): plain type edit, revert, regional LLM shim, LLM line
rewrite (`reformulate-kahan` / `reformulate-identity`), and chain-scoped
promotion (`chain_promote.py`: value closure, carrier widening, leaf
promotion via `clonable_leaf.py` + `integrator_base/shallow_wrapper.py`).
The whole-TU precision flip lives in `precision_flip.py` / `tu_emit.py` /
`flip_gate.py`. Gates: vanilla build + 1-sample smoke (compile / row-count /
NaN), variant wiring, no-silent-bypass, `nm` symbol presence. Bounded LLM
retry with backoff; deterministic failures never retry.

### Validator (`agents/validator/`)

Three builds on bit-identical inputs: DD ground truth (from the
`ddfun_enabled` qcdloop fork, with `third_party/include/` as the header
source of truth via `runner.materialize_dd_headers`), the current baseline,
and the candidate. Scores per-sample precise digits against the DD oracle
(`precise_digits.py`, per-sample `ref_scale` so analytic zeros don't
penalize). Verdict = regression guard on combined random+tail minima, plus
an absolute tolerance floor on the random battery. The **tail battery**
re-tests the characterizer-flagged adversarial offsets, verifies the frozen
input determinism hash (loud `DeterminismMismatch`, never a silent
fallback), and is regression-relative by design — adversarial offsets embed
workload physics ceilings no patch owns. `scorer.py` emits measurement-only
`(region, rung) → delta` manifest cells for the solver.

### Integrators (`agents/*_integrator/`, `agents/integrator_base/`)

| package | regional (per-region LLM shim) | whole-app |
|---|---|---|
| `tracked_integrator` | — | real (LLM interop shim + libclang line-scope injector) |
| `dd_integrator` | real | bounded stub (verifies the hand-written DD triple) |
| `ff_integrator` | real | not implemented |
| `float_integrator` | real (native target: no bridges/constants) | not implemented |
| `chain_integrator` | real (dd ruleset + chain-boundary rule C9) | n/a by design |

`integrator_base/` is the shared machinery: deterministic boundary-patch
synthesis (promote reads / demote writes, multi-limb reconstruction incl.
4-limb qf), the regional LLM engine with closed-include lint and
namespace-qualified math bridges, compiler-error-driven int↔tracked
annotation (C8), shim merging into one canonical per-family shim, and
source-hash staleness caching.

### Solver (`agents/solver/`) and per-integral orchestrator

`agents/solver/` greedily layers measured scorer-manifest cells
cheapest-first (`float < ff < dd`) onto an accumulated tree under a
regression-relative gate (see **Loop semantics**). Drivers:
`runs/qcdloop/run_solver_stage{1,2}.py`.
`agents/per_integral_orchestrator/` runs a filter → clone → build-gate →
pipeline pass per integral with a manifest (qcdloop wiring:
`runs/qcdloop/run_all_integrals.py`).

## Loop semantics

Locked policy (cited by `agents/solver/`):

- **Sequential layering** — each accepted patch becomes part of the new
  baseline; the next candidate is tested on top of the accumulated state.
- **No combining strategies** — one at a time, validate, keep or revert.
- **Fixed report** — characterize once, walk the queues; re-characterization
  between accepts is disabled (`recharacterize_after_n` is effectively ∞).
- Stops: queue exhaustion, per-phase iteration caps, wall-clock / LLM-token
  ceilings, or `diminishing_returns_k` consecutive non-accepts.

---

## Running

### Environment

- Python ≥ 3.10 (3.12 used in practice): `pip install -r requirements-langgraph.txt`
- CMake ≥ 3.18; for Kokkos-backed targets a Serial Kokkos install (default
  `~/kokkos-install`)
- An Anthropic-API-compatible endpoint for the LLM stages. Default
  configuration targets a local Argo proxy
  (`scripts/setup_argo_proxy.sh`): `ANTHROPIC_BASE_URL`
  (default `http://127.0.0.1:8083/argoapi/`), `ANTHROPIC_AUTH_TOKEN` (or
  `ARGO_USERNAME`), `ARGO_MODEL` (default `claudeopus47`). Direct Anthropic
  use = override base URL + model.
- `third_party/tracked/` is a git subtree (no submodule init); sync with
  `git subtree pull --prefix=third_party/tracked https://github.com/ReetBarik/Tracked-Error-Propagation-Datatype-Demo.git main --squash`

### Characterizer slice (single kernel)

```bash
python -m agents.cli characterize \
  --kernel tests/agents/fixtures/kernels/cancellation.cpp \
  --kernel-name cancellation_check \
  --ranges-yaml tests/agents/fixtures/input_ranges/cancellation.yaml \
  --out runs/out/cancellation
```

The six calibration fixtures (`runs/{cancellation,cln,kahan,lnrat,log_sum_exp,naive_variance}/`)
regenerate end-to-end with `scripts/regen_recall.sh`, which rebuilds each
fixture, re-parses its journal (`agents.shared.regen_profile`), and checks
hint recall (`agents.shared.recall_verifier` → `runs/recall_summary.json`).

### qcdloop strategy walk (whole app)

```bash
# characterize (chunked, sharded reduce):
python runs/qcdloop/run_chunked.py ...
# augment with adversarial tail offsets:
python runs/qcdloop/emit_tail_offsets.py --report runs/qcdloop/report_10k.json
# walk (production config wraps run_strategy_e2e.py):
runs/qcdloop/run_strategy_10k.sh
```

`runs/qcdloop/run_strategy_e2e.py` is the real strategy entry point: it
assembles a `PipelineState` with a fixed report path plus the injected
Patcher/Validator (region mode) or the TU provider (`--strategy-mode
tu_only`, default) and calls `agents.strategy.agent.run` directly. Artifacts
land under `runs/qcdloop/strategy/<run_id>/`.

### Tests

```bash
pytest -m "not llm"      # offline suite
pytest -m llm            # requires the live proxy (ANTHROPIC_AUTH_TOKEN set)
```

Markers are registered in `tests/conftest.py` (`llm`, `kokkos`);
kokkos-dependent tests self-skip when `~/kokkos-install` is absent.

---

## Repository layout

```
agents/
├── cli.py, __main__.py                  # characterizer front door
├── config.py                            # env defaults + PipelineConfig / StrategyConfig / StrategyBudget
├── orchestrator.py                      # LangGraph graph: characterize -> strategy -> END
├── state.py                             # PipelineState TypedDict
├── build_run/                           # micro-driver build+run (deterministic)
├── characterizer/                       # spec / driver-gen (LLM) / log parse / symbolic overlay (LLM)
├── strategy/                            # models (ladder), ranking, walk, tu_walk, agent (the loop)
├── patcher/                             # dispatch, edits, gates, rewrites, chain_promote, fanout,
│                                        #   call_graph, tu_emit, precision_flip, flip_gate, shim_synth
├── validator/                           # validate, runner (DD oracle staging), tail, scorer, precise_digits
├── integrator_base/                     # boundary, regional LLM engine, c8, shim_merge, shallow_wrapper, llm
├── tracked_integrator/ dd_integrator/ ff_integrator/ float_integrator/ chain_integrator/
├── per_integral_orchestrator/           # per-integral pass: filter -> clone -> gate -> manifest
├── shared/                              # stability_reducer, fast_merge, region_scan, bound_decomposition,
│                                        #   constant_derive, recall_verifier, regen_profile, type_resolve
└── solver/                              # greedy manifest layering (queue, solver, report)

docs/
├── strategy_patcher_design.md           # design of record (cited from code docstrings)
├── KNOWN_LIMITATIONS.md                 # current gaps, verified against code
└── slides/                              # 8-slide pipeline walkthrough (SVG)

runs/                                    # fixtures, qcdloop tooling + validated reports (see runs/README.md)
scripts/                                 # setup_argo_proxy.sh, regen_recall.sh, one_off/ generators
src/                                     # test-pinned header fixtures (kokkosUtils.h)
tests/                                   # offline suite; llm/kokkos-marked integration tests
third_party/
├── include/                             # vendored dd/ff/qf headers + ql:: enrichment (UPSTREAM.sha)
└── tracked/                             # Tracked<T> instrumentation library (git subtree)
```

## Status

| component | status |
|---|---|
| Characterizer (spec, LLM driver-gen, retry loop, parser, overlay) | implemented |
| Stability reducer + fast merge + tail-offset emitter | implemented |
| Strategy (tu_only + region modes, two-phase walk, chains, prunes) | implemented |
| Patcher (5 dispatch paths incl. chain promotion + whole-TU flip) | implemented |
| Validator (3-build precise-digits, tail battery, scorer manifests) | implemented |
| Solver + per-integral orchestrator | implemented |
| Regional integrators (dd / ff / float / chain) | implemented |
| Whole-app integrate() for ff / float; qf regional integrator | not implemented (see KNOWN_LIMITATIONS) |
| Build/run agent | deterministic wrapper (no LLM env detection) |

Validated runs and their write-ups live under `runs/qcdloop/`:
`PIPELINE_v1.md` (last full region-walk validation, Wave-3 prunes + tail
battery), `PHASE_2_TU_E2E_REFSCALE_2026-07-30.md` (whole-TU walk),
`HEADER_REFRESH_2026-08-13.md` (vendored-header provenance), and
`QF_INTEGRATION_2026-08-13.md` (the qf rung, current routing). The six
design docs of record for the chain/leaf/carrier/flip machinery are the
`*_DESIGN.md` files in the same directory.
