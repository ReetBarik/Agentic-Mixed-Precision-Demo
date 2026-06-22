# Plan: Characterizer agent — first vertical slice

**Branch:** `langgraph-agents` (continuing).
**Sibling dep:** `kokkos-extended-precision-demo` branch `tracked` at v1.1+
(must include the `opaque` fix — commit `36a94ad` or later).

## Goal

Implement the characterizer agent end-to-end with deterministic pass-through
stubs for every other agent in the pipeline. The slice runs the full
LangGraph orchestration on real kernels and produces a sensitivity profile
JSON. By the end:

- `python -m agents.cli characterize --kernel <path> --kernel-name <fn> \
   --ranges <spec>` produces a usable profile for both Tracked's calibration
  kernels (cancellation, naive variance, etc.) and for QCDLoop's `cLn`.
- The Tracked-calibration fixtures serve as the e2e smoke test (we know the
  expected per-op cond signatures from Tracked's own test assertions).
- `cln` exercises the interop-vs-opaque decision the characterizer must make
  for any real-world kernel using framework math (Kokkos/std/etc.).

## Out of scope

- Strategy / patcher / validator agents beyond identity stubs.
- LLM-driven build/run agent (deterministic subprocess wrapper for now).
- Framework detection beyond plain-C++17 and Kokkos-Serial.
- Re-characterization between accepted patches (deferred to v2 per top PLAN).
- Strategy combining (also v2).

## Architecture

LangGraph, shared state with reducers from day one on every accumulating
field. Single model (Claude Opus 4.7 via Argo) for all LLM-driven nodes in
this slice.

## Repository layout

```
agents/
  __init__.py
  state.py                  # PipelineState TypedDict + reducers
  config.py                 # PipelineConfig dataclass, CLI flag mapping
  cli.py                    # entrypoint
  characterizer/
    __init__.py
    agent.py                # LangGraph node: top-level characterizer run()
    spec.py                 # InstrumentationSpec dataclass
    driver_gen.py           # LLM-driven driver generation
    log_parser.py           # JSONL → SensitivityProfile (pure Python)
    symbolic_overlay.py     # LLM-driven idiom flagger
    profile.py              # SensitivityProfile + OpRecord dataclasses
    prompts/
      driver_gen.txt        # built from CHARACTERIZER_NOTES.md
      symbolic_overlay.txt
  build_run/
    __init__.py
    agent.py                # deterministic subprocess wrapper (no LLM yet)
    cmake_template.cmake    # rendered per run
  strategy/
    __init__.py
    agent.py                # stub: state | {strategy_queue: []}
  patcher/
    __init__.py
    agent.py                # stub: identity
  validator/
    __init__.py
    agent.py                # stub: identity
  orchestrator.py           # LangGraph wiring + conditional edges

third_party/
  tracked/                  # git submodule → kokkos-extended-precision-demo @ tracked

tests/
  agents/
    test_characterizer_e2e.py
    test_driver_gen.py
    test_log_parser.py
    test_symbolic_overlay.py
    test_build_run_stub.py
    fixtures/
      kernels/
        cancellation.cpp          # adapted from tracked/tests/tracked/test_cancellation
        naive_variance.cpp
        log_sum_exp.cpp
        kahan.cpp
        cln_kernel.hpp            # the cLn function alone, headers stripped
      input_ranges/
        cancellation.yaml
        naive_variance.yaml
        log_sum_exp.yaml
        kahan.yaml
        cln_kernel.yaml
  conftest.py

scripts/
  init_submodules.sh        # convenience wrapper
  run_characterizer.py      # slice CLI shortcut

requirements-langgraph.txt  # langgraph, pydantic, pyyaml, anthropic-via-argo

CHARACTERIZER_NOTES.md      # already in repo; lives at root
```

## Shared state schema (`agents/state.py`)

```python
from typing import Annotated, TypedDict
from operator import add
from agents.characterizer.profile import SensitivityProfile, SymbolicHint
from agents.characterizer.spec import InstrumentationSpec

class PipelineState(TypedDict):
    # Inputs
    source_files: list[str]
    kernel_name: str
    input_ranges: dict[str, tuple[float, float]]
    build_instructions: str
    whole_app_driver: str | None
    config: "PipelineConfig"

    # Characterizer outputs (LIST for fan-out support from day 1).
    sensitivity_profiles: Annotated[list[SensitivityProfile], add]
    symbolic_hints: Annotated[list[SymbolicHint], add]
    instrumentation_specs: Annotated[list[InstrumentationSpec], add]
    journal_paths: Annotated[list[str], add]

    # Strategy / patcher / validator (stubbed for this slice)
    strategy_queue: list                  # plain — single writer
    current_patch: dict | None            # plain — single writer
    validation_result: dict | None        # plain — single writer
    accepted_patches: Annotated[list[dict], add]
    rejected_patches: Annotated[list[dict], add]

    # Bookkeeping
    iteration: int
    errors: Annotated[list[str], add]
```

**Reducer policy:** every field that could plausibly receive writes from two
nodes in parallel uses `Annotated[..., add]` (list-append) or a custom
reducer. Scalar fields (`current_patch`, `validation_result`) stay plain
with a code comment noting "single writer in v1; if you parallelize the
strategy loop, this needs a reducer."

**Why lists for characterization outputs even though v1 only runs once per
kernel:** the orchestrator may be invoked with multiple kernels (whole-app
mode), and even single-kernel runs may eventually fan out across input-range
slices. Starting with lists costs nothing now, costs a refactor later.

## Sensitivity profile schema (`agents/characterizer/profile.py`)

```python
@dataclass
class OpRecord:
    op: str                      # "add", "sub", "mul", "opaque", etc.
    location: str                # "file:fn:line" or "" if not captured
    max_cond: float
    max_rel_err: float
    sample_count: int
    provenance_union: set[str]
    flagged: bool                # max_cond > config.flag_threshold

@dataclass
class SensitivityProfile:
    kernel: str
    samples_run: int
    per_op: list[OpRecord]                # sorted by max_cond desc
    per_line: dict[str, OpRecord]         # rolled up by source location
    per_variable: dict[str, float]        # var → max cond it appeared in
    top_hotspots: list[OpRecord]          # top-N by max_cond (default 10)
    opaque_coverage: float                # fraction of records that are opaque
    notes: list[str]                      # e.g. "opaque coverage 80% — under-characterized"

@dataclass
class SymbolicHint:
    idiom: str                   # "log_sum_exp_naive", "naive_variance", ...
    location: str                # source range "file:fn:start-end"
    severity: str                # "low" | "medium" | "high"
    suggested_rewrite: str       # short prose
```

## Characterizer internal flow

```
spec_build → driver_gen → invoke build_run → log_parse → symbolic_overlay → emit profile
```

### Step 1: `spec_build`

Pure Python. Reads kernel signature from `source_files` (parse via libclang
or regex for v1 — libclang is more robust but adds a system dep; regex is
fine for the limited signature shapes we need). Builds:

```python
@dataclass
class InstrumentationSpec:
    kernel_name: str
    kernel_signature: str                  # raw text
    parameter_types: list[tuple[str, str]] # [(arg_name, type), ...]
    input_ranges: dict[str, tuple[float, float]]
    template_instantiation: dict[str, str] # e.g., {"TOutput": "tracked::Complex<double>"}
    sample_count: int                      # default 512
    framework: str                         # "plain-cpp" | "kokkos-serial"
    detected_dispatchers: list[str]        # ["kAbs", "kLog", ...] if any
```

Template instantiation rules:
- Real scalar param → `Tracked<double>`
- Complex scalar param → `Complex<double>`
- Template params (`TOutput`, `TMass`, `TScale` etc.) inferred from how
  they're used in the body — if param is `TOutput const& z` and body
  calls `Imag(z)`/`Real(z)`, assume complex; otherwise real. This is
  heuristic for v1; the agent can override.

### Step 2: `driver_gen` (LLM-driven)

Prompt template at `agents/characterizer/prompts/driver_gen.txt`, built
**directly from `CHARACTERIZER_NOTES.md`** to avoid baking in QCDLoop
specifics. The prompt receives:

- `kernel_signature`
- `input_ranges`
- `framework` and detected dispatchers
- A list of non-templatable math calls the kernel performs (detected by
  scanning the body for known framework patterns: `Kokkos::*`, `std::*`,
  `cuda::*`)
- The `InstrumentationSpec`

The LLM must produce, in a single response:

1. A `.cpp` micro-driver source string.
2. A list of per-call decisions: for each non-templatable math call, did
   it choose `interop_shim`, `opaque_wrap`, or `inline_reimplementation`,
   with a one-line justification.
3. Any helper headers it decided to inline-reimplement (with the
   reimplementation source).

Output schema (Pydantic):

```python
class DriverGenOutput(BaseModel):
    driver_source: str
    interop_decisions: list[InteropDecision]
    inlined_helpers: dict[str, str]   # original_header_path → replacement_source
    notes: str

class InteropDecision(BaseModel):
    call_site: str                    # "Kokkos::log @ cln_kernel.hpp:12"
    strategy: Literal["interop_shim", "opaque_wrap", "inline_reimpl"]
    justification: str
```

Default policy (encoded in the prompt):
- Prefer `interop_shim` for stable, well-known framework math (log, exp,
  abs, sqrt, sin, cos, atan2).
- Use `opaque_wrap` when interior numerics are unlikely to matter for the
  characterization run (e.g., a vendor BLAS gemm where the kernel does
  one call and downstream math is what we care about).
- Use `inline_reimpl` for header-pollution cases where pulling in a header
  forces non-templatable instantiations that the shim can't reach.

The slice's CLI accepts `--strategy-override interop|opaque|inline` to
force a single strategy for testing.

### Step 3: invoke `build_run` (deterministic stub)

```python
class BuildRunStub:
    def __init__(self, tracked_root: Path, kokkos_root: Path | None):
        ...

    def build_and_run(self, driver_source: str, work_dir: Path,
                       framework: str) -> RunResult:
        # 1. Write driver_source to work_dir/src/micro_driver.cpp
        # 2. Render CMakeLists from agents/build_run/cmake_template.cmake.
        # 3. cmake -B build -DCMAKE_BUILD_TYPE=Release
        #    cmake --build build -j
        #    ./build/micro_driver
        # 4. Return RunResult{returncode, stdout, stderr, journal_path}
```

CMake template handles two cases:
- `framework="plain-cpp"` — links only Tracked headers from
  `third_party/tracked/include/`.
- `framework="kokkos-serial"` — also `find_package(Kokkos REQUIRED)`,
  links `Kokkos::kokkos`.

Build/run errors surface as `RunResult.returncode != 0` with stdout/stderr
captured; characterizer logs them to `state["errors"]` and aborts that
kernel cleanly without crashing the pipeline.

**No LLM in this stub.** The real LLM-driven build/run agent will replace
this with the same interface later — that's the contract.

### Step 4: `log_parse` (pure Python)

Reads `RunResult.journal_path` (JSONL), rolls up:

- By `(op, location)` → `OpRecord` with max cond, max rel_err, sum sample
  counts, union provenance.
- By location → `per_line`.
- By variable (from provenance sets) → `per_variable`.
- Top N by `max_cond` → `top_hotspots` (default N=10, configurable).
- `opaque_coverage` = `count(op="opaque") / total_records`.
- If `opaque_coverage > 0.5`, add note "kernel is heavily opaque; consider
  expanding Tracked op coverage or switching opaque→interop for major
  framework calls."

`flagged` threshold defaults to `cond > 1e8` (= 26+ bits lost), CLI
overridable.

### Step 5: `symbolic_overlay` (LLM-driven, optional)

Cheap LLM call. Reads the kernel source, returns a list of detected
unstable idioms with severity. Strict timeout (10s) and error tolerance:
on any failure, log a warning and continue with empty hints. Never gates
the pipeline.

Prompt at `agents/characterizer/prompts/symbolic_overlay.txt`. Asks for:
- Known patterns: catastrophic cancellation (`a - b` where `a ≈ b`),
  naive variance (`E[X²] - E[X]²`), naive log-sum-exp, large-magnitude
  summation, division by near-zero.
- Output as JSON list of `SymbolicHint`.

### Step 6: emit profile

Writes `<run_dir>/sensitivity_profile.json`, appends the typed object to
`state["sensitivity_profiles"]`.

## Build/run stub (`agents/build_run/`)

```python
@dataclass
class BuildSpec:
    framework: str                         # "plain-cpp" | "kokkos-serial"
    extra_include_dirs: list[str]
    extra_link_libs: list[str]
    cxx_standard: int                      # 17 or 20
    cmake_overrides: dict[str, str]

@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    journal_path: Path | None              # None if run failed before flush
    work_dir: Path                         # for debugging
```

`cmake_template.cmake` is rendered with Jinja2 (or just `str.format`):

```cmake
cmake_minimum_required(VERSION 3.18)
project(micro_driver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD {cxx_standard})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

{find_package_lines}

add_executable(micro_driver src/micro_driver.cpp)
target_include_directories(micro_driver PRIVATE
    {tracked_include_dir}
    {extra_include_dirs}
)
target_link_libraries(micro_driver PRIVATE {link_libs})
```

## Other agent stubs

```python
# agents/strategy/agent.py
def run(state: PipelineState) -> PipelineState:
    return {"strategy_queue": []}        # empty → orchestrator goes to END

# agents/patcher/agent.py
def run(state: PipelineState) -> PipelineState:
    return {}                            # identity, never called in this slice

# agents/validator/agent.py
def run(state: PipelineState) -> PipelineState:
    return {}                            # identity, never called in this slice
```

Defined so the graph topology compiles; the conditional edge after
`strategy` always routes to END in this slice.

## Orchestrator (`agents/orchestrator.py`)

```python
from langgraph.graph import StateGraph, END

def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("characterize", characterizer.run)
    g.add_node("strategy", strategy.run)
    g.add_node("patcher", patcher.run)
    g.add_node("validate", validator.run)

    g.set_entry_point("characterize")
    g.add_edge("characterize", "strategy")
    g.add_conditional_edges("strategy",
        lambda s: END if not s["strategy_queue"] else "patcher")
    g.add_edge("patcher", "validate")
    g.add_conditional_edges("validate",
        lambda s: END if not s["strategy_queue"] else "patcher")

    return g.compile()
```

For this slice the conditional always lands on END after `strategy`.

## CLI

```
python -m agents.cli characterize \
  --kernel third_party/tracked/tests/tracked/test_cancellation.cpp \
  --kernel-name cancellation_check \
  --ranges-yaml tests/agents/fixtures/input_ranges/cancellation.yaml \
  --samples 512 \
  --out runs/cancellation
```

Writes `runs/<name>/` containing:
- `micro_driver.cpp` — generated
- `interop_decisions.json` — LLM's per-call strategy choices
- `journal.jsonl` — Tracked output
- `sensitivity_profile.json` — characterizer output
- `symbolic_hints.json` — overlay output (or empty array)
- `build.log` — build/run subprocess stdout+stderr

## Test suite

### `test_log_parser.py` — pure unit, no LLM, no build

Synthetic JSONL fixtures → expected `SensitivityProfile`. Cover:
- Empty journal
- Single op
- Multiple ops, same location → roll-up
- Multiple samples, same op → max-cond and sample-count aggregation
- Heavy opaque coverage → notes triggered
- Provenance union and per-variable rollup

### `test_driver_gen.py` — LLM-required, marked `@pytest.mark.llm`

For each fixture kernel, run `driver_gen` and:
- Assert generated source compiles (via the build/run stub).
- Snapshot the source to `tests/agents/snapshots/` — fail loudly when the
  prompt changes; manual re-bless via `--bless`.
- Assert `interop_decisions` covers every non-templatable call in the
  kernel.

### `test_symbolic_overlay.py` — LLM-required

Feed kernels with known idioms, assert hint detection:
- `naive_variance.cpp` → hint with `idiom="naive_variance"`
- `log_sum_exp.cpp` → hint with `idiom="log_sum_exp_naive"`
- `cancellation.cpp` → hint with `idiom="catastrophic_cancellation"`

### `test_build_run_stub.py` — no LLM, requires cmake + Tracked submodule

Build a hand-written minimal driver, assert it runs and produces a JSONL.
Smoke test for the subprocess wrapper.

### `test_characterizer_e2e.py` — LLM + full pipeline

For each calibration fixture:
- Cancellation: assert `per_op` contains a `sub` record with `max_cond > 1e8`.
- Naive variance: assert a `sub` with `max_cond > 1e10`.
- Log-sum-exp: assert an `exp` with `max_cond ≥ 100`.
- Kahan: assert profile reports the well-conditioned chain plus the
  intentional cancellation in compensation.

For `cln_kernel`:
- Assert the generated driver builds against Kokkos (skipped if Kokkos
  unavailable — `@pytest.mark.kokkos`).
- Assert at least one `opaque` record with `prov` containing both
  `"Kokkos::log"` and `"z<n>"` (validates that provenance propagates
  through opaque, per the v1.1 opaque fix).
- Assert `interop_decisions` chose either `interop_shim` or `opaque_wrap`
  for `Kokkos::log` and `Kokkos::abs`, with non-empty justifications.

## Fixtures

Adapted from Tracked's calibration tests. Each fixture is a single `.cpp`
or `.hpp` file containing one kernel function with a clean signature. The
adaptation strips Catch2 boilerplate and reduces to:

```cpp
// tests/agents/fixtures/kernels/cancellation.cpp
#pragma once
template <class T>
T cancellation_check(T a, T b) {
    return (a + b) - a;   // catastrophic cancellation candidate
}
```

with a matching `cancellation.yaml`:
```yaml
ranges:
  a: [1.0, 1.0]      # constant
  b: [1e-15, 1e-10]  # small perturbation
```

`cln_kernel.hpp` is the body of `cLn` lifted from
`Agentic-Mixed-Precision-Demo/src/kokkosUtils.h` with the minimal `ql::`
surface inlined as a single header — same content as the manual
`cln_micro.cpp` minus the `main()`. The characterizer treats it as if it
were the user's kernel file.

## Milestones

1. **Scaffolding.** Drop `llm_agent/`, create `agents/` tree, state schema
   with reducers, deterministic stubs for strategy/patcher/validator,
   orchestrator wiring. Empty graph compiles and runs without error.
   Submodule Tracked under `third_party/tracked/`.

2. **Build/run stub + CMake template.** `BuildRunStub.build_and_run`
   compiles and executes a manually-written micro-driver against the
   submodule's Tracked headers. `test_build_run_stub.py` passes.

3. **Log parser + profile schema.** `test_log_parser.py` passing with
   synthetic JSONL.

4. **Driver gen + prompt.** Prompt built from `CHARACTERIZER_NOTES.md`.
   For at least one real-valued fixture (cancellation), generated driver
   compiles and runs end-to-end. `test_driver_gen.py` passes for that
   fixture.

5. **Driver gen handles complex/Kokkos.** Extend prompt and tests to the
   `cln_kernel.hpp` fixture. Confirms the interop-vs-opaque decision
   logic works against the same kind of kernel we hand-wrote.

6. **Symbolic overlay + prompt + tests.**

7. **Full e2e test suite passing** for all four Tracked calibration
   fixtures plus `cln_kernel`. CLI documented in repo README.

## Open items (not blocking, decide during implementation)

- **Kernel parsing in `spec_build`.** Regex vs libclang. Start with regex
  for v1, switch to libclang in v2 when kernel signatures get hairier.
- **Snapshot management.** Use `syrupy` or roll our own? Probably `syrupy`
  — it's pytest-native.
- **LLM client library.** Likely the in-repo Argo wrapper from
  `run-argo.sh` / `setup_argo_proxy.sh`. Confirm Pydantic schemas work
  through it for the structured outputs.
- **Kokkos availability in CI.** `test_characterizer_e2e.py::test_cln_kernel`
  is `@pytest.mark.kokkos` and skipped when `Kokkos_DIR` env var is unset.
  Local development requires the Kokkos install from
  `examples/cln_micro/build_kokkos_serial.sh` in the Tracked repo.
