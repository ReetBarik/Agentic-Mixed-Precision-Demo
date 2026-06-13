# Agentic Mixed-Precision Demo

An LLM-powered agent that automatically finds safe **mixed-precision optimizations** in C++ scientific computing functions. Given any C++ function, the agent identifies which local variables can be safely downcast from `double` to `float` without losing numerical accuracy beyond a configurable threshold.

No manual configuration is required — the agent reads the source file directly, reasons about the function, and verifies each proposed change numerically.

---

## What it does

High-performance computing code is often written in `double` precision throughout for safety. In practice, many intermediate variables tolerate `float` precision without affecting the final result. Finding these variables manually is tedious and error-prone. This project automates that search:

1. **Analyze** the target function to understand its signature and identify candidate local variables.
2. **Generate** a test driver that calls the function with random inputs and records outputs.
3. **Propose** a downcast for each candidate variable (e.g. change `double x` to `float x`).
4. **Verify** numerically: compile with the patch, run it, and compare output against the double-precision baseline. Accept if the result agrees to at least N decimal digits; reject otherwise.

---

## Agentic workflow

The system is built as a set of **LangGraph** subgraphs (skills), each handling one stage of the pipeline. An orchestrator graph wires them together.

```mermaid
flowchart TD
    A([User: --file path --function name]) --> B

    subgraph Orchestrator
        B[Load & validate target file]
        B --> C

        subgraph AnalyzeSkill["Analyze Skill"]
            C[Read source file]
            C --> D[LLM extracts function signature\nInput params · return type · framework\nLocal variable candidates\nSafe template instantiation types]
            D --> E{Valid?}
            E -- No, retry --> D
            E -- Yes --> F
        end

        F[Signature] --> G

        subgraph DriverSkill["Driver Skill"]
            G[LLM generates C++ test driver\n+ CMakeLists.txt]
            G --> H[Compile]
            H -- Failed --> I[Feed error back to LLM]
            I --> G
            H -- OK --> J[Run driver\nCollect baseline CSV]
        end

        J --> K

        subgraph DowncastSkill["Downcast Skill"]
            K[For each candidate variable...]
            K --> L[LLM proposes type change\ne.g. double → float]
            L --> M[Policy check\ne.g. must be lower-precision type]
            M -- Rejected --> L
            M -- OK --> N[Compile with patch]
            N -- Build failed --> L
            N -- OK --> O[Run & compare vs baseline]
            O -- meets digit threshold --> P[Accept patch]
            O -- fails --> Q{Retries left?}
            Q -- Yes --> L
            Q -- No --> R[Reject variable]
            P --> K
            R --> K
        end

        K --> S[Write summary JSON]
    end
```

### Skills

| Skill | Responsibility |
|-------|---------------|
| **Analyze** | Reads the full source file. Uses an LLM to extract the function signature (parameters, return type, portability framework), infer safe input domains for random testing, identify local floating-point variables as downcast candidates, and determine concrete template instantiation types that avoid compiler overload ambiguity. |
| **Driver** | Uses an LLM to generate a complete, self-contained C++ test driver and its `CMakeLists.txt`. Handles the detected portability framework (Kokkos, SYCL, OpenMP, CUDA, HIP, or plain C++). Feeds compilation errors back to the LLM and retries until the driver compiles and runs successfully, producing a baseline CSV of outputs. |
| **Downcast** | Iterates over each candidate local variable. For each one, asks an LLM to propose a source-level type change to a lower-precision type (e.g. `float`). Compiles the patched source, runs it with the same random inputs as the baseline, and compares outputs digit-by-digit. Accepts the patch if it meets the precision threshold, rejects it otherwise, and feeds verification results back to the LLM for the next attempt. |

---

## Repository layout

```
.
├── run-argo.sh                  # Entry point (handles tunnel + proxy + runs the agent)
├── src/                         # Example target: kokkosUtils.h (Kokkos C++ library)
├── scripts/
│   ├── compare_results.py       # Numerical comparison tool (baseline vs candidate CSV)
│   ├── prepare.sh               # Loads build environment modules
│   └── setup_argo_proxy.sh      # Starts the Argo SSH tunnel + proxy (used by run-argo.sh)
├── llm_agent/
│   ├── run.py                   # CLI entry point (called by run-argo.sh)
│   ├── config.py                # Model name, iteration limits
│   ├── client.py                # Anthropic API client factory
│   ├── state.py                 # TypedDicts for all graph states
│   ├── graphs/
│   │   └── orchestrator.py      # Top-level LangGraph graph
│   ├── skills/
│   │   ├── analyze/             # Signature extraction subgraph
│   │   ├── driver/              # Driver generation + compile-iterate subgraph
│   │   └── downcast/            # Patch proposal + verification subgraph
│   └── tools/
│       ├── build.py             # compile_driver(), run_driver(), build_and_run()
│       └── compare.py           # Wrapper around scripts/compare_results.py
└── experiments/                 # Output: baseline CSVs, candidate CSVs, summary JSONs
```

---

## Prerequisites

**Build environment:**
- C++17 compiler
- CMake ≥ 3.16
- The portability framework used by your target (e.g. Kokkos, OpenMP). For the included example (`kokkosUtils.h`) Kokkos must be on `CMAKE_PREFIX_PATH`.
- `scripts/prepare.sh` must set up the build environment (module loads, paths).

**Python:**
- Python 3.12
- Install dependencies: `pip install -r requirements-argo-agent.txt`

**LLM access (Argonne JLSE):**
The agent uses the Anthropic API routed through the Argo proxy on JLSE. `run-argo.sh` handles this automatically — it detects whether the SSH tunnel and proxy are already running (e.g. from an existing Claude Code session) and reuses them.

If running outside JLSE, set `ANTHROPIC_API_KEY` and omit `--base-url` from `llm_agent/run.py`, or point `ANTHROPIC_BASE_URL` at your own proxy.

---

## Usage

```bash
./run-argo.sh --file <repo-relative-path-to-header> \
              --function <function-name> \
              [--skills downcast] \
              [--min-digits 10] \
              [--batch 10] \
              [--seed 123] \
              [--max-iterations 3] \
              [--max-driver-retries 5] \
              [--output-dir experiments/]
```

**Example** — optimize `ddilog` from the included Kokkos utility header:

```bash
./run-argo.sh --file src/kokkosUtils.h --function ddilog --skills downcast --min-digits 10 --batch 10
```

**Output** — a JSON summary written to `experiments/<function>/generated/<function>_summary_<timestamp>.json`:

```json
{
  "function_name": "ddilog",
  "file_path": "src/kokkosUtils.h",
  "framework": "kokkos",
  "baseline_csv": "experiments/ddilog/generated/ddilog_baseline_10_123_<ts>.csv",
  "error": null,
  "skill_results": {
    "downcast": {
      "accepted_variables": ["S", "A", "B1", "B2", "B0"],
      "rejected_variables": ["T", "Y", "H", "ALFA"],
      "accepted_patches": [
        {
          "file_path": "src/kokkosUtils.h",
          "old_line": "        TMass S;",
          "new_line": "        float S;",
          "reasoning": "..."
        }
      ],
      "trace": "... (per-attempt records: proposal, policy_reject, verify_pass, min_precise_digits)"
    }
  }
}
```

### Key options

| Option | Default | Description |
|--------|---------|-------------|
| `--file` | required | Repo-relative path to the C++ header containing the target function |
| `--function` | required | Name of the function to optimize |
| `--skills` | `downcast` | Optimization skills to run |
| `--min-digits` | `10` | Minimum precise decimal digits required to accept a downcast |
| `--batch` | `10` | Number of random input samples per run |
| `--seed` | `123` | RNG seed (use the same seed across all runs for reproducibility) |
| `--max-iterations` | `3` | Max LLM retry attempts per variable in the downcast skill |
| `--max-driver-retries` | `5` | Max compile-fix attempts in the driver skill |

---

## How verification works

The agent uses **bitwise-reproducible** output comparison. For each run:

1. The driver generates `batch` random inputs using a fixed seed.
2. Outputs are serialized to CSV.
3. `scripts/compare_results.py` computes the minimum number of matching significant decimal digits across all samples (`min_precise_digits`).
4. A patch is accepted only if `min_precise_digits ≥ --min-digits` for all samples.

This makes accept/reject decisions deterministic and independent of platform floating-point rounding modes.

---

## Characterizer slice (v2, langgraph-agents branch)

The `agents/` tree implements the first vertical slice of the v2 multi-agent
pipeline: a LangGraph orchestrator with a real characterizer agent and
deterministic pass-through stubs for strategy / patcher / validator.  See
`agents/characterizer/PLAN.md` for the design and `CHARACTERIZER_NOTES.md`
for the framework-agnostic lessons feeding the prompt.

### Prerequisites

- Python deps:  `pip install -r requirements-langgraph.txt`
- Tracked submodule:  `git submodule update --init --recursive`
  (clones `kokkos-extended-precision-demo` @ `tracked` into
  `third_party/tracked/`).
- CMake (`>=3.18`) on PATH.
- For Kokkos-backed kernels: a Serial-only Kokkos install.  The Tracked
  repo ships `examples/cln_micro/build_kokkos_serial.sh` to produce one
  at `$HOME/kokkos-install`.
- Argo proxy running (the same `run-argo.sh` from the v1 workflow); the
  characterizer's `driver_gen` and `symbolic_overlay` nodes hit it for
  Claude Opus 4.7.

### One kernel, end to end

```bash
python -m agents.cli characterize \
  --kernel tests/agents/fixtures/kernels/cancellation.cpp \
  --kernel-name cancellation_check \
  --ranges-yaml tests/agents/fixtures/input_ranges/cancellation.yaml \
  --samples 512 \
  --out runs/cancellation
```

Artifacts land in `runs/cancellation/`:

- `src/micro_driver.cpp`         — LLM-generated driver
- `CMakeLists.txt`               — rendered build script
- `interop_decisions.json`       — per-call strategy choices (shim / opaque / inline)
- `journal.jsonl`                — raw Tracked output
- `sensitivity_profile.json`     — characterizer's roll-up
- `symbolic_hints.json`          — LLM idiom detection (best-effort)

### Running against all calibration fixtures

```bash
for k in cancellation naive_variance log_sum_exp kahan; do
  python -m agents.cli characterize \
    --kernel tests/agents/fixtures/kernels/${k}.cpp \
    --kernel-name $(python -c "print({'cancellation':'cancellation_check','naive_variance':'naive_variance','log_sum_exp':'log_sum_exp_naive','kahan':'kahan_sum'}['$k'])") \
    --ranges-yaml tests/agents/fixtures/input_ranges/${k}.yaml \
    --samples 512 \
    --out runs/${k}
done
```

Expected: each profile flags the predicted hotspot.  See
`runs/<name>/sensitivity_profile.json` — the `top_hotspots` list is sorted
by max condition number, descending.

### Kokkos kernel (Serial backend)

```bash
python -m agents.cli characterize \
  --kernel tests/agents/fixtures/kernels/cln_kernel.hpp \
  --kernel-name cLn \
  --ranges-yaml tests/agents/fixtures/input_ranges/cln_kernel.yaml \
  --samples 256 \
  --kokkos-root $HOME/kokkos-install \
  --out runs/cln
```

The characterizer detects the Kokkos framework from the source, picks
per-call strategies for `Kokkos::log` / `Kokkos::abs` (interop shim or
opaque wrap), and propagates provenance through the boundary.

### What's stubbed

- **Strategy / patcher / validator** agents return identity — the
  characterizer always exits the graph at strategy with an empty queue.
- **Build/run agent** is a deterministic subprocess wrapper (no LLM).
  Future work: LLM-driven build/run with framework detection and module
  loading.

### Tests

```bash
pytest tests/agents/test_log_parser.py
```

Pure-unit log parser tests (13).  E2E and driver-gen tests are deferred
— see `agents/characterizer/PLAN.md` for the planned test suite.

## Adding a new target

No catalog or spec file is needed. Just point the agent at any C++ header and function name:

```bash
./run-argo.sh --file path/to/mylib.h --function my_compute_function --skills downcast --min-digits 10 --batch 20
```

The analyze skill will automatically detect the portability framework, infer input domains from parameter names and types, and identify local variable candidates. The driver skill will generate and compile an appropriate test driver for the detected framework.
