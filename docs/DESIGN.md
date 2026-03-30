# Design notes: workflow, targets, and extensibility

This document records **why** the repository is structured the way it is, so you can recall the intent months later or onboard someone else without re-deriving the rationale.

## Overall goal

The project supports an **agentic mixed-precision workflow** (future: LangGraph or similar) that will:

1. Build and run **drivers** that exercise selected functions (e.g. `ddilog` in `kokkosUtils.h`) with reproducible random inputs.
2. Compare **double-precision baselines** to runs where individual locals are downcast to `float`.
3. Emit **accepted source changes** when error stays below a threshold.

The repository is intentionally split so that **environment/build** concerns stay separate from **which function** is under test. That separation makes it easy to add drivers without rewriting orchestration.

## Two JSON files: `workflow.json` vs `targets.json`

### `workflow.json` — generic build pipeline

**Role:** Describe *how* to prepare the machine and compile the project: modules, Kokkos bootstrap, incremental compile. These steps are **the same** regardless of whether you are testing `ddilog` or ten other functions.

**Reasoning:** If driver-specific details lived here, every new function would force edits to the same file that CI and humans use for “get a green build.” That couples unrelated concerns and increases merge noise. Keeping `workflow.json` function-agnostic means **orchestrators load it once** for any target.

**Placeholders:** `${PROJECT_ROOT}` (or env `AGENTIC_MIXED_PRECISION_DEMO_ROOT`) avoids hardcoded home paths so the same JSON works on laptops, CI, and clusters.

### `targets.json` — catalog of drivers and CLI conventions

**Role:** List each **driver** (`id`, executable path, source file), shared **CLI shape** (`default_cli`), **baseline file naming**, and placeholders for **numerical tolerances** used later by compare/agent logic.

**Reasoning:** Adding a new target function should be **data + one new `.cc` + one CMake id**, not a redesign. The catalog is the single place tools (Python, LangGraph) read to discover “what can I run?” CMake does not parse this file (to avoid bumping `cmake_minimum_required` for JSON or fragile regex); instead **`DEMO_DRIVERS` in `CMakeLists.txt` is kept in sync** with `drivers[]` in `targets.json`. That duplication is a small, explicit trade-off for portable CMake 3.16.

**`default_cli`:** All current drivers share `batch_size`, `output_file`, `seed` so one runner script can treat every driver the same. If a future driver needs a different shape, add optional fields on that driver entry or a `cli_override` pattern—without changing the global build workflow.

**`baseline_layout`:** Golden runs live under `baselines/<driver_id>/` with filenames **`{driver_id}_baseline_{batch_size}_{seed}.csv`** (example: `baselines/ddilog/ddilog_baseline_10_123.csv`). **Ad-hoc and smoke tests** use `experiments/<driver_id>/` (e.g. `ddilog_smoke_pass_10_123.csv`, `ddilog_smoke_fail_seed_mismatch_10_122.csv`) so they stay identifiable without opening files; future agent-generated outputs can use the same tree with a subfolder or `.gitignore` pattern so they do not collide with committed smoke files.

**`input_domain` (per driver):** Different targets live on different regions of parameter space. Each entry under `drivers[]` may include an **`input_domain`** block: named parameters, distribution (e.g. `uniform_real`), intervals or grids, and dtypes. That is the **documented contract** for baselines and for any future runner that must reproduce or vary inputs. The **driver implementation must stay aligned** with that block; if you later add CLI flags for min/max or multiple parameters, update `input_domain` in the same change. Global notes live under `input_domain_conventions` in `targets.json`.

## CMake: `DEMO_DRIVERS` and `driver/<id>_driver.cc`

**Convention:** For each `id` in `DEMO_DRIVERS`, the source is `driver/<id>_driver.cc` and the executable is `<id>_driver` in the build directory.

**Reasoning:** Predictable naming means no per-target `add_executable` boilerplate beyond the list, and the same pattern matches `targets.json` (`executable_basename`, `source_relative`). When you add a function, the checklist is mechanical (see below).

## Shell scripts (`scripts/prepare.sh`, `build_with_Kokkos.sh`, `compile.sh`)

**Role:** Site-specific environment (modules), one-time Kokkos build + initial project configure, vs **incremental** project rebuild.

**Incremental builds (normal edit–compile–run loop):** from the repo root, after Kokkos is available on `CMAKE_PREFIX_PATH`,

`source scripts/compile.sh ${PROJECT_ROOT}`

(e.g. `source scripts/compile.sh /path/to/Agentic-Mixed-Precision-Demo`). This is the step to use after changing `src/kokkosUtils.h` or drivers; it does **not** re-clone or rebuild Kokkos. **`build_with_Kokkos.sh`** is for bootstrapping Kokkos + project the first time (and is not written to skip Kokkos if the clone already exists—use **`compile.sh`** for day-to-day rebuilds).

**One-shot experiment:** `scripts/run_ddilog_experiment.sh` runs `compile.sh` (unless `--no-build`), then `build/ddilog_driver`, then `compare_results.py` against `baselines/ddilog/ddilog_baseline_<batch>_<seed>.csv` by default. CSV output defaults to `experiments/ddilog/generated/ddilog_run_<batch>_<seed>_<timestamp>.csv`; that directory’s contents are gitignored except `.gitkeep`.

**Mutation patches (`patches/ddilog/`):** Each candidate id has a unified diff `<id>.patch` (paths `a/src/kokkosUtils.h`, apply with `patch -p1` from repo root). Patches use **minimal context** (`n=0`) so multiple ids can be **stacked** in a fixed order (**`T` → `Y` → `S` → `A` → `H` → `ALFA` → `B1` → `B2` → `B0`**; see **`PATCH_ORDER`** in `ddilog_mutation_combo_greedy.py`). Locals **`Y`, `S`, `A`** and **`B1`, `B2`, `B0`** are each on **their own line** in `kokkosUtils.h` so their patches do not fight over one declaration line. **`scripts/apply_ddilog_patch.sh apply|revert <id>`** applies or reverses one patch. **`scripts/ddilog_mutation_sweep.py`** exercises **one id per run** (`--id` or `--all`). **`scripts/ddilog_mutation_combo_greedy.py`** runs a **deterministic greedy** search (sorted try order; **`--tie-break margin|first`**); it is **not** random—multi-start shuffling is not implemented. It writes JSON under **`experiments/ddilog/generated/`**. Regenerate patches if `ddilog` changes materially. **`targets.json`** lists **`patches_available`**.

**Reasoning:** These existed before the JSON manifest; `workflow.json` **documents** the intended `source ...` commands rather than hiding them inside Python. That keeps debugging familiar for HPC users and allows running the pipeline by hand without an agent.

## Result CSVs and `scripts/compare_results.py`

Driver output uses a **header row**, a **metadata row** (line 2, starts with `#`), then one row per sample.

- **Real-valued output:** `id,real hex` then `# target_id=… seed=… batch_size=…` plus optional domain keys (`x_min`, `x_max` for `ddilog`). Hex tokens use the same `0x` + hex digits convention as before; **16 hex digits** denotes IEEE-754 **binary64** (see also `hex_to_double` in the qcdloop [`csv_parser.py`](https://github.com/ReetBarik/qcdloop/blob/error_analysis/error_analysis/float/csv_parser.py); **≤8** digits denotes **binary32**).
- **Complex output (future):** `id,real hex,imag hex` with the same metadata line shape.

**Do not compare incompatible runs:** `compare_results.py` requires identical metadata for `target_id`, `seed`, and `batch_size`, and identical optional domain keys if present in either file. Mismatches exit with an error so baseline/candidate pairs from different seeds, batch sizes, targets, or input domains are never scored together.

**Precise digits** follow the `calculate_precise_digits` logic from [`plot_precision_analysis.py`](https://github.com/ReetBarik/qcdloop/blob/error_analysis/error_analysis/float/plot_precision_analysis.py), with a **ceiling of 16** decimal digits and a **1e-16** relative floor. **Pass** for real: `min` digits ≥ threshold (default 10). **Pass** for complex: minimum over the batch for **real** and for **imag** must both be ≥ threshold (Option A). Rows with NaN/Inf in baseline or candidate for a component are **skipped** for that component; if nothing finite remains, the script errors.

## What is intentionally *not* in this repo yet

- **LangGraph graph definition** — orchestration can call existing scripts (`workflow.json` → **`ddilog_mutation`**); no graph code in-tree yet.

Patch apply, single-id sweep, and greedy combo are **implemented** (`scripts/apply_ddilog_patch.sh`, `ddilog_mutation_sweep.py`, `ddilog_mutation_combo_greedy.py`).

## Checklist: adding a new target function

1. Implement `driver/<id>_driver.cc` (prefer the same three-arg CLI as `ddilog` for one code path in runners).
2. Append `<id>` to `DEMO_DRIVERS` in `CMakeLists.txt`.
3. Append a matching object to `targets.json` → `drivers`, including **`input_domain`** for every randomized (or grid-sampled) parameter.
4. Create `baselines/<id>/` (optional `.gitkeep`) for golden outputs.

## Cross-references

| Artifact | Purpose |
|----------|---------|
| `workflow.json` | Env + Kokkos + compile steps |
| `targets.json` | Runnable drivers + CLI + baseline naming + per-driver `input_domain` + `mutation_candidates` for `ddilog` |
| `CMakeLists.txt` | `DEMO_DRIVERS` → executables |
| `baselines/<id>/` | Stored double baselines per driver (`{id}_baseline_{batch}_{seed}.csv`) |
| `experiments/<id>/` | Smoke tests and non-golden experiment CSVs |
| `docs/DESIGN.md` | This rationale (human-oriented) |
| `scripts/compare_results.py` | Baseline vs candidate CSV validation |
| `scripts/run_ddilog_experiment.sh` | Optional incremental build, `ddilog_driver`, then `compare_results.py` |
| `scripts/apply_ddilog_patch.sh` | Apply or revert `patches/ddilog/<id>.patch` |
| `scripts/ddilog_mutation_sweep.py` | Apply → experiment → revert for ids with patch files |
| `scripts/ddilog_mutation_combo_greedy.py` | Greedy multi-float set under `--min-digits` |
| `patches/ddilog/*.patch` | Per-local diffs vs `src/kokkosUtils.h` (stackable for combo) |
| `experiments/ddilog/generated/` | Default output for ad-hoc runs (contents gitignored except `.gitkeep`) |

If the JSON and this document diverge, treat **the JSON as the operational contract** for tools and **this file as the explanation**—update both when you change conventions.
