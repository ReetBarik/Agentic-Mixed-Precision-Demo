# Future work

## LLM-based mixed-precision agent (not implemented)

**Goal:** An **in-repo** agent that uses an **LLM** to propose or revise source changes (unified diffs against `mutation_candidates.implementation_relative`), then **verifies** them with the existing pipeline: `compile.sh` → driver → `compare_results.py` vs double baseline, with a precision threshold from `compare_results.py` / `--min-digits`.

**Rough shape:**

- Read **`targets.json`** and the implementation file for the chosen driver (e.g. `ql::ddilog` in `src/kokkosUtils.h`).
- LLM proposes a patch (or iterates after failure); apply with `patch -p1` or equivalent; optional LangGraph-style graph is fine as long as the **LLM is the proposal step**.
- Reuse **`scripts/run_experiment.sh`**, **`scripts/apply_mutation_patch.py`**, **`scripts/compare_results.py`**, and hand-authored **`patches/<driver>/`** as references or starting points — the agent is **not** a rename of those scripts.

**Status:** Not in this repository yet. Patches under `patches/ddilog/` remain **human-curated**; sweep/combo scripts only **search** fixed patch sets.

## Improve proposal summarizer robustness

Make `run_episode` proposal summarization more generic and less regex-fragile:

- Parse unified diff structure first (`+++`, `@@`, added/removed lines) and treat it as the source of truth.
- Support multiple semantic rewrite patterns (direct `TMass -> float`, temp-float + cast-back, cast-only expression edits).
- Add graceful fallback with confidence tagging (`high|medium|low`) instead of over-specific summaries.
- Consider separating this into a dedicated `patch_summary` module with fixture-based tests for representative patch styles.
