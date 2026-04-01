# Future work

## LLM-based mixed-precision agent (implemented baseline; extend next)

**Current status:** The in-repo LLM workflow is available in `llm_agent/`:

- Propose-only patch generation (`patch_proposer`)
- Isolated verify (`patch_verify`)
- One-shot propose→verify episodes with retries (`run_episode`)
- Greedy accumulation over semantic vars (`run_greedy_episode`)

The current flow already reuses the existing compile/driver/compare pipeline and records per-attempt results under `experiments/<driver>/generated/`.

**Next goals:**

- Expand coverage beyond `ddilog` by adding more drivers in `targets.json` + `driver/<id>_driver.cc`.
- Improve coupled-variable handling so policy checks do not over-prune valid proposals for tightly linked symbols.
- Add richer run-level metrics (e.g., proposal format failures, policy rejects, numeric fails) for easier diagnostics across episodes.

## Improve proposal summarizer robustness

Make `run_episode` proposal summarization more generic and less regex-fragile:

- Parse unified diff structure first (`+++`, `@@`, added/removed lines) and treat it as the source of truth.
- Support multiple semantic rewrite patterns (direct `TMass -> float`, temp-float + cast-back, cast-only expression edits).
- Add graceful fallback with confidence tagging (`high|medium|low`) instead of over-specific summaries.
- Consider separating this into a dedicated `patch_summary` module with fixture-based tests for representative patch styles.
