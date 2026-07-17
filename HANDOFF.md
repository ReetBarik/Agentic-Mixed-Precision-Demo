# Strategy agent — implementation handoff

Scope: `agents/strategy/` implemented per `docs/strategy_patcher_design.md`
(commit 049faff). Patcher is **mocked** here — it's a separate follow-up task.
Branch: `langgraph-agents`.

## What landed

`agents/strategy/` (new modules):

| module | responsibility |
|---|---|
| `models.py` | precision ladder (cost-ordered), 11-kind vocabulary + derivation, `RegionTarget` / `RemediationIntent` dataclasses |
| `characterization.py` | load the fixed `stability_report` JSON → `RegionRecord`s, **merge per-line across integrals** |
| `ranking.py` | two class-driven queues (correctness 4-tier, speedup op_count-desc) |
| `walk.py` | per-target retry-walk state machine (up-ladder + reformulate + dd_ceiling; down-ladder + backoff; dd_untested) |
| `dispatch.py` | P6 dispatch dict — 8 Patcher statuses → Strategy response + log tag |
| `iteration_log.py` | append-only `iterations.jsonl` writer |
| `gitops.py` | branch create / reset-on-reject / cumulative diff (Strategy never commits code; Patcher does) |
| `report.py` | `report.json` (full) + `report.md` (Q4 projection) writers |
| `agent.py` | `run(state) -> {"strategy_result": …}` — the whole loop (correctness drain → speedup), stopping conditions, artifact emission |

Plumbing:
- `agents/config.py` — `StrategyConfig` + `StrategyBudget` dataclasses.
- `agents/state.py` — `PipelineState` extended with the fields Strategy reads
  (`characterization_report_path`, `strategy_repo_path`, `strategy_starting_sha`,
  `strategy_config`, `patcher_fn`, `validator_fn`) and writes (`strategy_result`).
- `.gitignore` + `runs/qcdloop/strategy/.gitkeep` for per-run artifacts.

Artifacts per run under `runs/qcdloop/strategy/<run_id>/`: `report.json`,
`report.md`, `iterations.jsonl`, `final.diff`. State delta is the thin Q5 bundle.

The orchestrator was **not** modified — Strategy now runs the loop internally and
leaves `strategy_queue` empty, so the existing conditional edge routes to END.
(See punts: the strategy→patcher→validate graph edges are now vestigial.)

## Design ambiguities flagged (not silently resolved)

1. **9 vs 8 transition kinds.** P3's kind table lists 8 transitions; a
   single-step cost-ladder up-walk from a **float** baseline needs `float-to-ff`
   (the 9th) — added to the vocabulary, which matches the task's stated "9
   transition kinds." A *fully general* from-baseline correctness walk from
   float to dd would additionally need `float-to-dd` (a 10th) that is **not** in
   the vocabulary; the walk caps a float baseline at `double` (status
   `exhausted`). This never occurs in the fixed-report workflow (correctness
   baselines are always `double`), so it's a latent edge, not a live gap. **P3's
   table should be amended to include `float-to-ff`.**

2. **Region identity — merged per source line.** Q1's region record is
   `(file, line_start, line_end, variables)` with **no integral**, but the
   characterization emits one region per `(integral, file, line)` and the same
   header line appears in many integrals. Since a line can only be promoted
   once, `load_regions(merge=True)` collapses same-line entries with worst-case
   signals (severity-max class, max cond/rel_err/pred_float, union of vars,
   highest-cond integral as representative). Without this, the same line would be
   promoted repeatedly with a stale `current_precision`. On `report_1k.json`:
   1188 raw regions → 456 code regions. **Confirm this merge policy is what you
   want** (alternative: keep per-integral and dedup only at assignment time).

3. **`prov_vars` ≠ region-local variables.** The report's only variable-ish
   field is `prov_vars`, an **input-provenance union** (thousands of entries
   like `m1[107]`), not the region-local set `ff_integrator`/Patcher needs to
   promote. It's currently passed through verbatim as `RegionTarget.variables`.
   This (a) bloats `report.json`/`iterations.jsonl` on real runs and (b) is the
   wrong input for regional promotion. Needs a characterizer-side region-local
   variable extraction (or Patcher-side narrowing).

4. **`cancellation_cascade` is non-localizable in the current report.** All 97
   cascade regions in `report_1k.json` have empty region keys (`""`) → skipped →
   correctness **tier 2 is empty in practice**, so the Kahan reformulate path is
   never exercised on real data yet. Ranking logic is correct; the data doesn't
   localize cascades. Characterization would need to localize them to feed tier 2.

5. **Rewrite layering (Q2) — resolved.** A reformulate is applied **on top of the
   retained DD** (`current_precision: dd`, final precision `dd`), not by
   reverting to `double` first. Matches Q2 ("a dd region can carry a rewrite at
   the ceiling") and keeps DD as branch HEAD so the git flow is simple. Recorded
   here because the design text ("At dd, still rejects → try rewrite") is silent
   on which tree the rewrite lands on.

6. **Speedup gate is float-based.** The speedup queue uses
   `predicted_rel_err_if_float ≤ 10^-tolerance` (faithful to the design). At
   tolerance 10, float (~7 digits) can never satisfy it, so the **speedup queue
   is empty on real data at tol=10**. `double→ff` speedups (ff ~14 digits, could
   meet tol 10) are never queued because the report has no
   `predicted_rel_err_if_ff`. A characterizer `predicted_rel_err_if_ff` signal
   would unlock ff speedups at high tolerance.

7. **Budget / diminishing-returns accounting.** `budget_iters` counts only
   iterations whose Patcher status has `counts_budget` (ok + Bucket A +
   timeout); `llm_gen_failed`, `patch_apply_failed`, `commit_failed` do not (P6).
   The diminishing-returns streak increments on any non-accepted iteration
   *except* `strategy_bug`, and resets on accept. Counting `llm_capacity` toward
   the DR streak is my call (a run that only produces LLM failures is stuck).

8. **LLM-token accounting source unspecified.** `budget.max_llm_tokens` is
   enforced against `resp.get("llm_tokens")` from the Patcher response, a field
   P2 does not define. Until the Patcher reports tokens, the token cap is inert.

9. **Callable contracts vs real signatures.** Strategy calls
   `patcher_fn(intent: dict, ctx: dict)` and `validator_fn(candidate_sha: str,
   ctx: dict)`. The real `validator.validate()` is **patch-based**
   (`base_state, candidate_patch, tolerance, snapshot`), whereas the design (P2)
   and this implementation are **SHA-based**. Adapters from these clean callables
   to the real Validator / Patcher are deferred (both mocked here).

## Test coverage

`tests/strategy/` — 49 tests, all passing (full repo suite: 134 passing).

- `test_ranking.py` (11) — tier order, local_cancellation always tier 1, error
  threshold gating, intra-tier cond-desc sort, speedup op_count sort +
  float-safety gate + correctness-exclusion + non-stable exclusion.
- `test_walk.py` (17) — correctness ladder (dd clear, ff-baseline walk,
  current_precision from baseline), DD ceiling (cascade→kahan,
  local_cancellation→identity catalog, rewrite-clears-on-dd, log_near_root
  no-rewrite), **dd_untested vs dd_ceiling (P6a)**, speedup ladder (backoff,
  first-reject, all-the-way-to-float), float-baseline exhaustion, protocol guards.
- `test_dispatch.py` (9) — all 8 statuses, buckets, dd_untested flags, unknown raises.
- `test_characterization.py` (5) — non-localizable skip (empty key + flag),
  per-line merge across integrals, no-merge mode, op_count sum.
- `test_loop.py` (8) — commit_failed→internal_error, budget max_iters,
  diminishing-returns→partial, dd_untested via build_failed, timeout
  retry-once→ok, timeout×2→fold-to-reject, patch_apply_failed strategy_bug+free,
  clean success.
- `test_e2e.py` (1) — full 3-region run on a **real temp git repo** with mocked
  Patcher (real commits) + Validator; asserts state-delta shape, `report.json`
  (precision distribution, assignments, ceiling region), iteration-log count,
  **branch state** (3 kept commits, rejects reset away), `final.diff` + markdown.

Mocks live in the tests; no dependency on the real Patcher (stub) or Validator.

## Punts / follow-ups

- **Wire real Patcher + Validator adapters** (SHA↔patch impedance, ctx plumbing).
  Patcher itself is the next task.
- **Simplify the orchestrator graph** to `characterize → strategy → END` — the
  patcher/validate nodes are now driven inside Strategy, not by the graph.
- **Region-local variable extraction** to replace the `prov_vars` passthrough
  (ambiguity 3) — also fixes `report.json` bloat.
- **Characterizer signals:** localize `cancellation_cascade` (ambiguity 4);
  add `predicted_rel_err_if_ff` to enable ff speedups at high tolerance
  (ambiguity 6).
- **Downstream-leverage ranking tiebreaker** — design-deferred, not implemented.
- **Re-characterization** — locked fixed-report-only (N=large); no infra built.
