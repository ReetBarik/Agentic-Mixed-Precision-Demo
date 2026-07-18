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

---

# Characterizer post-processing pass (2026-07-17)

Closes HANDOFF items 3, 4, 6 above. Four commits, post-processing only over the
existing reduced report — **no 100k re-run**. Full suite 134 → **148 passing**.

## What landed

| commit | change |
|---|---|
| `characterizer: add predicted_rel_err_if_ff` | Change 1 — unblocks item 6 |
| `characterizer: add region_local_vars` | Change 2 — addresses item 3 (with a data caveat, below) |
| `characterizer: localize cancellation cascades as chain regions` | Change 3 — unblocks item 4 |
| `strategy: required_by bookkeeping for cascade-chain promotions` | Change 4 |

**Change 1 — `predicted_rel_err_if_ff`.** `U_FF = 2**-46` (~1.42e-14, the
empirical float-float floor: nominal 2**-48 minus ~2 bits lost to the EFT
residual). Emitted in `_classify_region` + `_classify_variable` as a peer of
`predicted_rel_err_if_float`. Derived from `max_sensitivity` at finalize, so no
merge/shard-schema change. Strategy's speedup gate can now queue `double→ff` at
tolerance 10 (float never clears it; ff does).

**Change 2 — `region_local_vars`.** New peer of `prov_vars` in the report; the
source variables read as **direct leaf operands** by a region's ops. `prov_vars`
stays (full transitive union, for consumers that want it). Merges as a set union.

**Change 3 — `cascade_chain` records.** Per sample, walk the value DAG backward
from each victim (final-value DAG sink, high rel_err + low per-op cond), collect
add/sub ancestors whose operands nearly cancel, emit ONE chain record per victim
spanning the union of their source lines (multi-file allowed). Contributor test:
`|a-b|/(|a|+|b|) < cascade_cancel_ratio` (0.1, on `ReducerConfig`); val-based
from operand records, with a cond fallback (`1/cond`) for leaf operands with no
journaled value. `chain_id = cascade_<integral>_<sample_hash>_<victim_hash>`,
deterministic. Carried through merge/finalize **unmerged** (union by chain_id).
Additive — the old per-region cascade classification is untouched.

**Change 4 — Strategy `required_by`.** `load_chains` + `ChainRecord`; chains are
tier 2, drained before speedup. Each promoted chain's precision is distributed
across all its lines through a required_by ledger. `precision_assignment` entries
gain `required_by`; overlap resolves to one entry at max precision with all
chain_ids. `RetryWalk` gains a `floor` param enforcing the speedup floor rule.

## Where journal data was insufficient (Change 2 — READ THIS)

**`region_local_vars` is region-local *reads*, NOT declares/assigns.** The task
asked to "filter `prov_vars` to variables declared or assigned in-scope." That is
**not recoverable from the journal**, confirmed against the schema
(`third_party/tracked/include/tracked/journal.hpp`, `docs/PROVENANCE.md`):

1. **No LHS/output field.** A record's `id` is a synthesized op id
   (`<op>@<file>:<line>#<n>@<scope>`); the C++ assignment target (`k12` in
   `const TMass k12 = …`) appears nowhere. There is no `out`/`lhs`/`dst` field.
2. **`track()` emits no record.** Only ops emit; the declaration/assignment point
   of a source variable is simply not in the data.
3. **`prov_vars` entries carry no scope.** A `track()`-seeded source variable's
   id *is* its bare name (bypasses `make_id`, so no `line=` suffix). In the
   qcdloop pipeline they are whole-app inputs (`p1..p6`, `m1..m4`, `mu2`) seeded
   once at top level — essentially never per-line locals. A locality *filter* on
   `prov_vars` would therefore be almost always empty.

What **is** derivable (and is a true subset of `prov_vars`): the source vars a
region's ops read as **direct leaf operands** — the named inputs textually used
at the source line. That is what `region_local_vars` emits. Derived DAG values
*are* line-attributable (each carries its own `line=` scope), but they are
**unnamed**, so they can't populate a variable *name* list.

**Decision (flagging, per the task):** shipped reads-in-region as the honest,
tight, deterministic replacement for the `prov_vars` passthrough. If ff/dd
regional promotion needs the *written* variable set, the fix is characterizer-
upstream: add an LHS-name field to the journal record, or journal `track()` with
the active scope. Not doable in post-processing.

## Test coverage

- `tests/agents/test_stability_reducer.py` 14 → **21**: `predicted_rel_err_if_ff`
  (cascade + stable), `region_local_vars` (direct-reads-not-union, merge union,
  const/literal exclusion), cascade chains (localizes the right lines,
  deterministic chain_id, survives merge unmerged, no-chain on stable sample).
- `tests/strategy/` 49 → **56**: speedup floor (blocks below floor / allows down
  to floor / no-floor sanity), `load_chains` (multi-line record + rep target,
  empty when absent), e2e chain overlap (F.h:9 in two chains → one dd entry with
  both chain_ids; free stable region demotes; floor protects the overlap line),
  ledger max-precision unit (dd vs ff → dd, both ids).

## Punts from this pass

- **Real multi-line chain intents for Patcher.** Change 4 drives the retry walk
  on the chain's *representative* line (`ChainRecord.walk_record()`) and
  distributes the result to all chain lines. A real Patcher promoting a whole
  chain atomically (multi-region intent) is deferred with the rest of Patcher.
- **`region_local_vars` = reads, not writes** — see the data caveat above; the
  write set needs a journal schema change, out of scope for post-processing.
- **Strategy still reads `prov_vars`** for `RegionTarget.variables`
  (`characterization._one_record`); migrating single-span regions to consume
  `region_local_vars` is the follow-up noted in Change 2's brief.
- **Cascade victim = DAG sink.** Non-sink intermediates meeting the cascade
  criteria are not treated as separate victims (avoids a chain per intermediate);
  revisit if a real cascade's victim is consumed downstream.

---

# Patcher agent — implementation handoff

Scope: `agents/patcher/` implemented per `docs/strategy_patcher_design.md`
§P1–§P7 (+ the cascade-chain amendment), on branch `langgraph-agents`. Strategy
was already implemented (above); the Patcher is called from it as
`patcher_fn(intent, ctx) -> P2`. Full suite: **148 → 183** (all green), incl. a
real-build e2e.

## What landed

`agents/patcher/` (new modules):

| module | responsibility |
|---|---|
| `intent.py` | P1 parse → `RemediationIntent` (reuses `strategy.models` — the shared contract), P4 cheap pre-checks (file/line/variable), tree-path resolver |
| `result.py` | P2 return contract: the 8-status enum, `error.kind` vocab, `ok()` / `failure()` builders |
| `dispatch.py` | P3 four-path dispatch + the generators (regional, plain-edit, git-revert, llm-rewrite) + `is_retryable_misgen` (P4a) |
| `edits.py` | P3a plain-type-edit — comment/string/char-literal-aware keyword-token rewriter (see deviation below) |
| `rewrites.py` | P3/P3b llm-rewrite prompt build + region splice (kahan / identity) |
| `gates.py` | P5 build+smoke gate: vanilla driver, 21-row smoke, NaN/Inf scan, build/smoke timeouts, env-overridable module prelude |
| `gitops.py` | Patcher-owned git: apply / commit (Q3 message) / reset-on-fail / `revert --no-commit` / introducing-commit lookup |
| `agent.py` | `make_patcher_fn(...)` adapter + the per-intent flow: parse → precheck → dispatch → **N=3 bounded retry over generate+gate** (P4) → commit (P2) |

Plumbing:
- `agents/integrator_base/region.py` — shared `RegionIntegrationResult` (shim
  paths + boundary patch) returned by both regional integrators.
- `agents/dd_integrator/agent.py` — **`integrate_region`** added (P7, one-module
  two-functions); `integrate` (whole-app) untouched.
- `agents/ff_integrator/agent.py` — **`integrate_region`** added; `integrate`
  stub untouched.
- `agents/validator/agent.py` — **`make_validator_fn`** SHA↔patch adapter
  (Strategy HANDOFF item 9): turns a candidate SHA into the cumulative
  `starting_sha..candidate` diff `validate()` consumes.
- `agents/orchestrator.py` — simplified to **characterize → strategy → END**;
  the vestigial patcher/validate nodes + loop edges removed (Strategy drives both
  as callables, Q5).
- `agents/strategy/characterization.py` — single-span `RegionTarget.variables`
  migrated from `prov_vars` to **`region_local_vars`** (fallback to `prov_vars`
  for older reports); the tight region-local reads set the integrators want.

## Integrator scope decision — **(b)** (bounded stub)

Per the task's scope hedge, chose **(b)**: `ff_integrator.integrate_region` and
`dd_integrator.integrate_region` land the **locked signature, cheap validation,
and the `RegionIntegrationResult` return shape**, but the actual regional shim +
boundary-patch *generation* (LLM-driven, mirroring the tracked integrator's
hundreds of lines of prompt/C8/retry) is deferred — the default entry raises
`NotImplementedError`, which the Patcher maps to `llm_gen_failed`.

The Patcher's regional-integrator dispatch path is fully exercised through an
**injected** integrator: unit tests inject a mock that writes a shim into the
tree; the **e2e injects a hand-written qcdloop ff shim + boundary patch and does
a real vanilla build + smoke + NaN scan + commit**. So all Patcher orchestration
(dispatch, gates, retry, git ops, adapters, P2) is complete and tested; only the
integrator *codegen* is stubbed. Rationale: full regional codegen did not fit one
focused session alongside the whole Patcher; the injection seam means it drops in
later without touching Patcher.

## Design ambiguities / deviations flagged (not silently resolved)

1. **`libclang` for plain-type-edit (P3a) — bindings absent on this image.**
   The `clang` python bindings are not installed on the cluster. Shipped a small
   C++ **keyword-token rewriter** instead: it skips comments / string / char
   literals and rewrites only bare `float`/`double` keyword tokens on the target
   lines. Because both are C++ *reserved keywords* they can never be part of a
   longer identifier, so the identifier-collision corruption that motivated the
   AST decision (`float_traits`, `floating_point`) is impossible for a
   whole-token match; the lexer covers the comment/string cases. Same
   corruption-safety guarantee for this specific swap. `edits.py` is structured
   so a libclang backend slots in where the bindings exist. **Flag for review:**
   deviates from the locked "libclang" wording, though not from its intent.

2. **`integrate_region` signature — added `repo_path`.** The locked §P4 call
   shape omits it, but a real integrator needs it to resolve `working_tree` (a
   SHA) via `git show <sha>:<file>`. Added as a keyword; harmless to the stub.

3. **git-revert looks up the introducing commit from git history, not
   `iterations.jsonl`.** The design says "from Strategy's iteration log"; the
   per-patch commits *are* that log (same `[iter_N] <kind> <file>:<lines>`
   schema, Q3), so `gitops.find_introducing_commit` scans commit subjects. Avoids
   a log-ingestion + write-timing dependency; equivalent result.

4. **P4a "retry everything" consequence.** With `is_retryable_misgen == True`, a
   *persistent* build/NaN/crash failure on an **llm-driven** path (regional /
   rewrite) folds to `llm_gen_failed` after N (matches the §P4 pseudocode's final
   `return llm_gen_failed`). At the DD rung Strategy therefore logs `dd_untested`,
   not a physics ceiling (P6a) — conservative and correct until we can classify
   real compile errors vs misgen. **Deterministic** paths (plain-edit, revert) do
   *not* retry: a build failure there is a real `build_failed` (Bucket A).

5. **Timeout is returned immediately** (no internal retry) so Strategy's P6
   timeout-retry-once → fold-to-`build_failed` logic owns timeout policy.

## Test coverage (`tests/patcher/`, +35)

- `test_intent.py` — parse all 11 kinds; reject unknown kind / bad range /
  missing identity; the three P4 pre-check misses → `patch_apply_failed`;
  malformed intent → `patch_apply_failed`.
- `test_edits.py` — keyword rewriter: in-range swap, comments/strings/identifiers
  survive, float↔double round-trip, no-occurrence raises, out-of-range untouched.
- `test_paths.py` — all four dispatch paths through the Patcher (mock gate): ff/dd
  regional commit the shim; plain-edit double→float (identifier survives);
  git-revert strips a prior ff install; missing introducing commit →
  `patch_apply_failed`; llm-rewrite kahan + identity (identity reaches the prompt).
- `test_retry.py` — integrator recovers within budget / exhausts → `llm_gen_failed`;
  build recovers / exhausts (folds to `llm_gen_failed`); deterministic build fail →
  `build_failed`; timeout returned once (not retried); failure resets the tree.
- `test_gates.py` — `_scan_smoke`: ok / NaN / short-output-crash / nonzero-exit /
  integral-name-not-scanned; build timeout → `timeout`.
- `test_e2e.py` (`@kokkos`) — real git branch + real vanilla build + real smoke +
  NaN scan + real commit, mocked integrator returning a hand-written qcdloop shim;
  asserts P2 `ok`, artifacts on disk, one commit on the branch, Q3 commit subject,
  ≥21 RES rows.

## Punts from this pass

- **Regional integrator codegen** (ff + dd) — scope decision (b); the real
  LLM-driven / mechanical generation is the next task, dropping into the existing
  injection seam and `RegionIntegrationResult` contract.
- **libclang backend** for plain-type-edit — slot in where bindings exist; the
  keyword rewriter is the portable default.
- **Multi-line atomic chain intents** — Patcher treats a `cascade_chain` intent
  as a single-line (representative-line) intent, per the task brief; real
  multi-region atomic promotion is deferred with cascade e2e verification.
- **Validator adapter is untested end-to-end against a real regional patch** —
  it needs a real integrator (patch that adds a shim onto the include path) and
  the deferred master→ddfun line-map (`validate()` still requires
  `accepted_patches == []`). The SHA→diff→`validate()` wiring is in place and
  injection-testable.
