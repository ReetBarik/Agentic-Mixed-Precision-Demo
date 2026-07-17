# Strategy Agent — design notes

Working design captured 2026-07-16 with Reet. Locked pieces below. Open
questions at the bottom are the resume point.

## Objective (frames everything)

Given a precision tolerance (minimum precise digits, default 10), produce
the **lowest-precision assignment per region** such that global
min-precise-digits ≥ tolerance. Precision ladder:

```
float  ~7 digits   →   ff (float-float)  ~14 digits   →   double  ~15-16   →   dd (double-double)  ~30-31
```

**No quad.** CUDA fp128 requires Blackwell-only sm_100 hardware; DD is the
best portable ceiling.

Algorithmic rewrites (Kahan, algebraic identities) are orthogonal to
precision and applied when a region's signal class demands it.

Objective = "make it correct enough AND then as fast as possible" via
per-region precision choice + selective rewrites.

## Loop shape (locked)

Loop, not one-shot. Each iteration:

1. Read state (fixed report + working tree + accepted-patches log).
2. Select next target from ranked queue.
3. Choose remediation kind (precision move OR algorithmic rewrite).
4. Emit remediation intent → Patcher → compiled patch → Validator.
5. `accept` → fold patch into working tree; `reject` → mechanical retry
   (next kind in vocabulary walk).
6. Check stopping condition; loop.

**Two-phase execution (locked):**

- **Correctness mode first.** Drain until every target is at threshold
  OR at DD ceiling.
- **Speedup mode second.** Try demotions from `stable` regions, biggest
  op-count first; back off on first `reject` per target.

## Ranking function (locked: two-queue, class-driven)

**Correctness queue** (fixed order):

1. All `local_cancellation` regions (cond > 1e15).
2. `cancellation_cascade` regions with `max_rel_err > 10^-tolerance`.
3. `log_near_root` regions with `max_rel_err > 10^-tolerance`.
4. `stable` regions that surprisingly show `max_rel_err > 10^-tolerance`.

**Speedup queue:** `stable` regions with `predicted_rel_err_if_float`
well below tolerance, ranked by op_count descending.

Downstream-leverage tiebreaker (walk `prov_vars` DAG to prefer upstream
causes over downstream symptoms) is **deferred** — start simpler, add
if needed.

## Retry policy — mechanical vocabulary walk (locked)

**Per target in correctness mode:**

1. Walk up ladder: `current → next-up → ... → dd`.
2. If any level clears threshold → `accept`, done.
3. At `dd`, still rejects → try algorithmic rewrite (Kahan for cascade,
   identity for local-cancellation). Rewrites don't hurt — worst case
   they don't recover anything.
4. Rewrite also rejects → **accept `dd` version as-is**, log
   `dd_ceiling_reached`, move on.

**Per target in speedup mode:**

1. Walk down ladder: `current → next-down → ... → float`.
2. First `reject` → back off one level, stop for this target.

Retry regen with LLM temperature variation deferred; mechanical walk
comes first.

## Re-characterization policy (locked)

- **Fixed report** as the default (single characterization at loop
  start, reused all iterations).
- **Re-characterize after N accepted patches** — set N = large number
  we won't hit in practice. Infrastructure exists but production
  workflow is fixed-report-only. Reet: "let's see if fixed report
  only works."

## DD-ceiling acceptance (locked)

If DD + algorithmic rewrites can't clear threshold at a site, that site
IS the physics limit. Accept DD version as best-possible, do NOT block
the loop, log ceiling clearly in final report.

## Validator API — no changes needed (locked)

Strategy owns ladder awareness. Validator stays a black box:
input=(base, candidate_patch, tolerance, snapshot), output=verdict.

Strategy knows what precision level it just tried, so ceiling detection
is trivial: "I tried DD, got reject, tried rewrite, got reject → ceiling."

**No addendum to the Cluster-Claude Validator brief.** The per-sample
per-component precise-digits array is already listed as a secondary
output in the original brief. Strategy owns ceiling detection based on
what precision level it just tried; DD stays a black box.

## Stopping condition (locked)

Layered:

- **Hard stop:** correctness mode done AND speedup mode done → success.
- **Budget stop:** max iterations / wall-clock / LLM token cap.
- **Diminishing returns:** last K iterations all rejected → declare
  stuck, hand back to human.
- **User stop:** Reet says "we're done" (he's in the review loop live).

## Patcher exists as a separate agent (locked)

Strategy → Patcher → Validator (not Strategy → Validator direct).

- Strategy emits **remediation intents** (semantic).
- Patcher owns mechanical translation to git-apply diff, invokes
  `ff_integrator` for regional promotion, ensures compile + no NaN.
- Validator only sees patches that compile and run.

Rough intent schema (to refine when we design Patcher):

```json
{
  "target": {"file": "B2m.h", "line": 355, "variables": ["wlogsmu"], "region_span": [355, 360]},
  "kind": "promote-DD" | "promote-ff" | "promote-double" | "demote-double" | "demote-ff" | "demote-float" | "reformulate-kahan" | "reformulate-identity",
  "rationale": "cancellation_cascade at B2m.h:355, upstream of BIN2 hotspot"
}
```

Patcher return:

```json
{"status": "ok" | "build_failed" | "runtime_failed", "patch": "<unified diff>" | null, "error": "..." | null}
```

## Inputs to Strategy (proposed, not yet locked)

1. Characterization report (path to 13GB `report_100k.json`).
2. Precision tolerance (scalar, default 10).
3. Working tree (path or commit SHA).
4. Budget knobs (max iterations, wall-clock, LLM tokens).
5. Re-characterization trigger N (large, effectively disabled).
6. Handles to Patcher + Validator.

**NOT inputs:** DD baseline/shim (Validator's problem), ladder vocabulary
(Strategy's internal), ground-truth annotations.

## Outputs from Strategy (proposed, not yet locked)

```json
{
  "status": "success" | "partial" | "budget_exhausted",
  "final_working_tree": "<path or sha>",
  "precision_assignment": [
    {
      "file": "B2m.h",
      "line_start": 355,
      "line_end": 360,
      "variables": ["wlogsmu", "cs"],
      "precision": "ff",
      "rationale_id": "iter_23"
    },
    {"file": "B0m.h", "line_start": 230, "line_end": 230, "variables": ["lm"], "precision": "dd", "rationale_id": "iter_41"}
  ],
  "correctness_summary": {
    "regions_at_threshold": 1362,
    "regions_at_dd_ceiling": 5,
    "ceiling_regions": [
      {"location": "B14 B2m.h:401", "final_min_digits": 7.2, "signal_class": "local_cancellation", "attempted_rewrites": ["kahan"]}
    ]
  },
  "algorithmic_rewrites": [
    {
      "file": "B2m.h",
      "line_start": 355,
      "line_end": 360,
      "kind": "kahan",
      "rationale_id": "iter_47",
      "accepted": true
    }
  ],
  "speedup_summary": {
    "float": 407, "ff": 89, "double": 866, "dd": 5
  },
  "iteration_log_path": "runs/qcdloop/strategy/<run_id>/iterations.jsonl"
}
```

## Open questions (RESUME HERE)

Input/output contract sub-questions still to push on:

**Q1. `precision_assignment` key shape. LOCKED 2026-07-17: array of
region records `{file, line_start, line_end, variables, precision,
rationale_id}`.** Not a taste call — `ff_integrator/agent.py`
docstring already commits to a region contract
`(file, line_start, line_end, variables)`. Line-keyed strings would
force Patcher to reinvent the span + variable set for every intent
and break replay of the precision-map artifact against ff_integrator.
Array (not dict) so overlapping/nested spans stay expressible;
`rationale_id` cross-references the iteration log; single-line
regions just have `line_start == line_end`. Each accepted patch =
one entry (aligns with per-patch commits in Q3).

**Q2. `algorithmic_rewrites` shape. LOCKED 2026-07-17: top-level peer
field, same region-record shape as `precision_assignment`.** Rewrites
are orthogonal to precision (a `double` region can be Kahan-summed and
stay `double`; a `dd` region can carry a rewrite at the ceiling). One
region can carry BOTH — join by `(file, line_start, line_end)` for the
full remediation stack. Also removed from `speedup_summary`: rewrites
are correctness-mode remediations (Kahan for cascade, identity for
local-cancellation), not speedup moves.

**Q3. Patch commit strategy. LOCKED 2026-07-17: per-patch git commits
on `strategy/<run_id>` branch + cumulative diff at end.**

- Fully autonomous workflow (no human in the live loop). Commits are
  post-hoc forensics + downstream-agent consumption + `git bisect`
  target, not live-review UI.
- `run_id` = `YYYYMMDD_HHMMSS_<8char-hash>`, Strategy generates it.
  Caller supplies starting SHA; Strategy creates the branch and returns
  the name in output.
- Commit message schema (machine-parseable, one field per line):
  ```
  [iter_23] promote-ff B2m.h:355-360

  variables: wlogsmu, cs
  signal_class: cancellation_cascade
  max_rel_err: 3.2e-6 -> 8.1e-11
  min_precise_digits: 5.5 -> 10.1
  rationale_id: iter_23
  ```
- Cumulative diff written to
  `runs/qcdloop/strategy/<run_id>/final.diff` at end of run.
- Rejected patches: iteration log only, no commit.
- `git commit` failure = hard-fail run with `status: "internal_error"`.
  Silent recovery would corrupt the audit trail.
- No batching / toggles / opt-outs. Git overhead is noise vs Validator
  runtime.

**Q4. Markdown report alongside JSON. LOCKED 2026-07-17: yes, always,
single file at `runs/qcdloop/strategy/<run_id>/report.md`.**

Both consumers exist today: JSON for downstream agents, markdown for
Reet. Ceiling regions get top billing (that's the interesting output).
JSON is a strict superset of what markdown surfaces — markdown is a
pure projection, not additional data.

Sections:
- Header (status, tolerance, duration, starting SHA, final branch)
- **Ceiling regions** (top billing — physics limits needing attention)
- Precision distribution table
- Algorithmic rewrites accepted
- Iteration summary + pointer to `iterations.jsonl`

**Q5. Caller of Strategy. LOCKED 2026-07-17: LangGraph orchestrator,
single caller, no separate human CLI entry point.**

- Two entry points = two contracts to drift. The orchestrator IS the
  human interface (via `agents/cli.py`).
- Strategy signature stays as stubbed: `run(state: PipelineState) -> dict`.
- "Inputs" list above = PipelineState fields Strategy reads.
- Return value = state delta (thin pointer bundle). Fat artifacts
  live on disk under `runs/qcdloop/strategy/<run_id>/`.

State delta shape:
```python
{
  "strategy_result": {
    "status": "success" | "partial" | "budget_exhausted" | "internal_error",
    "run_id": "20260717_112200_a3f8b1c2",
    "final_branch": "strategy/20260717_112200_a3f8b1c2",
    "report_json_path": "runs/qcdloop/strategy/<run_id>/report.json",
    "report_md_path":   "runs/qcdloop/strategy/<run_id>/report.md",
    "cumulative_diff_path": "runs/qcdloop/strategy/<run_id>/final.diff",
  }
}
```

Disk artifacts under `runs/qcdloop/strategy/<run_id>/`:
- `report.json` — full output (precision_assignment, algorithmic_rewrites, summaries)
- `report.md` — human-readable projection of report.json
- `iterations.jsonl` — per-iteration log (accepted + rejected)
- `final.diff` — cumulative diff

---

## All open questions resolved 2026-07-17. Next: **design Patcher**
(its own contract, mechanical work, `ff_integrator` invocation,
build/runtime gates).

---

# Patcher Agent — design notes

Working design started 2026-07-17 with Reet. Same conventions as
Strategy: locked pieces above the open-questions list, no v1/v2
deferral, spec what's needed today.

## Locked pieces

### Regional DD is a real code path (Option A, locked 2026-07-17)

`promote-dd` on a region goes through a **regional-DD integrator**
that mirrors `ff_integrator` (same `(file, line_start, line_end,
variables)` input, same shim + boundary-patch output). The rest of
the app stays double.

The existing whole-app DD path stays — it's still the Validator's
ground truth. Two DD callers with different needs:

- **Validator:** whole-app DD build for ground truth (existing
  `agents/dd_integrator/` stub — today just points at the hand-written
  qcdloop DD triple).
- **Patcher:** regional DD promotion per remediation intent (new).

`ff_integrator` and the regional-DD integrator are structural twins;
shared boundary-patch machinery parameterized on scalar type
(`ffloat` vs `ddouble`) lives in `agents/integrator_base/`.

## P1. Remediation-intent schema (Strategy → Patcher) — LOCKED 2026-07-17

`target` reuses Strategy's `precision_assignment` region-record shape
(first four fields). `kind` derives from the current→target precision
delta; `rationale_id` is a passthrough into the iteration log.

```json
{
  "target": {
    "file": "B2m.h",
    "line_start": 355,
    "line_end": 360,
    "variables": ["wlogsmu", "cs"]
  },
  "kind": "promote-ff",
  "current_precision": "double",
  "rationale_id": "iter_23"
}
```

- **Same region shape everywhere.** Zero translation between Strategy
  output and Patcher input.
- **`current_precision` explicit, not derived.** Strategy owns
  ladder-walk state; Patcher shouldn't re-derive it from the working
  tree.
- **`kind` values — AMENDED in P3 (see below).** Original draft had
  eight `promote-/demote-<target>` values, but `demote-<target>` was
  ambiguous (double could demote to `float` OR to `ff`, both
  legitimate). Amended shape: transition-only `kind`
  (`<source>-to-<target>`) plus a peer `intent` field carrying
  `correctness` | `speedup`. See P3 for the full kind list and the
  `intent` addition to the intent schema.
- **No free-text `rationale` field.** Prose lives in iteration log,
  keyed by `rationale_id`; Patcher acts on structured fields only.

## P2. Return contract (Patcher → Strategy/Validator) — LOCKED 2026-07-17

Patcher returns a **candidate SHA on `strategy/<run_id>`**, not a diff
string. Patcher owns the git commit (parent = current branch HEAD);
Strategy resets branch tip to `parent_sha` on reject (option (a)).
Rejected commits become dangling, reachable via reflog for a while.

```json
{
  "status": "ok" | "llm_gen_failed" | "patch_apply_failed" |
            "commit_failed" | "build_failed" | "runtime_crashed" |
            "runtime_nan" | "timeout",
  "candidate_sha": "a3f8b1c2..." | null,
  "parent_sha": "b7e4d02f..." | null,
  "artifacts": {
    "shim_paths": ["runs/qcdloop/strategy/<run_id>/shims/B2m_ff.h"] | null,
    "boundary_patch_path": "runs/qcdloop/strategy/<run_id>/patches/iter_23.patch" | null,
    "build_log_path": "runs/qcdloop/strategy/<run_id>/logs/iter_23_build.log" | null,
    "runtime_log_path": "runs/qcdloop/strategy/<run_id>/logs/iter_23_runtime.log" | null
  },
  "error": {
    "kind": "compile" | "nan" | "crash" | "integrator" | "llm" | "apply" | "commit" | "timeout" | null,
    "detail": "<short structured message>" | null,
    "excerpt_path": "runs/qcdloop/strategy/<run_id>/errors/iter_23.txt" | null
  }
}
```

**Rationale for SHA-over-diff:**

- Aligns with Strategy Q3: Strategy commits per-patch on
  `strategy/<run_id>`; Patcher creating the commit avoids splitting
  git-write responsibility across two agents.
- Validator consumes a buildable tree; SHA means it does
  `git checkout && build && run`. Diff means Validator inherits an
  apply-conflict failure mode that isn't its problem.
- Rejected candidate SHAs stay reachable via reflog for forensics.
- Shim + boundary patch can run to hundreds of lines; SHA is 40 chars.

**Status enum (8 values, exhaustive):**

- `ok`
- `llm_gen_failed` — LLM/integrator produced no usable output
  (subsumes what a separate `integrator_failed` would cover;
  intent `kind` + `error.detail` disambiguates source)
- `patch_apply_failed` — diff doesn't apply to current tree
  (distinct from `build_failed`: apply-fail ⇒ regen with fresh
  context; build-fail ⇒ the code is wrong)
- `commit_failed` — git commit rejected; Strategy trips its Q3
  hard-fail (`internal_error`)
- `build_failed` — compile error
- `runtime_crashed` — process died / segfault during smoke run
- `runtime_nan` — smoke run produced NaN or Inf
  (separated from crash: crash = broken control flow, NaN = math
  went sideways; Strategy may treat them differently)
- `timeout` — wall-clock exceeded at any stage
  (kept standalone so retry policy can back off aggressively
  regardless of which stage timed out)

**Patcher's runtime contract:** compiles + doesn't crash + doesn't
emit NaN. No divergence sanity check — that's Validator's judgment
call. Keeps Patcher's responsibilities tight.

## P3. Dispatch by `kind` — LOCKED 2026-07-17

### Ladder is cost-ordered, not strictly precision-ordered

Nominal digits: float ~7, ff ~14, double ~15–16, dd ~30–31. But **ff
and double are within one digit and can trade in either direction**
(ff has 2× float mantissa but no extended exponent — in some regimes
double wins). The ladder `float → ff → double → dd` is a **cost**
ordering; precision is mostly-ordered with ff/double as effective
peers.

**Consequence:** `double → ff` is a first-class speedup move (cheaper
on hardware with fast float, precision usually holds). Strategy's
speedup walk `double → ff → float` is correct as written.

### Kind vocabulary — transition + intent (amends P1)

Original `promote-/demote-<target>` was ambiguous (`demote-double`
could target `ff` or `float`). Replaced with pure transition kinds +
separate `intent` field.

**Intent schema addendum:**

```json
{
  "target": {...},
  "kind": "double-to-ff",
  "intent": "correctness" | "speedup",
  "current_precision": "double",
  "rationale_id": "iter_23"
}
```

**Kind values (exhaustive):**

```
kind                    dispatch path              intent typically
───────────────────────────────────────────────────────────────
float-to-ff             regional-integrator (ff)   correctness   [added 2026-07-17]
float-to-double         plain-type-edit            correctness
double-to-ff            regional-integrator (ff)   correctness OR speedup
double-to-dd            regional-integrator (dd)   correctness
ff-to-double            git-revert (strip ff)      correctness (rare)
ff-to-dd                composite: revert ff, install dd  correctness
double-to-float         plain-type-edit            speedup
ff-to-float             composite: revert ff, then double→float  speedup
dd-to-double            git-revert (strip dd)      speedup
reformulate-kahan       llm-rewrite                correctness
reformulate-identity    llm-rewrite (identity picked by Strategy)  correctness
```

**Amendment 2026-07-17 (post-Cluster-Claude Strategy impl).** Original
table had 8 transition kinds and missed `float-to-ff` — needed for the
single-step up-walk from a float baseline. Added.

**Latent edge (not resolved):** `float-to-dd` is NOT in the vocabulary.
A fully general correctness walk from float would need it; today the
walk from a float baseline caps at `double` (walk status `exhausted`
if double still doesn't clear). Doesn't fire in the fixed-report
workflow (correctness baselines are always `double`), so latent, not
live. Revisit if a float-baseline correctness case ever surfaces.

### Four dispatch paths (not ten)

1. **Regional-integrator** — `float-to-ff`, `double-to-ff`,
   `double-to-dd`. Install a regional shim + boundary patch.
   `ff-to-dd` is composite (revert ff first, then install dd).
2. **Plain-type-edit** — `float-to-double`, `double-to-float`.
   Mechanical AST edit inside the region. Both types are native
   scalars; implicit conversion handles crossings.
3. **Git-revert** — `ff-to-double`, `dd-to-double`. Look up the
   introducing commit for the region from Strategy's iteration log,
   `git revert` on strategy branch. `ff-to-float` is composite
   (revert ff, then plain-edit double→float).
4. **LLM-rewrite** — `reformulate-kahan`, `reformulate-identity`.
   LLM generates a source rewrite; prompt gets region source +
   `variables` list + (for identity) the specific identity Strategy
   selected.

### P3a. Plain-type-edit implementation — LOCKED: AST-aware (libclang)

Use libclang to do type-node-only rewrites. Naive `sed`-style
substitution corrupts identifiers (`floating_point`, `float_traits`),
comments, string literals. Constraining intents to
"one-declaration-per-region" was the other option; rejected because
it distorts the region shape from characterization and pushes a
Patcher-side dependency onto Strategy — wrong direction.

### P3b. `reformulate-identity` — LOCKED: Strategy picks the identity

Add `identity` field to intent schema when
`kind == "reformulate-identity"`:

```json
{
  "kind": "reformulate-identity",
  "identity": "log1p" | "expm1" | "hypot" | "1-cos->2sin2" | ...,
  ...
}
```

Strategy has the signal-class context to know which identity fits.
Narrower LLM prompt = higher success rate. Retry walk becomes
deterministic ("tried log1p, next try hypot") instead of relying on
LLM sampling variety.

Starter identity catalog (extend as we find more):
- `log1p` — `log(1+x)` → `log1p(x)`
- `expm1` — `exp(x)-1` → `expm1(x)`
- `hypot` — `sqrt(x*x+y*y)` → `hypot(x,y)`
- `1-cos->2sin2` — `1 - cos(x)` → `2*sin(x/2)*sin(x/2)`

## P4. `ff_integrator` invocation — LOCKED 2026-07-17

### Call shape

```python
result = ff_integrator.integrate_region(
    file=intent.target.file,
    line_start=intent.target.line_start,
    line_end=intent.target.line_end,
    variables=intent.target.variables,
    working_tree=strategy_branch_head,   # SHA, not path
    scalar_type="ffloat",                # "ddouble" for regional-DD
    direction="in",                      # always "in" from Patcher (out = git-revert)
    out_dir=f"runs/qcdloop/strategy/{run_id}/shims/",
    attempt=attempt_index,               # for LLM seed/temperature variation
)
```

**Working tree passed as SHA, not path.** ff_integrator uses `git show
<sha>:<file>` or `git worktree add` internally. Pins the exact tree the
integrator saw so output is reproducible under parallel iterations (a
future scenario we don't want to preclude today).

### No vanilla pre-compile check

If characterization succeeded on this region, vanilla compiles by
construction. Re-verifying every intent doubles build time for zero
information in the common case. If the working tree HAS drifted, the
post-integrator build fails and surfaces as `build_failed` — correct
behavior, no special-casing needed.

### Cheap pre-check (file existence, line range, variable names)

Before calling the integrator:
- File exists in working tree at `intent.target.file`?
- File has ≥ `line_end` lines?
- All names in `variables` appear as identifiers somewhere in
  `[line_start, line_end]`?

Fail ⇒ return `patch_apply_failed` immediately (malformed intent is
same category as "diff doesn't apply"; reuses P2 status enum).

### Bounded retry loop around integrator + build (P4a, P4b resolved)

**Distinct from Strategy retry.** Two retry semantics that shouldn't mix:

- **Semantic retry** (Strategy): "this intent didn't work, try a
  different intent" (e.g., `double-to-ff` rejected → `double-to-dd`).
- **Mechanical retry** (Patcher): "same intent, LLM misgen — re-roll."

Patcher owns mechanical retries; Strategy never sees them. Matches the
whole-app tracked integrator precedent (MEMORY.md: C3/C6 misgens,
~5-10% wasted-attempt rate, retry-and-move-on wins over invalidating
shim cache).

```python
MAX_INTEGRATOR_RETRIES = 3   # single shared budget: integrator + build

for attempt in range(MAX_INTEGRATOR_RETRIES):
    result = ff_integrator.integrate_region(..., attempt=attempt)
    if result.status == "ok":
        build_result = build_and_smoke(...)
        if build_result.status == "ok":
            return commit_and_return_ok(...)
        if is_retryable_misgen(build_result.error):
            continue
        return build_failed(build_result.error)
    if result.status == "llm_failed":
        continue
    return llm_gen_failed(result.error)

return llm_gen_failed(f"integrator failed after {MAX_INTEGRATOR_RETRIES} attempts")
```

**P4a. What counts as "retryable misgen"? — LOCKED: everything, for now.**
Regional-integrator misgen patterns don't exist yet (we haven't run
regional integration at scale). Start with `is_retryable_misgen ==
lambda _: True`; collect real failure data during Stage 4 / Iteration
1, add pattern-matching later when we know what we're matching. Passes
the "spec what's needed today" test — what we can spec is bounded
retry with N=3; the classifier isn't specifiable yet.

**P4b. Retry budget: per integrator call, or per intent? — LOCKED: per
intent, shared with build failures.** Single shared budget of N=3
covers all combinations. Patcher can't eat 9 LLM calls on one intent
(3 integrator × 3 build).

### Dependency on P7

Call shape is stable regardless of how P7 (regional-DD integrator
module structure) resolves — only the import path changes. If P7
lands on "one module, two functions" (my lean), `double-to-dd` calls
`dd_integrator.integrate_region(..., scalar_type="ddouble")`. If P7
splits modules, the call becomes `regional_dd_integrator.integrate(...)`.
Same signature, same working-tree=SHA requirement, same bounded retry.

## P5. Build + runtime gates — LOCKED 2026-07-17

**Patcher builds only the vanilla driver (with regional shim(s)
installed) and runs a small deterministic smoke-test. No DD build,
no full 100k run.**

### What Patcher builds

Vanilla driver (`runs/qcdloop/src/boxGPU_vanilla.cpp`) linked against
qcdloop headers + whatever regional shim(s) Strategy has accumulated
on `strategy/<run_id>` branch. This is the *candidate* build — what
Validator later measures against DD ground truth.

**Not built by Patcher:**
- **DD driver.** Validator's job. Expensive whole-app build; result
  is only useful for numerical comparison, which is Validator's
  contract.
- **Tracked driver.** Characterizer territory. Fixed report already
  exists; tracked doesn't re-enter the loop.

Division: Patcher builds only what's needed to prove the candidate
compiles and doesn't crash.

### Smoke recipe

**1 sample per integral (21 samples total)**, deterministic seed
(`srand(12345)` per qcdloop convention, MEMORY.md), fixed recipe
from `boxGPU_app_recipes.hpp`.

Purpose is "does this compile-clean patch produce numbers?" — not
"is the patch numerically correct." Numerical correctness is
Validator's judgment over statistically meaningful samples.

21 samples ≈ 2 seconds runtime; noise vs build (~30-60s) and
Validator (~minutes). Catches: link errors that only fire at
runtime, NaN from divide-by-zero the region introduced, memory-access
crashes, infinite loops (via timeout).

### NaN/crash check

Scan smoke-run output for:
- Any `nan|NaN|NAN|inf|Inf|INF` token in coefficient columns →
  `runtime_nan`
- Fewer than 21 output rows (crash mid-run) → `runtime_crashed`

Both already in P2 status enum.

### Timeouts

- Build: **5 min hard** → `timeout`
- Smoke run: **30 sec hard** → `timeout`

Generous vs expected runtime, tight enough to catch runaway loops.
Tuneable after Stage 4 data lands.

### Environment plumbing

Reuse `build_and_run` agent's HPC module-load wrapper:
`bash -lc 'module use /soft/modulefiles && module load
gcc/13.3.0 cmake/3.28.3 && ...'`. Env-overridable via
`PIPELINE_MODULE_LIST`, `PIPELINE_MODULE_USE_PATH`. Same plumbing,
different caller — no duplication.

### Config notes (not design decisions)

- **One build per intent** that reaches the build stage — each
  accepted patch changes the tree.
- **Ensure ccache (or equivalent) is on** — repeated near-identical
  builds will hit the cache and drop rebuild time to seconds.
- **Smoke run stdout archived** to
  `runs/qcdloop/strategy/<run_id>/logs/iter_N_runtime.log` per P2's
  `artifacts.runtime_log_path`.

## P6. Failure modes back to Strategy — LOCKED 2026-07-17

P2's 8 status values split into three semantic buckets from Strategy's
perspective. This is the mapping table Strategy's retry walk uses.

### Three buckets

**Bucket A — Signal about the intent.** Intent was tried honestly, it
didn't work. Strategy treats like Validator `reject` and advances the
vocabulary walk.
- `build_failed` — transition doesn't compile for this region
- `runtime_nan` — transition breaks the math
- `runtime_crashed` — transition breaks control flow

**Bucket B — Patcher couldn't try the intent.** Signal about Patcher's
capacity, not the intent's viability.
- `llm_gen_failed` — Patcher's bounded N=3 retries exhausted
- `patch_apply_failed` — malformed intent (Strategy's bug)
- `timeout` — ambiguous stage; retry once then fold into Bucket A

**Bucket C — Fatal infrastructure.** Workflow can't continue.
- `commit_failed` — already locked in Q3 as `internal_error`, hard-abort

### Strategy retry response table

```
Patcher status         Strategy response                              log tag
─────────────────────────────────────────────────────────────────────────────
ok                     Hand SHA to Validator; verdict decides         (validator sets)
build_failed           Treat as reject, advance walk                  compile
runtime_nan            Treat as reject, advance walk                  runtime_nan
runtime_crashed        Treat as reject, advance walk                  runtime_crash
llm_gen_failed         Log, advance walk (don't count vs budget)      llm_capacity
patch_apply_failed     Log with strategy_bug=true, skip intent        strategy_bug
timeout                Retry same intent once;                        timeout
                       second timeout → advance walk (as build_failed)
commit_failed          Hard-abort run, status="internal_error"        fatal
```

### P6a. `llm_gen_failed` at DD level ≠ physics ceiling — LOCKED

If Strategy walks `double → dd` and Patcher returns `llm_gen_failed`
(integrator LLM couldn't generate the DD shim after 3 tries), Strategy
MUST NOT conclude "region at DD ceiling." That would falsely flag a
physics limit that's actually an LLM capacity limit.

**Rule:** DD ceiling is reached ONLY when Patcher returns `ok` AND
Validator returns `reject`. Any Patcher failure at the DD level → log
as `dd_untested`, not `dd_ceiling`. The `ceiling_regions` list in
Strategy's output distinguishes these two:

```json
"ceiling_regions": [
  {"location": "B14 B2m.h:401", "final_min_digits": 7.2,
   "signal_class": "local_cancellation",
   "ceiling_kind": "dd_ceiling",
   "attempted_rewrites": ["kahan"]},
  {"location": "B15 B0m.h:230", "final_min_digits": null,
   "ceiling_kind": "dd_untested",
   "reason": "llm_gen_failed after 3 attempts on double-to-dd"}
]
```

### P6b. Strategy does not retry `llm_gen_failed` — LOCKED

Patcher already spent 3 attempts. Strategy immediately reissuing the
same intent would burn 3 more — up to 3N calls per stuck region.

**Rule:** `llm_gen_failed` is terminal for the (intent, region) pair
within a single Strategy run. Strategy advances the walk immediately,
doesn't retry. If ff-gen failed at region R, Strategy doesn't retry
ff-gen at R even if it later tries `ff-to-dd` and reverts back to
double.

## P7. Regional-DD integrator module structure — LOCKED 2026-07-17

**Option (a): one module, two functions.** Extend
`agents/dd_integrator/` with an `integrate_region` sibling to
`integrate_whole_app`. No new module, no rename.

### Reasoning

1. **Code reuse is the point.** Whole-app and regional DD share more
   than ff and DD do: same underlying headers
   (`third_party/include/{dd_math.hpp, dd_complex.hpp}`), same
   `quad::ddfun::ddouble` scalar type, same DD-specific constant-table
   hex codegen (hex-encoded `(hi, lo)` double pairs per MEMORY.md).
   Splitting into two modules immediately spawns a shared-helper file
   — (a) with extra ceremony.
2. **Callers already disambiguate by signature.** Validator calls
   `integrate_whole_app(headers, driver)`; Patcher calls
   `integrate_region(file, line_start, line_end, variables, ...)`.
   Different signatures, different call sites, zero dispatch
   ambiguity. Module boundary would enforce nothing the function
   signatures don't already enforce.
3. **Symmetry with `ff_integrator`.** ff will eventually grow a
   whole-app mode too. Precedent set here propagates cleanly:
   `ff_integrator.integrate_whole_app` next to `.integrate_region`.
4. **Simpler mental model.** "DD integrator = where DD generation
   lives" beats "DD integrator does whole-app, regional-DD integrator
   does regions."

### Module shape

```python
# agents/dd_integrator/agent.py

def integrate_whole_app(
    target_library_headers: Path,
    driver_source_path: Path,
    ...
) -> Path:
    """Whole-app DD integration. Callers: Validator (ground-truth build)."""
    # today: stub that validates the qcdloop DD triple exists
    # future: LLM-driven codegen mirroring tracked_integrator

def integrate_region(
    file: str,
    line_start: int,
    line_end: int,
    variables: list[str],
    working_tree: str,
    scalar_type: str = "ddouble",
    direction: str = "in",
    out_dir: Path,
    attempt: int = 0,
) -> RegionIntegrationResult:
    """Regional DD promotion. Callers: Patcher (per remediation intent)."""
    # new; mirrors ff_integrator.integrate_region signature exactly
```

Shared internals (constant-table hex codegen, DD boundary-patch
generator) live as private module-level helpers in
`agents/dd_integrator/`, or lift into `agents/integrator_base/` if
`ff_integrator` needs them too. Refactor decision at implementation
time, not a design lock.

---

## All Patcher questions resolved 2026-07-17. Ready for implementation.

**Design surface complete:**
- Strategy Q1–Q5 (input/output contracts, patch commits, markdown
  report, caller)
- Patcher P1–P7 (intent schema, return contract, kind dispatch,
  integrator invocation, build/runtime gates, failure modes,
  module structure)
- P3a AST-aware type edits (libclang)
- P3b Strategy picks identity for `reformulate-identity`
- P4a retry classifier starts as "retry everything"
- P4b shared N=3 retry budget per intent
- P6a `llm_gen_failed` at DD level ≠ physics ceiling
- P6b Strategy never retries `llm_gen_failed`

