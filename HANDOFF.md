# HANDOFF — three independent tasks (2026-07-19, langgraph-agents)

Three unrelated tasks, shipped one commit each so review/revert is per-task.
Offline test-count deltas reported per task; no full pipeline / LLM rerun.

## Task 1 — shim-include ORDERING fix (`agents/integrator_base/boundary.py`)

**Symptom (from the e1d774a rerun):** `B0m.h:69` past R4 hit *"Constants is not
a class template"* because `boundary._insert_shim_include` spliced the shim
`#include` right after `#pragma once` — i.e. **before** the header's own app
includes that declare the primary templates the shim specializes. Ordering, not
include-set.

**Fix:** `_insert_shim_include` now defers to a new comment-aware preamble scan
`_shim_insert_index`, which places the shim **after the last `#include` in the
preamble** (any form — after the last is trivially after all app ones) and
**before the first code/decl line**. When the header has no includes it falls
back, in priority order, to *after* the last `#include` → include-guard
`#define` → `#pragma once` → top-of-file, so `#pragma once` / classic
`#ifndef`/`#define` guard semantics are preserved (shim never lands before the
guard). A leading `/* license */` block comment no longer truncates the scan
(block/line comments are skipped).

**Surprising bit:** the old fallback prepended at the very top when no
`#pragma once` was found — for a classic include-guard header that put the shim
*outside* the guard. The new fallback handles the `#ifndef`/`#define` idiom
explicitly.

**Verify:** `rerun_failing_regions.py`'s health report gained two per-region
fields — `placement` (`OK(after-app-includes)` vs `BAD(before-app-includes)`)
read off the patched header, and `build_sig` (classifies a build failure as the
`OLD-ORDERING-BLOCKER('is not a class template')` vs a `NEW-REASON: <first
error>` to flag). The summary now counts `shim-ordering blocker` regions (Task 1
target: 0). No full pipeline rerun forced — the script surfaces B0m.h:69's new
state when next run under the module+proxy env.

**Tests:** `tests/integrator_base/test_boundary.py` +5 (app-includes ordering,
`#pragma once`-only fallback, classic include-guard fallback, mixed system+app
includes with a license banner, idempotency). **Offline delta: 8 → 13** in that
file (full `tests/integrator_base/` 45 → 50).

## Task 2 — speedup gate: wire `predicted_rel_err_if_ff` into Strategy

**RECONCILIATION — the task has the two fields backwards; read this first.** The
task asked to *add* `predicted_rel_err_if_float` "peer to the existing
`predicted_rel_err_if_ff`". In the actual tree it is the **other way round**:

* `predicted_rel_err_if_float` (`U_FLOAT = 2**-24`, the textbook float unit
  roundoff — kept, *not* changed to the task's approximate `2**-23`) has existed
  since the first reducer commit (`bc4907f`), is emitted at every classify site
  (region / variable / chain), **is already carried into Strategy's
  `RegionRecord`, already gates `build_speedup_queue`, and already has reducer +
  ranking test coverage.** The double→float speedup gate the task describes was
  already live (fires at tolerance ≤ 6; excluded at 10 — correct).
* `predicted_rel_err_if_ff` (`U_FF = 2**-46`) was added *later* (`0e7233b`,
  2026-07-17, "unlock double→ff speedups at high tolerance") **in the reducer
  only** — Strategy dropped it at the boundary (`RegionRecord` never carried it;
  no gate referenced it). The HANDOFF for that change claims "Strategy's speedup
  gate can now queue double→ff at tolerance 10" — but that half was **never
  wired**. `grep predicted_rel_err_if_ff agents/strategy/` returned nothing.

So the literal Task-2 deliverable was already done; the genuine gap was the
**mirror image**. This commit wires the ff half — the change that actually makes
the speedup queue non-empty on qcdloop (tol=10), which is also the precondition
for Task 3's speedup phase to have anything to walk.

**What changed:**
- `characterization.py`: `RegionRecord` + `ChainRecord` now carry
  `predicted_rel_err_if_ff`; `load_regions` / `load_chains` read it (helper
  `_pred_ff`, **float-fallback** when absent — see below), the per-line merge
  propagates it worst-case, and `ChainRecord.walk_record()` forwards it.
- `ranking.build_speedup_queue`: admission now gates on
  `predicted_rel_err_if_ff <= thr` instead of `predicted_rel_err_if_float`. ff is
  the *loosest* cheaper-than-double rung (ff ~14 digits < float's ~7), so `pred_ff
  <= thr` **subsumes** the old float gate — strictly more regions admitted, never
  fewer. The walk + per-step Validator still decide how far down each region
  actually settles (double→ff, or on to float when float-safe).

**Float-fallback (why existing reports don't regress):** a report predating the
ff signal (`report_1k` / `report_100k`) has no `predicted_rel_err_if_ff`; the
loader falls back to that region's `predicted_rel_err_if_float` — a conservative
*upper* bound (ff is never worse than float), so a stale report admits **no ff
speedup it couldn't already make as a float speedup**. No silent new admissions
from missing data; the true (tighter) ff value requires the backfill or a fresh
run.

**Backfill utility:** `agents/shared/backfill_ff_prediction.py` (+ CLI) rewrites
`predicted_rel_err_if_ff` onto a frozen report without re-characterizing, derived
from `U_FF * max_sensitivity` (exact) or `pred_float * (U_FF/U_FLOAT)` when
sensitivity is absent. **Not run on the on-disk frozen reports** — they are
untracked and enormous (`report_100k.json` is **13.7 GB**, `report_1k.json`
247 MB; a whole-file `json.loads` is impractical). The float-fallback keeps them
usable as-is; the **50k / next characterization run emits `predicted_rel_err_if_ff`
natively**, so backfill is only for one-off use on a normal-sized frozen report.

**Not done / flagged:** I did **not** change `U_FLOAT` to `2**-23` (the existing
`2**-24` is the correct unit roundoff and is asserted exactly by reducer tests).
I did **not** touch the reducer emission (both fields already emitted). If you
*want* the strict float bar back for admission (only demote regions safe all the
way to float), revert the one-line gate in `build_speedup_queue`; the ff field is
still carried for reporting either way.

**Tests:** `tests/strategy/` 56 → **60** (ff carry-through + worst-case merge +
float-fallback in `test_characterization`; ff-only-safe admission, ff-subsumes-
float-at-low-tol, and the renamed not-even-ff-safe exclusion in `test_ranking`).
New `tests/shared/test_backfill_ff_prediction.py` **+8** (sensitivity-exact and
float-derived, idempotent, skip-no-float, report/variable/chain coverage,
dict-shaped chains, file round-trip, dry-run). **Offline delta: +12.**

---

# HANDOFF — Gap A (namespace-qualified bridge) + Gap B (source-derivable constants) (2026-07-18, langgraph-agents)

Two shim-completeness gaps the include-set hardening rerun unmasked (both generic,
not qcdloop-specific). Both were previously hidden behind the app-source-include
error / the R4 escape hatch firing first; with those resolved they became the next
accept-rate levers.

## Reproduced first (run 20260718_194556_67dbcf37, confirmed on a fresh rerun)

* **Gap B** — `_ieps50` (source definition `TScale(1e-50)`, a plain `double`
  literal) tripped the Rule R4 `#error` on B0m.h:69 / B2m.h:65: no vendored
  `dd_ieps50()`, no memorized hex pair. When the model *guessed*, it guessed
  **wrong** — hi `0x34F0EE0B102B7182` (correct is `0x358DEE7A4AD4B81F`) plus a
  spurious low word — then honestly bailed to R4. But the value is not a mystery:
  a source double literal carries only double precision, so its faithful dd value
  is `make_dd(<bits of the double>, 0x0)` — a zero low word.
* **Gap A** — a promoted `ddouble` flowing into a **namespace-qualified** math
  call `Ns::fn(x)` (e.g. `Kokkos::fabs`) skips ADL, so the vendored `quad::ddfun`
  overloads are never found and the value narrows to `double`
  (`cannot convert 'quad::ddfun::ddouble' to 'const double'`). Intermittent (the
  model often bridges via the app's own `ql::kAbs`/`ql::Max` overloads, which
  forward to `quad::ddfun::abs` and dodge Kokkos entirely).

## What landed

**Gap B — R3 becomes a 4-step cascade + a deterministic codegen helper.**
- `agents/shared/constant_derive.py` (new): a framework-agnostic derivation module.
  `resolve_constant_rhs(name, sources)` walks scan-reachable source to a constant's
  own definition (`#define` / `constexpr` / `const` / literal-returning accessor)
  and returns its RHS; `derive_from_rhs` runs the cascade — **(1)** vendored
  `dd_*()`/`ff_*()`, **(2)** known hex pair, **(3)** derive from source RHS (3a: a
  source `double`/`float` literal → `make_dd(bits, 0)` with a *zero* low word —
  correct, not truncation; 3b: a closed form over a small **catalog** of
  mathematical constants π/2π/π-2/e/√2/ln2/ln10/γ computed at import via the Bailey
  split and verified bit-exact against the vendored `dd_*`/`ff_*` pairs), **(4)**
  Rule R4 only if 1–3 all fail. `derive_literals_in` surfaces the literals of a
  composite RHS (e.g. the complex `{0, 1e-50}` of `_ieps50`) so the model assembles
  the container without guessing bits.
- `agents/integrator_base/regional.py`: scans the region for constant candidates
  (`derive_region_constants`), resolves + pre-derives them, and injects a
  **"Source-derivable constants"** section into the user turn with ready-made
  `make_dd(...)`/`make_ff(...)` values to use verbatim. The dd/ff system prompts'
  R3 (and the dd `constant_note`) were rewritten as the ordered cascade.

**Gap A — C3 refined + a deterministic bridge lint.**
- dd/ff system prompts: new **C3 (namespace-qualified math bridge)** paragraph —
  describes the ADL-skip *pattern* (`Ns::fn(promoted)` where `Ns` is neither the
  vendored nor the app namespace) with **(a)** inject an overload into `Ns`
  forwarding to the vendored op (preferred) / **(b)** a using-declaration fallback
  when injecting there is documented-forbidden. Framework-agnostic: no framework is
  named in the rule; Kokkos/std/sycl/cuda::std appear only as illustrations.
- `agents/integrator_base/regional.py`: `find_qualified_math_calls` (region scan
  keyed on a standard `<cmath>`/`<complex>` name set + the boundary dataflow's
  promoted-name set, factored out as `boundary.compute_promoted_names`) feeds a
  user-turn hint **and** a post-generation lint `_lint_qualified_bridges` that
  rejects a shim missing a bridge for a qualified math call on a promoted operand
  as a retryable `llm_failed` (same semantics as the C1 include lint — a failed gen
  never counts against the Strategy transition budget).

## Design choices

- **Gap A (a) vs (b).** The prompt prefers (a) namespace injection unless the
  target framework *documents* it as forbidden (a `static_assert` guard / an ADR
  against user additions), in which case (b) the using-declaration is the fallback
  with a comment. No framework is currently known to forbid it, so (a) is the live
  path; (b) is a documented escape, not a guess. The lint accepts either.
- **Gap A lint scope (few false positives).** The lint only fires on a *standard
  math* function name, invoked *qualified*, on an operand that the boundary
  dataflow actually promotes. App-specific wrappers (`ql::kAbs`) are deliberately
  outside the math-name set — they are bridged transitively and the region-text
  lint cannot see which level the shim chose, so flagging them would false-reject a
  buildable shim. Residual risk (a standard math call transitively bridged via a
  different namespace) is documented and rare in practice.
- **Gap B cascade / the "zero low word" subtlety.** Step 3a is the crux: a
  constant the *source* defines as a plain `double` literal has no precision below
  that double, so `make_dd(bits, 0)` is the faithful promotion — this is explicitly
  *not* the forbidden decimal-literal case (that rule guards π/e-style constants
  whose true value out-runs a double). The catalog exists only for step 3b closed
  forms, where taking the double bits *would* lose real precision. The helper
  reads generic C++ constants; no qcdloop symbols in the module or the catalog.

## Tests (+37 offline; full offline suite 244 → 281 passed; +3 @pytest.mark.llm)

- `tests/shared/test_constant_derive.py` (18): catalog matches vendored dd/ff
  pairs bit-for-bit; a double literal → zero dd low word (the exact `_ieps50` bits
  the model got wrong); cast/suffix parsing; source-walk over `#define`/`constexpr`/
  accessor/template-accessor forms (+ function-like-macro rejection); catalog
  closed forms (M_PI, 2·M_PI, M_PI·0.5, std::numbers::pi_v); composite-RHS literal
  enumeration.
- `tests/integrator_base/test_bridge_lint.py` (13): detector picks up Kokkos/std/
  sycl/`cuda::std` on promoted args, ignores `quad::`, non-promoted args, non-math
  fns, `Type<…>` accessors; lint rejects a missing bridge, passes for namespace-
  injection / using-decl / using-namespace, reports multiples.
- `tests/integrator_base/test_gap_integration.py` (6): both gaps end-to-end through
  the real engine with a canned LLM — Gap-A hint + lint (synthetic `std::sqrt`
  promoted), Gap-B hint (synthetic `constexpr double MY_TINY = 1e-40`), composite
  `_ieps50` derivation.
- `tests/dd_integrator/test_regional.py` (+3 `@pytest.mark.llm`): live-model
  reproduction — the `_ieps50` region generates with **no R4** and the correct
  `0x358dee7a4ad4b81f` bits; a synthetic derivable constant never R4s; a synthetic
  `std::sqrt(promoted)` gets a bridge. All pass against the Argo proxy (port 8084).

## End-to-end rerun (`runs/qcdloop/rerun_failing_regions.py`, extended with an
R4/bridge health report). Four regions through the **real Patcher** (generate →
build-gate → commit):

| region | before | after |
|---|---|---|
| `B3m.h:177` | builds (include fix) | **builds** |
| `B2m.h:65`  | R4 on `_ieps50` → `llm_gen_failed` | **builds** (Gap B derived `_ieps50`) |
| `B1m.h:62`  | R4 on `_ieps50` (HANDOFF) | **builds** (Gap B) |
| `B0m.h:69`  | R4 on `_ieps50` → `llm_gen_failed` | **derives `_ieps50` correctly (no R4, correct bits)** — now blocked by a NEW, out-of-scope error (below) |

**built (P2 ok): 2/4 → 3/4.** Gap B is demonstrably fixed on real regions (B2m
promoted; B0m's shim now emits `make_dd(0x358dee7a4ad4b81fULL, 0x0)` instead of an
R4 `#error`). LLM nondeterminism means an individual region's *before* status
varies run-to-run; the R4-on-`_ieps50` mode is the consistent signature.

## NEW failure mode flagged (needs Reet — out of scope for these two gaps)

Getting B0m.h:69 past R4 unmasked a **shim-include-ordering** bug (distinct from the
include-*set* fix): the boundary patch inserts `#include "<shim>"` immediately after
`#pragma once`, i.e. **before** the header's own `#include "box_common.h"` — which is
what transitively declares the `ql::Constants` primary template the shim
specializes. Result: `error: 'Constants' is not a class template / explicit
specialization of non-template 'ql::Constants'`. The shim is otherwise correct
(clean includes, no R4, correct `_ieps50` bits). Fixing it means changing
`agents/integrator_base/boundary._insert_shim_include` to place the shim **after**
the region file's existing app includes (not just after `#pragma once`) — a
boundary-patch design change with its own correctness surface (the shim must still
precede the region body, which it will). Flagged rather than fixed unilaterally, per
the task's stop-and-flag instruction. This is the next accept-rate lever for
header-file regions that specialize app class templates.

---

# HANDOFF — DD/FF shim include-set hardening (2026-07-18, langgraph-agents)

Prompt-hardened the regional integrators against **hallucinated app-source
includes** — the #1 accept-rate lever from the 2026-07-18 shakedown
(`runs/qcdloop/strategy/20260718_194556_67dbcf37`). In that run 11 DD regions
came back `dd_untested` (P6a Patcher failures); the dominant build-gate death was
DD shims emitting `#include "ql/constants.h"`, `<qcdloop/qcdloop.h>`,
`<qcdloop/types.h>`, `<Kokkos_Macros.hpp>`, … — app-source paths not on the
shim's include path, so every such build died with
`fatal error: <path>: No such file or directory` before the shim was ever
honestly tested.

## Root cause

Both regional prompts' `Output:` section ended with
`#include`s of the vendored headers **"(and any target headers the shim needs)"** —
an explicit invitation to pull app headers. The shim is textually included INTO
the region's own translation unit, where the library's declarations
(`ql::Constants<T>`, `ql::Max`, Kokkos macros) are already visible, so re-including
their headers is never needed and always breaks (the boundary patch owns all
caller-side wiring). Failure was intermittent (nondeterministic generation), so
this is prompt hygiene, not a structural bug.

## What landed

1. **Prompt (primary fix).** New numbered rule **C1 "closed include set"** in both
   `agents/dd_integrator/system_prompt.txt` and
   `agents/ff_integrator/system_prompt.txt` (mirrors the existing C-rule style —
   enumerated allowlist + rationale + what to do instead). The `Output:` clause
   `(and any target headers the shim needs)` was replaced with
   `vendored headers ONLY … (see C1: no app-source headers)`.
2. **Deterministic lint (safety net).** `agents/integrator_base/regional.py`
   (`_lint_include_set` / `_allowed_include_set`, `RegionalSpec.allowed_includes`,
   `_STDLIB_HEADERS`). Any `#include` outside {vendored headers ∪ stdlib} makes
   `run_integrate_region` return `RegionIntegrationResult.failed(...)` →
   `Gen(False, LLM_GEN_FAILED)` → the Patcher's **N=3 retry re-rolls** exactly as
   for any misgen. A lint reject never produces an accepted transition, so it does
   **not** count against the Strategy transition budget. Shared engine → both ff
   and dd inherit it. Stdlib headers are allowed (harmless, always on path) to
   avoid false-rejecting an otherwise-buildable shim.

## Tests (+11 offline; full offline suite 233 → 244 passed)

- `tests/integrator_base/test_include_lint.py` (9): each observed hallucination
  rejected, clean/stdlib/quoted-vendored pass, commented `#include` ignored,
  cross-vendored (ff header in a dd shim) rejected, multi-forbidden all reported.
- `tests/dd_integrator/test_regional.py` (+2 offline): ruleset carries C1;
  a forbidden-include shim becomes retryable `llm_failed` and is **not** persisted
  into the candidate tree.
- `tests/dd_integrator/test_regional.py` (+2 `@pytest.mark.llm`, skip if proxy
  absent): regenerate the exact previously-failing regions `B2m.h:65` /
  `B4m.h:163` with the live model → include set is clean. **Passed** against the
  Argo proxy (port 8084).

## End-to-end proof (`runs/qcdloop/rerun_failing_regions.py`)

Drove 4 previously-`dd_untested` regions through the real Patcher (generate →
build-gate → commit), not the full loop:

| region | prior failure | now |
|---|---|---|
| `B3m.h:177` | `<qcdloop/constants.h>` (include-only) | **builds (P2 ok)** |
| `B2m.h:65`  | `ql/constants.h` (+ `_ieps50`) | **builds (P2 ok)** |
| `B0m.h:69`  | `<qcdloop/qcdloop.h>` (+ `_ieps50`) | fails — Kokkos overload gap (not includes) |
| `B1m.h:62`  | `qcdloop/*` (+ `_ieps50`) | fails — **Rule R4 escape hatch on `_ieps50`** (by design) |

Across all reruns (up to 3 attempts each): **zero** `No such file` include errors,
**zero** forbidden includes in any generated shim. The include-hallucination mode
is eliminated. 2/4 build; the other 2 fail for reasons unrelated to this fix.

## Flag for the 50k run (other patterns the lint surfaced downstream)

Two failure modes remain in the `_ieps50`/`_2ipi` family that are **not** include
bugs and were previously masked by the include error firing first:

1. **R4 escape hatch on un-vendored DD constants.** `_ieps50` (1e-50),
   `_reps()`, `_2ipi` (2πi) have no vendored `dd_*()` and no known hex `(hi,lo)`,
   so the model correctly emits the Rule R4 `#error`. This is honest, not a
   regression — but it's a hard ceiling until `scripts/gen_dd_constants.cpp` emits
   those pairs (or they're added to the vendored table). Worth pre-generating the
   qcdloop constant set before the 50k run so these regions can actually promote.
2. **Kokkos math overload gap** (`B0m.h:69`): a promoted `ddouble` flows into
   `Kokkos::fabs` / `Kokkos_Complex.hpp` internals that have no `ddouble` overload
   (`cannot convert 'quad::ddfun::ddouble' to 'const double'`). This is a
   shim-completeness / C3 static-instantiation issue, orthogonal to includes —
   flagging as the next accept-rate lever after this one.

Reproduce: `python runs/qcdloop/rerun_failing_regions.py` under the venv +
`module load gcc/13.3.0 cmake/3.28.3`.

---

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

---

# Region-local write extraction (Fix C) — implementation handoff

Scope: `agents/shared/region_scan.py` — the source-scan companion to the
characterizer's region-local *reads*. Delivers `extract_region_writes(file,
line_start, line_end, working_tree, tracked_type="Tracked") -> list[str]`, the
tracked-typed local write set the ff/dd boundary patch demotes on region exit.
Closes the write-set gap flagged above ("Where journal data was insufficient" /
"`region_local_vars` = reads, not writes"): the write set is not recoverable from
the journal, so it is recovered from source. Branch `langgraph-agents`. Full
suite **183 → 207** (all green).

## Module placement decision

**`agents/shared/`, not `agents/patcher/`.** The call sites are
`ff_integrator.integrate_region` / `dd_integrator.integrate_region` (both build on
`agents/integrator_base/`), and the utility is plausibly useful to the
characterizer too. It is a shared source-scanning service, so it lives beside the
other shared services (`stability_reducer.py`, `fast_merge.py`, …). Tests live in
`tests/shared/test_region_scan.py` to match.

## Backends — libclang available here (unlike the P3a cluster image)

Mirrors P3a exactly: **libclang preferred (lazy import), keyword-token lexer
fallback.** The lexer reuses the P3a lexical state machine
(`agents/patcher/edits.py`) — skips comments / string / char literals, whole-token
matching, same "constrained subset of numerical kernels" scope.

- **libclang IS importable and functional in `.venv`** (`clang.cindex`, 
  `Index.create` + `parse` verified). So — unlike P3a, which shipped only the
  fallback because bindings were absent on the cluster — the **libclang backend is
  the exercised primary path here.** The system `/usr/bin/python` still lacks the
  bindings; run tests under `.venv`.
- Tests parameterize **every** behavioural case over both backends (`backend`
  fixture): `libclang` (skipped if bindings absent, so a bindings-less CI gives no
  false confidence) and `fallback` (forced via monkeypatching `_import_clang` to
  raise `ImportError`). 12 cases × 2 backends + backend-specific tests = 24.

## Corner case discovered — libclang present but type unresolved

Parsing a **single file without its include context** (the tracked type's header
missing) makes clang mis-recover `Tracked<double> a = …` as `VAR_DECL a
type='int'` and report **zero** writes — silently wrong for a real region. Guard:
an *empty* libclang result over a region whose text still contains
`tracked_type<` is treated as a resolution failure and routed to the include-free
lexer (`_region_has_tracked_text`). Covered by
`test_libclang_empty_over_unresolved_type_falls_back`. Practical consequence: on
real HPC files parsed without `-I` context, the **lexer is the effective
workhorse**; libclang's precision kicks in for self-contained files or when a
future caller supplies include dirs (the signature has no include-dirs param — a
possible extension when wiring lands).

## Semantics choices (documented in the module)

- **Write SET, source order.** Each written name appears once, ordered by first
  write within the region. `Tracked<double> a = …; a = …;` → `["a"]` (chosen over
  list semantics `["a","a"]`; the docstring calls it a write *set*).
- **Writes counted:** (a) `tracked_type<T>` VAR_DECLs, and (b) re-assignments
  (`name = …`, compound ops) of a tracked local — including a local declared
  *above* the region but written inside it (both backends build the tracked-name
  universe file-wide, then filter writes by line). A tracked-typed name
  immediately followed by `(` is treated as a function declaration, not a local
  write (excluded).
- **libclang assignment form:** a class-based tracked type's re-assignment shows
  up as `CALL_EXPR 'operator='` (overloaded), not `BINARY_OPERATOR`; that is what
  the AST path matches. If a future `tracked_type` is a builtin/typedef alias
  (plain `BINARY_OPERATOR` assignment), the AST path won't match it — but the
  lexer fallback catches it textually. Flagged rather than handled (no such type
  exists today).

## Not wired into the integrators (per task)

Deliberately **not** called from `ff_integrator` / `dd_integrator` — they are
scope-decision-(b) stubs; wiring lands with the regional codegen task. This pass
delivers the utility + tests only. No C++ / schema / journal / characterizer /
pipeline changes.

## Out-of-scope forms (flag only if a real region hits them)

Anonymous intermediates, by-reference-out params, and function-return-consumed
unnamed values are not treated as named writes — none appeared in the fixtures.
The fallback does not attempt full C++ (same subset bound as P3a's rewriter);
`>>` closing nested templates is handled but exotic template syntax is not.

---

# Regional ff/dd codegen — implementation handoff (2026-07-18)

Replaces the scope-(b) bounded stubs in `ff_integrator` / `dd_integrator` with
real LLM-driven regional generation. The agentic loop's forward slice now runs on
real qcdloop code: a real LLM shim + a deterministic boundary patch → real vanilla
build + smoke + commit, through the real Patcher with **no injection**.

## What landed

- **`agents/integrator_base/boundary.py`** (new, deterministic) — synthesizes the
  region boundary patch as a `git apply -p1` diff: promote reads to the extended
  scalar on entry (Rule R1), keep region locals extended (Rule R2), demote writes
  back on exit, and `#include` the shim after `#pragma once`. Comment/string/
  char-literal-aware, whole-word (reuses the P3a/region_scan lexical state machine).
- **`agents/integrator_base/regional.py`** (new) — the shared `run_integrate_region`
  engine the ff/dd twins wrap. Reads the region at the pinned SHA (`git show`),
  recovers writes (Fix C), SOURCE_HASH-caches, LLM-generates the shim
  (`attempt`-varied, cache bypassed on retry), writes it to `out_dir` **and** the
  candidate tree, and pairs it with the boundary patch. ff/dd differ only by a
  `RegionalSpec` (ruleset + C++ scalar/complex spelling + DD constant note).
- **`agents/integrator_base/cache.py`** — `compute_region_hash(region_src, ruleset,
  scalar_type, writes)`, the regional SOURCE_HASH analogue of `compute_source_hash`.
- **`agents/integrator_base/llm.py`** — `stream_shim` returns `(text, tokens)` so
  the result carries `llm_tokens`.
- **`agents/ff_integrator/{agent.py,system_prompt.txt}`** and
  **`agents/dd_integrator/{agent.py,system_prompt.txt}`** — `integrate_region` is
  now a thin wrapper over the engine (`quad::ffun::ffloat` / `quad::ddfun::ddouble`).
  `ff_integrator.integrate` (whole-app) and `dd_integrator.integrate` (whole-app
  qcdloop DD stub, Validator's ground truth) are **untouched**.
- **Vendored `third_party/include/{ff_math.hpp,ff_complex.hpp}`** from
  `kokkos-extended-precision-demo@fffunKokkos` + `third_party/include` added to the
  app CMake include path so shims resolve `<ff_math.hpp>`.
- **`agents/patcher/dispatch.py`** — `_gen_regional` passes `caller_type` (float for
  `float-to-ff`, else double).

## Prompt-authoring decisions (R1–R4)

The tracked Rules 1–9 + C1–C7 are the ancestor set; regional promotion adds:

- **R1 — boundary conversion is NOT the LLM's.** The promote/demote casts are
  synthesized deterministically; the shim must not emit them, nor reference the
  `__ff`/`__ext` renamed identifiers (it never sees them). This is the load-bearing
  split (design §P4): the LLM produces only the type/operator/constant surface, so a
  retry re-rolls the shim while the patch machinery stays fixed.
- **R2 — internals stay extended** (no mid-region round-trip through double/float).
- **R3 — named constants keep name + precision.** ff routes through vendored
  `ff_*()`; **DD must hex-encode** every constant as `make_dd(0x<hi>ULL,0x<lo>ULL)`
  — a decimal literal truncates the low word (ref `gen_dd_constants.cpp`). Codified
  in the DD prompt + reinforced in the DD `RegionalSpec.constant_note`.
- **R4 — `#error` escape hatch** (mirrors Rule 9), including "don't guess a
  constant's hex bits — surface it."

## Deviations from tracked_integrator's patterns (and why)

1. **Shared engine (`regional.py`), not per-integrator duplication.** The ff/dd
   regional paths are near-identical; the whole-app tracked path is not. Putting the
   flow in `integrator_base/regional.py` keeps the two `agent.py` files to a spec +
   a one-line delegation (design §P7 "code reuse is the point").
2. **Two-limb demotion, not `static_cast<caller>(x_extended)`.** Neither `ffloat`
   nor `ddouble` defines `operator double`; the vendored types' own conversion-out
   idiom is `(double)x.hi + (double)x.lo`. The boundary emits
   `static_cast<T>(x.hi) + static_cast<T>(x.lo)`.
3. **Dataflow-based local promotion, not a fixed `caller_type` token.** Real qcdloop
   locals are declared through template aliases (`TMass`/`TScale`, `double` at the
   vanilla instantiation), so matching the literal `caller_type="double"` finds
   nothing. The boundary promotes a local iff its initializer consumes an
   already-promoted value (chaining to a fixpoint) and demotes to the local's **own**
   declared type — which is what made the real e2e work through the Patcher. This is
   also why **Fix C returns `[]` on vanilla regions** (it keys on `tracked_type<…>`
   template syntax; bare `double`/`TMass` decls don't match): it is still wired in
   (`tracked_type=caller_type`) and feeds the pre-declared (Case B) write path, but
   the region-local write set comes from the boundary's own decl scan. Not a bug —
   the two sources are complementary; extending Fix C to bare/alias types is a
   follow-up below.
4. **Cache only on `attempt==0`.** A Patcher retry (`attempt>0`) must re-roll, so it
   bypasses the SOURCE_HASH cache (the shim filename/key is attempt-independent).

## LLM / token cost (from the real e2e)

`test_e2e_regional_ff_real_llm` (region `kokkosUtils.h:312`, `TMass arg = x1*x2`):
one integrator attempt, **~3.37–3.39k tokens** (input+output) per shim generation,
~2.5 min wall including the Kokkos vanilla build. The generated shim was minimal and
correct on the first attempt (no retry): `#pragma once` + vendored includes + a
Rule-2/C3 comment noting the vendored `ffloat*ffloat` operator suffices. Real-LLM
integration tests are marked `@pytest.mark.llm` (+`kokkos` for the e2e) and skip
without `ANTHROPIC_AUTH_TOKEN`.

## Test coverage

- `tests/integrator_base/test_boundary.py` (8) — promote/rename/demote, empty set,
  multi read/write chaining, Case-B seeding, whole-word/comment/string safety,
  body-local vs signature, template-alias demote-to-original-type, int-not-promoted.
- `tests/integrator_base/test_region_cache.py` (6) — `compute_region_hash` variation.
- `tests/ff_integrator/test_regional.py` (6+1 llm) / `tests/dd_integrator/test_regional.py`
  (5+1 llm) — shim+boundary generation, cache hit skips LLM, retry bypass + message
  variation, DD hex-constant note, bad-range failure, whole-app NotImplemented.
- `tests/patcher/test_e2e.py::test_e2e_regional_ff_real_llm` (llm+kokkos) — the
  no-injection forward slice, **verified passing here**. The offline injected e2e
  is preserved. Offline suite: **231 passing** (was 207).

## Punts / follow-ups

- **Whole-app ff/dd** (`integrate`) — not built; the regional path is what the
  Patcher drives. `dd_integrator.integrate` remains the qcdloop DD ground-truth stub.
- **Fix C on bare/alias types** — `extract_region_writes` matches only
  `tracked_type<…>`; extending it to `double`/`TMass` (or plumbing `-I` context so
  libclang resolves the vanilla types) would let it, rather than the boundary's decl
  scan, own the region-local write set. Not needed today.
- **Boundary kernel-subset bounds** — one name per decl statement (`double a,b;` and
  split `double r; r=…;` unhandled); single contiguous region. Multi-line atomic
  chain intents remain the Patcher HANDOFF punt (single-line intents only).
- **`caller_type` for `float-to-ff`** demotes Case-B writes to `float`; region-local
  writes always demote to their own declared type regardless.

---

# First real end-to-end run (2026-07-18)

First wiring of the full loop on real Stage-4 output: **characterize → Strategy →
Patcher (real LLM) → Validator (real 3-build precision test)**, driven by
`runs/qcdloop/run_strategy_e2e.py` (skips the characterize node, calls
`strategy.agent.run` directly per the Q5 seam).

- **run_id:** `20260718_194556_67dbcf37`  ·  branch `strategy/20260718_194556_67dbcf37`
- **report:** `runs/qcdloop/strategy/20260718_194556_67dbcf37/report.md` (+ `report.json`, `final.diff`, `iterations.jsonl`)
- **terminal status:** `budget_exhausted` — 8 accepted `double-to-dd` promotions
  (budget cap), 19 iteration-log entries, 1274 s, 84,959 LLM tokens, tolerance 8.0,
  snapshot seed 12345 / n=1000.
- **config note (budget deviation):** default budget (500 iters / 6 h) is
  intractable at real-Validator cost given the report's ~53k cascade chains, so the
  smoke ran `max_iters=8` / 4 h wall. Validator snapshot n=1000 (fast smoke). See
  "smells" below re: what `max_iters` actually counts.

## Terminal verdict per region

- **8 promoted to `dd` (accepted, ≥8 digits vs DD oracle):** `B2m.h:64`, `B0m.h:88`,
  `kokkosUtils.h:608`, `B1m.h:79`, `B2m.h:492`, `kokkosUtils.h:248`, `kokkosUtils.h:225`,
  `B4m.h:189`.
- **11 `dd_untested` (Patcher failure at the double-to-dd rung, P6a — NOT a physics
  ceiling):** `B2m.h:65`, `B1m.h:62/63`, `B4m.h:163/233`, `B0m.h:68/69`, `B2m.h:84`,
  `B3m.h:177`, `kokkosUtils.h:745/550`. All are the intermittent shim-hallucination
  failures below.
- **Remainder of the 66-long correctness-region queue + the cascade-chain tier:** not
  reached (budget bound first).

## What broke and what was fixed

1. **`fast_merge` dropped `cascade_chains`** (real bug, fixed `9cde89b`). The parallel
   shard merge never read `cascade_chains` from shards and omitted them from output, so
   any report built via the fast path (i.e. every 100k run) had an **empty correctness
   tier-2**. `region_local_vars` flowed correctly but was untested. Mirrored
   `finalize_report` (union chain_id-keyed dict → sorted list) + added fast/slow parity
   tests (`tests/agents/test_fast_merge.py`).
2. **Regional integrator read the wrong git path** (real bug, fixed `d6db3b0`).
   `_gen_regional` passed the characterization region key verbatim — a **bare basename**
   (`B2m.h`) — to the integrator, whose `_git_show` does `git show <sha>:<file>` and
   whose boundary patch labels `a/<file>`. The file is `box/B2m.h`, so **every** regional
   promotion died at `could not read B2m.h@<sha>` (→ `llm_gen_failed`). Fixed to pass the
   repo-relative resolved path (`deps.target_path` rel `repo_root`).
3. **Flat-tree assumption** (harness design). The regional integrator drops generated
   shims at the repo root and assumes `repo_root == QL_HEADERS`; the runner therefore
   gives the Patcher a **dedicated headers-rooted git repo** (a copy of
   `qcdloop_headers_full` with `boxGPU.h` at root) while the Validator baselines against
   the pristine main `qcdloop_headers_full`. Without this, `_install_in_tree` would drop
   shims off the include path.
4. **Frozen report was stale/oversized.** `runs/qcdloop/report_100k.json` is **13.7 GB**,
   lacks `region_local_vars` + `cascade_chains` (predates the reducer upgrade), and the
   filename in the brief (`results_100k.json`) doesn't exist. Per approval, re-ran the
   chunked characterization → `runs/qcdloop/report_smoke.json` (n=1000, both fields
   present, 53,603 cascade chains). The 13.7 GB frozen report was **not** touched.

## Remaining smells / follow-ups

- **Intermittent hallucinated app-header includes in DD shims.** The dominant failure
  (11/19 iters): the LLM shim adds e.g. `#include "ql/constants.h"` / `"ql/maths.h"`
  (nonexistent on the include path) → build gate fails → after retries folds to
  `llm_gen_failed` (mislabeled log_tag `llm_capacity`). Region-dependent and
  non-deterministic (`B2m.h:64` succeeds, adjacent `B2m.h:65` fails both attempts).
  Needs prompt hardening ("emit no app-header includes; only vendored `dd_math.hpp`/
  `dd_complex.hpp`; materialize constants inline via `make_dd`") and/or a misgen
  classifier. This is the #1 thing to fix to raise the accept rate.
- **Cascade chains not deduplicated across samples.** chain_ids embed the sample hash, so
  n=1000 → 53,603 chains; the tier-2 queue scales with samples×victims and
  `regions_at_threshold` in the report reads 54,047 (inflated by chain `region_status`
  entries). Needs span-level dedup before queueing or the cascade tier is unrunnable at
  scale.
- **`max_iters` counts only accepts + genuine rejects** (`llm_gen_failed` has
  `counts_budget=False`), so `max_iters=8` did ~19 log iterations. Add a separate hard
  cap on total iterations (incl. gen-failures) for bounded smoke runs.
- **`precision_distribution` dd=7 vs 8 `precision_assignment` entries** — off-by-one
  accounting mismatch between `region_final` and the accepted-assignment list; reconcile.
- **13.7 GB report / loader OOM.** `load_regions`/`load_chains` each `json.loads` the
  whole file, twice — fine at n=1000, would OOM on the 100k report. A streaming loader is
  needed before Strategy can consume a full-scale report.
