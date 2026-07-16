# Whole-app per-line attribution via per-statement `line=` scopes, then 100k/integral run

> Handoff plan. Branch `langgraph-agents`. Written 2026-07-16. The committed
> `agents/shared/stability_reducer.py` (policy-neutral characterizer reducer) is
> the map/merge step the 100k run depends on; this plan lands the per-line
> attribution that must precede that run.

## STATUS: IMPLEMENTED (2026-07-16)

All four steps landed on `langgraph-agents`. Design forks resolved: **libclang
AST** for statement detection (added as a `.venv` wheel); DAG "final-accumulation"
fallback **deferred** (the `""` bucket shrank to near-nothing once injection
landed).

- **Step 1** `655c83d` — reducer `line=` verified + basename/no-slash regression test.
- **Step 2** — proved on the existing journal (`BIN2` region `B2m.h:84`).
- **Step 3a** `e7e959f` — `tracked::push_scope`/`pop_scope` (+ Catch2 test).
- **Step 3b–d** `94f447a` — `agents/tracked_integrator/line_injector.py` (1089
  sites) + CLI + cache + unit tests. **256-sample bit-exactness gate PASSED**
  (coeff0 / per-integral max_cond / op counts identical C8-only vs C8+`line=`;
  35–102 real per-line regions/integral vs one `""` bucket). See
  `runs/qcdloop/VALIDATION.md`.
- **Step 3e** `ce135f2` — optional `build_run` integration (`inject_line_scopes`).
- **Step 4 infra** `d663e0a` (driver `--sample-offset`, bit-exact chunking) +
  `d1475f1` (bounded-parallel `run_chunked.py --workers W`) + `e002560`
  (`--resume`). The 100k run executes via `runs/qcdloop/run_chunked.py`.

**Deltas from the original sketch:** chunks are small (default 500, not 25k — an
instrumented 25k journal is ~245 GB) and reduced **in-process to JSON shards**
(not Parquet); parallelism is memory-bandwidth-bound, so the practical sweet spot
is `--workers ≈ 16–24`, not the full core count.

## Context

The consolidated qcdloop whole-app tracked driver is validated at 256 samples
(Stage-2 parity; `runs/qcdloop/VALIDATION.md`). The next milestone is the real
characterization run at **100k samples per integral** (21 integrals), executed as
chunked batches whose transient journals are reduced in-process by
`agents/shared/stability_reducer.py` (map) and merged (merge). Before that
expensive run we want **per-code-line attribution** so the Strategy Agent knows
WHERE in source error grows — not just which variable.

**The gap (confirmed against code):** the whole-app journal has no source
location. 100% of records have `"at": ""` and ids like
`abs@?#1@integral=B1/sample=0`. Root cause: location is a trailing defaulted
param on named `tracked::` ops, but the box arithmetic is dominated by C++
**operators** (`a-b`, `x*y`) that route through `Tracked::operator*` etc., which
**cannot carry a location** (binary operators can't take extra args; complex
decomposes into scalar operator ops too). So no `TRACKED_HERE`/parameter/
auto-capture scheme can attribute the operator ops.

**The mechanism that works — already proven.** Scopes attach to *every* op via
`current_scope_suffix()` (`third_party/tracked/include/tracked/tracked.hpp:69`),
operators included. The reducer already treats a `line=` sub-scope as the primary
per-line region key. The uncommitted 64-sample `runs/qcdloop/journal.jsonl`
carries `line=B2m.h:84` on ~81k ops — a hand-placed prototype scope on B2m.h's
`res(i,0)=(xspence... + kLog...)` accumulation write, since reset (no tree
trace). This retires the only real unknown: a `line=` scope DOES tag operator ops
end-to-end and the reducer buckets them. Format = **basename** (`B2m.h:84`).

**Target: every operator-bearing (value-producing) statement across the full
arithmetic call closure** — not just output-writes, and not just the top-level
box headers. Wrapping each statement shows *where along the computation* error
grows; instrumenting callees means `xspence`/`spence`/`xetatilde` get their OWN
`line=` regions rather than lumping into the calling box line.

The vendored tree is small and fixed: 11 headers, **no basename collisions**. The
arithmetic-bearing closure is `box/B0m–B4m.h`, `box/box_common.h`,
`kokkosMaths.h` (holds `xspence`/`xetatilde`/`spence`/`ddilog`), `kokkosUtils.h`.
Detection is **structural** (statements bearing `+ - * /` on tracked types), not
by header name — respecting the no-hardcoded-patterns rule. Basename `line=`
labels (`kokkosMaths.h:412`) are unambiguous here.

The pipeline code is already app-neutral (de-qcdloop); only `runs/qcdloop/`
artifacts are app-specific, which is correct.

## Goal

1. Make `line=` line-regions first-class in the reducer (largely DONE — verify +
   optional DAG fallback).
2. Prove end-to-end on one integral (B13 or the existing B2 journal): the reducer
   emits the accumulation line with its error growth and cascade class.
3. Generalize: teach the `tracked_integrator` to inject `line=` scopes around
   every operator-bearing statement across the full arithmetic header closure
   (`box/*.h` + `kokkosMaths.h` + `kokkosUtils.h`), deterministically, as a
   C8-style build-time patch (revives a `push_scope`/`pop_scope` primitive for
   declaration-safe wrap).
4. Then run the 100k-per-integral chunked characterization → reduce → merge.

## Design

### Generator pipeline facts (confirmed)
- The interop shim is generated by `agents/tracked_integrator/agent.py` entirely
  from `_SYSTEM_PROMPT` (no C++ codegen). C8 patches are derived deterministically
  by `derive_c8_patch` (agent.py:673) and emitted as `git apply`-able unified
  diffs via `_synthesize_patch`/`difflib`. `build_and_run`
  (`agents/build_run/agent.py` ~138-199) checks the header tree out pristine,
  generates the shim, builds, applies the C8 patch on int↔tracked failures,
  rebuilds; the tree is reset after. This is the delivery vehicle to reuse.
- Build chain: `module use /soft/modulefiles && module load gcc/13.3.0 cmake/3.28.3`.
- `agents/shared/regen_profile.py` only re-parses an existing journal → profile;
  rebuild+rerun first, then refresh.

### Step 1 — Reducer line-regions (mostly done)
`stability_reducer.py` already parses/strips `line=` (`_region_key`,
`_sample_key`) with a passing test (`test_line_scope_regions_and_cross_boundary_amp`).
Optional: a **DAG-structural "final-accumulation" fallback** — when a sample has
no `line=` tag, key the region by the DAG output-sink (highest-amp terminal node)
instead of dumping into `""`. Decide add-now vs defer.

HARD CONSTRAINT: `_parse_scope` splits the scope on `/`, so a `line=` value must
contain no `/` (`box/B2m.h:84` → truncates to `line=box`). Emit **basename**.
Verified safe: the 11 vendored headers have no basename collisions. Add a
regression test pinning this. (Only apps with duplicate basenames would later
need a non-`/` intra-suffix separator in `current_scope_suffix` + reducer.)

### Step 2 — End-to-end proof on one integral
Cheapest: run the reducer over the existing `runs/qcdloop/journal.jsonl` (already
has `line=B2m.h:84`) and show B2's region for that line with `max_rel_err`,
`max_amp`, and signal class. If a clean B13 demo is preferred, hand-inject one
`line=` scope at B13's res-write, rebuild (module chain), run low-sample, show the
same. No new engine needed for the proof.

### Step 3 — Generalize: per-statement `line=` injection (the real new work)
- **Primitive (`third_party/tracked`, ours):** add
  `tracked::push_scope(std::string)` / `tracked::pop_scope()` free functions (thin
  wrappers over `detail::scope_stack`; the RAII `scope` class already does this on
  ctor/dtor). Needed because general statements include **declarations**, which
  brace-wrapping would scope out. Trivial; no numeric-path change.
- **Injection = hybrid keyed on statement kind**, over each value-producing
  statement in the instrumented headers:
  - terminators (`return`/`break`/`continue`/`throw`) → RAII
    `{ tracked::scope _l("line=<basename>:<N>"); <stmt> }` (auto-pops on return);
  - all others (assignment/declaration/expression) →
    `tracked::push_scope("line=<basename>:<N>"); <stmt> tracked::pop_scope();`.
  Stack semantics restore the caller line on return from callees (correctness).
- **Instrumentation scope = the full arithmetic closure**: `box/*.h` (incl.
  `box_common.h`) + `kokkosMaths.h` + `kokkosUtils.h`, discovered by walking the
  include closure of `boxGPU.h` and filtering to headers carrying operator ops on
  tracked types (structural, not name-based). `timer.h`/`boxGPU.h` = no
  arithmetic, skipped.
- **Statement detection is the crux.** libclang is NOT installed (no
  `clang.cindex`, no system clang). Preferred: add the self-contained `libclang`
  pip wheel to `.venv` and walk the AST (exact statement bounds + kind). Fallback
  (no dep): a conservative brace-depth/`;`-boundary pass with a keyword check for
  statement kind — feasible to hand-validate because the header set is small/fixed.
- **Delivery = C8-style build-time patch.** New injector pass in
  `agents/tracked_integrator/agent.py` (sibling to `derive_c8_patch`) produces a
  `git apply`-able unified diff via `_synthesize_patch`/`difflib`, stored as
  `runs/qcdloop/src/ql_tracked_lines.patch`. `agents/build_run/agent.py` applies
  it in the pristine→build→run→reset flow (app source never permanently edited),
  ordered with the C8 patch.
- **Caching:** mirror the shim `SOURCE_HASH` — hash header tree + transform
  version into the patch; regenerate only on change.
- **Compose with C8:** both patch box/*.h; C8 touches B3m.h/B4m.h. Generate the
  line patch against the C8-patched tree and assert hunks don't overlap; apply in
  a fixed order.

### Step 4 — 100k/integral run (after 1–3)
Chunked: ~4 chunks × ~25k samples per integral (Serial, agent-level parallelism);
each chunk reduces its transient journal in-process via `reduce_journal` (never
materialize the full ~100s-of-GB journal) → shard report; `merge_reports` →
`finalize_report` → one consolidated per-integral report for the Strategy Agent.

## Verification

**Steps 1–2 (now, cheap):**
- `.venv/bin/pytest tests/agents/ -q` stays green; add the basename/`/`-constraint
  regression test.
- `python -m agents.shared.stability_reducer report runs/qcdloop/journal.jsonl -o
  /tmp/rep.json`; confirm B2's `regions` contains `B2m.h:84` with populated
  `max_rel_err`/`max_amp`/`signal_class`, forward-cone amp non-trivial.

**Step 3 gates (256 samples, vs `runs/qcdloop/VALIDATION.md`):**
1. **Numeric bit-exactness (critical):** scopes must not perturb values — compare
   per-integral max_cond to baseline (B1 ~2.1e3, B2 ~8.6e4, B15 ~2.4e7; gate-a
   atan2 sat only B14/B15/B16; cond>1e15 only BIN0–4). Any drift → fail.
2. **`line=` populated:** `grep -o 'line=[^"/]*' journal.jsonl | sort -u` shows
   `<basename>:<N>` across box + kokkosMaths + kokkosUtils, high op coverage.
3. **Reducer yields real per-line regions** (multiple keys per integral, not one
   `""` bucket), amp still populated.

**Step 4 gate:** merged 100k report reproduces the 256-sample class distribution
and surfaces additional gate-(b)/rare high-cond events.

## Risks
1. **Per-statement detection + kind classification** across templated, multi-line
   vendored C++ — the crux. Mitigate with libclang AST (add pip wheel); the
   brace-depth heuristic is viable only because the header set is small/fixed.
   Broader instrumentation = more injection sites = larger validation surface.
2. **Numeric perturbation** from injection — must be provably nil; scopes are
   value-neutral by construction, but gate #1 proves it.
3. **C8 × line-patch hunk conflict** on B3m.h/B4m.h — generate against C8-patched
   tree, assert non-overlap.
4. **Callee attribution lumping** — resolved by instrumenting the full closure
   (`kokkosMaths.h`/`kokkosUtils.h` too), so helper ops get their own regions.

## Effort
Steps 1–2: ~half-day (reducer verify + proof; mostly done). Step 3: ~1.5–2 days
(push/pop primitive + per-statement injector with kind classification + build
integration + validation; libclang setup swings it). Step 4: ~30–60 min wall.

## Key files
- `agents/shared/stability_reducer.py` — reducer (line= already handled)
- `third_party/tracked/include/tracked/tracked.hpp` — scope stack + `scope` RAII (add push/pop)
- `agents/tracked_integrator/agent.py` — `derive_c8_patch`/`_synthesize_patch` (add line-injector pass)
- `agents/build_run/agent.py` — pristine→build→reset flow (apply line patch)
- `runs/qcdloop_headers_full/{box/*.h,kokkosMaths.h,kokkosUtils.h}` — instrumentation targets (vendored; patch-only)
- `runs/qcdloop/src/{boxGPU_tracked.cpp,ql_tracked_interop.hpp,ql_tracked.patch}` — driver/shim/C8 patch
- `runs/qcdloop/VALIDATION.md` — 256-sample bit-exact baseline
