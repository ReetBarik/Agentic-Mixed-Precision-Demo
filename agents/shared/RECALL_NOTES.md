# Recall verifier — notes & findings

Companion to `agents/shared/recall_verifier.py`. Implements the
"End-to-end signal usefulness" layer of **PLAN_implementation.md §6**:
for each validation fixture, recall = fraction of `symbolic_hints[*].location`
covered by `sensitivity_profile.top_hotspots[*].location`, grouped by severity.

Pass criteria (§6): **≥80% recall on `high`, ≥50% on `medium`, precision
unbounded** (false positives acceptable). Surfaced as a non-blocking `pass`
status; the script never exits non-zero on a threshold miss (only on
structural errors — missing files, malformed JSON).

## How the artifacts get `location` strings

`location` originates from the `TRACKED_HERE` macro
(`::tracked::SourceLocation{__FILE__, __func__, __LINE__}`) attached to a
tracked op. Two attribution patterns exist across the fixtures:

| Pattern | `TRACKED_HERE` lives in | `__func__` is | Example profile location |
|---|---|---|---|
| **kernel-direct** (`cancellation`, `kahan`, `naive_variance`) | the kernel `.hpp` (named `tracked::add/sub/...` calls) | the kernel function | `…/cancellation.cpp:cancellation_check:10` |
| **driver-shim** (`log_sum_exp`, `lnrat`, `cln`) | the micro-driver's interop shims (kernel uses operator overloads / `std::`/`ql::` wrappers that can't carry `TRACKED_HERE`) | the *shim* function (`exp`, `log`, `kLog`, `kAbs`) | `src/micro_driver.cpp:exp:28` |

Path relativization (commit `918738e`) makes the file part relative to the run
dir **when the file is inside the run dir**. Driver-shim locations relativize
(`src/micro_driver.cpp:…`); kernel-direct locations point at
`tests/agents/fixtures/kernels/*.{hpp,cpp}`, which is *outside* the run dir, so
they stay absolute (the parser's documented fallback). This is expected and
fine for matching — the verifier matches on the function-name token, not the
path.

## The matcher (v1)

Hint and hotspot locations use different formats:

```
hint    cancellation_check:2-4        (func:line-range)
hint    file:log_sum_exp_naive:11-13  (literal-"file":func:line-range)
hint    Lnrat:complex_overload        (kernel:logical-overload, no line)
hotspot src/micro_driver.cpp:exp:28                       (relpath:func:line)
hotspot /abs/.../cancellation.cpp:cancellation_check:10
```

A hint matches a hotspot location if **either**:

- **(a) substring** — one string contains the other (after stripping a leading
  literal `file:` from the hint), or
- **(b) shared identifier token** — any C-identifier token of length ≥ 3 from
  the hint location (excluding the literal `file`) appears as a substring of the
  hotspot location.

Rule (b) is what lets kernel-direct hints match (shared function-name token,
e.g. `cancellation_check`). Precision is unbounded by spec, so this leniency is
acceptable. The matcher is intentionally simple and lives in `match_hint()`.

## Finding: driver-shim attribution does not align with kernel-level hints

This is the real, expected mismatch (anticipated in the task brief):

- **`log_sum_exp`** — hint location `file:log_sum_exp_naive:11-13` names the
  *kernel* function. The hotspots are attributed to the driver shims `exp` /
  `log` (`src/micro_driver.cpp:exp:28`, `:log:25`). No shared token →
  **no match**, despite the characterizer correctly flagging the right ops.
- **`lnrat`** — hints name `Lnrat:complex_overload` / `Lnrat:real_overload`
  (kernel + a *logical* overload label that exists in neither artifact as a
  symbol). Hotspots are attributed to shims `kLog` / `kAbs`. No shared token →
  **no match**.
- **`cln`** — no annotations (`symbolic_hints.json == []`), so no recall to
  compute regardless.

So the verifier passes on the kernel-direct fixtures (`cancellation`,
`naive_variance`) and *correctly surfaces* the shim/kernel attribution gap on
the driver-shim fixtures as `findings` rather than silently scoring them 0 with
no explanation.

### Two ways to close the gap (for human review — not decided here)

**Option A — tighten the `symbolic_hints` schema to require `file:func:line`.**
Force the LLM hint emitter (`symbolic_overlay`) to produce attributable
locations in the same `path:func:line` space the profile uses.

- *Pros:* one location grammar everywhere; matcher collapses to exact/substring;
  hints become directly clickable; removes logical-name ambiguity
  (`complex_overload`).
- *Cons:* the LLM only sees kernel source, so it can only name the *kernel*
  function + line — it cannot know that ops get attributed to *driver-shim*
  functions (`exp`, `kLog`). So this alone does **not** fix `log_sum_exp` /
  `lnrat`: the shim/kernel boundary remains. It would need to be paired with
  either (i) carrying `TRACKED_HERE` at the kernel call site rather than inside
  the shim, or (ii) Option B.

**Option B — add a logical-name → `file:func:line` mapping the verifier
consults.** Maintain a small per-fixture (or per-app) alias table mapping
hint-side names (`Lnrat:complex_overload`, `log_sum_exp_naive`) to the
shim/op locations the profile actually emits.

- *Pros:* no schema change; tolerates the shim/kernel boundary that is
  *inherent* to the operator-overload + interop-shim design; keeps hints
  human-readable (`Lnrat:complex_overload`).
- *Cons:* another artifact to maintain and keep in sync; pushes app-specific
  knowledge into the verifier; risks masking genuine misses behind a
  too-generous alias.

**Tradeoff in one line:** A makes the *data* uniform but can't see across the
shim boundary on its own; B leaves the data as-is and absorbs the boundary in
the *matcher* at the cost of a maintained mapping. A third hybrid — emit
`TRACKED_HERE` at the kernel call site (so `__func__` is the kernel function,
not the shim) — would make driver-shim locations carry the kernel function name
and let plain substring matching work without an alias table; it changes the
`driver_gen` prompt/codegen and is out of scope for this task.

## Reproducing

```
# 1. rebuild + rerun each micro-driver so journals carry TRACKED_HERE locations
#    (needs cmake + Kokkos for lnrat/cln) — see scripts/regen_recall.sh
# 2. re-parse journals -> sensitivity_profile.json (relativizes paths)
python -m agents.shared.regen_profile runs/*
# 3. recall verifier -> stdout + runs/recall_summary.json
python -m agents.shared.recall_verifier runs
```
