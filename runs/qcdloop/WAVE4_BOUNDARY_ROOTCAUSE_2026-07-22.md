# WAVE4 Boundary Root-Cause — the dedup regression is a lost per-region COMMIT artifact, not a lost app-source conversion

- **Date:** 2026-07-22
- **Branch:** `langgraph-agents` @ `917bf99`
- **Scope:** root-cause only (no code changes, no probe run, no 10k re-run — per Reet's call)
- **Supersedes the diagnosis in:** `WAVE3_10K_2026-07-21.md` §banner + §A (row "non-shim boundary-conversion lines")

---

## TL;DR

The WAVE3_10K report's headline diagnosis — *"the deterministic boundary patch that owns
every region-local precision rewrite never lands; PIPELINE_v1 had ~150 boundary-conversion
lines, Wave-3 has 0"* — **is not what the evidence shows.**

1. **Both** the good (PIPELINE_v1) and bad (Wave-3) runs have **0 app-source
   boundary-conversion lines.** The good run's app headers were *also* include-only.
2. The report's "159 hunks / ~150 conversion lines" for PIPELINE_v1 are **152 new
   per-region shim *files*** (`B2m_dd_L65_65_<hash>.h`), miscounted as conversions.
3. `regional.py` @ e11b788 **still calls `_boundary()`** in every path. Nothing was
   severed. The boundary patch is emitted exactly as before — and for qcdloop's
   template-typed regions it was *always* an `#include`-only patch.
4. The **real** regression: dedup replaced the unique per-region shim file (a distinct
   committable artifact per region) with a single merged canonical shim. A region whose
   `Constants<T>` members a sibling already merged — and whose header already `#include`s
   the canonical shim — now produces an **identical tree → `NothingToCommitError` →
   `empty_candidate`** (the 222). The per-region *commit record* was lost, not a rewrite.

Consequence: **Fix-1 as written is a no-op** for these regions, and the **probe's new
criterion is unsatisfiable** (there is no bare `double` at the computation lines).

---

## Evidence

### E1 — App source is include-only in BOTH runs

Per-file count of added lines in each run's cumulative `final.diff`, split into
`#include` additions vs. everything else (`awk` over the file sections):

| app header | GOOD `20260720_054121` includes / conversions | BAD `20260721_205813` includes / conversions |
| --- | --- | --- |
| box/B0m.h | 30 / **0** | 3 / **0** |
| box/B1m.h | 22 / **0** | 3 / **0** |
| box/B2m.h | 41 / **0** | 3 / **0** |
| box/B3m.h | 10 / **0** | 3 / **0** |
| box/B4m.h | 16 / **0** | 2 / **0** |
| boxGPU.h | 7 / **0** | 3 / **0** |
| kokkosUtils.h | 26 / **0** | 3 / **0** |
| per-region shim files created | **152** | 0 |
| canonical shims created | 0 | **3** |

There is **no `double`→`ddouble`/`ffloat`/`float` rewrite at any computation line in
either run.** The "boundary conversion" the report counts does not exist in the app
source; it is the set of new per-region shim files.

### E2 — Why there is no app-source conversion (by design of this codebase)

qcdloop box regions are template code, e.g. `box/B2m.h` around the probe's regions:

```cpp
const TOutput k12c = TOutput(k12 - ql::Max(ql::kAbs(k12), TMass(ql::Constants<TMass>::_one()))
                     * ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>());
```

The working types are the template parameters `TOutput / TMass / TScale`, not bare
`double`. The real intents (from `iterations.jsonl`) carry `"variables": []`. So
`boundary.synthesize_boundary_patch` finds nothing to promote (no reads, no writes, no
bare-`double` decls) and returns an **`#include`-only** patch. This is unchanged from
PIPELINE_v1 — it is how the module has always behaved on this codebase.

### E3 — `_boundary()` was not removed by e11b788

`git show e11b788 -- agents/integrator_base/regional.py` changes only the *shim install*
(`_install_in_tree` → `_install_canonical`) and the shim *filename* passed to `_boundary`
(`shim_name` → `canonical_name`). Both the cache-hit path and the normal path still call
`_boundary(...)`. The region-rewrite logic in `boundary.py` is byte-for-byte unchanged.

### E4 — The actual failure mode: lost per-region commit artifact

- **Old design:** each region → unique `Bxm_<fam>_L##_##_<hash>.h` (new file) + a unique
  `#include` in the app header. Every accepted region commits a **distinct** change.
  Result: 152 shim files, 0 `empty_candidate`.
- **Dedup:** all regions of a family merge into `ql_shim_{dd,ff,float}.h`. When a region's
  symbols are already present (a sibling merged them) *and* the header already includes the
  canonical shim, the tree is **identical** → `commit_all` raises `NothingToCommitError`
  → `_commit` maps it to `empty_candidate`. Cross-file first-includers commit a **hollow
  include-only** change. Bad-run iteration tally: `{ok: 69, empty_candidate: 222,
  llm_gen_failed: 9}`.

### E5 — "Demotion" has no numerical effect through the current gate/Validator driver

`runs/qcdloop/src/boxGPU_vanilla.cpp` (the build gate + Validator "candidate" driver)
instantiates `run_app<Kokkos::complex<double>, double, double, VanillaPrinter>` — every
type is `double`. The per-region shim's `Constants<ddouble>` specialization is therefore
never instantiated by the build. Including a shim changes nothing numerically through this
driver (the good run's iter_0 tail shows `cand_min_precise_digits == curr_min_precise_digits`).
In the current harness a "demotion" is a **bookkeeping artifact** (shim + include), not a
measured precision change.

---

## What this means for the WAVE-4 ticket

- **Fix 1 ("restore the boundary rewrite of computation lines double→ddouble")** — no-op
  here. There is no such rewrite to restore; the regions have no bare `double` and
  `variables: []`, so `synthesize_boundary_patch` correctly emits include-only.
- **Probe new criterion ("a non-#include line rewriting double→ddouble at the target
  computation line")** — **unsatisfiable** for B2m.h:64/84. Implementing it verbatim yields
  a probe that can never pass.
- **Fix 2 (guard: a demotion commit must touch a non-#include line)** — sound in spirit and
  worth keeping; but note that under the *good* design the "non-#include line" that a
  demotion touched was the **new per-region shim file body**, not an app-source line. The
  guard must be defined against the artifact that actually exists.

## Options to consider when revising the ticket

1. **Per-region committable marker + guard (keep dedup).** Keep one `Constants<T>` per TU;
   have the boundary patch emit a distinct line-anchored marker at each region's computation
   lines so every accepted demotion commits a distinct non-`#include` change (kills
   `empty_candidate`), and add the Fix-2 guard against that marker. Retarget the probe to
   assert the per-region marker rather than a `double→ddouble` rewrite. Smallest change that
   honestly matches the template architecture.
2. **Real per-region precision rewrite.** Make a demotion actually change the region's
   working precision in app source (per-region local type-alias override, or driver-side
   per-region instantiation at the extended type). This is a genuine design change and also
   forces the gate/Validator driver to instantiate regions at the extended type so the
   change is measurable — well beyond a "narrow restore".
3. **Revert dedup.** Unique per-region shim files demonstrably produced 85 clean demotions;
   the ticket forbids this, but it is the lowest-risk way to restore the pipeline if the
   dedup's TU-cleanliness win is not worth the lost per-region artifact.

## What was NOT done

Per Reet's decision (2026-07-22): no code changes, no probe run, no 10k re-run. The dedup
structural work (`shim_merge.py`) is untouched and remains sound. `dispatch.py:is_retryable_misgen`
untouched.

## Method

Cumulative `final.diff` of runs `20260720_054121_dd44d33c` (PIPELINE_v1, good) and
`20260721_205813_b5d5dbf7` (Wave-3, bad); `iterations.jsonl` status tally; direct read of
`agents/integrator_base/regional.py` + `boundary.py`, `git show e11b788`, and
`runs/qcdloop/src/boxGPU_vanilla.cpp` + `runs/qcdloop/app/CMakeLists.txt`.
</content>
