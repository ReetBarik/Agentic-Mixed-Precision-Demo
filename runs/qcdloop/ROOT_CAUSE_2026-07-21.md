# ROOT CAUSE — the WAVE3 dedup regression, and what the precision-walk is actually validating

- **Date written:** 2026-07-22 (covers the 2026-07-21 runs; filename dated to the run under analysis)
- **Branch:** `langgraph-agents` @ `917bf99`
- **Runs analyzed:** PIPELINE_v1 (good) `20260720_054121_dd44d33c`; Wave-3 post-dedup (bad) `20260721_205813_b5d5dbf7`
- **Status:** analysis only — no code changes, no probe run, no 10k. Supersedes the causal diagnosis in `WAVE3_10K_2026-07-21.md`.

---

## 0. One-paragraph summary

The dedup regression is real but it is **not** what `WAVE3_10K_2026-07-21.md` said. There
was never an app-source "boundary conversion" to lose: qcdloop's box regions are
template-typed and the intents carry `variables: []`, so the boundary patch has always been
`#include`-only. The report's "~150 boundary conversions in the baseline" are **152 new
per-region shim *files***, miscounted. What dedup actually broke is **bookkeeping**: it
collapsed the unique per-region shim file (the only distinct committable artifact each region
had) into one shared canonical shim, so regions that reuse an already-merged shim now commit
nothing → `empty_candidate`. Separately — and more importantly — the trace below shows the
Validator builds the candidate at **plain `double`**, so the shim's `Constants<ddouble>` is
**never instantiated**; the precision walk is, as currently wired, a **numerical no-op**, and
the "152/152 accepts" were confirming *non-regression of an unchanged double build*, not any
precision gain.

---

## 1. What the previous report got wrong

`WAVE3_10K_2026-07-21.md` §banner + §A row *"non-shim boundary-conversion lines in
final.diff"* claims **~150 for PIPELINE_v1 vs 0 for Wave-3**, and calls the 0 "the smoking
gun." Both halves are wrong.

**Direct evidence — the baseline also has zero app-source conversions.** Counting added lines
in each run's cumulative `final.diff`, split into `#include` additions vs. everything else:

| app header | GOOD `20260720_054121` (inc / conv) | BAD `20260721_205813` (inc / conv) |
| --- | --- | --- |
| box/B0m.h | 30 / **0** | 3 / **0** |
| box/B1m.h | 22 / **0** | 3 / **0** |
| box/B2m.h | 41 / **0** | 3 / **0** |
| box/B3m.h | 10 / **0** | 3 / **0** |
| box/B4m.h | 16 / **0** | 2 / **0** |
| boxGPU.h | 7 / **0** | 3 / **0** |
| kokkosUtils.h | 26 / **0** | 3 / **0** |
| **per-region shim files created** | **152** | 0 |
| **canonical shims created** | 0 | **3** |

Neither run rewrites a single computation line in the app source. The good `box/B2m.h` hunk
is 41 `#include` lines and nothing else.

**What the "159 hunks" actually are.** A cumulative `final.diff` counts one hunk per new
file. PIPELINE_v1 created **152 new per-region shim files** (`B2m_dd_L65_65_<hash>.h`, …),
each a `new file … @@ -0,0 +N @@` hunk. Those 152 file-creation hunks plus a handful of
app-header include hunks are the "159." They are shim *files*, not conversions. The report's
`grep`-level intuition ("lots of `+` lines with `ddouble`/`make_dd`") was reading the **body
of the new shim files**, which of course contain `Constants<ddouble>` and `make_dd(...)` —
because they *are* the shims.

**`e11b788` did not sever the boundary call.** `git show e11b788 -- regional.py` changes only
the shim *install* (`_install_in_tree` → `_install_canonical`) and the shim *basename* handed
to `_boundary` (`shim_name` → `canonical_name`). Both the cache-hit and normal paths still
call `_boundary(...)`, and `boundary.synthesize_boundary_patch` is byte-for-byte unchanged.
The boundary patch is still produced; for these regions it is (and always was) include-only.

---

## 2. What a "demotion" actually is today — one accepted demotion, end to end

Trace: **PIPELINE_v1 iter_0, `double-to-dd` at `B2m.h:65`, verdict `accept`.**

1. **Intent** (`iterations.jsonl`): `{kind: double-to-dd, target: {file: B2m.h, line_start: 65,
   line_end: 65, variables: []}}`. Note `variables: []`.
2. **Regional integrator** (`regional.py`): reads the region at the parent SHA. The region is
   ```cpp
   TOutput r14 = ql::Constants<TOutput>::_half() * (TOutput(k14) + …);
   ```
   — working types are the template params `TOutput/TMass/TScale`, no bare `double`. With
   `variables: []` and no bare-`double` decls, `compute_promotion` finds nothing to promote.
3. **LLM shim**: generates `B2m_dd_L65_65_3bdc4977.h`, which specializes
   `template<> struct ql::Constants<quad::ddfun::ddouble>` with `_one()` and a complex
   `_ieps50<…>()` (the pre-derived `make_dd` bit pairs). This is a **TU-global type
   specialization**, not a change to the region's code.
4. **Boundary patch** (`boundary.synthesize_boundary_patch`): nothing to promote → the only
   edit is inserting `#include "B2m_dd_L65_65_3bdc4977.h"` after the header preamble. The
   emitted `iter_0.patch` is exactly that one include line.
5. **Commit**: the candidate tree = pristine + `B2m_dd_L65_65_3bdc4977.h` (new file) +
   one `#include` line in `box/B2m.h`. **That new file + that include line are the entire
   "demotion."**
6. **Build gate** (`gates.py`): builds `boxGPU_vanilla.cpp` (see §3) against the candidate
   headers; compiles and smoke-runs 21 rows, no NaN. Passes.
7. **Validator** (`validate.py`): builds **current** (working tree, vanilla double) and
   **candidate** (working tree + this patch, vanilla double), and compares both to a **DD
   ground truth built from the external `qcdloop@ddfun_enabled` repo** — *not* from the
   candidate tree. `precise_digits(candidate)` vs `precise_digits(current)`: since the
   candidate is a double build whose only delta is an unused shim header, **candidate coeffs
   == current coeffs bit-for-bit**. iter_0's own tail line shows it:
   `cand_min_precise_digits == curr_min_precise_digits == 3.6906`. The reported
   `candidate_min_precise_digits: 8.8399` is the *baseline's* global-min, inherited unchanged.
8. **Verdict**: regression guard sees zero regression (identical), digit floor already met by
   the baseline → `accept`.

**So a "demotion" today = one new per-region shim file + one include line. Its `Constants<T>`
specialization is never instantiated by any build the Validator runs (§3), so it changes no
number. The accept means "adding this shim header did not break the double build," not "this
region got more precise."**

---

## 3. What the vanilla driver instantiates — and whether any shim ever fires

`runs/qcdloop/src/boxGPU_vanilla.cpp` (the build-gate driver **and** the Validator's `current`
and `candidate` driver, via `_run_vanilla` → `QL_MODE=vanilla`):

```cpp
return ql_app::run_app<Kokkos::complex<double>, double, double,
                       ql_app::VanillaPrinter>(argc, argv);
```

`TOutput = Kokkos::complex<double>`, `TMass = double`, `TScale = double`. Every box template
is instantiated at `double`. A shim that specializes `ql::Constants<quad::ddfun::ddouble>` is
**never referenced** by these instantiations — the specialization is dead code in this TU.

**Does any shim ever fire?** Not in the Validator. Its three runs are:
1. **DD ground truth** → `_run_dd(dd_repo, dd_ref, …)` builds `boxGPU_dd.cpp` against the
   **external `ddfun_enabled` repo**, not the candidate tree.
2. **current** → `_run_vanilla(vanilla_headers, accepted, None, …)` — double.
3. **candidate** → `_run_vanilla(vanilla_headers, accepted, candidate_patch, …)` — double.

`boxGPU_dd.cpp` *does* instantiate at extended precision —
`run_app<ql::ddfun::ddcomplex, ql::ddfun::ddouble, ql::ddfun::ddouble, DDPrinter>` — and the
app `CMakeLists.txt` *can* build `QL_MODE=dd` against an arbitrary `QL_HEADERS`. So the
machinery to make the candidate's shims fire **exists** (build the candidate tree with
`QL_MODE=dd`), but the Validator **does not use it**: it takes its DD truth from a fixed
external repo and never builds the candidate tree at `dd`.

**Therefore the candidate's shims never instantiate anywhere in the pipeline.** The
"152/152 accepts" were validating: *the accumulated set of (dead) shim headers still compiles
and the double build's numbers are unchanged from baseline.* No precision improvement was ever
measured on a candidate, because no candidate was ever built at a precision where its shim
matters.

---

## 4. What the dedup change actually broke (bookkeeping, not numerics)

- **Old design (per-region shim install).** Each region wrote a uniquely-named file
  `Bxm_<fam>_L##_##_<hash>.h` and a unique `#include`. Every accepted region therefore
  produced a **distinct, always-committable artifact**. `empty_candidate` ≈ 0. The
  per-region file *was* the demotion's identity in git.
- **Dedup (`e11b788`, `_install_canonical` + `shim_merge.py`).** All regions of a family merge
  their symbols into one `ql_shim_{dd,ff,float}.h`. When a region's `Constants<T>` members are
  already present (a sibling merged them) **and** the app header already `#include`s the
  canonical shim, the resulting tree is **identical to the parent** → `commit_all` raises
  `NothingToCommitError` → `_commit` maps it to `empty_candidate`. Cross-file first-includers
  commit a **hollow include-only** change. Bad-run tally: `{ok: 69, empty_candidate: 222,
  llm_gen_failed: 9}`.

This is a **lost commit artifact**, cleanly distinct from a **lost numerical change**:

| | old (per-region) | new (dedup) |
| --- | --- | --- |
| per-region committable artifact | ✅ unique shim file | ❌ collapses when shim already merged |
| numerical effect through Validator | **none** (double build) | **none** (double build) |
| `empty_candidate` | ~0 | 222 |

The `shim_merge.py` structural work is correct and should stay (one `Constants<T>` per TU,
keep-first dedup held — verified in the WAVE3_10K report §E). What regressed is that dedup
removed the *only* thing that made each region a distinct commit, and nothing replaced it.

---

## 5. The bigger question this surfaces

If a "demotion" changes no number in any build the Validator runs, **what has the
precision-walk been characterizing?**

- **As currently wired: nothing numerical.** The walk selects regions, generates shims,
  commits them, and the Validator confirms the *double* build is unchanged. It has been
  exercising and validating the **agentic plumbing** (characterize → strategy → patch → merge
  → commit → build-gate → validator bookkeeping), not precision. Every "accept" is really
  "this shim header compiles and doesn't perturb the double build."
- **Is there a path where shims *do* fire?** Architecturally yes: build the **candidate tree**
  with `QL_MODE=dd` (→ `boxGPU_dd.cpp`, which instantiates `ddouble`). Then a region whose box
  code is reached under `TMass/TScale/TOutput = ddouble` would pick up the shim's
  `Constants<ddouble>` and actually compute in double-double. **The Validator does not do
  this** — its DD run comes from the external `ddfun_enabled` repo, and its candidate run is
  vanilla double. So the wiring to make shims matter exists but is not connected to the
  candidate.
- **Why did earlier "speedup data" appear (CALIBRATION notes: "80 double→ff")?** That needs
  checking against the same lens — either a different driver/config instantiated the shims, or
  those numbers were also bookkeeping counts (regions demoted) rather than measured wall-clock
  / precision deltas. I did not re-derive them in this pass; flagging as a follow-up.

The uncomfortable implication: the dedup regression *hid* a pre-existing condition. The
pipeline was already a numerical no-op in PIPELINE_v1; it just committed a distinct file per
region so the walk *looked* productive. Fixing the bookkeeping alone would restore the
appearance without restoring (creating) real numerical effect.

---

## 6. What I need from you before proposing a fix

The right fix depends entirely on the pipeline's intended semantics, which the current wiring
does not pin down. Concrete questions:

1. **Is the precision walk supposed to change numerics on the candidate, or is it (for now) a
   plumbing/coverage exercise?** If the latter, the fix is purely to restore per-region
   committable artifacts (bookkeeping) and the Validator's role is coverage, not precision.
2. **Should the Validator build the candidate tree at `QL_MODE=dd`** (so shims instantiate and
   `Constants<ddouble>` actually fires), instead of / in addition to taking DD truth from the
   external `ddfun_enabled` repo? If yes, that — not the boundary patch — is the change that
   makes a "demotion" mean something, and it's a Validator change, not a Patcher change.
3. **How is a region's working type meant to become the extended type at all?** The box code is
   templated on `TOutput/TMass/TScale`. Three possibilities: (a) driver instantiates the whole
   app at the extended type (global, not per-region); (b) a per-region local type-alias / cast
   override injected into the source (a real boundary rewrite — but then the region needs bare
   `double`/alias sites to rewrite, which these don't have); (c) something else. Which is the
   intended mechanism?
4. **What is the acceptance criterion supposed to test?** Today it's "candidate double build
   doesn't regress vs baseline double build." Should it instead be "candidate at extended
   precision gains ≥ N digits vs baseline"? That reframes the entire verdict logic.
5. **Given all the above, is dedup even desirable?** If shims must fire per region under a `dd`
   build, one merged `Constants<ddouble>` per TU is correct C++ (no redefinition) — dedup
   stays. If the identity of a demotion must remain per-region for the walk, we need a separate
   per-region marker. These aren't in tension, but the answer determines whether the fix is
   "marker + guard" (option 1), "wire the dd candidate build" (option 2, larger), or both.

Until (1)–(4) are settled, any Patcher-side "restore the boundary rewrite" change is either a
no-op (nothing to rewrite) or cosmetic (a marker that makes the walk commit again without
making a demotion mean anything).

---

## Method / provenance

- Per-file conversion counts: `awk` over each run's `final.diff`, splitting `+` lines into
  `#include` vs. other, per app-header section.
- Iteration tallies: `iterations.jsonl` of each run.
- Boundary/merge behavior: read of `agents/integrator_base/{regional,boundary,shim_merge}.py`
  and `git show e11b788`.
- Instantiation + validator wiring: `runs/qcdloop/src/boxGPU_{vanilla,dd}.cpp`,
  `runs/qcdloop/app/CMakeLists.txt`, `agents/patcher/gates.py`,
  `agents/validator/{validate,runner,agent}.py`.
- No code was modified. `shim_merge.py` dedup left intact.
</content>
