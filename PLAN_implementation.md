# PLAN: Implementation — whole-app characterization

**Status:** Design discussed 2026-06-21. Implementation contracts locked 2026-06-28 (see §Implementation contracts at the bottom). Not yet implemented.

**Repo:** `ReetBarik/Agentic-Mixed-Precision-Demo` (branch `langgraph-agents`), targeting `ReetBarik/qcdloop` as the first whole-app integration.

> **Lower-level implementation plan.** Architectural context (agent decomposition, framework, catalog, loop semantics) is in [`PLAN_overview.md`](PLAN_overview.md).

---

## Goal

Characterize numerical stability of the qcdloop box integrals (BIN0–BIN4, B1–B16) under user-provided parameter ranges, producing actionable sensitivity profiles that the Strategy Agent can use to drive mixed-precision optimizations.

Operate at the **integral level** (B13, B11, etc.) as the primary unit of characterization, with leaf math kernels (Lnrat, cLn, Li2omx2, xspence, …) characterized separately and composed in.

---

## Constraints and decisions

- **Host-only Serial backend** across the entire characterization pipeline. Tracked type uses `std::fma`/`std::log2`/etc. — not device-callable. No benefit to GPU here; characterization is not throughput-bound.
- **GPU/device validation lives in Phase 3+** (Validator), separate from characterization.
- **Tracked datatype:** existing `TrackedComplexDouble` (in workspace as reference) is too primitive for production use — needs a real implementation. Companion `TrackedDouble` scalar version also needed.
- **No View element-type substitution.** Views stay vanilla (`Kokkos::complex<double>`). Tracking happens inside integral bodies operating on `Y[i][j]`, `mu2`, and derived scalars. The integral signature (`B13(res_view, Y, mu2, i)`) is operated on as: read inputs, tracked arithmetic, write `res(i, 0..2)` as plain `TOutput` at the end.
- **Integral body ↔ leaf boundary is the tracking boundary.** Bodies are tracked op-by-op; leaves are treated as opaque black boxes with empirically-derived input/output noise levels.

---

## Three-phase pipeline

```
User: whole-app + build instructions + param ranges + integrals-of-interest
   ↓
[Range Discovery Agent]     ── new agent (Phase 0)
   ├─→ per-leaf input distributions
   ├─→ per-leaf call frequencies
   └─→ raw sample dump (Parquet)
   ↓
[Characterizer Tier 1]      ── existing characterizer slice (Phase 1)
   └─→ per-leaf sensitivity profiles
   ↓
[Characterizer Tier 2]      ── extended characterizer (Phase 2)
   └─→ per-integral sensitivity profiles
       (body cancellations + leaf-injected noise)
   ↓
[Strategy Agent] → [Patcher] → [Validator] → walk loop
```

All three phases run host-only Serial. All three are parallelizable within phase (leaves in Phase 1, integrals in Phase 2).

---

## Phase 0: Range Discovery Agent (new)

### Why a separate agent

- **Different scope:** whole-app (driver + build + param ranges), not single-kernel
- **Different outputs:** input distributions and call frequencies, not sensitivity profiles
- **Different LLM work:** source parsing + logging-wrapper transformation, not micro-driver generation
- **Different cadence:** runs once per (whole-app, param-range) config; feeds N Tier 1 runs and N Tier 2 runs
- **Not qcdloop-specific:** any whole-app workload with a body→leaf decomposition can use it

### Inputs
- Whole-app source tree (qcdloop)
- Build instructions (qcdloop CMakeLists, needs Kokkos Serial)
- User-provided param ranges (mass/momentum/μ² distributions per integral case)
- List of leaf-call sites to instrument (auto-detected via LLM source scan, or user-specified)
- `batch_size = 100k` per integral (~22M total records, ~200 MB Parquet — fine)

### Steps
1. **Parse integral source files** (LLM): identify all leaf call sites, their templated signatures, return types, arg types. Output: list of `(file, line, leaf_name, arg_types)` tuples.
2. **Generate logging wrappers:** thin wrappers that log inputs and forward to real impl. Replace call sites via sed/template-based source patching.
3. **Build the instrumented binary:** calls the shared Build/Run agent in whole-app mode (Kokkos Serial config) on the patched source.
4. **Run with user ranges:** execute the existing whole-app driver. Logging wrappers append `(integral_id, leaf_name, inputs)` to per-leaf Parquet files (`runs/range_discovery/Lnrat.parquet`, etc.).
5. **Aggregate:** postprocess into per-leaf input distributions (marginals + optional bivariate joints for successive same-leaf calls), call counts, integral-level call frequencies.
6. **Emit artifacts:**
   - `leaf_input_ranges.json` — per-leaf min/max/percentiles per input dimension
   - `leaf_call_frequencies.json` — per-integral, per-leaf call counts
   - `leaf_input_samples.parquet` (per-leaf) — raw samples for Tier 1 to draw from

### Why 100k samples, not reservoir sampling
- 22 × 100k × ~10 leaf calls ≈ 22M records → ~200 MB Parquet — disk is cheap
- Preserves call ordering → recoverable joint distributions later
- Fully deterministic given seed
- Auditable (grep the log if "Tier 1 sees weird input range")
- Tier 1 subsamples (~5–10k per leaf characterization run); having the full set means resampling strategies can change without re-running Phase 0

### Implementation notes
- Use Parquet not JSONL (5–10× smaller, fast to load into Tier 1)
- One file per leaf, appended across integrals
- `std::ofstream` is legal — Serial backend, host code, no `Kokkos::printf` acrobatics
- Source-parsing for templated C++ across multiple files is the LLM-heavy step. For qcdloop specifically, regex over `ql::Lnrat<`, `ql::cLn<`, etc. is probably enough; for the general-purpose agent, the LLM does real work
- Logging wrappers must respect `KOKKOS_INLINE_FUNCTION` annotations — fine on Serial host where everything is host code anyway

### Byproducts (also useful)
- **Call-frequency-weighted leaf importance** → feeds Strategy Agent's prioritization
- **Dead-leaf detection** → leaves that never fire for the user's workload can be skipped in Tier 1
- **Joint distribution sketches** → partial info on leaf-input correlation across calls

---

## Phase 1: Characterizer Tier 1 (existing slice)

### What changes from today
- Inputs are **empirically-derived** from Phase 0, not guessed
- Per leaf: sample inputs from `leaf_input_samples.parquet` (or fit a distribution and resample)
- Otherwise unchanged: existing scalar-leaf characterization machinery applies directly

### Inputs
- Per-leaf input range JSON + raw samples from Phase 0
- Leaf source files (already in qcdloop tree)
- The existing characterizer config (sample count, retry budget, etc.)

### Outputs
- Per-leaf sensitivity profiles (`sensitivity_profile.json` per leaf)
- Per-leaf symbolic hints (`symbolic_hints.json` per leaf) — optional LLM overlay
- Same artifact schema as today's characterizer

### Effort
- ~zero new agent code — reuse existing characterizer
- Wiring change: orchestrator feeds Phase 0 outputs as ranges/samples instead of YAML

---

## Phase 2: Characterizer Tier 2 (extended characterizer)

### Strategy
Track every arithmetic op **inside the integral body** (B13's `m3sq = Y[2][2]`, `sibar * tabar - m3sqbar * m4sqbar`, `wlogsmu - wlog4mu` differences, etc.) but **stop at the leaf-call boundary** — leaves are still opaque, called with plain values, results re-enter tracking.

### Critically: post-leaf noise is initialized from Tier 1 profiles, not zero

Instead of:
```cpp
TOutput wlogsmu_plain = ql::Lnrat<...>(sibar.get(), mu2.get());
TrackedComplexDouble wlogsmu(wlogsmu_plain);  // ← resets lost-bits to 0
```

We do:
```cpp
TOutput wlogsmu_plain = ql::Lnrat<...>(sibar.get(), mu2.get());
TrackedComplexDouble wlogsmu(wlogsmu_plain);
double leaf_loss = leaf_profile_lookup("Lnrat", sibar.get(), mu2.get());
wlogsmu.lost_bits_rounding_real = leaf_loss;
wlogsmu.rounding_error_abs_real = std::pow(2.0, leaf_loss - 53.0) * std::abs(wlogsmu.real());
wlogsmu.update_total();
```

The body tracking then propagates this leaf-injected noise through subsequent ops automatically.

### What this catches
- **Body cancellations** (e.g. Gram determinant `sibar*tabar - m3sqbar*m4sqbar` in B13's `fac`)
- **`p3sq + m3sq - m4sq` and sqrt-of-difference cancellations**
- **Cancellation between leaf outputs** (`wlogsmu * wlogtmu - wlog3mu * wlog4mu`-style assemblies)
- **Pre-leaf cancellation in leaf inputs** (e.g. `m3sqbar = 2*Y[1][2]` accumulated from differences)
- **Leaf-introduced noise** at the level Tier 1 measured it

### What it still misses
- **Correlated leaf failures** across calls (e.g. `wlog3mu`/`wlog4mu` both functions of shared `mu2`+masses — their joint loss ≠ sum of marginals)
- **Branch-cut flips inside leaves** when `iszero(p3sq)` evaluation noise flips the branch
- **Leaf-internal cancellation when inputs are outside Tier 1's sampled regime** (mitigated by Phase 0 making Tier 1's sampling targeted, but tail regions can still escape)

### Required infrastructure changes

| Component | Change |
|---|---|
| `TrackedComplexDouble` | Productionize — clean rewrite, not the prototype |
| `TrackedDouble` | New companion scalar type (~200 lines, mirror of `TrackedComplexDouble`) |
| `ql::Constants<TrackedDouble>` and `<TrackedComplexDouble>` | Specializations: `_one()`, `_two()`, `_pi()`, `_half()`, `_pi2()`, etc. (~30 constants × 2 types) |
| `ql::kSqrt`, `kLog`, `kAbs`, `kPow`, `Real`, `Imag`, `Max`, `Sign`, `iszero` | Overloads for tracked types (~10 functions × short implementations) |
| Tracked integral variants (e.g. `B13_tracked.h`) | Per-integral, body-only Tracked promotion. Hand-written first; LLM-generated later |
| Leaf-profile lookup table | Indexed by leaf name + input bucket; loads Tier 1 outputs |
| Per-op CSV schema | Extension to record per-op cancellation/rounding bits during body tracking, not just final accumulation |
| Spec builder (extension) | Recognize `Kokkos::Array<T, N>` inputs, View-slot outputs, thread-index params (no View element-type substitution — that's the key simplification) |
| Driver generator | New template: Kokkos batch micro-driver (init, alloc Views, fill, deep_copy, parallel_for, fence, readback, finalize). One-shot template, not per-kernel reasoning |
| Log parser | Parse new per-op CSV schema from Tier 2 body tracking |

### Outputs
- Per-integral sensitivity profile: per-op cancellation/rounding rollup
- Annotated with leaf-input regimes (so Strategy Agent can correlate body loss with leaf-input quality)
- Same JSON schema family as Tier 1, with body-vs-leaf decomposition

---

## Two distinct signals for the Strategy Agent

Phase 1 and Phase 2 produce **different kinds of optimization signals**, and the Strategy Agent should treat them differently:

| Signal | Source | Example fix |
|---|---|---|
| Leaf in high-loss regime | Tier 1 | "Substitute analytic special-case formula for `Lnrat(m3sqbar, mu2)` when `m3sqbar/mu2 → 1`" |
| Body cancellation | Tier 2 | "Compute `fac = sibar*tabar - m3sqbar*m4sqbar` via Kahan summation" or "Use Denner §3.2 alternate form for this term" |
| Leaf called frequently in high-loss regime | Phase 0 freq × Tier 1 loss | Prioritize this leaf for replacement/rewrite |
| Body cancellation amplifying leaf noise | Tier 1 + Tier 2 composition | Coupled fix: clean up leaf input *and* restructure body |

---

## Implementation order

1. ~~Productionize `TrackedComplexDouble` + write `TrackedDouble`.~~ **Already done.** Adopt `third_party/tracked/` (vendored as a git subtree from `ReetBarik/Tracked-Error-Propagation-Datatype-Demo@main`) as-is. Already integrated and exercised by Phase 1 characterizer. Effort: 0 days.
2. **`ql::Constants` specializations + tracked overloads for `ql::*` math wrappers.** Tracked overloads already exist for the Phase 1 fixtures (see `runs/cln/src/micro_driver.cpp`); extend on demand when Phase 2 encounters a missing one. Effort: as needed, mostly mechanical.
3. **Phase 0 prototype: hand-written logging wrappers** for the dependencies the first target kernel touches. Compile-and-run on a Serial driver with batch_size=100k. Verify Parquet dump and per-dependency range aggregation. Effort: ~half-day.
4. **Wire Phase 1** to consume Phase 0 outputs as Tier 1 ranges. Reuse existing characterizer. Run Tier 1 on each instrumented dependency. Effort: ~half-day (mostly orchestration).
5. **Wire Phase 2** to consume Phase 1 outputs (provenance attribution via `tracked::opaque_at`; conservative `cond=1` for v1, see §3). Compare body-only vs full-pipeline loss estimates. Effort: ~half-day.
6. **Validate end-to-end** on the first target kernel. Per §6: schema validation (hard gate), tracking correctness (existing prototype tests), and hotspot recall against `symbolic_hints.json` annotations. Effort: ~1 hour beyond the prerequisites.
7. **Generalize:** apply to additional target kernels in the app. Open research question whether tracked-variant generation can be LLM-driven. Effort: 1–2 days per kernel hand-written, open-ended if LLM-driven.
8. **Build out the Range Discovery agent proper** (LLM-driven source parsing, wrapper generation, build/run integration) once the manual prototype validates the data flow. Effort: ~1 week.
9. **Build out Tier 2 characterizer extensions** (spec-builder for `Kokkos::Array` inputs, driver-gen Kokkos batch template, log parser per-op CSV). Effort: ~1 week.

Stop at any point if the signal isn't useful — steps 1–6 are the minimum viable proof of the architecture.

---

## Open questions / known limits

- **Correlated leaf failures.** Phase 1+2 composition gives a conservative upper bound under independence assumption. The `wlog3mu`/`wlog4mu` correlation in B13 will be undercounted. Acceptable for v1; revisit if Strategy Agent's recommendations are visibly wrong.
- **Branch-cut sensitivity in leaves.** Not caught by this architecture. Requires symbolic analysis or analytic bounds.
- **LLM-generated `Bxx_tracked.h`?** Open research question. Hand-writing one per integral is acceptable for v1 but doesn't scale to other libraries. The translation is structurally regular (body-only Tracked promotion, leaf calls left intact) which makes it a reasonable LLM target.
- **How to feed leaf-input regime info into Strategy Agent.** Body cancellation amplifying noisy-leaf-output is a *coupled* failure mode — needs joint profile artifact, not just two separate per-leaf and per-integral profiles. Schema TBD.
- **Mermaid workflow diagram update tabled (2026-06-28)** — trying to make a self-contained diagram that captures the phase structure without becoming unreadable is a losing battle. Current single-pipeline view stays in `improvement-plan/mermaid.md`; this plan is the authoritative reference.

---

## What this does NOT do

- Does not characterize qcdloop's GPU performance behavior (separate concern)
- Does not validate that patches preserve correctness (Validator's job, Phase 3+)
- Does not handle the "Bxx is selected by physics-deterministic input pattern" routing — Phase 0 just observes which Bxx fires for each input; if the user wants to characterize one specific Bxx in isolation, they set ranges to trigger it deterministically (which is what the existing test driver already does case-by-case)
- Does not replace the user's understanding of the physics. The Strategy Agent should propose fixes; the user reviews and accepts/rejects.

---

## Companion artifacts elsewhere

- Workflow diagram: `improvement-plan/mermaid.md` (single-pipeline view; update tabled per Open questions)
- Existing characterizer slice: `agents/characterizer/` in the `langgraph-agents` branch
- Build/Run agent (whole-app mode pending): `agents/build_run/`
- Tracked library: `third_party/tracked/` (vendored as a git subtree from `ReetBarik/Tracked-Error-Propagation-Datatype-Demo@main`)
- qcdloop integration (first target app): external repo `ReetBarik/qcdloop`. Dependency source: `src/qcdloop/box/B*.h`, `src/qcdloop/kokkosUtils.h`, `src/qcdloop/kokkosMaths.h`. Whole-app driver: `examples/boxGPU_test.cc`.

---

## Implementation contracts (locked 2026-06-28)

Outcome of a gap-by-gap audit on 2026-06-28. These decisions are the reference
during implementation; revisit only with justification.

### 1. Artifact schemas

**Per-dependency profile** — adopt the existing characterizer schema as the contract:
`runs/per_dependency/<dep>/sensitivity_profile.json` with fields `kernel`,
`samples_run`, `per_op[]`, `per_line{}`, `per_variable{}`, `top_hotspots[]`,
`opaque_coverage`. Companion `symbolic_hints.json` (separate file) carries
LLM-derived idioms with `location` + `severity` + `suggested_rewrite`. No new
fields invented.

**Per-kernel profile** (Phase 2) — `kernel_profile.json` extends the
per-dependency schema with:
- `dependency_profile_refs: {<dep>: {path, sha256}}` — pinned references to
  Phase 1 outputs consumed during tracking
- `per_output: {<out>: {max_cond, max_rel_err}}` — rollup at kernel return points
- `body_vs_dependency_decomposition: {body_max_cond, dependency_max_cond}` —
  headline number distinguishing body cancellations from dependency-injected loss

**Phase 0 outputs:**
- `dependency_input_ranges.json` — per-dependency, per-dim stats (`min`, `max`,
  `p01`, `p50`, `p99`, `n_nonfinite`, `n_negative`). For `complex<double>` dims:
  `stats_real` + `stats_imag` sub-objects (the only place real/imag keying lives
  in the schema; all other complex tracking decomposes at the op level).
- `dependency_call_frequencies.json` — `per_kernel{<kernel>: {total_evaluations,
  dependencies{}}}` and `per_dependency_totals{}`.
- `<dep>_input_samples.parquet` — one file per dependency. Columns:
  `kernel_id` (dictionary-encoded), `call_idx` (uint64), per-arg columns
  (`arg_<i>` for real, `arg_<i>_re` + `arg_<i>_im` for complex). ZSTD,
  row-group 64k.
- `dependency_manifest.json` — index across all per-dependency files.

All JSON artifacts carry `schema_version: 1`. Timestamps live only in
`run_config.json` / `run_metadata.json`, never in primary artifacts (keeps diffs
meaningful).

### 2. Tracked types

No new code. Adopt `tracked::Tracked<T>` and `tracked::Complex<T>` from
`third_party/tracked/` (vendored as a git subtree from
`ReetBarik/Tracked-Error-Propagation-Datatype-Demo@main`),
already integrated and exercised by the Phase 1 characterizer.

**Dependency re-entry pattern** (replaces all hand-wave injection snippets):
```cpp
auto wlogsmu_plain = ql::Lnrat<...>(sibar.value(), mu2.value());
auto wlogsmu = tracked::opaque_at("Lnrat", wlogsmu_plain,
                                  TRACKED_HERE, sibar, mu2);
```
The opaque barrier propagates `max(input_rel_errs) + u` with `cond=1`,
unions provenance, and adds `fn_name` to the provenance set. Worked example:
`runs/cln/src/micro_driver.cpp`.

**Convention:** the dependency function name MUST be the first argument to
`opaque` / `opaque_at` so provenance is greppable.

**Interop shim taxonomy** (already established in `runs/cln/`):
- `interop_shim`: dependency has a tracked-aware overload; delegate directly
  (e.g. `ql::kLog` on a `Tracked<T>` forwards to `tracked::log`).
- `opaque_wrap`: dependency is treated as a black box; call the native
  implementation on unwrapped values, re-wrap with `tracked::opaque` to
  preserve provenance.

### 3. Dependency-loss handoff at re-entry

v1: conservative `cond=1` from `tracked::opaque_at`. No lookup table; no
Phase 1 → Phase 2 numerical handoff beyond provenance attribution.

v2 (deferred): use `dependency_profile.max_cond` as a local cond override at
the opaque barrier. Requires a one-line extension to the tracked API to accept
an explicit `cond` argument. Add only if v1's conservative bound is provably
too loose for Strategy Agent decisions.

### 4. Determinism + seeding

Scope: fair comparison between baseline and patched runs on the **same input
set**. Not pursuing cross-machine or cross-compiler bit-reproducibility.

Per run: persist (a) the seed used to generate inputs, (b) a snapshot of the
generated input set. Both go in the run dir. Baseline and every candidate
patch in that run consume the snapshot. Cross-session reruns reuse the
snapshot, so PatchSet measurements are comparable across sessions.

### 5. Failure modes — warn loudly, fail rarely

Hard-fail only on:
- Disk full / OOM
- Tracked library can't be loaded

Everything else degrades gracefully and surfaces as a flag in the artifact
the Strategy Agent reads:

| Phase | Condition | Response |
|---|---|---|
| 0 | LLM mis-parses a dependency call site | Retry N times with error fed back; on giveup, flag site and skip |
| 0 | Wrapper patching breaks build | Same retry loop; surface patched-source diff |
| 0 | Dependency produces zero samples | Mark as dead-leaf-for-this-workload in `dependency_call_frequencies.json` |
| 1 | Dependency has no Phase 0 samples | Skip with warning; Phase 2 falls back to `opaque_at` default |
| 1 | Tracked op emits non-finite cond/rel_err | Already handled by tracked lib (emits null); aggregator counts + reports |
| 1 | Sample run exceeds timeout | Partial profile with `samples_run` reflecting completion; flag as partial |
| 2 | Phase 1 profile missing for a dependency the kernel calls | Fall back to `cond=1`; log the gap |
| 2 | Tracked body NaN-propagates | Record + continue; flag op as "tracking lost" |
| 2 | Tracked kernel variant fails to compile | Retry loop (same pattern as Phase 1 driver gen, commit `9f91f34`) |
| 2 | Op count explodes per sample | Truncate journal with warning; reduce samples-per-kernel instead of ops-per-sample |
| cross | Tool / source SHA mismatch between phases | Warn (don't fail); cross-run profile reuse is legitimate |

### 6. Validation — three layers

**Tracking correctness:** prototype's existing test suite at
`third_party/tracked/tests/` (cancellation, Kahan, log-sum-exp, naive variance,
Smith division, complex log/sqrt). No additions needed for v1.

**Schema correctness:** JSON Schema (draft-07) file per artifact type under
`schemas/`. CI runs each phase on a fixture, validates outputs. Pass criterion:
100% schema match. Hard gate.

**End-to-end signal usefulness:** recall verifier on top of existing
`sensitivity_profile.json` + `symbolic_hints.json`. For each validation
fixture, recall = fraction of `symbolic_hints[*].location` covered by
`top_hotspots[*].location`. Target: ≥80% on `high` severity, ≥50% on `medium`,
unbounded precision (false positives acceptable). Surface as non-blocking
status check.

**Prerequisite for the recall verifier:** add `TRACKED_HERE` at use sites in
`lnrat`, `cln`, `cancellation`, `kahan`, `naive_variance` fixtures (currently
emit empty `location` fields, blocking match). Verify path relativization
(commit `918738e`) actually fires on a fresh run.

### 7. Repo layout

```
agents/
  characterizer/             Phase 1 (per-dependency) — existing
  range_discovery/           Phase 0 — new
  kernel_characterizer/      Phase 2 (per-kernel body) — new
  build_run/                 existing
  shared/                    journal parsers, schema validators, interop helpers
  orchestrator/              Phase 3 (PatchSet loop) — landing later

runs/
  per_dependency/<dep>/      current runs/<kernel>/ moves here
  whole_app/<app>/<run_id>/
    run_config.json
    phase0/  dependency_input_ranges.json, dependency_call_frequencies.json,
             dependency_manifest.json, samples/<dep>.parquet
    phase1/  <dep>/sensitivity_profile.json, <dep>/symbolic_hints.json
    phase2/  <kernel>/kernel_profile.json, <kernel>/journal.jsonl
    phase3/  Strategy Agent outputs

tests/
  agents/fixtures/           existing unit-test fixtures
  validation/                annotated kernels for recall verifier (see §6)

third_party/tracked/         existing, vendored
schemas/                     JSON Schema files for every artifact type
```

App source (e.g. qcdloop) stays **external** — `run_config.json` carries an
`app_source_path` field, Phase 0's build step clones / references it. Keeps the
workflow app-agnostic.

### 8. Naming — workflow concepts vs qcdloop examples

Schema field names stay generic: `kernel` (the unit being characterized),
`dependencies` (functions a kernel calls and treats as opaque). qcdloop-
specific nouns (`B13`, `Lnrat`, `cLn`, `wlogsmu`) appear only in worked
examples, never as schema keys.
