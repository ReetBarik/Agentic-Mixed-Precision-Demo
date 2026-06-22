# PLAN: Whole-app characterization of qcdloop integrals

**Status:** Design discussed 2026-06-21. Not yet implemented.

**Repo:** `ReetBarik/Agentic-Mixed-Precision-Demo` (branch `langgraph-agents`), targeting `ReetBarik/qcdloop` as the first whole-app integration.

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

1. **Productionize `TrackedComplexDouble` + write `TrackedDouble`.** Clean rewrite from the prototype. Verify against vanilla on a few known test cases. Effort: ~1 day.
2. **`ql::Constants` specializations + tracked overloads for `ql::*` math wrappers.** Effort: ~half-day each, mostly mechanical.
3. **Hand-write `B13_tracked.h`** (Option A pattern, body-only Tracked, leaves as opaque). Validate that the build + Serial run produces sensible per-op rollups. Effort: ~1–2 hours.
4. **Phase 0 prototype: hand-written logging wrappers** for the leaves B13 actually touches (`Lnrat`, `cLn`, `Li2omx2`, `xspence`, `ratgam`, `ratreal`, `spencer`). Compile-and-run on the existing test driver with batch_size=100k. Verify Parquet dump and per-leaf range aggregation. Effort: ~half-day.
5. **Wire Phase 1** to consume Phase 0 outputs as Tier 1 ranges. Reuse existing characterizer. Run Tier 1 on each instrumented leaf. Effort: ~half-day (mostly orchestration).
6. **Wire Phase 2** to inject Phase 1 leaf profiles into `B13_tracked.h` at the post-leaf re-entry points. Compare body-only vs body+leaf-injected loss estimates. Effort: ~half-day.
7. **Validate end-to-end** on B13. Compare against vanilla output (correctness) and against the existing accumulation-only signal (precision delta). Effort: ~1 hour.
8. **Generalize:** templated `Bxx_tracked.h` for the other 21 integrals. Once one is hand-written, this is partly mechanical, partly a research question (can the LLM generate the tracked variant from the vanilla source?). Effort: 1–2 days hand-written, open-ended if LLM-driven.
9. **Build out the Range Discovery agent proper** (LLM-driven source parsing, wrapper generation, build/run integration) once the manual prototype validates the data flow. Effort: ~1 week.
10. **Build out Tier 2 characterizer extensions** (spec-builder for `Kokkos::Array` inputs, driver-gen Kokkos batch template, log parser per-op CSV). Effort: ~1 week.

Stop at any point if the signal isn't useful — steps 1–7 are the minimum viable proof of the architecture.

---

## Open questions / known limits

- **Correlated leaf failures.** Phase 1+2 composition gives a conservative upper bound under independence assumption. The `wlog3mu`/`wlog4mu` correlation in B13 will be undercounted. Acceptable for v1; revisit if Strategy Agent's recommendations are visibly wrong.
- **Branch-cut sensitivity in leaves.** Not caught by this architecture. Requires symbolic analysis or analytic bounds.
- **LLM-generated `Bxx_tracked.h`?** Open research question. Hand-writing one per integral is acceptable for v1 but doesn't scale to other libraries. The translation is structurally regular (body-only Tracked promotion, leaf calls left intact) which makes it a reasonable LLM target.
- **Whether Tier 2 needs per-op rollup or per-output rollup.** Per-op is richer but heavier on the log/parser. Per-output (just `res(i, 0..2)`) is what your prototype did and is much cheaper. Probably want both: per-op for the Characterizer's profile, per-output as the headline number.
- **How to feed leaf-input regime info into Strategy Agent.** Body cancellation amplifying noisy-leaf-output is a *coupled* failure mode — needs joint profile artifact, not just two separate per-leaf and per-integral profiles. Schema TBD.

---

## What this does NOT do

- Does not characterize qcdloop's GPU performance behavior (separate concern)
- Does not validate that patches preserve correctness (Validator's job, Phase 3+)
- Does not handle the "Bxx is selected by physics-deterministic input pattern" routing — Phase 0 just observes which Bxx fires for each input; if the user wants to characterize one specific Bxx in isolation, they set ranges to trigger it deterministically (which is what the existing test driver already does case-by-case)
- Does not replace the user's understanding of the physics. The Strategy Agent should propose fixes; the user reviews and accepts/rejects.

---

## Companion artifacts elsewhere

- Diagram for the LangGraph wiring: `/Users/rbarik/.openclaw/workspace/diagrams/agentic-workflow-v3.mmd` (current v2 architecture; Range Discovery Agent would slot in as a node before the Characterizer)
- Existing characterizer slice: `agents/characterizer/` in the `langgraph-agents` branch
- Build/Run agent (whole-app mode pending): `agents/build_run/`
- Leaf source: `src/qcdloop/box/B*.h`, `src/qcdloop/kokkosUtils.h`, `src/qcdloop/kokkosMaths.h` in qcdloop
- Whole-app driver: `examples/boxGPU_test.cc` in qcdloop
