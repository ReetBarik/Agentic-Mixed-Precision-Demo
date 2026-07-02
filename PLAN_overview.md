# Plan: Agentic Mixed-Precision Demo — Overview

**Branch:** `langgraph-agents`

> **High-level architecture of the agentic system.** For the current extension being implemented (whole-app characterization, Phase 0–2 + locked implementation contracts), see [`PLAN_implementation.md`](PLAN_implementation.md).

## Goal

LLM agent system that takes a user's C++ kernel(s), per-argument input ranges, and build instructions; characterizes numerical sensitivity per variable; and applies validated mixed-precision optimizations from a fixed catalog.

## Tracked library dependency

`Tracked<T>` C++ header-only library: overloads arithmetic ops to propagate condition number and accumulated relative error bound per variable, with per-op JSONL journaling and provenance tracking. Source of truth: `ReetBarik/Tracked-Error-Propagation-Datatype-Demo@main`. **Vendored as a git subtree** into this repo at `third_party/tracked/` and exercised end-to-end by the Phase 1 characterizer (`runs/cln/`, `runs/lnrat/`, etc.).

## Architecture

- **Framework:** LangGraph, shared `TypedDict` state.
- **Model:** Single model (Opus) for all agents in v1. Model-per-role deferred to v2 once we have data on where the reasoning bar lies.

## Agents

### 1. Build/Run agent

Owns the compilation environment: flags, framework detection (Kokkos / SYCL / OpenMP / CUDA / HIP / plain C++), module loads, include paths, link libraries.

Two modes:

- **Whole-app mode** — executes a build + run against a (possibly Range-Discovery-patched) source tree. User provides driver + build instructions; agent executes them on the source it's pointed at (original or instrumented).
- **Micro-driver mode** — agent wraps a single kernel given an *instrumentation spec* from the caller (which kernel, which input ranges, what telemetry to embed). Translates spec into actual driver source, compiles, runs, returns output.

The build/run agent is intentionally dumb about *what* the run means — it just executes what it's told.

### 2. Characterizer agent

Two tiers (whole-app pipeline splits this; single-kernel path collapses to Tier 1 only):

**Tier 1 — per-dependency** (implemented; this is the v1 characterizer slice):

1. Builds an instrumentation spec: instantiate the kernel template with `Tracked<double>`, sample inputs from user-provided per-argument ranges (or from Phase 0 samples in whole-app mode).
2. Invokes the build/run agent in micro-driver mode to produce a per-op condition / accumulated-error journal.
3. Parses the journal into a per-op / per-line / per-variable sensitivity profile (`sensitivity_profile.json`).
4. Cheap LLM symbolic overlay flags known unstable idioms (log-sum-exp, naive variance, catastrophic cancellation patterns) as `symbolic_hints.json` — diagnostic, not load-bearing.

**Tier 2 — per-kernel body** (designed, not yet implemented):

Body-only tracked variant of a top-level kernel; dependency calls remain opaque (handled via `tracked::opaque_at` with provenance attribution). Produces a `kernel_profile.json` with per-op body cancellation rollup + per-output condition + body-vs-dependency decomposition. Full design in [`PLAN_implementation.md`](PLAN_implementation.md).

Primary signal in both tiers is the `Tracked<T>` instrumented run. Shadow execution (FP64 vs FP128) is reserved for the validator, not the characterizer.

### 2a. Range Discovery agent (whole-app only, designed)

New agent introduced by the whole-app pipeline. Patches dependency call sites in the user's app with logging wrappers, builds the instrumented binary, runs the user's whole-app driver under realistic parameter ranges, and dumps per-dependency input distributions + call frequencies (`dependency_input_ranges.json`, `dependency_call_frequencies.json`, per-dep Parquet samples). Feeds Tier 1 and Tier 2 with empirically-derived inputs. Design + contracts: [`PLAN_implementation.md`](PLAN_implementation.md).

### 3. Strategy agent

Given the sensitivity profile, ranks applicable entries from the **fixed catalog**:

| Strategy | Patch shape | Risk |
|---|---|---|
| Downcast (`double` → `float`) | Type swap | Low |
| Float-float emulation (DD recovery) | Type swap | Low |
| FMA insertion | Line rewrite | Low |
| Algebraic rewrite (cancellation avoidance) | Line rewrite | Medium |
| Horner's method | Line rewrite | Medium |
| log-sum-exp rewrite | Multi-line | High |
| Kahan / compensated summation | Multi-line | Medium |
| Reassociation / pairwise summation | Multi-line | Medium |

Each catalog entry declares: preconditions, patch shape, risk, expected gain.

Output: ranked queue of strategy attempts.

Receives **structured failure feedback** from the validator (failing inputs, digits achieved vs required, per-variable error delta) to inform later proposals.

### 4. Patcher agent

Applies one strategy at a time. Supports:

- Type swaps
- Single-line rewrites
- Templated multi-line transformations (Kahan, log-sum-exp, etc.)

No free-form function-level rewrites in v1.

### 5. Validator agent

- Invokes the build/run agent in **whole-app mode**.
- Compares output against an FP128 reference.
- Acceptance metric is **configurable**: `min` / `p99` / `median` / `two-tier`. Default: **p99 ≥ 10 digits**.
- Reports the full distribution (min, p1, p50, p99, max) regardless of which metric gates acceptance.
- On failure, sends a structured failure report back to the strategy agent.

### 6. Orchestrator

LangGraph top-level. Wires the loop.

**Single-kernel path:**
```
characterize once
  → walk strategy queue
    → for each strategy:
        patch → validate
          accept: keep, retest next on top of new baseline
          reject: revert, try next
  → stop on queue exhaustion, iteration budget, or early-stop
```

**Whole-app path** (per [`PLAN_implementation.md`](PLAN_implementation.md)):
```
Phase 0: range discovery (once per app + param ranges)
  → Phase 1: characterize each dependency (parallel by process)
  → Phase 2: characterize each top-level kernel body (parallel by process)
  → for each top-level kernel:
      walk strategy queue (same loop as single-kernel path)
  → stop on queue exhaustion, iteration budget, or early-stop
```

## Loop semantics

- **Sequential layering** — each accepted patch becomes part of the new baseline; the next strategy is tested on top of the accumulated state.
- **No combining strategies in v1** — one at a time, validate, keep or revert.
- **Re-characterization between accepted patches deferred to v2** — characterize once at the start, walk the queue.

## Stop conditions

- Queue exhausted (natural stop).
- Iteration budget reached (default 50, configurable).
- Optional early stop after K consecutive failures (off by default).

Success = any accepted patch. Pipeline reports the full picture and lets the user judge.

## Inputs

User-provided:

- Source file(s) and kernel function name(s)
- Build instructions (script or commands)
- Whole-app driver
- Per-argument input ranges (min/max per argument)

CLI flags:

- `--acceptance-metric {min,p99,median,two-tier}` (default `p99`)
- `--min-digits N` (default 10)
- `--floor-digits M` (for two-tier mode)
- `--iteration-budget N` (default 50)
- `--early-stop-failures K` (default off)
- `--batch N`, `--seed N`, etc.

## Outputs

- **One** optimized kernel source file — cumulative result of all accepted patches.
- **JSON report** containing:
  - Accepted patches with reasoning
  - Rejected attempts with structured failure details
  - Sensitivity profile from characterization
  - Final accuracy distribution

## Deferred to a later cut

- Strategy combining (independence-class grouping using Tracked-type data flow).
- Re-characterization between accepted patches.
- Model-per-role assignment (cheap models for mechanical work).
- Free-form function-level rewrites.
- Additional secondary characterization strategies as overlays.
- v2 dependency-loss handoff between Tier 1 and Tier 2 (override `cond=1` at the opaque barrier with empirically-measured per-dependency `max_cond`). See [`PLAN_implementation.md`](PLAN_implementation.md) §3.
