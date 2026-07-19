# 10k Strategy Calibration v2 — Wave 1 + Wave 2 validation

Single faithful re-run validating the Wave-1 pipeline fixes (chain-representative
dedup, build-fuse, template-rung skip, `dr_k=60`, 200/250 caps) **and** the Wave-2
new capability (reachable `double→float` on template-typed code via the float
integrator; `_ieps50`/R4 complex-regulator constant fix). Companion to
`CALIBRATION.md` (Wave-1 Runs A/B) — read that first for pipeline mechanics; this
is terse.

**Date:** 2026-07-19 · **branch:** `langgraph-agents` @ `63e8986` (Wave-2 FF-merged) ·
**tolerance:** 7.0 · **model:** claudeopus47 · **validator:** real 3-build, seed 12345,
n=1000, DD oracle `~/qcdloop@ddfun_enabled` · **report:** `report_10k.json` (fixed input, reused)

## TL;DR

- **Faithful 200/250 drained → `success`.** The config that stopped `partial` in
  Wave-1 Run A now completes both phases in **101 min** (Run A: 78 min, correctness
  only, never reached speedup). Chain-dedup is the unlock — **527,523 chains skipped**.
- **Float fires and survives.** `double→float` was attempted **102×** on template
  regions (impossible in Wave-1: Run B had 0 float) and **accepted 86× (84%)** — above
  ff's 71% baseline. **Zero tolerance rejects** anywhere in the run: float survives
  tol=7 in **100% of cases where a patch was generated**. The residual gap is entirely
  a Patcher *generation* problem, not a numerical one (`dd_ceiling = 0`).
- **dd_untested: 47 → 40** (rate ~48% → ~36% per honestly-attempted dd line). Material,
  but **not** from the `_ieps50`/R4 group the ticket named — those bin-instances still
  fail at the dd generation rung (see Wave-2 §). The Wave-2 constant fix is correct but
  **unreached** on the specific ill-conditioned instances.

## Run

| | Wave2 faithful (this run) |
|---|---|
| run_id | `20260719_185132_033cbe69` |
| caps (corr/speedup) | **200 / 250** (speedup effective 379 w/ phase-1 spill) |
| dr_k | **60** (baked default, `agents/config.py:74`) |
| terminal status | **success** (both phases drained) |
| total iters | 229 (corr 111 + speedup 118) · budget-iters 159 |
| wall | 101 min (6076.83 s) · tokens 1.25M |
| reached speedup? | **yes**, drained (88/379 budget) |
| final precision dist | float **86** · ff **2** · dd **134** · double 266 (488 total) |

## Per-phase metrics (A / B / this run)

### Phase 1 — correctness (`double→dd`)

| metric | Run A (300/200, dr20) | Run B (probe 40, dr40) | **this (200/250, dr60)** |
|---|---|---|---|
| total iters | 128 | 63 | **111** |
| budget-iters (=accepts) | 80 | 40 (hit cap) | **71** |
| **distinct lines accepted** | 50 | ~30 | **71** |
| redundant re-promotions | **30** (B2m.h:64 ×21…) | — | **0** |
| llm_gen_failed | 48 | 23 | 40 |
| strategy_bug / empty / commit_failed | 0 / 0 / 0 | 0 / 0 / 0 | **0 / 0 / 0** |
| dd_untested | 47 | (partial) | **40** |
| terminal | partial @128 (dr) | budget_exhausted @40 | **drained (71/200)** |

`iters_per_accept` = 1.0 (budget); `llm_gen_failed` doesn't consume budget (P6). The
signal is **71 accepts across 71 distinct lines** — zero chain re-drive waste, vs Run A's
80/50 (30 wasted).

### Phase 2 — speedup

| metric | Run B (ff-first) | **this run (float-first)** |
|---|---|---|
| speedup queue size | 113 | **113** (identical — deterministic from report+tol) |
| total iters | 193 | **118** |
| `double→float` | **0 accept** (`ff→float` plain-edit, 80 fail: no bare `double`) | **102 att → 86 accept (84%)** |
| `double→ff` | 113 att → 80 accept (71%) | 16 att → 2 accept (fallback only) |
| **tolerance rejects** | 0 (float never generated) | **0** (float always survives when generated) |
| lines → float (final) | **0** | **86** |
| lines → ff (final) | 80 | 2 |
| iters / candidate | ~1.7 (ff + failed float) | **~1.16** (float one-shot) |
| terminal | success (80/200) | **success (88/379)** |

## Wave 1 validation

- **Chain-representative dedup (WI1) — CONFIRMED at scale.** Telemetry
  `regions_chain_dedup_skipped = 527,523` (≈ the full 10k chain population). Correctness
  produced **71 accepts / 71 distinct** (0 redundant), vs Run A's 80/50. The redundant
  ~30 chain re-promotions that consumed ~40% of Run A's correctness budget are **gone**.
- **Faithful config drains — the primary spec question is YES.** dr_k=60 held through the
  chain-phase `llm_gen_failed` streaks; correctness used only 71/200 and **completed**
  (Run A tripped `partial` @128 at dr_k=20). The 300-cap that Run A needed is obsolete.
- **Build-fuse — CONFIRMED.** ~38 s/budget-iter (6077 s / 159) vs Run A's ~58 s/budget-iter
  (4680 s / 80) — **~35% faster per productive iter**. (Not a clean 2× because the
  LLM-gen portion isn't fused; the build-bound portion roughly halved.)
- **Template-rung skip (WI3a/WI3b) — CONFIRMED.** `strategy_bug = 0` in **both** phases.
  Run B logged 80 `ff→float` no-ops mis-tagged `strategy_bug`; those are gone — the float
  integrator now owns template regions, so no plain-edit `-to-float` rung noise.

## Wave 2 validation

- **Reachable `double→float` (WI1/float integrator) — CONFIRMED.** 102 attempts, **86
  accepts (84%)**, all direct `double→float` on template-typed regions. This capability
  did not exist in Wave 1 (Run B: 0 float). 86 regions ended at float.
- **Float survival at tol=7 — the new scientific answer: YES.** Float accept rate 84% >
  ff 71%. **Every** float miss (16) is `llm_gen_failed` (generation), **not** a tolerance
  reject. Genuine/tolerance rejects across the whole run = **0**; `dd_ceiling = 0`. The
  binding constraint on precision coverage is Patcher *generation robustness*, not
  numerical precision.
- **ff-fallback (cheapest-first) — CONFIRMED.** For the 16 regions where float failed to
  generate, ff was attempted (16/16 landed exactly on the float-failed regions), and **2
  accepted** (kokkosUtils.h:270, B4m.h:126) → ff-demoted. The other 14 also
  `llm_gen_failed` as ff — i.e. those regions are region-intrinsically hard to generate
  (not float-specific, not transient: an easier transform issued moments later also
  failed). The wiring is correct and no Wave-1 win regressed (fallback caught what float
  missed, one rung shy).
- **`_ieps50`/R4 group — fix is correct but UNREACHED on the named instances.** Regions
  are **bin-scoped** (same source line runs per-bin as distinct regions). The Wave-2
  constant-derivation fix works **where the Patcher generates**: e.g. a non-BIN0
  instance of B0m.h:69 promoted to dd (iter 77, accept). But the specific ill-conditioned
  instances the ticket named — **BIN0 B0m.h:68/69, BIN1 B1m.h:62/63** — still fail at the
  `double→dd` **generation** rung (`llm_gen_failed`, P6a), *upstream* of where the
  constant fix operates, so the fix never runs there. B0m.h:68 / B1m.h:62 / B1m.h:63 fail
  **all** attempts; only B0m.h:69 has one passing bin-instance.
- **dd_untested delta: 47 → 40** (rate ~48% → ~36%). Material, but the improvement is from
  *other* regions + more distinct honest dd attempts (chain-dedup), **not** the `_ieps50`
  group. All 40 residual dd_untested are `llm_gen_failed` at the dd rung (P6a); **0** are
  physics ceilings.

## Surprises / not covered by expected outcomes

1. **`_ieps50` did not validate** (spec expected it would). Root cause: the blocker for
   the named bin-instances is Patcher *generation* (P6a), not constant derivation. The
   Wave-2 fix addressed a downstream step; the upstream generation gate still binds.
   → This is the single biggest deviation from the spec's expected outcome.
2. **Zero tolerance rejects, run-wide.** Neither dd nor float nor ff ever lost to the
   validator's 7-digit floor. The *entire* remaining opportunity (40 dd_untested + 16
   float gen-misses + 14 ff gen-misses) is a **generation-robustness** problem. Float is
   not precision-limited on this codebase at tol=7.
3. **Float is both cheaper and higher-yield than ff** (84% accept, ~1.16 iters/candidate
   vs Run B's ~1.7). Float-first ordering makes speedup materially lighter than the
   250-cap sizing (built on the ff-first assumption) anticipated.
4. **Generation-hard regions cluster as real/imag component pairs** (B1m.h:97/98,
   B2m.h:105/106, B2m.h:143/144) — a distinct cluster from the `_ieps50` dd group, and a
   concrete Wave-3 target.

## 50k budget recommendation — UNCHANGED (200 / 250 / 60 / 12h)

This run's numbers support the CALIBRATION.md §50k recommendation; keep it.

| knob | 50k value | basis (this run) |
|---|---|---|
| `max_iters_correctness` | **200** | 71 budget-iters for a 76-region corr queue drained cleanly; 50k corr queue ~191 → ~178 budget-iters projected, under 200 (margin thinning — bump to 220 only if 50k corr queue exceeds ~200). |
| `max_iters_speedup` | **250** | 88 budget-iters for a 113 queue, float-first (~1.16 iters/candidate — *cheaper* than the ff-first basis 250 was set on). 50k speedup queue ~200–300 → ~230 budget-iters at 84% float accept, under 250 (raise to 300 only if 50k speedup queue exceeds ~280). |
| `diminishing_returns_k` | **60** | held; correctness drained with no dr trip. |
| `max_wall_hours` | **12** | 10k = 1.7 h; 50k projects ~5–6 h at ~38 s/budget-iter (build-fuse). Ample margin. |

**Net:** the float rung does **not** blow up the speedup queue (float-first is cheaper
per candidate than assumed), so no cap resize is warranted. The one thing 50k should
watch is the correctness cap headroom (178/200 projected).

**Wave-3 pointer (not this session):** the ceiling is Patcher generation on
ill-conditioned bin-instances (`_ieps50`) and real/imag component-pair lines. Recommend
(a) retry-with-backoff on `llm_gen_failed` to clear any transient tail, (b) float/dd
prompt hardening for complex-component lines, then re-measure — that is the honest test
of "would float survive if we could generate it" (this run says: probably, but the
gen-failed set is enriched for hard regions, so don't assume the 84% rate carries).

## Repro

Merge + run: `git checkout langgraph-agents` (@ `63e8986`) → offline suite `349 passed`
→ `tmux new-session -d -s wave12-10k 'bash runs/qcdloop/run_strategy_10k.sh 2>&1 | tee run_loop_wave12.log'`.
Metrics: `.venv/bin/python runs/qcdloop/analyze_calibration.py --run-dir runs/qcdloop/strategy/20260719_185132_033cbe69 --report runs/qcdloop/report_10k.json`.
Per-iteration detail: `runs/qcdloop/strategy/20260719_185132_033cbe69/iterations.jsonl`.
