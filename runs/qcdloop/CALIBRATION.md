# 10k Strategy Calibration

Sizing data for the eventual 50k Strategy run. First real exercise of the
two-phase walk (correctness → speedup) on live 10k characterization data, with
all post-2026-07-17 fixes in (`a29477d` shim-include order, `5abaa11` speedup
gate, `2babbb8` two-phase split, `e1d774a` Gap A/B, `f6fbf46` include-set lint)
plus two bugs fixed *during* this calibration (see §Bugs).

**Date:** 2026-07-19 · **tolerance:** 7.0 · **model:** claudeopus47 · **validator:** real 3-build, seed 12345, n=1000, DD oracle `~/qcdloop@ddfun_enabled`

## TL;DR

- The shim-collision **fatal abort was fixed** (§Bug 1); the fix held — no
  `commit_failed`/`internal_error` in either run below.
- **Accept rate moved up** from the shakedown's 42% to ~62% (correctness), and the
  **dd_untested rate dropped** ~58% → ~48% (per dd-attempt): the shim hardening
  worked for the include-hallucination group. The residual dd_untested is
  dominated by the known `_ieps50`/Rule-R4 group (B0m.h:69, B1m.h:62, …).
- **Speedup phase produced first-ever data:** stable regions demote **double→ff**
  (80 wins), never to float (template-typed code has no literal `double` token for
  the plain-edit path).
- **The cascade-chain phase is the dominant 50k risk:** 527,558 chains at 10k
  (~2.6M projected at 50k), and the walk re-drives `double-to-dd` on each chain's
  *representative line* — the same lines are hammered dozens of times
  (B2m.h:64 accepted **21×**), which both wastes ~40% of correctness budget and
  trips diminishing-returns before speedup. **Fix chain-representative dedup before
  50k** (see §50k recommendation).

## Runs

Two runs were needed because the faithful 300/200 config never reaches speedup
(it stops `partial` in the chain phase). Run A is the faithful config (the primary
correctness result + the 300/200 pathology); Run B is a supplementary probe with
the correctness budget cap lowered so the walk hands off to speedup and yields
first speedup data.

| | Run A — faithful | Run B — speedup probe |
|---|---|---|
| run_id | `20260719_062642_85d7059b` | `20260719_075229_d8bedbcc` |
| caps (corr/speedup) | **300 / 200** | 40 / 200 |
| dr_k | 20 (default) | 40 |
| terminal status | **partial** (diminishing-returns) | **success** (drained) |
| total iters | 128 | 256 (63 corr + 193 speedup) |
| wall | 78 min | 100 min |
| reached speedup? | **no** | **yes** |

## Per-phase metrics

### Phase 1 — correctness (`double-to-dd`)

| metric | Run A (300/200) | Run B (probe, cap 40) | shakedown (ref) |
|---|---|---|---|
| total iterations | 128 | 63 | 19 |
| budget-iters used | 80 | 40 (hit cap) | 8 |
| accepts (events) | 80 | 40 | 8 |
| distinct lines accepted | **50** | ~30 | 8 |
| llm_gen_failed | 48 | 23 | 11 |
| empty_candidate | 0 | 0 | — |
| commit_failed / fatal | **0** | **0** | 0 |
| lines → dd (final) | 51 | 39 | 7 |
| dd_untested regions | 47 | (partial) | 11 |
| **iters/accept** (budget) | 1.0 | 1.0 | 1.0 |
| **iters/accept** (total) | 1.6 | 1.58 | 2.4 |
| **iters/region-attempted** | 128/76 = 1.68 | 63/63 = 1.0 | — |
| terminal | partial @128 (dr) | budget_exhausted @40 (soft handoff) | budget_exhausted @8 |

Notes:
- `llm_gen_failed` does **not** consume budget (P6), so budget-iters == accepts
  (iters/accept = 1.0 is structural, not a quality signal). The meaningful ratio
  is **total-iters/accept ≈ 1.6** and the **distinct-line accept rate**.
- Run A's 80 accepts cover only **50 distinct lines** — 30 are redundant chain
  re-promotions of already-dd lines (B2m.h:64 ×21, B2m.h:65 ×11). This is the
  chain-representative redundancy (§Bug/finding 3).

### Phase 2 — speedup (first-ever real data, Run B)

| metric | value |
|---|---|
| total iterations | 193 |
| budget-iters used | 80 (of 200) → **drained**, not capped |
| speedup queue size | 113 |
| **`double-to-ff`** (regional / LLM path) | 113 attempts → **80 accept**, 33 llm_gen_failed → **71% accept** |
| **`ff-to-float`** (revert + plain-edit path) | 80 attempts → **0 accept** (all `patch_apply_failed`: "no bare `double` token" — template-typed) |
| **`double-to-float`** direct (plain-edit) | never fires (walk reaches float only via ff-to-float, which fails) |
| lines → ff (final) | **80** |
| lines → float (final) | **0** |
| iters/accept (total) | 193/80 = 2.4 (each region = 1 ff attempt + 1 failed float attempt) |
| terminal | success (queue drained; used 80/200 budget) |

**Plain-edit vs LLM-path:** the speedup win is entirely the **LLM/regional
`double-to-ff`** path. The **plain-edit path (`-to-float`) is inapplicable** to
these template-typed HPC regions — there is no literal `double` token to rewrite —
so float is never reachable at tol=7 and the walk correctly settles at **ff**. The
80 `ff-to-float` failures are cheap (git-only, no build; they fail before the build
gate) but are mis-tagged `strategy_bug` in the log (cosmetic; see §Bug/finding 4).

### Phase-2 skip rate

`skipped_dd_promoted = 39`, but these are correctness-phase dd promotions
(local_cancellation / log_near_root regions), which are a **disjoint population**
from the 113 stable speedup candidates. **~0 candidates were actually dropped**
from the speedup set. (The analyzer's naive `skip_rate=34.5%` divides the two
disjoint counts and is misleading — real drop ≈ 0.)

## Accept-rate & dd_untested deltas vs shakedown

- **Accept rate:** shakedown 8/19 = **42%** → Run A correctness **62%** (80/128),
  or **66%** by distinct line (50/76). The three shim fixes moved the needle up.
- **dd_untested rate (per dd-attempt):** shakedown 11/19 = **58%** → Run A
  47/(47+51) = **48%**. Modest improvement. Absolute dd_untested is higher (47 vs
  11) only because Run A attempted all 76 regions vs the shakedown's ~8.
- Residual dd_untested is the known **`_ieps50` / Rule-R4 group** (B0m.h:68/69,
  B1m.h:62/63, …) that needs the un-vendored `1e-50` hex constant — a separate,
  already-identified blocker, not regressed by this work.

## Bugs / design issues surfaced

1. **[FIXED] Cross-region shim collision → fatal abort.** Regional shim filenames
   keyed on the content cache-key only (`{file}_{dd|ff}_{hash8}.h`), so two
   same-file regions with identical source collided on one filename; a later
   region cache-hit the earlier shim, netted no tree change, and `git commit`
   failed "nothing to commit" → escalated to fatal `internal_error`, killing the
   whole run at iter 26. Fixed by scoping the shim filename by line range
   (`20879dc`) and, as defense-in-depth, a non-fatal `empty_candidate` status so
   any empty commit advances the walk instead of aborting (`ba544a9`). Also fixed
   a latent clobber (one region overwriting another's accepted shim).
2. **[design, HIGH — fix before 50k] Cascade-chain representative hammering.**
   `ChainRecord.walk_record()` drives on `lines[0]`; with 527k chains sharing
   representatives, the same line is re-driven dozens of times — 30 of 80 Run-A
   accepts were redundant (B2m.h:64 ×21 = ~63 wasted builds), and 20 consecutive
   `llm_gen_failed` on B0m.h:69/B1m.h:62 tripped `dr_k=20` → `partial` before
   speedup. At 50k (~2.6M chains) this is the dominant cost. **Recommend: the
   chain phase should skip lines already at/above the target precision and
   deduplicate chains by representative line.**
3. **[design, LOW] `dr_k=20` too low for the chain phase.** Chain `llm_gen_failed`
   don't consume budget but do increment the DR streak, so unpromotable lines trip
   DR long before the correctness *budget* cap can bind. Raise `dr_k` (or exempt
   chain-representative re-attempts from the streak).
4. **[cosmetic] `ff-to-float` mis-tagged `strategy_bug`.** The plain-edit float
   demotion failing on template code is expected, not a strategy bug; consider
   skipping the `-to-float` rung for template-typed regions to cut 80 no-op
   iterations and clean the logs.
5. **[perf, note] Redundant patcher-gate + validator-candidate double-build.** Each
   accept builds the monolithic app TU twice (patcher gate, then validator
   candidate) — the app is one translation unit (~25–40 s/build, can't
   parallelize). Reusing the gate binary in the validator would ~halve accept
   cost. Not blocking, but material at 50k.

## Extrapolated 50k budget recommendation

50k queues (per ticket): ~191 non-stable (correctness) + ~200–300 stable speedup
candidates; chains ≈ 2.6M. Measured ratios: correctness ~66% of regions promote
(1 budget-iter each); speedup ~71% of candidates demote to ff (1 budget-iter each).

**Precondition:** fix the chain-representative dedup (§Bug 2). Without it the
correctness phase cannot cleanly drain and speedup is only reachable via the
low-cap trick used in Run B.

Assuming the dedup fix lands:

| knob | recommended 50k value | basis |
|---|---|---|
| `max_iters_correctness` | **200** | ~126 distinct dd promotions expected (66% of 191) + rejects/margin |
| `max_iters_speedup` | **250** | ~180 ff demotions expected (71% of ~250) + margin to drain |
| `diminishing_returns_k` | **60** | chain phase produces long non-accept streaks; 20 is too tight |
| `max_wall_hours` | **12** | ~25–37 s/iter build-bound; ~1000–1400 total iters projected |

If the dedup fix does **not** land before 50k: expect `partial` in correctness;
cap `max_iters_correctness` ≈ 150, set `dr_k` ≥ 80, and run a **separate**
speedup pass with a low correctness cap (as in Run B) to get speedup coverage.

## How to reproduce

Environment quirks:
- Use the repo `.venv` python (`/home/rbarik/Agentic-Mixed-Precision-Demo/.venv`).
- Argo proxy runs on **:8084** (not the config default :8083):
  `ANTHROPIC_BASE_URL=http://127.0.0.1:8084/argoapi/`, `ANTHROPIC_AUTH_TOKEN=rbarik`.
- Long runs go in a **detached tmux session** so they survive the driving
  shell/session dying: `tmux new-session -d -s <name> <script>`; watch with
  `tmux attach -t <name>`; completion is marked `__STRATEGY_EXIT__` in the tee'd log.

Steps:
1. **Characterize (10k):**
   `.venv/bin/python runs/qcdloop/run_chunked.py --total 10000 --chunk 500 --workers 16 --out runs/qcdloop/report_10k.json`
   (~18 min wall, ~80 GB peak journal on /tmp, emits region_local_vars /
   cascade_chains / predicted_rel_err_if_{ff,float} natively). Do **not**
   re-characterize to reproduce the walk — the report is fixed input.
2. **Faithful walk (Run A):** `runs/qcdloop/run_strategy_10k.sh` (caps 300/200,
   dr_k=20). Expect `partial` in correctness.
3. **Speedup probe (Run B):** `runs/qcdloop/run_strategy_10k_speedup.sh`
   (corr cap 40, dr_k 40) → reaches speedup, `success`.
4. **Metrics:** `.venv/bin/python runs/qcdloop/analyze_calibration.py --run-dir runs/qcdloop/strategy/<run_id> --report runs/qcdloop/report_10k.json`

Config for each walk is logged to the tee'd `run_loop*.log`; per-iteration detail
is in `runs/qcdloop/strategy/<run_id>/iterations.jsonl`.
