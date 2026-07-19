# Float Retro Probe — precision headroom of the Wave-1+2 10k float acceptances

**Date:** 2026-07-19 · **Branch:** `langgraph-agents` @ `be4dac9`
**Source run:** `20260719_185132_033cbe69` (10k re-run, CALIBRATION_v2, walk @ `3fd5ad3`)
**Method:** validator-only replay — no walk, no LLM, no re-characterization.

---

## TL;DR

| tolerance | survivors / 86 |
|-----------|:--------------:|
| tol=7 (source bar) | **86 / 86** |
| tol=8 | **86 / 86** |
| tol=9 | **0 / 86** |
| tol=10 | **0 / 86** |
| tol=11 | **0 / 86** |

**One-line interpretation:** float acceptance survival is a *cliff*, not a curve —
every one of the 86 float regions carries `candidate_min_precise_digits = 8.8399`
(identical to 4 dp), so all 86 clear tol=8 and all 86 fail tol=9. That 8.8399 is
**not float's headroom** — it is the double-precision floor set by the
ill-conditioned `BIN1 coeff0.imag` cancellation, an integral component that *no*
float region touches. Tightening the *global* validator bar past ~8.84 digits
measures the BIN conditioning wall (which limits plain `double` too), not float.
The current 84% float acceptance therefore has **~0.84 digits of margin against
tol=8 and zero against tol=9**, and that margin is a property of the double floor,
not of float.

**Recommendation: do NOT run a tighter-tolerance 10k walk.** tol=8 is free (all
current accepts already clear it, so it changes nothing); tol≥9 rejects every
candidate — float, ff, *and plain double* — because it sits above the 8.84-digit
conditioning floor. See [Recommendation](#recommendation).

---

## Method (and why 86 builds, not 344)

For each of the 86 float-accepted regions
(`kind=double-to-float`, `accepted=true` in `iterations.jsonl`; count cross-checked
against CALIBRATION_v2's "lines → float (final) = 86" ✓) the probe reconstructs the
*exact* cumulative candidate patch the walk validated —
`git diff <starting_sha>..<candidate_sha>` in the run's headers repo
`~/amp_strategy_headers_repo` (all 86 candidate commits still present; base
commit `93f57725` "base: qcdloop_headers_full snapshot") — and calls the same
`agents.validator.validate()` with the same `base_state` / snapshot the walk used
(seed 12345, n=1000, DD oracle `~/qcdloop@ddfun_enabled`, 3-build discipline,
`max_regression=0.5`).

**Cost collapse.** In `validate()`, tolerance enters *only* at the final gate —
`_decide(cand_min, curr_min, max_regression, floor=tolerance)`. It never touches
the DD/current/candidate builds nor the precise-digit scoring. So **one** build
per region yields the region's `cand_min`; survival at every tolerance is then a
pure threshold on that single number. This collapses the naive 86 × 4 = 344
validator invocations to **86 builds**, and makes monotonicity *structural*
(a downward threshold on a fixed number cannot be non-monotone) rather than merely
observed. The shared DD oracle and current baseline were already cached
(`seed12345_n1000`), so each region paid only its candidate build+run (~9.2 s;
total wall **804 s ≈ 13.4 min**).

Probe script: `runs/qcdloop/run_retro_probe.sh` → `runs/qcdloop/retro_probe.py`.
Per-region results: `runs/qcdloop/float_retro_probe_results.json`.

---

## Per-tolerance breakdown

All 86 regions are **identical** at the validator's reporting resolution:

- `cand_min_precise_digits` = **8.8399** for all 86 (distinct values: `{8.8399}`).
- `delta` (cand − current) = **0.0000** for all 86 — no float region moves the
  global minimum off the double baseline.
- global-min hotspot = **`BIN1 / coeff0.imag`** for all 86 (distinct hotspots:
  `{(BIN1, coeff0.imag)}`).

Consequently the "regions that fall out at each tolerance step" list is trivial:

| step | regions that first fail here |
|------|------------------------------|
| tol=8 | none (all 86 clear it: 8.8399 ≥ 8) |
| **tol=9** | **all 86** (8.8399 < 9) |
| tol=10 | — (already out at 9) |
| tol=11 | — (already out at 9) |

**Determinism cross-check:** every replayed `cand_min` equals the
`candidate_min_precise_digits` the walk recorded, within 1e-4 → **0 / 86
mismatches**. The validator is bit-reproducible on this workload.

---

## The 5–10 "weakest" float acceptances

The empirical validator **cannot rank the 86** — they are all pinned at 8.8399 on
the *same* non-float hotspot (`BIN1 coeff0.imag`), with delta exactly 0. At the
metric that gates acceptance, no float acceptance is measurably weaker than any
other. So "weakest" has to come from the **static** analyzer
(`report_10k.json`), which *does* score each region's own conditioning.

Of the 86 float locations, only **19 appear in the report's
`top_regions_by_rel_err` lists at all**; the other **67 are benign lines** with no
elevated-conditioning signal flagged (all of `B3m.h`, `B4m.h`, `kokkosUtils.h`,
and most of `B2m.h`). The weakest-on-paper (highest nominal condition number and
highest predicted float rel-err) among the 19:

| file:line | nominal max_cond | predicted rel-err if float | signal class | empirical Δdigits |
|-----------|:----------------:|:--------------------------:|--------------|:-----------------:|
| B0m.h:330 | 1.39e5 | 6.4e-2 | stable | **0.0** |
| B1m.h:106 | 7.11e4 | 6.7e-3 | stable | **0.0** |
| B0m.h:126 | 6.81e4 | 1.4e-1 | stable | **0.0** |
| boxGPU.h:99 | 2.72e4 | 2.1e-2 | stable | **0.0** |
| B0m.h:331 | 1.95e4 | 5.9e-2 | stable | **0.0** |
| B2m.h:151 | 1.47e4 | 1.0e-2 | stable | **0.0** |
| B0m.h:224 | 1.02e4 | 3.0e-2 | stable | **0.0** |
| B1m.h:133 | 6.16e3 | 1.2e-2 | stable | **0.0** |

**The striking part:** these are the regions the static conditioning model liked
*least* — `max_cond` up to 1.4e5 and predicted float rel-err up to **14%** — yet
every one cost **zero** global-min digits when actually built and run. The static
`predicted_rel_err_if_float` is a worst-case per-region amplification; empirically
the amplified error either cancels downstream, does not flow into the dominant
coefficient, or is dwarfed by the BIN cancellation that already caps the global min
at 8.84 digits. This is the direct answer to *"how much margin does the 84%
acceptance have"*: the on-paper-weakest float regions have **10%+ predicted
rel-err and ~1e5 conditioning, and still leave the accept metric untouched** — so
the acceptance margin is entirely the double floor's, and float's real cost on this
workload is below the 8.84-digit resolution of the global-min gate. There is no
sub-population of "risky" float acceptances that a tol=8 bar would have caught.

Note the clustering: the weakest-on-paper are concentrated in **`B0m.h`** (5 of
the top 8, incl. the three highest). If any future tighter-tol *per-component*
audit is done, `B0m.h` is where to point it — but the global-min gate does not see
a difference.

---

## The regions that survive to tol=11

**Count: 0.** No region survives past tol=8. There is no "float has real headroom
here" sub-population under the global-min metric — the plateau at 8.8399 means
every region is simultaneously (a) comfortably above tol=8 and (b) hard-blocked at
tol=9 by the double floor. The 67 benign lines (float on template/utility code
with `max_cond ≈ 1`, e.g. `kokkosUtils.h`, `B4m.h`) genuinely *do* have local
float headroom — the report predicts float rel-err ~6e-8 there — but that headroom
is invisible to a *global*-min tolerance because their contribution never sets the
global min. Their real headroom would only show up in a per-component, float-touched
audit, which is out of scope for this probe.

---

## Monotonicity

**0 violations across 86 regions × 4 tolerances.** As noted in [Method](#method-and-why-86-builds-not-344),
this is structural: survival at tol T is `cand_min ≥ T` on a single, tolerance-independent
`cand_min`, so a region passing a stricter tol while failing a looser one is
mathematically impossible. The probe still checks it explicitly against the derived
per-tol verdicts (and the underlying `cand_min` was independently reproduced, not
reused from the walk's records — see the determinism cross-check), and finds none.
No validator bug or workload-nondeterminism signal.

---

## Recommendation

**No tighter-tolerance 10k walk.** The task's rough rule ("if >50 survive at tol=8,
tol=8 is worth a walk") is nominally satisfied (86 survive) — but the *mechanism*
overrides the rule:

- **tol=8 is a no-op.** All 86 current accepts already clear 8.8399 ≥ 8, so a
  tol=8 walk re-accepts exactly the same set. Nothing to learn.
- **tol≥9 is scientifically unproductive on this codebase.** 8.8399 is the double
  floor of the `BIN1 coeff0.imag` cancellation cascade — a wall that constrains
  plain `double` itself, independent of any mixed-precision choice. A tol=9 walk
  would reject *every* candidate (float backfills to ff, ff has nothing better to
  offer on a component it doesn't touch, and even the correctness DD promotions
  only lift the components they cover). This matches RATIO_REPORT.md's conclusion
  that the float flop-share swing is ~0.1–1%: float is a thin backfill layer whose
  precision behaviour is decoupled from the global accuracy bottleneck.

The productive next question is **not** "tighten the global bar" but "measure
float headroom per-component on the lines float actually touches" — a different
instrument (per-component, float-touched digit comparison) and a separate
decision. The global-min gate has told us all it can: at its resolution, all 86
float acceptances are precision-neutral.

---

## Artifacts

- `runs/qcdloop/retro_probe.py` — probe (region enumeration, patch reconstruction, replay, thresholding, checks)
- `runs/qcdloop/run_retro_probe.sh` — launcher (module chain + venv)
- `runs/qcdloop/retro_probe.log` — full run log (86 regions + survival table)
- `runs/qcdloop/float_retro_probe_results.json` — per-region cand/curr/delta/hotspot/verdict + summary
- `runs/qcdloop/float_retro_condnums.json` — static condition numbers for the 19 report-flagged float locations
- `runs/qcdloop/FLOAT_RETRO_PROBE_SUMMARY.txt` — grep-able survival table + recommendation
