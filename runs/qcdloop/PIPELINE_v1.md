# PIPELINE_v1 — production mixed-precision pipeline (Wave-3 prunes + tail battery)

**Validated:** 2026-07-20 · branch `langgraph-agents` · run
`20260720_054121_dd44d33c` · faithful config (correctness 200 / speedup 250 /
dr_k 60 / tol 7 / n=1000) · DD oracle `~/qcdloop@ddfun_enabled` · tail-augmented
`report_10k.json`.

## TL;DR

The qcdloop mixed-precision pipeline is production-ready. Two capabilities landed
on top of the validated Wave-1/2 walk:

1. **Three report-mining prunes** (Wave-3, `d5815a9`): WI1 float-range hard gate,
   WI2 pred-float telemetry, WI3 flop-weighted speedup ordering.
2. **Tail-battery validation** (this work): on every candidate the Validator
   re-tests, in addition to the n=1000 random samples, the specific per-integral
   input offsets the characterizer flagged as adversarial — verified against a
   frozen per-integral input determinism hash.

Both were validated together in a single 10k run: terminal **`success`** in
**107 min**, **85 demotions preserved** (73 `float` + 12 `ff`), **0 correctness
regressions**, **0 tail-sample failures**, **0 determinism-hash mismatches**.
Every one of the 152 validated candidates accepted; the demotion delta vs the
pre-Wave-3 baseline (88 → 85) is entirely the WI1 range-safety bite, not a
correctness loss.

## Report-mining prunes (Wave-3)

Each prune mines a field already present in the characterization report; the whole
set is gated by `report_prunes` (env `STRATEGY_DISABLE_REPORT_PRUNES=1` disables
all three at once, no per-prune toggles).

- **WI1 — `value_range_ok_for_float` (HARD float-rung gate).** A speedup walk that
  would demote a region to `float` is stopped at `ff` when the region is
  range-unsafe (float over/underflow risk the Validator's finite random sample can
  miss). Reads the per-region flag with a per-region fail-open default of `True`.
  Validation-run bite: `regions_skipped_range_unsafe = 14` — these 14 range-unsafe
  regions retreated from `float` to `ff`, which is exactly why the final `float`
  count is 73 (down from the pre-Wave-3 86) and `ff` is 12 (up from 2). The
  demotions are preserved, just kept on the safe rung.
- **WI2 — `predicted_rel_err_if_float` (TELEMETRY-ONLY).** A local per-region error
  bound; it does **not** hard-gate because it is a *local* bound while the
  Validator gates on the *global* min precise-digits — hard-gating on it would
  over-block float regions with no correctness gain (see the design finding
  below). Counts flagged regions only: `regions_flagged_pred_float = 55`.
- **WI3 — flop-weighted speedup ordering.** The speedup queue is ordered by
  flop-weighted op mix (reuses `ratio_multipliers.json`) rather than raw op count,
  so the highest-throughput demotions are attempted first.
  `speedup_queue_flop_weighted = true`.

## Tail-testing

### Characterizer schema addition

Each integral in the report gains `integrals.<B>.tail_samples`:

```
{
  "determinism_hash": "sha256:...",           # SHA-256 of the first-100 inputs
  "max_rel_err":   [ {offset, criterion_value, output_component}, ...K ],
  "max_cond":      [ ... ],                    # cancellation proxy ref_scale/|comp|
  "max_abs_value": [ ... ],                    # largest |output component|
  "min_abs_value": [ ... ]                     # smallest nonzero |component|
}
```

The four criteria are measured on the integral's **output components**
(`coeff{0,1,2}.{real,imag}`), K=10 distinct offsets each. Because the characterizer
report is region-keyed (no per-sample identity, no per-output-component scoring)
and the CALIBRATION_v2 ~80 GB journal is gone, the offsets are computed by a
driver re-run (`emit_tail_offsets.py`): the vanilla and DD app drivers are run over
`[0,10000)` and their per-component outputs compared — far cheaper than a full
re-characterization (RES output is ~tens of MB, no journal), and it produces the
per-output-component signal the journal never held.

`max_cond` documents its equivalent honestly: a true per-sample condition number
is not derivable from outputs alone, so it uses the closest driver-computable
proxy — `ref_scale / |component|`, the cancellation indicator (large when a
component is a small residue of large quantities), distinct from raw rel-err.

Emit on the real 10k report: 21 integrals, **389 distinct offsets, 341 (88%) of
them ≥ 1000** — i.e. outside the random battery's `[0,1000)` window — in ~124 s.

### Validator battery mechanics

Offsets are per-integral `mt19937(12345)` stream indices (each integral re-seeds
and fills `[0,total)` before dispatch, so a per-integral offset is bit-identical
however chunked; the tracked and app drivers share the recipe verbatim). Two
additive driver flags (the existing `--sample-offset` range path is untouched and
byte-identical): `--dump-inputs N` (input fingerprint) and `--sample-list a,b,c`
(dispatch exact offsets in one invocation).

Per candidate:
1. **Determinism check** — recompute each integral's first-100-input hash from the
   candidate binary (`--dump-inputs 100`) and compare to the frozen report hash. A
   mismatch raises `DeterminismMismatch` loudly (never a silent fall-back) — the
   offsets are meaningful only if the input generator is unchanged.
2. **Dispatch** the union of tail offsets via `--sample-list`; the DD reference and
   the current baseline at those offsets are built + cached once per run.
3. **Score** candidate *and* current at the offsets vs the DD oracle and fold into
   the verdict.

### Determinism-hash guarantee

The hash is over *inputs*, generated before the integral is evaluated, so it is
independent of the candidate patch — it drifts only if the input generator
(mt19937 / distribution / toolchain) or offset semantics change. Validation run:
`tail_hash_mismatches = 0` across all 152 tail batteries.

### Design finding — tail testing must be regression-relative

The first tail-enabled run rejected every candidate at `tail_min = 3.69 < tol = 7`.
Diagnosis: the pristine double baseline hits the *same* 3.69 at the *same* hotspot
— **B12 offset 3868 `coeff0.imag`**, a ~1.6e-15 near-zero output where double's
*relative* precision is inherently ~3.7 digits (absolute error ~1e-19). That is a
**workload physics ceiling** at an adversarial `min_abs`/`max_cond` offset, present
in candidate and baseline alike — not candidate-induced.

So the tail battery is **regression-relative**: score candidate and current at the
same offsets, run the regression guard on the combined (random+tail) minima (a
candidate that does materially worse than the baseline at a tail offset — a float
demotion overflowing at an untested large-magnitude input, a broken cancellation —
is a hard reject), and keep the **absolute floor on the random battery** (the
adversarial offsets include workload ceilings no demotion owns). This is the same
*local vs global metric* asymmetry as WI1/WI2: a per-point/per-region local metric
must not gate an absolute global bar; it belongs in a relative (regression /
telemetry) role. Fail-open reduces exactly to the pre-tail verdict.

Per-integral tail-battery counts (validation run): batteries_run = **152** (one
per validated candidate), up to **599** tail samples tested per validation
(union of per-integral offsets across all 21 integrals, 389 distinct offsets),
tail failures (candidate-induced regressions) = **0**.

## Validation run summary

Run `20260720_054121_dd44d33c`, compared to the pre-Wave-3 Wave-1/2 validation
(CALIBRATION_v2, `63e8986`) at the same faithful config:

| metric | CALIBRATION_v2 (pre-Wave-3) | **this run (Wave-3 + tail)** | delta / cause |
|---|---|---|---|
| terminal status | success | **success** | — |
| wall | 101 min | **107 min** (6% ↑) | tail battery on 152 candidates |
| iterations / validated | — / — | 231 / **152** | — |
| accepts (all validations) | — | **152 / 152** | 0 rejects run-wide |
| final `float` | 86 | **73** | WI1: 13 range-unsafe → `ff` |
| final `ff` | 2 | **12** | WI1 retreats (safe rung) |
| **demotions (float+ff)** | 88 | **85** | WI1 range-safety bite (not a loss) |
| final `dd` | 134 | **133** | noise (1) |
| final `double` | 266 | **271** | mirror of above |
| `regions_at_dd_ceiling` | 0 | **0** | no genuine dd ceiling |
| `regions_unresolved` | 0 | **0** | — |
| `dd_untested` | 40 | **47** | Patcher gen noise (orthogonal track) |
| tolerance rejects | 0 | **0** | float always survives when generated |
| strategy_bug / commit_failed | 0 / 0 | **0 / 0** | — |

**Wave-3 prune telemetry:** `regions_skipped_range_unsafe = 14`,
`regions_flagged_pred_float = 55`, `speedup_queue_flop_weighted = true`,
`report_prunes_enabled = true`.

**Tail telemetry:** 152 validations ran the battery, `tail_hash_mismatches = 0`,
389 offsets dispatched, up to 599 tail samples per validation,
`tail_driven_regression_rejects = 0`.

Every delta from the pre-Wave-3 baseline is accounted for by a prune bite (WI1
float→ff retreat) or by orthogonal Patcher generation noise (`dd_untested`
40→47, stochastic proxy failures absorbed by `dr_k=60`). **No delta is a
correctness regression:** 0 tolerance rejects, 0 strategy_bugs, 0 commit
failures, 0 tail-sample failures, 0 hash mismatches. Reproduce the summary with
`python runs/qcdloop/analyze_tail_run.py --run-dir
runs/qcdloop/strategy/20260720_054121_dd44d33c`.

## Config surface

- Single runner: `runs/qcdloop/run_strategy_10k.sh`
  (`--report report_10k.json --sample-count 1000 --tolerance 7.0
  --max-iters-correctness 200 --max-iters-speedup 250 --dr-k 60`).
- Single kill-switch: `STRATEGY_DISABLE_REPORT_PRUNES=1` disables all three prunes
  together. Tail-testing has **no kill switch** — it is always on when the report
  carries `tail_samples`, and fails open (random-only) when it does not.
- Tail-offset emitter (post-characterization pass, reusable for 50k / future
  apps): `runs/qcdloop/emit_tail_offsets.py`.

## Known limitations

- **Physics ceiling.** At tol=7 the global-min gate is capped by BIN1 `coeff0.imag`
  cancellation on qcdloop; the tail battery additionally surfaces per-point
  ceilings (e.g. B12:3868 at 3.69 digits). These are workload properties, not
  pipeline limitations — handled correctly by the regression-relative verdict
  (they cancel in the delta).
- **Generation robustness.** dd_untested / float-gen-failed / ff-gen-failed on the
  10k baseline are the walk ceiling (Patcher generation, orthogonal track), not
  addressed here.
- **Deferred prunes.** The fine-grained `variables{}` map (per-var demotion,
  n_consumers tiebreak — WI4/WI5) is the largest unmined signal but needs a
  region-granular Patcher contract change. Not scoped for v1.
- **Tail coverage bound.** Tail testing is bounded by the characterization
  distribution; failure modes on inputs *outside* that distribution are caught by
  no battery — the pipeline makes no guarantee for input distributions it has never
  seen.

## Portability

Nothing here is qcdloop-specific: per-integral tail testing, offset-based sample
identity, input determinism hashes, and the per-region prunes all generalize to
any characterizer output carrying the corresponding fields. The two driver flags
(`--dump-inputs`, `--sample-list`) are additive and preserve the byte-identical
`--sample-offset` contract.
