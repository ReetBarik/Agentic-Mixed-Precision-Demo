# Report-field inventory — `report_10k.json` × Strategy usage

**Source:** `~/qcdloop-artifacts/report_10k.json` (1.7 GB, `schema_version:1`,
`kind:"stability_report"`, 20 integrals B1–B16 + BIN0–4, 10k samples each).
**Code baseline:** `langgraph-agents` @ `b1c4b2b`. Static analysis only — no walk,
no LLM, no builds.

**Method.** The report is a single-line 1.7 GB JSON dominated by per-instance
`variables{}` and `prov_vars[]`. Schema was recovered by brace-extracting the
first integral object (B1, 50 MB) for the full nested shape, then closing the
delta (populated `cascade_chains`, all `signal_class` values, the whole-file key
universe = **73 distinct keys**) with streamed greps. Consumption was traced
through `strategy/characterization.py` (the loader → `RegionRecord`/`ChainRecord`),
`ranking.py`, `walk.py`, `agent.py`, `models.py`, and confirmed absent from the
other consumers (`patcher/`, `float_integrator/`, `ff_integrator/`, `dd_integrator/`,
`integrator_base/`, `validator/`).

**Consumer vs generator.** `shared/stability_reducer.py` and `shared/fast_merge.py`
reference nearly every field, but they are the **characterizer reducer/merge that
*writes* the report** — not Strategy. Per the task's rule, generator-only touches
and telemetry-only reads count as **IGNORED** for pruning.

---

## ⚠ Drift / assumption flags (surface first)

- **`condition_numbers.mul` does not exist.** The task brief assumed a nested
  `...condition_numbers.mul` path. **No `condition_numbers` key appears anywhere**
  in the report, and no code references it. Conditioning is carried as the scalar
  `max_cond` per region/chain plus a free-text `note` (`"elevated per-op cond
  1.24e+09"`). This is a brief-assumption ↔ report mismatch, not a code drift.
- **`predicted_rel_err_if_float` is loaded-but-dead.** `characterization.py`
  parses, merges (`max`), and carries it on every `RegionRecord`/`ChainRecord`,
  but **no decision reads it** — `ranking.py`/`walk.py`/`agent.py` gate exclusively
  on `predicted_rel_err_if_ff`. It looks mined but is inert. Counted IGNORED.
- **`n` is loaded-but-dead.** Merged via `max`, never gates. Counted IGNORED.
- **`value_range_checked` (per-var) is always `false`** in this report (400k/400k
  sampled) — the per-variable range check was not run, so the field carries no
  signal even before asking whether Strategy reads it.
- No field-name mismatches were found between code accessor strings and real
  report paths (`region_local_vars`, `prov_vars` fallback, `chain`, `chain_id`,
  `ops`, `signal_class`, etc. all resolve).

---

## Part 1 — Field inventory (actual report paths)

Wildcards: `<B>` = integral name, `<file:line>` = region key (`""` = non-localizable
rollup), `<var>` = per-instance source-var key (`"m1[100]"`), `<op>` ∈
{abs,add,atan2,div,log,mul,neg,sqrt,sub}, `<sig>` ∈ {stable, cancellation_cascade,
local_cancellation, log_near_root}.

### Top level
| path | type | example | meaning |
|---|---|---|---|
| `schema_version` | int | `1` | report schema version |
| `kind` | str | `"stability_report"` | report type tag |
| `samples_seen.<B>` | int | `10000` | samples processed per integral |
| `no_id_records` | int | `0` | journal records with no region id |
| `integrals.<B>` | dict | — | per-integral container (20 keys) |

### `integrals.<B>`
| path | type | example | meaning |
|---|---|---|---|
| `.samples` | int | `10000` | samples for this integral |
| `.class_counts.<sig>` | int | `35` | region count per signal class |
| `.regions.<file:line>` | dict | — | localized code-region record |
| `.top_regions_by_rel_err[]` | list | len=10 | top-N regions by rel-err (redundant view of `.regions` + a `location` string) |
| `.cascade_chains[]` | list | len=0 (B1) | localized cascade victims (populated in cancellation integrals) |
| `.variables.<var>` | dict | — | per-source-variable record (~110k/integral) |

### `integrals.<B>.regions.<file:line>` (region record)
| path | type | example | meaning |
|---|---|---|---|
| `.signal_class` | str | `"stable"` | stability class (see `<sig>`) |
| `.max_rel_err` | float | `4.44e-16` | worst observed rel-err at line |
| `.p50_rel_err` | float | `1e-16` | median rel-err |
| `.p99_rel_err` | float | `1e-16` | 99th-pct rel-err |
| `.max_cond` | float | `1.0` | max per-op condition number |
| `.max_amp` | float | `1.0` | max forward amplification bound |
| `.max_sensitivity` | float | `1.0` | max output sensitivity to region |
| `.gate_a_count` | int | `0` | count of Gate-A elevated-conditioning events |
| `.predicted_rel_err_if_ff` | float | `1.42e-14` | model rel-err at ff |
| `.predicted_rel_err_if_float` | float | `5.96e-08` | model rel-err at float |
| `.value_range_ok_for_float` | bool | `true` | float exponent-range safety flag |
| `.abs_val_min` | float\|null | `null` | min \|value\| seen |
| `.abs_val_max` | float\|null | `null` | max \|value\| seen |
| `.n` | int | `40000` | op-samples contributing |
| `.non_localizable` | bool | `false` | true ⇒ rollup bucket (loader skips) |
| `.note` | str | `"elevated per-op cond 1.24e+09"` | free-text conditioning note |
| `.ops.<op>` | int | `add:20000` | dynamic op count per op-kind |
| `.prov_vars[]` | str | `"m1[0]"` (len 50k) | full transitive provenance var names |
| `.region_local_vars[]` | str | `"mu2[0]"` (len 0–N) | tight region-local read var names |

### `integrals.<B>.top_regions_by_rel_err[]`
Same fields as a region record **plus** `.location` (str `"boxGPU.h:140"`) in place
of the dict key. No fields unique to this view.

### `integrals.<B>.cascade_chains[]` (chain record)
| path | type | example | meaning |
|---|---|---|---|
| `.chain[]` | list | — | contributing sub-region spans |
| `.chain[].file` | str | `"B1m.h"` | span file |
| `.chain[].line_start` | int | `240` | span start line |
| `.chain[].line_end` | int | `240` | span end line |
| `.chain_id` | str | `"cascade_B10_003bcbcd_29db34c0"` | stable chain identity |
| `.kind` | str | `"cascade_chain"` | record tag |
| `.signal_class` | str | `"cancellation_cascade"` | chain class |
| `.max_cond` | float | `1.63e6` | max cond across chain |
| `.max_rel_err` | float | `1.25e-05` | worst rel-err |
| `.max_sensitivity` | float | `3.76e8` | max sensitivity |
| `.predicted_rel_err_if_ff` | float | `5.34e-06` | model rel-err at ff |
| `.predicted_rel_err_if_float` | float | `22.4` | model rel-err at float |
| `.n` | int | `8` | contributing op-samples |
| `.non_localizable` | bool | `false` | rollup flag |
| `.ops.<op>` | int | `sub:7` | dynamic op count per op-kind |
| `.region_local_vars[]` | str | — | chain-local read var names |

### `integrals.<B>.variables.<var>` (per-variable record)
| path | type | example | meaning |
|---|---|---|---|
| `.is_source_var` | bool | `true` | user-visible source var (merge keeps only these) |
| `.n_consumers` | int | `1` | downstream consumer count (leverage proxy) |
| `.max_amp` | float | `9.68` | per-var amplification |
| `.max_sensitivity` | float | `9.68` | per-var sensitivity |
| `.predicted_rel_err_if_ff` | float | `1.38e-13` | per-var model rel-err at ff |
| `.predicted_rel_err_if_float` | float | `5.77e-07` | per-var model rel-err at float |
| `.value_range_checked` | bool | `false` | **always false in this report** |

**`signal_class` population (whole report):** cancellation_cascade 527,682 ·
stable 1,307 · log_near_root 67 · local_cancellation 47.
**`value_range_ok_for_float`:** true 1,401 · **false 144** (~9%).
**`gate_a_count`:** ~84 nonzero / ~1,461 zero (tracks non-stable classes).

---

## Part 2 — Cross-reference: mined vs ignored

Legend: **USED** = drives a Strategy action (queue population, ordering,
admission, rung selection, skip). **IGNORED** = no consumer reads it (incl.
generator-only, telemetry-only, and loaded-but-dead). **PARTIAL** = a fraction of
the signal is mined.

### USED
| field | consumer · how |
|---|---|
| `signal_class` | `ranking.build_correctness_queue` (4-tier split) · `ranking.build_speedup_queue` (stable-only admission) · `walk._rewrites_for` (kahan vs identity) · `agent` chain tiering |
| `max_rel_err` | `ranking` correctness tiers 2/3/4 admission (`> 10^-tol`) · `agent.py:201` chain eligibility |
| `predicted_rel_err_if_ff` | `ranking.build_speedup_queue` admission (`<= 10^-tol`) — the gate that actually populates the speedup queue |
| `max_cond` | `ranking._correctness_sort_key` (intra-tier `desc`) · `agent.py:202` chain ordering |
| `ops.<op>` | **PARTIAL** — only `sum(ops.values())` → `op_count`, used for speedup ordering (`-op_count`); per-op *mix* discarded |
| `non_localizable` | `characterization.load_regions` skip filter (+ counts `non_localizable_skipped`) |
| `region_local_vars[]` | `characterization._region_vars` → `RegionTarget.variables` → ff/dd/float integrators |
| `cascade_chains[]` + `chain[]`/`chain_id` | `load_chains` → correctness tier-2 population; `chain_id` = identity/dedup + `agent.py:202` tiebreak |
| `regions` / `integrals` / `samples_seen`(keys) | structural iteration |

### IGNORED (no consumer; pruning candidates)
| field | why it's dark |
|---|---|
| `value_range_ok_for_float` | generator-only (`stability_reducer`); **not read by any consumer** |
| `predicted_rel_err_if_float` | loaded+merged+carried in `characterization.py`, **no decision reads it** |
| `n` (region/chain) | merged via `max`, never gates |
| `p50_rel_err`, `p99_rel_err` | generator-only |
| `max_amp` | generator-only |
| `max_sensitivity` | generator + `backfill_ff_prediction` (derives `pred_ff`); no consumer |
| `gate_a_count` | generator-only |
| `abs_val_min`, `abs_val_max` | generator-only |
| `note` | generator-only (patcher/c8 `note` hits are compiler-diagnostic text, unrelated) |
| `top_regions_by_rel_err[]` (+`.location`) | **no consumer** — redundant view of `.regions` |
| `variables.<var>.*` map (`is_source_var`, `n_consumers`, `max_amp`, `max_sensitivity`, per-var `predicted_*`, `value_range_checked`) | generator/merge only; Strategy takes only `region_local_vars` *names*, never the map |
| `class_counts.<sig>` | telemetry (loaded into no decision) |
| `samples`, `samples_seen`, `no_id_records` | telemetry |
| `schema_version`, `kind` | loaded into `meta` dict, telemetry-only |

### PARTIAL
| field | mined fraction / gap |
|---|---|
| `ops.<op>` | sum mined; per-op flop-weight (div/log ≫ add) discarded |
| `prov_vars[]` | consumed **only as fallback** when `region_local_vars` absent; in report_10k the tight set is present, so `prov_vars` is inert |
| `predicted_rel_err_if_float` | carried but dead (see IGNORED) |
| `max_cond` | mined for *ordering* only; never as an absolute *gate* |

---

## Part 3 — Pruning opportunity per unmined field

Format: **signal** → **decision it prunes** → cost / risk / interaction.

- **`value_range_ok_for_float`** → carries float exponent-range safety, a failure
  mode the error-model gate (`predicted_rel_err_if_ff`) is blind to (a
  well-conditioned value can still over/underflow float). → **Prune the float rung
  of the walk**: when `false`, stop the demotion walk at `ff`, don't attempt
  `double→…→float`. Cost: **trivial** (bool already per region, add one guard in
  the walk / `build_speedup_queue`). Risk: miscalibrated `false` ⇒ leaves float
  speedup on table (**no correctness risk** — Validator still guards accepts).
  Interaction: orthogonal to the ff error gate; complements, no conflict.

- **`ops.<op>` per-op mix** → true hardware cost of a region is flop-weighted, not
  op-count-flat (RATIO_REPORT: `div`=42×, `log`≈2350–4100× vs native in ff/dd). →
  **Reorder the speedup queue** by flop-weighted savings instead of raw
  `op_count desc`, so the div/log-heavy regions (where throughput actually lives)
  are tried first under the speedup budget cap. Cost: **moderate** (new weight fn
  replacing `sum` in `build_speedup_queue` sort key). Risk: reorder-only ⇒ leaves
  speedup on table if weights are off, **no correctness risk**. Interaction:
  refines the existing `op_count` sort — a strict upgrade of the same rule.

- **`predicted_rel_err_if_float`** → predicts float-rung reachability directly. →
  **Add a float-step gate** in the walk (skip the `→float` attempt when
  `pred_float > 10^-tol`, settle at ff), the float analog of the existing ff
  admission gate. Cost: **trivial** (field already loaded). Risk: pessimistic
  `pred_float` ⇒ leaves float on table, **no correctness risk**. Interaction:
  mirrors `predicted_rel_err_if_ff`; would finally *use* a currently-dead field.

- **`variables.<var>.n_consumers`** → downstream fan-out = leverage; `ranking.py`
  docstring explicitly names the **deferred "downstream-leverage tiebreaker (walk
  the prov-var DAG)"**. `n_consumers` is a cheap scalar proxy already emitted. →
  **Tie-break** within a correctness/speedup tier by max/sum `n_consumers` over a
  region's `region_local_vars`. Cost: **moderate** (must join the ignored
  `variables{}` map to regions by var name). Risk: reorder-only, no correctness
  risk. Interaction: fills the documented open tiebreaker slot.

- **`p99_rel_err`** → robust rel-err vs the single-sample `max_rel_err` outlier. →
  Could refine correctness-tier admission (currently `max_rel_err > thr`) to skip
  regions whose `max` is a lone spike but `p99` is clean. Cost: **moderate**.
  Risk: **CORRECTNESS** — `p99` can mask a genuine rare catastrophic cancellation
  (exactly the tier-1/2 target). Use only as informative telemetry / soft
  tiebreak, **never** as a replacement gate. Interaction: would weaken tier
  admission if it replaced `max`.

- **`gate_a_count`** → count of elevated-conditioning events. → Could gate
  "extra-safe stable" demotions. Cost: trivial. Risk: no-op — **duplicates
  `signal_class`** (`gate_a_count>0` tracks the ~84 non-stable regions). Low value.

- **`max_amp` / `max_sensitivity` (region)** → forward amplification bounds. →
  Could pre-filter demote candidates. Cost: trivial. Risk: no-op — **subsumed by
  `predicted_rel_err_if_ff`**, which is *derived from* these (see
  `backfill_ff_prediction`). Duplicate signal.

- **`abs_val_min` / `abs_val_max`** → the raw range behind
  `value_range_ok_for_float`. → Finer float-range gate (per-integral margin).
  Cost: moderate. Risk: null in this report (unfilled) — no bite until populated.
  Subsumed by the boolean flag.

- **`top_regions_by_rel_err[]`** → redundant top-N view of `.regions`. → No prune;
  adopting it would double-count (RATIO_REPORT already excludes it for op-counts).
  No-op.

- **`variables.<var>` fine-grained map** (per-var `predicted_*`, sensitivity) →
  enables **sub-region / per-variable demotion** (demote only the safe vars at a
  mixed line). Signal high, but cost **heavy** — breaks the region-granular
  `(file,line,line,variables)` Patcher contract; new logic, not a filter. Risk:
  new correctness surface. Defer.

- **`value_range_checked`** → per-var range validity. → No prune — **always
  `false`** here, zero signal.

- **`class_counts` / `samples` / telemetry** → could drive early-termination
  (skip integrals with 0 non-stable regions). Cost trivial, signal marginal.

For the **USED** fields, extra signal still on the table:
- `signal_class` — used **binary per tier**; could be *graded* by
  `predicted_rel_err_if_ff` margin for finer intra-tier prioritization.
- `max_cond` — used only for *ordering*; never as an absolute admission gate
  (e.g. a hard "cond > 1e15 ⇒ dd-mandatory" shortcut that skips the ff/float walk
  steps entirely).
- `op_count` — see the `ops.<op>` flop-weight upgrade above.

---

## Part 4 — Recommended prioritization (top 5 by signal × cost)

1. **`value_range_ok_for_float` → float-rung guard.** Trivial to wire, and it
   plugs a real hole: the speedup path admits on `predicted_rel_err_if_ff` (an
   *error* prediction) and the walk then steps down to float blind to exponent
   range. **144/1545 region-instances (~9%) are `false`** in report_10k, so the
   guard would actually fire — it stops the walk from spending a float attempt on
   range-unsafe regions that the error model would otherwise wave through.
   FLOAT_RETRO_PROBE shows float acceptance here is a *conditioning* cliff (8.84
   digits, the BIN1 double floor), i.e. the error model already governs the
   *conditioning* axis — leaving the *range* axis (this field) as the one distinct,
   unmined safety signal. Highest leverage per unit cost.

2. **`ops.<op>` → flop-weighted speedup ordering.** RATIO_REPORT is the direct
   evidence: post-walk float/ff throughput is only ~0.1–1.2% and is *concentrated*
   in a few `div`/`log`-heavy regions (div 42×, log ≈2350–4100× the flop weight of
   an `add`). The current `op_count desc` ordering treats an add-heavy and a
   log-heavy region of equal op-count as equal, so under the speedup budget cap
   (250) it can burn iterations on low-throughput regions first. A flop-weight
   sort front-loads the regions that move the needle — same accept count, more
   throughput captured. Moderate cost, high and *measured* bite.

3. **`predicted_rel_err_if_float` → float-step gate (revive the dead field).**
   Cheapest possible (already loaded), and symmetric with the working
   `predicted_rel_err_if_ff` gate. Honesty check from CALIBRATION_v2 /
   FLOAT_RETRO_PROBE: on *this* codebase at tol=7 there were **zero tolerance
   rejects** — the residual float gap is Patcher **gen-robustness**, not precision
   misprediction — so a `pred_float` gate would **not** shrink the 16 unreached
   instances today. Its value is (a) eliminating a misleading loaded-but-dead
   field and (b) future-proofing tighter-tolerance runs (tol≥9), where
   FLOAT_RETRO_PROBE predicts float falls off the cliff and a `pred_float` gate
   would pre-empt the doomed walk steps. High leverage-per-cost, deferred bite.

4. **`variables.<var>.n_consumers` → downstream-leverage tiebreak.** This is the
   one pruning rule the codebase *already asked for*: `ranking.py` documents the
   "downstream-leverage tiebreaker (walk the prov-var DAG)" as explicitly
   deferred. `n_consumers` is a ready-made cheap proxy, avoiding the full DAG walk.
   Cost is moderate only because it requires joining the currently-ignored
   `variables{}` map back to regions by name. Reorder-only, no correctness risk.
   Bite is unquantified on this report (no distribution pulled), so it ranks below
   the three measured wins but above the risky/duplicate fields.

5. **`p99_rel_err` → robust admission (telemetry-first, not a gate swap).** The
   correctness queue admits on `max_rel_err > thr`; a single-sample max can queue a
   region that is otherwise clean. `p99` offers a distribution-aware view. Ranked
   last and hedged deliberately: it is the **only** candidate with a correctness
   downside — `p99` can hide the rare catastrophic cancellation that tiers 1–2
   exist to catch — so the recommendation is to surface `p99` as telemetry / soft
   intra-tier tiebreak, and **not** to replace the `max` gate. Include it in the
   map because it is real unmined distribution signal; flag it because adopting it
   naively trades speedup gains against the correctness guarantee.

**Net:** the three measured, low-cost wins are `value_range_ok_for_float`,
`ops.<op>` flop-weighting, and reviving `predicted_rel_err_if_float`. All three
are *ordering/skip* prunes with no correctness exposure — they narrow what the
walk must empirically discover without touching the Validator's authority. The
`variables{}` map holds the most signal (per-var demotion, leverage tiebreak) but
also the most cost, and is where a future, larger pruning effort should aim.
