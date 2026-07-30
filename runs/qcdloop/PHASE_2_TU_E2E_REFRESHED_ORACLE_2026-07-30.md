# Phase-2.1 whole-TU-only e2e — REFRESHED dd oracle (hypot-aligned) — 2026-07-30

**What ran.** A re-run of the Strategy `strategy_mode="tu_only"` walk, byte-for-byte the same
invocation as the prior TU e2e (`20260730_002959_76e048b8`), against the **refreshed dd oracle**
(`~/qcdloop@ddfun_enabled` advanced from `2229ec4` → `d11a94b`). The refresh landed the
hypot-style scaled `abs()`/`sqrt()` in `dd_complex.hpp` (`9a0b8cf`) plus ff primitives
(`d11a94b`), so the oracle and the candidate flip TU now use **identical dd complex primitives**.

**Focus of interest (from the task):** *do B14/B15/B16/BIN0 restore to dd routing now that the
oracle and candidate use identical dd primitives?*

> ## ⟹ Headline: NO — they did NOT restore. Routing is bit-identical to the prior run.
>
> The expected `dd 0.000 → ~15.96` restoration and the expected `double → dd` moves **did not
> happen**. Every routing decision, every candidate-digit value, and the precision distribution are
> **identical** to `20260730_002959_76e048b8`. This is a genuine surprise vs the task's stated
> expectation ("Expected yes"), and the diagnosis below is the real signal of this run.

---

## 0. Step 1 — oracle refresh confirmed (tree hashes differ)

| | ref (`ddfun_enabled` HEAD) | resolved `dd_tree_hash` (`:src/qcdloop`) |
|---|---|---|
| **prior run** (`20260730_002959_76e048b8`) | `2229ec4` | `fcc69a93cf70a9d13a2560fc5422200167333f8c` |
| **this run** (`20260730_032849_4aad0c66`) | `d11a94b` | `bc3f792c96897bdb5eb147373a6191e8bfd90e4c` |

Tree hashes **differ** ✅ — the cache invalidated correctly. The provider (`tu_provider.py`) rebuilds
the oracle every run via `git archive ddfun_enabled:src/qcdloop` (no persistent oracle cache), and the
oracle tree extracted for this run (`tu_e2e_out_refreshed/dd_oracle_tree/src/qcdloop/dd_complex.hpp`)
was verified **byte-identical** to `ddfun_enabled:src/qcdloop/dd_complex.hpp @ d11a94b` and contains
the hypot-style `abs`/`sqrt`. I used a **fresh** `--tu-out-dir` (`tu_e2e_out_refreshed`) to preclude any
stale cmake/build reuse.

Refresh contents (both landed on the oracle branch since the prior run):

- `9a0b8cf` — `dd_complex.hpp`: hypot-style scaled `abs()`; `sqrt()` now calls `abs(z)`. Mirrors the
  candidate's `third_party/include/dd_complex.hpp` (verified: both carry the same "Hypot-style scaled
  magnitude" implementation).
- `d11a94b` — ff (`ff_math.hpp` + `ff_complex.hpp`) primitives added to the oracle (ff-route parity;
  no effect on the dd measure).

---

## 1. The 21-integral routing table (this run, 5000 samples, tol 7.0, OpenMP)

`base` = raw-double p100; `dd`/`ff` = candidate p100 (min over samples/components, **no `ref_scale`**,
exactly as `_min_digits` computes it); `float` is `notried` for all 21 (report pred-float prune, §4).

| integral | base | dd (status) | float | ff (status) | route |
|---|---|---|---|---|---|
| B1  | 11.808 | — (no_flip_needed) | notried | **9.264** (accepted) | **ff** |
| B2  | 12.142 | — (no_flip_needed) | notried | **10.045** (accepted) | **ff** |
| B3  | 12.271 | — (no_flip_needed) | notried | **9.502** (accepted) | **ff** |
| B4  | 10.250 | — (no_flip_needed) | notried | **8.423** (accepted) | **ff** |
| B5  | 11.585 | — (no_flip_needed) | notried | **9.045** (accepted) | **ff** |
| B6  | 12.269 | — (no_flip_needed) | notried | **10.105** (accepted) | **ff** |
| B7  | 11.626 | — (no_flip_needed) | notried | **10.182** (accepted) | **ff** |
| B8  | 10.139 | — (no_flip_needed) | notried | **8.593** (accepted) | **ff** |
| B9  | 11.530 | — (no_flip_needed) | notried | **8.642** (accepted) | **ff** |
| B10 | 10.093 | — (no_flip_needed) | notried | **7.891** (accepted) | **ff** |
| B11 |  9.460 | — (no_flip_needed) | notried | **7.769** (accepted) | **ff** |
| B12 |  3.691 | **14.331** (accepted) | notried | 2.406 (reject <7) | **dd** |
| B13 |  8.578 | — (no_flip_needed) | notried | **7.269** (accepted) | **ff** |
| B14 |  0.000 | 0.000 (reject <7)† | notried | 0.000 (reject <7) | **double** |
| B15 |  0.000 | 0.000 (reject <7)‡ | notried | 0.000 (reject <7) | **double** |
| B16 |  0.000 | 0.000 (reject <7)‡ | notried | 0.000 (reject <7) | **double** |
| B2m… | | | | | |
| BIN0|  0.000 | 0.000 (reject <7)‡ | notried | 0.000 (reject <7) | **double** |
| BIN1|  8.068 | — (no_flip_needed) | notried | 0.000 (reject <7) | **double** |
| BIN2|  9.383 | — (no_flip_needed) | notried | 0.000 (reject <7) | **double** |
| BIN3|  9.195 | — (no_flip_needed) | notried | **7.487** (accepted) | **ff** |
| BIN4|  9.038 | — (no_flip_needed) | notried | 0.000 (reject <7) | **double** |

† **B14** = pure metric artifact — genuine dd is ~15.5; only the missing `ref_scale` reads it as 0.0 (§3.1).
‡ **B15/B16/BIN0** = the metric-artifact story does **not** fully apply — the candidate dd flip is
genuinely wrong at one component even *with* `ref_scale` (§3.2). This is the new finding.

### Precision distribution

| precision | this run | prior run |
|---|---|---|
| float | 0 | 0 |
| ff | 13 | 13 |
| double | 7 | 7 |
| dd | 1 | 1 |
| **total** | 21 | 21 |

### Two-phase walk

| phase | measures | accepts |
|---|---|---|
| correctness (dd) | 21 | 1 (B12) |
| speedup (float→ff) | 21 | 13 |

---

## 2. Delta vs the prior TU e2e (`20260730_002959_76e048b8`) — **zero drift**

**The two runs are numerically indistinguishable.** Comparing every `tu_row` candidate:

| quantity | prior | this run | delta |
|---|---|---|---|
| routing (all 21) | ff=13 / dd=1 / double=7 | ff=13 / dd=1 / double=7 | **identical** |
| **max \|Δ\| ff-candidate digits** (all 21) | — | — | **0.00000** |
| **max \|Δ\| dd-candidate digits** (B12) | 14.331 | 14.331 | **0.00000** |
| base (double) digits (all 21) | — | — | **0.00000** |
| B14/B15/B16/BIN0 dd | 0.000 | 0.000 | **0.000 (NOT restored)** |

- **Expected `0.000 → ~15.96` for the four analytic-zero dd candidates: did NOT occur.**
- **Expected `double → dd` routing shifts for those four: did NOT occur.**
- **No other integral shifted tier or ε-drifted.** The 13 ff-accepted integrals are bit-identical to
  prior (max ff drift = 0.00000 digits, far under the 0.1-digit surface-it threshold). B12 dd unchanged
  at 14.331. The oracle port introduced **no** side effect on any genuine (non-analytic-zero) integral.

This is the constraint check the task asked for — *"if any of the previously-accepted 13 ff integrals
now shift tier or ε-drift meaningfully (>0.1 digits), that's evidence the oracle port introduced
unexpected side effects."* **They did not shift at all.** The port is side-effect-free on every genuine
integral.

---

## 3. Why the four did NOT restore — full diagnosis (the real signal)

The task's hypothesis was: aligning the oracle's `abs`/`sqrt` to the candidate's hypot primitives makes
candidate and oracle **bit-identical** at the analytic zeros → `err==0` → `MAX_DIGITS` → dd routes.
That hypothesis is **falsified by measurement.** Two independent reasons, and they differ per integral.

### 3.0 The oracle output DID change — but only at the noise floor

I built the **old** (naive-abs) oracle from `2229ec4` and diffed its output against the **new**
(hypot-abs) oracle for all analytic-zero integrals plus genuine controls:

| integral | oracle old == new? | max \|Δ\| (dd value) |
|---|---|---|
| B14  | **changed** | 3.2e-39 (4 comps) |
| B15  | **changed** | 1.7e-27 (6 comps) |
| B16  | **changed** | 1.7e-27 (6 comps) |
| BIN0 | **changed** | 4.9e-41 (231 comps) |
| B12  | identical | 0.0 |
| BIN1 | identical | 0.0 |
| BIN2 | identical | 0.0 |
| BIN3 | identical | 0.0 |

So the hypot port **did** move the oracle — but *only* at the deep-noise components of the four
analytic-zero integrals (~1e-27…1e-41), and it is **bit-identical at every genuine integral**. The
port is clean where it matters. But moving the oracle's noise floor from one roundoff path to another
did **not** make it match the candidate's noise floor — it just relocated the disagreement.

### 3.1 B14 — pure metric artifact (would restore with `ref_scale`, NOT with abs-sync)

Direct candidate-vs-oracle comparison of the `flip_B2m_dd` binary (B14 lives in group `box/B2m.h`),
5000 samples:

- per-component min digits (no `ref_scale`, as `_min_digits` runs): `[15.56, 15.63, 15.71, 0.00, 31.9, 31.9]`
- worst is **comp 3 @ sample 966**: candidate `-5.02e-40` vs oracle `-2.21e-40`, rel err **1.27 → 0 digits**.
- That component is an **analytic zero** relative to the sample scale, so with `ref_scale` supplied the
  metric returns MAX and **B14 min rises to 15.50**.

**B14 is exactly the documented dd metric artifact** (`project_phase2_routing_45197be` §3): the genuine
digits are ~15.5, and the 0.0 is purely `_min_digits` being called without `ref_scale`. The hypot
abs-sync did NOT fix it — the residual gap is a *value-level* roundoff at a ~1e-40 component, not an
abs/sqrt-primitive roundoff. Only `ref_scale` fixes B14.

### 3.2 B15 / B16 / BIN0 — NOT (only) a metric artifact: candidate is genuinely wrong

This is the new, important finding. For B15, B16, BIN0, the min digits stay **0.0 even WITH `ref_scale`**:

| integral | min NO ref_scale | min WITH ref_scale | worst component |
|---|---|---|---|
| B14  | 0.000 | **15.500** | comp3 s966: analytic zero (restores) |
| B15  | 0.000 | **0.000** | comp0 s0: cand `0.0` vs oracle `-7.14e-12`; true/scale = **0.055** (a REAL signal) |
| B16  | 0.000 | 0.000 | same group as B15 (`box/B3m.h`), same failure |
| BIN0 | 0.000 | **0.000** | comp1 s536: cand `1.5e-41` vs oracle `-4.24e-33`; true/scale = **9.3e-23** (just above the 1e-24 zero floor) |

- **B15/B16** (group `box/B3m.h`): the candidate dd flip emits a **hard `0.0`** at comp 0 where the
  oracle computes a genuine `-7.14e-12` whose magnitude is **5.5 % of the sample scale** — an
  unambiguously *real* value, not an analytic zero. `ref_scale` cannot rescue this: the candidate is
  simply computing the wrong number. The abs-sync had no bearing on it.
- **BIN0** (group `box/B0m.h`): candidate `~1e-41` vs oracle `~1e-33`, and `true/scale = 9.3e-23` sits
  **just above** `ZERO_REF_TOL = 1e-24`, so the metric (correctly, by its own rule) treats it as a
  genuine deep-noise value rather than an analytic zero, and the candidate misses it by 8 orders.

**Conclusion:** the "B14/B15/B16/BIN0 all restore to dd once primitives align" premise conflated four
integrals that turn out to have **three different** causes:

1. **B14** — genuine metric artifact; fixable only by passing `ref_scale` into `_min_digits`
   (abs-sync is irrelevant to it).
2. **B15/B16** — candidate dd flip produces a *wrong* value (hard 0.0 vs a 5%-of-scale real signal);
   not a metric problem at all.
3. **BIN0** — value disagreement at a component the metric classifies as genuine (true/scale just above
   the zero floor); not a metric artifact by the metric's own definition.

The oracle port was the right thing to do (it aligns primitives and is provably side-effect-free on all
genuine integrals, §2), but it was **never sufficient** to restore these four to dd, because the
residual disagreement lives at the value level (B15/B16/BIN0) or requires `ref_scale` (B14) — neither of
which the `abs`/`sqrt` sync touches.

### 3.3 Note on emission format

The candidate flip TU emits **single-word** components (dd `lo` folded/printed hi-only) while the oracle
emits `hi|lo`. For the genuine integrals this costs nothing (B12 measures 14.33). It is not the cause of
the four zeros — the hi words agree at the genuine components; the zeros come from the specific
components diagnosed in §3.1–§3.2.

---

## 4. Why `float` is `notried` everywhere (unchanged from prior)

Identical to the prior run: the report's worst-case `predicted_rel_err_if_float` ≫ 10⁻⁷ for all 21, so
the float rung is pruned up front (`_float_rung_ok`). `ff` is always attempted. No routing consequence —
float never wins any qcdloop box integral at tol 7.0. The oracle refresh does not touch the report, so
this is bit-identical to prior.

---

## 5. Wall-clock delta vs prior TU e2e

| run | oracle ref | duration (walk-internal) |
|---|---|---|
| prior (`20260730_002959_76e048b8`) | `2229ec4` | 109.6 s |
| this (`20260730_032849_4aad0c66`) | `d11a94b` | **106.16 s** |

**Δ ≈ −3.4 s** (−3.1 %). Within run-to-run OpenMP scheduling noise; the extra ff primitives in the
oracle tree add no measurable build/measure cost (the dd oracle build dominates and is unchanged in
scope). Same backend (OpenMP, `~/kokkos-install-openmp`, 32 threads spread/cores), same 5000 samples,
same seed 12345 — a genuine controlled comparison this time (unlike prior vs the serial L-measure
reference).

---

## 6. Regression gates

- **Full test suite:** `978 passed in 273.49s` ✅ (llm-marked live tests deselected as usual).
- **`third_party` pristine** — `git status --porcelain third_party` empty ✅ (candidate primitives
  unchanged; only the oracle branch moved).
- **Snapshot pristine** — `git status --porcelain runs/qcdloop_headers_full` empty ✅ (provider clones
  into `tu_e2e_out_refreshed/tree`, never writes the snapshot; STOP #Z holds).
- **flip_gate:** 18/18 ✅.
- **Oracle repo** `~/qcdloop` on `ddfun_enabled` (`d11a94b`); `git archive` reads the committed tree
  only (pre-existing untracked scratch files in `src/qcdloop` are irrelevant to the extraction).
- **No LLM on the walk path** — `patcher_fn`/`validator_fn` are `None` in tu_only; region/chain walk
  guarded off. Confirmed no LLM traffic.

## 7. New STOPs

**None as a build/run STOP.** But this run **falsifies the abs-sync-restores-dd hypothesis** and
surfaces a real, previously-conflated distinction:

> **The four analytic-zero integrals are not one bug.** B14 is a genuine `_min_digits`-missing-`ref_scale`
> metric artifact (restores with `ref_scale`). B15/B16/BIN0 are *not* metric artifacts — the candidate
> dd flip emits a value that disagrees with the oracle by many orders at a component the metric
> classifies as genuine (B15/B16: 5.5 %-of-scale real signal → hard 0.0 candidate; BIN0: true/scale
> just above the 1e-24 zero floor). Aligning the oracle's `abs`/`sqrt` to hypot moved the oracle's noise
> floor but did not close either gap, so routing is unchanged.

Follow-up decisions for Reet (not taken here — no strategy/metric change per task constraints):

1. **B14** — to route dd, `_min_digits` must pass `ref_scale` (per-sample max\|component\|) into
   `precise_digits_fast`, matching the Validator's `_score`. This is a metric-plumbing change in
   `tu_provider._min_digits`, orthogonal to the oracle port.
2. **B15/B16** — investigate why the candidate dd flip on `box/B3m.h` emits `0.0` at comp0 where the
   oracle has a 5%-of-scale value. This looks like a **candidate-flip correctness bug** (a real
   component being dropped/zeroed in the flip TU for the B3m group), not a metric artifact.
3. **BIN0** — the disagreement is at a component with true/scale = 9.3e-23, just above `ZERO_REF_TOL`.
   Decide whether that component is genuinely computable (then it's a candidate bug like B15/B16) or a
   near-zero the threshold should absorb.

---

## 8. Constraints honored

- **Strategy code not modified between runs** — only the oracle input (`ddfun_enabled` ref) changed.
  HEAD stayed `49490e2`; no source edits.
- **Tolerance not tuned** — held at 7.0 throughout.
- **Seed/count held** at 12345 / 5000 (task-mandated, characterizer's setup).
- **Report used as-is** (`report_5k.json`); not regenerated.
- **No metric change** — the `ref_scale` / candidate-flip findings are diagnosed and reported, **not
  fixed** in this task.

---

### Provenance

| | |
|---|---|
| run id | `20260730_032849_4aad0c66` |
| final branch | `strategy/20260730_032849_4aad0c66` |
| repo HEAD | `49490e2` (unchanged from prior TU e2e) |
| oracle | `~/qcdloop@ddfun_enabled` = `d11a94b`, `:src/qcdloop` tree `bc3f792` |
| prior oracle | `2229ec4`, tree `fcc69a9` |
| kokkos | `~/kokkos-install-openmp` (OpenMP, ZEN2), 32 threads spread/cores |
| scratch | `runs/qcdloop/tu_e2e_out_refreshed/` (fresh; snapshot untouched) |
| artifacts | `runs/qcdloop/strategy/20260730_032849_4aad0c66/{report.json,report.md,iterations.jsonl,final.diff}` |
