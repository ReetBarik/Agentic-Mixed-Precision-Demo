# Phase-2.1 whole-TU-only e2e — ref_scale wired into `_min_digits` — 2026-07-30

**What changed (metric only).** `tu_provider._min_digits` now computes a per-sample `ref_scale`
(the max `|DD-reference component|` across the sample's six coeffs, from the **oracle** values) and
passes it into every `precise_digits_fast` call — the **same convention** as the Validator's
`agents/validator/validate.py:_score`. Nothing else changed: strategy code, `tu_emit`, primitives,
oracle, tolerance, sample setup all identical to the prior refreshed-oracle run.

**Motivation.** `runs/qcdloop/PHASE_2_B15_TWO_LIMB_TRACE_2026-07-30.md` proved the analytic-zero
integrals' dd compute is correct on every double-resolvable component (12.5–15.6 digits); the 0.0 was
purely `_min_digits` running without `ref_scale`, reading roundoff at sub-scale analytic zeros as 0
digits.

**Commit:** `3760952` (metric plumbing + `tests/qcdloop/test_tu_provider_min_digits.py`, pushed).
Re-run: `20260730_053618_84184a19` (oracle `ddfun_enabled@d11a94b`, tree `bc3f792` — unchanged).

---

## 0. Headline — 3 tier moves, and the prediction was RIGHT for B16, PARTLY WRONG for B14/B15

| | prediction (task) | actual |
|---|---|---|
| **B16** | double → **dd** (~12.55) | double → **dd** (12.5508) ✅ **exactly as predicted** |
| **B14** | double → **dd** (~15.58) | double → **ff** ⚠️ *not dd* — see §3.1 |
| **B15** | **stay double** | double → **dd** ⚠️ *moved* — borderline accept, see §3.2 |
| **BIN0** | stay double | **double** ✅ as predicted |

> The two divergences are **real signal**, surfaced not absorbed (per the task's "if B14/B16 don't
> restore as expected, stop and diagnose"):
>
> - **B14 → ff, not dd.** ref_scale rescued B14's *baseline* too: raw double now measures **13.038**
>   (was the 0.000 artifact), which already clears 7.0. So the correctness phase correctly marks B14
>   `no_flip_needed` (never builds dd — double suffices), then ff wins the speedup downshift (10.82 ≥
>   7.0). The two-limb trace only measured the dd *candidate* (15.58) and didn't account for the
>   baseline being rescued in lockstep — with correct double already at 13 digits, dd is genuinely
>   unnecessary. **This is the walk doing the right thing.**
> - **B15 → dd, on a borderline "best-effort" accept.** B15's baseline AND dd candidate both still
>   measure **0.0** even with ref_scale (its worst noise cell sits above `ZERO_REF_TOL`, exactly as the
>   trace predicted). It routed dd only via `flip_gate` rule-4 ("strict lift > 0 → best-effort
>   progress") on a **7.5e-12-digit** lift — numerical noise, not a genuine dd win. Flagged in §3.2.

Final routing: **ff = 14 · dd = 3 · double = 4** (prior refreshed-oracle: ff=13/dd=1/double=7).

---

## 1. The 21-integral routing table (this run, ref_scale, 5000 samples, tol 7.0, OpenMP)

`base`/`dd`/`ff` = p100 (min over samples/components), **now computed WITH per-sample ref_scale**.
`—n` = dd not attempted (`no_flip_needed`); `notried` = float pruned by report; `A`/`r` = accept/reject.

| integral | base | dd | float | ff | route | Δ vs prior |
|---|---|---|---|---|---|---|
| B1  | 11.808 | —n | notried | 9.264 A | **ff** | — |
| B2  | 12.142 | —n | notried | 10.045 A | **ff** | — |
| B3  | 12.271 | —n | notried | 9.502 A | **ff** | — |
| B4  | 10.250 | —n | notried | 8.423 A | **ff** | — |
| B5  | 11.585 | —n | notried | 9.045 A | **ff** | — |
| B6  | 12.269 | —n | notried | 10.105 A | **ff** | — |
| B7  | 11.626 | —n | notried | 10.182 A | **ff** | — |
| B8  | 10.139 | —n | notried | 8.593 A | **ff** | — |
| B9  | 11.530 | —n | notried | 8.642 A | **ff** | — |
| B10 | 10.093 | —n | notried | 7.891 A | **ff** | — |
| B11 |  9.460 | —n | notried | 7.769 A | **ff** | — |
| B12 |  3.691 | **14.331 A** | notried | 2.406 r | **dd** | — (held) |
| B13 |  8.578 | —n | notried | 7.269 A | **ff** | — |
| **B14** | **13.038** | —n | notried | **10.821 A** | **ff** | **double → ff** |
| **B15** |  0.000 | **0.000 A**† | notried | 0.000 r | **dd** | **double → dd** † |
| **B16** |  6.564 | **12.551 A** | notried | 5.099 r | **dd** | **double → dd** |
| BIN0|  0.000 | 0.000 r | notried | 0.000 r | **double** | — (held) |
| BIN1|  8.068 | —n | notried | 0.000 r | **double** | — |
| BIN2|  9.383 | —n | notried | 0.000 r | **double** | — |
| BIN3|  9.195 | —n | notried | 7.487 A | **ff** | — |
| BIN4|  9.038 | —n | notried | 0.000 r | **double** | — |

† **B15 dd is a borderline accept, NOT a genuine dd result** (§3.2): both baseline and candidate are
0.0; it clears only the gate's rule-4 best-effort-lift branch on a 7.5e-12-digit difference in the
noise floor.

### Precision distribution

| precision | this run (ref_scale) | prior (refreshed-oracle) |
|---|---|---|
| float | 0 | 0 |
| ff | **14** | 13 |
| double | **4** | 7 |
| dd | **3** | 1 |
| **total** | 21 | 21 |

### Two-phase walk

| phase | measures | accepts | vs prior |
|---|---|---|---|
| correctness (dd) | 21 | **3** (B12, B15†, B16) | was 1 (B12) |
| speedup (float→ff) | 21 | **14** (adds B14) | was 13 |

---

## 2. Delta vs prior refreshed-oracle run (`20260730_032849_4aad0c66`)

**Exactly the 4 rows the metric touches moved; the other 17 are bit-identical.**

| integral | prior route | new route | why |
|---|---|---|---|
| **B14** | double | **ff** | ref_scale rescued baseline (0.000→13.038 ≥7) → no_flip_needed → ff downshift wins (§3.1) |
| **B15** | double | **dd** † | baseline+cand still 0.0; rule-4 best-effort accept on 7.5e-12 lift (§3.2) |
| **B16** | double | **dd** | dd candidate 0.000→**12.551** (ref_scale rescue); baseline 6.564 < 7 so genuinely needs dd |
| all 17 others | (unchanged) | (unchanged) | metric change is inert where no analytic-zero cell drives the min |

**Constraint check — the 13 ff-accepted integrals: MAX ff-drift = 0.000000 digits** (threshold 0.01).
ref_scale changed **nothing** on any genuine-signal integral. B12 dd held at 14.331 (bit-identical).
BIN0 held double (baseline+cand still 0.0). No unexpected side effect.

**Wall-clock:** 105.19 s vs prior 106.16 s (**Δ −0.97 s**, noise). The metric change is a per-component
arithmetic tweak off the build/measure hot path, as expected.

---

## 3. Diagnosis of the two divergences from prediction

### 3.1 B14 → ff (not dd) — ref_scale rescued the *baseline*, so double already suffices

The two-limb trace predicted B14 dd ≈ 15.58 and expected `double → dd`. What it did not fold in: the
**vanilla (raw-double) baseline is rescued by ref_scale in lockstep** with the dd candidate, because
the metric artifact was in the *reference comparison*, not the candidate:

| B14 | no ref_scale (prior) | WITH ref_scale (this run) |
|---|---|---|
| baseline (raw double) | 0.000 | **13.038** |
| dd candidate | 0.000 | 15.501 |

With the baseline at 13.038 ≥ 7.0, the correctness phase returns `no_flip_needed` (rule 1: "raw double
already clears tolerance — no flip produced") and never builds the dd TU. Then the speedup phase runs
ff (10.821 ≥ 7.0) and B14 routes **ff**. This is **correct**: a component that carries 13 correct
double digits does not need dd; the earlier "0.000" hid that raw double was already fine. The trace's
"dd ~15.58" was a true measurement of the dd candidate but not the routing driver — **the routing
driver is the baseline, and rescued double wins the ff downshift.**

### 3.2 B15 → dd — a borderline "best-effort" accept on noise (flag, not a genuine dd win)

B15's baseline and dd candidate both still measure **0.0 even with ref_scale** (confirmed: its worst
cell is s4553 c1, `|oracle|/scale = 8.6e-21`, above `ZERO_REF_TOL = 1e-24` — so ref_scale does NOT
classify it as an analytic zero, exactly as `PHASE_2_B15_TWO_LIMB_TRACE` §3 established). At that cell:

| B15 s4553 c1 | value | digits |
|---|---|---|
| oracle (dd) | −8.688e-28 (noise, 8.6e-21 of scale) | — |
| vanilla candidate | 1.701e-23 | 0.0 |
| dd candidate | −1.506e-38 | **7.5e-12** |

The dd candidate's noise value at that cell is a *hair* closer to the oracle's noise than vanilla's,
yielding a **7.527e-12-digit** "lift". `flip_gate._evaluate_upshift` rule 4 accepts any `lift > 0` as
"best-effort progress" even below tolerance — so B15 routes **dd** on a difference 12 orders below one
digit. **This is not a genuine dd win** — both precisions deliver 0.0 meaningful digits for B15's
noise-floor component; the "improvement" is meaningless roundoff jitter.

This is a **known-adjacent** case: it's the same B15/BIN0 sub-double-noise cell the two-limb trace
flagged for a separate resolution-floor follow-up. ref_scale (correctly, by its ZERO_REF_TOL contract)
does not rescue it, and the routing to dd is an artifact of rule-4's `lift > 0` threshold being
sensitive to sub-double noise. Surfaced here for the follow-up; **not fixed** (out of scope — this task
is ref_scale only, ZERO_REF_TOL untouched).

> **B15/BIN0 are the residual for the separate sub-double-resolution floor follow-up.** BIN0 rejects
> (its dd candidate noise happens NOT to beat vanilla → rule-5 reject → double); B15 accepts (its dd
> noise happens to beat vanilla by 7.5e-12 → rule-4 → dd). The coin-flip between them is pure noise —
> both should be handled by a resolution floor that ignores components below ~1e-16 of scale, at which
> point neither would route dd and the gate's rule-4 wouldn't see a spurious lift.

---

## 4. Regression gates

- **Full suite:** `973 passed, 10 deselected (llm)` in 98.5 s (run with `-m "not llm"`). A separate
  full run (llm included) showed one failure — `tests/dd_integrator/test_regional.py::
  test_real_llm_ieps50_derived_not_r4` — a **live-LLM test** (`@pytest.mark.llm`, gated on
  `ANTHROPIC_AUTH_TOKEN`) that flaked with `status='llm_failed'` ("retryable misgeneration",
  llm_tokens=7933). It exercises the real-LLM regional integrator, touches **no** `_min_digits` path,
  and is unrelated to this metric-only change. The non-llm suite (which covers `_min_digits`) is green.
- **New tests:** `tests/qcdloop/test_tu_provider_min_digits.py` — 5 pass (analytic-zero rescue;
  pre-fix 0.0 guard at the primitive level; genuine-signal non-inflation; per-sample oracle-scale
  convention; missing-integral contract). `flip_gate` 23/23.
- **Snapshot + `third_party` pristine** — porcelain empty ✅.
- **Oracle branch untouched** — `ddfun_enabled:src/qcdloop` = `bc3f792` (unchanged) ✅.

## 5. Constraints honored

- **Only `_min_digits` (+ its tests) modified.** Strategy code, `tu_emit`, primitives, oracle — all
  unchanged (`git diff` = `runs/qcdloop/tu_provider.py` `_min_digits` + new test file only).
- **Tolerance not tuned** (7.0). **Sample setup unchanged** (5000, seed 12345). **Oracle unchanged**
  (`d11a94b`). **`ZERO_REF_TOL` untouched** (still 1e-24 — the B15/BIN0 follow-up).
- **13 ff integrals: 0.000000 drift** (well under the 0.01 surface-it threshold) — no genuine-signal
  side effect.
- **Divergences surfaced, not absorbed** — B14→ff and B15→dd both diagnosed to root cause (§3).

## 6. New STOPs

None as a build/run STOP. One flag for the follow-up (already scoped, not this task):

> **`flip_gate` rule-4 (`lift > 0 → best-effort accept`) is sensitive to sub-double noise.** B15 routes
> dd on a 7.5e-12-digit lift between two 0.0-digit noise values. The clean fix is the sub-double-
> resolution floor already scoped for B15/BIN0 (ignore components < ~1e-16 of scale) — with it, B15's
> dd candidate would show no lift and rule-4 would not fire. Independent of the ref_scale change landed
> here.

---

### Provenance
| | |
|---|---|
| commit (metric+tests) | `3760952` |
| run id | `20260730_053618_84184a19` |
| oracle | `~/qcdloop@ddfun_enabled` = `d11a94b`, `:src/qcdloop` tree `bc3f792` (unchanged from prior) |
| prior run compared | `20260730_032849_4aad0c66` (refreshed-oracle, no ref_scale) |
| kokkos | `~/kokkos-install-openmp` (OpenMP), OMP_NUM_THREADS=32, spread/cores |
| scratch | `runs/qcdloop/tu_e2e_out_refscale/` (fresh; snapshot untouched) |
| artifacts | `strategy/20260730_053618_84184a19/{report.json,report.md,iterations.jsonl,final.diff}` |
