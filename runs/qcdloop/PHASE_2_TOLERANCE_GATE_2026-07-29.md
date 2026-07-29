# Phase-2 Tolerance-Gate Reframe + Re-run @ tolerance=7.0 — 2026-07-29

Pulls the `ff_math.hpp` unary-`operator+` enrichment (commit **3ab4aa6**), reframes the
acceptance gate from baseline-preserving to **tolerance-based**, and re-runs the full
correctness (UPSHIFT) and speedup (DOWNSHIFT) passes at **tolerance = 7.0** on all 21
integrals.

## 0. Executive verdict

| item | outcome |
|---|---|
| Source pull | **3ab4aa6** fast-forwarded; `third_party/` pristine after pull (1 line added to `ff_math.hpp`). Unblocks **B2m/B11 at ff**. |
| Gate rewrite | `flip_gate.py` now decides against `tolerance` (required, fail-loud), not the raw-double baseline. New `no_flip_needed` terminal-good state (double already clears the bar). |
| Tests | `test_flip_gate.py` rewritten to the tolerance contract: **15 → 18** (+3). Full suite **971 pass / 1 fail** — the one failure is a real-LLM test hitting a **GCP IAM 403** (`aiplatform.endpoints.predict` denied on the model endpoint), environmental, zero references to the gate. |
| Correctness (dd upshift) | **4 accept / 7 no_flip_needed / 0 reject.** dd accepts: B14, B15, B16, BIN0 (baseline 0.0). no_flip: B10, B12, B13, BIN1–4 (double already ≥ 7.0). |
| Speedup (downshift) | Raw-double set (B1–B9, B11): **float 0/10, ff 10/10 → ff.** dd-headroom set (the 7 no_flip): **ff 3/7 → ff** (B10, B12, B13); BIN1–4 stay double. |
| Final routing | **dd = 4** (B14,B15,B16,BIN0) · **ff = 13** (B1–B9,B11,B10,B12,B13) · **double = 4** (BIN1,BIN2,BIN3,BIN4). |
| Regression | snapshot + `third_party` pristine (STOP #Z); dd digits for the 4 dd accepts byte-identical to the prior run (tier unchanged). |
| New STOPs | **STOP #HHH** (BIN1/BIN2 ff = `nan` at runtime) and **STOP #III** (BIN3/BIN4 ff build-fails: `int`→`ffloat` ambiguous conversion, vendored ff-primitive gap). Both flag-and-stop; both integrals correctly fall back to double. |

**Bottom line.** The 7.0 bar changes the picture completely versus the old baseline-preserving
gate (which put all 11 dd candidates on dd and every raw-double integral back on double). At a
7-digit bar, **ff becomes the workhorse**: it clears 7.0 for 13 of 21 integrals — all 10
raw-double box integrals plus 3 dd candidates (B10/B12/B13) whose double baseline had headroom
above 7 but whose ff downshift still lands ≥ 7. Only 4 genuinely ill-conditioned integrals
(B14/B15/B16/BIN0, baseline 0.0 digits) still need dd. Four BIN integrals can't reach ff
(runtime nan or vendored build gap) and stay double.

---

## 1. Gate rewrite (`agents/patcher/flip_gate.py`)

The old gate accepted an upshift iff `lift > margin` and a downshift iff `lift >= -margin` —
both measured against the **raw-double baseline**. As `PHASE_2_FF_LANDED_2026-07-29.md` §3.3
documented, that framing rejects every downshift on a workload whose double baseline sits
above the bar (all precision above the bar is treated as sacred), and it wastes headroom.

The new gate decides against `StrategyConfig.tolerance` — the user's precise-digit bar. Both
`baseline_digits` and `candidate_digits` are p100 (min over samples/components), unchanged
`_min_digits`. `margin` survives as a strictness buffer: the effective bar is
`tolerance + margin`.

**UPSHIFT (double → dd/wider):**
1. `baseline_digits >= bar` → **no_flip_needed** (double already clears; no-op, terminal-good,
   *not* an accept — produces no flip TU).
2. built ∧ `candidate_digits >= bar` → **accept** (clears the bar, lift irrelevant).
3. built ∧ `lift > margin` → **accept** (below the bar, but strict progress toward it).
4. else → **reject**.

**DOWNSHIFT (double → float/ff):**
1. built ∧ `candidate_digits >= bar` → **accept** (bar cleared; negative lift is fine, that
   precision was headroom).
2. else → **reject**.

`GateInputs.tolerance` is **required** (no keyword default); `evaluate` raises `ValueError` if
it is `None`. `GateDecision` gains `no_flip_needed: bool` and a `terminal_good` property
(accept ∨ no_flip_needed). Reason strings carry stable tokens: `no_flip_needed`, `accept`,
`build_failed`, `unmeasurable`, `below_tolerance`. The module docstring was rewritten to the
tolerance contract; the "must preserve baseline" language is gone.

Callsites wired (each reads tolerance at the callsite, no hardcoding, fail-loud):
`runs/qcdloop/phase1_lmeasure.py` and `runs/qcdloop/phase2_lmeasure.py` both grow a
**required** `--tolerance` arg. (`agents/strategy/agent.py` does not import the gate; it uses
`error_threshold(tolerance)` directly and is unaffected.)

**Test delta:** `tests/patcher/test_flip_gate.py` rewritten, **15 → 18** tests. New cases per
the spec: UPSHIFT baseline≥tol→no-op / cand≥tol→accept / cand<tol,lift>0→accept /
cand<tol,lift≤0→reject; DOWNSHIFT cand≥tol→accept (incl. negative lift) / cand<tol→reject /
build_failed→reject; plus tolerance-required-fails-loud and margin-raises-the-bar (both
directions).

---

## 2. Correctness (Phase-1 UPSHIFT) @ tolerance=7.0

2000 samples, dd oracle `ddfun_enabled`. `runs/qcdloop/phase1_lmeasure_tol7_out/`.

| integral | baseline_digits | dd_digits | verdict | reason |
|---|---|---|---|---|
| B10  | 12.187 | 15.960 | **no_flip_needed** | double already clears 7.0 |
| B12  |  9.325 | 15.956 | **no_flip_needed** | double already clears 7.0 |
| B13  |  8.692 | 15.956 | **no_flip_needed** | double already clears 7.0 |
| B14  |  0.000 | 15.958 | **accept (dd)** | candidate clears 7.0, lift +15.96 |
| B15  |  0.000 | 15.963 | **accept (dd)** | candidate clears 7.0, lift +15.96 |
| B16  |  0.000 | 15.960 | **accept (dd)** | candidate clears 7.0, lift +15.96 |
| BIN0 |  0.000 | 15.964 | **accept (dd)** | candidate clears 7.0, lift +15.96 |
| BIN1 |  8.840 | 15.964 | **no_flip_needed** | double already clears 7.0 |
| BIN2 |  9.122 | 15.967 | **no_flip_needed** | double already clears 7.0 |
| BIN3 |  9.380 | 15.960 | **no_flip_needed** | double already clears 7.0 |
| BIN4 |  9.599 | 15.963 | **no_flip_needed** | double already clears 7.0 |

**4 accept / 7 no_flip_needed / 0 reject.** As predicted, the well-conditioned dd candidates
(baseline 8.69–12.19 ≥ 7.0) drop out of dd as no-ops; only the four with a 0.0-digit double
baseline still need dd. The 4 dd accepts' `dd_digits` are **byte-identical** to the prior
Phase-1 run (15.958 / 15.963 / 15.960 / 15.964) — tier unchanged, regression clean.

---

## 3. Speedup (Phase-2 DOWNSHIFT) @ tolerance=7.0

Walk = FLOAT then FF, first target ≥ 7.0 wins, else double. Every integral **not accepted for
dd** in §2 is a speedup candidate — that is the 10 raw-double integrals **plus** the 7
no_flip_needed dd candidates (run via the new `--targets` override). 2000 samples.
`runs/qcdloop/phase2_lmeasure_tol7_out/` + `runs/qcdloop/phase2_ddheadroom_tol7_out/`.

### 3a. Raw-double set (B1–B9, B11)

| integral | baseline | float_digits | ff_digits | final |
|---|---|---|---|---|
| B1  | 12.491 | 2.658 | 9.264  | **ff** |
| B2  | 12.921 | 4.055 | 10.507 | **ff** |
| B3  | 11.985 | 3.533 | 9.682  | **ff** |
| B4  | 11.372 | 3.789 | 10.158 | **ff** |
| B5  | 12.714 | 4.231 | 10.130 | **ff** |
| B6  | 12.269 | 3.325 | 10.773 | **ff** |
| B7  | 11.625 | 3.616 | 10.794 | **ff** |
| B8  | 10.765 | 2.640 | 8.966  | **ff** |
| B9  | 11.673 | 2.668 | 8.642  | **ff** |
| B11 |  9.460 | 0.478 | 7.805  | **ff** |

**float 0/10, ff 10/10 → ff.** B11 now builds at ff (commit 3ab4aa6's unary `operator+`) and
delivers 7.805 ≥ 7.0. Float is far too narrow (0.5–4.2 digits) for the box family and rejects
uniformly — the two-target ordering (cheaper float first) still pays for itself as the
rejection is free of an ff build when it would win, and here float simply never clears.

### 3b. dd-headroom set (the 7 no_flip_needed candidates)

| integral | baseline | float_digits | ff_digits | final | note |
|---|---|---|---|---|---|
| B10  | 12.187 | 3.152 | 9.761 | **ff** | |
| B12  |  9.325 | 0.696 | 7.944 | **ff** | |
| B13  |  8.692 | 0.607 | 7.495 | **ff** | |
| BIN1 |  8.840 | 0.000 | nan   | double | **STOP #HHH** (ff runtime nan) |
| BIN2 |  9.122 | 0.000 | nan   | double | **STOP #HHH** (ff runtime nan) |
| BIN3 |  9.380 | 0.527 | —     | double | **STOP #III** (B3m ff build-fail) |
| BIN4 |  9.599 | 0.000 | —     | double | **STOP #III** (B4m ff build-fail) |

**ff 3/7 → ff** (B10, B12, B13). The four BIN candidates cannot reach ff and correctly fall
back to double (their double baseline already clears 7.0, so no correctness is lost).

---

## 4. Final routing table (21 integrals)

| integral | previous run | this run @ tol=7.0 | moved? |
|---|---|---|---|
| B1  | double | **ff**     | double → ff |
| B2  | double | **ff**     | double → ff |
| B3  | double | **ff**     | double → ff |
| B4  | double | **ff**     | double → ff |
| B5  | double | **ff**     | double → ff |
| B6  | double | **ff**     | double → ff |
| B7  | double | **ff**     | double → ff |
| B8  | double | **ff**     | double → ff |
| B9  | double | **ff**     | double → ff |
| B11 | double | **ff**     | double → ff |
| B10 | dd     | **ff**     | dd → ff |
| B12 | dd     | **ff**     | dd → ff |
| B13 | dd     | **ff**     | dd → ff |
| B14 | dd     | **dd**     | unchanged |
| B15 | dd     | **dd**     | unchanged |
| B16 | dd     | **dd**     | unchanged |
| BIN0| dd     | **dd**     | unchanged |
| BIN1| dd     | **double** | dd → double |
| BIN2| dd     | **double** | dd → double |
| BIN3| dd     | **double** | dd → double |
| BIN4| dd     | **double** | dd → double |

**Tally:** dd = 4 · ff = 13 · double = 4.
**Moves from prior assignment:** 10 (double→ff) + 3 (dd→ff) + 4 (dd→double) = **17 of 21
integrals move**; 4 unchanged (the dd core B14/B15/B16/BIN0).

Why the wholesale shift: the prior report used the baseline-preserving gate at (effectively)
tolerance=10, which (a) accepted every dd candidate that built + lifted and (b) rejected every
downshift because ff/float land below the *delivered double* baseline. The 7.0 bar makes the
question "does the candidate clear 7 digits?" instead of "does it match double?" — and ff
clears 7 for most of the family while dd is only *needed* where double itself falls below 7.

---

## 5. New STOPs

- **STOP #HHH — BIN1/BIN2 ff runtime `nan`.** Both build clean at ff (B1m/B2m groups) but the
  ff coefficient measurement is `nan` → `_min_digits` yields nan → below-tolerance reject.
  This is a *runtime* ff behavior (not a build gap): the BIN integrand hits an ff code path
  that produces nan where dd/double do not. Correctly falls back to double (no correctness
  risk). Flag for investigation if a BIN ff tier is ever wanted; **no action** for the current
  routing (double is the right home).
- **STOP #III — BIN3/BIN4 ff build failure.** `box/B3m.h:117` and `box/B4m.h:172` raise
  `conversion from 'int' to 'quad::ffun::ffloat' is ambiguous`. This is the same *class* as the
  `operator+` gap (a vendored ff-primitive gap in `ff_math.hpp` / `ff_complex.hpp`, not the
  `kokkosMaths_ff.h` enrichment): the `ffloat` type needs an unambiguous `int` converting
  constructor (dd's `ddouble` has one). One-line-ish hand-back to Reet, mirroring the
  `operator+` fix. Until then BIN3/BIN4 stay double.

Neither STOP blocks the run — both integrals have a correct double fallback and the pass
completed with a full routing table.

---

## 6. Regression gates

- Full test suite: **971 pass / 1 fail**; the single failure
  (`test_regional.py::…[B4m.h:163]`) is a real-LLM test denied by GCP IAM (403,
  `aiplatform.endpoints.predict`), unrelated to this change (0 gate references).
- `flip_gate` tests: **18/18 green**.
- Snapshot `runs/qcdloop_headers_full` + `third_party/`: **pristine** (git porcelain clean).
- dd render byte-identical for the 4 dd accepts that don't change tier (B14/B15/B16/BIN0
  dd_digits identical to prior run).

## 7. Artifacts

- Gate: `agents/patcher/flip_gate.py`, `tests/patcher/test_flip_gate.py`.
- Runners: `runs/qcdloop/phase1_lmeasure.py`, `runs/qcdloop/phase2_lmeasure.py`
  (`--tolerance` required; phase2 gains `--targets`).
- Data: `runs/qcdloop/phase1_lmeasure_tol7_out/phase1_lmeasure.json`,
  `runs/qcdloop/phase2_lmeasure_tol7_out/phase2_lmeasure.json`,
  `runs/qcdloop/phase2_ddheadroom_tol7_out/phase2_lmeasure.json`.
