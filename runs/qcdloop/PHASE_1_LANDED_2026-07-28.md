# Phase-1 Template-Argument Promotion — LANDING (correctness) — 2026-07-28

Landing dispatch following the scoping report `TEMPLATE_ARG_PROMOTION_DESIGN.md`
(@6a03b2e) and the authorized vendored-primitive enrichment. Mechanism = **per-integral
whole-TU precision flip** (Decision 1): each promoted integral is compiled in its own
translation unit at its own precision via the pruned per-group `BO`, and its dd output is
narrowed to caller precision only at the app-output boundary reusing the acc1482
designed-exit transform.

## 0. Executive verdict

| item | outcome |
|---|---|
| Deliverables 1–6 (machinery) | **LANDED + unit-validated** (74 targeted tests, 568 patcher/integrator/shared + 65 validator green) |
| Per-group dd TU builds (11 groups→5 distinct) | **ALL CLEAN** at dd from the snapshot clone alone (Decision 2 honored) |
| App-boundary narrowing | **reuses shared acc1482 primitive** `narrow_two_limb_scalar` (STOP #TT — no one-off; 26 acc1482 tests byte-identical) |
| Snapshot pristine | **STOP #Z clean** — `runs/qcdloop_headers_full/` untouched; only the authorized 10-line `third_party/` enrichment |
| L-measure (Deliverable 7) | **RAN** — 2000 samples, all 11 build+measure |
| 🛑 **Acceptance instrument** | **STOP #WW FIRED** — the vs-dd instrument is **circular for dd-insufficient integrals**; 7/11 accepts are false positives. Machinery is correct; the *measurement reference* is wrong for those 7. |

**Bottom line:** the mechanism is built and works — every flagged integral builds honestly
at dd and narrows correctly. The L-measure is **valid for the 4 dd-sufficient integrals**
(B10/B12/B13/B14 → genuine lift) and **circular for the 7 dd-insufficient ones**
(B15/B16/BIN0-4 → false-positive accepts). This is exactly the instrument gap the scoping
report flagged as **handback #3** and the standing **STOP #A** gate-instrument question. It
is a hand-back, not a degrade — I did not adjust the gate to hide it.

---

## 1. What landed (Deliverables 1–6)

| # | deliverable | module | tests |
|---|---|---|---|
| 1 | detection + routing (shape-based, precision-parameterized) | `agents/patcher/precision_flip.py` | `test_precision_flip.py` (12) |
| 2 | per-integral dd TU + precision-parameterized wrapper generator | `agents/patcher/tu_emit.py` | `test_tu_emit.py` (15) |
| 3 | per-integral dispatch layer (RES-stream selector/merger) | `agents/patcher/flip_dispatch.py` | `test_flip_dispatch.py` (6) |
| 4 | app-boundary narrowing reusing acc1482 | `narrow_two_limb_scalar` in `boundary.py` + flip printer in `tu_emit.py` | `test_flip_boundary.py` (7) |
| 5 | acceptance gate (build AND lift > 0.0, uniform) | `agents/patcher/flip_gate.py` | `test_flip_gate.py` (8) |
| 6 | regression preservation | — | 568 + 65 green; acc1482 26 byte-identical |
| 7 | L-measure harness | `runs/qcdloop/phase1_lmeasure.py` | ran |

**Key design fidelity points:**
- **Decision 2 (no ddfun_enabled as build input):** the per-group dd TUs are built from a
  **clone of the pristine snapshot** + the vendored `third_party/` primitives. `ddfun_enabled`
  is materialized (`git archive`) **only** as the measurement oracle reference, never a
  build input. Confirmed by construction in `phase1_lmeasure.py`.
- **STOP #SS (precision-parameterized, not dd-hardcoded):** `TargetPrecision` + a
  `PROFILES` table drive routing (D1), emission (D2), and dispatch (D3). FF/FLOAT profiles
  are declared but marked unavailable (their maths headers aren't vendored) — selecting one
  **fails loud**, never silently degrades to dd.
- **STOP #TT (one boundary transform):** the dd→caller reconstruction is the single shared
  `narrow_two_limb_scalar`; the element-promotion designed exit and the flip printer both
  call it. Refactor kept all 26 acc1482 output strings byte-identical.
- **STOP #Z:** `emit_flip_tu` refuses any write resolving under `runs/qcdloop_headers_full/`.
  Verified: snapshot `git status` clean after the full run.
- **Group discovery is structural** (`B<k>m.h` shape / the header that defines the
  integral), no baked-in integral→group table (feedback_no_placeholder_patterns).

---

## 2. The per-group dd build result (Deliverable 2/4 — the honest dd builds)

All 5 distinct mass-group dd TUs built **clean** from the snapshot clone (first honest dd
builds for the whole box tree via the pipeline's own surface, not the oracle):

| group | integrals | build |
|---|---|---|
| B0m | BIN0 (+B1-B5) | clean |
| B1m | B10, BIN1 (+B6-B9) | clean |
| B2m | B12, B13, B14, B15, BIN2 (+B11) | clean |
| B3m | B16, BIN3 | clean |
| B4m | BIN4 | clean |

This validates the mechanism end-to-end: the pruned per-group `BO` (no
`QCDLOOP_BOX_FULL_DISPATCH`) isolates each group, so even the B3m/B4m int↔Tracked friction
the scoping flagged for Phase-3 does **not** block a single-precision Phase-1 compile — as
predicted (§3 `Y*`).

---

## 3. L-measure (Deliverable 7) — the numbers, honestly partitioned

2000 samples, min precise-digits per integral (min over all cells), gate margin 0.0.

### 3.1 VALIDATED — dd-sufficient (dd ≈ truth, so vs-dd is a valid reference)

| integral | baseline (raw double) | candidate (dd→double) | lift | verdict |
|---|---|---|---|---|
| **B10** | 12.19 | 15.96 | **+3.77** | genuine |
| **B12** | 9.32 | 15.96 | **+6.63** | genuine |
| **B13** | 8.69 | 15.96 | **+7.27** | genuine |
| **B14** | 0.00* | 15.96 | **+15.96** | genuine |

These are real: dd carries these integrals to ~30 digits (well above the cancellation
loss), so narrowing to double delivers a full-double result and the vs-dd measurement
faithfully reads the ~15.9-digit output floor. B10/B12/B13 are the three the design
targeted; all lift.

\* B14 min-cell baseline 0.00 here vs the design's whole-app ~13.19 — different *scope*
(worst single cell over 12k cells vs whole-app p100), same dd-sufficiency. The min-cell
instrument surfaces B14's own worst-conditioned coefficient, which raw double loses.

### 3.2 🛑 FALSE POSITIVE — dd-insufficient (dd ≠ truth, so vs-dd is CIRCULAR)

| integral | baseline | candidate | "lift" | reality |
|---|---|---|---|---|
| B15 | 0.00 | 15.96 | +15.96 | dd-insufficient: cancellation > dd budget |
| B16 | 0.00 | 15.96 | +15.96 | dd-insufficient + B3m friction |
| BIN0 | 0.00 | 15.96 | +15.96 | dd-insufficient |
| BIN1 | 8.84 | 15.96 | +7.12 | dd-insufficient |
| BIN2 | 9.12 | 15.97 | +6.85 | dd-insufficient |
| BIN3 | 9.38 | 15.96 | +6.58 | dd-insufficient |
| BIN4 | 9.60 | 15.96 | +6.36 | dd-insufficient |

The design (§3, handback #3) predicted these must **reject** — a clean dd build for them is
a false-positive fix because their cancellation exceeds dd's ~32-digit budget. The gate
accepted them anyway. **Why** is the load-bearing finding.

---

## 4. 🛑 STOP #WW — the acceptance instrument is circular for dd-insufficient integrals

**Root cause (proven, not inferred):** the candidate flip binary computes the integral at
dd internally, then narrows to double at the boundary. The L-measure references the **same
dd oracle**. So the candidate is `round_to_double(dd_result)` and the reference is
`dd_result` — the measured "candidate vs dd" error is *exactly the dd→double rounding*,
nothing else. Direct proof at sample 0:

```
B12: candidate = 7.82898288102895969e-12
     dd_oracle_narrowed = 7.82898288102895969e-12   rel diff = 0.00e+00
B15: candidate = -7.14217817574412199e-12
     dd_oracle_narrowed = -7.14217817574412199e-12  rel diff = 0.00e+00
```

`candidate ≡ round(dd_oracle)` bit-for-bit. Therefore `candidate_digits ≈ 15.9` is a
**constant** (spread across all 11 integrals = 0.011 digits), independent of whether dd is
accurate. When dd ≈ truth (§3.1) this constant coincides with the real accuracy and the
lift is genuine. When dd ≠ truth (§3.2) the candidate faithfully reproduces the *wrong* dd
answer to ~15.9 digits and the instrument reports a lift that does not exist against the
true value.

**This is the same instrument gap as STOP #A / handback #3** (whether the acceptance
instrument distinguishes build-success from real digit-lift). The fix is a **reference
change, not a mechanism change**: the dd-insufficient set needs a **quad (or analytic)
reference** to measure against — measuring a dd-computed candidate against a dd reference is
circular by construction for any integral where dd itself is not the truth.

**What I did NOT do (per the dispatch's bans):**
- Did **not** relax or special-case the gate to suppress the false positives (Decision 3 is
  uniform build-AND-lift; bending it to hide the circularity would be the falsifier trap).
- Did **not** synthesize a quad reference or coefficient (§3.4 ban).
- Did **not** claim the dd-insufficient integrals as wins.

---

## 5. Hand-backs for Reet

1. **STOP #WW — acceptance reference for dd-insufficient integrals.** The vs-dd instrument
   is valid only where dd ≈ truth (B10/B12/B13/B14 — confirmed genuine lift). B15/B16/BIN0-4
   need a **quad or analytic reference** before their flip can be accepted or rejected on
   merit. Options: (i) a quad oracle (`USE_QUAD_COMPLEX`, CUDA-only today) as a second
   reference; (ii) an analytic-zero / known-value battery for the BIN* series; (iii) a
   dd-sufficiency **pre-gate** that refuses to *measure* an integral whose static
   conditioning exceeds dd's budget (routes it to raw double untested, honestly marked
   "unvalidatable at dd"). Recommend (iii) as the immediate honest guard + (i)/(ii) for
   real Phase-2 measurement. **This is the STOP #A gate-instrument decision, now concrete.**

2. **B10/B12/B13 are ready to accept** (dd-sufficient, genuine measured lift +3.77/+6.63/
   +7.27). B14 also lifts (+15.96 min-cell) — consistent with STOP #A's "already accurate at
   whole-app scope" once you pick the acceptance scope (min-cell vs whole-app p100).

3. **Realizable Phase-1 lift is capped at double's output floor (~15.9), not +18.43.** The
   design's +18.43 B10 prediction assumed a dd *output*; Phase-1 correctness narrows to the
   caller's double contract at the boundary, so the deliverable is "raw-double baseline →
   full-double accuracy via dd internals," capped at ~15.9. The measured B10 +3.77 is the
   honest realizable lift for a double-output contract. A dd-*output* contract (keeping the
   two limbs across the app boundary) is a Phase-2 endpoint-lock question, not Phase-1.

---

## 6. Artifacts

- Machinery: `agents/patcher/{precision_flip,tu_emit,flip_dispatch,flip_gate}.py`;
  `narrow_two_limb_scalar` in `agents/integrator_base/boundary.py`.
- Tests: `tests/patcher/test_{precision_flip,tu_emit,flip_dispatch,flip_boundary,flip_gate}.py`
  (48 new) + acc1482 26 byte-identical.
- L-measure: `runs/qcdloop/phase1_lmeasure.py`; results
  `runs/qcdloop/phase1_lmeasure_out/phase1_lmeasure.json`; log
  `runs/qcdloop/phase1_lmeasure_run.log`.
- Authorized enrichment: `third_party/include/{dd_math,dd_complex}.hpp` (+10 lines, verbatim
  from `ddfun_enabled`).
