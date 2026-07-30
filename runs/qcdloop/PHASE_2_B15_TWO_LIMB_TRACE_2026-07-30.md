# Phase-2 — B15/B16 two-limb trace: is the hard-0.0 a dd-compute bug or a boundary consequence? — 2026-07-30

**Task.** Determine whether the B15/B16 "hard 0.0 at comp 0 sample 0" reported in
`PHASE_2_TU_E2E_REFRESHED_ORACLE_2026-07-30.md` §3.2 is (Case 1) a dd-compute bug in the flip TU, or
(Case 2) a legitimate consequence of narrowing dd→double at the app boundary. **Diagnose only, no fix.**

Working branch `langgraph-agents`, HEAD `0d7e000`. Experimental hi|lo printer patch applied to
`agents/patcher/tu_emit.py` for this run only, **reverted at the end (git clean, confirmed §6)**.

---

## 0. TL;DR — the premise was wrong; verdict is **neither Case 1 nor Case 2 as framed**

> **The "hard 0.0 at B15/B16 comp 0 sample 0" does not exist.** It was an artifact of the *prior
> diagnostic script* (`PHASE_2_TU_E2E_REFRESHED_ORACLE` §3.2), which read B15 from the **wrong group's
> flip binary**. B15 lives in `box/B2m.h`, not `box/B3m.h`; a flip TU that includes only `B3m.h` emits
> **all-zeros** for every out-of-group integral (its pruned `BO` is dead code). The prior script
> measured B15 against `flip_B3m_dd` → all-zero → "0.0". The production e2e provider maps groups
> **correctly** (`_group_for`), so this bug never affected routing — only the prior §3.2 narrative.
>
> **At comp 0 sample 0, the candidate dd compute matches the oracle to ~15+ digits** (B15:
> `-7.142178175744122e-12`, agreeing to rel 2e-23; B16: `-7.146737069161155e-12`, rel 1e-29). The
> value survives the boundary fine. There is no bug and no boundary loss at that cell.
>
> **What actually drives B15/B16/BIN0 to 0.0 min-digits** is a *different* set of cells — deep
> **sub-double noise-floor components** (|value|/scale ≈ 1e-21…1e-33, far below double's 1e-16
> resolution of the sample scale). There the candidate and oracle disagree, but **both values are
> numerical noise** that no double-typed caller could ever carry. This is closest to **Case 2 in
> spirit** (the disagreement is below the double boundary and irrelevant to the caller), but the
> mechanism is *sub-double-noise metric sensitivity*, not `(A, −A+tiny)` limb cancellation.

**Corrected finding:** B15/B16 are **not** candidate-flip correctness bugs. On every component double
can actually resolve, the candidate agrees with the oracle to 12.5–15.6 dd-digits. The 0.0 is a
**metric artifact at the sub-double noise floor** — the same class as B14, exposed by `_min_digits`
running **without `ref_scale`** *and* by the metric having no floor below which noise-vs-noise
disagreement is ignored.

---

## 1. Group-mapping correction (the root of the prior §3.2 error)

Structural group discovery (`tu_provider._group_for`, regex `\bvoid\s+<integral>\s*\(`) over the clone:

| integral | actual group | prior §3.2 assumed | flip binary that HAS it |
|---|---|---|---|
| **B15** | `box/B2m.h` | ❌ `box/B3m.h` | `flip_B2m_dd` |
| **B16** | `box/B3m.h` | ✓ `box/B3m.h` | `flip_B3m_dd` |
| BIN2 | `box/B2m.h` | — | `flip_B2m_dd` |
| BIN0 | `box/B0m.h` | ✓ | `flip_B0m_dd` |
| B14 | `box/B2m.h` | ✓ (it said B2m) | `flip_B2m_dd` |

Proof that a wrong-group binary emits all-zeros for an out-of-group integral (this is what the prior
script fell into for B15):

```
# flip_B3m_dd asked for B15 (WRONG group — B15 is in B2m):
RES,B15,0,0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000
# flip_B2m_dd asked for B15 (CORRECT group):
RES,B15,0,0xbd9f6960b3bf44b8,0xbde1ac3c9aa6a4ed,0xbdb1e8c73db61cb1,0x3d9b393505326cd2,0x0000000000000000,0x0000000000000000
```

The all-zero line is the pruned `BO` being dead in a TU that doesn't include B15's group header — a
build-topology fact, not a compute result. **The production provider never does this** (it builds
`flip_B2m_dd` for B15), so routing was correct; only the prior report's §3.2 cell-level story was wrong.

---

## 2. Two-limb trace at comp 0 sample 0 (the cell the task named)

Experimental hi|lo printer substituted into `_printer_struct` (dd branch), fresh flip TUs built into a
scratch tree (`runs/qcdloop/b15_two_limb_scratch`, snapshot untouched), run standalone at
`--sample-count 5000` (seed 12345 baked into the shared recipes). Oracle = refreshed
`ddfun_enabled@d11a94b` `boxGPU_app` (`dd_build`). Production single-token = prior TU e2e build.

### B15 — sample 0, comp 0 (`coeff0.real`)

| source | hi | lo | (double)hi+(double)lo | Decimal(hi)+Decimal(lo) |
|---|---|---|---|---|
| **Oracle** | `-7.142178175744122e-12` | `3.976585618844312e-28` | `-7.142178175744122e-12` | `-7.1421781757441215911348…e-12` |
| **Cand (hi\|lo)** | `-7.142178175744122e-12` | `3.9765870540571566e-28` | `-7.142178175744122e-12` | `-7.1421781757441215911347…e-12` |
| **Cand (prod, narrowed single token)** | — | — | `-7.142178175744122e-12` | — |

`hi` bit-identical; `lo` differs at ~1e-28 (the dd tail). **rel |cand−orc|/|orc| = 2.0e-23** → ~15.7
digits at the two-limb level. The **narrowed double delivered to the caller is the correct value** —
NOT 0.0.

### B16 — sample 0, comp 0 (`coeff0.real`)

| source | hi | lo | (double)hi+(double)lo | Decimal |
|---|---|---|---|---|
| **Oracle** | `-7.146737069161155e-12` | `-1.4459888054841047e-28` | `-7.146737069161155e-12` | `-7.14673706916115538…e-12` |
| **Cand (hi\|lo)** | `-7.146737069161155e-12` | `-1.4459888054832868e-28` | `-7.146737069161155e-12` | `-7.14673706916115538…e-12` |
| **Cand (prod, narrowed)** | — | — | `-7.146737069161155e-12` | — |

`hi` bit-identical; `lo` differs at ~1e-44. **rel = 1.1e-29** → ~29 digits. Boundary delivers the
correct value.

**Both named cells: candidate ≈ oracle to 15–29 digits, value survives narrowing. No bug, no 0.0.**

---

## 3. What actually drives the 0.0 — signal vs sub-double noise

Classifying **all 30 000 cells** (5000 samples × 6 comps) per integral by whether the oracle value is
resolvable by double at the sample scale (`|oracle|/scale > 1e-16`) vs sub-double noise, and measuring
the candidate's dd agreement on the **resolvable-signal** cells:

| integral | grp | resolvable-signal cells | sub-double-noise cells | **worst signal-cell dd-agreement** |
|---|---|---|---|---|
| B15 | B2m | 19 987 | 10 013 | **14.86 digits** (s4228 c1, \|orc\|/scale=2.4e-2, rel 1.4e-15) |
| B16 | B3m | 19 989 | 10 011 | **12.55 digits** (s4553 c1, \|orc\|/scale=3.0e-8, rel 2.8e-13) |
| BIN2 | B2m | 10 000 | 20 000 | **13.86 digits** (s4150 c1, \|orc\|/scale=6.8e-5, rel 1.4e-14) |
| BIN0 | B0m |  9 668 | 20 332 | **15.53 digits** (s1955 c0, \|orc\|/scale=3.7e-2, rel 3.0e-16) |
| B14 | B2m | 19 988 | 10 012 | **15.58 digits** (s3605 c3, \|orc\|/scale=1.3e-1, rel 2.6e-16) |

> **On every cell double can actually represent, the candidate dd compute agrees with the oracle to
> 12.5–15.6 dd-digits.** The candidate is computing correctly. (BIN2 and B14 are shown as controls —
> BIN2 routes fine at 13.86, and B14 is the known metric artifact; both confirm the classification.)

The `_min_digits` 0.0 comes only from the **sub-double-noise cells**. The two cells that set each
integral's min-with-`ref_scale` to 0.0:

| integral | cell | oracle dd | candidate dd | \|orc\|/scale | vs ZERO_REF_TOL (1e-24) |
|---|---|---|---|---|---|
| B15 | s4553 c1 (c0im) | `-8.69e-28` | `-1.51e-38` | 8.6e-21 | above → treated as "genuine" |
| BIN0 | s536 c1 (c0im) | `-4.24e-33` | `1.51e-41` | 9.3e-23 | above → treated as "genuine" |

Both oracle values sit at ~1e-27…1e-33 while the sample scale is ~1e-7…1e-11 — i.e. **8–23 orders of
magnitude below the scale**, far under double's 1e-16 resolution. A double-typed caller physically
cannot carry this component: it is rounding noise of the sample, and the oracle's own dd value here is
itself noise (the two dd compute paths' roundoff differs at ~1e-28, which the relative metric reads as
rel≈1 → 0 digits). The candidate's `1e-38`/`1e-41` and the oracle's `1e-28`/`1e-33` are **both noise**;
their disagreement is meaningless to the caller.

Note these sit **just above** `ZERO_REF_TOL = 1e-24` (8.6e-21, 9.3e-23), so the metric's analytic-zero
guard does **not** fire even with `ref_scale` — which is why B15/BIN0 read 0.0 while B14 (worst cell at
5.8e-33, below threshold... actually 5.8e-33 < 1e-24) and B16 (12.55) restore. The threshold is
catching some noise cells and missing others by a few orders of magnitude.

---

## 4. Verdict per integral

| integral | prior §3.2 claim | corrected verdict | Case |
|---|---|---|---|
| **B15** | candidate-flip correctness bug (hard 0.0 at c0 s0) | **NOT a bug.** c0 s0 = correct to 15.7 digits. 0.0 comes from a sub-double-noise cell (s4553 c1, 8.6e-21 of scale). Candidate agrees on all resolvable signal to ≥14.86 digits. | **Case 2-like** (sub-double noise, not a bug; not literal limb cancellation) |
| **B16** | same bug (same group as B15) | **NOT a bug**, and **not even in B15's group** (B16∈B3m, B15∈B2m). c0 s0 correct to 29 digits. With `ref_scale` B16 min = **12.55** — it would *route dd* if `_min_digits` passed `ref_scale`. | **Case 2-like** / metric artifact |
| **BIN0** | value miss above ZERO_REF_TOL | **Confirmed sub-double noise** (s536 c1, 9.3e-23 of scale). Resolvable signal agrees to 15.53 digits. | **Case 2-like** |

**Case 1 (dd-compute bug) is REJECTED for all three.** The `(hi, lo)` limbs of the candidate match the
oracle on every double-resolvable component; where they differ, both are sub-double noise.

**Case 2 as literally defined (a cancellation pair `(A, −A+tiny)` whose double sum rounds to 0.0) is
also not what happens** — the narrowed double at the named cells is the *correct* value, not 0.0. The
true mechanism is a third thing: **the metric (`_min_digits`) has no `ref_scale` and no sub-double-noise
floor, so it reports 0 digits for noise-vs-noise disagreement at components 8–23 orders below the
sample scale — components a double-typed caller never sees.** That is a *metric* limitation, faithful to
Case 2's spirit (the disagreement is below the app boundary and irrelevant), not a compute defect.

---

## 5. Why this matters for routing (no fix applied)

The four analytic-zero integrals split, corrected, as:

- **B14, B16** — restore to dd-worthy **with `ref_scale`** (15.58, 12.55). Pure `_min_digits`-missing-
  `ref_scale` metric artifact.
- **B15, BIN0** — do **not** restore even with `ref_scale`, because their worst noise cell (8.6e-21,
  9.3e-23 of scale) sits just **above** `ZERO_REF_TOL = 1e-24`. These need *either* a lower/rescaled
  zero threshold *or* a sub-double-noise floor (ignore components below ~1e-16 of scale, which double
  cannot represent anyway) to be recognized as the noise they are.

Follow-ups for Reet (NOT done here — diagnose-only):

1. Plumb `ref_scale` (per-sample max|component|) into `tu_provider._min_digits`'s
   `precise_digits_fast` call — matches the Validator's `_score`. Restores **B14 and B16** to dd.
2. For **B15/BIN0**, decide the sub-double-noise policy: the disagreeing components (~1e-21…1e-23 of
   scale) are unrepresentable by double. Options: (a) a sub-double-resolution floor in the metric
   (skip components with `|oracle|/scale < ~1e-16`), or (b) revisit `ZERO_REF_TOL` — but note 8.6e-21
   is genuinely 3 orders above the 1e-24 floor, so lowering the floor would also reclassify real
   deep-signal elsewhere; the resolution floor (a) is the cleaner instrument.
3. Correct `PHASE_2_TU_E2E_REFRESHED_ORACLE_2026-07-30.md` §3.2/§7: B15 is **not** a candidate-flip
   bug and is **not** in group B3m; the "5.5%-of-scale real signal → hard 0.0" claim was the
   wrong-group-binary artifact.

---

## 6. Constraints honored

- **Experimental printer patch NOT committed.** `git checkout agents/patcher/tu_emit.py` run at end;
  `git status --porcelain agents/patcher/tu_emit.py` **empty** ✅. Printer back to the
  `narrow_two_limb_scalar` single-token form (verified: lines 297/306).
- **Oracle + primitives pristine.** `third_party/` porcelain empty ✅; `~/qcdloop` on `ddfun_enabled`
  (`d11a94b`), no source touched — experiment reads the committed tree via the standalone binaries only.
- **Snapshot pristine.** `runs/qcdloop_headers_full` porcelain empty ✅; experiment built into
  `runs/qcdloop/b15_two_limb_scratch/tree` (a copy).
- **Tolerance / sample setup unchanged** (7.0, 5000, seed 12345).
- **Diagnose-only** — no metric, provider, or strategy change made.

### Provenance
| | |
|---|---|
| HEAD | `0d7e000` (unchanged) |
| oracle | `~/qcdloop@ddfun_enabled` = `d11a94b`, `:src/qcdloop` tree `bc3f792` |
| experimental builds | `runs/qcdloop/b15_two_limb_scratch/{flip_build_B2m_dd,flip_build_B3m_dd,flip_build_B0m_dd}` (hi\|lo printer, uncommitted) |
| production candidate | `runs/qcdloop/tu_e2e_out_refreshed/flip_build_*_dd` (prior TU e2e, narrowed single-token) |
| kokkos | `~/kokkos-install-openmp` (OpenMP), OMP_NUM_THREADS=16 |
