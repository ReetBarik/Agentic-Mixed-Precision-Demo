# WAVE3 CHARACTERIZATION — residual `llm_gen_failed` cluster inventory

**Date:** 2026-07-21 · read-only pass (no `agents/` or `tests/` changes) ·
characterization input to Wave-3 rule/classifier work.

## Run analyzed

- **Run id:** `20260720_054121_dd44d33c` (the PIPELINE_v1 validation run).
- **Config:** faithful — correctness 200 / speedup 250 / dr_k 60 / tol 7 / n=1000;
  DD oracle `~/qcdloop@ddfun_enabled`; tail-augmented `report_10k.json`.
- **Terminal status:** `success`, 231 iterations, 152 validated (152/152 accept).
- **Artifacts:** `runs/qcdloop/strategy/20260720_054121_dd44d33c/`
  (`iterations.jsonl`, per-iter `logs/iter_<N>_build.log`, `patches/`, `shims/`,
  `report.md`).

## Headline finding — the residual is ONE structural bug, not 70 precision gaps

The disproven "component-pair" hypothesis is replaced by hard evidence: **72 of the
79 `llm_gen_failed` events are the same C++ error — `redefinition of
struct ql::Constants<T>` (or a `ql::` helper overload)** — caused by the Patcher
emitting a *full* template specialization / free-function definition per region,
when **only one such definition may exist per translation unit per type**. All box
headers compile into a single TU (`boxGPU_vanilla.cpp` → `boxGPU.h` → B0m…B4m.h),
so the **first** region to materialize `Constants<T>` for a given `T` wins and
**every later region needing that same `T` collides** — regardless of file, line,
region kind, or which named constant it touches.

This is a **Patcher-contract / harness assembly bug**, orthogonal to precision, to
LLM capacity, and to the Wave-1 R3 constant-derivation work. The
`log_tag = "llm_capacity"` on every one of the 79 (dispatch.py:57) is a **misnomer**
— the LLM produced valid code in isolation (probe: 5/5 accept); it collides at TU
assembly.

## Count reconciliation vs PIPELINE_v1

| source | dd | float | ff | total |
|---|---|---|---|---|
| PIPELINE_v1 pre-tail baseline (task prompt) | 40 | 16 | 14 | 70 |
| **this run (tail-augmented, actual)** | **47** | **15** | **17** | **79** |

PIPELINE_v1's own summary table already records the drift (`dd_untested 40 → 47`,
attributed to "Patcher gen noise"). Use **79** as the real residual. 79 attempts
cover **74 distinct** `(file, line, target)` regions — 5 regions were retried once
(see §Transient).

## Step 1 — enumeration (grouped; full 79 rows in `wave3_gen_failures.csv`)

Every `llm_gen_failed` region, grouped by `(file, target)` with the failing lines.
All are `patcher_status=llm_gen_failed`, `log_tag=llm_capacity`, `dd_untested=True`
at the dd rung. Full per-iter rows (iter, kind, file, line, signature, raw error)
are in the committed `runs/qcdloop/wave3_gen_failures.csv`.

| file | target | # lines | lines |
|---|---|---|---|
| B0m.h | dd | 5 | 68, 69, 72, 74, 329 |
| B0m.h | float | 3 | 224, 229, 305 |
| B0m.h | ff | 2 | 229, 305 |
| B1m.h | dd | 12 | 62, 63, 66, 67, 134, 161, 162, 200, 204, 227, 234, 241 |
| B1m.h | float | 7 | 97, 98, 106, 132, 133, 138, 167 |
| B1m.h | ff | 8 | 61, 97, 98, 106, 132, 133, 138, 167 |
| B2m.h | dd | 9 | 64, 84, 204, 206, 300, 355, 400, 405, 493 |
| B2m.h | float | 4 | 105, 106, 109, 188 |
| B2m.h | ff | 4 | 105, 106, 109, 188 |
| B3m.h | dd | 4 | 76, 105, 177, 183 |
| B3m.h | float | 1 | 97 |
| B3m.h | ff | 1 | 97 |
| B4m.h | dd | 6 | 163, 184, 192, 195, 198, 233 |
| B4m.h | ff | 1 | 126 |
| kokkosUtils.h | dd | 6 | 183, 212, 401, 672, 704, 753 |
| kokkosUtils.h | ff | 1 | 580 |

Note the shape that gives the mechanism away: **B1m.h has 12 failing dd lines**,
B2m.h 9, B0m.h 5 — because each header is a single collision domain for its `T`.
Exactly **one** `Constants<T>` definition per `T` survives per TU (measured on the
accepted shims):

| type | accepted specs in TU | failed (collision) attempts |
|---|---|---|
| `Constants<quad::ddfun::ddouble>` | 1 | 36 |
| `Constants<float>` + `Constants<std::complex<float>>` | 2 | 15 |
| `Constants<quad::ffun::ffloat>` + `Constants<quad::ffun::ffcomplex>` | 2 | 16 |

(Of 67 accepted dd regions, **exactly 1** defines `Constants<ddouble>`; the other
66 are plain type-edits that reuse the one specialization. Same 1-per-type story
for float/ff.)

## Step 2 — cluster taxonomy (defined by the actual last-attempt error)

Signatures extracted by `slice_gen_failures.py` from each failed iter's build log.

| # | cluster | count | kinds | representative error |
|---|---|---|---|---|
| C-COLL | **TU symbol-redefinition collision** | **72** | 37 dd / 15 float / 16 ff (Constants) + 4 dd (helpers) | `error: redefinition of 'struct ql::Constants<quad::ddfun::ddouble>'` |
| C-DUP | Codegen defect — duplicate qualifier | 3 | dd | `error: duplicate 'inline'` |
| C-R4 | R4 constant-derivation escape | 3 | 2 dd / 1 ff | `#error "DD Regional Integrator: ql::cLn requires manual classification"` |
| C-TID | Codegen defect — template-id mismatch | 1 | dd | `template-id 'Real<quad::ddfun::ddouble>' ... does not match any template declaration` |

Detail on the clusters:

- **C-COLL (72) — the whole story.** 68 are `redefinition of Constants<T>`; 4 more
  are `redefinition of ql::Real / ql::Lnrat / ql::iszero` — the *same* root cause,
  a shim re-defining a `ql::`-namespace helper another shim already injected into
  the TU. Deterministic and structural: the second-and-later definition of any
  TU-global symbol always collides. (Name spelling is irrelevant — the LLM
  sometimes writes `Constants< ::quad::ddfun::ddouble >` with a leading `::`, but
  it is the same type to the compiler and still collides.)
- **C-DUP (3):** `duplicate 'inline'` — the shim emitted `inline` on a member that
  already carries a `KOKKOS_INLINE_FUNCTION`/`inline` macro. A pure text defect.
- **C-R4 (3):** genuine transcendental-valued "constants" — `ql::cLn`,
  `ql::Li2omx2`, `ql::Li2omrat` — where the value depends on a runtime argument, so
  the R3 cascade legitimately reaches the R4 manual-classification escape. These are
  *not* the `_ieps50` family Wave-1 targeted (see §Step 3).
- **Non-LLM cause: 0.** No env / header-ordering / upstream-break failures — the
  a29472d shim-ordering blocker and the 20879dc filename-collision fix are holding;
  every failure is content in an LLM-generated shim.

## Step 3 — Wave-1 cross-reference (stale vs actually-broken)

The Wave-1 R3 tightening (`d14e41b` sanitize + `97fca0a` R3-discipline +
`e07910d`) works on the `_ieps50` sub-cluster (its probe: 3/3 accept). Cross-
referencing that against this residual:

- **The `_ieps50` family is present but fails for a different reason.** B1m.h:62/63
  and B2m.h:64/65 have *identical* source (`ql::Constants<TScale>::_ieps50<...>()`
  + `_one()`). In this run they fail with **`redefinition`, not R4-escape** — the
  shim derives `_ieps50` correctly (`ddouble(1e-50)`, verified in the shim body);
  it simply collides. So Wave-1's `_ieps50` fix is **irrelevant to why they fail**.
  They are **not "stale-and-now-passing"** — they would collide identically today.
- **Only C-R4 (3 regions) is Wave-1-adjacent**, and those are cLn/dilog manual-
  classification, a harder class than `_ieps50`.
- **Net Wave-1-stale count for the residual: ≈ 0.** Re-running the 10k reproduces
  all 72 collisions bit-for-bit; Wave-1 cannot move them because the collision is
  *upstream* of (and binds before) any constant-derivation quality gain.

The two step-3 possibilities resolve cleanly: the collision regions were
**re-attempted and still failed** (deterministic), not stale.

## Step 4 — probe (5 regions, dd-integrator, isolation off pristine base)

`wave3_probe.py` drove `make_patcher_fn` directly, one dd intent per region, each
off the **pristine base** (so each shim is the *first* into its TU — no sibling).
This isolates region-intrinsic generation from the TU-assembly collision.

| region | 10k failure class | isolation result | interpretation |
|---|---|---|---|
| B1m.h:63 | C-COLL (Constants) | **ok** | structural — accepts alone, collides only with a sibling |
| B2m.h:84 | C-COLL (Constants) | **ok** | structural |
| B1m.h:62 | C-DUP (duplicate inline) | **ok** | gen defect no longer reproduces |
| B4m.h:184 | C-R4 (cLn manual class.) | **ok** | R4 escape no longer reproduces |
| B3m.h:105 | C-TID (Real template-id) | **ok** | gen defect no longer reproduces |

**5/5 accept in isolation.** The split:

- **Collision regions (C-COLL, 72):** accept in isolation → the failure is
  **sibling-context-dependent (structural)**, not region-intrinsic and not stale.
  Note the attribution caveat: for this cluster, "isolation accept" does **not**
  mean the failure was stale — it means the failure only exists when a sibling
  already occupies `Constants<T>` in the TU. The 10k build logs are the
  reproduction; isolation cannot reproduce it by construction.
- **Defect/R4 regions (C-DUP/C-R4/C-TID, 7):** now generate clean in isolation
  post-Wave-1 + backoff → their *generation* is fixed. **But** their generated
  shims still define `Constants<T>` for `_one`/`_two`, so on a real 10k they would
  **still collide** as the 2nd+ specialization. They are gated by C-COLL too.

**"Already-fixed-was-stale" vs "actually-broken" split:** generation-side, 7/7
sampled defect classes are fixed; residual-side, **79/79 are gated by the one
structural collision**, which is untouched by any generation fix.

## Step 5 — Wave-3 lever recommendation (ranked by coverage)

1. **[Primary — moves 72 directly, unblocks all 79] TU-global symbol dedup in the
   shim contract.** Stop emitting a *full* `template<> struct Constants<T>` (and
   full `ql::` helper overloads) per region. Instead accumulate members into **one
   merged `Constants<T>` specialization per (TU, T)**. Candidate mechanisms (Wave-3
   design choice — no code changed in this pass):
   - *Merge-into-existing:* when the region needs a `Constants<T>` member and a
     specialization for `T` already exists in the TU, append the member to it rather
     than emitting a new `template<>`.
   - *Harness assembly pass:* union all `*_dd_*`/`*_ff_*`/`*_float_*` Constants
     shims for a given `T` into a single generated header before the build gate.
   - *Avoid the specialization entirely:* route named constants through free
     functions keyed on `T` (ADL) instead of `Constants<T>` members, so there is no
     one-definition-per-TU constraint.
   Expected coverage: **72/79 direct**; combined with the already-landed Wave-1 gen
   fixes it plausibly clears the **full 79**.

2. **[Secondary — retry economics + honest telemetry] Misgen classifier at
   `dispatch.py:67`.** The ground-truth-flagged lever. Two parts:
   - *Re-tag:* classify the build error before tagging. `redefinition of
     Constants<T>` / `ql::` helper is a **known structural collision**, not
     `llm_capacity` and not a DD physics ceiling. Emitting `llm_capacity` +
     `dd_untested="investigate"` for these is misleading telemetry.
   - *Short-circuit:* collisions are deterministic, so retrying (and backoff between
     retries) is pure waste — recognize the signature and skip the retry loop until
     the dedup lands. Expected coverage: moves **retry economics** (wasted attempts
     + wall) on all 72; does **not** raise the accept count on its own.

3. **[Tertiary — already landed, just confirm] C-R4 / C-DUP / C-TID (7).** No new
   lever — the probe shows all 7 now generate clean in isolation post-Wave-1. They
   are unblocked automatically once C-COLL is fixed. Confirm on the next faithful
   10k (no separate work).

**Coverage honesty:** without lever #1, **no** lever moves the accept count — a
re-run reproduces all 72 collisions identically, Wave-1 is already landed and
cannot help (collision is upstream), and backoff is timing-only. Lever #1 is
load-bearing; #2 is efficiency/telemetry; #3 is free.

## What NOT to do (hypotheses this pass rules out)

- **Do not pursue the "component-pair" (split real/imag) hypothesis** — already
  disproven (Wave-2 grep: 0 matches) and re-confirmed: zero of the 79 failures
  involve a real/imag component pair; the taxonomy is entirely collision + a few
  gen defects.
- **Do not treat these as `llm_capacity`.** 0/79 are capacity/rate-limit/refusal.
  The proxy is a stable local mock; `llm_capacity` is a dispatch label misnomer for
  `llm_gen_failed`.
- **Do not add more R3/prompt tightening for the residual.** The shims already
  generate valid code (probe 5/5). More prompt discipline cannot fix a
  one-definition-per-TU C++ rule. Wave-1 R3 work is complete for this cluster.
- **Do not re-run the 10k expecting the dd_untested cluster to shrink.** The
  collision is deterministic; a re-run reproduces all 72. Only a dedup change moves
  it.
- **Do not tune backoff to raise accepts.** Backoff is timing-only and cannot
  change accept counts on a deterministic collision on a stable proxy (it only
  redistributes attempts; per-attempt logs confirm the terminal wall is always the
  collision).
- **Do not treat `dd_untested` as a DD physics/precision ceiling.** It is a
  code-assembly bug (`regions_at_dd_ceiling = 0` is correct — there is no genuine DD
  ceiling here).

## Artifacts committed with this report

- `runs/qcdloop/slice_gen_failures.py` — log-slicing helper (iterations.jsonl +
  build logs → clustered signatures + CSV).
- `runs/qcdloop/wave3_gen_failures.csv` — full 79-row enumeration.
- `runs/qcdloop/wave3_probe.py` + `runs/qcdloop/wave3_probe.log` — step-4 isolation
  probe (5/5 accept).
