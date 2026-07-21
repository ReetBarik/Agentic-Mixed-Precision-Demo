# Integrator system-prompt fix — sanitize + R3 cascade discipline (2026-07-21)

Two-part edit to the integrator system prompts, targeting the generation-robustness
ceiling in `runs/qcdloop/CALIBRATION_v2.md` (40 `dd_untested`, 16 float gen-misses,
14 ff gen-misses on the faithful 10k — all `llm_gen_failed`, zero tolerance rejects).
No new rules added this pass; we are testing whether tightening existing rules is
enough before adding C8/C9 for the split-component-complex pattern.

Branch: `langgraph-agents` (pulled — already up to date). Two commits, kept separate
for bisect:

- `d14e41b` prompts: sanitize qcdloop-flavored identifiers from integrator system prompts
- `97fca0a` prompts: tighten R3 cascade discipline (dd + ff) — mandatory step-by-step walk

---

## 1. Sanitization diff summary

Mechanical rename of qcdloop-specific identifiers to structural placeholders. Surrounding
sentence structure and worked-example shape unchanged — only the names swap.

| From | To |
|------|-----|
| `_ieps50` (dd, ff) / `_tiny` (float) | `_small_reg` |
| `TScale(1e-50)` (dd, ff) | `AppScalar(1e-50)` |
| `TScale(0.125)` (float) | `AppScalar(0.125)` |
| `ql/constants.h` | `<app>/constants.h` |
| `ql/maths.h` | `<app>/maths.h` |
| `qcdloop/*.h` | `<app>/*.h` |
| `ql::Constants<T>` | `<AppNs>::Constants<T>` |
| `ql::Max` | `<AppNs>::Max` |
| `ql::kAbs` | `<AppNs>::kAbs` |

`git diff --stat` (commit `d14e41b`):

```
 agents/dd_integrator/system_prompt.txt    | 4 ++--
 agents/ff_integrator/system_prompt.txt    | 4 ++--
 agents/float_integrator/system_prompt.txt | 4 ++--
 3 files changed, 6 insertions(+), 6 deletions(-)
```

Per-file string replacements (2 changed lines each — the R3.3a example line and the C1
include/namespace line):

| File | 3a example line | C1 line | total string swaps |
|------|-----------------|---------|--------------------|
| `agents/dd_integrator/system_prompt.txt` | `_ieps50`, `TScale(1e-50)` | `ql/constants.h`, `ql/maths.h`, `qcdloop/*.h`, `ql::Constants<T>`, `ql::Max`, `ql::kAbs` | 8 |
| `agents/ff_integrator/system_prompt.txt` | `_ieps50`, `TScale(1e-50)` | (same 6) | 8 |
| `agents/float_integrator/system_prompt.txt` | `_tiny`, `TScale(0.125)` | (same 6) | 8 |
| `agents/tracked_integrator/system_prompt.txt` | — | — | **0 (untouched)** |

**`tracked_integrator/system_prompt.txt` was NOT modified.** Its only match against the
sanitization list is the bare `TScale` on line 65 (Rule C7), which appears four times, all
as a generic `template <class TOutput, class TMass, class TScale>` template-*parameter*
name — kept per spec. It contains no `ql::…`, no `qcdloop/…`, no `_ieps50`. So the
"apply to all four where the string appears" rule yields zero edits for tracked. Confirmed
`grep` for `ql[:/]`, `qcdloop`, `_ieps50`, `_tiny` returns nothing in that file.

Post-edit `grep` for any residual qcdloop-flavored string across dd/ff/float returns clean.

---

## 2. R3 tightening (dd + ff prompts only)

The R3 intro sentence (line 15 in both `dd_integrator` and `ff_integrator`) was replaced.
The old intro ended with a permissive "…through the following cascade, IN ORDER — use the
FIRST step that applies, and never fall to the Rule R4 escape hatch (step 4) without stating
which of steps 1–3 you tried…". It now ends "…resolves to its ddouble/ffloat value through
the four-step cascade below." followed by a new **mandatory** block:

```
Cascade discipline — MANDATORY, no exceptions:
  - Walk steps 1 → 2 → 3 → 4 in order. Stop at the FIRST step that applies.
  - Step 4 (the R4 escape hatch) is FORBIDDEN unless you have explicitly considered and
    rejected steps 1, 2, AND 3 for this specific constant, and you write those three
    rejections as comments in the emitted shim.
  - "I am not sure whether step 3 applies" is NOT a valid rejection of step 3. If the
    constant's source definition is visible or supplied, step 3 applies — walk it.
  - A source literal defined as a plain double/float in the app source ALWAYS satisfies
    step 3a — no exception for small magnitudes, no exception for values near underflow,
    no exception for regulators. The double/float representation IS the exact intended
    value; a zero low word is CORRECT.      <-- dd
                                            ; its two-limb ffloat split is CORRECT.  <-- ff
```

- **Steps 1, 2, 3a, 3b, 4 are unchanged** modulo the commit-1 sanitization renames inside
  3a (`_ieps50`/`_tiny` → `_small_reg`, `TScale(…)` → `AppScalar(…)`). Verified by reading
  lines 15–35 of both files after the edit.
- **dd** uses the exact spec text (its existing parenthetical already matches: "carries
  only ~16 digits and silently truncates the low word…").
- **ff** — two deliberate, precision-faithful deviations from a verbatim dd copy, flagged
  here for review:
  1. Kept ff's own intro parenthetical ("narrows to ~7 float / ~16 double digits and
     defeats the promotion") rather than importing dd's low-word wording, which is
     dd-specific.
  2. Final bullet reads "**its two-limb ffloat split is CORRECT**" instead of "a zero low
     word is CORRECT". Rationale: a `double` source literal split across two `float` limbs
     generally has a **nonzero** low limb (matches ff's step 3a: "the ctor splits the value
     across the two float limbs"). "Zero low word" is a double-double–only property; copying
     it into ff would inject a factually wrong claim, which is the opposite of the
     robustness goal. This matches ff's already-existing 3a language.

`git diff --stat` (commit `97fca0a`):

```
 agents/dd_integrator/system_prompt.txt | 8 +++++++-
 agents/ff_integrator/system_prompt.txt | 8 +++++++-
 2 files changed, 14 insertions(+), 2 deletions(-)
```

---

## 3. Test results (steps 1–3, 5)

Baseline before edits: **386 passed** (full suite; the suite has grown past the ~349 in
CALIBRATION_v2 with the wave-3 + tail-testing work). No test reads `system_prompt.txt` or
asserts on prompt content — `grep -rl system_prompt tests/` returns nothing — so no test
needed touching. (The many `_ieps50` / `ql::…` hits under `tests/` are fixture kernels and
expected-shim assertions that use qcdloop as the *test application*; those are inputs, not
prompt-substring assertions, and are correctly left alone.)

| Step | Scope | Result |
|------|-------|--------|
| 1 | `pytest tests/` (full suite, post-edit) | **386 passed** (unchanged from baseline) |
| 2 | `tests/{dd,ff,float,tracked}_integrator/ tests/integrator_base/` | **91 passed** |
| 3 | `tests/patcher/` | **41 passed** |
| 5 | `tests/validator/` | **37 passed** |

No failures, no tests modified.

---

## 4. Step 4 — generation probe on the small-regulator cluster

Harness: `runs/qcdloop/rerun_failing_regions.py` (no arg surface — it drives
`make_patcher_fn` directly over a hardcoded region list). Rather than editing the harness
(out of scope per ground rules), I ran it as-is; its region list already contains three of
the `_ieps50`/small-regulator cluster regions named in the task
(`box/B0m.h:69`, `box/B2m.h:65`, `box/B1m.h:62`) plus one include-only control
(`box/B3m.h:177`). Env: `.venv` + `gcc/13.3.0` + `cmake/3.28.3` + proxy on `:8084`.

**Region picked for the detailed write-up: `box/B0m.h:69` (BIN0, `_ieps50` complex iε
regulator).**

**Verdict: accept.** All four regions: `status=ok`, clean include set, shim placed after
app includes, build gate passed, **`no-R4`** (no `#error` escape hatch emitted).

```
=== SUMMARY ===
  box/B3m.h:177    status=ok   placement=OK(after-app-includes)   sig=ok
  box/B2m.h:65     status=ok   placement=OK(after-app-includes)   sig=ok
  box/B0m.h:69     status=ok   placement=OK(after-app-includes)   sig=ok
  box/B1m.h:62     status=ok   placement=OK(after-app-includes)   sig=ok
  built (P2 ok)        : 4/4
  shim-ordering blocker: 0/4
```

Success criteria for the probe, checked against the emitted `B0m.h:69` shim:

- ✅ Generation completed without `llm_gen_failed` (`status=ok`).
- ✅ Emitted shim compiles (build gate `sig=ok`).
- ✅ Shim comments reference the constant by its source name — `_ieps50` (and `_one`),
  citing `Rule 5 / R3-step-3a` and `R3-step-3a` explicitly.
- ✅ Constant materialization uses `make_dd(...)` — **not** the R4 `#error` hatch.

Constant-materialization lines from the emitted shim (the exact pattern the fix targets —
the small `1e-50` regulator derived via R3.3a with a **zero low word**, the complex
regulator derived whole):

```cpp
  // _ieps50 : TOutput{_zero(), TScale(1e-50)} — a complex iε regulator.
  // Rule 3 (container of FP) + R3-step-3a (source double literal 1e-50 -> low word 0).
  template <class TOutput_, class TMass_, class TScale_>
  static inline quad::ddfun::ddcomplex _ieps50() {
    return quad::ddfun::ddcomplex(
      quad::ddfun::make_dd(0x0000000000000000ULL, 0x0000000000000000ULL),  // real = 0
      quad::ddfun::make_dd(0x358dee7a4ad4b81fULL, 0x0000000000000000ULL)   // imag = 1e-50, zero low word
    );
  }
```

`0x358dee7a4ad4b81f` is the IEEE-754 double bit pattern of `1e-50`; the zero low word is
exactly what the tightened R3.3a discipline now declares correct. (This is also the
bit-exact value documented as the Wave-2 `_ieps50` fix.) The shim identifiers
`ql::Constants` / `ql::Max` / `ql::kAbs` / `_ieps50` are the **real qcdloop app** symbols
(correctly preserved by the model) — only the prompt's illustrative *examples* were
sanitized, and the probe confirms that swap did not degrade the model's handling of the
concrete app names.

Probe log: `/tmp/prompt_probe_2026-07-21.log`; kept workdir:
`/tmp/rerun_failing_gv01v94e`.

---

## 5. Recommendation

**This looks sufficient to justify a faithful 10k re-measure; C8/C9 are not needed yet.**

The last faithful 10k left 40 `dd_untested` regions whose *whole* residual gap was the
Patcher escaping to R4 on exactly the small-regulator constants that R3.3a already named
(CALIBRATION_v2: "`_ieps50` still dd_untested … the whole residual gap is Patcher
gen-robustness, not precision"). The probe shows all three previously-`dd_untested` BIN0/BIN1
`_ieps50` regions now generate, materialize the regulator via `make_dd(…,0x0)`, and build —
with zero R4 escapes. That is the precise failure mode the R3 tightening was written to
close, and it closed on the sampled cluster.

Caveat before spending 10k: the probe covers 3 regions of the ~40-region cluster, single
seed each. Recommend the faithful 10k re-measure as the real test (reframed pass =
total demotions ≈ prior ~88 + zero correctness regressions, not the >10-bucket-deviation
metric). If any `_ieps50`-family region still lands `dd_untested` at 10k scale, *that*
is the signal to add C8/C9 for the split-component-complex pattern — but the evidence here
says try the existing-rule tightening at scale first.

No 10k re-run performed in this pass (per instructions).
