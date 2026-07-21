# Robustness Bundle #2 — C8 (dropped) + retry backoff (2026-07-21)

Second robustness bundle attacking the generation-robustness ceiling
(`runs/qcdloop/PIPELINE_v1.md` / `CALIBRATION_v2.md`: 40 `dd_untested` + 16 float
gen-misses + 14 ff gen-misses, all `llm_gen_failed`, zero tolerance rejects).
Baseline going in: `PROMPT_SANITIZE_2026-07-21.md` (Wave-1 R3 tightening + sanitize,
386 tests green).

**Outcome in one line:** the C8 pre-flight probe *failed its success criteria* —
the three cited "component-pair" regions are **not** split-component complex
representations, so C8 had no valid target and was **dropped** per the bundle's own
ground rule. **Backoff shipped alone.** The bundled faithful 10k was **deferred**
(owner decision): with the structural fix gone, only a timing-only change remained,
not worth a ~2 h run.

Branch: `langgraph-agents`. Commits (bisect order):

- ~~`prompts: add C8 rule …`~~ — **DROPPED** (probe failed; edits reverted, net zero)
- `79893a3` `patcher: add exponential backoff between LLM retry attempts`
- `<this file>` `runs/qcdloop: ROBUSTNESS_BUNDLE_2026-07-21 — C8 dropped, backoff shipped, 10k deferred`

---

## Part A — pre-10k

### A.1 — C8 diff summary (per file)

C8 ("split-component complex representation") was drafted into the three regional
prompts, probed, and then **reverted** when the probe showed no region exercises it.
Net change to every prompt file is **zero**.

| File | C8 added? | Final state | Reason |
|------|-----------|-------------|--------|
| `agents/dd_integrator/system_prompt.txt` | drafted, then reverted | **unchanged** | probe failed (see A.3) |
| `agents/ff_integrator/system_prompt.txt` | drafted, then reverted | **unchanged** | probe failed |
| `agents/float_integrator/system_prompt.txt` | drafted (extended-precision language stripped), then reverted | **unchanged** | probe failed |
| `agents/tracked_integrator/system_prompt.txt` | **skipped by design** | **unchanged** | already covered — see A.2 |

`grep -c C8` across dd/ff/float = `0/0/0` (revert confirmed). Working-tree diff for
the shipped bundle touches only the patcher (A.4).

### A.2 — tracked prompt: C8 skipped (already covered)

Per the ticket, before adding C8 to `tracked_integrator/system_prompt.txt` I read its
Rule 3 and C1/C5/C6/C7. The tracked integrator is a **different paradigm** from the
regional integrators, and the split-component failure mode cannot arise there:

- **Rule 3** already mandates that a complex value is returned as *one*
  `Complex<Tracked<T>>` container — "not `Tracked<Complex<T>>`".
- **C1** pins the exact tracked-complex spelling (components are the tracked scalar),
  so real and imag can never land in different precisions.
- **C6** ("discrete-vs-floating decided by USE") already forces a component that
  feeds a complex build to return the tracked scalar with provenance.
- The tracked integrator mirrors the target library's **function surface**
  (per-function shims following the driver), *not* per-line regional shims. The
  "two independent shims for two adjacent scalar lines" failure C8 targets is a
  regional artifact; the tracked path shims the library's complex-returning function
  as one container overload and never splits a complex value across two shims.

So C8 would be a duplicate rule in the tracked context. **Skipped, no edit.**

### A.3 — C8 pre-flight probe (the reason C8 was dropped)

Full Patcher (dd generate + real build gate) on all three ticket-cited regions,
targeted as 2-line ranges spanning each pair, with C8 present in the dd prompt.
Env: venv + gcc/13.3.0 + cmake/3.28.3, `~/kokkos-install`.

| Region | note | status | builds | cites C8 | ddcomplex/ddouble/#error |
|--------|------|--------|--------|----------|--------------------------|
| `B2m.h:143-144` | `ln43p`/`ln43m` = `ql::cLn(...)` | **ok** | yes | **False** | 12 / 9 / 0 |
| `B1m.h:97-98`   | `si`/`tabar` = `_two()*Y[..]` | **ok** | yes | **False** | 0 / 4 / 0 |
| `B2m.h:105-106` | `sibar`/`tabar` = `_two()*Y[..]` | **ok** | yes | **False** | 0 / 3 / 0 |

**Success criteria (from the ticket) vs. observed:**

- status=ok, shim builds — ✅ (all three)
- references BOTH source lines, ONE atomic complex op — ⚠️ N/A (no complex to unify)
- real & imag in the SAME container — ⚠️ N/A (no real/imag split exists)
- **comment cites Rule C8 — ❌ (0/3)**

The "cites C8" criterion fails in every region → **probe fails** → drop C8.

**Why it fails — the regions are misidentified.** None of the three is a
split-component real/imag pair:

- `B1m.h:97-98` and `B2m.h:105-106` are **not complex at all** — each is a pair of
  real `TMass` scalars (`si`/`tabar`, `sibar`/`tabar`) that merely *share the idiom*
  `ql::Constants<T>::_two() * Y[..][..]`. The correct shim is a single
  `Constants<ddouble>::_two()` specialization (Rule 5 / R3 step 2 / C5-C7). `0`
  `ddcomplex` in either shim.
- `B2m.h:143-144` **is** complex, but it is two **independent** complex values, each
  the full result of a separate `ql::cLn(...)` call sharing the `ieps2` selector —
  not the real and imaginary halves of one value. The integrator correctly shims it
  as a `ddcomplex`-returning `ql::cLn` overload via the **pre-existing** Rule 3 +
  C5/C7. C8 is irrelevant.

**Codebase-wide confirmation:** `grep -rE '\.real\(\)|\.imag\(\)|_re|_im|_r|_i|re|im'`
for an *assignment* split across `runs/qcdloop_headers_full/box/*.h` returns **zero**
matches. The "component-pair sub-cluster" C8 was written to fix **does not exist** as
a real/imag split in this codebase; complex parts are extracted through the
`ql::Real(...)` / `ql::Imag(...)` accessor functions, never as consecutive scalar
statements. C8 is not broken — it has no target.

Representative emitted shim (`B2m.h:143-144`, abridged — note Rule 3, no C8):

```cpp
namespace ql {
// Rule C5/C7 + Rule 3: ddcomplex-returning overload of ql::cLn for a real
// ddouble argument. Strictly more specialized than the library primary.
template <class TOutput, class TMass, class TScale>
inline quad::ddfun::ddcomplex cLn(quad::ddfun::ddouble x, int ieps) {
  if (x.hi > 0.0 || (x.hi == 0.0 && x.lo > 0.0)) {
    return quad::ddfun::ddcomplex(quad::ddfun::log(x), quad::ddfun::ddouble(0.0));
  }
  quad::ddfun::ddouble re = quad::ddfun::log(quad::ddfun::abs(x));
  quad::ddfun::ddouble im = quad::ddfun::dd_pi();       // R3 step 1: dd_pi()
  if (ieps < 0) im = quad::ddfun::ddouble(0.0) - im;
  return quad::ddfun::ddcomplex(re, im);
}
// … + a ddcomplex-arg overload for static-instantiation coverage (C3).
} // namespace ql
```

Container decision: `ln43p`/`ln43m` each → one `ddcomplex` (Rule 3). Both `_two()`
regions → one `Constants<ddouble>::_two()` = `make_dd(0x4000000000000000, 0x0)`
(Rule 5 / R3 step 2). No mixed-precision complex anywhere; nothing for C8 to fix.

Per the ground rule — *"If C8's pre-flight probe fails, drop C8 from the bundle, ship
backoff alone, and note it. Do not silently retry C8 with different wording — that's
Wave 3 territory."* — C8 was reverted and is not shipped.

### A.4 — Backoff diff summary

`79893a3 patcher: add exponential backoff between LLM retry attempts`

```
 agents/patcher/agent.py       |  76 +++++++++++++++++++
 tests/patcher/conftest.py     |  10 ++++
 tests/patcher/test_backoff.py | 125 +++++++++++++++++++++++++++++++
 3 files changed, 211 insertions(+)
```

- **Constants + pure helper** (`agent.py`): `BACKOFF_BASE_SEC = 2.0`,
  `BACKOFF_JITTER_SEC = 0.5`; `_backoff_delay(attempt) = BASE * 2**attempt +
  uniform(0, JITTER)` → 2.0–2.5 s after attempt 0, 4.0–4.5 s after attempt 1.
  Monotonic non-decreasing even under worst-case jitter overlap.
- **Placement in the retry loop**: a `time.sleep(_backoff_delay(attempt))` before
  *each* retry `continue` — the gen-failed path **and** the retryable build-failed
  path — and never after the final attempt. `MAX_INTEGRATOR_RETRIES` stays **3**;
  deterministic (non-llm) paths run one attempt and never sleep; the P6 timeout
  return is untouched (still standalone for Strategy's timeout-retry).
- **Per-attempt forensic trail**: `_append_attempt()` writes
  `<run_dir>/patcher_attempts.jsonl` (sibling to Strategy's `iterations.jsonl`) with
  `{iter_id, rationale_id, target, kind, attempt, outcome, status, elapsed_sec,
  backoff_sec}` for every llm-driven attempt. This is the backoff-credit source
  ("accepted on retry 2/3") and required **no** change to `agents/strategy/` —
  `_log_extra` only folds fixed fields, so the metadata lives in a patcher-owned
  file instead. Best-effort I/O: a logging failure never breaks a patch.
- **Testability**: an **autouse** `sleep_calls` fixture in `tests/patcher/conftest.py`
  records-and-skips the real sleeps, so the existing retry tests (and the new ones)
  run instantly instead of waiting real seconds; the backoff test inspects the
  recorded delays.

What was **not** touched (Wave-3-flagged): `dispatch.py:is_retryable_misgen`
(still "retry everything"), `MAX_INTEGRATOR_RETRIES`, the P6 timeout retry.

### A.5 — Test results

| Step | Command | Expected | Result |
|------|---------|----------|--------|
| 1 (baseline) | `pytest tests/ -x -q` | 386 | **386 passed** (165 s) |
| 2 (post-C8, before revert) | `pytest tests/{dd,ff,float,tracked}_integrator tests/integrator_base -x -q` | pass | **91 passed** (297 s) — no fixture asserts prompt content |
| 3 (backoff) | `pytest tests/patcher/ -q` | pass + new | **49 passed** (25 s) |
| 5 (full, after backoff; C8 reverted) | `pytest tests/ -q` | 386 + N | **394 passed** (251 s) — +8 backoff tests |

The 8 new tests (`tests/patcher/test_backoff.py`): delay is positive / exponential /
non-decreasing; sleep placed between attempts on eventual success; no sleep on
first-attempt success; no sleep after budget exhaustion; no sleep on the
deterministic path; attempt-log records each llm attempt with the winning (late)
attempt and per-attempt backoff; deterministic path writes no attempt log.

---

## Part B — 10k re-measure

**Status: DEFERRED (owner decision, 2026-07-21).**

The bundle's structural fix (C8) was dropped at the pre-flight gate, leaving only the
retry backoff — a **timing-only** change that can move the transient `llm_gen_failed`
tail but, by construction, cannot change the headline demotion / gen-miss counts on a
stable LLM proxy. Rather than spend a ~2 h faithful 10k to (very likely) re-print the
PIPELINE_v1 numbers (152/152 accept, 85 demotions, 40 `dd_untested`, 16 float / 14 ff
gen-misses, 0 tolerance rejects, 0 tail failures), the run was deferred until a
substantive **semantic** change is ready to measure.

No headline deltas, per-cluster attribution, or backoff-credit counts are reported
here — they require the deferred run. When it happens, backoff credit is a one-liner
over the new `patcher_attempts.jsonl` (`outcome == "ok" && attempt > 0`, grouped by
`iter_id`).

The hard-floor stop-the-line checks (tolerance rejects > 0, tail failures > 0,
demotions < 85) are therefore **not yet exercised** for this bundle.

---

## Recommendation for Wave 3

1. **Retire the "component-pair" hypothesis.** The probe + codebase grep show there
   is no split-component real/imag cluster in `box/`. C8 as written has no target
   here; do **not** revive it against this codebase without first finding a genuine
   `_re`/`_im` (or `.real()`/`.imag()`-assignment) split. The three regions the
   ticket cited already build to `status=ok`.

2. **Re-scope the residual `dd_untested` cluster from real data, not a guess.** The
   40 `dd_untested` at PIPELINE_v1 are *not* split-component. Characterize them
   directly from the last 10k's `iterations.jsonl` + error excerpts: how many are
   (a) structural (a genuinely new rule is needed — and *which* pattern), vs.
   (b) retry-exhausted transients (the **misgen classifier**, `dispatch.py:67`, is the
   Wave-3 lever), vs. (c) already-fixed by Wave-1 R3 tightening and merely stale.

3. **Backoff pays back only with the misgen classifier.** Backoff spaces the retries;
   it does not decide *which* failures to retry. Its transient-tail benefit is best
   measured **together** with the Wave-3 misgen classifier in the next faithful 10k,
   using `patcher_attempts.jsonl` for per-attempt attribution — that is the run worth
   paying for.
