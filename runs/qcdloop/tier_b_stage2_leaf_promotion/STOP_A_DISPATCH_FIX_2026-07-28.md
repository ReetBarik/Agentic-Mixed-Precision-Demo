# Subtask STOP #A dispatch fix — preprocessor-live root resolution

**Status: dispatch fix LANDED and VALIDATED as correct; re-run reveals a NEW, deeper
finding that supersedes L-measure v1's central claim.** The root reroute now lands on
the **preprocessor-live** `boxGPU.h` dispatch `BO` (was the `#ifndef`-dead `B0m.h:432`
copy — STOP #A). But making the promoted tree *reachable* also makes it *instantiated*
for the first time, and instantiation shows the L1′/L2/L3 emitted dd variant tree
**does not compile** — 89 dd/`double` type-binding errors for B10. **L-measure v1's
"builds clean at dd" was a dead-code false positive**: v1's reroute landed on the dead
`BO`, so the promoted `B1m_B10` templates were only *parsed*, never *instantiated*, and
uninstantiated dependent-typed C++ templates are not type-checked. B10 therefore does
**not** reach a measurable lift yet — not because the value-flow model is falsified
(the dd cancellation still never runs — it can't be compiled), but because the emission
is not yet type-correct under real instantiation. This is **NOT STOP #A** (that was a
lift < +8 *after* the dd cancellation executes; the cancellation still does not execute)
and it is **NOT STOP #AA** (the preprocessor filter is fully reliable on this tree). It
is a new emission-correctness finding, handed back per §2.4 falsifier discipline.
Date: 2026-07-28. Branch `langgraph-agents`, on top of L-measure v1 (`57d78f4`).

---

## 1. Fix summary

Structural, deterministic, no LLM in the decision loop — a two-stage
preprocessor-active + fail-loud definition selector, matching `call_graph.py`'s hybrid
discipline (libclang for what it recovers reliably; source-text scan for what it drops).

| file | change | LoC |
|---|---|---|
| `agents/patcher/preprocessor.py` | **NEW** — conditional-directive-aware `#include`-chain walk (`compute_active_lines`) + `-D` extractor (`defines_from_args`) | 210 |
| `agents/patcher/call_graph.py` | `CallGraph.active_lines` field; `def_is_active` / `active_defs` helpers; populate `active_lines` in `build_call_graph` (fail-open on `OSError`) | +48 / −1 |
| `agents/patcher/fanout.py` | `_pick_def` → preprocessor-active filter + `must_call` overload disambiguation + **fail-loud** on residual ≥2; `_resolve_root_file` delegates to `_pick_def`; `_body_calls` / `_ambiguous_def_message` helpers | +83 / −16 |
| `tests/patcher/fanout/test_preprocessor_dispatch.py` | **NEW** — 10 tests (real tree + 2 synthetic fixtures + walk unit) | 256 |

**466 LoC total.** No app identifier enters production code — the recognizer reads
whatever guard macros the source uses; `QCDLOOP_BOX_FULL_DISPATCH` appears only in the
walk's docstring/tests as the motivating example.

### Primary vs fallback path — a note on the hybrid

The brief's *primary path* was libclang's `isPreprocessorBranchActive`. **libclang's
Python bindings expose no such attribute** on `Cursor` (verified: the attr list is
empty), and its C-API branch info is unreliable on the broken-include, template-heavy
parses qcdloop needs anyway. So per the brief's own §(a) fallback clause — "*some
cursors libclang emits… have flaky `isPreprocessorBranchActive` info… fall through to:
walk `#include` from the driver TU*" — the **source-text include-chain walk is the
reliable path** here. This is exactly the hybrid the brief mandates (and the same split
`call_graph.py` already uses for edge/extent recovery): libclang owns definition +
extent recovery; the deterministic source walk owns preprocessor liveness. It is **not**
STOP #AA: the walk *does* disambiguate cleanly (§3), so the "structural query with a
right answer" premise holds for this tree.

### How the selector works (`_pick_def`)

1. **Single-def name** → return it (the common case; no change).
2. **Preprocessor-active filter** — keep only definitions whose head line is active
   under the app's build defines (`graph.active_defs`). Narrow-only: if the walk flags
   *every* candidate dead (a walk gap), fall back to the full set so the only candidate
   is never lost.
3. **Overload disambiguation** — among survivors, `must_call` selects the def whose body
   invokes the path's child (unchanged semantics, now applied to the *active* subset).
4. **Fail loud** — if ≥2 candidates still survive both filters, raise `FanoutError`
   naming every remaining candidate + file, explicitly citing STOP #A. **Never** a
   silent `defs[0]` — that blind pick is what landed the v1 reroute on dead code.

`_resolve_root_file` now delegates to `_pick_def(graph, graph.root)`, so the file the
root reroute is *written to* and the def it is *computed against* are always the same
live definition.

---

## 2. Test results

### Unit + structural (`test_preprocessor_dispatch.py`, 10 tests — all pass)

* **Real tree** — `_pick_def("BO")` returns `boxGPU.h:69-143` (the live full-dispatch),
  NOT the `#ifndef`-dead `B0m.h:432` at `defs[0]`; `_resolve_root_file` agrees. A
  precondition assert pins the fixture (defs[0] *is* the pruned copy) so the test can't
  silently rot.
* **Pruned copies inactive** — exactly 1 of the 6 `BO` defs is preprocessor-active
  (`boxGPU.h`); the 5 group-header copies are dead.
* **Chain frames unaffected** — `B1m`/`B10`/`B0m`/`B2m` (single-def) resolve unchanged;
  `Lnrat`/`Li2omx2` keep both live overloads (the overload choice is
  `_select_leaf_overload`'s job, not the preprocessor filter's).
* **Synthetic guarded fixture** (`#ifndef GUARD` copy + `#define GUARD` meta) — the
  guarded copy is filtered out; the pick is the live one. A twin test drives the guard
  via `extra_args=["-DGUARD"]` (build-time `-D`, not an in-source `#define`).
* **Synthetic two-live-candidate fixture** — two unguarded `BO`, both active →
  `_pick_def` raises with a diagnostic naming **both** files and citing STOP #A;
  `_resolve_root_file` inherits the same fail-loud. (Guards against silent `defs[0]`
  regression.)
* **Empty `active_lines`** (hand-built graph) → every def reported active (unfiltered
  fallback, byte-identical to pre-fix for tests / A-B runs).
* **Walk units** — `defines_from_args` forms; `#ifndef` honoured; real-tree walk yields
  exactly `boxGPU.h:69` active, `B0m.h:432` dead.

### Regression

* **`tests/patcher/` — 260 passed** (250 pre-existing + 10 new). No pre-existing test
  changed behaviour; the fail-loud path is only reachable on genuine ≥2-live ambiguity,
  which no fixture in the suite hits.
* **In the real run, the fail-loud path never triggered** (grep for `ambiguous
  definition` across all run logs is empty) — the selector resolved every frame cleanly.

---

## 3. Fix works in the live run (before the new finding)

Direct inspection of the B10 run tree (`lmeasure_run/B10/tree_B10/`) confirms the
reroute now lands correctly:

* **live `boxGPU.h:129`** — `ql::B1m_B10<TOutput, TMass, TScale>(res, xpi, musq, i);`
  (was `ql::B1m<…>` — now rerouted into the promoted tree). ✓
* **dead `B0m.h:432` pruned BO body** — still calls `ql::B0m<…>` / `ql::B1m<…>`, **not**
  rerouted (the v1 mis-landing site is now correctly untouched). ✓
* The compiler's own instantiation backtraces confirm the errors originate
  `required from … ql::BO … [TOutput = Kokkos::complex<double>; …]` at **`boxGPU.h:127`
  (B10) / `boxGPU.h:131` (B14)** — i.e. from the **live** meta-header BO, proving the
  promoted tree is now genuinely reached. In v1 it never was.

STOP #A's root cause — "root reroute lands on a preprocessor-dead `BO`" — is fixed and
verified end-to-end.

---

## 4. L-measure re-run results

Recipe: seed 12345, 5000 samples, mu2 = 91.2², range (100, 1e6) GeV²; B10/B13/B14;
leaf opt-in {B10, B13} (B12 not in the run set; B14 opts out — dd-sufficient). Kernel
scope + positive-lift gate, margin +0.5. `run_lmeasure.sh`, out
`lmeasure_run/`. `LMEASURE_DONE_EXIT_0`.

| integral | v1 (dead reroute) | v2 (live reroute) | change | measured lift |
|---|---|---|---|---|
| **B10** | `apply_failed`→**"ok build"**, `chain_no_lift`, lift **0.0** | `apply_failed` (**`build_failed`**, 89 dd/`double` type errors), 411s | **v1 build was a dead-code false positive** | **None** (cannot compile) |
| **B13** | `apply_failed` (`write_truncation`) | `apply_failed` (`write_truncation`), pre-build gate, 58s | **unchanged** | None |
| **B14** | `rejected/chain_no_lift`, 13.1855→13.1855, built+measured | `apply_failed` (**`build_failed`**, 5 dd/`double` type errors), 116s | **now instantiated → exposed same emission defect** | None |

* **B10 — `kernel_measured_lift` target ≈ +18.43, bar ≥ +8: NOT REACHED.** But *not*
  because the value-flow model is falsified — the dd cancellation still never executes
  (the promoted tree does not compile). 89 errors, dominant taxonomy:
  * 22× `no matching function for … Kokkos::complex<double>::complex(quad::ddfun::ddouble)`
  * 16× `invalid cast from 'quad::ddfun::ddouble' to 'double'`
  * 9× `conversion from 'quad::ddfun::ddcomplex' to … 'const Kokkos::complex<double>'`
  * 8× `cannot convert 'quad::ddfun::ddouble' to 'const double' in initialization`
  * 6× `Kokkos::complex<quad::ddfun::ddcomplex>` (nonsensical nested-complex)
  * 1× `#error "DD Chain Integrator: ql::ddilog(ddouble) requires manual classification"`

  Every error is inside a promoted `_B10` variant/clone body (`Lnrat_B10`, `B1m_B10`,
  `ddilog_*_B10`, `Li2omx2_*_B10`) — the L1′/L2/L3 emission — never in the dispatch
  selector. The emitted variants mix `ddouble`/`ddcomplex` against the box's
  `Kokkos::complex<double>` `TOutput` binding in ways g++ rejects once instantiated.

* **B13 — `write_truncation`, unchanged.** Gated *before* build (no build log) by the
  interior `chain_write_truncation` gate — exactly as the brief predicted ("probably
  still apply_failed/write_truncation — that's B13-selector's fault, not this
  subtask's"). No change from v1. B13-selector's problem.

* **B14 — the clincher.** B14 opted **out** of leaf promotion, yet now `build_failed`
  with the *same class* of dd/`double` errors (5 of them, in `B2m.h` `_B14` variants).
  Because the live BO now reroutes `massive==2` to B14's variant tree, B14's **base**
  chain-variant emission is instantiated for the first time too — and it doesn't compile
  either. **This proves the emission defect is NOT leaf-promotion-specific**: it is in
  the base chain-variant dd-binding emission that v1 never actually instantiated.

### STOP #B (B14 byte-identity)

B14 is **not** byte-identical to its v1 baseline: v1 measured `13.1855→13.1855`
(built), v2 is `build_failed`. **This is not a violation of the fix's scope** — the fix
touches only *which* definition the entry-point reroute lands on, and B14 opts out of
leaf promotion so no `_B14` leaf clone was emitted. What changed is upstream of B14's
own patch: the *shared live BO* now reroutes **all** its `massive==k` branches into the
per-integral variant trees, so B14's base chain variants (which existed in v1 but sat
unreferenced behind the dead BO) are now instantiated. B14's v1 "built + measured
13.1855" was the **same dead-code false positive** as B10's "builds clean". STOP #B's
intent — "the fix must not perturb B14's *own* result path" — holds; what it revealed is
that B14's v1 result was itself an artifact of the STOP #A defect. Flagged, not patched.

### STOP #Z (vendored snapshot)

**Pristine.** `git status --porcelain runs/qcdloop_headers_full/` is empty before and
after. All graph/emission ran over the per-run clones (`lmeasure_run/<I>/tree_<I>`).

---

## 5. Verdict against the §2.4 prediction

The brief's verdict gate has three branches. **None fits cleanly, because the premise of
all three — "the dd cancellation is finally reached" — is still not met.** The dispatch
fix makes the tree *reachable* (dispatched) but instantiation shows it is not
*compilable*, so the dd cancellation still never runs.

* **NOT "B10 ≈ +18.43 → Group A validated"** — no lift measured (no build).
* **NOT "B10 in +8..+18 → partial"** — no lift measured.
* **NOT "B10 < +8 → real STOP #A"** — this branch means "the dd cancellation is finally
  reached and still produces no lift." The dd cancellation is **not** reached: the
  promoted tree does not compile, so nothing executes. Reaching for coefficient
  synthesis here would be exactly the §3.4 falsifier error the arc has avoided twice.
* **NOT STOP #AA** — the preprocessor filter is fully reliable on this tree (§3); the
  structural-query premise holds.

**The honest verdict: the STOP #A dispatch defect is FIXED and validated, and fixing it
uncovered the next defect it was masking — the emitted dd variant tree is not
type-correct under real template instantiation.** L-measure v1's "builds clean at dd for
the first time" was a dead-code false positive; the true state is that the emission has
never been compiled under instantiation until now. This is a **new emission-correctness
finding (call it the "instantiation-binding" defect)**, not any existing STOP.

### What this is NOT (falsifier discipline, per §3.4)

* **Not the dispatch fix's bug.** Every error is in an emitted `_B10`/`_B14` variant
  body; the selector resolved cleanly and never fired its fail-loud path. The fix does
  exactly what it claims.
* **Not a value-flow / carrier falsification.** The value-flow model has *still* not been
  tested against a running dd cancellation — the code doesn't compile, so the carrier
  binding is untested at runtime. Coefficient synthesis (Option A / L4) is **not**
  reached for.
* **Not truncation.** The 43-coeff dd `_C` source-residence (probe P5) is unchanged and
  irrelevant to a compile failure.

---

## 6. Hand-back & recommended next step (Reet's call)

The dispatch fix should **land** (it is correct, tested, and unblocks the *reachability*
half of STOP #A). The new finding is the next blocker and belongs to the emission layer,
not this subtask:

**The L1′/L2/L3 dd-variant emission produces type-incorrect bindings under real
instantiation.** The emitted variants mix `quad::ddfun::ddouble` / `ddcomplex` against
the box's `TOutput = Kokkos::complex<double>` in ways that only manifest when the
templates are instantiated (which the STOP #A defect prevented until now). The dominant
shapes to triage:

1. **`Kokkos::complex<double>` ← `ddcomplex` / `ddouble` construction & assignment** (34
   of 89) — a promoted dd value flowing into an un-widened caller-precision
   `Kokkos::complex<double>` local/store with no conversion. This is the *designed exit
   boundary* truncation, but emitted as a raw assignment g++ rejects rather than an
   explicit narrowing.
2. **`ddouble → double` casts / `const double&` init** (24 of 89) — a dd value bound to a
   `double` reference/parameter inside a variant; a missing widen on a decl or callee
   parameter.
3. **`Kokkos::complex<quad::ddfun::ddcomplex>`** (nested-complex, ~7) — a container
   token widened *twice* (the complex-container promotion applied on top of an
   already-`ddcomplex` operand).
4. **`ql_shim_dd.h` `#error … requires manual classification`** (1) — a synthesized shim
   the emission left unclassified.

Because **B14 (no leaf promotion) exhibits the same class**, the fix belongs in the
**base chain-variant emission / boundary transform**, not rule (d). Recommended: a
minimal *instantiation gate* (compile the promoted tree with the live BO reroute — which
this fix now enables — as a fast pre-measurement check) so this class is caught
deterministically before a measurement pass, then triage the four shapes above in the
boundary/return-widen emission. This is a separate subtask (emission-binding
correctness), gated behind this dispatch fix landing.

**Everything downstream (B12-selector, B13-selector, merge-design) still defers** until
B10 both compiles *and* measures ≥ +8.

---

## 7. STOP audit

| STOP | condition | status |
|---|---|---|
| **STOP #A (root cause)** | reroute lands on preprocessor-dead BO | **FIXED** — reroute lands on live `boxGPU.h:69`; verified in tree + compiler backtraces |
| **STOP #A (post-fix, lift < +8)** | dd cancellation reached but no lift | **NOT APPLICABLE** — dd cancellation not reached (tree doesn't compile); did NOT reach for coefficient synthesis |
| **STOP #AA** | preprocessor filter unreliable on real tree | **NOT FIRED** — walk disambiguates cleanly; fail-loud never triggered in the run |
| **STOP #B** | B14 byte-identical to pre-leaf baseline | **surfaced, not violated** — B14 changed because its v1 "built" result was itself a dead-code artifact of STOP #A; fix scope (root-entry resolution only) respected; flagged for Reet |
| **STOP #Z** | vendored snapshot pristine | **HOLDS** — `runs/qcdloop_headers_full/` clean before + after |
| **NEW: instantiation-binding** | emitted dd variant tree type-incorrect under instantiation | **NEW FINDING** — 89 (B10) / 5 (B14) dd/`double` errors; emission layer, not dispatch; handed back |

Hard-constraint compliance: structural + deterministic + no LLM in the decision loop ✓;
hybrid matches `call_graph.py` (libclang defs/extents + source-text preprocessor walk) ✓;
fail-loud on ambiguity ✓; no app identifiers in production ✓; no new macros invented ✓;
no mutation of shared originals ✓; no L1′/L2/L3 changes ✓.

---

## MEMORY.md update block

> **STOP #A dispatch fix LANDED + validated 2026-07-28; re-run uncovered a deeper
> emission defect.** New `agents/patcher/preprocessor.py` (source-text `#include`-chain
> walk, `compute_active_lines` — libclang's Python `Cursor` exposes NO
> `isPreprocessorBranchActive`, so the brief's *fallback* walk is the reliable path;
> hybrid matches call_graph.py) + `CallGraph.active_lines`/`active_defs`/`def_is_active`
> + `_pick_def` two-stage (preprocessor-active filter → `must_call` → **fail-loud** on
> ≥2, never silent `defs[0]`); `_resolve_root_file` delegates to `_pick_def`. 466 LoC,
> 10 new tests, **260 patcher tests pass**. **Fix verified**: B10 run tree now reroutes
> the LIVE `boxGPU.h:129` BO → `B1m_B10` (was the `#ifndef`-dead `B0m.h:432` copy);
> compiler backtraces confirm instantiation from live BO. **But L-measure re-run: all 3
> `apply_failed`.** B10 `build_failed` (89 dd/`double` type errors), B13 `write_truncation`
> (unchanged pre-build gate, B13-selector's job), B14 `build_failed` (5 errors — and B14
> opted OUT of leaf promotion). **KEY: L-measure v1's "builds clean at dd" was a
> DEAD-CODE FALSE POSITIVE** — v1's reroute landed on the dead BO, so `B1m_B10` templates
> were parsed but never *instantiated* (uninstantiated dependent-typed C++ templates
> aren't type-checked). The fix makes them live → instantiated → the L1′/L2/L3 emission's
> dd/`double` binding errors surface (`Kokkos::complex<double> ← ddcomplex`, `ddouble→double`
> casts, nested `complex<ddcomplex>`). **B14 (no leaf promo) failing too proves the defect
> is in the BASE chain-variant emission, NOT rule (d).** Verdict: dispatch fix correct;
> did NOT reach for coefficient synthesis (dd cancellation still never runs — can't
> compile, so value-flow model untested at runtime). STOP #Z holds (snapshot pristine),
> fail-loud never fired. NEXT (Reet): emission-binding correctness subtask + an
> instantiation gate (compile the live-rerouted tree pre-measurement); B12/B13-selector +
> merge-design DEFER until B10 compiles AND measures ≥ +8. Report
> `STOP_A_DISPATCH_FIX_2026-07-28.md`. See [[project_l_measure_wiring]].
