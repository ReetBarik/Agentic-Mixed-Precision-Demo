# Tier-B Stage-2 — closure Subtask 5: deterministic forwarding-overload emitter — STOP #K at design verification (2026-07-26)

**Headline: the Subtask's core premise is empirically FALSE, and STOP #K fired before any
code was written.** The proposed durable fix — a deterministic emitter that forwards a
namespace-qualified math-bridge helper `Ns::fn<Ts…>(promoted)` to its visible primary
template as a "mechanical, semantics-free template forward" — **cannot be built soundly
for `ql::Lnrat` or `ql::ddilog`**: every mechanical forwarding overload either fails to
compile or infinitely self-recurses at runtime (stack-overflow segfault). This is exactly
the false-positive case STOP #K exists to catch. Per the conservative-parser discipline
(false negatives safe, false positives hard-fail), both symbols classify as
**NON-forwardable**, the emitter emits nothing for them, and **no source was changed.**

- run dir: `runs/qcdloop/tier_b_stage2_subtask5/`
- probe evidence (real headers + real Kokkos): `runs/qcdloop/tier_b_stage2_subtask5/probe_evidence/`
- offline baseline (unchanged, no code written): **391 pass** (`tests/integrator_base` + `tests/patcher` + `tests/shared`)
- head: `0446d3a` (no new commit — STOP before implementation)

## What was verified, and how

Before writing the emitter I did the one thing the Subtask explicitly demands (STOP #K:
"false positives would break the build → hard fail"): I checked, empirically, whether the
mechanical forward actually compiles **and terminates** at the real call-site types. I built
faithful single-TU probes against the **real** headers (`src/kokkosMaths.h`,
`src/kokkosUtils.h`, `third_party/include/dd_math.hpp`, `dd_complex.hpp`) and **real** Kokkos
(`~/kokkos-install`, `-std=c++20`, gcc 13.3.0), reproducing the exact promoted call that
`agents.integrator_base.boundary.promote_region_block` generates for `kokkosUtils.h:706`:

```cpp
// promoted reads v,x,w,y become ddouble; explicit template args stay <TOutput,TMass,TScale>:
ql::Lnrat<Kokkos::complex<double>, double, double>(v__ff /*ddouble*/, x__ff /*ddouble*/)
```

with the real entry types `TOutput = Kokkos::complex<double>`, `TMass = TScale = double`.

### Probe battery (all against real headers/types)

| candidate forwarding overload | compile | runtime |
|---|---|---|
| **A** — subtask-3 "working" B12 shim: 3 tparams, `(ddouble,ddouble)` args, `return ::ql::Lnrat<TOutput,TMass,TScale>(a,b)` | OK | **∞ RECURSION → segfault** |
| **D** — same, forward `<ddouble,TMass,TScale>` | OK | **∞ RECURSION** |
| **WIDEN** — forward `<TOutput,ddouble,ddouble>` (+ support surface) | OK | **∞ RECURSION** |
| **B** — task Step-2 example: 2 tparams `<TMass,TScale>`, `(ddouble,ddouble)` args | **FAIL** (call site names 3 explicit template args; 2-param overload not viable) | — |
| **C** — 2b hand-written: 2 tparams, `(const TMass&,const TMass&)` args | **FAIL** (ambiguous / arg-type mismatch at real types) | — |
| "recursion-safe" — cast args to `double`, call primary#2 at `TScale=double` | OK | runs — **but computes at DOUBLE precision (narrowed), a C9 violation** |

A depth counter inserted into Form A confirmed the mechanism directly: the overload
re-selects **itself** 6-deep and climbing (`RECURSION: overload re-selected itself 6 deep`),
not a data-dependent crash.

## Root cause — three independent structural walls (not LLM variance)

1. **Overload resolution ignores the explicit template-arg list.** C++ selects `ql::Lnrat`
   by the *argument types* `(ddouble, ddouble)`, not by the `<…>` a call writes. So **any**
   injected `ql::Lnrat(ddouble,ddouble)` overload whose body calls `ql::Lnrat(ddouble,ddouble)`
   re-binds to itself — the `<TOutput,TMass,TScale>` / `<ddouble,…>` the shim writes is
   irrelevant to which function is chosen. Infinite recursion is unavoidable for a same-name
   `(ddouble,ddouble)` overload that forwards to the same name with `ddouble` operands.

2. **There is no vendored `quad::ddfun::Lnrat` / `ddilog` to forward to.** The `_MATH_FN_NAMES`
   bridges work because `quad::ddfun::{abs,log,sqrt,…}` genuinely exist (grep: 14 such
   defs). For `Lnrat`/`ddilog`, `grep third_party/include` returns **nothing**. The only
   same-name target is the *app primary*, and that primary is **not instantiable at extended
   precision** without a large support surface — its body uses `.imag()`, `ql::Imag`,
   `ql::Real`, `ql::Sign`, `ql::iszero`, `ql::kLog`, `ql::kAbs`, `Constants<…>::_ipio2`, a
   complex-log branch, and a regulator constant. **That support surface *is* the "semantics"
   the premise assumed away.** A "mechanical, semantics-free forward" does not exist here.

3. **`Lnrat` is a leaf callee, never cloned into a variant.** The reroute machinery only
   clones frames that are *on* a chain's promoted line-set. `Lnrat`'s definition
   (`kokkosUtils.h:141–155`) is never in any chain line-set, so it is never renamed to a
   `Lnrat_B10`-style clone — verified: **`Lnrat` appears in ZERO fanout manifests across the
   entire run history**, whereas `ddilog`/`Li2omx2` *do* get cloned (their bodies *are* on the
   chain). So the rename-based escape from the self-recursion (which is what actually keeps the
   cloned `ddilog_…`/`Li2omx2_…` calls sound) is unavailable to `Lnrat`.

### The design already said "refuse"

`CLOSURE_SCOPED_CHAINS_DESIGN.md` §2.4 explicitly lists **`ql::Lnrat`** as a
callee-not-in-`F` "whose signatures we will not touch" → a `chain_closure_escapes` refusal.
Subtask 5's premise (emit a forward for it) directly contradicts the design's own refusal
rule. The empirical probes and the design agree: this is a frontier the chain does not cross.

## Why the earlier runs looked like it "could" forward

- **2b** (B10): the hand-written Form-C `Lnrat` overload was *present in a shim* but that
  build **failed anyway** (18 errors: `_pi2o6` #error + `T__ff` redecl) — the overload was
  **never validated by a passing build**.
- **subtask 3 / 4** (B12): the Form-A overload was present and the TU **built `ok`**, but the
  tree was **REVERTED** (`rejected`, `chain_no_lift`) — and, critically, B12's runtime samples
  never exercised `Li2omx2`'s else-branch (line 706), so the recursion bomb never detonated.
  "Built + ran to measurement" here means "compiled and the untaken branch stayed cold," not
  "the forward is correct."

So the Subtask-4 report's read ("a symbol the pipeline *can* forward, B12 did twice") was based
on a compile that hides a latent runtime bomb. Under the actual B10 codepath (which *does* take
line 706), the same overload segfaults. This is not variance the emitter removes — it is an
unsound transform the LLM's R4 `#error` was, in effect, **correctly refusing to emit**.

## STOP-condition audit (§ STOP-and-report)

- **STOP #K (emitted overload breaks build) — FIRED (pre-emptively, at verification).** The
  mechanical forward is a build/runtime-breaking overload for `Lnrat`/`ddilog`. Per discipline
  I did **not** ship it: I reclassified both symbols as **non-forwardable** and wrote no
  emitter. This is the conservative-parser contract working as designed (refuse when not
  confident; false positive is hard-fail).
- **STOP #A (measurement falsification) — did NOT fire.** No B10 measurement was reached; the
  closure-scoped design's core hypothesis is neither confirmed nor falsified this Subtask.
- **STOP #B (accept↔reject flip) — N/A.** No code changed, so B12/B13/B14 are byte-identical to
  their Subtask-4 baselines by construction (nothing was re-run — there was nothing to re-run).
- **STOP #L (R4 escape rate does not drop) — subsumed by STOP #K.** The premise for reducing the
  rate (a deterministic overload on the new path) does not exist for these symbols. R4 does not
  "drop"; it is arguably the **correct** honest outcome here, not a defect to engineer away.
- **STOP #M (new gen-defect class) — the real finding is upstream of this.** The blocker is not a
  new *generation* defect; it is that the *transform itself* is unsound. Reported here; proposal
  below; **not implemented — awaiting Reet.**

## R4 escape-rate before/after

Not applicable / intentionally unchanged. The Subtask's mechanism (move `Lnrat`/`ddilog` off
the LLM path onto a deterministic emitter) is unavailable, so the escape rate for these symbols
is **unchanged by design** — and, given the analysis above, the escape is the *sound* outcome
for a leaf helper with no extended-precision instantiation and no vendored target. Engineering
the rate down would mean shipping the recursion bomb.

## Verdict

Subtask 5's declared durable fix is **not implementable as specified** for the two symbols it
targets. The premise ("visible primary template ⇒ mechanical, semantics-free forward") holds for
the framework-agnostic `_MATH_FN_NAMES` vocabulary (which *does* have vendored extended
overloads) but **fails for app helpers `ql::Lnrat`/`ql::ddilog`**, which (1) have no vendored
target, (2) have a primary that is not instantiable at extended precision without their full
semantic support surface, and (3) — for `Lnrat` — are leaf callees the reroute never clones,
so a same-name `(ddouble,ddouble)` overload self-recurses. **STOP #K fired; no code written; no
regression possible.** All discipline honored: `_MATH_FN_NAMES` untouched, catalog/normaliser/
retry-budget untouched, shared `Li2omx2` untouched, offline baseline green (391).

## Proposed next moves (NOT implemented — Reet's call)

B10's first Group-A measured lift is **not reachable via helper-forwarding**. Two real paths,
smallest-blast-radius first:

1. **Wire the design's own `chain_closure_escapes` refusal at the `Lnrat`/`ddilog` frontier
   (recommended).** Per §2.4 these callees are out of scope for signature widening; the chain
   should **abandon cleanly** (no variants emitted, a clear `chain_closure_escapes` outcome)
   instead of driving the Patcher into 6 doomed retries against an unsound transform. This makes
   B10's outcome an *honest, cheap terminal* rather than an `llm_gen_failed` after a full retry
   budget — and stops presenting a solvable-looking blocker that isn't. It does **not** produce a
   B10 lift; it correctly records that this chain cannot be promoted without §2 change.

2. **Make `Lnrat`/`ddilog` chain-frames (larger, the only path to an actual B10 lift).** If B10's
   lift genuinely requires dd to flow *through* `Lnrat`/`ddilog`, their bodies must be **cloned
   and promoted** (like `Li2omx2`/`ddilog` already are on B12's chain), so the qualified calls
   route to promoted clones (`ql::Lnrat_B10<…>`) whose bodies compute in dd end-to-end and whose
   self-calls rename with them (no recursion, because the clone is a distinct name). This is a
   closure/reroute-scope change (extend `F` to include these callee frames + their support-surface
   constants), not an emitter — and it is exactly the "make it a frame" direction the design's
   rule (a)/(c) machinery already implements for on-chain frames. Non-trivial; needs Reet's
   go-ahead and a scoping pass on the support surface.

Further retry-budget bumps remain explicitly off the table (Subtask 4 STOP #H): the failure is
structural, not probabilistic.

## After success (NOT dispatched here)

Group B measurement-only (Phase 2f, needs Reet); B13 chain-selector scope refinement; B12
hotspot-covering chain selection (move the 3.6906 floor); cross-integral merge. STOP-after-Stage-2
holds; the emitter was **not** built (STOP #K), so nothing downstream advances this Subtask.
