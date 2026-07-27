# Subtask L1′ — Class-1 shallow-wrapper recognizer + synthesized-overload emitter

**Status: LANDED (design-tree only; not pushed). STOP before dispatching L2.**
Date: 2026-07-27. Branch `langgraph-agents`, on top of `0c9a3a3`.

Extends the existing Gap-A bridge machinery
(`agents/integrator_base/regional.py:64-199`) so the pipeline **synthesizes**
app-specific "shallow wrapper" dd/ff overloads (`ql::kAbs`, `ql::kLog`, `ql::kSqrt`,
`ql::kConj`, `ql::Real`, `ql::Imag`, `ql::Sign`, …) whenever a promoted region /
leaf-callee body names them — structurally, from each wrapper's own primary body +
the vendored `quad::ddfun` surface. No vendored app-specific primitive header; no
app identifiers baked into the recognizer/emitter (the qcdloop wrappers are an
**emergent consequence** of the body-shape recognizer, per
`[[feedback_no_placeholder_patterns]]`).

This is L1′ of the leaf-callee promotion dispatch (`LEAF_CALLEE_PROMOTION_DESIGN.md`
§2.2 / §8). L2 (rule (d) frame-discovery), L3 (emission plumbing), L-measure (B10
e2e) follow, **gated on this landing and Reet's dispatch call.**

## Deliverables

| artifact | path |
|---|---|
| recognizer + emitter + manifest API + region pass | `agents/integrator_base/shallow_wrapper.py` (new, 39 KB) |
| wire-in to the regional dispatch | `agents/integrator_base/regional.py` (+93 lines) |
| unit + compile + wire-up tests | `tests/integrator_base/test_shallow_wrapper.py` (new, 35 tests) |
| this report | `runs/qcdloop/tier_b_stage2_leaf_promotion/SUBTASK_L1_PRIME_2026-07-27.md` |

---

## 1. Recognizer scope + refusals table

The recognizer (`shallow_wrapper.recognize`) classifies a **single-`return`** primary
body into one of four shapes; anything else is a conservative refusal (`None` →
LLM fallback). All classification is by **body shape**, never by function name.

| # | shape | recognized form | dd transform | qcdloop example |
|---|---|---|---|---|
| 1 | `return Ns::mathfn(arg);` | delegation | redirect inner call to `quad::ddfun::mathfn` | `kAbs`→`Kokkos::abs`, `kLog`→`Kokkos::log`, `kSqrt`, `kConj` |
| 2 | `return arg.member();` | accessor | re-emit accessor on promoted param | `Real`→`.real()`, `Imag`→`.imag()` |
| 3 | `return <scalar-expr over param>;` | scalar-expr | **substitute param-type token** with promoted type | `Sign`→`(double(0)<x)-(x<double(0))` |
| 4 | `return <expr w/ Class-1 inner call>;` | transitive | same param-type substitution; inner wrapper emitted too | simple `iszero`→`kAbs(x)<T(cut)` |

**Refusals (all exercised by negative unit tests):**

| body shape | why refused | contract |
|---|---|---|
| multi-statement body | not a lone `return` | false-neg (safe) |
| control-flow (`if`/`for`/…) | not straight-line | false-neg |
| inner call to non-Class-1 symbol | can't classify the delegate | false-neg |
| delegation to non-`_MATH_FN_NAMES` op in unknown ns | not a math bridge | false-neg |
| multiple parameters | not a shallow-wrapper shape | false-neg |
| pointer / array parameter | not a scalar/container operand | false-neg |
| empty body | nothing to emit | false-neg |
| **delegation to a `_MATH_FN_NAMES` op the vendored surface lacks** | **STOP #S** — never invent a mapping | false-neg |
| **body names a template param other than the parameter's own type** | **STOP #T** — emitter does no full template-arg substitution | false-neg |

**Key empirical finding (drove the design):** the scalar-expr form does **not**
re-emit verbatim. `double(0) < ddouble` has no `operator<`; the parameter's own
type token must be substituted throughout the body (`double(0)` → `ddouble(0)`).
Verified by single-TU probe before coding and by the checked-in compile test.

**Conservative-parser contract held:** every uncertainty is a `None` (LLM fallback),
never a speculative emission. False positives are the STOP #P/#K hard-fail.

---

## 2. Emitter output samples (one per shape, whitespace-normalized)

All overloads use `auto` return deduction → **zero app-specific return-type
knowledge** (verified: `auto` deduces `ddouble`/`ddcomplex`/`int`/`bool` correctly).

**Delegation** (template param → BOTH scalar and complex, each STOP #S-guarded):
```cpp
namespace ql {
    // Subtask L1' shallow-wrapper synthesis: kAbs (delegation → quad::ddfun::abs); source primary re-emitted at quad::ddfun::ddouble.
    KOKKOS_INLINE_FUNCTION auto kAbs(quad::ddfun::ddouble const& x) { return quad::ddfun::abs(x); }
}
namespace ql {
    // Subtask L1' shallow-wrapper synthesis: kAbs (delegation → quad::ddfun::abs); source primary re-emitted at quad::ddfun::ddcomplex.
    KOKKOS_INLINE_FUNCTION auto kAbs(quad::ddfun::ddcomplex const& x) { return quad::ddfun::abs(x); }
}
```

**Accessor:**
```cpp
namespace ql {
    // Subtask L1' shallow-wrapper synthesis: Real (accessor .real()); source primary re-emitted at quad::ddfun::ddcomplex.
    KOKKOS_INLINE_FUNCTION auto Real(quad::ddfun::ddcomplex const& z) { return z.real(); }
}
```

**Scalar-expr** (param-type token widened — the load-bearing transform):
```cpp
namespace ql {
    // Subtask L1' shallow-wrapper synthesis: Sign (scalar-expr (param-type widened)); source primary re-emitted at quad::ddfun::ddouble.
    KOKKOS_INLINE_FUNCTION auto Sign(quad::ddfun::ddouble const& x) { return (quad::ddfun::ddouble(0) < x) - (x < quad::ddfun::ddouble(0)); }
}
```

**Transitive** (inner Class-1 call preserved; the dep's own overload is emitted too):
```cpp
namespace ql {
    // Subtask L1' shallow-wrapper synthesis: iszero (transitive → kAbs); source primary re-emitted at quad::ddfun::ddouble.
    KOKKOS_INLINE_FUNCTION auto iszero(quad::ddfun::ddouble const& x) { return ql::kAbs(x) < quad::ddfun::ddouble(1e-20) ? true : false; }
}
```

Each carries a comment naming its source primary + the Subtask ID. Deduplicated by
`(qualifier, fn, target)`; idempotent (byte-identical on re-add — STOP #Q guard in
`OverloadSet.add`).

---

## 3. Wire-up before/after

The pre-LLM pass (`regional.synthesize_shallow_wrappers`, called from
`run_integrate_region` §4b) runs on every extended (`emit_bridges`) region. On the
**real `Lnrat` body** (`kokkosUtils.h:140`, the B10 leaf):

```
region: return TOutput(ql::kLog(ql::kAbs(x / y))) - (ql::Constants<TScale>::template
        _ipio2<TOutput,TMass,TScale>() * TOutput(ql::Sign(-x) - ql::Sign(-y)));

BEFORE (pipeline today): ql::kLog / ql::kAbs / ql::Sign are app-qualified calls the
        Gap-A math scan ignores (not _MATH_FN_NAMES) → left entirely to the LLM;
        the promoted clone build fails on the missing dd overloads (P2 build-A: 5
        errors).

AFTER (L1′):
  recognized (deterministic, removed from LLM path): [Sign, kAbs, kLog]
  remaining (LLM path):                              []
  overloads emitted into shim:                       [Sign, kAbs, kLog]
```

Injection reuses `shim_merge.merge_into_canonical` (keep-first dedup), so a
synthesized overload never double-defines against LLM output. The existing bridge
lint (`_lint_qualified_bridges`) is unaffected — it only ever scanned
`_MATH_FN_NAMES`, so app wrappers were never lint targets; the wire-up test asserts
the shim carrying the synthesized overloads passes the lint clean.

**STOP #R (wire-up regression):** all 31 existing Gap-A / include-lint /
gap-integration tests stay green; the deterministic path claims **only** app-qualified
non-math calls, leaving the `<cmath>` math-bridge path byte-for-byte unchanged.

---

## 4. `is_class1_synthesizable` API surface (for L2)

```python
def is_class1_synthesizable(
    qualified_name: str,          # "ql::kAbs" or bare "kAbs"
    primary_body_source: str,     # the wrapper's primary definition text
    surface: VendoredSurface,     # from surface_from_spelling(...) / scan_vendored_ops
    *, is_synth_dep=None,         # predicate for transitive deps (or None to disable)
) -> bool
```

Pure, side-effect-free (no emission, no shim mutation). Returns `True` iff the
recognizer classifies `primary_body_source` as one of the four shapes AND the parsed
name matches `qualified_name`'s final component. L2's `clonable_leaf` predicate calls
this (clause (2)(ii)) to decide whether a leaf's promoted body names only
synthesizable / vendored / source symbols. Supporting surface helpers L2 also gets:

* `surface_from_spelling(cpp_scalar, cpp_complex, scalar_ops=, complex_ops=)` — build
  a `VendoredSurface` from concrete type spellings (framework-agnostic).
* `scan_vendored_ops(header_texts, scalar, complex_type)` — `(scalar_ops, complex_ops)`
  the vendored headers actually provide (grounds the STOP #S guard).
* `synthesize_for_region(region_text, promoted, sources, surface) -> SynthesisResult`
  — the region-level pass (also what the regional dispatch calls).

---

## 5. STOP audit

| STOP | condition | outcome |
|---|---|---|
| **#P** | recognizer classifies a non-shallow body as Class-1 (emitted overload won't compile) | **DISCHARGED** — checked-in compile test (`test_synthesized_overloads_compile_and_run`, `@kokkos`) builds the emitter's ACTUAL output for all four shapes against real vendored dd + Kokkos, exit 0. During development the probe caught a real false-positive (see #T) and the recognizer was tightened before it could ship. |
| **#Q** | emitter non-idempotent / non-dedup | **DISCHARGED** — `OverloadSet.add` raises `ShallowWrapperError` on a key collision with differing text; idempotence + dedup + stable-order tests green. |
| **#R** | existing Gap-A test flips path (math call → deterministic, or vice versa) | **DISCHARGED** — 31 Gap-A/lint tests green; deterministic path claims only NON-`_MATH_FN_NAMES` app calls; `_MATH_FN_NAMES` untouched. |
| **#S** | delegation to a `_MATH_FN_NAMES` op the vendored surface can't map | **HANDLED (refuse + fallback)** — `recognize` refuses when `inner_fn ∉ scalar_ops ∪ complex_ops`; the surface's op sets are scanned from the real headers, so a genuinely absent op is refused, not invented. Unit test `test_stop_s_refuses_unprovided_op`. |
| **#T** | a new gen-defect class downstream (e.g. wrapper needs template-arg substitution) | **FIRED, then SCOPED OUT (by design).** The compile probe surfaced that the **real** qcdloop `iszero` body names foreign template params (`TOutput`/`TMass`) inside a `Constants<TScale>::_x<...>()` accessor — a concrete emitted overload can't bind them, and the emitter does not do full template-arg substitution. Per the brief ("recognizer scope question, not an implementation bug"), the recognizer now **refuses** such bodies (STOP #T guard: body names a template param ≠ the parameter's own type → `None` → LLM path). This is a conservative false-negative, **safe**. Documented as design §7 item 4 (recognizer beyond straight-line delegation = future work). **B10 does not need `iszero`** (Lnrat names only kLog/kAbs/Sign/Real/Imag + source `_ipio2`, §2.7), so this does not block the headline case. A *simpler* transitive form (inner Class-1 call + cast to the param's own type, no foreign template param) IS synthesized and compile-tested. |
| **#K** | emitted transform breaks build/runtime (leaf-promotion parent STOP) | **guarded** — the compile test is the pre-emission proof for the shapes L1′ emits; L2/L-measure carry the full guard for the clone bodies. |

No STOP was left silently open. STOP #T fired exactly as the brief anticipated and
was resolved by tightening the recognizer (not by shipping a broken emitter) — the
compile probe is what caught it, validating the "make the P2 probe a permanent
regression" requirement.

---

## 6. Test + suite results

* `tests/integrator_base/test_shallow_wrapper.py` — **35 passed** (recognizer
  positives ×7, negatives ×9 incl. STOP #S/#T, emitter ×5, idempotence/dedup ×3,
  region synthesis ×5, manifest API ×3, wire-up ×3, **compile probe ×1 (ran, not
  skipped, 3.5 s)**).
* `tests/patcher tests/shared tests/integrator_base -m "not llm"` — **423 passed,
  2 pre-existing failures** (`test_source_hash_gate.py::{test_source_hash_is_preserved,
  test_committed_shim_is_a_cache_hit_and_untouched}`) — these assert the **whole-app
  tracked-integrator** committed shim (`runs/qcdloop/src/ql_tracked_interop.hpp`)
  regenerates byte-identically; they fail **identically on baseline `0c9a3a3`**
  (verified via `git stash`), are orthogonal to the regional integrator this Subtask
  touches, and reflect committed-fixture drift, not an L1′ regression.
* One live-LLM test (`test_real_llm_ieps50_derived_not_r4`, `@llm`) flaked on a
  `complex<ddouble>` LLM misgen this run — non-deterministic, unrelated (the
  `_ieps50` region has **zero** app-qualified calls, so L1′ is provably inert on it).

---

## 7. Hard-constraint compliance

* **No app-specific identifiers hardcoded** — recognizer works on body shape;
  `kAbs`/`kLog`/`ql`/`Lnrat` appear only in tests/comments as examples.
* **`_MATH_FN_NAMES` untouched** — imported read-only; the delegation target must be
  in it, but no app name was added.
* **Conservative parser** — false-negatives everywhere uncertain; the only hard-fails
  are STOP #P/#Q (emit-time invariants), both test-guarded.
* **No `runs/qcdloop_headers_full/` change** — vendored snapshot untouched
  (`e3d2e45` intact).
* **No rule (d) implementation** — this Subtask produces the machinery rule (d)
  consumes (`is_class1_synthesizable` + `synthesize_for_region`), not the
  frame-discovery logic.

---

## Verdict

**L1′ SUCCESS.** The shallow-wrapper recognizer + synthesized-overload emitter land
as a structural extension of the Gap-A machinery. All four recognized shapes produce
valid dd overloads that **compile + run** against the vendored surface (permanent
regression test). The recognizer's scope is now empirically pinned: straight-line
delegation / accessor / scalar-expr / simple-transitive are synthesized; a body
requiring full template-argument substitution (the real `iszero`) is a documented
STOP #T refusal that falls back to the LLM path — and **B10's `Lnrat` leaf needs none
of it** (kLog/kAbs/Sign/Real/Imag all synthesize cleanly). `is_class1_synthesizable`
is implemented + tested for L2 to consume in its `clonable_leaf` predicate.

**STOP before dispatching L2.** Reet decides L2 dispatch (rule (d) frame-discovery +
`clonable_leaf` consuming this synthesis manifest).
