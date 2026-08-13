# Leaf-Callee Promotion — Design Notes (Group A precision-lift unblock)

Status: **v3 — design + falsification probes done, STOP before implementation** (2026-07-27).
v2 was 2026-07-27 (earlier same day); v1 was 2026-07-26. Scope unchanged: extend
closure-scoped dd promotion (`CLOSURE_SCOPED_CHAINS_DESIGN.md`) so leaf callees like
`ql::Lnrat` / `ql::ddilog` become **cloned, promoted frames** instead of a
`chain_closure_escapes` refusal frontier.
Discipline: **design only** — no changes to `agents/`, `tests/`. The §6 probes were built
and run (Subtask-5-style single-TU); nothing else moved.

> ## What changed v2 → v3 (read this first)
>
> **The source input got richer.** Commit `e3d2e45`
> ("qcdloop-under-test: add kokkosMaths_dd.h (43-coeff Constants<T> at dd) as source
> input") added `runs/qcdloop_headers_full/kokkosMaths_dd.h` (402 lines) to the vendored
> qcdloop snapshot. This is **qcdloop's own dd-precision `Constants<T>`**, copied verbatim
> from `ReetBarik/qcdloop@ddfun_enabled` commit `2229ec4` — the same fork whose validation
> build is the oracle, but this *specific header* is qcdloop-authored **library data**, not
> oracle-only knowledge. It is exactly parallel to how the mainline
> `scarrazza/qcdloop:tools.cc` publishes both the 19-double and the 43-quadmath `_C` tables
> side by side. A one-line namespace shim at the top
> (`namespace ql { namespace ddfun = ::Kokkos::Experimental; }`) aliases the fork's `ql::ddfun` to
> this repo's vendored `Kokkos::Experimental` primitives (`third_party/include/dd_math.hpp` etc.);
> the body is otherwise verbatim upstream. `kokkosMaths.h` (the double primary) is
> **byte-identical to before** — the double build path is unchanged.
>
> **What this gives the pipeline (proven by §6 probe P5, `probe_constants_dd43.cpp`):**
>
> * `ql::Constants<Kokkos::Experimental::DoubleDouble>::_num_C()` = **43** (not 19).
> * `_C(i)` returns **bit-exact** dd Chebyshev coefficients from the source table
>   (`_C(0)/_C(18)/_C(42)` bit-verified against the header's `DoubleDouble::from_bits()` literals).
> * `_pi()` returns `ddfun::DoubleDouble_pi()` — **bit-exact dd π**, not `T(M_PI)`.
> * dd-appropriate scalar tolerances (`_eps=1e-12`, `_reps=1e-30`, `_neglig=1e-28`,
>   `_qlonshellcutoff=1e-20`, `_ieps50`, …) and a 25-term Bernoulli `_B` table, all as source.
>
> **Net effect on the plan — the v2 Class-2 problem dissolves for Group A:**
>
> 1. **§2.3 collapses.** v2's Option B ("accept the library's 19-coeff series at dd and
>    bound the truncation floor") no longer applies, because the pipeline now sees the
>    **43-coeff** table directly from source. Class 2 resolves via **source enrichment**, no
>    synthesis and no vendoring — the pipeline consumes qcdloop's published dd table as
>    source input, exactly as it consumes the double primary. Options A/B/C from v2 become
>    *historical alternatives* ("what we would have done if the library didn't publish dd
>    tables"), retained for the reasoning trail, not funded.
> 2. **§2.4 is repurposed.** The v2 "+8-to-+16 predicted band" + truncation-cancellation
>    caveat existed **only** because Option B forfeited the 43-coeff tail. With the 43-coeff
>    table available, ddilog's dd accuracy is **not truncation-limited** within dd's ~1e-32
>    arithmetic floor. v3 **restores the closure design's original +18.43-digit ceiling**
>    (Item 7's prediction) as the headline B10 prediction; the v2 band analysis is noted as
>    superseded. The falsifier keeps its shape (measured lift < +8 → STOP #A, value-flow
>    model wrong) but is **no longer diagnostic of a truncation problem**.
> 3. **The dispatch shape simplifies.** v2's optional **Subtask L4** (Option-A DCT
>    coefficient generator) is **removed from the Group-A plan** — no longer needed, because
>    the library publishes the table. It is retained only as a named contingency for a
>    *future* library-under-test that omits a needed dd table. Sequence becomes
>    L1′ → L2 → L3 → L-measure, **~10–15 days** (drops the ~2–3 wk L4).
> 4. **STOP #O softens.** The Class-2 capability-gap STOP is retained as a general safety
>    net but **does not fire for B10/B12/B13** under v3 — it is only reachable if a
>    library-under-test omits a dd-precision table the pipeline needs (not the case for the
>    current qcdloop-under-test after `e3d2e45`).
>
> **Everything from v2 that was independent of the coefficient-table question stands and is
> carried forward:** the Class-1 synthesized-wrapper story (§2.2, the whole point that the
> pipeline synthesizes the mechanical `ql::kAbs/kLog/…` overloads rather than vendoring
> them), the clone/STOP-#K rename discipline (§3), rules a/b/c interaction (§4), the
> termination/acyclicity proof (§2.6–2.8), per-integral clones, and the four v2 probes.

> ## What changed v1 → v2 (retained for the trail)
>
> v1's central resolution (§3.4) was to *vendor* a hand-ported `dd_ql_support.hpp` from
> `qcdloop@ddfun_enabled:src/qcdloop/kokkosMaths_dd.h` — importing qcdloop-specific dd
> knowledge (the Chebyshev series, the `ql::` dd helper overloads) the pipeline should
> synthesize. Reet's architectural line rejected this: `third_party/include/` is vendored,
> **app-independent** dd/ff primitives that know nothing about qcdloop; anything
> qcdloop-specific is the **pipeline's job to synthesize** from qcdloop's own source + the
> vendored primitives. `qcdloop@ddfun_enabled` is a **validation oracle, not a generation
> cheat-sheet**. v2 confronted this by splitting the category-(d) support surface into
> **Class 1 (pipeline-synthesizable mechanical wrappers)** and **Class 2 (precision-target
> coefficient tables)**, choosing Option B for Class 2 (19-coeff at dd from the unmodified
> double primary). v1's Subtask L1 ("vendor + port dd_ql_support.hpp") was deleted and
> replaced by L1′ (synthesize Class-1 in the agents tree).
>
> **v3 note on the v1/v2 vendoring debate:** the `e3d2e45` header is *not* a return to v1's
> `dd_ql_support.hpp`. v1 proposed vendoring a hand-ported support header into
> `third_party/include/` (app-independent primitives) and having the pipeline *depend on
> it as a primitive*. v3's header lives in `runs/qcdloop_headers_full/` — the **vendored
> snapshot of the library under test** — as qcdloop's own published data, consumed as
> *source input* exactly like `kokkosMaths.h`. `third_party/include/` remains app-independent
> (§4 architectural invariant preserved). The distinction is load-bearing: v1 wanted the
> pipeline's *primitive layer* to know qcdloop; v3 lets the pipeline *read qcdloop's own
> source*, which is what it is supposed to do.

Reads as the successor to two docs, both of which stand:
* `CLOSURE_SCOPED_CHAINS_DESIGN.md` — rules (a)/(b)/(c), the §2.4 refusal frontier, the
  §3 designed-exit gate, the §7 STOP discipline. **This design changes exactly one thing
  in it:** it moves `ql::Lnrat`/`ql::ddilog` from the §2.4 "callee-not-in-`F`" refusal set
  into `F` as clonable frames. Everything else in that doc is unchanged.
* `tier_b_stage2_subtask5/TIER_B_STAGE2_SUBTASK_5_2026-07-26.md` (STOP #K) — proved the
  *forwarding-overload* path is unsound (self-recursion). This design takes STOP #K's own
  recommended option 2 ("make `Lnrat`/`ddilog` chain-frames") and works out whether it is
  actually buildable. **The answer is: yes** — the support surface it needs is
  pipeline-synthesizable (Class 1) plus a **source-resident 43-coeff coefficient table**
  the library now publishes (Class 2, v3) — no vendored qcdloop-specific *primitive* header.

> **Load-bearing correction up front (read before §1).** The closure design's §2.4
> lists `ql::Lnrat` as a hard refusal because "its signature we will not touch." That is
> the wrong lens for a leaf callee. We are **not** touching `ql::Lnrat`'s signature — we
> **clone** it to `Lnrat_B10` (a new symbol, reachable only from the chain's rerouted
> call sites) and promote the clone's body, exactly as the pipeline already clones
> `ddilog`/`Li2omx2` whose bodies sit *on* the chain. STOP #K's recursion pit was a
> property of a same-name *overload*, not of a *renamed clone*. The §6 probes confirm the
> renamed clone builds and runs on the exact inputs that segfaulted the overload. The real
> blocker is not recursion and not signatures — it is the dd **support surface** the
> clone's body names. **v2's finding: that surface splits into a synthesizable class
> (Class 1) and a coefficient-table class (Class 2). v3's finding: the coefficient-table
> class is now source-resident at full 43-coeff precision, so it stops being a capability
> question at all.** §2 is the whole ballgame.

---

## 1. Scope of the extension

### 1.1 What the closure does today vs. what this adds

The closure (`compute_value_closure`, rules a/b/c) grows a carried-value set over a
**fixed** frame set `F`. `F` is the set of functions the dominant-chain selector's
line-set spans — i.e. functions **with at least one chain line in their body**. For B10,
`F = { B1m.h driver, Li2omx2, ddilog }`: `Li2omx2` and `ddilog` are in `F` because chain
lines fall inside them (`kokkosUtils.h:702-704` in `Li2omx2`; `:174,177,199,212` in
`ddilog`). They get cloned (`Li2omx2_B10`, `ddilog_B10`) and their bodies promoted.

`Lnrat` is **not** in `F`: no chain line falls inside `Lnrat`'s body (`kokkosUtils.h:141-155`).
It is a **leaf callee** — called *from* a chain line (`Li2omx2:701,706`) but its own
body is never selected. Today rule (b)/(2.4) hits the `Lnrat` call, sees a callee ∉ `F`,
and **refuses** (`chain_closure_escapes`). That refusal is what blocks B10: `Li2omx2_B10`
promoted to dd needs `ql::Lnrat<DoubleDoubleComplex,…>(dd,dd)` to exist, and it does not.

**This design grows `F`.** It adds a fourth rule to the closure fixed point:

> **(d) Leaf-callee promotion.** If a carried value is produced by (or flows as a promoted
> argument into) a call `ql::g<…>(…)` whose callee `g` is a **clonable leaf** (predicate
> §1.2) and `g ∉ F`, then add `g` to `F` as a new frame, clone it to `g_<integral>`, seed
> the clone's body into the closure, and reroute the call site to `g_<integral>`. Recurse:
> the clone's own body may call further clonable leaves, which rule (d) pulls in turn.

Rules (a)/(b)/(c) then run **unchanged** over the enlarged `F`. Rule (d) is purely a
*frame-discovery* rule; it does not change how any frame's body is promoted.

### 1.2 The clonable-leaf predicate

Reet's suggestion — "primary template compiles clean at dd against the vendored surface" —
is the right shape, and the Class-1/Class-2 split (§2) makes it decidable **without
vendoring a support header**. Refined predicate, all clauses required:

```
clonable_leaf(g) :=
   (1) g is a function template with a body available in the analysed headers
       (not an extern/vendored-binary symbol);              # can be cloned at all
 ∧ (2) g's body, with reads promoted to dd, calls ONLY:
         - other clonable_leaf callees (recurse), OR
         - symbols the dd TERMINATION BOUNDARY resolves at dd (§2.6):
             (i)   vendored Kokkos::Experimental ops (abs/log/…, DoubleDoubleComplex ops),
             (ii)  a Class-1 SYNTHESIZED wrapper (§2.2) — pipeline-emittable
                   mechanically from that wrapper's own primary + (i),
             (iii) a Class-2 data accessor the SOURCE instantiates at dd
                   (§2.3): the double primary at T=DoubleDouble, OR the enriched
                   dd source `kokkosMaths_dd.h` (43-coeff _C, dd _pi, …),
       i.e. rule (d)'s transitive closure over g terminates at the boundary;
                                                             # body instantiable at dd
 ∧ (3) g is NOT self-recursive under a SAME-NAME overload set that a rename
       cannot separate (STOP #K guard — §3);               # rename discipline safe
 ∧ (4) cloning g does not require widening a shared g PARAMETER that a
       non-chain caller also binds — the clone gets its own params, so this
       is automatically satisfied for a pure clone, but a leaf whose promotion
       demands INWARD dd on a shared original's signature is refused (§8.2 of
       the closure design still holds).
```

Clause (2) is decidable **against a synthesis manifest + a source-instantiation check, not
a vendored primitive header**. A leaf is clonable iff every dd symbol its promoted body
names is (i) vendored, (ii) a Class-1 wrapper the extended Gap-A machinery **can
synthesize** (predicate: primary body is a straight-line delegation to a vendored/ADL
op or a member accessor — §2.2), or (iii) a Class-2 accessor the **source instantiates at
dd** (§2.3 — either the double primary at `T=DoubleDouble`, or, for coefficient tables, the
enriched dd source `kokkosMaths_dd.h`). Anything else → the leaf is **not** clonable →
`chain_closure_escapes` (honest terminal, not a doomed emission). This keeps the
conservative-parser contract: false-negative (refuse a clonable leaf) is safe;
false-positive (clone an un-instantiable leaf) is the STOP #K hard-fail we must never ship.

### 1.3 Which frames become eligible

For the Group-A chains, rule (d) makes exactly these leaves eligible:

| leaf | called from | primary | promotable body? |
|------|-------------|---------|------------------|
| `ql::Lnrat` (TScale overload, `:153`) | `Li2omx2:701,706` | straight-line `kLog/kAbs/Sign/_ipio2` | yes — all Class-1 (§2.2) + source Constants (§2.3); **§6 probe: builds+runs** |
| `ql::ddilog` (`:163`) | `Li2omx2:702,708` (leaf on B12/B13) | Chebyshev series over `_C` | already IN `F` for B10 (chain lines inside it); **needed for B12/B13**; uses Class-2 `_C` (§2.3, now **43-coeff source**) |
| `ql::kfn`, `ql::ltspence`, `ql::cspence` | Group-B chains | series / branch | out of scope (Group-B dd-insufficient, §7) |

So for **B10** specifically, rule (d) is needed for **`Lnrat` only** — `ddilog`/`Li2omx2`
are already cloned frames. This narrows the headline case to a single leaf whose entire
support surface is Class-1 wrappers plus the source's own `Constants<DoubleDouble>::_ipio2`
(no `_C` — Lnrat has no series). **The B10 unblock therefore needs no Class-2 coefficient
work at all.** Class 2 becomes relevant only when `ddilog` is itself a rule-(d) leaf
(B12/B13), where under v3 it is served by the **43-coeff source table** (§2.3).

---

## 2. Support-surface scoping — the two classes (the crux)

This is the section the whole design turns on, and where the §6 probes did their work. v1
treated the category-(d) surface as one un-vendored monolith. v2 split it into Class 1
(synthesizable) and Class 2 (coefficient tables). **v3 resolves Class 2 by source
enrichment** — the library now publishes its dd table.

### 2.1 The B10 support-surface bill of materials, re-classified

Every dd symbol the B10 closure's promoted bodies name, sourced from `Lnrat` body
`kokkosUtils.h:141-155`, `ddilog` `:163-232`, `Li2omx2` `:692-712`; helper defs
`src/kokkosMaths.h:250-372`; vendored `third_party/include/*`; and — for coefficient
tables — the enriched dd source `runs/qcdloop_headers_full/kokkosMaths_dd.h`. **The
`qcdloop@ddfun_enabled` oracle is NOT consulted here except through that one vendored
header** — the classification is derived from qcdloop-under-test source + vendored
primitives only.

| symbol (at dd) | used by | vendored? | source primary at dd? | **class** |
|---|---|---|---|---|
| `ddadd/sub/mul/div`, `DoubleDouble`/`DoubleDoubleComplex` ops | all | ✅ `dd_math`/`dd_complex` | — | boundary (vendored) |
| `abs/log/sqrt/exp/pow` on dd | ddilog, Lnrat | ✅ `Kokkos::Experimental::*` | — | boundary (vendored) |
| `ql::kAbs(DoubleDouble/DoubleDoubleComplex)` | Lnrat, Li2omx2 | ❌ | primary `T kAbs(T){Kokkos::abs(x)}` — redirect | **Class 1** (§2.2) |
| `ql::kLog(DoubleDouble/DoubleDoubleComplex)` | Lnrat, ddilog, Li2omx2 | ❌ | primary `T kLog(T){Kokkos::log(x)}` — redirect | **Class 1** |
| `ql::kSqrt/kConj(dd)` | (Group A: not on chain) | ❌ | primary `T kSqrt(T){Kokkos::sqrt(x)}` — redirect | **Class 1** |
| `ql::Real/Imag(DoubleDoubleComplex)` | ddilog, Lnrat | ❌ | primary is `.real()/.imag()` accessor | **Class 1** |
| `ql::Sign(DoubleDouble)` | Lnrat, ddilog | ❌ | primary `(0<x)-(x<0)`, T-generic ±1/0 | **Class 1** |
| `ql::iszero<…>(DoubleDouble)` | ddilog (`:116`) | ❌ | template; body = `kAbs(x)<_qlonshellcutoff` | **Class 1** (transitive: `kAbs` + source cutoff) |
| `ql::kPow<…>(DoubleDouble,int)` | ddilog (`:117…`) | ❌ | template `TOutput(1.0); temp*=base` — clean at dd | **source (already instantiates)** |
| `_pi2o6/_ipio2/_half/_pi/_zero/_one` at dd | ddilog, Lnrat, Li2omx2 | partial (`DoubleDouble_pi()`) | double primary at dd; **or dd source (`_pi()`=`DoubleDouble_pi()` bit-exact)** | **Class 2 / source** (§2.3) |
| `Constants<DoubleDouble>::_C(i)`, `_num_C()` | ddilog | ❌ | **enriched dd source: 43 coeffs, bit-exact** (P5) | **Class 2 / source** (§2.3, v3) |

**Only the Class-1 wrappers are not vendored-boundary or already-in-source.** The Class-2
coefficient table `_C` — v2's one genuine capability question — is now **source-resident at
full 43-coeff precision** via `kokkosMaths_dd.h`. The whole re-classification is: Class 1 is
synthesized (§2.2), Class 2 is read from source (§2.3). v1's "single un-vendored file"
framing conflated these and concluded (wrongly) that a vendored primitive header was
required.

### 2.2 Class 1 — pipeline-synthesizable via extended Gap-A machinery

*(Unchanged from v2 — this class is orthogonal to the coefficient-table enrichment. Carried
forward; the §6 P2 probe re-confirms it against the current tree.)*

**Definition.** A Class-1 wrapper is a shallow app-specific function whose primary body is a
**straight-line delegation** to (a) a vendored/ADL-reachable op, or (b) a member accessor,
or (c) a type-generic scalar expression — such that its dd overload is a **mechanical
transform** of that primary body.

**For each Group-A wrapper — the existing primary, the mechanical transform, and Gap-A reach:**

| wrapper | primary (src/kokkosMaths.h) | mechanical dd transform | current Gap-A reach? |
|---|---|---|---|
| `ql::kAbs` | `:271` `T kAbs(T x){ return Kokkos::abs(x); }` (+`:279/285` double/cplx overloads) | emit `DoubleDouble kAbs(DoubleDouble){ return Kokkos::Experimental::abs(x); }` + `DoubleDouble kAbs(DoubleDoubleComplex){ return Kokkos::Experimental::abs(z); }` — redirect `Kokkos::abs`→`Kokkos::Experimental::abs` | **needs extension** (see below) |
| `ql::kLog` | `:289` `T kLog(T x){ return Kokkos::log(x); }` | `DoubleDouble kLog(DoubleDouble){ Kokkos::Experimental::log }`, `DoubleDoubleComplex kLog(DoubleDoubleComplex){ Kokkos::Experimental::log }` | **needs extension** |
| `ql::kSqrt`/`kConj` | `:295/301` `Kokkos::sqrt/conj` | analogous redirect to `Kokkos::Experimental::sqrt/conj` | **needs extension** |
| `ql::Real`/`Imag` | `:320-326` `.real()/.imag()` accessors on `complex<double>` | emit `DoubleDouble Real(DoubleDoubleComplex z){ return z.real(); }` etc. | **needs extension** (accessor form) |
| `ql::Sign` | `:328` `int Sign(double x){ return (0<x)-(x<0); }` | re-emit with dd operands: `int Sign(DoubleDouble x){ return (DoubleDouble(0.0)<x)-(x<DoubleDouble(0.0)); }` | **needs extension** (scalar-expr form) |
| `ql::iszero` | `:307` template, body `kAbs(x)<_qlonshellcutoff` | already a template — instantiates at dd once `kAbs(dd)` exists + `_qlonshellcutoff` (source literal) | **transitive** (falls out once the above land) |

**Why this is an *extension* of Gap-A, not a new capability.** The existing Gap-A bridge
(`regional.py:64-199`) already synthesizes overloads that inject into a namespace to
redirect a **namespace-qualified math call** onto the vendored `quad::` surface. Today it
fires only for the **standard `<cmath>` vocabulary** (`_MATH_FN_NAMES` = `abs/log/sqrt/…`)
and only for calls where the qualifier root is not vendored — precisely to bridge
`Ns::sqrt(promoted)` → the vendored op. The `ql::kAbs/kLog/kSqrt` wrappers are the **same
pattern one delegation-hop removed**: `ql::kAbs`'s *body* is `Kokkos::abs(x)`, a
`_MATH_FN_NAMES` call whose qualifier (`Kokkos`) has no dd overload.

**What specifically extends (design, not code):**

1. **A "shallow-wrapper" recognizer.** Extend the Gap-A scan so that, for an app-qualified
   call `ql::g(promoted)` where `g ∉ _MATH_FN_NAMES`, it inspects `g`'s **primary body**
   (already reachable via `region_scan`/`CallGraph`) and classifies it as Class-1 iff the
   body is a single `return <delegation>;` where `<delegation>` is one of:
   * `Kokkos::fn(arg)` / `quad::…::fn(arg)` with `fn ∈ _MATH_FN_NAMES` → **redirect** the
     inner call to `Kokkos::Experimental::fn` (the transform the bridge already knows);
   * `arg.real()` / `arg.imag()` / other member accessor → **accessor passthrough**;
   * a scalar comparison/arithmetic expression over the parameter with no non-boundary call
     → **re-emit verbatim at dd** (the parameter's type widens; the operators are vendored).
   Anything not matching these shapes → **not Class-1** → the leaf fails clause (2) → refuse.
   This recognizer is deterministic and conservative (unrecognized body ⇒ refuse), matching
   the parser contract.

2. **A synthesized-overload emitter** that, given `(g, primary_body, dd_target)`, produces
   the dd overload text and injects it into namespace `ql` alongside the existing shim —
   reusing the same injection/using-declaration remedy the Gap-A lint already sanctions
   (`_shim_bridges_qualifier`). The emitted overloads are exactly the §6 P2 probe's
   `WITH_SYNTH` block, produced mechanically instead of hand-written.

3. **`_MATH_FN_NAMES` stays a `<cmath>` vocabulary.** The extension does **not** bake in
   `kAbs/kLog/…` names (that would violate "no app-specific identifiers" —
   `[[feedback_no_placeholder_patterns]]`). It recognizes app wrappers **structurally**
   (body is a straight-line delegation to a `_MATH_FN_NAMES` op or an accessor), so it
   works for any app's shallow math wrappers, not qcdloop's specifically.

**A v3 subtlety worth stating explicitly.** The enriched `kokkosMaths_dd.h` *also* contains
the fork's own hand-written Class-1 dd wrappers (`kAbs`/`kLog`/`Sign`/`Real`/`Imag` at dd,
lines 294–400). **The pipeline does not consume those wrapper definitions** — it synthesizes
its own Class-1 overlay (§2.2 point 2) from the *double* primary's one-line bodies, keeping
Class-1 app-independent and the "pipeline synthesizes what it needs" claim intact. From
`kokkosMaths_dd.h` the pipeline consumes only the **data** the double primary cannot derive
from its own body: the 43-coeff `_C` table, the 25-term `_B` table, and the bit-exact dd
`_pi()` (§2.3). The wrapper *code* in that header is redundant with what §2.2 emits and is
ignored, so the two paths never collide (per-region shim vs source header; §4).

**Empirical proof (§6 P2).** Probe `probe_clone_synth.cpp` build B compiles and runs
`Lnrat_B10` with a `WITH_SYNTH` overlay that is **only** these mechanical Class-1 overloads
(no hand-written `Constants`, and NOT the fork's wrappers), against the source
`Constants<DoubleDouble>` primary. `|diff| = 0` vs the double primary. This is the exact surface
the extended Gap-A machinery would emit.

### 2.3 Class 2 — coefficient tables, resolved via source enrichment (v3)

**Definition.** Class-2 data is a value the primary template **cannot derive from its own
body** — precision-target-specific *data*, not code. The load-bearing example is
`Constants<DoubleDouble>::_C`, the Chebyshev series for Li₂: a dd-accurate ddilog wants **43**
coefficients (the DCT of Li₂ sampled at 43 Chebyshev nodes), which cannot be invented from a
19-coeff double table.

**v3 resolution — the library publishes its own dd table.** After commit `e3d2e45`, the
vendored qcdloop-under-test snapshot carries `runs/qcdloop_headers_full/kokkosMaths_dd.h`:
qcdloop's own dd-precision `Constants<T>`, with `_num_C()=43` and bit-exact dd `_C(i)`. This
is qcdloop-authored **library data**, exactly parallel to how the mainline
`scarrazza/qcdloop:tools.cc` publishes both the 19-double and the 43-quadmath `_C` tables
side by side. **The pipeline consumes it as source, no synthesis needed** — the same way it
consumes the double primary `kokkosMaths.h`. The namespace shim
(`namespace ql { namespace ddfun = ::Kokkos::Experimental; }`) lets the fork-authored header resolve
against this repo's vendored `Kokkos::Experimental` primitives unchanged (§4).

**Decisive discovery (probe `probe_constants_dd43.cpp`, P5).** `ql::Constants<DoubleDouble>`
instantiated from the enriched source:

```
num_C = 43  (expect 43)
C[0]  hi=0.42996693560813698   lo=-7.726e-18  bit-exact=1
C[18] hi=-1.4226020855112447e-16 lo=4.699e-33 bit-exact=1
C[42] hi=-1.11772e-35          lo=4.466e-52   bit-exact=1
_pi() hi=3.1415926535897931    lo=1.225e-16   bit-exact DoubleDouble_pi=1
sum_C(43) hi=0.8224670334241132  lo=1.520e-17
P5 PASS: enriched source provides 43-coeff dd table
```

`_num_C()=43`; `_C(0)/_C(18)/_C(42)` are **bit-exact** vs the source table's `DoubleDouble::from_bits()`
literals; `_pi()` is bit-exact `DoubleDouble_pi()` (not `T(M_PI)`). The 43-coeff sum
`0.8224670334241132` matches the v2 19-coeff sum's `hi` (both → π²/12) with a *refined* `lo`
tail — the "same value, more accurate" signature of the extra coefficients. **STOP #E
(source doesn't provide what the design claims) is discharged.**

**The three options v2 posed, now historical for Group A.** Retained for the reasoning
trail — these describe *what we would have done if the library did not publish dd tables*:

* **~~Option A — pipeline computes the 43-coeff table offline (chebfun-style DCT).~~**
  *(No longer needed for Group A.)* A real capability extension: sample Li₂ at 43 dd
  Chebyshev nodes, DCT, emit a `Constants<DoubleDouble>::_C` specialisation, run once per
  (function, precision) pair offline. **Superseded by source enrichment** — the library
  already ships the table. Retained only as the contingency an unspecified *future*
  library-under-test would trigger via STOP #O (§5).
* **~~Option B — accept the 19-coeff series at dd.~~** *(v2's choice; no longer applies.)*
  The double primary at `T=DoubleDouble` gives 19 coeffs; v2 measured the truncation ceiling this
  forfeits (v2 §2.4). **Superseded** — with 43 coeffs in source there is no truncation
  concession to bound.
* **~~Option C — declare precision-target tables an out-of-scope library-author
  pre-condition.~~** *(Fallback only.)* Under v3 the library-author *has* published the
  table, so C is satisfied *by the source itself* rather than by shrinking the claim.

**v3's Class-2 stance:** consume the source's 43-coeff dd table directly. Zero synthesis,
zero capability gap, preserves the architectural line (source + vendored primitives only,
no oracle-for-generation). If a *future* library-under-test omits a needed dd table, Option A
becomes live and STOP #O (§5) fires; that is not the case for qcdloop after `e3d2e45`.

### 2.4 Predicted lift ceiling (v3 — full +18.43 restored)

v2 §2.4 predicted a **+8-to-+16 band** with a truncation-cancellation caveat, *only*
because Option B forfeited the 43-coeff tail: the 19-term Chebyshev truncation floored
ddilog's absolute accuracy at ~1e-16, and v2 had to argue about how much of that cancels in
the `dilog4−dilog5` difference. **That entire analysis is superseded by v3's source
enrichment.**

With the 43-coeff table in source (§2.3), ddilog's dd accuracy is **not truncation-limited**
within dd's arithmetic floor. The two error sources v2's P4 probe isolated in the
Clenshaw-summed Chebyshev recurrence (`ddilog:220-227`):

1. **Recurrence roundoff (the cancellation dd is *for*).** The Clenshaw loop
   `B0 = C_i + ALFA·B1 − B2` accumulates catastrophic cancellation; at double this
   contributes ~1e-16 error. dd carries ~18 extra digits through the recurrence, shrinking
   this to ~1e-32. This is exactly the error B10's downstream `dilog4−dilog5` cancellation
   amplifies, and dd removes it in full.
2. **Series truncation.** With 43 coefficients the Chebyshev tail is driven to ~`|C[42]| ≈
   1e-35` (P5: `C[42].hi = -1.1e-35`) — **below dd's ~1e-32 arithmetic floor**. Truncation
   is therefore **no longer the binding constraint**; the 19-coeff floor (~1e-16) that
   forced v2's band is gone.

**Restored prediction (closure design Item 7).** With both error sources at or below dd's
~1e-32 floor, the design predicts B10 recovers the **full +18.43-digit** cancellation lift —
the closure design's original Item-7 ceiling, which implicitly assumed the 43-coeff series.
No truncation shortfall, no band.

**Falsifier (same shape as v2, different meaning).** Measured B10 `kernel_measured_lift`
**< +8** falsifies the design's **value-flow model** (STOP #A) — an intervening double
narrowing on the chain, a mis-scoped closure, or a promotion that didn't land — **not** a
truncation problem, since v3 removed truncation as a possible cause. The v2 "truncation
decorrelates across a Chebyshev branch boundary" caveat **no longer applies**: with 43
coeffs the truncation is below the dd floor regardless of which range-reduction branch the
two ddilog arguments land in. So a sub-+8 measurement points squarely at the value-flow
plumbing, which is the far more actionable diagnosis.

> **v2 §2.4 (the +8…+16 band + truncation-cancellation reasoning) is retained in the git
> history and the v2 probe P4 output for the trail, but is SUPERSEDED and does not govern
> the v3 prediction.**

### 2.5 Does B10's read flow through `DoubleDoubleComplex` or `Kokkos::complex<DoubleDouble>`?

Unchanged from v1/v2 (this was correct). The chain's `TOutput` is `Kokkos::complex<double>`;
the existing pipeline promotes complex containers to `Kokkos::Experimental::DoubleDoubleComplex`
(`dispatch.py:308`, `fanout.py:243/271`, `shim_normalise.py:60-63`), and the
`ddilog`/`Li2omx2` clones already do this (B12 built + executed, Subtask 3). So B10's reads
flow through **`Kokkos::Experimental::DoubleDoubleComplex` directly**, via vendored `dd_complex.hpp` — no
`Kokkos::complex<DoubleDouble>`, no container-axis bridging. The §6 probe confirms:
`Lnrat_B10<DoubleDoubleComplex,double,DoubleDouble>` compiles and runs with `DoubleDoubleComplex` as `TOutput`.

### 2.6 The termination boundary (updated for v3)

Rule (d) recurses; it terminates because every call in a promoted body resolves to exactly
one of four **boundary** kinds, none re-entering rule (d):

1. **Vendored `Kokkos::Experimental` math** — `abs/sqrt/log/exp/pow/…` on `DoubleDouble`/`DoubleDoubleComplex`
   (`dd_math.hpp`, `dd_complex.hpp`). Resolve at dd, no cloning. **Boundary.**
2. **Class-2 / source constants** — `_pi2o6`, `_ipio2`, `_C`, `_num_C` at dd, instantiated
   **from source**: either the double primary `Constants<T>` at `T=DoubleDouble`, or (for the
   coefficient tables and bit-exact dd π) the enriched dd source `kokkosMaths_dd.h`
   (§2.3, proven by `probe_constants_dd43.cpp`). **Boundary — a value, not a frame.**
3. **Class-1 synthesized wrappers** — `ql::{kAbs,kLog,kSqrt,Real,Imag,Sign,iszero}` at dd,
   **emitted by the extended Gap-A machinery** (§2.2), not vendored. Once emitted they are
   ordinary overloads that bottom out in boundary 1. **Boundary.**
4. **Vendored `DoubleDoubleComplex` container ops** — `+,−,*,/`, `.real()`, `.imag()`
   (`dd_complex.hpp`). **Boundary.**

Rule (d) adds a frame only for **none-of-the-above** = an app template whose body is
available and calls into these boundaries. Recursion adds a frame at most once per app
template in the finite header set.

### 2.7 Bounded, acyclic — the proof (unchanged from v1/v2)

* The universe of clonable app templates is **finite**.
* Rule (d) is **monotone** and records each app template as a clone at most once → halts
  after ≤ (#app-templates) rounds.
* **No cycle forces unbounded growth.** The qcdloop special-function call graph on these
  chains is a **DAG**. `Lnrat`'s body (`:141-155`) calls `kLog/kAbs/Sign/Imag/Real/_ipio2`
  — all boundary kinds, **no app-template callee** → `Lnrat` is a **sink**; rule (d) adds it
  and stops. `ddilog`'s body calls `kLog/kPow/Real/Sign/iszero/_C/_pi2o6` — all boundary
  (`kPow`/`iszero` instantiate from source at dd; `_C` now the 43-coeff source table) → also
  a sink. `Li2omx2` calls `Lnrat`/`ddilog`/`kLog`/`kAbs`/`_ipio2` → its only app-template
  callees are the two sinks. So the B10 rule-(d) frontier is depth-1 and closed:
  `{Li2omx2_B10 → Lnrat_B10 (sink), ddilog_B10 (sink)}`. **Bounded, acyclic. QED.**
* **Self-recursion is not a cycle in `F`** — `ddilog`/`Lnrat` have no self-call (verified);
  and the rename (§3) binds any hypothetical self-call to the clone name.

### 2.8 The circuit breaker (backstop, unchanged)

No size *cap* (a cap re-introduces the subset boundary), but keep a **circuit breaker**: if
rule (d) would grow `F` past a diagnostic threshold (8 frames or rule-(d) recursion depth
> 3), abort with `chain_closure_oversized`. For Group A this never fires (B10 frontier
depth-1, 3 frames). Graceful degradation, not a scope choice.

---

## 3. Rename discipline (how the clone avoids the Subtask-5 self-recursion pit)

*(Unchanged from v1/v2 — carried forward verbatim; the §6 probes re-confirm it.)*

### 3.1 Why the forwarding overload recursed, and why the clone does not

STOP #K's recursion was structural: an injected **same-name** overload
`ql::Lnrat(DoubleDouble,DoubleDouble)` whose body calls `ql::Lnrat(DoubleDouble,DoubleDouble)` — C++ selects by
*argument type*, ignoring the explicit `<…>`, so it re-selects itself forever.

A **clone** breaks every link:
* the clone is a **distinct symbol** `Lnrat_B10` — no overload set to re-enter;
* the clone's body names only `ql::kLog/kAbs/Sign/Real/Imag/Constants` + vendored ops — it
  **never names `Lnrat_B10` or `ql::Lnrat`** (verified; `Lnrat`'s body has no self-call);
* the call site `Li2omx2_B10:706` is **rerouted** to `Lnrat_B10` by the existing
  topological callee-before-caller reroute (`_reroute_in_function`).

The §6 probes are the empirical proof: `Lnrat_B10<DoubleDoubleComplex,double,DoubleDouble>(1.5,2.5)`
**builds and runs to completion (exit 0), no segfault**, on the exact inputs that made the
Subtask-5 forwarding overload stack-overflow — under both the v1 hand overlay and the v2
synthesized-surface overlay.

### 3.2 Self-recursive leaves (general rule, not needed for B10)

If a clonable leaf contained a self-call, the clone-and-rename discipline rewrites in-body
self-calls to the clone name in the same descending-line edit pass that promotes reads
(how `ddilog_B10`/`Li2omx2_B10` self-references are already handled). Vacuous for B10.
Clause (3) of the predicate refuses a leaf whose self-recursion crosses a **same-name
overload set** a rename cannot separate (STOP-#K guard, generalised).

### 3.3 Overlapping clones across integrals

**Per-integral clones** (recommended, and what the pipeline already does): `ddilog_B10` vs
`ddilog_B12`, `Lnrat_B10` vs `Lnrat_B12` live in distinct per-integral variant trees
(`per_integral_orchestrator`), never coexist in one TU → no collision, no shared-instantiation
hazard. Preserves the Appendix invariant. Shared dd instantiation **rejected** (re-introduces
cross-integral coupling). No new collision surface beyond
`variant_naming.py`/`assert_no_collisions`.

---

## 4. Interaction with the existing tree

### 4.1 Vendored-snapshot policy change (v3)

`runs/qcdloop_headers_full/README.md` was updated (commit `e3d2e45`) to document a
**two-source snapshot**:

| file(s) | source | role |
|---|---|---|
| `boxGPU.h`, `kokkosMaths.h`, `kokkosMaths_wrapper.h`, `kokkosUtils.h`, `timer.h`, `box/*.h` | `qcdloop@master` `8de2089` | double-precision primary — the library under test |
| `kokkosMaths_dd.h` | `qcdloop@ddfun_enabled` `2229ec4` (+ 1-line namespace shim) | qcdloop's own dd-precision `Constants<T>` (43-coeff `_C`, 25-term `_B`, dd `_pi()`, dd tolerances) — consumed as **source input** for dd support |

The vendored snapshot is now the pipeline's **canonical view of "the qcdloop under test,"
including its dd-precision `Constants<T>`.** The README's edit policy keeps both files as
verbatim upstream mirrors (modulo the documented shim) and states explicitly that any dd
support the tables *don't* cover (e.g. the Class-1 `ql::kAbs`/`ql::kLog` wrappers at dd) is
**synthesized by the pipeline, not written into the snapshot** — i.e. the §2.2 Class-1
machinery, not the fork's wrapper code (§2.2 v3 subtlety). `third_party/include/` remains
**app-independent** vendored primitives (`Kokkos::Experimental`), untouched by this change; the
namespace shim in `kokkosMaths_dd.h` bridges the fork's `ql::ddfun` authorship onto it.

### 4.2 Component-by-component

| component | change |
|---|---|
| **rule (c)** (`chain_promote._apply_rule_c`) | **unchanged** — rule (d) feeds it a larger `F`; `Lnrat_B10`'s dd return flows into `Li2omx2_B10` via the same rule-(c) return-widen already applied to `ddilog_B10`. |
| **rule (a)** (`_expand_value_closure`) | eligible-frame set grows to include rule-(d) frames; decl-widen logic unchanged, applied inside `Lnrat_B10`/`ddilog_B10` too. |
| **NEW rule (d)** (`chain_promote`) | frame-discovery fixed point: walk promoted-body calls, test `clonable_leaf`, add clones to `F`, seed bodies, record reroutes. Reuses `CallGraph` + `region_scan`. |
| **Gap-A bridge** (`regional.py`) | **EXTENDED (L1′)** — shallow-wrapper recognizer + synthesized-overload emitter (§2.2). This is where Class-1 support is *produced*. Lives in the agents tree as synthesis; no vendored primitive header. |
| **π-family catalog** (`constant_derive.py`) | **optional** — both the double primary and the enriched dd source supply `_pi2o6/_ipio2` at dd; the dd source's `_pi()` is bit-exact `DoubleDouble_pi()`. Catalog is an *optional* bit-exactness refinement, not a requirement (§2.6 boundary 2). |
| **`Constants<DoubleDouble>`** | **NOT specialised by the pipeline, NOT vendored as a primitive** — instantiates from **source**: coefficient tables + dd π from the enriched `kokkosMaths_dd.h` (43 coeffs, §2.3, P5); the rest from the double primary at `T=DoubleDouble`. |
| **enriched dd source** (`runs/qcdloop_headers_full/kokkosMaths_dd.h`) | **NEW source input** — consumed for its 43-coeff `_C`, 25-term `_B`, dd `_pi()`, dd tolerances. Its hand-written Class-1 wrapper *code* is NOT consumed (pipeline synthesizes its own, §2.2 v3 subtlety). |
| **shim normaliser** (`shim_normalise.py`) | **used more** (more clone bodies → more shims). No logic change. |
| **fanout manifest** (`fanout.py`) | **grows** — `Lnrat_B10` becomes a new `VariantSpec` with a `return_widen` (TOutput→DoubleDoubleComplex). First time `Lnrat` appears in a manifest. No schema change. |
| **clonable-leaf predicate** | **new predicate**, evaluated against the §2.2 synthesis manifest + the §2.3 source-instantiation check (double primary at dd + enriched dd source). The false-positive guard. |
| **`chain_closure_escapes`** (`result.py`) | **still fires** — for leaves failing `clonable_leaf` (body not a synthesizable shape, or demands inward param widening). Now a *smaller* set. |
| **support surface** (`third_party/include`) | **NO new primitive header.** v1's `dd_ql_support.hpp` stays deleted from the plan. Class-1 is synthesized into the per-region shim; Class-2 is source-resident in the vendored snapshot. |
| **kernel-scope + positive-lift gates** | **unchanged.** B10 now reaches the positive-lift gate for the first time. |

### 4.3 What tests break / what's new

* **Break (assertions invert):** any `test_chain_promote` case asserting `Lnrat` is a
  `chain_closure_escapes` frontier. Under rule (d), B10 emits `Lnrat_B10`.
* **New:**
  * `clonable_leaf` predicate unit tests (clonable sink `Lnrat`; transitively-clonable
    `Li2omx2`; **non**-clonable leaf whose body is not a synthesizable shape → refuse);
  * Class-1 **shallow-wrapper recognizer + emitter** unit tests (kAbs/kLog redirect;
    Real/Imag accessor; Sign scalar-expr; a non-delegating body → not Class-1);
  * a **source-instantiation** test for Class-2 (`Constants<DoubleDouble>::_num_C()==43`,
    `_C(i)` bit-exact from the enriched dd source — the §6 P5 probe made permanent);
  * rule-(d) frame-discovery + termination test (B10 frontier = `{Lnrat_B10}`, depth 1);
  * a synthesized-shim compile test (the §6 P2 `probe_clone_synth.cpp` made permanent);
  * e2e: B10 emits dd-returning `Lnrat_B10` + `ddilog_B10`/`Li2omx2_B10`, and the
    `dilog4−dilog5` cancellation at `B1m.h:240` executes at dd against a **43-coeff** series.
* **Stay green:** all Layer 0–5 mechanical tests; rules (a)/(b)/(c); scorer; non-chain path.

### 4.4 STOP-condition impact

* **STOP #A (measurement falsification)** — unchanged in meaning, now *reachable* for B10,
  and now wired to the **§2.4 restored prediction**: lift below +8 falsifies the
  **value-flow model** (not truncation — v3 removed truncation as a cause).
* **STOP #B (accept↔reject flip)** — unchanged; B13/B14 stay byte-identical unless rule (d)
  legitimately changes their `F`.
* **STOP #E (source doesn't provide what the design claims)** — **discharged** by §6 P5
  (`probe_constants_dd43.cpp`): the enriched source provides `_num_C()=43` and bit-exact dd
  `_C`. This was the v3-specific pre-implementation risk; it is closed.
* **STOP #K (emitted transform breaks build/runtime)** — **re-armed and central.** The
  `clonable_leaf` predicate + the synthesized-shim compile test are the guards. If a clone
  is emitted whose body names a symbol neither synthesizable (Class-1) nor source-resident
  (Class-2), the build fails → STOP #K. The predicate must refuse *before* emission. The §6
  probes are the pre-implementation discharge of STOP #K for `Lnrat`.
* **~~STOP #N (support-surface drift)~~ — DELETED (v2).** There is no vendored *primitive*
  header to drift. (The enriched dd source in the snapshot is a verbatim upstream mirror,
  refreshed by the README's script, not a pipeline-maintained artifact — no drift surface.)
* **STOP #O (Class-2 capability gap) — RETAINED as a safety net, DOES NOT FIRE for Group
  A.** Under v3 the library-under-test **publishes** its dd coefficient table, so a rule-(d)
  leaf's coefficient needs are met from source. STOP #O is only reachable if a *future*
  library-under-test omits a dd-precision table the pipeline needs — which would then
  trigger the (now-unfunded) Option-A DCT-synthesis path (§2.3). **It does not fire for
  B10/B12/B13** (B10 needs no `_C`; B12/B13 read the 43-coeff source table). See §5.

---

## 5. STOP #O — retained safety net, not reachable for Group A (v3)

v2 introduced STOP #O as the Class-2 capability-gap STOP. v3 keeps the *definition* but
records that it **does not fire** under the current source:

> **STOP #O (Class-2 capability gap).** If a rule-(d) leaf's body names a Class-2
> coefficient table that **(a)** neither the double primary instantiates at dd *nor* the
> vendored snapshot publishes as a dd source table, **and (b)** the leaf's dd accuracy is
> consequently bounded by data the pipeline cannot synthesize, then **STOP and decide:**
> fund **Option A** (build the offline DCT coefficient generator — sample the special
> function at N dd Chebyshev nodes → DCT → emit `Constants<DoubleDouble>::_C`, with a drift gate
> vs the oracle *for validation only*) or fall back to **Option C** (declare the table a
> library-author pre-condition). **Do not vendor a primitive support header.**

**Why it does not fire for the current qcdloop-under-test:** after `e3d2e45` the snapshot
publishes the 43-coeff dd `_C` table (P5). B10 needs no `_C` at all (§1.3); B12/B13's ddilog
reads the source 43-coeff table. Condition (a) is false — the table *is* published — so
STOP #O is unreachable for Group A. It remains armed for any future library that does not
publish a needed precision-target table; that scenario is not present today.

---

## 6. Falsification tests (built + run)

Full evidence: `runs/qcdloop/tier_b_stage2_leaf_promotion/probe_evidence/`. Single-TU,
built against **real** headers + **real** Kokkos, gcc 13.3.0, `-std=c++20` (ceiling probe
`-std=c++17`, no Kokkos). No changes to `agents/`/`tests/`. **Five probes** (P1–P4 from
v1/v2, P5 new in v3):

**(P1) `probe_clone.cpp` (v1) — clone-vs-forwarding + surface-gap enumeration.**
Confirms rename discipline and that the vendored-only surface fails.

| build | surface | compile | runtime on `(1.5, 2.5)` (the Subtask-5 segfault inputs) |
|---|---|---|---|
| A | vendored-only (= pipeline today) | **FAIL, 5 errors** | — |
| B | A + v1 **hand** overlay | OK | runs, exit 0, no segfault |

**(P2) `probe_clone_synth.cpp` (v2) — the overlay is what the pipeline would SYNTHESIZE.**
Class-1 mechanical wrappers only (kAbs/kLog redirects, Real/Imag accessors, Sign
scalar-expr), no hand-written `Constants`, and NOT the fork's wrappers. Re-run at HEAD
`e3d2e45`:

| build | surface | compile | runtime |
|---|---|---|---|
| A_synth | vendored-only | **FAIL, 5 errors** (`abs`,`log`,`Sign`,`Constants` enable_if) | — |
| B_synth | A + **Class-1 synthesized overlay** (no Constants hand-write) | **OK** | `Lnrat_B10(synth) dd re.hi = −0.51082562376599072  double re = −0.51082562376599072  \|diff\| = 0.000e+00` |

The exact surface the extended Gap-A machinery would emit + the source coefficient primary
is sufficient to compile and run the clone. **No vendored qcdloop-specific primitive header.**

**(P3) `probe_constants_dd.cpp` (v2) — the double-primary source-instantiation proof.**
Instantiates `ql::Constants<DoubleDouble>` from the *double* primary (`kokkosMaths.h`):

```
num_C=19  sum_C.hi=0.8224670334241132  sum_C.lo=-4.971e-17
```

Proves the double primary instantiates at dd (19-coeff series) — the v2 Option-B baseline.
Superseded as the *coefficient source* by P5, but retained: it pins that the build-A
`enable_if` error traces to `ql::kLog`→`Kokkos::log` (a Class-1 gap), **not** the table.

**(P4) `probe_optionB_ceiling.cpp` (v2) — Option-B lift ceiling. SUPERSEDED by v3.**
Sums the 19 coeffs at double vs dd; established the v2 truncation floor:

```
max |dd19 − double| over battery      = 1.110e-16   (roundoff dd buys back)
dd recurrence residual |lo/hi| @Y=0.55 = 1.037e-18   (~18 extra digits carried)
19-term truncation floor ~ |C[18]|     = 1.000e-16   (dd CANNOT reduce — needs 43 coeffs)
```

Under v3 the 43-coeff table drives the truncation floor to ~1e-35 (P5's `C[42]`), below dd's
~1e-32 arithmetic floor, so the P4 "band" no longer governs. Retained for the trail.

**(P5) `probe_constants_dd43.cpp` (v3, NEW) — enriched-source 43-coeff dd table.**
Instantiates `ql::Constants<DoubleDouble>` from the enriched dd source
(`runs/qcdloop_headers_full/kokkosMaths_dd.h`):

```
num_C = 43  (expect 43)
C[0]  hi=0.42996693560813698   lo=-7.726e-18  bit-exact=1
C[18] hi=-1.4226020855112447e-16 lo=4.699e-33 bit-exact=1
C[42] hi=-1.11772e-35          lo=4.466e-52   bit-exact=1
_pi() hi=3.1415926535897931    lo=1.225e-16   bit-exact DoubleDouble_pi=1
sum_C(43) hi=0.8224670334241132  lo=1.520e-17
P5 PASS: enriched source provides 43-coeff dd table
```

`_num_C()=43`; `_C(0)/_C(18)/_C(42)` bit-exact vs the source literals; `_pi()`=`DoubleDouble_pi()`
bit-exact. **Discharges STOP #E** — the source provides exactly the 43-coeff dd table the v3
design claims, with no synthesis and no vendored primitive header.

**What the probes establish (before committing weeks):**

1. **Rename discipline is sound (STOP-#K refutation)** — P1/P2, both overlays run to
   completion on the segfault inputs.
2. **The Class-1 support surface is pipeline-synthesizable, not vendor-only** — P2 clears
   the entire Class-1 gap with mechanical overloads.
3. **Class-2 is source-resident at full 43-coeff precision (v3)** — P5 proves the enriched
   source provides `_num_C()=43` and bit-exact dd `_C`; P3 shows even the double primary
   instantiates at dd (19 coeffs), so there is *no compile blocker* and *no truncation
   concession* to bound.
4. **The probes do NOT prove a lift** — `Lnrat`'s TScale branch has no cancellation, so
   dd==double here (`|diff|=0`). The lift is B10's `Li2omx2`/`dilog4−dilog5` story, measured
   only at a full e2e run — STOP #A's job, predicted (now **+18.43**, §2.4) not promised.

**What would still falsify the design at e2e (not cheaply pre-testable):** B10 emits all
clones dd and the cancellation executes at dd, but measures lift < +8 → an intervening
double narrowing or a mis-scoped closure (STOP #A / value-flow model wrong). Under v3 this
is **no longer confoundable with a truncation shortfall** (43 coeffs removed that), so a
sub-+8 measurement points squarely at the value-flow plumbing. The probes cannot pre-empt
this — it needs the full 5000-sample kinematic battery.

---

## 7. What this design does NOT solve

1. **Inward parameter widening on a shared original** — still refused (closure design §8.2).
   Rule (d) clones (own params), so it never *needs* to widen a shared signature. Not in
   Group A.
2. **B12's floor location** — B12's dominant chain does not cover its `coeff0.imag` hotspot
   (Subtask 3). Rule (d) lets B12's `Lnrat` clone but does not move the floor; orthogonal.
3. **Group B (B15/B16/BIN0–4)** — dd-insufficient (Item 7); rule (d) would clone their
   leaves (`kfn`/`ltspence`/`cspence`) and make them *measurable*, not *sufficient*. Out of
   scope; the circuit breaker + Group-B dd-insufficiency keep them from being accepted.
4. **Class-1 recognizer beyond straight-line delegation** — a wrapper whose body is a
   multi-statement computation (not a single delegation/accessor/scalar-expr) is **not**
   Class-1 and is refused (clause 2). Widening the recognizer to such bodies is future work;
   Group A's wrappers are all straight-line, so it is not needed now.
5. **Coefficient tables for a library that does NOT publish a dd table** — Option A (offline
   DCT synthesis) is specified (§2.3, §5) but **not funded**, because the current
   qcdloop-under-test *does* publish one. A future library-under-test lacking a needed dd
   table triggers STOP #O and the Option-A decision.

---

## 8. Implementation dispatch shape (proposal only — do NOT dispatch)

Rule (d) presupposes the closure design's Stages 1–2 (rules a/b/c), which are landed
(Subtasks 1a/1b/2a/2b). v3's dispatch (v2's L4 dropped from the Group-A plan):

* **Subtask L1′ — extend the Gap-A machinery to SYNTHESIZE Class-1 wrappers.** Add the
  shallow-wrapper recognizer (structural: body = single delegation to a `_MATH_FN_NAMES` op
  / member accessor / scalar-expr over the param) + the synthesized-overload emitter (§2.2),
  producing the dd overloads into the per-region shim. **Deliverable:** the extension +
  unit tests + the P2 probe made a permanent compile test. **No vendored primitive header.**
  Restores the demo's "pipeline synthesizes what it needs" claim; land it first (independent
  of rule (d)). *(~4–6 days.)*

* **Subtask L2 — rule (d) frame-discovery + `clonable_leaf` predicate.** Frame-discovery
  fixed point in `chain_promote` (walk promoted-body calls, test predicate against the L1′
  synthesis manifest + the source-instantiation check — double primary at dd **and** the
  enriched dd source for coefficient tables — add clones to `F`, seed bodies, record
  reroutes). Wire the circuit breaker (`chain_closure_oversized`) and the narrowed
  `chain_closure_escapes`. **Gate:** predicate false-positive = STOP #K; must refuse before
  emission. *(~4–6 days.)*

* **Subtask L3 — emission plumbing.** `Lnrat_B10` `VariantSpec` (reuses `return_widen` from
  Subtask 2a), test rewrites (inverted `Lnrat` assertions), the P5 source-instantiation test
  made permanent. *(~2–3 days.)*

* **Subtask L-measure — B10/B12/B13 e2e + §2.4 triage.** e2e re-run (seed 12345, 5000
  samples, kernel-scope + positive-lift gate). **Success = B10 reaches the positive-lift
  gate with `Li2omx2_B10`+`ddilog_B10`+`Lnrat_B10` all dd and measured
  `kernel_measured_lift ≈ +18.43`** (closure §7 bar ≥ +8; §2.4 predicts the full +18.43
  with 43 coeffs). **If lift < +8**, triage per §2.4: value-flow model wrong (STOP #A —
  intervening narrowing / mis-scoped closure), **not** truncation (v3 removed that cause).
  *(~2–3 days.)*

```
                                        subtotal  ≈ 12–18 days (~2.5–3.5 wks) → tighten to
                                        ~10–15 days: L4 dropped, Class-2 is zero-cost source read
```

**v2's L4 (Option-A DCT coefficient generator, +2–3 wks) is REMOVED from the Group-A plan.**
It would be required only if a *future* library-under-test failed to publish a needed dd
table (STOP #O). It is specified (§2.3, §5) but not funded. This dispatch is **on top of**
the closure design's Stages 1–2 (rules a/b/c, landed).

**If L1′ proves a Group-A wrapper is not synthesizable** (a body shape the recognizer cannot
handle soundly), **STOP at L1′** and hand the scope call back — that would mean the Class-1
premise is narrower than §2.2/§6 established. **If L-measure fires STOP #A**, the value-flow
model is wrong — debug the chain plumbing, do not reach for coefficient synthesis (43 coeffs
are already in source).

---

## Appendix — invariants carried from the closure design

* Variants are **per-integral clones**; the shared original (`Lnrat`, `ddilog`, `Li2omx2`)
  is never edited. Rule (d) obeys this — it clones, never mutates the primary.
* Refusals (`clonable_leaf` fail, oversized) computed **before any tree mutation**.
* Gate logic **unchanged**; rule (d) only enlarges the (non-inert) candidate the gates
  measure. B10 reaching the lift gate is the point.
* Conservative-parser contract: false-negative (refuse a clonable leaf) safe; false-positive
  (clone an un-instantiable leaf) is STOP #K — guarded by the predicate + the L1′ synthesis
  compile test + the §6 probes.
* **Architectural invariant (v2, refined in v3):** the pipeline **synthesizes**
  qcdloop-specific dd *code* (Class-1 wrappers) and **reads** qcdloop's own dd *data*
  (Class-2 coefficient tables) from the **vendored snapshot of the library under test**
  (`runs/qcdloop_headers_full/`, including `kokkosMaths_dd.h`). It **vendors nothing
  qcdloop-specific into the app-independent primitive layer** — `third_party/include/`
  (`Kokkos::Experimental`) stays app-independent; the namespace shim bridges the fork's `ql::ddfun`
  authorship onto it. The oracle `qcdloop@ddfun_enabled` is consulted for **validation
  drift only**, never for generation — with the single, documented exception of the one
  header the snapshot now vendors verbatim as source input (`kokkosMaths_dd.h`, commit
  `2229ec4`), which is qcdloop-authored library data, not oracle-derived generation.
```
