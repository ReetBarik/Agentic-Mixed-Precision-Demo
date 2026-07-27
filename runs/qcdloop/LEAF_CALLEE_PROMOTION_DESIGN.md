# Leaf-Callee Promotion — Design Notes (Group A precision-lift unblock)

Status: **v2 — design + falsification probes done, STOP before implementation** (2026-07-27).
v1 was 2026-07-26. Scope unchanged: extend closure-scoped dd promotion
(`CLOSURE_SCOPED_CHAINS_DESIGN.md`) so leaf callees like `ql::Lnrat` / `ql::ddilog`
become **cloned, promoted frames** instead of a `chain_closure_escapes` refusal frontier.
Discipline: **design only** — no changes to `agents/`, `tests/`. The §7 probes were built
and run (Subtask-5-style single-TU); nothing else moved.

> ## What changed v1 → v2 (read this first)
>
> **v1's central resolution (§3.4) was to *vendor* a hand-ported `dd_ql_support.hpp` from
> `qcdloop@ddfun_enabled:src/qcdloop/kokkosMaths_dd.h` — importing qcdloop-specific dd
> knowledge (the 43-term Chebyshev series, the `ql::` dd helper overloads) the pipeline
> should synthesize.** Reet's architectural line rejects this: `third_party/include/` is
> vendored, **app-independent** dd/ff primitives that know nothing about qcdloop and never
> will; anything qcdloop-specific is the **pipeline's job to synthesize** from qcdloop's own
> source + the vendored primitives. `qcdloop@ddfun_enabled` is a **validation oracle, not a
> generation cheat-sheet**.
>
> v2 confronts this by **splitting the category-(d) support surface into two classes**
> (§2), and — critically — by discovering through two new probes that **the split is far
> more favourable than v1 assumed**:
>
> * **Class 1 (mechanical wrappers — `kAbs/kLog/Real/Imag/Sign/iszero`)** is
>   *pipeline-synthesizable today* via an extension of the existing **Gap-A qualified-math
>   bridge** machinery (`agents/integrator_base/regional.py`). Each is a one-line delegation
>   whose dd overload is a namespace redirect (`Kokkos::abs`→`quad::ddfun::abs`) or a member
>   accessor (`.real()`) — exactly the bridge shape the shim generator already emits. **No
>   vendored header.**
> * **Class 2 (precision-target coefficient tables — the 43-term `Constants<ddouble>::_C`)**
>   is the one genuine capability question. v2 **chooses Option B** (§2.3): accept the
>   library's own 19-coeff series at dd. The load-bearing discovery: **`Constants<ddouble>`
>   already instantiates directly from the unmodified source primary** (probe
>   `probe_constants_dd.cpp`: `_num_C()=19`, `_C(i)` promotes to dd, `_pi/_half/_ipio2`
>   resolve at `T=ddouble`) — so Option B needs **zero synthesis and zero vendoring**. The
>   43-coeff table is *not* needed to compile or run; it is only needed to reach the full
>   theoretical accuracy, and §2.4 **bounds exactly how much lift the 19-coeff series
>   forfeits** (measured by `probe_optionB_ceiling.cpp`).
>
> **Net effect on the plan:** v1's Subtask **L1 ("vendor + port dd_ql_support.hpp") is
> deleted.** It is replaced by L1′ (extend Gap-A to synthesize Class-1 wrappers, in the
> agents tree) and a *measurement* subtask for Class 2 (Option B). The demo's claim is
> restored to "the pipeline synthesizes what it needs from source + primitives," with an
> honest, quantified precision demarcation for coefficient tables.
>
> Everything else in v1 (rules a/b/c interaction, the rename/STOP-#K refutation,
> per-integral clones, termination proof) **stands** and is carried forward.

Reads as the successor to two docs, both of which stand:
* `CLOSURE_SCOPED_CHAINS_DESIGN.md` — rules (a)/(b)/(c), the §2.4 refusal frontier, the
  §3 designed-exit gate, the §7 STOP discipline. **This design changes exactly one thing
  in it:** it moves `ql::Lnrat`/`ql::ddilog` from the §2.4 "callee-not-in-`F`" refusal set
  into `F` as clonable frames. Everything else in that doc is unchanged.
* `tier_b_stage2_subtask5/TIER_B_STAGE2_SUBTASK_5_2026-07-26.md` (STOP #K) — proved the
  *forwarding-overload* path is unsound (self-recursion). This design takes STOP #K's own
  recommended option 2 ("make `Lnrat`/`ddilog` chain-frames") and works out whether it is
  actually buildable. **The v2 answer is: yes, and the support surface it needs is
  pipeline-synthesizable (Class 1) plus a source-resident coefficient table used as-is
  (Class 2 / Option B) — no vendored qcdloop-specific header.**

> **Load-bearing correction up front (read before §1).** The closure design's §2.4
> lists `ql::Lnrat` as a hard refusal because "its signature we will not touch." That is
> the wrong lens for a leaf callee. We are **not** touching `ql::Lnrat`'s signature — we
> **clone** it to `Lnrat_B10` (a new symbol, reachable only from the chain's rerouted
> call sites) and promote the clone's body, exactly as the pipeline already clones
> `ddilog`/`Li2omx2` whose bodies sit *on* the chain. STOP #K's recursion pit was a
> property of a same-name *overload*, not of a *renamed clone*. The §7 probes confirm the
> renamed clone builds and runs on the exact inputs that segfaulted the overload. The real
> blocker is not recursion and not signatures — it is the dd **support surface** the
> clone's body names. **v2's finding: that surface is not an un-vendored monolith (as v1
> claimed) but two separable classes, one synthesizable and one already in source.** §2 is
> the whole ballgame.

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
promoted to dd needs `ql::Lnrat<ddcomplex,…>(dd,dd)` to exist, and it does not.

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
is the right shape, and v2's Class-1/Class-2 split (§2) makes it decidable **without
vendoring**. Refined predicate, all clauses required:

```
clonable_leaf(g) :=
   (1) g is a function template with a body available in the analysed headers
       (not an extern/vendored-binary symbol);              # can be cloned at all
 ∧ (2) g's body, with reads promoted to dd, calls ONLY:
         - other clonable_leaf callees (recurse), OR
         - symbols the dd TERMINATION BOUNDARY resolves at dd (§2.6):
             (i)   vendored quad::ddfun ops (abs/log/…, ddcomplex ops),
             (ii)  a Class-1 SYNTHESIZED wrapper (§2.2) — pipeline-emittable
                   mechanically from that wrapper's own primary + (i),
             (iii) a Class-2 data accessor that the SOURCE primary already
                   instantiates at dd (§2.3, Constants<ddouble>::…),
       i.e. rule (d)'s transitive closure over g terminates at the boundary;
                                                             # body instantiable at dd
 ∧ (3) g is NOT self-recursive under a SAME-NAME overload set that a rename
       cannot separate (STOP #K guard — §4);               # rename discipline safe
 ∧ (4) cloning g does not require widening a shared g PARAMETER that a
       non-chain caller also binds — the clone gets its own params, so this
       is automatically satisfied for a pure clone, but a leaf whose promotion
       demands INWARD dd on a shared original's signature is refused (§8.2 of
       the closure design still holds).
```

Clause (2) is now decidable **against a synthesis manifest, not a vendored header**. A leaf
is clonable iff every dd symbol its promoted body names is (i) vendored, (ii) a Class-1
wrapper the extended Gap-A machinery **can synthesize** (predicate: primary body is a
straight-line delegation to a vendored/ADL-reachable op or a member accessor — §2.2), or
(iii) a Class-2 accessor the **source primary already instantiates at dd** (§2.3). Anything
else → the leaf is **not** clonable → `chain_closure_escapes` (honest terminal, not a
doomed emission). This keeps the conservative-parser contract: false-negative (refuse a
clonable leaf) is safe; false-positive (clone an un-instantiable leaf) is the STOP #K
hard-fail we must never ship.

### 1.3 Which frames become eligible

For the Group-A chains, rule (d) makes exactly these leaves eligible:

| leaf | called from | primary | promotable body? |
|------|-------------|---------|------------------|
| `ql::Lnrat` (TScale overload, `:153`) | `Li2omx2:701,706` | straight-line `kLog/kAbs/Sign/_ipio2` | yes — all Class-1 (§2.2) + source Constants (§2.3); **§7 v2 probe: builds+runs** |
| `ql::ddilog` (`:163`) | `Li2omx2:702,708` (leaf on B12/B13) | Chebyshev series over `_C` | already IN `F` for B10 (chain lines inside it); **needed for B12/B13**; uses Class-2 `_C` (§2.3, Option B) |
| `ql::kfn`, `ql::ltspence`, `ql::cspence` | Group-B chains | series / branch | out of scope (Group-B dd-insufficient, §6) |

So for **B10** specifically, rule (d) is needed for **`Lnrat` only** — `ddilog`/`Li2omx2`
are already cloned frames. This narrows the headline case to a single leaf whose entire
support surface is Class-1 wrappers plus the source's own `Constants<ddouble>::_ipio2`
(no `_C` — Lnrat has no series). **The B10 unblock therefore needs no Class-2 coefficient
work at all.** Class 2 becomes relevant only when `ddilog` is itself a rule-(d) leaf
(B12/B13), where Option B applies (§2.3–2.4).

---

## 2. Support-surface scoping — the two classes (the crux)

This is the section the whole design turns on, and where the §7 probes did their work. v1
treated the category-(d) surface as one un-vendored monolith. v2 splits it.

### 2.1 The B10 support-surface bill of materials, re-classified

Every dd symbol the B10 closure's promoted bodies name, sourced from `Lnrat` body
`kokkosUtils.h:141-155`, `ddilog` `:163-232`, `Li2omx2` `:692-712`; helper defs
`src/kokkosMaths.h:250-372`; vendored `third_party/include/*`. **The `qcdloop@ddfun_enabled`
oracle is NOT a source here** — the classification is derived from qcdloop-under-test +
vendored primitives only.

| symbol (at dd) | used by | vendored? | source primary at dd? | **v2 class** |
|---|---|---|---|---|
| `ddadd/sub/mul/div`, `ddouble`/`ddcomplex` ops | all | ✅ `dd_math`/`dd_complex` | — | boundary (vendored) |
| `abs/log/sqrt/exp/pow` on dd | ddilog, Lnrat | ✅ `quad::ddfun::*` | — | boundary (vendored) |
| `ql::kAbs(ddouble/ddcomplex)` | Lnrat, Li2omx2 | ❌ | primary `T kAbs(T){Kokkos::abs(x)}` — redirect | **Class 1** (§2.2) |
| `ql::kLog(ddouble/ddcomplex)` | Lnrat, ddilog, Li2omx2 | ❌ | primary `T kLog(T){Kokkos::log(x)}` — redirect | **Class 1** |
| `ql::kSqrt/kConj(dd)` | (Group A: not on chain) | ❌ | primary `T kSqrt(T){Kokkos::sqrt(x)}` — redirect | **Class 1** |
| `ql::Real/Imag(ddcomplex)` | ddilog, Lnrat | ❌ | primary is `.real()/.imag()` accessor | **Class 1** |
| `ql::Sign(ddouble)` | Lnrat, ddilog | ❌ | primary `(0<x)-(x<0)`, T-generic ±1/0 | **Class 1** |
| `ql::iszero<…>(ddouble)` | ddilog (`:116`) | ❌ | template; body = `kAbs(x)<_qlonshellcutoff` | **Class 1** (transitive: `kAbs` + source cutoff) |
| `ql::kPow<…>(ddouble,int)` | ddilog (`:117…`) | ❌ | template `TOutput(1.0); temp*=base` — clean at dd | **source (already instantiates)** |
| `_pi2o6/_ipio2/_half/_pi/_zero/_one` at dd | ddilog, Lnrat, Li2omx2 | partial (`dd_pi()`) | source `Constants<T>` primary at dd (M_PI→ddouble) | **Class 2 / source** (§2.3) |
| `Constants<ddouble>::_C(i)`, `_num_C()` | ddilog | ❌ | source primary: **19** coeffs, promotes to dd | **Class 2** (§2.3, Option B) |

**Only two rows are not vendored-boundary or already-in-source: the Class-1 wrappers (which
the pipeline synthesizes, §2.2) and the Class-2 coefficient table `_C` (Option B, §2.3).**
This is the whole re-classification. v1's "single un-vendored file" framing conflated these
and concluded (wrongly) that a vendored header was required.

### 2.2 Class 1 — pipeline-synthesizable via extended Gap-A machinery

**Definition.** A Class-1 wrapper is a shallow app-specific function whose primary body is a
**straight-line delegation** to (a) a vendored/ADL-reachable op, or (b) a member accessor,
or (c) a type-generic scalar expression — such that its dd overload is a **mechanical
transform** of that primary body.

**For each Group-A wrapper — the existing primary, the mechanical transform, and Gap-A reach:**

| wrapper | primary (src/kokkosMaths.h) | mechanical dd transform | current Gap-A reach? |
|---|---|---|---|
| `ql::kAbs` | `:271` `T kAbs(T x){ return Kokkos::abs(x); }` (+`:279/285` double/cplx overloads) | emit `ddouble kAbs(ddouble){ return quad::ddfun::abs(x); }` + `ddouble kAbs(ddcomplex){ return quad::ddfun::abs(z); }` — redirect `Kokkos::abs`→`quad::ddfun::abs` | **needs extension** (see below) |
| `ql::kLog` | `:289` `T kLog(T x){ return Kokkos::log(x); }` | `ddouble kLog(ddouble){ quad::ddfun::log }`, `ddcomplex kLog(ddcomplex){ quad::ddfun::log }` | **needs extension** |
| `ql::kSqrt`/`kConj` | `:295/301` `Kokkos::sqrt/conj` | analogous redirect to `quad::ddfun::sqrt/conj` | **needs extension** |
| `ql::Real`/`Imag` | `:320-326` `.real()/.imag()` accessors on `complex<double>` | emit `ddouble Real(ddcomplex z){ return z.real(); }` etc. | **needs extension** (accessor form) |
| `ql::Sign` | `:328` `int Sign(double x){ return (0<x)-(x<0); }` | re-emit with dd operands: `int Sign(ddouble x){ return (ddouble(0.0)<x)-(x<ddouble(0.0)); }` | **needs extension** (scalar-expr form) |
| `ql::iszero` | `:307` template, body `kAbs(x)<_qlonshellcutoff` | already a template — instantiates at dd once `kAbs(dd)` exists + `_qlonshellcutoff` (source literal `T(1e-10)`) | **transitive** (falls out once the above land) |

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
     inner call to `quad::ddfun::fn` (the transform the bridge already knows);
   * `arg.real()` / `arg.imag()` / other member accessor → **accessor passthrough**;
   * a scalar comparison/arithmetic expression over the parameter with no non-boundary call
     → **re-emit verbatim at dd** (the parameter's type widens; the operators are vendored).
   Anything not matching these shapes → **not Class-1** → the leaf fails clause (2) → refuse.
   This recognizer is deterministic and conservative (unrecognized body ⇒ refuse), matching
   the parser contract.

2. **A synthesized-overload emitter** that, given `(g, primary_body, dd_target)`, produces
   the dd overload text and injects it into namespace `ql` alongside the existing shim —
   reusing the same injection/using-declaration remedy the Gap-A lint already sanctions
   (`_shim_bridges_qualifier`). The emitted overloads are exactly the §7 v2 probe's
   `WITH_SYNTH` block, produced mechanically instead of hand-written.

3. **`_MATH_FN_NAMES` stays a `<cmath>` vocabulary.** The extension does **not** bake in
   `kAbs/kLog/…` names (that would violate "no app-specific identifiers" —
   `[[feedback_no_placeholder_patterns]]`). It recognizes app wrappers **structurally**
   (body is a straight-line delegation to a `_MATH_FN_NAMES` op or an accessor), so it
   works for any app's shallow math wrappers, not qcdloop's specifically.

**Empirical proof (§7 v2).** Probe `probe_clone_synth.cpp` build B compiles and runs
`Lnrat_B10` with a `WITH_SYNTH` overlay that is **only** these mechanical Class-1 overloads
(no hand-written `Constants`), against the unmodified source `Constants<ddouble>` primary.
`|diff| = 0` vs the double primary. This is the exact surface the extended Gap-A machinery
would emit.

### 2.3 Class 2 — coefficient tables the primary can't derive from its body

**Definition.** Class-2 data is a value the primary template **cannot derive from its own
body** — precision-target-specific *data*, not code. The load-bearing example is
`Constants<ddouble>::_C`, the Chebyshev series for Li₂: the source primary
(`kokkosMaths.h:26`) carries **19** coefficients chosen for double precision; a dd-accurate
ddilog would want **~43** (the count the oracle uses). The pipeline **cannot invent 43
dd-appropriate coefficients from the 19-coeff primary alone** — they are the DCT of Li₂
sampled at more Chebyshev nodes, external data.

**Decisive discovery (probe `probe_constants_dd.cpp`).** The 19-coeff table is **not a
compile blocker and needs no synthesis to exist at dd.** The source primary
`template<typename T> struct Constants` instantiates *directly* at `T = ddouble`:
`_num_C()` returns 19, `_C(i)` returns `ddouble(coeffs[i])` (the double literal promoted
honestly to `make_dd(bits, 0)`), and `_pi()/_half()/_ipio2()` all resolve. Build + run:

```
num_C=19  sum_C.hi=0.8224670334241132  sum_C.lo=-4.971e-17
```

So the 19-coeff series **already runs at dd from source, zero synthesis, zero vendoring.**
The only question Class-2 raises is *accuracy*: how much of ddilog's dd headroom does the
19-coeff truncation forfeit?

**The three options the brief poses, and v2's choice.**

* **Option A — pipeline computes the 43-coeff table offline (chebfun-style DCT).** A real
  capability extension: a coefficient-generation utility that, given a primary annotated
  "this `_C` is the Chebyshev series for Li₂ on `[−1,1]`", samples Li₂ at N Chebyshev nodes
  in dd, DCTs to N coefficients, emits a `Constants<ddouble>::_C` specialisation. Run once
  per (function, precision) pair, offline, not per-integral LLM synthesis. **Cost: ~2–3
  weeks** (dd-accurate Li₂ node sampler + DCT + emitter + a bit-exactness gate vs the oracle
  *for drift detection only*). **Demonstrates the pipeline can bootstrap coefficient tables
  from qcdloop source + math** — the strongest form of the demo's claim.

* **Option B — accept the 19-coeff series at dd (v2's CHOICE for the first cut).** Compute
  ddilog at dd using the source's own 19 coefficients. Zero synthesis, zero vendoring, ships
  with L1′ + the measurement subtask. The 19-coeff truncation caps ddilog's dd accuracy;
  §2.4 bounds exactly where. **Cost: ~0 beyond the measurement run.**

* **Option C — declare precision-target tables an out-of-scope pre-condition.** State the
  pipeline synthesizes everything derivable from source + primitives, and coefficient tables
  at target precision are the **library author's** responsibility to publish alongside the
  primary (an annotated `Constants<ddouble>` as *source input*). Ships fastest, shrinks the
  claim, honest about the demarcation.

**v2 chooses Option B as the first cut, with Option A named as the principled follow-on.**
Justification:

1. **B10 — the headline case — needs *no* Class-2 work at all** (§1.3: Lnrat has no series;
   `ddilog`/`Li2omx2` are already-cloned frames, not rule-(d) leaves, for B10). So the
   coefficient-table question is **orthogonal to the B10 unblock** and must not gate it.
2. **B is measurable now and bounds the promise honestly.** Option A is weeks of work whose
   payoff is unknown until measured; Option B measures the payoff first. If the 19-coeff
   ceiling (§2.4) already clears B10's lift bar, Option A is unnecessary. If it does not,
   Option A's cost is justified by a *measured* gap, not a hoped-for one.
3. **B preserves the architectural line** — it uses only source + vendored primitives, no
   oracle knowledge. It neither vendors (v1's sin) nor over-claims.
4. **Option A is the honest upgrade path** when a coefficient table is the proven binding
   constraint. It is real synthesis-from-source (sample + DCT), not a port. v2 **specifies**
   it (§2.3 above, §9 L4-optional) so it is ready if §2.4's measurement demands it, but does
   **not** fund it speculatively.

Option C is rejected as the *default* because it shrinks the claim further than necessary —
Option A shows the claim is *achievable*, so conceding the table to the library author is
premature. C remains the correct fallback **iff** Option A proves intractable for some
function (a series with no closed-form node sampler), guarded by **STOP #O** (§5).

### 2.4 Bounding the Option-B lift ceiling (the brief's explicit demand)

The closure design's Item-7 assumed the full **+18.43-digit** dd ceiling, which *implicitly
assumes the 43-coeff series*. Option B uses 19 coeffs, so v2 must **predict a numerical
ceiling for the achievable lift, not promise the full +18.43**.

Probe `probe_optionB_ceiling.cpp` isolates the two error sources in the Clenshaw-summed
Chebyshev recurrence (`ddilog:220-227`) by summing the **identical 19 coeffs** at double vs
dd across a battery of arguments:

```
max |dd19 - double| over battery      = 1.110e-16   (roundoff dd BUYS BACK)
dd recurrence residual |lo/hi| @Y=0.55 = 1.037e-18   (dd carries ~18 extra digits of the SUM)
19-term truncation floor ~ |C[18]|     = 1.000e-16   (dd CANNOT reduce — needs 43 coeffs)
```

**Interpretation — the two components of ddilog's error, and which dd fixes:**

1. **Recurrence roundoff (the cancellation dd is *for*).** The Clenshaw loop
   `B0 = C_i + ALFA·B1 − B2` accumulates catastrophic cancellation; at double this
   contributes ~1e-16 error. dd carries ~18 extra digits through the recurrence (residual
   `|lo/hi| ≈ 1e-18`), shrinking this component to ~1e-32. **This is exactly the error
   B10's downstream `dilog4−dilog5` cancellation amplifies, and Option B removes it in
   full.** dd buys back the roundoff regardless of coefficient count.

2. **Series truncation (a property of the 19 coeffs, *not* the arithmetic).** The 19-term
   Chebyshev tail is bounded by `|C[18]| ≈ 1e-16`. This is **independent of arithmetic
   width** — dd cannot shrink it. It caps ddilog's *absolute* accuracy at ~1e-16 **at the
   point where the truncation, not the roundoff, dominates.**

**Predicted ceiling.** Option B's ddilog is accurate to **≈ max(1e-32 roundoff, 1e-16
truncation) = ~1e-16** in the regions where the 19-term tail is the floor. **But B10's lift
does not come from ddilog's absolute accuracy — it comes from removing the *cancellation
roundoff* in the `dilog4−dilog5` difference at `B1m.h:240`.** The two ddilog calls share the
same truncation error (same series, nearby arguments), so **the truncation error largely
*cancels in the difference*, while the roundoff error (uncorrelated between the two calls)
does not.** Option B removes the uncorrelated roundoff (component 1) — which is the part
that survives the difference — and leaves the correlated truncation (component 2), which
mostly cancels anyway.

**Therefore the design predicts:** Option B recovers **most** of B10's cancellation lift —
bounded **below** by "the roundoff component that survives the `dilog4−dilog5` difference"
and **above** by the full +18.43 only in the idealized no-truncation limit. A conservative
design prediction: **Option B yields a lift in the +8 to +16 digit band** (clears the
closure design's **≥ +8** acceptance bar), with the residual gap to +18.43 attributable to
the truncation error that does *not* cancel in the difference. **If the measured B10 lift
lands below +8, that falsifies the "truncation cancels in the difference" premise and is the
signal to fund Option A** (43 coeffs, which drives component 2 to ~1e-32 and recovers the
full ceiling). This is the STOP-#A measurement wired to a concrete numerical prediction —
exactly the "bound, don't promise" the brief demands.

> **Caveat the measurement must check.** The above assumes the two ddilog arguments in
> `dilog4−dilog5` fall in the *same* Chebyshev range-reduction branch (`:174-211`), so their
> truncation errors are correlated. If B10's kinematics straddle a branch boundary, the
> truncation errors decorrelate and the ceiling drops toward +8. The e2e run measures which
> regime B10 is in; the design does not assume — it predicts a band and names the falsifier.

### 2.5 Does B10's read flow through `ddcomplex` or `Kokkos::complex<ddouble>`?

Unchanged from v1 (this was correct). The chain's `TOutput` is `Kokkos::complex<double>`;
the existing pipeline promotes complex containers to `quad::ddfun::ddcomplex`
(`dispatch.py:308`, `fanout.py:243/271`, `shim_normalise.py:60-63`), and the
`ddilog`/`Li2omx2` clones already do this (B12 built + executed, Subtask 3). So B10's reads
flow through **`quad::ddfun::ddcomplex` directly**, via vendored `dd_complex.hpp` — no
`Kokkos::complex<ddouble>`, no container-axis bridging. The §7 v2 probe confirms:
`Lnrat_B10<ddcomplex,double,ddouble>` compiles and runs with `ddcomplex` as `TOutput`.

### 2.6 The termination boundary (updated)

Rule (d) recurses; it terminates because every call in a promoted body resolves to exactly
one of four **boundary** kinds, none re-entering rule (d):

1. **Vendored `quad::ddfun` math** — `abs/sqrt/log/exp/pow/…` on `ddouble`/`ddcomplex`
   (`dd_math.hpp`, `dd_complex.hpp`). Resolve at dd, no cloning. **Boundary.**
2. **Class-2 / source constants** — `_pi2o6`, `_ipio2`, `_C`, `_num_C` at dd, instantiated
   **from the source `Constants<T>` primary** (§2.3, proven by `probe_constants_dd.cpp`);
   π-family entries may additionally be routed through the Subtask-3 catalog for
   bit-exactness where the primary's `M_PI` literal is insufficiently precise (an optional
   refinement, not required to build). **Boundary — a value, not a frame.**
3. **Class-1 synthesized wrappers** — `ql::{kAbs,kLog,kSqrt,Real,Imag,Sign,iszero}` at dd,
   **emitted by the extended Gap-A machinery** (§2.2), not vendored. Once emitted they are
   ordinary overloads that bottom out in boundary 1. **Boundary.**
4. **Vendored `ddcomplex` container ops** — `+,−,*,/`, `.real()`, `.imag()`
   (`dd_complex.hpp`). **Boundary.**

Rule (d) adds a frame only for **none-of-the-above** = an app template whose body is
available and calls into these boundaries. Recursion adds a frame at most once per app
template in the finite header set.

### 2.7 Bounded, acyclic — the proof (unchanged from v1)

* The universe of clonable app templates is **finite**.
* Rule (d) is **monotone** and records each app template as a clone at most once → halts
  after ≤ (#app-templates) rounds.
* **No cycle forces unbounded growth.** The qcdloop special-function call graph on these
  chains is a **DAG**. `Lnrat`'s body (`:141-155`) calls `kLog/kAbs/Sign/Imag/Real/_ipio2`
  — all boundary kinds, **no app-template callee** → `Lnrat` is a **sink**; rule (d) adds it
  and stops. `ddilog`'s body calls `kLog/kPow/Real/Sign/iszero/_C/_pi2o6` — all boundary
  (`kPow`/`iszero` instantiate from source at dd) → also a sink. `Li2omx2` calls
  `Lnrat`/`ddilog`/`kLog`/`kAbs`/`_ipio2` → its only app-template callees are the two sinks.
  So the B10 rule-(d) frontier is depth-1 and closed:
  `{Li2omx2_B10 → Lnrat_B10 (sink), ddilog_B10 (sink)}`. **Bounded, acyclic. QED.**
* **Self-recursion is not a cycle in `F`** — `ddilog`/`Lnrat` have no self-call (verified);
  and the rename (§4) binds any hypothetical self-call to the clone name.

### 2.8 The circuit breaker (backstop, unchanged)

No size *cap* (a cap re-introduces the subset boundary), but keep a **circuit breaker**: if
rule (d) would grow `F` past a diagnostic threshold (8 frames or rule-(d) recursion depth
> 3), abort with `chain_closure_oversized`. For Group A this never fires (B10 frontier
depth-1, 3 frames). Graceful degradation, not a scope choice.

---

## 3. Rename discipline (how the clone avoids the Subtask-5 self-recursion pit)

*(Unchanged from v1 — carried forward verbatim; the §7 v2 probe re-confirms it.)*

### 3.1 Why the forwarding overload recursed, and why the clone does not

STOP #K's recursion was structural: an injected **same-name** overload
`ql::Lnrat(ddouble,ddouble)` whose body calls `ql::Lnrat(ddouble,ddouble)` — C++ selects by
*argument type*, ignoring the explicit `<…>`, so it re-selects itself forever.

A **clone** breaks every link:
* the clone is a **distinct symbol** `Lnrat_B10` — no overload set to re-enter;
* the clone's body names only `ql::kLog/kAbs/Sign/Real/Imag/Constants` + vendored ops — it
  **never names `Lnrat_B10` or `ql::Lnrat`** (verified; `Lnrat`'s body has no self-call);
* the call site `Li2omx2_B10:706` is **rerouted** to `Lnrat_B10` by the existing
  topological callee-before-caller reroute (`_reroute_in_function`).

The §7 probes are the empirical proof: `Lnrat_B10<ddcomplex,double,ddouble>(1.5,2.5)`
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
hazard. Preserves the Appendix invariant. Shared dd instantiation **rejected for v1/v2**
(re-introduces cross-integral coupling). No new collision surface beyond
`variant_naming.py`/`assert_no_collisions`.

---

## 4. Interaction with the existing tree

| component | change |
|---|---|
| **rule (c)** (`chain_promote._apply_rule_c`) | **unchanged** — rule (d) feeds it a larger `F`; `Lnrat_B10`'s dd return flows into `Li2omx2_B10` via the same rule-(c) return-widen already applied to `ddilog_B10`. |
| **rule (a)** (`_expand_value_closure`) | eligible-frame set grows to include rule-(d) frames; decl-widen logic unchanged, applied inside `Lnrat_B10`/`ddilog_B10` too. |
| **NEW rule (d)** (`chain_promote`) | frame-discovery fixed point: walk promoted-body calls, test `clonable_leaf`, add clones to `F`, seed bodies, record reroutes. Reuses `CallGraph` + `region_scan`. |
| **Gap-A bridge** (`regional.py`) | **EXTENDED (v2's L1′)** — shallow-wrapper recognizer + synthesized-overload emitter (§2.2). This is where Class-1 support is *produced*. Was v1's vendored-header work; now lives in the agents tree as synthesis. |
| **π-family catalog** (`constant_derive.py`) | **used, optionally** — the source `Constants<ddouble>` primary already supplies `_pi2o6/_ipio2` at dd; the catalog is an *optional* bit-exactness refinement, not a requirement (§2.6 boundary 2). |
| **`Constants<ddouble>`** | **NOT specialised, NOT vendored** — instantiates from the source primary at dd (§2.3, proven). 19-coeff `_C` used as-is (Option B). |
| **shim normaliser** (`shim_normalise.py`) | **used more** (more clone bodies → more shims). No logic change. |
| **fanout manifest** (`fanout.py`) | **grows** — `Lnrat_B10` becomes a new `VariantSpec` with a `return_widen` (TOutput→ddcomplex). First time `Lnrat` appears in a manifest. No schema change. |
| **clonable-leaf predicate** | **new predicate**, evaluated against the §2.2 synthesis manifest + §2.3 source-instantiation check. The false-positive guard. |
| **`chain_closure_escapes`** (`result.py`) | **still fires** — for leaves failing `clonable_leaf` (body not a synthesizable shape, or demands inward param widening). Now a *smaller* set. |
| **support surface** (`third_party/include`) | **NO new header.** v1's `dd_ql_support.hpp` is deleted from the plan. Class-1 is synthesized into the per-region shim; Class-2 is source-resident. |
| **kernel-scope + positive-lift gates** | **unchanged.** B10 now reaches the positive-lift gate for the first time. |

### What tests break / what's new

* **Break (assertions invert):** any `test_chain_promote` case asserting `Lnrat` is a
  `chain_closure_escapes` frontier. Under rule (d), B10 emits `Lnrat_B10`.
* **New:**
  * `clonable_leaf` predicate unit tests (clonable sink `Lnrat`; transitively-clonable
    `Li2omx2`; **non**-clonable leaf whose body is not a synthesizable shape → refuse);
  * Class-1 **shallow-wrapper recognizer + emitter** unit tests (kAbs/kLog redirect;
    Real/Imag accessor; Sign scalar-expr; a non-delegating body → not Class-1);
  * rule-(d) frame-discovery + termination test (B10 frontier = `{Lnrat_B10}`, depth 1);
  * a synthesized-shim compile test (the §7 v2 `probe_clone_synth.cpp` made permanent);
  * e2e: B10 emits dd-returning `Lnrat_B10` + `ddilog_B10`/`Li2omx2_B10`, and the
    `dilog4−dilog5` cancellation at `B1m.h:240` executes at dd.
* **Stay green:** all Layer 0–5 mechanical tests; rules (a)/(b)/(c); scorer; non-chain path.

### STOP-condition impact

* **STOP #A (measurement falsification)** — unchanged in meaning, now *reachable* for B10,
  and now wired to the **§2.4 numerical prediction**: lift below +8 falsifies the
  "truncation cancels in the difference" premise → fund Option A.
* **STOP #B (accept↔reject flip)** — unchanged; B13/B14 stay byte-identical unless rule (d)
  legitimately changes their `F`.
* **STOP #K (emitted transform breaks build/runtime)** — **re-armed and central.** The
  `clonable_leaf` predicate + the synthesized-shim compile test are the guards. If a clone
  is emitted whose body names a symbol neither synthesizable (Class-1) nor source-resident
  (Class-2), the build fails → STOP #K. The predicate must refuse *before* emission. The §7
  v2 probe is the pre-implementation discharge of STOP #K for `Lnrat`.
* **~~STOP #N (support-surface drift)~~ — DELETED.** v1's STOP #N guarded drift between a
  *vendored* `dd_ql_support.hpp` and the oracle. There is no vendored header in v2, so there
  is nothing to drift. (If Option A is later funded, its emitter gets its own drift check —
  see STOP #O.)
* **NEW STOP #O (Class-2 capability gap)** — if a rule-(d) leaf's body names a Class-2
  coefficient table that (a) the source primary does **not** instantiate at dd *and* (b)
  Option B's as-is series is measured insufficient (lift < +8 traced to truncation, §2.4),
  then the leaf's dd accuracy is bounded by data the pipeline cannot yet synthesize. **STOP
  and decide: fund Option A (build the DCT coefficient generator) or fall back to Option C
  (declare the table a library-author pre-condition).** For Group A this never fires (B10
  needs no `_C`; B12/B13 lift is not promised — closure design §7 outcome ii).

---

## 5. Cost estimate

### 5.1 Frames added per integral

| integral | clonable leaves rule (d) adds | new frames | Class-2 needed? | past "surgical" threshold? |
|---|---|---|---|---|
| **B10** | `Lnrat` only (`ddilog`/`Li2omx2` already in `F`) | **+1** (`Lnrat_B10`) | **no** — Lnrat has no series; source Constants suffices | no (3 frames, depth-1) |
| **B12** | `Lnrat` (leaf via `Li2omx2`); `ddilog` already cloned | **+1** (`Lnrat_B12`) | Option B `_C` (source, as-is) | no |
| **B13** | `Lnrat`, possibly `ddilog` if leaf on B13's chain | **+1–2** | Option B `_C` (source) | no (B13 lift not promised) |

Class-1 synthesis is **shared machinery** (one Gap-A extension, works for every wrapper),
paid once. Class-2 for Group A is **zero cost** (source-resident 19-coeff series). No
integral grows past the circuit breaker.

### 5.2 Runtime

Rule (d) adds `Lnrat`'s dd arithmetic (a handful of dd ops per call) inside frames already
paying dd cost. Negligible delta on the closure's ~+5–15% per Group-A integral. Unmeasured
until a full e2e build completes.

### 5.3 Implementation size (v2 — L1 dissolved)

```
L1′ Extend Gap-A: shallow-wrapper recognizer + synthesized-overload
    emitter (Class-1) + clonable-leaf synthesis manifest ...........  4–6 days
L2  Rule (d) frame-discovery + clonable_leaf predicate +
    reroute wiring + circuit breaker ..............................  4–6 days
L3  Emission plumbing (Lnrat_B10 VariantSpec, return_widen reuse)
    + test rewrites ...............................................  2–3 days
L-measure  Option-B e2e (B10 +B12/B13) + §2.4 ceiling triage ......  2–3 days
                                                    subtotal  ≈ 12–18 days (~2.5–3.5 wks)
L4 (OPTIONAL, only if STOP #O fires) Option-A DCT coefficient
    generator + drift gate ........................................  +2–3 wks
```

v1's L1 ("vendor + port dd_ql_support.hpp", 3–4 days) is **deleted**. L1′ replaces it with
synthesis in the agents tree (comparable size, but produces a *capability*, not a vendored
artifact). This is **on top of** the closure design's Stages 1–2 (rules a/b/c, landed).

---

## 6. Falsification tests (built + run — v2 rewritten overlay)

Full evidence: `runs/qcdloop/tier_b_stage2_leaf_promotion/probe_evidence/`. Single-TU,
built against **real** headers + **real** Kokkos, gcc 13.3.0, `-std=c++20` (ceiling probe
`-std=c++17`, no Kokkos). No changes to `agents/`/`tests/`. Four probes:

**(P1) `probe_clone.cpp` (v1, retained) — clone-vs-forwarding + surface-gap enumeration.**
Confirms rename discipline and that the vendored-only surface fails.

| build | surface | compile | runtime on `(1.5, 2.5)` (the Subtask-5 segfault inputs) |
|---|---|---|---|
| A | vendored-only (= pipeline today) | **FAIL, 5 errors** | — |
| B | A + v1 **hand** overlay | OK | runs, exit 0, no segfault |

**(P2) `probe_clone_synth.cpp` (v2, NEW) — the overlay is what the pipeline would SYNTHESIZE.**
This replaces v1's hand-written overlay with **Class-1 mechanical wrappers only** (kAbs/kLog
redirects, Real/Imag accessors, Sign scalar-expr) and **no hand-written `Constants`** — the
source `Constants<ddouble>` primary is used as-is (Option B).

| build | surface | compile | runtime |
|---|---|---|---|
| A_synth | vendored-only | **FAIL, 5 errors** (`abs`,`log`,`Sign`,`Constants` enable_if) | — |
| B_synth | A + **Class-1 synthesized overlay** (no Constants hand-write) | **OK** | `Lnrat_B10(synth) dd re.hi = −0.51082562376599072  double re = −0.51082562376599072  \|diff\| = 0.000e+00` |

**This is the decisive v2 result:** the exact surface the extended Gap-A machinery would emit
(mechanical wrappers) + the unmodified source coefficient primary is sufficient to compile
and run the clone. **No vendored qcdloop-specific header.**

**(P3) `probe_constants_dd.cpp` (v2, NEW) — the Class-2 source-instantiation proof.**
Instantiates `ql::Constants<ddouble>::_num_C()` and `_C(i)` directly from the source primary:

```
num_C=19  sum_C.hi=0.8224670334241132  sum_C.lo=-4.971e-17
```

Proves the 19-coeff table exists at dd from source alone — Option B needs no synthesis. Also
pins that the probe's `enable_if` build error (P1/P2 build A) traces to `ql::kLog`→`Kokkos::log`
(a Class-1 gap), **not** to the coefficient table — the table was never the compile blocker
v1 implied.

**(P4) `probe_optionB_ceiling.cpp` (v2, NEW) — bounds the Option-B lift (§2.4).**
Sums the identical 19 coeffs at double vs dd (inline two-sum/two-prod dd, no Kokkos):

```
max |dd19 − double| over battery      = 1.110e-16   (roundoff dd buys back)
dd recurrence residual |lo/hi| @Y=0.55 = 1.037e-18   (~18 extra digits carried)
19-term truncation floor ~ |C[18]|     = 1.000e-16   (dd CANNOT reduce)
```

Establishes the §2.4 prediction: Option B removes the cancellation roundoff (the part that
survives the `dilog4−dilog5` difference) but not the series truncation (which mostly cancels
in the difference) → **predicted B10 lift +8 to +16**, falsifier = measured lift < +8.

**What the probes establish (before committing weeks):**

1. **Rename discipline is sound (STOP-#K refutation)** — P1/P2, both overlays run to
   completion on the segfault inputs.
2. **The support surface is pipeline-synthesizable, not vendor-only** — P2 clears the entire
   Class-1 gap with mechanical overloads; P3 shows Class-2 is source-resident. This is the
   v2 correction to v1's §3.4.
3. **Option B is bounded, not hand-waved** — P4 gives a numerical ceiling and a concrete
   falsifier for the lift, discharging the brief's "predict a ceiling, don't promise +18.43."
4. **The probes do NOT prove a lift** — `Lnrat`'s TScale branch has no cancellation, so
   dd==double here (`|diff|=0`). The lift is B10's `Li2omx2`/`dilog4−dilog5` story, measured
   only at a full e2e run — STOP #A's job, predicted (not promised) by §2.4.

**What would still falsify the design at e2e (not cheaply pre-testable):** B10 emits all
clones dd and the cancellation executes at dd, but measures lift < +8 → either an intervening
double narrowing (STOP #A / value-flow model wrong) or the §2.4 truncation-decorrelation
caveat (→ fund Option A / STOP #O). The probes cannot pre-empt this — it needs the full
5000-sample kinematic battery.

---

## 7. What this design does NOT solve

1. **Inward parameter widening on a shared original** — still refused (closure design §8.2).
   Rule (d) clones (own params), so it never *needs* to widen a shared signature. Not in
   Group A.
2. **B12's floor location** — B12's dominant chain does not cover its `coeff0.imag` hotspot
   (Subtask 3). Rule (d) lets B12's `Lnrat` clone but does not move the floor; orthogonal.
3. **Full +18.43 ddilog ceiling under Option B** — the 19-coeff truncation caps ddilog's
   *absolute* accuracy at ~1e-16 (§2.4). Only Option A (43-coeff DCT synthesis) recovers the
   full ceiling; funded only if STOP #O fires.
4. **Group B (B15/B16/BIN0–4)** — dd-insufficient (Item 7); rule (d) would clone their
   leaves (`kfn`/`ltspence`/`cspence`) and make them *measurable*, not *sufficient*. Out of
   scope; the circuit breaker + Group-B dd-insufficiency keep them from being accepted.
5. **Class-1 recognizer beyond straight-line delegation** — a wrapper whose body is a
   multi-statement computation (not a single delegation/accessor/scalar-expr) is **not**
   Class-1 and is refused (clause 2). Widening the recognizer to such bodies is future work;
   Group A's wrappers are all straight-line, so it is not needed now.

---

## 8. Implementation dispatch shape (proposal only — do NOT dispatch)

Rule (d) presupposes the closure design's Stages 1–2 (rules a/b/c), which are landed
(Subtasks 1a/1b/2a/2b). v2's dispatch (v1's L1 dissolved):

* **Subtask L1′ — extend the Gap-A machinery to SYNTHESIZE Class-1 wrappers.** Add the
  shallow-wrapper recognizer (structural: body = single delegation to a `_MATH_FN_NAMES` op
  / member accessor / scalar-expr over the param) + the synthesized-overload emitter (§2.2),
  producing the dd overloads into the per-region shim. **Deliverable:** the extension +
  unit tests + the P2 probe made a permanent compile test. **No vendored header.** This is
  the piece that restores the demo's "pipeline synthesizes what it needs" claim; land it
  first (it is independent of rule (d)).

* **Subtask L2 — rule (d) frame-discovery + `clonable_leaf` predicate.** Frame-discovery
  fixed point in `chain_promote` (walk promoted-body calls, test predicate against the L1′
  synthesis manifest + source-instantiation check, add clones to `F`, seed bodies, record
  reroutes). Wire the circuit breaker (`chain_closure_oversized`) and the narrowed
  `chain_closure_escapes`. **Gate:** predicate false-positive = STOP #K; must refuse before
  emission.

* **Subtask L3 — emission plumbing.** `Lnrat_B10` `VariantSpec` (reuses `return_widen` from
  Subtask 2a), test rewrites (inverted `Lnrat` assertions).

* **Subtask L-measure — Option-B e2e + §2.4 triage.** B10/B12/B13 e2e re-run (seed 12345,
  5000 samples, kernel-scope + positive-lift gate). **Success = B10 reaches the positive-lift
  gate with `Li2omx2_B10`+`ddilog_B10`+`Lnrat_B10` all dd and measured
  `kernel_measured_lift ≥ +8`** (closure §7 bar; §2.4 predicts +8…+16). **If lift < +8**,
  triage per §2.4: intervening narrowing (STOP #A) vs truncation-decorrelation (STOP #O →
  scope Option A).

* **Subtask L4 (OPTIONAL, gated on STOP #O) — Option-A coefficient synthesis.** Only if
  L-measure proves the 19-coeff truncation is the binding constraint. Build the offline
  chebfun-style DCT generator (sample Li₂ at N dd Chebyshev nodes → DCT → emit
  `Constants<ddouble>::_C`) + a drift gate that compares generated coefficients to the oracle
  *for validation only*. This is real synthesis-from-source, not a port.

**If L1′ proves a Group-A wrapper is not synthesizable** (a body shape the recognizer cannot
handle soundly), **STOP at L1′** and hand the scope call back — that would mean the Class-1
premise is narrower than §2.2/§7 established. **If L-measure fires STOP #O**, the coefficient
table is the real gap: decide Option A vs C (§2.3) — do not vendor.

---

## Appendix — invariants carried from the closure design

* Variants are **per-integral clones**; the shared original (`Lnrat`, `ddilog`, `Li2omx2`)
  is never edited. Rule (d) obeys this — it clones, never mutates the primary.
* Refusals (`clonable_leaf` fail, oversized) computed **before any tree mutation**.
* Gate logic **unchanged**; rule (d) only enlarges the (non-inert) candidate the gates
  measure. B10 reaching the lift gate is the point.
* Conservative-parser contract: false-negative (refuse a clonable leaf) safe; false-positive
  (clone an un-instantiable leaf) is STOP #K — guarded by the predicate + the L1′ synthesis
  compile test + the §7 probes.
* **v2 architectural invariant:** the pipeline **synthesizes** qcdloop-specific dd support
  (Class-1 wrappers) or uses **source-resident** data (Class-2 tables at dd), and **vendors
  nothing qcdloop-specific**. `third_party/include/` stays app-independent. The oracle
  `qcdloop@ddfun_enabled` is consulted for **validation drift only**, never for generation.
```
