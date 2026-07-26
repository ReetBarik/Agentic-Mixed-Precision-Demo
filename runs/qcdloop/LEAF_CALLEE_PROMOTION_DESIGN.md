# Leaf-Callee Promotion — Design Notes (Group A precision-lift unblock)

Status: **design + falsification probe done, STOP before implementation** (2026-07-26).
Scope: extend closure-scoped dd promotion (`CLOSURE_SCOPED_CHAINS_DESIGN.md`) so leaf
callees like `ql::Lnrat` / `ql::ddilog` become **cloned, promoted frames** instead of a
`chain_closure_escapes` refusal frontier. Discipline: **design only** — no changes to
`agents/`, `tests/`. The §7 probe was built and run (Subtask-5-style single-TU); nothing
else moved.

Reads as the successor to two docs, both of which stand:
* `CLOSURE_SCOPED_CHAINS_DESIGN.md` — rules (a)/(b)/(c), the §2.4 refusal frontier, the
  §3 designed-exit gate, the §7 STOP discipline. **This design changes exactly one thing
  in it:** it moves `ql::Lnrat`/`ql::ddilog` from the §2.4 "callee-not-in-`F`" refusal set
  into `F` as clonable frames. Everything else in that doc is unchanged.
* `tier_b_stage2_subtask5/TIER_B_STAGE2_SUBTASK_5_2026-07-26.md` (STOP #K) — proved the
  *forwarding-overload* path is unsound (self-recursion). This design takes STOP #K's own
  recommended option 2 ("make `Lnrat`/`ddilog` chain-frames") and works out whether it is
  actually buildable. **The answer is: yes, but only after a bounded support-surface port
  the pipeline does not yet have (§3, category (d)).**

> **Load-bearing correction up front (read before §1).** The closure design's §2.4
> lists `ql::Lnrat` as a hard refusal because "its signature we will not touch." That is
> the wrong lens for a leaf callee. We are **not** touching `ql::Lnrat`'s signature — we
> **clone** it to `Lnrat_B10` (a new symbol, reachable only from the chain's rerouted
> call sites) and promote the clone's body, exactly as the pipeline already clones
> `ddilog`/`Li2omx2` whose bodies sit *on* the chain. STOP #K's recursion pit was a
> property of a same-name *overload*, not of a *renamed clone*. The §7 probe confirms the
> renamed clone builds and runs on the exact inputs that segfaulted the overload. The real
> blocker is not recursion and not signatures — it is that the dd **support surface**
> `Lnrat_B10`'s body needs (`ql::kAbs/kLog/Real/Imag/Sign` at `ddouble`, `Constants<ddouble>`)
> is **not vendored** into this repo. §3 is the whole ballgame.

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
It is a **leaf callee** — called *from* a chain line (`Li2omx2:706`, `:701`) but its own
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
is the right shape but the §7 probe shows it must be checked against the surface the
pipeline **actually has**, not the surface that exists upstream. Refined predicate, all
clauses required:

```
clonable_leaf(g) :=
   (1) g is a function template with a body available in the analysed headers
       (not an extern/vendored-binary symbol);              # can be cloned at all
 ∧ (2) g's body, with reads promoted to dd, calls ONLY:
         - other clonable_leaf callees (recurse), OR
         - symbols the dd SUPPORT SURFACE resolves at dd
           (vendored quad::ddfun ops, ql:: dd helpers, Constants<ddouble>),
       i.e. rule (d)'s transitive closure over g terminates at the
       termination boundary (§2);                            # body instantiable at dd
 ∧ (3) g is NOT self-recursive under a SAME-NAME overload set that a rename
       cannot separate (STOP #K guard — §4);               # rename discipline safe
 ∧ (4) cloning g does not require widening a shared g PARAMETER that a
       non-chain caller also binds — the clone gets its own params, so this
       is automatically satisfied for a pure clone, but a leaf whose promotion
       demands INWARD dd on a shared original's signature is refused (§8.2 of
       the closure design still holds).
```

Clause (2) is not decidable by "does the primary compile clean at dd" alone, because the
pipeline's dd surface is **incomplete** (§3). The predicate must be evaluated against a
**declared support-surface manifest** (§3.4): a leaf is clonable iff every dd symbol its
promoted body names is either (a) vendored, (b) in the Subtask-3 catalog, or (c) itself a
clonable leaf. Anything else → the leaf is **not** clonable → `chain_closure_escapes`
(honest terminal, not a doomed emission). This keeps the conservative-parser contract:
false-negative (refuse a clonable leaf) is safe; false-positive (clone an un-instantiable
leaf) is the STOP #K hard-fail we must never ship.

### 1.3 Which frames become eligible

For the Group-A chains, rule (d) makes exactly these leaves eligible:

| leaf | called from | primary | promotable body? |
|------|-------------|---------|------------------|
| `ql::Lnrat` (TScale overload, `:152`) | `Li2omx2:701,706` | straight-line `log/abs/sign` | yes, **iff** dd `kLog/kAbs/Sign` + `Constants<ddouble>::_ipio2` exist (§3) |
| `ql::ddilog` (`:162`) | `Li2omx2:702,708` | Chebyshev series | already IN `F` for B10 (chain lines inside it) — rule (d) not needed for B10; **needed for B12/B13** where ddilog is a leaf |
| `ql::kfn`, `ql::ltspence`, `ql::cspence` | Group-B chains | series / branch | out of scope (Group-B dd-insufficient, §6) |

So for **B10** specifically, rule (d) is needed for **`Lnrat` only** — `ddilog`/`Li2omx2`
are already cloned frames. This narrows the headline case to a single leaf.

---

## 2. Termination of the transitive closure

### 2.1 The termination boundary

Rule (d) recurses (a clone's body may call further leaves). It terminates because every
call in a promoted body resolves to exactly one of four **boundary** kinds, none of which
re-enter rule (d):

1. **Vendored standard math via the `quad::ddfun` surface** — `abs, sqrt, log, exp, pow,
   sin, cos, …` on `ddouble`/`ddcomplex` (`third_party/include/dd_math.hpp`,
   `dd_complex.hpp`). These are the `_MATH_FN_NAMES` bridge vocabulary; they resolve at dd
   with no cloning. **Boundary — not a frame.**
2. **Constants via the Subtask-3 π-family catalog** — `_pi2o6`, `_ipio2`, etc. derived
   deterministically as bit-exact `make_dd(…)` (`agents/shared/constant_derive.py`,
   `TIER_B_STAGE2_SUBTASK_3`). **Boundary — a value, not a frame.**
3. **The dd helper layer `ql::{kAbs,kLog,Real,Imag,Sign,iszero}` at dd** — *if* it exists
   (§3: it does not yet; category (d)). Once supplied, these are ordinary overloads, **not
   clones** (they are the leaves' leaves; they bottom out in boundary 1). **Boundary.**
4. **`Kokkos::complex<ddouble>` / vendored `ddcomplex` container ops** — `+,-,*,/`,
   `real()`, `imag()`. Provided by `dd_complex.hpp` (for `ddcomplex`) — §3 decides whether
   B10's reads flow through `ddcomplex` directly or `Kokkos::complex<ddouble>`. **Boundary.**

Rule (d) only ever adds a frame for boundary-kind **none-of-the-above** = an app template
whose body is available and calls into these boundaries. The recursion adds a frame at most
once per app template in the (finite) header set.

### 2.2 Bounded, acyclic — the proof

* The universe of clonable app templates is **finite** (the analysed headers contain
  finitely many function templates).
* Rule (d) is **monotone** (only adds to `F`) and records each app template as a clone at
  most once → the `F`-growth iteration halts after ≤ (#app-templates) rounds.
* **No cycle can force unbounded growth.** The qcdloop special-function call graph on these
  chains is a **DAG** — verified in the closure design §2.5 and re-verified here:
  `Lnrat`'s body (`:141-155`) calls `kLog/kAbs/Sign/Imag/iszero/Constants` — **all
  boundary kinds, no app-template callee** → `Lnrat` is a **sink**; rule (d) adds it and
  stops. `ddilog`'s body calls `kLog/kPow/Real/Constants` — all boundary → also a sink.
  `Li2omx2` calls `Lnrat`, `ddilog`, `kLog`, `kAbs`, `Constants` → its only app-template
  callees are the two sinks. So the B10 rule-(d) frontier is depth-1 and closed:
  `{Li2omx2_B10 → Lnrat_B10 (sink), ddilog_B10 (sink)}`. **Bounded, acyclic. QED.**
* **Self-recursion is not a cycle in `F`.** `ddilog`'s Chebyshev loop and `Lnrat`'s
  branches contain no self-call (verified: no `ql::ddilog`/`ql::Lnrat` token inside their
  own bodies). Even if a leaf *were* self-recursive, the rename (§4) binds its self-calls
  to the clone name, so it is still one frame, not an infinite family.

### 2.3 The circuit breaker (backstop, not a design lever)

Carry over the closure design's §6.1 stance: no size *cap* (a cap re-introduces the subset
boundary). But keep a **circuit breaker**: if rule (d) would grow `F` past a diagnostic
threshold (say 8 frames or a rule-(d) recursion depth > 3), abort the chain with
`chain_closure_oversized` rather than emit a giant fragile patch. For Group-A this never
fires (B10 frontier is depth-1, 3 frames total); it exists only to bound a pathological
selector seed. This is graceful degradation (§6), not a scope choice.

---

## 3. Support-surface scoping — the category-(d) blocker (the crux)

This is the section the whole design turns on, and where the §7 probe did its work.

### 3.1 The B10 support-surface bill of materials

Every dd symbol the B10 closure's promoted bodies name, cross-checked against the four
questions the brief asks. Sources: `Lnrat` body `kokkosUtils.h:141-155`, `ddilog`
`:162-232`, `Li2omx2` `:692-712`; helper defs `src/kokkosMaths.h:270-370`; vendored
`third_party/include/*`; oracle fork `qcdloop@ddfun_enabled:src/qcdloop/kokkosMaths_dd.h`.

| symbol (at dd) | used by | (a) vendored? | (b) catalog? | (c) primary clean at dd? | (d) bridging needed? |
|---|---|---|---|---|---|
| `ddadd/sub/mul/div`, `ddouble`/`ddcomplex` ops | all | ✅ `dd_math`/`dd_complex` | — | — | none |
| `abs/log/sqrt/exp/pow/sin/cos` on dd | ddilog, Lnrat | ✅ `quad::ddfun::*` | — | — | none (via `_MATH_FN_NAMES`) |
| `_pi2o6`, `_ipio2`, `_half`, `_pi` at dd | ddilog, Lnrat, Li2omx2 | partial (`dd_pi()`) | ✅ π-family catalog | — | **catalog wiring (exists, Subtask 3)** |
| **`ql::kAbs(ddouble/ddcomplex)`** | Lnrat, Li2omx2 | ❌ | ❌ | ❌ wraps `Kokkos::abs`, no dd overload | **🔴 (d) — MISSING** |
| **`ql::kLog(ddouble/ddcomplex)`** | Lnrat, ddilog, Li2omx2 | ❌ | ❌ | ❌ wraps `Kokkos::log`, no dd overload | **🔴 (d) — MISSING** |
| **`ql::Real/Imag(ddcomplex)`** | ddilog, Lnrat | ❌ | ❌ | ❌ `double`/`complex<double>` overloads only | **🔴 (d) — MISSING** |
| **`ql::Sign(ddouble/ddcomplex)`** | Lnrat | ❌ | ❌ | ❌ `double`/`complex<double>` overloads only | **🔴 (d) — MISSING** |
| **`ql::iszero<…>(ddouble)`** | Lnrat (`:145`) | ❌ | ❌ | template, but calls `kAbs(dd)` + `Constants<dd>` | **🔴 (d) — transitively missing** |
| **`ql::kPow<…>(ddouble,int)`** | ddilog (`:179,199,211`) | ❌ | ❌ | template `TMass temp=TMass(1.0); temp*=base` — **clean at dd** | ⚠️ instantiates iff `ddouble(1.0)`+`*=` exist (they do, `dd_math`) → **OK, no bridge** |
| **`Constants<ddouble>::_C(i)`, `_num_C()`** | ddilog | ❌ | ❌ | ❌ pipeline primary has **19** coeffs (`kokkosMaths.h:23`); dd-accurate ddilog needs **43** (oracle `kokkosMaths_dd.h`) | **🔴 (d) — MISSING (accuracy, not just compile)** |

**Every 🔴 row is a category-(d) blocker.** They cluster into one artifact.

### 3.2 The blocker, named: the un-vendored `ql` dd helper layer

All 🔴 rows are provided, upstream, by a **single file**:
`qcdloop@ddfun_enabled:src/qcdloop/kokkosMaths_dd.h` (393 LOC, namespace `ql`, backed by
`ql::ddfun`). It defines exactly `ql::kAbs/kLog/kSqrt/kConj/Real/Imag/Sign/Max/Min/Htheta`
on `ddouble`/`ddcomplex`, plus `Constants<ddouble>` with the **43-term** dd Chebyshev
series and dd `_pi2o6/_ipio2/_half/…`. It is the dd twin of `src/kokkosMaths.h`.

**It is NOT vendored into this repo.** `third_party/include/` vendored only the
*framework-agnostic* arithmetic (`quad::ddfun` math + `dd_complex`) — deliberately, because
that layer is app-independent. The `ql::` helper layer is **qcdloop-specific** (it knows
`Constants`, `ql::Real`, the on-shell cutoff), so it was left out. That omission is exactly
the "semantics the premise assumed away" that STOP #K identified — but STOP #K read it as
*impossible*; it is actually *un-vendored*, i.e. **a bounded porting task**, because the
code demonstrably exists and works (it is what the dd oracle itself is built from).

Note a **namespace mismatch** the port must resolve: the vendored arithmetic is
`quad::ddfun::ddouble`; the oracle helper layer is written against `ql::ddfun::ddouble`
(a fork with `Kokkos::bit_cast` portability + extra operators — see
`third_party/include/README.md`). The port cannot be a verbatim copy; it must be
**re-expressed against the vendored `quad::ddfun`** types. This is the one non-mechanical
part of the port (§6 costs it).

### 3.3 Does B10's `Kokkos::complex<double>` read flow through `ddcomplex` or
`Kokkos::complex<ddouble>`? (the brief's §3 empirical question)

The chain's `TOutput` is `Kokkos::complex<double>`. Promoting it to dd has two candidate
targets: the vendored `quad::ddfun::ddcomplex`, or `Kokkos::complex<quad::ddfun::ddouble>`
via Kokkos's primary `complex<T>` template. The existing pipeline **already commits to
`ddcomplex`** — `fanout.py`/`boundary.py`/`shim_normalise.py` all promote complex
containers to `quad::ddfun::ddcomplex` (grep: `dispatch.py:308`, `fanout.py:243/271`,
`shim_normalise.py:60-63`), and the `ddilog`/`Li2omx2` clones already do this successfully
(B12 built + executed end-to-end, Subtask 3). So B10's reads flow through
**`quad::ddfun::ddcomplex` directly**, via the vendored `dd_complex.hpp` surface — **no
`Kokkos::complex<ddouble>` and no bridging on the container axis.** The §7 probe confirms:
`Lnrat_B10<ddcomplex,double,ddouble>` compiles and runs with `ddcomplex` as `TOutput`.
(Whether `Kokkos::complex<ddouble>` *also* works is moot — the design does not need it, and
committing to `ddcomplex` keeps B10 consistent with every clone the pipeline already emits.)

### 3.4 Resolution of the category-(d) blocker

The port is **required** and is the design's first implementation subtask (§9, Subtask L1).
Concretely, vendor a new header — call it `third_party/include/dd_ql_support.hpp` — that
provides, against the vendored `quad::ddfun` types:

* `ql::kAbs/kLog/kSqrt/kConj(ddouble|ddcomplex)` bridging to `quad::ddfun::{abs,log,sqrt,conj}`;
* `ql::Real/Imag/Sign(ddouble|ddcomplex)`;
* `ql::iszero<…>(ddouble)` (needs `Constants<ddouble>::_qlonshellcutoff`);
* a `Constants<ddouble>` specialisation (or a partial-specialisation path) with the 43-term
  dd Chebyshev series (`_C`/`_num_C`) and the dd `_pi2o6/_ipio2/_half/_pi/_zero/_one/…`
  the bodies use — the π-family entries can reuse the Subtask-3 catalog derivations for
  bit-exactness.

The §7 probe's `WITH_OVERLAY` block (~30 LOC) is a working miniature of this header for the
`Lnrat` subset — it proves the shape compiles and runs. The full port is ~150–250 LOC
(a subset of the oracle's 393, re-namespaced; `Max/Min/Htheta/kConj` are not on the Group-A
chains and can be omitted or ported for completeness).

**This is a design blocker resolved by scoping it as owned work, not by hand-waving.** It is
category (d), it is bounded, the source exists and is proven (it builds the oracle), and the
§7 probe de-risks it. It is *not* new shim-generation logic and *not* an LLM capability —
it is a one-time vendored header, deterministic forever after.

---

## 4. Rename discipline (how the clone avoids the Subtask-5 self-recursion pit)

### 4.1 Why the forwarding overload recursed, and why the clone does not

STOP #K's recursion was structural: an injected **same-name** overload
`ql::Lnrat(ddouble,ddouble)` whose body calls `ql::Lnrat(ddouble,ddouble)` — C++ selects by
*argument type*, ignoring the explicit `<…>`, so it re-selects itself forever.

A **clone** breaks every link of that chain:
* the clone is a **distinct symbol** `Lnrat_B10` (or `ql::Lnrat_B10`), so there is no
  overload set to re-enter — the name that would recurse does not exist as a same-name
  sibling;
* the clone's body names only `ql::kLog/kAbs/Sign/Constants` and vendored ops — it **never
  names `Lnrat_B10` or `ql::Lnrat`** (verified in the §7 probe source and confirmed by the
  fact that `Lnrat`'s body has no self-call to begin with);
* the call site `Li2omx2_B10:706` is **rerouted** to `Lnrat_B10` by the existing
  topological callee-before-caller reroute (`_reroute_in_function`, the same machinery that
  already reroutes `ddilog`→`ddilog_B10`).

The §7 probe is the empirical proof: `Lnrat_B10<ddcomplex,double,ddouble>(1.5,2.5)` **builds
and runs to completion (exit 0), no segfault**, on the exact inputs that made the Subtask-5
forwarding overload stack-overflow.

### 4.2 Self-recursive leaves (the general rule, not needed for B10)

If a clonable leaf *did* contain a self-call (e.g. a hypothetical recursive series), the
clone-and-rename discipline handles it the way it already handles `ddilog`/`Li2omx2` self-
references: **rewrite in-body self-calls to the clone name** in the same descending-line
edit pass that promotes reads. `ddilog_B10`'s Chebyshev recurrence is straight-line (no
self-call), so this is vacuous for B10 — but the rule must be stated so a future leaf with a
self-call binds to the clone, not the primary. This is clause (3) of the predicate (§1.2):
a leaf whose self-recursion crosses a **same-name overload set** a rename cannot separate is
**refused** (the STOP-#K guard, generalised).

### 4.3 Overlapping clones across integrals (B10 and B12 both need `ddilog_*`)

Both B10 and B12 chains touch `ddilog`. Two options:

* **Per-integral clones (recommended, and what the pipeline already does).**
  `ddilog_B10` and `ddilog_B12` are distinct symbols in distinct per-integral variant trees
  (each integral runs in its own isolated tree — `per_integral_orchestrator`). They never
  coexist in one TU, so there is no collision and no shared-instantiation hazard. `Lnrat_B10`
  and `Lnrat_B12` likewise. This preserves the Appendix invariant "variants are per-integral
  clones; the shared original is never edited," and it is byte-consistent with how
  `Li2omx2_B10` vs `Li2omx2_B12` already work.
* **Shared dd instantiation (rejected for v1).** A single `Lnrat<ddcomplex,…>` shared across
  integrals would be smaller code but re-introduces cross-integral coupling the whole
  per-integral architecture exists to avoid (a shared instantiation measured on B10 would
  perturb B12's tree). Defer; per-integral clones are correct and already supported.

So: **per-integral clones**, no new collision surface beyond what `variant_naming.py` /
`assert_no_collisions` already guard. `Lnrat_B10` is just another name in that namespace.

---

## 5. Interaction with the existing tree

| component | change |
|---|---|
| **rule (c)** (`chain_promote._apply_rule_c`) | **unchanged** — stays exactly as landed in Subtask 2b. Rule (d) feeds it a larger `F`; rule (c) still propagates returns across chain-internal call edges. `Lnrat_B10`'s return (dd) flows into `Li2omx2_B10` via the *same* rule-(c) return-widen it already applies to `ddilog_B10`. |
| **rule (a)** (`_expand_value_closure`) | eligible-frame set grows to include rule-(d) frames; the rule-(a) logic (decl-widen for carriers) is unchanged, just applied inside `Lnrat_B10`/`ddilog_B10` too. |
| **NEW rule (d)** (`chain_promote`) | frame-discovery fixed point: walk promoted-body calls, test `clonable_leaf`, add clones to `F`, seed clone bodies, record reroutes. Reuses `CallGraph` + `region_scan` (no new source-analysis machinery). |
| **π-family catalog** (`constant_derive.py`) | **used more** — `Lnrat_B10`/`ddilog_B10` bodies pull `_pi2o6`/`_ipio2` at dd. Already proven end-to-end (Subtask 3); the 43-coeff `_C` series is the one new catalog/support-surface entry (§3.4). |
| **shim normaliser** (`shim_normalise.py`) | **used more** (more clone bodies → more shims to normalise). No logic change; its dedup/unary-canonicalisation applies to `Lnrat_B10`'s shim as-is. |
| **fanout manifest** (`fanout.py`) | **grows** — `Lnrat_B10` becomes a new `VariantSpec` with a `return_widen` (its `TOutput` widens to `ddcomplex`). This is the *first time `Lnrat` appears in a manifest* (Subtask 5 verified it appears in zero manifests today). No schema change — a clone is a clone. |
| **support surface** (`third_party/include`) | **NEW header** `dd_ql_support.hpp` (§3.4). The one genuinely new artifact. |
| **closure-scoped predicate** (`clonable_leaf`) | **new predicate**, evaluated against the support-surface manifest. This is the false-positive guard (§1.2). |
| **`chain_closure_escapes`** (`result.py`) | **still fires** — for leaves that fail `clonable_leaf` (body names an un-portable symbol, or demands inward param widening). Now a *smaller* set (Lnrat/ddilog move out of it). |
| **kernel-scope + positive-lift gates** | **unchanged.** B10 now reaches the positive-lift gate for the first time (today it dies at `apply_failed` upstream). |

### What tests break / what's new

* **Break (assertions invert):** any `test_chain_promote` case asserting `Lnrat` is *not*
  cloned / is a `chain_closure_escapes` frontier. Under rule (d), B10 now emits `Lnrat_B10`.
* **New:**
  * `clonable_leaf` predicate unit tests (clonable sink `Lnrat`; transitively-clonable
    `Li2omx2`; **non**-clonable leaf whose body names an un-ported symbol → refuse);
  * rule-(d) frame-discovery + termination test (B10 frontier = `{Lnrat_B10}`, depth 1,
    circuit breaker not tripped);
  * `dd_ql_support.hpp` compile test (a checked-in TU that `#include`s it and instantiates
    `ql::kLog(ddouble)` etc. — the §7 probe promoted to a permanent regression);
  * e2e: B10 emits a dd-returning `Lnrat_B10` **and** `ddilog_B10`/`Li2omx2_B10`, and the
    `dilog4-dilog5` cancellation at `B1m.h:240` executes at dd.
* **Stay green:** all Layer 0–5 mechanical tests; rule (a)/(b)/(c) tests; scorer; the whole
  non-chain path.

### STOP-condition impact

* **STOP #A (measurement falsification)** — unchanged in meaning; now *reachable* for B10.
  If B10 emits (rule (d)+(c) fire, all clones dd) but measures `chain_no_lift`, the
  value-flow model is wrong — strongest disconfirmation.
* **STOP #B (accept↔reject flip)** — unchanged; guard B13/B14 stay byte-identical unless
  rule (d) legitimately changes their `F`.
* **STOP #K (emitted transform breaks build/runtime)** — **re-armed and central.** The
  `clonable_leaf` predicate + the `dd_ql_support.hpp` compile test are the two guards. If a
  clone is emitted whose body names a symbol the support surface lacks, the build fails →
  STOP #K. The predicate must refuse *before emission* (conservative-parser contract). The
  §7 probe is the pre-implementation discharge of STOP #K for `Lnrat`.
* **NEW STOP #N (support-surface drift)** — if the vendored `dd_ql_support.hpp` and the
  oracle's `kokkosMaths_dd.h` diverge (upstream changes the Chebyshev coeffs, say), a clone
  could compile but compute a subtly wrong dd value → a silent precision regression the
  lift gate might not catch (it compares to the dd oracle, which uses the *oracle's* series;
  if the vendored series matches, they agree — so the guard is a **bit-exactness test of the
  vendored series vs the oracle's**, analogous to Subtask-3's STOP #C). Add it.

---

## 6. Cost estimate

### 6.1 Frames added per integral

| integral | clonable leaves rule (d) adds | new frames | category (d)? | past "surgical" threshold? |
|---|---|---|---|---|
| **B10** | `Lnrat` only (`ddilog`/`Li2omx2` already in `F`) | **+1** (`Lnrat_B10`) | yes — needs `dd_ql_support.hpp` (§3.4), shared one-time | no (3 frames total, depth-1 frontier) |
| **B12** | `Lnrat` (leaf on B12 chain via `Li2omx2`); `ddilog` already cloned | **+1** (`Lnrat_B12`) | same shared header | no |
| **B13** | `Lnrat`, and possibly `ddilog` if it is a leaf on B13's chain | **+1–2** | same shared header | no (but B13's lift is not promised — closure design §7 outcome ii) |

The category-(d) work is **shared across all three** (one `dd_ql_support.hpp`), so it is
paid once, not per integral. No integral's closure grows past the circuit-breaker threshold;
graceful degradation (§2.3 → fall through to whole-app dd routing, or
`chain_closure_oversized`) is designed but **never triggered for Group A**. Whether to wire a
literal whole-app-dd fallback for an over-threshold integral is **deferred as scope-creep** —
Group A does not need it, and Group B is dd-insufficient anyway (§6.2 of closure design).

### 6.2 Runtime — inherits the closure design's §6.2 estimate unchanged

Rule (d) adds `Lnrat`'s dd arithmetic (a handful of dd ops per call) inside frames already
paying dd cost. Negligible delta on top of the closure's ~+5–15% per Group-A integral.
Unmeasured until a full e2e build completes (none has, for B10).

### 6.3 Implementation size

```
L1  Vendor + port dd_ql_support.hpp (re-namespace ql::ddfun→quad::ddfun,
    43-coeff series, bit-exactness test vs oracle) ..............  3–4 days
L2  Rule (d) frame-discovery + clonable_leaf predicate +
    support-surface manifest + reroute wiring ...................  4–6 days
L3  Manifest/emission plumbing (Lnrat_B10 VariantSpec, return_widen
    reuse) + test rewrites .....................................  2–3 days
e2e B10 (+B12/B13) re-run + triage .............................  2–3 days
                                                        total  ≈ 11–16 working days (~2–3 wks)
```
This is **on top of** the closure design's Stages 1–2 (which must land first — rule (d)
presupposes rule (c)). It is comparable to one closure stage.

---

## 7. Falsification tests (built + run — Subtask-5-style, cheap)

Full evidence: `runs/qcdloop/tier_b_stage2_leaf_promotion/probe_evidence/`
(`probe_clone.cpp`, `build_A.err`, `README.md`). Single TU, ~90 LOC + ~30 LOC overlay, no
changes to `agents/`/`tests/`. Built against **real** headers + **real** Kokkos, gcc 13.3.0,
`-std=c++20`.

**The probe clones `Lnrat` → `Lnrat_B10` (dd body, renamed, no self-call) and asks two
questions the design lives or dies on:**

| build | surface | compile | runtime on `(1.5, 2.5)` — the Subtask-5 segfault inputs |
|---|---|---|---|
| **A** | vendored-only (= pipeline today) | **FAIL, 5 errors** | — |
| **B** | A + `dd_ql_support` overlay (§3.4 miniature) | **OK** | **runs, exit 0, NO segfault** |

Build-A errors are the category-(d) inventory, verbatim:
`no matching function for call to 'abs(const quad::ddfun::ddouble&)'`,
`… 'log(const quad::ddfun::ddouble&)'`, `… 'Sign(quad::ddfun::ddouble)'`,
`no type named 'type' in 'struct std::enable_if<false, double>'` (Constants<ddouble>).

Build-B run output:
`Lnrat_B10 dd re.hi = -0.51082562376599072   double re = -0.51082562376599072   |diff| = 0.000e+00`.

**What the probe establishes (before committing weeks):**

1. **Rename discipline is sound (the STOP-#K refutation).** A renamed clone builds and runs
   to completion on the exact inputs that segfaulted the forwarding overload. The design's
   core premise — "clone instead of forward" — is empirically valid, not just asserted.
2. **The blocker is precisely the support surface (category (d)), and it is bounded.** Build
   A fails *only* on the four 🔴 symbol classes of §3.1; Build B, adding a ~30-LOC overlay,
   clears all of them. This confirms §3.4's claim that the blocker is a bounded vendored
   header, not an open-ended semantic gap.
3. **It does NOT prove a lift.** `Lnrat`'s `TScale` branch has no cancellation, so dd==double
   here (`|diff|=0`). The probe proves *compilability + termination*, which is exactly what
   STOP #K put in doubt. The precision lift is B10's `Li2omx2`/`ddilog4-ddilog5` cancellation
   story, measurable only at a full e2e re-run — that is STOP #A's job, not the probe's.

**What would still falsify the design at e2e (not cheaply pre-testable):** B10 emits all
clones dd and the cancellation executes at dd, but measures `chain_no_lift` → the value-flow
model missed an intervening double narrowing (STOP #A). The probe cannot pre-empt this
because it needs the full 5000-sample kinematic battery; it is the headline measurement the
implementation exists to reach.

---

## 8. What this design does NOT solve

1. **Inward parameter widening on a shared original** — still refused (closure design §8.2).
   Rule (d) clones, giving each clone its own params, so it never *needs* to widen a shared
   signature. A leaf whose promotion demanded inward dd on a param a non-chain caller binds
   is `chain_closure_escapes` (predicate clause 4). Not in Group A.
2. **B12's floor location** — B12's dominant chain does not cover its `coeff0.imag` hotspot
   (Subtask 3). Rule (d) lets B12's `Lnrat` clone, but does not move the floor; that is a
   chain-selection question, orthogonal.
3. **Group B (B15/B16/BIN0–4)** — dd-insufficient (Item 7); rule (d) would clone their
   leaves (`kfn`/`ltspence`/`cspence`) and make them *measurable*, but not *sufficient*.
   Out of scope; the circuit breaker + Group-B dd-insufficiency keep them from being
   accepted.
4. **The `ql::ddfun` vs `quad::ddfun` fork reconciliation** — the port (§3.2) re-expresses
   the oracle helper layer against vendored types. If upstream `quad::ddfun` lacks an op the
   oracle's `ql::ddfun` had, that op must be added to the vendored surface too — flagged as a
   L1 risk, bounded by the §7 overlay (which needed nothing beyond `abs/log/dd_pi`).

---

## 9. Implementation dispatch shape (proposal only — do NOT dispatch)

Rule (d) presupposes the closure design's Stages 1–2 (rules a/b/c) are landed — they are
(Subtasks 1a/1b/2a/2b). So this is a clean follow-on, in **three subtasks**, strictly
ordered:

* **Subtask L1 — vendor `third_party/include/dd_ql_support.hpp` (the category-(d) port).**
  Re-express the oracle `kokkosMaths_dd.h` helper layer against vendored `quad::ddfun`:
  `ql::kAbs/kLog/kSqrt/kConj/Real/Imag/Sign/iszero(dd…)` + `Constants<ddouble>` with the
  43-term dd Chebyshev series and dd π-family (reuse Subtask-3 catalog for bit-exactness).
  **Deliverable:** the header + a checked-in compile/bit-exactness test (the §7 probe made
  permanent + STOP #N series check). **No agent changes.** This is the riskiest, most
  independent piece; land and prove it first.

* **Subtask L2 — rule (d) frame-discovery + `clonable_leaf` predicate.** Add the frame-
  discovery fixed point to `chain_promote` (walk promoted-body calls, test predicate against
  the L1 support-surface manifest, add clones to `F`, seed bodies, record reroutes). Wire the
  circuit breaker (`chain_closure_oversized`) and the narrowed `chain_closure_escapes`.
  **Gate:** predicate false-positive = STOP #K; must refuse before emission.

* **Subtask L3 — emission plumbing + e2e.** `Lnrat_B10` `VariantSpec` (reuses `return_widen`
  from Subtask 2a), test rewrites (inverted `Lnrat` assertions), then the B10/B12/B13 e2e
  re-run (seed 12345, 5000 samples, kernel-scope + positive-lift gate, STOP-and-report).
  **Success = B10 reaches the positive-lift gate with `Li2omx2_B10`+`ddilog_B10`+`Lnrat_B10`
  all dd and a measured `kernel_measured_lift ≥ +8`** (closure design §7 bar).

If L1 proves the port is not as bounded as §7 suggests (an oracle op has no vendored twin
and is itself unportable), **STOP at L1** and hand the scope call back — that would mean the
support surface is genuinely open-ended, not just un-vendored, and the design's central
claim would need revisiting.

---

## Appendix — invariants carried from the closure design

* Variants are **per-integral clones**; the shared original (`Lnrat`, `ddilog`, `Li2omx2`)
  is never edited. Rule (d) obeys this — it clones, never mutates the primary.
* Refusals (`clonable_leaf` fail, oversized) computed **before any tree mutation**.
* Gate logic **unchanged**; rule (d) only enlarges the (non-inert) candidate the gates
  measure. B10 reaching the lift gate is the point.
* Conservative-parser contract: false-negative (refuse a clonable leaf) safe; false-positive
  (clone an un-instantiable leaf) is STOP #K — guarded by the predicate + the L1 compile
  test + the §7 probe.
