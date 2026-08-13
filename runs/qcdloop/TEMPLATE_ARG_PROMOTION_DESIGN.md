# Template-Argument Promotion — Phase 1 (Correctness) — SCOPING

**Date:** 2026-07-28
**Branch:** langgraph-agents @ `acc1482`
**Dispatch:** Phase-1 template-argument promotion — scoping only (design, feasibility, blast-radius, validation plan)
**Status:** **DESIGN PROPOSED — AWAITING AUTHORIZATION.** No production code written. Landing is a separate dispatch.
**Reference proof:** `github.com/ReetBarik/qcdloop @ ddfun_enabled` (local checkout `/home/rbarik/qcdloop`, read via `git show ddfun_enabled:<path>` — working tree untouched)

---

## 0. Executive verdict

| gate | outcome |
|---|---|
| (a) feasibility — template-parametricity at every call site | **CONFIRMED** (no STOP #OO, no STOP #RR for B0m/B1m/B2m) |
| (d) rule-(d) subsumption | **SUBSUMED — 100% B10 (71/71), 100% B12 (27/27)** |
| STOP #OO (template surface not clean) | **not fired** |
| STOP #PP (ODR collision) | **FIRED — but resolvable by design** (see §1.4, §2). The two precision headers are mutually exclusive *by construction*; the resolution is a **per-integral whole-TU precision flip**, not a sibling call in one TU. |
| STOP #QQ (dispatch requires driver-source mutation) | **CONDITIONAL** — the pipeline already owns the driver TU (`boxGPU_dd.cpp` / `boxGPU_vanilla.cpp`) and the `QL_MODE=dd` build path, so no *user* driver mutation is required; but the mechanism is build-orchestration, not source-emission. See §2.2. |
| STOP #RR (Phase-3 inner parametricity gap) | **PARTIAL FLAG** — B0m/B1m/B2m subtrees clean at every inner call site; **B3m/B4m carry int↔Tracked conversion crossings** (`boxGPU.h:25-29`) that Phase 3 must handle separately. |
| STOP #Z (vendored snapshot pristine) | **clean** — read-only investigation, `third_party/` untouched |

**The one load-bearing discovery that reshapes the mechanism:** the dispatch as literally worded ("emit a sibling `BO<DoubleDoubleComplex,DoubleDouble,DoubleDouble>(...)` call *alongside* the `BO<complex<double>,double,double>(...)` call") **is not legal C++** in this codebase. `kokkosMaths.h` (double) and `kokkosMaths_dd.h` (dd) each define `ql::Constants<T>` as a **primary template**, a namespace-scope `using complex = …`, and primary `kAbs/kLog/kSqrt` templates. The two headers are **mutually exclusive within one translation unit** — `kokkosMaths_wrapper.h` selects exactly one via `USE_DD_COMPLEX`. Therefore `BO<dd,…>` and `BO<double,…>` cannot coexist in a single TU. The correct — and *already partially built* — mechanism is a **whole-TU precision flip per integral**, exactly the model the Validator's DD oracle (`boxGPU_dd.cpp`, `QL_MODE=dd`) already uses. This is good news: the emission surface shrinks from "synthesize dd shims into a double TU" (the source of every current boundary error) to "compile the flagged integral's TU at dd and narrow only at the app output boundary."

---

## 1. Deliverable (a) — Feasibility: template-parametricity at every depth

Checked against `runs/qcdloop_headers_full/` (the master snapshot the pipeline consumes), **not** ddfun_enabled. The ddfun_enabled branch is used only as the *proof of compilability*; the feasibility claim rests on the master snapshot.

### 1.1 The call graph is fully parametric at every internal call site

| frame | file:line | signature | type args flow through? |
|---|---|---|---|
| `BO` | `boxGPU.h:69` | `template<TOutput,TMass,TScale> void BO(View<TOutput*[3]>&, View<TScale*>&, View<TMass*[4]>&, View<TScale*[6]>&, int)` | **Y** — every View is over a type param; calls `B0m..B4m<T,T,T>`, `Constants<TOutput>`, `iszero<T,T,T>` |
| `B0m/B1m/B2m/B3m/B4m` | `box/B*m.h` | `template<TOutput,TMass,TScale> void Bxm(View<TOutput*[3]>&, Array<TMass,13>&, TScale&, int)` | **Y** — dispatch to BIN*/B6-B16<T,T,T> with args forwarded verbatim |
| `BIN1,B6,B7,B8,B9,B10,…` | `box/B1m.h:45-250` etc. | `template<TOutput,TMass,TScale> void B10(View<TOutput*[3]>&, Array<Array<TMass,4>,4>&, TScale&, int)` | **Y** — locals typed `TMass`/`TOutput`; leaves called `<TOutput,TMass,TScale>` |
| `Ycalc` | `box/box_common.h:18` | `template<TOutput,TMass,TScale> void Ycalc(Array<Array<TMass,4>,4>&, …)` | **Y** |
| `Lnrat` | `kokkosUtils.h:127,139` | `template<TOutput,TMass,TScale> TOutput Lnrat(TOutput&,TOutput&)` and `(TScale&,TScale&)` | **Y** |
| `ddilog` | `kokkosUtils.h:149` | `template<TOutput,TMass,TScale> TMass ddilog(TMass&)` | **Y** |
| `Li2omrat` | `kokkosUtils.h:619,630` | `template<TOutput,TMass,TScale> TOutput Li2omrat(…)` | **Y** |
| `Li2omx2` | `kokkosUtils.h:688,722` | `template<TOutput,TMass,TScale> TOutput Li2omx2(…)` | **Y** |
| `cLi2omx2`, `xspence`, `spence`, `kfn`, `kPow` | `kokkosUtils.h:606,655,…` `kokkosMaths.h:255` | all `template<TOutput,TMass,TScale>` | **Y** |
| `iszero` | `kokkosMaths.h:314` | `template<TOutput,TMass,TScale> bool iszero(TScale&)` | **Y** |

No frame in the chain hard-codes `double` or `complex<double>` in a way that blocks dd instantiation. Every internal call site propagates `<TOutput,TMass,TScale>`.

### 1.2 The scalar-leaf layer resolves by overload, and the dd overloads exist

`kAbs/kLog/kSqrt/kConj` are generic `template<typename T>` (dispatch to `Kokkos::abs/log/sqrt`). `Sign/Real/Imag/Max/Min/Htheta` are **non-template** overloads on `double` and `Kokkos::complex<double>` (`kokkosMaths.h:284-378`). Their dd counterparts — `kAbs/kLog/kSqrt` dd template specializations plus `Sign/Real/Imag/Max/Min/Htheta` overloads on `ql::ddfun::ddouble`/`ddcomplex` — are all present in `kokkosMaths_dd.h:273-399`. So when the TU is compiled with `USE_DD_COMPLEX`, overload resolution binds the dd forms.

### 1.3 The `Constants<T>` leaf has a dd specialization — but as a *rival primary*, not a partial specialization

`Constants<T>` is where the parametricity story gets subtle. **Both** headers define `template<typename T> struct Constants` as a **primary template**:

- `kokkosMaths.h:18` — `_C` = 19-term **double** Chebyshev, `_num_C()==19`, `_pi()` = `constexpr double` literal.
- `kokkosMaths_dd.h:32` — `_C` = 43-term **dd** Chebyshev (`ql::ddfun::make_dd(hi,lo)`), `_num_C()==43`, `_pi()` = `ql::ddfun::dd_pi()`.

These are the same fully-qualified name `ql::Constants<T>`. They can never appear in one TU (redefinition). This is the mechanism-defining fact of §1.4.

**Numerical corollary (why this matters beyond ODR):** if you *could* instantiate `BO<DoubleDoubleComplex,DoubleDouble,DoubleDouble>` against the *double* `kokkosMaths.h`, it would **compile** (every member is parametric on `T`) but silently return a 19-term-Chebyshev, `constexpr-double-π` result — i.e. ~16 significant digits wearing a dd type. The dd accuracy comes *specifically* from `kokkosMaths_dd.h`'s 43-term table and `DoubleDouble_pi()`. So the whole-TU flip is mandatory for **correctness**, not merely to dodge the ODR error.

### 1.4 STOP #OO verdict: NOT fired

The template surface is clean: every frame is parametric and every leaf either is parametric or has a dd overload/specialization available in `kokkosMaths_dd.h`. No source enrichment is required beyond what `e3d2e45` already vendored (`kokkosMaths_dd.h` + `third_party/include/dd_math.hpp`, `dd_complex.hpp`). The `ddfun_enabled` branch — whose box/util headers are byte-identical to the master snapshot for the leaf sources — is the standing proof that these exact templates compile and run at full dd.

### 1.5 Phase-3 free-byproduct check (STOP #RR)

Per the dispatch's instruction to check *every* inner call site (Phase 3 reuses this mechanism at inner sites):

- **B0m / B1m / B2m subtrees** — clean at every inner call site. Phase 3 can instantiate an inner `Lnrat<float,float,float>` inside an otherwise-double B10, or an inner `Lnrat<double,…>` inside an otherwise-dd chain, with no parametricity gap.
- **B3m / B4m subtrees (B16, BIN3, BIN4)** — `boxGPU.h:25-29` documents that these bodies "contain int↔Tracked crossings that Tracked deliberately lacks implicit conversions for," resolved by the C8 library patch at build time. This is a **type-conversion friction point** that a naive inner-site dd/float re-instantiation could re-expose. **STOP #RR flag raised for Phase 3 only** — it does not block Phase-1 whole-subtree promotion (the whole-TU flip compiles B3m/B4m at a single precision, so no *mixed* crossing forms; the C8 patch already handles the int↔type edges within one precision). Phase 3's *inner-site mixed* promotion of B16/BIN3/BIN4 will need a different mechanism for those sub-regions.

---

## 2. Deliverable (b) — Mechanism design

### 2.1 What the current pipeline does, and why it fails

The existing chain_promote/fanout path emits **dd-typed source into a `double` TU**: it clones leaves (rule d, `Lnrat_B10`), widens carriers, and inserts dd values through shims. Because the surrounding instantiation is `TOutput=Kokkos::complex<double>`, every dd value that reaches a caller-precision decl/store/return is a boundary error — this is the entire `instantiation_gate.py` Shape-1/2/3/4 taxonomy, and it is exactly the 71 (B10) / 27 (B12) errors that deliverable (d) shows are artifacts of *this embedding*, not of the math.

### 2.2 What template-argument promotion does instead

**Flip the whole translation unit for the flagged integral to dd**, exactly as `boxGPU_dd.cpp` already does for the oracle, then narrow only at the app output boundary. Concretely:

1. **Detection (chain_promote).** A region flagged by characterization for dd, whose enclosing subtree is fully template-parametric (the (a) check, evaluated per-integral), is routed to **template-arg promotion** rather than element/rule-d promotion. Phase-1 default: *any* dd-flagged region in a parametric subtree → whole-subtree template-arg. Element promotion (acc1482) becomes the reserved path for regions **not** inside a template-parametric subtree.

2. **Emission = build-mode selection, not source rewrite.** The pipeline already has the two ingredients:
   - `runs/qcdloop/src/boxGPU_dd.cpp` — `#define USE_DD_COMPLEX; run_app<DoubleDoubleComplex,DoubleDouble,DoubleDouble,DDPrinter>(...)`, includes `boxGPU.h` which via `kokkosMaths_wrapper.h` pulls `kokkosMaths_dd.h`.
   - `runs/qcdloop/app/CMakeLists.txt` — `QL_MODE=dd` selects that driver; `QL_HEADERS=<ddfun_enabled/src/qcdloop>` supplies the dd-capable headers.

   Phase-1 promotion of integral `X` = **build `X`'s TU at `QL_MODE=dd`** (against the promoted/master dd-capable tree) and route the caller's dispatch for `X` to that binary. No `BO<dd>` call is spliced next to a `BO<double>` call; the *whole* `BO` instantiation for `X` is dd because its TU selected the dd headers.

   **Two candidate granularities** (design decision below):
   - **(2a) whole-app dd for the promoted set** — one TU, all promoted integrals at dd, all others at double is *impossible in one TU* (the ODR fact). So a mixed app = **multiple TUs**, one per precision class, linked together, with the top-level dispatch selecting the right symbol per integral.
   - **(2b) per-integral TU** — each integral gets its own TU compiled at its own precision; a thin dispatch layer calls the right one. This is the natural extension of the existing `QL_MODE` split.

   **Chosen: (2b) per-integral TU.** Justification: (i) it is the *only* ODR-legal way to have B10@dd and B1@double in one program; (ii) it is a direct generalization of the `QL_MODE=vanilla|dd` split the harness already drives (one configure per binary, `validator/runner.py:68`); (iii) it keeps the instantiation-gate validation trivial — each TU is a clean single-precision compile, so "does B10 build at dd?" is `QL_MODE=dd` restricted to B10's recipe, no shim classification needed; (iv) it makes each integral's precision independently selectable, which is exactly what Phase 2 (endpoint-lock) and Phase 3 (intra-integral) will need to vary.

3. **Where the dd output lands.** The promoted TU produces `res` as `View<DoubleDoubleComplex*[3]>`. The caller-facing contract is `View<complex<double>*[3]>`. **Land the dd output via a narrowing at the app boundary** — the point where `res_dd(i,k)` is read back for the caller — using acc1482's designed-exit transform:
   - `demote_exit_carriers_line` (deliverable b of acc1482) demotes the dd read at the projection site to caller precision (`complex<double>(double(v.real().hi + …), …)` component reconstruction).
   - `widen_carrier_assign_line` (deliverable c) handles the mirror case where a caller-precision value must feed a dd carrier at the entry boundary.

   This is the **single wiring decision that keeps Phase 3 additive**: the dd→caller narrowing is the *same* boundary transform whether the boundary is the whole-app output (Phase 1) or an inner demoted call site (Phase 3). **Do not emit a one-off narrowing.** Phase 1 threads the flip's output boundary through `boundary.demote_exit_carriers_line`/`widen_carrier_assign_line` so Phase 3 inner-site demotions reuse the identical code path with a different boundary location.

4. **Interaction with rule (d).** For Phase-1 correctness, template-arg promotion **replaces** rule (d) for every case where the subtree is fully template-parametric — which deliverable (d) shows is **100%** of B10's and B12's rule-d error classes. Rule-(d) clone emission is **not invoked** on the correctness path. The machinery is **retained** (`fanout.py` clone emission, `clonable_leaf.py`, `closure_decls`) — Phase 3 reactivates it in *demotion* mode (clone a callee, narrow args, widen return, reroute a *specific* inner call site) when inner-site template-arg instantiation hits granularity limits.

5. **Interaction with acc1482 element promotion.** Element promotion remains correct and is **used for non-template-parametric contexts** (a region whose enclosing subtree is *not* fully parametric — e.g. a hand-written kernel with baked-in `double`). It is no longer the primary correctness mechanism for the box tree, but it is not superseded: it is the fallback the detection step (2.2.1) routes to.

### 2.3 Emission-simplicity + gate-validation argument for (2b)

Under (2b) the instantiation gate becomes *structurally* a non-issue for the correctness path: a per-integral dd TU is a clean single-precision compile identical in shape to the existing `boxGPU_dd.cpp` oracle build, which already compiles and runs for all 21 integrals. The Shape-1/2/3/4 boundary taxonomy (which exists only because dd is embedded in a double TU) has **no surface to appear on** — there is no dd/double mixing *inside* the TU; the only dd↔double crossing is the app output boundary, owned by one reused transform. `instantiation_gate.py` stays in the tree as the regression detector for the *element-promotion* path and for any residual non-parametric context, but the template-arg path routes around its failure modes by construction.

---

## 3. Deliverable (c) — Blast radius (21-integral sweep)

Baseline gate outcomes for {B10,B12,B13,B14} are from measured Tier-B artifacts under `runs/qcdloop/tier_b_stage2_leaf_promotion/`; B15/B16 co-built (not independently gated); the other 15 have no dd-gate artifact and carry only static conditioning floors (`runs/qcdloop/bound_decomposition_all_21.json`). Mass-group map confirmed from `boxGPU.h:12-16`; all five group headers verified `template<TOutput,TMass,TScale>`.

| integral | mass-group | baseline gate outcome | current promotion mechanism | subtree parametric? | predicted Phase-1 touch |
|---|---|---|---|---|---|
| B1  | B0m | unmeasured; floor 12.16, STABLE_ALREADY | none | Y | unchanged |
| B2  | B0m | unmeasured; floor 11.5, STABLE_ALREADY | none | Y | unchanged |
| B3  | B0m | unmeasured; floor 10.32, STABLE_ALREADY | none | Y | unchanged |
| B4  | B0m | unmeasured; floor 11.5, STABLE_ALREADY | none | Y | unchanged |
| B5  | B0m | unmeasured; floor 10.39, STABLE_ALREADY | none | Y | unchanged |
| B6  | B1m | unmeasured; floor 13.24, STABLE_ALREADY | none | Y | unchanged |
| B7  | B1m | unmeasured; floor 9.72, STABLE_ALREADY | none | Y | unchanged |
| B8  | B1m | unmeasured; floor 10.14, STABLE_ALREADY | none | Y | unchanged |
| B9  | B1m | unmeasured; floor 11.53, STABLE_ALREADY | none | Y | unchanged |
| **B10** | B1m | **`apply_failed`/build_failed** (measured) | out-of-scope region-core (`res(i,1)` View + `Constants<TOutput>` returns) + 71 rule-d clones | Y | **promoted-via-template-arg** |
| B11 | B2m | unmeasured; floor 10.09, STABLE_ALREADY | none | Y | unchanged |
| **B12** | B2m | **`apply_failed`/build_failed** (byte-identical pre/post landing) | rule-d `Lnrat` leaf + own `complex<DoubleDoubleComplex>` | Y | **promoted-via-template-arg** |
| **B13** | B2m | **`apply_failed`/`write_truncation`** (built OK, gate reject) | closure/chain promotion | Y | **promoted-via-template-arg** (uniform interior carrier retype removes truncation) |
| B14 | B2m | `rejected`(`chain_no_lift`), clean build, lift 0.0, pred +16.66 | element (fixed-size aggregate) + chain — works | Y | unchanged (already-accurate; STOP #A) |
| B15 | B2m | co-built clean in B14 run; floor 0.0, dd-INSUFFICIENT | element (shares B14 B2m chain) | Y | **at-risk** (dd-insufficient: cancellation > dd budget) |
| B16 | B3m | co-built clean in B14 run; floor 0.0, dd-INSUFFICIENT | element (shares B3m chain) | Y* | **at-risk** (dd-insufficient + B3m int↔Tracked friction) |
| BIN0 | B0m | unmeasured; floor 0.0, dd-INSUFFICIENT | none | Y | at-risk (dd-insufficient) |
| BIN1 | B1m | unmeasured; floor 0.0, dd-INSUFFICIENT | none | Y | at-risk (dd-insufficient) |
| BIN2 | B2m | unmeasured; floor 0.0, dd-INSUFFICIENT | none | Y | at-risk (dd-insufficient) |
| BIN3 | B3m | unmeasured; floor 0.0, dd-INSUFFICIENT | none | Y* | at-risk (dd-insufficient + B3m friction) |
| BIN4 | B4m | unmeasured; floor 0.0, dd-INSUFFICIENT | none | Y* | at-risk (dd-insufficient + B4m friction) |

`Y*` = header is parametric but the B3m/B4m subtree carries int↔Tracked crossings (`boxGPU.h:25-29`) — a Phase-3 concern (STOP #RR), not a Phase-1 blocker (single-precision TU compiles fine).

**Go/no-go read:** the 3 measured build-failures (B10, B12, B13) are exactly the integrals template-arg promotion is designed to fix. B14 is already-accurate (nothing to lift). The 7 "at-risk" are **dd-INSUFFICIENT** — a clean dd build for them would be a *false-positive fix* (cancellation loss exceeds dd's ~32-digit budget; they need quad/rewrite). The mechanism must **not** claim these as wins: a clean build ≠ a lift. B1-B9/B11 are unchanged (STABLE_ALREADY, no dd need).

---

## 4. Deliverable (d) — Rule-(d) subsumption analysis

**Verdict: SUBSUMED — 100% for both B10 and B12.** (Full evidence below; cross-referenced against build logs + ddfun_enabled leaf call sites.)

### 4.1 B10 — 71/71 dissolve

The 71 clone errors (in the emitted `_B10` clone bodies of `kokkosUtils.h`) partition into 8 classes, every one a **caller/callee mixed-precision boundary** created by rule (d) embedding a dd body inside a `TOutput=complex<double>, TMass=TScale=double` instantiation:

| # | count | signature | dissolves at dd caller because |
|---|---:|---|---|
| C1 | 22 | `complex<double>::complex(DoubleDouble)` | `complex<double>` is `TOutput` → becomes `DoubleDoubleComplex`; `DoubleDoubleComplex(DoubleDouble)` well-formed |
| C2 | 16 | `invalid cast DoubleDouble → double` | `double` is `TScale`/`TMass` → becomes `DoubleDouble`; dd→dd identity |
| C3 | 8 | `cannot convert DoubleDouble → const double init` | as C2 |
| C4 | 6 | `complex<double>::complex(DoubleDoubleComplex)` | as C1 |
| C5 | 6 | `operator= DoubleDoubleComplex = complex<DoubleDoubleComplex>` (Shape-3) | `complex<DoubleDoubleComplex>` forms only under *partial* promotion (dd local × `cxs[k]` caller-complex read); uniform dd makes `cxs` a `DoubleDoubleComplex` array → never forms |
| C6 | 6 | `operator= DoubleDoubleComplex = complex<double>` | `complex<double>` is `TOutput` → `DoubleDoubleComplex` |
| C7 | 6 | `const double& ← DoubleDouble` | `TScale` → `DoubleDouble` |
| C8 | 1 | `complex<double> → DoubleDoubleComplex` | `TOutput` → `DoubleDoubleComplex` |

Sum = 71. **No residual, no source-enrichment gap** — the leaf templates exist unmodified in `runs/qcdloop_headers_full/kokkosUtils.h` (byte-identical to ddfun_enabled), dd primitives vendored in `third_party/include/`. The one clone-related item *outside* the 71 (Shape-4 `ddilog(DoubleDouble)` shim) also dissolves: `ddilog<DoubleDoubleComplex,DoubleDouble,DoubleDouble>` instantiates straight from source, no forwarding shim.

### 4.2 B12 — 27/27 dissolve

| count | signature | disposition |
|---:|---|---|
| 12 | `wrong number of template arguments (3, should be 2)` | **Removed, not fixed.** Caused by the rule-d `ql_shim_dd.h` forwarding overload `Lnrat(const DoubleDouble&, const DoubleDouble&)` (2 params) that the 3-arg site can't bind. `ql_shim_dd.h` does not exist in ddfun_enabled — it is a rule-d artifact. Template-arg promotion eliminates its cause. |
| 12 | `Lnrat<complex<double>,double,double>(DoubleDouble&, DoubleDouble&)` no match | dissolves — at dd the TScale overload `Lnrat<DoubleDoubleComplex,DoubleDouble,DoubleDouble>(DoubleDouble,DoubleDouble)` matches |
| 2 | `complex<DoubleDoubleComplex> → const DoubleDoubleComplex` | dissolves — same partial-promotion artifact as B10 C5 |
| 1 | `static_assert Kokkos::complex floating point` | dissolves — downstream of the `complex<DoubleDoubleComplex>` above |

**No residual.** `boxGPU_test_dd_B12.cc:196` (`BO<DoubleDoubleComplex,DoubleDouble,DoubleDouble>`) is the standing proof the B12 chain compiles + runs at dd.

### 4.3 Consequence

Both integrals clear the ≥90% threshold at 100%. **Rule (d) is confirmed subsumed for Phase-1 correctness.** It remains in the tree for Phase-3 demotion mode. The 12 "wrong-arg-count" B12 errors are notable: template-arg promotion doesn't *fix* them, it *removes their cause* (the forwarding shim), which is strictly cleaner.

---

## 5. Deliverable (e) — Validation plan

### 5.1 Minimum validation (pre-implementation, hand-constructed PoC)
Prove the ddfun_enabled mechanism compiles under *the current pipeline's toolchain* (module chain: `gcc/13.3.0`, `cmake/3.28.3`, Kokkos), not just under Reet's original build:
1. Configure the app CMake with `QL_MODE=dd`, `QL_HEADERS=<ddfun_enabled/src/qcdloop>` (or the master snapshot once `kokkosMaths_wrapper.h` is present in it — see §5.5).
2. Build `boxGPU_app` (= `boxGPU_dd.cpp`) and run it restricted to B10, then B12.
3. **Expected:** clean build (no Shape-1/2/3/4 errors), non-NaN dd output for B10 and B12 — the first honest dd builds for both. This is a *build-orchestration* PoC; it writes no pipeline code.

### 5.2 21-integral gate sweep (post-implementation)
Run each integral's per-integral dd TU through the build gate. **Expected:** all 21 build clean at dd (the oracle already does). Regression check = no integral that built clean before now fails.

### 5.3 L-measure prediction
- **B10 lift target restored to +18.43** (the original static-tightness prediction) — because the promoted tree now instantiates on the *live* dd path (992e209 dispatch fix stands) and the whole chain is honestly dd, not dead code and not a partial embedding.
- **B12** — honest dd baseline measurable for the first time (was build_failed pre & post acc1482).
- **Caveat (from §3):** B15/B16/BIN* are dd-INSUFFICIENT; a clean dd build there is **not** a lift. L-measure must gate on measured digit-lift, not build success, for those.

### 5.4 Regression coverage (must stay green)
- acc1482 element-promotion suite: `tests/integrator_base/test_region_core_element_promotion.py` (26 tests).
- Full patcher/integrator/shared suites (520 tests).
- B14/B15/B16 clean-build discipline: B14 stays `chain_no_lift` clean (or upgrades to template-arg with identical output); B15/B16 stay co-built clean. **Byte-identity requirement:** integrals not routed to template-arg promotion must emit byte-identically to acc1482 (the strict no-op guards on the element/reconcile paths must remain inert when template-arg owns the integral).

### 5.5 Prerequisite check surfaced by feasibility (source enrichment, if any)
The master snapshot `runs/qcdloop_headers_full/kokkosMaths_wrapper.h` currently only branches `USE_QUAD_COMPLEX` vs `kokkosMaths.h` — it does **not** have the `USE_DD_COMPLEX → kokkosMaths_dd.h` arm that ddfun_enabled's wrapper has. `kokkosMaths_dd.h` **is** vendored in the snapshot (per `README.md`), but the wrapper does not select it. **This is the one enrichment Phase-1 landing needs**: either (i) build the promoted integrals against a `ddfun_enabled/src/qcdloop` tree (as the oracle already does — no snapshot change), or (ii) add the `USE_DD_COMPLEX` arm to the snapshot's `kokkosMaths_wrapper.h` (a human-authored commit analogous to `e3d2e45`, **not** a pipeline mutation, per the non-goals). Recommend (i) for Phase-1 (reuses the oracle's proven tree); (ii) only if the pipeline must self-host the dd headers. **This is not STOP #OO** (the templates are clean); it is a build-wiring choice for the landing dispatch.

---

## 6. Deliverable (f) — Ordered landing plan (if authorized)

1. **Detection + routing (chain_promote).** Add the template-parametric-subtree check to region routing; dd-flagged regions in parametric subtrees route to template-arg promotion, else to element promotion. No source emission yet.
2. **Per-integral dd TU emission (fanout + CMake).** Generalize the `QL_MODE=vanilla|dd` split to per-integral TUs: emit/select the dd TU for each promoted integral; thin dispatch layer routes each integral to its precision-specific symbol. Reuse `boxGPU_dd.cpp`'s recipe (it already parameterizes on `<TOutput,TMass,TScale,Printer>`).
3. **Designed-exit narrowing at the app output boundary.** Wire the dd→caller projection through `boundary.demote_exit_carriers_line` / `widen_carrier_assign_line` (acc1482) — **reused, not re-implemented**. This is the Phase-3-enabling wiring decision.
4. **Gate-validate B10 build clean at dd** — first honest B10 dd build.
5. **Gate-validate B12 build clean at dd.**
6. **21-integral gate sweep** for regression; confirm byte-identity for non-promoted integrals.
7. **L-measure re-run** — honest B10 lift (target +18.43) + B12 baseline. B14/B15/B16 stay byte-identical to acc1482 baseline unless the mechanism-selection rule (step 1) prefers template-arg for them; if so, output must match. **Do not** claim dd-INSUFFICIENT integrals (B15/B16/BIN*) as lifts.

---

## 7. STOP audit

| STOP | condition | status |
|---|---|---|
| #OO | template surface not clean | **not fired** — every frame parametric, every leaf has dd overload/spec available |
| #PP | ODR collision (dd spec collides with `complex<double>` spec) | **fired-and-resolved** — the two `Constants<T>`/`using complex`/`kAbs` primaries are mutually exclusive per TU *by construction*; resolved by the per-integral whole-TU flip (§2.2b), which never puts both in one TU. Not a blocker; it *is* the mechanism constraint. |
| #QQ | dispatch requires user-driver-source mutation | **not fired for Phase 1** — the pipeline owns the driver TU (`boxGPU_dd.cpp`) and the `QL_MODE=dd` build path; no user source is touched. (992e209 dispatch fix stands.) |
| #RR | Phase-3 inner-parametricity gap | **flagged** — B0m/B1m/B2m clean; **B3m/B4m int↔Tracked crossings** (B16/BIN3/BIN4) need a separate Phase-3 mechanism. Does not block Phase 1. |
| #Z | vendored snapshot pristine | **clean** — read-only; `third_party/` untouched |

---

## 8. Handbacks / decisions for Reet

1. **Mechanism reframe (the big one).** The dispatch's "sibling `BO<dd>` call alongside `BO<double>` in the driver" is not ODR-legal here — the two precision headers are mutually exclusive per TU, and the double `Constants<T>` would also be *numerically wrong* at dd (19-term vs 43-term). The correct mechanism is the **per-integral whole-TU precision flip** you already built for the DD oracle (`boxGPU_dd.cpp`, `QL_MODE=dd`). Phase 1 = build the flagged integral's TU at dd + narrow at the app output boundary (reusing acc1482). **Confirm this reframe before the implementation dispatch.**
2. **Build-wiring choice (§5.5).** Phase-1 landing needs the dd headers include-reachable. Recommend building promoted integrals against the `ddfun_enabled/src/qcdloop` tree (as the oracle does — zero snapshot change). Alternative: add the `USE_DD_COMPLEX` arm to the master snapshot's `kokkosMaths_wrapper.h` (human-authored, like `e3d2e45`). Your call.
3. **dd-INSUFFICIENT false-positive guard.** B15/B16/BIN0-4 will *build* clean at dd but won't *lift* (cancellation > dd budget). The L-measure gate must reject "clean build, no lift" for these, not accept them. Confirm the acceptance instrument distinguishes build-success from digit-lift (this is the same STOP #A gate-instrument question from B14).
4. **Rule (d) 100% subsumed** for B10/B12 correctness — machinery retained for Phase-3 demotion. No action needed; noted for the record.

---

## 9. Non-goals honored
No production code. No Phase-2/Phase-3 work. No deletion of rule-(d)/closure/carrier-reconcile machinery. No dispatch-selector changes beyond reaching the dd instantiation (992e209 stands). No modification to `runs/qcdloop_headers_full/` (any enrichment is a separate human commit). No coefficient synthesis (§3.4 falsifier trap remains banned).
