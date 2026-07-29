# Phase-2 Float Downshift — Pipeline-Authored Shim Synthesis (SCOPING) — 2026-07-29

Scoping-only subtask following STOP #XX (`PHASE_2_STOP_XX_2026-07-29.md`, @9ff9ba8).
Inventory + feasibility + design for a **pipeline-generated float leaf shim** that lets the
Phase-2 double→float downshift proceed with **zero source enrichment**. No production code
here; landing is a separate authorized dispatch.

**Reet's reframe (load-bearing):** instead of authoring a static `kokkosMaths_float.h`, the
pipeline synthesizes the missing float leaf overloads itself, per-integral-TU, as a
patcher-generated shim header included alongside the double reference `kokkosMaths.h`. This
works *only* because the target precision (IEEE float) has **library-native** support
(`Kokkos::abs`/`log`/`sqrt`/`conj` on `Kokkos::complex<float>`); it is a strict subset of
"downshift to any library-supported precision," and does **not** generalize to ff (§8).

## 0. Executive verdict

| item | outcome |
|---|---|
| Leaf inventory (deliv. 1) | **13 non-template leaves** in `kokkosMaths.h`; **all** are library-pass-through / member-access / scalar-ternary bodies — **zero hand-authored math** → **no STOP #BBB predicted** |
| Constants<T> | **NOT a blocker at float** (primary template narrows from double cleanly) — confirmed, not enumerated as a synthesis concern |
| kokkosUtils.h helpers (Lnrat/Li2*/xspence/L0/L1/kfn/iszero/Ycalc/kPow/kLog/kSqrt/kConj) | **all templates** → auto-instantiate at float, **no shim needed** |
| Path (b) native-float shim | **FEASIBLE for all leaves** called by B1–B9, B11 — proven by compile probe |
| **STOP #CCC** (Kokkos lacks native complex<float>) | **DID NOT FIRE** — probe: B0m/B1m/B2m groups all compile **clean** at float; binaries run; float output bit-differs from double (genuine float compute) |
| **STOP #DDD** (shim ODR / double-shadowing) | **DID NOT FIRE** — probe: a double TU with the float shim present compiles clean; strictly-float signatures do not shadow the double overloads |
| **STOP #Z** (snapshot pristine) | **clean** — all probes in `/tmp` clones; `runs/qcdloop_headers_full/`, `third_party/`, `runs/qcdloop/src/` untouched |
| Group interleaving | every Phase-2 group also holds a Phase-1 dd candidate — **already solved** by `flip_dispatch.py`'s per-integral RES-stream selection (no STOP #ZZ) |

**Bottom line:** path (b) is **viable for every leaf reachable from B1–B9, B11**, empirically
proven end-to-end (compile + run + genuine-float-compute + no-ODR). Per this subtask's
verdict gate, this scopes to **"Reet authorizes an implementation dispatch; Phase 2 lands as
pipeline-authored shim synthesis, zero source enrichment."** No STOP #BBB leaf was found. The
only per-integral nuance is **BIN0** (a massless *finite* box), which is grouped with the
Phase-2 targets but is itself dd-relevant — routing detail in §6.

---

## 1. Leaf function inventory (deliverable 1)

Every `ql::` symbol reachable from `kokkosMaths.h` (the double reference header) +
`kokkosUtils.h` (included transitively via `box_common.h`). Source lines are
`runs/qcdloop_headers_full/kokkosMaths.h` unless noted.

### 1.1 Non-template overloads on double / complex<double> — **need a synthesized float sibling**

| leaf | line | double body | body kind | path (b) synthesis |
|---|---|---|---|---|
| `kAbs(double)` | 286 | `return Kokkos::abs(x);` | library pass-through | `kAbs(float){ return Kokkos::abs(x); }` |
| `kAbs(Kokkos::complex<double>)` | 292 | `return Kokkos::abs(x);` (returns real) | library pass-through | `kAbs(complex<float>)→float{ return Kokkos::abs(x); }` |
| `Imag(double)` | 319 | `return 0.0;` | scalar literal | `Imag(float)→float{ return 0.0f; }` |
| `Imag(Kokkos::complex<double>)` | 323 | `return x.imag();` | member access | `Imag(complex<float>)→float{ return x.imag(); }` |
| `Real(double)` | 327 | `return x;` | identity | `Real(float)→float{ return x; }` |
| `Real(Kokkos::complex<double>)` | 331 | `return x.real();` | member access | `Real(complex<float>)→float{ return x.real(); }` |
| `Sign(double)` | 335 | `(0<x)-(x<0)` | scalar ternary | `Sign(float)→int{ (0<x)-(x<0) }` |
| `Sign(Kokkos::complex<double>)` | 339 | `return x / ql::kAbs(x);` | library + arith | `Sign(complex<float>)→complex<float>{ x/ql::kAbs(x) }` |
| `Max(double,double)` | 344 | `kAbs(a)>kAbs(b)?a:b` | scalar ternary | `Max(float,float)` identical modulo type |
| `Max(Kokkos::complex<double>,…)` | 352 | `kAbs(a)>kAbs(b)?a:b` | scalar ternary | `Max(complex<float>,…)` identical modulo type |
| `Min(double,double)` | 360 | `kAbs(a)>kAbs(b)?b:a` | scalar ternary | `Min(float,float)` identical modulo type |
| `Min(Kokkos::complex<double>,…)` | 368 | `kAbs(a)>kAbs(b)?b:a` | scalar ternary | `Min(complex<float>,…)` identical modulo type |
| `Htheta(double)` | 375 | `0.5*(1+Sign(x))` | scalar arith | `Htheta(float)→float{ 0.5f*(1+Sign(x)) }` |

**13 leaves, all synthesizable via path (b).** Every body is (a) a library call whose float
instantiation exists (`Kokkos::abs`), (b) a member/identity access valid on `complex<float>`,
or (c) a scalar ternary/arith identical modulo the scalar type. **None is hand-authored
non-trivial math → no STOP #BBB.**

### 1.2 Generic templates — instantiate at float automatically (no shim)

| leaf | line | note |
|---|---|---|
| `kAbs<T>` | 280 | generic; the double/complex overloads above are the *specialized* ones — the generic covers other T at float |
| `kLog<T>` | 298 | `Kokkos::log(x)` — float instantiation library-native (probe-confirmed) |
| `kSqrt<T>` | 304 | `Kokkos::sqrt(x)` — float instantiation library-native |
| `kConj<T>` | 310 | `Kokkos::conj(x)` — float instantiation library-native |
| `kPow<TOutput,…>` | 256/266 | template; pure `*=` loop |
| `iszero<TOutput,…>` | 315 | template |
| `Constants<T>` (all members: `_pi`/`_zero`/`_one`/`_two`/`_half`/`_C`/`_B`/`_eps*`/`_ieps50`/`_pi2*`/…) | 18–252 | **primary template with a body**; `Constants<float>` narrows the double literals/tables cleanly. **NOT a blocker** (contra the dd `_C[43]` concern, which was about *accuracy above double* — narrowing double→float is correct, merely over-computed). |

`kokkosUtils.h` helpers — **all 42 template heads**, every one `template<typename TOutput,
typename TMass, typename TScale>` or `template<typename TMass>`: `Lnrat` (both overloads),
`ddilog`, `denspence`, `ltspence`, `cspence`, `xspence`, `Li2omrat` (both), `cLi2omx2`,
`Li2omx2` (both), `L0`, `L1`, `kfn`, plus `Ycalc` (box_common.h). **All instantiate at float
automatically; none needs a shim.** (Sole non-template in kokkosUtils.h is the debug
`printDoubleBits(double)`, unreachable from the box compute path.)

### 1.3 Pass-through to scalar builtin needing no synthesis

None separate from §1.2 — the `Kokkos::abs/log/sqrt/conj` calls are reached through the
§1.1 float siblings (for the specialized complex/real overloads) or the §1.2 generic
templates.

---

## 2. Per-integral leaf usage (deliverable 2)

Direct-body counts of the §1.1 non-template leaves per integral (grep of each integral's
body), plus the **transitive** leaves every integral inherits through the shared machinery:

- **`BO` wrapper** (per group, `#ifndef QCDLOOP_BOX_FULL_DISPATCH`): calls `Max` + `kAbs`
  (scalefac reduction) — so **every** integral transitively needs `Max(complex?)`/`kAbs`.
  (In practice the `BO` scalefac operates on `TScale`=float, so `Max(float)`/`kAbs(float)`.)
- **`Lnrat` / `Li2omrat` / `Li2omx2` / `xspence`** (templates) internally call
  `kLog`/`kAbs`/`Real`/`Imag`/`Sign` — so any integral using these inherits the full
  real/complex leaf set at float.

| integral | direct non-template leaves (body) | transitive (via BO + Lnrat/Li2*) | net shim need |
|---|---|---|---|
| **B1** | — | Lnrat→{kLog,kAbs,Real,Imag,Sign}; BO→{Max,kAbs} | full leaf set |
| **B2** | — | Lnrat, Li2omrat, BO | full leaf set |
| **B3** | kAbs·1, Real·4, Sign·4 | Lnrat, Li2omrat, Li2omx2, L0, L1, BO | full leaf set |
| **B4** | — | Lnrat, Li2omrat, BO | full leaf set |
| **B5** | kAbs·1, Real·4, Sign·4 | Lnrat, Li2omrat, Li2omx2, L0, L1, BO | full leaf set |
| **B6** | — | Lnrat, BO | full leaf set |
| **B7** | — | Lnrat, Li2omrat, BO | full leaf set |
| **B8** | — | Lnrat, Li2omrat, Li2omx2, kPow, BO | full leaf set |
| **B9** | — | Lnrat, Li2omx2, Li2omrat, kSqrt, BO | full leaf set |
| **B11** | Real·4, Imag·1 | (B2m group machinery) | full leaf set |

**Conclusion:** all 10 Phase-2 targets need the **same** full float leaf set, and every leaf
is path-(b)-synthesizable (§1.1). There is no integral that needs a leaf outside the
synthesizable 13. Predicted per-integral outcome for **all of B1–B9, B11:
`downshiftable-via-path-b`** (no path-(a) fallback needed, no hand-authored-leaf block).

*Note (per the dispatch): `Constants<T>` touches are pervasive and fine at float — not
reported as a synthesis concern.*

---

## 3. Path (b) mechanism design — the synthesized-shim generator (deliverable 3)

### 3.1 Shape

A new patcher module (proposed `agents/patcher/shim_synth.py`) renders a header
`kokkosMaths_float_shim.hpp` into the **cloned** per-integral TU tree (never the snapshot),
included by the flip driver's wrapper **after** the double reference `kokkosMaths.h`:

```
// clone kokkosMaths_wrapper.h  (generated into the CLONE only — STOP #Z guard reused)
#pragma once
#include "kokkosMaths.h"              // the double REFERENCE header (unchanged)
#include "kokkosMaths_float_shim.hpp" // pipeline-synthesized float leaf siblings
```

For each §1.1 non-template leaf, the generator emits its float sibling by **binding to the
library-native float instantiation**, deriving the body class from the reference:

- library pass-through (`kAbs`, and the `kLog/kSqrt/kConj` complex specializations if a
  precision ever needs them): `return Kokkos::abs(x);` etc.
- member/identity (`Real`, `Imag`): `return x.real();` / `return x;`
- scalar ternary/arith (`Sign`, `Max`, `Min`, `Htheta`): body identical modulo the scalar
  type token.

The generator's **input** is the leaf inventory of §1.1 extracted structurally from the
reference header (parse the non-template `ql::` overloads whose signatures name the reference
scalar/complex type), **not** a baked-in leaf list — so if the reference header gains/loses a
non-template leaf, the shim tracks it. The **output** signatures are strictly float-typed
(`float` / `Kokkos::complex<float>`), which is what makes STOP #DDD safe (§7).

### 3.2 Where it hooks into `tu_emit.py`

Today `PrecisionProfile` carries `maths_header` (the single header the wrapper's arm
`#include`s) and `available` (vendored?). Extend the profile with two fields (design, not
code):

- `maths_reference_header: str` — the **double** reference header the shim is built on
  (`kokkosMaths.h`).
- `shim_synthesis: bool` — when `True`, the profile is served by **generating** a shim
  alongside `maths_reference_header` rather than selecting a static `maths_header`. This is
  the new trigger: *"the target precision has no static wrapper header, but library-native
  instantiations exist for the referenced leaves."*

`render_wrapper` gains a branch: for a `shim_synthesis` profile, emit the two-line wrapper
(reference + shim include) instead of the `#if defined(macro)` ladder arm. `emit_flip_tu`
gains a step: after writing the wrapper, call `shim_synth.render(reference_header,
leaf_inventory)` and write `kokkosMaths_float_shim.hpp` into the clone (reusing the existing
`_refuse_snapshot` guard — STOP #Z). The `FLOAT` profile flips to
`shim_synthesis=True, maths_reference_header="kokkosMaths.h", available=True`; `printer_name`,
`cpp_output=Kokkos::complex<float>`, `cpp_scalar=float`, `two_limb=False` are already correct
from Phase-1's declared-but-unavailable float profile. **No new emission code path for the
driver itself** — `render_group_driver` is unchanged (it already rendered a correct float
driver in the STOP #XX probe); only the wrapper/shim step is new. This keeps the STOP #YY
guarantee: precision-parameterized, not float-hardcoded.

### 3.3 Precision-parameterization (STOP #SS discipline)

The generator is **not** float-hardcoded. Its parameters are (target scalar type, target
complex type, reference header). The trigger is a profile property (`shim_synthesis`), not a
precision literal. Any future precision with library-native leaf instantiations (e.g. `half`,
`bfloat16`, where Kokkos/library support exists) selects the same generator by setting
`shim_synthesis=True` + its scalar/complex tokens. Precisions **without** library-native
support (ff, dd) leave `shim_synthesis=False` and require a static wrapper header (enrichment)
— the fail-loud path is preserved.

### 3.4 Invalidation / regeneration

The shim is derived from the reference header's non-template leaf set. Regeneration key:
**sha256 of the extracted leaf inventory** (the sorted list of `{name, signature, body-kind}`
tuples parsed from `kokkosMaths.h`), written as a first-line comment in the generated shim:

```
// @shim-inventory-sha256: <hex>  reference=kokkosMaths.h precision=float
```

On each TU emit, recompute the inventory sha from the current reference header; if it differs
from the shim's embedded sha (or the shim is absent), regenerate. This invalidates correctly
when the double reference header changes a leaf signature/body, and is a cheap no-op when it
has not. (Mirrors the existing shim-hash discipline in the fanout path.)

---

## 4. Path (a) fallback — promote-and-narrow (deliverable 4)

Path (a): a float leaf body that **calls the double leaf on promoted args and narrows the
return** — e.g. `Sign(complex<float> x){ return complex<float>(ql::Sign(complex<double>(x))); }`.

- **Correctness:** preserved (double result narrowed to float).
- **Speedup:** **lost** for that leaf (double compute + two narrowings > native float).
- **When it matters:** if a single leaf falls to (a) and the integral's compute is dominated
  by that leaf, the integral's *whole* downshift yields no speedup → Phase 2 should
  **reject** that integral's downshift (keep it at raw double) rather than ship a slower-float
  no-win. Accept path (a) only when the (a)-leaf is not compute-dominant.

**Recommendation (matches the dispatch's default): do NOT emit path (a) automatically.**
Default to **STOP #BBB — hand back per-leaf.** The generator emits path (a) for a specific
leaf **only** when Reet explicitly authorizes it for that leaf (a per-leaf allow-list
parameter on the generator, off by default). This keeps "no silent no-speedup degrade" — a
path-(a) leaf is a deliberate, logged trade, never an invisible fallback. **For B1–B9, B11 no
path (a) is needed at all** (§2): every leaf is path-(b)-native.

---

## 5. `to_d` recipe extension (deliverable 5)

`runs/qcdloop/src/boxGPU_app_recipes.hpp:193` (`to_d`) currently handles only `double`
(identity) and two-limb `.hi`. Float needs a middle arm:

```cpp
auto to_d = [](auto v) -> double {
    if constexpr (std::is_same_v<decltype(v), double>) return v;
    else if constexpr (std::is_same_v<decltype(v), float>) return static_cast<double>(v);
    else return v.hi;                       // two-limb (dd/ff)
};
```

This is **pipeline-owned application code** under `runs/qcdloop/src/` (not snapshot, not
enrichment) — a 1-line addition. The STOP #XX probe's secondary error
(`boxGPU_app_recipes.hpp:195 request for member 'hi' in float`) is exactly this gap; the
Phase-2-shim probe applied this arm to a `/tmp` copy and it compiled clean. Landing puts it
in the tracked `runs/qcdloop/src/boxGPU_app_recipes.hpp` (small, reviewed).

---

## 6. 21-integral scope table (deliverable 6)

Group membership drives the whole-TU flip; per-integral RES selection (`flip_dispatch.py`)
routes each integral independently, so a Phase-2 float group and a Phase-1 dd candidate can
coexist in the same group header without conflict.

| integral | group | Phase-1 state | Phase-2 action | leaves needing synth | path (b) viable | predicted outcome |
|---|---|---|---|---|---|---|
| **B1** | B0m.h | raw double | try float | full set (transitive) | ✅ | downshift-to-float |
| **B2** | B0m.h | raw double | try float | full set | ✅ | downshift-to-float |
| **B3** | B0m.h | raw double | try float | full set + Sign/Real direct | ✅ | downshift-to-float |
| **B4** | B0m.h | raw double | try float | full set | ✅ | downshift-to-float |
| **B5** | B0m.h | raw double | try float | full set + Sign/Real direct | ✅ | downshift-to-float |
| **B6** | B1m.h | raw double | try float | full set | ✅ | downshift-to-float |
| **B7** | B1m.h | raw double | try float | full set | ✅ | downshift-to-float |
| **B8** | B1m.h | raw double | try float | full set + kPow | ✅ | downshift-to-float |
| **B9** | B1m.h | raw double | try float | full set + kSqrt | ✅ | downshift-to-float |
| **B11** | B2m.h | raw double | try float | full set + Real/Imag direct | ✅ | downshift-to-float |
| B10 | B1m.h | **dd (accepted)** | **untouched** | — | — | stays dd |
| B12 | B2m.h | **dd (accepted)** | **untouched** | — | — | stays dd |
| B13 | B2m.h | **dd (accepted)** | **untouched** | — | — | stays dd |
| B14 | B2m.h | **dd (accepted)** | **untouched** | — | — | stays dd |
| B15 | B2m.h | **dd (accepted)** | **untouched** | — | — | stays dd |
| B16 | B3m.h | **dd (accepted)** | **untouched** | — | — | stays dd |
| BIN0 | B0m.h | **dd (accepted)** | **untouched** | — | — | stays dd |
| BIN1 | B1m.h | **dd (accepted)** | **untouched** | — | — | stays dd |
| BIN2 | B2m.h | **dd (accepted)** | **untouched** | — | — | stays dd |
| BIN3 | B3m.h | **dd (accepted)** | **untouched** | — | — | stays dd |
| BIN4 | B4m.h | **dd (accepted)** | **untouched** | — | — | stays dd |

**Phase-1 dd candidates (11): confirmed untouched — stay at dd (STOP #ZZ guard).** The
Phase-2 float flip for a group builds a *separate* float TU; `flip_dispatch.py` takes B10's
RES from its dd binary and B6's RES from the float binary even though both live in B1m.h. The
group's float TU still *compiles* B10 at float (unavoidable — it's in the header), but its
RES stream for B10 is **discarded** by the dispatch plan (B10 is owned by the dd source). No
Phase-1 result changes.

*Prediction caveat (honest):* the table's "downshift-to-float" is a **build-and-route**
prediction. The actual Phase-2 **acceptance** (lift ≥ 0.0 vs raw double under `flip_gate` with
`lift_direction=downshift`) is an L-measure question for the landing dispatch — float will
*hold* precision for well-conditioned integrals and *lose* it for the ill-conditioned ones
(some of B1–B9/B11 may reject on precision and stay at double). Scoping proves the *mechanism*
is viable and buildable for all 10; which ones *accept* is measured at landing.

---

## 7. STOP audit (deliverable 7)

| STOP | definition | state | evidence |
|---|---|---|---|
| **#BBB** | a non-template leaf has substantive hand-authored math (no library pass-through) | **not fired** | all 13 §1.1 leaves are library-pass-through / member / scalar-ternary; none hand-authored |
| **#CCC** | path (b) leaf fails to compile because the library-native float instantiation is absent on this Kokkos | **not fired** | `/tmp` probe: B0m/B1m/B2m float TUs compile **clean**, run, and produce genuine-float output (`Kokkos::abs/log/sqrt/conj<complex<float>>` all present) |
| **#DDD** | the shim, in a TU with the double reference header, causes ODR collision / shadows a double call site via implicit conversion | **not fired** | probe: a **double** TU with the float shim present compiles **clean**; shim signatures are strictly `float`/`complex<float>`, no viable implicit-conversion tie with the double overloads |
| **#ZZ** | Phase-2 routing regresses a Phase-1 dd accept | **not applicable** | per-integral RES selection keeps all 11 dd candidates on their dd source; scoping ran no routing |
| **#Z** | vendored snapshot pristine | **clean** | probes used `/tmp` clones; `git status` on `runs/qcdloop_headers_full/`, `third_party/`, `runs/qcdloop/src/` empty before and after |

Empirical probe method (read-only, reproducible): clone snapshot → `/tmp`; write a
**hand-authored probe** shim (13 float leaves) + a two-line clone wrapper (reference + shim);
apply the §5 `to_d` float arm to a `/tmp` copy of the recipe; compile a per-group float driver
(`run_app<complex<float>, float, float, FloatPrinter>`) for B0m/B1m/B2m; run; diff float vs
double RES. The probe shim is **not production code** — production path (b) generates it from
the reference inventory (§3).

---

## 8. ff-hypothetical note (deliverable 8)

**ff has NO library-native support.** The ff scalar/complex are custom types
(`quad::ffun::ffloat` / `quad::ffun::ffcomplex`, `third_party/include/ff_math.hpp`,
`ff_complex.hpp`); there is no `Kokkos::abs<ffcomplex>` / `Kokkos::log<ffcomplex>` — the leaf
bodies path (b) binds to **do not exist** for ff. Path (b) is therefore **fundamentally
unavailable for ff**: a float shim can bind `kAbs(complex<float>)→Kokkos::abs`, but an ff shim
would have nothing library-native to bind to. ff downshift genuinely requires an **ff-wrapper
header** (a static `kokkosMaths_ff.h` analogous to `kokkosMaths_dd.h`, layering ql:: leaves on
the `quad::ffun` primitives) if Reet ever wants it — i.e. the original STOP #XX enrichment
ask, unchanged for ff. **Float is the special case**: IEEE float is library-native, so the
pipeline can route around the missing wrapper. The shim-synthesis capability is a
**strict subset** of "downshift to library-supported precisions"; dd and ff remain
enrichment-required by construction.

---

## 9. Ordered landing plan (if authorized — NOT executed here)

1. **`agents/patcher/shim_synth.py`** — leaf-inventory extractor (parse non-template `ql::`
   overloads from the reference header, structural — no baked-in leaf names) + float-sibling
   renderer + inventory-sha stamping. Unit tests: inventory parse on the real `kokkosMaths.h`
   (expect the 13 §1.1 leaves), rendered-shim golden, sha-invalidation round-trip.
2. **`tu_emit.py`** — add `maths_reference_header` + `shim_synthesis` to `PrecisionProfile`;
   `render_wrapper` shim branch; `emit_flip_tu` shim-write step (reusing `_refuse_snapshot`);
   flip `FLOAT` profile to `shim_synthesis=True, available=True`. Tests: float wrapper shape,
   snapshot-write refusal, profile-parameterization (a synthetic `half`-like profile selects
   the same path).
3. **`boxGPU_app_recipes.hpp`** — add the `float` arm to `to_d` (§5). 1 line, in tracked
   pipeline source.
4. **`flip_gate.py`** — add `lift_direction` (`upshift`: `> margin`; `downshift`:
   `>= margin`) per the Phase-2 dispatch's gate extension. Tests: downshift accepts lift 0.0,
   rejects lift < 0.0.
5. **Phase-2 L-measure** — extend `phase1_lmeasure.py` (or a `phase2_lmeasure.py`) to build
   the float group TUs for B0m/B1m/B2m, route B1–B9/B11 to float via `DispatchPlan`, measure
   float-vs-dd-ref digits vs raw-double-vs-dd-ref, apply the downshift gate. Report per-integral
   accept/reject + final 21-integral precision assignment.
6. **Regression:** acc1482 26 + Phase-1 48 + full suites green; snapshot + third_party
   pristine; all 11 dd candidates unchanged (byte-identical RES from their dd sources).

---

## 10. Verdict (this scoping subtask)

Per the subtask's verdict gate:

> Path (b) viable for all leaves called by B1–B9, B11 → Reet authorizes implementation
> dispatch. Phase 2 lands as pipeline-authored shim synthesis, zero source enrichment.

**This is that outcome.** All 13 non-template leaves are path-(b)-synthesizable; the compile
probe proved B0m/B1m/B2m float TUs build clean, run, and do genuine float compute; STOP #CCC
and STOP #DDD did not fire; no STOP #BBB leaf exists. **Recommend authorizing the
implementation dispatch** (§9 plan). Zero source enrichment for float. ff stays
enrichment-required (§8); Phase 3 and ff are separate future dispatches.

---

## 11. Artifacts

- This report: `runs/qcdloop/PHASE_2_SHIM_SYNTHESIS_DESIGN_2026-07-29.md`.
- Prior: `runs/qcdloop/PHASE_2_STOP_XX_2026-07-29.md` (the STOP #XX that motivated this).
- Read (unchanged): `runs/qcdloop_headers_full/{kokkosMaths.h, kokkosMaths_dd.h, kokkosUtils.h,
  box/{B0m,B1m,B2m,box_common}.h}`, `agents/patcher/tu_emit.py`,
  `runs/qcdloop/src/boxGPU_app_recipes.hpp`.
- Compile probe: `/tmp` clone (transient, removed); snapshot + third_party + `runs/qcdloop/src`
  pristine (verified).
