# Region-Core Element-Level Promotion — Design + Blast Radius (2026-07-28)

**Subtask:** scope the master blocker of STOP #CC (emission-binding). **Design + blast
radius only — no production code, no tests, no pipeline changes.** Landing
authorization is a separate dispatch after Reet reviews.

**Pinned commit:** `992e209` (STOP #A dispatch fix; instantiation newly real).
The STOP #CC report (`EMISSION_BINDING_2026-07-28.md`) and this note are uncommitted
design/report files on top of it.

**Commit lineage**
- `992e209` STOP #A dispatch fix — live BO reroute; the promoted dd tree instantiates
  at the real box binding for the first time, surfacing the dd/`double` binding errors.
- `57d78f4` L-measure v1 — the dead-code false positive that made "validated as an
  uninstantiated template" untrustworthy (the lesson governing why this subtask is
  design-only).
- `57651ac` Phase 2d-A — **introduced the d1 guard** this design proposes to carve out.

**Verdict of this scoping pass (no STOP fired):**
- **STOP #DD does NOT fire** — the Phase-2d d1 rationale is fully recovered (§1) and the
  element-only carve-out provably preserves it (§2.1).
- **STOP #EE does NOT fire** — the sweep (§4) shows **3** integrals with confirmed/
  predicted non-trivial emission diffs (B14, B15, B16), plus one bounded, cleanly
  enumerable conditional class (the `x4` root-solver blocks, historically never
  promoted). Well under the >8 threshold.
- **STOP #FF does NOT fire** — the semantic model (§3) is unambiguous in every context
  because **the array declaration is never retyped**; only individual read/write
  *occurrences* are wrapped, so a "partially-promoted array" never exists.
- **STOP #Z respected** — this pass touches only design/report files; the vendored
  snapshot `runs/qcdloop_headers_full/` is read-only and untouched.

**Headline scoping boundary (important):** the fixed-size-aggregate element promotion
designed here **unblocks B14 and B16** (and the decl-init read in B15) — the `cxs[k]`
direct-arithmetic integrals — but **does NOT unblock B10**. B10's region-core Shape-3
(`B1m.h:439`) is formed by a promoted dd value multiplying a **`Kokkos::View` accessor
`res(i,1)` and `Constants<TOutput>::_two()` returns** — neither is a fixed-size
aggregate element, so it falls in an adjacent class that is explicitly a **non-goal**
here (no dynamic-container support). See §4 and §7.

---

## 1. The d1 guard rationale (recovered — STOP #DD cleared)

`agents/shared/region_scan.py:155-170`, in `region_reads_from_function`:

```python
# Phase 2d (d1): names used as an array subscript BASE anywhere in the region
# (``name[``) are aggregates/pointers, not promotable scalars — promoting one
# yields ``FloatFloat[int]`` / ``operator[](FloatFloat,int)`` build failures (the
# xpi_in-style Kokkos::Array reads).  Exclude them from the derived reads.
subscripted = _subscripted_names(toks, rs, re_)
...
if t.text in subscripted:
    continue
```

Introduced in **`57651ac` (Phase 2d-A, 2026-07-24)**. Recovered rationale, corroborated
across three sources:

- **The guard comment itself:** a name used as a subscript base (`cxs[`) is an
  aggregate. If `region_reads_from_function` returned `cxs`, `promote_region_block`
  would rename/retype the **whole name** → `FloatFloat cxs` (or `FloatFloatComplex cxs`), and every
  `cxs[k]` in the body becomes `FloatFloat[int]` / `operator[](FloatFloat,int)` — a hard build
  failure. The motivating case is named: "xpi_in-style `Kokkos::Array` reads."
- **`PHASE_2C_2026-07-24.md:89-92`:** `boxGPU.h:99-101, 114-115` were correctly reported
  as `promotion_no_op` because "those lines touch only `xpi[...]` (a `View`/array
  aggregate, deliberately excluded from scalar promotion), so the derived scalar-read
  set is genuinely empty."
- **`region_scan.py:458-480` (`_local_decl_scalar_names`) + `_core_type_is_scalar`
  (483-495):** the scalar-name universe *already* excludes `Kokkos::Array<…>` decls
  (comment: "`Kokkos::Array<…> xpi = …` are skipped"; and "a pointer (`*`) or C-array
  (`[`) anywhere makes it non-scalar"). The d1 guard is the read-site companion to this
  decl-site exclusion.

**Why element-only reversal preserves the protection:** the d1 regression is caused by
promoting the **whole array name**, which makes the *declaration* `FloatFloat cxs` and thus
every `cxs[k]` ill-typed. Element promotion never touches the array name or its decl —
it wraps the **read expression** `cxs[k]` in an entry cast (`DoubleDoubleComplex(DoubleDouble(cxs[k].real()),
DoubleDouble(cxs[k].imag()))`), leaving `cxs` typed `Kokkos::Array<TOutput,3>`. `FloatFloat[int]`
can never arise because no `FloatFloat cxs` decl is ever emitted. The d1 whole-name
exclusion stays exactly as-is; the carve-out is strictly additive at the element
read/write occurrence.

## 2. The defect and the narrow fix

### 2.1 What forms `Kokkos::complex<Kokkos::Experimental::DoubleDoubleComplex>`

Ground truth (B14, `box/B2m.h`, function at 373-406; confirmed in
`lmeasure_run/B14/.../iter_0_build.log`):

```cpp
Kokkos::Array<TOutput, 3> cxs;                       // B2m.h:391 — fixed-size complex aggregate
ql::kfn<TOutput, TMass, TScale>(cxs, ieps, -si, m2, m4);
...
TOutput fac;                                         // B2m.h:396 — closure carrier (widened → DoubleDoubleComplex)
...
fac = TOutput(ql::Constants<TMass>::_two() / (m2 * m4 * ta)) * cxs[0] / (cxs[1] * cxs[2]) * xlog;
//    ^-- region promotes m2/m4/ta/xlog to dd; TOutput(...) cast → DoubleDoubleComplex(...)      ^-- B2m.h:401
//        DoubleDoubleComplex(...)  *  cxs[k]   →   Kokkos::complex<Kokkos::Experimental::DoubleDoubleComplex>
```

The `TOutput(...)` functional cast is rewritten to `DoubleDoubleComplex(...)` (existing
`_complex_cast_indices`), but `cxs[0]`, `cxs[1]`, `cxs[2]` stay `Kokkos::complex<double>`
because `cxs` is excluded from `reads` (d1). Overload resolution promotes both operands
of `DoubleDoubleComplex(...) * cxs[0]` to the common `Kokkos::complex<DoubleDoubleComplex>`, which fails
`Kokkos::complex`'s floating-point `static_assert` (`Kokkos_Complex.hpp:35`) and poisons
every downstream use, including the `res(i,k) = fac` stores (B14 errors 761 → 35 → 765).

### 2.2 The fix in one sentence

Give the region promotion **element-level** read/write promotion for **fixed-size
complex aggregates**: when a `cxs[k]` occurrence of a `Kokkos::Array<TOutput,N>` (or
equivalent fixed-size complex array) appears in a promoted region, wrap that read
occurrence in an entry promotion to `complex_type`, and demote element **stores** on
exit — never retyping the array declaration.

## 3. Proposed change surface (design only — not implemented)

| file | change | est. LoC | new schema/fields |
|------|--------|---------:|-------------------|
| `agents/shared/region_scan.py` | new `region_element_reads_from_function(...)` returning subscripted reads of *fixed-size complex aggregates* that d1 drops, as `(base, index_text, occurrence_span)` records; array-type detection helper `_fixed_complex_array_decls(toks)` (scan `Kokkos::Array< <complex-elem> , <int-literal> > name;`). d1 guard **unchanged**. | ~60 | `ElementRead` record (`base:str, index:str, start:int, end:int`) |
| `agents/integrator_base/boundary.py` | `promote_region_block`: accept `element_reads` param; emit per-occurrence read wrap via `_promote_complex_entry(expr=cxs[k])` **in-place** (span edit, not a rename); element **store** demotion via `_demote_complex_expr` on `cxs[k] = <dd>` targets; guard so a whole-array pass `f(cxs)` is left verbatim. | ~50 | `element_reads` kwarg; `_ELEM_*` edit kind |
| `agents/shared/type_resolve.py` | resolve `Kokkos::Array<T,N>` element type → complex classification of `T` (reuse existing template-param binding from `57651ac`). | ~20 | — |
| call sites (fanout / regional wiring) | thread `element_reads` from the derivation into `promote_region_block`. | ~15 | — |

**No new emission machinery is validated by construction** — the wrap reuses the
existing, build-exercised `_promote_complex_entry` / `_demote_complex_expr` helpers, and
lands only through the same instantiated-build gate that surfaced these errors.

### 3.1 Array-type detection (bounding the scope)

Element promotion fires **only** when the subscript base resolves to a **fixed-size
complex aggregate**:
- `Kokkos::Array<TOutput, N>` / `Kokkos::Array<Kokkos::complex<…>, N>` where `N` is an
  integer literal (the QCD case: `Array<TOutput,3> cxs`, `Array<TOutput,2> x4`).
- C-style fixed arrays `TOutput a[N]` and `std::array<TOutput,N>` — same shape,
  admissible; included for generality (no hardcoded `cxs`/`x4` names — the detector keys
  on the *decl shape*, honoring the no-placeholder-patterns constraint).

Explicitly **excluded** (non-goal; left at caller precision, as today):
- **Dynamic containers** — `Kokkos::View<…>`, `std::vector<…>`, and any `()`-accessor
  (`res(i,k)`). This is why B10's `res(i,1)` case (§4) is out of scope.
- **Nested aggregates** `Kokkos::Array<Kokkos::Array<…>,M>` (the `Y[i][j]` mass matrix)
  — double-subscript; deferred.

### 3.2 Semantic invariant (resolves STOP #FF)

**The array declaration is never retyped. dd-ness lives only in wrapped read
temporaries and demoted store expressions — never in an array element's storage.**
Consequences, each well-defined:

| context | emitted form | precision of the array |
|---------|--------------|------------------------|
| read in arithmetic `cxs[k] * dd` | `DoubleDoubleComplex(DoubleDouble(cxs[k].real()), DoubleDouble(cxs[k].imag())) * dd` | unchanged (caller) |
| whole-array pass `f(cxs)` | `f(cxs)` verbatim | unchanged (caller) — callee runs at caller precision, no dd leak |
| element store `x4[1] = <dd expr>` | `x4[1] = <demote(dd expr) to element type>` | unchanged (caller) |
| element read into scalar decl `TOutput xs = cxs[0]` | `DoubleDoubleComplex xs = DoubleDoubleComplex(DoubleDouble(cxs[0].real()), …)` (decl retyped by existing decl path; **RHS wrapped by element read**) | unchanged (caller) |

There is no "mixed-precision array" state to reason about — a `Kokkos::Array<TOutput,3>`
is always uniformly `TOutput`. This is the single design decision that removes the
STOP #FF ambiguity; it is chosen deliberately over "promote all elements of an array"
(which *would* create a mixed/retyped aggregate and re-open the d1 failure mode).

### 3.3 Interaction with the boundary / designed-exit transform (Shape 1)

Element-promoted reads still flow into a `res(i,k) = fac` store where the carrier `fac`
is dd. That store is **Shape 1 designed-exit narrowing** (held deliverable b) — element
promotion fixes the *interior* `complex<DoubleDoubleComplex>` (761) but the *store* (764/765) still
needs the designed-exit demote. **The two must land together** (§6): element promotion
alone leaves 764/765; Shape-1 alone can't build because 761 still poisons the type. This
is the STOP #CC entanglement, now with a concrete joint-landing plan.

### 3.4 Interaction with rule (a) closure_decls

A closure carrier written from an element-promoted expression (`fac = DoubleDoubleComplex(...) *
cxs[k]...`) is already widened at its decl by the existing `ClosureDecl` path (carrier
`fac` decl 396 → `DoubleDoubleComplex`). Element promotion needs **no per-index carrier
tracking**: because the array is never retyped, "any element promoted → widen the
*carrier* that receives the expression" is the existing rule-a behavior and suffices.
The array itself is never a carrier. **Invariant: element promotion adds carriers only
for the LHS the promoted expression is assigned to, never for the array.**

## 4. Blast-radius sweep — 21-integral touch table

Method. Touch surface = a region the pipeline **promotes to dd** that contains a
**fixed-size complex-aggregate subscript occurrence** (read in arithmetic, or read in a
decl-init, or an element store). Determined by intersecting each integral's landing
regions (`BOUND_DECOMPOSITION_ALL_21_2026-07-25.md`) with a static grep of the pristine
source (`runs/qcdloop_headers_full/box/*.h`) for the pattern, cross-checked against the
two instantiated build logs that exist (B14, B10) and the recorded `final.diff`s.
Baseline gate status is **measured** only for B10/B14 (the only integrals built under
the post-992e209 instantiated pipeline); all others are **predicted** from static
analysis and are marked so — this pass did **not** run 21 fresh pipelines (a scoping
pass; a full sweep is validation-plan step 4, §6).

Legend: **Touch** = would element promotion emit a non-trivial diff? `direct` = `cxs[k]`
in arithmetic (needs the fix); `decl-init` = `T xs = cxs[k]` (needs RHS wrap); `x4-cond`
= only the `x4` root-solver block, historically never promoted (0 `final.diff`s retype
`x4`); `none` = no fixed-size complex subscript in any dd-promoted region; `oos` = region-
core exists but is out-of-scope (View/`Constants`).

| # | integral | class | landing region(s) | fixed-size subscript in promoted region? | region-core Shape-3? | **Touch** |
|---|----------|-------|-------------------|------------------------------------------|----------------------|-----------|
| 1 | B1  | STABLE | boxGPU/box scalar | no (stable-gated, no dd promo) | no | none |
| 2 | B2  | STABLE | — | no | no | none |
| 3 | B3  | STABLE | — | no | no | none |
| 4 | B4  | STABLE | — | no | no | none |
| 5 | B5  | STABLE | — | no | no | none |
| 6 | B6  | STABLE | — | no | no | none |
| 7 | B7  | STABLE | — | no | no | none |
| 8 | B8  | STABLE | — | no | no | none |
| 9 | B9  | STABLE | — | no | no | none |
| 10 | B11 | STABLE | — | no | no | none |
| 11 | B12 | dd-suff | B2m.h:206,207,241 (`ga43*`, `wlog*` scalars) | no | no | none |
| 12 | B13 | dd-suff | B2m.h:300-355,533 (`ddilog` scalars) | no | no | none |
| 13 | **B14** | dd-suff | **B2m.h:401**,405,578 (`fac=…*cxs[k]…`) | **yes (direct)** | **yes — MEASURED** (761/35) | **direct** |
| 14 | **B10** | dd-suff | B1m.h:227,240,241; **439** | no (fixed-size); `res(i,1)` View + `Constants` | **yes — MEASURED** (439) | **oos** |
| 15 | **B15** | dd-insuf | B2m.h:492,496,578 (`TOutput xs = cxs[0]`, then `fac=xs/…`) | **yes (decl-init)** | predicted | **decl-init** |
| 16 | **B16** | dd-insuf | **B3m.h:177**,183,230 (`fac=…*cxs[0]/…`) | **yes (direct)** | predicted | **direct** |
| 17 | BIN0 | dd-insuf | B0m.h:68,88 (`cspence`/`ltspence`; `x4` solver nearby) | only `x4` block | predicted (if promoted) | x4-cond |
| 18 | BIN1 | dd-insuf | B1m.h:62,63,79 | only `x4` block | predicted (if promoted) | x4-cond |
| 19 | BIN2 | dd-insuf | B2m.h:64,65,84 | only `x4` block | predicted (if promoted) | x4-cond |
| 20 | BIN3 | dd-insuf | B3m.h:76,78,109 (+`x1`,`l4`,`ix4` reads) | only `x4`/`x1`/`l4` blocks | predicted (if promoted) | x4-cond |
| 21 | BIN4 | dd-insuf | B4m.h:119,195,198,233 (`x[i][j]` nested, `x_in` construction) | nested (`x[i][j]` — deferred) / construction | predicted | none (nested = deferred) |

**Go/no-go count.** Non-trivial in-scope emission diffs: **3 confirmed/predicted**
(B14 direct + B16 direct + B15 decl-init). The `x4-cond` tail (BIN0-3) is (a) a single,
cleanly enumerable class — element promotion of `x4[k]`/`x1[k]`/`l4[k]` stores/reads in
the quadratic-root solver — and (b) **historically never promoted**: no recorded
`final.diff` retypes `x4` (`grep x4__ff|x4__w|x4__dd` over all `final.diff`s = empty),
because the root-solver is stable-conditioned geometric setup, not a cancellation
hotspot. It only becomes a touch if a future run's Strategy selects that region for dd,
which the conditioning data does not motivate. Even counting it worst-case, the total
stays at/under the STOP #EE threshold and every class is enumerable — **STOP #EE does
not fire.**

## 5. Regression risk enumeration (per touched integral / class)

For each class in §4, the concrete regression surfaces:

**(R1) Emission-granularity change — B14, B16, B15.** A `cxs[k]` read that was passed
through verbatim now becomes a wrapped `DoubleDoubleComplex(DoubleDouble(cxs[k].real()), …)`
expression. *Semantics unchanged* (full caller precision is preserved into the wrap;
`_promote_complex_entry` wraps each component in the extended **scalar** first, exactly
as the existing read path does), but the **emitted variant text changes** for these
three regions. Detected by: the instantiation gate (must build clean) + a `final.diff`
review showing only the intended wrap edits.

**(R2) Carrier-widen interaction — B14, B16.** An element-promoted expression assigned
to a carrier (`fac = … * cxs[k]`) relies on the carrier already being widened by rule-a.
Risk: element promotion firing on a read whose result is assigned to a carrier that is
*not* in the closure set → a new `DoubleDoubleComplex`-valued expression stored into a
`Kokkos::complex<double>` carrier (a fresh Shape-1). Mitigation/contract: element
promotion must fire **only** inside a region already selected for dd promotion, so the
carrier is in scope for rule-a; enforced by the invariant in §3.4. Detected by: gate.

**(R3) Boundary-transform interaction — B14, B16 (new store sites).** Element promotion
makes the carrier expression dd, so the `res(i,k) = fac` designed exits (B14 764/765)
now definitely need Shape-1 demotion. This is *expected* and is why the joint landing
(§6) pairs element promotion with deliverable b. Risk if landed alone: 764/765 persist.

**(R4) Element-store demotion — x4-cond class (BIN0-3, conditional).** If a promoted
region ever contains `x4[1] = c/(TOutput(a)*x4[0])` with `a` promoted, the store target
`x4[1]` (a caller-precision element) needs element-store demotion. This is designed
(§3.2 row 3) but **untested against a real promoted x4 region** (none exists). Risk is
latent, not active. Contract: if a future run promotes an `x4` region, the store-demote
path activates; until then it is dead but specified.

**(R5) d1-motivated regression re-opening — NONE.** The original d1 failure
(`FloatFloat[int]` from whole-array promotion) cannot recur: the array decl is never
retyped (§2.1, §3.2). This is the load-bearing safety property. Detected by: the
existing d1 regression test must stay green (§6), plus a new test asserting the array
decl token is never in the edit set.

**(R6) Whole-array-pass leak — NONE by construction.** `f(cxs)` is left verbatim
(§3.2), so a callee never receives a dd-typed array. Detected by: a new test that a
region passing `cxs` whole to a call emits no wrap on that occurrence.

No regression class is un-enumerable → **STOP #FF does not fire.**

## 6. Validation plan (for the landing dispatch, if authorized)

**Build validation (instantiation gate is the sole arbiter — no "trust me"):**
1. **B14 clean build** — the decisive test. After element promotion **+** deliverables
   b (designed-exit narrow) **+** c (rule-c receiving-local), B14 must reach
   `instantiation gate: 0 errors, 0 unknown`. This is the first honest clean dd build.
2. **B16 clean build** — second `cxs`-direct integral; confirms generality beyond B14.
3. **B12 + B13 unchanged** — regression check that `none`-touch integrals emit
   byte-identical variants (diff the `final.diff` against pre-change).
4. **Full 21 instantiation-gate sweep** — run the current pipeline per integral, gate
   each build, produce the measured analogue of §4's predicted table. Go/no-go for the
   conditional `x4-cond` tail: if any BIN integral now promotes an `x4` region, R4
   activates and must gate clean.

**Unit-test surface (new, in the landing dispatch — none added in this scoping pass):**
- element read wrap: `cxs[k] * dd` → wrapped, array decl unchanged (R1, R5).
- element store demote: `x4[1] = <dd>` → demoted to element type (R4).
- whole-array pass untouched: `f(cxs)` emits no wrap (R6).
- decl-init read: `TOutput xs = cxs[0]` with `xs` promoted → RHS wrapped (B15).
- **d1 preserved:** the original whole-array motivating case still yields *no* whole-name
  promotion and the array decl token is never edited (R5) — this is the guard-preservation
  test the dispatch requires.
- fixed-size detection negative: `Kokkos::View`/`res(i,k)`/`std::vector` reads are **not**
  wrapped (scope bound; keeps B10's `res(i,1)` out).

**Gate coverage:** the instantiation gate (`agents/patcher/instantiation_gate.py`,
wired at `tier_b_stage1.py:254`) already classifies the four shapes; the changed
variants build through the same gate with no config change. Confirm the gate still
reports `0 unknown` on the new clean builds (a non-zero unknown = STOP #BB).

## 7. Explicit landing plan (ordered — only if Reet authorizes)

1. **Element-level promotion** in `region_scan` + `promote_region_block` +
   `type_resolve` (§3). Fixed-size complex aggregates only.
2. **Shape 1 store-narrowing (b) + Shape 2 / rule-c receiving-local widen (c)** — the
   previously-held emission fixes, now able to instantiation-validate because step 1
   clears the interior `complex<DoubleDoubleComplex>`.
3. **Instantiation-gate validate on B14** (§6.1) — must build clean end-to-end.
4. **Instantiation-gate sweep on all 21** (§6.4) — regression check; realize the
   measured table.
5. **L-measure re-run** — **B14 now an honest dd baseline** (measure lift vs bar ≥+8).
   **B10 remains blocked** on (i) its 71 rule-d leaf-clone errors *and* (ii) its
   out-of-scope View/`Constants` region-core (`B1m.h:439`) — **no B10 lift expected from
   this subtask.** Both are separate future dispatches (rule-d; and a broader
   caller-precision-read wrapping that would extend beyond fixed-size aggregates).

## 8. Non-goals honored

- No production code / test / pipeline changes in this pass — design + report only.
- No rule-(d) work (B10's 71 leaf-clone errors — separate dispatch).
- No dispatch-selector changes (`992e209` stands).
- No dynamic-container support (`Kokkos::View`, `std::vector`, `res(i,k)` accessors) —
  which is precisely why B10's region-core is out of scope here.
- No re-litigating the d1 guard beyond the element-level carve-out — whole-name
  subscript-base exclusion stays for its original case.
- No L-measure re-run in this subtask.

## 9. Hand-back to Reet

**One decision:** authorize the ordered landing plan (§7) for **fixed-size complex
aggregates only**. Blast radius is bounded and enumerable (§4: 3 in-scope integrals + a
never-promoted conditional tail; §5: 6 regression classes, all detected by the
instantiation gate; R5/R6 are safe by construction). The clean B14 dd build (§6.1) is
the go/no-go proof and the first honest measurement the emission-binding arc can produce.

**Explicitly flagged, not folded in:** B10 needs *two* further, out-of-scope pieces —
its 71 rule-d clones and a broader caller-precision-read wrapping (View accessors +
`Constants` returns) that goes past fixed-size aggregates. Do not expand this subtask to
cover them; they are their own conversations.
