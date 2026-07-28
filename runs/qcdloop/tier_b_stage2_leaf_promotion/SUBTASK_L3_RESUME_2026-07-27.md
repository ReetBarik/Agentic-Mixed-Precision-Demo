# Subtask L3-resume — emission plumbing under Resolution A (LANDED)

**Status: LANDED. Production code written to `agents/`; permanent tests added under
`tests/`; vendored snapshot `runs/qcdloop_headers_full/` byte-for-byte pristine.**
Started 2026-07-27, completed 2026-07-28. Branch `langgraph-agents`, on top of the
STOP report `SUBTASK_L3_2026-07-27.md`, L2 (`94a8e4f`), L1′ (`45b384d`),
integrator_base re-pin (`7b30d78`), source enrichment (`e3d2e45`).

This resumes the subtask that `SUBTASK_L3_2026-07-27.md` **STOPPED** at its verdict
gate. That gate handed the scope call back to Reet. **Reet ratified Resolution A**, and
this report records the implementation of L3 under it.

---

## 0. The gate, and the decision that discharged it

The L3 dispatch carried:

> *"If the L1′ P2 compile test fails when adapted, STOP at L3 and hand the scope call
> back (integration seam warranting joint L1′/L2/L3 review, not an L3 code fix)."*

It fired because source enrichment `e3d2e45` ships qcdloop's **own** dd surface in the
vendored snapshot's `kokkosMaths_dd.h` — not just Class-2 `Constants<T>` DATA but
concrete **Class-1 dd wrappers** (kAbs/kLog/kSqrt/kConj/Real/Imag/Sign at
ddouble+ddcomplex, lines 296–368). So L1′ synthesizing those same overloads at dd is a
**redundant, ODR-conflicting duplicate** of the source definitions.

**Resolution A (ratified):** treat a source-provided dd Class-1 wrapper as a **dd
boundary** — do NOT synthesize when the source already defines the symbol at dd. This
mirrors exactly how the same enrichment dissolved Class-2 (consume the source table, do
not vendor/synthesize). Under Resolution A the synth overlay shrinks to ∅, so the leaf
clone builds against the enriched source alone. **The verdict gate now PASSES** (§3).

Resolutions B (make the synthesizer idempotent vs an already-dd primary) and C
(distinct-ns / inline coexistence) were NOT taken; the verdict-gate discipline was
honored — no fall-back to B/C without the explicit Resolution-A go-ahead.

---

## 1. What landed (deliverables a–d)

### (a) L1′ — source-boundary awareness · `agents/integrator_base/shallow_wrapper.py`
A new `SourceProvided` / `SOURCE_PROVIDED` outcome: when the recognizer finds the source
already defines a concrete dd overload for a Class-1 symbol, it classifies that symbol as
a **boundary** and does NOT emit a synthesized overload. This is the "boundary, do not
synthesize" signal Resolution A needs, and it is what prevents the ODR collision at the
root instead of papering over it downstream.

### (b) L2 — boundary reclassification · `agents/patcher/clonable_leaf.py`
`is_dd_boundary` gains a fifth kind: a source-provided dd overload is a boundary (clause
2(ii)). A leaf whose callees are all boundaries (now including source-provided Class-1)
remains clonable; the clone body keeps calling those source dd wrappers unchanged.

### (c) L3 — emission plumbing · `agents/patcher/chain_promote.py`
Two functions, added immediately after `_attach_return_widens` and wired into
`chain_promote` right after the `_attach_return_widens(...)` call:

* **`_select_leaf_overload(graph, name, arg_cores)`** — picks the correct `Lnrat`
  overload by **subset match**: an overload qualifies when its parameter core-types are a
  subset of the observed call-site argument core-types; the most-specific (largest param
  set) qualifying overload wins; ties or no-match → `FanoutError` (STOP). This is
  load-bearing: `Lnrat` has a control-flow `(TOutput,TOutput)` overload @126 and the
  shallow `(TScale,TScale)` overload @138. Both the B10 box integral (`box/B1m.h`, calls
  at :228–230, mixed `TScale`/`TMass` args) and `Li2omx2` (`kokkosUtils.h:687–708`,
  `TScale`/`TScale` args) resolve to **@138** at instantiation. `_find_primary_defs[0]`
  and `_pick_def` both return the WRONG @126; subset matching selects @138 robustly and
  frameworkagnostically ([[feedback_no_placeholder_patterns]] — no baked names).

* **`_materialize_leaf_variants(closure, *, manifest, graph, scalar_type, complex_type,
  complex_tokens, shim_include, new_specs, root_reroutes)`** — for each
  `(g, g_clone)` in `closure.leaf_reroutes` (e.g. `Lnrat → Lnrat_B10`): scans the chain
  frame bodies to collect the callers and the arg core-types, selects the overload via
  `_select_leaf_overload`, builds a `VariantSpec(variant_name=g_clone, orig_name=g,
  file=leaf_fd.file, orig_start/orig_end=@138 extent, promotes=[])`, recovers the
  signature line via `_return_type_signature`, and — if the dd carrier type differs from
  the original return type (`_carrier_dd_type`) — attaches a
  `ReturnWiden(return_line, orig_type, dd_type, function_name=g_clone)`. The body is
  emitted verbatim (renamed), so it never names itself (defeats the STOP #K
  self-recursion premise). It then reroutes every chain caller's `g(...)` call to
  `g_clone(...)`, and, if the entry point itself calls `g`, records `root_reroutes[g] =
  g_clone`. Returns early (inert) when `closure.leaf_reroutes` is empty — i.e. no
  `leaf_ctx` opt-in → byte-identical to pre-L3 behavior (STOP #B).

The dd-ness of the clone is carried **solely** by the `ReturnWiden` on the signature
line; `promotes=[]` means the body is not retyped. Whether the return widens to
`ddouble` or `ddcomplex` depends on the caller's complex binding (`_carrier_dd_type` with
the same `complex_tokens` used to widen the caller's receiving local) — the invariant
"leaf return type == caller receiving-local widen" holds either way, so the
ddcomplex-vs-ddouble choice is an **L-measure value-flow question, not an L3 bug**.

### (d) MEMORY + this report
`project_leaf_callee_promotion_design.md` extended with the L3-resume block and a
correction of the v2 "pipeline IGNORES that code, synthesizes its own, consumes only
DATA" subtlety (FALSIFIED at L3 — the enriched source ships the Class-1 dd wrappers too;
Resolution A is the corrected policy). `MEMORY.md` index one-liner updated.

---

## 2. Overload selection — why subset match, not equality

The first cut used strict equality (`param-cores == arg-cores`) and raised
`FanoutError: arg_cores=['TMass','TScale'] matched neither overload`. Root cause: the
chain reaches `Lnrat` from **two** sites with different argument core-types (the B10 box
integral contributes `{TScale, TMass}`; `Li2omx2` contributes `{TScale}`), and the union
never equals either overload's parameter list exactly. Both sites nonetheless bind to the
@138 `(TScale, TScale)` overload at instantiation (`TScale == TMass == ddouble` → exact
match; @126 would require a `ddcomplex` conversion). Subset match (param-cores ⊆
arg-cores, most-specific wins) selects @138 uniquely and degrades to a clean STOP only
when two sites genuinely require different overloads.

---

## 3. Verdict gate — re-verified PASSING (permanent tests)

The throwaway `/tmp` probes are promoted to permanent, environment-gated regression
tests in `tests/patcher/fanout/test_l3_compile_gate.py` (gcc/13.3.0 via `module load` +
Kokkos + vendored dd headers + the enriched snapshot):

* **P2 (Resolution A).** `render_variant`'s byte-exact `Lnrat_B10` (with
  `ReturnWiden(139, TOutput → quad::ddfun::ddcomplex)`) compiles with **no synth
  overlay** against the enriched source and computes
  `log(1.5/2.5) = log(0.6) = -0.51082562376599072` in dd, imag 0, rc=0. This is the
  exact leaf STOP #K said could not be synthesized — under Resolution A it comes from
  source and builds.
* **P2-negative.** Re-introducing the pre-Resolution-A synth overlay (a redundant
  `ql::kLog`/`kAbs` at dd) fails with an **ambiguating redeclaration** against the source
  wrappers (STOP #K). This pins Resolution A as *load-bearing*: the fix is *removing* the
  synthesis, and only that.
* **P5.** The enriched source ships the 43-coeff dd `Constants<T>` table bit-exactly
  (`num_C=43`, `_pi` bit-exact, π²/12 sum `0.8224670334241132`) — Class-2 is
  source-resident too (STOP #E discharged), the twin of the Class-1 dissolution P2 relies
  on.

---

## 4. Emission unit tests (no compiler) · `test_rule_d_leaf_promotion.py`

Three tests exercise `_materialize_leaf_variants` over a **complete** `shutil.copytree`
of the vendored tree (including `box/`), with the call graph rooted on the tmp copy:

* `test_l3_emits_lnrat_clone_and_reroutes_callers` — with the rule-(d) opt-in,
  `chain_promote` declares `Lnrat_B10`; the rendered `kokkosUtils.h` contains the @138
  **shallow** overload clone (`Lnrat_B10(TScale const& x, TScale const& y)`), its body is
  verbatim (`ql::kLog(ql::kAbs(x / y))`, no self-call), and the `Li2omx2` chain call is
  rerouted (`ql::Lnrat_B10<TOutput, TMass, TScale>(v, x)`).
* `test_l3_no_leaf_ctx_emits_no_clone` — STOP #B at the emission layer: no opt-in →
  `_materialize_leaf_variants` inert, no `Lnrat_B10` anywhere, `leaf_reroutes == {}`.
* `test_l3_leaves_vendored_snapshot_pristine` — the security invariant: an emission run
  leaves every `.h` in `runs/qcdloop_headers_full/` byte-for-byte unchanged.

---

## 5. Safety & discipline

* **Vendored snapshot pristine.** All graph/emission work runs over `tmp` copies. A
  mid-session mistake (graph built over the pristine FULL tree while writing to `tmp` —
  `chain_promote` keys `new_specs` by `FuncDef.file` and `_merge_into_file` writes there,
  so it mutated the real snapshot) was caught immediately and reverted with
  `git checkout -- runs/qcdloop_headers_full/`. The fix is encoded in the tests and in
  the memory ("build the graph over the working tree"). Final `git status` of the
  snapshot: clean.
* **No app-specific identifiers in production code** — overload selection is structural
  (subset match on core-types); `Lnrat`/`kLog`/etc. appear only in comments and tests
  ([[feedback_no_placeholder_patterns]]).
* **Tests green:** `tests/patcher/` + `tests/integrator_base/` + the compile gate = 357
  passed; the rule-(d) file = 11 passed (8 discovery + 3 new emission).

---

## 6. Next

**L-measure** (separate debugging conversation): wire the `LeafPromotionContext` into the
real per-integral run and measure B10's lift end-to-end. Falsifier unchanged: a measured
lift **below +8** means the value-flow model (leaf return type / carrier binding) is
wrong, not the L3 plumbing.
