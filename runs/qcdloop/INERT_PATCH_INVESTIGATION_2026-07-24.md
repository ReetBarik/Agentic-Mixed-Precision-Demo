# Inert-patch investigation on B1 (pre-2c diagnostic)

**Scope:** find out *why* the 8 `measured` cells in Phase 2b's B1 manifest
(`per_integral_out_b1_2b/B1/manifest_scorer_B1.jsonl`) all carry
`delta_effective == baseline_delta_effective` (patches compile/wire/execute but
produce bit-for-bit the unpatched B1 output). This task **does not** fix them
(that's 2c), design joint measurement (2d), or touch the scorer/solver/graph.

**Verdict: the inertness is a mix, and both causes are present in every measured
cell — but they are not equal.** There is one *mechanical* cause that makes the
patches literal no-ops (dominant, universal), sitting on top of one *semantic*
cause that would make them pointless even if the mechanical bug were fixed
(B1-specific, degenerate).

- **Mechanical (dominant, reproduces on every integral): empty promotion
  payload.** The region intent reaches the Patcher with `reads=[]` and
  `writes=[]`, so the shared promotion transform `promote_region_block` has
  nothing to retype and returns the region **verbatim** (`promoted=False`). The
  variant body (or the in-place region) is therefore a bit-identical copy of the
  original, instantiated at `double` → bit-identical output.
- **Semantic (B1-specific, degenerate): region inactivity, already flagged by
  the characterizer.** Every one of B1's 35 regions is `signal_class="stable"`
  ("no elevated conditioning or accumulated-error signal"). B1's own error floor
  is ~6.9e-13 in plain double. There is **no error to fix** on B1, so even a
  correct promotion of these regions could not move B1's output.

The "residual regional no-op = *dispatch lands on the original path*" hypothesis
from the task prompt is **ruled out by a runtime tracer** (below): dispatch
correctly enters the variant. The wiring is sound; the payload is empty.

---

## 1. Reproduction & representative cell

Recommended cell **`B0m.h:126` at `dd`** (intent 0, iter_0). Manifest row:

```
region_id=B0m.h:126 rung=dd status=measured
delta_effective = 6.942342260996819e-13
baseline_delta_effective = 6.942342260996819e-13   → inert=yes
patcher_metadata={"kind":"double-to-dd","intent":"correctness","via":"plain"}
```

`delta == baseline` reproduced directly from the committed manifest; the 2b
report notes a second run produced a bit-identical manifest. All 8 measured
cells carry the **same** number `6.942342260996819e-13` — the tell that the
candidate output equals the baseline in every case.

---

## 2. Symbol-/runtime-level dispatch check (rules out "dispatch-miss")

The emitted fan-out for `B0m.h:126` (from `final.diff` / the tree) is
structurally correct:

- `boxGPU.h`: entry point `BO` body rerouted in place —
  `ql::B0m<…>` → `ql::B0m_B1<…>` (massive==0 branch). ✔
- variant `B0m_B1` (copy of `B0m`): `offshell==0` branch rerouted
  `ql::B1<…>` → `ql::B1_B0m_B1<…>`. ✔
- variant `B1_B0m_B1` (copy of `B1`): emitted, topo-ordered before its caller. ✔

I confirmed the reroute is *exercised at runtime*, not just present in source.
Instrumented both the original `B1` and the variant `B1_B0m_B1` with distinct
`fprintf(stderr,…)` sentinels, rebuilt the vanilla app against the patched tree
(`QL_MODE=vanilla`, `QL_HEADERS=tree_B1`), ran `--sample-count 3`:

```
VARIANT  B1_B0m_B1 hits: 3      (i=0, i=1, i=2)
ORIGINAL B1        hits: 0
```

**Dispatch enters the variant on every B1 sample; the original is never called.**
This is definitional — B1 *is* the massive==0/offshell==0 path — and it is now
observed, not merely inferred. Phase 2a's `nm` gate already confirmed the variant
symbol is present; the tracer confirms it is *reached*. Tracer edits were
reverted after the run; the tree is clean.

Conclusion for this cell: **cause #2 in the "dispatch-miss" sense is false.** The
variant is called; it is inert because its **body was never promoted**.

### Why the body is inert — the empty payload

`render_variant` → `boundary.promote_region_block(region_text, reads, writes, …)`.
The emitted `AMP-FANOUT-MANIFEST` for this variant carries:

```
"promotes":[{"region_start":126,"region_end":126,"reads":[],"writes":[],
             "scalar_type":"quad::ddfun::ddouble","two_limb":true, …}]
```

`promote_region_block` (agents/integrator_base/boundary.py:335) short-circuits:

```python
if not pure_reads and not decl_writes and not caseB:
    return region_text.split("\n"), False   # region verbatim, promoted=False
```

With `reads=[]`, `writes=[]`, and no region-local *declaration* on line 126
(`res(i,1) = fac * …` is an assignment to a `Kokkos::View`, no `T name = …`),
all three sets are empty → **verbatim copy, nothing retyped.** The variant body
is therefore character-for-character identical to the original `B1`, and the
whole app instantiates it at `<Kokkos::complex<double>, double, double>` → the
same bits as vanilla.

Verified the variant body is byte-identical to `git show HEAD:box/B0m.h` lines
113–127. No `ddouble` appears anywhere in the variant body; the only `ddouble` in
the tree is the unused `Constants<ddouble>::_pi2()` specialization in
`ql_shim_dd.h` (never instantiated at `double`).

---

## 3. Where the empty payload comes from (causal chain)

```
report region_local_vars=[]            (characterizer)
  → characterization._region_vars() returns []   (strategy/characterization.py:251)
  → RegionTarget.variables = []
  → dispatch.py:306  reads=list(intent.target.variables) = []
  → writes: extract_region_writes() = []  (res(i,1) is an indexed View write, not a scalar)
  → promote_region_block(reads=[], writes=[]) → promoted=False, region verbatim
  → variant body == original → double instantiation → bit-identical output
```

`_region_vars` returns `region_local_vars` **verbatim when the field is present**
(it only falls back to `prov_vars` for reports predating the field). Here the
field is present as `[]`, so the empty set flows straight through. The write set
is empty because Fix-C finds no promotable scalar write (the region writes
`res(i,1)`, an indexed output-View element).

---

## 4. The characterizer already knows B1 has no signal

Every measured cell is `signal_class="stable"`, and so is **every** B1 region:

```
signal_class distribution across 35 B1 regions: {'stable': 35}   (zero non-stable)
class_counts (report): {'stable': 35}
```

| region_id     | rung(s)     | signal_class | max_rel_err | max_cond | region_local_vars |
|---------------|-------------|--------------|-------------|----------|-------------------|
| B0m.h:126     | dd          | stable       | 1.77e-09    | 5.11e+04 | `[]`              |
| boxGPU.h:100  | float, ff   | stable       | 9.99e-16    | 1.00     | `[]`              |
| boxGPU.h:140  | dd          | stable       | 1.77e-09    | 1.00     | `[]`              |
| boxGPU.h:141  | float, ff   | stable       | 1.92e-13    | 1.00     | `[]`              |
| boxGPU.h:142  | float, ff   | stable       | 2.11e-15    | 1.00     | `[]`              |

(`note` on all: *"no elevated conditioning or accumulated-error signal"*;
`non_localizable=False`.)

B1 is a **fully well-conditioned integral** — a degenerate test subject for a
correctness pass. Its scorer delta (6.94e-13) *is* the baseline; there is no
headroom for dd/ff/float to recover. Strategy queued these regions because it
ranks `top_regions_by_rel_err` and dispatches P6 correctness intents to the top
of that list **without gating on `signal_class`** — so on an all-stable integral
it queues stable regions whose promotion cannot help.

---

## 5. Generalization across the 8 cells (uniform, two structural flavors)

The finding is **uniform** across all 8 measured cells (all `region_local_vars=[]`,
all `stable`, all delta==baseline==6.94e-13), realized through two structural
paths that reduce to the *same* empty-payload cause:

- **`B0m.h:126` — variant fan-out path.** `B1` is *not* the entry point, so a
  real variant `B1_B0m_B1` is created, wired, and runtime-entered (§2) — but its
  body is a verbatim clone.
- **`boxGPU.h:100/140/141/142` — entry-point in-place path.** These regions live
  inside `BO` (the entry point, boxGPU.h:72–147). The fan-out's degenerate branch
  `_promote_in_place` calls the same `promote_region_block`; with empty reads/
  writes it returns `promoted=False` and `_promote_in_place` only splices `if
  promoted` → **no edit at all.** Confirmed on the committed `boxGPU.h:140`
  intent (iter_1): its entire diff is *one `#include "ql_shim_dd.h"` line + a
  shim comment* — the region `res(i,0)/=scalefac2` is untouched. A pure no-op.

So: one path emits a real-but-inert variant, the other emits no computation edit
at all. Both are inert for the identical reason (empty promotion payload).

---

## 6. Does this generalize beyond B1?

- **Empty-payload no-op: UNIVERSAL across qcdloop.** All qcdloop regions are
  *template* regions; the characterizer emits `region_local_vars=[]` for 23/35 B1
  regions and, for the other 12, only **provenance-indexed pseudo-names**
  (`mu2[0]…mu2[4999]`, `p1[…]`, `m1[…]`) with **zero bare source identifiers**.
  Neither form is a usable `reads` set for `promote_region_block` (which needs
  `si`, `ta`, `fac`, …). This matches the prior root-cause note
  ("qcdloop app source never gets computation-line rewrites — template regions,
  variables:[]"). **Expect this to reproduce on every integral**, including the
  high-signal ones (B12, BIN*). It is the primary reason no B1 patch has ever
  moved the needle.
- **Region inactivity: B1-SPECIFIC / degenerate.** B1 being all-stable is *not*
  representative. Integrals with genuine cancellation (B12 rel-err ~2e-4, the
  BINs) *do* have high-signal regions worth promoting. On those, the semantic
  cause disappears and only the mechanical cause remains — meaning **once the
  empty-payload bug is fixed, high-signal integrals should finally show a
  rung-discriminating delta, while B1 should correctly stay flat.**

---

## 7. Recommendation for 2c scope

Because the mechanical cause is universal and blocks *all* payoff, and the
semantic cause is degenerate-to-B1, **2c must fix both, prioritized as follows.**

### 2c-A (primary, blocking): make the promotion payload non-empty

The fan-out/boundary transform cannot promote a region when it receives no
variables. Give it a real reads set for template regions. Two options (pick in
the spec):

1. **Derive reads from the region source** at patch time — libclang/token scan
   of the region's RHS identifiers that are function-scope locals/params
   (`si`, `ta`, `fac`, `lnrat_*`), which is exactly what `promote_region_block`
   needs. This is Patcher-local and does not depend on re-running the
   characterizer.
2. **Fix the characterizer's `region_local_vars`** to emit source identifiers
   for template regions instead of `[]` / indexed provenance. Cleaner long-term,
   but a characterizer change + full re-run.

Recommend **(1)** for 2c (fast, self-contained, testable on the existing tree)
and file (2) as follow-up. Add a **hard gate**: a fan-out variant / in-place
promotion whose rendered block is byte-identical to the original is a
`promotion_no_op` failure, not a silent `measured` cell — so an empty payload can
never again masquerade as a real patch. (2a's wiring gate checks the symbol
*exists*; it does not check the body *changed*.)

Note on semantics: even with a correct reads set, promoting a *single assignment
line* whose inputs are already double-rounded recovers little — meaningful dd
promotion of B1-like code wants the **whole function/call-chain** promoted (i.e.
instantiate the variant at `<ddouble,ddouble,ddouble>`), which the variant
mechanism already supports structurally. Whether 2c goes line-region or
whole-variant-instantiation is the main design fork worth Reet's call.

### 2c-B (secondary, cheap): gate the *correctness* queue on signal — keep the speedup queue

`signal_class="stable"` cuts **opposite ways** for the two queues, so the gate
must be queue-specific, not a wholesale integral skip:

- **Correctness (dd/upcast) queue — skip stable regions.** A stable region has
  no error to fix, so promoting it can only be inert. Strategy should **not queue
  a correctness intent for a `signal_class="stable"` region on the integral being
  scored**, and an integral whose regions are *all* stable (B1) should produce an
  **empty correctness queue** ("nothing to fix on B1") instead of 8 inert cells.
- **Speedup (float/ff/demote) queue — KEEP stable regions.** Stable, well-
  conditioned regions are exactly the *right* demotion targets — everything to
  gain, low risk. B1's 6 speedup intents (`B0m.h:118/124/125` × float/ff) are
  precisely these "demote a well-conditioned integral" attempts. They must stay
  queued. **Note they did not come back "correctly skipped" — they came back
  `llm_gen_failed`** (the separate float-rung Patcher-gen robustness gap), which
  is a Patcher generation problem, *not* an inertness or queueing problem, and is
  out of scope for 2c's inert-patch fix (tracked separately).

So 2c-B is a **correctness-queue-only** `signal_class` gate: reuse the
`signal_class` already in the record, drop stable regions from the P6 correctness
dispatch, leave the speedup dispatch untouched. This makes the manifest honest on
B1's correctness side (empty, not inert) while preserving the demote attempts.

### Proposed 2c prompt-shaped outline (for Reet to accept/iterate)

> **2c — Close the inert-patch gap.**
> 1. Patcher: derive a region reads set from source identifiers when
>    `intent.target.variables` is empty (token/libclang scan of the region RHS,
>    function-scope locals+params only). Feed it into `fan_out_region` /
>    `_promote_in_place`.
> 2. Patcher gate: add `promotion_no_op` — reject (terminal, deterministic) any
>    variant/in-place block byte-identical to its original. Wire it into the
>    manifest failure_mode enum next to `variant_name_collision`.
> 3. Strategy: gate the P6 **correctness** dispatch on `signal_class != "stable"`
>    for the scored integral (all-stable integrals → empty correctness queue).
>    **Leave the speedup dispatch untouched** — stable regions are the intended
>    demotion targets; B1's 6 speedup intents must still be attempted.
> 4. Re-run B1 (correctness side should now be *clean/empty*, not 8 inert cells;
>    speedup side unchanged) **and** one high-signal integral (B12 or a BIN) to
>    confirm correctness deltas now discriminate rung.
> 5. Tests: fan-out emits a non-verbatim body for a region with derived reads;
>    `promotion_no_op` fires on an empty-payload region; Strategy drops a stable
>    region from the correctness queue **but keeps it in the speedup queue**.

---

## 8. Telemetry / tests worth keeping

- **`promotion_no_op` gate (recommended to land in 2c, not now):** the single
  most valuable guard this investigation motivates — a byte-identical
  variant/in-place block must be a typed failure, not a `measured` cell. Had it
  existed, 2b would have reported 8 `promotion_no_op` instead of 8 inert
  `measured`.
- **Optional fan-out dispatch tracer flag:** the `fprintf` sentinel technique
  (guarded by an env var / build define, e.g. `AMP_FANOUT_TRACE`) is a cheap way
  to assert at runtime that a variant is entered. Useful as an opt-in debug
  build option; **not** needed in the steady-state pipeline (the `nm` gate +
  now-recommended body-changed gate cover the static side). I did **not** land it
  (out of scope; would touch fan-out rendering = 2c territory). Flagging it so
  2c can decide.
- No scorer/graph/solver changes were made; tree instrumentation was reverted.

---

## 9. Handoff to Reet

- **Which cause dominates:** *mixed, but decisively.* The **mechanical
  empty-payload no-op dominates and is universal** (blocks all payoff on every
  integral); **region inactivity is real but B1-specific/degenerate** (B1 has no
  error to fix — the characterizer already says so). The prompt's "dispatch lands
  on the original path" sub-hypothesis is **false** — runtime tracer shows the
  variant is entered on every B1 sample.
- **2c scope:** fix the payload (Patcher-derived reads + a `promotion_no_op`
  gate) **and** gate the *correctness* queue on signal (Strategy drops stable
  regions from the correctness dispatch only — **keeps them for the speedup
  queue**, since stable/well-conditioned regions are the intended demotion
  targets; B1's 6 speedup intents came back `llm_gen_failed`, a separate
  Patcher-gen gap, not a queueing problem). Payload fix is the blocking,
  universal one; the Strategy gate is the cheap, honesty one. Full outline in §7.
  Main design fork for you: line-region promotion vs. whole-variant `<ddouble,…>`
  instantiation.
- **Surprises:**
  1. Two *different* structural paths (variant fan-out vs. entry-point in-place)
     both collapse to the same empty-payload no-op — the `boxGPU.h:*` cells never
     received a computation edit at all (only an `#include`), while `B0m.h:126`
     got a real-but-verbatim variant.
  2. `region_local_vars` is not just sometimes empty — when non-empty it's
     **provenance-indexed junk** (`mu2[1000]`), never source identifiers. The
     field as currently emitted is unusable as a reads set for *any* qcdloop
     region, which is why deriving reads from source (2c-A option 1) is the
     pragmatic path.
  3. B1 is a genuinely bad demo for a *correctness* pass — it's fully stable.
     Once 2c lands, the interesting e2e signal will come from a high-signal
     integral (B12/BIN), where the mechanical fix should finally produce
     rung-discriminating deltas.
