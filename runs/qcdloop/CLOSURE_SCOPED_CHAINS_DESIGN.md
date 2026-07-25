# Closure-Scoped Chain Promotion — Design Notes

Status: **design finalized** (2026-07-25). Supersedes the Blocker-A carrier framing
(`BLOCKER_A_CARRIER_DESIGN.md`), which survives as one special case (§5, §7).
Scope: Phase 2f Tier-B Stage-1 chain-scoped double-double (dd) promotion.
Discipline: **design only** — no implementation in this pass.

> **Load-bearing correction up front (read this before §2).** The obvious fix
> the last run's report sketched — "extend the B10 chain to cover `Li2omx2`'s
> `return` at `:707` and let the outermost-exemption apply there" — is
> **numerically inert** and this design does **not** adopt it. `Li2omx2` sits at
> call-graph depth 3; it is **not** the chain's outermost region (the `B1m.h`
> driver at depth 0 is). Exempting `Li2omx2`'s return means the emitted patch
> rounds the dd result back to `TOutput` (double) *at the return*, i.e. **before**
> the near-equal cancellation `dilog4 - dilog5` that happens in the caller
> (`B1m.h:240`). The value the chain exists to protect is destroyed one line
> before it is used. That is the same inert-patch class Phase 2b/2c were built to
> kill (`promotion_no_op` / delta==baseline), wearing a new hat. The sound fix is
> to keep the value dd **across** the return — return-type widening of the
> per-integral variant (§2 rule (c), §7) — not to exempt the truncation. Every
> section below is written against that correction.

---

## 1. Problem statement

### 1.1 The class

`chain_promote` widens a **subset of the value flow** — the lines the dominant-
cascade selector emitted — and rounds back to caller precision at the boundary of
that subset. Wherever the subset boundary falls in the middle of a value's
lifetime (rather than at the value's true exit to the shared driver output), the
dd value is truncated to double *before* the cancellation it was widened for, and
the fix is inert. The interior 2d-B gate (`chain_write_truncation`) correctly
observes the truncation and rejects the patch (`patcher_status: write_truncation`).

Every "Blocker" in this arc has been the **same gap in a new disguise** — the
subset boundary sitting one construct short of the value's real consumer:

| disguise | where the subset boundary falls | example |
|----------|--------------------------------|---------|
| **A — frame-local carrier** | a local *declared outside* the chain line-set, written by one chain link and read by another | B10 `ddilog` `Y,S,A` (`kokkosUtils.h:157`) |
| **B — return-value severance** | a function *return* whose value a caller chain line consumes | B10 `Li2omx2` `:704`→`:707`→`B1m.h:240` |
| **C — consumer-store severance** | an in-frame *extract / output store* the selector omitted | B13 `ga34*`→`ql::Real`; B14 `fac`→`res(i,k)` |

> Note on nomenclature: this document's **A / B / C** are three disguises of ONE
> gap (the chain-scope boundary severing dd value flow). They are **distinct**
> from the auto-memory's "Blocker B" = *B12's LLM shim-generation defect*
> (re-declared promoted locals, malformed unary `operator+`). That is a separate
> axis; §8 addresses it and explains why it stays orthogonal.

Blocker A landed (`@e1971d0`, carrier decl-widening; 671 tests). The Subtask-5
re-run (`TIER_B_STAGE1_2026-07-25.md`) proved the carrier fix works — B10's
`Y/S/A` seam cleared, the chain advanced `105s→287s` through full shim generation —
and then surfaced disguises **B** and **C** as the next residuals. That is the
tell: local fixes keep exposing the next exclusion because each one widens *one
more construct* rather than reformulating scope to the value flow itself.

### 1.2 The three concrete residuals (worked examples, verified against source)

**B10 — disguise B (return-value severance).** Chain
`cascade_B10_612f1391_494252c4`, 10 lines:
`B1m.h:{227,240,241}` + `kokkosUtils.h:{174,177,199,212,702,703,704}`.

```
kokkosUtils.h:688  TOutput Li2omx2(TScale const& v,w,x,y) {        // returns TOutput (double complex)
kokkosUtils.h:691    TOutput prod, Li2omx2;                        // decl OUTSIDE chain line-set
kokkosUtils.h:704    Li2omx2 = -TOutput(pi2o6 - ql::ddilog<...>(arg2)) + lnarg*lnomarg - ...;  // CHAIN line: dd write
kokkosUtils.h:707    return Li2omx2;                               // NON-chain: rounds dd -> TOutput
...
B1m.h:236   const TOutput dilog4 = ql::Li2omx2<...>(m3sqbar, tabar, mp2sq, msq);   // NON-chain: receives return
B1m.h:237   const TOutput dilog5 = ql::Li2omx2<...>(m3sqbar, m4sqbar, si, msq);    // NON-chain: receives return
B1m.h:240   res(i,0) = dilog4 - dilog5 - 2*dilog1 + 2*dilog2 + 2*dilog3 + ...;     // CHAIN line: THE cancellation
```

The near-equal cancellation is `dilog4 - dilog5` at `B1m.h:240` — in the driver,
the chain's **outermost** (depth-0) region. But `dilog4`/`dilog5` are `Li2omx2`
**return values**, received into caller locals `B1m.h:{236,237}` that are **not in
the chain line-set**. The dd value must survive: `:704` write → `:707` return →
`B1m.h:{236,237}` locals → `:240` cancellation → `res(i,0)` store (the true exit).
The selector cut the chain at `:704`; carrier-widening (Blocker A) cannot help
because `Li2omx2` is written on a chain line but read only at a **non-chain**
`return` — it fails carrier condition 2. Genuine, correctly-diagnosed
`write_truncation`.

**B13 — disguise C (consumer-store severance).** Chain
`cascade_B13_79fc5b8f_f080f240`, 8 lines: `B2m.h:{300,301,305,306,355,533}` +
`kokkosUtils.h:{212,702}`.

```
B2m.h:282   TOutput root, ga34p, ga34pm1, ga34m, ga34mm1;   // decl OUTSIDE chain line-set
B2m.h:301   ga34m   = TOutput(+p3sq + m4sq - m3sq) - root;   // CHAIN line: near-equal (A - root) dd write
B2m.h:311   x34m = ql::Real(ga34m);                          // NON-chain: extract real part -> TMass (double)
...
B2m.h:331   dilog3 = ql::Li2omx2<...>(m3sqbar, x34m, tabar, x34mm1);  // NON-chain: x34* enters callee as arg
B2m.h:355   res(i,0) = ... - dilog2 - dilog3 - dilog5 - dilog6 + ...; // CHAIN line: dilog sum
```

`ga34*` are the cancellation victims (`A ± root`, near-equal when `root ≈ A`).
Their decl (`:282-283`) is outside the chain; they are read only at the
**non-chain** extracts `x34* = ql::Real(ga34*)` (`:309-317`) → **not carriers**.
The dd subtraction at `:301` lands in a double `ga34m` → truncates → genuine
`write_truncation`.

**B14 — disguise C.** Chain `cascade_B14_3429b1d4_01bf2ff3`, 3 lines:
`B2m.h:{401,578}` + `kokkosUtils.h:1208`.

```
B2m.h:396   TOutput fac;                       // decl OUTSIDE chain line-set
B2m.h:401   fac = TOutput(...) * cxs[0]/(cxs[1]*cxs[2]) * xlog;   // CHAIN line: dd write
B2m.h:404   res(i,1) = fac;                    // NON-chain: output store
B2m.h:405   res(i,0) = fac * wlogtmu;          // NON-chain: output store
```

`fac` is written on a chain line, read only at the **non-chain** kernel output
stores `res(i,{0,1})` → not a carrier → genuine `write_truncation`.

### 1.3 Why local fixes keep exposing the next exclusion

Each fix so far enumerated **one construct** the promotion must widen: region
bodies (2a), pre-declared writes (Case-B), complex containers (2d-A), carrier
decls (Blocker A). The value flow is a *graph*; enumerating constructs chases its
edges one kind at a time, and there is always one more edge kind (a return, an
extract, an out-param) at the frontier. The fix is to stop enumerating constructs
and instead compute the **transitive closure of the dd-carrying value flow** — so
the promotion envelope is defined by *where the value goes*, and the only
truncation is at the value's true exit (the driver's output store / an out-param
that leaves the chain's function set).

---

## 2. The closure algorithm

### 2.1 Objects

For a chain `C` with seed line-set `L` (today's dominant-selector output) spanning
a set of functions `F` on the call graph rooted at the entry point:

* a **frame** is one function `f ∈ F`;
* a **carried value** is a (frame, local-name) or (frame, return) or
  (frame, out-param) that must hold dd;
* a **closure edge** connects two carried values by same-frame dataflow
  (write→read of a local) or by a **chain-internal call edge** (a callee's return
  → the caller's receiving local), i.e. a call edge whose callee is in `F`.

The closure is the least fixed point of the carried-value set, seeded by `L` and
grown by three rules to `dryness`.

### 2.2 Seed

```
seed = { (frame(ℓ), name) : ℓ ∈ L, name written or read on ℓ that is a
                            floating / template / complex-container value }
```

i.e. exactly what today's per-region `promote_region_block` already promotes on
the seed lines. The dominant-chain selector is henceforth **only** a seed
generator (§5).

### 2.3 Closure rules (fixed point)

Let `W` = set of carried values (initialised to `seed`). Repeat until `W`
unchanged:

**(a) Intra-frame carrier** *(generalises Blocker A)*.
For a local `v` in frame `f` whose declaration lies **outside** `f`'s chain
line-set: if `v` is **written** by any carried value's line in `f` **and read on
any other line inside `f`**, then `v`'s decl widens and every read of `v` in `f`
joins `W`.
*(Blocker A's strict-carrier test required the read to be on another **chain**
line. This drops that restriction to "any line in the same frame" — the exact
generalisation that captures B13's `ga34*` (read at the non-chain extracts) and
B14's `fac` (read at the non-chain output stores).)*

**(b) Forward flow.**
For a line that **reads** a carried value and **writes** to (i) another local,
(ii) an out-parameter, or (iii) a `return`: the written entity joins `W`; recurse
on it. Sub-cases:
* → local: rule (a) widens its decl.
* → out-param / return **whose consumer is inside the chain's function set**: this
  is a **chain-internal boundary**, handled by rule (c).
* → out-param / return **whose consumer leaves the chain's function set**, or a
  kernel-output container (`res(i,k)`): this is a **chain exit** — the value's
  designed landing at caller precision. It does **not** widen further; it becomes
  a *designed-exit* marker for the gate (§3).

**(c) Chain-internal return/out-param propagation** *(cross-frame, the new
capability B10 forces)*.
If rule (b) reaches a `return X` (or an out-param write) in a callee frame `g ∈ F`
whose returned value is consumed by a carried value in a **caller frame** `f ∈ F`
(the call edge `f → g` is chain-internal), then:
1. `g`'s **variant return type** widens to the dd type (realised on the
   per-integral variant `g_<integral>`, never the shared original — §7);
2. every call site `f → g_<integral>` now produces a dd value, so the caller's
   **receiving local** re-enters rule (a) in `f` and widens.
Recurse: if `f` in turn returns that local, apply (c) up the call edge `f`'s
caller. The recursion climbs the (acyclic) call graph and terminates at the
outermost frame, whose store to the shared output is the chain exit.

Rule (c) is the load-bearing addition. It is what makes the closure follow the
value **across** `Li2omx2`'s return into `B1m.h`'s cancellation, instead of
rounding at the return.

### 2.4 Refusals (terminal, before any tree mutation)

A carried value cannot be widened when it would require editing shared state or a
shared signature. These terminate the chain cleanly (no variants emitted), exactly
as Blocker A's two terminals do today:

* **decl is a function parameter of a frame in `F`** → `chain_carrier_unwidenable`
  (v1 does not rewrite input signatures — §8).
* **decl is a global / class member / output container**, or the value must enter
  a **callee not in `F`** as an argument (e.g. `x34*` passed into `ql::Real` /
  `ql::Lnrat`, whose signatures we will not touch) →
  `chain_carrier_external` / a new `chain_closure_escapes` (§8).

Return-type widening under rule (c) is **not** a refusal: the callee is a
per-integral *clone* (`Li2omx2_B10`), reachable only from the chain's own rerouted
call sites, so widening *its* return type is contained (§7 proves safety). This is
the crucial asymmetry — **outward return flow is in scope; inward parameter flow
is not** (§8).

### 2.5 Termination proof

The carried-value universe is
`U = (locals ∪ returns ∪ out-params) over the frames F`, which is **finite**
(`F` is finite — the seed spans finitely many functions, and rule (c) only
propagates along the finitely many chain-internal call edges; it never discovers a
new frame outside `F`). Each rule application only **adds** to `W` (monotone); a
carried value, a widened decl, and a widened return type are each recorded at most
once. Therefore `W` is a monotone sequence in the finite lattice `2^U`, and the
fixed-point iteration halts after `≤ |U|` rounds. Rule (c)'s cross-frame recursion
climbs the **call graph, which is a DAG** in qcdloop (no recursive special
functions on these chains — verified: `ddilog` Chebyshev, `Li2omx2`, `kfn`,
`ltspence`/`cspence` are all straight-line/iterative, not self-recursive), so the
climb has bounded depth `≤ |F|`. **QED.**

### 2.6 Complexity

Per closure round: a token scan of each frame's source, `O(Σ_f |f|)` where `|f|`
is the frame's token count. Rounds `≤ |U| = O(Σ_f locals_f)`. So
`O((Σ_f |f|) · (Σ_f locals_f))` per chain — quadratic in a **bounded** quantity
(Item 7: chains are 3–10 seed lines over ≤ 5 frames of ≤ ~50 lines each). In
practice a few thousand token comparisons per chain: negligible against one LLM
shim generation. Reuses `region_scan` + `CallGraph` (no new source-analysis
machinery), exactly as `compute_carrier_closure` does today.

### 2.7 Worked traces

**B10 (rules a → b → c → exit).**
```
seed              : {Li2omx2 body @704, ddilog body @{174,177,199,212}, B1m.h cancellation @{240,241}, fac @227}
(a) in ddilog     : Y,S,A decl @157 widens          [Blocker A, now a special case]
(a) in Li2omx2    : prod,Li2omx2 decl @691 widens   (Li2omx2 written @704, read @707)
(b) in Li2omx2    : @707 `return Li2omx2` reads carried Li2omx2, writes RETURN -> join
(c) f=B1m.h g=Li2omx2 : Li2omx2_B10 return type -> ddcomplex;
                        callers B1m.h:{236,237} produce dd -> dilog4,dilog5 decls widen (rule a in B1m.h)
      cancellation @240 : dilog4 - dilog5 now dd  [the point of the whole chain]
      exit @res(i,0)    : store to res(i,k) container -> DESIGNED EXIT (round to caller precision)
```
The dd value now flows unbroken `ddilog → Li2omx2 → B1m.h cancellation → res`, and
the only truncation is the final `res(i,0)` store — the chain's designed exit.
This is the qcdloop-with-dd-throughout behaviour Reet verified, reproduced
regionally. **Within-function closure alone (rules a,b, no c) does NOT achieve
this** — it stops at `Li2omx2`'s return, which still rounds to `TOutput`; only
rule (c) carries dd across. This is the correction from the header box.

**B13 (rules a → b → exit; no rule (c) needed, one refusal at the frontier).**
```
seed          : {ga34* near-equal writes @{300,301,305,306}, dilog sum @355, ...}
(a) in B2m    : ga34* decl @282-283 widens (written @301.., read @309-317) -> dd subtraction recovered
(b) @311..    : `x34* = ql::Real(ga34*)` reads carried ga34*, writes x34* (local)
                x34* would join, BUT its producer is `ql::Real` (a callee NOT in F,
                returning a real projection). Two possibilities, decided by §3:
   - if the ga34* cancellation residual fits caller precision (Item 7: B13 loss 7.3
     < 15.95 double digits), the extract to double x34* is a DESIGNED EXIT for that
     sub-value: the cancellation is already recovered in dd @301; rounding the
     real projection to double loses nothing double could have kept. Closure stops.
   - if it does not fit, x34* -> ql::Real -> callee-arg is a `chain_closure_escapes`
     refusal (would need ql::Real's signature widened — §8).
```
The design **emits** B13 (ga34* recovered at dd, extract as designed exit) and lets
the positive-lift gate measure whether the double extract suffices. It does **not**
promise B13 lifts — see §7 falsification.

**B14 (rules a → b → exit; the clean within-frame case).**
```
seed        : {fac write @401, ...}
(a) in B2m  : fac decl @396 widens (written @401, read @404,405)
(b) @404,405: `res(i,{0,1}) = fac ...` reads carried fac, writes res(i,k) container -> DESIGNED EXIT
```
`fac` is computed at dd (its cancellation recovered); `res(i,0)=fac*wlogtmu` rounds
to caller precision at the kernel output — the designed exit. Clean win, no rule
(c) required (fac and its stores are in one frame).

---

## 3. Interior-gate reformulation

### 3.1 Today

`chain_write_truncation(region_meta, …)` (`chain_promote.py:127`) skips the chain's
**outermost region by call-graph depth** (`min(m["depth"])`) and runs the
per-region `boundary.write_truncation_inert` on every interior region; it fires if
any interior region truncates all landings to caller precision. The exemption key
is **depth**. That is wrong for two reasons the residuals expose:

* a value's real exit (`res(i,k)`) may be written by a function at **interior
  depth** (a callee coefficient routine writing the out-param `res`) — today it is
  not exempt (B14's `fac` fired at interior depth even though its consumer is the
  kernel output);
* a truncation at an interior **return** may be a genuine severance (B10's `:707`)
  or, after rule (c) widens the return type, not a truncation at all — depth
  cannot tell these apart.

### 3.2 New

Fire iff a **carried value is truncated to caller precision at a line that is not
the closure's designed exit**, where *designed exit* is defined structurally, not
by depth:

```
designed_exit(landing) :=
      landing writes a kernel-output container (res(i,k)) or an out-param that
        leaves the chain's function set F                          # true exit
   OR landing is a `return` of a frame whose return type the closure WIDENED
        under rule (c)                                             # carries dd, no truncation
   OR landing's narrowed value is not read by any later carried value
        (its downstream, until the next chain line, passes only through
         lines the closure did not widen) AND its cancellation residual
         is already resolved                                       # benign extract (B13 x34*)
```

Concretely, the gate's condition rewrites from

```
FIRE  ⇔  ∃ region r with depth(r) > min_depth
             ∧ write_truncation_inert(r, …, carrier_names)
```

to

```
FIRE  ⇔  ∃ landing t of a carried value
             ∧ truncates_to_caller_precision(t)
             ∧ ¬ designed_exit(t)
```

`carrier_names` (Blocker A) generalises to `closure_names` — the full carried-value
set — and threads through `_compute_promotion` / `promote_region_block` /
`write_truncation_inert` exactly as today (§4). The per-region
`write_truncation_inert` detector is **unchanged**; it is simply applied to the
closure's non-exit landings instead of to depth-interior regions.

### 3.3 Correctness (the gate stays a real gate)

* **Still rejects genuine round-back-to-double at a real boundary.** If rule (c)
  *cannot* widen a return the value needs (refusal — the callee escapes `F`, or
  the value flows in via a param), then that return is not a designed exit and the
  gate fires. B10 with rule (c) *disabled* still rejects at `:707` — the
  correction from the header box is enforced by the gate, not merely asserted.
* **Still permits the outermost-exemption where it should apply.** The old
  depth-min region's store to `res(i,k)` is a kernel-output container → matches the
  first `designed_exit` clause → exempt. The reformulation **subsumes** the
  current exemption (every case it exempted, the new rule exempts) and **adds** the
  interior-depth output store (B14) and the rule-(c) widened return (B10).
* **Conservative on uncertainty.** The third `designed_exit` clause (benign
  extract) exempts only when the downstream is provably non-carrier *and* the
  residual is resolved; when it cannot be proven, the gate does **not** exempt →
  falls back to today's reject. So the reformulation cannot turn a currently-
  correct rejection into a false accept. (And there is nothing to regress on the
  accept side: Tier-B has **zero** chain accepts today.)
* **No gate is weakened.** As with Blocker A, the gate logic is unchanged; it sees
  a corrected classification (closure members are no longer read as truncating
  sinks because their decls/returns are widened). The only *new* exemption is
  structurally justified (a widened return carries dd; a kernel-output store is the
  designed answer landing).

---

## 4. Composition with the existing gates

Closure-scoped chains slot **below** the acceptance machinery — they change *what
patch is emitted*, not *how it is judged*. The judging gates are unchanged in
policy; they simply now receive a non-inert candidate to measure.

* **Kernel-scope acceptance gate** (`afd334c`, `9a65c81`;
  `solver.py` `cand_per_kernel` / `curr_per_kernel`, `target_kernel`). A
  closure-scoped candidate still carries `target_kernel = K` and is gated against
  `K`'s own baseline + accumulated floor. Closure changes only *which lines the
  candidate touches*; the per-kernel min is measured once per candidate as before.
  Bigger envelopes do **not** cross kernels (variants stay per-integral). **No
  change** to the kernel-scope gate.
* **Positive-lift acceptance** (`solver.py:64`, `MIN_CHAIN_LIFT`, Reet 2026-07-24).
  Unchanged. This is now the *first gate that ever runs on B10/B13/B14* — today
  they die at the Patcher (`apply_failed`) upstream of it. The design's whole point
  is to let the true precision outcome reach this gate. A closure that emits an
  inert patch (rule (c) refused, cancellation not recovered) is caught here as
  `chain_no_lift` — a real measured terminal, not a spurious Patcher reject.
* **`chain_promote` coordination envelope** (2c `chain_promotion_no_op`, 2d-B
  `chain_write_truncation`). `chain_promotion_no_op` is unchanged (fires iff the
  whole closure promotes nothing). `chain_write_truncation` is reformulated per §3.

### What changes, by module

| module | change |
|--------|--------|
| `agents/patcher/chain_promote.py` | `compute_carrier_closure` → **`compute_value_closure`** (rules a/b/c fixed point, §2); emits `closure_names`, decl-widen records, **and return-type-widen records** (rule c). `chain_write_truncation` reformulated to the designed-exit condition (§3). New refusal `chain_closure_escapes`. |
| `agents/integrator_base/boundary.py` | `carrier_names` param → `closure_names` (same threading through `_compute_promotion`/`promote_region_block`/`write_truncation_inert`). `write_truncation_inert` gains a `designed_exit` predicate hook. Add `widen_return_type_line` (mirror of the existing `widen_decl_type_line`, §7). |
| `agents/patcher/fanout.py` | `VariantSpec` gains a `return_widen: ReturnWiden \| None` field (orig→dd return type); `render_variant` rewrites the variant's return-type token in the same descending-line pass as `promotes`/`carrier_decls`; `merge`/`to_json`/`from_json` round-trip it. `CarrierDecl` generalises to `ClosureDecl` (unchanged shape). |
| `agents/patcher/dispatch.py` | `_gen_chain` wires the new `chain_closure_escapes` terminal and the return-widen records into the variants (analogue of the existing carrier-decl attach loop, `chain_promote.py:651`). |
| `agents/patcher/result.py` | keep `CHAIN_CARRIER_UNWIDENABLE` / `CHAIN_CARRIER_EXTERNAL`; add `CHAIN_CLOSURE_ESCAPES` (value must enter a callee ∉ F, or a shared helper signature) + `err.kind`. |
| `agents/strategy/*` | selector demoted to seed generator (§5); `ChainRecord.lines` remains the seed. `dispatch.py` `VIA_CHAIN` and `RemediationIntent.chain_lines` unchanged (still the seed line-set). |
| `agents/validator/scorer.py` | **no logic change.** `canonical_region_id` / `rung_from_kind` already key on `(file, line, rung)`; closure variants share the seed region's id exactly as fan-out variants do today (`scorer.py:25`). The per-`(region_id, rung)` delta manifest just has more non-inert cells to reduce. |
| `agents/chain_integrator/*` | prompt rule **C10** (carrier invariant) generalises to **C11** (closure invariant): a value carried across a widened return/out-param stays dd end-to-end; never re-narrow at a return whose type the emission layer widened. |

---

## 5. What migrates vs what dissolves

### Dissolves

* **"carrier vs non-carrier" framing.** A carrier was "written by one chain link,
  read by another." That is exactly rule (a) restricted to reads-on-chain-lines.
  Under the closure, a carrier is just a rule-(a) name whose reads happen to fall
  on chain lines — no longer a distinguished category. `BLOCKER_A_CARRIER_DESIGN.md`
  §2's four-condition strict-carrier test **dissolves** into rule (a)'s two
  conditions (decl outside line-set; written-and-read in the frame).

### Migrates (code physically moves)

* **Dominant-chain selector → seed generator.**
  `ranking.build_chain_dd_queue` + `characterization.load_chains`/`ChainRecord`
  keep their jobs (identify COMPUTED cascade chains, rank by predicted lift) but
  their output `ChainRecord.lines` is now *only the seed* handed to
  `compute_value_closure`. **New interface:** unchanged type
  (`list[RegionTarget]`), new contract ("a seed, not the promotion envelope").
  `ChainManifest.lines` stays the seed; the closure computes the envelope inside
  `chain_promote` before emission. No selector code changes — only its documented
  role.
* **Fix A carrier-closure module (Subtasks 1–4) → special case of rule (a).**
  `compute_carrier_closure` → `compute_value_closure` **in the same file**
  (`chain_promote.py`). `_local_decls` / `_DeclStmt` / `_carrier_dd_type` /
  `_names_read_in_region` / `_region_depth` **survive verbatim** — they are the
  primitives rule (a)/(b) call. `CarrierClosure` grows `return_widens` alongside
  `decl_widens`. `CarrierDecl`/`carrier_decls` (fanout) → `ClosureDecl`/
  `closure_decls` (rename; shape unchanged). `widen_decl_type_line` (boundary)
  survives; `widen_return_type_line` is added beside it.

### Untouched (scope-agnostic mechanical layers — Phase 2f Layers 0–5)

The layers that turn a *line/decl set* into emitted variants act on **whatever set
the closure hands them**; they have no notion of "chain scope" and need no change:

* call-graph resolution + path enumeration (`call_graph.py`, incl. the Layer-6
  template-extent fix `@41f0391`);
* variant naming + collision check (`variant_naming.py`, `assert_no_collisions`);
* per-region promotion transform (`boundary.promote_region_block`);
* shim generation + merge (`chain_integrator`, `shim_merge.py`, `_merge_into_file`);
* topological callee-before-caller emission + reroute (`_topo_order`,
  `_accumulate_region_specs`, `_reroute_in_function`).

These stay green. Return-type widening is the one place a mechanical layer grows a
field (`VariantSpec.return_widen`), but the *emission* reuses the existing
descending-line edit pass.

### Tests: rewrite vs stay green

* **Rewrite:** `test_chain_promote.py` carrier tests
  (`test_real_b13_has_no_carriers`, `test_real_b14_has_no_carriers`, the B10
  carrier-fires test) — their *assertions invert*: B13/B14 now have **closure
  members** (ga34*/fac decls widen), and B10 grows a **return-widen** record. The
  test *scaffolding* (real-source fixtures) is reused verbatim.
* **New:** rule-(c) return-propagation unit tests; `designed_exit` predicate tests
  (kernel-output store, widened return, benign extract, genuine severance);
  `widen_return_type_line` tests; an e2e integration test asserting B10 emits a
  dd-returning `Li2omx2_B10` and a dd `dilog4/dilog5` in the caller.
* **Stay green:** all Layer 0–5 tests; `boundary` per-region transform tests;
  scorer tests; the region (non-chain) fan-out suite; the whole non-chain Patcher
  path.

---

## 6. Cost surface (stated honestly)

### 6.1 Bigger chains → shim surface → Blocker-B stress. **Stance: deterministic
post-processing sweep; no chain-size cap.**

The closure grows the *number of frames/lines* an envelope touches, hence the
number of dd shims the chain integrator (LLM) generates. That directly stresses
the memory's "Blocker B" (B12: the LLM re-declared already-promoted locals
`p3sq__ff/m3sq__ff`, emitted a malformed unary `operator+`). More shims = more
draws against that defect.

Three options were considered:

1. **Prompt hardening** — already partially landed (`@97fca0a` R3 discipline,
   `@d14e41b` sanitisation). Necessary but not sufficient: it lowers the per-shim
   defect *rate*, but the closure raises the shim *count*, so the expected
   defect count per chain is roughly flat — not a scaling answer.
2. **Bounded chain-size cap with fallback** — **rejected as the primary lever.**
   A cap reintroduces exactly the subset boundary this design exists to remove:
   capping the closure means rounding dd→double somewhere in the middle of the
   value flow again → the inert-patch class returns under a size threshold. A cap
   is philosophically incompatible with closure-scoping. (It survives only as a
   *safety backstop*: if a closure exceeds, say, 8 frames or 60 lines, abort with a
   diagnostic `chain_closure_oversized` rather than emit a giant fragile patch —
   a circuit breaker, not a design choice.)
3. **Deterministic post-processing sweep** — **committed.** The closure and all
   widening (decl, return-type, boundary) are **deterministic**, source-derived
   emission — the LLM's remaining job is only the per-helper dd shim *body*. A
   deterministic normalisation pass over each generated shim (drop re-declarations
   of already-promoted locals; canonicalise unary operators; both are exactly
   B12's two defects) fixes the Blocker-B class **at the source** and its cost is
   `O(shim size)`, so it scales with the closure. This moves robustness from
   "hope the LLM behaves N times" to "normalise N times deterministically."

**Justification:** the design's own thesis — determinism is the lever that made
carrier-widening work — extends to gen robustness. The right response to "more
shims" is "make each shim's acceptance deterministic," not "generate fewer shims."

### 6.2 Runtime cost delta on the 21-integral consolidated driver

The closure widens more locals/returns to dd *within the chain's already-dd
frames*. dd scalar ops cost ~8–20× a double op; the closure roughly **2–3×** the
widened line count per chain (adds decls, returns, caller-receiving locals) but
does **not** widen new *functions'* worth of arithmetic — the special-function dd
cost (`ddilog`/`Li2omx2`/`kfn`) is already paid the moment the chain is dd; the
closure adds a handful more dd mults per call in the same frames.

Estimate (explicitly an estimate — **no chain has ever built to completion**, so
this is not measured):

* **10 STABLE integrals**: untouched (no chain) → **0%**.
* **4 Group-A integrals** (B10/B12/B13/B14): the dd-active frames grow ~20–40% in
  dd-op count; those frames are a fraction of per-integral runtime → **~+5–15%**
  per affected integral evaluation.
* **7 Group-B integrals**: if attempted, similar per-frame growth, but they are
  dd-*insufficient* (§8) and should not be accepted — cost is incurred only during
  the (rejected) measurement, not in the shipped driver.
* **Consolidated 21-driver aggregate**: bounded by the Group-A share → **low
  single-digit percent** on the shipped, accepted set. Confirm at re-run
  (`--dump-inputs` battery); do not trust this number until measured.

### 6.3 Migration path: **staged rollout (recommended)**

| stage | content | unblocks | risk |
|-------|---------|----------|------|
| **1** | rules (a)+(b) fixed point (generalise Fix A) + §3 gate reformulation | B13, B14 | gate change — validate "no gate weakened" on the currently-emitting chains **first** |
| **2** | rule (c): return-type/out-param propagation + `VariantSpec.return_widen` + caller re-closure | B10 | new emission capability; template return-type rewrite |
| **3** | deterministic post-gen normalisation sweep (§6.1) | Blocker-B robustness at scale | low |

**Why staged, not one refactor:** Stage 1 carries the riskiest change (the gate
reformulation could, if wrong, turn a correct rejection into a false accept), so it
must be isolated and proven on chains that already reach emission before the
cross-frame complexity of Stage 2 is layered on. Stage 2 is independently
measurable (B10 alone validates rule (c) and the core thesis). Stage 3 is
orthogonal and can land any time.

### 6.4 Is this "a week of focused work"? **No — honestly ~2–3 weeks.**

Stage 1 is genuinely Fix-A-sized (Blocker A was Subtasks 1–5). Stage 2 is
**comparable in size again** — a new `VariantSpec` field, return-type emission,
the caller re-closure loop, and the gate's designed-exit logic for widened returns
are each non-trivial and template-C++-fiddly. Stage 3 is smaller. Rough breakdown:

```
Stage 1  rules a/b + gate reformulation + test rewrites .......  4–6 days
Stage 2  rule c (return-widen emission + caller re-closure) ...  5–7 days
Stage 3  deterministic normalisation sweep ...................   2 days
e2e re-runs (B10/B13/B14) + triage per stage ................    2–3 days
                                                        total  ≈ 13–18 working days  (~2.5–3.5 weeks)
```

If the standing estimate given to Reet was "a week," that covers **Stage 1 only**
(which delivers B13/B14 emission + the gate fix). Full class-closure — including
B10, the headline case — is materially more. This is flagged per the design brief:
the migration cost is larger than a week, with numbers.

---

## 7. Success criterion

Concrete and measurable, per the Blocker-A precedent ("NOT all accept" — a *real
diagnostic terminal* is the bar; a measured lift is the win). Re-run: B10, B13,
B14 only (skip B12 — its gen defect is out of scope, §8), seed 12345,
sample_count 5000, entry BO, kernel-scope gate, `MIN_CHAIN_LIFT` (+0.5),
STOP-and-report.

**"Closure-scoped chains work" =**

* **B10 (the headline):** the emitted variant `Li2omx2_B10` **returns ddcomplex**,
  `B1m.h:{236,237}` `dilog4/dilog5` are **dd**, the cancellation at `:240` executes
  at dd, and B10 reaches the **positive-lift gate** with a measured
  `kernel_measured_lift > 0`. **Validation:** a lift toward Item 7's prediction —
  floor `9.88 → ~25.8`, predicted `+16` (chain-scope predicted `+18.4`). A measured
  lift of **≥ +8 digits** on B10's kernel counts as the design validated (dd
  ceiling ~26; anything materially positive proves rule (c) preserved the
  cancellation across the return, which is the whole thesis).
* **B14:** `fac` widened, `res(i,0)=fac*wlogtmu` as the designed exit; reaches the
  lift gate. **Validation:** measured lift toward floor `5.21 → ~21.2` (predicted
  `+16`); **≥ +8** counts.
* **B13:** `ga34*` recovered at dd; the `ql::Real` extract as designed exit.
  Two acceptable outcomes: (i) measured lift toward `8.62 → ~24.6` if the double
  extract suffices; **or** (ii) a precise `chain_closure_escapes` terminal naming
  the `ql::Real`/`Li2omx2`-arg frontier — a *diagnostic* win (today's opaque
  `write_truncation` becomes a named, correct scope-out). B13 is **not** promised a
  lift.
* **Group B (B15/B16/BIN0–4):** if run, the closure should make them **emit and
  measure** (no spurious `write_truncation`) and then correctly **fail the lift
  bar** or land below 10 digits — dd is insufficient by Item 7 (loss 24–47 > dd's
  ~32). "Closure works" here means *correct measurement*, not a lift.

**What falsifies the design:**

* B10 emits (rule (c) fires, `Li2omx2_B10` returns dd) but measures
  **`chain_no_lift`** → the cross-frame-return hypothesis is wrong; the residual
  is elsewhere (e.g. an intervening double narrowing rule (c) missed). This would
  be the strongest disconfirmation and would demand re-examining the value-flow
  model, not the emission.
* Any of B10/B13/B14 still terminates `apply_failed @ write_truncation` → the gate
  reformulation and closure did not compose (a bug, not a design refutation).
* A **previously-correct rejection flips to a false accept** (a chain that should
  round at double now measured as lifting spuriously) → the `designed_exit`
  predicate is too permissive; §3.3's conservatism failed.

---

## 8. What this design does NOT solve (surgical)

1. **B12's LLM shim-generation defect (memory's "Blocker B") — orthogonal.**
   B12 dies at `llm_gen_failed` (re-declared promoted locals, malformed unary
   `operator+`), *upstream* of any scoping question. Closure-scoping neither helps
   nor needs it; it makes B12's class **more likely** (more shims), which §6.1's
   deterministic sweep addresses. B12's precision path is *not* known to be a
   closure problem and is explicitly out of this design's scope. (If, after the
   sweep, B12 still fails to *lift*, that becomes a fresh measurement question —
   Item 7 rates B12 dd-sufficient, floor `3.69 → 19.6`.)

2. **Inward cross-function value flow (through parameters) — the genuine next
   class.** The design carries dd **outward** across a return (rule (c), safe
   because the callee is a per-integral clone). It does **not** carry dd **inward**
   — a value that must enter a callee as a dd **argument** requires widening the
   callee's **parameter** signature. On a *shared* helper (`Li2omx2`'s
   `TScale const&` params, `ql::Real`, `ql::Lnrat`) that is either a shared-original
   edit (forbidden — would re-measure benign B8/B9, Item 7 §3) or a per-call-path
   *parameter-specialised* clone with every caller's argument expression rewritten
   — a strictly larger fan-out. v1 refuses (`chain_carrier_unwidenable` /
   `chain_closure_escapes`). **Reasoning for scoping out:** the cancellation
   patterns in this suite produce their victims as function *results* (dilog
   returns, `ga34* ± root`), combined in the caller — the flow is outward. Inward
   dd-argument flow does **not** arise for the Group-A cancellations (the special
   functions take kinematic doubles as inputs; only their *outputs* carry the
   cancellation). B13's `x34* → Li2omx2` is the one place it *could* bite, and only
   if the double extract proves insufficient (§7 outcome ii) — which the re-run
   measures. Parameter-specialisation is the correct **next** phase, scoped by
   evidence, not aspiration.

3. **Value flow that escapes the chain's function set `F` entirely.** A value
   stored into a shared/global container, or passed to a function the selector did
   not include, cannot be followed without *expanding* `F` (re-running selection) —
   which would break the closure's termination guarantee (§2.5 depends on `F`
   fixed). Terminal `chain_closure_escapes`. Widening `F` is a selector concern
   (chain-definition completeness), not a closure concern.

4. **Group B beyond-dd (B15/B16/BIN0–4).** dd's ~32-digit budget cannot cover a
   24–47-digit cancellation loss (Item 7). Closure delivers the *full dd value
   flow* for these integrals, but full dd is still not enough — they need quad or
   an algorithmic rewrite. Closure makes them *correctly measurable* (§7); it does
   not make dd sufficient. Out of Tier-B's dd scope by construction.

5. **Chain-definition completeness (the selector itself).** If the dominant-chain
   selector seeds a line-set whose value flow's true exit lies in a frame it never
   named, the closure faithfully follows the flow to the frontier of `F` and
   refuses (case 3) — correctly, but the *fix* is a better seed, not a bigger
   closure. This design assumes the seed's function set contains the value's exit;
   where it does not, that is a selector refinement, named and deferred.

---

## Appendix — invariants preserved from Blocker A

* Variants are **per-integral clones**; the shared original special function is
  never edited (Item 7 §3 correctness requirement). Return-type widening obeys this
  — it edits `Li2omx2_B10`, not `Li2omx2`.
* Refusals are computed **before any tree mutation** (no half-emitted chains).
* The gate logic is **unchanged**; only the classification it sees is corrected.
  No gate is weakened — §3.3.
* Conservative over-widening (multi-declarator siblings, potential/conditional
  writes) carries over: over-widening a same-type value never truncates.
