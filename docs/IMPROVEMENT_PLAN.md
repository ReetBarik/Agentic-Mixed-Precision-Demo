# Improvement Plan (2026-08-20)

> **Forward-looking document.** Unlike the rest of `docs/`, this describes planned work, not
> as-built behavior. Every factual claim about the current code was verified against source
> (file:line) and survived an adversarial review pass; design proposals are marked as such.
> Companion: the Tracked librarization plan, which lives in its own repo:
> [Tracked-Error-Propagation-Datatype `docs/TRACKED_LIBRARY_PLAN.md`](https://github.com/ReetBarik/Tracked-Error-Propagation-Datatype-Demo/blob/main/docs/TRACKED_LIBRARY_PLAN.md).

Seven objectives, then a dependency-ordered roadmap.

---

## Objective 1 — Region-level precision assignment (functions/lines), rung-pruned, cast-aware

### 1.A Build on the Phase 2e/2f solver path

Extend `agents/solver/` + `agents/per_integral_orchestrator/` + scorer cells — not the
historical `strategy_mode="region"` walk. The solver path is measure-then-decide: a
per-integral measurement fan-out produces measured whole-app digit outcomes per
(region, rung) cell; a pure queue builder excludes INERT cells
(`|delta_effective − baseline_delta_effective| ≤ 1e-18`, `agents/solver/queue.py`); a greedy
cheapest-first solve with region-locking and a regression-relative gate
(accept ⇔ `cand_min ≥ baseline_min − 0.5`) stacks patches on the accumulated tree (the
cumulative-diff adapter sidesteps `validate()`'s `accepted_patches` NotImplementedError).
The old region walk ran once at production scale (PIPELINE_v1: 231 iters, 152/152 accepts,
85 demotions) but is serial, order-dependent, and pre-qf.

Solver blockers to fix first:
- `run_solver_stage1.py:214-219` builds `ValidateResult` without per-kernel floors while
  `queue.py:120-135` sets `target_kernel` from single-element `integrals_scope` — a rerun
  rejects every scoped candidate as `kernel_scope_unmeasured`.
- The all-21 Stage-2 solve has never been executed (no `SOLVER_STAGE2.md` was ever committed).
- No cross-integral merge policy exists (`run_solver_stage2.py` explicitly does not merge trees).

### 1.B The principled rung prune ("never try float at tol 10")

No digit-ceiling-vs-tolerance prune exists today; predictions exist only for float/ff
(`predicted_rel_err_if_* = U_r × max_sensitivity`, `stability_reducer.py:911-912`).

Design (corrected after adversarial review):

1. **Emit `predicted_rel_err_if_<rung>` for all five rungs.**
   `agents/shared/bound_decomposition.predict()` already computes four from one sensitivity;
   add U_QF. **U_QF calibration requires a limb-printing probe**: flip candidates print
   single doubles (`boxGPU_vanilla.cpp:19-21`), so qf measurements pin at ~15.95 (QF_INTEGRATION
   measured exactly 15.81/15.96) — the probe must print qf limbs the way `boxGPU_dd.cpp`
   prints `hi|lo`, scored against the DD oracle (~3 digits of headroom over qf's ~28.9).
2. **One gate-aware feasibility function, screening-first**:
   `feasible(R, r, gate) ⇔ −log10(U_r × max_sensitivity(R))` clears the gate's bar.
   - This implies the intrinsic rung ceiling exactly when `sens ≥ 1` (the common case — a
     tol-10 run prunes float as desired). It is **not** universally sound to prune on the bare
     ceiling: `_cond_eff` floors at >0, not ≥1 (`stability_reducer.py:399-409`), so damping
     regions (sqrt cond 0.5, log cond 1/|ln x|) can legitimately have `sens < 1`.
   - **Gate-aware**: under the solver's regression-relative gate, compare against
     `baseline_min − 0.5`, not tolerance (tolerance is decorative in the solver path;
     with B12's 3.69 physics floor, "infeasible at tol 10" demotions legitimately pass).
   - Apply as a **build-skipping screen** with logged skip counts and the existing
     `STRATEGY_DISABLE_REPORT_PRUNES` kill switch — not an unappealable verdict (the model
     both over-predicts, per WI2's history, and under-predicts, per Objective 5's audit).
     This also unifies the WI2 hard-vs-telemetry split: both modes get the same screen in
     the same role.
3. Keep existing hard prunes: WI1 FP32-family range guard, `required_by` chain floors,
   `REGION_REALIZABLE` (until the qf integrator lands).
4. **Measurement ceiling**: the vanilla driver prints doubles, so candidate digits cap at
   ~15.95 — valid tolerances are t ≤ 15 through the current output contract.

### 1.C Cast-overhead-aware assignment

Ground truth: **no cast cost, no cast counts, no kernel-timing harness exist anywhere.**
The journal op vocabulary cannot represent casts; the only C++ timing artifact
(`runs/qcdloop_headers_full/timer.h`) is dead code. Casts are paid per region execution,
unhoisted, per-element for aggregates; chain interiors pay zero casts; the whole-TU flip
pays one narrow per output (the minimal-cast realization). Hand-derived entry/exit costs:
dd entry ~free / exit ~3 ops; ff entry ~4 ops; qf entry ~7 ops — small vs emulated per-op
costs (ff add=11, mul=32) unless crossings dominate short regions.

Three layers, cheapest first:
1. **Static net-gain objective**: a unified cost table (see 1-E) gains a `qf` column and
   per-family `cast_entry`/`cast_exit` rows. Crossings = (|reads| + |writes|) × executions,
   where executions come from **per-region ops-per-sample histograms or the exact trip counts
   latent in the id `#counter` suffixes** — not `reg["n"]/samples`, which over-counts by the
   ops-per-execution factor. Drop net-negative candidates pre-measurement.
2. **Structural minimization (the bigger lever)**: function-granularity regions (casts
   amortize; the fan-out clone machinery is the mechanism); fuse adjacent same-rung regions;
   hoist casts via the existing closure decl-widen (make the chain path the default multi-line
   representation — `chain_lines` plumbing exists at `intent.py:71-75`); add **direct ext↔ext
   converters** (today ff→dd rounds through double — a correctness hazard, not just cost).
3. **Timing gate (new work)**: per-integral chrono in the drivers, median-of-repeats, a
   `timing` column on scorer cells; report-first, gate later.

### 1.D Granularity + qf

- **Function-level regions are the next granularity**: characterizer emits function-scoped
  records (the merge machinery at `characterization.py:315-354` keyed on function is the
  template); patcher mechanism = fan-out clone with region = whole body.
- Per-line exists (solver cells are per-line); its work item is re-validation on the qf-era
  ladder plus statement-atomic slicing (line_injector's AST statement set).
- **Build the qf_integrator**, then drop `"qf"` from `REGION_REALIZABLE` (`models.py:82`;
  the guard code is ready — "holds the moment a qf integrator lands", `walk.py:142`).
  Evidence it matters: at TU level qf absorbed the entire dd set (QF_INTEGRATION_2026-08-13).

### 1.E Architecture-aware cost model (a pipeline input)

`ArchProfile` JSON (`{name, fp32_tflops, fp64_tflops, fma}`; ship `gb300.json` = 77.5/1.25
and a measured `cpu-serial.json`), consumed by a single unified cost table
(`agents/shared/cost_model.py`, absorbing `ratio_multipliers.json` and
`gb300_cost_model.py`, which contradict each other on native transcendental costs and lack
qf/cast entries):

    t(region, rung, arch) = Σ_op ops[op]·mult[rung][op] / thr[base(rung)]
                          + crossings·cast[rung] / thr[base(rung)]
    base(float|ff|qf) = FP32;  base(double|dd) = FP64

Consequence (verified arithmetic): on GB300, dd-add ≈ 8.8 units vs qf-add ≈ 0.39 (~20×),
and ff (0.14) undercuts native double (0.8); a CPU profile flips both. Therefore the
hardcoded orders — `LADDER`, `CORRECTNESS_WALK=(qf,dd)`, `_SPEEDUP_WALK=(float,ff)`, solver
`RUNG_RANK` — become **derived, per region** (op mix matters: log-heavy regions shift qf/dd
via the 2350/4100-class multipliers). Upshift picks argmin-cost among feasible rungs;
downshift requires net gain > 0 under the profile. Honesty clauses: no memory/occupancy/
register-pressure modeling (hurts multi-word types most); host-serial characterization vs
GPU numerics (FTZ, fma); ranking input only, cross-checked by the timing gate.

---

## Objective 2 — Algorithmic-rewrite phase BEFORE the precision walks

### Current inversion (verified)

Rewrites fire only after the dd ceiling (`walk.py:186-198`); the patcher marks cancellation
regions `awaiting_algorithmic_rewrite` and the walk then terminates without reaching the
reformulate phase (`dispatch.py:110-176`) — actively stranding the targets; `log_near_root`
has no catalog entry; the symbolic overlay's `suggested_rewrite` hints are consumed by
nothing; the rewrite path has never produced a committed accept.

### Why rewrite-first

Rungs cost runtime forever; rewrites shrink κ once — simultaneously shrinking the upshift
set and expanding the downshift set. Direct evidence of promotion-prediction failure: B14
chain, predicted +16.66 digits, measured 0.0.

### Phase 0 loop

Queue cancellation + log-near-root + cascade hotspots by −max_cond → LLM rewrite via
`PATH_LLM_REWRITE` upgraded with error-feedback and best-of-N (Objective 3) → gates:
build/smoke → validator (regression-relative global + per-integral floors) →
re-characterize → accept on digit or conditioning improvement → terminate on K dry rounds →
**algorithmic-ceiling certificate** (residual hotspots with κ = the precision walk's input).
Then run upshift/downshift on the post-rewrite report.

### Enabling work (corrected after adversarial review)

1. **Tracer op coverage is a hard prerequisite** (moved to Wave 1): `log1p/expm1/hypot` are
   not in the Tracked op vocabulary — an accepted identity rewrite gets wrapped opaque with
   cond=1, so a κ-gate would auto-pass mechanically. Kahan reads as *high* cond
   ("compensation doing useful work") — gate Kahan on measured digits and downstream rel_err,
   never raw cond.
2. **Blast radius is per-file, not per-integral**: rewrites in shared headers
   (`kokkosUtils.h`, `kokkosMaths.h`, `box_common.h` are instrumented) shift keys and
   conditioning for every integral touching the file. The loop needs a rewritten-file →
   affected-integrals map; shared-header rewrites degenerate to near-full re-characterization.
   Realistic per-round refresh: minutes (line-patch regen + rebuild + run + reduce).
3. Enablers: `--integral` filter on the tracked driver (~10 lines; per-integral mt19937(12345)
   reseed already supports isolation); fast_merge splice-by-integral; C8 re-derivation is
   **already automated** (`c8.py` + `derive_c8_patch`) — just re-run it per accepted rewrite.
4. No "cheap dd identity guard": on cond>1e15 regions a true identity evaluated at dd agrees
   to only ~(31.9 − log10 κ) ≈ 17 digits, and no region-local eval harness exists. The
   DD-oracle digit gate + tail battery + determinism hash remain the semantic gate.
5. Kahan adds flops — feed rewritten regions back through 1.C's net-gain model.

---

## Objective 3 — LLM staffing (team of Sonnets vs one Opus)

**Provenance settled: production traffic was live claudeopus47 via the Argo tunnel** (real
usage-object token accounting; tunnel-required runner; no mock code exists or ever existed).
The WAVE3 "stable local mock" sentence is a wording error. The record therefore stands:

- Accept rates 42% → 62% → 71% per attempt across runs; **0/79 failures were
  capacity/rate-limit/refusal**; 72/79 were one deterministic harness bug (TU shim
  redefinition collision — fixed by the canonical shim merge); 3 were the model *correctly
  refusing* a structurally unsound transform (the 3→6 retry bump chased this and was reverted).
- Retries at the shim/rewrite sites are feedback-free re-rolls (only
  `// regeneration attempt N` changes); only characterizer driver-gen feeds stderr back —
  and converges in ≤1 retry.
- All LLM calls serialized, single-candidate, no temperature, one global `ARGO_MODEL` knob;
  region iterations are build-bound (~25-37 s); current tu_only production mode uses zero
  LLM calls in the walk.

**Model capability is not the observed bottleneck.** Ordered levers:

1. **Error-feedback retries at the regional-shim and rewrite sites** (port the driver-gen
   multi-turn pattern). Also add a misgen-signature short-circuit for deterministic failures
   (WAVE3's actual lever 2).
2. **Parallel best-of-N with deterministic gate selection**: llm seams are injectable; first
   candidate to clear lints+build+smoke wins; diversity via distinct prompt framings.
3. **Model mix by A/B, not prior.** Pricing (Anthropic list, checked 2026-08-20): Opus 5
   $5/$25 per MTok, Sonnet 5 $3/$15 (intro $2/$10 through 2026-08-31) — Sonnet ≈ 0.6× Opus,
   best-of-2 Sonnet ≈ 1.2× one Opus per round; the decision variable is cost-per-accepted-
   patch. Hypothesis mirroring the precision ladder: cheapest-sufficient model first — Sonnet
   for rule-constrained, heavily-gated shim generation (strong verifier ⇒ pass@N favors
   cheaper models), escalate to Opus on exhaustion; Opus-tier for weakly-gated judgment sites
   (rewrites, chain promotion, whole-app driver synthesis). No LLM judges/committees.
4. Wiring: per-site model map; model id + candidate index in the attempts log; proxy must
   expose Sonnet ids (fix `setup_argo_proxy.sh` default `MODEL=gpt4o`).
5. A/B harness: `rerun_failing_regions.py` + the 10 `@pytest.mark.llm` tests + one
   per-integral measurement pass. Arms: Opus-single / Opus+feedback / Sonnet+feedback /
   Sonnet best-of-3+feedback. Metrics: accepts/attempt, tokens/accept, wall/accept,
   dd_untested residue.

---

## Objective 4 — Generalizing beyond QCDLoop

Audited reality: onboarding a new codebase is a 14-step manual process; six steps have zero
code support (snapshot vendoring; the consolidated tracked driver; its CMake; the
measurement drivers + output contract — `coeffs.py` hardcodes `N_COMPONENTS=6`; the
whole-app oracle; `tu_emit` PROFILES/conventions). Deepest coupling: the oracle (hand-written
`qcdloop@ddfun_enabled` fork with layout-specific staging surgery; whole-app ff/float
integrators `NotImplementedError`; dd a verify-only stub). `agents/` is not app-clean:
`runner.py` `MODULE_PRELUDE` hardcoded with no env override, `_APP_CMAKE_DIR`, `boxGPU_app`,
`validate.py` defaults, `tu_emit` boxGPU.h/box/ conventions, basename-only region ids.

Plan:
1. **TargetAdapter contract** (config + per-app module): header snapshot; kernel manifest;
   driver contract (INP/RES schema, seed/offset semantics, output component labels + an
   app-defined reduction — implement the scorer's doc-only identity default); build recipe
   (make `runner.py` respect `PIPELINE_MODULE_LIST` like `build_run` already does); oracle
   recipe; flop table; ArchProfile. Acceptance: zero executable app identifiers in `agents/`.
2. **Oracle strategy, ranked**: (a) template-instantiation oracle — productize the
   designed-but-unwired `baseline_spec {"kind":"instantiate_at"}` hook and turn `tu_emit`
   PROFILES into adapter data (removes the fork dependency for precision-templated apps);
   (b) TU-flip-built oracle; (c) MPFR/float128 shadow build (later); (d) Tracked bounds are
   NOT an oracle.
3. **LLM-assisted onboarding** for the worst zero-support step (whole-app tracked-driver
   synthesis, extending characterizer driver-gen) — but the existing bit-exactness gate
   cannot validate a synthesized driver's semantics on a new app (it compares two builds of
   the same driver); require an independent reference (the app's own tests / native-run
   comparison).
4. Portability items: **path-qualified region ids are NOT a cheap fix** — `/` is the
   scope-stack separator and the injector documents that a `/` in a value truncates the
   region; this is a coordinated tracer+reducer+format change (scoped properly, not Wave 1).
   Regenerate the stale Python 3.9 venv (≥3.10 is already documented). Kokkos: instrumentation
   is Kokkos-free, but promoted builds of a non-Kokkos target still gain a Kokkos
   include/link dependency (every extended-precision header includes `<Kokkos_Core.hpp>`) —
   acceptable if declared.
5. **Forcing function**: onboard a second small precision-templated codebase early;
   success = zero-support steps 6 → ≤2.

---

## Objective 5 — Tracked/report signal extraction (what's left on the table)

Verdict (adversarially confirmed): hard-gating consumption of the ~850 MB report reduces to
five region scalars + two ordering scalars + chain spans + tail offsets + the variables
lists (`region_local_vars` IS live-consumed by patcher intents). Everything else is computed,
merged at scale, and read by no decision.

### 5.A Live bugs (fix regardless of roadmap)

1. `log_parser.py:50` reads prov keys absent from v0.3 journals — the legacy per-variable
   rollup is silently empty. Disposition: **retire the legacy rollup**; the reducer's
   `variables{}` map is the one to keep and start consuming.
2. Six distinct 1/u-cap causes (log/sin/cos/atan2/add/sub-underflow) collide into
   `atan2_saturation` — add a `cap:"<cause>"` journal field (Tracked-library change).
3. NaN cond → JSON null → parsed as 0.0 = "maximally stable" (sin(0); overflow-poisoned
   atan2 at qcdloop magnitudes vanish from the signal). ±Inf clamps to 1.798e308 (forensic
   loss only — the float-range verdict is unaffected since DBL_MAX > FLT_MAX).
4. Gate-a leaks into max_rel_err/histograms (999-saturated-ops + 1 mul can classify
   `cancellation_cascade`); `atan2_saturation` regions land in neither queue — never demoted
   despite being demotion-friendly. Also: per-variable `max_amp ≡ max_sensitivity`
   (overwritten at `stability_reducer.py:692-694`).

### 5.B Merge-associative reducer upgrades (ranked)

1. **Argmax sample identity** per region stat — the report names each region's adversarial
   input directly (largely replaces the driver-rerun tail emitter; gives the validator a
   counterexample-first check and the patcher a repro).
2. **Amplification argmax paths** — rewrite targeting at the amplifier; cast-fence placement.
3. **Distributions, not maxima** (sens/cond/|val| LogHists; re-emit the rel_err_hist that
   finalize drops) → p99-based feasibility; FTZ-tolerant range decisions (today one transient
   subnormal in 100k vetoes the FP32 family).
4. Ops-per-sample histograms (cast-cost execution weight; trip counts also recoverable from
   id `#counter` suffixes).
5. Per-sample sink-error distribution per integral — a distribution of **modeled bounds**
   (not measured error): useful for ranking and model-consistency, not as an acceptance
   baseline.
6. Structural cascade frequency (span-set-hash rollup) + raw sample keys on chains.
7. The dead `prov_consts` union (reader exists as dead code).

### 5.C Report fields to start consuming (smallest exploits)

p99 in queue admission; **max_amp for cast-boundary gating** (error-at-output =
U_target × max_amp — the exact boundary-injection model; zero readers today);
**`build_chain_dd_queue`** (implemented, tested, zero callers — its COMPUTED-band gate would
have pre-classified the B14 failure); derived guards at load time
(`predicted_rel_err_if_qf` once U_QF exists; `ff_limb_ok = abs_val_min ≥ FLT_MIN_NORMAL·2^24`);
tail criterion metadata (use range-extreme offsets to dynamically test the WI1 prune — 14
regions statically forfeited in the 10k run); wire the chain range flag; delete dead weight
(`class_counts`, `top_regions_by_rel_err`, the legacy per_variable rollup) and fix the
triple full-report JSON parse.

### 5.D Tracer upgrades (Tracked-library roadmap; ranked)

1. **Shadow-precision execution** — measured per-op deviation at a lower rung; converts rung
   predictions from modeled (max-gated, swamping-blind — the library's own Kahan test asserts
   the blind spot) to measured; the only route to a per-region qf signal.
2. Op coverage `log1p/expm1/hypot/fma` (Objective 2's prerequisite).
3. Cap-cause flag + exact-tie marker + fetestexcept flags.
4. Complex source component id split (zero cost; Re/Im consumers currently conflated).
5. Assignment binds + leaf value records (kills the write-set source-scan hack; unlocks
   WI4/WI5 — "the largest unmined signal").
6. Branch-margin records (demotion-flipped branches are invisible to rel_err).
7. Journal interning (~10-30× smaller → per-sample retention feasible).

Inherited-model honesty: swamping blindness, mul/div κ=1, opaque κ=1, Kahan-reads-high-cond
are declared library limitations — feasibility margins (Obj 1) and rewrite acceptance rules
(Obj 2) must account for them explicitly.

---

## Objective 6 — Parallel characterization over samples (no journal materialization)

Ground truth (verified): the library is thread-clean (journal buffer, id counters, scope
stack all `thread_local`; zero shared mutable state); the tracked path never enters Kokkos
parallel regions (`ql::BO` is a pure per-index host call); each of the 21 integrals reseeds
mt19937(12345) — **integral-level parallelism is the cleanest unit**; the reducer consumes
FIFOs today with zero changes; merge is associative. Traps: workers must push their own
`integral=/sample=` scopes (main-thread scopes are invisible; unscoped ids collapse into one
pseudo-sample); `flush` truncates and writes only the caller's buffer; `clear()` resets id
counters. The binding constraint today is **RAM** (no intermediate flush exists; chunk=500
holds ~5 GB of records per worker).

Staged:
1. **`flush_and_clear(std::ostream&)`** (preserve counters) at sample-scope pop, streaming
   to FIFOs consumed by concurrent reducers. Zero journal bytes materialized; RAM per worker
   drops to one (integral, sample) — sub-MB average. **Shard-commit guard required**: a
   mid-stream crash yields clean EOF and a valid-parsing partial shard that `--resume` would
   silently reuse; gate shard commit on driver exit status or a samples_seen check.
2. **Thread-per-batch in one process** (~60-120 lines C++): work-stealing over the 21
   integrals, private Views, per-thread RNG + prefix refill, per-thread FIFOs; reducers:
   producers > 1 (Python json.loads is the ceiling). Values bit-identical; report
   partition-invariant except chain_id strings (already true under process chunking).
3. **Native C++ map-step reduction** (~500-700 lines) when 100k+ runs are routine: port the
   per-sample map only (DAG/amp/aggregation/cascades incl. their map-step class stamps); keep
   Python merge/finalize. Parity gotchas: replicate the NaN/Inf JSON clamps; compare parsed
   values. Differential testing is trivial (merge associativity: per-chunk equality implies
   any-partition equality).

Stages 1-2 also make Objective 2's per-rewrite refresh cheap.

---

## Objective 7 — Robust consumption of the extended-precision ladder (KEP)

KEP (`../kokkos-extended-precision-demo`) ratified a restructure arc on 2026-08-20
(`docs/UPSTREAM_PLAN.md`): repo rename dropping "kokkos"; namespace `Kokkos::Experimental` →
`eplib::` (placeholder); headers `third_party/include/*.hpp` → `include/EPLIB/*.hpp` with
compat wrappers at old paths; macro renames; packaging/version macros only at S6; numerics
frozen by a byte-identical gate — **the gate protects values, not spellings, and spellings
are what AMP consumes** (12 fact categories across ~43 Python files, ~30 test files, four
LLM prompts, one C++ string template). One KEP rename already happened
(`quad::ddfun` → `Kokkos::Experimental`) and left dual vocabularies in 3+ places.

Vendoring reality: hand-`cp` + hand re-application of nine local patch blocks (AMP is ahead
on 5 of 6 files, incl. the load-bearing hypot-safe complex `abs` the DD oracle depends on) +
a hand-edited `UPSTREAM.sha`; upstream history was rewritten once (dangling SHA); contrast
the clean git-subtree used for `third_party/tracked`.

Mechanism (layered):
1. **Pin now**: ask KEP to cut the planned `kokkos-native-freeze` tag (the arc has not
   started — there is a window); record content hashes, not just SHAs. Vendor the two
   license texts (trivial).
2. **Rung Registry**: one machine-readable per-rung descriptor (`agents/shared/rungs.py` +
   data file) consolidating all 12 fact categories — spellings **plus historical aliases**,
   namespaces, limb members, ctor/converter facts, `from_bits` formats, constant-fn naming,
   header set + macros, U constants, range family, math coverage, op-cost column, cast costs.
   All consumers become projections: `tu_emit.PROFILES`, integrator SPECs, `instantiation_gate`
   vocab, `constant_derive` emission strings, `shim_normalise` defaults, `chain_promote`'s
   prefix test (any-known-namespace, fail-loud on none), the oracle shim rendering, and the
   **system prompts as templates** with registry facts injected at build time. A KEP rename
   becomes a registry edit + probe run, not a 43-file sweep.
3. **Conformance probe suite**: compile-and-run probes per registry claim on every refresh —
   limb members, `from_bits` round-trips, ctor semantics, no-`operator double`, complex
   members/accessors/narrowing trap, constant functions, math coverage, and gb300's
   calibration invariants (ff add=11, dd mul=31) as a **cost-table freshness check** (nothing
   runs it today).
4. **Mechanize vendoring**: git subtree (like `third_party/tracked`) + local patches as
   `.patch` files applied by script + regenerated ledger. Better: **upstream the nine patches
   to KEP** (the freeze is lifted for the arc; the hypot `abs` fixes a bug KEP's own ledger
   marks latent for QF).
5. **eplib arc strategy**: build the registry **before S2 lands**; then the migration is a
   registry flip supporting old + new vocabularies simultaneously (which gate log
   classification needs anyway).
6. **Upstream asks** (same owner): the S0 tag; a machine-readable export manifest shipped by
   KEP (formalized into S6 packaging later); version macros earlier than S6; wrappers kept
   through ≥1 AMP refresh cycle; adopt the local patches; a stability ruling on the
   constant-fn naming (flagged provisional in the header itself).

Fix regardless: `instantiation_gate` has **no FloatFloat spellings** (an ff binding failure
STOPs #BB today); `chain_promote.py:595`'s literal prefix test silently stops guarding after
a namespace move; validator `dd_ref` defaults to a floating branch name; missing qf column
makes qf flop-weighting silently degenerate to op_count; the include-order shadowing trap
(one silent frozen-oracle incident already).

---

## Roadmap (dependency-ordered)

**Wave 1 — foundations**: `--integral` driver flag + fast_merge splice; `flush_and_clear` +
FIFO streaming with the shard-commit guard; the four reducer/parser bug fixes (5.A); tracer
op coverage (`log1p/expm1/hypot/fma`) + cap-cause flag; U_QF limb-printing calibration probe;
all-rung prediction emission + RegionRecord carriage; unified cost table (qf column + cast
rows); error-feedback retries + per-site model config; `runner.py` MODULE_PRELUDE env
override; regenerate the 3.9 venv; Rung Registry + conformance probes + subtree conversion
for `third_party/include` (Objective 7 — do before KEP's S2).

**Wave 2 — Objective 1 core**: qf_integrator; gate-aware feasibility screen in solver
candidate generation; cast-aware net-gain ranking with ops-per-sample weights; timing
harness + scorer timing column; stage-1 per-kernel-floor fix; first real all-21 Stage-2
solve + cross-integral merge policy; ArchProfile plumbed into walk orders/RUNG_RANK.

**Wave 3 — Objective 2**: Phase 0 rewrite-to-ceiling on qcdloop (catalog + log_near_root
entries; overlay hints wired; best-of-N + feedback; file→integrals invalidation map;
per-round re-characterization); then re-run upshift/downshift on the post-rewrite report.
Measurable claim: the qf/dd set shrinks and the float/ff set grows.

**Wave 4 — Objective 3 A/B** piggybacked on Wave 2/3 runs.

**Wave 5 — Objective 4**: TargetAdapter extraction + instantiation oracle + LLM driver
synthesis (with an independent correctness reference), validated by a second codebase.
Shadow-precision execution and the WI4/WI5 per-variable unlock slot in as long-poles.

**Parallel track**: the Tracked librarization (own repo) — see
[`docs/TRACKED_LIBRARY_PLAN.md` in Tracked-Error-Propagation-Datatype](https://github.com/ReetBarik/Tracked-Error-Propagation-Datatype-Demo/blob/main/docs/TRACKED_LIBRARY_PLAN.md).
Objectives 5.D and 6 land there and flow back to AMP via subtree.
