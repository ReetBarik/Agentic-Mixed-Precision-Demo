# Tier-B Stage-2 — closure Subtask 4: Patcher retry-budget bump for R4 #error variance (2026-07-26)

Single-knob gen-robustness change (Subtask 3 proposed fix (1), the smallest-blast-radius
unblock): bump the Patcher integrator retry budget 3→6 to absorb the LLM
non-determinism Subtask 3's STOP #F diagnosed on the `ql::Lnrat<ddouble>` R4 `#error`
"requires manual classification" escape hatch. No new classification logic, no
normaliser expansion, no gate/closure/design changes — the retry budget is the single
knob turned. Plus a diagnostic-only R4 `#error` per-attempt log so the variance rate is
measurable.

- gate: positive lift ≥ 0.5 digits vs accumulated-min (kernel-scope); tolerance 6.0 (reporting-only)
- seed 12345, sample_count 5000, entry BO
- run dir: `runs/qcdloop/tier_b_stage2_subtask4/`
- offline: **319 tests pass** (`tests/patcher` + `tests/shared`; +9 new in `test_r4_escape_log.py`)
- code: `fba36a0` (Steps 1-2, pushed on langgraph-agents)

## Per-integral outcome (kernel-scoped gate)

| I | outcome | patcher_status | kernel lift | Subtask-3 baseline | Δ vs Subtask-3 |
|---|---|---|---|---|---|
| B10 | apply_failed | llm_gen_failed | — | apply_failed (llm_gen_failed) | **budget 3→6 applied (6 attempts ran); still lost all draws to R4 variance — headline NOT achieved** |
| B12 | rejected | **ok** | +0.00 (3.6906→3.6906) | rejected/ok (3.6906) | **byte-identical; still builds `ok` on attempt 1 (STOP #I clear)** |
| B13 | apply_failed | write_truncation | — | write_truncation | **byte-identical** (final.diff sha match) |
| B14 | rejected | ok (chain_no_lift) | +0.00 (13.1855→13.1855) | rejected/ok (13.1855) | **byte-identical** (final.diff sha match) |

## Headline: budget bump applied and R4 log works, but B10 still lost all 6 draws

The single knob took effect exactly as intended and B10 ran the **full 6-attempt budget**
(Subtask 3 was capped at 3). But B10 lost every draw against its chain-shim gen defect and
never reached a build, so the first Group-A measured lift is **not achieved this Subtask**.
STOP #H did NOT fire numerically (R4 fired 4/6 < 5), and the variance pattern *strengthens*
the non-determinism hypothesis rather than falsifying it — see the audit below.

### B10 per-attempt variance (budget=6, `patcher_attempts.jsonl`)

| attempt | outcome | status | R4 escape | escape symbol | backoff (s) |
|---|---|---|---|---|---|
| 0 | build_failed | build_failed | ✔ | `ql::Lnrat<TOutput, TMass, TScale>` | 2.17 |
| 1 | build_failed | build_failed | ✔ | `ql::Lnrat<TOutput, TMass, TScale>` | 4.28 |
| 2 | gen_failed | llm_gen_failed | — | — (no build reached) | 8.00 |
| 3 | gen_failed | llm_gen_failed | — | — (no build reached) | 16.19 |
| 4 | build_failed | build_failed | ✔ | `ql::ddilog<ddouble,...>` | 32.15 |
| 5 | build_failed | llm_gen_failed | ✔ | `ql::Lnrat<TOutput, TMass, TScale>` | 0.00 |

Backoff spacing widened monotonically (2→4→8→16→32s) across the extended budget, exactly as
designed — the Wave-2 constants needed no change.

## Variance-rate table (R4 #error escape hatch)

| integral | attempts | R4-escape attempts | distinct escape symbols | recovered to `ok`? |
|---|---|---|---|---|
| B10 | 6 | 4 (0,1,4,5) | 2 — `ql::Lnrat` ×3, `ql::ddilog` ×1 | **no** (lost all 6) |
| B12 | 2 | 1 (attempt 0) | 1 — `ql::Lnrat<ddouble,ddouble,ddouble>` | **yes** (attempt 1) |

Two independent facts here confirm the escape is **non-deterministic**, not a capability gap:

1. **B12 recovered on the same symbol.** B12's dominant chain hits the *same* `ql::Lnrat`
   region (kokkosUtils.h:702). Its attempt 0 took the R4 escape; its attempt 1 emitted the
   working forwarding overload and built `ok`. A deterministic escape would have repeated the
   `#error` on attempt 1 too. This is the identical recovery pattern Subtask 3 saw for B12.
2. **B10's escapes are not even a single fixed symbol.** Across B10's 6 draws the escape hit
   `ql::Lnrat` three times *and* `ql::ddilog` once, with two draws failing at generation
   before any build. A deterministic escape on one symbol/silo would have produced the same
   `#error` all six times. It did not.

B10 simply drew the escape (or a plain gen failure) on all six attempts — a worse run of luck
than B12's, on a probabilistic failure. Budget=6 lowered but did not eliminate the loss
probability on this integral.

## STOP-condition audit (§ STOP-and-report)

- **STOP #H (R4 escape deterministic — fires ≥5/6 on B10's L702)** — **did NOT fire.** R4
  fired **4 of 6** attempts (< 5), and across **two different symbols** plus two non-R4 gen
  failures. The non-determinism hypothesis is *corroborated*, not refuted: a deterministic
  escape would repeat one symbol's `#error` on all six draws, and B12 recovered on the same
  symbol. The budget was NOT bumped further (per discipline); the durable fix (option 2,
  deterministic forwarding-overload emitter) is the correct next move — see below.
- **STOP #A (measurement falsification)** — did NOT fire. B10 never reached measurement
  (build failed upstream at the `Lnrat`/`ddilog` chain-shim gen), so "builds cleanly but
  measures chain_no_lift / lift < +8" cannot apply. The closure-scoped design's core
  hypothesis is neither confirmed nor falsified by a B10 headline this Subtask — the blocker
  remains the orthogonal Patcher gen-robustness defect, exactly as Subtask 3 scoped it.
- **STOP #B (accept ↔ reject flip)** — did NOT fire. B13 identical (`apply_failed`/
  `write_truncation`, final.diff **byte-identical**, sha `01ba4719…`); B14 identical
  (`rejected`/`ok`, 13.1855→13.1855, final.diff **byte-identical**, sha `01ba4719…`); B12
  identical (`rejected`/`ok`, 3.6906→3.6906). No currently-correct rejection became a false
  accept; no accepting chain regressed.
- **STOP #I (budget bump breaks a previously-succeeding path)** — did NOT fire. B12 built
  `ok` on attempt 1 under budget=6 (its Subtask-3 win was on attempt 2 of a 3-budget run);
  B14 still built `ok`. More attempts strictly extended the retry window; nothing that
  accepted under budget=3 failed under budget=6.
- **STOP #J (new gen-defect class)** — did NOT fire. Every B10/B12 build failure this run is
  in the R4 escape-hatch family (`ql::Lnrat`, `ql::ddilog` — both math-bridge helpers taking
  the `#error "… requires manual classification"` escape) or a plain LLM gen failure. No new
  defect class (no #include break, no redeclaration, no `_pi2o6`-style constant defect —
  those stay cleared by Subtask 3's catalog + normaliser).

## Verdict

Subtask 4's single declared change is **done and verified**: `MAX_INTEGRATOR_RETRIES` 3→6
took effect (B10 ran all 6 attempts), and the R4 `#error` diagnostic log lands
`r4_escape`/`r4_symbol` in `patcher_attempts.jsonl` for every escape-hatch build failure
(the variance-rate table above is read directly from it). Backoff spacing widened naturally
(2→32s) with no constant change. **No regression** (STOP #B/#I clear; B12/B13/B14 unchanged
from Subtask-3 baselines, B13/B14 byte-identical). **No new defect class** (STOP #J clear).

**The headline — B10's first Group-A measured lift — is NOT achieved.** The budget bump
reduced but did not eliminate the loss probability: B10 drew the R4 escape or a plain gen
failure on all 6 attempts, while B12 recovered on the same `Lnrat` symbol in 2. STOP #H did
not fire (4/6 < 5, two distinct symbols), so this is **not** a deterministic escape to be
worked around by more attempts — it is exactly the LLM non-determinism the Subtask
hypothesised, and the retry budget is a mitigation with diminishing returns, not a fix.

**Recommendation: escalate to the durable fix.** Subtask 3 option (2) — the deterministic
forwarding-overload emitter for namespace-qualified math-bridge helpers (`ql::Lnrat<ext,…>`
→ `::ql::Lnrat<ext,…>`, the same mechanical transform B12 and 2b produced by hand) — removes
the LLM from the helper-forwarding decision entirely, converting a probabilistic `#error`
into a deterministic overload. This run is the empirical case for it: a symbol the pipeline
*can* forward (B12 did, twice now across Subtask 3 and 4) but *probabilistically won't*,
where six attempts were not enough on B10. Further budget bumps are explicitly not
recommended (STOP #H discipline). This is Patcher gen-robustness work, tracked separately
from the closure track, and needs Reet's go-ahead before implementation.

## After success (NOT dispatched here)

- Subtask 3 option (2) — deterministic forwarding-overload emitter (durable fix; **now the
  recommended next step** given B10 lost 6/6).
- Group B measurement-only (Phase 2f, needs Reet).
- B13 chain-selector scope refinement (narrow Li2omx2 scope).
- B12 hotspot-covering chain selection (move the 3.6906 floor).

STOP-after-Stage-2 holds; the retry budget was the single knob turned.
