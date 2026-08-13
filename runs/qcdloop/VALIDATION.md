# qcdloop consolidated driver — validation (256 samples, Stage-2 parity)

**Date:** 2026-07-15. **Branch:** `langgraph-agents`. **Backend:** Kokkos::Serial.
**qcdloop:** `ReetBarik/qcdloop@8de2089`. **Sample count:** 256 per integral
(21 integrals, 4,977,435 records total).

The single `boxGPU_tracked` driver dispatches all 21 box integrals (B1–B16,
BIN0–BIN4) through `ql::BO<TOutput,TMass,TScale>()`, each in a nested
`integral=<name>` / `sample=<i>` scope, one journal, one shim, one C8 patch.

## Integrator whole-app invocation

The interop shim was regenerated once by the tracked-integrator against the
un-pruned `runs/qcdloop_headers_full/` tree, and the C8 patch derived once from
the whole-app compile diagnostics.

- **C8 patch: byte-identical to the Stage-2 B16/BIN3/BIN4 patch**
  (`md5 f0726269…`) — 9 sites, `{a:3, b:5, c:1}`, files `box/B3m.h`,
  `box/B4m.h`. Passes the STOP-#1 byte-identity gate.
- **Shim:** `Sign(Tracked<T>) → Tracked<T>` (527 lines, `SOURCE_HASH 25f2b895…`
  — re-pinned from `cfad2410…` when e3d2e45 enriched the source snapshot, then
  from `247c8b86…` by the 95ce538 header refresh.  That sweep changed only
  README.md and kokkosMaths_dd.h's alias block inside the hashed tree; the shim
  carries no dd/ff vocabulary, so its body was re-stamped, not regenerated).

### STOP #1 — Sign classification non-determinism (resolved)

The *first* whole-app generation classified `Sign → int` (Rule 1), contradicting
all 21 Stage-2 shims (`Sign → Tracked`, Rule 2/C6). That single inversion
cascaded into ~32 int↔tracked errors across B0m.h, kokkosUtils.h, B3m.h, B4m.h,
and would have ballooned the derived patch. Per directive, a bounded retry loop
(≤5, cache bypassed) accepted the first shim with `Sign → Tracked` **and** a
patch byte-identical to `f0726269`.

**Converged on attempt 1/5.** Whole-app misgen rate for the record: 1 bad
generation (pre-loop) then converged on the first fresh retry. At whole-app scale
one bad `Sign` classification cascades to ~32 errors (Sign is called from many
box code paths) vs ~3 at per-target scale — the same C6 fragility, amplified.
This escalates the deferred "C6 wording-hardening" candidate (cite the direct
`TScale x = ql::Sign(...)` assignment site) but did **not** block: retries cleared
it within budget.

## Characterization parity (the accepted consolidation gate)

For every integral, the worst-case conditioning reproduces the Stage-2
per-target reference bit-for-bit:

- **All 21 integrals: `max(cond)` bit-identical** to the per-target journals
  (spot-checked final `coeff0` values bit-identical too, e.g. B1[0]).
- **`cond>1e15` gate:** records occur only in the massive boxes — B14/B15/B16
  cap at exactly `2^53 = 9007199254740992` (atan2 saturation, class a);
  BIN0–BIN4 exceed `2^53` (~1.7e16, genuine cancellation, class b). No third
  category, consistent with the Stage-2 final gate.

## STOP #2 — op-count divergence (understood; accepted as benign)

9/21 integrals have bit-exact op counts vs Stage-2; 12/21 differ (per-sample
deltas −28…+64), localized to **complex-arithmetic decomposition granularity**
(`abs`/`mul`/`div` and their `add`/`sub`/`neg` sub-terms — how `|z|`, `z·w`,
`z/w` factor into real ops).

- **Characterization-neutral:** final values, `max(cond)`, and `cond>1e15`
  hotspot counts are bit-identical despite the op-count differences.
- **Cause:** shim-decomposition non-determinism between the regenerated
  consolidated shim and the per-target shims — **not** the callsite-counter
  scope machinery the spec hypothesized, and **not** a seed-reproduction failure
  (a fresh per-target rebuild reproduces its own journal exactly).
- **Structural note:** one consolidated shim matches 9 per-target journals and
  differs from 12 ⇒ the 21 per-target shims are mutually inconsistent in
  decomposition (≥2 factorizations in the corpus). **All-21 op-count parity is
  therefore unsatisfiable by any single shim.** The correct consolidation gate is
  value/cond/hotspot bit-identity (which passes), not op-count parity.

Decision (2026-07-15): accept the consolidated shim's op counts as the new
baseline; op-count divergence documented as benign.

## Artifacts

`journal.jsonl` (scope-tagged, ~1.6 GB at 256) and `journal_meta.json` are
regenerated locally and gitignored. Rebuild: apply `src/ql_tracked.patch` to
`runs/qcdloop_headers_full`, `cmake`/build, run `./build/boxGPU_tracked
--sample-count 256`, then reset the header tree.

## Per-line attribution (`line=` scope injection)

C++ operator ops carry no source location, so per-*line* attribution comes from
`line=<basename>:<N>` scopes injected around every value-producing statement
across the arithmetic closure (`box/*.h`, `box_common.h`, `kokkosMaths.h`,
`kokkosUtils.h`, and the `boxGPU.h` dispatch). The injector
(`agents/tracked_integrator/line_injector.py`) walks the libclang AST of
`boxGPU_tracked.cpp`, selects statements structurally (subtree bears an
operator/call — not by name; the box functions are templates so types are
dependent), and emits `src/ql_tracked_lines.patch` (1089 sites). Declarations
wrap with `tracked::push_scope/pop_scope` (a lexical block would scope the name
out); other statements wrap with an RAII `{ tracked::scope … ; <stmt>; }`.

Regenerate (composes with the C8 patch — applied to the tree, then reset):

```
module use /soft/modulefiles && module load gcc/13.3.0 cmake/3.28.3
python -m agents.tracked_integrator.line_injector \
  --driver runs/qcdloop/src/boxGPU_tracked.cpp \
  --headers runs/qcdloop_headers_full \
  --tracked-include third_party/tracked/include \
  --kokkos-include $HOME/kokkos-install/include \
  --repo-root . --c8-patch runs/qcdloop/src/ql_tracked.patch \
  --out runs/qcdloop/src/ql_tracked_lines.patch
```

Build with attribution: apply `src/ql_tracked.patch` **then**
`src/ql_tracked_lines.patch`, build, run, reset the tree.

### Bit-exactness gate (256 samples, 2026-07-16) — PASS

Scopes are value-neutral by construction; proven empirically by comparing a
C8-only build to a C8+`line=` build at 256 samples:

- **`coeff0` bit-identical** across all 21 integrals (`diff` of run logs).
- **`max(cond)` and per-integral op counts bit-identical** (`==`) for all 21
  integrals (reducer over both journals).
- **Attribution works:** the reducer now yields **35–102 `line=` regions per
  integral** (vs one `""` bucket in the C8-only journal), no `line=` value
  contains `/`.
