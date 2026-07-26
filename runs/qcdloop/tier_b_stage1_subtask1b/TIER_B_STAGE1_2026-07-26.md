# Tier-B Stage-1 — chain-scoped dd promotion (2026-07-26)

Phase 2f coordinated whole-chain double-double promotion on the 4 measured Tier-B integrals. v1 promotes the dominant COMPUTED cascade chain per integral (one coordinated envelope).

- gate: positive lift >= 0.5 digits vs accumulated-min (chain_dd); tolerance 6.0 (reporting-only)
- seed 12345, sample_count 5000, entry BO

## Per-integral outcome (kernel-scoped gate)

The gate now scores each chain against ITS integral's own p100 floor (kernel-scope, Reet 2026-07-25), not the whole-app min pinned by the worst kernel (B12's hotspot). Whole-app columns are kept for cross-kernel visibility.

| I | kernel baseline | kernel final | kernel lift | predicted lift | app baseline | app final | outcome | chain | lines |
|---|---|---|---|---|---|---|---|---|---|
| B13 | — | — | — | +17.10 | — | — | apply_failed | cascade_B13_79fc5b8f_f080f240 | 8 |
| B14 | 13.1855 | 13.1855 | +0.00 | +16.66 | 3.6906 | 3.6906 | rejected (chain_no_lift) | cascade_B14_3429b1d4_01bf2ff3 | 3 |

## Predicted vs measured lift (kernel-scoped)

- **B13** (cascade_B13_79fc5b8f_f080f240): predicted +17.10, kernel-measured — (— -> —), whole-app lift —, tightness 0.07080121254580928, patcher_status=write_truncation, declared_dd=False
    - lines: B2m.h:300, B2m.h:301, B2m.h:305, B2m.h:306, B2m.h:355, B2m.h:533, kokkosUtils.h:212, kokkosUtils.h:702
- **B14** (cascade_B14_3429b1d4_01bf2ff3): predicted +16.66, kernel-measured +0.00 (13.1855 -> 13.1855), whole-app lift +0.00, tightness 0.19860180300800165, patcher_status=ok, declared_dd=False
    - lines: B2m.h:401, B2m.h:578, kokkosUtils.h:1208

## Notes
- Kernel-scope gate (Reet 2026-07-25): each chain gated against its own integral's p100 floor, not the whole-app min (which B12's hotspot pins). The whole-app gate rejected B14 as chain_no_lift because it couldn't move the global min; the kernel-scope gate measures B14's own coefficient lift.
- Chain-scope 2d-B (Fix 1): the gate now fires only on INTERIOR chain regions; the outermost region's exit-truncation is the designed output boundary and is exempt (was false-positiving B10/B12 pre-build).
- STOP after Stage-1 for review; Group B / all-21 not run.
- v1 = dominant chain per integral; multi-chain union deferred to Stage-2.
