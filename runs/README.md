# runs/ layout

## Active

- **`qcdloop/`** — the consolidated qcdloop application driver. A single
  `boxGPU_tracked` main() dispatches all 21 in-scope box integrals
  (B1–B16, BIN0–BIN4) through the public `ql::BO<TOutput,TMass,TScale>()`
  entry point, each wrapped in a nested `tracked::scope("integral=<name>")`
  / `tracked::scope("sample=<i>")` pair. One shim
  (`src/ql_tracked_interop.hpp`), one C8 patch (`src/ql_tracked.patch`,
  byte-identical to the Stage-2 B16/BIN3/BIN4 patch), one build. Emits
  `journal.jsonl` (scope-tagged) + `journal_meta.json`.
- **`qcdloop_headers_full/`** — vendored upstream qcdloop headers
  (`ReetBarik/qcdloop@8de2089`), all box families B0m..B4m. The consolidated
  driver builds against this tree; the C8 patch is applied at build time and
  reset afterward.
- Other directories (`cancellation/`, `cln/`, `kahan/`, `lnrat/`,
  `log_sum_exp/`, `naive_variance/`) are unrelated calibration micro-fixtures.

## Archived — `archive/stage2/`

The Stage-2 per-target scaffolding: 21 directories, one per integral
(B1–B16, BIN0–BIN4), each with its own driver, minimized interop shim,
CMakeLists, and (B16/BIN3/BIN4) C8 patch. These were used to isolate
compile/validation per target during Stage 2 and are retired now that the
consolidated `qcdloop/` driver exercises the whole surface in one build.
Their (gitignored) journals travelled with the move.

- **`archive/stage2/B13/`** — the **Stage-1 locked reference**. Unlike the
  thin Stage-2 scaffolds, it carries unique committed artifacts: compressed
  journal snapshots (`journal*.jsonl.gz`), analysis tooling
  (`trace_sources.py`, `analysis_queries.sh`), a Stage-1 `README.md`, its own
  pruned `qcdloop_headers/`, and a `reference/`. Its integral is also present
  in the consolidated journal as `integral=B13` (validated bit-identical max
  cond). Kept for the Stage-1 data + tooling; any future Phase 0 work resumes
  from here.

> Note: the consolidation spec cited a Phase 0 resume checklist at
> `todos/iteration1_phase0_b13.md` as the reason to keep B13 in place; that
> file is not present in the repo, so B13 was archived with the rest and this
> pointer left in its stead.
