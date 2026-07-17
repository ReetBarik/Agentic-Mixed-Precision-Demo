# Handoff — qcdloop 100k merge optimization (branch `langgraph-agents`)

Repo: `/home/rbarik/Agentic-Mixed-Precision-Demo`

## What was accomplished
The 100k-sample (200-shard) stability-report merge in `runs/qcdloop/run_chunked.py`
was catastrophically slow (2.5 hr O(N²) fold; later attempts still ran 34+ min and
never finished). Root bottlenecks were diagnosed and a fast parallel merge was built
that does the full 100k merge in **~128 s**, producing **byte-identical output** to
`finalize_report(merge_reports(all_shards))`.

Root causes fixed (all four stacked into the new merge):
1. **Source-var filter** — `finalize_report` keeps only `is_source_var` vars (~8.7%);
   the other ~91% are sample-scoped intermediates it discards. Drop them up front →
   ~11× less data.
2. **Partition by integral** — the report has 21 independent integrals; merge +
   finalize + serialize each in its own worker so no process ever builds the full
   ~23M-variable structure.
3. **`prov_vars` set-merge** — BIN4 has regions with ~35k `prov_vars` each;
   `_merge_region`'s `sorted(set|set)` per shard is O(N²). Accumulate into a set once,
   sort once.
4. **orjson + `gc.disable()`** — fast (de)serialization, no cyclic-GC rescans of the
   large object graphs being built.

Also fixed a real correctness bug: `finalize_report`'s `top_regions_by_rel_err` used
an unstable tie-break (order-dependent output). Now sorts by `(-max_rel_err, location)`.

## Deliverables already produced
Validated: 21 integrals, `samples_seen: [100000]`, `no_id_records: 0`.
- `runs/qcdloop/report_100k.json` (13.8 GB) and `report_100k.json.gz` (2.43 GB) — both gitignored.
- 200 valid shards on `/tmp/qcdloop_shards/` (83 GB).

## Uncommitted working-tree changes (the point — NOT yet committed)
- **NEW** `agents/shared/fast_merge.py` — parallel partition merge, orjson-optional.
  Public fn: `merge_shard_files(paths, out_path, workers=None, tmp_dir=None, cfg=None) -> samples_seen`.
- **MODIFIED** `agents/shared/stability_reducer.py` — deterministic `top_regions` tie-break.
- **MODIFIED** `runs/qcdloop/run_chunked.py` — merge now calls
  `fast_merge.merge_shard_files(shard_paths, args.out)` (import added). Supersedes the
  committed `d52a5e3` O(N²) fold.
- ⚠️ `git status` also shows `runs/qcdloop_headers_full/*.h` modified — these are
  **transient** (an interrupted run's applied patches). Do NOT commit them; reset with
  `git checkout -- runs/qcdloop_headers_full/`.

## Immediate next steps
1. Clean up leftover state: kill any running `run_chunked`/`boxGPU`/`merge_*` process,
   then `git checkout -- runs/qcdloop_headers_full/`.
2. Re-verify fast_merge is identical on a 4-shard test dir: run
   `fast_merge.merge_shard_files` on 4 shards from `/tmp/qcdloop_shards` vs
   `sr.finalize_report(sr.merge_reports(...))`, diff with `json.dumps(sort_keys=True)`.
3. (Optional, was in progress) small e2e:
   `PYTHONPATH=$PWD .venv/bin/python runs/qcdloop/run_chunked.py --total 2000 --chunk 500 --workers 4 --shard-dir /tmp/e2e_shards --out /tmp/e2e_report.json`
   NOTE: the **compute phase** (tracked `boxGPU`) was stalling — unrelated to the merge;
   investigate separately or skip, since the merge is validated on real data.
4. **Commit** and push to `langgraph-agents` (project convention: commit + push
   directly, no PR). Intended commit set (path-scope these; do NOT `git add -A` —
   that risks the transient headers): `agents/shared/fast_merge.py`,
   `agents/shared/stability_reducer.py`, `runs/qcdloop/run_chunked.py`,
   `runs/qcdloop/.gitignore`, `runs/qcdloop/HANDOFF.md`. End commit message with
   `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## Env notes
- Always `PYTHONPATH=/home/rbarik/Agentic-Mixed-Precision-Demo .venv/bin/python`.
- Build chain for compute phase: `module use /soft/modulefiles` + gcc/13.3.0 + cmake/3.28.3.
- Box: 128 cores, 502 GB RAM. Home quota 100 GB — keep big artifacts on `/tmp`.
- User constraints: **keep JSON shard format** and **keep report semantics** (they
  deferred collapsing the 23M per-sample source-var entries — a possible future
  "rethink report size" win).
- Standalone validated script (if job-tmp still exists):
  `/home/rbarik/.claude/jobs/c054aab7/tmp/merge_partition.py`.
