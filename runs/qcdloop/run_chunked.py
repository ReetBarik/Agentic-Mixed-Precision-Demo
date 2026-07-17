#!/usr/bin/env python3
"""Chunked, optionally-parallel characterization runner for the consolidated
qcdloop tracked driver.

A full 100k-samples/integral run would materialize ~1 TB of journal (the
instrumented 256-sample journal is ~2.5 GB → ~10 MB/sample across 21 integrals),
so it must run in bounded-journal chunks reduced in-process.  Two levels:

* **inner (chunk):** the driver runs one global sample range
  ``[offset, offset+chunk)``, its transient journal is reduced to a small shard
  with ``stability_reducer.reduce_journal``, and the journal is deleted.
* **outer (width W = --workers):** up to W chunks run concurrently, each in its
  OWN working dir (so ``journal.jsonl`` never collides) and reduced in its OWN
  process (so the CPU-bound reduce parallelizes across cores).  A pool of W
  processes runs the N chunks in waves of W.

Peak journal-on-disk is therefore ``W × one-chunk-journal`` (≈ W × chunk ×
10 MB) — tune W and --chunk together to a disk budget.  Shards are merged +
finalized into one consolidated per-integral report for the Strategy Agent.

Chunking is bit-exact: ``--sample-offset`` fills the skipped prefix so the
mt19937 draws and input ids match a single ``[0, total)`` run, and ``track()``
emits no records, so chunk ``[offset, offset+chunk)`` is byte-identical to the
same samples in one big run (merge == reduce-of-concatenation).  W does not
affect results — the merge is order-independent.

Prepares the header tree (C8 patch → line= patch), builds once, runs all chunks,
then resets the tree — mirroring the manual procedure in VALIDATION.md.

Usage (under the module env):
    module use /soft/modulefiles && module load gcc/13.3.0 cmake/3.28.3
    python runs/qcdloop/run_chunked.py --total 100000 --chunk 500 --workers 8 \
        --out runs/qcdloop/report_100k.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SRC = HERE / "src"
HEADERS = REPO / "runs" / "qcdloop_headers_full"
C8_PATCH = SRC / "ql_tracked.patch"
LINE_PATCH = SRC / "ql_tracked_lines.patch"
DRIVER_BIN = HERE / "build" / "boxGPU_tracked"

# Module prelude for driver subprocesses (matches the build chain).  reduce is
# pure Python and needs no module env.
MODULE_PRELUDE = ("module use /soft/modulefiles && "
                  "module load gcc/13.3.0 cmake/3.28.3")

# Rough per-sample journal size (instrumented, all 21 integrals) for the disk
# estimate printed at startup; measured ~9.8 MB/sample at 256 and 1k.
_BYTES_PER_SAMPLE = 10_000_000

sys.path.insert(0, str(REPO))
from agents.shared import stability_reducer as sr  # noqa: E402
from agents.shared import fast_merge  # noqa: E402


def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(REPO), capture_output=True, text=True)


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run under the module env (matches the build chain)."""
    inner = " ".join(cmd)
    return subprocess.run(["bash", "-lc", f"{MODULE_PRELUDE} && {inner}"],
                          capture_output=True, text=True, **kw)


def prepare_tree() -> None:
    _git("checkout", "--", str(HEADERS))
    for patch in (C8_PATCH, LINE_PATCH):
        r = _git("apply", str(patch))
        if r.returncode != 0:
            _git("checkout", "--", str(HEADERS))
            raise RuntimeError(f"apply {patch.name} failed:\n{r.stderr}")


def build() -> None:
    r = _run(["cmake", "--build", str(HERE / 'build'), "-j"])
    if r.returncode != 0:
        raise RuntimeError(f"build failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def _process_chunk(task: dict) -> dict:
    """Worker: run one chunk in an isolated dir, reduce it, write its shard.

    Runs in a separate process (ProcessPoolExecutor).  Each chunk gets a fresh
    temp dir so the driver's hardcoded ``journal.jsonl`` never collides with a
    concurrent chunk; the journal is deleted before returning, so at most W
    journals exist at once.
    """
    offset, count, shard_path = task["offset"], task["count"], task["shard_path"]
    tmp = Path(tempfile.mkdtemp(prefix=f"qcdloop_c{offset:08d}_"))
    try:
        cmd = [str(DRIVER_BIN), "--sample-count", str(count),
               "--sample-offset", str(offset)]
        r = subprocess.run(
            ["bash", "-lc", f"{MODULE_PRELUDE} && {' '.join(cmd)}"],
            cwd=str(tmp), capture_output=True, text=True,
        )
        journal = tmp / "journal.jsonl"
        if r.returncode != 0 or not journal.exists():
            return {"offset": offset, "ok": False, "err": (r.stderr or "")[-1500:]}
        jsize = journal.stat().st_size
        shard = sr.reduce_journal(str(journal))
        sr._write_json(shard, shard_path)
        return {"offset": offset, "count": count, "ok": True, "jsize": jsize}
    except Exception as exc:  # noqa: BLE001 - surface any worker failure to main
        return {"offset": offset, "ok": False, "err": repr(exc)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total", type=int, default=100000, help="total samples/integral")
    ap.add_argument("--chunk", type=int, default=500, help="samples/integral per chunk")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent chunks (W); peak disk ≈ W × chunk journal")
    ap.add_argument("--out", default=str(HERE / "report_100k.json"))
    ap.add_argument("--shard-dir", default=str(HERE / "shards"))
    ap.add_argument("--keep-shards", action="store_true")
    ap.add_argument("--no-prepare", action="store_true",
                    help="assume the tree is already patched+built")
    ap.add_argument("--resume", action="store_true",
                    help="reuse any already-written, valid shard in --shard-dir "
                         "(skip recomputing those chunks); implies --keep-shards")
    args = ap.parse_args(argv)
    if args.resume:
        args.keep_shards = True

    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(exist_ok=True)
    offsets = list(range(0, args.total, args.chunk))
    workers = max(1, args.workers)
    peak_gb = workers * min(args.chunk, args.total) * _BYTES_PER_SAMPLE / 1e9
    print(f"chunked run: total={args.total} chunk={args.chunk} "
          f"({len(offsets)} chunks) workers={workers} "
          f"est. peak journal on disk ≈ {peak_gb:.0f} GB -> {args.out}", flush=True)

    def _valid_shard(p: Path) -> bool:
        """A shard is reusable only if it parses (guards against a shard whose
        write was interrupted by a hard kill mid-run)."""
        try:
            json.loads(p.read_text())
            return True
        except Exception:  # noqa: BLE001
            return False

    all_tasks = [
        {"offset": off, "count": min(args.chunk, args.total - off),
         "shard_path": str(shard_dir / f"shard_{off:08d}.json")}
        for off in offsets
    ]
    done_shard_paths: list[str] = []
    tasks = all_tasks
    if args.resume:
        tasks = []
        for t in all_tasks:
            sp = Path(t["shard_path"])
            if sp.is_file() and _valid_shard(sp):
                done_shard_paths.append(t["shard_path"])
            else:
                tasks.append(t)
        print(f"resume: {len(done_shard_paths)} shard(s) reused, "
              f"{len(tasks)} chunk(s) to run", flush=True)

    prepared = False
    try:
        if not args.no_prepare:
            prepare_tree()
            prepared = True
            build()
            print("prepared tree (C8 + line=) and built", flush=True)

        shard_paths: list[str] = list(done_shard_paths)
        failures: list[dict] = []
        t0 = time.monotonic()
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_process_chunk, t): t for t in tasks}
            for fut in as_completed(futs):
                res = fut.result()
                done += 1
                if not res.get("ok"):
                    failures.append(res)
                    print(f"  chunk [{res['offset']},…) FAILED: {res.get('err','')}",
                          flush=True)
                    continue
                shard_paths.append(futs[fut]["shard_path"])
                print(f"  chunk {done}/{len(tasks)} "
                      f"[{res['offset']},{res['offset']+res['count']}) "
                      f"journal={res['jsize']/1e9:.2f}GB "
                      f"(elapsed {time.monotonic()-t0:.0f}s)", flush=True)

        if failures:
            raise RuntimeError(f"{len(failures)} chunk(s) failed; aborting merge")

        # Parallel, partition-by-integral merge (see agents/shared/fast_merge).
        # Merges + finalizes + serializes each integral in its own worker so no
        # process ever builds the full multi-million-variable structure; a 100k
        # (200-shard) run merges in ~2 min instead of hours. The merge picks its
        # own worker count (min(32, cpu)); it is independent of the chunk
        # --workers used during the compute phase.
        seen = fast_merge.merge_shard_files(shard_paths, args.out)
        print(f"wrote consolidated report: {args.out} "
              f"(total wall {time.monotonic()-t0:.0f}s)", flush=True)
        print(f"samples_seen (should all be {args.total}): "
              f"{sorted(set(seen.values()))}", flush=True)
        if not args.keep_shards:
            shutil.rmtree(shard_dir, ignore_errors=True)
    finally:
        if prepared:
            _git("checkout", "--", str(HEADERS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
