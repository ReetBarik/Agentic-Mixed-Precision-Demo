#!/usr/bin/env python3
"""Chunked characterization runner for the consolidated qcdloop tracked driver.

A full 100k-samples/integral run would materialize ~1 TB of journal (the
instrumented 256-sample journal is ~2.5 GB → ~10 MB/sample across 21 integrals).
This runner never holds more than one chunk on disk: for each chunk it runs the
driver over a distinct global sample range ``[offset, offset+chunk)``, reduces
that transient journal to a small shard with ``stability_reducer.reduce``, and
deletes the journal before the next chunk.  The shards are then merged +
finalized into one consolidated per-integral report for the Strategy Agent.

Chunking is bit-exact: ``--sample-offset`` fills the skipped prefix so the
mt19937 draws and input ids match a single ``[0, total)`` run, and ``track()``
emits no records, so chunk ``[offset, offset+chunk)`` is byte-identical to the
same samples in one big run (validated: merge == reduce-of-concatenation).

Prepares the header tree (C8 patch → line= patch), builds once, runs all chunks,
then resets the tree — mirroring the manual procedure in VALIDATION.md.

Usage (under the module env):
    module use /soft/modulefiles && module load gcc/13.3.0 cmake/3.28.3
    python runs/qcdloop/run_chunked.py --total 100000 --chunk 500 \
        --out runs/qcdloop/report_100k.json
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SRC = HERE / "src"
HEADERS = REPO / "runs" / "qcdloop_headers_full"
C8_PATCH = SRC / "ql_tracked.patch"
LINE_PATCH = SRC / "ql_tracked_lines.patch"
DRIVER_BIN = HERE / "build" / "boxGPU_tracked"

sys.path.insert(0, str(REPO))
from agents.shared import stability_reducer as sr  # noqa: E402


def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(REPO), capture_output=True, text=True)


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run under the module env (matches the build chain)."""
    prelude = "module use /soft/modulefiles && module load gcc/13.3.0 cmake/3.28.3"
    inner = " ".join(cmd)
    return subprocess.run(["bash", "-lc", f"{prelude} && {inner}"],
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


def run_chunk(offset: int, count: int) -> Path:
    journal = HERE / "journal.jsonl"
    if journal.exists():
        journal.unlink()
    r = _run([str(DRIVER_BIN), "--sample-count", str(count),
              "--sample-offset", str(offset)], cwd=str(HERE))
    if r.returncode != 0 or not journal.exists():
        raise RuntimeError(f"chunk [{offset},{offset+count}) run failed:\n{r.stderr[-2000:]}")
    return journal


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total", type=int, default=100000, help="total samples/integral")
    ap.add_argument("--chunk", type=int, default=500, help="samples/integral per chunk")
    ap.add_argument("--out", default=str(HERE / "report_100k.json"))
    ap.add_argument("--shard-dir", default=str(HERE / "shards"))
    ap.add_argument("--keep-shards", action="store_true")
    ap.add_argument("--no-prepare", action="store_true",
                    help="assume the tree is already patched+built")
    args = ap.parse_args(argv)

    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(exist_ok=True)
    offsets = list(range(0, args.total, args.chunk))
    print(f"chunked run: total={args.total} chunk={args.chunk} "
          f"({len(offsets)} chunks) -> {args.out}", flush=True)

    prepared = False
    try:
        if not args.no_prepare:
            prepare_tree()
            prepared = True
            build()
            print("prepared tree (C8 + line=) and built", flush=True)

        shard_paths: list[Path] = []
        for k, offset in enumerate(offsets):
            count = min(args.chunk, args.total - offset)
            t0 = time.monotonic()
            journal = run_chunk(offset, count)
            jsize = journal.stat().st_size
            shard = sr.reduce_journal(str(journal))
            sp = shard_dir / f"shard_{offset:08d}.json"
            sr._write_json(shard, str(sp))
            shard_paths.append(sp)
            journal.unlink()
            print(f"  chunk {k+1}/{len(offsets)} [{offset},{offset+count}) "
                  f"journal={jsize/1e9:.2f}GB reduced in {time.monotonic()-t0:.0f}s",
                  flush=True)

        shards = [__import__("json").loads(p.read_text()) for p in shard_paths]
        report = sr.finalize_report(sr.merge_reports(shards))
        sr._write_json(report, args.out)
        print(f"wrote consolidated report: {args.out}", flush=True)
        seen = report.get("samples_seen", {})
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
