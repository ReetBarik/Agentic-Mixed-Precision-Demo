"""Build + run the Validator's app drivers, aggregating coeffs into flat arrays.

Two responsibilities:

* :func:`build_driver` — cmake-configure + build ``runs/qcdloop/app`` for one mode
  (``vanilla`` / ``dd``) against a given header tree, returning the binary path.
* :func:`run_and_aggregate` — run the binary over ``[0, total)`` samples in
  bit-exact chunks (``--sample-offset``), optionally across a process pool, and
  fold the ``RES`` output into per-integral flat ``array('d')`` buffers indexed by
  ``sample*6 + component``.  ``(hi, lo)`` per component (vanilla ``lo == 0``), so
  the reference at 100k is ~200 MB, not multi-GB.

Chunking here is for parallelism + bounded per-chunk stdout, not journal size (the
app drivers emit no journal) — it mirrors runs/qcdloop/run_chunked.py's
``--sample-offset`` contract, which is byte-identical to a single ``[0, total)``
run (proven in step 5a).
"""

from __future__ import annotations

import subprocess
import tempfile
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from agents.validator.coeffs import N_COMPONENTS, parse_component

_REPO = Path(__file__).resolve().parents[2]
_APP_CMAKE_DIR = _REPO / "runs" / "qcdloop" / "app"

# Module prelude matching the build chain (see agents/build_run + run_chunked.py).
MODULE_PRELUDE = ("module use /soft/modulefiles && "
                  "module load gcc/13.3.0 cmake/3.28.3")

# Per-integral coeff arrays: {integral: (hi_array, lo_array)}, both length
# total*N_COMPONENTS, index = sample*N_COMPONENTS + component.
CoeffArrays = dict


def _bash(cmd: str, **kw) -> subprocess.CompletedProcess:
    """Run a command under the module env in a login shell."""
    return subprocess.run(["bash", "-lc", f"{MODULE_PRELUDE} && {cmd}"],
                          capture_output=True, text=True, **kw)


def build_driver(
    tree_headers: Path,
    mode: str,
    build_dir: Path,
    kokkos_root: Path,
) -> Path:
    """Configure + build the app driver for ``mode`` against ``tree_headers``.

    ``tree_headers`` is the directory containing ``boxGPU.h`` + ``box/`` (e.g.
    ``runs/qcdloop_headers_full`` for vanilla, a ddfun_enabled ``src/qcdloop`` for
    dd).  Returns the path to the built ``boxGPU_app`` binary.
    """
    if mode not in ("vanilla", "dd"):
        raise ValueError(f"mode must be 'vanilla' or 'dd', got {mode!r}")
    tree_headers = Path(tree_headers).resolve()
    build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    cfg = _bash(
        f"cmake -S {_APP_CMAKE_DIR} -B {build_dir} "
        f"-DCMAKE_PREFIX_PATH={Path(kokkos_root).resolve()} "
        f"-DQL_MODE={mode} -DQL_HEADERS={tree_headers} "
        f"-DCMAKE_BUILD_TYPE=Release"
    )
    if cfg.returncode != 0:
        raise RuntimeError(f"cmake configure ({mode}) failed:\n{cfg.stdout[-2000:]}\n{cfg.stderr[-2000:]}")

    bld = _bash(f"cmake --build {build_dir} -j")
    if bld.returncode != 0:
        raise RuntimeError(f"build ({mode}) failed:\n{bld.stdout[-3000:]}\n{bld.stderr[-3000:]}")

    binary = build_dir / "boxGPU_app"
    if not binary.is_file():
        raise RuntimeError(f"build ({mode}) produced no binary at {binary}")
    return binary


def _run_chunk(task: dict) -> str:
    """Worker: run one [offset, offset+count) chunk, return its stdout file path.

    Writes stdout to a temp file (RES lines) rather than returning it, so large
    chunk output never crosses the process boundary.
    """
    binary, offset, count = task["binary"], task["offset"], task["count"]
    out_fd, out_path = tempfile.mkstemp(prefix=f"qcdloop_res_{offset:08d}_", suffix=".txt")
    import os
    os.close(out_fd)
    cmd = f"{binary} --sample-count {count} --sample-offset {offset}"
    with open(out_path, "w") as fh:
        r = subprocess.run(["bash", "-lc", f"{MODULE_PRELUDE} && {cmd}"],
                           stdout=fh, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        Path(out_path).unlink(missing_ok=True)
        raise RuntimeError(f"driver chunk [{offset},{offset+count}) failed:\n{r.stderr[-1500:]}")
    return out_path


def run_and_aggregate(
    binary: Path,
    total: int,
    *,
    chunk: int = 0,
    workers: int = 1,
) -> CoeffArrays:
    """Run ``binary`` over ``[0, total)`` and fold RES output into flat arrays.

    ``chunk`` samples per chunk (0 or >= total → a single [0, total) run);
    ``workers`` concurrent chunks.  Returns ``{integral: (hi, lo)}`` with
    ``hi``/``lo`` ``array('d')`` of length ``total * N_COMPONENTS``.
    """
    if chunk <= 0 or chunk >= total:
        offsets = [0]
        counts = [total]
    else:
        offsets = list(range(0, total, chunk))
        counts = [min(chunk, total - off) for off in offsets]

    tasks = [{"binary": str(binary), "offset": off, "count": cnt}
             for off, cnt in zip(offsets, counts)]

    result: CoeffArrays = {}

    def _ingest(path: str) -> None:
        with open(path) as fh:
            for line in fh:
                if not line.startswith("RES,"):
                    continue
                parts = line.rstrip("\n").split(",")
                if len(parts) != 3 + N_COMPONENTS:
                    raise ValueError(f"malformed RES line: {line!r}")
                integral = parts[1]
                idx = int(parts[2])
                if integral not in result:
                    result[integral] = (
                        array("d", bytes(8 * total * N_COMPONENTS)),
                        array("d", bytes(8 * total * N_COMPONENTS)),
                    )
                hi, lo = result[integral]
                base = idx * N_COMPONENTS
                for c, tok in enumerate(parts[3:]):
                    h, l = parse_component(tok)
                    hi[base + c] = h
                    lo[base + c] = l
        Path(path).unlink(missing_ok=True)

    if workers <= 1 or len(tasks) == 1:
        for t in tasks:
            _ingest(_run_chunk(t))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_chunk, t): t for t in tasks}
            for fut in as_completed(futs):
                _ingest(fut.result())

    return result
