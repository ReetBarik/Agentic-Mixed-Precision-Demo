"""Build + run the Validator's app drivers, aggregating coeffs into flat arrays.

Three responsibilities:

* :func:`build_driver` — cmake-configure + build ``runs/qcdloop/app`` for one mode
  (``vanilla`` / ``dd``) against a given header tree, returning the binary path.
* :func:`stage_dd_headers` — repoint a freshly-archived ``ddfun_enabled`` tree at
  the repo-vendored DD primitives, so the DD oracle and the candidate builds share
  one set of extended-precision headers.
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


# ---------------------------------------------------------------------------
# DD oracle header staging
# ---------------------------------------------------------------------------
# The qcdloop@ddfun_enabled fork ships its OWN dd_math/dd_complex/ff_math/
# ff_complex under ``namespace ql::ddfun``, sitting next to kokkosMaths_dd.h.
# Those shadow the repo-vendored copies two ways: app/CMakeLists.txt lists
# ${QL_HEADERS} ahead of ${_vendored_include}, AND kokkosMaths_dd.h reaches them
# with a QUOTED #include, which searches the includer's own directory before any
# -I path — so include-order alone cannot dislodge them.
#
# The effect was that the DD oracle silently ignored third_party/include: a
# refresh there moved the candidate builds but left the oracle on the fork's
# frozen copies (documented in reports/HEADER_REFRESH_2026-08-13.md).
#
# Fix: delete the shadowing primitives from the ARCHIVED tree (never from
# ~/qcdloop — the archive is a throwaway staging dir), which makes the quoted
# include fall through to -I third_party/include, then inject the ql::ddfun
# alias namespace the fork's sources expect. The fork authors ql::ddfun natively
# via its own dd_math.hpp, so removing that file removes the namespace too; the
# shim below restores it over Kokkos::Experimental. Same shape as the checked-in
# runs/qcdloop_headers_full/kokkosMaths_dd.h mirror.
#
# Injected rather than overwriting kokkosMaths_dd.h wholesale, so that any future
# change on the fork side (new Chebyshev tables, tolerances) is preserved.
_DD_SHADOWING_PRIMITIVES = ("dd_math.hpp", "dd_complex.hpp",
                            "ff_math.hpp", "ff_complex.hpp")

_QL_DDFUN_SHIM = """
// ---- injected by agents/validator/runner.stage_dd_headers ----
// The vendored primitives live under Kokkos::Experimental; this restores the
// ql::ddfun spelling the fork's sources are written against. A real namespace,
// not an alias: an alias cannot host using-declarations or the make_dd/dd_pi
// wrappers, and the fork calls both.
namespace ql {
namespace ddfun {
using namespace ::Kokkos::Experimental;
using ddouble   = ::Kokkos::Experimental::DoubleDouble;
using ddcomplex = ::Kokkos::Experimental::DoubleDoubleComplex;
KOKKOS_INLINE_FUNCTION ddouble make_dd(uint64_t hi_bits, uint64_t lo_bits) {
    return ::Kokkos::Experimental::DoubleDouble::from_bits(hi_bits, lo_bits);
}
KOKKOS_INLINE_FUNCTION ddouble dd_pi() { return ::Kokkos::Experimental::DoubleDouble_pi(); }
}  // namespace ddfun
}  // namespace ql
// ---- end injected shim ----
"""

_SHIM_MARKER = "injected by agents/validator/runner.stage_dd_headers"


def git_archive(repo: Path, ref: str, subpath: str, dest: Path) -> None:
    """Extract ``repo@ref:subpath`` into ``dest`` (repo stays on its branch).

    The DD oracle is always materialized from the pinned ref, never read from the
    repo's on-disk working tree.
    """
    proc = subprocess.run(["git", "-C", str(repo), "archive", ref, subpath],
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"git archive {ref}:{subpath} failed:\n{proc.stderr.decode()[-1500:]}")
    tar = subprocess.run(["tar", "-x", "-C", str(dest)], input=proc.stdout,
                         capture_output=True)
    if tar.returncode != 0:
        raise RuntimeError(f"tar extract failed:\n{tar.stderr.decode()[-1500:]}")


def stage_dd_headers(dd_headers: Path) -> Path:
    """Repoint an archived ``ddfun_enabled:src/qcdloop`` tree at the vendored DD headers.

    Mutates ``dd_headers`` in place (it must be a throwaway archive dir, never a
    real checkout) and returns it. Prefer :func:`materialize_dd_headers`, which
    pairs this with the archive step; call this directly only if you already have
    an archived tree in hand.

    Idempotent. Raises if the tree does not look like a ddfun_enabled checkout.
    """
    dd_headers = Path(dd_headers).resolve()
    maths_dd = dd_headers / "kokkosMaths_dd.h"
    if not maths_dd.is_file():
        raise RuntimeError(
            f"{dd_headers} has no kokkosMaths_dd.h — not a ddfun_enabled src/qcdloop tree")

    for name in _DD_SHADOWING_PRIMITIVES:
        (dd_headers / name).unlink(missing_ok=True)

    text = maths_dd.read_text()
    if _SHIM_MARKER in text:
        return dd_headers                      # already staged

    # Anchor after the primitive includes so the shim sees the types it aliases.
    anchor = '#include "dd_complex.hpp"'
    if anchor not in text:
        raise RuntimeError(
            f'{maths_dd} lacks the expected `{anchor}` include — the fork layout '
            "changed; update stage_dd_headers rather than guessing an anchor.")
    text = text.replace(anchor, anchor + "\n" + _QL_DDFUN_SHIM, 1)
    maths_dd.write_text(text)
    return dd_headers


def materialize_dd_headers(dd_repo: Path, dd_ref: str, dest: Path) -> Path:
    """Archive ``dd_repo@dd_ref:src/qcdloop`` into ``dest`` and stage it for building.

    Returns the header dir to hand to :func:`build_driver` as ``QL_HEADERS``.

    The two steps are fused deliberately: archiving without staging is what left
    the DD oracle silently reading the fork's own shadowing dd_/ff_ headers
    instead of third_party/include. Keeping them together means a new caller
    cannot reintroduce that by forgetting the second call.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    git_archive(Path(dd_repo), dd_ref, "src/qcdloop", dest)
    return stage_dd_headers(dest / "src" / "qcdloop")


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
