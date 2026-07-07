"""Tracked-integrator agent — shared service (no LangGraph node).

Owns the responsibility of integrating the ``Tracked<T>`` error-propagation
datatype into an arbitrary scientific application.  Given a target library's
headers and a driver source that exercises them, it produces a single C++
interop shim header (``<app>_interop.hpp``) that makes the library callable with
``T = Tracked<double>`` (and ``Complex<Tracked<double>>`` where applicable), so
its floating-point computations emit a condition-number / error-propagation
journal.  The generated header is self-auditing via comments; there is no
separate manifest file.

Structurally symmetric with ``agents/build_run/agent.py``: this module exposes a
plain callable (:func:`integrate`) rather than a LangGraph node, and is intended
to be invoked from ``build_and_run(...)`` as a prerequisite step whenever a
target uses Tracked and no up-to-date shim exists (see revision #2 of the task
spec: shared service, no new graph edges, no ``PipelineState`` changes).

**This is the STRUCTURE-ONLY scaffold (Part 1).**  :func:`integrate` performs
the real staleness bookkeeping — it hashes the target-library header directory
and compares that against the ``// SOURCE_HASH:`` line embedded in an existing
shim — but instead of calling the LLM it writes a benign placeholder header.
Part 2 replaces the placeholder body with LLM-driven generation; the signature
and the hash/caching contract are stable across that change, so callers wired up
now keep working unchanged.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

# Files under the target-library header directory that participate in the
# SOURCE_HASH.  Kept broad on purpose: any C/C++ header flavor invalidates the
# cached shim when its bytes change.  Non-header files (README, etc.) are
# ignored so documentation churn does not force regeneration.
_HEADER_SUFFIXES = {".h", ".hpp", ".hh", ".hxx", ".ipp", ".inc", ".cuh", ".tcc"}

_SOURCE_HASH_RE = re.compile(r"//\s*SOURCE_HASH:\s*(\S+)")

# Written verbatim by Part 1; Part 2's post-processing replaces PENDING with the
# real hash.  The scaffold writes the real hash directly (no LLM round-trip).
_SOURCE_HASH_PENDING = "PENDING"


def integrate(
    target_library_headers,
    driver_source_path,
    tracked_repo_path=None,
    existing_shim=None,
    *,
    cfg=None,
    out_path=None,
    app_name=None,
) -> Path:
    """Produce (or reuse) the ``<app>_interop.hpp`` shim for a target library.

    Parameters
    ----------
    target_library_headers:
        Path to the target library's header directory.  Its contents are hashed
        into the shim's ``// SOURCE_HASH:`` line for staleness detection.
    driver_source_path:
        Path to the driver source file that exercises the library.  Named
        ``driver_source_path`` (not ``driver_source``) to disambiguate from
        ``driver_gen.driver_source``, which is C++ *text*, not a path.  When no
        explicit ``out_path`` is given, the generated shim is written alongside
        this file.
    tracked_repo_path:
        Path to the Tracked upstream checkout.  Defaults to the vendored subtree
        at ``third_party/tracked`` when ``None``.  Unused by the scaffold body;
        Part 2 embeds the Tracked headers' include path.
    existing_shim:
        Path to a pre-existing shim to extend / refresh in place.  If it exists
        and its embedded ``SOURCE_HASH`` matches the freshly computed hash, the
        shim is considered up to date and returned untouched (cache hit).
    cfg:
        Optional :class:`~agents.config.PipelineConfig`.  Unused by the scaffold;
        Part 2 reads ``cfg.model`` for the LLM call (no hardcoded model names).
    out_path:
        Optional explicit output path for the shim.  Overrides the default
        (``<driver_dir>/<app>_interop.hpp``) and ``existing_shim`` for placement.
    app_name:
        Optional application name used to build the default filename.  Derived
        from the header directory name when ``None``.

    Returns
    -------
    Path
        The path to the up-to-date shim (freshly written, or the cached one).
    """
    headers_dir = Path(target_library_headers).resolve()
    driver_path = Path(driver_source_path).resolve()
    if tracked_repo_path is None:
        tracked_repo_path = Path(__file__).parent.parent.parent / "third_party" / "tracked"
    tracked_repo_path = Path(tracked_repo_path)

    if not headers_dir.is_dir():
        raise NotADirectoryError(
            f"target_library_headers is not a directory: {headers_dir}"
        )

    source_hash = _hash_header_dir(headers_dir)

    resolved_app_name = app_name or _derive_app_name(headers_dir)

    # Resolve the output path.  Precedence: explicit out_path > existing_shim
    # (refresh in place) > default alongside the driver.
    if out_path is not None:
        shim_path = Path(out_path).resolve()
    elif existing_shim is not None:
        shim_path = Path(existing_shim).resolve()
    else:
        shim_path = driver_path.parent / f"{resolved_app_name}_interop.hpp"

    # Staleness check: an existing shim whose embedded hash matches the current
    # header contents is up to date — return it without rewriting.
    cache_candidate = Path(existing_shim).resolve() if existing_shim is not None else shim_path
    if cache_candidate.exists():
        cached_hash = _extract_source_hash(cache_candidate.read_text(encoding="utf-8"))
        if cached_hash == source_hash:
            return cache_candidate

    # (Re)generate.  Part 2 replaces this placeholder with the LLM call and
    # post-processes `// SOURCE_HASH: PENDING` -> source_hash; the scaffold
    # writes the real hash directly.
    shim_path.parent.mkdir(parents=True, exist_ok=True)
    shim_path.write_text(
        _render_placeholder(resolved_app_name, source_hash), encoding="utf-8"
    )
    return shim_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hash_header_dir(headers_dir: Path) -> str:
    """SHA-256 over the header files under ``headers_dir`` (recursive).

    The digest folds in each header's path relative to ``headers_dir`` and its
    bytes, walked in sorted order, so a rename, move, edit, add, or delete of any
    header changes the hash.  Non-header files (see :data:`_HEADER_SUFFIXES`) are
    skipped so documentation churn does not invalidate the cached shim.
    """
    h = hashlib.sha256()
    files = sorted(
        p for p in headers_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _HEADER_SUFFIXES
    )
    for path in files:
        rel = path.relative_to(headers_dir).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _extract_source_hash(text: str) -> str | None:
    """Return the hash on the ``// SOURCE_HASH:`` line, or ``None`` if absent.

    A ``PENDING`` placeholder is treated as "no hash" so it never counts as a
    cache hit.
    """
    m = _SOURCE_HASH_RE.search(text)
    if not m:
        return None
    value = m.group(1)
    return None if value == _SOURCE_HASH_PENDING else value


def _derive_app_name(headers_dir: Path) -> str:
    """Best-effort application name from the header directory name.

    Strips common packaging suffixes (``qcdloop_headers`` -> ``qcdloop``); falls
    back to the raw directory name.  Part 2 may refine this from the driver's
    includes, but the scaffold only needs a stable, sensible default.
    """
    name = headers_dir.name
    for suffix in ("_headers", "-headers", "_include", "_includes", "-include", "_inc"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)] or name
    return name


def _render_placeholder(app_name: str, source_hash: str) -> str:
    """A benign, valid header standing in for the not-yet-implemented shim.

    Deliberately *not* an ``#error``: the scaffold placeholder must be a compilable
    no-op so wiring it into ``build_and_run`` cannot break an unrelated build.
    Part 2's escape hatch (Rule 9) is what emits ``#error`` for genuinely
    unclassifiable functions.
    """
    return (
        f"// {app_name}_interop.hpp — Tracked<T> interop shim (SCAFFOLD PLACEHOLDER)\n"
        f"//\n"
        f"// Generated by agents/tracked_integrator (structure-only pass, Part 1).\n"
        f"// LLM-driven shim generation is not implemented yet (Part 2); this is a\n"
        f"// compilable no-op so the caching/staleness plumbing can be exercised.\n"
        f"// SOURCE_HASH: {source_hash}\n"
        f"#pragma once\n"
    )
