"""Regression gate for the integrator_base refactor.

The tracked interop shim's ``SOURCE_HASH`` is a cache key over (target headers ⊕
ruleset).  The extraction of the shared machinery into ``agents.integrator_base``
MUST preserve it byte-for-byte, or the committed qcdloop shim
(``runs/qcdloop/src/ql_tracked_interop.hpp``) would silently regenerate.  These
tests pin the historical hashes and prove the cache-hit path returns the
committed shim untouched.

The cache-hit test runs against a tmp_path COPY of the committed shim: a miss
rewrites ``existing_shim`` in place, so asserting on the tracked file directly
made the failure mode destructive rather than merely red.
"""

import shutil
from pathlib import Path

import pytest

from agents.integrator_base import cache
from agents.tracked_integrator import agent as ti

REPO = Path(__file__).resolve().parents[2]
HEADERS_FULL = REPO / "runs" / "qcdloop_headers_full"
COMMITTED_SHIM = REPO / "runs" / "qcdloop" / "src" / "ql_tracked_interop.hpp"

# Locked values (see runs/qcdloop/VALIDATION.md).  SOURCE_HASH was re-pinned
# cfad2410… → 247c8b86… when e3d2e45 added kokkosMaths_dd.h to qcdloop_headers_full
# (a legitimate source-snapshot enrichment; RULESET_HASH is unchanged).
#
# Re-pinned again 247c8b86… → 25f2b895… by the 95ce538 header refresh
# (quad::ddfun/quad::ffun → Kokkos::Experimental).  That sweep touched exactly two
# files inside the hashed tree — README.md and kokkosMaths_dd.h's alias block —
# and the committed shim references NEITHER: it contains no dd/ff vocabulary at
# all, only the ql:: Tracked-interop surface (ql::Real, ql::kAbs, ql::Constants,
# …).  So the shim is stale only in the bookkeeping sense; its content cannot
# depend on what changed.  Re-pinning here records that, rather than regenerating
# a validated 526-line artifact from a non-deterministic LLM pass.  RULESET_HASH
# is again unchanged, confirming the ruleset itself did not move.
RULESET_HASH = "473ccee3385392101f03d66f7d3fe8f6be11b3a57c38d9abe16e4b7a65fc914c"
SOURCE_HASH = "25f2b895f28aa7fe2d953e3184974ba497a1a3dc934b31f64c048867df6b43ec"

_needs_tree = pytest.mark.skipif(
    not HEADERS_FULL.is_dir() or not COMMITTED_SHIM.is_file(),
    reason="qcdloop headers_full tree / committed shim not present",
)


def test_ruleset_hash_is_preserved():
    assert ti._ruleset_hash() == RULESET_HASH
    assert cache.ruleset_hash(ti._SYSTEM_PROMPT) == RULESET_HASH


@_needs_tree
def test_source_hash_is_preserved():
    assert ti._compute_source_hash(HEADERS_FULL) == SOURCE_HASH
    assert cache.compute_source_hash(HEADERS_FULL, ti._SYSTEM_PROMPT) == SOURCE_HASH


@_needs_tree
def test_committed_shim_is_a_cache_hit_and_untouched(tmp_path):
    """The committed shim must be a cache hit and be returned unmodified.

    Exercised against a *copy* in tmp_path, never the tracked file itself.  On a
    cache MISS ``integrate`` rewrites ``existing_shim`` in place, and with
    ``cfg=None`` it writes the 7-line offline placeholder — so pointing this test
    at the real path meant a single stale hash silently replaced the committed
    526-line shim with a no-op, leaving a dirty tracked file for the next
    ``git commit -a`` to pick up.  A copy fails the same assertions without
    putting the artifact at risk.
    """
    before = COMMITTED_SHIM.read_bytes()
    committed_mtime = COMMITTED_SHIM.stat().st_mtime_ns

    probe = tmp_path / COMMITTED_SHIM.name
    shutil.copy2(COMMITTED_SHIM, probe)
    mtime_before = probe.stat().st_mtime_ns

    ret = ti.integrate(
        target_library_headers=HEADERS_FULL,
        driver_source_path=REPO / "runs" / "qcdloop" / "src" / "boxGPU_tracked.cpp",
        existing_shim=probe,
        cfg=None,  # cache hit short-circuits before any generation path
    )

    assert Path(ret).resolve() == probe.resolve()
    assert probe.read_bytes() == before  # byte-identical
    assert probe.stat().st_mtime_ns == mtime_before  # not rewritten
    assert cache.extract_source_hash(before.decode()) == SOURCE_HASH

    # The tracked file was never a write target.
    assert COMMITTED_SHIM.read_bytes() == before
    assert COMMITTED_SHIM.stat().st_mtime_ns == committed_mtime
