"""Regression gate for the integrator_base refactor.

The tracked interop shim's ``SOURCE_HASH`` is a cache key over (target headers ⊕
ruleset).  The extraction of the shared machinery into ``agents.integrator_base``
MUST preserve it byte-for-byte, or the committed qcdloop shim
(``runs/qcdloop/src/ql_tracked_interop.hpp``) would silently regenerate.  These
tests pin the historical hashes and prove the cache-hit path returns the
committed shim untouched.
"""

from pathlib import Path

import pytest

from agents.integrator_base import cache
from agents.tracked_integrator import agent as ti

REPO = Path(__file__).resolve().parents[2]
HEADERS_FULL = REPO / "runs" / "qcdloop_headers_full"
COMMITTED_SHIM = REPO / "runs" / "qcdloop" / "src" / "ql_tracked_interop.hpp"

# Locked historical values (see runs/qcdloop/VALIDATION.md).
RULESET_HASH = "473ccee3385392101f03d66f7d3fe8f6be11b3a57c38d9abe16e4b7a65fc914c"
SOURCE_HASH = "cfad2410c3ddc32ab520cc03f18dd5e38f62b9fd0359678851e50da9f40a0ac8"

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
def test_committed_shim_is_a_cache_hit_and_untouched():
    before = COMMITTED_SHIM.read_bytes()
    mtime_before = COMMITTED_SHIM.stat().st_mtime_ns

    ret = ti.integrate(
        target_library_headers=HEADERS_FULL,
        driver_source_path=REPO / "runs" / "qcdloop" / "src" / "boxGPU_tracked.cpp",
        existing_shim=COMMITTED_SHIM,
        cfg=None,  # cache hit short-circuits before any generation path
    )

    assert Path(ret).resolve() == COMMITTED_SHIM.resolve()
    assert COMMITTED_SHIM.read_bytes() == before  # byte-identical
    assert COMMITTED_SHIM.stat().st_mtime_ns == mtime_before  # not rewritten
    assert cache.extract_source_hash(before.decode()) == SOURCE_HASH
