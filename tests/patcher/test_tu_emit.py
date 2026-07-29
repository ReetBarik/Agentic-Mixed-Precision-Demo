"""Deliverable 2 — per-integral TU emission + precision-parameterized wrapper.

Unit-level: renders + group discovery + snapshot guard + precision parameterization.
The end-to-end g++ compile of the emitted TU against a snapshot clone is exercised by
the harness (it needs the module env + Kokkos); here we assert the *shape* the compile
depends on (fork-shape wrapper arm, pruned-group include, profile-driven template args).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.patcher.precision_flip import TargetPrecision
from agents.patcher.tu_emit import (
    PROFILES, TUEmitError, emit_flip_tu, group_header_for_files,
    profile_for, render_group_driver, render_wrapper)


# --------------------------------------------------------------------------- #
# wrapper generator
# --------------------------------------------------------------------------- #

def test_wrapper_has_dd_arm_the_snapshot_lacks():
    w = render_wrapper(TargetPrecision.DD)
    # The load-bearing §5.5 arm: USE_DD_COMPLEX -> kokkosMaths_dd.h.
    assert "#if defined(USE_DD_COMPLEX)" in w
    assert '#include "kokkosMaths_dd.h"' in w
    # Preserves the quad arm + double default (fork shape).
    assert "USE_QUAD_COMPLEX" in w
    assert '#include "kokkosMaths.h"' in w
    assert w.count("#pragma once") == 1


def test_wrapper_ladder_is_wellformed():
    w = render_wrapper(TargetPrecision.DD)
    # Exactly one opening #if, and the dd arm precedes the quad arm as #elif.
    assert w.index("#if defined(USE_DD_COMPLEX)") < w.index("#elif defined(USE_QUAD_COMPLEX)")
    # Balanced #if/#endif (2 opens: dd-ladder head + inner CUDA guard -> matches design).
    assert w.count("#endif") == 2


# --------------------------------------------------------------------------- #
# group driver generator
# --------------------------------------------------------------------------- #

def test_group_driver_includes_group_header_not_meta_header():
    d = render_group_driver("box/B1m.h", TargetPrecision.DD)
    assert '#include "box/B1m.h"' in d
    # Must NOT pull the meta-header (that would define QCDLOOP_BOX_FULL_DISPATCH and
    # instantiate every group -> loses per-group isolation).
    assert '#include "boxGPU.h"' not in d
    # The full-dispatch macro must never be DEFINED in the driver (a comment naming it
    # is fine); defining it would activate the meta-dispatch and lose isolation.
    assert "#define QCDLOOP_BOX_FULL_DISPATCH" not in d


def test_group_driver_dd_template_args_and_define():
    d = render_group_driver("box/B2m.h", TargetPrecision.DD)
    assert "#define USE_DD_COMPLEX" in d
    assert "ql::ddfun::ddcomplex" in d
    assert "ql::ddfun::ddouble" in d
    assert "DDPrinter" in d
    assert "run_app<" in d


def test_group_driver_reuses_shared_recipe():
    d = render_group_driver("box/B1m.h", TargetPrecision.DD)
    assert '#include "boxGPU_app_recipes.hpp"' in d


# --------------------------------------------------------------------------- #
# group discovery (structural, no integral->group table)
# --------------------------------------------------------------------------- #

def test_group_discovery_selects_box_group_header():
    assert group_header_for_files(["B1m.h", "kokkosUtils.h"]) == "box/B1m.h"
    assert group_header_for_files(["/x/y/B2m.h"]) == "box/B2m.h"


def test_group_discovery_ignores_non_group_headers():
    # kokkosMaths.h / kokkosUtils.h are not B<k>m.h group headers.
    assert group_header_for_files(["B3m.h", "kokkosMaths.h", "timer.h"]) == "box/B3m.h"


def test_group_discovery_rejects_no_group():
    with pytest.raises(TUEmitError):
        group_header_for_files(["kokkosUtils.h", "timer.h"])


def test_group_discovery_rejects_multi_group():
    with pytest.raises(TUEmitError):
        group_header_for_files(["B1m.h", "B2m.h"])


# --------------------------------------------------------------------------- #
# precision parameterization (STOP #SS)
# --------------------------------------------------------------------------- #

def test_dd_profile_available():
    assert profile_for(TargetPrecision.DD).precision is TargetPrecision.DD


def test_unavailable_precision_fails_loud_not_dd_fallback():
    # ff/float profiles exist (parameterization) but their headers aren't vendored yet:
    # selecting one must fail, never silently degrade to dd (reverse-STOP #SS).
    assert not PROFILES[TargetPrecision.FF].available
    with pytest.raises(TUEmitError):
        profile_for(TargetPrecision.FF)
    with pytest.raises(TUEmitError):
        render_group_driver("box/B1m.h", TargetPrecision.FF)


def test_profiles_declare_all_three_targets():
    # The table is parameterized across all three precisions from the start.
    assert set(PROFILES) == {TargetPrecision.DD, TargetPrecision.FF, TargetPrecision.FLOAT}


# --------------------------------------------------------------------------- #
# emission + snapshot guard (STOP #Z)
# --------------------------------------------------------------------------- #

def _fake_tree(root: Path) -> Path:
    (root / "box").mkdir(parents=True)
    (root / "boxGPU.h").write_text("// meta\n")
    (root / "box" / "B1m.h").write_text("// group\n")
    (root / "kokkosMaths_wrapper.h").write_text("// old wrapper\n")
    return root


def test_emit_writes_wrapper_and_driver_into_clone(tmp_path):
    clone = _fake_tree(tmp_path / "tree")
    drv = tmp_path / "drv"
    tu = emit_flip_tu(clone, "box/B1m.h", drv, TargetPrecision.DD)
    assert tu.wrapper_path == clone / "kokkosMaths_wrapper.h"
    assert "USE_DD_COMPLEX" in tu.wrapper_path.read_text()
    assert tu.driver_path.exists()
    assert '#include "box/B1m.h"' in tu.driver_path.read_text()


def test_emit_refuses_snapshot_write():
    # Point at the real pristine snapshot: the guard must fire before any write.
    snap = Path(__file__).resolve().parents[2] / "runs" / "qcdloop_headers_full"
    with pytest.raises(TUEmitError):
        emit_flip_tu(snap, "box/B1m.h", Path("/tmp/should_not_matter"))


def test_emit_rejects_non_tree(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(TUEmitError):
        emit_flip_tu(empty, "box/B1m.h", tmp_path / "drv")
