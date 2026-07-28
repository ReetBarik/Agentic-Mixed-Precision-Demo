"""STOP #A dispatch fix — preprocessor-active definition selection.

``graph.defs[name]`` can hold several definitions of one name where only ONE
survives preprocessing under the app's build defines.  qcdloop's box dispatch is the
motivating case: each group header carries a pruned ``BO`` guarded by
``#ifndef QCDLOOP_BOX_FULL_DISPATCH``, and ``boxGPU.h`` ``#define``s that macro before
including them, so only the meta-header's full-dispatch ``BO`` is live.  Before this
fix ``_pick_def`` / ``_resolve_root_file`` blindly returned ``defs[0]`` — the first
pruned copy — and the root reroute landed on dead code (STOP #A: B10 lift 0.0).

These tests pin the fix at three levels:

* the real qcdloop tree — ``_pick_def("BO")`` returns the live ``boxGPU.h`` def;
* a synthetic guarded fixture — the ``-D``-killed copy is filtered out;
* a synthetic two-live-candidate fixture — the pick fails loud (no silent ``defs[0]``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.patcher import fanout
from agents.patcher.call_graph import build_call_graph
from agents.patcher.fanout import FanoutError, _pick_def, _resolve_root_file
from agents.patcher.preprocessor import (compute_active_lines, defines_from_args)
from tests.patcher.fanout.conftest import (requires_libclang, requires_qcdloop_full)

# --------------------------------------------------------------------------- #
# (b.1) real tree: _pick_def("BO") -> the live boxGPU.h def, not the pruned copy
# --------------------------------------------------------------------------- #


@requires_libclang
@requires_qcdloop_full
def test_pick_def_bo_selects_live_dispatch(qcdloop_full_graph):
    """On the real tree, ``BO`` has 6 defs (5 pruned + 1 live); the pick is the live
    ``boxGPU.h:69`` full-dispatch, NOT the ``#ifndef``-dead ``B0m.h:432`` at defs[0]."""
    g = qcdloop_full_graph
    # precondition: the exact defect the fix targets — several BO defs, defs[0] pruned.
    assert len(g.defs["BO"]) >= 2
    assert Path(g.defs["BO"][0].file).name != "boxGPU.h", (
        "fixture drift: defs[0] is no longer the pruned copy this test guards")

    fd = _pick_def(g, "BO")
    assert Path(fd.file).name == "boxGPU.h", (
        f"_pick_def(BO) landed on {Path(fd.file).name}:{fd.line_start}, "
        f"want the live boxGPU.h dispatch")
    assert fd.line_start == 69

    # _resolve_root_file must agree (the reroute file and the picked def are the same)
    assert Path(_resolve_root_file(g)).name == "boxGPU.h"


@requires_libclang
@requires_qcdloop_full
def test_pruned_bo_copies_are_inactive(qcdloop_full_graph):
    """Every pruned group-header ``BO`` is preprocessor-dead; only the live one is active."""
    g = qcdloop_full_graph
    active = [fd for fd in g.defs["BO"] if g.def_is_active(fd)]
    assert len(active) == 1, [f"{Path(f.file).name}:{f.line_start}" for f in active]
    assert Path(active[0].file).name == "boxGPU.h"


@requires_libclang
@requires_qcdloop_full
def test_chain_frames_still_resolve_singly(qcdloop_full_graph):
    """The active filter narrows only the ambiguous entry point; single-def chain
    frames (B1m/B10/B0m/B2m) and both live helper overloads are unaffected."""
    g = qcdloop_full_graph
    for name in ("B1m", "B10", "B0m", "B2m"):
        fd = _pick_def(g, name)
        assert fd is not None and fd.name == name
    # Lnrat/Li2omx2 have two overloads, BOTH preprocessor-active (the overload choice
    # is _select_leaf_overload's job, not the preprocessor filter's).
    for name in ("Lnrat", "Li2omx2"):
        assert len(g.active_defs(name)) == 2, name


# --------------------------------------------------------------------------- #
# (b.2) synthetic: a guarded copy killed by -DX is filtered out
# --------------------------------------------------------------------------- #

# Two BO definitions: one guarded by ``#ifndef GUARD`` (the group-header analogue),
# one unguarded (the live full-dispatch analogue).  The meta-header ``#define``s GUARD
# before including the guarded one, exactly like boxGPU.h does for the real tree.
_GUARDED_H = """\
#pragma once
namespace app {
#ifndef GUARD
template<class T>
void BO(T& r) { r = pruned<T>(r); }
#endif

template<class T>
void pruned(T x) {}
}  // namespace app
"""

_META_H = """\
#pragma once
#define GUARD
#include "guarded.h"
namespace app {

template<class T>
void live(T x) {}

template<class T>
void BO(T& r) { live<T>(r); }
}  // namespace app
"""


def _write(tmp_path: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    return tmp_path


@requires_libclang
def test_defined_guard_prunes_copy(tmp_path):
    """With ``GUARD`` ``#define``d before the guarded header, the guarded ``BO`` copy is
    preprocessor-dead; ``_pick_def`` returns the unguarded (live) one — no ambiguity."""
    tree = _write(tmp_path, {"guarded.h": _GUARDED_H, "meta.h": _META_H})
    fanout.clear_graph_cache()
    g = build_call_graph("BO", tree, tu_file=tree / "meta.h")

    # both defs are enumerated by libclang...
    assert len(g.defs["BO"]) == 2
    # ...but only the unguarded (meta.h) one is preprocessor-active.
    active = g.active_defs("BO")
    assert len(active) == 1 and Path(active[0].file).name == "meta.h"

    fd = _pick_def(g, "BO")
    assert Path(fd.file).name == "meta.h"


@requires_libclang
def test_extra_args_define_prunes_copy(tmp_path):
    """A ``-D`` from ``extra_args`` (not an in-source ``#define``) also seeds the walk:
    guarding on a build-time macro prunes the copy the same way."""
    # guarded header included WITHOUT an in-source #define of GUARD; the guard is set
    # only via extra_args -DGUARD, so the walk must honour the build define.
    meta = ("#pragma once\n#include \"guarded.h\"\nnamespace app {\n"
            "template<class T>\nvoid live(T x) {}\n"
            "template<class T>\nvoid BO(T& r) { live<T>(r); }\n}\n")
    tree = _write(tmp_path, {"guarded.h": _GUARDED_H, "meta.h": meta})
    fanout.clear_graph_cache()
    g = build_call_graph("BO", tree, tu_file=tree / "meta.h",
                         extra_args=["-DGUARD"])
    active = g.active_defs("BO")
    assert len(active) == 1 and Path(active[0].file).name == "meta.h"
    assert Path(_pick_def(g, "BO").file).name == "meta.h"


# --------------------------------------------------------------------------- #
# (b.3) synthetic: two LIVE candidates -> fail loud with a diagnostic (no defs[0])
# --------------------------------------------------------------------------- #

# Two BO definitions, NEITHER guarded — both preprocessor-active.  ``_pick_def`` must
# refuse to guess and raise a diagnostic naming both, rather than silently pick defs[0]
# (the exact regression that caused STOP #A).
_TWO_LIVE_A = """\
#pragma once
namespace app {
template<class T>
void BO(T& r) { r = one<T>(r); }
template<class T>
void one(T x) {}
}  // namespace app
"""

_TWO_LIVE_META = """\
#pragma once
#include "a.h"
namespace app {
template<class T>
void two(T x) {}
template<class T>
void BO(T& r) { two<T>(r); }
}  // namespace app
"""


@requires_libclang
def test_two_live_candidates_fail_loud(tmp_path):
    """Two unguarded ``BO`` defs both survive preprocessing; ``_pick_def`` raises with a
    diagnostic naming both files — it must never fall back to a silent ``defs[0]``."""
    tree = _write(tmp_path, {"a.h": _TWO_LIVE_A, "meta.h": _TWO_LIVE_META})
    fanout.clear_graph_cache()
    g = build_call_graph("BO", tree, tu_file=tree / "meta.h")

    assert len(g.active_defs("BO")) == 2, "fixture: both BO must be active"
    with pytest.raises(FanoutError) as ei:
        _pick_def(g, "BO")
    msg = str(ei.value)
    assert "ambiguous" in msg.lower()
    assert "a.h" in msg and "meta.h" in msg          # both candidates named
    assert "STOP #A" in msg                            # the guardrail is called out
    # _resolve_root_file goes through the same pick -> same fail-loud contract.
    with pytest.raises(FanoutError):
        _resolve_root_file(g)


# --------------------------------------------------------------------------- #
# fallback: an empty active_lines map (hand-built graph) -> unfiltered, pre-fix behaviour
# --------------------------------------------------------------------------- #


def test_empty_active_lines_reports_all_active():
    """A graph with no preprocessor walk (empty ``active_lines``) reports every def
    active — the selection degrades to the unfiltered path (tests / A-B runs)."""
    from agents.patcher.call_graph import CallGraph, FuncDef
    g = CallGraph(root="BO", tu_file="t.h")
    g.defs["BO"] = [FuncDef("BO", "/x/a.h", 1, 5, True),
                    FuncDef("BO", "/x/b.h", 1, 5, True)]
    assert g.def_is_active(g.defs["BO"][0])
    assert len(g.active_defs("BO")) == 2


# --------------------------------------------------------------------------- #
# unit: the preprocessor walk + -D extraction in isolation
# --------------------------------------------------------------------------- #


def test_defines_from_args_forms():
    assert defines_from_args(["-DFOO"]) == {"FOO"}
    assert defines_from_args(["-DFOO=1"]) == {"FOO"}
    assert defines_from_args(["-D", "BAR"]) == {"BAR"}
    assert defines_from_args(["-I/x", "-DA", "-DB=2", "-std=c++17"]) == {"A", "B"}
    assert defines_from_args([]) == set()


def test_compute_active_lines_honours_ifndef(tmp_path):
    """The walk marks a ``#ifndef``-guarded body dead once the guard is defined."""
    _write(tmp_path, {"guarded.h": _GUARDED_H, "meta.h": _META_H})
    active = compute_active_lines(tmp_path / "meta.h", [str(tmp_path)])
    guarded = str((tmp_path / "guarded.h").resolve())
    # the guarded BO body (line 5 in _GUARDED_H) is inside #ifndef GUARD -> dead.
    assert 5 not in active.get(guarded, set())
    # the unguarded pruned() def (line 9) is live.
    assert 9 in active.get(guarded, set())


@requires_qcdloop_full
def test_compute_active_lines_real_tree():
    """On the real tree the walk yields exactly one active BO head line (boxGPU.h:69)."""
    from tests.patcher.fanout.conftest import _QCDLOOP_FULL
    tree = _QCDLOOP_FULL.resolve()
    active = compute_active_lines(tree / "boxGPU.h", [str(tree), str(tree / "box")])
    boxgpu = str((tree / "boxGPU.h").resolve())
    b0m = str((tree / "box" / "B0m.h").resolve())
    assert 69 in active.get(boxgpu, set())        # live full-dispatch BO
    assert 432 not in active.get(b0m, set())       # pruned B0m BO is dead
