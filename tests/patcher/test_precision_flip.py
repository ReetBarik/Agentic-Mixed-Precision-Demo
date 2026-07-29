"""Deliverable 1 — detection + routing for the per-integral precision flip.

Shape-based, deterministic; no app identifiers in the predicates.  The tests build
hand-constructed :class:`CallGraph` fixtures (no libclang) so parametricity and routing
are exercised in isolation.  App-like names (``BO``/``B10``/``Lnrat``) appear here only
as *test data* — the module under test never pattern-matches them.
"""

from __future__ import annotations

from agents.patcher.call_graph import CallGraph, FuncDef
from agents.patcher.precision_flip import (
    FlipDecision, ParametricityResult, Route, TargetPrecision,
    route_integral, subtree_is_parametric)


def _graph(defs: dict[str, list[FuncDef]], edges: dict[str, set[str]],
           root: str = "ENTRY", active=None) -> CallGraph:
    g = CallGraph(root=root, tu_file="tu.h")
    g.defs = defs
    g.edges = {k: set(v) for k, v in edges.items()}
    if active is not None:
        g.active_lines = active
    return g


def _fd(name, file="h.h", ls=1, le=5, tmpl=True) -> FuncDef:
    return FuncDef(name, file, ls, le, tmpl)


# --------------------------------------------------------------------------- #
# subtree_is_parametric
# --------------------------------------------------------------------------- #

def test_fully_parametric_subtree_passes():
    # ENTRY -> MID -> LEAF, all templates; scalar-lib callee (undefined) is unresolved.
    defs = {"ENTRY": [_fd("ENTRY")], "MID": [_fd("MID")], "LEAF": [_fd("LEAF")]}
    edges = {"ENTRY": {"MID"}, "MID": {"LEAF", "sqrt"}, "LEAF": set()}
    g = _graph(defs, edges)
    res = subtree_is_parametric(g, ["LEAF"])
    assert res.parametric is True
    assert set(res.frames_checked) == {"ENTRY", "MID", "LEAF"}
    assert res.non_template == ()
    assert "sqrt" in res.unresolved


def test_non_template_frame_breaks_parametricity():
    defs = {"ENTRY": [_fd("ENTRY")], "MID": [_fd("MID", tmpl=False)],
            "LEAF": [_fd("LEAF")]}
    edges = {"ENTRY": {"MID"}, "MID": {"LEAF"}}
    g = _graph(defs, edges)
    res = subtree_is_parametric(g, ["LEAF"])
    assert res.parametric is False
    assert "MID" in res.non_template


def test_only_frames_on_path_to_target_are_checked():
    # SIDE is a non-template but NOT on the path to LEAF -> must not break the check.
    defs = {"ENTRY": [_fd("ENTRY")], "MID": [_fd("MID")], "LEAF": [_fd("LEAF")],
            "SIDE": [_fd("SIDE", tmpl=False)]}
    edges = {"ENTRY": {"MID", "SIDE"}, "MID": {"LEAF"}}
    g = _graph(defs, edges)
    res = subtree_is_parametric(g, ["LEAF"])
    assert res.parametric is True
    assert "SIDE" not in res.frames_checked


def test_missing_target_frame_fails():
    defs = {"ENTRY": [_fd("ENTRY")]}
    edges = {"ENTRY": set()}
    g = _graph(defs, edges)
    res = subtree_is_parametric(g, ["NOPE"])
    assert res.parametric is False
    assert "NOPE" in res.non_template


def test_empty_targets_is_not_parametric():
    g = _graph({"ENTRY": [_fd("ENTRY")]}, {"ENTRY": set()})
    res = subtree_is_parametric(g, [])
    assert res.parametric is False


def test_active_non_template_overload_breaks_but_inactive_ignored():
    # MID has two overloads: a template (active) and a non-template (inactive under the
    # build defines).  active_defs filters to the template -> parametric.
    defs = {"ENTRY": [_fd("ENTRY", ls=1, le=3)],
            "MID": [_fd("MID", file="m.h", ls=10, le=20, tmpl=True),
                    _fd("MID", file="m.h", ls=30, le=40, tmpl=False)],
            "LEAF": [_fd("LEAF", file="m.h", ls=50, le=60)]}
    edges = {"ENTRY": {"MID"}, "MID": {"LEAF"}}
    from pathlib import Path
    mabs = str(Path("m.h").resolve())
    eabs = str(Path("h.h").resolve())
    active = {mabs: {10, 50}, eabs: {1}}   # only the template MID overload active
    g = _graph(defs, edges, active=active)
    res = subtree_is_parametric(g, ["LEAF"])
    assert res.parametric is True


def test_cycle_is_safe():
    defs = {"ENTRY": [_fd("ENTRY")], "A": [_fd("A")], "B": [_fd("B")]}
    edges = {"ENTRY": {"A"}, "A": {"B"}, "B": {"A"}}   # A<->B cycle, no target reachable
    g = _graph(defs, edges)
    res = subtree_is_parametric(g, ["A"])
    assert res.parametric is True   # A is a template and reachable


# --------------------------------------------------------------------------- #
# route_integral
# --------------------------------------------------------------------------- #

def test_dd_flagged_parametric_routes_to_flip():
    defs = {"ENTRY": [_fd("ENTRY")], "MID": [_fd("MID")]}
    edges = {"ENTRY": {"MID"}}
    g = _graph(defs, edges)
    d = route_integral("B10", dd_flagged=True, graph=g, target_frames=["MID"])
    assert isinstance(d, FlipDecision)
    assert d.route is Route.PRECISION_FLIP
    assert d.target is TargetPrecision.DD


def test_not_dd_flagged_routes_to_raw_double():
    defs = {"ENTRY": [_fd("ENTRY")], "MID": [_fd("MID")]}
    g = _graph(defs, {"ENTRY": {"MID"}})
    d = route_integral("B1", dd_flagged=False, graph=g, target_frames=["MID"])
    assert d.route is Route.RAW_DOUBLE
    assert d.target is None
    assert "not dd-flagged" in d.reason


def test_flagged_but_non_parametric_routes_to_raw_double():
    defs = {"ENTRY": [_fd("ENTRY")], "MID": [_fd("MID", tmpl=False)]}
    g = _graph(defs, {"ENTRY": {"MID"}})
    d = route_integral("B16", dd_flagged=True, graph=g, target_frames=["MID"])
    assert d.route is Route.RAW_DOUBLE
    assert d.parametricity is not None and not d.parametricity.parametric
    assert "not fully template-parametric" in d.reason


def test_target_precision_is_parameterized_not_hardcoded():
    # STOP #SS: routing must honor a non-dd target (Phase 2/3 extensibility).
    defs = {"ENTRY": [_fd("ENTRY")], "MID": [_fd("MID")]}
    g = _graph(defs, {"ENTRY": {"MID"}})
    d = route_integral("B10", dd_flagged=True, graph=g, target_frames=["MID"],
                       target=TargetPrecision.FF)
    assert d.route is Route.PRECISION_FLIP
    assert d.target is TargetPrecision.FF


def test_decision_is_deterministic():
    defs = {"ENTRY": [_fd("ENTRY")], "MID": [_fd("MID")]}
    g = _graph(defs, {"ENTRY": {"MID"}})
    a = route_integral("B10", dd_flagged=True, graph=g, target_frames=["MID"])
    b = route_integral("B10", dd_flagged=True, graph=g, target_frames=["MID"])
    assert a == b


# --------------------------------------------------------------------------- #
# deliverable 5 (Phase-2) — downshift routing
# --------------------------------------------------------------------------- #

from agents.patcher.precision_flip import route_downshift, DOWNSHIFT_PREFERENCE  # noqa: E402


def test_downshift_parametric_raw_double_routes_to_first_available():
    defs = {"ENTRY": [_fd("ENTRY")], "MID": [_fd("MID")], "LEAF": [_fd("LEAF")]}
    g = _graph(defs, {"ENTRY": {"MID"}, "MID": {"LEAF"}})
    d = route_downshift("B1", dd_candidate=False, graph=g, target_frames=["LEAF"],
                        available_targets={TargetPrecision.FLOAT})
    assert d.route is Route.PRECISION_FLIP
    assert d.target is TargetPrecision.FLOAT


def test_downshift_never_touches_a_dd_candidate():
    # STOP #ZZ: a Phase-1 dd accept is never downshifted, regardless of parametricity.
    defs = {"ENTRY": [_fd("ENTRY")], "MID": [_fd("MID")]}
    g = _graph(defs, {"ENTRY": {"MID"}})
    d = route_downshift("B10", dd_candidate=True, graph=g, target_frames=["MID"],
                        available_targets={TargetPrecision.FLOAT})
    assert d.route is Route.RAW_DOUBLE
    assert "dd candidate" in d.reason


def test_downshift_non_parametric_stays_double():
    defs = {"ENTRY": [_fd("ENTRY")], "MID": [_fd("MID", tmpl=False)]}
    g = _graph(defs, {"ENTRY": {"MID"}})
    d = route_downshift("B5", dd_candidate=False, graph=g, target_frames=["MID"],
                        available_targets={TargetPrecision.FLOAT})
    assert d.route is Route.RAW_DOUBLE
    assert "not fully template-parametric" in d.reason


def test_downshift_ff_unavailable_falls_to_float():
    # Preference is (FLOAT, FF); with only FLOAT available FF is filtered out (STOP #EEE).
    assert DOWNSHIFT_PREFERENCE[0] is TargetPrecision.FLOAT
    defs = {"ENTRY": [_fd("ENTRY")], "MID": [_fd("MID")]}
    g = _graph(defs, {"ENTRY": {"MID"}})
    d = route_downshift("B2", dd_candidate=False, graph=g, target_frames=["MID"],
                        available_targets={TargetPrecision.FLOAT})
    assert d.target is TargetPrecision.FLOAT


def test_downshift_no_available_target_stays_double():
    defs = {"ENTRY": [_fd("ENTRY")], "MID": [_fd("MID")]}
    g = _graph(defs, {"ENTRY": {"MID"}})
    d = route_downshift("B3", dd_candidate=False, graph=g, target_frames=["MID"],
                        available_targets=set())
    assert d.route is Route.RAW_DOUBLE
    assert "no available downshift target" in d.reason


def test_downshift_prefers_cheapest_when_multiple_available():
    # If both FLOAT and FF were available, the cheapest (FLOAT, first in preference) wins.
    defs = {"ENTRY": [_fd("ENTRY")], "MID": [_fd("MID")]}
    g = _graph(defs, {"ENTRY": {"MID"}})
    d = route_downshift("B4", dd_candidate=False, graph=g, target_frames=["MID"],
                        available_targets={TargetPrecision.FLOAT, TargetPrecision.FF})
    assert d.target is TargetPrecision.FLOAT
