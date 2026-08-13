"""Instantiation-gate wiring in the tier_b_stage1 harness (the pre-measurement gate).

Loads the ``_instantiation_gate`` helper from the run harness by file path (the
harness is a script under runs/, not an importable package) and pins its three
branches: a clean binding failure → ``instantiation_binding`` tag, an unknown shape
→ STOP #BB, an errorless log → no tag.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_HARNESS = _ROOT / "runs" / "qcdloop" / "tier_b_stage1.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("tier_b_stage1_h", _HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load_harness()


def _write_log(tmp_path, body: str) -> Path:
    p = tmp_path / "iter_0_build.log"
    p.write_text(body, encoding="utf-8")
    return p


def test_binding_failure_gets_instantiation_tag(tmp_path):
    log = _write_log(tmp_path,
        "b.h:1:1: error: invalid cast from type "
        "‘Kokkos::Experimental::DoubleDouble’ to type ‘double’\n"
        "b.h:2:1: error: no matching function for call to "
        "‘Kokkos::complex<double>::complex(Kokkos::Experimental::DoubleDouble)’\n")
    tag, report = H._instantiation_gate(str(log), tmp_path, 0)
    assert tag == "instantiation_binding"
    assert report.total == 2
    assert not report.has_unknown
    # A per-shape report file is written alongside the run.
    assert (tmp_path / "instantiation_gate_iter_0.json").is_file()


def test_unknown_shape_raises_stop_bb(tmp_path):
    log = _write_log(tmp_path,
        "b.h:1:1: error: use of undeclared identifier ‘frobnicate’\n")
    with pytest.raises(H.InstantiationStopBB):
        H._instantiation_gate(str(log), tmp_path, 0)


def test_errorless_log_is_not_a_binding_failure(tmp_path):
    log = _write_log(tmp_path, "[100%] Built target boxGPU_app\n")
    tag, report = H._instantiation_gate(str(log), tmp_path, 0)
    assert tag is None
    assert report.ok


def test_missing_log_is_not_a_binding_failure(tmp_path):
    tag, report = H._instantiation_gate(None, tmp_path, 0)
    assert tag is None
    assert report.ok
