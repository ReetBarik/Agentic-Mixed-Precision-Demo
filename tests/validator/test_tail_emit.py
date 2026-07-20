"""emit_tail_offsets.py — tail-offset selection + report augmentation.

Selection is tested on synthetic vanilla/DD coeff buffers with planted worst
samples.  The augmentation path is exercised end-to-end with the four driver
seams (vanilla build, DD build, run, determinism hash) monkeypatched, so no
Kokkos / compiled driver is needed — it confirms the ``tail_samples`` field is
written with the expected offsets + a determinism hash and that the original
report is preserved as ``*.pre_tail.json``.
"""

import importlib.util
import json
from array import array
from pathlib import Path

import pytest

from agents.validator import runner, tail
from agents.validator.coeffs import N_COMPONENTS

_REPO = Path(__file__).resolve().parents[2]
_EMIT_PATH = _REPO / "runs" / "qcdloop" / "emit_tail_offsets.py"


def _load_emit():
    spec = importlib.util.spec_from_file_location("emit_tail_offsets", _EMIT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


emit_mod = _load_emit()


def _buffers(total: int, samples: dict) -> tuple:
    """Build (hi, lo) arrays of length total*6 for a *DD* integral.

    ``samples[s]`` is a dict {component_index: value}; unset components are 0.
    """
    hi = array("d", bytes(8 * total * N_COMPONENTS))
    lo = array("d", bytes(8 * total * N_COMPONENTS))
    for s, comps in samples.items():
        for c, v in comps.items():
            hi[s * N_COMPONENTS + c] = v
    return (hi, lo)


def test_select_tail_picks_planted_extremes():
    total = 8
    # DD reference: plant one worst sample per criterion on coeff0.real (c=0),
    # using coeff1.real (c=2) to raise ref_scale where needed.
    dd_samples = {
        0: {0: 1.0},
        1: {0: 1.0},               # worst rel-err lives here (candidate off by 1e-2)
        2: {0: 1e30},              # max abs value
        3: {0: 1e-20},             # min nonzero abs value
        4: {0: 1e-10, 2: 1e5},     # worst cancellation cond: 1e5 / 1e-10 = 1e15
        5: {0: 1.0},
        6: {0: 1.0},               # 2nd-worst rel-err (candidate off by 1e-4)
        7: {0: 1.0},
    }
    dd = _buffers(total, dd_samples)
    # vanilla == DD everywhere except the two rel-err samples.
    v_hi = array("d", dd[0])
    v_lo = array("d", bytes(8 * total * N_COMPONENTS))
    v_hi[1 * N_COMPONENTS + 0] = 1.0 + 1e-2
    v_hi[6 * N_COMPONENTS + 0] = 1.0 + 1e-4
    van = (v_hi, v_lo)

    ts = emit_mod._select_tail(van, dd, total, k=2)

    assert ts["max_rel_err"][0]["offset"] == 1
    assert ts["max_rel_err"][0]["output_component"] == "coeff0.real"
    assert ts["max_rel_err"][0]["criterion_value"] == pytest.approx(1e-2, rel=1e-3)
    assert ts["max_rel_err"][1]["offset"] == 6           # 2nd worst

    assert ts["max_cond"][0]["offset"] == 4
    assert ts["max_cond"][0]["criterion_value"] == pytest.approx(1e15, rel=1e-6)

    assert ts["max_abs_value"][0]["offset"] == 2
    assert ts["max_abs_value"][0]["criterion_value"] == pytest.approx(1e30, rel=1e-9)

    assert ts["min_abs_value"][0]["offset"] == 3
    assert ts["min_abs_value"][0]["criterion_value"] == pytest.approx(1e-20, rel=1e-9)


def test_select_tail_excludes_analytic_zeros_from_min_abs():
    # An analytic zero (ref far below ZERO_REF_TOL * ref_scale) must NOT win
    # min_abs_value — that would preserve noise, not a genuine tiny output.
    total = 3
    dd_samples = {
        0: {0: 1.0, 1: 1e-30},   # coeff0.imag is an analytic zero vs scale 1.0
        1: {0: 1.0},
        2: {0: 1e-12, 1: 1.0},   # coeff0.real 1e-12 is a genuine tiny value
    }
    dd = _buffers(total, dd_samples)
    van = (array("d", dd[0]), array("d", bytes(8 * total * N_COMPONENTS)))
    ts = emit_mod._select_tail(van, dd, total, k=1)
    # the genuine 1e-12 wins, not the 1e-30 analytic zero.
    assert ts["min_abs_value"][0]["offset"] == 2
    assert ts["min_abs_value"][0]["criterion_value"] == pytest.approx(1e-12, rel=1e-6)


def test_emit_augments_report_and_preserves_original(tmp_path, monkeypatch):
    total = 6
    # synthetic report with two integrals
    report = {
        "schema_version": 1,
        "integrals": {
            "B1": {"samples": total, "regions": {}},
            "B12": {"samples": total, "regions": {}},
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    vanilla_headers = tmp_path / "headers"
    vanilla_headers.mkdir()
    (vanilla_headers / "boxGPU.h").write_text("// dummy\n")

    dd_samples = {s: {0: 1.0 + s} for s in range(total)}
    dd = _buffers(total, dd_samples)
    van = (array("d", dd[0]), array("d", bytes(8 * total * N_COMPONENTS)))
    synthetic = {"B1": (array("d", van[0]), array("d", van[1])),
                 "B12": (array("d", van[0]), array("d", van[1]))}
    synthetic_dd = {"B1": (array("d", dd[0]), array("d", dd[1])),
                    "B12": (array("d", dd[0]), array("d", dd[1]))}

    monkeypatch.setattr(emit_mod.runner, "build_driver",
                        lambda *a, **k: Path("/fake/vanilla_bin"))
    monkeypatch.setattr(emit_mod._validate, "_build_dd_binary",
                        lambda *a, **k: Path("/fake/dd_bin"))

    def _fake_run(binary, tot, **kw):
        return synthetic_dd if "dd" in str(binary) else synthetic
    monkeypatch.setattr(emit_mod.runner, "run_and_aggregate", _fake_run)
    monkeypatch.setattr(emit_mod.tail, "determinism_hash",
                        lambda binary, n=tail.DETERMINISM_N:
                        {"B1": "sha256:h1", "B12": "sha256:h12"})

    summary = emit_mod.emit(report_path, total, k=2,
                            dd_repo=Path("/x"), dd_ref="ref",
                            kokkos_root=Path("/k"),
                            vanilla_headers=vanilla_headers,
                            chunk=0, workers=1)

    assert summary["integrals_augmented"] == 2
    assert summary["hash_present"] == 2

    # original preserved
    pre = report_path.with_suffix(".pre_tail.json")
    assert pre.exists()
    assert "tail_samples" not in json.loads(pre.read_text())["integrals"]["B1"]

    # augmented report carries tail_samples + determinism hash
    aug = json.loads(report_path.read_text())
    for b in ("B1", "B12"):
        ts = aug["integrals"][b]["tail_samples"]
        assert ts["determinism_hash"].startswith("sha256:")
        for crit in tail.CRITERIA:
            assert crit in ts
    assert aug["tail_schema_version"] == 1
