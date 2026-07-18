"""P5 gate classification — NaN / short output / crash / build timeout."""

import subprocess

import pytest

from agents.patcher import gates, result as R


def _res(n, bad_at=None):
    """n RES rows (6 coeff columns); optionally inject 'nan' at row bad_at."""
    lines = []
    for i in range(n):
        coeffs = ["1.0"] * 6
        if bad_at == i:
            coeffs[2] = "nan"
        lines.append(f"RES,B{i},{i}," + ",".join(coeffs))
    return "\n".join(lines) + "\n"


def _cp(stdout, rc=0):
    return subprocess.CompletedProcess(["x"], rc, stdout, "")


def test_scan_ok():
    g = gates._scan_smoke(_cp(_res(21)), 21, None, None)
    assert g.status == R.OK


def test_scan_nan():
    g = gates._scan_smoke(_cp(_res(21, bad_at=5)), 21, None, None)
    assert g.status == R.RUNTIME_NAN and g.err_kind == R.ERR_NAN


def test_scan_short_output_is_crash():
    g = gates._scan_smoke(_cp(_res(10)), 21, None, None)
    assert g.status == R.RUNTIME_CRASHED
    assert "10 result rows" in g.detail


def test_scan_nonzero_exit_is_crash():
    g = gates._scan_smoke(_cp(_res(21), rc=139), 21, None, None)
    assert g.status == R.RUNTIME_CRASHED and g.err_kind == R.ERR_CRASH


def test_scan_ignores_nan_in_integral_name():
    # a token like the integral label must not be scanned as a coeff NaN
    out = "RES,BINF,0," + ",".join(["1.0"] * 6) + "\n" + _res(20)
    g = gates._scan_smoke(_cp(out), 21, None, None)
    assert g.status == R.OK


def test_build_timeout(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="cmake", timeout=1)

    monkeypatch.setenv("PIPELINE_MODULE_LIST", "")
    monkeypatch.setattr(gates.subprocess, "run", boom)
    g = gates.run_gate(tmp_path / "hdr", tmp_path / "build", tmp_path / "logs", 1,
                       build_timeout=1)
    assert g.status == R.TIMEOUT and g.err_kind == R.ERR_TIMEOUT
