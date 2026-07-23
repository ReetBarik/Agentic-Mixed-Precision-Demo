"""Symbol-presence / variant-wiring gate — source-level orphan guard (always) plus
nm telemetry (needs g++)."""

from __future__ import annotations

from pathlib import Path

from agents.patcher import gates
from tests.patcher.fanout.conftest import gxx_compile, requires_gxx

# --- a self-contained, compilable tree with a real (non-inline) variant symbol ---
GOOD_CPP = """\
namespace app {
double f_g_h_B1(double x) { return x * 2.0; }   // variant, external linkage
double g_h_B1(double x)   { return f_g_h_B1(x) + 1.0; }
}
int main() { return (int) app::g_h_B1(3.0); }
"""

# --- a tree whose variant is orphaned: defined (inline, so no symbol) but uncalled ---
ORPHAN_CPP = """\
namespace app {
inline double orphan_B1(double x) { return x; }   // never called -> no symbol
double real(double x) { return x + 1.0; }
}
int main() { return (int) app::real(2.0); }
"""


def test_wiring_ok_source_level(tmp_path):
    (tmp_path / "app.cpp").write_text(GOOD_CPP)
    assert gates.check_variant_wiring(tmp_path, ["f_g_h_B1", "g_h_B1"]) is None


def test_wiring_flags_orphan_source_level(tmp_path):
    (tmp_path / "app.cpp").write_text(ORPHAN_CPP)
    err = gates.check_variant_wiring(tmp_path, ["orphan_B1"])
    assert err is not None and "rename_cascade_incomplete" in err
    assert "orphan_B1" in err


def test_referenced_variants(tmp_path):
    (tmp_path / "app.cpp").write_text(GOOD_CPP)
    ref = gates.referenced_variants(tmp_path, ["f_g_h_B1", "g_h_B1"])
    assert ref == {"f_g_h_B1", "g_h_B1"}


def test_no_silent_bypass_ok(tmp_path):
    (tmp_path / "app.cpp").write_text(GOOD_CPP)
    # f_g_h_B1 is called by g_h_B1 -> the reroute landed
    assert gates.check_no_silent_bypass(tmp_path, {"f": "f_g_h_B1"}) is None


def test_no_silent_bypass_detects_missed_reroute(tmp_path):
    (tmp_path / "app.cpp").write_text(GOOD_CPP)
    err = gates.check_no_silent_bypass(tmp_path, {"ghost": "ghost_B1"})
    assert err is not None and "silent_bypass" in err


@requires_gxx
def test_nm_finds_non_inline_variant(tmp_path):
    src = tmp_path / "app.cpp"
    src.write_text(GOOD_CPP)
    obj = tmp_path / "app.o"
    r = gxx_compile(src, obj)
    assert r.returncode == 0, r.stderr
    present, absent = gates.variant_symbols_present(obj, ["f_g_h_B1", "g_h_B1"])
    assert "f_g_h_B1" in present         # external variant symbol emitted


@requires_gxx
def test_nm_absent_for_inlined_orphan(tmp_path):
    src = tmp_path / "app.cpp"
    src.write_text(ORPHAN_CPP)
    obj = tmp_path / "app.o"
    r = gxx_compile(src, obj)
    assert r.returncode == 0, r.stderr
    present, absent = gates.variant_symbols_present(obj, ["orphan_B1"])
    # inline + uncalled -> no emitted symbol (why nm alone can't be the guard);
    # the source-level wiring check is what catches this orphan.
    assert "orphan_B1" in absent
    assert gates.check_variant_wiring(tmp_path, ["orphan_B1"]) is not None
