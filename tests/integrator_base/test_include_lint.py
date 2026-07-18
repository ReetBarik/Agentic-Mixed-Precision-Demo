"""Unit tests for the C1 include-set lint (agents/integrator_base/regional.py).

The lint is the deterministic safety net behind the system-prompt C1 rule: a
regional shim's `#include` set is closed to the vendored extended-precision
headers (plus harmless stdlib headers).  Any app-source `#include`
(``ql/constants.h``, ``qcdloop/types.h``, ``Kokkos_*.hpp``) is a misgeneration —
that path is not on the shim's include path and guarantees a hard build failure.
This reproduces the dominant Patcher failure of run 20260718_194556_67dbcf37.
"""

from __future__ import annotations

from agents.dd_integrator import agent as dd
from agents.ff_integrator import agent as ff
from agents.integrator_base import regional as r

# The exact hallucinated includes pulled from the 2026-07-18 shakedown shims.
_OBSERVED_BAD = [
    '#include "ql/constants.h"',      # B2m_dd_4ca3edc1.h:28
    '#include "ql/maths.h"',          # B2m_dd_4ca3edc1.h:29
    "#include <qcdloop/types.h>",     # B0m_dd_d6ffdd47.h:25
    "#include <qcdloop/qcdloop.h>",   # B0m_dd_8501ce27.h:29
    "#include <qcdloop/constants.h>", # B3m_dd_c5440ce5.h:5
    "#include <Kokkos_Macros.hpp>",   # B2m_dd_3328f5c9.h:5
    "#include <Kokkos_Array.hpp>",    # B2m_dd_3328f5c9.h:6
]

_DD_ALLOWED = r._allowed_include_set(dd._SPEC)
_FF_ALLOWED = r._allowed_include_set(ff._SPEC)


def _shim(*include_lines: str) -> str:
    body = ["#pragma once", "// SOURCE_HASH: PENDING"]
    body.extend(include_lines)
    body.append("namespace quad { namespace ddfun {} }")
    return "\n".join(body) + "\n"


def test_clean_dd_shim_passes():
    shim = _shim("#include <dd_math.hpp>", "#include <dd_complex.hpp>")
    assert r._lint_include_set(shim, _DD_ALLOWED) is None


def test_clean_ff_shim_passes():
    shim = _shim("#include <ff_math.hpp>", "#include <ff_complex.hpp>")
    assert r._lint_include_set(shim, _FF_ALLOWED) is None


def test_stdlib_headers_allowed():
    # Harmless: always on the include path, never the failure mode.
    shim = _shim("#include <dd_math.hpp>", "#include <cstdint>", "#include <complex>")
    assert r._lint_include_set(shim, _DD_ALLOWED) is None


def test_quoted_vendored_header_allowed():
    # Basename resolves whether angle- or quote-included; don't false-reject.
    shim = _shim('#include "dd_math.hpp"', '#include "dd_complex.hpp"')
    assert r._lint_include_set(shim, _DD_ALLOWED) is None


def test_each_observed_hallucination_rejected():
    for bad in _OBSERVED_BAD:
        shim = _shim("#include <dd_math.hpp>", bad)
        msg = r._lint_include_set(shim, _DD_ALLOWED)
        assert msg is not None, f"lint failed to reject {bad!r}"
        assert "C1 include lint" in msg


def test_ff_headers_are_app_source_for_dd_and_rejected():
    # A dd shim must not pull ff headers and vice-versa — cross-vendored is still
    # outside the dd allowlist.
    shim = _shim("#include <dd_math.hpp>", "#include <ff_math.hpp>")
    assert r._lint_include_set(shim, _DD_ALLOWED) is not None


def test_commented_include_is_ignored():
    # A `//`-commented include is not a real directive — must not trip the lint.
    shim = _shim("#include <dd_math.hpp>", '// #include "ql/constants.h" (do NOT)')
    assert r._lint_include_set(shim, _DD_ALLOWED) is None


def test_multiple_forbidden_all_reported():
    shim = _shim("#include <dd_math.hpp>",
                 '#include "ql/constants.h"', "#include <qcdloop/types.h>")
    msg = r._lint_include_set(shim, _DD_ALLOWED)
    assert msg is not None
    assert "ql/constants.h" in msg and "qcdloop/types.h" in msg


def test_allowed_set_is_vendored_plus_stdlib():
    assert "dd_math.hpp" in _DD_ALLOWED and "dd_complex.hpp" in _DD_ALLOWED
    assert "ff_math.hpp" in _FF_ALLOWED and "ff_complex.hpp" in _FF_ALLOWED
    assert "cstdint" in _DD_ALLOWED  # stdlib always permitted
    assert "ql/constants.h" not in _DD_ALLOWED
