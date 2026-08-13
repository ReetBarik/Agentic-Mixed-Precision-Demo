"""Unit tests for the Gap-A namespace-qualified bridge scan + lint
(agents/integrator_base/regional.py).

A namespace-qualified math call ``Ns::fn(promoted)`` skips ADL, so the shim must
inject a bridge into ``Ns`` (or a using-declaration).  These tests pin the
detector (which calls need a bridge) and the deterministic lint (a shim missing a
bridge is a retryable misgeneration).  Framework-agnostic: the trigger is the
pattern, exercised here with Kokkos / std / sycl / cuda::std / a made-up namespace.
"""

from __future__ import annotations

from agents.integrator_base import regional as r

_PROMOTED = frozenset({"x", "k34", "a"})


def _calls(region: str, promoted=_PROMOTED):
    return r.find_qualified_math_calls(region, promoted)


# --------------------------------------------------------------------------- #
# detector
# --------------------------------------------------------------------------- #

def test_detects_kokkos_fabs_on_promoted():
    calls = _calls("y = Kokkos::fabs(x);")
    assert ("Kokkos", "fabs", "Kokkos") in calls


def test_detects_std_and_sycl_and_nested_cuda():
    assert ("std", "sqrt", "std") in _calls("std::sqrt(x)")
    assert ("sycl", "exp", "sycl") in _calls("sycl::exp(x)")
    # nested qualifier: root is the leftmost component
    calls = _calls("cuda::std::pow(x, 2)")
    assert ("cuda", "pow", "cuda::std") in calls


def test_ignores_vendored_quad_namespace():
    assert _calls("Kokkos::Experimental::abs(x)") == []
    assert _calls("Kokkos::Experimental::sqrt(x)") == []
    # Pre-refresh spelling: still present in trees the rename sweep excluded.
    assert _calls("quad::ddfun::abs(x)") == []
    assert _calls("quad::ffun::sqrt(x)") == []


def test_vendored_allowlist_does_not_swallow_its_own_root():
    """``Kokkos::Experimental`` is exempt; the enclosing ``Kokkos`` is NOT.

    The allowlist must key on the full qualifier chain.  Keying on the root would
    exempt every ``Kokkos::fn(promoted)`` call — precisely the bridge-needing case
    this scan exists to catch (the 95ce538 header-refresh regression).
    """
    assert ("Kokkos", "fabs", "Kokkos") in _calls("Kokkos::fabs(x)")
    assert ("Kokkos", "sqrt", "Kokkos") in _calls("Kokkos::sqrt(x)")
    # ...while a namespace nested inside the vendored one stays exempt.
    assert _calls("Kokkos::Experimental::detail::abs(x)") == []


def test_ignores_call_without_promoted_arg():
    # literal / non-promoted operands never narrow an extended value
    assert _calls("std::sqrt(2.0)") == []
    assert _calls("Kokkos::fabs(unpromoted_var)") == []


def test_ignores_non_math_functions():
    # printf etc. are not math ops on the extended type
    assert _calls("Kokkos::printf(\"%d\", x)") == []


def test_ignores_class_template_accessor():
    # Ns::Type<...>::member() is not a free-function math call (handled elsewhere)
    assert _calls("ql::Constants<TScale>::template _ieps50<A,B,C>()") == []


def test_dedupes_repeated_calls():
    calls = _calls("Kokkos::fabs(x) + Kokkos::fabs(a)")
    assert calls.count(("Kokkos", "fabs", "Kokkos")) == 1


# --------------------------------------------------------------------------- #
# lint
# --------------------------------------------------------------------------- #

_REGION = "y = Kokkos::fabs(x);"


def test_lint_rejects_missing_bridge():
    shim = ("#pragma once\n#include <dd_math.hpp>\n"
            "namespace quad { namespace ddfun {\n"
            "  DoubleDouble myabs(DoubleDouble v){ return abs(v); }\n} }\n")   # ADL-only, no Kokkos bridge
    msg = r._lint_qualified_bridges(_REGION, shim, _PROMOTED)
    assert msg is not None
    assert "C3 bridge lint" in msg
    assert "Kokkos::fabs" in msg


def test_lint_passes_with_namespace_injection_bridge():
    shim = ("#pragma once\n#include <dd_math.hpp>\n"
            "namespace Kokkos { KOKKOS_INLINE_FUNCTION Kokkos::Experimental::DoubleDouble "
            "fabs(Kokkos::Experimental::DoubleDouble x){ return Kokkos::Experimental::abs(x); } }\n")
    assert r._lint_qualified_bridges(_REGION, shim, _PROMOTED) is None


def test_lint_passes_with_using_declaration_fallback():
    shim = ("#pragma once\n#include <dd_math.hpp>\n"
            "using Kokkos::Experimental::fabs;  // (b) fallback bridge\n")
    assert r._lint_qualified_bridges(_REGION, shim, _PROMOTED) is None


def test_lint_passes_with_using_namespace_fallback():
    region = "y = std::sqrt(x);"
    shim = "#pragma once\nnamespace std { using namespace Kokkos::Experimental; }\n"
    # using namespace std form
    shim2 = "#pragma once\nusing namespace std;\n"
    assert r._lint_qualified_bridges(region, shim2, _PROMOTED) is None


def test_lint_clean_region_passes():
    # no qualified math calls at all
    assert r._lint_qualified_bridges("y = a * x;", "#pragma once\n", _PROMOTED) is None


def test_lint_reports_multiple_missing_bridges():
    region = "y = Kokkos::fabs(x) + std::sqrt(a);"
    shim = "#pragma once\n#include <dd_math.hpp>\n"
    msg = r._lint_qualified_bridges(region, shim, _PROMOTED)
    assert msg is not None
    assert "Kokkos::fabs" in msg and "std::sqrt" in msg
