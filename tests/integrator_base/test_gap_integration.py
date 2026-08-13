"""Engine-level offline tests for Gap A (namespace-qualified bridge) and Gap B
(source-derivable constants) through agents.integrator_base.regional.

No LLM and no compiler: a canned ``llm_fn`` captures the user turn and returns a
shim we control, so we can assert (1) the deterministic hints reach the model and
(2) the lints flip a bad shim to a retryable ``llm_failed``.  The synthetic cases
(``std::sqrt(promoted)`` / ``constexpr double MY_TINY = 1e-40``) exercise the
generic path with zero qcdloop symbols.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from agents.dd_integrator import agent as dd
from agents.integrator_base import regional


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args],
                          capture_output=True, text=True, check=True)


def _init_repo(root: Path, files: dict[str, str]) -> str:
    root.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t.t")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    return _git(root, "rev-parse", "HEAD").stdout.strip()


class _CapturingLLM:
    """Returns a fixed shim; records the (system, user) turns it was given."""

    def __init__(self, shim: str):
        self.shim = shim
        self.calls: list[tuple[str, str, int]] = []

    def __call__(self, system, user, attempt):
        self.calls.append((system, user, attempt))
        return self.shim


_CLEAN_SHIM = (
    "#pragma once\n// SOURCE_HASH: PENDING\n"
    "#include <dd_math.hpp>\n#include <dd_complex.hpp>\n"
)


# --------------------------------------------------------------------------- #
# Gap A — synthetic std::sqrt(promoted)
# --------------------------------------------------------------------------- #

_GAPA_FILE = (
    "#pragma once\n"
    "double g(double x) {\n"
    "    double r = std::sqrt(x);\n"   # line 3 — qualified math call on a promoted read
    "    return r;\n"
    "}\n"
)


def _run_gapa(tmp_path, shim):
    root = tmp_path / "cand"
    sha = _init_repo(root, {"kernel.h": _GAPA_FILE})
    llm = _CapturingLLM(shim)
    res = dd.integrate_region(
        file="kernel.h", line_start=3, line_end=3, variables=["x"],
        working_tree=sha, out_dir=tmp_path / "shims", repo_path=str(root), llm_fn=llm,
    )
    return res, llm


def test_gapa_user_message_lists_qualified_call(tmp_path):
    _, llm = _run_gapa(tmp_path, _CLEAN_SHIM + "namespace std { }\n")
    user = llm.calls[0][1]
    assert "Namespace-qualified calls needing a bridge" in user
    assert "std::sqrt" in user


def test_gapa_missing_bridge_rejected_as_misgen(tmp_path):
    # shim provides only ADL overloads in Kokkos::Experimental -> no std:: bridge -> reject
    bad = _CLEAN_SHIM + "namespace quad { namespace ddfun { } }\n"
    res, _ = _run_gapa(tmp_path, bad)
    assert not res.ok and res.status == "llm_failed"
    assert "C3 bridge lint" in (res.error or "")
    assert "std::sqrt" in (res.error or "")
    # a rejected shim is not persisted into the tree
    assert not list((tmp_path / "cand").glob("kernel_dd_*.h"))


def test_gapa_with_bridge_accepted(tmp_path):
    good = _CLEAN_SHIM + (
        "namespace std {\n"
        "  Kokkos::Experimental::DoubleDouble sqrt(Kokkos::Experimental::DoubleDouble x){ return Kokkos::Experimental::sqrt(x); }\n"
        "}\n")
    res, _ = _run_gapa(tmp_path, good)
    assert res.ok, res.error


# --------------------------------------------------------------------------- #
# Gap B — synthetic constexpr double MY_TINY = 1e-40
# --------------------------------------------------------------------------- #

_GAPB_KERNEL = (
    "#pragma once\n"
    '#include "consts.h"\n'
    "double g(double a) {\n"
    "    double c = MY_TINY * a;\n"     # line 4 — reads the derivable constant
    "    return c;\n"
    "}\n"
)
_GAPB_CONSTS = "#pragma once\nconstexpr double MY_TINY = 1e-40;\n"


def test_gapb_user_message_carries_derived_constant(tmp_path):
    root = tmp_path / "cand"
    sha = _init_repo(root, {"kernel.h": _GAPB_KERNEL, "consts.h": _GAPB_CONSTS})
    llm = _CapturingLLM(_CLEAN_SHIM)
    dd.integrate_region(
        file="kernel.h", line_start=4, line_end=4, variables=["a"],
        working_tree=sha, out_dir=tmp_path / "shims", repo_path=str(root), llm_fn=llm,
    )
    user = llm.calls[0][1]
    # the dynamic hint section (not the static constant_note mention)
    assert "## Source-derivable constants (Rule R3, step 3)" in user
    assert "MY_TINY" in user
    assert "1e-40" in user
    # the derived value is a real DoubleDouble::from_bits pair with a zero low word (source literal)
    assert "Kokkos::Experimental::DoubleDouble::from_bits(0x" in user
    assert "0x0000000000000000ULL" in user


def test_gapb_no_hint_when_constant_not_derivable(tmp_path):
    # a runtime value (no visible literal definition) must not fabricate a hint
    kernel = ("#pragma once\ndouble g(double a){\n"
              "  double c = runtime_lookup() * a;\n  return c;\n}\n")
    root = tmp_path / "cand"
    sha = _init_repo(root, {"kernel.h": kernel})
    llm = _CapturingLLM(_CLEAN_SHIM)
    dd.integrate_region(
        file="kernel.h", line_start=3, line_end=3, variables=["a"],
        working_tree=sha, out_dir=tmp_path / "shims", repo_path=str(root), llm_fn=llm,
    )
    assert "## Source-derivable constants (Rule R3, step 3)" not in llm.calls[0][1]


# --------------------------------------------------------------------------- #
# derive_region_constants helper (directly)
# --------------------------------------------------------------------------- #

_IEPS50_REGION = ("const TOutput k = ql::Constants<TScale>::template "
                  "_ieps50<TOutput, TMass, TScale>();")
_IEPS50_SOURCES = [
    _IEPS50_REGION,
    "template<class A,class B,class C>\n"
    "static TOutput _ieps50() { return TOutput{Constants<TScale>::_zero(), "
    "TScale(1e-50)}; }\n",
    "static constexpr T _zero() { return T(0.0); }\n",
]


# --------------------------------------------------------------------------- #
# Wave-3 dedup — two regions merge into ONE canonical per-family shim (engine)
# --------------------------------------------------------------------------- #

_MULTI_REGION_FILE = (
    "#pragma once\n"
    "template<class TMass> TMass fa(TMass a) {\n"
    "    return a * ql::Constants<TMass>::_one();\n"   # line 3 — region A needs _one
    "}\n"
    "template<class TMass> TMass fb(TMass b) {\n"
    "    return b + ql::Constants<TMass>::_two();\n"   # line 6 — region B needs _two
    "}\n"
)

_SHIM_ONE = (
    "#pragma once\n// SOURCE_HASH: PENDING\n#include <dd_math.hpp>\n"
    "namespace ql {\n"
    "template <class T> struct Constants;\n"
    "template <>\nstruct Constants< ::Kokkos::Experimental::DoubleDouble > {\n"
    "    static inline ::Kokkos::Experimental::DoubleDouble _one() { return ::Kokkos::Experimental::DoubleDouble(1.0); }\n"
    "};\n} // namespace ql\n"
)
_SHIM_TWO = (
    "#pragma once\n// SOURCE_HASH: PENDING\n#include <dd_math.hpp>\n"
    "namespace ql {\n"
    "template <class T> struct Constants;\n"
    "template <>\nstruct Constants<Kokkos::Experimental::DoubleDouble> {\n"   # no leading :: — same type
    "    static inline ::Kokkos::Experimental::DoubleDouble _two() { return ::Kokkos::Experimental::DoubleDouble(2.0); }\n"
    "};\n} // namespace ql\n"
)


def test_two_regions_merge_into_one_canonical_dd_shim(tmp_path):
    root = tmp_path / "cand"
    sha = _init_repo(root, {"kernel.h": _MULTI_REGION_FILE})
    common = dict(file="kernel.h", working_tree=sha, variables=[],
                  out_dir=tmp_path / "shims", repo_path=str(root))

    # Region A lands first (first into the TU → creates the canonical shim).
    ra = dd.integrate_region(line_start=3, line_end=3, llm_fn=_CapturingLLM(_SHIM_ONE),
                             **common)
    assert ra.ok, ra.error
    # Region B lands against the tree carrying A's canonical shim → must MERGE, not
    # emit a second Constants<DoubleDouble> (the WAVE3 collision).
    rb = dd.integrate_region(line_start=6, line_end=6, llm_fn=_CapturingLLM(_SHIM_TWO),
                             **common)
    assert rb.ok, rb.error

    canonical = (root / "ql_shim_dd.h").read_text()
    # exactly ONE specialization, both members present (::-agnostic count)
    assert re.sub(r"\s+", "", canonical).replace("::", "").count(
        "structConstants<Kokkos::Experimental::DoubleDouble>".replace("::", "")) == 1
    assert canonical.count("_one()") == 1
    assert canonical.count("_two()") == 1
    # no per-region shim leaked into the tree; both boundaries include the canonical
    assert not list(root.glob("kernel_dd_*.h"))
    assert '#include "ql_shim_dd.h"' in (ra.boundary_patch or "")
    assert '#include "ql_shim_dd.h"' in (rb.boundary_patch or "")


def test_derive_region_constants_composite_complex_legacy_literals():
    # No complex_type given → legacy behavior: surface the bare literal(s) only.
    got = regional.derive_region_constants(_IEPS50_REGION, _IEPS50_SOURCES, "dd")
    entry = next(c for c in got if c["name"] == "_ieps50")
    lit_exprs = " ".join(l["expr"] for l in entry["literals"])
    assert "358dee7a4ad4b81f" in lit_exprs   # correct 1e-50 hi bits


def test_derive_region_constants_assembles_full_complex_value():
    # Wave 2: with the complex type, the imaginary iε regulator is derived WHOLE —
    # a ready-made DoubleDoubleComplex(re, im) the model uses verbatim (no collapse to real).
    got = regional.derive_region_constants(
        _IEPS50_REGION, _IEPS50_SOURCES, "dd", "Kokkos::Experimental::DoubleDoubleComplex")
    entry = next(c for c in got if c["name"] == "_ieps50")
    assert entry["how"] == "complex"
    assert entry["literals"] == []          # not the fallback literals hint
    assert entry["expr"] == (
        "Kokkos::Experimental::DoubleDoubleComplex("
        "Kokkos::Experimental::DoubleDouble::from_bits(0x0000000000000000ULL, 0x0000000000000000ULL), "
        "Kokkos::Experimental::DoubleDouble::from_bits(0x358dee7a4ad4b81fULL, 0x0000000000000000ULL))"
    )


def test_dd_integrate_region_hint_carries_full_complex_value(tmp_path):
    # End-to-end through dd.integrate_region: the user turn hands the model the
    # complete complex value and the "return the FULL complex value" instruction.
    kernel = ("#pragma once\n"
              "template<class TOutput, class TMass, class TScale>\n"
              "TOutput f() {\n"
              "  const TOutput k = ql::Constants<TScale>::template "
              "_ieps50<TOutput, TMass, TScale>();\n"
              "  return k;\n}\n")
    defs = ("#pragma once\n"
            "template<class A,class B,class C>\n"
            "static TOutput _ieps50() { return TOutput{Constants<TScale>::_zero(), "
            "TScale(1e-50)}; }\n"
            "static constexpr T _zero() { return T(0.0); }\n")
    root = tmp_path / "cand"
    sha = _init_repo(root, {"kernel.h": kernel, "consts.h": defs})
    llm = _CapturingLLM(_CLEAN_SHIM)
    dd.integrate_region(
        file="kernel.h", line_start=4, line_end=4, variables=[],
        working_tree=sha, out_dir=tmp_path / "shims", repo_path=str(root), llm_fn=llm,
    )
    user = llm.calls[0][1]
    assert "Kokkos::Experimental::DoubleDoubleComplex(" in user
    assert "0x358dee7a4ad4b81fULL" in user           # imaginary limb = 1e-50
    assert "COMPLEX container" in user                # the preserve-imaginary note
    assert "do NOT collapse it to a real scalar" in user
