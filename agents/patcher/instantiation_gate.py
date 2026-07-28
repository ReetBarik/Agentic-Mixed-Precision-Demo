"""Instantiation gate — deterministic classifier for dd/``double`` binding errors.

Motivation (STOP #A dispatch fix, 2026-07-28).  C++ does **not** type-check the
body of a dependent-typed template until it is *instantiated* with concrete type
arguments.  Every "does it compile?" probe in the leaf-promotion arc before this
gate compiled the emitted dd variant tree only as an *uninstantiated* template
(the STOP #A dispatch defect left the promoted tree unreachable, so it was parsed
but never instantiated), which is a **false positive**: a variant body can be
syntactically valid yet type-incorrect once the real box binding
(``TOutput = Kokkos::complex<double>``) is substituted.

The Patcher's build gate (:func:`agents.patcher.gates.run_gate`) already builds
the *vanilla driver*, which instantiates ``ql::BO`` at the real box binding — so
once the dispatch reroute lands on the live ``BO`` the promoted tree IS
instantiated and the binding errors surface.  This module does not add a second
build; it **classifies** the g++ error log of that build into the four known
emission-binding error shapes so a binding failure is tagged distinctly
(``instantiation_binding``) instead of a generic ``build_failed``, and so an
*unknown* error shape trips a hard STOP (STOP #BB) rather than being papered over.

Deterministic, no LLM: every classification is a fixed pattern match on the g++
error text.  The four shapes (taxonomy: ``STOP_A_DISPATCH_FIX_2026-07-28.md`` §6):

* **Shape 1 — exit-boundary narrowing** (``dd_to_caller_complex``): a promoted dd
  value (``ddcomplex``/``ddouble``) flows into an un-narrowed caller-precision
  ``Kokkos::complex<double>`` local / store / return, emitted as a raw
  assignment / construction g++ rejects.
* **Shape 2 — missing interior widen** (``dd_to_double``): a dd value bound to a
  ``double`` / ``const double&`` decl or parameter inside a variant — a closure
  decl or callee parameter the emission left at caller precision.
* **Shape 3 — nested complex** (``nested_complex``): ``Kokkos::complex<...
  ddcomplex>`` — a complex container widened twice (the complex-container
  promotion applied on top of an already-``ddcomplex`` operand).
* **Shape 4 — unclassified shim** (``shim_unclassified``): a synthesized shim the
  emission left as a Rule-R4 ``#error "... requires manual classification"``.

Anything the classifier cannot bucket is :data:`SHAPE_UNKNOWN` — the caller MUST
treat that as STOP #BB (catalog the class, hand back), never a silent fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# The reason tag a binding failure carries (distinct from a generic build_failed).
INSTANTIATION_BINDING = "instantiation_binding"

# The four known emission-binding error shapes + the unknown sentinel.
SHAPE_1_EXIT_NARROW = "shape1_dd_to_caller_complex"     # designed-exit narrowing hole
SHAPE_2_INTERIOR_WIDEN = "shape2_dd_to_double"          # missing interior/callee widen
SHAPE_3_NESTED_COMPLEX = "shape3_nested_complex"        # complex<ddcomplex> double-widen
SHAPE_4_SHIM = "shape4_shim_unclassified"              # #error manual classification
SHAPE_UNKNOWN = "unknown"                               # STOP #BB

KNOWN_SHAPES = frozenset({
    SHAPE_1_EXIT_NARROW, SHAPE_2_INTERIOR_WIDEN,
    SHAPE_3_NESTED_COMPLEX, SHAPE_4_SHIM,
})

# g++ prints diagnostics with U+2018/U+2019 curly quotes by default; normalise both
# those and the ASCII fallback to a plain apostrophe so the patterns match either.
_CURLY = {"‘": "'", "’": "'", "“": '"', "”": '"'}

# The dd type spellings the chain integrator emits (agents/chain_integrator/agent.py).
_DD_SCALAR = "quad::ddfun::ddouble"
_DD_COMPLEX = "quad::ddfun::ddcomplex"
# Any Kokkos/std complex container wrapping the dd complex — the double-widen symptom.
_NESTED_RE = re.compile(r"(?:Kokkos|std)::complex<\s*quad::ddfun::ddcomplex\s*>")


def _normalise(text: str) -> str:
    for k, v in _CURLY.items():
        text = text.replace(k, v)
    return text


def classify_error(message: str) -> str:
    """Bucket a single g++ ``error:`` message into a shape (or :data:`SHAPE_UNKNOWN`).

    Order matters: Shape 3 (nested complex) is tested FIRST because a doubly-widened
    ``Kokkos::complex<quad::ddfun::ddcomplex>`` operand also matches the raw-assignment
    / static-assert signatures of the other shapes; the nested container is the root
    cause and the more specific bucket.  Shape 4 (the ``#error``) is unambiguous.
    Shapes 1 and 2 are then distinguished by the *target* precision — a dd value
    landing in a caller ``complex<double>`` (Shape 1) vs a bare ``double`` (Shape 2).
    """
    m = _normalise(message)

    # --- Shape 3: nested complex<ddcomplex> (double-widened container) --------
    if _NESTED_RE.search(m):
        return SHAPE_3_NESTED_COMPLEX
    # The Kokkos::complex static_assert fires only when instantiated on a non-FP type,
    # i.e. on ddcomplex — the same double-widen root cause seen from the library side.
    if ("Kokkos::complex can only be instantiated for a cv-unqualified floating point"
            in m):
        return SHAPE_3_NESTED_COMPLEX

    # --- Shape 4: the Rule-R4 unclassified shim ------------------------------
    if "requires manual classification" in m:
        return SHAPE_4_SHIM

    # --- Shape 1: dd value -> caller-precision complex<double> ---------------
    # construction: Kokkos::complex<double>::complex(quad::ddfun::dd{ouble,complex})
    if re.search(r"complex<double>::complex\(\s*quad::ddfun::dd(?:ouble|complex)\s*\)", m):
        return SHAPE_1_EXIT_NARROW
    # decl/return conversion: ddcomplex/ddouble -> (const) Kokkos::complex<double>
    if (("conversion from '" + _DD_COMPLEX) in m or
            ("conversion from '" + _DD_SCALAR) in m) and "Kokkos::complex<double>" in m:
        return SHAPE_1_EXIT_NARROW
    # assignment either direction between ddcomplex/ddouble and complex<double>
    if "operator=" in m and "Kokkos::complex<double>" in m and (
            _DD_COMPLEX in m or _DD_SCALAR in m):
        return SHAPE_1_EXIT_NARROW
    # could-not-convert complex<double> -> ddcomplex (a caller value into a dd sink at
    # the exit line — the mirror of the narrowing hole, still an exit-boundary defect)
    if "could not convert" in m and "Kokkos::complex<double>" in m and _DD_COMPLEX in m:
        return SHAPE_1_EXIT_NARROW

    # --- Shape 2: dd value -> bare double (missing interior/callee widen) -----
    if re.search(r"invalid cast from type '" + re.escape(_DD_SCALAR) +
                 r"' to type 'double'", m):
        return SHAPE_2_INTERIOR_WIDEN
    if re.search(r"cannot convert '" + re.escape(_DD_SCALAR) +
                 r"' to 'const double'", m):
        return SHAPE_2_INTERIOR_WIDEN
    if re.search(r"reference of type 'const double&'.*'" + re.escape(_DD_SCALAR) + r"'", m):
        return SHAPE_2_INTERIOR_WIDEN
    if ("cannot convert '" + _DD_SCALAR) in m and "'double'" in m:
        return SHAPE_2_INTERIOR_WIDEN

    return SHAPE_UNKNOWN


@dataclass
class InstantiationReport:
    """Per-shape classification of a build log's g++ errors.

    ``by_shape`` maps each shape key to the list of raw error messages in it;
    ``unknown`` collects any message the classifier could not bucket (STOP #BB when
    non-empty).  ``ok`` is True only when there were zero errors at all.
    """

    total: int = 0
    by_shape: dict[str, list[str]] = field(default_factory=dict)
    unknown: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.total == 0

    @property
    def has_unknown(self) -> bool:
        return bool(self.unknown)

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in sorted(self.by_shape.items())}

    def summary(self) -> str:
        if self.ok:
            return "instantiation gate: no errors"
        parts = [f"{k}={len(v)}" for k, v in sorted(self.by_shape.items())]
        if self.unknown:
            parts.append(f"{SHAPE_UNKNOWN}={len(self.unknown)}")
        return (f"instantiation_binding: {self.total} error(s) — "
                + ", ".join(parts))


_ERROR_LINE_RE = re.compile(r"\berror:\s*(.*)")


def classify_build_log(log_text: str) -> InstantiationReport:
    """Classify every ``error:`` line in a g++ build log into shapes.

    Deterministic line scan: each ``... error: <message>`` line is bucketed by
    :func:`classify_error`.  The ``#error`` directive (Shape 4) is emitted by the
    preprocessor as ``error: #error "..."`` / ``error: "..."`` — both carry the
    ``requires manual classification`` marker so they classify correctly.
    """
    report = InstantiationReport()
    for line in log_text.splitlines():
        m = _ERROR_LINE_RE.search(line)
        if not m:
            continue
        message = m.group(1).strip()
        if not message:
            continue
        report.total += 1
        shape = classify_error(message)
        if shape == SHAPE_UNKNOWN:
            report.unknown.append(message)
        else:
            report.by_shape.setdefault(shape, []).append(message)
    return report


def classify_build_log_file(path: str | Path) -> InstantiationReport:
    """Classify a build log read from ``path`` (empty report if unreadable)."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return InstantiationReport()
    return classify_build_log(text)
