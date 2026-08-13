"""Source-derivable extended-precision constants (Gap B) — the deterministic
companion to the regional integrators' Rule R3 constant cascade.

Motivation
----------
A regional ff/dd shim must materialize any named constant the promoted region
reads as a *two-limb* ``DoubleDouble::from_bits(0x<hi>, 0x<lo>)`` / ``FloatFloat::from_bits(0x<hi>, 0x<lo>)``
pair — a decimal literal truncates the low word (Rule R3).  The original R3 gave
the model only two ways to get those bits: a vendored ``dd_*()`` / ``ff_*()``
factory, or a hex pair it already knew.  Everything else fell through to the
Rule R4 ``#error`` escape hatch — even when the constant was *trivially
derivable* from its own source definition.

The 2026-07-18 rerun made this concrete: ``_ieps50`` (a physics infinitesimal
whose source definition is just the double literal ``1e-50``) tripped R4 on every
box region, because the model had no vendored factory and no memorized hex pair —
and when it *guessed*, it guessed wrong (a spurious low word, and even a wrong hi
word).  But the constant's faithful extended value is not a mystery: a source
``double`` literal carries **exactly** double precision by construction, so its
honest double-double promotion is ``DoubleDouble::from_bits(<bits of the double>, 0x0)`` — there
is no hidden low word to recover.  This module computes those bits deterministically.

What it does
------------
Two capabilities, both framework-agnostic (no app symbols anywhere):

* :func:`derive_literal` — turn a numeric literal (``1e-50``, ``0.125``, ``2.0``,
  optionally wrapped in a cast like ``TScale(1e-50)`` / ``static_cast<T>(...)``)
  into the exact ``(hi, lo)`` bit pair for dd / ff.  A source literal is a
  ``double`` (or narrower) value, so the dd low word is always ``0``; the ff
  split captures the double value across two ``float`` limbs.

* :data:`KNOWN_CONSTANTS` — a small catalog of *mathematical* constants (π, 2π,
  π/2, e, √2, ln 2, ln 10, Euler γ) whose dd/ff ``(hi, lo)`` pairs are computed
  at import from high-precision decimal strings via the standard Bailey split.
  Used when a source RHS is a closed form over one of these (``2.0*M_PI`` →
  the 2π entry) rather than a bare literal — the one case where the "just take
  the double bits" rule would lose real precision.

:func:`resolve_constant_rhs` walks scan-reachable source to the *definition* of a
named constant and returns its right-hand side text; :func:`derive_from_rhs`
classifies that RHS through the cascade (literal → catalog closed-form → give up).
The regional engine uses these to hand the model ready-made ``DoubleDouble::from_bits(...)`` /
``FloatFloat::from_bits(...)`` values so a derivable constant never reaches R4.

Nothing here is qcdloop-specific: the catalog holds only standard mathematical
values, and the source walk reads generic C++ ``constexpr`` / ``const`` /
``#define`` / literal-returning-accessor declarations.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from decimal import Decimal, getcontext
from fractions import Fraction

# High-precision decimal strings for the catalog (≥ 40 significant digits — more
# than a double-double needs).  Computed to (hi, lo) at import via the Bailey
# split, which reproduces the vendored dd_*/ff_* pairs bit-for-bit (see tests).
getcontext().prec = 80

_CONST_DECIMALS: dict[str, Decimal] = {
    # canonical name -> high-precision value
    "pi":          Decimal("3.14159265358979323846264338327950288419716939937510582097494459"),
    "two_pi":      Decimal("6.28318530717958647692528676655900576839433879875021164194988918"),
    "half_pi":     Decimal("1.57079632679489661923132169163975144209858469968755291048747230"),
    "e":           Decimal("2.71828182845904523536028747135266249775724709369995957496696763"),
    "sqrt2":       Decimal("1.41421356237309504880168872420969807856967187537694807317667974"),
    "ln2":         Decimal("0.69314718055994530941723212145817656807550013436025525412068001"),
    "ln10":        Decimal("2.30258509299404568401799145468436420760110148862877297603332790"),
    "euler_gamma": Decimal("0.57721566490153286060651209008240243104215933593992359880576723"),
}

# The π family (π², π/3, π/6, π²/3, π²/6, π²/12).  These are DERIVED from the
# canonical ``pi`` entry above at prec=80 rather than transcribed as decimal
# literals: composing from the 63-digit ``pi`` string is exact to ~63 significant
# digits (far beyond the ~32 a double-double carries) and removes the hand-copy
# error that a literal string would risk (STOP #C).  The stored value is still the
# *true* constant's Bailey split — never the result of dd arithmetic — exactly like
# ``two_pi``/``half_pi``.  Upstream a kernel may define these compositionally
# (``_pi()*_pi()``, ``_pi()/TScale(6)`` …); :func:`derive_from_catalog` recognizes
# those RHS shapes and maps them onto these entries.
_pi_hp = _CONST_DECIMALS["pi"]
_CONST_DECIMALS.update({
    "pi_squared":         _pi_hp * _pi_hp,
    "pi_over_3":          _pi_hp / Decimal(3),
    "pi_over_6":          _pi_hp / Decimal(6),
    "pi_squared_over_3":  _pi_hp * _pi_hp / Decimal(3),
    "pi_squared_over_6":  _pi_hp * _pi_hp / Decimal(6),
    "pi_squared_over_12": _pi_hp * _pi_hp / Decimal(12),
})

# Aliases the source RHS may spell a catalog constant with.  Keys are matched
# case-sensitively against a bare identifier / macro; values index _CONST_DECIMALS.
# Framework-agnostic: only standard math spellings (C macros, std::numbers, and
# the plain names).  App-specific constant names are resolved by source walk, not
# by this table.
_CATALOG_ALIASES: dict[str, str] = {
    "M_PI": "pi", "pi": "pi", "PI": "pi", "pi_v": "pi",
    "M_E": "e", "e": "e", "E": "e", "e_v": "e",
    "M_SQRT2": "sqrt2", "sqrt2": "sqrt2", "SQRT2": "sqrt2", "sqrt2_v": "sqrt2",
    "M_LN2": "ln2", "ln2": "ln2", "LN2": "ln2", "ln2_v": "ln2", "log2e": "ln2",
    "M_LN10": "ln10", "ln10": "ln10", "LN10": "ln10", "ln10_v": "ln10",
    "egamma": "euler_gamma", "euler_gamma": "euler_gamma", "gamma": "euler_gamma",
    "egamma_v": "euler_gamma",
}

# Accessor-name aliases for the π family, registered so a *composition* RHS
# (``_pi() * _pio6<...>()``) can resolve each accessor to its canonical catalog
# name without a source walk.  These are standard mathematical π-family accessor
# spellings (the same convention as ``M_PI`` / ``std::numbers::pi_v`` above) — they
# name a mathematical value, not any app symbol.  A caller may pass its own map to
# :func:`derive_from_rhs` to override / extend this default (kernel-specific
# spellings stay caller-supplied — the catalog itself is library-agnostic).
_PI_FAMILY_ACCESSOR_ALIASES: dict[str, str] = {
    "_pi": "pi",
    "_pi2": "pi_squared",
    "_pio3": "pi_over_3",
    "_pio6": "pi_over_6",
    "_pi2o3": "pi_squared_over_3",
    "_pi2o6": "pi_squared_over_6",
    "_pi2o12": "pi_squared_over_12",
}
# Public: the accessor-alias default the integrator engine registers.
PI_FAMILY_ACCESSOR_ALIASES = dict(_PI_FAMILY_ACCESSOR_ALIASES)

# Symbolic form of each catalog entry as (π-power, rational coefficient) so a
# composition (``A() * B()`` / ``A() / k``) can be reduced algebraically and matched
# back to a catalog name.  Only the π family (plus the plain ``pi`` bases) carries a
# symbolic form; other catalog constants (e, √2, …) are absent → composition over
# them returns ``None`` (we never invent a value).
_CONST_SYMBOLIC: dict[str, tuple[int, Fraction]] = {
    # canonical name -> (power of pi, rational coefficient)
    "pi":                 (1, Fraction(1)),
    "two_pi":             (1, Fraction(2)),
    "half_pi":            (1, Fraction(1, 2)),
    "pi_squared":         (2, Fraction(1)),
    "pi_over_3":          (1, Fraction(1, 3)),
    "pi_over_6":          (1, Fraction(1, 6)),
    "pi_squared_over_3":  (2, Fraction(1, 3)),
    "pi_squared_over_6":  (2, Fraction(1, 6)),
    "pi_squared_over_12": (2, Fraction(1, 12)),
}
# Reverse map: a reduced (power, coeff) -> catalog name, so a composed symbolic
# form lands on a catalog entry (or None → not derivable, never invented).
_SYMBOLIC_TO_NAME: dict[tuple[int, Fraction], str] = {
    sym: name for name, sym in _CONST_SYMBOLIC.items()
}


# --------------------------------------------------------------------------- #
# IEEE-754 split helpers
# --------------------------------------------------------------------------- #

def _f64_bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _f32(x: float) -> float:
    """Round a Python float to the nearest IEEE-754 single."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def dd_bits_from_decimal(value: Decimal) -> tuple[int, int]:
    """Bailey double-double split of an exact ``Decimal`` -> (hi_bits, lo_bits).

    ``hi`` is the nearest double to ``value``; ``lo`` is the nearest double to the
    residual ``value - hi``.  Reproduces the vendored ``dd_*()`` pairs bit-for-bit.
    """
    hi = float(value)
    lo = float(value - Decimal(hi))
    return _f64_bits(hi), _f64_bits(lo)


def ff_bits_from_decimal(value: Decimal) -> tuple[int, int]:
    """Bailey float-float split of an exact ``Decimal`` -> (hi_bits, lo_bits)."""
    hi = _f32(float(value))
    lo = _f32(float(value - Decimal(hi)))
    return _f32_bits(hi), _f32_bits(lo)


def dd_bits_from_double(x: float) -> tuple[int, int]:
    """(hi, lo) bits for a value that is *already a double literal* — lo is 0.

    A source ``double`` literal has no precision below the double it denotes, so
    its faithful double-double promotion is ``DoubleDouble::from_bits(bits(x), 0x0)``.  This is
    the point of Gap B: do NOT invent a low word for a source literal.
    """
    return _f64_bits(x), 0


def ff_bits_from_double(x: float) -> tuple[int, int]:
    """(hi, lo) bits promoting a source ``double`` literal to float-float.

    Unlike the dd case, a double value generally does NOT fit one ``float``, so
    the double is split across the two ``float`` limbs to preserve its value.
    """
    hi = _f32(x)
    lo = _f32(x - hi)
    return _f32_bits(hi), _f32_bits(lo)


# Catalog: canonical name -> {"dd": (hi, lo), "ff": (hi, lo)} bit pairs.
KNOWN_CONSTANTS: dict[str, dict[str, tuple[int, int]]] = {
    name: {"dd": dd_bits_from_decimal(dec), "ff": ff_bits_from_decimal(dec)}
    for name, dec in _CONST_DECIMALS.items()
}


# --------------------------------------------------------------------------- #
# derivation result
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Derivation:
    """A derived extended-precision constant value ready to paste into a shim.

    ``expr`` is the ``DoubleDouble::from_bits(...)`` / ``FloatFloat::from_bits(...)`` call; ``how`` records which
    cascade step produced it (for the rule-justification comment and telemetry).
    """

    name: str            # source identifier the constant is read by
    scalar: str          # "dd" | "ff"
    expr: str            # e.g. "Kokkos::Experimental::DoubleDouble::from_bits(0x358dee7a4ad4b81fULL, 0x0ULL)"
    how: str             # "literal" | "catalog:pi" | ...
    rhs: str = ""        # the source RHS the derivation came from (provenance)


def _make_call(scalar: str, hi: int, lo: int) -> str:
    if scalar == "dd":
        return f"Kokkos::Experimental::DoubleDouble::from_bits(0x{hi:016x}ULL, 0x{lo:016x}ULL)"
    if scalar == "ff":
        return f"Kokkos::Experimental::FloatFloat::from_bits(0x{hi:08x}U, 0x{lo:08x}U)"
    raise ValueError(f"unknown scalar {scalar!r} (expected 'dd' or 'ff')")


# --------------------------------------------------------------------------- #
# literal parsing / derivation
# --------------------------------------------------------------------------- #

# A C++ floating (or integer-used-as-floating) literal, optional suffix.  Kept
# deliberately narrow: decimal only (a hex float like 0x1p-4 -> None -> R4/model).
_FLOAT_LITERAL_RE = re.compile(
    r"""^[+-]?(
          (?:\d+\.\d*|\.\d+|\d+)   # 12.  .5  12  12.5
          (?:[eE][+-]?\d+)?        # optional exponent
        )[fFlL]*$""",
    re.VERBOSE,
)

# ``Name( <inner> )`` functional cast, or ``static_cast< T >( <inner> )``.
_CAST_RE = re.compile(r"^\s*(?:static_cast\s*<[^>]*>|[A-Za-z_]\w*)\s*\(\s*(.*?)\s*\)\s*$",
                      re.DOTALL)


_CAST_PREFIX_RE = re.compile(r"^\s*(?:static_cast\s*<[^>]*>|[A-Za-z_]\w*)\s*\(")


def _matching_paren(text: str, open_idx: int) -> int | None:
    """Index of the ``)`` matching the ``(`` at ``open_idx``, or ``None``."""
    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def _strip_casts(text: str) -> str:
    """Peel functional / static_cast wrappers: ``TScale(1e-50)`` -> ``1e-50``.

    Only peels when the wrapper's ``(`` matches a ``)`` that is the LAST non-space
    character — i.e. the cast genuinely brackets the whole expression.  A product
    like ``_pi() * _pio6()`` (whose first ``(`` closes mid-expression) and a
    genuine multi-arg call (``foo(a, b)``) or braced init are left intact.
    """
    prev = None
    cur = text.strip()
    while cur != prev:
        prev = cur
        m = _CAST_PREFIX_RE.match(cur)
        if not m:
            break
        open_idx = m.end() - 1               # position of the wrapper's '('
        close_idx = _matching_paren(cur, open_idx)
        if close_idx is None or close_idx != len(cur.rstrip()) - 1:
            break                            # '(' does not bracket the whole expr
        inner = cur[open_idx + 1:close_idx]
        # Reject if the inner text has a top-level comma (a real call, not a cast).
        if _has_top_level_comma(inner):
            break
        cur = inner.strip()
    return cur


def _has_top_level_comma(text: str) -> bool:
    depth = 0
    for ch in text:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            return True
    return False


def parse_float_literal(text: str) -> float | None:
    """Parse a numeric literal (optionally cast-wrapped) to a Python float."""
    inner = _strip_casts(text)
    m = _FLOAT_LITERAL_RE.match(inner)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def derive_literal(name: str, rhs: str, scalar: str) -> Derivation | None:
    """Derive a constant whose RHS is a numeric literal (Gap B cascade step 3a)."""
    value = parse_float_literal(rhs)
    if value is None:
        return None
    hi, lo = (dd_bits_from_double(value) if scalar == "dd"
              else ff_bits_from_double(value))
    return Derivation(name=name, scalar=scalar, expr=_make_call(scalar, hi, lo),
                      how="literal", rhs=rhs.strip())


# A closed form ``[k *] <catalog-name> [* k]`` where k is an exact small factor
# (a power of two keeps the split exact; other integers are accepted because the
# catalog value already carries full precision and the multiply stays in dd/ff).
_CATALOG_TOKEN_RE = re.compile(r"[A-Za-z_]\w*(?:_v)?")


def _lookup_catalog_alias(token: str,
                          alias_map: dict[str, str] | None = None) -> str | None:
    """Resolve ``token`` to a canonical catalog name.

    Consults the built-in math-spelling table first, then the caller-supplied
    ``alias_map`` (kernel-specific accessor spellings — library-agnostic values,
    caller-supplied names).  ``std::numbers::pi_v`` style qualifiers fall back to
    the last path component.
    """
    if token in _CATALOG_ALIASES:
        return _CATALOG_ALIASES[token]
    if alias_map and token in alias_map:
        return alias_map[token]
    # tolerate ``std::numbers::pi_v`` style: last path component
    tail = token.rsplit("::", 1)[-1]
    if tail in _CATALOG_ALIASES:
        return _CATALOG_ALIASES[tail]
    if alias_map and tail in alias_map:
        return alias_map[tail]
    return None


# An accessor call token, template args tolerated: ``_pio6<TOutput,TMass,TScale>()``
# or ``Constants<T>::_pi()`` -> the leading identifier (``_pio6`` / ``_pi``).
_ACCESSOR_NAME_RE = re.compile(
    r"^\s*(?:[A-Za-z_]\w*\s*(?:<[^;{}]*>)?\s*::\s*)*"   # optional qualifier(s)
    r"(?:template\s+)?([A-Za-z_]\w*)\s*(?:<[^;{}]*>)?\s*\(\s*\)\s*$")


def _resolve_symbolic(token: str,
                      alias_map: dict[str, str] | None) -> tuple[int, Fraction] | None:
    """Reduce one operand (an accessor call or bare alias) to (π-power, coeff).

    Handles a catalog alias (``M_PI`` -> pi -> (1, 1)) and an accessor call with
    optional template args (``_pio6<...>()`` -> pi_over_6 -> (1, 1/6)).  Returns
    ``None`` when the operand does not name a symbolic catalog constant.
    """
    tok = token.strip()
    m = _ACCESSOR_NAME_RE.match(tok)
    if m is not None:
        tok = m.group(1)
    canon = _lookup_catalog_alias(tok, alias_map)
    if canon is None:
        return None
    return _CONST_SYMBOLIC.get(canon)


def _catalog_from_symbolic(name: str, sym: tuple[int, Fraction], scalar: str,
                           rhs: str) -> Derivation | None:
    """Map a reduced (power, coeff) onto a catalog entry, or ``None``."""
    canon = _SYMBOLIC_TO_NAME.get(sym)
    if canon is None or canon not in KNOWN_CONSTANTS:
        return None
    hi, lo = KNOWN_CONSTANTS[canon][scalar]
    return Derivation(name=name, scalar=scalar, expr=_make_call(scalar, hi, lo),
                      how=f"catalog:{canon}", rhs=rhs.strip())


def _derive_composition(name: str, inner: str, scalar: str,
                        alias_map: dict[str, str] | None) -> Derivation | None:
    """Recognize an algebraic composition of catalog accessors / aliases.

    Supported shapes (each operand an accessor call or bare alias):
      * ``A() * B()``           -> symbolic product     (``_pi()*_pio6()`` -> π²/6)
      * ``A() / TScale(k)``     -> divide by integer k   (``_pi()/TScale(6)`` -> π/6)
      * ``A() / k``             -> divide by integer k   (``_pi2()/TScale(12)`` -> π²/12)

    Each operand is reduced to a (π-power, rational-coeff) symbolic form; the
    composition is reduced algebraically and matched back to a catalog entry.  A
    composition that does not land on a catalog name returns ``None`` — the value
    is never invented.
    """
    # A() / <divisor>  — divisor is a small integer, optionally cast-wrapped.
    parts = _split_top_level_binop(inner, "/")
    if parts is not None:
        lhs, rhs_div = parts
        sym = _resolve_symbolic(lhs, alias_map)
        if sym is None:
            return None
        k = _parse_int_divisor(rhs_div)
        if k is None or k == 0:
            return None
        power, coeff = sym
        return _catalog_from_symbolic(name, (power, coeff / k), scalar, inner)

    # A() * B()  — symbolic product (powers add, coefficients multiply).
    parts = _split_top_level_binop(inner, "*")
    if parts is not None:
        lhs, rhs_mul = parts
        sym_l = _resolve_symbolic(lhs, alias_map)
        sym_r = _resolve_symbolic(rhs_mul, alias_map)
        if sym_l is None or sym_r is None:
            return None
        power = sym_l[0] + sym_r[0]
        coeff = sym_l[1] * sym_r[1]
        return _catalog_from_symbolic(name, (power, coeff), scalar, inner)
    return None


def _split_top_level_binop(text: str, op: str) -> tuple[str, str] | None:
    """Split ``text`` on a SINGLE top-level binary ``op``; ``None`` if not exactly one."""
    depth = 0
    pos = -1
    for i, ch in enumerate(text):
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth = max(0, depth - 1)
        elif ch == op and depth == 0:
            if pos != -1:
                return None            # more than one top-level op — not a simple pair
            pos = i
    if pos == -1:
        return None
    return text[:pos].strip(), text[pos + 1:].strip()


def _parse_int_divisor(text: str) -> int | None:
    """Parse a small positive integer divisor, optionally cast-wrapped (``TScale(6)``)."""
    inner = _strip_casts(text).strip()
    if re.fullmatch(r"\d+", inner):
        return int(inner)
    if re.fullmatch(r"\d+\.0*", inner):   # 6. / 6.0 → integer
        return int(float(inner))
    return None


def derive_from_catalog(name: str, rhs: str, scalar: str,
                        alias_map: dict[str, str] | None = None) -> Derivation | None:
    """Derive a constant whose RHS is a catalog constant or a composition (step 3b).

    Handles the exact-name case (``M_PI`` -> π), a scaled form carried directly by
    the catalog (``2.0*M_PI`` -> ``two_pi``), and an algebraic composition of
    catalog accessors (``_pi() * _pio6<...>()`` -> π²/6, ``_pi() / TScale(6)`` ->
    π/6).  Returns ``None`` when nothing lands on a catalog entry — the value is
    never invented.  ``alias_map`` supplies caller-specific accessor spellings.
    """
    inner = _strip_casts(rhs)
    # bare catalog constant
    canon = _lookup_catalog_alias(inner.strip(), alias_map)
    # bare accessor call (``_pio6<...>()``) — strip template args + () and look up
    if canon is None:
        m_acc = _ACCESSOR_NAME_RE.match(inner.strip())
        if m_acc is not None:
            canon = _lookup_catalog_alias(m_acc.group(1), alias_map)
    if canon and canon in KNOWN_CONSTANTS:
        hi, lo = KNOWN_CONSTANTS[canon][scalar]
        return Derivation(name=name, scalar=scalar, expr=_make_call(scalar, hi, lo),
                          how=f"catalog:{canon}", rhs=rhs.strip())
    # k * <catalog> with a direct scaled catalog entry (numeric scalar factor)
    m = re.match(r"^\s*([\d.]+)\s*\*\s*([A-Za-z_][\w:]*)\s*$", inner)
    if not m:
        m2 = re.match(r"^\s*([A-Za-z_][\w:]*)\s*\*\s*([\d.]+)\s*$", inner)
        if m2:
            k_text, tok = m2.group(2), m2.group(1)
        else:
            k_text = tok = None
    else:
        k_text, tok = m.group(1), m.group(2)
    if k_text is not None:
        base = _lookup_catalog_alias(tok.strip(), alias_map)
        try:
            k = float(k_text)
        except ValueError:
            k = None
        if k is not None:
            if base == "pi" and k == 2.0:
                hi, lo = KNOWN_CONSTANTS["two_pi"][scalar]
                return Derivation(name=name, scalar=scalar,
                                  expr=_make_call(scalar, hi, lo),
                                  how="catalog:two_pi", rhs=rhs.strip())
            if base == "pi" and k == 0.5:
                hi, lo = KNOWN_CONSTANTS["half_pi"][scalar]
                return Derivation(name=name, scalar=scalar,
                                  expr=_make_call(scalar, hi, lo),
                                  how="catalog:half_pi", rhs=rhs.strip())
    # algebraic composition of catalog accessors (π family)
    return _derive_composition(name, inner, scalar, alias_map)


def derive_from_rhs(name: str, rhs: str, scalar: str,
                    alias_map: dict[str, str] | None = None) -> Derivation | None:
    """Run the Gap B derivation cascade over a resolved RHS.

    Order: numeric literal (3a) -> catalog closed form / composition (3b) ->
    ``None`` (the caller falls through to Rule R4).  ``scalar`` is ``"dd"`` or
    ``"ff"``; ``alias_map`` supplies caller-specific accessor spellings for the
    catalog composition branch.
    """
    if not rhs or not rhs.strip():
        return None
    return (derive_literal(name, rhs, scalar)
            or derive_from_catalog(name, rhs, scalar, alias_map))


# --------------------------------------------------------------------------- #
# complex-container derivation (Gap B, Rule 3 for a container constant)
# --------------------------------------------------------------------------- #
# A named constant whose RHS is a 2-element complex container — the iε-prescription
# regulators the box kernels read, ``_ieps50 = TOutput{_zero(), TScale(1e-50)}``
# (an *imaginary* infinitesimal 0 + 1e-50·i) and its siblings ``_ieps`` /
# ``_2ipi`` / ``_ipi`` / ``_ipio2``.  The earlier cascade surfaced only the bare
# scalar literal (``1e-50``) as a "composite" hint and left the model to assemble
# the container itself — which it botched, collapsing ``{0, 1e-50}`` to a *real*
# ``DoubleDouble(1e-50)`` (dropping the imaginary axis the iε prescription lives on) or
# returning the wrong container type.  This derives BOTH limbs of the container so
# the engine can hand the model the complete complex value.


@dataclass(frozen=True)
class ComplexDerivation:
    """A derived complex-container constant: the two component ``make_*`` exprs.

    ``real``/``imag`` are the derived scalar component expressions (a
    ``DoubleDouble::from_bits(...)`` / ``FloatFloat::from_bits(...)`` call each); the regional engine wraps them
    in the concrete complex type spelling it owns.  ``how`` records the component
    provenance for the rule-justification comment.
    """

    name: str
    scalar: str          # "dd" | "ff"
    real: str            # component expr for the real part
    imag: str            # component expr for the imaginary part
    how: str             # e.g. "complex(literal, catalog:pi)"
    rhs: str = ""


def _split_top_level(text: str) -> list[str]:
    """Split ``text`` on top-level commas (ignoring nested ``()[]{}<>``)."""
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts]


# ``[Type]{ ... }`` or ``[Type]( ... )`` — a (optionally type-named) container
# initializer.  The captured inner text is split into components downstream.
_CONTAINER_RE = re.compile(r"^\s*(?:[A-Za-z_][\w:<>]*\s*)?[\{(](.*)[\})]\s*$", re.DOTALL)

# Trailing accessor call in a component (``Constants<TScale>::_zero()`` -> ``_zero``).
_ACCESSOR_CALL_RE = re.compile(r"(?:::\s*)?(?:template\s+)?([A-Za-z_]\w*)\s*\(")


def _derive_component(text: str, scalar: str, sources: list[str],
                      depth: int = 0) -> str | None:
    """Derive one container component to a ``DoubleDouble::from_bits``/``FloatFloat::from_bits`` expr.

    Handles a literal / cast-wrapped literal / catalog constant directly, and a
    named accessor (``Constants<T>::_zero()``) by resolving its own source RHS and
    recursing (bounded).  Returns ``None`` when the component is not derivable
    (an opaque expression, a product of accessors we don't compose, …).
    """
    text = text.strip()
    if not text or depth > 3:
        return None
    direct = derive_from_rhs("_c", text, scalar)
    if direct is not None:
        return direct.expr
    m = _ACCESSOR_CALL_RE.search(text)
    if m is not None:
        inner_rhs = resolve_constant_rhs(m.group(1), sources)
        if inner_rhs is not None:
            return _derive_component(inner_rhs, scalar, sources, depth + 1)
    return None


def derive_complex_from_rhs(name: str, rhs: str, scalar: str,
                            sources: list[str]) -> ComplexDerivation | None:
    """Derive a 2-element complex-container RHS to its real/imag component exprs.

    Recognizes ``[Type]{re, im}`` / ``[Type](re, im)`` where BOTH components are
    themselves derivable (a literal, a catalog constant, or a named accessor that
    resolves to one).  Returns ``None`` for anything else (not a 2-part container,
    or a component we cannot derive) so the caller falls back to the literal-hint /
    Rule R4 path unchanged.
    """
    if not rhs or not rhs.strip():
        return None
    m = _CONTAINER_RE.match(rhs)
    if m is None:
        return None
    parts = _split_top_level(m.group(1))
    if len(parts) != 2:
        return None
    real = _derive_component(parts[0], scalar, sources)
    imag = _derive_component(parts[1], scalar, sources)
    if real is None or imag is None:
        return None
    return ComplexDerivation(
        name=name, scalar=scalar, real=real, imag=imag,
        how="complex", rhs=rhs.strip())


# --------------------------------------------------------------------------- #
# source RHS resolution (walk to a constant's definition by name)
# --------------------------------------------------------------------------- #

def resolve_constant_rhs(name: str, sources: list[str]) -> str | None:
    """Find the definition of constant ``name`` in ``sources`` and return its RHS.

    ``sources`` is a list of already-read source texts (the region file plus any
    scan-reachable headers).  Recognizes the generic C++ constant-definition forms
    a numerical kernel uses; returns the first match's RHS text (without the
    trailing ``;``) or ``None`` when the declaration is not in reach.

    Forms recognized (``NAME`` is the requested identifier):

    * ``#define NAME <rhs>`` (rest of line)
    * ``[static] [constexpr|const] <type> NAME = <rhs> ;``
    * ``[static] [constexpr] <type> NAME ( ) { return <rhs> ; }``            (accessor)
    * ``[static] [constexpr] <type> NAME < ... > ( ) { return <rhs> ; }``    (template accessor)
    * ``template < ... > [static] <type> NAME ( ) { return <rhs> ; }``       (template accessor)
    """
    esc = re.escape(name)

    # #define NAME rhs   (function-like macros NAME(...) are skipped)
    define_re = re.compile(r"^[ \t]*#[ \t]*define[ \t]+" + esc + r"(?![\w(])[ \t]+(.+?)[ \t]*$",
                           re.MULTILINE)
    # <...> TYPE NAME = rhs ;
    assign_re = re.compile(r"\b" + esc + r"\s*=\s*(.+?)\s*;", re.DOTALL)
    # <...> NAME [<...>] ( ) { ... return rhs ; }
    accessor_re = re.compile(
        esc + r"\s*(?:<[^{};]*>)?\s*\(\s*\)\s*\{[^{}]*?\breturn\b\s*(.+?)\s*;",
        re.DOTALL,
    )

    for text in sources:
        m = define_re.search(text)
        if m:
            return _clean_rhs(m.group(1))
        m = accessor_re.search(text)
        if m:
            return _clean_rhs(m.group(1))
        m = _assign_match(assign_re, esc, text)
        if m is not None:
            return _clean_rhs(m)
    return None


def _assign_match(assign_re: re.Pattern, esc: str, text: str) -> str | None:
    """First ``NAME = rhs ;`` whose ``NAME`` is a *declaration* (preceded by a type
    token), not an assignment to a pre-existing variable or a ``==`` comparison."""
    decl_prefix = re.compile(r"[A-Za-z_]\w*[ \t*&>]+$")
    for m in assign_re.finditer(text):
        # guard against '==' : the char right after NAME must be a lone '='
        start = m.start()
        # preceding non-space run should look like a type (an identifier)
        before = text[max(0, start - 64):start]
        if decl_prefix.search(before) or re.search(r"\b(?:constexpr|const|static|inline)\s*$", before):
            return m.group(1)
    return None


def _clean_rhs(rhs: str) -> str:
    """Trim a resolved RHS: drop a line comment tail and surrounding whitespace."""
    # strip a trailing // comment (best-effort; RHS should be a single expression)
    rhs = re.split(r"//", rhs, maxsplit=1)[0]
    return rhs.strip().rstrip(";").strip()


# --------------------------------------------------------------------------- #
# numeric-literal enumeration (for surfacing partial hints on composite RHS)
# --------------------------------------------------------------------------- #

_NUM_IN_TEXT_RE = re.compile(r"(?<![\w.])[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fFlL]*")


def derive_literals_in(rhs: str, scalar: str) -> list[Derivation]:
    """Derive every distinct numeric literal appearing in a composite RHS.

    For a braced / complex RHS like ``TOutput{_zero(), TScale(1e-50)}`` the whole
    expression is not a single scalar constant, but its literals (``1e-50``) are
    each derivable — surfacing them lets the model assemble the container value
    (Rule 3) without guessing bits.  Integer array indices etc. are excluded by
    requiring a fractional part or exponent, or an explicit float-ish context.
    """
    out: list[Derivation] = []
    seen: set[str] = set()
    for m in _NUM_IN_TEXT_RE.finditer(rhs):
        tok = m.group(0)
        # skip pure integers with no float character — likely indices/counts.
        if not re.search(r"[.eE]", tok):
            continue
        if tok in seen:
            continue
        seen.add(tok)
        d = derive_literal(tok, tok, scalar)
        if d is not None:
            out.append(d)
    return out
