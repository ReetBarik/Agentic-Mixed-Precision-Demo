"""Source-derivable extended-precision constants (Gap B) — the deterministic
companion to the regional integrators' Rule R3 constant cascade.

Motivation
----------
A regional ff/dd shim must materialize any named constant the promoted region
reads as a *two-limb* ``make_dd(0x<hi>, 0x<lo>)`` / ``make_ff(0x<hi>, 0x<lo>)``
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
honest double-double promotion is ``make_dd(<bits of the double>, 0x0)`` — there
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
The regional engine uses these to hand the model ready-made ``make_dd(...)`` /
``make_ff(...)`` values so a derivable constant never reaches R4.

Nothing here is qcdloop-specific: the catalog holds only standard mathematical
values, and the source walk reads generic C++ ``constexpr`` / ``const`` /
``#define`` / literal-returning-accessor declarations.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from decimal import Decimal, getcontext

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
    its faithful double-double promotion is ``make_dd(bits(x), 0x0)``.  This is
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

    ``expr`` is the ``make_dd(...)`` / ``make_ff(...)`` call; ``how`` records which
    cascade step produced it (for the rule-justification comment and telemetry).
    """

    name: str            # source identifier the constant is read by
    scalar: str          # "dd" | "ff"
    expr: str            # e.g. "quad::ddfun::make_dd(0x358dee7a4ad4b81fULL, 0x0ULL)"
    how: str             # "literal" | "catalog:pi" | ...
    rhs: str = ""        # the source RHS the derivation came from (provenance)


def _make_call(scalar: str, hi: int, lo: int) -> str:
    if scalar == "dd":
        return f"quad::ddfun::make_dd(0x{hi:016x}ULL, 0x{lo:016x}ULL)"
    if scalar == "ff":
        return f"quad::ffun::make_ff(0x{hi:08x}U, 0x{lo:08x}U)"
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


def _strip_casts(text: str) -> str:
    """Peel functional / static_cast wrappers: ``TScale(1e-50)`` -> ``1e-50``.

    Only peels when the parentheses balance around the whole expression, so a
    genuine multi-arg call (``foo(a, b)``) or a braced init is left intact.
    """
    prev = None
    cur = text.strip()
    while cur != prev:
        prev = cur
        m = _CAST_RE.match(cur)
        if not m:
            break
        inner = m.group(1)
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


def _lookup_catalog_alias(token: str) -> str | None:
    if token in _CATALOG_ALIASES:
        return _CATALOG_ALIASES[token]
    # tolerate ``std::numbers::pi_v`` style: last path component
    tail = token.rsplit("::", 1)[-1]
    return _CATALOG_ALIASES.get(tail)


def derive_from_catalog(name: str, rhs: str, scalar: str) -> Derivation | None:
    """Derive a constant whose RHS is a bare catalog constant (Gap B step 3b).

    Handles the exact-name case (``M_PI`` -> π).  A scaled form (``2.0*M_PI``)
    is matched only when the catalog carries the scaled value directly (``two_pi``
    for ``2*pi``); otherwise we return ``None`` and let the model compose from the
    catalog base value we still surface in the hint.
    """
    inner = _strip_casts(rhs)
    # bare catalog constant
    canon = _lookup_catalog_alias(inner.strip())
    if canon and canon in KNOWN_CONSTANTS:
        hi, lo = KNOWN_CONSTANTS[canon][scalar]
        return Derivation(name=name, scalar=scalar, expr=_make_call(scalar, hi, lo),
                          how=f"catalog:{canon}", rhs=rhs.strip())
    # k * <catalog> with a direct scaled catalog entry (only 2*pi today)
    m = re.match(r"^\s*([\d.]+)\s*\*\s*([A-Za-z_][\w:]*)\s*$", inner)
    if not m:
        m2 = re.match(r"^\s*([A-Za-z_][\w:]*)\s*\*\s*([\d.]+)\s*$", inner)
        if m2:
            k_text, tok = m2.group(2), m2.group(1)
        else:
            return None
    else:
        k_text, tok = m.group(1), m.group(2)
    base = _lookup_catalog_alias(tok.strip())
    try:
        k = float(k_text)
    except ValueError:
        return None
    if base == "pi" and k == 2.0:
        hi, lo = KNOWN_CONSTANTS["two_pi"][scalar]
        return Derivation(name=name, scalar=scalar, expr=_make_call(scalar, hi, lo),
                          how="catalog:two_pi", rhs=rhs.strip())
    if base == "pi" and k == 0.5:
        hi, lo = KNOWN_CONSTANTS["half_pi"][scalar]
        return Derivation(name=name, scalar=scalar, expr=_make_call(scalar, hi, lo),
                          how="catalog:half_pi", rhs=rhs.strip())
    return None


def derive_from_rhs(name: str, rhs: str, scalar: str) -> Derivation | None:
    """Run the Gap B derivation cascade over a resolved RHS.

    Order: numeric literal (3a) -> catalog closed form (3b) -> ``None`` (the
    caller falls through to Rule R4).  ``scalar`` is ``"dd"`` or ``"ff"``.
    """
    if not rhs or not rhs.strip():
        return None
    return derive_literal(name, rhs, scalar) or derive_from_catalog(name, rhs, scalar)


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
