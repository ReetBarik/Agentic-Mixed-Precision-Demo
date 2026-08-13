"""Deterministic post-generation source normaliser (Subtask 3, design §6.1).

The closure-scoped chain promotion grows the number of dd shims the chain
integrator (LLM) emits, which stresses the memory's "Blocker B" — a small set of
*generation-robustness* defects that are orthogonal to the closure/gate/emission
machinery.  This module is the deterministic sweep that design §6.1 proposed in
place of a chain-size cap: a pure, idempotent source transform run over the tree
files a chain touched, AFTER the LLM shims are generated and the boundary/variant
promotion is spliced, but BEFORE the build gate.

It fixes three narrow, well-defined syntactic defect classes.  Each transform is:

* **deterministic** — no LLM, no randomness;
* **idempotent** — running it twice is byte-identical to running it once;
* **semantically null on clean input** — a shim/source with none of the defects
  is returned unchanged (byte-for-byte).

The three classes (design §8 point 1, memory's "Blocker B"):

N1 — **redeclaration of an already-promoted local** (the ``T__ff`` class).  When a
     function body contains more than one chain region, each region's boundary
     entry emits ``<ext> r__ff = <ext>(r);`` for the same read ``r``; two such
     lines in one scope are a C++ redeclaration (``error: redeclaration of
     'DoubleDouble T__ff'``).  The redeclaration is dropped down to a plain assignment
     ``r__ff = <ext>(r);`` (the first declaration still introduces the name).

N2 — **malformed unary ``operator+``**.  The LLM sometimes emits a redundant
     unary ``+`` on an extended operand (``x = + y;``, ``return + expr;``); the
     vendored ``DoubleDouble``/``FloatFloat`` define no unary ``operator+``, so this is a
     hard compile error.  A redundant unary ``+`` is semantically null for every
     numeric type, so it is simply removed.

N3 — **``_ieps50``-residual constant spelling** (WAVE-1+2).  A two-limb extended
     constant written through the *scalar constructor from a decimal literal*
     (``DoubleDouble(1e-50)``) is canonicalised to the equivalent full-precision bit
     pair (``DoubleDouble::from_bits(<bits(1e-50)>, 0x0)``).  ``DoubleDouble(double h)`` sets
     ``{hi=h, lo=0}``, so the rewrite is BIT-IDENTICAL — a defensive
     canonicalisation that removes the truncation-prone decimal-literal spelling
     the WAVE-1 model leaked, never a value change.  (The *complex* iε collapse —
     dropping the imaginary axis — is fixed upstream at generation by
     :mod:`agents.shared.constant_derive`; N3 does NOT attempt that semantic fix.)

Nothing here is app-specific: the extended-type spellings are supplied by the
caller (the chain flow passes its ``scalar``/``complex`` C++ spellings); the
defaults name only the vendored ``Kokkos::Experimental`` / ``Kokkos::Experimental`` families the
integrators target.
"""

from __future__ import annotations

import re
from pathlib import Path

from agents.shared import constant_derive as cderive

# The vendored extended-precision scalar/complex spellings the integrators emit.
# Callers pass their own via ``normalise_source(..., ext_types=...)``; this default
# is only the fallback used by tests and the classic dd/ff families.
DEFAULT_EXT_SCALARS: frozenset[str] = frozenset({
    "Kokkos::Experimental::DoubleDouble", "Kokkos::Experimental::FloatFloat",
})
DEFAULT_EXT_COMPLEX: frozenset[str] = frozenset({
    "Kokkos::Experimental::DoubleDoubleComplex", "Kokkos::Experimental::FloatFloatComplex",
})


# --------------------------------------------------------------------------- #
# N1 — drop redeclarations of an already-promoted local within one scope
# --------------------------------------------------------------------------- #

def _decl_re(ext_types: frozenset[str]) -> re.Pattern:
    """A ``<ext-type> <name> = <rhs>;`` declaration line for one of ``ext_types``."""
    alt = "|".join(re.escape(t) for t in sorted(ext_types, key=len, reverse=True))
    return re.compile(
        r"^(?P<indent>\s*)(?P<type>(?:" + alt + r"))\s+"
        r"(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<rhs>.*?;)(?P<trail>\s*(?://.*)?)$")


def _drop_redeclarations(text: str, ext_types: frozenset[str]) -> str:
    """N1: within each brace scope, keep the first ``<ext> name = …`` declaration
    of a given ``(name, type)`` and demote any later identical-scope declaration of
    the same name+type to a plain assignment (drop the type prefix).

    Scope is tracked by brace depth; a declaration seen at depth ``d`` shadows only
    within depth ``d`` (a re-declaration at a deeper depth is legal C++ shadowing
    and is left intact).  Only an exact same-scope, same-name, same-type repeat —
    the ``T__ff`` redeclaration error — is rewritten.
    """
    decl_re = _decl_re(ext_types)
    lines = text.split("\n")
    out: list[str] = []
    # stack of scopes; each scope maps declared name -> type spelling
    scopes: list[dict[str, str]] = [{}]

    for line in lines:
        m = decl_re.match(line)
        if m is not None:
            name, typ = m.group("name"), m.group("type")
            cur = scopes[-1]
            if cur.get(name) == typ:
                # redeclaration in the SAME scope at the SAME type -> assignment
                out.append(f"{m.group('indent')}{name} = "
                           f"{m.group('rhs')}{m.group('trail')}")
                # brace bookkeeping still needs to run on this (rewritten) line
                _track_braces(out[-1], scopes)
                continue
            cur[name] = typ
        out.append(line)
        _track_braces(line, scopes)

    return "\n".join(out)


def _strip_noncode(line: str) -> str:
    """Best-effort removal of line comments + string/char literals for brace counting."""
    line = re.split(r"//", line, maxsplit=1)[0]
    line = re.sub(r"/\*.*?\*/", "", line)
    line = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
    line = re.sub(r"'(?:\\.|[^'\\])*'", "''", line)
    return line


def _track_braces(line: str, scopes: list[dict[str, str]]) -> None:
    """Push/pop scope dicts as ``{``/``}`` open and close (comments/strings stripped)."""
    code = _strip_noncode(line)
    for ch in code:
        if ch == "{":
            scopes.append({})
        elif ch == "}":
            if len(scopes) > 1:
                scopes.pop()


# --------------------------------------------------------------------------- #
# N2 — canonicalise a malformed / redundant unary operator+
# --------------------------------------------------------------------------- #

# A unary ``+`` sits right after an operator / opener / ``return`` (a position where
# a value, not a binary operand, is expected).  We require the ``+`` NOT to be part
# of ``++`` or ``+=`` (guard both neighbours) and to directly precede an operand.
_UNARY_PLUS_RE = re.compile(
    r"(?P<lead>(?:[-=(,{\[?:*/%<>&|^!~]|\breturn\b|\bcase\b)\s*)"
    r"\+"
    r"(?P<sp>\s*)"
    r"(?=[\w(])")


def _drop_redundant_unary_plus(text: str) -> str:
    """N2: remove a redundant unary ``+`` (``= + y`` -> ``= y``, ``return + e`` ->
    ``return e``).  Semantically null for any numeric type; fixes the extended-type
    ``no match for operator+`` build error.  ``++`` / ``+=`` are never touched."""
    def _sub(m: re.Match) -> str:
        # do not eat a '+' that is actually '++' (guard is the lookbehind of lead's
        # last char being '+', which the char class already excludes) — safe here.
        return f"{m.group('lead')}{m.group('sp')}"

    prev = None
    cur = text
    # iterate to a fixed point so ``+ + x`` collapses fully (idempotent either way)
    while cur != prev:
        prev = cur
        cur = _UNARY_PLUS_RE.sub(_sub, cur)
    return cur


# --------------------------------------------------------------------------- #
# N3 — canonicalise a decimal-literal extended-scalar constructor (_ieps50 residual)
# --------------------------------------------------------------------------- #

_MAKE_FOR_SCALAR = {
    "Kokkos::Experimental::DoubleDouble": ("dd", "Kokkos::Experimental::DoubleDouble::from_bits"),
    "Kokkos::Experimental::FloatFloat":   ("ff", "Kokkos::Experimental::FloatFloat::from_bits"),
}


def _canonicalise_literal_ctors(text: str, ext_scalars: frozenset[str]) -> str:
    """N3: rewrite ``<ext-scalar>(<decimal-literal>)`` to the equivalent
    ``DoubleDouble::from_bits/FloatFloat::from_bits(<bits>, 0x0)`` bit pair.

    ``DoubleDouble(double h)`` initialises ``{hi=h, lo=0}``; ``DoubleDouble::from_bits(bits(h), 0)``
    reconstructs the same value, so the rewrite is BIT-IDENTICAL (a defensive
    canonicalisation, never a value change) and removes the decimal-literal
    spelling the WAVE-1 model leaked.  A ctor whose argument is anything but a plain
    decimal literal (an identifier, an expression, an already-``make_*`` call) is
    left untouched → semantically null on clean input, idempotent.
    """
    for scalar_type in ext_scalars:
        info = _MAKE_FOR_SCALAR.get(scalar_type)
        if info is None:
            continue
        which, make_fn = info
        # <scalar>( <literal> )  — literal only (no comma, no nested call)
        ctor_re = re.compile(re.escape(scalar_type) + r"\s*\(\s*([^(),;]*?)\s*\)")

        def _sub(m: re.Match, which=which, make_fn=make_fn) -> str:
            arg = m.group(1).strip()
            d = cderive.derive_literal("_c", arg, which)
            if d is None:
                return m.group(0)          # not a bare decimal literal — leave as-is
            # d.expr is already ``<make_fn>(0x..., 0x0...)`` for this scalar family
            return d.expr

        text = ctor_re.sub(_sub, text)
    return text


# --------------------------------------------------------------------------- #
# public entry points
# --------------------------------------------------------------------------- #

def normalise_source(text: str, *,
                     ext_scalars: frozenset[str] = DEFAULT_EXT_SCALARS,
                     ext_complex: frozenset[str] = DEFAULT_EXT_COMPLEX) -> str:
    """Run all three deterministic normalisations over ``text`` and return the result.

    Idempotent and byte-identical to the input when no defect is present.
    ``ext_scalars`` / ``ext_complex`` supply the caller's extended-type spellings
    (both scalar spellings feed N1's redeclaration scan; the scalar family also
    feeds N3's literal-ctor canonicalisation)."""
    ext_types = ext_scalars | ext_complex
    text = _drop_redeclarations(text, ext_types)
    text = _drop_redundant_unary_plus(text)
    text = _canonicalise_literal_ctors(text, ext_scalars)
    return text


def normalise_file(path: str | Path, *,
                   ext_scalars: frozenset[str] = DEFAULT_EXT_SCALARS,
                   ext_complex: frozenset[str] = DEFAULT_EXT_COMPLEX) -> bool:
    """Normalise the file at ``path`` in place.  Returns ``True`` iff it changed."""
    p = Path(path)
    try:
        original = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    normalised = normalise_source(original, ext_scalars=ext_scalars,
                                  ext_complex=ext_complex)
    if normalised == original:
        return False
    p.write_text(normalised, encoding="utf-8")
    return True
