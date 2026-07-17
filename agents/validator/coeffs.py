"""Parse the app drivers' ``RES,...`` output into per-(integral, sample) coeffs.

Each driver prints one line per dispatched sample:

    RES,<integral>,<global_sample_idx>,c0re,c0im,c1re,c1im,c2re,c2im

where each of the six coefficient components is a hex token:

* vanilla : ``0x<16-hex>``            → the double's IEEE-754 bit pattern
* dd      : ``0x<16-hex>|0x<16-hex>`` → the DD value's ``(hi, lo)`` doubles

Both are decoded to a uniform ``(hi, lo)`` float pair (vanilla ``lo == 0.0``) so
the precise-digits computation treats candidate and reference identically.
"""

from __future__ import annotations

import struct
from typing import Iterable

# Six components per sample, in emission order.
COMPONENT_LABELS = (
    "coeff0.real", "coeff0.imag",
    "coeff1.real", "coeff1.imag",
    "coeff2.real", "coeff2.imag",
)
N_COMPONENTS = len(COMPONENT_LABELS)


def hex_to_double(tok: str) -> float:
    """Decode a ``0x<16-hex>`` IEEE-754 bit pattern to a Python float."""
    return struct.unpack("<d", struct.pack("<Q", int(tok, 16)))[0]


def parse_component(tok: str) -> tuple[float, float]:
    """Decode one component token to a ``(hi, lo)`` double pair.

    ``0xAAAA`` → ``(value, 0.0)`` (vanilla); ``0xAAAA|0xBBBB`` → ``(hi, lo)`` (DD).
    """
    if "|" in tok:
        hi_s, lo_s = tok.split("|", 1)
        return hex_to_double(hi_s), hex_to_double(lo_s)
    return hex_to_double(tok), 0.0


# One parsed sample: 6 components, each a (hi, lo) pair.
Sample = tuple  # tuple[tuple[float, float], ...] of length N_COMPONENTS


def parse_res_lines(lines: Iterable[str]) -> dict[str, dict[int, Sample]]:
    """Aggregate ``RES`` lines into ``{integral: {sample_idx: (6×(hi,lo))}}``.

    Non-``RES`` lines (Kokkos banners, the driver's stderr echo) are ignored.
    A malformed ``RES`` line (wrong field count) raises ``ValueError`` — silent
    truncation of a driver run would corrupt the ground truth.
    """
    out: dict[str, dict[int, Sample]] = {}
    for line in lines:
        if not line.startswith("RES,"):
            continue
        parts = line.rstrip("\n").split(",")
        # RES, integral, idx, + 6 components
        if len(parts) != 3 + N_COMPONENTS:
            raise ValueError(f"malformed RES line ({len(parts)} fields): {line!r}")
        integral = parts[1]
        idx = int(parts[2])
        comps = tuple(parse_component(tok) for tok in parts[3:])
        out.setdefault(integral, {})[idx] = comps
    return out
