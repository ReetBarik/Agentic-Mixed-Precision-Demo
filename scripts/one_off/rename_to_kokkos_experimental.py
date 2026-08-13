#!/usr/bin/env python3
"""T3 — rename the Agentic tree to the upstream Kokkos::Experimental API.

Run after third_party/include was refreshed from
kokkos-extended-precision-demo@5ae2f80:

    quad::ddfun / quad::ffun  ->  Kokkos::Experimental
    ddouble   -> DoubleDouble           ffloat    -> FloatFloat
    ddcomplex -> DoubleDoubleComplex    ffcomplex -> FloatFloatComplex
    make_dd(a, b) -> DoubleDouble::from_bits(a, b)
    make_ff(a, b) -> FloatFloat::from_bits(a, b)
    dd_pi() ...   -> DoubleDouble_pi() ...
    ff_pi() ...   -> FloatFloat_pi() ...

WHY THIS IS ONE REGEX PASS AND NOT A PIPELINE OF seds
-----------------------------------------------------
The qcdloop-facing surface calls ql::ddfun::ddouble, ql::ddfun::make_dd (68
sites) and friends. Those resolve through the alias namespaces in
kokkosMaths_dd.h / kokkosMaths_ff.h, which are hand-edited to keep exporting
the OLD names, so every ql:: call site must survive verbatim.

Two sed-based attempts failed on exactly this:

  1. Protecting only the prefix (ql::ddfun:: -> @@QLDD@@) leaves
     "@@QLDD@@ddouble"; \bddouble\b still matches there, because @ is a
     non-word character and the word boundary holds.
  2. Folding the identifier into the sentinel (-> "@@QLDD:make_dd@@") fails
     too: the sentinel's own payload is a renameable token, so the factory
     pass rewrites it to "@@QLDD:DoubleDouble::from_bits@@".

Any protect-then-substitute scheme has this re-entrancy problem. A single
left-to-right pass does not: the ql:: alternative below matches the whole
qualified name first and returns it unchanged, so the scanner resumes AFTER
it and the inner identifier is never re-examined.

Ordering inside the pattern is load-bearing — ql:: must come first, and the
namespace prefix must precede the bare identifiers so that
quad::ddfun::make_dd becomes Kokkos::Experimental::DoubleDouble::from_bits
(prefix rewritten, then the scanner lands on make_dd).

SCOPE — deliberately not the whole tree:
    included : agents/, tests/, live scripts, live design docs, 2 READMEs
    excluded : the 4 vendored headers (upstream + LOCAL PATCH blocks + shim)
               kokkosMaths_dd.h / kokkosMaths_ff.h (hand-edited alias blocks)
               runs/qcdloop/src/, runs/qcdloop_headers_full/ (ql:: surface)
               runs/archive/, runs/qcdloop/tier_b_stage2_* (frozen)
               *.log, *.csv, and any dated report — historical record; a
               2026-07-29 report must not claim an API that postdates it.

Usage:  scripts/one_off/rename_to_kokkos_experimental.py [--dry-run]
"""
from __future__ import annotations

import re
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(
    subprocess.run(["git", "rev-parse", "--show-toplevel"],
                   capture_output=True, text=True, check=True).stdout.strip()
)

IDENT = {
    "make_dd": "DoubleDouble::from_bits",
    "make_ff": "FloatFloat::from_bits",
    "ddcomplex": "DoubleDoubleComplex",
    "ddouble": "DoubleDouble",
    "ffcomplex": "FloatFloatComplex",
    "ffloat": "FloatFloat",
    "dd_euler_gamma": "DoubleDouble_euler_gamma",
    "ff_euler_gamma": "FloatFloat_euler_gamma",
    "dd_log10": "DoubleDouble_log10",
    "ff_log10": "FloatFloat_log10",
    "dd_log2": "DoubleDouble_log2",
    "ff_log2": "FloatFloat_log2",
    "dd_sqrt2": "DoubleDouble_sqrt2",
    "ff_sqrt2": "FloatFloat_sqrt2",
    "dd_pi": "DoubleDouble_pi",
    "ff_pi": "FloatFloat_pi",
    "dd_e": "DoubleDouble_e",
    "ff_e": "FloatFloat_e",
}
# longest first so dd_e cannot shadow dd_euler_gamma
_IDENT_ALT = "|".join(sorted(map(re.escape, IDENT), key=len, reverse=True))

TOKEN_RE = re.compile(
    r"(?P<ql>\bql::(?:ddfun|ffun)::[A-Za-z_][A-Za-z0-9_]*)"   # keep verbatim
    r"|(?P<qual>\bquad::(?:ddfun|ffun)::)"                    # namespace, qualified
    r"|(?P<nsbare>\bquad::(?:ddfun|ffun)\b)"                  # namespace, in prose
    rf"|(?P<ident>\b(?:{_IDENT_ALT})\b)"
)

SCAN_ROOTS = ["agents", "tests", "docs", "HANDOFF.md",
              "runs/qcdloop", "runs/qcdloop_headers_full", "third_party/include"]

EXCLUDE_EXACT = {
    "third_party/include/dd_math.hpp", "third_party/include/ff_math.hpp",
    "third_party/include/dd_complex.hpp", "third_party/include/ff_complex.hpp",
    "third_party/include/kokkosMaths_ff.h",
    "runs/qcdloop_headers_full/kokkosMaths_dd.h",
}
EXCLUDE_GLOB = [
    "runs/archive/*", "runs/qcdloop/tier_b_stage2_*",
    "runs/qcdloop/src/*", "runs/qcdloop_headers_full/*",
    "*.log", "*.csv",
]
DATED_MD = re.compile(r"20\d\d-\d\d-\d\d.*\.md$")


def replace(m: re.Match) -> str:
    if m.group("ql"):
        return m.group("ql")                      # ql:: surface — untouched
    if m.group("qual") or m.group("nsbare"):
        return "Kokkos::Experimental::" if m.group("qual") else "Kokkos::Experimental"
    return IDENT[m.group("ident")]


def in_scope(path: str) -> bool:
    if path in EXCLUDE_EXACT or DATED_MD.search(path):
        return False
    return not any(fnmatch(path, g) for g in EXCLUDE_GLOB)


def main() -> int:
    dry = "--dry-run" in sys.argv
    pattern = (r"quad::(ddfun|ffun)"
               r"|\b(ddouble|ddcomplex|ffloat|ffcomplex|make_dd|make_ff)\b"
               r"|\b(dd|ff)_(pi|e|log2|log10|sqrt2|euler_gamma)\b")
    out = subprocess.run(["git", "grep", "-lE", pattern, "--", *SCAN_ROOTS],
                         cwd=ROOT, capture_output=True, text=True)
    files = sorted(f for f in out.stdout.splitlines() if in_scope(f))

    print(f"T3 rename — {len(files)} files in scope")
    if dry:
        print("\n".join(f"  {f}" for f in files))
        return 0

    changed = 0
    for rel in files:
        p = ROOT / rel
        src = p.read_text()
        dst = TOKEN_RE.sub(replace, src)
        if dst != src:
            p.write_text(dst)
            changed += 1

    # -------- self-check: fail loud rather than leave a half-renamed tree
    bad = []
    for rel in files:
        text = (ROOT / rel).read_text()
        for pat, why in (
            (r"ql::(ddfun|ffun)::(DoubleDouble|FloatFloat)", "ql:: surface rewritten"),
            (r"Kokkos::Experimental::(make_dd|make_ff|ddouble|ddcomplex"
             r"|ffloat|ffcomplex|dd_pi|ff_pi)", "half-renamed symbol"),
            (r"quad::(ddfun|ffun)", "old namespace survived"),
        ):
            for ln, line in enumerate(text.splitlines(), 1):
                if re.search(pat, line):
                    bad.append(f"{why}: {rel}:{ln}: {line.strip()[:120]}")
    if bad:
        print("SELF-CHECK FAILED", file=sys.stderr)
        print("\n".join(bad[:40]), file=sys.stderr)
        return 1

    print(f"done — {changed} files rewritten, self-check clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
