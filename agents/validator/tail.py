"""Tail-sample battery primitives, shared by the offset emitter and the Validator.

The tail battery is the *adversarial* complement to the Validator's n=1000 random
battery: on every candidate it re-tests the specific per-integral input offsets
that the characterization pass flagged as the hardest — worst achieved relative
error, worst cancellation-conditioning, and magnitude extremes on the integral's
output components.  A retro-probe of accepted float demotions showed the finite random battery can
miss failure modes at untested inputs; rather than pile on preemptive gates, this
upgrades the Validator's guarantee to test the known-adversarial points directly.

Offset identity
---------------
An "offset i for integral B" is the i-th sample of B's *own* input stream, from
the top of B's fresh ``std::mt19937(12345)`` reseed (see
``boxGPU_app_recipes.hpp::run_integral`` — every integral re-seeds and fills
``[0,total)`` before dispatch, so a per-integral offset is bit-identical however
it is chunked).  The tracked (characterizer) and app (validator) drivers share
that recipe header verbatim, so offsets transfer between them.  Two driver flags
added for tail testing:

* ``--dump-inputs N`` — print the raw double inputs (``INP`` lines) for ``[0,N)``,
  no integral evaluated.  Used to fingerprint the input *generator* (determinism
  hash), independent of the candidate's algorithm.
* ``--sample-list a,b,c`` — dispatch only the listed per-integral offsets in one
  invocation (fills ``[0,max+1)`` so each listed offset's draw stream is identical
  to a full run).  Regenerates a sparse tail set for every integral at once.

Determinism hash
----------------
``determinism_hash`` for an integral is the SHA-256 of that integral's first-100
canonical input lines.  The emitter freezes it into the report; the Validator
recomputes it from the candidate binary and compares.  Because inputs are
generated *before* the integral is evaluated, the hash is independent of the
candidate patch — it drifts only if the input generator (mt19937 / distribution /
toolchain) or the offset semantics change, which is exactly what must invalidate
the preserved offsets.  A mismatch is a hard, loud failure (never a silent
fall-back to random-only).
"""

from __future__ import annotations

import hashlib
import subprocess
from array import array

from agents.validator import runner
from agents.validator.coeffs import N_COMPONENTS, parse_component

# The four tail criteria, in report/schema order.
CRITERIA = ("max_rel_err", "max_cond", "max_abs_value", "min_abs_value")

# Default determinism-hash window: first N inputs per integral.
DETERMINISM_N = 100


class DeterminismMismatch(RuntimeError):
    """Raised when a candidate's input generator disagrees with the report.

    Carries the integral, expected, and actual hashes so the caller can surface
    the exact ``DETERMINISM_MISMATCH: <integral> hash <actual> != <expected>``.
    """

    def __init__(self, integral: str, expected: str, actual: str):
        self.integral = integral
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"DETERMINISM_MISMATCH: {integral} hash {actual} != {expected}")


def _run_driver(binary, extra_args: list[str]) -> str:
    """Run ``binary`` with ``extra_args`` under the module env, return stdout.

    Mirrors ``runner._run_chunk`` (login shell + module prelude) but returns the
    stdout directly — the tail/dump payloads are small (a few thousand lines).
    """
    cmd = f"{binary} " + " ".join(extra_args)
    r = subprocess.run(
        ["bash", "-lc", f"{runner.MODULE_PRELUDE} && {cmd}"],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"driver {extra_args} failed:\n{r.stderr[-2000:]}")
    return r.stdout


# ---------------------------------------------------------------------------
# Determinism hash (input fingerprint)
# ---------------------------------------------------------------------------

def dump_inputs(binary, n: int = DETERMINISM_N) -> dict[str, list[str]]:
    """Run ``--dump-inputs n``; return ``{integral: [canonical_line, ...]}``.

    Each canonical line is the ``INP`` payload *after* the ``INP,<integral>,``
    prefix — i.e. ``<i>,<mu2>,<m0..3>,<p0..5>`` (index + 11 hex input tokens) —
    so the fingerprint pins both the draw values and their per-integral order.
    Lines are returned in emission (index) order.
    """
    out = _run_driver(binary, ["--dump-inputs", str(int(n))])
    per: dict[str, list[str]] = {}
    for line in out.splitlines():
        if not line.startswith("INP,"):
            continue
        # INP,<integral>,<rest...>
        _, integral, rest = line.split(",", 2)
        per.setdefault(integral, []).append(rest)
    return per


def determinism_hash(binary, n: int = DETERMINISM_N) -> dict[str, str]:
    """``{integral: "sha256:<hex>"}`` over the first ``n`` inputs of each integral."""
    per = dump_inputs(binary, n)
    return {integral: _hash_lines(lines) for integral, lines in per.items()}


def _hash_lines(lines: list[str]) -> str:
    h = hashlib.sha256()
    h.update("\n".join(lines).encode("utf-8"))
    return "sha256:" + h.hexdigest()


def verify_determinism(binary, expected: dict[str, str], integrals: list[str],
                       n: int = DETERMINISM_N) -> None:
    """Recompute hashes from ``binary`` and compare to ``expected`` for ``integrals``.

    Raises :class:`DeterminismMismatch` on the first divergence (loud, no silent
    fall-back).  Integrals in ``integrals`` that are absent from ``expected`` are
    the caller's fail-open responsibility (they should be filtered out first).
    """
    actual = determinism_hash(binary, n)
    for integral in integrals:
        exp = expected.get(integral)
        if exp is None:
            continue  # fail-open handled upstream
        act = actual.get(integral)
        if act != exp:
            raise DeterminismMismatch(integral, exp, act or "<absent>")


# ---------------------------------------------------------------------------
# Sparse offset dispatch (--sample-list)
# ---------------------------------------------------------------------------

def run_offsets(binary, offsets: list[int]) -> dict[str, dict[int, list[tuple[float, float]]]]:
    """Dispatch ``offsets`` via ``--sample-list``; return per-integral coeffs.

    Returns ``{integral: {offset: [(hi,lo) x N_COMPONENTS]}}``.  ``offsets`` is
    the union across all integrals (the driver dispatches the same list for every
    integral in one invocation); callers pick the offsets that matter per
    integral.  An empty ``offsets`` returns ``{}`` without spawning the driver.
    """
    offsets = sorted({int(o) for o in offsets if int(o) >= 0})
    if not offsets:
        return {}
    out = _run_driver(binary, ["--sample-list", ",".join(str(o) for o in offsets)])
    per: dict[str, dict[int, list[tuple[float, float]]]] = {}
    for line in out.splitlines():
        if not line.startswith("RES,"):
            continue
        parts = line.split(",")
        if len(parts) != 3 + N_COMPONENTS:
            raise ValueError(f"malformed RES line: {line!r}")
        integral = parts[1]
        idx = int(parts[2])
        comps = [parse_component(tok) for tok in parts[3:]]
        per.setdefault(integral, {})[idx] = comps
    return per


def integral_offsets(tail_samples: dict) -> list[int]:
    """Union of every offset across the four criteria of one integral's tail spec."""
    offs: set[int] = set()
    for crit in CRITERIA:
        for entry in tail_samples.get(crit, []):
            off = entry.get("offset")
            if off is not None:
                offs.add(int(off))
    return sorted(offs)


def all_offsets(report_tail: dict[str, dict]) -> list[int]:
    """Union of tail offsets across all integrals — the driver's ``--sample-list``."""
    offs: set[int] = set()
    for ts in report_tail.values():
        offs.update(integral_offsets(ts))
    return sorted(offs)


def load_tail_samples(report_path) -> dict[str, dict]:
    """Extract ``{integral: tail_samples}`` from a (possibly large) report.

    Returns ``{}`` when the report predates the tail schema (no integral carries a
    ``tail_samples`` field) — the Validator then fails open to random-only.  Reads
    the whole report once at run setup (small, one-time cost); the Strategy agent
    re-reads it for region records independently.
    """
    import json
    with open(report_path) as fh:
        report = json.load(fh)
    out: dict[str, dict] = {}
    for integral, obj in report.get("integrals", {}).items():
        ts = obj.get("tail_samples")
        if ts:
            out[integral] = ts
    return out
