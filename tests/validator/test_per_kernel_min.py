"""Phase 2f kernel-scope — Validator ``_score`` per-integral min_precise_digits.

The kernel-scope acceptance gate reads a per-kernel floor from the same whole-app
scoring sweep the verdict already runs.  These tests pin that the per-integral min
is a correct by-product: the whole-app min is the min ACROSS integrals, and each
integral's own entry is the min WITHIN that integral.
"""

from array import array

from agents.validator.validate import _score
from agents.validator.coeffs import N_COMPONENTS


def _coeffs(hi_by_integral):
    """Build a CoeffArrays {integral: (hi, lo)} with lo == 0 (vanilla), one sample.

    ``hi_by_integral`` maps integral -> list of N_COMPONENTS hi values (one sample).
    """
    out = {}
    for integ, hi in hi_by_integral.items():
        assert len(hi) == N_COMPONENTS
        out[integ] = (array("d", hi), array("d", [0.0] * N_COMPONENTS))
    return out


def test_per_integral_min_is_min_within_each_integral():
    # DD reference: both integrals have all components = 1.0 (well-conditioned).
    ref = _coeffs({"B12": [1.0] * N_COMPONENTS, "B14": [1.0] * N_COMPONENTS})
    # Candidate: B12 has a badly-off component (rel err 1e-2 -> ~2 digits); B14 exact.
    cand_hi_b12 = [1.0] * N_COMPONENTS
    cand_hi_b12[0] = 1.0 + 1e-2      # ~2 correct digits on coeff0.real
    cand = _coeffs({"B12": cand_hi_b12, "B14": [1.0] * N_COMPONENTS})

    stats = _score(cand, ref, "candidate", None)
    pk = stats["per_integral_min_precise_digits"]
    assert set(pk) == {"B12", "B14"}
    # B12 dragged down to ~2 digits; B14 exact -> capped.
    assert pk["B12"] < 3.0
    assert pk["B14"] > 30.0
    # Whole-app min == min across integrals == B12's floor.
    assert stats["min_precise_digits"] == pk["B12"]


def test_whole_app_min_is_min_across_kernels():
    ref = _coeffs({"B12": [1.0] * N_COMPONENTS, "B14": [1.0] * N_COMPONENTS})
    # B12 slightly off (~4 digits), B14 worse (~1 digit) -> whole-app == B14.
    b12 = [1.0] * N_COMPONENTS; b12[0] = 1.0 + 1e-4
    b14 = [1.0] * N_COMPONENTS; b14[0] = 1.0 + 1e-1
    cand = _coeffs({"B12": b12, "B14": b14})

    stats = _score(cand, ref, "candidate", None)
    pk = stats["per_integral_min_precise_digits"]
    assert pk["B14"] < pk["B12"]
    assert stats["min_precise_digits"] == min(pk.values()) == pk["B14"]
