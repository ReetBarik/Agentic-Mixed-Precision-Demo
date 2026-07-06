#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <tracked/tracked.hpp>

using Catch::Approx;
using tracked::track;

// Calibration: Kahan (compensated) summation vs naive accumulation.
//
// Two distinct pathologies — one that Tracked<T> catches, one it misses:
//
//  SWAMPING: small value is silently absorbed into a large accumulator.
//    1e16 + 1.0 = 1e16 in double (ULP(1e16) = 2 > 1.0).
//    cond of the add is ≈ 1 (no cancellation), so Tracked<T> gives no warning.
//    This is a LIMITATION of the first-order condition-number model.
//
//  CANCELLATION in the Kahan compensation step: the computation
//    c = (t - s) - y deliberately extracts rounding error via cancellation.
//    Tracked<T> flags it — correctly, but the cancellation here is intentional.
//
// Takeaway: a high condition-number flag does not always mean "bug"; it means
// "this operation is sensitive to input error."  Kahan uses that sensitivity
// on purpose to improve accuracy.

TEST_CASE("kahan: swamping is invisible to Tracked<T>") {
    // 1e16 + 1.0: ULP(1e16) = 2, so 1.0 < ULP and is rounded away.
    // cond of add = (1e16 + 1) / 1e16 ≈ 1 — Tracked sees no problem.
    auto s  = track("s",  1e16);
    auto x1 = track("x1", 1.0);
    auto r  = s + x1;

    // Condition number ≈ 1 — swamping is undetected by the first-order model.
    REQUIRE(r.max_cond_seen_ < 2.0);

    // Yet the FP result is exactly 1e16: the 1.0 was silently lost.
    REQUIRE(r.value_ == 1e16);
}

TEST_CASE("kahan: fully-swamped compensation step is exact — cond == 1") {
    // Kahan update for s=1e16, y=1.0. y is entirely swamped (1.0 < ULP(1e16)=2):
    //   t   = s + y   = 1e16   (== s, bit-identical; the swamping add has cond ≈ 1)
    //   err = t - s   = 0      bit-identical operands → EXACT cancellation, cond = 1
    //   c   = err - y = -1.0   cond ≈ 1
    // The information in y was destroyed by swamping in the *addition* — a
    // documented blind spot of the first-order model (see the swamping test
    // above). The subtraction t - s itself recovers nothing and loses nothing,
    // so its cond is correctly 1, NOT the old spurious 1/u ≈ 9e15 sentinel that
    // this fix removes. (A *partially*-swamped step, where t ≠ s, is a genuine
    // near-cancellation and still reports high cond — see test_cancellation.cpp.)
    auto s   = track("s", 1e16);
    auto y   = track("y", 1.0);
    auto t   = s + y;
    auto err = t - s;
    auto c   = err - y;

    REQUIRE(t.value_ == s.value_);              // y fully swamped: t is bit-identical to s
    REQUIRE(err.value_ == 0.0);
    REQUIRE(err.max_cond_seen_ == Approx(1.0)); // exact cancellation → no false alarm
    REQUIRE(c.max_cond_seen_   == Approx(1.0)); // stays low through the chain
}

TEST_CASE("kahan: well-conditioned chain certified safe by Tracked<T>") {
    // Positive values, similar magnitudes — no cancellation anywhere.
    // Tracked<T> confirms: max_cond ≈ 1, rel_err stays near a few × u.
    auto a = track("a", 1.0);
    auto b = track("b", 2.0);
    auto c = track("c", 3.0);
    auto d = track("d", 4.0);
    auto r = ((a + b) + c) + d;

    REQUIRE(r.value_ == Approx(10.0));
    REQUIRE(r.max_cond_seen_ < 2.0);

    double u = tracked::unit_roundoff<double>();
    REQUIRE(r.rel_err_bound_ < 10.0 * u);
}
