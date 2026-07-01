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

TEST_CASE("kahan: compensation step has cond >> 1 — intentional cancellation") {
    // Kahan update for s=1e16, y=1.0:
    //   t   = s + y   = 1e16   (y swamped)
    //   err = t - s   = 0      cond = 1/u ≈ 9e15 — deliberately high
    //   c   = err - y = -1.0   cond ≈ 1
    // The high cond on (t - s) is correct: Kahan deliberately cancels to
    // expose the rounding error.  This shows cond is a signal, not a verdict.
    auto s   = track("s", 1e16);
    auto y   = track("y", 1.0);
    auto t   = s + y;
    auto err = t - s;
    auto c   = err - y;

    REQUIRE(err.max_cond_seen_ > 1e8);   // high cond on the compensation step
    REQUIRE(c.max_cond_seen_   > 1e8);   // carries forward through the chain
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
