#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <tracked/tracked.hpp>

#include <cmath>

using Catch::Approx;
using tracked::track;

// Calibration: naive one-pass variance formula.
//
// Naive:  Var = E[X²] - (E[X])²
//
// When mean >> std dev, both terms are ≈ mean², and their difference
// is the tiny variance.  In double precision the subtraction catastrophically
// cancels, producing 0 or garbage instead of the true variance.
//
// Dataset: {1e6, 1e6+1, 1e6-1}
//   mean = 1e6,  true variance = 2/3 ≈ 0.667
//   E[X²] = 1e12 + 2/3,  (E[X])² = 1e12  →  difference = 2/3
//   Here 2/3 stays well above ULP(1e12) ≈ 1.2e-4, so sum_x2/n and mean² are
//   DISTINCT — the subtraction is a genuine near-cancellation (not the
//   fully-swamped 1e8 case, which rounds sum_x2/n down to exactly 1e12 and
//   collapses to an exact, lossless subtraction that Tracked reports as cond=1).
//   Condition ≈ (2e12) / (2/3) ≈ 3e12: catastrophic, but mid-range — this
//   exercises the general (|a|+|b|)/|a-b| heuristic, not the 1/u sentinel.

TEST_CASE("naive_variance: catastrophic cancellation in E[X²] - (E[X])²") {
    auto d0 = track("d0", 1e6);
    auto d1 = track("d1", 1e6 + 1.0);
    auto d2 = track("d2", 1e6 - 1.0);
    auto n  = track("n",  3.0);

    auto sum_x2    = d0*d0 + d1*d1 + d2*d2;
    auto sum_x     = d0 + d1 + d2;
    auto mean      = sum_x / n;
    auto var_naive = sum_x2 / n - mean * mean;

    // Tracked surfaces the catastrophic cancellation with a mid-range cond:
    // clearly above 1e10, but well below the 1/u ≈ 9e15 exact-cancel sentinel.
    // This proves the general cancellation heuristic fired, not the sentinel.
    REQUIRE(var_naive.max_cond_seen_ > 1e10);
    REQUIRE(var_naive.max_cond_seen_ < 1e15);

    // Many significant digits are lost: rel_err_bound is inflated far above u.
    double u = tracked::unit_roundoff<double>();
    REQUIRE(var_naive.rel_err_bound_ > 1e6 * u);

    // The value is still ≈ 2/3 in this regime (unlike the fully-swamped 1e8
    // dataset that collapses to 0), but Tracked flags its degraded accuracy.
    REQUIRE(var_naive.value_ == Approx(2.0 / 3.0).epsilon(1e-2));
}

TEST_CASE("naive_variance: well-conditioned when mean is small") {
    // Dataset: {1, 2, 3} — mean = 2, true variance = 2/3
    // E[X²] = 14/3 ≈ 4.667,  (E[X])² = 4.0  →  difference 2/3  (cond ≈ 13)
    // No catastrophic cancellation; Tracked confirms it.
    auto d0 = track("d0", 1.0);
    auto d1 = track("d1", 2.0);
    auto d2 = track("d2", 3.0);
    auto n  = track("n",  3.0);

    auto sum_x2    = d0*d0 + d1*d1 + d2*d2;
    auto sum_x     = d0 + d1 + d2;
    auto mean      = sum_x / n;
    auto var_naive = sum_x2 / n - mean * mean;

    REQUIRE(var_naive.max_cond_seen_ < 1e4);
    REQUIRE(var_naive.value_ == Approx(2.0 / 3.0).epsilon(1e-10));
}
