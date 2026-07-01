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
// Dataset: {1e8, 1e8+1, 1e8-1}
//   mean = 1e8,  true variance = 2/3 ≈ 0.667
//   E[X²] ≈ 1e16 + 2/3,  (E[X])² = 1e16  →  difference = 2/3
//   In FP:  (1e16 + 2/3) rounds to 1e16  →  result = 0  (all digits lost)

TEST_CASE("naive_variance: catastrophic cancellation in E[X²] - (E[X])²") {
    auto d0 = track("d0", 1e8);
    auto d1 = track("d1", 1e8 + 1.0);
    auto d2 = track("d2", 1e8 - 1.0);
    auto n  = track("n",  3.0);

    auto sum_x2    = d0*d0 + d1*d1 + d2*d2;
    auto sum_x     = d0 + d1 + d2;
    auto mean      = sum_x / n;
    auto var_naive = sum_x2 / n - mean * mean;

    // Tracked surfaces the catastrophic cancellation.
    // Condition of the final subtraction ≈ 2e16 / (2/3) ≈ 3e16.
    REQUIRE(var_naive.max_cond_seen_ > 1e10);

    // rel_err_bound >> 1: all significant digits are lost.
    REQUIRE(var_naive.rel_err_bound_ > 1.0);

    // In double precision the +2/3 is below the ULP of 1e16 (≈ 2),
    // so sum_x2/n rounds to exactly 1e16 and the result is 0.
    REQUIRE(var_naive.value_ == Approx(0.0).margin(1e-6));
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
