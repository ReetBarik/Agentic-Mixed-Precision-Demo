#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <tracked/tracked.hpp>

using Catch::Approx;
using tracked::track;

// Calibration: catastrophic cancellation detection.
//
// Cancellation occurs in (a - b) when a ≈ b: large operands, tiny result.
// Condition number = (|a|+|b|)/|a-b| >> 1 amplifies any input error.
// Tracked<T> flags this automatically.

TEST_CASE("cancellation: (1 + eps) - 1 has cond >> 1") {
    // Exact result = eps.  FP keeps the correct value here (eps is representable),
    // but all relative accuracy in the result came from cancellation.
    // cond = (|1+eps| + |1|) / |eps| ≈ 2/eps = 2e10 >> 1e8.
    double eps = 1e-10;
    auto one   = track("one", 1.0);
    auto eps_t = track("eps", eps);
    auto sum   = one + eps_t;   // cond ≈ 1, no cancellation
    auto diff  = sum - one;     // cond ≈ 2e10 — catastrophic

    REQUIRE(diff.max_cond_seen_ > 1e8);
}

TEST_CASE("cancellation: rel_err_bound grows by factor of cond") {
    // After a cancelling subtraction with cond ≈ 2e10, the relative error
    // bound on the result is cond × 2u ≈ 2e10 × 2.2e-16 ≈ 4.4e-6.
    // That means ~10 decimal digits have been lost.
    double eps = 1e-10;
    auto a = track("a", 1.0 + eps);
    auto b = track("b", 1.0);
    auto r = a - b;

    double u = tracked::unit_roundoff<double>();
    REQUIRE(r.rel_err_bound_ > 1e4 * u);   // at least 10 000× fresh unit_roundoff
}

TEST_CASE("cancellation: two cancellations in sequence compound the error") {
    // r1 = (1 + eps) - 1  →  cond ≈ 2e10
    // r2 = (r1 + r1) - r1  →  same cond again; max_cond_seen carries forward
    double eps = 1e-9;
    auto a = track("a", 1.0 + eps);
    auto b = track("b", 1.0);
    auto r1 = a - b;
    auto r2 = r1 + r1;
    auto r3 = r2 - r1;   // another cancelling subtraction on already-damaged values

    // max_cond_seen accumulates across the chain
    REQUIRE(r3.max_cond_seen_ > 1e8);
    // rel_err_bound has grown dramatically from fresh u ≈ 1.1e-16
    REQUIRE(r3.rel_err_bound_ > 1e-8);
}
