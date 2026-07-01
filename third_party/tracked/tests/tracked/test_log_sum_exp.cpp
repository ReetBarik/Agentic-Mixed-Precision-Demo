#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>

#include <cmath>

using Catch::Approx;
using tracked::track;

// Calibration: log-sum-exp and the exp condition number.
//
// Naive form:  log(exp(a) + exp(b))
//   cond(exp(x)) = |x|  — large arguments amplify input errors.
//
// Stable form: shift by m = max(a, b) first.
//   m + log(exp(a-m) + exp(b-m))
//   Shifted arguments are ≤ 0 and bounded in magnitude, so cond(exp) ≤ |b-a|.
//
// Tracked<T> surfaces max_cond from exp as the dominant risk in the naive form
// and confirms the stable form stays well-conditioned.

TEST_CASE("log_sum_exp: exp condition number equals |x|") {
    // exp(100): any relative error in the input is amplified 100-fold.
    auto a = track("a", 100.0);
    auto r = tracked::exp(a);
    REQUIRE(r.max_cond_seen_ == Approx(100.0));

    // rel_err_bound = cond * (u + u) = 100 * 2u
    double u = tracked::unit_roundoff<double>();
    REQUIRE(r.rel_err_bound_ == Approx(100.0 * 2.0 * u).epsilon(1e-6));
}

TEST_CASE("log_sum_exp naive: max_cond dominated by large exp argument") {
    // log(exp(100) + exp(99))
    // exp(100): cond = 100 — this propagates through the add and log.
    auto a = track("a", 100.0);
    auto b = track("b",  99.0);
    auto result = tracked::log(tracked::exp(a) + tracked::exp(b));
    REQUIRE(result.max_cond_seen_ >= 100.0);
}

TEST_CASE("log_sum_exp stable: shift keeps exp argument small, cond stays low") {
    // Shift a=100, b=99 by m=100.  Shifted: a-m=0, b-m=-1.
    // exp(0):  cond = 0     (no amplification)
    // exp(-1): cond = 1     (1× amplification)
    // vs. exp(100): cond = 100  in the naive form.
    auto a_s   = track("a_s", 0.0);   // a - m
    auto b_s   = track("b_s", -1.0);  // b - m
    auto m     = track("m", 100.0);
    auto result = m + tracked::log(tracked::exp(a_s) + tracked::exp(b_s));

    // max_cond is from exp(-1)=1 and log: well below the naive form's 100.
    REQUIRE(result.max_cond_seen_ < 10.0);
}

TEST_CASE("log_sum_exp: log condition blows up near x=1") {
    // log(1 + eps): log(x) → 0 as x → 1, so cond = 1/|log(x)| → ∞.
    // This arises in the stable form when exp(a-m) + exp(b-m) ≈ 1,
    // i.e. when a ≈ b ≈ m and exp(b-m) ≈ 0.
    auto x = track("x", 1.0 + 1e-12);
    auto r = tracked::log(x);
    REQUIRE(r.max_cond_seen_ > 1e10);
}
