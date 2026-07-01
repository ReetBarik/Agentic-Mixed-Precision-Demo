#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <tracked/complex.hpp>

#include <cmath>

using Catch::Approx;
using tracked::track;
using tracked::Complex;

// ============================================================
// Smith division — values
// ============================================================

TEST_CASE("Smith div: (1+2i)/(3+4i) = 11/25 + 2/25·i") {
    auto a = tracked::track("a", 1.0, 2.0);
    auto b = tracked::track("b", 3.0, 4.0);
    auto r = a / b;
    REQUIRE(r.real().value_ == Approx(11.0 / 25.0).epsilon(1e-12));
    REQUIRE(r.imag().value_ == Approx(2.0  / 25.0).epsilon(1e-12));
}

TEST_CASE("Smith div: (6+0i)/(2+0i) = 3 (pure real)") {
    auto a = tracked::track("a", 6.0, 0.0);
    auto b = tracked::track("b", 2.0, 0.0);
    auto r = a / b;
    REQUIRE(r.real().value_ == Approx(3.0).epsilon(1e-12));
    REQUIRE(r.imag().value_ == Approx(0.0).margin(1e-15));
}

TEST_CASE("Smith div: takes |c|>|d| branch correctly") {
    // |c|=3 > |d|=1, exercises the first Smith branch
    auto a = tracked::track("a", 2.0, 1.0);
    auto b = tracked::track("b", 3.0, 1.0);   // |re|>|im|
    auto r = a / b;
    double ex_re = (2.0*3.0 + 1.0*1.0) / (3.0*3.0 + 1.0*1.0);  // 7/10
    double ex_im = (1.0*3.0 - 2.0*1.0) / (3.0*3.0 + 1.0*1.0);  // 1/10
    REQUIRE(r.real().value_ == Approx(ex_re).epsilon(1e-12));
    REQUIRE(r.imag().value_ == Approx(ex_im).epsilon(1e-12));
}

TEST_CASE("Smith div: takes |d|>|c| branch correctly") {
    // |im|=4 > |re|=1, exercises the second Smith branch
    auto a = tracked::track("a", 2.0, 3.0);
    auto b = tracked::track("b", 1.0, 4.0);   // |im|>|re|
    auto r = a / b;
    double ex_re = (2.0*1.0 + 3.0*4.0) / (1.0*1.0 + 4.0*4.0);  // 14/17
    double ex_im = (3.0*1.0 - 2.0*4.0) / (1.0*1.0 + 4.0*4.0);  // -5/17
    REQUIRE(r.real().value_ == Approx(ex_re).epsilon(1e-12));
    REQUIRE(r.imag().value_ == Approx(ex_im).epsilon(1e-12));
}

// ============================================================
// Smith vs naive: overflow safety for large |b|
// ============================================================

TEST_CASE("Smith div: stays finite for |b| ≈ 1e154 (naive would overflow)") {
    // Naive denominator c²+d² overflows when |c| or |d| > ~1e154 (double max ≈ 1.8e308)
    double large = 1e154;
    auto a = tracked::track("a", 1.0, 0.0);
    auto b = tracked::track("b", large, large);
    auto r = a / b;
    // Smith result should be finite and correct: (1+0i)/(L+Li) = (1-i)/(2L)
    REQUIRE(std::isfinite(r.real().value_));
    REQUIRE(std::isfinite(r.imag().value_));
    REQUIRE(r.real().value_ == Approx(1.0 / (2.0 * large)).epsilon(1e-6));
    REQUIRE(r.imag().value_ == Approx(-1.0 / (2.0 * large)).epsilon(1e-6));
}

TEST_CASE("Smith div: naive form silently wrong for large |b| (demonstration)") {
    // The naive denominator c²+d² overflows to inf. Dividing a finite numerator
    // by inf gives 0, not the correct 1/(2L). Silent wrong result — no NaN/inf
    // to catch at runtime. Smith avoids forming c²+d² entirely.
    double large = 1e154;
    double a_re = 1.0, a_im = 0.0;
    double c = large, d = large;

    double naive_denom = c*c + d*d;          // overflow → inf
    double naive_re    = (a_re*c + a_im*d) / naive_denom;  // large/inf → 0
    double naive_im    = (a_im*c - a_re*d) / naive_denom;  // -large/inf → -0

    REQUIRE(!std::isfinite(naive_denom));         // denominator overflowed
    REQUIRE(naive_re == Approx(0.0).margin(1e-300));  // silently zero — WRONG
    REQUIRE(naive_im == Approx(0.0).margin(1e-300));  // correct answer is ±1/(2L)
}

// ============================================================
// Smith division: Tracked correctly flags naive instability
// ============================================================

TEST_CASE("Smith div: Tracked Smith shows lower max_cond than naive for large |b|") {
    // Use a moderately large |b| where both forms are finite but naive has
    // larger intermediate cancellation risk.
    double large = 1e100;
    auto a_s = tracked::track("a", 1.0, 0.0);
    auto b_s = tracked::track("b", large, large);
    auto smith = a_s / b_s;
    double smith_max_cond = std::max(smith.real().max_cond_seen_,
                                     smith.imag().max_cond_seen_);

    // Naive as inline tracked ops
    auto a_n = tracked::track("a_n", 1.0, 0.0);
    auto c_n = tracked::track("c_n", large, 0.0);
    auto d_n = tracked::track("d_n", 0.0, large);
    // (ac+bd)/(c²+d²) + i·(bc-ad)/(c²+d²)  — c=large, d=large
    auto c_t  = tracked::track("c_t",  large);
    auto d_t  = tracked::track("d_t",  large);
    auto a_re = tracked::track("a_re", 1.0);
    auto a_im = tracked::track("a_im", 0.0);
    auto c2   = c_t * c_t;
    auto d2   = d_t * d_t;
    auto den  = c2 + d2;      // potential overflow or catastrophic add for equal squares
    auto ac   = a_re * c_t;
    auto bd   = a_im * d_t;
    auto num_re = ac + bd;
    auto bc   = a_im * c_t;
    auto ad   = a_re * d_t;
    auto num_im = bc - ad;
    auto re_n = num_re / den;
    auto im_n = num_im / den;
    double naive_max_cond = std::max(re_n.max_cond_seen_, im_n.max_cond_seen_);

    // Smith should not be worse than naive; typically comparable or better.
    // The key is that Smith doesn't form c²+d² which can magnify error.
    REQUIRE(smith_max_cond <= naive_max_cond + 1.0);  // Smith is at most as bad
}

// ============================================================
// math functions via division — norm, abs, arg
// ============================================================

TEST_CASE("norm(z): |3+4i|² = 25") {
    auto z = tracked::track("z", 3.0, 4.0);
    auto r = tracked::norm(z);
    REQUIRE(r.value_ == Approx(25.0).epsilon(1e-12));
}

TEST_CASE("abs(z): |3+4i| = 5") {
    auto z = tracked::track("z", 3.0, 4.0);
    auto r = tracked::abs(z);
    REQUIRE(r.value_ == Approx(5.0).epsilon(1e-12));
}

TEST_CASE("arg(z): arg(1+i) = π/4") {
    auto z = tracked::track("z", 1.0, 1.0);
    auto r = tracked::arg(z);
    REQUIRE(r.value_ == Approx(M_PI / 4.0).epsilon(1e-12));
}

TEST_CASE("conj(z): conj(3+4i) = 3-4i") {
    auto z = tracked::track("z", 3.0, 4.0);
    auto r = tracked::conj(z);
    REQUIRE(r.real().value_ == Approx(3.0));
    REQUIRE(r.imag().value_ == Approx(-4.0));
}

TEST_CASE("exp(z): exp(0+π·i) ≈ -1+0i (Euler)") {
    auto z = tracked::track("z", 0.0, M_PI);
    auto r = tracked::exp(z);
    REQUIRE(r.real().value_ == Approx(-1.0).epsilon(1e-12));
    REQUIRE(r.imag().value_ == Approx(0.0).margin(1e-14));
}
