#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <tracked/complex.hpp>

#include <cmath>

using Catch::Approx;
using tracked::track;

static constexpr double PI = M_PI;

// ============================================================
// log — values
// ============================================================

TEST_CASE("log(z): log(1+0i) = 0") {
    auto z = tracked::track("z", 1.0, 0.0);
    auto r = tracked::log(z);
    REQUIRE(r.real().value_ == Approx(0.0).margin(1e-15));
    REQUIRE(r.imag().value_ == Approx(0.0).margin(1e-15));
}

TEST_CASE("log(z): log(-1+0i) = i·π") {
    auto z = tracked::track("z", -1.0, 0.0);
    auto r = tracked::log(z);
    REQUIRE(r.real().value_ == Approx(0.0).margin(1e-14));
    REQUIRE(r.imag().value_ == Approx(PI).epsilon(1e-12));
}

TEST_CASE("log(z): log(e+0i) = 1") {
    auto z = tracked::track("z", std::exp(1.0), 0.0);
    auto r = tracked::log(z);
    REQUIRE(r.real().value_ == Approx(1.0).epsilon(1e-12));
    REQUIRE(r.imag().value_ == Approx(0.0).margin(1e-15));
}

TEST_CASE("log(z): log(0+i) = i·π/2") {
    auto z = tracked::track("z", 0.0, 1.0);
    auto r = tracked::log(z);
    REQUIRE(r.real().value_ == Approx(0.0).margin(1e-14));
    REQUIRE(r.imag().value_ == Approx(PI / 2.0).epsilon(1e-12));
}

// ============================================================
// log — pathology 1: |z| ≈ 1 amplifies cond from log factor
// ============================================================

TEST_CASE("log(z): |z|≈1 causes large max_cond from real log component") {
    // z = 1 + 1e-8·i, |z| ≈ 1, log|z| ≈ 5e-17 → log cond blows up
    auto z = tracked::track("z", 1.0, 1e-8);
    auto r = tracked::log(z);
    // The real part (log|z|) should have seen a very high condition
    REQUIRE(r.real().max_cond_seen_ > 1e6);
}

// ============================================================
// log — pathology 2: near negative real axis, arg component dominates
// ============================================================

TEST_CASE("log(z): near negative real axis, arg cond dominates") {
    // z = -1 + 1e-15·i → atan2(1e-15, -1) ≈ π - 1e-15, well away from 0
    // so arg cond is moderate; but |z| ≈ 1 so log cond is huge.
    // Both pathologies: we just confirm max_cond is large.
    auto z = tracked::track("z", -1.0, 1e-15);
    auto r = tracked::log(z);
    REQUIRE(r.real().max_cond_seen_ > 1e10);
}

TEST_CASE("log(z): near positive real axis (small im), atan2 cond large") {
    // z = 10 + 1e-20·i → atan2(1e-20, 10) ≈ 1e-21 → atan2 cond → 1/u
    auto z = tracked::track("z", 10.0, 1e-20);
    auto r = tracked::log(z);
    // The imaginary part (arg) should carry the large atan2 condition
    REQUIRE(r.imag().max_cond_seen_ > 1e10);
}
