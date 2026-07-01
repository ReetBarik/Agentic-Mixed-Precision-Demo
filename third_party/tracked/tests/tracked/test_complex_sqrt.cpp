#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <tracked/complex.hpp>

#include <cmath>

using Catch::Approx;
using tracked::Complex;
using tracked::track;

// ============================================================
// sqrt — values
// ============================================================

TEST_CASE("sqrt(z): sqrt(4+0i) = 2") {
    auto z = tracked::track("z", 4.0, 0.0);
    auto r = tracked::sqrt(z);
    REQUIRE(r.real().value_ == Approx(2.0).epsilon(1e-12));
    REQUIRE(r.imag().value_ == Approx(0.0).margin(1e-15));
}

TEST_CASE("sqrt(z): sqrt(-1+0i) = i") {
    auto z = tracked::track("z", -1.0, 0.0);
    auto r = tracked::sqrt(z);
    REQUIRE(r.real().value_ == Approx(0.0).margin(1e-15));
    REQUIRE(r.imag().value_ == Approx(1.0).epsilon(1e-12));
}

TEST_CASE("sqrt(z): sqrt(3+4i) = 2+i") {
    // |3+4i| = 5, sqrt = (2+i) since (2+i)² = 4-1+4i = 3+4i
    auto z = tracked::track("z", 3.0, 4.0);
    auto r = tracked::sqrt(z);
    REQUIRE(r.real().value_ == Approx(2.0).epsilon(1e-12));
    REQUIRE(r.imag().value_ == Approx(1.0).epsilon(1e-12));
}

// ============================================================
// sqrt — stable form near negative real axis
// ============================================================

TEST_CASE("sqrt(z): stable formula keeps max_cond low near negative real axis") {
    // z = -1 + 1e-10·i  — the case where arg/2 ≡ sin/cos form is catastrophic
    auto z = tracked::track("z", -1.0, 1e-10);
    auto r = tracked::sqrt(z);

    // Stable result: re ≈ 1e-10/(2·1) ≈ 5e-11, im ≈ 1
    REQUIRE(r.imag().value_ == Approx(1.0).epsilon(1e-6));
    REQUIRE(r.real().value_ == Approx(5e-11).epsilon(1e-3));

    // max_cond_seen should stay reasonable (< 1e10)
    REQUIRE(r.real().max_cond_seen_ < 1e10);
    REQUIRE(r.imag().max_cond_seen_ < 1e10);
}

TEST_CASE("sqrt(z): naive arg/2 form produces higher max_cond than stable form") {
    // z = -1 + 1e-10·i
    // Naive: w = sqrt(|z|)·(cos(arg/2) + i·sin(arg/2))
    // Near negative real axis, arg ≈ π, arg/2 ≈ π/2, cos(π/2) ≈ 0 → catastrophic.

    auto zy = tracked::track("zy", -1.0, 1e-10);
    auto zx = tracked::track("zx", -1.0, 1e-10);

    // Stable form
    auto stable = tracked::sqrt(zy);
    double stable_max_cond = std::max(stable.real().max_cond_seen_,
                                      stable.imag().max_cond_seen_);

    // Naive form — inline, not exposed in the API
    auto r_n   = tracked::abs(zx);
    auto a_n   = tracked::arg(zx);
    auto half  = tracked::track("half", 0.5);
    auto ha    = half * a_n;
    auto re_n  = tracked::mul(r_n, tracked::cos(ha));
    auto im_n  = tracked::mul(r_n, tracked::sin(ha));
    double naive_max_cond = std::max(re_n.max_cond_seen_, im_n.max_cond_seen_);

    // The naive path triggers cos(arg/2) ≈ cos(π/2) → high cond.
    // Assert the gap is visible.
    REQUIRE(naive_max_cond > stable_max_cond * 10.0);
}

// ============================================================
// sqrt — negative imaginary part
// ============================================================

TEST_CASE("sqrt(z): sqrt(-1 - 1e-10·i) has negative imaginary part") {
    auto z = tracked::track("z", -1.0, -1e-10);
    auto r = tracked::sqrt(z);
    // sign(im) is negative, so im_out = -w < 0
    REQUIRE(r.imag().value_ < 0.0);
    REQUIRE(r.real().value_ > 0.0);
}

// ============================================================
// sqrt — TRACKED_HERE propagation
// ============================================================

TEST_CASE("sqrt(z): TRACKED_HERE attributes sub-ops to the call site") {
    tracked::journal::clear();
    auto z = tracked::track("z", 3.0, 4.0);
    tracked::sqrt(z, TRACKED_HERE);
    // All records should carry the same non-empty location
    const auto& recs = tracked::journal::records();
    REQUIRE(!recs.empty());
    std::string loc0 = recs[0].at;
    REQUIRE(!loc0.empty());
    for (const auto& rec : recs) {
        REQUIRE(rec.at == loc0);
    }
}
