#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>

#include <cmath>

using Catch::Approx;
using tracked::track;
using tracked::journal::records;
using tracked::journal::clear;

#define SINGLE_OP(expr)  ([&]{ clear(); (expr); return records().back(); }())

static constexpr double PI = M_PI;

// ============================================================
// sin
// ============================================================

TEST_CASE("sin: correct value") {
    auto x = track("x", 0.1);
    auto r = tracked::sin(x);
    REQUIRE(r.value_ == Approx(std::sin(0.1)).epsilon(1e-15));
}

TEST_CASE("sin: condition number near 0.1 is |x·cot(x)| ≈ 1") {
    auto x = track("x", 0.1);
    auto rec = SINGLE_OP(tracked::sin(x));
    // cond = |0.1 * cos(0.1) / sin(0.1)|
    double expected = std::abs(0.1 * std::cos(0.1) / std::sin(0.1));
    REQUIRE(rec.cond == Approx(expected).epsilon(1e-10));
}

TEST_CASE("sin: condition number near π is capped at 1/u") {
    auto x = track("x", PI);
    auto rec = SINGLE_OP(tracked::sin(x));
    // |sin(π)| is tiny relative to |x|·u — should trigger the cap
    double u = tracked::unit_roundoff<double>();
    REQUIRE(rec.cond == Approx(1.0 / u).epsilon(1e-6));
}

TEST_CASE("sin: op name is 'sin'") {
    auto x = track("x", 1.0);
    auto rec = SINGLE_OP(tracked::sin(x));
    REQUIRE(rec.op == "sin");
}

TEST_CASE("sin: TRACKED_HERE captures location") {
    clear();
    auto x = track("x", 0.5);
    tracked::sin(x, TRACKED_HERE);
    REQUIRE(!records().back().at.empty());
}

// ============================================================
// cos
// ============================================================

TEST_CASE("cos: correct value") {
    auto x = track("x", 0.1);
    auto r = tracked::cos(x);
    REQUIRE(r.value_ == Approx(std::cos(0.1)).epsilon(1e-15));
}

TEST_CASE("cos: condition number at x=0 is 0 (sin(0)=0)") {
    auto x = track("x", 0.0);
    auto rec = SINGLE_OP(tracked::cos(x));
    // cond = |x·sin(x)/cos(x)| = 0·0/1 = 0
    REQUIRE(rec.cond == Approx(0.0).margin(1e-30));
}

TEST_CASE("cos: condition number near π/2 is capped at 1/u") {
    auto x = track("x", PI / 2.0);
    auto rec = SINGLE_OP(tracked::cos(x));
    // |cos(π/2)| is tiny relative to |x|·u — should trigger the cap
    double u = tracked::unit_roundoff<double>();
    REQUIRE(rec.cond == Approx(1.0 / u).epsilon(1e-6));
}

TEST_CASE("cos: op name is 'cos'") {
    auto x = track("x", 1.0);
    auto rec = SINGLE_OP(tracked::cos(x));
    REQUIRE(rec.op == "cos");
}

// ============================================================
// atan2
// ============================================================

TEST_CASE("atan2: correct value") {
    auto y = track("y", 1.0);
    auto x = track("x", 1.0);
    auto r = tracked::atan2(y, x);
    REQUIRE(r.value_ == Approx(std::atan2(1.0, 1.0)).epsilon(1e-15));
}

TEST_CASE("atan2: well-conditioned away from axes") {
    // atan2(1, 1) = π/4, none of the pathological regions
    auto y = track("y", 1.0);
    auto x = track("x", 1.0);
    auto rec = SINGLE_OP(tracked::atan2(y, x));
    // cond = 2|xy| / ((x²+y²)·|atan2|) = 2 / (2·π/4) = 4/π ≈ 1.27
    double expected = 2.0 * std::abs(1.0 * 1.0)
                    / ((1.0 + 1.0) * std::abs(std::atan2(1.0, 1.0)));
    REQUIRE(rec.cond == Approx(expected).epsilon(1e-10));
}

TEST_CASE("atan2: cond large near positive real axis (atan2 → 0)") {
    // atan2(1e-20, 1) ≈ 1e-20, cond blows up
    auto y = track("y", 1e-20);
    auto x = track("x", 1.0);
    auto rec = SINGLE_OP(tracked::atan2(y, x));
    REQUIRE(rec.cond > 1e10);
}

TEST_CASE("atan2: cond capped at 1/u when atan2 < u") {
    // Use values where |atan2| is extremely small
    auto y = track("y", 1e-200);
    auto x = track("x", 1.0);
    auto rec = SINGLE_OP(tracked::atan2(y, x));
    double u = tracked::unit_roundoff<double>();
    REQUIRE(rec.cond == Approx(1.0 / u).epsilon(1e-6));
}

TEST_CASE("atan2: prov_vars is union of y and x") {
    auto y = track("y", 1.0);
    auto x = track("x", 1.0);
    auto r = tracked::atan2(y, x);
    REQUIRE(r.prov_vars_ == (std::set<std::string>{"x", "y"}));
}

TEST_CASE("atan2: in field carries both input primary ids") {
    auto y = track("y", 1.0);
    auto x = track("x", 1.0);
    auto rec = SINGLE_OP(tracked::atan2(y, x));
    REQUIRE(rec.in.size() == 2);
    REQUIRE(rec.in[0] == "y");
    REQUIRE(rec.in[1] == "x");
}

TEST_CASE("atan2: op name is 'atan2'") {
    auto y = track("y", 0.5);
    auto x = track("x", 0.5);
    auto rec = SINGLE_OP(tracked::atan2(y, x));
    REQUIRE(rec.op == "atan2");
}

TEST_CASE("atan2: max_cond_seen propagates from inputs") {
    auto y = track("y", 1.0 + 1e-9);
    auto b = track("b", 1.0);
    auto diff = y - b;  // catastrophic cancellation, high max_cond_seen
    auto x = track("x", 1.0);
    auto r = tracked::atan2(diff, x);
    REQUIRE(r.max_cond_seen_ > 1e8);
}
