#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <tracked/complex.hpp>

#include <cmath>
#include <set>
#include <string>

using Catch::Approx;
using tracked::track;
using tracked::Complex;
using tracked::journal::records;
using tracked::journal::clear;

// ============================================================
// Construction and accessors
// ============================================================

TEST_CASE("Complex: default constructor gives (0,0)") {
    Complex<double> z;
    REQUIRE(z.real().value_ == 0.0);
    REQUIRE(z.imag().value_ == 0.0);
}

TEST_CASE("Complex: construct from raw scalars") {
    Complex<double> z(3.0, 4.0);
    REQUIRE(z.real().value_ == 3.0);
    REQUIRE(z.imag().value_ == 4.0);
}

TEST_CASE("Complex: construct from Tracked reals") {
    auto re = track("re", 1.5);
    auto im = track("im", 2.5);
    Complex<double> z(re, im);
    REQUIRE(z.real().value_ == 1.5);
    REQUIRE(z.imag().value_ == 2.5);
}

TEST_CASE("Complex: explicit construct from single Tracked (im=0)") {
    auto re = track("re", 7.0);
    Complex<double> z(re);
    REQUIRE(z.real().value_ == 7.0);
    REQUIRE(z.imag().value_ == 0.0);
}

TEST_CASE("Complex: track() factory — both components share provenance id") {
    auto z = tracked::track("z", 1.0, 2.0);
    REQUIRE(z.real().provenance_ == std::set<std::string>{"z"});
    REQUIRE(z.imag().provenance_ == std::set<std::string>{"z"});
}

TEST_CASE("Complex: track() factory with explicit im=0") {
    auto z = tracked::track("z", 5.0, 0.0);
    REQUIRE(z.real().value_ == 5.0);
    REQUIRE(z.imag().value_ == 0.0);
}

// ============================================================
// Arithmetic — values
// ============================================================

TEST_CASE("Complex: addition value") {
    auto a = tracked::track("a", 1.0, 2.0);
    auto b = tracked::track("b", 3.0, 4.0);
    auto r = a + b;
    REQUIRE(r.real().value_ == Approx(4.0));
    REQUIRE(r.imag().value_ == Approx(6.0));
}

TEST_CASE("Complex: subtraction value") {
    auto a = tracked::track("a", 5.0, 7.0);
    auto b = tracked::track("b", 2.0, 3.0);
    auto r = a - b;
    REQUIRE(r.real().value_ == Approx(3.0));
    REQUIRE(r.imag().value_ == Approx(4.0));
}

TEST_CASE("Complex: multiplication value (1+2i)*(3+4i) = -5+10i") {
    auto a = tracked::track("a", 1.0, 2.0);
    auto b = tracked::track("b", 3.0, 4.0);
    auto r = a * b;
    REQUIRE(r.real().value_ == Approx(-5.0));
    REQUIRE(r.imag().value_ == Approx(10.0));
}

TEST_CASE("Complex: division value (1+2i)/(3+4i) = 11/25 + 2/25·i") {
    auto a = tracked::track("a", 1.0, 2.0);
    auto b = tracked::track("b", 3.0, 4.0);
    auto r = a / b;
    REQUIRE(r.real().value_ == Approx(11.0 / 25.0).epsilon(1e-12));
    REQUIRE(r.imag().value_ == Approx(2.0  / 25.0).epsilon(1e-12));
}

TEST_CASE("Complex: unary negation") {
    auto z = tracked::track("z", 2.0, -3.0);
    auto r = -z;
    REQUIRE(r.real().value_ == Approx(-2.0));
    REQUIRE(r.imag().value_ == Approx(3.0));
}

// ============================================================
// Arithmetic — journal record counts
// ============================================================

TEST_CASE("Complex: addition emits 2 records (2 adds)") {
    clear();
    auto a = tracked::track("a", 1.0, 2.0);
    auto b = tracked::track("b", 3.0, 4.0);
    auto r = a + b;
    REQUIRE(records().size() == 2);
    REQUIRE(records()[0].op == "add");
    REQUIRE(records()[1].op == "add");
}

TEST_CASE("Complex: multiplication emits 6 records (4 muls, 1 sub, 1 add)") {
    clear();
    auto a = tracked::track("a", 1.0, 2.0);
    auto b = tracked::track("b", 3.0, 4.0);
    auto r = a * b;
    REQUIRE(records().size() == 6);
}

// ============================================================
// Compound assignment
// ============================================================

TEST_CASE("Complex: compound assignment +=") {
    auto a = tracked::track("a", 1.0, 2.0);
    auto b = tracked::track("b", 3.0, 4.0);
    a += b;
    REQUIRE(a.real().value_ == Approx(4.0));
    REQUIRE(a.imag().value_ == Approx(6.0));
}

TEST_CASE("Complex: compound assignment -=") {
    auto a = tracked::track("a", 5.0, 7.0);
    auto b = tracked::track("b", 2.0, 3.0);
    a -= b;
    REQUIRE(a.real().value_ == Approx(3.0));
    REQUIRE(a.imag().value_ == Approx(4.0));
}

TEST_CASE("Complex: compound assignment *= scalar Tracked") {
    auto z = tracked::track("z", 1.0, 2.0);
    auto s = track("s", 3.0);
    z *= s;
    REQUIRE(z.real().value_ == Approx(3.0));
    REQUIRE(z.imag().value_ == Approx(6.0));
}

TEST_CASE("Complex: compound assignment /= scalar Tracked") {
    auto z = tracked::track("z", 6.0, 4.0);
    auto s = track("s", 2.0);
    z /= s;
    REQUIRE(z.real().value_ == Approx(3.0));
    REQUIRE(z.imag().value_ == Approx(2.0));
}

// ============================================================
// Comparisons
// ============================================================

TEST_CASE("Complex: equality on matching values") {
    auto a = Complex<double>(track("x", 1.0), track("y", 2.0));
    auto b = Complex<double>(track("x", 1.0), track("y", 2.0));
    REQUIRE(a == b);
}

TEST_CASE("Complex: inequality on differing values") {
    auto a = Complex<double>(track("x", 1.0), track("y", 2.0));
    auto b = Complex<double>(track("x", 1.0), track("y", 3.0));
    REQUIRE(a != b);
}

// ============================================================
// Provenance union
// ============================================================

TEST_CASE("Complex: add — result provenance is union of both inputs") {
    auto a = tracked::track("a", 1.0, 0.0);
    auto b = tracked::track("b", 0.0, 1.0);
    auto r = a + b;
    auto prov_re = r.real().provenance_;
    auto prov_im = r.imag().provenance_;
    REQUIRE(prov_re == (std::set<std::string>{"a", "b"}));
    REQUIRE(prov_im == (std::set<std::string>{"a", "b"}));
}

TEST_CASE("Complex: mul — result provenance is union of both inputs") {
    auto a = tracked::track("a", 1.0, 2.0);
    auto b = tracked::track("b", 3.0, 4.0);
    auto r = a * b;
    REQUIRE(r.real().provenance_ == (std::set<std::string>{"a", "b"}));
    REQUIRE(r.imag().provenance_ == (std::set<std::string>{"a", "b"}));
}

// ============================================================
// Scalar overloads
// ============================================================

TEST_CASE("Complex: Complex + T scalar") {
    auto z = tracked::track("z", 1.0, 2.0);
    auto r = z + 3.0;
    REQUIRE(r.real().value_ == Approx(4.0));
    REQUIRE(r.imag().value_ == Approx(2.0));
}

TEST_CASE("Complex: T scalar + Complex") {
    auto z = tracked::track("z", 1.0, 2.0);
    auto r = 3.0 + z;
    REQUIRE(r.real().value_ == Approx(4.0));
    REQUIRE(r.imag().value_ == Approx(2.0));
}

TEST_CASE("Complex: Complex * T scalar") {
    auto z = tracked::track("z", 2.0, 3.0);
    auto r = z * 2.0;
    REQUIRE(r.real().value_ == Approx(4.0));
    REQUIRE(r.imag().value_ == Approx(6.0));
}

// ============================================================
// Opaque barrier
// ============================================================

TEST_CASE("opaque(Complex): emits two 'opaque' records") {
    clear();
    auto z = tracked::opaque<double>("my_fn", 1.0, 2.0);
    REQUIRE(records().size() == 2);
    REQUIRE(records()[0].op == "opaque");
    REQUIRE(records()[1].op == "opaque");
}

TEST_CASE("opaque(Complex): both records carry fn_name in 'in' field") {
    clear();
    auto z = tracked::opaque<double>("my_fn", 1.0, 2.0);
    REQUIRE(records()[0].in == std::vector<std::string>{"my_fn"});
    REQUIRE(records()[1].in == std::vector<std::string>{"my_fn"});
}

TEST_CASE("opaque(Complex): result carries fn_name as provenance") {
    auto z = tracked::opaque<double>("ext", 3.0, 4.0);
    REQUIRE(z.real().provenance_ == std::set<std::string>{"ext"});
    REQUIRE(z.imag().provenance_ == std::set<std::string>{"ext"});
}

TEST_CASE("opaque(Complex): values are passed through") {
    auto z = tracked::opaque<double>("fn", 5.0, 6.0);
    REQUIRE(z.real().value_ == 5.0);
    REQUIRE(z.imag().value_ == 6.0);
}
