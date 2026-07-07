#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>

#include <cmath>
#include <string>

using Catch::Approx;
using tracked::track;
using tracked::journal::records;
using tracked::journal::clear;

// Helper: run an op inside a cleared scope and return the single log record.
// Used for single-op condition-number checks.
#define SINGLE_OP(expr)  ([&]{ clear(); (expr); return records().back(); }())

// ============================================================
// Arithmetic — condition numbers
// ============================================================

TEST_CASE("add: condition number (|a|+|b|)/|a+b|") {
    auto a = track("a", 3.0);
    auto b = track("b", 1.0);
    clear();
    auto r = add(a, b, TRACKED_HERE);
    REQUIRE(r.value_ == 4.0);
    // cond = (3+1)/4 = 1.0
    REQUIRE(records().size() == 1);
    REQUIRE(records()[0].cond == Approx(1.0).epsilon(1e-12));
    REQUIRE(records()[0].op == "add");
    REQUIRE(!records()[0].at.empty());  // TRACKED_HERE captured a location
}

TEST_CASE("add: condition number amplified when result is small") {
    // a = 1e6, b = -1e6 + 1 → sum = 1, cond = 2e6/1 = 2e6
    auto a = track("a", 1e6);
    auto b = track("b", -1e6 + 1.0);
    auto rec = SINGLE_OP(add(a, b));
    REQUIRE(rec.val == Approx(1.0).epsilon(1e-9));
    REQUIRE(rec.cond == Approx(2e6 - 1.0).epsilon(1e-4));
}

TEST_CASE("sub: condition number matches add formula") {
    auto a = track("a", 5.0);
    auto b = track("b", 2.0);
    auto rec = SINGLE_OP(sub(a, b));
    // cond = (5+2)/3
    REQUIRE(rec.cond == Approx(7.0 / 3.0).epsilon(1e-12));
}

TEST_CASE("sub: catastrophic cancellation detected by cond > 1e8") {
    // a - b where a ≈ b: large operands, tiny difference
    // a = 1.0 + 1e-9, b = 1.0 → result = 1e-9, cond ≈ 2/1e-9 = 2e9
    auto a = track("a", 1.0 + 1e-9);
    auto b = track("b", 1.0);
    auto rec = SINGLE_OP(sub(a, b));
    REQUIRE(rec.cond > 1e8);
}

TEST_CASE("mul: condition number is 1") {
    auto a = track("a", 3.0);
    auto b = track("b", 7.0);
    auto rec = SINGLE_OP(mul(a, b));
    REQUIRE(rec.val == Approx(21.0));
    REQUIRE(rec.cond == Approx(1.0));
}

TEST_CASE("div: condition number is 1") {
    auto a = track("a", 6.0);
    auto b = track("b", 2.0);
    auto rec = SINGLE_OP(div(a, b));
    REQUIRE(rec.val == Approx(3.0));
    REQUIRE(rec.cond == Approx(1.0));
}

TEST_CASE("neg: condition number is 1") {
    auto a = track("a", -4.5);
    auto rec = SINGLE_OP(neg(a));
    REQUIRE(rec.val == Approx(4.5));
    REQUIRE(rec.cond == Approx(1.0));
}

// ============================================================
// Math ops — condition numbers
// ============================================================

TEST_CASE("sqrt: condition number is 0.5") {
    auto a = track("a", 9.0);
    auto rec = SINGLE_OP(tracked::sqrt(a, TRACKED_HERE));
    REQUIRE(rec.val == Approx(3.0));
    REQUIRE(rec.cond == Approx(0.5));
    REQUIRE(rec.op == "sqrt");
}

TEST_CASE("exp: condition number is |x|") {
    {
        auto a = track("a", 2.0);
        auto rec = SINGLE_OP(tracked::exp(a));
        REQUIRE(rec.cond == Approx(2.0));
    }
    {
        auto a = track("a", -3.0);
        auto rec = SINGLE_OP(tracked::exp(a));
        REQUIRE(rec.cond == Approx(3.0));
    }
    {
        auto a = track("a", 0.0);
        auto rec = SINGLE_OP(tracked::exp(a));
        REQUIRE(rec.cond == Approx(0.0).margin(1e-30));
    }
}

TEST_CASE("log: condition number is 1/|log(x)|") {
    // log(e^2) = 2 → cond = 1/2
    auto a = track("a", std::exp(2.0));
    auto rec = SINGLE_OP(tracked::log(a));
    REQUIRE(rec.val == Approx(2.0).epsilon(1e-12));
    REQUIRE(rec.cond == Approx(0.5).epsilon(1e-10));
    REQUIRE(rec.op == "log");
}

TEST_CASE("log: condition number large near x=1") {
    // log(1 + 1e-15) ≈ 1e-15, cond = 1/1e-15 = 1e15 >> 1
    auto a = track("a", 1.0 + 1e-15);
    auto rec = SINGLE_OP(tracked::log(a));
    REQUIRE(rec.cond > 1e10);
}

TEST_CASE("abs: condition number is 1") {
    auto a = track("a", -7.5);
    auto rec = SINGLE_OP(tracked::abs(a));
    REQUIRE(rec.val == Approx(7.5));
    REQUIRE(rec.cond == Approx(1.0));
}

// ============================================================
// Error bound propagation
// ============================================================

TEST_CASE("fresh Tracked<double> has rel_err == unit_roundoff") {
    auto a = track("a", 1.0);
    REQUIRE(a.rel_err_bound_ == Approx(tracked::unit_roundoff<double>()));
}

TEST_CASE("mul/div: error propagates without amplification") {
    auto a = track("a", 3.0);
    auto b = track("b", 4.0);
    double u = tracked::unit_roundoff<double>();
    auto r = a * b;
    // cond=1, new_err = 1*(u + u) = 2u
    REQUIRE(r.rel_err_bound_ == Approx(2.0 * u).epsilon(1e-6));
}

TEST_CASE("sub with cancellation: error bound grows with condition") {
    // a = large, b = large - small: result is small, cond >> 1
    auto a = track("a", 1.0 + 1e-9);
    auto b = track("b", 1.0);
    auto r = a - b;
    // rel_err_bound >> unit_roundoff (~1.1e-16); actual ≈ cond*2u ≈ 2e9*2.2e-16 ≈ 4.4e-7
    REQUIRE(r.rel_err_bound_ > 1e-8);
}

TEST_CASE("max_cond_seen accumulates across a chain of ops") {
    auto a = track("a", 1.0 + 1e-9);
    auto b = track("b", 1.0);
    auto c = track("c", 2.0);
    auto r1 = a - b;          // catastrophic cancellation, high cond
    auto r2 = r1 * c;         // mul cond=1, but carries forward max_cond
    REQUIRE(r2.max_cond_seen_ == r1.max_cond_seen_);
    REQUIRE(r2.max_cond_seen_ > 1e8);
}

// ============================================================
// Provenance
// ============================================================

TEST_CASE("track: sets id and prov_vars to given name") {
    auto a = track("alpha", 1.0);
    REQUIRE(a.id_ == "alpha");
    REQUIRE(a.prov_vars_ == std::set<std::string>{"alpha"});
    REQUIRE(a.prov_consts_.empty());
}

TEST_CASE("binary op: prov_vars is union of inputs") {
    auto a = track("a", 1.0);
    auto b = track("b", 2.0);
    auto r = a + b;
    REQUIRE(r.prov_vars_ == (std::set<std::string>{"a", "b"}));
}

TEST_CASE("chain: prov_vars accumulates transitively") {
    auto a = track("a", 1.0);
    auto b = track("b", 2.0);
    auto c = track("c", 3.0);
    auto r = (a + b) * c;
    REQUIRE(r.prov_vars_ == (std::set<std::string>{"a", "b", "c"}));
}

TEST_CASE("log record prov_vars field matches result provenance") {
    auto a = track("x", 2.0);
    auto b = track("y", 3.0);
    auto rec = SINGLE_OP(add(a, b, TRACKED_HERE));
    REQUIRE(std::set<std::string>(rec.prov_vars.begin(), rec.prov_vars.end())
            == (std::set<std::string>{"x", "y"}));
    REQUIRE(rec.prov_consts.empty());
}

// ============================================================
// Source location (M3)
// ============================================================

TEST_CASE("TRACKED_HERE captures non-empty file:function:line") {
    auto a = track("a", 1.0);
    auto b = track("b", 2.0);
    clear();
    auto r = add(a, b, TRACKED_HERE);  // line is captured here
    REQUIRE(!records()[0].at.empty());
    // at contains this source file name
    REQUIRE(records()[0].at.find("test_basic_ops") != std::string::npos);
}

TEST_CASE("operator+ uses empty source location") {
    auto a = track("a", 1.0);
    auto b = track("b", 2.0);
    auto rec = SINGLE_OP(a + b);
    REQUIRE(rec.at.empty());
}

// ============================================================
// Opaque barrier
// ============================================================

TEST_CASE("opaque: emits barrier record with op=opaque") {
    clear();
    auto r = tracked::opaque("external_fn", 3.14);
    REQUIRE(records().size() == 1);
    REQUIRE(records()[0].op == "opaque");
    REQUIRE(records()[0].in == std::vector<std::string>{"external_fn"});
}

TEST_CASE("opaque: fn_name is a boundary marker in 'in', not provenance") {
    // v0.3: the fn_name marker is neither a source variable nor a named
    // constant, so it stays out of prov_vars/prov_consts (retiring v0.2, which
    // folded it into provenance). It survives as the leading entry of `in`.
    clear();
    auto r = tracked::opaque("ext", 1.0);
    REQUIRE(records().back().in == std::vector<std::string>{"ext"});
    REQUIRE(r.prov_vars_.empty());
    REQUIRE(r.prov_consts_.empty());
}

TEST_CASE("opaque_at with TRACKED_HERE: location captured") {
    clear();
    auto r = tracked::opaque_at("ext_fn", 2.0, TRACKED_HERE);
    REQUIRE(!records()[0].at.empty());
}

TEST_CASE("opaque: cond is 1 (pass-through, not 0)") {
    clear();
    auto r = tracked::opaque("fn", 5.0);
    REQUIRE(records()[0].cond == Approx(1.0));
    REQUIRE(r.max_cond_seen_ == Approx(1.0));
}

TEST_CASE("opaque with no tracked inputs: rel_err = u") {
    clear();
    auto r = tracked::opaque("fn", 5.0);
    REQUIRE(records()[0].rel_err == Approx(tracked::unit_roundoff<double>()));
    REQUIRE(r.rel_err_bound_ == Approx(tracked::unit_roundoff<double>()));
}

TEST_CASE("opaque with tracked inputs: rel_err propagates from max input") {
    clear();
    // Build inputs with non-trivial accumulated error.
    auto a = track("a", 1e8);
    auto b = track("b", 1.0);
    auto diff = a - b;   // catastrophic cancellation candidate? no, well-cond here
    // Force a high-error tracked value via cancellation:
    auto x = track("x", 1.0);
    auto y = track("y", 1.0 - 1e-12);
    auto cancel = x - y;   // big rel_err
    REQUIRE(cancel.rel_err_bound_ > 1e-5);

    clear();
    auto r = tracked::opaque("fn", 42.0, cancel);
    REQUIRE(records()[0].cond == Approx(1.0));
    // rel_err = cond * (max_input_rel_err + u) ~= cancel.rel_err + u
    REQUIRE(records()[0].rel_err == Approx(cancel.rel_err_bound_ + tracked::unit_roundoff<double>()));
}

TEST_CASE("opaque with tracked inputs: prov_vars is union(inputs); fn_name leads 'in'") {
    clear();
    auto a = track("a", 1.0);
    auto b = track("b", 2.0);
    auto r = tracked::opaque("my_fn", 99.0, a, b);
    // Provenance is the forwarded inputs only — fn_name is a marker, not a var.
    REQUIRE(r.prov_vars_ == (std::set<std::string>{"a", "b"}));
    REQUIRE(r.prov_consts_.empty());
    // `in` = [fn_name, input_ids...] so the value graph stays traversable.
    REQUIRE(records().back().in == (std::vector<std::string>{"my_fn", "a", "b"}));
}

// ============================================================
// Operator overloads — functional correctness
// ============================================================

TEST_CASE("operators produce correct values") {
    auto a = track("a", 6.0);
    auto b = track("b", 2.0);
    REQUIRE((a + b).value_ == Approx(8.0));
    REQUIRE((a - b).value_ == Approx(4.0));
    REQUIRE((a * b).value_ == Approx(12.0));
    REQUIRE((a / b).value_ == Approx(3.0));
    REQUIRE((-a).value_   == Approx(-6.0));
}

TEST_CASE("compound assignment operators work") {
    auto a = track("a", 10.0);
    auto b = track("b", 3.0);
    a += b; REQUIRE(a.value_ == Approx(13.0));
    a -= b; REQUIRE(a.value_ == Approx(10.0));
    a *= b; REQUIRE(a.value_ == Approx(30.0));
    a /= b; REQUIRE(a.value_ == Approx(10.0));
}

TEST_CASE("comparison operators use underlying value") {
    auto a = track("a", 5.0);
    auto b = track("b", 3.0);
    REQUIRE(a > b);
    REQUIRE(b < a);
    REQUIRE(a != b);
    REQUIRE(a == track("a2", 5.0));
}
