#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/complex.hpp>

#include <algorithm>
#include <regex>
#include <set>
#include <string>
#include <vector>

using Catch::Approx;
using tracked::track;
using tracked::literal;
using tracked::Tracked;
using tracked::journal::records;
using tracked::journal::clear;

// ---- helpers ----------------------------------------------------------------

static bool starts_with(const std::string& s, const std::string& pre) {
    return s.size() >= pre.size() && s.compare(0, pre.size(), pre) == 0;
}

static bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// A fixed call site so the (file,line,op) callsite counter is shared across
// invocations — used by the uniqueness test.
static Tracked<double> lit_here(double v) {
    return literal(v, TRACKED_HERE);
}

// ============================================================
// 1. literal() gets an auto id and no provenance
// ============================================================

TEST_CASE("literal_gets_auto_id_and_empty_provenance") {
    auto x = literal(3.14);
    REQUIRE(!x.id().empty());
    REQUIRE(starts_with(x.id(), "_lit@"));
    REQUIRE(x.prov_vars().empty());
    REQUIRE(x.prov_consts().empty());
}

// ============================================================
// 2. literal() with an explicit source location captures it
// ============================================================

TEST_CASE("literal_with_source_location_captures_it") {
    auto x = literal(1.0, TRACKED_HERE);
    // id contains this file and matches _lit@<file>:<line>#<n>
    REQUIRE(x.id().find(__FILE__) != std::string::npos);
    std::regex shape(R"(^_lit@.*:[0-9]+#[0-9]+$)");
    REQUIRE(std::regex_search(x.id(), shape));
}

// ============================================================
// 3. ids are unique per call at the same site (counter suffix)
// ============================================================

TEST_CASE("literal_ids_are_unique_per_call") {
    clear();
    auto a = lit_here(1.0);   // #1
    auto b = lit_here(2.0);   // #2 (same file:line:op)
    auto c = lit_here(3.0);   // #3
    REQUIRE(ends_with(a.id(), "#1"));
    REQUIRE(ends_with(b.id(), "#2"));
    REQUIRE(ends_with(c.id(), "#3"));
    // all three distinct
    REQUIRE(a.id() != b.id());
    REQUIRE(b.id() != c.id());
    REQUIRE(a.id() != c.id());
}

// ============================================================
// 4. literal() respects the lexical scope stack
// ============================================================

TEST_CASE("literal_respects_scope") {
    clear();
    tracked::scope s("s=42");
    auto x = literal(1.0, TRACKED_HERE);
    REQUIRE(ends_with(x.id(), "@s=42"));
}

// ============================================================
// 5. a literal operand produces a fully-traceable record
// ============================================================

TEST_CASE("literal_used_as_operand_produces_traceable_record") {
    clear();
    auto x = track("m", 5.0);
    auto k = literal(2.0);
    auto r = x - k;

    const auto& rec = records().back();
    REQUIRE(rec.op == "sub");
    REQUIRE(rec.in == (std::vector<std::string>{"m", k.id()}));
    // no empty string anywhere in `in`
    for (const auto& inp : rec.in) REQUIRE_FALSE(inp.empty());
}

// ============================================================
// 6. trace_sources excludes literals (not source variables)
// ============================================================

TEST_CASE("trace_sources_excludes_literals") {
    clear();
    auto y = literal(2.0) * literal(3.0);
    auto z = track("m", 1.0) + y;
    REQUIRE(tracked::journal::trace_sources(z.id()) == std::set<std::string>{"m"});
}

// ============================================================
// 7. trace_ancestors includes literals (valid graph nodes)
// ============================================================

TEST_CASE("trace_ancestors_includes_literals") {
    clear();
    auto a = literal(2.0);
    auto b = literal(3.0);
    auto y = a * b;
    auto z = track("m", 1.0) + y;

    auto order = tracked::journal::trace_ancestors(z.id());
    REQUIRE(std::find(order.begin(), order.end(), a.id()) != order.end());
    REQUIRE(std::find(order.begin(), order.end(), b.id()) != order.end());
}

// ============================================================
// 8. regression: complex ops never emit empty `in` ids
// ============================================================

TEST_CASE("complex_ops_produce_no_empty_in_ids") {
    clear();
    using Cplx = tracked::Complex<double>;
    auto z1 = Cplx(track("re1", 1.0), track("im1", 2.0));
    auto z2 = z1 * Cplx(3.0);              // scalar-to-complex promotion
    auto z3 = z2 + tracked::literal(0.5);  // explicit literal path
    (void)z3;
    for (const auto& rec : records()) {
        for (const auto& inp : rec.in) {
            REQUIRE_FALSE(inp.empty());
        }
    }
}
