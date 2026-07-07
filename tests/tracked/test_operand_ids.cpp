#include <catch2/catch_test_macros.hpp>

#include <tracked/tracked.hpp>

#include <string>
#include <vector>

using tracked::track;
using tracked::sub;
using tracked::mul;
using tracked::Tracked;
using tracked::journal::records;
using tracked::journal::clear;

static bool starts_with(const std::string& s, const std::string& pre) {
    return s.size() >= pre.size() && s.compare(0, pre.size(), pre) == 0;
}

// ============================================================
// The one-arg Tracked(T) ctor must synthesize a real literal id.
// Regression: a value promoted through Tracked<T>(v) used to carry an empty
// id_, which then surfaced as an empty operand slot in every op it fed.
// ============================================================

TEST_CASE("tracked_value_ctor_synthesizes_literal_id") {
    Tracked<double> x(3.14);
    REQUIRE_FALSE(x.id().empty());
    REQUIRE(starts_with(x.id(), "_lit@"));
}

// ============================================================
// An op with a Tracked(T)-constructed operand records real ids on both sides.
// ============================================================

TEST_CASE("op_with_bare_ctor_operand_has_no_empty_ids") {
    clear();
    auto a = track("m", 5.0);
    auto r = mul(a, Tracked<double>(2.0));

    const auto& rec = records().back();
    REQUIRE(rec.op == "mul");
    REQUIRE(rec.in.size() == 2);
    REQUIRE(rec.in[0] == "m");
    REQUIRE_FALSE(rec.in[1].empty());
    REQUIRE(starts_with(rec.in[1], "_lit@"));
    for (const auto& inp : rec.in) REQUIRE_FALSE(inp.empty());
}

// ============================================================
// sub(a, a) records a.id() on both operand slots — not "".
// ============================================================

TEST_CASE("sub_of_identical_operand_records_id_twice") {
    clear();
    auto a = track("q", 7.0);
    auto r = sub(a, a);

    const auto& rec = records().back();
    REQUIRE(rec.op == "sub");
    REQUIRE(rec.in == (std::vector<std::string>{"q", "q"}));
    REQUIRE(rec.in[0] == a.id());
    REQUIRE(rec.in[1] == a.id());
    for (const auto& inp : rec.in) REQUIRE_FALSE(inp.empty());
}
