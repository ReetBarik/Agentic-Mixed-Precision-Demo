#pragma once
// SOURCE_HASH: 4c24bd3e2b330ec2117d525aef9702ed3a9a59c4656366303a3b1eae12d9e6ad
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Rule 5 / R3 step 3a: ql::Constants<TMass>::_two() — source RHS T(2.0), a plain
// double literal. Faithful ddouble value has a zero low word (correct, not a
// truncation). We provide a partial specialization of the library's
// Constants<T> class template keyed on the extended scalar so the region's
// call `ql::Constants<TMass>::_two()` with TMass = quad::ddfun::ddouble
// resolves to the ddouble-returning member (C5/C7).
//
// Rejection log for R3 cascade:
//   Step 1: no vendored dd_two() factory exists.
//   Step 2: hex pair supplied by "Source-derivable constants" table.
//   Step 3: applies — source literal T(2.0); using the given hex pair.
namespace ql {
template <class T> struct Constants; // primary declared by the app; forward-decl is harmless here
template <>
struct Constants<quad::ddfun::ddouble> {
    // Rule 5 / C5: ddouble specialization of the library's constant dispatcher.
    static inline quad::ddfun::ddouble _two() {
        // R3 step 3a: T(2.0) → exact double, low word zero.
        return quad::ddfun::make_dd(0x4000000000000000ULL, 0x0000000000000000ULL);
    }
};
} // namespace ql