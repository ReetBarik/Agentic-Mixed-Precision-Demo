#pragma once
// SOURCE_HASH: da9bb06a31fc72c7a54278481266cbac7f1217772c4bfa15813d7542f059eb62
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Rule 5 / R3 step 2: ql::Constants<TMass>::_half() named-constant wrapper.
// Region uses ql::Constants<TMass>::_half() with TMass promoted to
// quad::ddfun::ddouble; supply a partial specialization keyed on the extended
// scalar (C5/C7) so this call resolves to a ddouble-returning factory rather
// than the library primary that would narrow to double.
// Source RHS was T(0.5) — a source double literal → make_dd(<bits>, 0x0)
// with a zero low word (correct per R3 step 3a; pre-derived).
namespace ql {
template <>
struct Constants<quad::ddfun::ddouble> {
    // Rule 2: floating-point named constant participating in region arithmetic
    // -> return quad::ddfun::ddouble.
    static inline quad::ddfun::ddouble _half() {
        // source: T(0.5)
        return quad::ddfun::make_dd(0x3fe0000000000000ULL, 0x0000000000000000ULL);
    }
};
} // namespace ql