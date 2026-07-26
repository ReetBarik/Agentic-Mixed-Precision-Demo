#pragma once
// SOURCE_HASH: da9bb06a31fc72c7a54278481266cbac7f1217772c4bfa15813d7542f059eb62
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Rule C5/C7: Partial specialization of the library-owned class template
// ql::Constants<T> keyed on the extended scalar quad::ddfun::ddouble, so
// that ql::Constants<quad::ddfun::ddouble>::_half() resolves to a ddouble
// (not the primary's double), keeping the region's arithmetic extended
// end-to-end (Rule 2 / C9 chain-internal contract).
namespace ql {
template <>
struct Constants<quad::ddfun::ddouble> {
    // Rule 5 + Rule R3 step 3a: source RHS is T(0.5), a plain double literal,
    // so the faithful ddouble carries a zero low word. Materialized via the
    // pre-derived hex pair rather than a decimal literal (mandatory).
    // Source name preserved: ql::Constants<T>::_half
    static KOKKOS_INLINE_FUNCTION quad::ddfun::ddouble _half() {
        return quad::ddfun::make_dd(0x3fe0000000000000ULL,
                                    0x0000000000000000ULL);
    }
};
} // namespace ql