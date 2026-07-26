#pragma once
// SOURCE_HASH: 2864b4f1a9e0fd7c6db16fb879185fbfd8675d07fc87f6fd969a31df4f845ab2
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Region: kokkosUtils.h:703
//   const TOutput lnomarg = TOutput(ql::kLog(ql::Constants<TScale>::_one() - arg2));
//
// After promotion TScale/TOutput become quad::ddfun::ddouble. The region
// references two ql:: symbols that must resolve for the extended scalar:
//   - ql::Constants<ddouble>::_one()  (Rule 5 / C5 — named-constant dispatch)
//   - ql::kLog(ddouble)               (Rule 2 / C7 — function-template overload
//                                      whose result stays on the chain, C9)

namespace ql {

// Rule 5 / C5: partial specialization of the library's Constants<T> template
// keyed on the extended scalar so ql::Constants<ddouble>::_one() resolves to
// the full-precision ddouble one, not a narrowed double.
//
// Rule R3 cascade for _one:
//   step 1: no vendored dd_one() factory.
//   step 2: pre-derived hex pair supplied by the caller — use verbatim.
// Source spelling: T(1.0)
template <>
struct Constants<quad::ddfun::ddouble> {
    // Rule 5 / R3 step 2: 1.0 as an exact ddouble (low word zero — a source
    // double literal carries only double precision, so this is faithful).
    static KOKKOS_INLINE_FUNCTION quad::ddfun::ddouble _one() {
        return quad::ddfun::make_dd(0x3ff0000000000000ULL, 0x0000000000000000ULL);
    }
};

// Rule 2 / C7 / C9: ddouble overload of ql::kLog. The result feeds directly
// into the region's TOutput temporary (a chain-internal value) so it MUST
// return ddouble — narrowing here would reintroduce the cancellation the
// chain promotion exists to remove.
KOKKOS_INLINE_FUNCTION
inline quad::ddfun::ddouble kLog(quad::ddfun::ddouble x) {
    return quad::ddfun::log(x);
}

} // namespace ql