#pragma once
// SOURCE_HASH: 88d25afce75d3cf13e32b1c0dd3905868412771237493c66e0b71992c6de1850
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Rule 5 / R3 step 2: ql::Constants<TOutput>::_two() specialized for ddouble.
// Source RHS T(2.0) -> exact bit pair make_dd(0x4000...ULL, 0x0ULL).
// C5/C7: partial specialization of the library's Constants<T> class template,
// keyed on the extended scalar so it wins over the primary template.
// C1: we do NOT #include the app header; the primary template is already
// visible at the shim's include site inside the region's TU.
namespace ql {
template <>
struct Constants<quad::ddfun::ddouble> {
    // Rule 5 / R3: _two (source RHS T(2.0))
    static KOKKOS_INLINE_FUNCTION quad::ddfun::ddouble _two() {
        return quad::ddfun::make_dd(0x4000000000000000ULL, 0x0000000000000000ULL);
    }
};
} // namespace ql