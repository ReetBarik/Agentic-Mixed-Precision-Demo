#pragma once
// SOURCE_HASH: dc98d3a6808acc7e51ed85272f5759ee04843f9409c4430b6dadf89a8ac946ab
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Rule 5 / R3 step 3a: ql::Constants<T>::_one() — source RHS T(1.0), a plain
// double literal, so its faithful ddouble value has hi=0x3ff0000000000000,
// lo=0x0. Specialize the library's Constants class template on the extended
// scalar so the region's qualified call ql::Constants<TMass>::_one() resolves
// here and returns ddouble (C5/C7). The specialization attaches by name to
// ql::Constants already declared at the include site (C1: no app header
// re-include). Chain-internal return stays ddouble (C9).
namespace ql {
template <class T> struct Constants; // primary template already declared by the app at the include site

template <>
struct Constants<quad::ddfun::ddouble> {
    // Rule 5 / R3 step 3a: _one derived from source literal T(1.0).
    // Annotated KOKKOS_INLINE_FUNCTION per Rule 8/C4 to match kokkosUtils.h dispatch;
    // the macro is defined at the include site.
    KOKKOS_INLINE_FUNCTION
    static quad::ddfun::ddouble _one() {
        return quad::ddfun::make_dd(0x3ff0000000000000ULL, 0x0000000000000000ULL);
    }
};
} // namespace ql