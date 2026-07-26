#pragma once
// SOURCE_HASH: 1d04ac0e402b30e1c9c171bb35f8e96eec18c038815fb27c26101c6be96d31ae
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Shim for kokkosUtils.h:704 — Li2omx2 computation promoted to double-double.
// Region references:
//   ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>()
//   ql::Constants<TScale>::_half()
//   ql::ddilog<TOutput, TMass, TScale>(arg2)
//   TOutput(...) conversions on ddouble/ddcomplex values
//
// Rule 5 / R3: named constants must be materialized at full ddouble precision.
// Rule C5/C7: specialize the target library's Constants<T> class template on
// the extended scalar so the region's qualified calls resolve to ddouble
// producers rather than the primary (double) template.

namespace ql {

// Rule C5/C7 + Rule 5: partial specialization of ql::Constants keyed on the
// extended scalar quad::ddfun::ddouble. Rule R3 step 2 (known hex pair)
// supplies each constant verbatim from the "Source-derivable constants" table.
template <>
struct Constants<quad::ddfun::ddouble> {

    // Rule 5 / R3 step 2: _pi2o6 from source RHS _pi() * _pio6<...>()
    // (catalog:pi_squared_over_6). Chain-internal (C9): returns ddouble.
    template <class TOutput, class TMass, class TScale>
    KOKKOS_INLINE_FUNCTION
    static quad::ddfun::ddouble _pi2o6() {
        return quad::ddfun::make_dd(0x3ffa51a6625307d3ULL,
                                    0x3c81873d8912200cULL);
    }

    // Rule 5 / R3 step 3a: _half from source RHS T(0.5) (literal).
    // Zero low word is correct — the source literal carries only double precision.
    KOKKOS_INLINE_FUNCTION
    static quad::ddfun::ddouble _half() {
        return quad::ddfun::make_dd(0x3fe0000000000000ULL,
                                    0x0000000000000000ULL);
    }
};

} // namespace ql