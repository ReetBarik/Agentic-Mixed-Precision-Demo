#pragma once
// SOURCE_HASH: 6c905d16be2e12006d0ba065b5a8a1ed25022bd8d945b6afd4d2962fef44a7dc
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Rule 5 / R3 step 3a: ql::Constants<TMass>::_one() — source RHS T(1.0),
// a plain double literal, faithful ddouble has a zero low word.
// Rule C5/C7: partial specialization of the library's Constants<T> class
// template keyed on the extended scalar wins over the primary. The
// specialization is injected into namespace ql where the primary is
// declared at the region's include site (C1: do NOT include app headers).
// Rule C4/Rule 8: KOKKOS_INLINE_FUNCTION already defined at include site.
// Rule C9: chain-internal producer must return ddouble, never narrow.
namespace ql {
template <>
struct Constants<quad::ddfun::ddouble> {
    // Rule 5 / R3 step 3a: _one from T(1.0) — zero low word is correct.
    static KOKKOS_INLINE_FUNCTION quad::ddfun::ddouble _one() {
        return quad::ddfun::make_dd(0x3ff0000000000000ULL, 0x0000000000000000ULL);
    }
};
} // namespace ql