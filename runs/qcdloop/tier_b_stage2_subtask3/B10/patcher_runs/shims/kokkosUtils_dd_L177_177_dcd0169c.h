#pragma once
// SOURCE_HASH: dcd0169cbaea1be9621cb07188dcc04493ac39f0fa3916cf556c4c5fa7642fb4
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Region:
//   A = TMass(-ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>()
//        + A * (TMass(-ql::Constants<TMass>::_half()) * A
//               + ql::kLog(TMass(ql::Constants<TMass>::_one()) + T)));
//
// With TMass/TScale/TOutput all promoted to quad::ddfun::ddouble, we must supply:
//   - ql::Constants<ddouble>::_pi2o6<...>()   (Rule 5 / C5)
//   - ql::Constants<ddouble>::_half()         (Rule 5 / C5)
//   - ql::Constants<ddouble>::_one()          (Rule 5 / C5)
//   - ql::kLog(ddouble)                       (Rule 2 / C3, namespace-qualified)
//   - TMass(ddouble) i.e. ddouble(ddouble)    — already the identity ctor, no shim needed.

namespace ql {

// Rule 5 / C5: partial specialization of the library's Constants<T> primary
// template on the extended scalar. Values resolved per Rule R3 from
// "Source-derivable constants" — verbatim bit pairs, never decimal literals.
template <>
struct Constants<quad::ddfun::ddouble> {
    // Rule 5 / R3 step 2 — source RHS `T(0.5)` (literal), zero low word is correct.
    KOKKOS_INLINE_FUNCTION
    static quad::ddfun::ddouble _half() {
        return quad::ddfun::make_dd(0x3fe0000000000000ULL, 0x0000000000000000ULL);
    }

    // Rule 5 / R3 step 2 — source RHS `T(1.0)` (literal), zero low word is correct.
    KOKKOS_INLINE_FUNCTION
    static quad::ddfun::ddouble _one() {
        return quad::ddfun::make_dd(0x3ff0000000000000ULL, 0x0000000000000000ULL);
    }

    // Rule 5 / R3 step 3b — source RHS `_pi() * _pio6<...>()`, catalog:pi_squared_over_6.
    // C5: function-template member; keep the same leading explicit template
    // parameters the call site names (`template _pi2o6<TOutput, TMass, TScale>()`).
    template <class TOutput, class TMass, class TScale>
    KOKKOS_INLINE_FUNCTION
    static quad::ddfun::ddouble _pi2o6() {
        return quad::ddfun::make_dd(0x3ffa51a6625307d3ULL, 0x3c81873d8912200cULL);
    }
};

// Rule 2 / C3 (namespace-qualified math bridge): the region calls
// `ql::kLog(ddouble)`. A qualified call is looked up only in `ql` and its
// enclosing scopes, so a ddouble overload of `log` in `quad::ddfun` is not
// visible here — we must inject a ddouble overload directly into `ql` that
// forwards to the vendored `quad::ddfun::log`. Returns ddouble to honor the
// chain-internal contract (C9): its result feeds the enclosing expression
// which continues in extended precision.
KOKKOS_INLINE_FUNCTION
inline quad::ddfun::ddouble kLog(quad::ddfun::ddouble x) {
    return quad::ddfun::log(x);
}

} // namespace ql