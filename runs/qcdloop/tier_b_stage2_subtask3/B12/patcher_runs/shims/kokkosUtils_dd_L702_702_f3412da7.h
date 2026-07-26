#pragma once
// SOURCE_HASH: f3412da736ac75f36f52c06723e6cb406988a423aa165475d6198e41e6ba1eba
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Region:
//   const TOutput lnarg = TOutput(-ql::Lnrat<TOutput, TMass, TScale>(v, x)
//                                 - ql::Lnrat<TOutput, TMass, TScale>(w, y));
//
// After promotion, TOutput along this chain link is quad::ddfun::ddouble
// (and TMass / TScale — the mass and scale template parameters — are also
// ddouble because they participate in the chain-internal arithmetic).
//
// The region calls ql::Lnrat<TOutput, TMass, TScale>(...) with a namespace
// qualifier. Per C3 (namespace-qualified math bridge), ADL cannot rescue
// qualified calls into a foreign namespace, so we must make a ddouble
// overload of ql::Lnrat visible inside namespace ql itself. Per C9 (chain
// boundary), the result of Lnrat is consumed by this link's own arithmetic
// (unary minus, subtraction, TOutput(...) construction) and then written
// into `lnarg` which flows onward — so the overload MUST return ddouble,
// never a value narrowed back to double.
//
// The library's Lnrat is a function template with three explicit type
// parameters <TOutput, TMass, TScale>; we provide a function-template
// overload whose value-parameter types are the concrete
// quad::ddfun::ddouble, which is strictly more specialized than the
// library's bare-template primary under partial ordering (C5/C7). We keep
// the same three leading explicit template parameters the call site names
// so `ql::Lnrat<TOutput, TMass, TScale>(v, x)` continues to resolve.
//
// We forward to the primary ql::Lnrat with the same explicit template
// arguments — the primary is already visible at the shim's include site
// (the region's TU includes the app header before our shim), so ADL/name
// lookup finds it — and the primary's body, re-instantiated with ddouble
// for TOutput/TMass/TScale, computes in ddouble end-to-end via the vendored
// operators (Rule R2, C3 for any unqualified math inside it).

namespace ql {

// Rule 2 + C3 + C5/C7 + C9: ddouble-typed overload of the chain-internal
// helper ql::Lnrat so the qualified call ql::Lnrat<...>(v, x) on promoted
// (ddouble) operands stays in extended precision and returns ddouble to
// the next chain step (unary minus + subtraction + TOutput(...) ctor).
// Kokkos annotation per Rule 8/C4 — kokkosUtils.h dispatch is Kokkos.
template <class TOutput, class TMass, class TScale>
KOKKOS_INLINE_FUNCTION
quad::ddfun::ddouble Lnrat(const quad::ddfun::ddouble& a,
                           const quad::ddfun::ddouble& b) {
    // Delegate to the primary template re-instantiated at ddouble; every
    // internal op is provided by the vendored ddouble operator set
    // (dd_math.hpp) so the computation never leaves extended precision.
    return ::ql::Lnrat<TOutput, TMass, TScale>(a, b);
}

} // namespace ql