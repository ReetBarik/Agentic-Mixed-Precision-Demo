#pragma once
// SOURCE_HASH: 7c6e60d3c842c5e59405012e0482358fb7f178f78969a81dc505331bffc051f2
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Rule 5 / R3 step 2: named constant ql::Constants<TScale>::_pi2o12 materialized
// from pre-derived exact hex (hi, lo) pair (source RHS: _pi2() / TScale(12)).
// The region invokes ql::Constants<TScale>::template _pi2o12<TOutput, TMass, TScale>().
// TOutput is promoted to quad::ddfun::ddouble on the chain; per C5/C7 we provide a
// partial specialization of ql::Constants keyed on the extended scalar so the
// qualified call resolves to a ddouble-returning member (C9 chain-internal contract).
//
// Cascade rejections for _pi2o12:
//   step 1: no vendored dd_pi2o12() free function exists.
//   step 2: exact (hi, lo) bit pair supplied under 'Source-derivable constants' — used here.
//   step 3: not needed; step 2 applied.

namespace ql {

template <>
struct Constants<quad::ddfun::ddouble> {
    // Rule 2 / R3 step 2 — return ddouble (chain-internal, C9); hex pair verbatim from prompt.
    // Source name: _pi2o12  (RHS: _pi2() / TScale(12))
    template <class TOutput, class TMass, class TScale>
    static KOKKOS_INLINE_FUNCTION quad::ddfun::ddouble _pi2o12() {
        return quad::ddfun::make_dd(0x3fea51a6625307d3ULL, 0x3c71873d8912200cULL);
    }
};

} // namespace ql