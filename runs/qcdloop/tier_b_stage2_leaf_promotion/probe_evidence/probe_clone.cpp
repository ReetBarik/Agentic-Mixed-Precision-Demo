// Falsification probe for LEAF_CALLEE_PROMOTION_DESIGN §7.
//
// Subtask 5 proved a *forwarding overload* ql::Lnrat(ddouble,ddouble) self-recurses
// (overload resolution re-selects itself). This probe tests the DESIGN's alternative:
// a clone-and-rename promoted frame `Lnrat_B10` whose body computes in dd and whose
// calls are all to renamed/vendored symbols — never to ql::Lnrat.
//
// Two questions:
//   Q1 (rename discipline): does a renamed clone avoid the self-recursion pit?  (By
//       construction it must — Lnrat_B10's body never names Lnrat_B10 or ql::Lnrat.)
//   Q2 (support surface):   does the clone COMPILE against the surface actually
//       available in the pipeline?  The body uses ql::kLog / ql::kAbs / ql::Sign /
//       ql::Constants<TScale>::_ipio2 with TScale = ddouble.  If those have no dd
//       overload, the clone cannot compile — that is the category-(d) blocker.
//
// Build A (vendored-only, mirrors the pipeline): expected to FAIL — enumerates the
//   missing support surface.
// Build B (adds -DWITH_OVERLAY, a hand-written dd support surface): expected to
//   COMPILE + RUN and match the double result on the exact 706-branch inputs.

#include <Kokkos_Core.hpp>
#include "kokkosMaths.h"
#include "kokkosUtils.h"
#include <dd_math.hpp>
#include <dd_complex.hpp>

using quad::ddfun::ddouble;
using quad::ddfun::ddcomplex;

#ifdef WITH_OVERLAY
// -----------------------------------------------------------------------------
// Support-surface overlay: the MINIMUM dd helper layer the clone body needs.
// This mirrors what qcdloop@ddfun_enabled:src/qcdloop/kokkosMaths_dd.h provides
// but which is NOT vendored into this repo's third_party/include.  Its existence
// (and size) is the whole point: it is category-(d) work the design must fund.
// -----------------------------------------------------------------------------
namespace ql {
    // kAbs / kLog on dd real+complex (vendored quad::ddfun has abs/log, but ql::kAbs
    // wraps Kokkos::abs which has no dd overload — so we must bridge explicitly).
    KOKKOS_INLINE_FUNCTION ddouble  kAbs(ddouble  const& x) { return quad::ddfun::abs(x); }
    KOKKOS_INLINE_FUNCTION ddouble  kAbs(ddcomplex const& z) { return quad::ddfun::abs(z); }
    KOKKOS_INLINE_FUNCTION ddouble  kLog(ddouble  const& x) { return quad::ddfun::log(x); }
    KOKKOS_INLINE_FUNCTION ddcomplex kLog(ddcomplex const& z) { return quad::ddfun::log(z); }
    // Real / Imag / Sign on dd (kokkosMaths.h only has double / Kokkos::complex<double>).
    KOKKOS_INLINE_FUNCTION ddouble Real(ddcomplex const& z) { return z.real(); }
    KOKKOS_INLINE_FUNCTION ddouble Imag(ddcomplex const& z) { return z.imag(); }
    KOKKOS_INLINE_FUNCTION int     Sign(ddouble const& x) {
        return (ddouble(0.0) < x) - (x < ddouble(0.0));
    }
    // Constants<ddouble>::_pi() / _half() etc. must resolve at TScale = ddouble.
    // The pipeline Constants<T> primary uses T(0.5) literals and _pi() from a double
    // bit-pattern; here we route pi through the vendored dd_pi() for dd accuracy.
    template<> struct Constants<ddouble> {
        KOKKOS_INLINE_FUNCTION static ddouble _zero() { return ddouble(0.0); }
        KOKKOS_INLINE_FUNCTION static ddouble _half() { return ddouble(0.5); }
        KOKKOS_INLINE_FUNCTION static ddouble _pi()   { return quad::ddfun::dd_pi(); }
        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static TOutput _ipio2() {
            return TOutput{_zero(), _pi() * _half()};
        }
    };
}
#endif

// -----------------------------------------------------------------------------
// The proposed clone: Lnrat_B10, a promoted frame.  Body copied VERBATIM from the
// TScale overload ql::Lnrat (kokkosUtils.h:152-155) — the overload Li2omx2:706
// selects — with the frame renamed and reads promoted to dd.  TOutput := ddcomplex,
// TScale := ddouble.  NOTE: the body names ql::kLog / ql::kAbs / ql::Sign /
// ql::Constants<ddouble> — it does NOT name Lnrat_B10 or ql::Lnrat (Q1: no recursion).
// -----------------------------------------------------------------------------
namespace ql {
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput Lnrat_B10(TScale const& x, TScale const& y) {
        return TOutput(ql::kLog(ql::kAbs(x / y)))
             - (ql::Constants<TScale>::template _ipio2<TOutput, TMass, TScale>()
                * TOutput(ql::Sign(-x) - ql::Sign(-y)));
    }
}

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    double out = 0.0;
    {
        // The exact inputs the Subtask-5 forwarding overload segfaulted on (v=1.5,x=2.5),
        // promoted to dd.
        ddouble v(1.5), x(2.5);
        auto r = ql::Lnrat_B10<ddcomplex, double, ddouble>(v, x);
        out = r.real().hi + r.imag().hi;

        // Correctness cross-check: the double primary on the same inputs.
        auto rd = ql::Lnrat<Kokkos::complex<double>, double, double>(1.5, 2.5);
        double dd_val = r.real().hi;
        double d_val  = rd.real();
        Kokkos::printf("Lnrat_B10 dd re.hi = %.17g   double re = %.17g   |diff| = %.3e\n",
                       dd_val, d_val, (dd_val - d_val < 0 ? d_val - dd_val : dd_val - d_val));
    }
    Kokkos::finalize();
    return ((int)out) & 1;
}
