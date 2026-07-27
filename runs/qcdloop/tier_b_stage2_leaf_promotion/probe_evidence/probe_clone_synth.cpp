// v2 falsification probe for LEAF_CALLEE_PROMOTION_DESIGN §7 (revised overlay).
//
// v1's WITH_OVERLAY block HAND-WROTE a Constants<ddouble> specialisation — which the
// revised architecture forbids (that is importing qcdloop-specific dd knowledge).
// This v2 probe replaces the overlay with what the PIPELINE WOULD ACTUALLY SYNTHESIZE:
//
//   (Class 1) mechanical dd overloads of the shallow ql:: wrappers, each a straight
//             namespace redirect Kokkos::fn -> quad::ddfun::fn or a member accessor —
//             exactly the Gap-A bridge shape the shim machinery already emits.  These
//             are GENERATED, not vendored: every line is derivable from the primary's
//             one-line body in src/kokkosMaths.h + the vendored quad::ddfun surface.
//
//   (Class 2) NOTHING is hand-written.  Constants<ddouble>::_C / _num_C / _pi / _half
//             / _ipio2 instantiate DIRECTLY from the unmodified source primary
//             (kokkosMaths.h) at T = ddouble — proven by probe_constants_dd.cpp.  The
//             19 double coeffs promote honestly to make_dd(bits,0).  This is Option B
//             (§2): accept the 19-coeff series at dd, no 43-coeff synthesis.
//
// Build A (vendored-only, = pipeline today): FAIL, enumerates the Class-1 gap.
// Build B (-DWITH_SYNTH, = Class-1 overlay only): COMPILE + RUN, matches double.
//
//   g++ -std=c++20 -w -DWITH_SYNTH -Isrc -Ithird_party/include -I~/kokkos-install/include \
//       probe_clone_synth.cpp -L~/kokkos-install/lib64 -lkokkoscore -lkokkoscontainers -ldl -o /tmp/Bs && /tmp/Bs

#include <Kokkos_Core.hpp>
#include "kokkosMaths.h"
#include "kokkosUtils.h"
#include <dd_math.hpp>
#include <dd_complex.hpp>

using quad::ddfun::ddouble;
using quad::ddfun::ddcomplex;

#ifdef WITH_SYNTH
// ===========================================================================
// CLASS-1 SYNTHESIZED SURFACE.  Each overload below is what the extended Gap-A
// machinery would emit for one shallow ql:: wrapper, derived MECHANICALLY from
// that wrapper's own primary body + the vendored quad::ddfun ops.  No qcdloop dd
// knowledge is imported — only "the primary delegates to Kokkos::fn; redirect
// that one call to quad::ddfun::fn".  NB: NO Constants<ddouble> here — the source
// primary already instantiates at dd (Class 2 handled by Option B, zero synthesis).
// ===========================================================================
namespace ql {
    // primary: template<T> T kAbs(T x){ return Kokkos::abs(x); }  (kokkosMaths.h:271)
    //   -> redirect Kokkos::abs -> quad::ddfun::abs  (dd real returns ddouble; dd
    //      complex returns ddouble, mirroring the double overload's return kind)
    KOKKOS_INLINE_FUNCTION ddouble  kAbs(ddouble  const& x) { return quad::ddfun::abs(x); }
    KOKKOS_INLINE_FUNCTION ddouble  kAbs(ddcomplex const& z) { return quad::ddfun::abs(z); }
    // primary: template<T> T kLog(T x){ return Kokkos::log(x); }  (kokkosMaths.h:289)
    KOKKOS_INLINE_FUNCTION ddouble   kLog(ddouble   const& x) { return quad::ddfun::log(x); }
    KOKKOS_INLINE_FUNCTION ddcomplex kLog(ddcomplex const& z) { return quad::ddfun::log(z); }
    // primary: Real/Imag are member accessors (kokkosMaths.h:320-326) -> .real()/.imag()
    KOKKOS_INLINE_FUNCTION ddouble Real(ddcomplex const& z) { return z.real(); }
    KOKKOS_INLINE_FUNCTION ddouble Imag(ddcomplex const& z) { return z.imag(); }
    // primary: int Sign(double x){ return (0<x)-(x<0); } (kokkosMaths.h:328) -> same
    //   ±1/0 logic, T-generic; re-emit with the dd operators (already defined).
    KOKKOS_INLINE_FUNCTION int Sign(ddouble const& x) {
        return (ddouble(0.0) < x) - (x < ddouble(0.0));
    }
    // NOTE: Constants<ddouble> is NOT specialised here.  ql::Lnrat's body calls
    // ql::Constants<TScale>::_ipio2<...>() with TScale=ddouble, which resolves via
    // the UNMODIFIED source primary Constants<T> (kokkosMaths.h:18) — instantiating
    // _pi()=T(M_PI), _half()=T(0.5) at T=ddouble.  Class 2 = source, not synthesis.
}
#endif

// ---------------------------------------------------------------------------
// The proposed clone Lnrat_B10 — body VERBATIM from the ql::Lnrat TScale overload
// (kokkosUtils.h:153-155), renamed, reads promoted to dd.  Identical to v1's probe;
// the ONLY thing v2 changes is what stands behind ql::kLog/kAbs/Sign/Constants.
// ---------------------------------------------------------------------------
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
        ddouble v(1.5), x(2.5);
        auto r = ql::Lnrat_B10<ddcomplex, double, ddouble>(v, x);
        out = r.real().hi + r.imag().hi;
        auto rd = ql::Lnrat<Kokkos::complex<double>, double, double>(1.5, 2.5);
        double dd_val = r.real().hi, d_val = rd.real();
        Kokkos::printf("Lnrat_B10(synth) dd re.hi = %.17g   double re = %.17g   |diff| = %.3e\n",
                       dd_val, d_val, (dd_val - d_val < 0 ? d_val - dd_val : dd_val - d_val));
    }
    Kokkos::finalize();
    return ((int)out) & 1;
}
