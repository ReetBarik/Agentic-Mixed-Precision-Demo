//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//
// Quad-float precision version of kokkosMaths (QFFUN backend, ql::qfun::).
// Portable across all Kokkos backends (CUDA / HIP / SYCL / OpenMP / Serial).
//
// Structural mirror of third_party/include/kokkosMaths_ff.h (itself a mirror of
// runs/qcdloop_headers_full/kokkosMaths_dd.h) with:
//   ql::ffun   → ql::qfun
//   ffloat     → qfloat
//   ffcomplex  → qfcomplex
//   43-term Chebyshev / 25-term Bernoulli → same shapes at qf precision
// Constants are QF-encoded by a four-way FP32 split of the EXACT dd source pair;
// see the note on _C below for why the dd pair and not its double rounding.
// Custom container ql::qfun::qfcomplex (not Kokkos::complex<qfloat>) sidesteps
// Kokkos::complex's is_floating_point_v static_assert — QF sits in exactly the
// position FF did at STOP #EEE (Phase-2-float landing 0e580fe), and clears it the
// same way, on the vendored QuadFloatComplex.
//
// PRECISION / RANGE.  QF is four-word FP32: ~28.9 decimal digits (~96-bit
// significand), between double (~15.9) and dd (~31.9).  Its EXPONENT range is
// NOT widened — it stays FP32-bounded at ~3.4e38, far tighter than double or dd.
// Every other rung on the ladder either matches or exceeds the double baseline's
// range; qf and ff are the two that do not.

#pragma once

#include <math.h>
#include <type_traits>

#include "qf_math.hpp"
#include "qf_complex.hpp"

// Namespace shim (Agentic pipeline vendored snapshot; QF headers vendored from
// ReetBarik/kokkos-extended-precision-demo@e67d7da, 2026-08-13).
// The qf primitives live in third_party/include/qf_*.hpp under
// `Kokkos::Experimental`, as QuadFloat / QuadFloatComplex, with the bit-pattern
// constructor exposed as the static factory QuadFloat::from_bits().
//
// Mirrors kokkosMaths_ff.h / kokkosMaths_dd.h exactly: a REAL namespace rather
// than a namespace alias, because an alias cannot host the using-declarations or
// the make_qf/qf_pi compatibility wrappers that this header's ql:: call sites
// use.  The `using namespace ::Kokkos::Experimental;` is what makes the qualified
// spellings below (ql::qfun::abs, ql::qfun::sqrt, ...) resolve — qualified lookup
// in a namespace follows its using-directives.
namespace ql {
namespace qfun {

using namespace ::Kokkos::Experimental;

using qfloat    = ::Kokkos::Experimental::QuadFloat;
using qfcomplex = ::Kokkos::Experimental::QuadFloatComplex;

// Four limbs, not two — the QF analogue of make_ff/make_dd.
KOKKOS_INLINE_FUNCTION qfloat make_qf(uint32_t b0, uint32_t b1,
                                      uint32_t b2, uint32_t b3) {
    return ::Kokkos::Experimental::QuadFloat::from_bits(b0, b1, b2, b3);
}
KOKKOS_INLINE_FUNCTION qfloat qf_pi() { return ::Kokkos::Experimental::QuadFloat_pi(); }

}  // namespace qfun
}  // namespace ql

namespace ql
{
    using complex = ql::qfun::qfcomplex;

    template<typename T>
    struct Constants {

        // Number of Chebyshev coefficients for ddilog (must match coeffs array in _C)
        KOKKOS_INLINE_FUNCTION
        static constexpr int _num_C() { return 43; }

        KOKKOS_INLINE_FUNCTION
        static T _C(int i) {
            // QF-encoded Chebyshev coefficients (43 terms) — four-way FP32 split of
            // the EXACT dd value (hi+lo, ~31 digits) from kokkosMaths_dd.h, generated
            // by scripts/one_off/gen_qf_constants.py.  Splitting from the exact dd pair
            // rather than from its double approximation matters here: QF resolves ~29
            // digits, well past a single double's ~16, so a double-sourced split would
            // cap the table ~13 digits below what QF can hold.  (The ff table could
            // safely split from the double — ff resolves only ~14 digits.)
            const T coeffs[43] = {
                ql::qfun::make_qf(0x3edc24a0U, 0x321b3172U, 0xa491d086U, 0x965d5012U),  // C[0]   (~ 0.429966935608137)
                ql::qfun::make_qf(0x3ed1cc0cU, 0xb181ee2fU, 0xa5108f7cU, 0x187e5cb9U),  // C[1]   (~ 0.4097598753307711)
                ql::qfun::make_qf(0xbc9846c7U, 0x2f00fcacU, 0x227b9016U, 0x95bf8635U),  // C[2]   (~ -0.01858843665014592)
                ql::qfun::make_qf(0x3abf09f3U, 0xadd901a0U, 0x21669633U, 0x14c16a6cU),  // C[3]   (~ 0.0014575108406226786)
                ql::qfun::make_qf(0xb915fd81U, 0x2c9988c5U, 0x1fb06a6cU, 0x92c92b90U),  // C[4]   (~ -0.00014304184442340048)
                ql::qfun::make_qf(0x37853ef7U, 0xaad6da28U, 0x9d56de23U, 0x10c333cbU),  // C[5]   (~ 1.5884155418795532e-05)
                ql::qfun::make_qf(0xb600089bU, 0xa98001adU, 0x1adbd0dbU, 0x0d93e2c5U),  // C[6]   (~ -1.9078495938658273e-06)
                ql::qfun::make_qf(0x3481e59aU, 0x286b28bcU, 0x1bd4f2fbU, 0x8e27b30eU),  // C[7]   (~ 2.419518085416475e-07)
                ql::qfun::make_qf(0xb3092729U, 0x26c8b547U, 0x9a21c3fdU, 0x0dbc7fcfU),  // C[8]   (~ -3.1933412742517834e-08)
                ql::qfun::make_qf(0x31954efdU, 0xa434c326U, 0x97f7bbaaU, 0x8b49644aU),  // C[9]   (~ 4.345450626769123e-09)
                ql::qfun::make_qf(0xb0268451U, 0x2338d07aU, 0x96c26bfcU, 0x0a05612fU),  // C[10]  (~ -6.057848011840744e-10)
                ql::qfun::make_qf(0x2ebd61cdU, 0xa0f262f8U, 0x13d61386U, 0x06c3b685U),  // C[11]  (~ 8.612097799359498e-11)
                ql::qfun::make_qf(0xad5ae7b7U, 0xa0979609U, 0x128f7f47U, 0x05a8077dU),  // C[12]  (~ -1.2443316599388679e-11)
                ql::qfun::make_qf(0x2c004044U, 0x1fec141dU, 0x93158cb3U, 0x068e63ffU),  // C[13]  (~ 1.822556962357363e-12)
                ql::qfun::make_qf(0xaa9808daU, 0x9c805545U, 0x901b6d59U, 0x83944b22U),  // C[14]  (~ -2.7006766049114654e-13)
                ql::qfun::make_qf(0x29360b80U, 0x9b16e4a8U, 0x0e161bf0U, 0x019229bdU),  // C[15]  (~ 4.0422092631526645e-14)
                ql::qfun::make_qf(0xa7dbe48fU, 0x99c613e1U, 0x8bc12cedU, 0x800fb11eU),  // C[16]  (~ -6.1032514526918794e-15)
                ql::qfun::make_qf(0x2685d464U, 0x18b42196U, 0x0c031d1eU, 0x0000f09fU),  // C[17]  (~ 9.286297533019577e-16)
                ql::qfun::make_qf(0xa52403cdU, 0x98e71434U, 0x8acf33a4U, 0x00006badU),  // C[18]  (~ -1.4226020855112447e-16)
                ql::qfun::make_qf(0x23ca3c0dU, 0x17784653U, 0x8a01f1acU, 0x8002f565U),  // C[19]  (~ 2.1926317181539574e-17)
                ql::qfun::make_qf(0xa27ab9e6U, 0x95b1c725U, 0x893de62cU, 0x000002a4U),  // C[20]  (~ -3.3979732421589787e-18)
                ql::qfun::make_qf(0x211c30e0U, 0x1449313aU, 0x86ccf4ebU, 0x0000004dU),  // C[21]  (~ 5.291954244833147e-19)
                ql::qfun::make_qf(0x9fc378f3U, 0x93256963U, 0x86b22387U, 0x00000451U),  // C[22]  (~ -8.278580814278998e-20)
                ql::qfun::make_qf(0x1e75a21eU, 0x11b91520U, 0x84aa5e01U, 0x00000048U),  // C[23]  (~ 1.3003717345455603e-20)
                ql::qfun::make_qf(0x9d1ae911U, 0x103e81afU, 0x035d216bU, 0x8000000bU),  // C[24]  (~ -2.050222242552825e-21)
                ql::qfun::make_qf(0x1bc40ff2U, 0x0f578bb2U, 0x8255c2b2U, 0x80000002U),  // C[25]  (~ 3.2435785491489305e-22)
                ql::qfun::make_qf(0x9a78eeadU, 0x8cbf1e58U, 0x0006b53eU, 0x00000000U),  // C[26]  (~ -5.147799903343207e-23)
                ql::qfun::make_qf(0x191e7e1eU, 0x8c988202U, 0x000e617aU, 0x80000000U),  // C[27]  (~ 8.193877477171578e-24)
                ql::qfun::make_qf(0x97ca5ea5U, 0x0b0ea5f4U, 0x0006c30eU, 0x00000000U),  // C[28]  (~ -1.3077835405712668e-24)
                ql::qfun::make_qf(0x168185fcU, 0x8a1bb498U, 0x000150ddU, 0x80000000U),  // C[29]  (~ 2.09256293057989e-25)
                ql::qfun::make_qf(0x95263076U, 0x08197a87U, 0x800007bcU, 0x00000000U),  // C[30]  (~ -3.35616615054383e-26)
                ql::qfun::make_qf(0x13d5b454U, 0x07197737U, 0x8000078cU, 0x80000000U),  // C[31]  (~ 5.3946577714317e-27)
                ql::qfun::make_qf(0x9289af8eU, 0x85760a34U, 0x80000034U, 0x80000000U),  // C[32]  (~ -8.68919320869e-28)
                ql::qfun::make_qf(0x1131c2a4U, 0x04f4222bU, 0x8000006aU, 0x80000000U),  // C[33]  (~ 1.402281686966e-28)
                ql::qfun::make_qf(0x8fe5eabbU, 0x032592cdU, 0x80000008U, 0x80000000U),  // C[34]  (~ -2.26715578131e-29)
                ql::qfun::make_qf(0x0e94f18cU, 0x820fa1bcU, 0x80000000U, 0x80000000U),  // C[35]  (~ 3.6717416991e-30)
                ql::qfun::make_qf(0x8d4149c6U, 0x008b271cU, 0x80000000U, 0x80000000U),  // C[36]  (~ -5.956151695e-31)
                ql::qfun::make_qf(0x0bfb3833U, 0x0008c5c8U, 0x80000000U, 0x80000000U),  // C[37]  (~ 9.67662432e-32)
                ql::qfun::make_qf(0x8aa37e5cU, 0x00073221U, 0x00000000U, 0x00000000U),  // C[38]  (~ -1.57438595e-32)
                ql::qfun::make_qf(0x0955187fU, 0x80006d70U, 0x00000000U, 0x00000000U),  // C[39]  (~ 2.565046e-33)
                ql::qfun::make_qf(0x880b0df3U, 0x800031f8U, 0x00000000U, 0x00000000U),  // C[40]  (~ -4.184519e-34)
                ql::qfun::make_qf(0x06b5b431U, 0x800000c4U, 0x80000000U, 0x80000000U),  // C[41]  (~ 6.83494e-35)
                ql::qfun::make_qf(0x856db677U, 0x80000027U, 0x80000000U, 0x80000000U)   // C[42]  (~ -1.11772e-35)
            };
            return coeffs[i];
        }

        // Number of Bernoulli coefficients for li2series (must match coeffs array in _B)
        KOKKOS_INLINE_FUNCTION
        static constexpr int _num_B() { return 25; }

        KOKKOS_INLINE_FUNCTION
        static T _B(int i) {
            // QF-encoded Bernoulli coefficients (25 terms) — same exact-dd four-way
            // split as _C above.
            //
            // RANGE, NOT PRECISION, IS THE FLOOR HERE.  QF widens the significand
            // (4x24 bits) but inherits FP32's EXPONENT range, so entries near the
            // FP32 denormal floor (~1.4e-45) cannot spend their low limbs: B[24]
            // (~4.8e-42) is itself subnormal and reconstructs to only ~5 digits
            // (rel_err 2.1e-5), and C[42] (~-1.1e-35) to ~10 (rel_err 5.7e-11).
            // QF buys nothing over ff on those tail entries.  Harmless in context —
            // they sit 35+ orders below C[0]/B[0], far under even QF's 29-digit
            // resolution of the series sum — but it is the same FP32 range ceiling
            // that the strategy walk's fp32-family guard exists to catch, seen here
            // in the constant tables rather than in the data.
            const T coeffs[25] = {
                ql::qfun::make_qf(0x3ce38e39U, 0xaf638e39U, 0x21e38e39U, 0x94638e39U),  // B[0]   (~ 0.027777777777777776)
                ql::qfun::make_qf(0xb991a2b4U, 0x2ceca864U, 0x1f7edcbbU, 0x92cf1358U),  // B[1]   (~ -0.0002777777777777778)
                ql::qfun::make_qf(0x369e83d0U, 0xa9922184U, 0x1d435b73U, 0x90aa0c27U),  // B[2]   (~ 4.72411186696901e-06)
                ql::qfun::make_qf(0xb3c54352U, 0xa71f629dU, 0x99193eadU, 0x0cdcae44U),  // B[3]   (~ -9.185773074661964e-08)
                ql::qfun::make_qf(0x31026bfbU, 0x24395e6eU, 0x17a9328fU, 0x8ae9bee6U),  // B[4]   (~ 1.8978869988971e-09)
                ql::qfun::make_qf(0xae32c526U, 0x2161d2b1U, 0x14c79a39U, 0x0744f077U),  // B[5]   (~ -4.0647616451442256e-11)
                ql::qfun::make_qf(0x2b7b1f8fU, 0x1dc46b09U, 0x0ffdf1d9U, 0x0311e07dU),  // B[6]   (~ 8.921691020456452e-13)
                ql::qfun::make_qf(0xa8b398e3U, 0x1bc48504U, 0x0ecd2e0dU, 0x8247949eU),  // B[7]   (~ -1.9939295860721074e-14)
                ql::qfun::make_qf(0x26024030U, 0x988c61faU, 0x0bf0aac7U, 0x001a0a0aU),  // B[8]   (~ 4.518980029619918e-16)
                ql::qfun::make_qf(0xa33f0b46U, 0x16ea2887U, 0x89b93bc3U, 0x8001a629U),  // B[9]   (~ -1.0356517612181247e-17)
                ql::qfun::make_qf(0x208d6385U, 0x13ec3158U, 0x875f797eU, 0x80000bedU),  // B[10]  (~ 2.395218621026187e-19)
                ql::qfun::make_qf(0x9dd2dfb8U, 0x9165eefdU, 0x841e19eaU, 0x8000003fU),  // B[11]  (~ -5.581785874325009e-21)
                ql::qfun::make_qf(0x1b1e4441U, 0x8df48a5dU, 0x002677cdU, 0x80000000U),  // B[12]  (~ 1.3091507554183213e-22)
                ql::qfun::make_qf(0x986ee0a6U, 0x8b868f2eU, 0x0013eafcU, 0x00000000U),  // B[13]  (~ -3.0874198024267403e-24)
                ql::qfun::make_qf(0x15b52281U, 0x08bf90d4U, 0x00004dcbU, 0x00000000U),  // B[14]  (~ 7.315975652702203e-26)
                ql::qfun::make_qf(0x9309ec8bU, 0x86dfdf3fU, 0x00000461U, 0x80000000U),  // B[15]  (~ -1.740845657234001e-27)
                ql::qfun::make_qf(0x1052d132U, 0x037370d8U, 0x80000001U, 0x80000000U),  // B[16]  (~ 4.1576356446139e-29)
                ql::qfun::make_qf(0x8da1a525U, 0x812b89c4U, 0x00000000U, 0x00000000U),  // B[17]  (~ -9.962148488284622e-31)
                ql::qfun::make_qf(0x0af89c5aU, 0x000030a6U, 0x00000000U, 0x00000000U),  // B[18]  (~ 2.3940344248961652e-32)
                ql::qfun::make_qf(0x883fafacU, 0x80000d5cU, 0x00000000U, 0x00000000U),  // B[19]  (~ -5.76834735536739e-34)
                ql::qfun::make_qf(0x059425f4U, 0x80000043U, 0x00000000U, 0x00000000U),  // B[20]  (~ 1.393179479647008e-35)
                ql::qfun::make_qf(0x82e57ea7U, 0x00000007U, 0x00000000U, 0x00000000U),  // B[21]  (~ -3.3721219654850894e-37)
                ql::qfun::make_qf(0x00590d85U, 0x80000000U, 0x80000000U, 0x80000000U),  // B[22]  (~ 8.178208777562102e-39)
                ql::qfun::make_qf(0x800229e6U, 0x00000000U, 0x00000000U, 0x00000000U),  // B[23]  (~ -1.987010831152386e-40)
                ql::qfun::make_qf(0x00000d7bU, 0x80000000U, 0x80000000U, 0x80000000U)   // B[24]  (~ 4.8357785180405507e-42)
            };
            return coeffs[i];
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static T _qlonshellcutoff() {
            return T(1e-20);
        }

        KOKKOS_INLINE_FUNCTION
        static T _pi() {
            return ql::qfun::qf_pi();
        }

        KOKKOS_INLINE_FUNCTION
        static T _pi2() {
            return _pi() * _pi();
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static T _pio3() {
            return _pi() / ql::Constants<TScale>::_three();
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static T _pio6() {
            return _pi() / ql::Constants<TScale>::_six();
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static T _pi2o3() {
            return _pi() * _pio3<TOutput, TMass, TScale>();
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static T _pi2o6() {
            return _pi() * _pio6<TOutput, TMass, TScale>();
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static T _pi2o12() {
            return _pi2() / ql::Constants<TScale>::_twelve();
        }

        KOKKOS_INLINE_FUNCTION
        static constexpr T _zero() { return T(0.0); }

        KOKKOS_INLINE_FUNCTION
        static constexpr T _half() { return T(0.5); }

        KOKKOS_INLINE_FUNCTION
        static constexpr T _one() { return T(1.0); }

        KOKKOS_INLINE_FUNCTION
        static constexpr T _two() { return T(2.0); }

        KOKKOS_INLINE_FUNCTION
        static constexpr T _three() { return T(3.0); }

        KOKKOS_INLINE_FUNCTION
        static constexpr T _four() { return T(4.0); }

        KOKKOS_INLINE_FUNCTION
        static constexpr T _five() { return T(5.0); }

        KOKKOS_INLINE_FUNCTION
        static constexpr T _six() { return T(6.0); }

        KOKKOS_INLINE_FUNCTION
        static constexpr T _ten() { return T(10.0); }

        KOKKOS_INLINE_FUNCTION
        static constexpr T _twelve() { return T(12.0); }

        KOKKOS_INLINE_FUNCTION
        static constexpr T _eps() { return T(1e-12); }

        KOKKOS_INLINE_FUNCTION
        static constexpr T _eps4() { return T(1e-4); }

        KOKKOS_INLINE_FUNCTION
        static constexpr T _eps7() { return T(1e-7); }

        KOKKOS_INLINE_FUNCTION
        static constexpr T _eps10() { return T(1e-10); }

        KOKKOS_INLINE_FUNCTION
        static constexpr T _eps14() { return T(1e-14); }

        KOKKOS_INLINE_FUNCTION
        static constexpr T _eps15() { return T(1e-15); }

        KOKKOS_INLINE_FUNCTION
        static constexpr T _xloss() { return T(0.125); }

        KOKKOS_INLINE_FUNCTION
        static constexpr T _neglig() { return T(1e-28); }

        KOKKOS_INLINE_FUNCTION
        static constexpr T _reps() { return T(1e-30); }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static TOutput _2ipi() {
            return TOutput{Constants<TScale>::_zero(), Constants<TScale>::_two() * Constants<TScale>::_pi()};
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static TOutput _ipio2() {
            return TOutput{Constants<TScale>::_zero(), Constants<TScale>::_pi() * Constants<TScale>::_half()};
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static TOutput _ipi() {
            return TOutput{Constants<TScale>::_zero(), Constants<TScale>::_pi()};
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static TOutput _ieps() {
            return TOutput{Constants<TScale>::_zero(), Constants<TScale>::_reps()};
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static TOutput _ieps2() {
            return TOutput{Constants<TScale>::_zero(), Constants<TScale>::_reps() * Constants<TScale>::_reps()};
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static TOutput _ieps50() {
            return TOutput{Constants<TScale>::_zero(), TScale(1e-50)};
        }
    };

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput kPow(TOutput const& base, int const& exponent) {
        TOutput temp = TOutput(1.0);

        for (int i = 0; i < exponent; i++)
            temp *= base;

        return temp;
    }

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TMass kPow(TMass const& base, int const& exponent) {
        TMass temp = TMass(1.0);

        for (int i = 0; i < exponent; i++)
            temp *= base;

        return temp;
    }

    // Math dispatch functions - base templates
    template<typename T>
    KOKKOS_INLINE_FUNCTION
    T kAbs(T const& x) {
        return Kokkos::abs(x);
    }

    template<typename T>
    KOKKOS_INLINE_FUNCTION
    T kLog(T const& x) {
        return Kokkos::log(x);
    }

    template<typename T>
    KOKKOS_INLINE_FUNCTION
    T kSqrt(T const& x) {
        return Kokkos::sqrt(x);
    }

    // FF specializations
    template<>
    KOKKOS_INLINE_FUNCTION
    ql::qfun::qfloat kAbs(ql::qfun::qfloat const& x) {
        return ql::qfun::abs(x);
    }

    template<>
    KOKKOS_INLINE_FUNCTION
    ql::qfun::qfloat kLog(ql::qfun::qfloat const& x) {
        return ql::qfun::log(x);
    }

    template<>
    KOKKOS_INLINE_FUNCTION
    ql::qfun::qfloat kSqrt(ql::qfun::qfloat const& x) {
        return ql::qfun::sqrt(x);
    }

    // Overloads for qfcomplex
    KOKKOS_INLINE_FUNCTION
    ql::qfun::qfloat kAbs(ql::qfun::qfcomplex const& z) {
        return ql::qfun::abs(z);
    }

    KOKKOS_INLINE_FUNCTION
    ql::qfun::qfcomplex kLog(ql::qfun::qfcomplex const& z) {
        return ql::qfun::log(z);
    }

    KOKKOS_INLINE_FUNCTION
    ql::qfun::qfcomplex kSqrt(ql::qfun::qfcomplex const& z) {
        return ql::qfun::sqrt(z);
    }

    KOKKOS_INLINE_FUNCTION
    ql::qfun::qfcomplex kConj(ql::qfun::qfcomplex const& z) {
        return ql::qfun::conj(z);
    }

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION bool iszero(TScale const& x) {
        return (ql::kAbs(x) < ql::Constants<TScale>::template _qlonshellcutoff<TOutput, TMass, TScale>()) ? true : false;
    }

    KOKKOS_INLINE_FUNCTION ql::qfun::qfloat Imag(ql::qfun::qfloat const& /*x*/) {
        return ql::qfun::qfloat(0.0f);
    }

    KOKKOS_INLINE_FUNCTION ql::qfun::qfloat Imag(ql::qfun::qfcomplex const& x) {
        return x.imag();
    }

    KOKKOS_INLINE_FUNCTION ql::qfun::qfloat Real(ql::qfun::qfloat const& x) {
        return x;
    }

    KOKKOS_INLINE_FUNCTION ql::qfun::qfloat Real(ql::qfun::qfcomplex const& x) {
        return x.real();
    }

    KOKKOS_INLINE_FUNCTION ql::qfun::qfloat Sign(ql::qfun::qfloat const& x) {
        const ql::qfun::qfloat zero = ql::Constants<ql::qfun::qfloat>::_zero();
        const ql::qfun::qfloat one  = ql::Constants<ql::qfun::qfloat>::_one();
        if (zero < x) {
            return one;
        } else if (x < zero) {
            return -one;
        } else {
            return zero;
        }
    }

    KOKKOS_INLINE_FUNCTION ql::qfun::qfcomplex Sign(ql::qfun::qfcomplex const& x) {
        return x / ql::kAbs(x);
    }

    KOKKOS_INLINE_FUNCTION ql::qfun::qfloat Max(ql::qfun::qfloat const& a, ql::qfun::qfloat const& b) {
        if (ql::kAbs(a) > ql::kAbs(b))
            return a;
        else
            return b;
    }

    KOKKOS_INLINE_FUNCTION ql::qfun::qfcomplex Max(ql::qfun::qfcomplex const& a, ql::qfun::qfcomplex const& b) {
        if (ql::kAbs(a) > ql::kAbs(b))
            return a;
        else
            return b;
    }

    KOKKOS_INLINE_FUNCTION ql::qfun::qfloat Min(ql::qfun::qfloat const& a, ql::qfun::qfloat const& b) {
        if (ql::kAbs(a) > ql::kAbs(b))
            return b;
        else
            return a;
    }

    KOKKOS_INLINE_FUNCTION ql::qfun::qfcomplex Min(ql::qfun::qfcomplex const& a, ql::qfun::qfcomplex const& b) {
        if (ql::kAbs(a) > ql::kAbs(b))
            return b;
        else
            return a;
    }

    KOKKOS_INLINE_FUNCTION ql::qfun::qfloat Htheta(ql::qfun::qfloat const& x) {
        return ql::qfun::qfloat(0.5f) * (ql::qfun::qfloat(1.0f) + ql::Sign(x));
    }

}
