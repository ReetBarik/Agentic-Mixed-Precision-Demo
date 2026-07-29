//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//
// Float-float precision version of kokkosMaths (FFFUN backend, ql::ffun::).
// Portable across all Kokkos backends (CUDA / HIP / SYCL / OpenMP / Serial).
//
// Structural mirror of runs/qcdloop_headers_full/kokkosMaths_dd.h with:
//   ql::ddfun  → ql::ffun
//   ddouble    → ffloat
//   ddcomplex  → ffcomplex
//   43-term dd Chebyshev / 25-term dd Bernoulli → same shapes at ff precision
// Constants are ff-encoded via Dekker split of the source double values that
// the dd bit-patterns represent (hi float + residual float), preserving the
// full ~14-digit ff precision at each entry. Custom container ql::ffun::ffcomplex
// (not Kokkos::complex<ffloat>) sidesteps Kokkos::complex's is_floating_point_v
// static_assert (STOP #EEE from Phase-2-float landing 0e580fe).

#pragma once

#include <math.h>
#include <type_traits>

#include "ff_math.hpp"
#include "ff_complex.hpp"

// Namespace shim (Agentic pipeline vendored snapshot, 2026-07-29).
// The ff primitives live under `quad::ffun` in third_party/include/ff_*.hpp
// (byte-identical to ReetBarik/kokkos-extended-precision-demo@fffunKokkos).
// Alias `ql::ffun` to `quad::ffun` so this header mirrors kokkosMaths_dd.h's
// alias pattern (`ql::ddfun` → `quad::ddfun`) exactly.
namespace ql { namespace ffun = ::quad::ffun; }

namespace ql
{
    using complex = ql::ffun::ffcomplex;

    template<typename T>
    struct Constants {

        // Number of Chebyshev coefficients for ddilog (must match coeffs array in _C)
        KOKKOS_INLINE_FUNCTION
        static constexpr int _num_C() { return 43; }

        KOKKOS_INLINE_FUNCTION
        static T _C(int i) {
            // FF-encoded Chebyshev coefficients (43 terms) — Dekker split of the
            // source double values from kokkosMaths_dd.h (scripts/gen_ff_constants).
            const T coeffs[43] = {
                ql::ffun::make_ff(0x3edc24a0U, 0x321b3172U),  // C[0]   (~ 0.42996693560813698)
                ql::ffun::make_ff(0x3ed1cc0cU, 0xb181ee2fU),  // C[1]   (~ 0.40975987533077107)
                ql::ffun::make_ff(0xbc9846c7U, 0x2f00fcacU),  // C[2]   (~ -0.01858843665014592)
                ql::ffun::make_ff(0x3abf09f3U, 0xadd901a0U),  // C[3]   (~ 0.0014575108406226786)
                ql::ffun::make_ff(0xb915fd81U, 0x2c9988c5U),  // C[4]   (~ -0.00014304184442340048)
                ql::ffun::make_ff(0x37853ef7U, 0xaad6da28U),  // C[5]   (~ 1.5884155418795532e-05)
                ql::ffun::make_ff(0xb600089bU, 0xa98001adU),  // C[6]   (~ -1.9078495938658273e-06)
                ql::ffun::make_ff(0x3481e59aU, 0x286b28bcU),  // C[7]   (~ 2.4195180854164751e-07)
                ql::ffun::make_ff(0xb3092729U, 0x26c8b547U),  // C[8]   (~ -3.1933412742517834e-08)
                ql::ffun::make_ff(0x31954efdU, 0xa434c326U),  // C[9]   (~ 4.3454506267691229e-09)
                ql::ffun::make_ff(0xb0268451U, 0x2338d07aU),  // C[10]  (~ -6.0578480118407444e-10)
                ql::ffun::make_ff(0x2ebd61cdU, 0xa0f262f8U),  // C[11]  (~ 8.6120977993594977e-11)
                ql::ffun::make_ff(0xad5ae7b7U, 0xa0979609U),  // C[12]  (~ -1.2443316599388679e-11)
                ql::ffun::make_ff(0x2c004044U, 0x1fec141dU),  // C[13]  (~ 1.8225569623573632e-12)
                ql::ffun::make_ff(0xaa9808daU, 0x9c805546U),  // C[14]  (~ -2.7006766049114654e-13)
                ql::ffun::make_ff(0x29360b80U, 0x9b16e4a8U),  // C[15]  (~ 4.0422092631526645e-14)
                ql::ffun::make_ff(0xa7dbe48fU, 0x99c613e1U),  // C[16]  (~ -6.1032514526918794e-15)
                ql::ffun::make_ff(0x2685d464U, 0x18b42196U),  // C[17]  (~ 9.2862975330195768e-16)
                ql::ffun::make_ff(0xa52403cdU, 0x98e71434U),  // C[18]  (~ -1.4226020855112447e-16)
                ql::ffun::make_ff(0x23ca3c0dU, 0x17784653U),  // C[19]  (~ 2.1926317181539574e-17)
                ql::ffun::make_ff(0xa27ab9e6U, 0x95b1c725U),  // C[20]  (~ -3.3979732421589787e-18)
                ql::ffun::make_ff(0x211c30e0U, 0x1449313aU),  // C[21]  (~ 5.291954244833147e-19)
                ql::ffun::make_ff(0x9fc378f3U, 0x93256963U),  // C[22]  (~ -8.2785808142789982e-20)
                ql::ffun::make_ff(0x1e75a21eU, 0x11b91520U),  // C[23]  (~ 1.3003717345455603e-20)
                ql::ffun::make_ff(0x9d1ae911U, 0x103e81afU),  // C[24]  (~ -2.0502222425528248e-21)
                ql::ffun::make_ff(0x1bc40ff2U, 0x0f578bb2U),  // C[25]  (~ 3.2435785491489305e-22)
                ql::ffun::make_ff(0x9a78eeadU, 0x8cbf1e58U),  // C[26]  (~ -5.1477999033432072e-23)
                ql::ffun::make_ff(0x191e7e1eU, 0x8c988202U),  // C[27]  (~ 8.193877477171578e-24)
                ql::ffun::make_ff(0x97ca5ea5U, 0x0b0ea5f4U),  // C[28]  (~ -1.3077835405712668e-24)
                ql::ffun::make_ff(0x168185fcU, 0x8a1bb498U),  // C[29]  (~ 2.0925629305798902e-25)
                ql::ffun::make_ff(0x95263076U, 0x08197a87U),  // C[30]  (~ -3.3561661505438297e-26)
                ql::ffun::make_ff(0x13d5b454U, 0x07197737U),  // C[31]  (~ 5.3946577714316998e-27)
                ql::ffun::make_ff(0x9289af8eU, 0x85760a34U),  // C[32]  (~ -8.6891932086899993e-28)
                ql::ffun::make_ff(0x1131c2a4U, 0x04f4222bU),  // C[33]  (~ 1.4022816869659999e-28)
                ql::ffun::make_ff(0x8fe5eabbU, 0x032592cdU),  // C[34]  (~ -2.26715578131e-29)
                ql::ffun::make_ff(0x0e94f18cU, 0x820fa1bcU),  // C[35]  (~ 3.6717416991000001e-30)
                ql::ffun::make_ff(0x8d4149c6U, 0x008b271cU),  // C[36]  (~ -5.9561516950000003e-31)
                ql::ffun::make_ff(0x0bfb3833U, 0x0008c5c8U),  // C[37]  (~ 9.6766243200000001e-32)
                ql::ffun::make_ff(0x8aa37e5cU, 0x00073221U),  // C[38]  (~ -1.5743859500000001e-32)
                ql::ffun::make_ff(0x0955187fU, 0x80006d70U),  // C[39]  (~ 2.5650459999999999e-33)
                ql::ffun::make_ff(0x880b0df3U, 0x800031f8U),  // C[40]  (~ -4.184519e-34)
                ql::ffun::make_ff(0x06b5b431U, 0x800000c4U),  // C[41]  (~ 6.8349399999999998e-35)
                ql::ffun::make_ff(0x856db677U, 0x80000027U)   // C[42]  (~ -1.11772e-35)
            };
            return coeffs[i];
        }

        // Number of Bernoulli coefficients for li2series (must match coeffs array in _B)
        KOKKOS_INLINE_FUNCTION
        static constexpr int _num_B() { return 25; }

        KOKKOS_INLINE_FUNCTION
        static T _B(int i) {
            // FF-encoded Bernoulli coefficients (25 terms) — Dekker split of the
            // source double values from kokkosMaths_dd.h. Tail entries (B[20]+)
            // approach float's denormal range and lose precision at that scale;
            // this is the honest ff floor for these tables, not a defect.
            const T coeffs[25] = {
                ql::ffun::make_ff(0x3ce38e39U, 0xaf638e39U),  // B[0]   (~ 0.027777777777777776)
                ql::ffun::make_ff(0xb991a2b4U, 0x2ceca864U),  // B[1]   (~ -0.00027777777777777778)
                ql::ffun::make_ff(0x369e83d0U, 0xa9922184U),  // B[2]   (~ 4.7241118669690098e-06)
                ql::ffun::make_ff(0xb3c54352U, 0xa71f629dU),  // B[3]   (~ -9.1857730746619641e-08)
                ql::ffun::make_ff(0x31026bfbU, 0x24395e6eU),  // B[4]   (~ 1.8978869988971001e-09)
                ql::ffun::make_ff(0xae32c526U, 0x2161d2b1U),  // B[5]   (~ -4.0647616451442256e-11)
                ql::ffun::make_ff(0x2b7b1f8fU, 0x1dc46b09U),  // B[6]   (~ 8.9216910204564523e-13)
                ql::ffun::make_ff(0xa8b398e3U, 0x1bc48504U),  // B[7]   (~ -1.9939295860721074e-14)
                ql::ffun::make_ff(0x26024030U, 0x988c61faU),  // B[8]   (~ 4.5189800296199183e-16)
                ql::ffun::make_ff(0xa33f0b46U, 0x16ea2887U),  // B[9]   (~ -1.0356517612181247e-17)
                ql::ffun::make_ff(0x208d6385U, 0x13ec3158U),  // B[10]  (~ 2.395218621026187e-19)
                ql::ffun::make_ff(0x9dd2dfb8U, 0x9165eefdU),  // B[11]  (~ -5.581785874325009e-21)
                ql::ffun::make_ff(0x1b1e4441U, 0x8df48a5dU),  // B[12]  (~ 1.3091507554183213e-22)
                ql::ffun::make_ff(0x986ee0a6U, 0x8b868f2eU),  // B[13]  (~ -3.0874198024267403e-24)
                ql::ffun::make_ff(0x15b52281U, 0x08bf90d4U),  // B[14]  (~ 7.3159756527022029e-26)
                ql::ffun::make_ff(0x9309ec8bU, 0x86dfdf3fU),  // B[15]  (~ -1.7408456572340009e-27)
                ql::ffun::make_ff(0x1052d132U, 0x037370d8U),  // B[16]  (~ 4.1576356446138999e-29)
                ql::ffun::make_ff(0x8da1a525U, 0x812b89c4U),  // B[17]  (~ -9.9621484882846217e-31)
                ql::ffun::make_ff(0x0af89c5aU, 0x000030a6U),  // B[18]  (~ 2.3940344248961652e-32)
                ql::ffun::make_ff(0x883fafacU, 0x80000d5cU),  // B[19]  (~ -5.7683473553673897e-34)
                ql::ffun::make_ff(0x059425f4U, 0x80000043U),  // B[20]  (~ 1.393179479647008e-35)
                ql::ffun::make_ff(0x82e57ea7U, 0x00000007U),  // B[21]  (~ -3.3721219654850894e-37)
                ql::ffun::make_ff(0x00590d85U, 0x80000000U),  // B[22]  (~ 8.1782087775621025e-39)
                ql::ffun::make_ff(0x800229e6U, 0x00000000U),  // B[23]  (~ -1.9870108311523859e-40)
                ql::ffun::make_ff(0x00000d7bU, 0x80000000U)   // B[24]  (~ 4.8357785180405507e-42)
            };
            return coeffs[i];
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static T _qlonshellcutoff() {
            return T(1e-20);
        }

        KOKKOS_INLINE_FUNCTION
        static T _pi() {
            return ql::ffun::ff_pi();
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
    ql::ffun::ffloat kAbs(ql::ffun::ffloat const& x) {
        return ql::ffun::abs(x);
    }

    template<>
    KOKKOS_INLINE_FUNCTION
    ql::ffun::ffloat kLog(ql::ffun::ffloat const& x) {
        return ql::ffun::log(x);
    }

    template<>
    KOKKOS_INLINE_FUNCTION
    ql::ffun::ffloat kSqrt(ql::ffun::ffloat const& x) {
        return ql::ffun::sqrt(x);
    }

    // Overloads for ffcomplex
    KOKKOS_INLINE_FUNCTION
    ql::ffun::ffloat kAbs(ql::ffun::ffcomplex const& z) {
        return ql::ffun::abs(z);
    }

    KOKKOS_INLINE_FUNCTION
    ql::ffun::ffcomplex kLog(ql::ffun::ffcomplex const& z) {
        return ql::ffun::log(z);
    }

    KOKKOS_INLINE_FUNCTION
    ql::ffun::ffcomplex kSqrt(ql::ffun::ffcomplex const& z) {
        return ql::ffun::sqrt(z);
    }

    KOKKOS_INLINE_FUNCTION
    ql::ffun::ffcomplex kConj(ql::ffun::ffcomplex const& z) {
        return ql::ffun::conj(z);
    }

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION bool iszero(TScale const& x) {
        return (ql::kAbs(x) < ql::Constants<TScale>::template _qlonshellcutoff<TOutput, TMass, TScale>()) ? true : false;
    }

    KOKKOS_INLINE_FUNCTION ql::ffun::ffloat Imag(ql::ffun::ffloat const& /*x*/) {
        return ql::ffun::ffloat(0.0f);
    }

    KOKKOS_INLINE_FUNCTION ql::ffun::ffloat Imag(ql::ffun::ffcomplex const& x) {
        return x.imag();
    }

    KOKKOS_INLINE_FUNCTION ql::ffun::ffloat Real(ql::ffun::ffloat const& x) {
        return x;
    }

    KOKKOS_INLINE_FUNCTION ql::ffun::ffloat Real(ql::ffun::ffcomplex const& x) {
        return x.real();
    }

    KOKKOS_INLINE_FUNCTION ql::ffun::ffloat Sign(ql::ffun::ffloat const& x) {
        const ql::ffun::ffloat zero = ql::Constants<ql::ffun::ffloat>::_zero();
        const ql::ffun::ffloat one  = ql::Constants<ql::ffun::ffloat>::_one();
        if (zero < x) {
            return one;
        } else if (x < zero) {
            return -one;
        } else {
            return zero;
        }
    }

    KOKKOS_INLINE_FUNCTION ql::ffun::ffcomplex Sign(ql::ffun::ffcomplex const& x) {
        return x / ql::kAbs(x);
    }

    KOKKOS_INLINE_FUNCTION ql::ffun::ffloat Max(ql::ffun::ffloat const& a, ql::ffun::ffloat const& b) {
        if (ql::kAbs(a) > ql::kAbs(b))
            return a;
        else
            return b;
    }

    KOKKOS_INLINE_FUNCTION ql::ffun::ffcomplex Max(ql::ffun::ffcomplex const& a, ql::ffun::ffcomplex const& b) {
        if (ql::kAbs(a) > ql::kAbs(b))
            return a;
        else
            return b;
    }

    KOKKOS_INLINE_FUNCTION ql::ffun::ffloat Min(ql::ffun::ffloat const& a, ql::ffun::ffloat const& b) {
        if (ql::kAbs(a) > ql::kAbs(b))
            return b;
        else
            return a;
    }

    KOKKOS_INLINE_FUNCTION ql::ffun::ffcomplex Min(ql::ffun::ffcomplex const& a, ql::ffun::ffcomplex const& b) {
        if (ql::kAbs(a) > ql::kAbs(b))
            return b;
        else
            return a;
    }

    KOKKOS_INLINE_FUNCTION ql::ffun::ffloat Htheta(ql::ffun::ffloat const& x) {
        return ql::ffun::ffloat(0.5f) * (ql::ffun::ffloat(1.0f) + ql::Sign(x));
    }

}
