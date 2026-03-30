//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//
// Double precision version of kokkosMaths

#pragma once

#include <math.h>
#include <type_traits>

namespace ql
{
    using complex = Kokkos::complex<double>;

    template<typename T>
    struct Constants {

        // Number of Chebyshev coefficients for ddilog (must match coeffs array in _C)
        KOKKOS_INLINE_FUNCTION
        static constexpr int _num_C() { return 19; }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _C(int i) {
            // Double precision Chebyshev coefficients (19 terms)
            constexpr double coeffs[19] = {
                0.4299669356081370,
                0.4097598753307711,
                -0.0185884366501460,
                0.0014575108406227,
                -0.0001430418444234,
                0.0000158841554188,
                -0.0000019078495939,
                0.0000002419518085,
                -0.0000000319334127,
                0.0000000043454506,
                -0.0000000006057848,
                0.0000000000861210,
                -0.0000000000124433,
                0.0000000000018226,
                -0.0000000000002701,
                0.0000000000000404,
                -0.0000000000000061,
                0.0000000000000009,
                -0.0000000000000001
            };
            return T(coeffs[i]);
        }

        // Number of Bernoulli coefficients for li2series (must match coeffs array in _B)
        KOKKOS_INLINE_FUNCTION
        static constexpr int _num_B() { return 25; }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _B(int i) {
            // Double precision Bernoulli coefficients (25 terms)
            constexpr double coeffs[25] = {
                0.02777777777777777777777777777777777777777778774E0,
                -0.000277777777777777777777777777777777777777777778E0,
                4.72411186696900982615268329554043839758125472E-6,
                -9.18577307466196355085243974132863021751910641E-8,
                1.89788699889709990720091730192740293750394761E-9,
                -4.06476164514422552680590938629196667454705711E-11,
                8.92169102045645255521798731675274885151428361E-13,
                -1.993929586072107568723644347793789705630694749E-14,
                4.51898002961991819165047655285559322839681901E-16,
                -1.035651761218124701448341154221865666596091238E-17,
                2.39521862102618674574028374300098038167894899E-19,
                -5.58178587432500933628307450562541990556705462E-21,
                1.309150755418321285812307399186592301749849833E-22,
                -3.087419802426740293242279764866462431595565203E-24,
                7.31597565270220342035790560925214859103339899E-26,
                -1.740845657234000740989055147759702545340841422E-27,
                4.15763564461389971961789962077522667348825413E-29,
                -9.96214848828462210319400670245583884985485196E-31,
                2.394034424896165300521167987893749562934279156E-32,
                -5.76834735536739008429179316187765424407233225E-34,
                1.393179479647007977827886603911548331732410612E-35,
                -3.372121965485089470468473635254930958979742891E-37,
                8.17820877756210262176477721487283426787618937E-39,
                -1.987010831152385925564820669234786567541858996E-40,
                4.83577851804055089628705937311537820769430091E-42
            };
            return T(coeffs[i]);
        }
        
        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static T _qlonshellcutoff() {
            return T(1e-10);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _pi() {
            return T(M_PI);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _pi2() { 
            return _pi() * _pi(); 
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static T _pio3() { 
            return _pi() / TScale(3); 
        }

        template<typename TOutput, typename TMass, typename TScale>
        KOKKOS_INLINE_FUNCTION static T _pio6() { 
            return _pi() / TScale(6); 
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
            return _pi2() / TScale(12); 
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _zero() {
            return T(0.0);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _half() {
            return T(0.5);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _one() {
            return T(1.0);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _two() {
            return T(2.0);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _three() {
            return T(3.0);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _four() {
            return T(4.0);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _five() {
            return T(5.0);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _six() {
            return T(6.0);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _ten() {
            return T(10.0);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _eps() {
            return T(1e-6);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _eps4() {
            return T(1e-4);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _eps7() {
            return T(1e-7);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _eps10() {
            return T(1e-10);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _eps14() {
            return T(1e-14);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _eps15() {
            return T(1e-15);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _xloss() {
            return T(0.125);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _neglig() {
            return T(1e-14);
        }

        KOKKOS_INLINE_FUNCTION 
        static constexpr T _reps() {
            return T(1e-16);
        }

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
            return TOutput{Constants<TScale>::_zero(), 1e-50};
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

    // Math dispatch functions
    template<typename T>
    KOKKOS_INLINE_FUNCTION
    T kAbs(T const& x) {
        return Kokkos::abs(x);
    }

    // Explicit overloads for double and Kokkos::complex<double>
    KOKKOS_INLINE_FUNCTION
    double kAbs(double const& x) {
        return Kokkos::abs(x);
    }

    // Overload for Kokkos::complex<double> - Kokkos::abs(complex) returns double, not complex
    KOKKOS_INLINE_FUNCTION
    double kAbs(Kokkos::complex<double> const& x) {
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

    template<typename T>
    KOKKOS_INLINE_FUNCTION
    T kConj(T const& x) {
        return Kokkos::conj(x);
    }

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION bool iszero(TScale const& x) {
        return (ql::kAbs(x) < ql::Constants<TScale>::template _qlonshellcutoff<TOutput, TMass, TScale>()) ? true : false;
    }

    KOKKOS_INLINE_FUNCTION double Imag(double const& x) {
        return 0.0;
    }
        
    KOKKOS_INLINE_FUNCTION double Imag(Kokkos::complex<double> const& x) {
        return x.imag();
    }

    KOKKOS_INLINE_FUNCTION double Real(double const& x) {
        return x;   
    }

    KOKKOS_INLINE_FUNCTION double Real(Kokkos::complex<double> const& x) {
        return x.real();
    }

    KOKKOS_INLINE_FUNCTION int Sign(double const& x) {
        return (double(0) < x) - (x < double(0));
    }

    KOKKOS_INLINE_FUNCTION Kokkos::complex<double> Sign(Kokkos::complex<double> const& x) {
        return x / ql::kAbs(x);
    }

    
    KOKKOS_INLINE_FUNCTION double Max(double const& a, double const& b) {
        if (ql::kAbs(a) > ql::kAbs(b)) 
            return a;
        else 
            return b;
    }


    KOKKOS_INLINE_FUNCTION Kokkos::complex<double> Max(Kokkos::complex<double> const& a, Kokkos::complex<double> const& b) {
        if (ql::kAbs(a) > ql::kAbs(b)) 
            return a;
        else 
            return b;
    }

    
    KOKKOS_INLINE_FUNCTION double Min(double const& a, double const& b) {
        if (ql::kAbs(a) > ql::kAbs(b)) 
            return b;
        else 
            return a;
    }


    KOKKOS_INLINE_FUNCTION Kokkos::complex<double> Min(Kokkos::complex<double> const& a, Kokkos::complex<double> const& b) {
        if (ql::kAbs(a) > ql::kAbs(b)) 
            return b;
        else 
            return a;
    }

    KOKKOS_INLINE_FUNCTION double Htheta(double const& x) { 
        return 0.5 * (1 + ql::Sign(x)); 
    }

}
