// ql_tracked_interop.hpp
// Tracked interop shim for QCDLoop+Kokkos (B14 spike / two-mass box family).
//
// SOURCE_HASH: 551c835fad3d9551d32c5ef4332d393770cb8055772c6a865bd4be0d3a0dd06f
//
// This shim makes ql::BO callable with T = tracked::Tracked<double>
// (TMass, TScale) and tracked::Complex<double> (TOutput).
//
// Include order note (from driver): this header is included BEFORE
// kokkosMaths.h / kokkosUtils.h / boxGPU.h so that qualified `ql::foo(...)`
// calls inside those templates resolve to our tracked overloads by partial
// ordering (Rule C7).

#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/complex.hpp>
#include <tracked/journal.hpp>

#include <cmath>
#include <type_traits>

// ---- Forward declarations of library class templates we specialize --------
// Rule C5: our Constants<T> specializations must parse even though our shim
// is included BEFORE kokkosMaths.h. Forward-declare the primary in ql's
// namespace so the specialization below is well-formed; the library supplies
// the primary later in the same TU.
namespace ql {
    template <typename T> struct Constants;
}

// ============================================================================
// Free operators the qcdloop templates apply to tracked values that the
// Tracked library does not define. Placed in namespace tracked so ADL finds
// them.
// Rule C3: statically-instantiated call graph uses unary + on tracked scalars
// (e.g. TOutput{+re, +im}-style constructions inside kokkosMaths). Identity
// operator: no rounding, no journal entry.
// ============================================================================
namespace tracked {

    // Rule C3: unary + identity, no journal record.
    template <class T>
    inline Tracked<T> operator+(const Tracked<T>& a) { return a; }

    // Rule C3: unary + on Complex identity.
    template <class T>
    inline Complex<T> operator+(const Complex<T>& a) { return a; }

} // namespace tracked

// ============================================================================
// ql::Constants specialization for tracked scalar (Rule 5 / Rule C5).
// The library dispatches every named numeric constant through the class
// template ql::Constants<T>::_zero() / _one() / ... / _pi() / _pi2() / etc.
// We mirror the FULL leaf-accessor interface, routing each leaf through
// tracked::constant("<name>", T(value)) so every named constant retains its
// name in the journal. Partial specialization keyed on tracked::Tracked<T>
// covers the tracked type generically (Rule C5).
// ============================================================================
namespace ql {

    template <class T>
    struct Constants< ::tracked::Tracked<T> > {
        using Tr = ::tracked::Tracked<T>;

        // ---- Chebyshev/Bernoulli table sizes: integer, stay integer (Rule 1)
        KOKKOS_INLINE_FUNCTION
        static constexpr int _num_C() { return 19; }
        KOKKOS_INLINE_FUNCTION
        static constexpr int _num_B() { return 25; }

        // ---- Chebyshev coefficient: floating-point leaf -> named constant.
        // Rule 5: named constant table entry, one id per index.
        static Tr _C(int i) {
            constexpr double coeffs[19] = {
                0.4299669356081370, 0.4097598753307711, -0.0185884366501460,
                0.0014575108406227, -0.0001430418444234, 0.0000158841554188,
                -0.0000019078495939, 0.0000002419518085, -0.0000000319334127,
                0.0000000043454506, -0.0000000006057848, 0.0000000000861210,
                -0.0000000000124433, 0.0000000000018226, -0.0000000000002701,
                0.0000000000000404, -0.0000000000000061, 0.0000000000000009,
                -0.0000000000000001
            };
            return ::tracked::constant<T>("C[" + std::to_string(i) + "]", T(coeffs[i]));
        }

        // Rule 5: named constant table entry (Bernoulli-series).
        static Tr _B(int i) {
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
            return ::tracked::constant<T>("B[" + std::to_string(i) + "]", T(coeffs[i]));
        }

        // Rule 5: onshell cutoff — named constant.
        template <typename TOutput, typename TMass, typename TScale>
        static Tr _qlonshellcutoff() {
            return ::tracked::constant<T>("qlonshellcutoff", T(1e-10));
        }

        // Rule 5: pi and derived pi constants.
        static Tr _pi()  { return ::tracked::constant<T>("pi", T(M_PI)); }
        static Tr _pi2() { auto p = _pi(); return p * p; }

        template <typename TOutput, typename TMass, typename TScale>
        static Tr _pio3()   { return _pi() / ::tracked::constant<T>("three", T(3)); }
        template <typename TOutput, typename TMass, typename TScale>
        static Tr _pio6()   { return _pi() / ::tracked::constant<T>("six",   T(6)); }
        template <typename TOutput, typename TMass, typename TScale>
        static Tr _pi2o3()  { return _pi() * _pio3<TOutput,TMass,TScale>(); }
        template <typename TOutput, typename TMass, typename TScale>
        static Tr _pi2o6()  { return _pi() * _pio6<TOutput,TMass,TScale>(); }
        template <typename TOutput, typename TMass, typename TScale>
        static Tr _pi2o12() { return _pi2() / ::tracked::constant<T>("twelve", T(12)); }

        // Rule 5: small integer / half constants — named so cancellation
        // provenance survives.
        static Tr _zero()  { return ::tracked::constant<T>("zero",  T(0.0)); }
        static Tr _half()  { return ::tracked::constant<T>("half",  T(0.5)); }
        static Tr _one()   { return ::tracked::constant<T>("one",   T(1.0)); }
        static Tr _two()   { return ::tracked::constant<T>("two",   T(2.0)); }
        static Tr _three() { return ::tracked::constant<T>("three", T(3.0)); }
        static Tr _four()  { return ::tracked::constant<T>("four",  T(4.0)); }
        static Tr _five()  { return ::tracked::constant<T>("five",  T(5.0)); }
        static Tr _six()   { return ::tracked::constant<T>("six",   T(6.0)); }
        static Tr _ten()   { return ::tracked::constant<T>("ten",   T(10.0)); }

        // Rule 5: epsilons / xloss / neglig / reps — named constants.
        static Tr _eps()    { return ::tracked::constant<T>("eps",    T(1e-6));  }
        static Tr _eps4()   { return ::tracked::constant<T>("eps4",   T(1e-4));  }
        static Tr _eps7()   { return ::tracked::constant<T>("eps7",   T(1e-7));  }
        static Tr _eps10()  { return ::tracked::constant<T>("eps10",  T(1e-10)); }
        static Tr _eps14()  { return ::tracked::constant<T>("eps14",  T(1e-14)); }
        static Tr _eps15()  { return ::tracked::constant<T>("eps15",  T(1e-15)); }
        static Tr _xloss()  { return ::tracked::constant<T>("xloss",  T(0.125)); }
        static Tr _neglig() { return ::tracked::constant<T>("neglig", T(1e-14)); }
        static Tr _reps()   { return ::tracked::constant<T>("reps",   T(1e-16)); }

        // Rule 3: complex-typed named constants — return the tracked complex
        // container (Complex<T>), not Tracked<Complex<T>>.
        // Rule 5: components are named ("zero", "pi", ...) via constant().
        // Rule 6: intermediate scalings (0.5, 2.0) that are structural to the
        // container construction go through literal() — they're not user-named
        // constants at the qcdloop call site, just Complex-assembly scaffolding.
        template <typename TOutput, typename TMass, typename TScale>
        static ::tracked::Complex<T> _2ipi() {
            auto zero = ::tracked::constant<T>("zero", T(0));
            auto two  = ::tracked::constant<T>("two",  T(2));
            auto pi   = ::tracked::constant<T>("pi",   T(M_PI));
            return ::tracked::Complex<T>(zero, two * pi);
        }
        template <typename TOutput, typename TMass, typename TScale>
        static ::tracked::Complex<T> _ipio2() {
            auto zero = ::tracked::constant<T>("zero", T(0));
            auto half = ::tracked::constant<T>("half", T(0.5));
            auto pi   = ::tracked::constant<T>("pi",   T(M_PI));
            return ::tracked::Complex<T>(zero, pi * half);
        }
        template <typename TOutput, typename TMass, typename TScale>
        static ::tracked::Complex<T> _ipi() {
            auto zero = ::tracked::constant<T>("zero", T(0));
            auto pi   = ::tracked::constant<T>("pi",   T(M_PI));
            return ::tracked::Complex<T>(zero, pi);
        }
        template <typename TOutput, typename TMass, typename TScale>
        static ::tracked::Complex<T> _ieps() {
            auto zero = ::tracked::constant<T>("zero", T(0));
            auto reps = ::tracked::constant<T>("reps", T(1e-16));
            return ::tracked::Complex<T>(zero, reps);
        }
        template <typename TOutput, typename TMass, typename TScale>
        static ::tracked::Complex<T> _ieps2() {
            auto zero = ::tracked::constant<T>("zero", T(0));
            auto reps = ::tracked::constant<T>("reps", T(1e-16));
            return ::tracked::Complex<T>(zero, reps * reps);
        }
        template <typename TOutput, typename TMass, typename TScale>
        static ::tracked::Complex<T> _ieps50() {
            auto zero  = ::tracked::constant<T>("zero",   T(0));
            auto ieps50v = ::tracked::constant<T>("ieps50", T(1e-50));
            return ::tracked::Complex<T>(zero, ieps50v);
        }
    };

} // namespace ql

// ============================================================================
// ql:: free-function shims for tracked scalar + tracked complex.
//
// Rule C7: each concrete-typed overload here must outrank the library's own
// same-named function template under partial ordering. For functions with
// leading explicit template parameters (e.g. Lnrat<TOutput,TMass,TScale>),
// we mirror those on the tracked overloads so qualified calls bind directly.
// ============================================================================

namespace ql {

    // ---- Real / Imag ---------------------------------------------------------
    // Rule 2: floating-point return participating in downstream error propagation.
    // The scalar's imaginary part is a literal 0 (not a named constant, so Rule 6).
    template <class T>
    inline ::tracked::Tracked<T> Real(const ::tracked::Tracked<T>& x) { return x; }

    template <class T>
    inline ::tracked::Tracked<T> Imag(const ::tracked::Tracked<T>& /*x*/) {
        return ::tracked::literal<T>(T(0)); // Rule 6: anonymous zero.
    }

    // Rule 2/3: complex Real/Imag return tracked scalar components verbatim.
    template <class T>
    inline ::tracked::Tracked<T> Real(const ::tracked::Complex<T>& z) { return z.real(); }
    template <class T>
    inline ::tracked::Tracked<T> Imag(const ::tracked::Complex<T>& z) { return z.imag(); }

    // ---- Sign ---------------------------------------------------------------
    // Rule C6: `Sign` yields ±1/0 that then flows into floating-point arithmetic
    // (multiplied into tracked expressions, added to condition-branches). Return
    // the tracked scalar (Rule 2), NOT raw int, so provenance survives.
    template <class T>
    inline ::tracked::Tracked<T> Sign(const ::tracked::Tracked<T>& x) {
        T v = x.value();
        T s = (T(0) < v) - (v < T(0));
        return ::tracked::literal<T>(s); // Rule 6: unnamed ±1/0 sentinel.
    }

    // Rule C6: complex Sign = z/|z|; return tracked complex container (Rule 3).
    template <class T>
    inline ::tracked::Complex<T> Sign(const ::tracked::Complex<T>& z) {
        auto a = ::tracked::abs(z);
        return z / a;
    }

    // ---- kAbs ---------------------------------------------------------------
    // Rule 2: |x| is floating-point, propagate through tracked abs.
    template <class T>
    inline ::tracked::Tracked<T> kAbs(const ::tracked::Tracked<T>& x) {
        return ::tracked::abs(x);
    }
    // Rule 2/3: complex |z| yields a real tracked scalar.
    template <class T>
    inline ::tracked::Tracked<T> kAbs(const ::tracked::Complex<T>& z) {
        return ::tracked::abs(z);
    }

    // ---- kLog / kSqrt / kConj -----------------------------------------------
    // Rule 2/3: tracked math functions dispatch through tracked:: overloads.
    template <class T>
    inline ::tracked::Tracked<T> kLog(const ::tracked::Tracked<T>& x) {
        return ::tracked::log(x);
    }
    template <class T>
    inline ::tracked::Complex<T> kLog(const ::tracked::Complex<T>& z) {
        return ::tracked::log(z);
    }

    template <class T>
    inline ::tracked::Tracked<T> kSqrt(const ::tracked::Tracked<T>& x) {
        return ::tracked::sqrt(x);
    }
    template <class T>
    inline ::tracked::Complex<T> kSqrt(const ::tracked::Complex<T>& z) {
        return ::tracked::sqrt(z);
    }

    // Rule 3: complex conjugate returns tracked complex container.
    template <class T>
    inline ::tracked::Complex<T> kConj(const ::tracked::Complex<T>& z) {
        return ::tracked::conj(z);
    }
    // Rule 2: real "conjugate" is identity.
    template <class T>
    inline ::tracked::Tracked<T> kConj(const ::tracked::Tracked<T>& x) { return x; }

    // ---- Max / Min ----------------------------------------------------------
    // Rule 2: qcdloop's Max/Min compare by |value| and return one of the
    // ORIGINAL tracked operands unchanged — preserving provenance. Rule 7:
    // comparison itself is on plain .value().
    template <class T>
    inline ::tracked::Tracked<T> Max(const ::tracked::Tracked<T>& a,
                                     const ::tracked::Tracked<T>& b) {
        using std::abs;
        return (abs(a.value()) > abs(b.value())) ? a : b;
    }
    template <class T>
    inline ::tracked::Complex<T> Max(const ::tracked::Complex<T>& a,
                                     const ::tracked::Complex<T>& b) {
        auto aa = ::tracked::abs(a);
        auto ab = ::tracked::abs(b);
        return (aa.value() > ab.value()) ? a : b;
    }
    template <class T>
    inline ::tracked::Tracked<T> Min(const ::tracked::Tracked<T>& a,
                                     const ::tracked::Tracked<T>& b) {
        using std::abs;
        return (abs(a.value()) > abs(b.value())) ? b : a;
    }
    template <class T>
    inline ::tracked::Complex<T> Min(const ::tracked::Complex<T>& a,
                                     const ::tracked::Complex<T>& b) {
        auto aa = ::tracked::abs(a);
        auto ab = ::tracked::abs(b);
        return (aa.value() > ab.value()) ? b : a;
    }

    // ---- Htheta -------------------------------------------------------------
    // Rule C6: Htheta returns a numeric 0/1 that feeds tracked arithmetic
    // downstream (multiplied into 2ipi products, etc.). Return tracked scalar.
    template <class T>
    inline ::tracked::Tracked<T> Htheta(const ::tracked::Tracked<T>& x) {
        // 0.5 * (1 + sign(x)); build through tracked ops so the record chain
        // stays consistent. "half" / "one" are structural constants used only
        // to assemble the step (Rule 5 named where semantically meaningful).
        auto half = ::tracked::constant<T>("half", T(0.5));
        auto one  = ::tracked::constant<T>("one",  T(1));
        auto s    = Sign(x);
        return half * (one + s);
    }

    // ---- iszero -------------------------------------------------------------
    // Rule 1: predicate returns raw bool (used only as a branch condition).
    // Rule 7: comparison on tracked value uses .value().
    template <class TOutput, class TMass, class TScale, class T>
    inline bool iszero(const ::tracked::Tracked<T>& x) {
        using std::abs;
        // Match the library primary: |x| < onshell cutoff.
        return abs(x.value()) < T(1e-10);
    }
    // (Some call sites pass a raw scalar; the library primary handles those.)

    // ---- kPow (integer exponent) --------------------------------------------
    // Rule 2/3: repeated multiplication in tracked land; no std::pow available
    // for tracked, so implement as a multiply loop.
    template <class TOutput, class TMass, class TScale, class T>
    inline ::tracked::Tracked<T> kPow(const ::tracked::Tracked<T>& base, const int& exponent) {
        const int n = exponent < 0 ? -exponent : exponent;
        // "one" is a named structural constant seeding the accumulator (Rule 5).
        ::tracked::Tracked<T> temp = ::tracked::constant<T>("one", T(1));
        for (int i = 0; i < n; ++i) temp = temp * base;
        if (exponent < 0) {
            auto one = ::tracked::constant<T>("one", T(1));
            return one / temp;
        }
        return temp;
    }
    template <class TOutput, class TMass, class TScale, class T>
    inline ::tracked::Complex<T> kPow(const ::tracked::Complex<T>& base, const int& exponent) {
        const int n = exponent < 0 ? -exponent : exponent;
        // Rule 3: complex accumulator built from tracked components.
        auto one_re = ::tracked::constant<T>("one",  T(1));
        auto zero_im= ::tracked::constant<T>("zero", T(0));
        ::tracked::Complex<T> temp(one_re, zero_im);
        for (int i = 0; i < n; ++i) temp = temp * base;
        if (exponent < 0) {
            ::tracked::Complex<T> one_c(::tracked::constant<T>("one", T(1)),
                                        ::tracked::constant<T>("zero", T(0)));
            return one_c / temp;
        }
        return temp;
    }

} // namespace ql

// UNCLASSIFIED items — none surfaced for the B14/B2m/B0m/B1m path exercised
// by this driver. The pruned dispatchers (QCDLOOP_BOX_FULL_DISPATCH is
// defined by boxGPU.h) mean B3m/B4m instantiations are NOT part of the
// statically compiled call graph here, so we do not need their auxiliary
// overloads (e.g. eta/xeta/xetatilde on tracked types) provided beyond what
// the shared Constants/Real/Imag/Sign/kAbs/kLog/kSqrt/kPow surface covers.