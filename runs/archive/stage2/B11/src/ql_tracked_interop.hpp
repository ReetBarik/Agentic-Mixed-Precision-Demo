// ql_tracked_interop.hpp
// Tracked interop shim for QCDLoop+Kokkos box integrals.
//
// SOURCE_HASH: 551c835fad3d9551d32c5ef4332d393770cb8055772c6a865bd4be0d3a0dd06f
//
// This shim makes ql::BO callable with T = tracked::Tracked<double>
// (TMass, TScale) and tracked::Complex<double> (TOutput). It provides:
//   - ql::Constants<Tracked<T>> partial specialization        (Rule 5 / C5)
//   - ql::Constants<Complex<T>> partial specialization        (Rule 5 / C5)
//   - Tracked-typed overloads for ql::Real/Imag/Sign/kAbs/
//     kLog/kSqrt/kConj/Max/Min/Htheta/iszero/kPow/cLn/Lnrat/
//     ratreal/ratgam/kfn                                      (Rules 1/2/3, C6, C7)
//   - Complex<T> overloads for Real/Imag/Sign/kAbs/kLog/
//     kSqrt/kConj/Max/Min                                     (Rules 1/2/3, C7)
//   - Missing operators for Tracked types (unary +)           (C3)
//
// Include order (from driver): this header MUST come before any qcdloop
// header, because qcdloop's templates make QUALIFIED ql::foo(x) calls that
// bypass ADL; the tracked overloads must be visible at each template's
// definition point.

#pragma once

#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/complex.hpp>
#include <tracked/journal.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include <cmath>
#include <string>
#include <type_traits>

// ---------------------------------------------------------------------------
// Forward-declare ql::Constants inside the ql namespace so our
// partial specializations below parse before qcdloop's own header supplies
// the primary definition. (C5)
// ---------------------------------------------------------------------------
namespace ql {
    template <typename T> struct Constants;   // C5: forward decl of library primary
}

// ---------------------------------------------------------------------------
// C3: Missing operators the tracked API doesn't itself define but which
// qcdloop templates may instantiate. Provide them as free functions in
// namespace tracked so ADL / direct-call resolution finds them.
// Identity operations emit no journal record.
// ---------------------------------------------------------------------------
namespace tracked {

    // C3: unary operator+ on Tracked<T> — identity, no journal record.
    template <class T>
    inline Tracked<T> operator+(const Tracked<T>& a) {
        return a;
    }

    // C3: unary operator+ on Complex<T> — identity, no journal record.
    template <class T>
    inline Complex<T> operator+(const Complex<T>& a) {
        return a;
    }

} // namespace tracked

// ---------------------------------------------------------------------------
// ql:: shim overloads
// ---------------------------------------------------------------------------
namespace ql {

    // ======================================================================
    // Rule 5 / C5: ql::Constants<T> partial specialization on the tracked
    // scalar. Mirrors every accessor the library's call graph reaches; each
    // named leaf scalar routes through tracked::constant("<name>", T(v)) so
    // the constant keeps its name in the journal. Static (non-template)
    // members return tracked::Tracked<T>; the templated accessors return the
    // tracked scalar too.
    // ======================================================================
    template <class T>
    struct Constants<tracked::Tracked<T>> {

        using Tk = tracked::Tracked<T>;

        // Rule 5: number of Chebyshev / Bernoulli coefficients — plain int,
        // used only as loop bound (Rule 1 / C6: discrete count).
        KOKKOS_INLINE_FUNCTION
        static constexpr int _num_C() { return 19; }
        KOKKOS_INLINE_FUNCTION
        static constexpr int _num_B() { return 25; }

        // Rule 5: named Chebyshev coefficient — floating-point, participates
        // in tracked arithmetic. Wrap via tracked::constant so its identity
        // survives into the journal.
        static Tk _C(int i) {
            static const double coeffs[19] = {
                0.4299669356081370, 0.4097598753307711, -0.0185884366501460,
                0.0014575108406227, -0.0001430418444234, 0.0000158841554188,
                -0.0000019078495939, 0.0000002419518085, -0.0000000319334127,
                0.0000000043454506, -0.0000000006057848, 0.0000000000861210,
                -0.0000000000124433, 0.0000000000018226, -0.0000000000002701,
                0.0000000000000404, -0.0000000000000061, 0.0000000000000009,
                -0.0000000000000001
            };
            return tracked::constant<T>("C[" + std::to_string(i) + "]", T(coeffs[i]));
        }

        static Tk _B(int i) {
            static const double coeffs[25] = {
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
            return tracked::constant<T>("B[" + std::to_string(i) + "]", T(coeffs[i]));
        }

        // Rule 5: onshell cutoff, named constant.
        template <typename TOutput, typename TMass, typename TScale>
        static Tk _qlonshellcutoff() {
            return tracked::constant<T>("qlonshellcutoff", T(1e-10));
        }

        // Rule 5: named mathematical constants. Each is wrapped once with a
        // stable name so the journal reflects the source identifier.
        static Tk _pi()   { return tracked::constant<T>("pi",   T(M_PI)); }
        static Tk _pi2()  { return tracked::constant<T>("pi2",  T(M_PI) * T(M_PI)); }

        template <typename TOutput, typename TMass, typename TScale>
        static Tk _pio3() { return tracked::constant<T>("pio3", T(M_PI) / T(3)); }

        template <typename TOutput, typename TMass, typename TScale>
        static Tk _pio6() { return tracked::constant<T>("pio6", T(M_PI) / T(6)); }

        template <typename TOutput, typename TMass, typename TScale>
        static Tk _pi2o3() { return tracked::constant<T>("pi2o3", T(M_PI) * T(M_PI) / T(3)); }

        template <typename TOutput, typename TMass, typename TScale>
        static Tk _pi2o6() { return tracked::constant<T>("pi2o6", T(M_PI) * T(M_PI) / T(6)); }

        template <typename TOutput, typename TMass, typename TScale>
        static Tk _pi2o12() { return tracked::constant<T>("pi2o12", T(M_PI) * T(M_PI) / T(12)); }

        static Tk _zero()  { return tracked::constant<T>("zero",  T(0.0)); }
        static Tk _half()  { return tracked::constant<T>("half",  T(0.5)); }
        static Tk _one()   { return tracked::constant<T>("one",   T(1.0)); }
        static Tk _two()   { return tracked::constant<T>("two",   T(2.0)); }
        static Tk _three() { return tracked::constant<T>("three", T(3.0)); }
        static Tk _four()  { return tracked::constant<T>("four",  T(4.0)); }
        static Tk _five()  { return tracked::constant<T>("five",  T(5.0)); }
        static Tk _six()   { return tracked::constant<T>("six",   T(6.0)); }
        static Tk _ten()   { return tracked::constant<T>("ten",   T(10.0)); }

        static Tk _eps()    { return tracked::constant<T>("eps",    T(1e-6)); }
        static Tk _eps4()   { return tracked::constant<T>("eps4",   T(1e-4)); }
        static Tk _eps7()   { return tracked::constant<T>("eps7",   T(1e-7)); }
        static Tk _eps10()  { return tracked::constant<T>("eps10",  T(1e-10)); }
        static Tk _eps14()  { return tracked::constant<T>("eps14",  T(1e-14)); }
        static Tk _eps15()  { return tracked::constant<T>("eps15",  T(1e-15)); }
        static Tk _xloss()  { return tracked::constant<T>("xloss",  T(0.125)); }
        static Tk _neglig() { return tracked::constant<T>("neglig", T(1e-14)); }
        static Tk _reps()   { return tracked::constant<T>("reps",   T(1e-16)); }

        // Rule 3 / C1: complex-valued named constants return
        // tracked::Complex<T> (NOT Tracked<Complex<T>>).
        template <typename TOutput, typename TMass, typename TScale>
        static TOutput _2ipi() {
            // TOutput is tracked::Complex<T> in the tracked instantiation.
            return TOutput(
                tracked::constant<T>("zero", T(0)),
                tracked::constant<T>("two_pi", T(2) * T(M_PI))
            );
        }

        template <typename TOutput, typename TMass, typename TScale>
        static TOutput _ipio2() {
            return TOutput(
                tracked::constant<T>("zero", T(0)),
                tracked::constant<T>("pi_o2", T(M_PI) * T(0.5))
            );
        }

        template <typename TOutput, typename TMass, typename TScale>
        static TOutput _ipi() {
            return TOutput(
                tracked::constant<T>("zero", T(0)),
                tracked::constant<T>("pi", T(M_PI))
            );
        }

        template <typename TOutput, typename TMass, typename TScale>
        static TOutput _ieps() {
            return TOutput(
                tracked::constant<T>("zero", T(0)),
                tracked::constant<T>("reps", T(1e-16))
            );
        }

        template <typename TOutput, typename TMass, typename TScale>
        static TOutput _ieps2() {
            return TOutput(
                tracked::constant<T>("zero", T(0)),
                tracked::constant<T>("reps2", T(1e-16) * T(1e-16))
            );
        }

        template <typename TOutput, typename TMass, typename TScale>
        static TOutput _ieps50() {
            return TOutput(
                tracked::constant<T>("zero", T(0)),
                tracked::constant<T>("ieps50", T(1e-50))
            );
        }
    };

    // ======================================================================
    // Rule 5 / C5: ql::Constants<Complex<T>> partial specialization. Some
    // library sites request Constants<TOutput>::_one() / _half() / _two() /
    // _zero() etc. where TOutput is the complex type. Return the tracked
    // complex container (Rule 3 / C1), routing the real component through a
    // named constant.
    // ======================================================================
    template <class T>
    struct Constants<tracked::Complex<T>> {

        using Ck = tracked::Complex<T>;

        // Rule 1 / C6: coefficient counts are discrete loop bounds.
        KOKKOS_INLINE_FUNCTION
        static constexpr int _num_C() { return 19; }
        KOKKOS_INLINE_FUNCTION
        static constexpr int _num_B() { return 25; }

        // Rule 3 / C1: scalar-valued complex constants — the real component
        // is a named tracked scalar, the imaginary component is a padding
        // literal (per Complex ctor semantics).
        static Ck _zero()  { return Ck(tracked::constant<T>("zero",  T(0.0))); }
        static Ck _half()  { return Ck(tracked::constant<T>("half",  T(0.5))); }
        static Ck _one()   { return Ck(tracked::constant<T>("one",   T(1.0))); }
        static Ck _two()   { return Ck(tracked::constant<T>("two",   T(2.0))); }
        static Ck _three() { return Ck(tracked::constant<T>("three", T(3.0))); }
        static Ck _four()  { return Ck(tracked::constant<T>("four",  T(4.0))); }
        static Ck _five()  { return Ck(tracked::constant<T>("five",  T(5.0))); }
        static Ck _six()   { return Ck(tracked::constant<T>("six",   T(6.0))); }
        static Ck _ten()   { return Ck(tracked::constant<T>("ten",   T(10.0))); }

        static Ck _pi()   { return Ck(tracked::constant<T>("pi",   T(M_PI))); }
        static Ck _pi2()  { return Ck(tracked::constant<T>("pi2",  T(M_PI) * T(M_PI))); }

        // Rule 5: complex-typed i*pi constants — imaginary component named.
        template <typename TOutput, typename TMass, typename TScale>
        static Ck _2ipi() {
            return Ck(
                tracked::constant<T>("zero",   T(0)),
                tracked::constant<T>("two_pi", T(2) * T(M_PI))
            );
        }

        template <typename TOutput, typename TMass, typename TScale>
        static Ck _ipio2() {
            return Ck(
                tracked::constant<T>("zero",  T(0)),
                tracked::constant<T>("pi_o2", T(M_PI) * T(0.5))
            );
        }

        template <typename TOutput, typename TMass, typename TScale>
        static Ck _ipi() {
            return Ck(
                tracked::constant<T>("zero", T(0)),
                tracked::constant<T>("pi",   T(M_PI))
            );
        }

        template <typename TOutput, typename TMass, typename TScale>
        static Ck _ieps() {
            return Ck(
                tracked::constant<T>("zero", T(0)),
                tracked::constant<T>("reps", T(1e-16))
            );
        }

        template <typename TOutput, typename TMass, typename TScale>
        static Ck _ieps2() {
            return Ck(
                tracked::constant<T>("zero",  T(0)),
                tracked::constant<T>("reps2", T(1e-16) * T(1e-16))
            );
        }

        template <typename TOutput, typename TMass, typename TScale>
        static Ck _ieps50() {
            return Ck(
                tracked::constant<T>("zero",   T(0)),
                tracked::constant<T>("ieps50", T(1e-50))
            );
        }

        // Rule 5: pi-scaled reals as complex — real component named.
        template <typename TOutput, typename TMass, typename TScale>
        static Ck _pi2o6() {
            return Ck(tracked::constant<T>("pi2o6", T(M_PI) * T(M_PI) / T(6)));
        }
        template <typename TOutput, typename TMass, typename TScale>
        static Ck _pi2o12() {
            return Ck(tracked::constant<T>("pi2o12", T(M_PI) * T(M_PI) / T(12)));
        }
        template <typename TOutput, typename TMass, typename TScale>
        static Ck _pi2o3() {
            return Ck(tracked::constant<T>("pi2o3", T(M_PI) * T(M_PI) / T(3)));
        }
        template <typename TOutput, typename TMass, typename TScale>
        static Ck _pio3() {
            return Ck(tracked::constant<T>("pio3", T(M_PI) / T(3)));
        }
        template <typename TOutput, typename TMass, typename TScale>
        static Ck _pio6() {
            return Ck(tracked::constant<T>("pio6", T(M_PI) / T(6)));
        }
    };

    // ======================================================================
    // Rule 1 / C6: Real / Imag return the raw scalar the library consumes as
    // a plain floating-point number (assigned to TScale, compared, or fed
    // into further tracked arithmetic). Because the *use* is floating-point
    // (C6), these must return tracked::Tracked<T>, NOT bare double — the
    // library assigns their result into TScale variables which are the
    // tracked scalar itself.
    // ======================================================================

    // Rule 2 / C6: Real of a tracked scalar is the tracked value itself
    // (identity for real inputs).
    template <class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> Real(const tracked::Tracked<T>& x) {
        return x;
    }

    // Rule 2 / C6: Imag of a tracked real scalar is a literal zero (padding).
    template <class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> Imag(const tracked::Tracked<T>& /*x*/) {
        return tracked::literal(T(0));
    }

    // Rule 2 / C6: Real / Imag of a tracked complex — return the tracked
    // real/imag components (they are themselves tracked scalars, per C1).
    template <class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> Real(const tracked::Complex<T>& z) {
        return z.real();
    }

    template <class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> Imag(const tracked::Complex<T>& z) {
        return z.imag();
    }

    // ======================================================================
    // Rule 2 / C6: Sign. In the library, Sign() is multiplied into tracked
    // arithmetic (e.g. `TOutput(ql::Sign(ql::Real(k14)))`), so the result
    // MUST flow as a tracked scalar to preserve provenance. The sign itself
    // is a runtime ±1/0, entered as an anonymous literal (see complex.hpp
    // sqrt() precedent).
    // ======================================================================
    template <class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> Sign(const tracked::Tracked<T>& x) {
        T v = x.value();
        T s = (T(0) < v) ? T(1) : ((v < T(0)) ? T(-1) : T(0));
        return tracked::literal(s);
    }

    // Rule 3 / C7: Sign of complex — z / |z|, returned as tracked complex
    // container (Rule 3). Result feeds back into complex arithmetic.
    template <class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Complex<T> Sign(const tracked::Complex<T>& z) {
        auto a = tracked::abs(z);
        return z / tracked::Complex<T>(a);
    }

    // ======================================================================
    // Rule 2: kAbs — floating-point return, participates in downstream
    // tracked arithmetic and comparisons.
    // ======================================================================
    template <class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x) {
        return tracked::abs(x);
    }

    // Rule 2 / C1: kAbs of complex — |z| is a real (tracked) scalar.
    template <class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> kAbs(const tracked::Complex<T>& z) {
        return tracked::abs(z);
    }

    // ======================================================================
    // Rule 2 / C7: elementary transcendental functions on tracked scalars
    // and tracked complex.
    // ======================================================================
    template <class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> kLog(const tracked::Tracked<T>& x) {
        return tracked::log(x);
    }

    template <class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Complex<T> kLog(const tracked::Complex<T>& z) {
        return tracked::log(z);
    }

    template <class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> kSqrt(const tracked::Tracked<T>& x) {
        return tracked::sqrt(x);
    }

    template <class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Complex<T> kSqrt(const tracked::Complex<T>& z) {
        return tracked::sqrt(z);
    }

    // Rule 2: kConj on a real tracked scalar is the identity.
    template <class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> kConj(const tracked::Tracked<T>& x) {
        return x;
    }

    // Rule 3: kConj on tracked complex — returns tracked complex.
    template <class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Complex<T> kConj(const tracked::Complex<T>& z) {
        return tracked::conj(z);
    }

    // ======================================================================
    // Rule 2 / C7: Max / Min — return by value using |a|>|b| comparison. The
    // comparison itself is a bool over unwrapped magnitudes (Rule 7), but
    // the RETURN participates in further tracked arithmetic, so the returned
    // type is the tracked scalar / complex (Rule 2 / Rule 3).
    // ======================================================================
    template <class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> Max(const tracked::Tracked<T>& a, const tracked::Tracked<T>& b) {
        // Rule 7: compare via .value() (through tracked::abs value).
        auto absa = tracked::abs(a);
        auto absb = tracked::abs(b);
        return (absa.value() > absb.value()) ? a : b;
    }

    template <class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> Min(const tracked::Tracked<T>& a, const tracked::Tracked<T>& b) {
        auto absa = tracked::abs(a);
        auto absb = tracked::abs(b);
        return (absa.value() > absb.value()) ? b : a;
    }

    template <class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Complex<T> Max(const tracked::Complex<T>& a, const tracked::Complex<T>& b) {
        auto absa = tracked::abs(a);
        auto absb = tracked::abs(b);
        return (absa.value() > absb.value()) ? a : b;
    }

    template <class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Complex<T> Min(const tracked::Complex<T>& a, const tracked::Complex<T>& b) {
        auto absa = tracked::abs(a);
        auto absb = tracked::abs(b);
        return (absa.value() > absb.value()) ? b : a;
    }

    // ======================================================================
    // Rule 2 / C6: Htheta returns a numeric 0/1 that flows into tracked
    // arithmetic — MUST return the tracked scalar to preserve provenance.
    // ======================================================================
    template <class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> Htheta(const tracked::Tracked<T>& x) {
        // 0.5*(1 + sign(x)) — done as tracked ops so the result is a
        // real tracked scalar with a full derivation chain.
        auto s = Sign(x);
        auto one = tracked::constant<T>("one", T(1));
        auto half = tracked::constant<T>("half", T(0.5));
        return half * (one + s);
    }

    // ======================================================================
    // Rule 1 / C6: iszero returns a plain bool — it is consumed *only* as
    // a discrete selector (`if (iszero(...))`, `!iszero(...) ? ... : ...`),
    // never mixed into arithmetic. Unwrap via .value() (Rule 7).
    // ======================================================================
    template <class TOutput, class TMass, class TScale, class T>
    KOKKOS_INLINE_FUNCTION
    bool iszero(const tracked::Tracked<T>& x) {
        // Threshold matches Constants<TScale>::_qlonshellcutoff = 1e-10.
        using std::abs;
        return abs(x.value()) < T(1e-10);
    }

    // Rule 1 / C6: iszero of a complex — used as a discrete selector.
    template <class TOutput, class TMass, class TScale, class T>
    KOKKOS_INLINE_FUNCTION
    bool iszero(const tracked::Complex<T>& z) {
        auto a = tracked::abs(z);
        return a.value() < T(1e-10);
    }

    // ======================================================================
    // Rule 2 / C7: kPow — integer exponent, tracked base. Implement as a
    // multiply loop since the tracked API defines no pow(). Constrain the
    // base type to the concrete tracked type so we win partial ordering
    // against the library's own kPow template (C7). Provide overloads for
    // both tracked scalar and tracked complex.
    // ======================================================================
    template <class TOutput, class TMass, class TScale, class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> kPow(const tracked::Tracked<T>& base, const int& exponent) {
        const int n = exponent < 0 ? -exponent : exponent;
        // Rule 5: seed with the named constant "one" so provenance names it.
        tracked::Tracked<T> temp = tracked::constant<T>("one", T(1));
        for (int i = 0; i < n; ++i) temp = temp * base;
        if (exponent < 0) {
            tracked::Tracked<T> one = tracked::constant<T>("one", T(1));
            return one / temp;
        }
        return temp;
    }

    template <class TOutput, class TMass, class TScale, class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Complex<T> kPow(const tracked::Complex<T>& base, const int& exponent) {
        const int n = exponent < 0 ? -exponent : exponent;
        // Seed as tracked complex constant one.
        tracked::Complex<T> temp(tracked::constant<T>("one", T(1)));
        for (int i = 0; i < n; ++i) temp = temp * base;
        if (exponent < 0) {
            tracked::Complex<T> one(tracked::constant<T>("one", T(1)));
            return one / temp;
        }
        return temp;
    }

    // ======================================================================
    // Rule 2 / C7: cLn — complex log with sign-of-imaginary branch handling.
    // Two overloads: (Complex, Tracked) and (Tracked, Tracked). Return
    // tracked complex (Rule 3).
    //
    // The library's own cLn is a function template with template params
    // <TOutput, TMass, TScale>; qualified calls pass all three explicitly.
    // Per C7, we carry those leading explicit params on our concrete-typed
    // overloads so `ql::cLn<A,B,C>(x, sig)` binds to us.
    // ======================================================================
    template <class TOutput, class TMass, class TScale, class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Complex<T> cLn(const tracked::Complex<T>& z, const tracked::Tracked<T>& isig) {
        // Rule 7: branch decision via unwrapped .value().
        if (z.imag().value() == T(0) && z.real().value() <= T(0)) {
            // log(-z) + i*pi*sign(isig)
            auto neg_z = -z;
            auto lg = tracked::log(neg_z);
            T sgn = (T(0) < isig.value()) ? T(1) : ((isig.value() < T(0)) ? T(-1) : T(0));
            auto sgn_lit = tracked::literal(sgn);
            auto pi = tracked::constant<T>("pi", T(M_PI));
            // i*pi*sign(isig) is imaginary; build a tracked complex.
            tracked::Complex<T> imag_term(tracked::literal(T(0)), pi * sgn_lit);
            return lg + imag_term;
        }
        return tracked::log(z);
    }

    template <class TOutput, class TMass, class TScale, class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Complex<T> cLn(const tracked::Tracked<T>& x, const tracked::Tracked<T>& isig) {
        if (x.value() > T(0)) {
            // Real positive log — real component only.
            return tracked::Complex<T>(tracked::log(x));
        } else {
            auto neg_x = -x;
            auto lg = tracked::log(neg_x);
            T sgn = (T(0) < isig.value()) ? T(1) : ((isig.value() < T(0)) ? T(-1) : T(0));
            auto sgn_lit = tracked::literal(sgn);
            auto pi = tracked::constant<T>("pi", T(M_PI));
            tracked::Complex<T> imag_term(tracked::literal(T(0)), pi * sgn_lit);
            return tracked::Complex<T>(lg) + imag_term;
        }
    }

    // ======================================================================
    // Rule 3 / C7: Lnrat — returns tracked complex.
    // Signatures we need per library primary:
    //   Lnrat<TOutput,TMass,TScale>(TOutput const&, TOutput const&)
    //   Lnrat<TOutput,TMass,TScale>(TScale  const&, TScale  const&)
    // In the tracked instantiation, TOutput == Complex<T> and TScale == TMass
    // == Tracked<T>. Provide both concrete shapes (C7).
    // ======================================================================
    template <class TOutput, class TMass, class TScale, class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Complex<T> Lnrat(const tracked::Complex<T>& x, const tracked::Complex<T>& y) {
        auto r = x / y;
        // Rule 7: examine the imag component's .value() for branch.
        if (r.imag().value() == T(0)) {
            auto abs_r = tracked::abs(r);
            auto lg = tracked::log(abs_r);  // real
            // ipio2 * (sign(-Re x) - sign(-Re y))
            T sx = -x.real().value();
            T sy = -y.real().value();
            T ssx = (T(0) < sx) ? T(1) : ((sx < T(0)) ? T(-1) : T(0));
            T ssy = (T(0) < sy) ? T(1) : ((sy < T(0)) ? T(-1) : T(0));
            auto diff = tracked::literal(ssx - ssy);
            auto pi_o2 = tracked::constant<T>("pi_o2", T(M_PI) * T(0.5));
            // ipio2 = (0, pi/2); subtract ipio2 * diff.
            auto imag_part = pi_o2 * diff;
            tracked::Complex<T> ipio2_diff(tracked::literal(T(0)), imag_part);
            return tracked::Complex<T>(lg) - ipio2_diff;
        }
        return tracked::log(r);
    }

    template <class TOutput, class TMass, class TScale, class T>
    KOKKOS_INLINE_FUNCTION
    tracked::Complex<T> Lnrat(const tracked::Tracked<T>& x, const tracked::Tracked<T>& y) {
        auto abs_ratio = tracked::abs(x / y);
        auto lg = tracked::log(abs_ratio);
        T sx = -x.value();
        T sy = -y.value();
        T ssx = (T(0) < sx) ? T(1) : ((sx < T(0)) ? T(-1) : T(0));
        T ssy = (T(0) < sy) ? T(1) : ((sy < T(0)) ? T(-1) : T(0));
        auto diff = tracked::literal(ssx - ssy);
        auto pi_o2 = tracked::constant<T>("pi_o2", T(M_PI) * T(0.5));
        auto imag_part = pi_o2 * diff;
        tracked::Complex<T> ipio2_diff(tracked::literal(T(0)), imag_part);
        return tracked::Complex<T>(lg) - ipio2_diff;
    }

    // ======================================================================
    // Rule 2 / C7: ratreal — writes rat and ieps by reference. The tracked
    // signature mirrors the library primary; TMass == TScale == Tracked<T>
    // in the tracked instantiation.
    // ======================================================================
    template <class TOutput, class TMass, class TScale, class T>
    KOKKOS_INLINE_FUNCTION
    void ratreal(const tracked::Tracked<T>& si,
                 const tracked::Tracked<T>& ta,
                 tracked::Tracked<T>& rat,
                 tracked::Tracked<T>& ieps) {
        rat = si / ta;
        // Rule 7: branch via unwrapped values.
        if (rat.value() > T(0)) {
            ieps = tracked::literal(T(0));
        } else if (si.value() < T(0)) {
            ieps = tracked::literal(T(-1));
        } else if (ta.value() < T(0)) {
            ieps = tracked::literal(T(1));
        } else if (ta.value() == T(0)) {
            Kokkos::printf("error in ratreal\n");
            ieps = tracked::literal(T(0));
        } else {
            ieps = tracked::literal(T(0));
        }
    }

    // ======================================================================
    // Rule 2 / Rule 3 / C7: ratgam — writes complex ratp, ratm and scalar
    // ieps by reference. Signature mirrors library primary with all-tracked
    // arguments.
    // ======================================================================
    template <class TOutput, class TMass, class TScale, class T>
    KOKKOS_INLINE_FUNCTION
    void ratgam(tracked::Complex<T>& ratp,
                tracked::Complex<T>& ratm,
                tracked::Tracked<T>& ieps,
                const tracked::Tracked<T>& p3sq,
                const tracked::Tracked<T>& m3sq,
                const tracked::Tracked<T>& m4sq) {
        // root = sqrt((p3sq - m3sq + m4sq)^2 - 4*m4sq*p3sq), promoted to complex.
        auto diff = p3sq - m3sq + m4sq;
        auto diff2 = diff * diff;
        auto four = tracked::constant<T>("four", T(4));
        auto disc = diff2 - four * m4sq * p3sq;
        // Real->Complex promotion then sqrt.
        tracked::Complex<T> disc_c(disc);
        auto root = tracked::sqrt(disc_c);

        auto num_re = p3sq + m4sq - m3sq;
        auto den_re = tracked::neg(p3sq) + m4sq - m3sq;

        tracked::Complex<T> num_c(num_re);
        tracked::Complex<T> den_c(den_re);

        ratp = (num_c + root) / (den_c + root);
        ratm = (num_c - root) / (den_c - root);
        ieps = tracked::literal(T(0));
    }

    // ======================================================================
    // Rule 2 / Rule 3 / C7: kfn — writes a 3-array of tracked complex plus a
    // tracked scalar ieps. Signature mirrors library primary; used by B14
    // / B15 / B16. Provide the tracked overload so qualified calls resolve
    // here. This overload is required to exist by static instantiation of
    // the B1m..B3m dispatchers even if the test kinematics never reach it
    // (C3 static-instantiation rule).
    // ======================================================================
    template <class TOutput, class TMass, class TScale, class T>
    KOKKOS_INLINE_FUNCTION
    void kfn(Kokkos::Array<tracked::Complex<T>, 3>& res,
             tracked::Tracked<T>& ieps,
             const tracked::Tracked<T>& xpi,
             const tracked::Tracked<T>& xm,
             const tracked::Tracked<T>& xmp) {
        if (xm.value() == T(0) || xmp.value() == T(0)) {
            Kokkos::printf("Error in kfn,xm,xmp");
        }
        auto one_c  = tracked::constant<T>("one", T(1));
        auto two_c  = tracked::constant<T>("two", T(2));
        auto four_c = tracked::constant<T>("four", T(4));

        auto diff = xm - xmp;
        auto xx1 = xpi - diff * diff;
        auto rat = xx1 / (four_c * xm * xmp);

        // Rule 7: branch via unwrapped value.
        if (rat.value() == T(0)) {
            // res[1] = -2 * sqrt(rat) * i + 2*rat
            tracked::Complex<T> rat_c(rat);
            auto srat = tracked::sqrt(rat_c);
            tracked::Complex<T> two_i(tracked::literal(T(0)), two_c);
            auto rat2 = two_c * rat;
            tracked::Complex<T> rat2_c(rat2);
            res[1] = tracked::Complex<T>(tracked::literal(T(0)), tracked::literal(T(0))) - two_i * srat + rat2_c;
            tracked::Complex<T> one_cc(one_c);
            tracked::Complex<T> two_cc(two_c);
            res[0] = one_cc - res[1];
            res[2] = two_cc - res[1];
        } else {
            auto inner = (rat - one_c) / rat;
            tracked::Complex<T> inner_c(inner);
            auto root = tracked::sqrt(inner_c);
            tracked::Complex<T> one_cc(one_c);
            auto invopr = one_cc / (one_cc + root);
            tracked::Complex<T> rat_c(rat);
            res[0] = -(invopr * invopr) / rat_c;
            tracked::Complex<T> two_cc(two_c);
            res[1] = two_cc * invopr;
            res[2] = two_cc * root * invopr;
        }
        ieps = tracked::literal(T(1));
    }

    // ======================================================================
    // Note on execution-space annotations (C4):
    // The driver invokes ql::BO from a plain host loop (not inside a
    // Kokkos::parallel_for or CUDA kernel launch). Tracked ops are host-only
    // (they use std::string / thread-local journals). Per C4, this means the
    // shim's overloads carry KOKKOS_INLINE_FUNCTION only for lexical
    // compatibility with the library's own annotated templates (which forces
    // uniform annotation on any overload they call); no device-side
    // execution is intended or supported.
    // ======================================================================

} // namespace ql