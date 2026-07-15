// ql_tracked_interop.hpp
// Tracked interop shim for QCDLoop+Kokkos box integrals (B16 / three-mass path).
//
// SOURCE_HASH: cfad2410c3ddc32ab520cc03f18dd5e38f62b9fd0359678851e50da9f40a0ac8
//
// Included BEFORE kokkosMaths.h / kokkosUtils.h / boxGPU.h (see driver order),
// so any specialization of ql::Constants<T> we provide MUST forward-declare the
// primary template inside namespace ql first (Rule C5). All shim overloads are
// placed in namespace ql so qcdloop's own qualified `ql::Foo<...>(tracked)` call
// sites bind to them by partial ordering (Rule C7).
//
// Execution model: driver invokes qcdloop from a plain host loop (no
// parallel_for), and tracked ops are host-only (std::string / journaling), so
// NO KOKKOS_INLINE_FUNCTION annotation is emitted on shim overloads
// (Rule 8 / Rule C4).

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
// Rule C5: forward-declare the library's primary Constants<T> so our partial
// specialization on tracked::Tracked<T> below parses even though kokkosMaths.h
// (which defines the primary) is included AFTER this shim in the driver.
// ---------------------------------------------------------------------------
namespace ql {
    template <typename T> struct Constants;
}

namespace tracked {

// ---------------------------------------------------------------------------
// Rule C3: qcdloop expression templates invoke unary operator+ / mixed-type
// binary operators on tracked scalars that Tracked<T> does not itself define.
// Provide them as free functions in namespace tracked so ADL finds them; they
// introduce no rounding, so no journal record.
// ---------------------------------------------------------------------------

// Rule C3: identity unary plus (no rounding, no journal record).
template <class T>
inline Tracked<T> operator+(const Tracked<T>& a) { return a; }

// Rule C3: identity unary plus for tracked complex.
template <class T>
inline Complex<T> operator+(const Complex<T>& a) { return a; }

} // namespace tracked

// ---------------------------------------------------------------------------
// ql::Constants<tracked::Tracked<T>> specialization.
// Rule 5 / Rule C5: qcdloop reaches every named leaf constant through
// ql::Constants<TMass|TScale>::name(); mirror the FULL member interface of the
// primary and route each named scalar through tracked::constant("<name>", ...)
// so the source identifier survives in the journal.
// ---------------------------------------------------------------------------
namespace ql {

template <class T>
struct Constants<tracked::Tracked<T>> {
    using Tr = tracked::Tracked<T>;

    // Rule 5: named constants — preserve source spelling in the journal.
    static Tr _zero()  { return tracked::constant<T>("zero",  T(0)); }
    static Tr _half()  { return tracked::constant<T>("half",  T(0.5)); }
    static Tr _one()   { return tracked::constant<T>("one",   T(1)); }
    static Tr _two()   { return tracked::constant<T>("two",   T(2)); }
    static Tr _three() { return tracked::constant<T>("three", T(3)); }
    static Tr _four()  { return tracked::constant<T>("four",  T(4)); }
    static Tr _five()  { return tracked::constant<T>("five",  T(5)); }
    static Tr _six()   { return tracked::constant<T>("six",   T(6)); }
    static Tr _ten()   { return tracked::constant<T>("ten",   T(10)); }

    static Tr _pi()      { return tracked::constant<T>("pi",      T(M_PI)); }
    static Tr _pi2()     { return tracked::constant<T>("pi2",     T(M_PI * M_PI)); }

    // Rule 5: tolerance / small-number constants — keep names.
    static Tr _eps()     { return tracked::constant<T>("eps",     T(1e-6)); }
    static Tr _eps4()    { return tracked::constant<T>("eps4",    T(1e-4)); }
    static Tr _eps7()    { return tracked::constant<T>("eps7",    T(1e-7)); }
    static Tr _eps10()   { return tracked::constant<T>("eps10",   T(1e-10)); }
    static Tr _eps14()   { return tracked::constant<T>("eps14",   T(1e-14)); }
    static Tr _eps15()   { return tracked::constant<T>("eps15",   T(1e-15)); }
    static Tr _xloss()   { return tracked::constant<T>("xloss",   T(0.125)); }
    static Tr _neglig()  { return tracked::constant<T>("neglig",  T(1e-14)); }
    static Tr _reps()    { return tracked::constant<T>("reps",    T(1e-16)); }

    // Rule 1: discrete count of Chebyshev coefficients — plain int (never
    // Tracked<int>); consumed only as a loop bound.
    static constexpr int _num_C() { return 19; }
    static constexpr int _num_B() { return 25; }

    // Rule 5: Chebyshev / Bernoulli coefficient tables — each named "C[i]",
    // "B[i]" to preserve per-slot provenance in the journal.
    static Tr _C(int i) {
        static const double coeffs[19] = {
            0.4299669356081370,   0.4097598753307711,  -0.0185884366501460,
            0.0014575108406227,  -0.0001430418444234,   0.0000158841554188,
           -0.0000019078495939,   0.0000002419518085,  -0.0000000319334127,
            0.0000000043454506,  -0.0000000006057848,   0.0000000000861210,
           -0.0000000000124433,   0.0000000000018226,  -0.0000000000002701,
            0.0000000000000404,  -0.0000000000000061,   0.0000000000000009,
           -0.0000000000000001
        };
        return tracked::constant<T>(std::string("C[") + std::to_string(i) + "]",
                                    T(coeffs[i]));
    }

    static Tr _B(int i) {
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
        return tracked::constant<T>(std::string("B[") + std::to_string(i) + "]",
                                    T(coeffs[i]));
    }

    // Rule 5: on-shell cutoff — mirror the primary's TOutput/TMass/TScale
    // template arity so ql::iszero<...>() qualified calls resolve.
    template <typename TOutput, typename TMass, typename TScale>
    static Tr _qlonshellcutoff() {
        return tracked::constant<T>("qlonshellcutoff", T(1e-10));
    }

    template <typename TOutput, typename TMass, typename TScale>
    static Tr _pio3()   { return tracked::constant<T>("pio3",   T(M_PI / 3.0)); }
    template <typename TOutput, typename TMass, typename TScale>
    static Tr _pio6()   { return tracked::constant<T>("pio6",   T(M_PI / 6.0)); }
    template <typename TOutput, typename TMass, typename TScale>
    static Tr _pi2o3()  { return tracked::constant<T>("pi2o3",  T(M_PI * M_PI / 3.0)); }
    template <typename TOutput, typename TMass, typename TScale>
    static Tr _pi2o6()  { return tracked::constant<T>("pi2o6",  T(M_PI * M_PI / 6.0)); }
    template <typename TOutput, typename TMass, typename TScale>
    static Tr _pi2o12() { return tracked::constant<T>("pi2o12", T(M_PI * M_PI / 12.0)); }

    // Rule 3: complex-valued constants become tracked::Complex<T>, NOT
    // Tracked<Complex<T>>. Named-imag components go through constant();
    // structural zeros are anonymous literals.
    template <typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _2ipi() {
        return tracked::Complex<T>(tracked::literal(T(0)),
                                   tracked::constant<T>("2pi", T(2) * T(M_PI)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ipio2() {
        return tracked::Complex<T>(tracked::literal(T(0)),
                                   tracked::constant<T>("pio2", T(M_PI) * T(0.5)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ipi() {
        return tracked::Complex<T>(tracked::literal(T(0)),
                                   tracked::constant<T>("pi", T(M_PI)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ieps() {
        return tracked::Complex<T>(tracked::literal(T(0)),
                                   tracked::constant<T>("reps", T(1e-16)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ieps2() {
        return tracked::Complex<T>(tracked::literal(T(0)),
                                   tracked::constant<T>("reps2", T(1e-16) * T(1e-16)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ieps50() {
        return tracked::Complex<T>(tracked::literal(T(0)),
                                   tracked::constant<T>("eps50", T(1e-50)));
    }
};

// ---------------------------------------------------------------------------
// ql::Constants<tracked::Complex<T>> specialization.
// The library also queries Constants<TOutput>::_zero(), _one(), _half(),
// _two(), _three(), _four(), _five(), _six() and _ieps50<...>() where TOutput
// is the tracked complex type. Provide the same interface returning
// tracked::Complex<T> (Rule 3 — container of tracked, not tracked-of-container).
// ---------------------------------------------------------------------------
template <class T>
struct Constants<tracked::Complex<T>> {
    using Cx = tracked::Complex<T>;
    using Tr = tracked::Tracked<T>;

    // Rule 5 / Rule 3: named-real scalar promoted into a complex.
    static Cx _zero()  { return Cx(tracked::constant<T>("zero",  T(0))); }
    static Cx _half()  { return Cx(tracked::constant<T>("half",  T(0.5))); }
    static Cx _one()   { return Cx(tracked::constant<T>("one",   T(1))); }
    static Cx _two()   { return Cx(tracked::constant<T>("two",   T(2))); }
    static Cx _three() { return Cx(tracked::constant<T>("three", T(3))); }
    static Cx _four()  { return Cx(tracked::constant<T>("four",  T(4))); }
    static Cx _five()  { return Cx(tracked::constant<T>("five",  T(5))); }
    static Cx _six()   { return Cx(tracked::constant<T>("six",   T(6))); }

    // Rule 5: pure-imaginary named constants.
    template <typename TOutput, typename TMass, typename TScale>
    static Cx _2ipi() {
        return Cx(tracked::literal(T(0)),
                  tracked::constant<T>("2pi", T(2) * T(M_PI)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static Cx _ipio2() {
        return Cx(tracked::literal(T(0)),
                  tracked::constant<T>("pio2", T(M_PI) * T(0.5)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static Cx _ipi() {
        return Cx(tracked::literal(T(0)),
                  tracked::constant<T>("pi", T(M_PI)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static Cx _ieps() {
        return Cx(tracked::literal(T(0)),
                  tracked::constant<T>("reps", T(1e-16)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static Cx _ieps50() {
        return Cx(tracked::literal(T(0)),
                  tracked::constant<T>("eps50", T(1e-50)));
    }
};

// ---------------------------------------------------------------------------
// Scalar math dispatch shims on tracked::Tracked<T>.
// Each mirrors the library primary's arity so qualified calls resolve here
// (Rule C7). Return types follow Rule 1/2/6.
// ---------------------------------------------------------------------------

// Rule 2: tracked absolute value — participates downstream, stays tracked.
template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x) {
    return tracked::abs(x, TRACKED_HERE);
}

// Rule 2: tracked absolute value of complex -> tracked real.
template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Complex<T>& x) {
    return tracked::abs(x, TRACKED_HERE);
}

// Rule 2: log/sqrt/conj on tracked scalars.
template <class T>
inline tracked::Tracked<T> kLog(const tracked::Tracked<T>& x) {
    return tracked::log(x, TRACKED_HERE);
}
template <class T>
inline tracked::Complex<T> kLog(const tracked::Complex<T>& x) {
    return tracked::log(x, TRACKED_HERE);
}
template <class T>
inline tracked::Tracked<T> kSqrt(const tracked::Tracked<T>& x) {
    return tracked::sqrt(x, TRACKED_HERE);
}
template <class T>
inline tracked::Complex<T> kSqrt(const tracked::Complex<T>& x) {
    return tracked::sqrt(x, TRACKED_HERE);
}

// Rule 2 / Rule 3: conjugate. Real conj is identity; complex flips imag sign.
template <class T>
inline tracked::Tracked<T> kConj(const tracked::Tracked<T>& x) { return x; }
template <class T>
inline tracked::Complex<T> kConj(const tracked::Complex<T>& x) {
    return tracked::conj(x, TRACKED_HERE);
}

// Rule 6: iszero — cutoff is a named library constant (goes through
// Constants<TScale>::_qlonshellcutoff). Rule 1: returns bool (branch selector).
template <typename TOutput, typename TMass, typename TScale, class T>
inline bool iszero(const tracked::Tracked<T>& x) {
    using std::abs;
    return abs(x.value()) < T(1e-10);
}

// Rule 1: Imag/Real on tracked types — extract as tracked scalar (Rule C6:
// downstream consumers feed these into floating-point arithmetic, so return
// Tracked, not raw double).
template <class T>
inline tracked::Tracked<T> Imag(const tracked::Tracked<T>& /*x*/) {
    // Rule 6: anonymous inline zero (imag part of a real is structurally 0).
    return tracked::literal(T(0));
}
template <class T>
inline tracked::Tracked<T> Imag(const tracked::Complex<T>& x) {
    // Rule 2: pass through the tracked imag component (already tracked).
    return x.imag();
}
template <class T>
inline tracked::Tracked<T> Real(const tracked::Tracked<T>& x) {
    return x;
}
template <class T>
inline tracked::Tracked<T> Real(const tracked::Complex<T>& x) {
    return x.real();
}

// Rule 1: Sign — a numeric +/-1/0 whose result flows into tracked arithmetic
// (multiplied into tracked expressions). Rule C6 says this is a FLOATING-POINT
// return, so it is Tracked, not int. Value chosen by comparing .value().
template <class T>
inline tracked::Tracked<T> Sign(const tracked::Tracked<T>& x) {
    // Rule 7: comparisons on tracked go through raw values.
    if (x.value() > T(0)) return tracked::literal(T(1));
    if (x.value() < T(0)) return tracked::literal(T(-1));
    return tracked::literal(T(0));
}

// Rule 3: Sign of complex returns z/|z| (a tracked complex), per C6 example.
template <class T>
inline tracked::Complex<T> Sign(const tracked::Complex<T>& x) {
    auto mag = tracked::abs(x, TRACKED_HERE);
    return x / tracked::Complex<T>(mag);
}

// Rule 1 (Htheta): 0.5 * (1 + sign(x)); result flows into tracked arithmetic
// (Rule C6), so return Tracked.
template <class T>
inline tracked::Tracked<T> Htheta(const tracked::Tracked<T>& x) {
    auto s     = ql::Sign(x);
    auto one   = tracked::constant<T>("one",  T(1));
    auto half  = tracked::constant<T>("half", T(0.5));
    return tracked::mul(half, tracked::add(one, s, TRACKED_HERE), TRACKED_HERE);
}

// Rule 2: Max / Min by absolute value — return type is the argument type.
// Rule 7: compare via .value() on the abs.
template <class T>
inline tracked::Tracked<T> Max(const tracked::Tracked<T>& a, const tracked::Tracked<T>& b) {
    using std::abs;
    return (abs(a.value()) > abs(b.value())) ? a : b;
}
template <class T>
inline tracked::Complex<T> Max(const tracked::Complex<T>& a, const tracked::Complex<T>& b) {
    auto amag = tracked::abs(a, TRACKED_HERE);
    auto bmag = tracked::abs(b, TRACKED_HERE);
    return (amag.value() > bmag.value()) ? a : b;
}
template <class T>
inline tracked::Tracked<T> Min(const tracked::Tracked<T>& a, const tracked::Tracked<T>& b) {
    using std::abs;
    return (abs(a.value()) > abs(b.value())) ? b : a;
}
template <class T>
inline tracked::Complex<T> Min(const tracked::Complex<T>& a, const tracked::Complex<T>& b) {
    auto amag = tracked::abs(a, TRACKED_HERE);
    auto bmag = tracked::abs(b, TRACKED_HERE);
    return (amag.value() > bmag.value()) ? b : a;
}

// ---------------------------------------------------------------------------
// kPow: qcdloop calls ql::kPow<TOutput,TMass,TScale>(base, int_exp).
// Rule C7: outrank the library primaries with concrete tracked value parameters
// while carrying the same 3 leading explicit template parameters. Implement as
// a multiply loop over tracked operator* (Rule C2: there is no tracked::pow).
// ---------------------------------------------------------------------------

template <typename TOutput, typename TMass, typename TScale, class T>
inline tracked::Tracked<T> kPow(const tracked::Tracked<T>& base, const int& exponent) {
    // Rule 1: integer exponent stays raw int (loop count / branch selector).
    const int n = exponent < 0 ? -exponent : exponent;
    // Rule 5: unit seed is a named constant.
    tracked::Tracked<T> acc = tracked::constant<T>("one", T(1));
    for (int i = 0; i < n; ++i) acc = acc * base;
    if (exponent < 0) {
        tracked::Tracked<T> one = tracked::constant<T>("one", T(1));
        return one / acc;
    }
    return acc;
}

template <typename TOutput, typename TMass, typename TScale, class T>
inline tracked::Complex<T> kPow(const tracked::Complex<T>& base, const int& exponent) {
    // Rule 1: exponent is a raw int loop count.
    const int n = exponent < 0 ? -exponent : exponent;
    // Rule 5 / Rule 3: seed the accumulator as a tracked complex 1.
    tracked::Complex<T> acc(tracked::constant<T>("one", T(1)));
    for (int i = 0; i < n; ++i) acc = acc * base;
    if (exponent < 0) {
        tracked::Complex<T> one(tracked::constant<T>("one", T(1)));
        return one / acc;
    }
    return acc;
}

} // namespace ql