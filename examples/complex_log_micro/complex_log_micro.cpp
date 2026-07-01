// Manual micro-driver: instrument a complex-log routine with tracked::Complex<double>.
//
// `complex_log` is a representative "complex log with branch-cut handling"
// kernel, the kind of function copied verbatim from a user's codebase and
// run through Tracked to characterize where it loses accuracy.
//
// Kokkos::log and Kokkos::abs are treated as opaque (single-record black
// boxes): the driver documents this limitation in the journal via op="opaque"
// records carrying fn_name="Kokkos::log" / "Kokkos::abs".

#include <Kokkos_Core.hpp>
#include <tracked/tracked.hpp>
#include <tracked/complex.hpp>
#include <tracked/journal.hpp>

#include <cstdio>

// --------------------------------------------------------------------------
// Minimal external-library surface that complex_log needs.  Reimplementing
// it here (rather than including the original framework headers) avoids
// forcing Kokkos::log on tracked::Complex, which would fail to compile.
// --------------------------------------------------------------------------

namespace extmath {

template <class T>
struct Constants {
    // NOTE: not constexpr — Tracked<T> holds a std::set<std::string> and is
    // not a literal type, so constexpr would be ill-formed for T = Tracked<>.
    static T _zero() { return T(0); }
    static T _pi()   { return T(3.14159265358979323846); }
};

// --- Accessors (return Tracked<T> so comparisons stay Tracked-vs-Tracked) ---

template <class T>
inline tracked::Tracked<T> Imag(const tracked::Complex<T>& z) { return z.imag(); }

template <class T>
inline tracked::Tracked<T> Real(const tracked::Complex<T>& z) { return z.real(); }

// Sign on a plain scalar — returns int by convention.
template <class T>
inline int Sign(const T& x) { return (T(0) < x) - (x < T(0)); }

// --- Opaque math wrappers ---
// These stand in for the framework's complex abs/log.  The interior of
// Kokkos::log / Kokkos::abs is treated as a single opaque op in the journal.

template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Complex<T>& z) {
    Kokkos::complex<T> raw{z.real().value(), z.imag().value()};
    T result = Kokkos::abs(raw);
    // Forward tracked inputs so rel_err + provenance propagate through opaque.
    return tracked::opaque("Kokkos::abs", result, z.real(), z.imag());
}

template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x) {
    return tracked::abs(x);   // real abs is cheap and visible — no need to hide
}

template <class T>
inline tracked::Complex<T> kLog(const tracked::Complex<T>& z) {
    Kokkos::complex<T> raw{z.real().value(), z.imag().value()};
    Kokkos::complex<T> r = Kokkos::log(raw);
    // Both output components depend on both input components.
    return tracked::opaque<T>("Kokkos::log", r.real(), r.imag(),
                              z.real(), z.imag());
}

template <class T>
inline tracked::Tracked<T> kLog(const tracked::Tracked<T>& x) {
    return tracked::log(x);   // real log stays visible
}

} // namespace extmath

// --------------------------------------------------------------------------
// complex_log — a representative complex-log kernel with branch-cut handling.
//
// Computes log of a complex number z, with a branch-cut +iπ term added
// when z is on the negative real axis.
// --------------------------------------------------------------------------

template <class TComplex, class TReal>
TComplex complex_log(const TComplex& z, const TReal& branch_sign) {
    TComplex result;
    if (extmath::Imag(z) == extmath::Constants<TReal>::_zero() &&
        extmath::Real(z) <= extmath::Constants<TReal>::_zero()) {
        // On (or just below) the negative real axis — add the branch term.
        TComplex temp(extmath::Constants<TReal>::_zero(),
                      extmath::Constants<TReal>::_pi() * TReal(extmath::Sign(branch_sign)));
        result = extmath::kLog(-z) + temp;
    } else {
        result = extmath::kLog(z);
    }
    return result;
}

// --------------------------------------------------------------------------

int main() {
    Kokkos::initialize();
    {
        using V  = tracked::Tracked<double>;
        using CV = tracked::Complex<double>;

        std::puts("Sample 1: ordinary case  z = 1.5 + 0.3i");
        {
            auto z = tracked::track("z1", 1.5, 0.3);
            auto s = tracked::track("s1", 1.0);
            auto r = complex_log<CV, V>(z, s);
            std::printf("  result = (%.17g, %.17g)\n",
                        r.real().value(), r.imag().value());
        }

        std::puts("Sample 2: near branch cut  z = -1.0 + 1e-15·i");
        {
            auto z = tracked::track("z2", -1.0, 1e-15);
            auto s = tracked::track("s2", 1.0);
            auto r = complex_log<CV, V>(z, s);
            std::printf("  result = (%.17g, %.17g)\n",
                        r.real().value(), r.imag().value());
        }

        std::puts("Sample 3: on negative real axis  z = -2.0 + 0i");
        {
            auto z = tracked::track("z3", -2.0, 0.0);
            auto s = tracked::track("s3", 1.0);
            auto r = complex_log<CV, V>(z, s);
            std::printf("  result = (%.17g, %.17g)\n",
                        r.real().value(), r.imag().value());
        }

        tracked::journal::flush("complex_log.jsonl");
        std::puts("\nJournal flushed to complex_log.jsonl");
    }
    Kokkos::finalize();
    return 0;
}
