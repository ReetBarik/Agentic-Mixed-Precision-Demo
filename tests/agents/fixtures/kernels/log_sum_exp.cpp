#pragma once
#include <cmath>
#include <tracked/journal.hpp>   // TRACKED_HERE / tracked::SourceLocation

// Naive log-sum-exp: numerically unstable when inputs differ by many orders
// of magnitude.  Use the max-shift trick in production.
//
// Uses std:: calls to exercise the characterizer's interop-shim path.  The
// driver injects location-forwarding std::exp / std::log overloads (a trailing
// tracked::SourceLocation), so passing TRACKED_HERE at these call sites
// attributes the ops to *this* kernel function rather than to the shim.
template <class T>
T log_sum_exp_naive(T a, T b) {
    return std::log(std::exp(a, TRACKED_HERE) + std::exp(b, TRACKED_HERE),
                    TRACKED_HERE);
}
