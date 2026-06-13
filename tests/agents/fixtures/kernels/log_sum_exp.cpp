#pragma once
#include <cmath>

// Naive log-sum-exp: numerically unstable when inputs differ by many orders
// of magnitude.  Use the max-shift trick in production.
//
// Uses std:: calls to exercise the characterizer's interop-shim path.
// Per-line attribution for std:: calls comes from the shim itself.
template <class T>
T log_sum_exp_naive(T a, T b) {
    return std::log(std::exp(a) + std::exp(b));
}
