#pragma once
#include <cmath>

// Naive log-sum-exp: numerically unstable when inputs differ by many orders
// of magnitude.  Use the max-shift trick in production.
template <class T>
T log_sum_exp_naive(T a, T b) {
    return std::log(std::exp(a) + std::exp(b));
}
