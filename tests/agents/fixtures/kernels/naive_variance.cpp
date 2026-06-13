#pragma once
#include <tracked/tracked.hpp>

// Naive two-pass variance: E[X^2] - E[X]^2 — catastrophically cancellates
// when variance is small relative to the mean.
// Uses named Tracked functions with TRACKED_HERE for per-line attribution.
template <class T>
T naive_variance(T sum_x, T sum_x2, T n) {
    auto mean    = tracked::div(sum_x,  n, TRACKED_HERE);
    auto mean_sq = tracked::div(sum_x2, n, TRACKED_HERE);
    auto mean2   = tracked::mul(mean,   mean, TRACKED_HERE);
    return tracked::sub(mean_sq, mean2, TRACKED_HERE);
}
