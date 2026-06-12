#pragma once
// Naive two-pass variance: E[X^2] - E[X]^2 — catastrophically cancellates
// when variance is small relative to the mean.
template <class T>
T naive_variance(T sum_x, T sum_x2, T n) {
    T mean = sum_x / n;
    T mean_sq = sum_x2 / n;
    return mean_sq - mean * mean;
}
