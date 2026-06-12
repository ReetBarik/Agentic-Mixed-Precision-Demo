#pragma once
// Kahan compensated summation — intentional cancellation in the compensation
// step is the point; the accumulated sum should be well-conditioned.
template <class T>
T kahan_sum(T a, T b, T c) {
    T sum = a;
    T comp = T(0);
    // Add b
    T y = b - comp;
    T t = sum + y;
    comp = (t - sum) - y;   // intentional cancellation here
    sum = t;
    // Add c
    y = c - comp;
    t = sum + y;
    comp = (t - sum) - y;   // intentional cancellation here
    sum = t;
    return sum;
}
