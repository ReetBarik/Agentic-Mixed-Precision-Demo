#pragma once
#include <tracked/tracked.hpp>

// Kahan compensated summation — intentional cancellation in the compensation
// step is the point; the accumulated sum should be well-conditioned.
// Uses named Tracked functions with TRACKED_HERE for per-line attribution.
template <class T>
T kahan_sum(T a, T b, T c) {
    auto sum  = a;
    auto comp = T(0);

    // Add b
    auto y1   = tracked::sub(b, comp, TRACKED_HERE);
    auto t1   = tracked::add(sum, y1, TRACKED_HERE);
    auto d1   = tracked::sub(t1, sum, TRACKED_HERE);
    comp      = tracked::sub(d1, y1, TRACKED_HERE);   // intentional cancellation
    sum       = t1;

    // Add c
    auto y2   = tracked::sub(c, comp, TRACKED_HERE);
    auto t2   = tracked::add(sum, y2, TRACKED_HERE);
    auto d2   = tracked::sub(t2, sum, TRACKED_HERE);
    comp      = tracked::sub(d2, y2, TRACKED_HERE);   // intentional cancellation
    sum       = t2;

    return sum;
}
