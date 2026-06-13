#pragma once
#include <tracked/tracked.hpp>

// Catastrophic cancellation in (a + b) - a when b << a.
// Uses named Tracked functions with TRACKED_HERE so the journal captures
// per-line source attribution.
template <class T>
T cancellation_check(T a, T b) {
    auto s = tracked::add(a, b, TRACKED_HERE);
    return tracked::sub(s, a, TRACKED_HERE);
}
