#pragma once
#include <tracked/tracked.hpp>

// Output-by-reference variant of the catastrophic-cancellation kernel.
// Returns void and writes (a + b) - a into the output parameter `out`
// instead of returning it.  Exercises the driver generator's handling of
// void-returning kernels: `out` must be default-constructed as a Tracked
// instance and passed by reference, never wrapped with tracked::track().
template <class T>
void cancellation_out(T a, T b, T& out) {
    auto s = tracked::add(a, b, TRACKED_HERE);
    out = tracked::sub(s, a, TRACKED_HERE);
}
