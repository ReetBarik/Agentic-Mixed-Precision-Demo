#pragma once
template <class T>
T cancellation_check(T a, T b) {
    return (a + b) - a;   // catastrophic cancellation when b << a
}
