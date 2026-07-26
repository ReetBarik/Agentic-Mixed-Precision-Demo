#pragma once
// SOURCE_HASH: 8a25ba67078785aee3cf807a7556c2ad0cc5a10beabcb60fd565f1c16a78c18f
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Region: box/B2m.h:306 — ga43m = TOutput(+p3sq + m3sq - m4sq) - root;
// The region uses only built-in arithmetic operators (+, -) and a
// TOutput(...) conversion around the sum. The vendored ddouble type
// already provides operator+, operator-, and a constructor from double
// (and identity for ddouble). No named constants, no math functions,
// no qualified-namespace math calls, and no container types appear in
// this region — so no additional overloads, specializations, or
// constant wrappers are required beyond what dd_math.hpp already
// supplies. (Rules R2, C2, C3: nothing missing to bridge.)