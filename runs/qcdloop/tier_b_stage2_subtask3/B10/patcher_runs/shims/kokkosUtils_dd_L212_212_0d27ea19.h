#pragma once
// SOURCE_HASH: 0d27ea19805f68e86b3233256455c70ee596554d9b998b8eaca7e8d7c0225364
// Shim for kokkosUtils.h:212 region promoted to quad::ddfun::ddouble.
// Region: return -(S * (B0 - H * B2) + A);
// All operands (S, B0, H, B2, A) are promoted to ddouble by the boundary patch.
// The vendored ddouble type already provides operator*, operator-, operator+,
// and unary operator- for ddouble op ddouble. No additional overloads,
// specializations, or named constants are referenced by this region.

#include <dd_math.hpp>
#include <dd_complex.hpp>

// No shim symbols required:
//   - Rule 2: arithmetic (*, -, +, unary -) on ddouble is supplied by the vendored header.
//   - Rule 5: region references no named constants.
//   - C3: region uses no qualified math calls needing an ADL/namespace bridge.
//   - C5/C7: region instantiates no library class/function templates on ddouble.