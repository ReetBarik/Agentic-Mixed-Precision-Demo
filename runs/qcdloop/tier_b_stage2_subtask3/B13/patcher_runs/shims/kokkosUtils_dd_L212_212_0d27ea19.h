#pragma once
// SOURCE_HASH: 0d27ea19805f68e86b3233256455c70ee596554d9b998b8eaca7e8d7c0225364
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Region: return -(S * (B0 - H * B2) + A);
// All operands are promoted to quad::ddfun::ddouble by the boundary patch.
// The vendored dd_math.hpp already provides:
//   - operator*(ddouble, ddouble)
//   - operator-(ddouble, ddouble)   (binary)
//   - operator+(ddouble, ddouble)
//   - operator-(ddouble)            (unary negate)
// so no additional operators, overloads, specializations, or named
// constants are required for this region to compile.
//
// No named constants appear in the region -> Rule 5 not triggered.
// No qualified math calls appear         -> C3 bridge not triggered.
// No library templates are resolved      -> C5/C7 not triggered.
// No container-of-float appears          -> Rule 3 not triggered.