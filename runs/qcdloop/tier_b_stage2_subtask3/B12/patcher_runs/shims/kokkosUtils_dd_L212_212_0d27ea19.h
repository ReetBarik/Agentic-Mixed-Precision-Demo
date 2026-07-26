#pragma once
// SOURCE_HASH: 0d27ea19805f68e86b3233256455c70ee596554d9b998b8eaca7e8d7c0225364
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Region: return -(S * (B0 - H * B2) + A);
// Operators used: binary * (ddouble,ddouble), binary - (ddouble,ddouble),
// binary + (ddouble,ddouble), unary - (ddouble).
// All of these are already provided by the vendored quad::ddfun::ddouble type
// (see dd_math.hpp). No additional overloads, specializations, or named
// constants are referenced by this region.
//
// No reads, no writes, no named constants, no qualified math calls,
// no library templates to specialize. Nothing to emit beyond the include.