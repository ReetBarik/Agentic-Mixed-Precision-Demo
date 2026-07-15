//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//
// Wrapper header that conditionally includes the appropriate precision version
// of kokkosMaths based on USE_QUAD_COMPLEX define

#pragma once

#ifdef USE_QUAD_COMPLEX
#ifdef KOKKOS_ENABLE_CUDA
#include "kokkosMaths_quad.h"
#else
#error "USE_QUAD_COMPLEX requires KOKKOS_ENABLE_CUDA to be defined"
#endif
#else
#include "kokkosMaths.h"
#endif
