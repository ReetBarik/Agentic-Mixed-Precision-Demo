#include <Kokkos_Core.hpp>
#include "kokkosMaths.h"
#include "kokkosUtils.h"
#include <dd_math.hpp>
#include <dd_complex.hpp>
#include "candidate_overload.inc"
int main(int argc, char** argv){ Kokkos::initialize(argc,argv); double out=0;
 { quad::ddfun::ddouble v(1.5), x(2.5);
   auto r = ql::Lnrat<Kokkos::complex<double>, double, double>(v, x); out=r.hi+r.lo; }
 Kokkos::finalize(); return ((int)out)&1; }
