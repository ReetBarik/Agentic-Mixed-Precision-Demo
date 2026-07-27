#include <Kokkos_Core.hpp>
#include "kokkosMaths.h"
#include <dd_math.hpp>
using quad::ddfun::ddouble;
int main(int argc,char**argv){
  Kokkos::initialize(argc,argv);
  { // Does Constants<ddouble>::_C(i) and _num_C() instantiate at dd at RUNTIME?
    int n = ql::Constants<ddouble>::_num_C();
    ddouble s(0.0);
    for(int i=0;i<n;++i) s = s + ql::Constants<ddouble>::_C(i);
    Kokkos::printf("num_C=%d  sum_C.hi=%.17g  sum_C.lo=%.3e\n", n, s.hi, s.lo);
  }
  Kokkos::finalize(); return 0;
}
