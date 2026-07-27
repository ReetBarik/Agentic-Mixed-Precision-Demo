// v3 falsification probe for LEAF_CALLEE_PROMOTION_DESIGN §7 (P5).
//
// PURPOSE: pre-implementation discharge of STOP #E ("source doesn't provide what
// the design claims").  v2's probe_constants_dd.cpp instantiated the *double-primary*
// Constants<T> at dd and saw the 19-coeff series.  After commit e3d2e45 the vendored
// qcdloop-under-test snapshot ALSO carries kokkosMaths_dd.h — qcdloop's own dd-precision
// Constants<T> (43-term Chebyshev _C, 25-term Bernoulli _B, dd _pi() via dd_pi()).
// This probe confirms the enriched source now hands the pipeline the 43-coeff table
// directly:
//
//   * ql::Constants<ddouble>::_num_C() == 43   (not 19)
//   * _C(0), _C(18), _C(42) are BIT-EXACT vs the source table's make_dd() literals
//   * _pi() is bit-exact dd pi (dd_pi()), not T(M_PI)
//
// Build (single-TU, no Kokkos runtime object needed beyond init):
//   g++ -std=c++20 -w -Ithird_party/include -I<KI>/include \
//       -I runs/qcdloop_headers_full \
//       probe_constants_dd43.cpp -L<KI>/lib64 -lkokkoscore -lkokkoscontainers -ldl -o /tmp/pC43 && /tmp/pC43

#include <Kokkos_Core.hpp>
#include "dd_math.hpp"
#include "dd_complex.hpp"
// The enriched dd-precision Constants<T> from the vendored qcdloop-under-test snapshot.
#include "kokkosMaths_dd.h"

using quad::ddfun::ddouble;

static bool bit_eq(ddouble const& a, uint64_t hi_bits, uint64_t lo_bits) {
    ddouble ref = quad::ddfun::make_dd(hi_bits, lo_bits);
    uint64_t a_hi, a_lo, r_hi, r_lo;
    __builtin_memcpy(&a_hi, &a.hi, 8);  __builtin_memcpy(&a_lo, &a.lo, 8);
    __builtin_memcpy(&r_hi, &ref.hi, 8); __builtin_memcpy(&r_lo, &ref.lo, 8);
    return a_hi == r_hi && a_lo == r_lo;
}

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    int rc = 0;
    {
        int n = ql::Constants<ddouble>::_num_C();
        Kokkos::printf("num_C = %d  (expect 43)\n", n);
        if (n != 43) { Kokkos::printf("  FAIL: _num_C() != 43\n"); rc |= 1; }

        // Bit-exact spot checks against the source table (kokkosMaths_dd.h C[0]/C[18]/C[42]).
        ddouble c0  = ql::Constants<ddouble>::_C(0);
        ddouble c18 = ql::Constants<ddouble>::_C(18);
        ddouble c42 = ql::Constants<ddouble>::_C(42);
        bool ok0  = bit_eq(c0 , 0x3fdb849409b3171fULL, 0xbc61d08606ea8094ULL);
        bool ok18 = bit_eq(c18, 0xbca48079ae714341ULL, 0x3938662e035d673fULL);
        bool ok42 = bit_eq(c42, 0xb8adb6cee2774df0ULL, 0x354562db4ea6970aULL);
        Kokkos::printf("C[0]  hi=%.17g lo=%.3e  bit-exact=%d\n", c0.hi , c0.lo , (int)ok0);
        Kokkos::printf("C[18] hi=%.17g lo=%.3e  bit-exact=%d\n", c18.hi, c18.lo, (int)ok18);
        Kokkos::printf("C[42] hi=%.17g lo=%.3e  bit-exact=%d\n", c42.hi, c42.lo, (int)ok42);
        if (!(ok0 && ok18 && ok42)) { Kokkos::printf("  FAIL: coeff bit mismatch\n"); rc |= 2; }

        // dd pi is bit-exact dd_pi(), not T(M_PI).
        ddouble pi = ql::Constants<ddouble>::_pi();
        bool okpi = bit_eq(pi, 0x400921fb54442d18ULL, 0x3ca1a62633145c07ULL);
        Kokkos::printf("_pi() hi=%.17g lo=%.3e  bit-exact dd_pi=%d\n", pi.hi, pi.lo, (int)okpi);
        if (!okpi) { Kokkos::printf("  FAIL: _pi() not dd_pi()\n"); rc |= 4; }

        // Sum the 43 coeffs (sanity: series value, cf. v2's 19-coeff sum 0.82246703...).
        ddouble s(0.0);
        for (int i = 0; i < n; ++i) s = s + ql::Constants<ddouble>::_C(i);
        Kokkos::printf("sum_C(43) hi=%.17g lo=%.3e\n", s.hi, s.lo);

        Kokkos::printf(rc == 0 ? "P5 PASS: enriched source provides 43-coeff dd table\n"
                               : "P5 FAIL: rc=%d\n", rc);
    }
    Kokkos::finalize();
    return rc;
}
