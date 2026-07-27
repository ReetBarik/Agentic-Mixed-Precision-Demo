// Option-B ceiling probe for LEAF_CALLEE_PROMOTION_DESIGN v2 §2 (Class-2 choice).
//
// Question the design must answer BEFORE promising a lift: if ddilog is computed at
// dd using the SAME 19-coeff Chebyshev series the pipeline can see (Constants<T>::_C,
// kokkosMaths.h:26-50) — i.e. WITHOUT synthesizing the oracle's 43-coeff dd table —
// how much accuracy does dd actually buy back?
//
// This isolates the two error sources in a Clenshaw-summed Chebyshev series:
//   (1) ROUNDOFF in the recurrence  B0 = C_i + ALFA*B1 - B2   (the cancellation the
//       chain suffers).  dd shrinks this from ~1e-16 to ~1e-32.
//   (2) TRUNCATION of the series at 19 terms.  This is a property of the COEFFS, not
//       the arithmetic — dd cannot shrink it.  coeffs[18] ~= -1e-16 (kokkosMaths.h:47)
//       so the 19-term tail bounds the achievable accuracy at ~1e-16 REGARDLESS of
//       arithmetic width.
//
// Method (no Kokkos, no headers — pure self-contained):
//   * sum the identical 19 coeffs three ways at a battery of Y in [-1,1]:
//       - double  Clenshaw           (baseline: roundoff + truncation, both ~1e-16)
//       - dd      Clenshaw, 19 coeff (Option B: roundoff ~1e-32, truncation ~1e-16)
//       - "hi"    reference = dd Clenshaw treated as truth for the ROUNDOFF delta only
//   * the double-vs-dd19 gap is the roundoff dd removes; the dd19-vs-analytic gap is
//     the truncation floor dd CANNOT remove.
//
// A minimal double-double built inline (two-sum / two-prod) so the probe needs no
// third_party include and no toolchain flags beyond -std=c++17.
//
// Build/run:
//   g++ -std=c++17 -O2 probe_optionB_ceiling.cpp -o /tmp/ceil && /tmp/ceil

#include <cstdio>
#include <cmath>

// ---- minimal double-double (Dekker two-sum / two-prod) --------------------
struct dd {
    double hi, lo;
    dd(double h=0.0, double l=0.0): hi(h), lo(l) {}
};
static inline dd two_sum(double a, double b){
    double s=a+b; double bb=s-a; double e=(a-(s-bb))+(b-bb); return dd(s,e);
}
static inline dd two_prod(double a, double b){
    double p=a*b; double e=std::fma(a,b,-p); return dd(p,e);
}
static inline dd add(dd a, dd b){
    dd s=two_sum(a.hi,b.hi); s.lo+=a.lo+b.lo;
    double h=s.hi+s.lo; double l=s.lo-(h-s.hi); return dd(h,l);
}
static inline dd sub(dd a, dd b){ return add(a, dd(-b.hi,-b.lo)); }
static inline dd mul(dd a, dd b){
    dd p=two_prod(a.hi,b.hi); p.lo+=a.hi*b.lo+a.lo*b.hi;
    double h=p.hi+p.lo; double l=p.lo-(h-p.hi); return dd(h,l);
}

// ---- the 19 double-precision Chebyshev coeffs (verbatim, kokkosMaths.h:28-48) --
static const double C19[19] = {
    0.4299669356081370, 0.4097598753307711, -0.0185884366501460, 0.0014575108406227,
   -0.0001430418444234, 0.0000158841554188, -0.0000019078495939, 0.0000002419518085,
   -0.0000000319334127, 0.0000000043454506, -0.0000000006057848, 0.0000000000861210,
   -0.0000000000124433, 0.0000000000018226, -0.0000000000002701, 0.0000000000000404,
   -0.0000000000000061, 0.0000000000000009, -0.0000000000000001
};

// Clenshaw sum of C19 at argument ALFA (= 2*H, H=2Y-1) — mirrors ddilog's loop.
static double clenshaw_double(double ALFA){
    double B1=0,B2=0,B0=0;
    for(int i=18;i>=0;--i){ B0=C19[i]+ALFA*B1-B2; B2=B1; B1=B0; }
    double H=ALFA*0.5; return B0 - H*B2;
}
static dd clenshaw_dd(dd ALFA){
    dd B1,B2,B0;
    for(int i=18;i>=0;--i){ B0=sub(add(dd(C19[i]),mul(ALFA,B1)),B2); B2=B1; B1=B0; }
    dd H=mul(ALFA,dd(0.5)); return sub(B0, mul(H,B2));
}

int main(){
    // A battery of Y across the reduced Chebyshev domain [0,1] (ddilog maps its
    // argument here after range reduction).  H=2Y-1, ALFA=2H.
    const double Ys[] = {0.02,0.1,0.25,0.4,0.55,0.7,0.85,0.97};
    printf("  Y        double_sum          dd19_sum(hi)        |dd19-double| (roundoff dd removes)\n");
    double max_roundoff=0.0;
    for(double Y: Ys){
        double H=2*Y-1, ALFA=2*H;
        double d = clenshaw_double(ALFA);
        dd     q = clenshaw_dd(dd(ALFA));
        double gap = std::fabs(q.hi - d);
        if(gap>max_roundoff) max_roundoff=gap;
        printf("  %.2f   %.17g   %.17g   %.3e\n", Y, d, q.hi, gap);
    }
    // The dd internal residual q.lo at a representative point = how many extra digits
    // the dd recurrence actually carries (the roundoff headroom dd provides).
    dd qmid = clenshaw_dd(dd(2*(2*0.55-1)));
    printf("\n  dd recurrence residual |lo/hi| at Y=0.55 = %.3e  (dd carries ~%.1f extra digits of the SUM)\n",
           std::fabs(qmid.lo/qmid.hi), -std::log10(std::fabs(qmid.lo/qmid.hi)+1e-300));
    printf("  max |dd19 - double| over battery = %.3e  (= roundoff dd buys back)\n", max_roundoff);

    // Truncation floor: |last coeff| bounds the 19-term Chebyshev tail — a property of
    // the COEFFS, independent of arithmetic width.  dd CANNOT shrink this.
    printf("  19-term truncation floor ~ |C[18]| = %.3e  (dd CANNOT reduce — needs 43-coeff table)\n",
           std::fabs(C19[18]));
    printf("\n  CEILING for Option B ddilog accuracy = max(roundoff_removed_floor, truncation_floor)\n");
    printf("  => Option B lifts ONLY the cancellation-roundoff component; series accuracy stays ~%.0e\n",
           std::fabs(C19[18]));
    return 0;
}
