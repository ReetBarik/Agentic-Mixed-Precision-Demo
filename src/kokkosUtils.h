//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov

#pragma once
#include <math.h>
#include <inttypes.h>
#include <cstdio>
#include <ostream>
#include "kokkosMaths.h"

namespace ql
{
    // complex is defined in kokkosMaths.h

    void printDoubleBits(double x)
    {
        // We'll copy the double bits into a 64-bit integer.
        // A union is a common trick, or we can use memcpy.
        union {
            double d;
            uint64_t u;
        } conv;

        conv.d = x;

        // Use C99's PRIx64 for a portable 64-bit hex format.
        // %.16g prints up to 16 significant digits in decimal (just for reference).
        // std::printf("decimal=%.16g\n", x);
        std::printf("0x%016" PRIx64, conv.u);
    }

    inline void printDoubleBits(double x, std::ostream& os)
    {
        union {
            double d;
            uint64_t u;
        } conv;
        conv.d = x;
        char buf[32];
        if (std::snprintf(buf, sizeof(buf), "0x%016" PRIx64, conv.u) > 0)
            os << buf;
    }


    template<typename TMass>
    KOKKOS_INLINE_FUNCTION void printDoubleBits(TMass x) {
        union {
            TMass d;
            uint64_t u;
        } conv;

        conv.d = x;
        Kokkos::printf("0x%016" PRIx64, conv.u);
    }

    /*!
    * Computes the log of a complex number z.
    * If the imag(z)=0 and real(z)<0 and extra ipi term is included.
    * \param z the complex argument of the logarithm
    * \param isig the sign of the imaginary part
    * \return the complex log(z)
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput cLn(TOutput const& z, TScale const& isig) {
        TOutput cln;
        
        if (ql::Imag(z) == ql::Constants<TScale>::_zero() && ql::Real(z) <= ql::Constants<TScale>::_zero()) {
            TOutput temp(ql::Constants<TScale>::_zero(), ql::Constants<TScale>::_pi() * ql::Sign(isig));
            cln = ql::kLog(-z) + temp;
        }
        else
            cln = ql::kLog(z);
        return cln;
    }

    /*!
    * Computes the log of a real number x.
    * If the x<0 and extra ipi term is included.
    * \param x the real argument of the logarithm
    * \param isig the sign of the imaginary part
    * \return the complex log(x)
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput cLn(TScale const& x, TScale const& isig) {
        TOutput ln;
        if (x > ql::Constants<TScale>::_zero())
            ln = TOutput(ql::kLog(x));
        else {
            TOutput temp(ql::Constants<TScale>::_zero(), ql::Constants<TScale>::_pi() * ql::Sign(isig));
            ln = TOutput(ql::kLog(-x)) + temp;
        }
        return ln;
    }

    /*!
    * Implementation of the formulae of Denner and Dittmaier \cite Denner:2005nn.
    * \f[
    * f_{n}(x) = \ln \left( 1 - \frac{1}{x} \right) + \sum_{l=n+1}^{\infty} \frac{x^{n-l}}{l+1}
    *   \f]
    *
    * \param n the lower index of
    * \param x the argument of the function
    * \param iep the epsilon value
    * \return function DD from Eq. 4.11 of \cite Denner:2005nn.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput fndd(int const& n, TOutput const& x, TScale const& iep) {
        const int infty = 16;
        TOutput res = TOutput(ql::Constants<TScale>::_zero());
        
        if (ql::kAbs(x) < ql::Constants<TScale>::_ten()) {
            
            if (!ql::iszero<TOutput, TMass, TScale>(ql::kAbs(x - TOutput(ql::Constants<TScale>::_one())))) {
                res = (TOutput(ql::Constants<TScale>::_one()) - ql::kPow<TOutput, TMass, TScale>(x, n + 1)) * (ql::cLn<TOutput, TMass, TScale>(x - TOutput(ql::Constants<TScale>::_one()), iep) - ql::cLn<TOutput, TMass, TScale>(x, iep)); 
            }
            for (int j = 0; j <= n; j++) {
                res -= ql::kPow<TOutput, TMass, TScale>(x, n - j) / TScale(j + ql::Constants<TScale>::_one());
            }
        } else {

            res = ql::cLn<TOutput, TMass, TScale>(TOutput(ql::Constants<TScale>::_one()) - TOutput(ql::Constants<TScale>::_one()) / x, iep); 
            for (int j = n + 1; j <= n + infty; j++)
                res += ql::kPow<TOutput, TMass, TScale>(x, n - j) / TScale(j + ql::Constants<TScale>::_one());
        }
        
        return res;
    }



    /*!
    * Computes \f${\rm Lnrat}(x,y) = \log(x-i \epsilon)-\log(y-i \epsilon)\f$
    * \param x TMass object for the numerator
    * \param y TMass object for the denumerator
    * \return returns the ratio of logs
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput Lnrat(TOutput const& x, TOutput const& y) {
        
        const TOutput r = x / y;
        auto imag_r = (std::is_same<TOutput, double>::value) ? 0 : r.imag();
        if (ql::iszero<TOutput, TMass, TScale>(ql::Imag(r))) {
            return TOutput(ql::kLog(ql::kAbs(r))) - ql::Constants<TScale>::template _ipio2<TOutput, TMass, TScale>() * TOutput(ql::Sign(-ql::Real(x)) - ql::Sign(-ql::Real(y)));
        }  
        else
            return ql::kLog(r);
    }

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput Lnrat(TScale const& x, TScale const& y) {
        return TOutput(ql::kLog(ql::kAbs(x / y))) - (ql::Constants<TScale>::template _ipio2<TOutput, TMass, TScale>() * TOutput(ql::Sign(-x) - ql::Sign(-y)));
    }

    /*!
    * The dilog function for real argument
    * \param x the argument of the dilog
    * \return the dilog
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TMass ddilog(TMass const& x) {

        if (x == TMass(ql::Constants<TMass>::_one()))
            return ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>();
        else if (x == TMass(-ql::Constants<TMass>::_one()))
            return - TMass(ql::Constants<TMass>::_half()) * ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>();

        const TMass T = -x;
        TMass Y;
        TMass S;
        TMass A;

        if (ql::Real(T) <= TMass(-ql::Constants<TMass>::_two())) {

            Y = TMass(-ql::Constants<TMass>::_one()) / (TMass(ql::Constants<TMass>::_one()) + T);
            S = TMass(ql::Constants<TMass>::_one());
            A = TMass(-ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>() + TMass(ql::Constants<TMass>::_half()) * ql::Real(ql::kPow<TOutput, TMass, TScale>(ql::kLog(-T), 2) - ql::kPow<TOutput, TMass, TScale>(ql::kLog(TMass(ql::Constants<TMass>::_one()) + TMass(ql::Constants<TMass>::_one()) / T), 2)));

        } else if (ql::Real(T) < TMass(-ql::Constants<TMass>::_one())) {

            Y = TMass(-ql::Constants<TMass>::_one()) - T;
            S = TMass(-ql::Constants<TMass>::_one());
            A = ql::kLog(-T);
            A = TMass(-ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>() + A * (A + ql::kLog(TMass(ql::Constants<TMass>::_one()) + TMass(ql::Constants<TMass>::_one()) / T)));            

        } else if (ql::Real(T) <= TMass(-ql::Constants<TMass>::_half())) {

            Y = (TMass(-ql::Constants<TMass>::_one()) - T) / T;
            S = TMass(ql::Constants<TMass>::_one());
            A = ql::kLog(-T);
            A = TMass(-ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>() + A * (TMass(-ql::Constants<TMass>::_half()) * A + ql::kLog(TMass(ql::Constants<TMass>::_one()) + T)));

        } else if (ql::Real(T) < TMass(ql::Constants<TMass>::_zero())) {

            Y = -T / (TMass(ql::Constants<TMass>::_one()) + T);
            S = TMass(-ql::Constants<TMass>::_one());
            A = TMass(TMass(ql::Constants<TMass>::_half()) * ql::Real(ql::kPow<TOutput, TMass, TScale>(ql::kLog(TMass(ql::Constants<TMass>::_one()) + T),2)));

        } else if (ql::Real(T) <= TMass(ql::Constants<TMass>::_one())) {

            Y = T;
            S = TMass(ql::Constants<TMass>::_one());
            A = TMass(ql::Constants<TMass>::_zero());

        } else {

            Y = TMass(ql::Constants<TMass>::_one()) / T;
            S = TMass(-ql::Constants<TMass>::_one());
            A = TMass(ql::Constants<TMass>::template _pi2o6<TOutput, TMass, TScale>() + TMass(ql::Constants<TMass>::_half()) * ql::Real(ql::kPow<TOutput, TMass, TScale>(ql::kLog(T),2)));

        }
        
        const TMass H = Y + Y - TMass(ql::Constants<TMass>::_one());
        const TMass ALFA = H + H;
        TMass B1 = TMass(ql::Constants<TMass>::_zero());
        TMass B2 = TMass(ql::Constants<TMass>::_zero());
        TMass B0 = TMass(ql::Constants<TMass>::_zero());

        
        for (int i = ql::Constants<TScale>::_num_C() - 1; i >= 0; --i) {
            
            B0 = ql::Constants<TScale>::_C(i) + ALFA * B1 - B2;
            B2 = B1;
            B1 = B0;

        }

        return -(S * (B0 - H * B2) + A);

    }


    /*!
    * \param z
    * \param isig
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput li2series(TOutput const& z, TScale const& isig) {
        
        TOutput xm = -ql::cLn<TOutput, TMass, TScale>(TOutput(ql::Constants<TScale>::_one()) - z, -isig);
        const TOutput x2 = xm * xm;
        TOutput res = xm - x2/ TOutput(ql::Constants<TScale>::_four());

        for (int j = 0; j < ql::Constants<TScale>::_num_B(); j++)
        {
            xm *= x2;
            const TOutput n = res + xm *  ql::Constants<TScale>::_B(j);
            if (n == res) return res;
            else res = n;
        }
        Kokkos::printf("li2series: bad convergence\n");
        return TOutput(ql::Constants<TScale>::_zero());
    }


    /*!
    * \param z1
    * \param s
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput ltli2series(TOutput const& z1, TScale const& s) {
        TOutput xm = -ql::cLn<TOutput, TMass, TScale>(z1, -s);
        const TOutput x2 = xm * xm;
        TOutput res = xm - x2 / TOutput(ql::Constants<TScale>::_four());

        for (int i = 0; i < ql::Constants<TScale>::_num_B(); i++) {
            xm *= x2;
            const TOutput n = res + xm * ql::Constants<TScale>::_B(i);
            if (n == res) return res;
            else res = n;
        }
        Kokkos::printf("ltli2series: bad convergence\n");
        return TOutput(ql::Constants<TScale>::_zero());
    }


    /*!
    * \param z the argument
    * \param isig the sign of z
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput denspence(TOutput const& z, TScale const& isig) {
        const TOutput z1 = TOutput(ql::Constants<TScale>::_one()) - z;
        const TScale az1 = ql::kAbs(z1);

        if (isig == ql::Constants<TScale>::_zero() && ql::Imag(z) == ql::Constants<TScale>::_zero() && ql::kAbs(ql::Real(z1)) < ql::Constants<TScale>::template _qlonshellcutoff<TOutput, TMass, TScale>()){
            Kokkos::printf("denspence: argument on cut\n");
        }
            
        if (az1 < ql::Constants<TScale>::_eps15())
            return TOutput{ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>(), ql::Constants<TScale>::_zero()};

   
        else if (ql::Real(z) < ql::Constants<TScale>::_half())
        {
            if (ql::kAbs(z) < ql::Constants<TScale>::_one())
                return ql::li2series<TOutput, TMass, TScale>(z, isig);
            else
                return -ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>() - TOutput(ql::Constants<TScale>::_half()) * ql::kPow<TOutput, TMass, TScale>(ql::cLn<TOutput, TMass, TScale>(-z, -isig),2) - ql::li2series<TOutput, TMass, TScale>(TOutput(ql::Constants<TScale>::_one()) / z, -isig);
        }
        else
        {
            if (az1 < ql::Constants<TScale>::_one())
                return ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>() - ql::cLn<TOutput, TMass, TScale>(z, isig) * ql::cLn<TOutput, TMass, TScale>(z1, -isig) - ql::li2series<TOutput, TMass, TScale>(z1, -isig);
            else
                return TOutput(ql::Constants<TScale>::_two()) * ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>() + TOutput(ql::Constants<TScale>::_half()) * ql::kPow<TOutput, TMass, TScale>(ql::cLn<TOutput, TMass, TScale>(-z1, -isig),2) - ql::cLn<TOutput, TMass, TScale>(z, isig) * ql::cLn<TOutput, TMass, TScale>(z1, -isig) + ql::li2series<TOutput, TMass, TScale>(TOutput(ql::Constants<TScale>::_one()) / z1, isig);
        }
    }


    /*!
    * Calculate Li[2](1-(x1+ieps1)*(x2+ieps2)) for real x1,x2
    * Using +Li2(1-x1*x2)                           for x1*x2<1
    * and   -Li2(1-1/(x1*x2))-1/2*(ln(x1)+ln(x2))^2 for x1*x2>1
    * \param x1 numerator
    * \param x2 denominator
    * \param ieps1 sign of x1
    * \param ieps2 sign of x2
    * \return the ratio Li2
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput Li2omx(TMass const& x1, TMass const& x2, TScale const& ieps1, TScale const& ieps2) {
        
        TOutput prod, Li2omx;

        TMass arg = x1 * x2;
        const TScale ieps = ql::Sign(ql::Real(x2) * ieps1 + ql::Real(x1) * ieps2);
        if (ql::Real(arg) <= TMass(ql::Constants<TMass>::_one())) {
            if (ql::Real(arg) == TMass(ql::Constants<TMass>::_one()) || ql::Real(arg) == TMass(ql::Constants<TMass>::_zero()))
                prod = TOutput(ql::Constants<TScale>::_zero());
            else {
                const TOutput lnarg = ql::cLn<TOutput, TMass, TScale>(ql::Real(x1), ieps) + ql::cLn<TOutput, TMass, TScale>(ql::Real(x2), ieps2);
                const TOutput lnomarg = TOutput(ql::kLog(TMass(ql::Constants<TMass>::_one()) - arg));
                prod = lnarg * lnomarg;
            }
            Li2omx = TOutput(ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>()) - ql::denspence<TOutput, TMass, TScale>(TOutput(arg), ieps) - prod;
        }
        else if (ql::Real(arg) > TMass(ql::Constants<TMass>::_one())) {
            arg = TMass(ql::Constants<TMass>::_one()) / (x1 * x2);
            const TOutput lnarg = - ql::cLn<TOutput, TMass, TScale>(ql::Real(x1), ieps1) - ql::cLn<TOutput, TMass, TScale>(ql::Real(x2), ieps2);
            const TOutput lnomarg = TOutput(ql::kLog(TMass(ql::Constants<TMass>::_one()) - arg));
            Li2omx = -TOutput(ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>()) + ql::denspence<TOutput, TMass, TScale>(TOutput(arg), ieps) + lnarg * lnomarg - TOutput(ql::Constants<TScale>::_half()) * lnarg * lnarg;
        }
        
        return Li2omx;
    }


    /*!
    * \param zrat1 first argument
    * \param zrat2 second argument
    * \param ieps1 sign for zrat1
    * \param ieps2 sign for zrat2
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput spencer(TOutput const& zrat1, TOutput const& zrat2, TScale const&ieps1, TScale const&ieps2) {
        
        const TScale x1 = ql::Real(zrat1);
        const TScale x2 = ql::Real(zrat2);
        const TScale y1 = ql::Imag(zrat1);
        const TScale y2 = ql::Imag(zrat2);

            TOutput res, prod;
        if (ql::iszero<TOutput, TMass, TScale>(y1) && ql::iszero<TOutput, TMass, TScale>(y2))
            res = ql::Li2omx<TOutput, TMass, TScale>(x1, x2, ieps1, ieps2);
        else {
            TOutput arg = zrat1 * zrat2;
            const TScale ieps = ql::Constants<TScale>::_zero();
            if (ql::kAbs(arg) <= ql::Constants<TScale>::_one()) {
                if (arg == TOutput(ql::Constants<TScale>::_zero()) || arg == TOutput(ql::Constants<TScale>::_one()))
                    prod = TOutput(ql::Constants<TScale>::_zero());
                else{
                    const TOutput lnarg = ql::cLn<TOutput, TMass, TScale>(zrat1, ieps1) + ql::cLn<TOutput, TMass, TScale>(zrat2, ieps2);
                    const TOutput lnomarg = ql::kLog(TOutput(ql::Constants<TScale>::_one()) - arg);
                    prod = lnarg * lnomarg;
                }
                res = TOutput(ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>()) - ql::denspence<TOutput, TMass, TScale>(arg, ieps) - prod;
            }
            else if (ql::kAbs(arg) > ql::Constants<TScale>::_one()) {
                arg = TOutput(ql::Constants<TScale>::_one()) / (zrat1 * zrat2);
                const TOutput lnarg = -ql::cLn<TOutput, TMass, TScale>(zrat1, ieps1) - ql::cLn<TOutput, TMass, TScale>(zrat2, ieps2);
                const TOutput lnomarg = ql::kLog(TOutput(ql::Constants<TScale>::_one()) - arg);
                res = -TOutput(ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>()) + ql::denspence<TOutput, TMass, TScale>(arg, ieps) + lnarg * lnomarg - TOutput(ql::Constants<TScale>::_half()) * ql::kPow<TOutput, TMass, TScale>(lnarg,2);
            }
        }
        return res;
    }

    /*!
    * \param z1
    * \param s1
    * \param z2
    * \param s2
    * \param s12
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION int eta(TOutput const& z1, TScale const& s1, TOutput const& z2, TScale const& s2, TScale const& s12) {
        
        TScale im1 = ql::Imag(z1), im2 = ql::Imag(z2), im12 = ql::Imag(z1 * z2);
        if (im1 == ql::Constants<TScale>::_zero()) im1 = s1;
        if (im2 == ql::Constants<TScale>::_zero()) im2 = s2;
        if (im12 == ql::Constants<TScale>::_zero()) im12 = s12;

        int eta;
        if (im1 < ql::Constants<TScale>::_zero() && im2 < ql::Constants<TScale>::_zero() && im12 > ql::Constants<TScale>::_zero())
            eta = 1;
        else if (im1 > ql::Constants<TScale>::_zero() && im2 > ql::Constants<TScale>::_zero() && im12 < ql::Constants<TScale>::_zero())
            eta = -1;
        else
            eta = 0;

        return eta;
    }


    /*!
    * \param z2
    * \param im2
    * \param im12
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput xeta(const Kokkos::Array<TOutput, 2> &z1, const Kokkos::Array<TScale, 2> &im1, TOutput const& z2, TScale const& im2, TScale const& im12, const Kokkos::Array<TOutput, 2> &l1) {  
        
            return l1[1] * TOutput(ql::eta<TOutput, TMass, TScale>(z1[1], im1[1], z2, im2, im12)) - l1[0] * TOutput(ql::eta<TOutput, TMass, TScale>(z1[0], im1[0], z2, im2, im12));
    }
    

    /*!
    * \param c1
    * \param im1x
    * \param c2
    * \param im2x
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION int etatilde(TOutput const& c1, TScale const& im1x, TOutput const& c2, TScale const& im2x) {
        int etatilde;
        TScale im1 = ql::Imag(c1);
        TScale im2 = ql::Imag(c2);

        if (im1 == ql::Constants<TScale>::_zero())
            im1 = im1x;
        
        if (im2 != ql::Constants<TScale>::_zero())
            etatilde = ql::eta<TOutput, TMass, TScale>(c1, im1x, c2, ql::Constants<TScale>::_zero(), ql::Constants<TScale>::_zero());
        else if ( ql::Real(c2) > ql::Constants<TScale>::_zero())
            etatilde = 0;
        else if (im1 > ql::Constants<TScale>::_zero() && im2x > ql::Constants<TScale>::_zero())
            etatilde = -1;
        else if (im1 < ql::Constants<TScale>::_zero() && im2x < ql::Constants<TScale>::_zero())
            etatilde = 1;
        else
            etatilde = 0;

        return etatilde;
    }


    /*!
    * \param z2
    * \param im2
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput xetatilde(const Kokkos::Array<TOutput, 2> &z1, const Kokkos::Array<TScale, 2> &im1, TOutput const& z2, TScale const& im2, const Kokkos::Array<TOutput, 2> &l1) { 
    
            return l1[1] * TOutput(ql::etatilde<TOutput, TMass, TScale>(z1[1], im1[1], z2, im2)) - l1[0] * TOutput(ql::etatilde<TOutput, TMass, TScale>(z1[0], im1[0], z2, im2));
    }


    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput eta5(TOutput const& a, TOutput const& b, TOutput const& c, TOutput const& d, TOutput const& e) {
        TOutput res = TOutput(ql::Constants<TScale>::_zero());
        const TScale ima = ql::Sign(ql::Imag(a));
        const TScale imb = ql::Sign(ql::Imag(b));
        const TScale imc = ql::Sign(ql::Imag(c));
        const TScale imd = ql::Sign(ql::Imag(d));
        const TScale ime = ql::Sign(ql::Imag(e));

        if (ima == imb)
        {
            if (ima == imd)
            {
                if (imc == ime)      res = TOutput(ql::Constants<TScale>::_zero()); 
                else if (ima != imc) res = ql::Constants<TScale>::template _2ipi<TOutput, TMass, TScale>() * TOutput(imc);
                else                 res = ql::Constants<TScale>::template _2ipi<TOutput, TMass, TScale>() * TOutput(-ime);
            }
            else if (ima != imc) res = ql::Constants<TScale>::template _2ipi<TOutput, TMass, TScale>() * TOutput(imc);
            else res = TOutput(ql::Constants<TScale>::_zero());
        }
        else if (ima == imd && ima != ime) res =  ql::Constants<TScale>::template _2ipi<TOutput, TMass, TScale>() * TOutput(-ime);
        else res = TOutput(ql::Constants<TScale>::_zero());

        return res;
    }


    /*!
    * \param a
    * \param b
    * \param c
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput eta3(TOutput const& a,TOutput const& b, TOutput const& c) {
        TOutput res = TOutput(ql::Constants<TScale>::_zero());
        const TScale ima = ql::Sign(ql::Imag(a));
        const TScale imb = ql::Sign(ql::Imag(b));
        const TScale imc = ql::Sign(ql::Imag(c));
        if (ima == imb && ima != imc) res = ql::Constants<TScale>::template _2ipi<TOutput, TMass, TScale>() * TOutput(imc);
        return res;
    }

    /*!
    * \param a
    * \param b
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput eta2(TOutput const& a,TOutput const& b) {
        const TScale ima = ql::Imag(a);
        const TScale imb = ql::Imag(b);
        const TScale imab = ql::Imag(a * b);
        return ql::Constants<TScale>::template _2ipi<TOutput, TMass, TScale>() * (ql::Htheta(-ima) * ql::Htheta(-imb) * ql::Htheta(imab) - ql::Htheta(ima) * ql::Htheta(imb) * ql::Htheta(-imab));
    }


    /*!
    * \param i_in
    * \param z_in
    * \param s
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput ltspence(int const& i_in, TOutput const& z_in, TScale const& s) {
        
        TOutput z[2]; //TODO:: change to hardcoded array of size 2
        z[i_in] = z_in;
        z[1 - i_in] = TOutput(ql::Constants<TScale>::_one()) - z_in;

        TOutput ltspence;
        if (ql::Real(z[0]) < ql::Constants<TScale>::_half()){
            if (ql::kAbs(z[0]) < ql::Constants<TScale>::_one())
                ltspence = ql::ltli2series<TOutput, TMass, TScale>(z[1],s);
            else {
                const TOutput clnz = ql::cLn<TOutput, TMass, TScale>(-z[0], -s);
                ltspence = TOutput(-ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>()) - TOutput(ql::Constants<TScale>::_half()) * clnz * clnz - ql::ltli2series<TOutput, TMass, TScale>(-z[1] / z[0], -s);
            }
        }
        else {
            const TScale az1 = ql::kAbs(z[1]);
            if (az1 < ql::Constants<TScale>::_eps15())
                ltspence = TOutput(ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>());
            else if (az1 < ql::Constants<TScale>::_one())
                ltspence = TOutput(ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>()) - ql::cLn<TOutput, TMass, TScale>(z[0],s) * ql::cLn<TOutput, TMass, TScale>(z[1], -s) - ql::ltli2series<TOutput, TMass, TScale>(z[0],-s);
            else {
                const TOutput clnz = ql::cLn<TOutput, TMass, TScale>(-z[0], -s);
                ltspence = TOutput(ql::Constants<TScale>::_two() * ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>()) + TOutput(ql::Constants<TScale>::_half()) * clnz * clnz - ql::cLn<TOutput, TMass, TScale>(z[0], s) * ql::cLn<TOutput, TMass, TScale>(z[1], -s) + ql::ltli2series<TOutput, TMass, TScale>(-z[0] / z[1], s);
            }
        }

        return ltspence;
    }


    /*!
    * \param z1 input argument
    * \param im1 sign of z1
    * \param z2 input argument
    * \param im2 sign of z2
    * \return the complex Spence's function
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput cspence(TOutput const& z1, TScale const& im1, TOutput const& z2, TScale const& im2) {
        TOutput cspence = TOutput(ql::Constants<TScale>::_zero());
        const TOutput z12 = z1 * z2;
        const TScale im12 = im2 * ql::Sign(ql::Real(z1));

        if (ql::Real(z12) > ql::Constants<TScale>::_half()) {
            cspence = ql::ltspence<TOutput, TMass, TScale>(1, z12, ql::Constants<TScale>::_zero());
            const int etas = ql::eta<TOutput, TMass, TScale>(z1, im1, z2, im2, im12);
            if (etas != 0) 
                cspence += TOutput(etas) * ql::cLn<TOutput, TMass, TScale>(TOutput(ql::Constants<TScale>::_one()) - z12, -im12) * ql::Constants<TScale>::template _2ipi<TOutput, TMass, TScale>();
        }
        else if (ql::kAbs(z12) < ql::Constants<TScale>::_eps4()) {
            cspence = TOutput(ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>());
            if (ql::kAbs(z12) >  ql::Constants<TScale>::_eps14())
                cspence += -ql::ltspence<TOutput, TMass, TScale>(0, z12, ql::Constants<TScale>::_zero()) + (ql::cLn<TOutput, TMass, TScale>(z1,im1) + ql::cLn<TOutput, TMass, TScale>(z2,im2)) * z12 * (TOutput(ql::Constants<TScale>::_one()) + z12 * (TOutput(ql::Constants<TScale>::_half()) + z12 * (TOutput(ql::Constants<TScale>::_one()) / TOutput(ql::Constants<TScale>::_three()) + z12 / TOutput(ql::Constants<TScale>::_four()))));
        }
        else
            cspence = TOutput(ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>()) - ql::ltspence<TOutput, TMass, TScale>(0, z12, ql::Constants<TScale>::_zero()) - (ql::cLn<TOutput, TMass, TScale>(z1, im1) + ql::cLn<TOutput, TMass, TScale>(z2, im2)) * ql::cLn<TOutput, TMass, TScale>(TOutput(ql::Constants<TScale>::_one()) - z12, ql::Constants<TScale>::_zero());

        return cspence;
    }


    /*!
    * \param z2 input arguments
    * \param im2 input signs.
    * \return the difference of cspence functions 
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput xspence(const Kokkos::Array<TOutput, 2>& z1, const Kokkos::Array<TScale, 2>& im1, TOutput const& z2, TScale const& im2) { 
        
        return ql::cspence<TOutput, TMass, TScale>(z1[1], im1[1], z2, im2) - ql::cspence<TOutput, TMass, TScale>(z1[0], im1[0], z2, im2);

    }
    

    /*!
    * \param x numerator
    * \param y denominator
    * \return the Li2 ratio
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput Li2omrat(TScale const& x, TScale const& y) {
        const TScale omarg = x / y;
        const TScale arg = ql::Constants<TScale>::_one() - omarg;
        if (arg > ql::Constants<TScale>::_one())
            return TOutput(ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>() - ql::ddilog<TOutput, TMass, TScale>(omarg)) - ql::kLog(arg) * ql::Lnrat<TOutput, TMass, TScale>(x, y);
        else
            return TOutput(ql::ddilog<TOutput, TMass, TScale>(arg));
    }


    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput Li2omrat(TOutput const& x, TOutput const& y, TScale const& ieps1, TScale const& ieps2) {
        const TOutput omarg = x / y;
        const TOutput arg = TOutput(ql::Constants<TScale>::_one()) - omarg;
        const TScale isarg = ql::Sign(ql::Real(x) * ieps2 - ql::Real(y) * ieps1);

        if (ql::kAbs(arg) > ql::Constants<TScale>::_one()) {
            const TScale isomarg = -isarg;
            return TOutput(ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>()) - ql::denspence<TOutput, TMass, TScale>(omarg, isomarg) - ql::cLn<TOutput, TMass, TScale>(omarg, isomarg) * ql::cLn<TOutput, TMass, TScale>(arg, isarg);
        }
        else
            return ql::denspence<TOutput, TMass, TScale>(arg, isarg);
    }


    /*!
    * Calculates Li[2](1-(z1+ieps1)*(z2+ieps2)) for complex z1,z2
    * Using +Li2(1-z1*z2)                           for z1*z2<1
    * and   -Li2(1-1/(z1*z2))-1/2*(ln(z1)+ln(z2))^2 for z1*z2>1
    * \param z1 input argument
    * \param z2 input argument
    * \param ieps1 sign of z1
    * \param ieps2 sign of z2
    * \return Li2 of the 1-product
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput cLi2omx2(TOutput const& z1, TOutput const& z2, TScale const& ieps1, TScale const& ieps2) {
        const TOutput arg = z1 * z2;
        const TScale ieps = ql::Sign(ql::Real(z2) * ieps1 + ql::Real(z1) * ieps2);

        TOutput prod, res;
        if (ql::kAbs(arg) <= ql::Constants<TScale>::_one())
        {
            if (arg == TOutput(ql::Constants<TScale>::_zero()) || arg == TOutput(ql::Constants<TScale>::_one()))
                prod = TOutput(ql::Constants<TScale>::_zero());
            else
                prod = (ql::cLn<TOutput, TMass, TScale>(z1, ieps1)+ ql::cLn<TOutput, TMass, TScale>(z2, ieps2)) * ql::cLn<TOutput, TMass, TScale>(TOutput(ql::Constants<TScale>::_one()) - arg, -ieps);
            res = TOutput(ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>()) - ql::denspence<TOutput, TMass, TScale>(arg, ieps) - prod;
        }
        else if (ql::kAbs(arg) > ql::Constants<TScale>::_one())
        {
            const TOutput arg2 = TOutput(ql::Constants<TScale>::_one()) / (z1 * z2);
            const TOutput lnomarg = ql::cLn<TOutput, TMass, TScale>(TOutput(ql::Constants<TScale>::_one()) - arg2, -ieps);
            const TOutput lnarg = -ql::cLn<TOutput, TMass, TScale>(z1, ieps1) - ql::cLn<TOutput, TMass, TScale>(z2, ieps2);
            res = TOutput(-ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>()) + ql::denspence<TOutput, TMass, TScale>(arg2, ieps) + lnarg * lnomarg - TOutput(ql::Constants<TScale>::_half()) * lnarg * lnarg;
        }
        return res;
    }


    /*!
    * expression for dilog(1-(v-i*ep)*(w-i*ep)/(x-i*ep)/(y-i*ep)) for real v,w,x and y
    * \param v numerator
    * \param w numerator
    * \param x denominator
    * \param y denominator
    * \return the dilog ratio
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput Li2omx2(TScale const& v, TScale const& w, TScale const& x, TScale const& y) {
        const TScale arg = (v * w) / (x * y);
        const TScale omarg = ql::Constants<TScale>::_one() - arg;
        TOutput prod, Li2omx2;

        if (ql::kAbs(arg) <= ql::Constants<TScale>::_one()) {
            if (ql::kAbs(arg) == ql::Constants<TScale>::_zero() || ql::kAbs(arg) == ql::Constants<TScale>::_one())
                prod = TOutput(ql::Constants<TScale>::_zero());
            else
                prod = (ql::Lnrat<TOutput, TMass, TScale>(v, x) + ql::Lnrat<TOutput, TMass, TScale>(w, y)) * TOutput(ql::kLog(omarg));
            Li2omx2 = TOutput(ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>() - ql::ddilog<TOutput, TMass, TScale>(arg)) - prod;
        }
        else if (ql::kAbs(arg) > ql::Constants<TScale>::_one()) {
            const TScale arg2 = (x * y) / (v * w);
            const TOutput lnarg = TOutput(-ql::Lnrat<TOutput, TMass, TScale>(v, x) - ql::Lnrat<TOutput, TMass, TScale>(w, y));
            const TOutput lnomarg = TOutput(ql::kLog(ql::Constants<TScale>::_one() - arg2));
            Li2omx2 = -TOutput(ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>() - ql::ddilog<TOutput, TMass, TScale>(arg2)) + lnarg * lnomarg - TOutput(ql::Constants<TScale>::_half()) * lnarg * lnarg;
        }

        return Li2omx2;
    }


    /*!
    * Calculates Li[2](1-(z1+ieps1)*(z2+ieps2)) for complex z1,z2
    * Using +Li2(1-z1*z2)                           for z1*z2<1
    * and   -Li2(1-1/(z1*z2))-1/2*(ln(z1)+ln(z2))^2 for z1*z2>1
    * \param z1 input argument
    * \param z2 input argument
    * \param ieps1 sign of z1
    * \param ieps2 sign of z2
    * \return Li2 of the 1-product
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput Li2omx2(TOutput const& v, TOutput const& w, TOutput const& x, TOutput const& y, TScale const& ieps1, TScale const& ieps2) {
        return ql::cLi2omx2<TOutput, TMass, TScale>(v / x, w / y, ieps1, ieps2);
    }


    /*!
    * Generalization of cLi2omx2 for 3 arguments.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput cLi2omx3(TOutput const& z1, TOutput const& z2, TOutput const& z3, TScale const& ieps1, TScale const& ieps2, TScale const& ieps3) {
        const TOutput arg = z1 * z2 * z3;

        TScale ieps = ql::Constants<TScale>::_zero();
        if (ql::iszero<TOutput, TMass, TScale>(ql::Imag(arg)))
            ieps = ql::Sign(ql::Real(z2 * z3) * ieps1 + ql::Real(z1 * z3) * ieps2 + ql::Real(z1 * z2) * ieps3);

        TOutput res = TOutput(ql::Constants<TScale>::_zero());
        if (ql::kAbs(arg) <= ql::Constants<TScale>::_one()) {
            TOutput prod;
            if (arg == TOutput(ql::Constants<TScale>::_zero()) || arg == TOutput(ql::Constants<TScale>::_one()))
                prod = TOutput(ql::Constants<TScale>::_zero());
            else {
                const TOutput lnarg = ql::cLn<TOutput, TMass, TScale>(z1, ieps1) + ql::cLn<TOutput, TMass, TScale>(z2, ieps2) + ql::cLn<TOutput, TMass, TScale>(z3, ieps3);
                const TOutput lnomarg = ql::cLn<TOutput, TMass, TScale>(TOutput(ql::Constants<TScale>::_one()) - arg, ql::Constants<TScale>::_zero());
                prod = lnarg * lnomarg;
            }
            res = TOutput(ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>()) - ql::denspence<TOutput, TMass, TScale>(arg, ieps) - prod;
        }
        else {
            const TOutput arg2 = TOutput(ql::Constants<TScale>::_one()) / (z1 * z2 * z3);
            const TOutput lnarg = -ql::cLn<TOutput, TMass, TScale>(z1, ieps1) - ql::cLn<TOutput, TMass, TScale>(z2, ieps2) - ql::cLn<TOutput, TMass, TScale>(z3, ieps3);
            const TOutput lnomarg = ql::cLn<TOutput, TMass, TScale>(TOutput(ql::Constants<TScale>::_one()) - arg2, ql::Constants<TScale>::_zero());
            res = - TOutput(ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>()) + ql::denspence<TOutput, TMass, TScale>(arg2, ieps) + lnarg * lnomarg - TOutput(ql::Constants<TScale>::_half()) * lnarg * lnarg;
        }

        return res;
    }


    /*!
    * \param x input argument
    * \param y input argument
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput L0(TMass const& x, TMass const& y) {
        TOutput L0;
        const TMass denom = TMass(ql::Constants<TMass>::_one()) - x / y;
        if (ql::kAbs(denom) < ql::Constants<TScale>::_eps7())
            L0 = -TOutput(ql::Constants<TScale>::_one()) - TOutput(denom * (TMass(ql::Constants<TMass>::_half()) + denom / TMass(ql::Constants<TMass>::_three())));
        else
            L0 = ql::Lnrat<TOutput, TMass, TScale>(x, y) / TOutput(denom);

        return L0;
    }


    /*!
    * \param x input argument
    * \param y input argument
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput L1(TMass const& x, TMass const& y) {
        TOutput L1;
        const TMass denom = TMass(ql::Constants<TMass>::_one()) - x / y;
        if (ql::kAbs(denom) < ql::Constants<TScale>::_eps7())
            L1 = -TOutput(ql::Constants<TScale>::_one()) * TOutput(ql::Constants<TScale>::_half()) - TOutput(denom / TMass(ql::Constants<TMass>::_three()) * (TMass(ql::Constants<TMass>::_one()) + TMass(ql::Constants<TMass>::_three()) * denom / TMass(ql::Constants<TMass>::_four())));
        else
            L1 = (ql::L0<TOutput, TMass, TScale>(x, y) + TOutput(ql::Constants<TScale>::_one())) / TOutput(denom);

        return L1;
    }


    /*!
    * Solution of a quadratic equation a*z^2+b*z+c=0.
    * \param a coefficient
    * \param b coefficient
    * \param c coefficient
    * \param l result [-im,+im]
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION void solveabc(TMass const& a, TMass const&b, TMass const& c, Kokkos::Array<TOutput, 2>& z) {
        const TMass discr = b * b - TMass(ql::Constants<TMass>::_four()) * a * c;

        if (ql::iszero<TOutput, TMass, TScale>(a)) {
            Kokkos::printf("solveabc -- equation is not quadratic");
        }

        if (ql::iszero<TOutput, TMass, TScale>(ql::Imag(discr))) {
            
            const TMass sgnb = ql::Sign(ql::Real(b));
            if (ql::iszero<TOutput, TMass, TScale>(ql::Real(b))) {
                z[0] = -(TOutput(b) - ql::kSqrt(TOutput(discr))) / (TOutput(ql::Constants<TScale>::_two()) * a);
                z[1] = -(TOutput(b) + ql::kSqrt(TOutput(discr))) / (TOutput(ql::Constants<TScale>::_two()) * a);
            }
            else {
                if (ql::Real(discr) > TMass(ql::Constants<TMass>::_zero())) {
                    const TMass q = TMass(-ql::Constants<TMass>::_half()) * (b + sgnb * ql::kSqrt(discr));
                    if (ql::Real(b) > 0) {
                        z[0] = TOutput(c / q);
                        z[1] = TOutput(q / a);
                    }
                    else {
                        z[0] = TOutput(q / a);
                        z[1] = TOutput(c / q);
                    }
                }
                else {
                    z[1] = -(TOutput(b) + sgnb * ql::kSqrt(TOutput(discr))) / (TOutput(ql::Constants<TScale>::_two()) * a);
                    z[0] = ql::kConj(z[1]);
                    if (ql::Real(b) < 0) {
                        z[0] = z[1];
                        z[1] = ql::kConj(z[0]);
                    }
                }
            }
        }
        else {
            TOutput qq = -TOutput(b) + ql::kSqrt(TOutput(discr));
            TOutput hh = -TOutput(b) - ql::kSqrt(TOutput(discr));

            z[0] = qq * TOutput(ql::Constants<TScale>::_half()) / TOutput(a);
            z[1] = (TOutput(ql::Constants<TScale>::_two()) * TOutput(c)) / qq;

            if (ql::Imag(z[0]) > ql::Constants<TScale>::_zero()) {
                z[0] = hh * TOutput(ql::Constants<TScale>::_half()) / TOutput(a);
                z[1] = (TOutput(ql::Constants<TScale>::_two()) * TOutput(c)) / hh;
            }
        }
    }


    /*!
    * Solution of a quadratic equation a*z^2+b*z+c=0, with a give input discriminant.
    * \param a coefficient
    * \param b coefficient
    * \param c coefficient
    * \param d discriminant
    * \param l result [-im,+im]
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION void solveabcd(TOutput const& a, TOutput const&b, TOutput const& c, TOutput const& d, Kokkos::Array<TOutput, 2>& z) {
        if (a == TOutput(ql::Constants<TScale>::_zero())) {
            if (b == TOutput(ql::Constants<TScale>::_zero())) {
                Kokkos::printf("solveabcd - no possible solution\n");
            }
            z[0] = - c / b; z[1] = z[0];
        }
        else if (c == TOutput(ql::Constants<TScale>::_zero())) {
            z[0] = d / a; z[1] = TOutput(ql::Constants<TScale>::_zero());
        }
        else {
            const TOutput up = - b + d;
            const TOutput dn = - b - d;
            if (ql::kAbs(up) >= ql::kAbs(dn)) {
                z[0] = TOutput(ql::Constants<TScale>::_half()) * up / a;
                z[1] = TOutput(ql::Constants<TScale>::_two()) * c / up;
            }
            else {
                z[1] = TOutput(ql::Constants<TScale>::_half()) * dn / a;
                z[0] = TOutput(ql::Constants<TScale>::_two()) * c / dn;
            }
        }
    }


    /*!
    * Solution of a quadratic equation a*z^2+b*z+c=0, with a give input discriminant.
    * \param a coefficient
    * \param b coefficient
    * \param c coefficient
    * \param d discriminant
    * \param l result [-im,+im]
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION void solveabcd(TOutput const& a, TOutput const&b, TOutput const& c, Kokkos::Array<TOutput, 2>& z) {
        if (a == TOutput(ql::Constants<TScale>::_zero())) {
            if (b == TOutput(ql::Constants<TScale>::_zero())) {
                Kokkos::printf("solveabcd - no possible solution\n");
            }
            z[0] = -c / b; z[1] = z[0];
        }
        else if (c == TOutput(ql::Constants<TScale>::_zero())) {
            z[0] = TOutput(ql::Constants<TScale>::_zero()); z[1] = TOutput(ql::Constants<TScale>::_zero());
        }
        else {
            const TOutput d = ql::kSqrt(b * b - TOutput(ql::Constants<TScale>::_four()) * a * c);
            const TOutput up = - b + d;
            const TOutput dn = - b - d;
            if (ql::kAbs(up) >= ql::kAbs(dn)) {
                z[0] = TOutput(ql::Constants<TScale>::_half()) * up / a;
                z[1] = TOutput(ql::Constants<TScale>::_two()) * c / up;
            }
            else {
                z[1] = TOutput(ql::Constants<TScale>::_half()) * dn / a;
                z[0] = TOutput(ql::Constants<TScale>::_two()) * c / dn;
            }
        }
    }


    /*!
    * Calculate the function
    * \f[
    * R(p^2,m,m_p) = \frac{p_3^2+m_4^2-m_3^2+\sqrt{(p_3^2+m_4^2-m_3^2)^2-4 p_3^2 m_4^2}}{-p_3^2+m_4^2-m_3^2+\sqrt{(p_3^2+m_4^2-m_3^2)^2-4 p_3^2 m_4^2}}
    * \f]
    * where the roots are allowed to be imaginary.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION void ratgam(TOutput &ratp, TOutput &ratm, TScale &ieps, TMass const& p3sq, TMass const& m3sq, TMass const& m4sq) {
        
        const TOutput root = ql::kSqrt(TOutput(ql::kPow<TOutput, TMass, TScale>(p3sq - m3sq + m4sq, 2) - TMass(ql::Constants<TMass>::_four()) * m4sq * p3sq));
        ratp = (TOutput(p3sq + m4sq - m3sq) + root) / (TOutput(-p3sq + m4sq - m3sq) + root);
        ratm = (TOutput(p3sq + m4sq - m3sq) - root) / (TOutput(-p3sq + m4sq - m3sq) - root);
        ieps = ql::Constants<TScale>::_zero();
    }



    /*!
    * Calculate the function
    * \f[
    * R = \frac{\sigma-i \epsilon}{\tau-i \epsilon}
    * \f]
    * where sigma and tau are real and ieps give the sign if i*pi.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION void ratreal(TMass const& si, TMass const& ta, TMass &rat, TScale &ieps) {
        rat = si / ta;
        if (ql::Real(rat) > ql::Constants<TScale>::_zero())
            ieps = ql::Constants<TScale>::_zero();
        else if (ql::Real(si) < ql::Constants<TScale>::_zero())
            ieps = -ql::Constants<TScale>::_one();
        else if (ql::Real(ta) < ql::Constants<TScale>::_zero())
            ieps = ql::Constants<TScale>::_one();
        else if (ql::Real(ta) == ql::Constants<TScale>::_zero()) {
            Kokkos::printf("error in ratreal\n");
        }
    }
    

    /*!
    * \brief Tools<TOutput, TMass, TScale>::qlZlogint
    * \param z
    * \param ieps
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput Zlogint(TOutput const& z, TScale const& ieps) {
        const TOutput omz = TOutput(ql::Constants<TScale>::_one()) - z;
        return omz * (ql::cLn<TOutput, TMass, TScale>(omz, ieps) - TOutput(ql::Constants<TScale>::_one())) - (-z) * (ql::cLn<TOutput, TMass, TScale>(-z, ieps) - TOutput(ql::Constants<TScale>::_one()));
    }


    /*!
    * Finite Triangle Li2 sum.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput R3int(TOutput const& a, TOutput const& s1, TOutput const& s2, TOutput const& t1, TOutput const& t2, TOutput const& t3, TOutput const& t4) {
        
        const TOutput b = (s1 + s2) * (s1 - s2) - a;
        const TOutput c = s2 * s2;
        const TOutput d = ql::kSqrt((a - (s1 + s2) * (s1 + s2)) * (a - (s1 - s2) * (s1 - s2)));
        Kokkos::Array<TOutput, 2> y;
        Kokkos::Array<TOutput, 2> s;
        TOutput res;
        ql::solveabcd<TOutput, TMass, TScale>(a, b, c, d, y);
        ql::solveabcd<TOutput, TMass, TScale>(a, t2, t3, t4, s);

        const TOutput y0 = -(t1 + b * s[0]) / t4;
        const TOutput dq0 = (y0 - y[0]);
        const TOutput dq1 = (y0 - y[1]);

        const TOutput OneOdq0 = TOutput(ql::Constants<TScale>::_one()) / dq0;
        const TOutput OneMy0 = TOutput(ql::Constants<TScale>::_one()) - y[0];
        const TScale SignImagOneOdq0 = ql::Sign(ql::Imag(OneOdq0));

        const TOutput OneOdq1 = TOutput(ql::Constants<TScale>::_one()) / dq1;
        const TOutput OneMy1 = TOutput(ql::Constants<TScale>::_one()) - y[1];
        const TScale SignImagOneOdq1 = ql::Sign(ql::Imag(OneOdq1));

        res = ql::cspence<TOutput, TMass, TScale>(-y[0], ql::Sign(ql::Imag(-y[0])), OneOdq0, SignImagOneOdq0)
             -ql::cspence<TOutput, TMass, TScale>(OneMy0, ql::Sign(ql::Imag(OneMy0)), OneOdq0, SignImagOneOdq0)
             +ql::cspence<TOutput, TMass, TScale>(-y[1], ql::Sign(ql::Imag(-y[1])), OneOdq1, SignImagOneOdq1)
             -ql::cspence<TOutput, TMass, TScale>(OneMy1, ql::Sign(ql::Imag(OneMy1)), OneOdq1, SignImagOneOdq1);

        TOutput zz = y0 * (a * y0 + b);

        if (ql::kAbs(ql::Real(zz)) * ql::Constants<TScale>::_reps() * ql::Constants<TScale>::_reps() <= ql::kAbs(ql::Imag(zz)) * ql::Constants<TScale>::_neglig() && ql::kAbs(ql::Imag(zz)) <= ql::kAbs(ql::Real(zz)) * ql::Constants<TScale>::_neglig())
            zz = (TOutput(ql::Real(zz)) + c) / a;
        else
            zz = (zz + c) / a;

        // ajust complex logs
        TOutput extra = ql::eta3<TOutput, TMass, TScale>(-y[0], -y[1], c / a) - ql::eta3<TOutput, TMass, TScale>(dq0, dq1, zz);
        if (ql::Real(a) < ql::Constants<TScale>::_zero() && ql::Imag(zz) < ql::Constants<TScale>::_zero()) 
            extra -= ql::Constants<TScale>::template _2ipi<TOutput, TMass, TScale>();
        if (extra != TOutput(ql::Constants<TScale>::_zero())) {
            const TOutput arg4 = (y0 - TOutput(ql::Constants<TScale>::_one())) / y0;
            res += extra * ql::cLn<TOutput, TMass, TScale>(arg4, ql::Sign(ql::Imag(arg4)));
        }

        return res;
    }


    /*!
    * Finite Triangle Li2 sum.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput R3int(TOutput const& a, TOutput const& s1, TOutput const& s2, TOutput const& t1) {
        const TOutput b = (s1 + s2) * (s1 - s2) - a;
        const TOutput c = s2 * s2;
        const TOutput d = ql::kSqrt((a - (s1 + s2) * (s1 + s2)) * (a - (s1 - s2) * (s1 - s2)));
        
        Kokkos::Array<TOutput, 2> y;
        TOutput res;
        ql::solveabcd<TOutput, TMass, TScale>(a, b, c, d, y);

        const TOutput y0 = t1;
        const TOutput dq0 = (y0 - y[0]);
        const TOutput dq1 = (y0 - y[1]);

        const TOutput OneOdq0 = TOutput(ql::Constants<TScale>::_one()) / dq0;
        const TOutput OneMy0 = TOutput(ql::Constants<TScale>::_one()) - y[0];
        const TScale SignImagOneOdq0 = ql::Sign(ql::Imag(OneOdq0));

        const TOutput OneOdq1 = TOutput(ql::Constants<TScale>::_one()) / dq1;
        const TOutput OneMy1 = TOutput(ql::Constants<TScale>::_one()) - y[1];
        const TScale SignImagOneOdq1 = ql::Sign(ql::Imag(OneOdq1));

        res = ql::cspence<TOutput, TMass, TScale>(-y[0], ql::Sign(ql::Imag(-y[0])), OneOdq0, SignImagOneOdq0)
             -ql::cspence<TOutput, TMass, TScale>(OneMy0, ql::Sign(ql::Imag(OneMy0)), OneOdq0, SignImagOneOdq0)
             +ql::cspence<TOutput, TMass, TScale>(-y[1], ql::Sign(ql::Imag(-y[1])), OneOdq1, SignImagOneOdq1)
             -ql::cspence<TOutput, TMass, TScale>(OneMy1, ql::Sign(ql::Imag(OneMy1)), OneOdq1, SignImagOneOdq1);

        TOutput zz = y0 * (a * y0 + b);

        if (ql::kAbs(ql::Real(zz)) * ql::Constants<TScale>::_reps() * ql::Constants<TScale>::_reps() <= ql::kAbs(ql::Imag(zz)) * ql::Constants<TScale>::_neglig() && ql::kAbs(ql::Imag(zz)) <= ql::kAbs(ql::Real(zz)) * ql::Constants<TScale>::_neglig())
            zz = (TOutput(ql::Real(zz)) + c) / a;
        else
            zz = (zz + c) / a;

        // ajust complex logs
        TOutput extra = ql::eta3<TOutput, TMass, TScale>(-y[0], -y[1], c / a) - ql::eta3<TOutput, TMass, TScale>(dq0, dq1, zz);
        if (ql::Real(a) < ql::Constants<TScale>::_zero() && ql::Imag(zz) < ql::Constants<TScale>::_zero())
            extra -= ql::Constants<TScale>::template _2ipi<TOutput, TMass, TScale>();
        if (extra != TOutput(ql::Constants<TScale>::_zero())) {
            const TOutput arg4 = (y0 - TOutput(ql::Constants<TScale>::_one()))/y0;
            res += extra * ql::cLn<TOutput, TMass, TScale>(arg4, ql::Sign(ql::Imag(arg4)));
        }

        return res;
    }


    /*!
    * Finite Triangle Li2 sum.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput R2int(TOutput const& a, TOutput const& b, TOutput const& y0) {
        
        const TOutput y1 = -b / a;
        const TOutput dq0 = y0 - y1;

        const TOutput oneOdq0 = TOutput(ql::Constants<TScale>::_one()) / dq0;
        const TScale  SignImagOneOdq0 = ql::Sign(ql::Imag(oneOdq0));
        const TOutput oneMy1 = TOutput(ql::Constants<TScale>::_one()) - y1;

        TOutput res = ql::cspence<TOutput, TMass, TScale>(-y1, ql::Sign(ql::Imag(-y1)), oneOdq0, SignImagOneOdq0)
                     -ql::cspence<TOutput, TMass, TScale>(oneMy1, ql::Sign(ql::Imag(oneMy1)), oneOdq0, SignImagOneOdq0);

        TOutput extra = ql::eta5<TOutput, TMass, TScale>(a, -y1, b, dq0, a * dq0);
        if (extra != TOutput(ql::Constants<TScale>::_zero())) {
            const TOutput arg4 = (y0 - TOutput(ql::Constants<TScale>::_one())) / y0;
            res += extra * ql::cLn<TOutput, TMass, TScale>(arg4, ql::Sign(ql::Imag(arg4)));
        }
        return res;
    }


    /*!
    * \param y
    * \param z
    * \param ieps
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION TOutput Rint(TOutput const& y, TOutput const& z, TScale const& ieps) {
        
        const TOutput omz = TOutput(ql::Constants<TScale>::_one()) - z;
        const TOutput ymone = y - TOutput(ql::Constants<TScale>::_one());
        const TOutput invymz = TOutput(ql::Constants<TScale>::_one())/(y - z);
        const TOutput c[2] = { y * invymz, ymone * invymz }; 

        if (ql::Imag(z) != ql::Constants<TScale>::_zero()) {
            const TOutput c2ipi = TOutput{ql::Constants<TScale>::_zero(), ql::Constants<TScale>::_two() * ql::Constants<TScale>::_pi()};
            const TOutput a = -z;
            const TOutput b = invymz;
            const TOutput ab= -z * invymz;
            const TOutput eta1 = c2ipi * TOutput( ql::Htheta(-ql::Imag(a)) * ql::Htheta(-ql::Imag(b)) * ql::Htheta(Imag(ab))
                                                 -ql::Htheta(ql::Imag(a)) * ql::Htheta(Imag(b)) * ql::Htheta(-ql::Imag(ab)) );
            const TOutput a2 = omz;
            const TOutput ab2= omz * invymz;
            const TOutput eta2 = c2ipi * TOutput( ql::Htheta(-ql::Imag(a2)) * ql::Htheta(-ql::Imag(b)) * ql::Htheta(ql::Imag(ab2))
                                                 -ql::Htheta(ql::Imag(a2)) * ql::Htheta(ql::Imag(b)) * ql::Htheta(-ql::Imag(ab2)) );

            TOutput logc1 = TOutput(ql::Constants<TScale>::_zero()), logc2 = TOutput(ql::Constants<TScale>::_zero());
            if (eta1 != TOutput(ql::Constants<TScale>::_zero())) logc1 = TOutput(ql::kLog(c[0]));
            if (eta2 != TOutput(ql::Constants<TScale>::_zero())) logc2 = TOutput(ql::kLog(c[1]));

            return ql::denspence<TOutput, TMass, TScale>(c[0], ql::Constants<TScale>::_zero()) - ql::denspence<TOutput, TMass, TScale>(c[1], ql::Constants<TScale>::_zero()) + eta1 * logc1 - eta2 * logc2;
        }
        else {
            const TScale ieps1 = -ieps * ql::Sign(ql::Real(y));
            const TScale ieps2 = -ieps * ql::Sign(ql::Real(ymone));
            return ql::denspence<TOutput, TMass, TScale>(c[0], ieps1) - ql::denspence<TOutput, TMass, TScale>(c[1], ieps2);
        }
    }


    /*!
    * \brief Tools<TOutput, TMass, TScale>::R
    * \param q
    * \return
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION void R(TOutput &r, TOutput &d, TOutput const& q) {
        d = ql::kSqrt(q * q - TOutput(ql::Constants<TScale>::_four()));
        r  = q + d;
        TOutput r2 = q - d;
        TScale a = ql::kAbs(r), b = ql::kAbs(r2);
        if (b > a) { 
            r = r2; 
            d = -d;
        }
        a = ql::Imag(q);
        b = ql::Imag(r);
        if (a == ql::Constants<TScale>::_zero()) {
            if (b <= ql::Constants<TScale>::_zero())
                r /= TOutput(ql::Constants<TScale>::_two());
            else {
                r = TOutput(ql::Constants<TScale>::_two()) / r;
                d = -d;
            }
        }
        else {
            const TScale ik = ql::Sign(ql::Real(a));
            const TScale ir = ql::Sign(ql::Real(b));
            if (ir == ik)
                r /= TOutput(ql::Constants<TScale>::_two());
            else {
                r = TOutput(ql::Constants<TScale>::_two()) / r;
                d = -d;
            }
        }
        return;
    }


    /*!
    * Calculate the K-function give in Eq. 2.7 of \cite Beenakker:1988jr
    * \f[
    *  K(p^2,m,m_p) = \frac{1-\sqrt{1-4m m_p / (z-(m-m_p)^2)}}{1+\sqrt{1-4m m_p / (z-(m-m_p)^2)}}
    * \f]
    * and fill x[0] = -K, x[1] = 1+K, x[2] = 1-K, the roots are allowed to be imaginary.
    * ieps gives the sign of the imaginary part of -K: 1 -> +i*eps
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION void kfn(Kokkos::Array<TOutput, 3> &res, TScale& ieps, TMass const& xpi, TMass const& xm, TMass const& xmp) {
        
        if (xm == TMass(ql::Constants<TMass>::_zero()) || xmp == TMass(ql::Constants<TMass>::_zero())) {
            Kokkos::printf("Error in kfn,xm,xmp");
        }
        const TOutput xx1 = TOutput(xpi - (xm - xmp) * (xm - xmp));
        const TOutput rat = TOutput(xx1 / (TMass(ql::Constants<TMass>::_four()) * xm * xmp));
        if (ql::iszero<TOutput, TMass, TScale>(ql::Real(rat))) {
            res[1] = -TOutput(ql::Constants<TScale>::_two()) * ql::kSqrt(rat) * TOutput{ql::Constants<TScale>::_zero(), ql::Constants<TScale>::_one()} + TOutput(ql::Constants<TScale>::_two()) * rat;
            res[0] = TOutput(ql::Constants<TScale>::_one()) - res[1];
            res[2] = TOutput(ql::Constants<TScale>::_two()) - res[1];
        } 
        else {
            const TOutput root = ql::kSqrt((rat - TOutput(ql::Constants<TScale>::_one())) / rat);
            const TOutput invopr = TOutput(ql::Constants<TScale>::_one())/(TOutput(ql::Constants<TScale>::_one()) + root);
            res[0] = -invopr * invopr / rat;
            res[1] = TOutput(ql::Constants<TScale>::_two()) * invopr;
            res[2] = TOutput(ql::Constants<TScale>::_two()) * root * invopr;
        }
        ieps = ql::Constants<TScale>::_one();
    }

}
