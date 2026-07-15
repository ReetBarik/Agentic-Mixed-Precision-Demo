//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//
// Box integrals with 3 internal masses.
// Contains: BIN3, B16, B3m dispatcher, pruned BO.

#pragma once

#include "box_common.h"


namespace ql
{

    KOKKOS_INLINE_FUNCTION
    constexpr int swap_b3m(int i, int j) {
        constexpr int val[13][4] = { 
            {0, 3, 2, 1},
            {1, 0, 3, 2},
            {2, 1, 0, 3},
            {3, 2, 1, 0},
            {4, 7, 6, 5},
            {5, 4, 7, 6},
            {6, 5, 4, 7},
            {7, 6, 5, 4},
            {8, 9, 8, 9},
            {9, 8, 9, 8},
            {10, 10, 10, 10},
            {11, 12, 11, 12},
            {12, 11, 12, 11},
        };
        return val[i][j];
    }

    /*!
    * Finite box with 3 non-zero masses. Formulae from \cite Denner:1991qq.
    * \param res output object res[0,1,2] the coefficients in the Laurent series, following the LoopTools implementation \cite Hahn:2006qw.
    * \param Y the modified Cayley matrix.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void BIN3(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        const int i) {

        const TMass m2 = Y[1][1];
        const TMass m3 = Y[2][2];
        const TMass m4 = Y[3][3];
        const TMass m_0 = ql::kSqrt(m3 * m4);
        const TMass m_1 = ql::kSqrt(m2 * m3);
        const TMass m_2 = ql::kSqrt(m2 * m4);

        const TMass k12 = ql::Constants<TMass>::_two() * Y[2][3] / m_0;
        const TMass k13 = ql::Constants<TMass>::_two() * Y[0][2] / m3;
        const TMass k14 = ql::Constants<TMass>::_two() * Y[1][2] / m_1;
        const TMass k23 = ql::Constants<TMass>::_two() * Y[0][3] / m_0;
        const TMass k24 = ql::Constants<TMass>::_two() * Y[1][3] / m_2;
        const TMass k34 = ql::Constants<TMass>::_two() * Y[0][1] / m_1;

        int ir12 = 0, ir14 = 0, ir24 = 0;
        const TOutput r12 = ql::Constants<TOutput>::_half() * (TOutput(k12) + TOutput(ql::Sign(ql::Real(k12))) * ql::kSqrt(TOutput((k12 - ql::Constants<TMass>::_two()) * (k12 + ql::Constants<TMass>::_two()))));
        const TOutput r14 = ql::Constants<TOutput>::_half() * (TOutput(k14) + TOutput(ql::Sign(ql::Real(k14))) * ql::kSqrt(TOutput((k14 - ql::Constants<TMass>::_two()) * (k14 + ql::Constants<TMass>::_two()))));
        const TOutput r24 = ql::Constants<TOutput>::_half() * (TOutput(k24) + TOutput(ql::Sign(ql::Real(k24))) * ql::kSqrt(TOutput((k24 - ql::Constants<TMass>::_two()) * (k24 + ql::Constants<TMass>::_two()))));
        if (ql::Real(k12) < -ql::Constants<TMass>::_two()) ir12 = ql::Constants<TScale>::_ten() * ql::Sign(ql::Constants<TScale>::_one() - ql::kAbs(r12));
        if (ql::Real(k14) < -ql::Constants<TMass>::_two()) ir14 = ql::Constants<TScale>::_ten() * ql::Sign(ql::Constants<TScale>::_one() - ql::kAbs(r14));
        if (ql::Real(k24) < -ql::Constants<TMass>::_two()) ir24 = ql::Constants<TScale>::_ten() * ql::Sign(ql::Constants<TScale>::_one() - ql::kAbs(r24));

        const TOutput q24 = r24 - ql::Constants<TOutput>::_one() / r24;
        const TOutput q12 = TOutput(k12) - r24 * TOutput(k14);

        const TOutput a = TOutput(k34) / r24 - TOutput(k23);
        const TOutput b = TOutput(k12 * k34) - TOutput(k13) * q24 - TOutput(k14 * k23);
        const TOutput c = TOutput(k13) * q12 + r24 * TOutput(k34) - TOutput(k23);
        const TOutput d = TOutput((k12 * k34 - k13 * k24 - k14 * k23) * (k12 * k34 - k13 * k24 - k14 * k23)) -
                        TOutput(ql::Constants<TScale>::_four()) * TOutput(k13 * (k13 - k23 * (k12 - k14 * k24)) +
                                        k23 * (k23 - k24 * k34) + k34 * (k34 - k13 * k14));
        const TOutput discr = ql::kSqrt(d);

        Kokkos::Array<TOutput, 2> x4;
        Kokkos::Array<TOutput, 2> x1;
        Kokkos::Array<TOutput, 2> l4;
        x4[0] = ql::Constants<TOutput>::_half() * (b - discr) / a;
        x4[1] = ql::Constants<TOutput>::_half() * (b + discr) / a;
        if (ql::kAbs(x4[0]) > ql::kAbs(x4[1]))
            x4[1] = c / (a * x4[0]);
        else
            x4[0] = c / (a * x4[1]);

        const TOutput dd = -TOutput(k34) * r24 + TOutput(k23);
        Kokkos::Array<TScale, 2> ix4;
        Kokkos::Array<TScale, 2> ix1;
        ix4[0] = ql::Sign(ql::Real(dd));
        ix4[1] = -ix4[0];

        x1[0] = x4[0] / r24;
        x1[1] = x4[1] / r24;
        ix1[0] = ql::Sign(ix4[0] * ql::Real(r24));
        ix1[1] = -ix1[0];

        const TOutput cc = ql::cLn<TOutput, TMass, TScale>(ql::Real(k13), -ql::Constants<TScale>::_one());
        l4[0] = cc + ql::cLn<TOutput, TMass, TScale>((q12 + q24 * x4[0]) / dd, ql::Real(q24 * ix4[0] / dd));
        l4[1] = cc + ql::cLn<TOutput, TMass, TScale>((q12 + q24 * x4[1]) / dd, ql::Real(q24 * ix4[1] / dd));

        res(i, 2) = res(i, 1) = ql::Constants<TOutput>::_zero();
        res(i, 0) = (
            ql::xspence<TOutput, TMass, TScale>(x4, ix4, r14, ir14) +
            ql::xspence<TOutput, TMass, TScale>(x4, ix4, ql::Constants<TOutput>::_one() / r14, -ir14) -
            ql::xspence<TOutput, TMass, TScale>(x4, ix4, TOutput(k34 / k13), -ql::Real(k13)) -
            ql::xspence<TOutput, TMass, TScale>(x1, ix1, r12, ir12) -
            ql::xspence<TOutput, TMass, TScale>(x1, ix1, ql::Constants<TOutput>::_one() / r12, -ir12) +
            ql::xspence<TOutput, TMass, TScale>(x1, ix1, TOutput(k23 / k13), -ql::Real(k13)) -
            TOutput{ql::Constants<TScale>::_zero(), ql::Constants<TScale>::_two() * ql::Constants<TScale>::_pi()} *
                ql::xetatilde<TOutput, TMass, TScale>(x4, ix4, ql::Constants<TOutput>::_one() / r24, -ir24, l4)
        ) / (TOutput(m3 * m_2) * discr);
    }

    /*!
    * Implementation of the formulae from Beenakker et al. \cite Beenakker:1988jr.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B16(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass m2sq = Y[1][1];
        const TMass m3sq = Y[2][2];
        const TMass m4sq = Y[3][3];
        const TMass tabar = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass si = ql::Constants<TMass>::_two() * Y[1][3] - Y[1][1] - Y[3][3];
        const TMass mp2sq = ql::Constants<TMass>::_two() * Y[1][2] - m3sq - m2sq;
        const TMass mp3sq = ql::Constants<TMass>::_two() * Y[2][3] - m3sq - m4sq;
        const TMass m2 = ql::kSqrt(m2sq);
        const TMass m3 = ql::kSqrt(m3sq);
        const TMass m4 = ql::kSqrt(m4sq);
        const TMass mean = ql::kSqrt(ql::Real(m3sq) * mu2);

        TScale ieps = ql::Constants<TScale>::_zero(), iep2 = ql::Constants<TScale>::_zero(), iep3 = ql::Constants<TScale>::_zero();
        Kokkos::Array<TOutput, 3> cxs;
        Kokkos::Array<TOutput, 3> cx2;
        Kokkos::Array<TOutput, 3> cx3;
        ql::kfn<TOutput, TMass, TScale>(cxs, ieps, -si, m2, m4);
        ql::kfn<TOutput, TMass, TScale>(cx2, iep2, -mp2sq, m2, m3);
        ql::kfn<TOutput, TMass, TScale>(cx3, iep3, -mp3sq, m3, m4);

        const TOutput xs = cxs[0];
        const TScale rexs = ql::Real(xs);
        const TScale imxs = ql::Imag(xs);

        TOutput fac;
        if (ql::iszero<TOutput, TMass, TScale>(rexs - ql::Constants<TScale>::_one()) && ql::iszero<TOutput, TMass, TScale>(imxs)) {
            fac = TOutput(-ql::Constants<TMass>::_half() / (m2 * m4 * tabar));
            res(i,2) = ql::Constants<TOutput>::_zero();
            res(i,1) = ql::Constants<TOutput>::_one();
            res(i,0) = ql::Constants<TOutput>::_two() * ql::Lnrat<TOutput, TMass, TScale>(mean, tabar) - ql::Constants<TOutput>::_two();

            if (ql::iszero<TOutput, TMass, TScale>(ql::Real(cx2[0] - cx3[0])) && ql::iszero<TOutput, TMass, TScale>(ql::Imag(cx2[0] - cx3[0]))
                && ql::iszero<TOutput, TMass, TScale>(ql::Real(cx2[0] - ql::Constants<TOutput>::_one())) && ql::iszero<TOutput, TMass, TScale>(ql::Imag(cx2[0])))
                res(i,0) += ql::Constants<TOutput>::_four();
            else if (ql::iszero<TOutput, TMass, TScale>(ql::Real(cx2[0] - cx3[0])) && ql::iszero<TOutput, TMass, TScale>(ql::Imag(cx2[0] - cx3[0])))
                res(i,0) += ql::Constants<TOutput>::_two() + ql::Constants<TOutput>::_two() * (cx2[0] * cx2[0] + ql::Constants<TOutput>::_one()) * ql::cLn<TOutput, TMass, TScale>(cx2[0], iep2) / (cx2[0] * cx2[0] - ql::Constants<TOutput>::_one());
            else {
                const TOutput cln_cx2_iep2 = ql::cLn<TOutput, TMass, TScale>(cx2[0], iep2);
                const TOutput cln_cx3_iep3 = ql::cLn<TOutput, TMass, TScale>(cx3[0], iep3);
                res(i,0) += -(ql::Constants<TOutput>::_one() + cx2[0] * cx3[0]) / (ql::Constants<TOutput>::_one() - cx2[0] * cx3[0])
                    * (cln_cx2_iep2 + cln_cx3_iep3)
                    -(ql::Constants<TOutput>::_one() + cx2[0] / cx3[0]) / (ql::Constants<TOutput>::_one() - cx2[0] / cx3[0])
                    *(cln_cx2_iep2 - cln_cx3_iep3);
            }
        } else {
            fac = TOutput(-ql::Constants<TMass>::_one() / (m2 * m4 * tabar)) * cxs[0] / (ql::Constants<TOutput>::_one() - cxs[0] * cxs[0]);
            const TOutput xlog = ql::cLn<TOutput, TMass, TScale>(xs, ieps);
            const TOutput cln_cx2_iep2 = ql::cLn<TOutput, TMass, TScale>(cx2[0], iep2);
            const TOutput cln_cx3_iep3 = ql::cLn<TOutput, TMass, TScale>(cx3[0], iep3);
            res(i,2) = ql::Constants<TOutput>::_zero();
            res(i,1) = -xlog;
            res(i,0) = -ql::Constants<TOutput>::_two() * xlog * ql::Lnrat<TOutput, TMass, TScale>(mean, tabar)
                + cln_cx2_iep2 * cln_cx2_iep2 + cln_cx3_iep3 * cln_cx3_iep3
                - ql::cLi2omx2<TOutput, TMass, TScale>(xs, xs, ieps, ieps)
                + ql::cLi2omx3<TOutput, TMass, TScale>(cxs[0], cx2[0], cx3[0], ieps, iep2, iep3)
                + ql::cLi2omx3<TOutput, TMass, TScale>(cxs[0], ql::Constants<TOutput>::_one() / cx2[0], ql::Constants<TOutput>::_one() / cx3[0], ieps, -iep2, -iep3)
                + ql::cLi2omx3<TOutput, TMass, TScale>(cxs[0], cx2[0], ql::Constants<TOutput>::_one() / cx3[0], ieps, iep2, -iep3)
                + ql::cLi2omx3<TOutput, TMass, TScale>(cxs[0], ql::Constants<TOutput>::_one() / cx2[0], cx3[0], ieps, -iep2, iep3);            
        }

        for (size_t j = 0; j < 3; j++)
            res(i,j) *= TOutput(fac);

    }

    /*!
    * This function triggers the topologies with 3 internal masses.
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param xpi array with masses and momenta squared
    * \param mu2 is the square of the scale mu
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B3m(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<TMass, 13>& xpi, 
        const TScale& mu2,
        const int i) {

        int jsort = 0;
        for (int j = 0; j < 4; j++) 
            if (ql::iszero<TOutput, TMass, TScale>(xpi[j])) jsort = j;
            

        Kokkos::Array<TMass, 13> xpo;
        for (size_t j = 0; j < 13; j++) 
            xpo[ql::swap_b3m(j, jsort)] = xpi[j];
        

        Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Y;
        Y[0][0] = xpo[0];
        Y[1][1] = xpo[1];
        Y[2][2] = xpo[2];
        Y[3][3] = xpo[3];
        Y[0][1] = Y[1][0] = ql::Constants<TMass>::_half() * (xpo[0] + xpo[1] - xpo[4]);
        Y[0][2] = Y[2][0] = ql::Constants<TMass>::_half() * (xpo[0] + xpo[2] - xpo[8]);
        Y[0][3] = Y[3][0] = ql::Constants<TMass>::_half() * (xpo[0] + xpo[3] - xpo[7]);
        Y[1][2] = Y[2][1] = ql::Constants<TMass>::_half() * (xpo[1] + xpo[2] - xpo[5]);
        Y[1][3] = Y[3][1] = ql::Constants<TMass>::_half() * (xpo[1] + xpo[3] - xpo[9]);
        Y[2][3] = Y[3][2] = ql::Constants<TMass>::_half() * (xpo[2] + xpo[3] - xpo[6]);

        if (ql::iszero<TOutput, TMass, TScale>(Y[0][0]) && ql::iszero<TOutput, TMass, TScale>(Y[0][1]) && ql::iszero<TOutput, TMass, TScale>(Y[0][3])) 
            ql::B16<TOutput, TMass, TScale>(res, Y, mu2, i);
        else 
            ql::BIN3<TOutput, TMass, TScale>(res, Y, i);
    }

#ifndef QCDLOOP_BOX_FULL_DISPATCH
    /*!
    * Pruned BO for massive == 3 (3 internal masses).
    * Includes only the B3m dispatch path.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void BO(
        const Kokkos::View<TOutput* [3]>& res,
        const Kokkos::View<TScale*>& mu2,
        const Kokkos::View<TMass* [4]>& m,
        const Kokkos::View<TScale* [6]>& p,
        const int i) {
        
        const TScale scalefac = ql::Max(
            ql::kAbs(p(i, 4)),
            ql::Max(ql::kAbs(p(i, 5)),
            ql::Max(ql::kAbs(p(i, 0)),
            ql::Max(ql::kAbs(p(i, 1)),
            ql::Max(ql::kAbs(p(i, 2)),
            ql::kAbs(p(i, 3)))))));
        
        Kokkos::Array<TMass, 13> xpi;
        xpi[0] = m(i, 0) / scalefac;
        xpi[1] = m(i, 1) / scalefac;
        xpi[2] = m(i, 2) / scalefac;
        xpi[3] = m(i, 3) / scalefac;
        xpi[4] = TMass(p(i, 0) / scalefac);
        xpi[5] = TMass(p(i, 1) / scalefac);
        xpi[6] = TMass(p(i, 2) / scalefac);
        xpi[7] = TMass(p(i, 3) / scalefac);
        xpi[8] = TMass(p(i, 4) / scalefac);
        xpi[9] = TMass(p(i, 5) / scalefac);
        xpi[10] = xpi[4] + xpi[5] + xpi[6] + xpi[7] - xpi[8] - xpi[9];
        xpi[11] = -xpi[4] + xpi[5] - xpi[6] + xpi[7] + xpi[8] + xpi[9];
        xpi[12] = xpi[4] - xpi[5] + xpi[6] - xpi[7] + xpi[8] + xpi[9];
        
        const TScale musq = mu2(i) / scalefac;
        
        int massive = 0;
        for (size_t j = 0; j < 4; j++) {
            if (!ql::iszero<TOutput, TMass, TScale>(ql::kAbs(xpi[j]))) 
                massive += 1;
        }
        
        const TMass y13 = xpi[0] + xpi[2] - xpi[8];
        const TMass y24 = xpi[1] + xpi[3] - xpi[9];
        
        if (ql::iszero<TOutput, TMass, TScale>(y13) || 
            ql::iszero<TOutput, TMass, TScale>(y24)) {
            res(i, 0) = ql::Constants<TOutput>::_zero();
            res(i, 1) = ql::Constants<TOutput>::_zero();
            res(i, 2) = ql::Constants<TOutput>::_zero();
            return;
        }
        
        if (massive == 3) {
            ql::B3m<TOutput, TMass, TScale>(res, xpi, musq, i);
        } else {
            res(i, 0) = ql::Constants<TOutput>::_zero();
            res(i, 1) = ql::Constants<TOutput>::_zero();
            res(i, 2) = ql::Constants<TOutput>::_zero();
            return;
        }
        
        const TScale scalefac2 = scalefac * scalefac;
        res(i, 0) /= scalefac2;
        res(i, 1) /= scalefac2;
        res(i, 2) /= scalefac2;
    }
#endif

}
