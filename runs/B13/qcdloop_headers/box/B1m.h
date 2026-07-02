//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//
// Box integrals with 1 internal mass.
// Contains: BIN1, B6, B7, B8, B9, B10, B1m dispatcher, pruned BO.

#pragma once

#include "box_common.h"


namespace ql
{

    KOKKOS_INLINE_FUNCTION
    constexpr int swap_b1m(int i, int j) {
        constexpr int val[13][5] = { 
            {3, 2, 1, 0, 2},
            {0, 3, 2, 1, 1},
            {1, 0, 3, 2, 0},
            {2, 1, 0, 3, 3},
            {7, 6, 5, 4, 5},
            {4, 7, 6, 5, 4},
            {5, 4, 7, 6, 7},
            {6, 5, 4, 7, 6},
            {9, 8, 9, 8, 8},
            {8, 9, 8, 9, 9},
            {10, 10, 10, 10, 10},
            {12, 11, 12, 11, 11},
            {11, 12, 11, 12, 12},
        };
        return val[i][j];
    }

    /*!
    * Finite box with 1 non-zero mass. Formulae from \cite Denner:1991qq.
    * \param res output object res[0,1,2] the coefficients in the Laurent series, following the LoopTools implementation \cite Hahn:2006qw.
    * \param Y the modified Cayley matrix.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void BIN1(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        const int i) {

        const TMass m4 = Y[3][3];
        const TMass k34 = ql::Constants<TMass>::_two() * Y[0][1] / m4;
        const TMass k23 = ql::Constants<TMass>::_two() * Y[0][2] / m4;
        const TMass k13 = ql::Constants<TMass>::_two() * Y[0][3] / m4;
        const TMass k24 = ql::Constants<TMass>::_two() * Y[1][2] / m4;
        const TMass k14 = ql::Constants<TMass>::_two() * Y[1][3] / m4;
        const TMass k12 = ql::Constants<TMass>::_two() * Y[2][3] / m4;

        const TOutput k12c = TOutput(k12 - ql::Max(ql::kAbs(k12), TMass(ql::Constants<TMass>::_one())) * ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>());
        const TOutput k13c = TOutput(k13 - ql::Max(ql::kAbs(k13), TMass(ql::Constants<TMass>::_one())) * ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>());
        const TOutput k14c = TOutput(k14 - ql::Max(ql::kAbs(k14), TMass(ql::Constants<TMass>::_one())) * ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>());
        const TOutput k23c = TOutput(k23 - ql::Max(ql::kAbs(k23), TMass(ql::Constants<TMass>::_one())) * ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>());
        const TOutput k24c = TOutput(k24 - ql::Max(ql::kAbs(k24), TMass(ql::Constants<TMass>::_one())) * ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>()) / k12c;
        const TOutput k34c = TOutput(k34 - ql::Max(ql::kAbs(k34), TMass(ql::Constants<TMass>::_one())) * ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>()) / k13c;

        const TMass a = k34 * k24;
        const TMass b = k13 * k24 + k12 * k34 - k14 * k23;
        const TOutput c = TOutput(k13 * k12) - TOutput(k23) * (ql::Constants<TOutput>::_one() - ql::Constants<TOutput>::template _ieps50<TOutput, TMass, TScale>());
        const TOutput disc = ql::kSqrt(TOutput(b * b) - TOutput(ql::Constants<TScale>::_four()) * TOutput(a) * c);

        Kokkos::Array<TOutput, 2> x4;
        x4[0] = ql::Constants<TOutput>::_half() * (TOutput(b) - disc) / TOutput(a);
        x4[1] = ql::Constants<TOutput>::_half() * (TOutput(b) + disc) / TOutput(a);
        if (ql::kAbs(x4[0]) > ql::kAbs(x4[1]))
            x4[1] = c / (TOutput(a) * x4[0]);
        else
            x4[0] = c / (TOutput(a) * x4[1]);

        const Kokkos::Array<TScale, 2> imzero = {ql::Constants<TScale>::_zero(), ql::Constants<TScale>::_zero()};
        res(i, 0) = (
            ql::xspence<TOutput, TMass, TScale>(x4, imzero, k14c, ql::Constants<TScale>::_zero()) -
            ql::xspence<TOutput, TMass, TScale>(x4, imzero, k34c, ql::Constants<TScale>::_zero()) -
            ql::xspence<TOutput, TMass, TScale>(x4, imzero, k24c, ql::Constants<TScale>::_zero()) +
            (ql::kLog(x4[1]) - ql::kLog(x4[0])) * (ql::kLog(k12c) + ql::kLog(k13c) - ql::kLog(k23c))
            ) / (TOutput(ql::kPow<TOutput, TMass, TScale>(m4, 2)) * disc);

        res(i, 1) = res(i, 2) = ql::Constants<TOutput>::_zero();
    }

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B6(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass si = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass tabar = ql::Constants<TMass>::_two() * Y[1][3];
        const TMass msq = Y[3][3];
        const TOutput wlogs = ql::Lnrat<TOutput, TMass, TScale>(si, msq);
        const TOutput wlogt = ql::Lnrat<TOutput, TMass, TScale>(tabar, msq);
        const TOutput wlogm = ql::Lnrat<TOutput, TMass, TScale>(mu2, msq);

        res(i,2) = ql::Constants<TOutput>::_two();  
        res(i,1) = ql::Constants<TOutput>::_two() * (wlogm - wlogt) - wlogs;
        res(i,0) = wlogm * wlogm - wlogm * (ql::Constants<TOutput>::_two() * wlogt + wlogs) 
                + ql::Constants<TOutput>::_two() * wlogt * wlogs - ql::Constants<TOutput>::_half() * ql::Constants<TScale>::_pi2();

        const TOutput d = TOutput(si * tabar);
        for (size_t j = 0; j < 3; j++) {
            res(i,j) /= d;
        }
    }

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B7(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {
    
        const TMass tabar = ql::Constants<TMass>::_two() * Y[1][3];
        const TMass p4sqbar = ql::Constants<TMass>::_two() * Y[0][3];
        const TMass si = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass msq = Y[3][3];
        const TOutput wlogs = ql::Lnrat<TOutput, TMass, TScale>(si, msq);
        const TOutput wlogt = ql::Lnrat<TOutput, TMass, TScale>(tabar, msq);
        const TOutput wlogp = ql::Lnrat<TOutput, TMass, TScale>(p4sqbar, msq);
        const TOutput wlogm = ql::Lnrat<TOutput, TMass, TScale>(mu2, msq);
    
        res(i,2) = ql::Constants<TOutput>::_three() / ql::Constants<TOutput>::_two();
        res(i,1) = (ql::Constants<TOutput>::_three() / ql::Constants<TOutput>::_two()) * wlogm - ql::Constants<TOutput>::_two() * wlogt - wlogs + wlogp;
        res(i,0) = ql::Constants<TOutput>::_two() * wlogs * wlogt - wlogp * wlogp - TOutput(ql::Constants<TScale>::template _pi2o12<TOutput, TMass, TScale>() * ql::Constants<TScale>::_five())
                 + (ql::Constants<TOutput>::_three() / ql::Constants<TOutput>::_four()) * wlogm * wlogm + wlogm * (-ql::Constants<TOutput>::_two() * wlogt - wlogs + wlogp)
                 - ql::Constants<TOutput>::_two() * ql::Li2omrat<TOutput, TMass, TScale>(p4sqbar, tabar);
    
        const TOutput d = TOutput(si * tabar);
        for (size_t j = 0; j < 3; j++) {
            res(i,j) /= d;
        }
    }

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B8(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass msq = Y[3][3];
        const TMass tabar = ql::Constants<TMass>::_two() * Y[1][3];
        const TMass si = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass p3sqbar = ql::Constants<TMass>::_two() * Y[2][3];
        const TMass p4sqbar = ql::Constants<TMass>::_two() * Y[0][3];
        const TOutput wlogs = ql::Lnrat<TOutput, TMass, TScale>(si, mu2);
        const TOutput wlogp3 = ql::Lnrat<TOutput, TMass, TScale>(p3sqbar, tabar);
        const TOutput wlogp4 = ql::Lnrat<TOutput, TMass, TScale>(p4sqbar, tabar);

        const TOutput dilog3 = ql::Li2omrat<TOutput, TMass, TScale>(p3sqbar, tabar);
        const TOutput dilog4 = ql::Li2omrat<TOutput, TMass, TScale>(p4sqbar, tabar);
        const TOutput dilog34 = ql::Li2omx2<TOutput, TMass, TScale>(p3sqbar, p4sqbar, si, msq);
        const TOutput ln_si_mu2 = ql::Lnrat<TOutput, TMass, TScale>(si, mu2);

        res(i,2) = ql::Constants<TOutput>::_one();
        res(i,1) = wlogp3 + wlogp4 - wlogs;
        res(i,0) = -ql::Constants<TOutput>::_two() * dilog3 - ql::Constants<TOutput>::_two() * dilog4 - dilog34
                - TOutput(ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>()) + ql::Constants<TOutput>::_half() * (ln_si_mu2 * ln_si_mu2 - ql::kPow<TOutput, TMass, TScale>(ql::Lnrat<TOutput, TMass, TScale>(si, msq), 2))
                + ql::Constants<TOutput>::_two() * ln_si_mu2 * ql::Lnrat<TOutput, TMass, TScale>(tabar, msq)
                - ql::Lnrat<TOutput, TMass, TScale>(p3sqbar, mu2) * ql::Lnrat<TOutput, TMass, TScale>(p3sqbar, msq)
                - ql::Lnrat<TOutput, TMass, TScale>(p4sqbar, mu2) * ql::Lnrat<TOutput, TMass, TScale>(p4sqbar, msq);

        const TOutput d = TOutput(si * tabar);
        for (size_t j = 0; j < 3; j++) {
            res(i,j) /= d;
        }
    }

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B9(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass msq = Y[3][3];
        const TMass mean = ql::kSqrt(TMass(mu2 * msq));
        const TMass tabar = ql::Constants<TMass>::_two() * Y[1][3];
        const TMass si = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass m3sqbar = ql::Constants<TMass>::_two() * Y[2][3];
        const TMass mp2sq = ql::Constants<TMass>::_two() * Y[1][2];
        const TOutput fac = TOutput(si * tabar);

        const TOutput wlogt = ql::Lnrat<TOutput, TMass, TScale>(tabar, mean);
        const TOutput wlog2 = ql::Lnrat<TOutput, TMass, TScale>(si, mp2sq);

        const TOutput dilog1 = ql::Li2omx2<TOutput, TMass, TScale>(m3sqbar, tabar, mp2sq, msq);
        const TOutput dilog2 = ql::Li2omrat<TOutput, TMass, TScale>(si, mp2sq);

        res(i,2) = ql::Constants<TOutput>::_half();
        res(i,1) = -wlogt - wlog2;
        res(i,0) = dilog1 + ql::Constants<TOutput>::_two() * dilog2 + TOutput(ql::Constants<TScale>::template _pi2o12<TOutput, TMass, TScale>()) + (wlogt + wlog2) * (wlogt + wlog2);

        for (size_t j = 0; j < 3; j++) {
            res(i,j) /= fac;
        }
    }

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B10(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass msq = Y[3][3];
        const TMass si = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass tabar = ql::Constants<TMass>::_two() * Y[1][3];
        const TMass m4sqbar = ql::Constants<TMass>::_two() * Y[0][3];
        const TMass m3sqbar = ql::Constants<TMass>::_two() * Y[2][3];
        const TMass mp2sq = ql::Constants<TMass>::_two() * Y[1][2];
        const TMass mean = ql::kSqrt(mu2 * msq);

        const TOutput fac = TOutput(si * tabar - mp2sq * m4sqbar);
        const TOutput wlogsmu = ql::Lnrat<TOutput, TMass, TScale>(si, mu2);
        const TOutput wlogtmu = ql::Lnrat<TOutput, TMass, TScale>(tabar, mu2);
        const TOutput wlog2mu = ql::Lnrat<TOutput, TMass, TScale>(mp2sq, mu2);
        const TOutput wlog4mu = ql::Lnrat<TOutput, TMass, TScale>(m4sqbar, mu2);

        const TOutput dilog1 = ql::Li2omrat<TOutput, TMass, TScale>(mp2sq, si);
        const TOutput dilog2 = ql::Li2omrat<TOutput, TMass, TScale>(tabar, m4sqbar);
        const TOutput dilog3 = ql::Li2omx2<TOutput, TMass, TScale>(mp2sq, m4sqbar, si, tabar);
        const TOutput dilog4 = ql::Li2omx2<TOutput, TMass, TScale>(m3sqbar, tabar, mp2sq, msq);
        const TOutput dilog5 = ql::Li2omx2<TOutput, TMass, TScale>(m3sqbar, m4sqbar, si, msq);

        res(i,2) = ql::Constants<TOutput>::_zero();
        res(i,1) = wlog2mu + wlog4mu - wlogsmu - wlogtmu;
        res(i,0) = dilog4 - dilog5
            - ql::Constants<TOutput>::_two() * dilog1
            + ql::Constants<TOutput>::_two() * dilog2
            + ql::Constants<TOutput>::_two() * dilog3
            + ql::Constants<TOutput>::_two() * res(i,1) * ql::Lnrat<TOutput, TMass, TScale>(mean, tabar);
           
        for (size_t j = 0; j < 3; j++) {
            res(i,j) /= fac;
        }
    }

    /*!
    * This function triggers the topologies with 1 internal mass.
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param xpi array with masses and momenta squared
    * \param mu2 is the square of the scale mu
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B1m(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<TMass, 13>& xpi, 
        const TScale& mu2,
        const int i) {

        int jsort = 0;
        for (int j = 0; j < 4; j++) {
            if (!ql::iszero<TOutput, TMass, TScale>(xpi[j])) {
                jsort = j;
            }
        }

        Kokkos::Array<TMass, 13> xpi_in;
        for (size_t j = 0; j < 13; j++) {
            xpi_in[ql::swap_b1m(j, jsort)] = xpi[j];
        }

        Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Y;
        Y[0][0] = xpi_in[0];
        Y[1][1] = xpi_in[1];
        Y[2][2] = xpi_in[2];
        Y[3][3] = xpi_in[3];
        Y[0][1] = Y[1][0] = ql::Constants<TMass>::_half() * (xpi_in[0] + xpi_in[1] - xpi_in[4]);
        Y[0][2] = Y[2][0] = ql::Constants<TMass>::_half() * (xpi_in[0] + xpi_in[2] - xpi_in[8]);
        Y[0][3] = Y[3][0] = ql::Constants<TMass>::_half() * (xpi_in[0] + xpi_in[3] - xpi_in[7]);
        Y[1][2] = Y[2][1] = ql::Constants<TMass>::_half() * (xpi_in[1] + xpi_in[2] - xpi_in[5]);
        Y[1][3] = Y[3][1] = ql::Constants<TMass>::_half() * (xpi_in[1] + xpi_in[3] - xpi_in[9]);
        Y[2][3] = Y[3][2] = ql::Constants<TMass>::_half() * (xpi_in[2] + xpi_in[3] - xpi_in[6]);

        if (!ql::iszero<TOutput, TMass, TScale>(Y[0][0]) || 
            !ql::iszero<TOutput, TMass, TScale>(Y[1][1]) || 
            !ql::iszero<TOutput, TMass, TScale>(Y[2][2])) {
            Kokkos::printf("Box::B1m - Wrong ordering.");
        }

        const bool zY01 = ql::iszero<TOutput, TMass, TScale>(Y[0][1]);
        const bool zY12 = ql::iszero<TOutput, TMass, TScale>(Y[1][2]);
        const bool zY23 = ql::iszero<TOutput, TMass, TScale>(Y[2][3]);
        const bool zY30 = ql::iszero<TOutput, TMass, TScale>(Y[3][0]);

        if (zY01 && zY12 && zY23 && zY30) 
            ql::B6<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (zY01 && zY12 && zY23) 
            ql::B7<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (zY01 && zY12 && zY30) {
            Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Yalt;
            ql::Ycalc<TOutput, TMass, TScale>(Y, Yalt, 1);
            ql::B7<TOutput, TMass, TScale>(res, Yalt, mu2, i);
        } else if (zY01 && zY12) 
            ql::B8<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (zY01 && zY30) 
            ql::B9<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (zY12 && zY23) {
            Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Yalt;
            ql::Ycalc<TOutput, TMass, TScale>(Y, Yalt, 1);
            ql::B9<TOutput, TMass, TScale>(res, Yalt, mu2, i);
        } else if (zY01)
            ql::B10<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (zY12) {
            Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Yalt;
            ql::Ycalc<TOutput, TMass, TScale>(Y, Yalt, 1);
            ql::B10<TOutput, TMass, TScale>(res, Yalt, mu2, i);
        } else 
            ql::BIN1<TOutput, TMass, TScale>(res, Y, i);
    }           

#ifndef QCDLOOP_BOX_FULL_DISPATCH
    /*!
    * Pruned BO for massive == 1 (1 internal mass).
    * Includes only the B1m dispatch path.
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
        
        if (massive == 1) {
            ql::B1m<TOutput, TMass, TScale>(res, xpi, musq, i);
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
