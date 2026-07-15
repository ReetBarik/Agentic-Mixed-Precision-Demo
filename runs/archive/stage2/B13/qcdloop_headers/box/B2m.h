//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//
// Box integrals with 2 internal masses.
// Contains: BIN2, B11, B12, B13, B14, B15, B2ma, B2mo, B2m dispatcher, pruned BO.

#pragma once

#include "box_common.h"


namespace ql
{

    KOKKOS_INLINE_FUNCTION
    constexpr int swap_b2m(int i, int j) {
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
            {11, 11, 11, 11, 11},
            {12, 12, 12, 12, 12},
        };
        return val[i][j];
    }

    /*!
    * Finite box with 2 non-zero masses. Formulae from \cite Denner:1991qq.
    * \param res output object res[0,1,2] the coefficients in the Laurent series, following the LoopTools implementation \cite Hahn:2006qw.
    * \param Y the modified Cayley matrix.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void BIN2(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        const int i) {

        const TMass m3 = Y[2][2];
        const TMass m4 = Y[3][3];
        const TMass m = ql::kSqrt(m3 * m4);

        const TMass k12 = ql::Constants<TMass>::_two() * Y[1][2] / m3;
        const TMass k13 = ql::Constants<TMass>::_two() * Y[0][2] / m3;
        const TMass k14 = ql::Constants<TMass>::_two() * Y[2][3] / m;
        const TMass k23 = ql::Constants<TMass>::_two() * Y[0][1] / m3;
        const TMass k24 = ql::Constants<TMass>::_two() * Y[1][3] / m;
        const TMass k34 = ql::Constants<TMass>::_two() * Y[0][3] / m;

        const TOutput k12c = TOutput(k12 - ql::Max(ql::kAbs(k12), TMass(ql::Constants<TMass>::_one())) * ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>());
        const TOutput k13c = TOutput(k13 - ql::Max(ql::kAbs(k13), TMass(ql::Constants<TMass>::_one())) * ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>());
        const TOutput k23c = TOutput(k23 - ql::Max(ql::kAbs(k23), TMass(ql::Constants<TMass>::_one())) * ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>());
        const TOutput k24c = TOutput(k24 - ql::Max(ql::kAbs(k24), TMass(ql::Constants<TMass>::_one())) * ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>()) / k12c;
        const TOutput k34c = TOutput(k34 - ql::Max(ql::kAbs(k34), TMass(ql::Constants<TMass>::_one())) * ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>()) / k13c;

        TOutput r14 = ql::Constants<TOutput>::_half() * (TOutput(k14) + TOutput(ql::Sign(ql::Real(k14))) * ql::kSqrt(TOutput((k14 - ql::Constants<TMass>::_two()) * (k14 + ql::Constants<TMass>::_two()))));
        r14 *= (ql::Constants<TOutput>::_one() + ql::Constants<TOutput>::template _ieps50<TOutput, TMass, TScale>() * TOutput(ql::Sign(ql::Real(ql::Constants<TOutput>::_one() / r14 - r14))));

        const TMass a = k34 * k24 - k23;
        const TMass b = k13 * k24 + k12 * k34 - k14 * k23;
        const TOutput c = TOutput(k13 * k12) - TOutput(k23) * (ql::Constants<TOutput>::_one() - ql::Constants<TOutput>::template _ieps50<TOutput, TMass, TScale>());
        const TOutput disc = ql::kSqrt(TOutput(b * b) - TOutput(ql::Constants<TMass>::_four() * a) * c);

        Kokkos::Array<TOutput, 2> x4;
        x4[0] = ql::Constants<TOutput>::_half() * (TOutput(b) - disc) / TOutput(a);
        x4[1] = ql::Constants<TOutput>::_half() * (TOutput(b) + disc) / TOutput(a);
        if (ql::kAbs(x4[0]) > ql::kAbs(x4[1]))
            x4[1] = c / (TOutput(a) * x4[0]);
        else
            x4[0] = c / (TOutput(a) * x4[1]); 

        const Kokkos::Array<TScale, 2> imzero = {ql::Constants<TScale>::_zero(), ql::Constants<TScale>::_zero()};
        res(i, 0) = (
            ql::xspence<TOutput, TMass, TScale>(x4, imzero, r14, ql::Constants<TScale>::_zero()) +
            ql::xspence<TOutput, TMass, TScale>(x4, imzero, ql::Constants<TOutput>::_one() / r14, ql::Constants<TScale>::_zero()) -
            ql::xspence<TOutput, TMass, TScale>(x4, imzero, k34c, ql::Constants<TScale>::_zero()) -
            ql::xspence<TOutput, TMass, TScale>(x4, imzero, k24c, ql::Constants<TScale>::_zero()) +
            (ql::kLog(x4[1]) - ql::kLog(x4[0])) * (ql::kLog(k12c) + ql::kLog(k13c) - ql::kLog(k23c))
        ) / (m3 * m * disc);

        res(i, 1) = res(i, 2) = ql::Constants<TOutput>::_zero();
    }

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B11(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass m3sq = Y[2][2];
        const TMass m4sq = Y[3][3];
        const TMass sibar = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass tabar = ql::Constants<TMass>::_two() * Y[1][3];
        const TMass p3sq = -(ql::Constants<TMass>::_two() * Y[2][3] - Y[2][2] - Y[3][3]);
        const TMass m3mu = ql::kSqrt(m3sq * mu2);
        const TMass m4mu = ql::kSqrt(m4sq * mu2);

        const TOutput wlogt = ql::Lnrat<TOutput, TMass, TScale>(tabar, m4mu);
        const TOutput wlogs = ql::Lnrat<TOutput, TMass, TScale>(sibar, m3mu);

        TOutput root;
        TMass x43p, x43pm1, x43m, x43mm1;
        if (ql::iszero<TOutput, TMass, TScale>(p3sq)) {
            root = ql::Constants<TOutput>::_one();
            x43p = -ql::Constants<TMass>::_one();
            x43pm1 = -ql::Constants<TMass>::_one();
            x43m = m3sq;
            x43mm1 = m4sq;
        } else {
            root = ql::kSqrt(TOutput(ql::kPow<TOutput, TMass, TScale>(p3sq + m3sq - m4sq, 2) - ql::Constants<TMass>::_four() * m3sq * p3sq));
            const TOutput ga43p   = TOutput(+p3sq + m3sq - m4sq) + root;
            const TOutput ga43pm1 = TOutput(-p3sq + m3sq - m4sq) + root;
            const TOutput ga43m   = TOutput(+p3sq + m3sq - m4sq) - root;
            const TOutput ga43mm1 = TOutput(-p3sq + m3sq - m4sq) - root;

            x43p = -ql::Real(ga43p);
            x43pm1 = -ql::Real(ga43pm1);
            x43m = ql::Real(ga43m);
            x43mm1 = ql::Real(ga43mm1);
        }

        // deal with real roots
        TOutput ln43p, ln43m, rat2p, rat2m;
        TScale ieps2;
        if (ql::iszero<TOutput, TMass, TScale>(ql::Imag(root))) {
            ln43p = ql::Lnrat<TOutput, TMass, TScale>(x43p, x43pm1);
            ln43m = ql::Lnrat<TOutput, TMass, TScale>(x43m, x43mm1);
        } else {
            ql::ratgam<TOutput, TMass, TScale>(rat2p, rat2m, ieps2, p3sq, m4sq, m3sq);
            ln43p = ql::cLn<TOutput, TMass, TScale>(rat2p, ieps2);
            ln43m = ql::cLn<TOutput, TMass, TScale>(rat2m, ieps2);
        }

        TOutput intbit;
        if (ql::iszero<TOutput, TMass, TScale>(p3sq)) {
            intbit = -ql::Constants<TOutput>::_half() * TOutput(ql::kPow<TOutput, TMass, TScale>(ql::kLog(m3sq / m4sq), 2));
        } else {
            intbit = -ql::Constants<TOutput>::_half() * (ql::kPow<TOutput, TMass, TScale>(ln43p, 2) + ql::kPow<TOutput, TMass, TScale>(ln43m, 2));
        }

        res(i,2) = ql::Constants<TOutput>::_one();
        res(i,1) = -wlogt - wlogs;
        res(i,0) = intbit
            + ql::Constants<TOutput>::_two() * wlogt * wlogs - TOutput(ql::Constants<TScale>::_half() * ql::Constants<TScale>::_pi2())
            + TOutput(ql::kPow<TOutput, TMass, TScale>(ql::kLog(m3sq / m4sq), 2) / ql::Constants<TScale>::_four());

        const TOutput d = TOutput(sibar * tabar);
        for (size_t j = 0; j < 3; j++) {
            res(i,j) /= d;
        }
    }

    /*!
    * Implementation of the formulae from Ellis et al. \cite Ellis:2007qk.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B12(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass m3sq = Y[2][2];
        const TMass m4sq = Y[3][3];
        const TMass sibar = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass tabar = ql::Constants<TMass>::_two() * Y[1][3];
        const TMass m4sqbar = ql::Constants<TMass>::_two() * Y[0][3];
        const TMass p3sq = -(ql::Constants<TMass>::_two() * Y[2][3] - Y[2][2] - Y[3][3]);

        const TMass mean = ql::kSqrt(mu2 * ql::Real(m3sq));
        const TOutput fac = TOutput(sibar * tabar);

        const TOutput wlogsmu = ql::Lnrat<TOutput, TMass, TScale>(sibar, mean);
        const TOutput wlogtmu = ql::Lnrat<TOutput, TMass, TScale>(tabar, mean);
        const TOutput wlog4mu = ql::Lnrat<TOutput, TMass, TScale>(m4sqbar, mean);
        const TOutput wlog = wlogsmu + wlogtmu - wlog4mu;

        TOutput root;
        TMass x43p, x43pm1, x43m, x43mm1;
        if (ql::iszero<TOutput, TMass, TScale>(p3sq)) {
            root = ql::Constants<TOutput>::_one();
            x43p = -ql::Constants<TMass>::_one();
            x43pm1 = -ql::Constants<TMass>::_one();
            x43m = m3sq;
            x43mm1 = m4sq;
        } else {
            root = ql::kSqrt(TOutput(ql::kPow<TOutput, TMass, TScale>(p3sq + m3sq - m4sq, 2) - ql::Constants<TMass>::_four() * m3sq * p3sq));
            const TOutput ga43p   = TOutput(+p3sq + m3sq - m4sq) + root;
            const TOutput ga43pm1 = TOutput(-p3sq + m3sq - m4sq) + root;
            const TOutput ga43m   = TOutput(+p3sq + m3sq - m4sq) - root;
            const TOutput ga43mm1 = TOutput(-p3sq + m3sq - m4sq) - root;

            x43p = -ql::Real(ga43p);
            x43pm1 = -ql::Real(ga43pm1);
            x43m = ql::Real(ga43m);
            x43mm1 = ql::Real(ga43mm1);
        }

        const TOutput dilog1 = ql::Li2omrat<TOutput, TMass, TScale>(m4sqbar, tabar);
        TOutput dilog2, dilog3;

        // deal with real roots
        TScale ieps2, ieps1;
        TMass rat1;
        TOutput ln43p, ln43m, rat2p, rat2m, zrat1;
        if (ql::iszero<TOutput, TMass, TScale>(ql::Imag(root))) {
            ln43p = ql::Lnrat<TOutput, TMass, TScale>(x43p, x43pm1);
            ln43m = ql::Lnrat<TOutput, TMass, TScale>(x43m, x43mm1);
            dilog2 = ql::Li2omx2<TOutput, TMass, TScale>(m4sqbar, x43p, sibar, x43pm1);
            dilog3 = ql::Li2omx2<TOutput, TMass, TScale>(m4sqbar, x43m, sibar, x43mm1);
        } else {
            ql::ratreal<TOutput, TMass, TScale>(m4sqbar, sibar, rat1, ieps1);
            ql::ratgam<TOutput, TMass, TScale>(rat2p, rat2m, ieps2, p3sq, m4sq, m3sq);
            zrat1 = TOutput(rat1);
            ln43p = ql::cLn<TOutput, TMass, TScale>(rat2p, ieps2);
            ln43m = ql::cLn<TOutput, TMass, TScale>(rat2m, ieps2);

            dilog2 = ql::spencer<TOutput, TMass, TScale>(zrat1, rat2p, ieps1, ieps2);
            dilog3 = ql::spencer<TOutput, TMass, TScale>(zrat1, rat2m, ieps1, ieps2);
        }

        res(i,2) = ql::Constants<TOutput>::_half();
        res(i,1) = -wlog;
        res(i,0) = -TOutput(ql::Constants<TScale>::template _pi2o12<TOutput, TMass, TScale>())
            + ql::Constants<TOutput>::_two() * wlogsmu * wlogtmu - wlog4mu * wlog4mu
            + (wlog4mu - wlogsmu) * ql::kLog(m4sq / m3sq) - ql::Constants<TOutput>::_half() * (ln43p * ln43p + ln43m * ln43m)
            - ql::Constants<TOutput>::_two() * dilog1 - dilog2 - dilog3;

        for (size_t j = 0; j < 3; j++)
            res(i,j) /= fac;
    }


    /*!
    * Implementation of the formulae from Ellis et al. \cite Ellis:2007qk.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B13(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {
            
        const TMass m3sq = Y[2][2];
        const TMass m4sq = Y[3][3];
        const TMass sibar = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass tabar = ql::Constants<TMass>::_two() * Y[1][3];
        const TMass m4sqbar = ql::Constants<TMass>::_two() * Y[0][3];
        const TMass m3sqbar = ql::Constants<TMass>::_two() * Y[1][2];
        const TMass p3sq = -(ql::Constants<TMass>::_two() * Y[2][3] - Y[2][2] - Y[3][3]);

        const TOutput fac = TOutput(sibar * tabar - m3sqbar * m4sqbar);
        const TOutput wlogsmu = ql::Lnrat<TOutput, TMass, TScale>(sibar, TMass(mu2));
        const TOutput wlogtmu = ql::Lnrat<TOutput, TMass, TScale>(tabar, TMass(mu2));
        const TOutput wlog3mu = ql::Lnrat<TOutput, TMass, TScale>(m3sqbar, TMass(mu2));
        const TOutput wlog4mu = ql::Lnrat<TOutput, TMass, TScale>(m4sqbar, TMass(mu2));
        const TOutput dilog1 = ql::Li2omrat<TOutput, TMass, TScale>(m3sqbar, sibar);
        const TOutput dilog4 = ql::Li2omrat<TOutput, TMass, TScale>(m4sqbar, tabar);
        const TOutput dilog7 = ql::Li2omx2<TOutput, TMass, TScale>(m3sqbar, m4sqbar, sibar, tabar);

        TOutput root, ga34p, ga34pm1, ga34m, ga34mm1;
        TOutput ga43p, ga43pm1, ga43m, ga43mm1;
        TMass x34p, x34pm1, x34m, x34mm1;
        TMass x43p, x43pm1, x43m, x43mm1;
        if (ql::iszero<TOutput, TMass, TScale>(p3sq)) {
            root = ql::Constants<TOutput>::_one();
            x34p = -ql::Constants<TMass>::_one();
            x34pm1 = -ql::Constants<TMass>::_one();
            x34m = m4sq;
            x34mm1 = m3sq;

            x43p = m3sq;
            x43pm1 = m4sq;
            x43m = -ql::Constants<TMass>::_one();
            x43mm1 = -ql::Constants<TMass>::_one();
        } else {
            root = ql::kSqrt(TOutput(ql::kPow<TOutput, TMass, TScale>(p3sq-m3sq+m4sq, 2) - ql::Constants<TMass>::_four() * m4sq * p3sq));
            ga34p   = TOutput(+p3sq + m4sq - m3sq) + root;
            ga34pm1 = TOutput(-p3sq + m4sq - m3sq) + root;
            ga34m   = TOutput(+p3sq + m4sq - m3sq) - root;
            ga34mm1 = TOutput(-p3sq + m4sq - m3sq) - root;

            ga43p   = TOutput(+p3sq + m3sq - m4sq) + root;
            ga43pm1 = TOutput(-p3sq + m3sq - m4sq) + root;
            ga43m   = TOutput(+p3sq + m3sq - m4sq) - root;
            ga43mm1 = TOutput(-p3sq + m3sq - m4sq) - root;

            x34p = -ql::Real(ga34p);
            x34pm1 = -ql::Real(ga34pm1);
            x34m = ql::Real(ga34m);
            x34mm1 = ql::Real(ga34mm1);

            x43p = -ql::Real(ga43p);
            x43pm1 = -ql::Real(ga43pm1);
            x43m = ql::Real(ga43m);
            x43mm1 = ql::Real(ga43mm1);
        }

        TMass rat3t, rat4s;
        TOutput ln43p, ln43m, dilog2, dilog3, dilog5, dilog6;
        TOutput zrat3t, zrat4s, rat34p, rat34m, rat43p, rat43m;
        TScale ieps3t, ieps4s, ieps34, ieps43;

        if (ql::iszero<TOutput, TMass, TScale>(ql::Imag(root))) {
            ln43p = ql::Lnrat<TOutput, TMass, TScale>(x43p, x43pm1);
            ln43m = ql::Lnrat<TOutput, TMass, TScale>(x43m, x43mm1);

            dilog2 = ql::Li2omx2<TOutput, TMass, TScale>(m3sqbar, x34p, tabar, x34pm1);
            dilog3 = ql::Li2omx2<TOutput, TMass, TScale>(m3sqbar, x34m, tabar, x34mm1);
            dilog5 = ql::Li2omx2<TOutput, TMass, TScale>(m4sqbar, x43p, sibar, x43pm1);
            dilog6 = ql::Li2omx2<TOutput, TMass, TScale>(m4sqbar, x43m, sibar, x43mm1);
        } else {
            ql::ratreal<TOutput, TMass, TScale>(m3sqbar, tabar, rat3t, ieps3t);
            ql::ratreal<TOutput, TMass, TScale>(m4sqbar, sibar, rat4s, ieps4s);

            ql::ratgam<TOutput, TMass, TScale>(rat34p, rat34m, ieps34, p3sq, m3sq, m4sq);
            ql::ratgam<TOutput, TMass, TScale>(rat43p, rat43m, ieps43, p3sq, m4sq, m3sq);

            zrat3t = TOutput(rat3t);
            zrat4s = TOutput(rat4s);

            dilog2 = ql::spencer<TOutput, TMass, TScale>(zrat3t, rat34p, ieps3t, ieps34);
            dilog3 = ql::spencer<TOutput, TMass, TScale>(zrat3t, rat34m, ieps3t, ieps34);
            dilog5 = ql::spencer<TOutput, TMass, TScale>(zrat4s, rat43p, ieps4s, ieps43);
            dilog6 = ql::spencer<TOutput, TMass, TScale>(zrat4s, rat43m, ieps4s, ieps43);

            ln43p = ql::cLn<TOutput, TMass, TScale>(rat43p, ql::Constants<TScale>::_zero());
            ln43m = ql::cLn<TOutput, TMass, TScale>(rat43m, ql::Constants<TScale>::_zero());
        }


        res(i,2) = ql::Constants<TOutput>::_zero();
        res(i,1) = wlog3mu + wlog4mu - wlogsmu - wlogtmu;
        res(i,0) = -ql::Constants<TOutput>::_two() * dilog1 - dilog2 - dilog3
                 -ql::Constants<TOutput>::_two() * dilog4 - dilog5 - dilog6
                 +ql::Constants<TOutput>::_two() * dilog7
                 +ql::Constants<TOutput>::_two() * wlogsmu * wlogtmu - wlog3mu * wlog3mu - wlog4mu * wlog4mu
                 +(wlog3mu - wlogtmu) * ql::kLog(m3sq / mu2)
                 +(wlog4mu - wlogsmu) * ql::kLog(m4sq / mu2)
                 -ql::Constants<TOutput>::_half() * (ln43p * ln43p + ln43m * ln43m);

        for (size_t j = 0; j < 3; j++)
            res(i,j) /= fac;
    }

    /*!
    * Implementation of the formulae from Beenakker et al. \cite Beenakker:1988jr.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B14(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass m2sq = Y[1][1];
        const TMass m4sq = Y[3][3];
        const TMass ta = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass si = ql::Constants<TMass>::_two() * Y[1][3] - Y[1][1] - Y[3][3];
        const TMass m2 = ql::kSqrt(m2sq);
        const TMass m4 = ql::kSqrt(m4sq);

        const TOutput wlogtmu = ql::Lnrat<TOutput, TMass, TScale>(mu2, ta);

        TScale ieps = ql::Constants<TScale>::_zero();
        Kokkos::Array<TOutput, 3> cxs;
        ql::kfn<TOutput, TMass, TScale>(cxs, ieps, -si, m2, m4);
        const TScale xs = ql::Real(cxs[0]);
        const TScale imxs = ql::Imag(cxs[0]);

        TOutput fac;
        if ( ql::iszero<TOutput, TMass, TScale>(xs - ql::Constants<TScale>::_one()) && ql::iszero<TOutput, TMass, TScale>(imxs))
            fac = TOutput(-xs / (m2 * m4 * ta));
        else {
            const TOutput xlog = ql::cLn<TOutput, TMass, TScale>(cxs[0], ieps);
            fac = TOutput(ql::Constants<TMass>::_two() / (m2 * m4 * ta)) * cxs[0] / (cxs[1] * cxs[2]) * xlog;
        }
        res(i,2) = ql::Constants<TOutput>::_zero();
        res(i,1) = fac;
        res(i,0) = fac * wlogtmu;
    }

    /*!
    * Implementation of the formulae from Beenakker et al. \cite Beenakker:1988jr.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B15(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass m2sq = Y[1][1];
        const TMass m4sq = Y[3][3];
        const TMass m2sqbar = ql::Constants<TMass>::_two() * Y[1][2];
        const TMass m4sqbar = ql::Constants<TMass>::_two() * Y[2][3];
        const TMass si = ql::Constants<TMass>::_two() * Y[1][3] - Y[1][1] - Y[3][3];
        const TMass ta = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass m2 = ql::kSqrt(m2sq);
        const TMass m4 = ql::kSqrt(m4sq);

        TScale ieps = ql::Constants<TScale>::_zero();
        Kokkos::Array<TOutput, 3> cxs;
        ql::kfn<TOutput, TMass, TScale>(cxs, ieps, -si, m2, m4);
        const TOutput xs = cxs[0];

        TOutput fac;
        if (ql::iszero<TOutput, TMass, TScale>(m2sqbar) && !ql::iszero<TOutput, TMass, TScale>(m4sqbar)) {
            TMass yi;
            TScale iepyi;
            ql::ratreal<TOutput, TMass, TScale>(m4 * m2sqbar, m2 * m4sqbar, yi, iepyi);
            TOutput cyi = TOutput(yi);
            fac = xs / (ql::Constants<TOutput>::_one() - xs * xs) / TOutput(-m2 * m4 * ta);
            const TOutput xlog = ql::cLn<TOutput, TMass, TScale>(xs, ieps);
            res(i,2) = ql::Constants<TOutput>::_zero();
            res(i,1) = -xlog;
            res(i,0) = xlog * (-xlog - TOutput(ql::kLog(mu2 / m4sq))
                -ql::Constants<TOutput>::_two() * ql::Lnrat<TOutput, TMass, TScale>(m4sqbar, ta))
                -ql::cLi2omx2<TOutput, TMass, TScale>(xs, xs, ieps, ieps)
                +ql::cLi2omx2<TOutput, TMass, TScale>(xs, cyi, ieps, iepyi)
                -ql::cLi2omx2<TOutput, TMass, TScale>(ql::Constants<TOutput>::_one() / xs, cyi, -ieps, iepyi);

            for (size_t j = 0; j < 3; j++)
                res(i,j) *= TOutput(fac);

            return;
        } else if (ql::iszero<TOutput, TMass, TScale>(m4sqbar) && !ql::iszero<TOutput, TMass, TScale>(m2sqbar)) {
            TMass yy;
            TScale iepsyy;
            ql::ratreal<TOutput, TMass, TScale>(m2 * m4sqbar, m4 * m2sqbar, yy, iepsyy);
            TOutput cyy = TOutput(yy);
            fac = xs / (ql::Constants<TOutput>::_one() - xs * xs) / TOutput(-m2 * m4 * ta);
            const TOutput xlog = ql::cLn<TOutput, TMass, TScale>(xs, ieps);
            res(i,2) = ql::Constants<TOutput>::_zero();
            res(i,1) = -xlog;
            res(i,0) = xlog * (-xlog - TOutput(ql::kLog(mu2 / m2sq))
                      -ql::Constants<TOutput>::_two() * ql::Lnrat<TOutput, TMass, TScale>(m2sqbar, ta))
                -ql::cLi2omx2<TOutput, TMass, TScale>(xs, xs, ieps, ieps)
                +ql::cLi2omx2<TOutput, TMass, TScale>(xs, cyy, ieps, iepsyy)
                -ql::cLi2omx2<TOutput, TMass, TScale>(ql::Constants<TOutput>::_one() / xs, cyy, -ieps, iepsyy);

            for (size_t j = 0; j < 3; j++)
                res(i,j) *= TOutput(fac);

            return;
        } else if (ql::iszero<TOutput, TMass, TScale>(m4sqbar) && ql::iszero<TOutput, TMass, TScale>(m2sqbar)) {
            Kokkos::printf("Box::B15 wrong kinematics, this is really B14.");
        }
        TMass yy;
        TScale iepsyy;
        ql::ratreal<TOutput, TMass, TScale>(m2*m4sqbar, m4*m2sqbar, yy, iepsyy);
        const TScale rexs = ql::Real(xs);
        const TScale imxs = ql::Imag(xs);

        if (ql::iszero<TOutput, TMass, TScale>(rexs-ql::Constants<TScale>::_one()) && ql::iszero<TOutput, TMass, TScale>(imxs)) {
            fac = TOutput(ql::Constants<TOutput>::_half() / (m2 * m4 * ta));
            res(i,2) = ql::Constants<TOutput>::_zero();
            res(i,1) = ql::Constants<TOutput>::_one();
            res(i,0) = TOutput(ql::kLog(mu2 / (m2 * m4)))
                      -ql::Lnrat<TOutput, TMass, TScale>(m2sqbar, ta) - ql::Lnrat<TOutput, TMass, TScale>(m4sqbar, ta) - ql::Constants<TOutput>::_two()
                     -TOutput((ql::Constants<TMass>::_one() + yy) / (ql::Constants<TMass>::_one() - yy)) * ql::Lnrat<TOutput, TMass, TScale>(m2 * m4sqbar, m4 * m2sqbar);
        } else {
            fac = xs / (ql::Constants<TOutput>::_one() - xs * xs) / TOutput(-m2 * m4 * ta);
            const TOutput xlog = ql::cLn<TOutput, TMass, TScale>(xs, ieps);
            res(i,2) = ql::Constants<TOutput>::_zero();
            res(i,1) = -xlog;
            res(i,0) = xlog * (-ql::Constants<TOutput>::_half() * xlog - TOutput(ql::kLog(mu2 / (m2 * m4)))
                    -ql::Lnrat<TOutput, TMass, TScale>(m2sqbar, ta) - ql::Lnrat<TOutput, TMass, TScale>(m4sqbar, ta))
                    -ql::cLi2omx2<TOutput, TMass, TScale>(xs, xs, ieps, ieps)
                    +ql::Constants<TOutput>::_half() * ql::kPow<TOutput, TMass, TScale>(ql::Lnrat<TOutput, TMass, TScale>(m2 * m4sqbar, m4 * m2sqbar), 2)
                    +ql::cLi2omx2<TOutput, TMass, TScale>(xs, TOutput(yy), ieps, iepsyy)
                    +ql::cLi2omx2<TOutput, TMass, TScale>(xs, TOutput(ql::Constants<TMass>::_one() / yy), ieps, -iepsyy);
        }

        for (size_t j = 0; j < 3; j++)
            res(i,j) *= TOutput(fac);
    }

    /*!
    * This function trigger the topologies with 2-offshell external lines (adjacent).
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param xpi array with masses and momenta squared
    * \param mu2 is the square of the scale mu
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B2ma(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<TMass, 13>& xpi, 
        const TScale& mu2,
        const int i) {

        Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Y;
        Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Yalt;

        Y[0][0] = xpi[0];
        Y[1][1] = xpi[1];
        Y[2][2] = xpi[2];
        Y[3][3] = xpi[3];
        Y[0][1] = Y[1][0] = ql::Constants<TMass>::_half() * (xpi[0] + xpi[1] - xpi[4]);
        Y[0][2] = Y[2][0] = ql::Constants<TMass>::_half() * (xpi[0] + xpi[2] - xpi[8]);
        Y[0][3] = Y[3][0] = ql::Constants<TMass>::_half() * (xpi[0] + xpi[3] - xpi[7]);
        Y[1][2] = Y[2][1] = ql::Constants<TMass>::_half() * (xpi[1] + xpi[2] - xpi[5]);
        Y[1][3] = Y[3][1] = ql::Constants<TMass>::_half() * (xpi[1] + xpi[3] - xpi[9]);
        Y[2][3] = Y[3][2] = ql::Constants<TMass>::_half() * (xpi[2] + xpi[3] - xpi[6]);

        ql::Ycalc<TOutput, TMass, TScale>(Y, Yalt, 2, ql::iszero<TOutput, TMass, TScale>(xpi[2]));

        const bool zY01 = ql::iszero<TOutput, TMass, TScale>(Y[0][1]);
        const bool zY12 = ql::iszero<TOutput, TMass, TScale>(Y[1][2]);
        const bool zY03 = ql::iszero<TOutput, TMass, TScale>(Y[0][3]);

        if (zY01 && zY12 && zY03) 
            ql::B11<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (zY01 && zY12 && !zY03) 
            ql::B12<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (ql::iszero<TOutput, TMass, TScale>(Yalt[0][1]) && ql::iszero<TOutput, TMass, TScale>(Yalt[1][2]) && !ql::iszero<TOutput, TMass, TScale>(Yalt[0][3]))
            ql::B12<TOutput, TMass, TScale>(res, Yalt, mu2, i);
        else if (zY01 && !zY12 && !zY03)
            ql::B13<TOutput, TMass, TScale>(res, Y, mu2, i);
        else
            ql::BIN2<TOutput, TMass, TScale>(res, Y, i);
    }

    /*!
    * This function trigger the topologies with 2-offshell external lines (opposite).
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param xpi array with masses and momenta squared
    * \param mu2 is the square of the scale mu
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B2mo(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<TMass, 13>& xpi, 
        const TScale& mu2,
        const int i) {

        Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Y;

        Y[0][0] = xpi[0];
        Y[1][1] = xpi[1];
        Y[2][2] = xpi[2];
        Y[3][3] = xpi[3];
        Y[0][1] = Y[1][0] = ql::Constants<TMass>::_half() * (xpi[0] + xpi[1] - xpi[4]);
        Y[0][2] = Y[2][0] = ql::Constants<TMass>::_half() * (xpi[0] + xpi[2] - xpi[8]);
        Y[0][3] = Y[3][0] = ql::Constants<TMass>::_half() * (xpi[0] + xpi[3] - xpi[7]);
        Y[1][2] = Y[2][1] = ql::Constants<TMass>::_half() * (xpi[1] + xpi[2] - xpi[5]);
        Y[1][3] = Y[3][1] = ql::Constants<TMass>::_half() * (xpi[1] + xpi[3] - xpi[9]);
        Y[2][3] = Y[3][2] = ql::Constants<TMass>::_half() * (xpi[2] + xpi[3] - xpi[6]);

        const bool zY00 = ql::iszero<TOutput, TMass, TScale>(Y[0][0]);
        const bool zY22 = ql::iszero<TOutput, TMass, TScale>(Y[2][2]);
        const bool zY01 = ql::iszero<TOutput, TMass, TScale>(Y[0][1]);
        const bool zY03 = ql::iszero<TOutput, TMass, TScale>(Y[0][3]);
        const bool zY12 = ql::iszero<TOutput, TMass, TScale>(Y[1][2]);
        const bool zY23 = ql::iszero<TOutput, TMass, TScale>(Y[1][2]);

        if (zY00 && zY22 && zY01 && zY03 && zY12 && zY23) 
            ql::B14<TOutput, TMass, TScale>(res, Y, mu2, i); 
        else if (zY00 && zY22 && zY01 && zY03) 
            ql::B15<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (zY00 && zY22 && zY12 && zY23) {
            Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Yalt;
            ql::Ycalc<TOutput, TMass, TScale>(Y, Yalt, 2, ql::iszero<TOutput, TMass, TScale>(xpi[2]));
            ql::B15<TOutput, TMass, TScale>(res, Yalt, mu2, i);
        } else {
            Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Yadj;
            const int swap23[4] = {0, 2, 1, 3};
            for (int j = 0; j < 4; j++) 
                for (int k = 0; k < 4; k++) 
                    Yadj[j][k] = Y[swap23[j]][swap23[k]];

            ql::BIN2<TOutput, TMass, TScale>(res, Yadj, i);
        }
    }

    /*!
    * This function trigger the topologies with 2 internal masses.
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param xpi array with masses and momenta squared
    * \param mu2 is the square of the scale mu
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B2m(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<TMass, 13>& xpi, 
        const TScale& mu2,
        const int i) {

        int jsort1 = -1, jsort2 = -1;
        for (int j = 0; j < 4; j++) {
            if (!ql::iszero<TOutput, TMass, TScale>(xpi[j])) {
                if (jsort1 == -1) {
                    jsort1 = j;
                } else {
                    jsort2 = j;
                }
            }
        }

        int jdiff = jsort2 - jsort1;
        Kokkos::Array<TMass, 13> xpiout;

        if (jdiff == 1 || jdiff == 2) {
            for (size_t j = 0; j < 13; j++) {
                xpiout[ql::swap_b2m(j, jsort2)] = xpi[j];
            }
        } else if (jdiff == 3) {
            for (size_t j = 0; j < 13; j++) {
                xpiout[ql::swap_b2m(j, 0)] = xpi[j];
            }
        }

        if (jdiff == 2) {
            ql::B2mo<TOutput, TMass, TScale>(res, xpiout, mu2, i);
        } else {
            ql::B2ma<TOutput, TMass, TScale>(res, xpiout, mu2, i);
        }
    }

#ifndef QCDLOOP_BOX_FULL_DISPATCH
    /*!
    * Pruned BO for massive == 2 (2 internal masses).
    * Includes only the B2m dispatch path.
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
        
        if (massive == 2) {
            ql::B2m<TOutput, TMass, TScale>(res, xpi, musq, i);
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
