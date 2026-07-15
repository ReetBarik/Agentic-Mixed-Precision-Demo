//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//
// Box integrals with 0 internal masses.
// Contains: BIN0, B1, B2, B3, B4, B5, B0m dispatcher, pruned BO.

#pragma once

#include "box_common.h"


namespace ql
{

    KOKKOS_INLINE_FUNCTION
    constexpr int swap_b0m(int i, int j) {
        constexpr int val[13][4] = {
            {3, 2, 1, 0},
            {0, 3, 2, 1},
            {1, 0, 3, 2},
            {2, 1, 0, 3},
            {7, 6, 5, 4},
            {4, 7, 6, 5},
            {5, 4, 7, 6},
            {6, 5, 4, 7},
            {9, 8, 9, 8},
            {8, 9, 8, 9},
            {10, 10, 10, 10},
            {12, 11, 12, 11},
            {11, 12, 11, 12},
        };
        return val[i][j];
    }

    KOKKOS_INLINE_FUNCTION
    constexpr int jsort_b0m(int i) {
        constexpr int val[4] = {3,0,1,2};
        return val[i];
    }

    /*!
    * Finite box with zero masses. Formulae from \cite Denner:1991qq.
    * \param res output object res[0,1,2] the coefficients in the Laurent series, following the LoopTools implementation \cite Hahn:2006qw.
    * \param Y the modified Cayley matrix.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void BIN0(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        const int i) {

        const TMass m2 = ql::kAbs(Y[1][3]);
        const TMass k12 = ql::Constants<TMass>::_two() * Y[0][1] / m2;
        const TMass k13 = ql::Constants<TMass>::_two() * Y[0][2] / m2;
        const TMass k14 = ql::Constants<TMass>::_two() * Y[0][3] / m2;
        const TMass k23 = ql::Constants<TMass>::_two() * Y[1][2] / m2;
        const TMass k24 = ql::Constants<TMass>::_two() * Y[1][3] / m2;
        const TMass k34 = ql::Constants<TMass>::_two() * Y[2][3] / m2;

        const TOutput k12c = TOutput(k12 - ql::Max(ql::kAbs(k12), TMass(ql::Constants<TMass>::_one())) * ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>());
        const TOutput k13c = TOutput(k13 - ql::Max(ql::kAbs(k13), TMass(ql::Constants<TMass>::_one())) * ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>());
        const TOutput k14c = TOutput(k14 - ql::Max(ql::kAbs(k14), TMass(ql::Constants<TMass>::_one())) * ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>());
        const TOutput k23c = TOutput(k23 - ql::Max(ql::kAbs(k23), TMass(ql::Constants<TMass>::_one())) * ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>());
        const TOutput k24c = TOutput(k24 - ql::Max(ql::kAbs(k24), TMass(ql::Constants<TMass>::_one())) * ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>()) / k12c;
        const TOutput k34c = TOutput(k34 - ql::Max(ql::kAbs(k34), TMass(ql::Constants<TMass>::_one())) * ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>()) / k13c;

        const TMass a = k34 * k24;
        const TMass b = k13 * k24 + k12 * k34 - k14 * k23;
        const TOutput c = TOutput(k13 * k12) + TOutput(k23) * ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>();
        const TOutput disc = ql::kSqrt(TOutput(b * b) - TOutput(ql::Constants<TMass>::_four() * a) * c);

        Kokkos::Array<TOutput, 2> x4;
        x4[0] = ql::Constants<TOutput>::_half() * (TOutput(b) - disc) / TOutput(a);
        x4[1] = ql::Constants<TOutput>::_half() * (TOutput(b) + disc) / TOutput(a);
        if (ql::kAbs(x4[0]) > ql::kAbs(x4[1]))
            x4[1] = c / (TOutput(a) * x4[0]);
        else
            x4[0] = c / (TOutput(a) * x4[1]);

        const Kokkos::Array<TScale, 2> imzero = {ql::Constants<TScale>::_zero(), ql::Constants<TScale>::_zero()};
        const TOutput log_x40 = ql::kLog(x4[0]);
        const TOutput log_x41 = ql::kLog(x4[1]);

        res(i, 0) = (
            (log_x41 - log_x40) * (-ql::Constants<TOutput>::_half() * (log_x41 + log_x40) +
            ql::kLog(k12c) + ql::kLog(k13c) - ql::kLog(k23c) - ql::kLog(k14c)) -
            ql::xspence<TOutput, TMass, TScale>(x4, imzero, k34c, ql::Constants<TScale>::_zero()) -
            ql::xspence<TOutput, TMass, TScale>(x4, imzero, k24c, ql::Constants<TScale>::_zero())
            ) / (m2 * m2 * disc);

        res(i, 1) = res(i, 2) = ql::Constants<TOutput>::_zero();
    }

    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(0,0,0,0;s_{12},s_{23};0,0,0,0) = \frac{\mu^{2\epsilon}}{s_{12}s_{23}}\left[ \frac{2}{\epsilon^2}\left( (-s_{12}-i\epsilon)^{-\epsilon} + (-s_{23}-i \epsilon)^{-\epsilon} \right) - \ln^2 \left( \frac{-s_{12}-i\epsilon}{-s_{23}-i\epsilon} \right) - \pi^2 \right] + \mathcal{O}(\epsilon)
    * \f]
    * Implementation of the formulae from Bern et al. \cite Bern:1993kr.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B1(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass si = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass ta = ql::Constants<TMass>::_two() * Y[1][3];
        const TOutput fac = ql::Constants<TOutput>::_one() / (si * ta);

        const TOutput lnrat_tamu2 = ql::Lnrat<TOutput, TMass, TScale>(ta, mu2);
        const TOutput lnrat_simu2 = ql::Lnrat<TOutput, TMass, TScale>(si, mu2);
        const TOutput lnrat_tasi = ql::Lnrat<TOutput, TMass, TScale>(ta, si);

        res(i,2) = fac * ql::Constants<TOutput>::_two() * ql::Constants<TOutput>::_two();  
        res(i,1) = fac * ql::Constants<TOutput>::_two() * (-lnrat_tamu2 - lnrat_simu2);
        res(i,0) = fac * (lnrat_tamu2 * lnrat_tamu2 + lnrat_simu2 * lnrat_simu2 - lnrat_tasi * lnrat_tasi - TOutput(ql::Constants<TScale>::_pi2()));
    }

    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(0,0,0,p_4^2;s_{12},s_{23};0,0,0,0) = \frac{\mu^{2\epsilon}}{s_{12}s_{23}}\left[ \frac{2}{\epsilon^2}\left( (-s_{12})^{-\epsilon} + (-s_{23})^{-\epsilon} -(-p_4^2)^{-\epsilon} \right) - 2 {\rm Li}_2 \left( 1-\frac{p_4^2}{s_{12}} \right) - 2 {\rm Li}_2 \left( 1-\frac{p_4^2}{s_{23}} \right) -  \ln^2 \left( \frac{-s_{12}}{-s_{23}} \right) - \frac{\pi^2}{3} \right] + \mathcal{O}(\epsilon)
    * \f]
    * Implementation of the formulae from Bern et al. \cite Bern:1993kr.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B2(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass si = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass ta = ql::Constants<TMass>::_two() * Y[1][3];
        const TMass mp4sq = ql::Constants<TMass>::_two() * Y[0][3];
        const TOutput fac = ql::Constants<TOutput>::_one() / (si * ta);

        const TOutput ln_mp4_mu2 = ql::Lnrat<TOutput, TMass, TScale>(mp4sq, mu2);
        const TOutput ln_ta_mu2 = ql::Lnrat<TOutput, TMass, TScale>(ta, mu2);
        const TOutput ln_si_mu2 = ql::Lnrat<TOutput, TMass, TScale>(si, mu2);
        const TOutput ln_mp4_ta = ql::Lnrat<TOutput, TMass, TScale>(mp4sq, ta);
        const TOutput ln_mp4_si = ql::Lnrat<TOutput, TMass, TScale>(mp4sq, si);
        const TOutput ln_ta_si = ql::Lnrat<TOutput, TMass, TScale>(ta, si);

        res(i,2) = fac * ql::Constants<TOutput>::_two();
        res(i,1) = res(i,2) * (ln_mp4_mu2 - ln_ta_mu2 - ln_si_mu2);
        res(i,0) = fac * (-ln_mp4_mu2 * ln_mp4_mu2 + ln_ta_mu2 * ln_ta_mu2 + ln_si_mu2 * ln_si_mu2
                    + ql::Constants<TOutput>::_two() * (ql::Li2omrat<TOutput, TMass, TScale>(ta, mp4sq) + ql::Li2omrat<TOutput, TMass, TScale>(si, mp4sq) - TOutput(ql::Constants<TScale>::template _pi2o6<TOutput, TMass, TScale>()))
                    + ln_mp4_ta * ln_mp4_ta + ln_mp4_si * ln_mp4_si - ln_ta_si * ln_ta_si);
    }

    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(0,p_2^2,0,p_4^2;s_{12},s_{23};0,0,0,0) = \frac{\mu^{2\epsilon}}{s_{12}s_{23}-p_4^2 p_2^2} \\
    *    \left[ \frac{2}{\epsilon^2}\left( (-s_{12})^{-\epsilon} + (-s_{23})^{-\epsilon}-(-p_2^2)^{-\epsilon}-(-p_4^2)^{-\epsilon} \right) \\
    * - 2 {\rm Li}_2 \left( 1-\frac{p_2^2}{s_{12}} \right) - 2 {\rm Li}_2 \left( 1-\frac{p_2^2}{s_{23}} \right) - 2 {\rm Li}_2 \left( 1-\frac{p_4^2}{s_{12}} \right) - 2 {\rm Li}_2 \left( 1-\frac{p_4^2}{s_{23}} \right) \\
    * + 2 {\rm Li}_2 \left( 1-\frac{p_4^2 p_2^2}{s_{23}s_{12}} \right) -  \ln^2 \left( \frac{-s_{12}}{-s_{23}} \right) \right] + \mathcal{O}(\epsilon)
    * \f]
    * Implementation of the formulae from Bern et al. \cite Bern:1993kr.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B3(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass si = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass ta = ql::Constants<TMass>::_two() * Y[1][3];
        const TMass mp4sq = ql::Constants<TMass>::_two() * Y[0][3];
        const TMass mp2sq = ql::Constants<TMass>::_two() * Y[1][2];
        const TMass r = ql::Constants<TMass>::_one() - mp2sq * mp4sq / (si * ta);
        const TScale SignRealsi = ql::Sign(ql::Real(si));
        const TScale SignRealmp2sq = ql::Sign(ql::Real(mp2sq));

        // Use expansion only in cases where signs are not ++-- or --++
        const bool landau = ((SignRealsi == ql::Sign(ql::Real(ta))) &&
                            (SignRealmp2sq == ql::Sign(ql::Real(mp4sq))) &&
                            (SignRealsi != SignRealmp2sq));

        if (ql::kAbs(r) < ql::Constants<TScale>::_eps() && landau == false) {
            // Expanded case
            const TOutput fac = ql::Constants<TOutput>::_one() / (si * ta);
            const TOutput ln_si_mu2 = ql::Lnrat<TOutput, TMass, TScale>(si, mu2);
            const TOutput ln_ta_mp4 = ql::Lnrat<TOutput, TMass, TScale>(ta, mp4sq);
            const TOutput l0_mp4_ta = ql::L0<TOutput, TMass, TScale>(mp4sq, ta);
            const TOutput l0_mp4_si = ql::L0<TOutput, TMass, TScale>(mp4sq, si);
            const TOutput l1_mp4_ta = ql::L1<TOutput, TMass, TScale>(mp4sq, ta);
            const TOutput l1_mp4_si = ql::L1<TOutput, TMass, TScale>(mp4sq, si);

            res(i,2) = ql::Constants<TOutput>::_zero();
            res(i,1) = -(ql::Constants<TOutput>::_two() + r) * fac;
            res(i,0) = fac * ( ql::Constants<TOutput>::_two() - ql::Constants<TOutput>::_half() * r + (ql::Constants<TOutput>::_two() + r) * (ln_si_mu2 + ln_ta_mp4)
                        + ql::Constants<TOutput>::_two() * (l0_mp4_ta + l0_mp4_si) + r * (l1_mp4_ta + l1_mp4_si));
        } else {
            // General case
            const TOutput fac = ql::Constants<TOutput>::_one() / (si * ta - mp2sq * mp4sq);
            const TOutput ln_mp2_mu2 = ql::Lnrat<TOutput, TMass, TScale>(mp2sq, mu2);
            const TOutput ln_mp4_mu2 = ql::Lnrat<TOutput, TMass, TScale>(mp4sq, mu2);
            const TOutput ln_si_mu2 = ql::Lnrat<TOutput, TMass, TScale>(si, mu2);
            const TOutput ln_ta_mu2 = ql::Lnrat<TOutput, TMass, TScale>(ta, mu2);
            const TOutput ln_s_t = ql::Lnrat<TOutput, TMass, TScale>(si, ta);
            const TOutput li2_1 = ql::Li2omrat<TOutput, TMass, TScale>(mp2sq, si);
            const TOutput li2_2 = ql::Li2omrat<TOutput, TMass, TScale>(mp2sq, ta);
            const TOutput li2_3 = ql::Li2omrat<TOutput, TMass, TScale>(mp4sq, si);
            const TOutput li2_4 = ql::Li2omrat<TOutput, TMass, TScale>(mp4sq, ta);
            const TOutput li2_5 = ql::Li2omx2<TOutput, TMass, TScale>(mp2sq, mp4sq, si, ta);

            res(i,2) = ql::Constants<TOutput>::_zero();
            res(i,1) = fac * ql::Constants<TOutput>::_two() * (ql::Lnrat<TOutput, TMass, TScale>(mp2sq, si) + ql::Lnrat<TOutput, TMass, TScale>(mp4sq, ta));
            res(i,0) = fac * (ln_si_mu2 * ln_si_mu2 + ln_ta_mu2 * ln_ta_mu2
                        - ln_mp2_mu2 * ln_mp2_mu2 - ln_mp4_mu2 * ln_mp4_mu2
                        + ql::Constants<TOutput>::_two() * (li2_5 - li2_1 - li2_2 - li2_3 - li2_4 - ql::Constants<TOutput>::_half() * ln_s_t * ln_s_t));
        }
    }


    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(0,0,p_3^2,p_4^2;s_{12},s_{23};0,0,0,0) = \frac{\mu^{2\epsilon}}{s_{12}s_{23}} \\
    *     \left[ \frac{2}{\epsilon^2}\left( (-s_{12})^{-\epsilon} + (-s_{23})^{-\epsilon}-(-p_3^2)^{-\epsilon}-(-p_4^2)^{-\epsilon} \right) + \frac{1}{\epsilon^2} \left( (-p_3^2)^{-\epsilon}(-p_4)^{-\epsilon} \right) / (-s_{12})^{-\epsilon} \\
    * - 2 {\rm Li}_2 \left( 1-\frac{p_3^2}{s_{23}} \right) - 2 {\rm Li}_2 \left( 1-\frac{p_4^2}{s_{23}} \right) -  \ln^2 \left( \frac{-s_{12}}{-s_{23}} \right) \right] + \mathcal{O}(\epsilon)
    * \f]
    * Implementation of the formulae from Bern et al. \cite Bern:1993kr.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B4(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass si = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass ta = ql::Constants<TMass>::_two() * Y[1][3];
        const TMass mp4sq = ql::Constants<TMass>::_two() * Y[0][3];
        const TMass mp3sq = ql::Constants<TMass>::_two() * Y[2][3];
        const TOutput fac = ql::Constants<TOutput>::_one() / (si * ta);

        const TOutput ln_ta_mu2 = ql::Lnrat<TOutput, TMass, TScale>(ta, mu2);
        const TOutput ln_si_mu2 = ql::Lnrat<TOutput, TMass, TScale>(si, mu2);
        const TOutput ln_mp3_mu2 = ql::Lnrat<TOutput, TMass, TScale>(mp3sq, mu2);
        const TOutput ln_mp4_mu2 = ql::Lnrat<TOutput, TMass, TScale>(mp4sq, mu2);
        const TOutput ln_si_ta = ql::Lnrat<TOutput, TMass, TScale>(si, ta);
        const TOutput ln_si_mp3 = ql::Lnrat<TOutput, TMass, TScale>(si, mp3sq);

        res(i,2) = fac;
        res(i,1) = -fac * (ln_si_mp3 + ql::Lnrat<TOutput, TMass, TScale>(ta, mp4sq) + ln_ta_mu2);
        res(i,0) = fac * (ln_ta_mu2 * ln_ta_mu2
                    + ql::Constants<TOutput>::_half() * ln_si_mu2 * ln_si_mu2
                    - ql::Constants<TOutput>::_half() * ln_mp3_mu2 * ln_mp3_mu2
                    - ql::Constants<TOutput>::_half() * ln_mp4_mu2 * ln_mp4_mu2
                    + ql::Constants<TOutput>::_two() * (-ql::Li2omrat<TOutput, TMass, TScale>(mp3sq, ta) - ql::Li2omrat<TOutput, TMass, TScale>(mp4sq, ta) +
                                    ql::Constants<TOutput>::_half() * (ln_si_mp3 * ql::Lnrat<TOutput, TMass, TScale>(si, mp4sq) - ln_si_ta * ln_si_ta)));
    }


    /*!
    * The integral is defined as:
    * \f[
    * I_4^{D=4-2\epsilon}(0,p_2^2,p_3^2,p_4^2;s_{12},s_{23};0,0,0,0) = \frac{\mu^{2\epsilon}}{s_{12}s_{23}-p_2^2 p_4^2} \\
    *     \left[ \frac{2}{\epsilon^2}\left( (-s_{12})^{-\epsilon} + (-s_{23})^{-\epsilon}-(-p_2^2)^{-\epsilon}-(-p_3^2)^{-\epsilon} -(-p_4^2)^{-\epsilon} \right) \\
    *  + \frac{1}{\epsilon^2} \left( (-p_2^2)^{-\epsilon}(-p_3)^{-\epsilon} \right) / (-s_{23})^{-\epsilon}+ \frac{1}{\epsilon^2} \left( (-p_3^2)^{-\epsilon}(-p_4)^{-\epsilon} \right) / (-s_{12})^{-\epsilon} \\
    * - 2 {\rm Li}_2 \left( 1-\frac{p_2^2}{s_{12}} \right) - 2 {\rm Li}_2 \left( 1-\frac{p_4^2}{s_{23}} \right) + 2 {\rm Li}_2 \left( 1-\frac{p_2^2 p_4^2}{s_{12} s_{23}} \right) -  \ln^2 \left( \frac{-s_{12}}{-s_{23}} \right) \right] + \mathcal{O}(\epsilon)
    * \f]
    * Implementation of the formulae from Bern et al. \cite Bern:1993kr.
    *
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param mu2 the energy scale squared.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B5(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        TScale const& mu2,
        const int i) {

        const TMass si = ql::Constants<TMass>::_two() * Y[0][2];
        const TMass ta = ql::Constants<TMass>::_two() * Y[1][3];
        const TMass mp2sq = ql::Constants<TMass>::_two() * Y[1][2];
        const TMass mp3sq = ql::Constants<TMass>::_two() * Y[2][3];
        const TMass mp4sq = ql::Constants<TMass>::_two() * Y[0][3];
        const TMass r = ql::Constants<TMass>::_one() - mp2sq * mp4sq / (si * ta);
        const TScale SignRealsi = ql::Sign(ql::Real(si));
        const TScale SignRealmp2sq = ql::Sign(ql::Real(mp2sq));

        // Use expansion only in cases where signs are not ++-- or --++
        const bool landau = ((SignRealsi == ql::Sign(ql::Real(ta))) &&
                            (SignRealmp2sq == ql::Sign(ql::Real(mp4sq))) &&
                            (SignRealsi != SignRealmp2sq));

        if (ql::kAbs(r) < ql::Constants<TScale>::_eps() && landau == false) {
            const TOutput l0_mp4_ta = ql::L0<TOutput, TMass, TScale>(mp4sq, ta);
            const TOutput l1_mp4_ta = ql::L1<TOutput, TMass, TScale>(mp4sq, ta);

            // Expanded case
            res(i,2) = ql::Constants<TOutput>::_zero();
            res(i,1) = -(ql::Constants<TOutput>::_one() + ql::Constants<TOutput>::_half() * r) / (si * ta);
            res(i,0) = res(i,1) * (ql::Lnrat<TOutput, TMass, TScale>(mu2, si) +
                                ql::Lnrat<TOutput, TMass, TScale>(mp3sq, ta) - ql::Constants<TOutput>::_two() -
                                (ql::Constants<TOutput>::_one() + mp4sq / ta) * l0_mp4_ta) +
                    (r / (si * ta)) * (l1_mp4_ta - l0_mp4_ta - ql::Constants<TOutput>::_one());
        } else {
            // General case
            const TOutput fac = ql::Constants<TOutput>::_one() / (si * ta - mp2sq * mp4sq);
            const TOutput li2_1 = ql::Li2omrat<TOutput, TMass, TScale>(mp2sq, si);
            const TOutput li2_2 = ql::Li2omrat<TOutput, TMass, TScale>(mp4sq, ta);
            const TOutput li2_3 = ql::Li2omx2<TOutput, TMass, TScale>(mp2sq, mp4sq, si, ta);
            const TOutput ln_ta_mp2 = ql::Lnrat<TOutput, TMass, TScale>(ta, mp2sq);
            const TOutput ln_si_mp4 = ql::Lnrat<TOutput, TMass, TScale>(si, mp4sq);
            const TOutput ln_si_ta = ql::Lnrat<TOutput, TMass, TScale>(si, ta);

            res(i,2) = ql::Constants<TOutput>::_zero();
            res(i,1) = -ln_ta_mp2 - ln_si_mp4;
            res(i,0) = -ql::Constants<TOutput>::_half() * (ln_ta_mp2 * ln_ta_mp2 + ln_si_mp4 * ln_si_mp4) -
                    (ql::Lnrat<TOutput, TMass, TScale>(mp3sq, ta) +
                        ql::Lnrat<TOutput, TMass, TScale>(mu2, ta)) * ln_ta_mp2 -
                    (ql::Lnrat<TOutput, TMass, TScale>(mp3sq, si) +
                        ql::Lnrat<TOutput, TMass, TScale>(mu2, si)) * ln_si_mp4 -
                    ql::Constants<TOutput>::_two() * (li2_1 + li2_2 - li2_3) - ln_si_ta * ln_si_ta;

            res(i,1) *= fac;
            res(i,0) *= fac;
        }
    }

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B0m(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<TMass, 13>& xpi, 
        const TScale& mu2,
        const int i) {

        int offshell = 0, jsort0 = 0, jsort1 = 0, jsort2 = 0;
        for (int j = 0; j < 4; j++) {
            if (!ql::iszero<TOutput, TMass, TScale>(xpi[j + 4])) {
                offshell += 1;
                if (jsort1 == 0) jsort1 = j + 1;
                else jsort2 = j + 1;
            } else 
                jsort0 = j;
        }

        Kokkos::Array<TMass, 13> xpiout;
        bool swapped = true;
        const int jdiff = jsort2 - jsort1;

        if (offshell == 1) {
            for (size_t j = 0; j < 13; j++) {
                xpiout[ql::swap_b0m(j, jsort1 - 1)] = xpi[j];
            }
        } else if (offshell == 2 && (jdiff == 2 || jdiff == 1)) {
            for (size_t j = 0; j < 13; j++) {
                xpiout[ql::swap_b0m(j, jsort2 - 1)] = xpi[j];
            }
        } else if (offshell == 2 && jdiff == 3) {
            for (size_t j = 0; j < 13; j++) {
                xpiout[ql::swap_b0m(j, 0)] = xpi[j];
            }
        } else if (offshell == 3) {
            for (size_t j = 0; j < 13; j++) {
                xpiout[ql::swap_b0m(j, ql::jsort_b0m(jsort0))] = xpi[j];
            }
        } else {
            swapped = false;
        }

        Kokkos::Array<Kokkos::Array<TMass, 4>, 4> Y;
        Kokkos::Array<TMass, 13> xpi_in;
        if (swapped) 
            xpi_in = xpiout;
        else 
            xpi_in = xpi;

        
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

        if (offshell == 4) 
            ql::BIN0<TOutput, TMass, TScale>(res, Y, i);
        else if (offshell == 3) 
            ql::B5<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (offshell == 2) {
            if (!ql::iszero<TOutput, TMass, TScale>(xpi_in[5])) 
                ql::B3<TOutput, TMass, TScale>(res, Y, mu2, i);
            else if (!ql::iszero<TOutput, TMass, TScale>(xpi_in[6]) && !ql::iszero<TOutput, TMass, TScale>(xpi_in[7])) 
                ql::B4<TOutput, TMass, TScale>(res, Y, mu2, i);
        } else if (offshell == 1)
            ql::B2<TOutput, TMass, TScale>(res, Y, mu2, i);
        else if (offshell == 0) 
            ql::B1<TOutput, TMass, TScale>(res, Y, mu2, i);
    }           

#ifndef QCDLOOP_BOX_FULL_DISPATCH
    /*!
    * Pruned BO for massive == 0 (0 internal masses).
    * Includes only the B0m dispatch path.
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
        
        if (massive == 0) {
            ql::B0m<TOutput, TMass, TScale>(res, xpi, musq, i);
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
