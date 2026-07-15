//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//
// Box integrals with 4 internal masses.
// Contains: BIN4, B4m dispatcher, pruned BO.

#pragma once

#include "box_common.h"


namespace ql
{

    /*! 
    * Finite box with 4 non-zero masses. Formulae from \cite Denner:1991qq.
    * \param res output object res[0,1,2] the coefficients in the Laurent series, following the LoopTools implementation \cite Hahn:2006qw.
    * \param Y the modified Cayley matrix.
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void BIN4(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        const int i) {
    
        TMass tmp;
        TMass M[4];
        for (int j = 0; j < 4; j++)
            M[j] = Y[j][j];

        TMass k12 = ql::Constants<TMass>::_two() * Y[0][1] / ql::kSqrt(M[0] * M[1]);
        TMass k23 = ql::Constants<TMass>::_two() * Y[1][2] / ql::kSqrt(M[1] * M[2]);
        TMass k34 = ql::Constants<TMass>::_two() * Y[2][3] / ql::kSqrt(M[2] * M[3]);
        TMass k14 = ql::Constants<TMass>::_two() * Y[0][3] / ql::kSqrt(M[0] * M[3]);
        TMass k13 = ql::Constants<TMass>::_two() * Y[0][2] / ql::kSqrt(M[0] * M[2]);
        TMass k24 = ql::Constants<TMass>::_two() * Y[1][3] / ql::kSqrt(M[1] * M[3]);

        if (ql::kAbs(k13) >= ql::Constants<TMass>::_two()) { /*do nothing*/ 
        } else if (ql::kAbs(k12) >= ql::Constants<TMass>::_two()) {
            // 2 <-> 3
            tmp = k12;
            k12 = k13;
            k13 = tmp;
            tmp = k24;
            k24 = k34;
            k34 = tmp;
        } else if (ql::kAbs(k14) >= ql::Constants<TMass>::_two()) {
            tmp = k13;
            k13 = k14;
            k14 = tmp;
            tmp = k23;
            k23 = k24;
            k24 = tmp;
        } else if (ql::kAbs(k23) >= ql::Constants<TMass>::_two()) {
            tmp = k13;
            k13 = k23;
            k23 = tmp;
            tmp = k14;
            k14 = k24;
            k24 = tmp;
        } else if (ql::kAbs(k24) >= ql::Constants<TMass>::_two()) {
            tmp = k12;
            k12 = k23;
            k23 = k34;
            k34 = k14;
            k14 = tmp;
            tmp = k13;
            k13 = k24;
            k24 = tmp;
        } else if (ql::kAbs(k34) >= ql::Constants<TMass>::_two()) {
            tmp = k12;
            k12 = k24;
            k24 = tmp;
            tmp = k13;
            k13 = k34;
            k34 = tmp;
        }
        // else nothing found, all r_ij on the complex unit circle

        TMass kij[6];
        kij[0] = k12;
        kij[1] = k23;
        kij[2] = k34;
        kij[3] = k14;
        kij[4] = k13;
        kij[5] = k24;

        TOutput rij[6];
        TOutput r12 = rij[0] = ql::Constants<TOutput>::_half() * (TOutput(k12) + TOutput(ql::Sign(ql::Real(k12))) * ql::kSqrt(TOutput((k12 - ql::Constants<TMass>::_two()) * (k12 + ql::Constants<TMass>::_two()))));
        TOutput r23 = rij[1] = ql::Constants<TOutput>::_half() * (TOutput(k23) + TOutput(ql::Sign(ql::Real(k23))) * ql::kSqrt(TOutput((k23 - ql::Constants<TMass>::_two()) * (k23 + ql::Constants<TMass>::_two()))));
        TOutput r34 = rij[2] = ql::Constants<TOutput>::_half() * (TOutput(k34) + TOutput(ql::Sign(ql::Real(k34))) * ql::kSqrt(TOutput((k34 - ql::Constants<TMass>::_two()) * (k34 + ql::Constants<TMass>::_two()))));
        TOutput r14 = rij[3] = ql::Constants<TOutput>::_half() * (TOutput(k14) + TOutput(ql::Sign(ql::Real(k14))) * ql::kSqrt(TOutput((k14 - ql::Constants<TMass>::_two()) * (k14+ql::Constants<TMass>::_two()))));
        TOutput r13 = rij[4] = ql::Constants<TOutput>::_one() / (ql::Constants<TOutput>::_half() * (TOutput(k13) + TOutput(ql::Sign(ql::Real(k13))) * ql::kSqrt(TOutput((k13 - ql::Constants<TMass>::_two()) * (k13 + ql::Constants<TMass>::_two())))));
        TOutput r24 = rij[5] = ql::Constants<TOutput>::_one() / (ql::Constants<TOutput>::_half() * (TOutput(k24) + TOutput(ql::Sign(ql::Real(k24))) * ql::kSqrt(TOutput((k24 - ql::Constants<TMass>::_two()) * (k24 + ql::Constants<TMass>::_two())))));

        TScale irij[6];
        for (int j = 0; j < 6; j++) {
            if (ql::Imag(rij[j]) == ql::Constants<TScale>::_zero()) {

                const TOutput ki = TOutput(kij[j]) - ql::Constants<TOutput>::template _ieps50<TOutput,TMass,TScale>();
                const TOutput kk = ql::Constants<TOutput>::_half() * (ki + TOutput(ql::Sign(ql::Real(ki))) * ql::kSqrt((ki - ql::Constants<TOutput>::_two()) * (ki + ql::Constants<TOutput>::_two())));
                auto val = (ql::kAbs(rij[j]) - ql::Constants<TScale>::_one()) * ql::Imag(kk);
                irij[j] = TScale{val};

            } else {
                irij[j] = ql::Constants<TScale>::_zero();
            }
        }

        const TScale ir13 = irij[4];
        const TScale ir24 = irij[5];
        const TScale ir1324 = ql::Sign(ql::Real(r24)) * ir13 - ql::Sign(ql::Real(r13)) * ir24;

        const TOutput a = TOutput(k34) / r24 - TOutput(k23) + (TOutput(k12) - TOutput(k14) / r24) * r13;
        const TOutput b = (ql::Constants<TOutput>::_one() / r13 - r13) * (ql::Constants<TOutput>::_one() / r24 - r24) + TOutput(k12 * k34) - TOutput(k14 * k23);
        const TOutput c = TOutput(k34) * r24 - TOutput(k23) + (TOutput(k12) - TOutput(k14) * r24) / r13;
        const TOutput d = TOutput(k23) + (r24 * TOutput(k14) - TOutput(k12)) * r13 - r24 * TOutput(k34);
        TOutput disc = ql::kSqrt(b * b - TOutput(ql::Constants<TScale>::_four()) * a * (c + d * ql::Constants<TOutput>::template _ieps50<TOutput, TMass, TScale>()));

        TScale ix[2][4];
        ix[0][3] = ql::Imag(ql::Constants<TOutput>::_half() / a * (b - disc));
        ix[1][3] = ql::Imag(ql::Constants<TOutput>::_half() / a * (b + disc));

        disc = ql::kSqrt(b * b - TOutput(ql::Constants<TScale>::_four()) * a * c);
        TOutput x[2][4];
        x[0][3] = ql::Constants<TOutput>::_half() / a * (b - disc);
        x[1][3] = ql::Constants<TOutput>::_half() / a * (b + disc);
        if (ql::kAbs(x[0][3]) > ql::kAbs(x[1][3]))
            x[1][3] = c / (a * x[0][3]);
        else
            x[0][3] = c / (a * x[1][3]);

        x[0][0] = x[0][3] / r24;
        x[1][0] = x[1][3] / r24;
        x[0][1] = x[0][3] * r13 / r24;
        x[1][1] = x[1][3] * r13 / r24;
        x[0][2] = x[0][3] * r13;
        x[1][2] = x[1][3] * r13;

        const TScale s1 = ql::Sign(ql::Real(x[0][3]));
        const TScale s2 = ql::Sign(ql::Real(x[1][3]));
        ix[0][0] = ix[0][3] * ql::Real(x[0][0]) * s1;
        ix[1][0] = ix[1][3] * ql::Real(x[1][0]) * s2;
        ix[0][1] = ix[0][3] * ql::Real(x[0][1]) * s1;
        ix[1][1] = ix[1][3] * ql::Real(x[1][1]) * s2;
        ix[0][2] = ix[0][3] * ql::Real(x[0][2]) * s1;
        ix[1][2] = ix[1][3] * ql::Real(x[1][2]) * s2;

        res(i,0) = ql::Constants<TOutput>::_zero();
        for (int j = 0; j < 4; j++) {
            const Kokkos::Array<TOutput, 2> x_in = { x[0][j], x[1][j] }; 
            const Kokkos::Array<TScale, 2> ix_in = { ix[0][j], ix[1][j] };
            res(i,0) += ql::kPow<TOutput, TMass, TScale>(-ql::Constants<TOutput>::_one() ,j+1) * (
                    ql::xspence<TOutput, TMass, TScale>(x_in, ix_in, rij[j], irij[j]) +
                    ql::xspence<TOutput, TMass, TScale>(x_in, ix_in, ql::Constants<TOutput>::_one() / rij[j], -irij[j])
                    );
        }

        const TScale gamma = ql::Sign(ql::Real(a*(x[1][3] - x[0][3])) + ql::Constants<TScale>::_reps());
        TOutput l[2][4];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 4; j++)
                l[i][j] = ql::Constants<TOutput>::_zero();
        l[0][3] = ql::Constants<TScale>::template _2ipi<TOutput, TMass, TScale>() * TOutput(ql::eta<TOutput, TMass, TScale>(r13, ir13, ql::Constants<TOutput>::_one() / r24, -ir24, ir1324));
        l[1][3] = l[0][3];

        TOutput etas = ql::Constants<TOutput>::_zero();
        if (ql::Imag(r13) == 0) {
            r12 = TOutput(k12) - r24 * TOutput(k14);
            r23 = TOutput(k23) - r24 * TOutput(k34);
            r34 = TOutput(k34) - r13 * TOutput(k14);
            r14 = TOutput(k23) - r13 * TOutput(k12);
            const TOutput q13 = TOutput(k13) - ql::Constants<TOutput>::_two() * r13;
            const TOutput q24 = TOutput(k24) - ql::Constants<TOutput>::_two() * r24;

            TScale cc = gamma * ql::Sign(ql::Imag(r24) + ir24);
            l[0][0] = ql::cLn<TOutput, TMass, TScale>(-x[0][0], -ix[0][0]) +
                ql::cLn<TOutput, TMass, TScale>(r14 - q13/x[0][0], -ql::Constants<TScale>::_one()) +
                ql::cLn<TOutput, TMass, TScale>((r12 - q24 * x[0][3])/d, cc);
            l[1][0] = ql::cLn<TOutput, TMass, TScale>(-x[1][0], -ix[1][0]) +
                ql::cLn<TOutput, TMass, TScale>(r14 - q13/x[1][0], -ql::Constants<TScale>::_one()) +
                ql::cLn<TOutput, TMass, TScale>((r12 - q24 * x[1][3])/d, -cc);

            cc = gamma * ql::Sign(ql::Real(r13) * (ql::Imag(r24) + ir24));
            l[0][1] = ql::cLn<TOutput, TMass, TScale>(-x[0][1], -ix[0][1]) +
                ql::cLn<TOutput, TMass, TScale>(r14 - q13/x[0][0], -ql::Constants<TScale>::_one()) +
                ql::cLn<TOutput, TMass, TScale>((r23 - q24 * x[0][2]) / d, cc);
            l[1][1] = ql::cLn<TOutput, TMass, TScale>(-x[1][1], -ix[1][1]) +
                ql::cLn<TOutput, TMass, TScale>(r14 - q13 / x[1][0], -ql::Constants<TScale>::_one()) +
                ql::cLn<TOutput, TMass, TScale>((r23 - q24 * x[1][2]) / d, -cc);
            l[0][2] = ql::cLn<TOutput, TMass, TScale>(-x[0][2], -ix[0][2]) +
                ql::cLn<TOutput, TMass, TScale>(r34 - q13 / x[0][3], -ql::Constants<TScale>::_one()) +
                ql::cLn<TOutput, TMass, TScale>((r23 - q24 * x[0][2]) / d, cc);
            l[1][2] = ql::cLn<TOutput, TMass, TScale>(-x[1][2], -ix[1][2]) +
                ql::cLn<TOutput, TMass, TScale>(r34 - q13 / x[1][3], -ql::Constants<TScale>::_one()) +
                ql::cLn<TOutput, TMass, TScale>((r23 - q24 * x[1][2]) / d, -cc);

            const Kokkos::Array<TOutput, 2> x_in = {x[0][3], x[1][3]}; 
            const Kokkos::Array<TScale, 2> ix_in = {ix[0][3], ix[1][3]};
            const Kokkos::Array<TOutput, 2> l_in_a = {l[0][2], l[1][2]};
            const Kokkos::Array<TOutput, 2> l_in_b = {l[0][0], l[1][0]};
            const Kokkos::Array<TOutput, 2> l_in_c = {l[0][1], l[1][1]};
            const Kokkos::Array<TOutput, 2> l_in_d = {l[0][3], l[1][3]};

            etas = ql::xetatilde<TOutput, TMass, TScale>(x_in, ix_in, r13, ir13, l_in_a) +
                ql::xetatilde<TOutput, TMass, TScale>(x_in, ix_in, ql::Constants<TOutput>::_one() / r24, -ir24, l_in_b) -
                ql::xetatilde<TOutput, TMass, TScale>(x_in, ix_in, r13 / r24, ir1324, l_in_c) +
                ql::xetatilde<TOutput, TMass, TScale>(x_in, ix_in, -r13 / r24, -ir1324, l_in_d);
        } else {
            for (int j = 0; j < 3; j++) {
                l[0][j] = ql::kLog(-x[0][j]) + ql::cLn<TOutput, TMass, TScale>(TOutput(kij[j]) - ql::Constants<TOutput>::_one() / x[0][j]-x[0][j], -ql::Real(x[0][j] * b * TOutput(gamma)));
                l[1][j] = ql::kLog(-x[1][j]) + ql::cLn<TOutput, TMass, TScale>(TOutput(kij[j]) - ql::Constants<TOutput>::_one() / x[1][j]-x[1][j], -ql::Real(x[1][j] * b * TOutput(gamma)));
            }

            const Kokkos::Array<TOutput, 2> x_in = {x[0][3], x[1][3]}; 
            const Kokkos::Array<TScale, 2> ix_in = {ix[0][3], ix[1][3]};
            const Kokkos::Array<TOutput, 2> l_in_a = {l[0][2], l[1][2]};
            const Kokkos::Array<TOutput, 2> l_in_b = {l[0][0], l[1][0]};
            const Kokkos::Array<TOutput, 2> l_in_c = {l[0][1], l[1][1]};
            const Kokkos::Array<TOutput, 2> l_in_d = {l[0][3], l[1][3]};

            etas = ql::xeta<TOutput, TMass, TScale>(x_in, ix_in, r13, ir13, ix[0][2], l_in_a) +
                ql::xeta<TOutput, TMass, TScale>(x_in, ix_in, ql::Constants<TOutput>::_one() / r24, -ir24, ix[0][0], l_in_b) -
                ql::xeta<TOutput, TMass, TScale>(x_in, ix_in, r13 / r24, ir1324, ix[0][1], l_in_c) +
                ql::xeta<TOutput, TMass, TScale>(x_in, ix_in, -r13 / r24, -ir1324, ix[0][3], l_in_d) *
                (ql::Constants<TOutput>::_one() - TOutput(ql::Sign(ql::Real(b)) * gamma));
            }

        res(i,0) = (res(i,0) - ql::Constants<TScale>::template _2ipi<TOutput, TMass, TScale>() * TOutput(etas) +
            (l[1][1] - l[0][1]) * l[0][3]) / (TOutput(ql::kSqrt(M[0] * M[1] * M[2] * M[3])) * disc) ;

        res(i,2) = res(i,1) = ql::Constants<TOutput>::_zero();
    }

    /*!
    * This function triggers the topologies with 4 internal masses.
    * \param res output object res[0,1,2] the coefficients in the Laurent series
    * \param xpi array with masses and momenta squared
    */
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void B4m(
        const Kokkos::View<TOutput* [3]>& res, 
        const Kokkos::Array<TMass, 13>& xpi, 
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

        ql::BIN4<TOutput, TMass, TScale>(res, Y, i);
    }

#ifndef QCDLOOP_BOX_FULL_DISPATCH
    /*!
    * Pruned BO for massive == 4 (4 internal masses).
    * Includes only the B4m dispatch path.
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
        
        if (massive == 4) {
            ql::B4m<TOutput, TMass, TScale>(res, xpi, i);
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
