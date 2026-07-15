//
// QCDLoop + Kokkos 2025
//
// Authors: Reet Barik      : rbarik@anl.gov
//          Taylor Childers : jchilders@anl.gov
//
// Common utilities shared across box integral headers.

#pragma once

#include "kokkosUtils.h"


namespace ql
{
    // complex is defined in kokkosMaths.h

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION
    void Ycalc(
        Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Y,
        Kokkos::Array<Kokkos::Array<TMass, 4>, 4>& Yalt,
        int const& massive, 
        bool const& opposite= false) {
            
        if (massive == 1) {
            //C---exchange (1<-->3)
            Yalt[0][0] = Y[2][2];
            Yalt[1][1] = Y[1][1];
            Yalt[2][2] = Y[0][0];
            Yalt[3][3] = Y[3][3];
            Yalt[0][1] = Yalt[1][0] = Y[1][2];
            Yalt[0][2] = Yalt[2][0] = Y[0][2];
            Yalt[0][3] = Yalt[3][0] = Y[2][3];
            Yalt[1][2] = Yalt[2][1] = Y[0][1];
            Yalt[1][3] = Yalt[3][1] = Y[1][3];
            Yalt[2][3] = Yalt[3][2] = Y[0][3];
        } else if (massive == 2 && opposite) {
            //C---exchange (2<-->4) .and (1<-->3)
            Yalt[0][0] = Y[2][2];
            Yalt[1][1] = Y[3][3];
            Yalt[2][2] = Y[0][0];
            Yalt[3][3] = Y[1][1];
            Yalt[0][1] = Yalt[1][0] = Y[2][3];
            Yalt[0][2] = Yalt[2][0] = Y[0][2];
            Yalt[0][3] = Yalt[3][0] = Y[1][2];
            Yalt[1][2] = Yalt[2][1] = Y[0][3];
            Yalt[1][3] = Yalt[3][1] = Y[1][3];
            Yalt[2][3] = Yalt[3][2] = Y[0][1];
        } else if (massive == 2 && !opposite) {
            //C---exchange (1<-->2)and(3<-->4)
            Yalt[0][0] = Y[1][1];
            Yalt[1][1] = Y[0][0];
            Yalt[2][2] = Y[3][3];
            Yalt[3][3] = Y[2][2];
            Yalt[0][1] = Yalt[1][0] = Y[0][1];
            Yalt[2][3] = Yalt[3][2] = Y[2][3];
            Yalt[0][2] = Yalt[2][0] = Y[1][3];
            Yalt[0][3] = Yalt[3][0] = Y[1][2];
            Yalt[1][2] = Yalt[2][1] = Y[0][3];
            Yalt[1][3] = Yalt[3][1] = Y[0][2];
        } else {
            Kokkos::printf("Box::Ycalc - massive value not implemented");
        }
    }

}
