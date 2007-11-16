//----------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.4
// Copyright (C) 2002-2005 Maxim Shemanarev (http://www.antigrain.com)
//
// Permission to copy, use, modify, sell and distribute this software 
// is granted provided this copyright notice appears in all copies. 
// This software is provided "as is" without express or implied
// warranty, and with no claim as to its suitability for any purpose.
//
//----------------------------------------------------------------------------
// Contact: mcseem@antigrain.com
//          mcseemagg@yahoo.com
//          http://www.antigrain.com
//----------------------------------------------------------------------------
//
// Solving simultaneous equations
//
//----------------------------------------------------------------------------
#ifndef AGG_SIMUL_EQ_INCLUDED
#define AGG_SIMUL_EQ_INCLUDED

#include <math.h>
#include "agg_basics.h"

namespace agg
{

    //=============================================================swap_arrays
    template<class T> void swap_arrays(T* a1, T* a2, unsigned n)
    {
        unsigned i;
        for(i = 0; i < n; i++)
        {
            T tmp = *a1;
            *a1++ = *a2;
            *a2++ = tmp;
        }
    }


    //============================================================matrix_pivot
    template<unsigned Rows, unsigned Cols>
    struct matrix_pivot
    {
        static int pivot(double m[Rows][Cols], unsigned row)
        {
            int k = int(row);
            double max_val, tmp;

            max_val = -1.0;
            unsigned i;
            for(i = row; i < Rows; i++)
            {
                if((tmp = fabs(m[i][row])) > max_val && tmp != 0.0)
                {
                    max_val = tmp;
                    k = i;
                }
            }

            if(m[k][row] == 0.0)
            {
                return -1;
            }

            if(k != int(row))
            {
                swap_arrays(m[k], m[row], Cols);
                return k;
            }
            return 0;
        }
    };
    


    //===============================================================simul_eq
    template<unsigned Size, unsigned RightCols>
    struct simul_eq
    {
        static bool solve(const double left[Size][Size], 
                          const double right[Size][RightCols],
                          double result[Size][RightCols])
        {
            unsigned i, j, k;
            double a1;

            double tmp[Size][Size + RightCols];

            for(i = 0; i < Size; i++)
            {
                for(j = 0; j < Size; j++)
                {
                    tmp[i][j] = left[i][j];
                } 
                for(j = 0; j < RightCols; j++)
                {
                    tmp[i][Size + j] = right[i][j];
                }
            }

            for(k = 0; k < Size; k++)
            {
                if(matrix_pivot<Size, Size + RightCols>::pivot(tmp, k) < 0)
                {
                    return false; // Singularity....
                }

                a1 = tmp[k][k];

                for(j = k; j < Size + RightCols; j++)
                {
                    tmp[k][j] /= a1;
                }

                for(i = k + 1; i < Size; i++)
                {
                    a1 = tmp[i][k];
                    for (j = k; j < Size + RightCols; j++)
                    {
                        tmp[i][j] -= a1 * tmp[k][j];
                    }
                }
            }


            for(k = 0; k < RightCols; k++)
            {
                int m;
                for(m = int(Size - 1); m >= 0; m--)
                {
                    result[m][k] = tmp[m][Size + k];
                    for(j = m + 1; j < Size; j++)
                    {
                        result[m][k] -= tmp[m][j] * result[j][k];
                    }
                }
            }
            return true;
        }

    };


}

#endif
