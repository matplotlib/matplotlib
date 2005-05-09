//----------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.3
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
// Perspective 2D transformations
//
//----------------------------------------------------------------------------
#ifndef AGG_TRANS_PERSPECTIVE_INCLUDED
#define AGG_TRANS_PERSPECTIVE_INCLUDED

#include "agg_basics.h"
#include "agg_simul_eq.h"

namespace agg
{
    //=======================================================trans_perspective
    class trans_perspective
    {
    public:
        //--------------------------------------------------------------------
        trans_perspective() : m_valid(false) {}


        //--------------------------------------------------------------------
        // Arbitrary quadrangle transformations
        trans_perspective(const double* src, const double* dst) 
        {
            quad_to_quad(src, dst);
        }


        //--------------------------------------------------------------------
        // Direct transformations 
        trans_perspective(double x1, double y1, double x2, double y2, 
                          const double* quad)
        {
            rect_to_quad(x1, y1, x2, y2, quad);
        }


        //--------------------------------------------------------------------
        // Reverse transformations 
        trans_perspective(const double* quad, 
                          double x1, double y1, double x2, double y2)
        {
            quad_to_rect(quad, x1, y1, x2, y2);
        }


        //--------------------------------------------------------------------
        // Set the transformations using two arbitrary quadrangles.
        void quad_to_quad(const double* src, const double* dst)
        {

            double left[8][8];
            double right[8][1];

            unsigned i;
            for (i = 0; i < 4; i++)
            {
                unsigned ix = i * 2;
                unsigned iy = ix + 1;

                left[ix][0]  =  1.0;
                left[ix][1]  =  src[ix];
                left[ix][2]  =  src[iy];
                left[ix][3]  =  0.0;
                left[ix][4]  =  0.0;
                left[ix][5]  =  0.0;
                left[ix][6]  = -src[ix] * dst[ix];
                left[ix][7]  = -src[iy] * dst[ix];
                right[ix][0] =  dst[ix];

                left[iy][0]  =  0.0;
                left[iy][1]  =  0.0;
                left[iy][2]  =  0.0;
                left[iy][3]  =  1.0;
                left[iy][4]  =  src[ix];
                left[iy][5]  =  src[iy];
                left[iy][6]  = -src[ix] * dst[iy];
                left[iy][7]  = -src[iy] * dst[iy];
                right[iy][0] =  dst[iy];
            }
            m_valid = simul_eq<8, 1>::solve(left, right, m_mtx);
        }


        //--------------------------------------------------------------------
        // Set the direct transformations, i.e., rectangle -> quadrangle
        void rect_to_quad(double x1, double y1, double x2, double y2, 
                          const double* quad)
        {
            double src[8];
            src[0] = src[6] = x1;
            src[2] = src[4] = x2;
            src[1] = src[3] = y1;
            src[5] = src[7] = y2;
            quad_to_quad(src, quad);
        }


        //--------------------------------------------------------------------
        // Set the reverse transformations, i.e., quadrangle -> rectangle
        void quad_to_rect(const double* quad, 
                          double x1, double y1, double x2, double y2)
        {
            double dst[8];
            dst[0] = dst[6] = x1;
            dst[2] = dst[4] = x2;
            dst[1] = dst[3] = y1;
            dst[5] = dst[7] = y2;
            quad_to_quad(quad, dst);
        }

        //--------------------------------------------------------------------
        // Check if the equations were solved successfully
        bool is_valid() const { return m_valid; }

        //--------------------------------------------------------------------
        // Transform a point (x, y)
        void transform(double* x, double* y) const
        {
            double tx = *x;
            double ty = *y;
            double d = 1.0 / (m_mtx[6][0] * tx + m_mtx[7][0] * ty + 1.0);
            *x = (m_mtx[0][0] + m_mtx[1][0] * tx + m_mtx[2][0] * ty) * d;
            *y = (m_mtx[3][0] + m_mtx[4][0] * tx + m_mtx[5][0] * ty) * d;
        }

        //--------------------------------------------------------------------
        class iterator_x
        {
            double den;
            double den_step;
            double nom_x;
            double nom_x_step;
            double nom_y;
            double nom_y_step;

        public:
            double x;
            double y;

            iterator_x() {}
            iterator_x(double tx, double ty, double step, const double m[8][1]) :
                den(m[6][0] * tx + m[7][0] * ty + 1.0),
                den_step(m[6][0] * step),
                nom_x(m[0][0] + m[1][0] * tx + m[2][0] * ty),
                nom_x_step(m[1][0] * step),
                nom_y(m[3][0] + m[4][0] * tx + m[5][0] * ty),
                nom_y_step(m[4][0] * step),
                x(nom_x / den),
                y(nom_y / den)
            {
            }

            void operator ++ ()
            {
                den   += den_step;
                nom_x += nom_x_step;
                nom_y += nom_y_step;
                double d = 1.0 / den;
                x = nom_x * d;
                y = nom_y * d;
            }
        };

        //--------------------------------------------------------------------
        iterator_x begin(double x, double y, double step) const
        {
            return iterator_x(x, y, step, m_mtx);
        }

    private:
        double m_mtx[8][1];
        bool   m_valid;
    };

}

#endif
