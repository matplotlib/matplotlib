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
// class bspline
//
//----------------------------------------------------------------------------

#ifndef AGG_BSPLINE_INCLUDED
#define AGG_BSPLINE_INCLUDED

#include "agg_array.h"

namespace agg
{
    //----------------------------------------------------------------bspline
    // A very simple class of Bi-cubic Spline interpolation.
    // First call init(num, x[], y[]) where num - number of source points, 
    // x, y - arrays of X and Y values respectively. Here Y must be a function 
    // of X. It means that all the X-coordinates must be arranged in the ascending
    // order. 
    // Then call get(x) that calculates a value Y for the respective X. 
    // The class supports extrapolation, i.e. you can call get(x) where x is
    // outside the given with init() X-range. Extrapolation is a simple linear 
    // function.
    //
    //  See Implementation agg_bspline.cpp
    //------------------------------------------------------------------------
    class bspline 
    {
    public:
        bspline();
        bspline(int num);
        bspline(int num, const double* x, const double* y);

        void   init(int num);
        void   add_point(double x, double y);
        void   prepare();

        void   init(int num, const double* x, const double* y);

        double get(double x) const;
        double get_stateful(double x) const;
    
    private:
        bspline(const bspline&);
        const bspline& operator = (const bspline&);

        static void bsearch(int n, const double *x, double x0, int *i);
        double extrapolation_left(double x) const;
        double extrapolation_right(double x) const;
        double interpolation(double x, int i) const;

        int               m_max;
        int               m_num;
        double*           m_x;
        double*           m_y;
        pod_array<double> m_am;
        mutable int       m_last_idx;
    };


}

#endif
