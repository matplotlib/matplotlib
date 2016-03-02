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
#ifndef AGG_LINE_AA_BASICS_INCLUDED
#define AGG_LINE_AA_BASICS_INCLUDED

#include <stdlib.h>
#include "agg_basics.h"

namespace agg
{

    // See Implementation agg_line_aa_basics.cpp 

    //-------------------------------------------------------------------------
    enum line_subpixel_scale_e
    {
        line_subpixel_shift = 8,                          //----line_subpixel_shift
        line_subpixel_scale  = 1 << line_subpixel_shift,  //----line_subpixel_scale
        line_subpixel_mask  = line_subpixel_scale - 1,    //----line_subpixel_mask
        line_max_coord      = (1 << 28) - 1,              //----line_max_coord
        line_max_length = 1 << (line_subpixel_shift + 10) //----line_max_length
    };

    //-------------------------------------------------------------------------
    enum line_mr_subpixel_scale_e
    {
        line_mr_subpixel_shift = 4,                           //----line_mr_subpixel_shift
        line_mr_subpixel_scale = 1 << line_mr_subpixel_shift, //----line_mr_subpixel_scale 
        line_mr_subpixel_mask  = line_mr_subpixel_scale - 1   //----line_mr_subpixel_mask 
    };

    //------------------------------------------------------------------line_mr
    AGG_INLINE int line_mr(int x) 
    { 
        return x >> (line_subpixel_shift - line_mr_subpixel_shift); 
    }

    //-------------------------------------------------------------------line_hr
    AGG_INLINE int line_hr(int x) 
    { 
        return x << (line_subpixel_shift - line_mr_subpixel_shift); 
    }

    //---------------------------------------------------------------line_dbl_hr
    AGG_INLINE int line_dbl_hr(int x) 
    { 
        return x << line_subpixel_shift;
    }

    //---------------------------------------------------------------line_coord
    struct line_coord
    {
        AGG_INLINE static int conv(double x)
        {
            return iround(x * line_subpixel_scale);
        }
    };

    //-----------------------------------------------------------line_coord_sat
    struct line_coord_sat
    {
        AGG_INLINE static int conv(double x)
        {
            return saturation<line_max_coord>::iround(x * line_subpixel_scale);
        }
    };

    //==========================================================line_parameters
    struct line_parameters
    {
        //---------------------------------------------------------------------
        line_parameters() {}
        line_parameters(int x1_, int y1_, int x2_, int y2_, int len_) :
            x1(x1_), y1(y1_), x2(x2_), y2(y2_), 
            dx(abs(x2_ - x1_)),
            dy(abs(y2_ - y1_)),
            sx((x2_ > x1_) ? 1 : -1),
            sy((y2_ > y1_) ? 1 : -1),
            vertical(dy >= dx),
            inc(vertical ? sy : sx),
            len(len_),
            octant((sy & 4) | (sx & 2) | int(vertical))
        {
        }

        //---------------------------------------------------------------------
        unsigned orthogonal_quadrant() const { return s_orthogonal_quadrant[octant]; }
        unsigned diagonal_quadrant()   const { return s_diagonal_quadrant[octant];   }

        //---------------------------------------------------------------------
        bool same_orthogonal_quadrant(const line_parameters& lp) const
        {
            return s_orthogonal_quadrant[octant] == s_orthogonal_quadrant[lp.octant];
        }

        //---------------------------------------------------------------------
        bool same_diagonal_quadrant(const line_parameters& lp) const
        {
            return s_diagonal_quadrant[octant] == s_diagonal_quadrant[lp.octant];
        }

        //---------------------------------------------------------------------
        void divide(line_parameters& lp1, line_parameters& lp2) const
        {
            int xmid = (x1 + x2) >> 1;
            int ymid = (y1 + y2) >> 1;
            int len2 = len >> 1;

            lp1 = *this;
            lp2 = *this;

            lp1.x2  = xmid;
            lp1.y2  = ymid;
            lp1.len = len2;
            lp1.dx  = abs(lp1.x2 - lp1.x1);
            lp1.dy  = abs(lp1.y2 - lp1.y1);

            lp2.x1  = xmid;
            lp2.y1  = ymid;
            lp2.len = len2;
            lp2.dx  = abs(lp2.x2 - lp2.x1);
            lp2.dy  = abs(lp2.y2 - lp2.y1);
        }
        
        //---------------------------------------------------------------------
        int x1, y1, x2, y2, dx, dy, sx, sy;
        bool vertical;
        int inc;
        int len;
        int octant;

        //---------------------------------------------------------------------
        static const int8u s_orthogonal_quadrant[8];
        static const int8u s_diagonal_quadrant[8];
    };



    // See Implementation agg_line_aa_basics.cpp 

    //----------------------------------------------------------------bisectrix
    void bisectrix(const line_parameters& l1, 
                   const line_parameters& l2, 
                   int* x, int* y);


    //-------------------------------------------fix_degenerate_bisectrix_start
    void inline fix_degenerate_bisectrix_start(const line_parameters& lp, 
                                               int* x, int* y)
    {
        int d = iround((double(*x - lp.x2) * double(lp.y2 - lp.y1) - 
                        double(*y - lp.y2) * double(lp.x2 - lp.x1)) / lp.len);
        if(d < line_subpixel_scale/2)
        {
            *x = lp.x1 + (lp.y2 - lp.y1);
            *y = lp.y1 - (lp.x2 - lp.x1);
        }
    }


    //---------------------------------------------fix_degenerate_bisectrix_end
    void inline fix_degenerate_bisectrix_end(const line_parameters& lp, 
                                             int* x, int* y)
    {
        int d = iround((double(*x - lp.x2) * double(lp.y2 - lp.y1) - 
                        double(*y - lp.y2) * double(lp.x2 - lp.x1)) / lp.len);
        if(d < line_subpixel_scale/2)
        {
            *x = lp.x2 + (lp.y2 - lp.y1);
            *y = lp.y2 - (lp.x2 - lp.x1);
        }
    }


}

#endif
