//----------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.1
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

#ifndef AGG_WARP_MAGNIFIER_INCLUDED
#define AGG_WARP_MAGNIFIER_INCLUDED

#include <math.h>
#include "agg_basics.h"


namespace agg
{

    class trans_warp_magnifier
    {
    public:
        trans_warp_magnifier() : m_xc(0.0), m_yc(0.0), m_magn(1.0), m_radius(1.0), m_warp(false) {}
 
        void center(double x, double y) { m_xc = x; m_yc = y; }
        void magnification(double m)    { m_magn = m;         }
        void radius(double r)           { m_radius = r;       }
        void warp(bool w)               { m_warp = w;         }

        void transform(double* x, double* y) const
        {
            double dx = *x - m_xc;
            double dy = *y - m_yc;
            double r = sqrt(dx * dx + dy * dy);
            double rm = m_radius / m_magn;
            if(r < rm)
            {
                *x = m_xc + dx * m_magn;
                *y = m_yc + dy * m_magn;
                return;
            }

            if(m_warp)
            {
                double m = (r + rm * (m_magn - 1.0)) / r;
                *x = m_xc + dx * m;
                *y = m_yc + dy * m;
                return;
            }

            if(r < m_radius)
            {
                double m = m_radius / r;
                *x = m_xc + dx * m;
                *y = m_yc + dy * m;
            }
        }

    private:
        double m_xc;
        double m_yc;
        double m_magn;
        double m_radius;
        bool   m_warp;
    };



}


#endif

