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

#ifndef AGG_WARP_MAGNIFIER_INCLUDED
#define AGG_WARP_MAGNIFIER_INCLUDED


namespace agg
{

    //----------------------------------------------------trans_warp_magnifier
    //
    // See Inmplementation agg_trans_warp_magnifier.cpp
    //
    class trans_warp_magnifier
    {
    public:
        trans_warp_magnifier() : m_xc(0.0), m_yc(0.0), m_magn(1.0), m_radius(1.0) {}
 
        void center(double x, double y) { m_xc = x; m_yc = y; }
        void magnification(double m)    { m_magn = m;         }
        void radius(double r)           { m_radius = r;       }

        double xc()            const { return m_xc; }
        double yc()            const { return m_yc; }
        double magnification() const { return m_magn;   }
        double radius()        const { return m_radius; }

        void transform(double* x, double* y) const;
        void inverse_transform(double* x, double* y) const;

    private:
        double m_xc;
        double m_yc;
        double m_magn;
        double m_radius;
    };


}


#endif

