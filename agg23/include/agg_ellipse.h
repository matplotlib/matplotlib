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
// class ellipse
//
//----------------------------------------------------------------------------

#ifndef AGG_ELLIPSE_INCLUDED
#define AGG_ELLIPSE_INCLUDED

#include "agg_basics.h"
#include <math.h>

namespace agg
{

    //----------------------------------------------------------------ellipse
    class ellipse
    {
    public:
        ellipse() : m_x(0.0), m_y(0.0), m_rx(1.0), m_ry(1.0), m_num(4), m_step(0), m_cw(false) {}
        ellipse(double x, double y, double rx, double ry, unsigned num_steps, bool cw=false) 
            : m_x(x), m_y(y), m_rx(rx), m_ry(ry), m_num(num_steps), m_step(0), m_cw(cw) {}

        void init(double x, double y, double rx, double ry, unsigned num_steps, 
                  bool cw=false);
        void approximation_scale(double scale);
        void rewind(unsigned path_id);
        unsigned vertex(double* x, double* y);

    private:
        double m_x;
        double m_y;
        double m_rx;
        double m_ry;
        unsigned m_num;
        unsigned m_step;
        bool m_cw;
    };


    //------------------------------------------------------------------------
    inline void ellipse::init(double x, double y, double rx, double ry, 
                              unsigned num_steps, bool cw)
    {
        m_x = x;
        m_y = y;
        m_rx = rx;
        m_ry = ry;
        m_num = num_steps;
        m_step = 0;
        m_cw = cw;
    }

    //------------------------------------------------------------------------
    inline void ellipse::approximation_scale(double scale)
    {   
       m_num = unsigned((fabs(m_rx) + fabs(m_ry) + 6.0) * scale);
       if(m_num < 6) m_num = 6;
    }

    //------------------------------------------------------------------------
    inline void ellipse::rewind(unsigned)
    {
        m_step = 0;
    }

    //------------------------------------------------------------------------
    inline unsigned ellipse::vertex(double* x, double* y)
    {
        if(m_step == m_num) 
        {
            ++m_step;
            return path_cmd_end_poly | path_flags_close | path_flags_ccw;
        }
        if(m_step > m_num) return path_cmd_stop;
        double angle = double(m_step) / double(m_num) * 2.0 * pi;
        if(m_cw) angle = 2.0 * pi - angle;
        *x = m_x + cos(angle) * m_rx;
        *y = m_y + sin(angle) * m_ry;
        m_step++;
        return ((m_step == 1) ? path_cmd_move_to : path_cmd_line_to);
    }

}



#endif


