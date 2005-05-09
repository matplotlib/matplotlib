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

#include <math.h>
#include "agg_vpgen_clip_polyline.h"

namespace agg
{
    static double clip_epsilon = 1e-10;


    //----------------------------------------------------------------------------
    void vpgen_clip_polyline::reset()
    {
        m_vertex = 0;
        m_num_vertices = 0;
    }

    //----------------------------------------------------------------------------
    void vpgen_clip_polyline::move_to(double x, double y)
    {
        m_vertex = 0;
        m_num_vertices = 0;
        m_f1 = clipping_flags(x, y);
        if(m_f1 == 0)
        {
            m_x[0] = x;
            m_y[0] = y;
            m_cmd[0] = path_cmd_move_to;
            m_num_vertices = 1;
        }
        m_x1 = x;
        m_y1 = y;
    }


    //----------------------------------------------------------------------------
    bool vpgen_clip_polyline::move_point(double& x, double& y, unsigned& flags)
    {
       double bound;

       if(flags & (clip_x1 | clip_x2)) 
       {
           bound = (flags & clip_x1) ? m_clip_box.x1 : m_clip_box.x2;
           y = (bound - m_x1) * (m_y2 - m_y1) / (m_x2 - m_x1) + m_y1;
           x = bound;
           flags = clipping_flags_y(y);
       }
       if(fabs(m_y2 - m_y1) < clip_epsilon && fabs(m_x2 - m_x1) < clip_epsilon) 
       {
           return false;
       }
       if(flags & (clip_y1 | clip_y2)) 
       {
           bound = (flags & clip_y1) ? m_clip_box.y1 : m_clip_box.y2;
           x = (bound - m_y1) * (m_x2 - m_x1) / (m_y2 - m_y1) + m_x1;
           y = bound;
       }
       flags = 0;
       return true;
    }

    //----------------------------------------------------------------------------
    void vpgen_clip_polyline::clip_line_segment()
    {
        if((m_f1 & m_f2) == 0)
        {
            if(m_f1) 
            {   
                if(!move_point(m_x1, m_y1, m_f1)) return;
                if(m_f1) return;
                m_x[0] = m_x1;
                m_y[0] = m_y1;
                m_cmd[0] = path_cmd_move_to;
                m_num_vertices = 1;
            }
            if(m_f2) 
            {                    // Move Point 2
                if(!move_point(m_x2, m_y2, m_f2)) return;
            }
            m_x[m_num_vertices] = m_x2;
            m_y[m_num_vertices] = m_y2;
            m_cmd[m_num_vertices++] = path_cmd_line_to;
        }
    }



    //----------------------------------------------------------------------------
    void vpgen_clip_polyline::line_to(double x, double y)
    {
        m_vertex = 0;
        m_num_vertices = 0;
        unsigned f = m_f2 = clipping_flags(m_x2 = x, m_y2 = y);

        if(m_f2 == m_f1)
        {
            if(m_f2 == 0)
            {
                m_x[0] = x;
                m_y[0] = y;
                m_cmd[0] = path_cmd_line_to;
                m_num_vertices = 1;
            }
        }
        else
        {
            clip_line_segment();
        }

        m_f1 = f;
        m_x1 = x;
        m_y1 = y;
    }


    //----------------------------------------------------------------------------
    unsigned vpgen_clip_polyline::vertex(double* x, double* y)
    {
        if(m_vertex < m_num_vertices)
        {
            *x = m_x[m_vertex];
            *y = m_y[m_vertex];
            return m_cmd[m_vertex++];
        }
        return path_cmd_stop;
    }


}
