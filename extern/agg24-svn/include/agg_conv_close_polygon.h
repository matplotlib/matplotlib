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

#ifndef AGG_CONV_CLOSE_POLYGON_INCLUDED
#define AGG_CONV_CLOSE_POLYGON_INCLUDED

#include "agg_basics.h"

namespace agg
{

    //======================================================conv_close_polygon
    template<class VertexSource> class conv_close_polygon
    {
    public:
        explicit conv_close_polygon(VertexSource& vs) : m_source(&vs) {}
        void attach(VertexSource& source) { m_source = &source; }

        void rewind(unsigned path_id);
        unsigned vertex(double* x, double* y);

    private:
        conv_close_polygon(const conv_close_polygon<VertexSource>&);
        const conv_close_polygon<VertexSource>& 
            operator = (const conv_close_polygon<VertexSource>&);

        VertexSource* m_source;
        unsigned      m_cmd[2];
        double        m_x[2];
        double        m_y[2];
        unsigned      m_vertex;
        bool          m_line_to;
    };



    //------------------------------------------------------------------------
    template<class VertexSource> 
    void conv_close_polygon<VertexSource>::rewind(unsigned path_id)
    {
        m_source->rewind(path_id);
        m_vertex = 2;
        m_line_to = false;
    }


    
    //------------------------------------------------------------------------
    template<class VertexSource> 
    unsigned conv_close_polygon<VertexSource>::vertex(double* x, double* y)
    {
        unsigned cmd = path_cmd_stop;
        for(;;)
        {
            if(m_vertex < 2)
            {
                *x = m_x[m_vertex];
                *y = m_y[m_vertex];
                cmd = m_cmd[m_vertex];
                ++m_vertex;
                break;
            }

            cmd = m_source->vertex(x, y);

            if(is_end_poly(cmd))
            {
                cmd |= path_flags_close;
                break;
            }

            if(is_stop(cmd))
            {
                if(m_line_to)
                {
                    m_cmd[0]  = path_cmd_end_poly | path_flags_close;
                    m_cmd[1]  = path_cmd_stop;
                    m_vertex  = 0;
                    m_line_to = false;
                    continue;
                }
                break;
            }

            if(is_move_to(cmd))
            {
                if(m_line_to)
                {
                    m_x[0]    = 0.0;
                    m_y[0]    = 0.0;
                    m_cmd[0]  = path_cmd_end_poly | path_flags_close;
                    m_x[1]    = *x;
                    m_y[1]    = *y;
                    m_cmd[1]  = cmd;
                    m_vertex  = 0;
                    m_line_to = false;
                    continue;
                }
                break;
            }

            if(is_vertex(cmd))
            {
                m_line_to = true;
                break;
            }
        }
        return cmd;
    }

}

#endif
