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

#include "agg_vcgen_bspline.h"

namespace agg
{

    //------------------------------------------------------------------------
    vcgen_bspline::vcgen_bspline() :
        m_src_vertices(),
        m_spline_x(),
        m_spline_y(),
        m_interpolation_step(1.0/50.0),
        m_closed(0),
        m_status(initial),
        m_src_vertex(0)
    {
    }


    //------------------------------------------------------------------------
    void vcgen_bspline::remove_all()
    {
        m_src_vertices.remove_all();
        m_closed = 0;
        m_status = initial;
        m_src_vertex = 0;
    }


    //------------------------------------------------------------------------
    void vcgen_bspline::add_vertex(double x, double y, unsigned cmd)
    {
        m_status = initial;
        if(is_move_to(cmd))
        {
            m_src_vertices.modify_last(point_d(x, y));
        }
        else
        {
            if(is_vertex(cmd))
            {
                m_src_vertices.add(point_d(x, y));
            }
            else
            {
                m_closed = get_close_flag(cmd);
            }
        }
    }


    //------------------------------------------------------------------------
    void vcgen_bspline::rewind(unsigned)
    {
        m_cur_abscissa = 0.0;
        m_max_abscissa = 0.0;
        m_src_vertex = 0;
        if(m_status == initial && m_src_vertices.size() > 2)
        {
            if(m_closed)
            {
                m_spline_x.init(m_src_vertices.size() + 8);
                m_spline_y.init(m_src_vertices.size() + 8);
                m_spline_x.add_point(0.0, m_src_vertices.prev(m_src_vertices.size() - 3).x);
                m_spline_y.add_point(0.0, m_src_vertices.prev(m_src_vertices.size() - 3).y);
                m_spline_x.add_point(1.0, m_src_vertices[m_src_vertices.size() - 3].x);
                m_spline_y.add_point(1.0, m_src_vertices[m_src_vertices.size() - 3].y);
                m_spline_x.add_point(2.0, m_src_vertices[m_src_vertices.size() - 2].x);
                m_spline_y.add_point(2.0, m_src_vertices[m_src_vertices.size() - 2].y);
                m_spline_x.add_point(3.0, m_src_vertices[m_src_vertices.size() - 1].x);
                m_spline_y.add_point(3.0, m_src_vertices[m_src_vertices.size() - 1].y);
            }
            else
            {
                m_spline_x.init(m_src_vertices.size());
                m_spline_y.init(m_src_vertices.size());
            }
            unsigned i;
            for(i = 0; i < m_src_vertices.size(); i++)
            {
                double x = m_closed ? i + 4 : i;
                m_spline_x.add_point(x, m_src_vertices[i].x);
                m_spline_y.add_point(x, m_src_vertices[i].y);
            }
            m_cur_abscissa = 0.0;
            m_max_abscissa = m_src_vertices.size() - 1;
            if(m_closed)
            {
                m_cur_abscissa = 4.0;
                m_max_abscissa += 5.0;
                m_spline_x.add_point(m_src_vertices.size() + 4, m_src_vertices[0].x);
                m_spline_y.add_point(m_src_vertices.size() + 4, m_src_vertices[0].y);
                m_spline_x.add_point(m_src_vertices.size() + 5, m_src_vertices[1].x);
                m_spline_y.add_point(m_src_vertices.size() + 5, m_src_vertices[1].y);
                m_spline_x.add_point(m_src_vertices.size() + 6, m_src_vertices[2].x);
                m_spline_y.add_point(m_src_vertices.size() + 6, m_src_vertices[2].y);
                m_spline_x.add_point(m_src_vertices.size() + 7, m_src_vertices.next(2).x);
                m_spline_y.add_point(m_src_vertices.size() + 7, m_src_vertices.next(2).y);
            }
            m_spline_x.prepare();
            m_spline_y.prepare();
        }
        m_status = ready;
    }






    //------------------------------------------------------------------------
    unsigned vcgen_bspline::vertex(double* x, double* y)
    {
        unsigned cmd = path_cmd_line_to;
        while(!is_stop(cmd))
        {
            switch(m_status)
            {
            case initial:
                rewind(0);

            case ready:
                if(m_src_vertices.size() < 2)
                {
                    cmd = path_cmd_stop;
                    break;
                }

                if(m_src_vertices.size() == 2)
                {
                    *x = m_src_vertices[m_src_vertex].x;
                    *y = m_src_vertices[m_src_vertex].y;
                    m_src_vertex++;
                    if(m_src_vertex == 1) return path_cmd_move_to;
                    if(m_src_vertex == 2) return path_cmd_line_to;
                    cmd = path_cmd_stop;
                    break;
                }

                cmd = path_cmd_move_to;
                m_status = polygon;
                m_src_vertex = 0;

            case polygon:
                if(m_cur_abscissa >= m_max_abscissa)
                {
                    if(m_closed)
                    {
                        m_status = end_poly;
                        break;
                    }
                    else
                    {
                        *x = m_src_vertices[m_src_vertices.size() - 1].x;
                        *y = m_src_vertices[m_src_vertices.size() - 1].y;
                        m_status = end_poly;
                        return path_cmd_line_to;
                    }
                }

                *x = m_spline_x.get_stateful(m_cur_abscissa);
                *y = m_spline_y.get_stateful(m_cur_abscissa);
                m_src_vertex++;
                m_cur_abscissa += m_interpolation_step;
                return (m_src_vertex == 1) ? path_cmd_move_to : path_cmd_line_to;

            case end_poly:
                m_status = stop;
                return path_cmd_end_poly | m_closed;

            case stop:
                return path_cmd_stop;
            }
        }
        return cmd;
    }


}

