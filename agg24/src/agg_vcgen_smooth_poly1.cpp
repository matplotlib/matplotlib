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
// Smooth polygon generator
//
//----------------------------------------------------------------------------

#include "agg_vcgen_smooth_poly1.h"

namespace agg
{

    //------------------------------------------------------------------------
    vcgen_smooth_poly1::vcgen_smooth_poly1() :
        m_src_vertices(),
        m_smooth_value(0.5),
        m_closed(0),
        m_status(initial),
        m_src_vertex(0)
    {
    }


    //------------------------------------------------------------------------
    void vcgen_smooth_poly1::remove_all()
    {
        m_src_vertices.remove_all();
        m_closed = 0;
        m_status = initial;
    }


    //------------------------------------------------------------------------
    void vcgen_smooth_poly1::add_vertex(double x, double y, unsigned cmd)
    {
        m_status = initial;
        if(is_move_to(cmd))
        {
            m_src_vertices.modify_last(vertex_dist(x, y));
        }
        else
        {
            if(is_vertex(cmd))
            {
                m_src_vertices.add(vertex_dist(x, y));
            }
            else
            {
                m_closed = get_close_flag(cmd);
            }
        }
    }


    //------------------------------------------------------------------------
    void vcgen_smooth_poly1::rewind(unsigned)
    {
        if(m_status == initial)
        {
            m_src_vertices.close(m_closed != 0);
        }
        m_status = ready;
        m_src_vertex = 0;
    }


    //------------------------------------------------------------------------
    void vcgen_smooth_poly1::calculate(const vertex_dist& v0, 
                                       const vertex_dist& v1, 
                                       const vertex_dist& v2,
                                       const vertex_dist& v3)
    {

        double k1 = v0.dist / (v0.dist + v1.dist);
        double k2 = v1.dist / (v1.dist + v2.dist);

        double xm1 = v0.x + (v2.x - v0.x) * k1;
        double ym1 = v0.y + (v2.y - v0.y) * k1;
        double xm2 = v1.x + (v3.x - v1.x) * k2;
        double ym2 = v1.y + (v3.y - v1.y) * k2;

        m_ctrl1_x = v1.x + m_smooth_value * (v2.x - xm1);
        m_ctrl1_y = v1.y + m_smooth_value * (v2.y - ym1);
        m_ctrl2_x = v2.x + m_smooth_value * (v1.x - xm2);
        m_ctrl2_y = v2.y + m_smooth_value * (v1.y - ym2);
    }


    //------------------------------------------------------------------------
    unsigned vcgen_smooth_poly1::vertex(double* x, double* y)
    {
        unsigned cmd = path_cmd_line_to;
        while(!is_stop(cmd))
        {
            switch(m_status)
            {
            case initial:
                rewind(0);

            case ready:
                if(m_src_vertices.size() <  2)
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
                if(m_closed)
                {
                    if(m_src_vertex >= m_src_vertices.size())
                    {
                        *x = m_src_vertices[0].x;
                        *y = m_src_vertices[0].y;
                        m_status = end_poly;
                        return path_cmd_curve4;
                    }
                }
                else
                {
                    if(m_src_vertex >= m_src_vertices.size() - 1)
                    {
                        *x = m_src_vertices[m_src_vertices.size() - 1].x;
                        *y = m_src_vertices[m_src_vertices.size() - 1].y;
                        m_status = end_poly;
                        return path_cmd_curve3;
                    }
                }

                calculate(m_src_vertices.prev(m_src_vertex), 
                          m_src_vertices.curr(m_src_vertex), 
                          m_src_vertices.next(m_src_vertex),
                          m_src_vertices.next(m_src_vertex + 1));

                *x = m_src_vertices[m_src_vertex].x;
                *y = m_src_vertices[m_src_vertex].y;
                m_src_vertex++;

                if(m_closed)
                {
                    m_status = ctrl1;
                    return ((m_src_vertex == 1) ? 
                             path_cmd_move_to : 
                             path_cmd_curve4);
                }
                else
                {
                    if(m_src_vertex == 1)
                    {
                        m_status = ctrl_b;
                        return path_cmd_move_to;
                    }
                    if(m_src_vertex >= m_src_vertices.size() - 1)
                    {
                        m_status = ctrl_e;
                        return path_cmd_curve3;
                    }
                    m_status = ctrl1;
                    return path_cmd_curve4;
                }
                break;

            case ctrl_b:
                *x = m_ctrl2_x;
                *y = m_ctrl2_y;
                m_status = polygon;
                return path_cmd_curve3;

            case ctrl_e:
                *x = m_ctrl1_x;
                *y = m_ctrl1_y;
                m_status = polygon;
                return path_cmd_curve3;

            case ctrl1:
                *x = m_ctrl1_x;
                *y = m_ctrl1_y;
                m_status = ctrl2;
                return path_cmd_curve4;

            case ctrl2:
                *x = m_ctrl2_x;
                *y = m_ctrl2_y;
                m_status = polygon;
                return path_cmd_curve4;

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

