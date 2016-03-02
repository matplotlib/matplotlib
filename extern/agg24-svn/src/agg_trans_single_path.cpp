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

#include "agg_math.h"
#include "agg_vertex_sequence.h"
#include "agg_trans_single_path.h"

namespace agg
{

    //------------------------------------------------------------------------
    trans_single_path::trans_single_path() :
        m_base_length(0.0),
        m_kindex(0.0),
        m_status(initial),
        m_preserve_x_scale(true)
    {
    }

    //------------------------------------------------------------------------
    void trans_single_path::reset()
    {
        m_src_vertices.remove_all();
        m_kindex = 0.0;
        m_status = initial;
    }

    //------------------------------------------------------------------------
    void trans_single_path::move_to(double x, double y)
    {
        if(m_status == initial)
        {
            m_src_vertices.modify_last(vertex_dist(x, y));
            m_status = making_path;
        }
        else
        {
            line_to(x, y);
        }
    }

    //------------------------------------------------------------------------
    void trans_single_path::line_to(double x, double y)
    {
        if(m_status == making_path)
        {
            m_src_vertices.add(vertex_dist(x, y));
        }
    }


    //------------------------------------------------------------------------
    void trans_single_path::finalize_path()
    {
        if(m_status == making_path && m_src_vertices.size() > 1)
        {
            unsigned i;
            double dist;
            double d;

            m_src_vertices.close(false);
            if(m_src_vertices.size() > 2)
            {
                if(m_src_vertices[m_src_vertices.size() - 2].dist * 10.0 < 
                   m_src_vertices[m_src_vertices.size() - 3].dist)
                {
                    d = m_src_vertices[m_src_vertices.size() - 3].dist + 
                        m_src_vertices[m_src_vertices.size() - 2].dist;

                    m_src_vertices[m_src_vertices.size() - 2] = 
                        m_src_vertices[m_src_vertices.size() - 1];

                    m_src_vertices.remove_last();
                    m_src_vertices[m_src_vertices.size() - 2].dist = d;
                }
            }

            dist = 0.0;
            for(i = 0; i < m_src_vertices.size(); i++)
            {
                vertex_dist& v = m_src_vertices[i];
                double d = v.dist;
                v.dist = dist;
                dist += d;
            }
            m_kindex = (m_src_vertices.size() - 1) / dist;
            m_status = ready;
        }
    }



    //------------------------------------------------------------------------
    double trans_single_path::total_length() const
    {
        if(m_base_length >= 1e-10) return m_base_length;
        return (m_status == ready) ? 
            m_src_vertices[m_src_vertices.size() - 1].dist :
            0.0;
    }


    //------------------------------------------------------------------------
    void trans_single_path::transform(double *x, double *y) const
    {
        if(m_status == ready)
        {
            if(m_base_length > 1e-10)
            {
                *x *= m_src_vertices[m_src_vertices.size() - 1].dist / 
                      m_base_length;
            }

            double x1 = 0.0;
            double y1 = 0.0;
            double dx = 1.0;
            double dy = 1.0;
            double d  = 0.0;
            double dd = 1.0;
            if(*x < 0.0)
            {
                // Extrapolation on the left
                //--------------------------
                x1 = m_src_vertices[0].x;
                y1 = m_src_vertices[0].y;
                dx = m_src_vertices[1].x - x1;
                dy = m_src_vertices[1].y - y1;
                dd = m_src_vertices[1].dist - m_src_vertices[0].dist;
                d  = *x;
            }
            else
            if(*x > m_src_vertices[m_src_vertices.size() - 1].dist)
            {
                // Extrapolation on the right
                //--------------------------
                unsigned i = m_src_vertices.size() - 2;
                unsigned j = m_src_vertices.size() - 1;
                x1 = m_src_vertices[j].x;
                y1 = m_src_vertices[j].y;
                dx = x1 - m_src_vertices[i].x;
                dy = y1 - m_src_vertices[i].y;
                dd = m_src_vertices[j].dist - m_src_vertices[i].dist;
                d  = *x - m_src_vertices[j].dist;
            }
            else
            {
                // Interpolation
                //--------------------------
                unsigned i = 0;
                unsigned j = m_src_vertices.size() - 1;
                if(m_preserve_x_scale)
                {
                    unsigned k;
                    for(i = 0; (j - i) > 1; ) 
                    {
                        if(*x < m_src_vertices[k = (i + j) >> 1].dist) 
                        {
                            j = k; 
                        }
                        else 
                        {
                            i = k;
                        }
                    }
                    d  = m_src_vertices[i].dist;
                    dd = m_src_vertices[j].dist - d;
                    d  = *x - d;
                }
                else
                {
                    i = unsigned(*x * m_kindex);
                    j = i + 1;
                    dd = m_src_vertices[j].dist - m_src_vertices[i].dist;
                    d = ((*x * m_kindex) - i) * dd;
                }
                x1 = m_src_vertices[i].x;
                y1 = m_src_vertices[i].y;
                dx = m_src_vertices[j].x - x1;
                dy = m_src_vertices[j].y - y1;
            }
            double x2 = x1 + dx * d / dd;
            double y2 = y1 + dy * d / dd;
            *x = x2 - *y * dy / dd;
            *y = y2 + *y * dx / dd;
        }
    }


}

