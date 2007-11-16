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
#include "agg_trans_double_path.h"

namespace agg
{

    //------------------------------------------------------------------------
    trans_double_path::trans_double_path() :
        m_kindex1(0.0),
        m_kindex2(0.0),
        m_base_length(0.0),
        m_base_height(1.0),
        m_status1(initial),
        m_status2(initial),
        m_preserve_x_scale(true)
    {
    }


        //------------------------------------------------------------------------
    void trans_double_path::reset()
    {
        m_src_vertices1.remove_all();
        m_src_vertices2.remove_all();
        m_kindex1 = 0.0;
        m_kindex1 = 0.0;
        m_status1 = initial;
        m_status2 = initial;
    }


    //------------------------------------------------------------------------
    void trans_double_path::move_to1(double x, double y)
    {
        if(m_status1 == initial)
        {
            m_src_vertices1.modify_last(vertex_dist(x, y));
            m_status1 = making_path;
        }
        else
        {
            line_to1(x, y);
        }
    }


    //------------------------------------------------------------------------
    void trans_double_path::line_to1(double x, double y)
    {
        if(m_status1 == making_path)
        {
            m_src_vertices1.add(vertex_dist(x, y));
        }
    }


    //------------------------------------------------------------------------
    void trans_double_path::move_to2(double x, double y)
    {
        if(m_status2 == initial)
        {
            m_src_vertices2.modify_last(vertex_dist(x, y));
            m_status2 = making_path;
        }
        else
        {
            line_to2(x, y);
        }
    }


    //------------------------------------------------------------------------
    void trans_double_path::line_to2(double x, double y)
    {
        if(m_status2 == making_path)
        {
            m_src_vertices2.add(vertex_dist(x, y));
        }
    }


    //------------------------------------------------------------------------
    double trans_double_path::finalize_path(vertex_storage& vertices)
    {
        unsigned i;
        double dist;
        double d;

        vertices.close(false);
        if(vertices.size() > 2)
        {
            if(vertices[vertices.size() - 2].dist * 10.0 < 
               vertices[vertices.size() - 3].dist)
            {
                d = vertices[vertices.size() - 3].dist + 
                    vertices[vertices.size() - 2].dist;

                vertices[vertices.size() - 2] = 
                    vertices[vertices.size() - 1];

                vertices.remove_last();
                vertices[vertices.size() - 2].dist = d;
            }
        }

        dist = 0;
        for(i = 0; i < vertices.size(); i++)
        {
            vertex_dist& v = vertices[i];
            d = v.dist;
            v.dist = dist;
            dist += d;
        }

        return (vertices.size() - 1) / dist;
    }


    //------------------------------------------------------------------------
    void trans_double_path::finalize_paths()
    {
        if(m_status1 == making_path && m_src_vertices1.size() > 1 &&
           m_status2 == making_path && m_src_vertices2.size() > 1)
        {
            m_kindex1 = finalize_path(m_src_vertices1);
            m_kindex2 = finalize_path(m_src_vertices2);
            m_status1 = ready;
            m_status2 = ready;
        }
    }


    //------------------------------------------------------------------------
    double trans_double_path::total_length1() const
    {
        if(m_base_length >= 1e-10) return m_base_length;
        return (m_status1 == ready) ? 
            m_src_vertices1[m_src_vertices1.size() - 1].dist :
            0.0;
    }


    //------------------------------------------------------------------------
    double trans_double_path::total_length2() const
    {
        if(m_base_length >= 1e-10) return m_base_length;
        return (m_status2 == ready) ? 
            m_src_vertices2[m_src_vertices2.size() - 1].dist :
            0.0;
    }


    //------------------------------------------------------------------------
    void trans_double_path::transform1(const vertex_storage& vertices, 
                                       double kindex, double kx, 
                                       double *x, double* y) const
    {
        double x1 = 0.0;
        double y1 = 0.0;
        double dx = 1.0;
        double dy = 1.0;
        double d  = 0.0;
        double dd = 1.0;
        *x *= kx;
        if(*x < 0.0)
        {
            // Extrapolation on the left
            //--------------------------
            x1 = vertices[0].x;
            y1 = vertices[0].y;
            dx = vertices[1].x - x1;
            dy = vertices[1].y - y1;
            dd = vertices[1].dist - vertices[0].dist;
            d  = *x;
        }
        else
        if(*x > vertices[vertices.size() - 1].dist)
        {
            // Extrapolation on the right
            //--------------------------
            unsigned i = vertices.size() - 2;
            unsigned j = vertices.size() - 1;
            x1 = vertices[j].x;
            y1 = vertices[j].y;
            dx = x1 - vertices[i].x;
            dy = y1 - vertices[i].y;
            dd = vertices[j].dist - vertices[i].dist;
            d  = *x - vertices[j].dist;
        }
        else
        {
            // Interpolation
            //--------------------------
            unsigned i = 0;
            unsigned j = vertices.size() - 1;
            if(m_preserve_x_scale)
            {
                unsigned k;
                for(i = 0; (j - i) > 1; ) 
                {
                    if(*x < vertices[k = (i + j) >> 1].dist) 
                    {
                        j = k; 
                    }
                    else 
                    {
                        i = k;
                    }
                }
                d  = vertices[i].dist;
                dd = vertices[j].dist - d;
                d  = *x - d;
            }
            else
            {
                i = unsigned(*x * kindex);
                j = i + 1;
                dd = vertices[j].dist - vertices[i].dist;
                d = ((*x * kindex) - i) * dd;
            }
            x1 = vertices[i].x;
            y1 = vertices[i].y;
            dx = vertices[j].x - x1;
            dy = vertices[j].y - y1;
        }
        *x = x1 + dx * d / dd;
        *y = y1 + dy * d / dd;
    }


    //------------------------------------------------------------------------
    void trans_double_path::transform(double *x, double *y) const
    {
        if(m_status1 == ready && m_status2 == ready)
        {
            if(m_base_length > 1e-10)
            {
                *x *= m_src_vertices1[m_src_vertices1.size() - 1].dist / 
                      m_base_length;
            }

            double x1 = *x;
            double y1 = *y;
            double x2 = *x;
            double y2 = *y;
            double dd = m_src_vertices2[m_src_vertices2.size() - 1].dist /
                        m_src_vertices1[m_src_vertices1.size() - 1].dist;

            transform1(m_src_vertices1, m_kindex1, 1.0, &x1, &y1);
            transform1(m_src_vertices2, m_kindex2, dd,  &x2, &y2);

            *x = x1 + *y * (x2 - x1) / m_base_height;
            *y = y1 + *y * (y2 - y1) / m_base_height;
        }
    }

}

