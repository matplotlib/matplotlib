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
// classes polygon_ctrl_impl
//
//----------------------------------------------------------------------------

#include "ctrl/agg_polygon_ctrl.h"

namespace agg
{

    polygon_ctrl_impl::polygon_ctrl_impl(unsigned np, double point_radius) :
        ctrl(0, 0, 1, 1, false),
        m_polygon(np * 2),
        m_num_points(np),
        m_node(-1),
        m_edge(-1),
        m_vs(&m_polygon[0], m_num_points, false),
        m_stroke(m_vs),
        m_point_radius(point_radius),
        m_status(0),
        m_dx(0.0),
        m_dy(0.0),
        m_in_polygon_check(true)
    {
        m_stroke.width(1.0);
    }


    void polygon_ctrl_impl::rewind(unsigned)
    {
        m_status = 0;
        m_stroke.rewind(0);
    }

    unsigned polygon_ctrl_impl::vertex(double* x, double* y)
    {
        unsigned cmd = path_cmd_stop;
        double r = m_point_radius;
        if(m_status == 0)
        {
            cmd = m_stroke.vertex(x, y);
            if(!is_stop(cmd)) 
            {
                transform_xy(x, y);
                return cmd;
            }
            if(m_node >= 0 && m_node == int(m_status)) r *= 1.2;
            m_ellipse.init(xn(m_status), yn(m_status), r, r, 32);
            ++m_status;
        }
        cmd = m_ellipse.vertex(x, y);
        if(!is_stop(cmd)) 
        {
            transform_xy(x, y);
            return cmd;
        }
        if(m_status >= m_num_points) return path_cmd_stop;
        if(m_node >= 0 && m_node == int(m_status)) r *= 1.2;
        m_ellipse.init(xn(m_status), yn(m_status), r, r, 32);
        ++m_status;
        cmd = m_ellipse.vertex(x, y);
        if(!is_stop(cmd)) 
        {
            transform_xy(x, y);
        }
        return cmd;
    }


    bool polygon_ctrl_impl::check_edge(unsigned i, double x, double y) const
    {
       bool ret = false;

       unsigned n1 = i;
       unsigned n2 = (i + m_num_points - 1) % m_num_points;
       double x1 = xn(n1);
       double y1 = yn(n1);
       double x2 = xn(n2);
       double y2 = yn(n2);

       double dx = x2 - x1;
       double dy = y2 - y1;

       if(sqrt(dx*dx + dy*dy) > 0.0000001)
       {
          double x3 = x;
          double y3 = y;
          double x4 = x3 - dy;
          double y4 = y3 + dx;

          double den = (y4-y3) * (x2-x1) - (x4-x3) * (y2-y1);
          double u1 = ((x4-x3) * (y1-y3) - (y4-y3) * (x1-x3)) / den;

          double xi = x1 + u1 * (x2 - x1);
          double yi = y1 + u1 * (y2 - y1);

          dx = xi - x;
          dy = yi - y;

          if (u1 > 0.0 && u1 < 1.0 && sqrt(dx*dx + dy*dy) <= m_point_radius)
          {
             ret = true;
          }
       }
       return ret;
    }



    bool polygon_ctrl_impl::in_rect(double x, double y) const
    {
        return false;
    }


    bool polygon_ctrl_impl::on_mouse_button_down(double x, double y)
    {
        unsigned i;
        bool ret = false;
        m_node = -1;
        m_edge = -1;
        inverse_transform_xy(&x, &y);
        for (i = 0; i < m_num_points; i++)
        {
            if(sqrt( (x-xn(i)) * (x-xn(i)) + (y-yn(i)) * (y-yn(i)) ) < m_point_radius)
            {
                m_dx = x - xn(i);
                m_dy = y - yn(i);
                m_node = int(i);
                ret = true;
                break;
            }
        }

        if(!ret)
        {
            for (i = 0; i < m_num_points; i++)
            {
                if(check_edge(i, x, y))
                {
                    m_dx = x;
                    m_dy = y;
                    m_edge = int(i);
                    ret = true;
                    break;
                }
            }
        }

        if(!ret)
        {
            if(point_in_polygon(x, y))
            {
                m_dx = x;
                m_dy = y;
                m_node = int(m_num_points);
                ret = true;
            }
        }
        return ret;
    }


    bool polygon_ctrl_impl::on_mouse_move(double x, double y, bool button_flag)
    {
        bool ret = false;
        double dx;
        double dy;
        inverse_transform_xy(&x, &y);
        if(m_node == int(m_num_points))
        {
            dx = x - m_dx;
            dy = y - m_dy;
            unsigned i;
            for(i = 0; i < m_num_points; i++)
            {
                xn(i) += dx;
                yn(i) += dy;
            }
            m_dx = x;
            m_dy = y;
            ret = true;
        }
        else
        {
            if(m_edge >= 0)
            {
                unsigned n1 = m_edge;
                unsigned n2 = (n1 + m_num_points - 1) % m_num_points;
                dx = x - m_dx;
                dy = y - m_dy;
                xn(n1) += dx;
                yn(n1) += dy;
                xn(n2) += dx;
                yn(n2) += dy;
                m_dx = x;
                m_dy = y;
                ret = true;
            }
            else
            {
                if(m_node >= 0)
                {
                    xn(m_node) = x - m_dx;
                    yn(m_node) = y - m_dy;
                    ret = true;
                }
            }
        }
        return ret;
    }

    bool polygon_ctrl_impl::on_mouse_button_up(double x, double y)
    {
        bool ret = (m_node >= 0) || (m_edge >= 0);
        m_node = -1;
        m_edge = -1;
        return ret;
    }


    bool polygon_ctrl_impl::on_arrow_keys(bool left, bool right, bool down, bool up)
    {
        return false;
    }


    //======= Crossings Multiply algorithm of InsideTest ======================== 
    //
    // By Eric Haines, 3D/Eye Inc, erich@eye.com
    //
    // This version is usually somewhat faster than the original published in
    // Graphics Gems IV; by turning the division for testing the X axis crossing
    // into a tricky multiplication test this part of the test became faster,
    // which had the additional effect of making the test for "both to left or
    // both to right" a bit slower for triangles than simply computing the
    // intersection each time.  The main increase is in triangle testing speed,
    // which was about 15% faster; all other polygon complexities were pretty much
    // the same as before.  On machines where division is very expensive (not the
    // case on the HP 9000 series on which I tested) this test should be much
    // faster overall than the old code.  Your mileage may (in fact, will) vary,
    // depending on the machine and the test data, but in general I believe this
    // code is both shorter and faster.  This test was inspired by unpublished
    // Graphics Gems submitted by Joseph Samosky and Mark Haigh-Hutchinson.
    // Related work by Samosky is in:
    //
    // Samosky, Joseph, "SectionView: A system for interactively specifying and
    // visualizing sections through three-dimensional medical image data",
    // M.S. Thesis, Department of Electrical Engineering and Computer Science,
    // Massachusetts Institute of Technology, 1993.
    //
    // Shoot a test ray along +X axis.  The strategy is to compare vertex Y values
    // to the testing point's Y and quickly discard edges which are entirely to one
    // side of the test ray.  Note that CONVEX and WINDING code can be added as
    // for the CrossingsTest() code; it is left out here for clarity.
    //
    // Input 2D polygon _pgon_ with _numverts_ number of vertices and test point
    // _point_, returns 1 if inside, 0 if outside.
    bool polygon_ctrl_impl::point_in_polygon(double tx, double ty) const
    {
        if(m_num_points < 3) return false;
        if(!m_in_polygon_check) return false;

        unsigned j;
        int yflag0, yflag1, inside_flag;
        double  vtx0, vty0, vtx1, vty1;

        vtx0 = xn(m_num_points - 1);
        vty0 = yn(m_num_points - 1);

        // get test bit for above/below X axis
        yflag0 = (vty0 >= ty);

        vtx1 = xn(0);
        vty1 = yn(0);

        inside_flag = 0;
        for (j = 1; j <= m_num_points; ++j) 
        {
            yflag1 = (vty1 >= ty);
            // Check if endpoints straddle (are on opposite sides) of X axis
            // (i.e. the Y's differ); if so, +X ray could intersect this edge.
            // The old test also checked whether the endpoints are both to the
            // right or to the left of the test point.  However, given the faster
            // intersection point computation used below, this test was found to
            // be a break-even proposition for most polygons and a loser for
            // triangles (where 50% or more of the edges which survive this test
            // will cross quadrants and so have to have the X intersection computed
            // anyway).  I credit Joseph Samosky with inspiring me to try dropping
            // the "both left or both right" part of my code.
            if (yflag0 != yflag1) 
            {
                // Check intersection of pgon segment with +X ray.
                // Note if >= point's X; if so, the ray hits it.
                // The division operation is avoided for the ">=" test by checking
                // the sign of the first vertex wrto the test point; idea inspired
                // by Joseph Samosky's and Mark Haigh-Hutchinson's different
                // polygon inclusion tests.
                if ( ((vty1-ty) * (vtx0-vtx1) >=
                      (vtx1-tx) * (vty0-vty1)) == yflag1 ) 
                {
                    inside_flag ^= 1;
                }
            }

            // Move to the next pair of vertices, retaining info as possible.
            yflag0 = yflag1;
            vtx0 = vtx1;
            vty0 = vty1;

            unsigned k = (j >= m_num_points) ? j - m_num_points : j;
            vtx1 = xn(k);
            vty1 = yn(k);
        }
        return inside_flag != 0;
    }
}

