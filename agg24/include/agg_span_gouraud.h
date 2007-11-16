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

#ifndef AGG_SPAN_GOURAUD_INCLUDED
#define AGG_SPAN_GOURAUD_INCLUDED

#include "agg_basics.h"
#include "agg_math.h"

namespace agg
{

    //============================================================span_gouraud
    template<class ColorT> class span_gouraud
    {
    public:
        typedef ColorT color_type;

        struct coord_type
        {
            double x;
            double y;
            color_type color;
        };

        //--------------------------------------------------------------------
        span_gouraud() : 
            m_vertex(0)
        {
            m_cmd[0] = path_cmd_stop;
        }

        //--------------------------------------------------------------------
        span_gouraud(const color_type& c1,
                     const color_type& c2,
                     const color_type& c3,
                     double x1, double y1,
                     double x2, double y2,
                     double x3, double y3,
                     double d) : 
            m_vertex(0)
        {
            colors(c1, c2, c3);
            triangle(x1, y1, x2, y2, x3, y3, d);
        }

        //--------------------------------------------------------------------
        void colors(ColorT c1, ColorT c2, ColorT c3)
        {
            m_coord[0].color = c1;
            m_coord[1].color = c2;
            m_coord[2].color = c3;
        }

        //--------------------------------------------------------------------
        // Sets the triangle and dilates it if needed.
        // The trick here is to calculate beveled joins in the vertices of the 
        // triangle and render it as a 6-vertex polygon. 
        // It's necessary to achieve numerical stability. 
        // However, the coordinates to interpolate colors are calculated
        // as miter joins (calc_intersection).
        void triangle(double x1, double y1, 
                      double x2, double y2,
                      double x3, double y3,
                      double d)
        {
            m_coord[0].x = m_x[0] = x1; 
            m_coord[0].y = m_y[0] = y1;
            m_coord[1].x = m_x[1] = x2; 
            m_coord[1].y = m_y[1] = y2;
            m_coord[2].x = m_x[2] = x3; 
            m_coord[2].y = m_y[2] = y3;
            m_cmd[0] = path_cmd_move_to;
            m_cmd[1] = path_cmd_line_to;
            m_cmd[2] = path_cmd_line_to;
            m_cmd[3] = path_cmd_stop;

            if(d != 0.0)
            {   
                dilate_triangle(m_coord[0].x, m_coord[0].y,
                                m_coord[1].x, m_coord[1].y,
                                m_coord[2].x, m_coord[2].y,
                                m_x, m_y, d);

                calc_intersection(m_x[4], m_y[4], m_x[5], m_y[5],
                                  m_x[0], m_y[0], m_x[1], m_y[1],
                                  &m_coord[0].x, &m_coord[0].y);

                calc_intersection(m_x[0], m_y[0], m_x[1], m_y[1],
                                  m_x[2], m_y[2], m_x[3], m_y[3],
                                  &m_coord[1].x, &m_coord[1].y);

                calc_intersection(m_x[2], m_y[2], m_x[3], m_y[3],
                                  m_x[4], m_y[4], m_x[5], m_y[5],
                                  &m_coord[2].x, &m_coord[2].y);
                m_cmd[3] = path_cmd_line_to;
                m_cmd[4] = path_cmd_line_to;
                m_cmd[5] = path_cmd_line_to;
                m_cmd[6] = path_cmd_stop;
            }
        }

        //--------------------------------------------------------------------
        // Vertex Source Interface to feed the coordinates to the rasterizer
        void rewind(unsigned)
        {
            m_vertex = 0;
        }

        //--------------------------------------------------------------------
        unsigned vertex(double* x, double* y)
        {
            *x = m_x[m_vertex];
            *y = m_y[m_vertex];
            return m_cmd[m_vertex++];
        }

    protected:
        //--------------------------------------------------------------------
        void arrange_vertices(coord_type* coord) const
        {
            coord[0] = m_coord[0];
            coord[1] = m_coord[1];
            coord[2] = m_coord[2];

            if(m_coord[0].y > m_coord[2].y)
            {
                coord[0] = m_coord[2]; 
                coord[2] = m_coord[0];
            }

            coord_type tmp;
            if(coord[0].y > coord[1].y)
            {
                tmp      = coord[1];
                coord[1] = coord[0];
                coord[0] = tmp;
            }

            if(coord[1].y > coord[2].y)
            {
                tmp      = coord[2];
                coord[2] = coord[1];
                coord[1] = tmp;
            }
       }

    private:
        //--------------------------------------------------------------------
        coord_type m_coord[3];
        double m_x[8];
        double m_y[8];
        unsigned m_cmd[8];
        unsigned m_vertex;
    };

}

#endif

