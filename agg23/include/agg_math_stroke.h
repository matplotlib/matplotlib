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
// Stroke math
//
//----------------------------------------------------------------------------

#ifndef AGG_STROKE_MATH_INCLUDED
#define AGG_STROKE_MATH_INCLUDED

#include "agg_math.h"
#include "agg_vertex_sequence.h"

namespace agg
{
    //-------------------------------------------------------------line_cap_e
    enum line_cap_e
    {
        butt_cap,
        square_cap,
        round_cap
    };

    //------------------------------------------------------------line_join_e
    enum line_join_e
    {
        miter_join,
        miter_join_revert,
        round_join,
        bevel_join
    };

    // Minimal angle to calculate round joins, less than 0.1 degree.
    const double stroke_theta = 0.001;       //----stroke_theta

    //--------------------------------------------------------stroke_calc_arc
    template<class VertexConsumer>
    void stroke_calc_arc(VertexConsumer& out_vertices,
                         double x,   double y, 
                         double dx1, double dy1, 
                         double dx2, double dy2,
                         double width,
                         double approximation_scale)
    {
        typedef typename VertexConsumer::value_type coord_type;

        //// Check if we actually need the arc (this optimization works bad)
        ////-----------------
        //double dd = calc_distance(dx1, dy1, dx2, dy2);
        //if(dd < 1.0/approximation_scale)
        //{
        //    out_vertices.add(coord_type(x + dx1, y + dy1));
        //    if(dd > 0.25/approximation_scale)
        //    {
        //        out_vertices.add(coord_type(x + dx2, y + dy2));
        //    }
        //    return;
        //}

        double a1 = atan2(dy1, dx1);
        double a2 = atan2(dy2, dx2);
        double da = a1 - a2;

        if(fabs(da) < stroke_theta)
        {
            out_vertices.add(coord_type((x + x + dx1 + dx2) * 0.5, 
                                        (y + y + dy1 + dy2) * 0.5));
            return;
        }

        bool ccw = da > 0.0 && da < pi;

        if(width < 0) width = -width;
        da = fabs(1.0 / (width * approximation_scale));
        if(!ccw)
        {
            if(a1 > a2) a2 += 2 * pi;
            while(a1 < a2)
            {
                out_vertices.add(coord_type(x + cos(a1) * width, y + sin(a1) * width));
                a1 += da;
            }
        }
        else
        {
            if(a1 < a2) a2 -= 2 * pi;
            while(a1 > a2)
            {
                out_vertices.add(coord_type(x + cos(a1) * width, y + sin(a1) * width));
                a1 -= da;
            }
        }
        out_vertices.add(coord_type(x + dx2, y + dy2));
    }



    //-------------------------------------------------------stroke_calc_miter
    template<class VertexConsumer>
    void stroke_calc_miter(VertexConsumer& out_vertices,
                           const vertex_dist& v0, 
                           const vertex_dist& v1, 
                           const vertex_dist& v2,
                           double dx1, double dy1, 
                           double dx2, double dy2,
                           double width,
                           bool revert_flag,
                           double miter_limit)
    {
        typedef typename VertexConsumer::value_type coord_type;

        double xi = v1.x;
        double yi = v1.y;

        if(calc_intersection(v0.x + dx1, v0.y - dy1,
                             v1.x + dx1, v1.y - dy1,
                             v1.x + dx2, v1.y - dy2,
                             v2.x + dx2, v2.y - dy2,
                             &xi, &yi))
        {
            // Calculation of the intersection succeeded
            //---------------------
            double d1 = calc_distance(v1.x, v1.y, xi, yi);
            double lim = width * miter_limit;
            if(d1 <= lim)
            {
                // Inside the miter limit
                //---------------------
                out_vertices.add(coord_type(xi, yi));
            }
            else
            {
                // Miter limit exceeded
                //------------------------
                if(revert_flag || d1 < intersection_epsilon)
                {
                    // For the compatibility with SVG, PDF, etc, 
                    // we use a simple bevel join instead of
                    // "smart" bevel
                    //-------------------
                    out_vertices.add(coord_type(v1.x + dx1, v1.y - dy1));
                    out_vertices.add(coord_type(v1.x + dx2, v1.y - dy2));
                }
                else
                {
                    // Smart bevel that cuts the miter at the limit point
                    //-------------------
                    d1  = lim / d1;
                    double x1 = v1.x + dx1;
                    double y1 = v1.y - dy1;
                    double x2 = v1.x + dx2;
                    double y2 = v1.y - dy2;

                    x1 += (xi - x1) * d1;
                    y1 += (yi - y1) * d1;
                    x2 += (xi - x2) * d1;
                    y2 += (yi - y2) * d1;
                    out_vertices.add(coord_type(x1, y1));
                    out_vertices.add(coord_type(x2, y2));
                }
            }
        }
        else
        {
            // Calculation of the intersection failed, most probaly
            // the three points lie one straight line. 
            // First check if v0 and v2 lie on the opposite sides of vector: 
            // (v1.x, v1.y) -> (v1.x+dx1, v1.y-dy1), that is, the perpendicular
            // to the line determined by vertices v0 and v1.
            // This condition deternines whether the next line segments continues
            // the previous one or goes back.
            //----------------
            double x2 = v1.x + dx1;
            double y2 = v1.y - dy1;
            if(((x2 - v0.x)*dy1 - (v0.y - y2)*dx1 < 0.0) !=
               ((x2 - v2.x)*dy1 - (v2.y - y2)*dx1 < 0.0))
            {
                // This case means that the next segment continues 
                // the previous one (straight line)
                //-----------------
                out_vertices.add(coord_type(v1.x + dx1, v1.y - dy1));
            }
            else
            {
                // This case means that the next segment goes back  
                //-----------------
                if(revert_flag)
                {
                    out_vertices.add(coord_type(v1.x + dx1, v1.y - dy1));
                    out_vertices.add(coord_type(v1.x + dx2, v1.y - dy2));
                }
                else
                {
                    // If no miter-revert, calcuate new dx1, dy1, dx2, dy2
                    out_vertices.add(coord_type(v1.x + dx1 + dy1 * miter_limit, 
                                                v1.y - dy1 + dx1 * miter_limit));
                    out_vertices.add(coord_type(v1.x + dx2 - dy2 * miter_limit, 
                                                v1.y - dy2 - dx2 * miter_limit));
                }
            }
        }
    }






    //--------------------------------------------------------stroke_calc_cap
    template<class VertexConsumer>
    void stroke_calc_cap(VertexConsumer& out_vertices,
                         const vertex_dist& v0, 
                         const vertex_dist& v1, 
                         double len,
                         line_cap_e line_cap,
                         double width,
                         double approximation_scale)
    {
        typedef typename VertexConsumer::value_type coord_type;

        out_vertices.remove_all();

        double dx1 = (v1.y - v0.y) / len;
        double dy1 = (v1.x - v0.x) / len;
        double dx2 = 0;
        double dy2 = 0;

        dx1 *= width;
        dy1 *= width;

        if(line_cap != round_cap)
        {
            if(line_cap == square_cap)
            {
                dx2 = dy1;
                dy2 = dx1;
            }
            out_vertices.add(coord_type(v0.x - dx1 - dx2, v0.y + dy1 - dy2));
            out_vertices.add(coord_type(v0.x + dx1 - dx2, v0.y - dy1 - dy2));
        }
        else
        {
            double a1 = atan2(dy1, -dx1);
            double a2 = a1 + pi;
            double da = fabs(1.0 / (width * approximation_scale));
            while(a1 < a2)
            {
                out_vertices.add(coord_type(v0.x + cos(a1) * width, 
                                            v0.y + sin(a1) * width));
                a1 += da;
            }
            out_vertices.add(coord_type(v0.x + dx1, v0.y - dy1));
        }
    }



    //-------------------------------------------------------stroke_calc_join
    template<class VertexConsumer>
    void stroke_calc_join(VertexConsumer& out_vertices,
                          const vertex_dist& v0, 
                          const vertex_dist& v1, 
                          const vertex_dist& v2,
                          double len1, 
                          double len2,
                          double width, 
                          line_join_e line_join,
                          line_join_e inner_line_join,
                          double miter_limit,
                          double inner_miter_limit,
                          double approximation_scale)
    {
        typedef typename VertexConsumer::value_type coord_type;

        double dx1, dy1, dx2, dy2;

        dx1 = width * (v1.y - v0.y) / len1;
        dy1 = width * (v1.x - v0.x) / len1;

        dx2 = width * (v2.y - v1.y) / len2;
        dy2 = width * (v2.x - v1.x) / len2;

        out_vertices.remove_all();

        if(calc_point_location(v0.x, v0.y, v1.x, v1.y, v2.x, v2.y) > 0)
        {
            // Inner join
            //---------------
            stroke_calc_miter(out_vertices,
                              v0, v1, v2, dx1, dy1, dx2, dy2, 
                              width,                                   
                              inner_line_join == miter_join_revert, 
                              inner_miter_limit);
        }
        else
        {
            // Outer join
            //---------------
            switch(line_join)
            {
            case miter_join:
                stroke_calc_miter(out_vertices, 
                                  v0, v1, v2, dx1, dy1, dx2, dy2, 
                                  width,                                   
                                  false, 
                                  miter_limit);
                break;

            case miter_join_revert:
                stroke_calc_miter(out_vertices, 
                                  v0, v1, v2, dx1, dy1, dx2, dy2, 
                                  width,                                   
                                  true, 
                                  miter_limit);
                break;

            case round_join:
                stroke_calc_arc(out_vertices, 
                                v1.x, v1.y, dx1, -dy1, dx2, -dy2, 
                                width, approximation_scale);
                break;

            default: // Bevel join
                out_vertices.add(coord_type(v1.x + dx1, v1.y - dy1));
                if(calc_distance(dx1, dy1, dx2, dy2) > approximation_scale * 0.25)
                {
                    out_vertices.add(coord_type(v1.x + dx2, v1.y - dy2));
                }
                break;
            }
        }
    }




}

#endif
