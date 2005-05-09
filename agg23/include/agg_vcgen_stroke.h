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

#ifndef AGG_VCGEN_STROKE_INCLUDED
#define AGG_VCGEN_STROKE_INCLUDED

#include "agg_math_stroke.h"


namespace agg
{

    //============================================================vcgen_stroke
    //
    // See Implementation agg_vcgen_stroke.cpp
    // Stroke generator
    //
    //------------------------------------------------------------------------
    class vcgen_stroke
    {
        enum status_e
        {
            initial,
            ready,
            cap1,
            cap2,
            outline1,
            close_first,
            outline2,
            out_vertices,
            end_poly1,
            end_poly2,
            stop
        };

    public:
        typedef vertex_sequence<vertex_dist, 6> vertex_storage;
        typedef pod_deque<point_type, 6>        coord_storage;

        vcgen_stroke();

        void line_cap(line_cap_e lc) { m_line_cap = lc; }
        void line_join(line_join_e lj) { m_line_join = lj; }
        void inner_line_join(line_join_e lj) { m_inner_line_join = lj; }

        line_cap_e line_cap() const { return m_line_cap; }
        line_join_e line_join() const { return m_line_join; }
        line_join_e inner_line_join() const { return m_inner_line_join; }

        void width(double w) { m_width = w * 0.5; }
        void miter_limit(double ml) { m_miter_limit = ml; }
        void miter_limit_theta(double t);
        void inner_miter_limit(double ml) { m_inner_miter_limit = ml; }
        void approximation_scale(double as) { m_approx_scale = as; }

        double width() const { return m_width * 2.0; }
        double miter_limit() const { return m_miter_limit; }
        double inner_miter_limit() const { return m_inner_miter_limit; }
        double approximation_scale() const { return m_approx_scale; }

        void shorten(double s) { m_shorten = s; }
        double shorten() const { return m_shorten; }

        // Vertex Generator Interface
        void remove_all();
        void add_vertex(double x, double y, unsigned cmd);

        // Vertex Source Interface
        void     rewind(unsigned path_id);
        unsigned vertex(double* x, double* y);

    private:
        vcgen_stroke(const vcgen_stroke&);
        const vcgen_stroke& operator = (const vcgen_stroke&);

        vertex_storage m_src_vertices;
        coord_storage  m_out_vertices;
        double         m_width;
        double         m_miter_limit;
        double         m_inner_miter_limit;
        double         m_approx_scale;
        double         m_shorten;
        line_cap_e     m_line_cap;
        line_join_e    m_line_join;
        line_join_e    m_inner_line_join;
        unsigned       m_closed;
        status_e       m_status;
        status_e       m_prev_status;
        unsigned       m_src_vertex;
        unsigned       m_out_vertex;
    };


}

#endif
