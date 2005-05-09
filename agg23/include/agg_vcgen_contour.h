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

#ifndef AGG_VCGEN_CONTOUR_INCLUDED
#define AGG_VCGEN_CONTOUR_INCLUDED

#include "agg_math_stroke.h"

namespace agg
{

    //----------------------------------------------------------vcgen_contour
    //
    // See Implementation agg_vcgen_contour.cpp
    //
    class vcgen_contour
    {
        enum status_e
        {
            initial,
            ready,
            outline,
            out_vertices,
            end_poly,
            stop
        };

    public:
        typedef vertex_sequence<vertex_dist, 6> vertex_storage;
        typedef pod_deque<point_type, 6>        coord_storage;

        vcgen_contour();

        void line_join(line_join_e lj) { m_line_join = lj; }
        void inner_line_join(line_join_e lj) { m_inner_line_join = lj; }
        void width(double w) { m_width = w * 0.5; }
        void miter_limit(double ml) { m_miter_limit = ml; }
        void miter_limit_theta(double t);
        void inner_miter_limit(double ml) { m_inner_miter_limit = ml; }
        void approximation_scale(double as) { m_approx_scale = as; }
        void auto_detect_orientation(bool v) { m_auto_detect = v; }

        line_join_e line_join() const { return m_line_join; }
        line_join_e inner_line_join() const { return m_inner_line_join; }
        double width() const { return m_width * 2.0; }
        double miter_limit() const { return m_miter_limit; }
        double inner_miter_limit() const { return m_inner_miter_limit; }
        double approximation_scale() const { return m_approx_scale; }
        bool   auto_detect_orientation() const { return m_auto_detect; }

        // Generator interface
        void remove_all();
        void add_vertex(double x, double y, unsigned cmd);

        // Vertex Source Interface
        void     rewind(unsigned path_id);
        unsigned vertex(double* x, double* y);

    private:
        vcgen_contour(const vcgen_contour&);
        const vcgen_contour& operator = (const vcgen_contour&);

        vertex_storage m_src_vertices;
        coord_storage  m_out_vertices;
        double         m_width;
        line_join_e    m_line_join;
        line_join_e    m_inner_line_join;
        double         m_approx_scale;
        double         m_abs_width;
        double         m_signed_width;
        double         m_miter_limit;
        double         m_inner_miter_limit;
        status_e       m_status;
        unsigned       m_src_vertex;
        unsigned       m_out_vertex;
        unsigned       m_closed;
        unsigned       m_orientation;
        bool           m_auto_detect;
    };

}

#endif
