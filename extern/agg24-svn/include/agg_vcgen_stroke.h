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
        typedef pod_bvector<point_d, 6>         coord_storage;

        vcgen_stroke();

        void line_cap(line_cap_e lc)     { m_stroker.line_cap(lc); }
        void line_join(line_join_e lj)   { m_stroker.line_join(lj); }
        void inner_join(inner_join_e ij) { m_stroker.inner_join(ij); }

        line_cap_e   line_cap()   const { return m_stroker.line_cap(); }
        line_join_e  line_join()  const { return m_stroker.line_join(); }
        inner_join_e inner_join() const { return m_stroker.inner_join(); }

        void width(double w) { m_stroker.width(w); }
        void miter_limit(double ml) { m_stroker.miter_limit(ml); }
        void miter_limit_theta(double t) { m_stroker.miter_limit_theta(t); }
        void inner_miter_limit(double ml) { m_stroker.inner_miter_limit(ml); }
        void approximation_scale(double as) { m_stroker.approximation_scale(as); }

        double width() const { return m_stroker.width(); }
        double miter_limit() const { return m_stroker.miter_limit(); }
        double inner_miter_limit() const { return m_stroker.inner_miter_limit(); }
        double approximation_scale() const { return m_stroker.approximation_scale(); }

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

        math_stroke<coord_storage> m_stroker;
        vertex_storage             m_src_vertices;
        coord_storage              m_out_vertices;
        double                     m_shorten;
        unsigned                   m_closed;
        status_e                   m_status;
        status_e                   m_prev_status;
        unsigned                   m_src_vertex;
        unsigned                   m_out_vertex;
    };


}

#endif
