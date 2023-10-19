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
        typedef pod_bvector<point_d, 6>         coord_storage;

        vcgen_contour();

        void line_cap(line_cap_e lc)     { m_stroker.line_cap(lc); }
        void line_join(line_join_e lj)   { m_stroker.line_join(lj); }
        void inner_join(inner_join_e ij) { m_stroker.inner_join(ij); }

        line_cap_e   line_cap()   const { return m_stroker.line_cap(); }
        line_join_e  line_join()  const { return m_stroker.line_join(); }
        inner_join_e inner_join() const { return m_stroker.inner_join(); }

        void width(double w) { m_stroker.width(m_width = w); }
        void miter_limit(double ml) { m_stroker.miter_limit(ml); }
        void miter_limit_theta(double t) { m_stroker.miter_limit_theta(t); }
        void inner_miter_limit(double ml) { m_stroker.inner_miter_limit(ml); }
        void approximation_scale(double as) { m_stroker.approximation_scale(as); }

        double width() const { return m_width; }
        double miter_limit() const { return m_stroker.miter_limit(); }
        double inner_miter_limit() const { return m_stroker.inner_miter_limit(); }
        double approximation_scale() const { return m_stroker.approximation_scale(); }

        void auto_detect_orientation(bool v) { m_auto_detect = v; }
        bool auto_detect_orientation() const { return m_auto_detect; }

        // Generator interface
        void remove_all();
        void add_vertex(double x, double y, unsigned cmd);

        // Vertex Source Interface
        void     rewind(unsigned path_id);
        unsigned vertex(double* x, double* y);

    private:
        vcgen_contour(const vcgen_contour&);
        const vcgen_contour& operator = (const vcgen_contour&);

        math_stroke<coord_storage> m_stroker;
        double                     m_width;
        vertex_storage             m_src_vertices;
        coord_storage              m_out_vertices;
        status_e                   m_status;
        unsigned                   m_src_vertex;
        unsigned                   m_out_vertex;
        unsigned                   m_closed;
        unsigned                   m_orientation;
        bool                       m_auto_detect;
    };

}

#endif
