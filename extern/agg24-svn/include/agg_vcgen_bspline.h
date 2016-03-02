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

#ifndef AGG_VCGEN_BSPLINE_INCLUDED
#define AGG_VCGEN_BSPLINE_INCLUDED

#include "agg_basics.h"
#include "agg_array.h"
#include "agg_bspline.h"


namespace agg
{

    //==========================================================vcgen_bspline
    class vcgen_bspline
    {
        enum status_e
        {
            initial,
            ready,
            polygon,
            end_poly,
            stop
        };

    public:
        typedef pod_bvector<point_d, 6> vertex_storage;

        vcgen_bspline();

        void interpolation_step(double v) { m_interpolation_step = v; }
        double interpolation_step() const { return m_interpolation_step; }

        // Vertex Generator Interface
        void remove_all();
        void add_vertex(double x, double y, unsigned cmd);

        // Vertex Source Interface
        void     rewind(unsigned path_id);
        unsigned vertex(double* x, double* y);

    private:
        vcgen_bspline(const vcgen_bspline&);
        const vcgen_bspline& operator = (const vcgen_bspline&);

        vertex_storage m_src_vertices;
        bspline        m_spline_x;
        bspline        m_spline_y;
        double         m_interpolation_step;
        unsigned       m_closed;
        status_e       m_status;
        unsigned       m_src_vertex;
        double         m_cur_abscissa;
        double         m_max_abscissa;
    };

}


#endif

