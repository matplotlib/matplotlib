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

#ifndef AGG_VCGEN_MARKERS_TERM_INCLUDED
#define AGG_VCGEN_MARKERS_TERM_INCLUDED

#include "agg_basics.h"
#include "agg_vertex_sequence.h"

namespace agg
{

    //======================================================vcgen_markers_term
    //
    // See Implemantation agg_vcgen_markers_term.cpp
    // Terminal markers generator (arrowhead/arrowtail)
    //
    //------------------------------------------------------------------------
    class vcgen_markers_term
    {
    public:
        vcgen_markers_term() : m_curr_id(0), m_curr_idx(0) {}

        // Vertex Generator Interface
        void remove_all();
        void add_vertex(double x, double y, unsigned cmd);

        // Vertex Source Interface
        void rewind(unsigned path_id);
        unsigned vertex(double* x, double* y);

    private:
        vcgen_markers_term(const vcgen_markers_term&);
        const vcgen_markers_term& operator = (const vcgen_markers_term&);

        struct coord_type
        {
            double x, y;

            coord_type() {}
            coord_type(double x_, double y_) : x(x_), y(y_) {}
        };

        typedef pod_bvector<coord_type, 6> coord_storage; 

        coord_storage m_markers;
        unsigned      m_curr_id;
        unsigned      m_curr_idx;
    };


}

#endif
