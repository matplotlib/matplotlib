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

#ifndef AGG_CONV_UNCLOSE_POLYGON_INCLUDED
#define AGG_CONV_UNCLOSE_POLYGON_INCLUDED

#include "agg_basics.h"

namespace agg
{
    //====================================================conv_unclose_polygon
    template<class VertexSource> class conv_unclose_polygon
    {
    public:
        explicit conv_unclose_polygon(VertexSource& vs) : m_source(&vs) {}
        void attach(VertexSource& source) { m_source = &source; }

        void rewind(unsigned path_id)
        {
            m_source->rewind(path_id);
        }

        unsigned vertex(double* x, double* y)
        {
            unsigned cmd = m_source->vertex(x, y);
            if(is_end_poly(cmd)) cmd &= ~path_flags_close;
            return cmd;
        }

    private:
        conv_unclose_polygon(const conv_unclose_polygon<VertexSource>&);
        const conv_unclose_polygon<VertexSource>& 
            operator = (const conv_unclose_polygon<VertexSource>&);

        VertexSource* m_source;
    };

}

#endif
