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

#ifndef AGG_CONV_SHORTEN_PATH_INCLUDED
#define AGG_CONV_SHORTEN_PATH_INCLUDED

#include "agg_basics.h"
#include "agg_conv_adaptor_vcgen.h"
#include "agg_vcgen_vertex_sequence.h"

namespace agg
{

    //=======================================================conv_shorten_path
    template<class VertexSource>  class conv_shorten_path : 
    public conv_adaptor_vcgen<VertexSource, vcgen_vertex_sequence>
    {
    public:
        typedef conv_adaptor_vcgen<VertexSource, vcgen_vertex_sequence> base_type;

        conv_shorten_path(VertexSource& vs) : 
            conv_adaptor_vcgen<VertexSource, vcgen_vertex_sequence>(vs)
        {
        }

        void shorten(double s) { base_type::generator().shorten(s); }
        double shorten() const { return base_type::generator().shorten(); }

    private:
        conv_shorten_path(const conv_shorten_path<VertexSource>&);
        const conv_shorten_path<VertexSource>& 
            operator = (const conv_shorten_path<VertexSource>&);
    };


}

#endif
