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
//
// Polygon clipping converter
// There an optimized Liang-Basky algorithm is used. 
// The algorithm doesn't optimize the degenerate edges, i.e. it will never
// break a closed polygon into two or more ones, instead, there will be 
// degenerate edges coinciding with the respective clipping boundaries.
// This is a sub-optimal solution, because that optimization would require 
// extra, rather expensive math while the rasterizer tolerates it quite well, 
// without any considerable overhead.
//
//----------------------------------------------------------------------------
#ifndef AGG_CONV_CLIP_POLYGON_INCLUDED
#define AGG_CONV_CLIP_POLYGON_INCLUDED

#include "agg_basics.h"
#include "agg_conv_adaptor_vpgen.h"
#include "agg_vpgen_clip_polygon.h"

namespace agg
{

    //=======================================================conv_clip_polygon
    template<class VertexSource> 
    struct conv_clip_polygon : public conv_adaptor_vpgen<VertexSource, vpgen_clip_polygon>
    {
        typedef conv_adaptor_vpgen<VertexSource, vpgen_clip_polygon> base_type;

        conv_clip_polygon(VertexSource& vs) : 
            conv_adaptor_vpgen<VertexSource, vpgen_clip_polygon>(vs) {}

        void clip_box(double x1, double y1, double x2, double y2)
        {
            base_type::vpgen().clip_box(x1, y1, x2, y2);
        }

        double x1() const { return base_type::vpgen().x1(); }
        double y1() const { return base_type::vpgen().y1(); }
        double x2() const { return base_type::vpgen().x2(); }
        double y2() const { return base_type::vpgen().y2(); }

    private:
        conv_clip_polygon(const conv_clip_polygon<VertexSource>&);
        const conv_clip_polygon<VertexSource>& 
            operator = (const conv_clip_polygon<VertexSource>&);
    };

}

#endif
