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
#ifndef AGG_CONV_BSPLINE_INCLUDED
#define AGG_CONV_BSPLINE_INCLUDED

#include "agg_basics.h"
#include "agg_vcgen_bspline.h"
#include "agg_conv_adaptor_vcgen.h"


namespace agg
{

    //---------------------------------------------------------conv_bspline
    template<class VertexSource> 
    struct conv_bspline : public conv_adaptor_vcgen<VertexSource, vcgen_bspline>
    {
        typedef conv_adaptor_vcgen<VertexSource, vcgen_bspline> base_type;

        conv_bspline(VertexSource& vs) : 
            conv_adaptor_vcgen<VertexSource, vcgen_bspline>(vs) {}

        void   interpolation_step(double v) { base_type::generator().interpolation_step(v); }
        double interpolation_step() const { return base_type::generator().interpolation_step(); }

    private:
        conv_bspline(const conv_bspline<VertexSource>&);
        const conv_bspline<VertexSource>& 
            operator = (const conv_bspline<VertexSource>&);
    };

}


#endif

