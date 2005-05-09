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

#ifndef AGG_SPAN_CONVERTER_INCLUDED
#define AGG_SPAN_CONVERTER_INCLUDED

#include "agg_basics.h"

namespace agg
{
    //----------------------------------------------------------span_converter
    template<class SpanGenerator, class Conv> class span_converter
    {
    public:
        typedef typename SpanGenerator::color_type color_type;

        span_converter(SpanGenerator& span_gen, Conv& conv) : 
            m_span_gen(&span_gen), m_conv(&conv) {}

        //--------------------------------------------------------------------
        void prepare(unsigned max_span_len) 
        {
            m_span_gen->prepare(max_span_len);
        }

        //--------------------------------------------------------------------
        color_type* generate(int x, int y, unsigned len)
        {
            color_type* span = m_span_gen->generate(x, y, len);
            m_conv->convert(span, x, y, len);
            return span;
        }

    private:
        SpanGenerator* m_span_gen;
        Conv*          m_conv;
    };

}

#endif
