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
// span_solid_rgba8
//
//----------------------------------------------------------------------------

#ifndef AGG_SPAN_SOLID_INCLUDED
#define AGG_SPAN_SOLID_INCLUDED

#include "agg_basics.h"

namespace agg
{
    //--------------------------------------------------------------span_solid
    template<class ColorT> class span_solid
    {
    public:
        typedef ColorT color_type;

        //--------------------------------------------------------------------
        void color(const color_type& c) { m_color = c; }
        const color_type& color() const { return m_color; }

        //--------------------------------------------------------------------
        void prepare() {}

        //--------------------------------------------------------------------
        void generate(color_type* span, int x, int y, unsigned len)
        {   
            do { *span++ = m_color; } while(--len);
        }

    private:
        color_type m_color;
    };


}

#endif
