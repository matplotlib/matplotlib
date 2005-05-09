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
//
// span_solid_rgba8
//
//----------------------------------------------------------------------------

#ifndef AGG_SPAN_SOLID_INCLUDED
#define AGG_SPAN_SOLID_INCLUDED

#include "agg_basics.h"
#include "agg_span_generator.h"

namespace agg
{
    //--------------------------------------------------------------span_solid
    template<class ColorT, class Allocator = span_allocator<ColorT> >
    class span_solid : public span_generator<ColorT, Allocator>
    {
    public:
        typedef Allocator alloc_type;
        typedef ColorT color_type;
        typedef span_generator<color_type, alloc_type> base_type;

        //--------------------------------------------------------------------
        span_solid(alloc_type& alloc) : base_type(alloc) {}

        //--------------------------------------------------------------------
        void color(const color_type& c) { m_color = c; }
        const color_type& color() const { return m_color; }

        //--------------------------------------------------------------------
        color_type* generate(int x, int y, unsigned len)
        {   
            color_type* span = base_type::allocator().span();
            do
            {
                *span++ = m_color;
            }
            while(--len);
            return base_type::allocator().span();
        }

    private:
        color_type m_color;
    };


}

#endif
