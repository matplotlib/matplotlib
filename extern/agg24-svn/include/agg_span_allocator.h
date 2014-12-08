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

#ifndef AGG_SPAN_ALLOCATOR_INCLUDED
#define AGG_SPAN_ALLOCATOR_INCLUDED

#include "agg_array.h"

namespace agg
{
    //----------------------------------------------------------span_allocator
    template<class ColorT> class span_allocator
    {
    public:
        typedef ColorT color_type;

        //--------------------------------------------------------------------
        AGG_INLINE color_type* allocate(unsigned span_len)
        {
            if(span_len > m_span.size())
            {
                // To reduce the number of reallocs we align the 
                // span_len to 256 color elements. 
                // Well, I just like this number and it looks reasonable.
                //-----------------------
                m_span.resize(((span_len + 255) >> 8) << 8);
            }
            return &m_span[0];
        }

        AGG_INLINE color_type* span()               { return &m_span[0]; }
        AGG_INLINE unsigned    max_span_len() const { return m_span.size(); }

    private:
        pod_array<color_type> m_span;
    };
}


#endif


