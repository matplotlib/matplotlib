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

#ifndef AGG_SPAN_ALLOCATOR_INCLUDED
#define AGG_SPAN_ALLOCATOR_INCLUDED

#include "agg_basics.h"

namespace agg
{
    //----------------------------------------------------------span_allocator
    template<class ColorT> class span_allocator
    {
    public:
        typedef ColorT color_type;

        //--------------------------------------------------------------------
        ~span_allocator()
        {
            delete [] m_span;
        }

        //--------------------------------------------------------------------
        span_allocator() :
            m_max_span_len(0),
            m_span(0)
        {
        }

        //--------------------------------------------------------------------
        color_type* allocate(unsigned max_span_len)
        {
            if(max_span_len > m_max_span_len)
            {
                delete [] m_span;
                m_span = new color_type[m_max_span_len = max_span_len];
            }
            return m_span;
        }

        //--------------------------------------------------------------------
        color_type* span()
        { 
            return m_span; 
        }

    private:
        //--------------------------------------------------------------------
        span_allocator(const span_allocator<ColorT>&);
        const span_allocator<ColorT>& operator = (const span_allocator<ColorT>&);

        unsigned    m_max_span_len;
        color_type* m_span;
    };
}


#endif


