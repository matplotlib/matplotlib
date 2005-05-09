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

#ifndef AGG_SPAN_GENERATOR_INCLUDED
#define AGG_SPAN_GENERATOR_INCLUDED

#include "agg_basics.h"
#include "agg_span_allocator.h"

namespace agg
{

    //==========================================================span_generator
    template<class ColorT, class Allocator> class span_generator
    {
    public:
        typedef ColorT color_type;
        typedef Allocator alloc_type;

        //--------------------------------------------------------------------
        span_generator(alloc_type& alloc) : m_alloc(&alloc) {}

        //--------------------------------------------------------------------
        void allocator(alloc_type& alloc) { m_alloc = &alloc; }
        alloc_type& allocator() { return *m_alloc; }

        //--------------------------------------------------------------------
        void prepare(unsigned max_span_len) 
        {
            m_alloc->allocate(max_span_len);
        }

    private:
        alloc_type* m_alloc;
    };
}

#endif
