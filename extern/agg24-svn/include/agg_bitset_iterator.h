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

#ifndef AGG_BITSET_ITERATOR_INCLUDED
#define AGG_BITSET_ITERATOR_INCLUDED

#include "agg_basics.h"

namespace agg
{
    
    class bitset_iterator
    {
    public:
        bitset_iterator(const int8u* bits, unsigned offset = 0) :
            m_bits(bits + (offset >> 3)),
            m_mask(0x80 >> (offset & 7))
        {}

        void operator ++ ()
        {
            m_mask >>= 1;
            if(m_mask == 0)
            {
                ++m_bits;
                m_mask = 0x80;
            }
        }

        unsigned bit() const
        {
            return (*m_bits) & m_mask;
        }

    private:
        const int8u* m_bits;
        int8u        m_mask;
    };

}

#endif
