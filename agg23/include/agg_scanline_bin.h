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
// Class scanline_bin - binary scanline.
//
//----------------------------------------------------------------------------
#ifndef AGG_SCANLINE_BIN_INCLUDED
#define AGG_SCANLINE_BIN_INCLUDED

#include "agg_basics.h"

namespace agg
{

    //=============================================================scanline_bin
    // 
    // This is binary scaline container which supports the interface 
    // used in the rasterizer::render(). See description of agg_scanline_u8 
    // for details.
    // 
    // Rendering:
    //-------------------------------------------------------------------------
    //  
    //  int y = sl.y();
    //  
    // ************************************
    // ...Perform vertical clipping here...
    // ************************************
    //
    //  unsigned num_spans = sl.num_spans();
    //  const agg::scanline_bin::span* cur_span = sl.spans();
    //
    //  do
    //  {
    //      x = cur_span->x;
    //      len = cur_span->len;
    // 
    //      **************************************
    //      ...Perform horizontal clipping here...
    //      **************************************
    //      
    //      hor_line(x, y, len)
    //      ++cur_span;
    //  }
    //  while(--num_spans);
    // 
    //------------------------------------------------------------------------
    class scanline_bin
    {
    public:
        struct span
        {
            int16 x;
            int16 len;
        };

        typedef const span* const_iterator;

        ~scanline_bin()
        {
            delete [] m_spans;
        }

        scanline_bin() :
            m_max_len(0),
            m_last_x(0x7FFF),
            m_spans(0),
            m_cur_span(0)
        {
        }

        void reset(int min_x, int max_x);
        void add_cell(int x, unsigned);
        void add_cells(int x, unsigned len, const void*);
        void add_span(int x, unsigned len, unsigned);
        void finalize(int y) { m_y = y; }
        void reset_spans();

        int            y()         const { return m_y; }
        unsigned       num_spans() const { return unsigned(m_cur_span - m_spans); }
        const_iterator begin()     const { return m_spans + 1; }

    private:
        scanline_bin(const scanline_bin&);
        const scanline_bin operator = (const scanline_bin&);

        unsigned  m_max_len;
        int       m_last_x;
        int       m_y;
        span*     m_spans;
        span*     m_cur_span;
    };


    //------------------------------------------------------------------------
    inline void scanline_bin::reset(int min_x, int max_x)
    {
        unsigned max_len = max_x - min_x + 3;
        if(max_len > m_max_len)
        {
            delete [] m_spans;
            m_spans   = new span [max_len];
            m_max_len = max_len;
        }
        m_last_x    = 0x7FFF;
        m_cur_span  = m_spans;
    }


    //------------------------------------------------------------------------
    inline void scanline_bin::reset_spans()
    {
        m_last_x    = 0x7FFF;
        m_cur_span  = m_spans;
    }


    //------------------------------------------------------------------------
    inline void scanline_bin::add_cell(int x, unsigned)
    {
        if(x == m_last_x+1)
        {
            m_cur_span->len++;
        }
        else
        {
            ++m_cur_span;
            m_cur_span->x = (int16)x;
            m_cur_span->len = 1;
        }
        m_last_x = x;
    }


    //------------------------------------------------------------------------
    inline void scanline_bin::add_span(int x, unsigned len, unsigned)
    {
        if(x == m_last_x+1)
        {
            m_cur_span->len = (int16)(m_cur_span->len + len);
        }
        else
        {
            ++m_cur_span;
            m_cur_span->x = (int16)x;
            m_cur_span->len = (int16)len;
        }
        m_last_x = x + len - 1;
    }

    //------------------------------------------------------------------------
    inline void scanline_bin::add_cells(int x, unsigned len, const void*)
    {
        add_span(x, len, 0);
    }
}


#endif
