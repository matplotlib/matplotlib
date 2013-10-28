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
// Class scanline_bin - binary scanline.
//
//----------------------------------------------------------------------------
//
// Adaptation for 32-bit screen coordinates (scanline32_bin) has been sponsored by 
// Liberty Technology Systems, Inc., visit http://lib-sys.com
//
// Liberty Technology Systems, Inc. is the provider of
// PostScript and PDF technology for software developers.
// 
//----------------------------------------------------------------------------

#ifndef AGG_SCANLINE_BIN_INCLUDED
#define AGG_SCANLINE_BIN_INCLUDED

#include "agg_array.h"

namespace agg
{

    //=============================================================scanline_bin
    // 
    // This is binary scaline container which supports the interface 
    // used in the rasterizer::render(). See description of agg_scanline_u8 
    // for details.
    // 
    //------------------------------------------------------------------------
    class scanline_bin
    {
    public:
        typedef int32 coord_type;

        struct span
        {
            int16 x;
            int16 len;
        };

        typedef const span* const_iterator;

        //--------------------------------------------------------------------
        scanline_bin() :
            m_last_x(0x7FFFFFF0),
            m_spans(),
            m_cur_span(0)
        {
        }

        //--------------------------------------------------------------------
        void reset(int min_x, int max_x)
        {
            unsigned max_len = max_x - min_x + 3;
            if(max_len > m_spans.size())
            {
                m_spans.resize(max_len);
            }
            m_last_x   = 0x7FFFFFF0;
            m_cur_span = &m_spans[0];
        }

        //--------------------------------------------------------------------
        void add_cell(int x, unsigned)
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

        //--------------------------------------------------------------------
        void add_span(int x, unsigned len, unsigned)
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

        //--------------------------------------------------------------------
        void add_cells(int x, unsigned len, const void*)
        {
            add_span(x, len, 0);
        }

        //--------------------------------------------------------------------
        void finalize(int y) 
        { 
            m_y = y; 
        }

        //--------------------------------------------------------------------
        void reset_spans()
        {
            m_last_x    = 0x7FFFFFF0;
            m_cur_span  = &m_spans[0];
        }

        //--------------------------------------------------------------------
        int            y()         const { return m_y; }
        unsigned       num_spans() const { return unsigned(m_cur_span - &m_spans[0]); }
        const_iterator begin()     const { return &m_spans[1]; }

    private:
        scanline_bin(const scanline_bin&);
        const scanline_bin operator = (const scanline_bin&);

        int             m_last_x;
        int             m_y;
        pod_array<span> m_spans;
        span*           m_cur_span;
    };






    //===========================================================scanline32_bin
    class scanline32_bin
    {
    public:
        typedef int32 coord_type;

        //--------------------------------------------------------------------
        struct span
        {
            span() {}
            span(coord_type x_, coord_type len_) : x(x_), len(len_) {}

            coord_type x;
            coord_type len;
        };
        typedef pod_bvector<span, 4> span_array_type;


        //--------------------------------------------------------------------
        class const_iterator
        {
        public:
            const_iterator(const span_array_type& spans) :
                m_spans(spans),
                m_span_idx(0)
            {}

            const span& operator*()  const { return m_spans[m_span_idx];  }
            const span* operator->() const { return &m_spans[m_span_idx]; }

            void operator ++ () { ++m_span_idx; }

        private:
            const span_array_type& m_spans;
            unsigned               m_span_idx;
        };


        //--------------------------------------------------------------------
        scanline32_bin() : m_max_len(0), m_last_x(0x7FFFFFF0) {}

        //--------------------------------------------------------------------
        void reset(int min_x, int max_x)
        {
            m_last_x = 0x7FFFFFF0;
            m_spans.remove_all();
        }

        //--------------------------------------------------------------------
        void add_cell(int x, unsigned)
        {
            if(x == m_last_x+1)
            {
                m_spans.last().len++;
            }
            else
            {
                m_spans.add(span(coord_type(x), 1));
            }
            m_last_x = x;
        }

        //--------------------------------------------------------------------
        void add_span(int x, unsigned len, unsigned)
        {
            if(x == m_last_x+1)
            {
                m_spans.last().len += coord_type(len);
            }
            else
            {
                m_spans.add(span(coord_type(x), coord_type(len)));
            }
            m_last_x = x + len - 1;
        }

        //--------------------------------------------------------------------
        void add_cells(int x, unsigned len, const void*)
        {
            add_span(x, len, 0);
        }

        //--------------------------------------------------------------------
        void finalize(int y) 
        { 
            m_y = y; 
        }

        //--------------------------------------------------------------------
        void reset_spans()
        {
            m_last_x = 0x7FFFFFF0;
            m_spans.remove_all();
        }

        //--------------------------------------------------------------------
        int            y()         const { return m_y; }
        unsigned       num_spans() const { return m_spans.size(); }
        const_iterator begin()     const { return const_iterator(m_spans); }

    private:
        scanline32_bin(const scanline32_bin&);
        const scanline32_bin operator = (const scanline32_bin&);

        unsigned        m_max_len;
        int             m_last_x;
        int             m_y;
        span_array_type m_spans;
    };





}


#endif
