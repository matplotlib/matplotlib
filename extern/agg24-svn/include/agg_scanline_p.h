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
// Class scanline_p - a general purpose scanline container with packed spans.
//
//----------------------------------------------------------------------------
//
// Adaptation for 32-bit screen coordinates (scanline32_p) has been sponsored by 
// Liberty Technology Systems, Inc., visit http://lib-sys.com
//
// Liberty Technology Systems, Inc. is the provider of
// PostScript and PDF technology for software developers.
// 
//----------------------------------------------------------------------------
#ifndef AGG_SCANLINE_P_INCLUDED
#define AGG_SCANLINE_P_INCLUDED

#include "agg_array.h"

namespace agg
{

    //=============================================================scanline_p8
    // 
    // This is a general purpose scaline container which supports the interface 
    // used in the rasterizer::render(). See description of scanline_u8
    // for details.
    // 
    //------------------------------------------------------------------------
    class scanline_p8
    {
    public:
        typedef scanline_p8 self_type;
        typedef int8u       cover_type;
        typedef int16       coord_type;

        //--------------------------------------------------------------------
        struct span
        {
            coord_type        x;
            coord_type        len; // If negative, it's a solid span, covers is valid
            const cover_type* covers;
        };

        typedef span* iterator;
        typedef const span* const_iterator;

        scanline_p8() :
            m_last_x(0x7FFFFFF0),
            m_covers(),
            m_cover_ptr(0),
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
                m_covers.resize(max_len);
            }
            m_last_x    = 0x7FFFFFF0;
            m_cover_ptr = &m_covers[0];
            m_cur_span  = &m_spans[0];
            m_cur_span->len = 0;
        }

        //--------------------------------------------------------------------
        void add_cell(int x, unsigned cover)
        {
            *m_cover_ptr = (cover_type)cover;
            if(x == m_last_x+1 && m_cur_span->len > 0)
            {
                m_cur_span->len++;
            }
            else
            {
                m_cur_span++;
                m_cur_span->covers = m_cover_ptr;
                m_cur_span->x = (int16)x;
                m_cur_span->len = 1;
            }
            m_last_x = x;
            m_cover_ptr++;
        }

        //--------------------------------------------------------------------
        void add_cells(int x, unsigned len, const cover_type* covers)
        {
            memcpy(m_cover_ptr, covers, len * sizeof(cover_type));
            if(x == m_last_x+1 && m_cur_span->len > 0)
            {
                m_cur_span->len += (int16)len;
            }
            else
            {
                m_cur_span++;
                m_cur_span->covers = m_cover_ptr;
                m_cur_span->x = (int16)x;
                m_cur_span->len = (int16)len;
            }
            m_cover_ptr += len;
            m_last_x = x + len - 1;
        }

        //--------------------------------------------------------------------
        void add_span(int x, unsigned len, unsigned cover)
        {
            if(x == m_last_x+1 && 
               m_cur_span->len < 0 && 
               cover == *m_cur_span->covers)
            {
                m_cur_span->len -= (int16)len;
            }
            else
            {
                *m_cover_ptr = (cover_type)cover;
                m_cur_span++;
                m_cur_span->covers = m_cover_ptr++;
                m_cur_span->x      = (int16)x;
                m_cur_span->len    = (int16)(-int(len));
            }
            m_last_x = x + len - 1;
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
            m_cover_ptr = &m_covers[0];
            m_cur_span  = &m_spans[0];
            m_cur_span->len = 0;
        }

        //--------------------------------------------------------------------
        int            y()         const { return m_y; }
        unsigned       num_spans() const { return unsigned(m_cur_span - &m_spans[0]); }
        const_iterator begin()     const { return &m_spans[1]; }

    private:
        scanline_p8(const self_type&);
        const self_type& operator = (const self_type&);

        int                   m_last_x;
        int                   m_y;
        pod_array<cover_type> m_covers;
        cover_type*           m_cover_ptr;
        pod_array<span>       m_spans;
        span*                 m_cur_span;
    };








    //==========================================================scanline32_p8
    class scanline32_p8
    {
    public:
        typedef scanline32_p8 self_type;
        typedef int8u         cover_type;
        typedef int32         coord_type;

        struct span
        {
            span() {}
            span(coord_type x_, coord_type len_, const cover_type* covers_) :
                x(x_), len(len_), covers(covers_) {}

            coord_type x;
            coord_type len; // If negative, it's a solid span, covers is valid
            const cover_type* covers;
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
        scanline32_p8() :
            m_max_len(0),
            m_last_x(0x7FFFFFF0),
            m_covers(),
            m_cover_ptr(0)
        {
        }

        //--------------------------------------------------------------------
        void reset(int min_x, int max_x)
        {
            unsigned max_len = max_x - min_x + 3;
            if(max_len > m_covers.size())
            {
                m_covers.resize(max_len);
            }
            m_last_x    = 0x7FFFFFF0;
            m_cover_ptr = &m_covers[0];
            m_spans.remove_all();
        }

        //--------------------------------------------------------------------
        void add_cell(int x, unsigned cover)
        {
            *m_cover_ptr = cover_type(cover);
            if(x == m_last_x+1 && m_spans.size() && m_spans.last().len > 0)
            {
                m_spans.last().len++;
            }
            else
            {
                m_spans.add(span(coord_type(x), 1, m_cover_ptr));
            }
            m_last_x = x;
            m_cover_ptr++;
        }

        //--------------------------------------------------------------------
        void add_cells(int x, unsigned len, const cover_type* covers)
        {
            memcpy(m_cover_ptr, covers, len * sizeof(cover_type));
            if(x == m_last_x+1 && m_spans.size() && m_spans.last().len > 0)
            {
                m_spans.last().len += coord_type(len);
            }
            else
            {
                m_spans.add(span(coord_type(x), coord_type(len), m_cover_ptr));
            }
            m_cover_ptr += len;
            m_last_x = x + len - 1;
        }

        //--------------------------------------------------------------------
        void add_span(int x, unsigned len, unsigned cover)
        {
            if(x == m_last_x+1 && 
               m_spans.size() &&
               m_spans.last().len < 0 && 
               cover == *m_spans.last().covers)
            {
                m_spans.last().len -= coord_type(len);
            }
            else
            {
                *m_cover_ptr = cover_type(cover);
                m_spans.add(span(coord_type(x), -coord_type(len), m_cover_ptr++));
            }
            m_last_x = x + len - 1;
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
            m_cover_ptr = &m_covers[0];
            m_spans.remove_all();
        }

        //--------------------------------------------------------------------
        int            y()         const { return m_y; }
        unsigned       num_spans() const { return m_spans.size(); }
        const_iterator begin()     const { return const_iterator(m_spans); }

    private:
        scanline32_p8(const self_type&);
        const self_type& operator = (const self_type&);

        unsigned              m_max_len;
        int                   m_last_x;
        int                   m_y;
        pod_array<cover_type> m_covers;
        cover_type*           m_cover_ptr;
        span_array_type       m_spans;
    };


}


#endif

