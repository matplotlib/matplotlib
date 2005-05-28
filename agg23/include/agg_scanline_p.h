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

    //==============================================================scanline_p
    // 
    // This is a general purpose scaline container which supports the interface 
    // used in the rasterizer::render(). See description of agg_scanline_u
    // for details.
    // 
    //------------------------------------------------------------------------
    template<class CoverT> class scanline_p
    {
    public:
        typedef CoverT cover_type;
        typedef int16  coord_type;

        //--------------------------------------------------------------------
        struct span
        {
            coord_type        x;
            coord_type        len; // If negative, it's a solid span, covers is valid
            const cover_type* covers;
        };

        typedef span* iterator;
        typedef const span* const_iterator;

        //--------------------------------------------------------------------
        ~scanline_p()
        {
            delete [] m_spans;
            delete [] m_covers;
        }

        scanline_p() :
            m_max_len(0),
            m_last_x(0x7FFFFFF0),
            m_covers(0),
            m_cover_ptr(0),
            m_spans(0),
            m_cur_span(0)
        {
        }

        //--------------------------------------------------------------------
        void reset(int min_x, int max_x)
        {
            unsigned max_len = max_x - min_x + 3;
            if(max_len > m_max_len)
            {
                delete [] m_spans;
                delete [] m_covers;
                m_covers  = new CoverT [max_len];
                m_spans   = new span [max_len];
                m_max_len = max_len;
            }
            m_last_x    = 0x7FFFFFF0;
            m_cover_ptr = m_covers;
            m_cur_span  = m_spans;
            m_cur_span->len = 0;
        }

        //--------------------------------------------------------------------
        void add_cell(int x, unsigned cover)
        {
            *m_cover_ptr = (CoverT)cover;
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
        void add_cells(int x, unsigned len, const CoverT* covers)
        {
            memcpy(m_cover_ptr, covers, len * sizeof(CoverT));
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
                *m_cover_ptr = (CoverT)cover;
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
            m_cover_ptr = m_covers;
            m_cur_span  = m_spans;
            m_cur_span->len = 0;
        }

        //--------------------------------------------------------------------
        int            y()         const { return m_y; }
        unsigned       num_spans() const { return unsigned(m_cur_span - m_spans); }
        const_iterator begin()     const { return m_spans + 1; }

    private:
        scanline_p(const scanline_p<CoverT>&);
        const scanline_p<CoverT>& operator = (const scanline_p<CoverT>&);

        unsigned m_max_len;
        int      m_last_x;
        int      m_y;
        CoverT*  m_covers;
        CoverT*  m_cover_ptr;
        span*    m_spans;
        span*    m_cur_span;
    };


    //=============================================================scanline_p8
    typedef scanline_p<int8u> scanline_p8;

    //============================================================scanline_p16
    typedef scanline_p<int16u> scanline_p16;

    //============================================================scanline_p32
    typedef scanline_p<int32u> scanline_p32;









    //===========================================================scanline32_p
    template<class CoverT> class scanline32_p
    {
    public:
        typedef CoverT cover_type;
        typedef int32  coord_type;
        typedef scanline32_p<cover_type> scanline_type;

        struct span
        {
            span() {}
            span(coord_type x_, coord_type len_, const cover_type* covers_) :
                x(x_), len(len_), covers(covers_) {}

            coord_type x;
            coord_type len; // If negative, it's a solid span, covers is valid
            const cover_type* covers;
        };
        typedef pod_deque<span, 4> span_array_type;


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
        ~scanline32_p()
        {
            delete [] m_covers;
        }

        scanline32_p() :
            m_max_len(0),
            m_last_x(0x7FFFFFF0),
            m_covers(0),
            m_cover_ptr(0)
        {
        }

        //--------------------------------------------------------------------
        void reset(int min_x, int max_x)
        {
            unsigned max_len = max_x - min_x + 3;
            if(max_len > m_max_len)
            {
                delete [] m_covers;
                m_covers  = new cover_type[max_len];
                m_max_len = max_len;
            }
            m_last_x    = 0x7FFFFFF0;
            m_cover_ptr = m_covers;
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
            m_cover_ptr = m_covers;
            m_spans.remove_all();
        }

        //--------------------------------------------------------------------
        int            y()         const { return m_y; }
        unsigned       num_spans() const { return m_spans.size(); }
        const_iterator begin()     const { return const_iterator(m_spans); }

    private:
        scanline32_p(const scanline_type&);
        const scanline_type& operator = (const scanline_type&);

        unsigned        m_max_len;
        int             m_last_x;
        int             m_y;
        cover_type*     m_covers;
        cover_type*     m_cover_ptr;
        span_array_type m_spans;
    };



    //===========================================================scanline32_p8
    typedef scanline32_p<int8u> scanline32_p8;

    //==========================================================scanline32_p16
    typedef scanline32_p<int16u> scanline32_p16;

    //==========================================================scanline32_p32
    typedef scanline32_p<int32u> scanline32_p32;




}


#endif

