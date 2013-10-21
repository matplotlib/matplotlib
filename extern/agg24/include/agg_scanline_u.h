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
// Adaptation for 32-bit screen coordinates (scanline32_u) has been sponsored by
// Liberty Technology Systems, Inc., visit http://lib-sys.com
//
// Liberty Technology Systems, Inc. is the provider of
// PostScript and PDF technology for software developers.
//
//----------------------------------------------------------------------------

#ifndef AGG_SCANLINE_U_INCLUDED
#define AGG_SCANLINE_U_INCLUDED

#include "agg_array.h"

namespace agg
{
    //=============================================================scanline_u8
    //
    // Unpacked scanline container class
    //
    // This class is used to transfer data from a scanline rasterizer
    // to the rendering buffer. It's organized very simple. The class stores
    // information of horizontal spans to render it into a pixel-map buffer.
    // Each span has staring X, length, and an array of bytes that determine the
    // cover-values for each pixel.
    // Before using this class you should know the minimal and maximal pixel
    // coordinates of your scanline. The protocol of using is:
    // 1. reset(min_x, max_x)
    // 2. add_cell() / add_span() - accumulate scanline.
    //    When forming one scanline the next X coordinate must be always greater
    //    than the last stored one, i.e. it works only with ordered coordinates.
    // 3. Call finalize(y) and render the scanline.
    // 3. Call reset_spans() to prepare for the new scanline.
    //
    // 4. Rendering:
    //
    // Scanline provides an iterator class that allows you to extract
    // the spans and the cover values for each pixel. Be aware that clipping
    // has not been done yet, so you should perform it yourself.
    // Use scanline_u8::iterator to render spans:
    //-------------------------------------------------------------------------
    //
    // int y = sl.y();                    // Y-coordinate of the scanline
    //
    // ************************************
    // ...Perform vertical clipping here...
    // ************************************
    //
    // scanline_u8::const_iterator span = sl.begin();
    //
    // unsigned char* row = m_rbuf->row(y); // The the address of the beginning
    //                                      // of the current row
    //
    // unsigned num_spans = sl.num_spans(); // Number of spans. It's guaranteed that
    //                                      // num_spans is always greater than 0.
    //
    // do
    // {
    //     const scanline_u8::cover_type* covers =
    //         span->covers;                     // The array of the cover values
    //
    //     int num_pix = span->len;              // Number of pixels of the span.
    //                                           // Always greater than 0, still it's
    //                                           // better to use "int" instead of
    //                                           // "unsigned" because it's more
    //                                           // convenient for clipping
    //     int x = span->x;
    //
    //     **************************************
    //     ...Perform horizontal clipping here...
    //     ...you have x, covers, and pix_count..
    //     **************************************
    //
    //     unsigned char* dst = row + x;  // Calculate the start address of the row.
    //                                    // In this case we assume a simple
    //                                    // grayscale image 1-byte per pixel.
    //     do
    //     {
    //         *dst++ = *covers++;        // Hypotetical rendering.
    //     }
    //     while(--num_pix);
    //
    //     ++span;
    // }
    // while(--num_spans);  // num_spans cannot be 0, so this loop is quite safe
    //------------------------------------------------------------------------
    //
    // The question is: why should we accumulate the whole scanline when we
    // could render just separate spans when they're ready?
    // That's because using the scanline is generally faster. When is consists
    // of more than one span the conditions for the processor cash system
    // are better, because switching between two different areas of memory
    // (that can be very large) occurs less frequently.
    //------------------------------------------------------------------------
    class scanline_u8
    {
    public:
        typedef scanline_u8 self_type;
        typedef int8u       cover_type;
        typedef int16       coord_type;

        //--------------------------------------------------------------------
        struct span
        {
            coord_type  x;
            coord_type  len;
            cover_type* covers;
        };

        typedef span* iterator;
        typedef const span* const_iterator;

        //--------------------------------------------------------------------
        scanline_u8() :
            m_min_x(0),
            m_last_x(0x7FFFFFF0),
            m_cur_span(0)
        {}

        //--------------------------------------------------------------------
        void reset(int min_x, int max_x)
        {
            unsigned max_len = max_x - min_x + 2;
            if(max_len > m_spans.size())
            {
                m_spans.resize(max_len);
                m_covers.resize(max_len);
            }
            m_last_x   = 0x7FFFFFF0;
            m_min_x    = min_x;
            m_cur_span = &m_spans[0];
        }

        //--------------------------------------------------------------------
        void add_cell(int x, unsigned cover)
        {
            x -= m_min_x;
            m_covers[x] = (cover_type)cover;
            if(x == m_last_x+1)
            {
                m_cur_span->len++;
            }
            else
            {
                m_cur_span++;
                m_cur_span->x      = (coord_type)(x + m_min_x);
                m_cur_span->len    = 1;
                m_cur_span->covers = &m_covers[x];
            }
            m_last_x = x;
        }

        //--------------------------------------------------------------------
        void add_cells(int x, unsigned len, const cover_type* covers)
        {
            x -= m_min_x;
            memcpy(&m_covers[x], covers, len * sizeof(cover_type));
            if(x == m_last_x+1)
            {
                m_cur_span->len += (coord_type)len;
            }
            else
            {
                m_cur_span++;
                m_cur_span->x      = (coord_type)(x + m_min_x);
                m_cur_span->len    = (coord_type)len;
                m_cur_span->covers = &m_covers[x];
            }
            m_last_x = x + len - 1;
        }

        //--------------------------------------------------------------------
        void add_span(int x, unsigned len, unsigned cover)
        {
            x -= m_min_x;
            memset(&m_covers[x], cover, len);
            if(x == m_last_x+1)
            {
                m_cur_span->len += (coord_type)len;
            }
            else
            {
                m_cur_span++;
                m_cur_span->x      = (coord_type)(x + m_min_x);
                m_cur_span->len    = (coord_type)len;
                m_cur_span->covers = &m_covers[x];
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
            m_cur_span  = &m_spans[0];
        }

        //--------------------------------------------------------------------
        int      y()           const { return m_y; }
        unsigned num_spans()   const { return unsigned(m_cur_span - &m_spans[0]); }
        const_iterator begin() const { return &m_spans[1]; }
        iterator       begin()       { return &m_spans[1]; }

    private:
        scanline_u8(const self_type&);
        const self_type& operator = (const self_type&);

    private:
        int                   m_min_x;
        int                   m_last_x;
        int                   m_y;
        pod_array<cover_type> m_covers;
        pod_array<span>       m_spans;
        span*                 m_cur_span;
    };




    //==========================================================scanline_u8_am
    //
    // The scanline container with alpha-masking
    //
    //------------------------------------------------------------------------
    template<class AlphaMask>
    class scanline_u8_am : public scanline_u8
    {
    public:
        typedef scanline_u8           base_type;
        typedef AlphaMask             alpha_mask_type;
        typedef base_type::cover_type cover_type;
        typedef base_type::coord_type coord_type;

        scanline_u8_am() : base_type(), m_alpha_mask(0) {}
        scanline_u8_am(const AlphaMask& am) : base_type(), m_alpha_mask(&am) {}

        //--------------------------------------------------------------------
        void finalize(int span_y)
        {
            base_type::finalize(span_y);
            if(m_alpha_mask)
            {
                typename base_type::iterator span = base_type::begin();
                unsigned count = base_type::num_spans();
                do
                {
                    m_alpha_mask->combine_hspan(span->x,
                                                base_type::y(),
                                                span->covers,
                                                span->len);
                    ++span;
                }
                while(--count);
            }
        }

    private:
        const AlphaMask* m_alpha_mask;
    };




    //===========================================================scanline32_u8
    class scanline32_u8
    {
    public:
        typedef scanline32_u8 self_type;
        typedef int8u         cover_type;
        typedef int32         coord_type;

        //--------------------------------------------------------------------
        struct span
        {
            span() {}
            span(coord_type x_, coord_type len_, cover_type* covers_) :
                x(x_), len(len_), covers(covers_) {}

            coord_type  x;
            coord_type  len;
            cover_type* covers;
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
        class iterator
        {
        public:
            iterator(span_array_type& spans) :
                m_spans(spans),
                m_span_idx(0)
            {}

            span& operator*()  { return m_spans[m_span_idx];  }
            span* operator->() { return &m_spans[m_span_idx]; }

            void operator ++ () { ++m_span_idx; }

        private:
            span_array_type& m_spans;
            unsigned         m_span_idx;
        };



        //--------------------------------------------------------------------
        scanline32_u8() :
            m_min_x(0),
            m_last_x(0x7FFFFFF0),
            m_covers()
        {}

        //--------------------------------------------------------------------
        void reset(int min_x, int max_x)
        {
            unsigned max_len = max_x - min_x + 2;
            if(max_len > m_covers.size())
            {
                m_covers.resize(max_len);
            }
            m_last_x = 0x7FFFFFF0;
            m_min_x  = min_x;
            m_spans.remove_all();
        }

        //--------------------------------------------------------------------
        void add_cell(int x, unsigned cover)
        {
            x -= m_min_x;
            m_covers[x] = cover_type(cover);
            if(x == m_last_x+1)
            {
                m_spans.last().len++;
            }
            else
            {
                m_spans.add(span(coord_type(x + m_min_x), 1, &m_covers[x]));
            }
            m_last_x = x;
        }

        //--------------------------------------------------------------------
        void add_cells(int x, unsigned len, const cover_type* covers)
        {
            x -= m_min_x;
            memcpy(&m_covers[x], covers, len * sizeof(cover_type));
            if(x == m_last_x+1)
            {
                m_spans.last().len += coord_type(len);
            }
            else
            {
                m_spans.add(span(coord_type(x + m_min_x),
                                 coord_type(len),
                                 &m_covers[x]));
            }
            m_last_x = x + len - 1;
        }

        //--------------------------------------------------------------------
        void add_span(int x, unsigned len, unsigned cover)
        {
            x -= m_min_x;
            memset(&m_covers[x], cover, len);
            if(x == m_last_x+1)
            {
                m_spans.last().len += coord_type(len);
            }
            else
            {
                m_spans.add(span(coord_type(x + m_min_x),
                                 coord_type(len),
                                 &m_covers[x]));
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
            m_last_x = 0x7FFFFFF0;
            m_spans.remove_all();
        }

        //--------------------------------------------------------------------
        int      y()           const { return m_y; }
        unsigned num_spans()   const { return m_spans.size(); }
        const_iterator begin() const { return const_iterator(m_spans); }
        iterator       begin()       { return iterator(m_spans); }

    private:
        scanline32_u8(const self_type&);
        const self_type& operator = (const self_type&);

    private:
        int                   m_min_x;
        int                   m_last_x;
        int                   m_y;
        pod_array<cover_type> m_covers;
        span_array_type       m_spans;
    };




    //========================================================scanline32_u8_am
    //
    // The scanline container with alpha-masking
    //
    //------------------------------------------------------------------------
    template<class AlphaMask>
    class scanline32_u8_am : public scanline32_u8
    {
    public:
        typedef scanline32_u8         base_type;
        typedef AlphaMask             alpha_mask_type;
        typedef base_type::cover_type cover_type;
        typedef base_type::coord_type coord_type;


        scanline32_u8_am() : base_type(), m_alpha_mask(0) {}
        scanline32_u8_am(const AlphaMask& am) : base_type(), m_alpha_mask(&am) {}

        //--------------------------------------------------------------------
        void finalize(int span_y)
        {
            base_type::finalize(span_y);
            if(m_alpha_mask)
            {
                typename base_type::iterator span = base_type::begin();
                unsigned count = base_type::num_spans();
                do
                {
                    m_alpha_mask->combine_hspan(span->x,
                                                base_type::y(),
                                                span->covers,
                                                span->len);
                    ++span;
                }
                while(--count);
            }
        }

    private:
        const AlphaMask* m_alpha_mask;
    };



}

#endif

