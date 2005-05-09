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

#ifndef AGG_SCANLINE_U_INCLUDED
#define AGG_SCANLINE_U_INCLUDED

#include <string.h>
#include "agg_basics.h"

namespace agg
{
    //==============================================================scanline_u
    //
    // Unpacked scanline container class
    //
    // This class is used to transfer data from a scanline rastyerizer 
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
    // That's because using the scaline is generally faster. When is consists 
    // of more than one span the conditions for the processor cash system
    // are better, because switching between two different areas of memory 
    // (that can be very large) occures less frequently.
    //------------------------------------------------------------------------
    template<class CoverT, class CoordT=int16> class scanline_u
    {
    public:
        typedef scanline_u<CoverT, CoordT> self_type;
        typedef CoverT cover_type;
        typedef CoordT coord_type;

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
        ~scanline_u();
        scanline_u();

        void     reset(int min_x, int max_x);
        void     add_cell(int x, unsigned cover);
        void     add_cells(int x, unsigned len, const CoverT* covers);
        void     add_span(int x, unsigned len, unsigned cover);
        void     finalize(int y) { m_y = y; }
        void     reset_spans();

        int      y()           const { return m_y; }
        unsigned num_spans()   const { return unsigned(m_cur_span - m_spans); }
        const_iterator begin() const { return m_spans + 1; }
        iterator       begin()       { return m_spans + 1; }

    private:
        scanline_u(const self_type&);
        const self_type& operator = (const self_type&);

    private:
        int           m_min_x;
        unsigned      m_max_len;
        int           m_last_x;
        int           m_y;
        cover_type*   m_covers;
        span*         m_spans;
        span*         m_cur_span;
    };



    //------------------------------------------------------------------------
    template<class CoverT, class CoordT> 
    scanline_u<CoverT, CoordT>::~scanline_u()
    {
        delete [] m_spans;
        delete [] m_covers;
    }


    //------------------------------------------------------------------------
    template<class CoverT, class CoordT> scanline_u<CoverT, CoordT>::scanline_u() :
        m_min_x(0),
        m_max_len(0),
        m_last_x(0x7FFFFFF0),
        m_covers(0),
        m_spans(0),
        m_cur_span(0)
    {
    }


    //------------------------------------------------------------------------
    template<class CoverT, class CoordT> 
    void scanline_u<CoverT, CoordT>::reset(int min_x, int max_x)
    {
        unsigned max_len = max_x - min_x + 2;
        if(max_len > m_max_len)
        {
            delete [] m_spans;
            delete [] m_covers;
            m_covers  = new cover_type [max_len];
            m_spans   = new span       [max_len];
            m_max_len = max_len;
        }
        m_last_x        = 0x7FFFFFF0;
        m_min_x         = min_x;
        m_cur_span      = m_spans;
    }


    //------------------------------------------------------------------------
    template<class CoverT, class CoordT> 
    inline void scanline_u<CoverT, CoordT>::reset_spans()
    {
        m_last_x    = 0x7FFFFFF0;
        m_cur_span  = m_spans;
    }


    //------------------------------------------------------------------------
    template<class CoverT, class CoordT> 
    inline void scanline_u<CoverT, CoordT>::add_cell(int x, unsigned cover)
    {
        x -= m_min_x;
        m_covers[x] = (unsigned char)cover;
        if(x == m_last_x+1)
        {
            m_cur_span->len++;
        }
        else
        {
            m_cur_span++;
            m_cur_span->x      = (coord_type)(x + m_min_x);
            m_cur_span->len    = 1;
            m_cur_span->covers = m_covers + x;
        }
        m_last_x = x;
    }


    //------------------------------------------------------------------------
    template<class CoverT, class CoordT> 
    void scanline_u<CoverT, CoordT>::add_cells(int x, unsigned len, const CoverT* covers)
    {
        x -= m_min_x;
        memcpy(m_covers + x, covers, len * sizeof(CoverT));
        if(x == m_last_x+1)
        {
            m_cur_span->len += (coord_type)len;
        }
        else
        {
            m_cur_span++;
            m_cur_span->x      = (coord_type)(x + m_min_x);
            m_cur_span->len    = (coord_type)len;
            m_cur_span->covers = m_covers + x;
        }
        m_last_x = x + len - 1;
    }


    //------------------------------------------------------------------------
    template<class CoverT, class CoordT> 
    void scanline_u<CoverT, CoordT>::add_span(int x, unsigned len, unsigned cover)
    {
        x -= m_min_x;
        memset(m_covers + x, cover, len);
        if(x == m_last_x+1)
        {
            m_cur_span->len += (coord_type)len;
        }
        else
        {
            m_cur_span++;
            m_cur_span->x      = (coord_type)(x + m_min_x);
            m_cur_span->len    = (coord_type)len;
            m_cur_span->covers = m_covers + x;
        }
        m_last_x = x + len - 1;
    }


    //=============================================================scanline_u8
    typedef scanline_u<int8u, int16> scanline_u8;

    //============================================================scanline_u16
    typedef scanline_u<int16u, int16> scanline_u16;

    //============================================================scanline_u32
    typedef scanline_u<int32u, int16> scanline_u32;


    //=============================================================scanline_am
    // 
    // The scanline container with alpha-masking
    // 
    //------------------------------------------------------------------------
    template<class AlphaMask, class CoverT, class CoordT=int16> 
    class scanline_am : public scanline_u<CoverT, CoordT>
    {
    public:
        typedef AlphaMask alpha_mask_type;
        typedef CoverT cover_type;
        typedef CoordT coord_type;
        typedef scanline_u<CoverT, CoordT> scanline_type;

        scanline_am() : scanline_type(), m_alpha_mask(0) {}
        scanline_am(const AlphaMask& am) : scanline_type(), m_alpha_mask(&am) {}

        //--------------------------------------------------------------------
        void finalize(int span_y)
        {
            scanline_type::finalize(span_y);
            if(m_alpha_mask)
            {
                typename scanline_type::iterator span = scanline_type::begin();
                unsigned count = scanline_type::num_spans();
                do
                {
                    m_alpha_mask->combine_hspan(span->x, 
                                                scanline_type::y(), 
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


    //==========================================================scanline_u8_am
    template<class AlphaMask> 
    class scanline_u8_am : public scanline_am<AlphaMask, int8u, int16>
    {
    public:
        typedef AlphaMask alpha_mask_type;
        typedef int8u cover_type;
        typedef scanline_am<alpha_mask_type, cover_type> self_type;

        scanline_u8_am() : self_type() {}
        scanline_u8_am(const AlphaMask& am) : self_type(am) {}
    };

}

#endif

