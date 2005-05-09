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
// class rendering_buffer_dynarow
//
//----------------------------------------------------------------------------

#ifndef AGG_RENDERING_BUFFER_DYNAROW_INCLUDED
#define AGG_RENDERING_BUFFER_DYNAROW_INCLUDED

#include "agg_basics.h"

namespace agg
{

    //===============================================rendering_buffer_dynarow
    // Rendering buffer class with dynamic allocation of the rows.
    // The rows are allocated as needed when requesting for span_ptr().
    // The class automatically calculates min_x and max_x for each row.
    // Generally it's more efficient to use this class as a temporary buffer
    // for rendering a few lines and then to blend it with another buffer.
    //
    template<unsigned PixWidth> class rendering_buffer_dynarow
    {
    public:
        //-------------------------------------------------------------------
        struct row_data
        {
            int8u*   ptr;
            int      x1;
            int      x2;
        };

        //--------------------------------------------------------------------
        struct span_data
        {
            int x;
            unsigned len;
            int8u* ptr;
            span_data() {}
            span_data(int) : x(0), len(0), ptr(0) {}
            span_data(int x_, unsigned len_, int8u* ptr_) : 
                x(x_), len(len_), ptr(ptr_) {}
        };

        //-------------------------------------------------------------------
        ~rendering_buffer_dynarow()
        {
            init(0,0);
        }

        //-------------------------------------------------------------------
        rendering_buffer_dynarow() :
            m_rows(0),
            m_width(0),
            m_height(0)
        {
        }

        // Allocate and clear the buffer
        //--------------------------------------------------------------------
        rendering_buffer_dynarow(unsigned width, unsigned height) :
            m_rows(new row_data[height]),
            m_width(width),
            m_height(height)
        {
            memset(m_rows, 0, sizeof(row_data) * height);
        }

        // Allocate and clear the buffer
        //--------------------------------------------------------------------
        void init(unsigned width, unsigned height)
        {
            unsigned i;
            for(i = 0; i < m_height; ++i) delete [] m_rows[i].ptr;
            delete [] m_rows;
            m_rows = 0;
            if(width && height)
            {
                m_width  = width;
                m_height = height;
                m_rows = new row_data[height];
                memset(m_rows, 0, sizeof(row_data) * height);
            }
        }

        //--------------------------------------------------------------------
        unsigned width()  const { return m_width;  }
        unsigned height() const { return m_height; }

        // Get pointer to the beginning of the row. Memory for the row
        // is allocated as needed.
        //--------------------------------------------------------------------
        int8u* row(int y)
        {
            row_data* r = m_rows + y;
            if(r->ptr == 0)
            {
                r->ptr = new int8u [m_width * PixWidth];
                memset(r->ptr, 0, m_width * PixWidth);
            }
            return r->ptr;
        }

        // Get const pointer to the row. The caller must check it for null.
        //--------------------------------------------------------------------
        const int8u* row(int y) const
        {
            return m_rows[y].ptr;
        }

        // Get the Y-th row. The pointer r.ptr is automatically adjusted
        // to the actual beginning of the row. Use this function as follows:
        //
        // rendering_buffer_dynarow::row_data r = rbuf.row(x, y);
        // if(r.ptr)
        // {
        //    do { blend(r.ptr); r.ptr += PixWidth } while(++r.x1 < r.x2);
        // }
        //--------------------------------------------------------------------
        row_data row(int x, int y) const 
        { 
            row_data r = m_rows[y];
            if(r.ptr)
            {
                if(x < r.x1) x = r.x1;
                r.ptr += x * PixWidth;
            }
            return r; 
        }


        // The main function used for rendering. Returns pointer to the 
        // pre-allocated span. Memory for the row is allocated as needed.
        //--------------------------------------------------------------------
        int8u* span_ptr(int x, int y, unsigned len)
        {
            row_data* r = m_rows + y;
            int x2 = x + len - 1;
            if(r->ptr)
            {
                if(x  < r->x1) { r->x1 = x;  }
                if(x2 > r->x2) { r->x2 = x2; }
            }
            else
            {
                r->ptr = new int8u [m_width * PixWidth];
                r->x1 = x;
                r->x2 = x2;
                memset(r->ptr, 0, m_width * PixWidth);
            }
            return r->ptr + x * PixWidth;
        }

        // Get const pointer to the span. Used mostly in GetPixel function
        // The caller must check the returned pointer for null.
        //--------------------------------------------------------------------
        const int8u* span_ptr(int x, int y, unsigned) const
        {
            row_data* r = m_rows + y;
            return r->ptr ? r->ptr + x * PixWidth : 0;
        }

        // Pre-allocate (if neccesary) and return span.
        //--------------------------------------------------------------------
        span_data span(int x, int y, unsigned len)
        {
            return span_data(x, len, span_ptr(x, y, len));
        }


    private:
        //--------------------------------------------------------------------
        // Prohibit copying
        rendering_buffer_dynarow(const rendering_buffer_dynarow<PixWidth>&);
        const rendering_buffer_dynarow<PixWidth>& 
            operator = (const rendering_buffer_dynarow<PixWidth>&);

    private:
        //--------------------------------------------------------------------
        row_data* m_rows;       // Pointers to each row of the buffer
        unsigned  m_width;      // Width in pixels
        unsigned  m_height;     // Height in pixels
    };


}


#endif
