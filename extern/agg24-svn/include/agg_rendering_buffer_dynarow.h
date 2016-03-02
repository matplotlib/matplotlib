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
// class rendering_buffer_dynarow
//
//----------------------------------------------------------------------------

#ifndef AGG_RENDERING_BUFFER_DYNAROW_INCLUDED
#define AGG_RENDERING_BUFFER_DYNAROW_INCLUDED

#include "agg_array.h"

namespace agg
{

    //===============================================rendering_buffer_dynarow
    // Rendering buffer class with dynamic allocation of the rows.
    // The rows are allocated as needed when requesting for span_ptr().
    // The class automatically calculates min_x and max_x for each row.
    // Generally it's more efficient to use this class as a temporary buffer
    // for rendering a few lines and then to blend it with another buffer.
    //
    class rendering_buffer_dynarow
    {
    public:
        typedef row_info<int8u> row_data;

        //-------------------------------------------------------------------
        ~rendering_buffer_dynarow()
        {
            init(0,0,0);
        }

        //-------------------------------------------------------------------
        rendering_buffer_dynarow() :
            m_rows(),
            m_width(0),
            m_height(0),
            m_byte_width(0)
        {
        }

        // Allocate and clear the buffer
        //--------------------------------------------------------------------
        rendering_buffer_dynarow(unsigned width, unsigned height, 
                                 unsigned byte_width) :
            m_rows(height),
            m_width(width),
            m_height(height),
            m_byte_width(byte_width)
        {
            memset(&m_rows[0], 0, sizeof(row_data) * height);
        }

        // Allocate and clear the buffer
        //--------------------------------------------------------------------
        void init(unsigned width, unsigned height, unsigned byte_width)
        {
            unsigned i;
            for(i = 0; i < m_height; ++i) 
            {
                pod_allocator<int8u>::deallocate((int8u*)m_rows[i].ptr, m_byte_width);
            }
            if(width && height)
            {
                m_width  = width;
                m_height = height;
                m_byte_width = byte_width;
                m_rows.resize(height);
                memset(&m_rows[0], 0, sizeof(row_data) * height);
            }
        }

        //--------------------------------------------------------------------
        unsigned width()      const { return m_width;  }
        unsigned height()     const { return m_height; }
        unsigned byte_width() const { return m_byte_width; }

        // The main function used for rendering. Returns pointer to the 
        // pre-allocated span. Memory for the row is allocated as needed.
        //--------------------------------------------------------------------
        int8u* row_ptr(int x, int y, unsigned len)
        {
            row_data* r = &m_rows[y];
            int x2 = x + len - 1;
            if(r->ptr)
            {
                if(x  < r->x1) { r->x1 = x;  }
                if(x2 > r->x2) { r->x2 = x2; }
            }
            else
            {
                int8u* p = pod_allocator<int8u>::allocate(m_byte_width);
                r->ptr = p;
                r->x1  = x;
                r->x2  = x2;
                memset(p, 0, m_byte_width);
            }
            return (int8u*)r->ptr;
        }

        //--------------------------------------------------------------------
        const int8u* row_ptr(int y) const { return m_rows[y].ptr; }
              int8u* row_ptr(int y)       { return row_ptr(0, y, m_width); }
        row_data     row    (int y) const { return m_rows[y]; }

    private:
        //--------------------------------------------------------------------
        // Prohibit copying
        rendering_buffer_dynarow(const rendering_buffer_dynarow&);
        const rendering_buffer_dynarow& operator = (const rendering_buffer_dynarow&);

    private:
        //--------------------------------------------------------------------
        pod_array<row_data> m_rows;       // Pointers to each row of the buffer
        unsigned            m_width;      // Width in pixels
        unsigned            m_height;     // Height in pixels
        unsigned            m_byte_width; // Width in bytes
    };


}


#endif
