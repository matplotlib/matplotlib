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
// class rendering_buffer
//
//----------------------------------------------------------------------------

#ifndef AGG_RENDERING_BUFFER_INCLUDED
#define AGG_RENDERING_BUFFER_INCLUDED

#include "agg_array.h"

namespace agg
{

    //===========================================================row_accessor
    template<class T> class row_accessor
    {
    public:
        typedef const_row_info<T> row_data;

        //-------------------------------------------------------------------
        row_accessor() :
            m_buf(0),
            m_start(0),
            m_width(0),
            m_height(0),
            m_stride(0)
        {
        }

        //--------------------------------------------------------------------
        row_accessor(T* buf, unsigned width, unsigned height, int stride) :
            m_buf(0),
            m_start(0),
            m_width(0),
            m_height(0),
            m_stride(0)
        {
            attach(buf, width, height, stride);
        }


        //--------------------------------------------------------------------
        void attach(T* buf, unsigned width, unsigned height, int stride)
        {
            m_buf = m_start = buf;
            m_width = width;
            m_height = height;
            m_stride = stride;
            if(stride < 0) 
            { 
                m_start = m_buf - int(height - 1) * stride;
            }
        }

        //--------------------------------------------------------------------
        AGG_INLINE       T* buf()          { return m_buf;    }
        AGG_INLINE const T* buf()    const { return m_buf;    }
        AGG_INLINE unsigned width()  const { return m_width;  }
        AGG_INLINE unsigned height() const { return m_height; }
        AGG_INLINE int      stride() const { return m_stride; }
        AGG_INLINE unsigned stride_abs() const 
        {
            return (m_stride < 0) ? unsigned(-m_stride) : unsigned(m_stride); 
        }

        //--------------------------------------------------------------------
        AGG_INLINE       T* row_ptr(int, int y, unsigned) 
        { 
            return m_start + y * m_stride; 
        }
        AGG_INLINE       T* row_ptr(int y)       { return m_start + y * m_stride; }
        AGG_INLINE const T* row_ptr(int y) const { return m_start + y * m_stride; }
        AGG_INLINE row_data row    (int y) const 
        { 
            return row_data(0, m_width-1, row_ptr(y)); 
        }

        //--------------------------------------------------------------------
        template<class RenBuf>
        void copy_from(const RenBuf& src)
        {
            unsigned h = height();
            if(src.height() < h) h = src.height();
        
            unsigned l = stride_abs();
            if(src.stride_abs() < l) l = src.stride_abs();
        
            l *= sizeof(T);

            unsigned y;
            unsigned w = width();
            for (y = 0; y < h; y++)
            {
                memcpy(row_ptr(0, y, w), src.row_ptr(y), l);
            }
        }

        //--------------------------------------------------------------------
        void clear(T value)
        {
            unsigned y;
            unsigned w = width();
            unsigned stride = stride_abs();
            for(y = 0; y < height(); y++)
            {
                T* p = row_ptr(0, y, w);
                unsigned x;
                for(x = 0; x < stride; x++)
                {
                    *p++ = value;
                }
            }
        }

    private:
        //--------------------------------------------------------------------
        T*            m_buf;    // Pointer to renrdering buffer
        T*            m_start;  // Pointer to first pixel depending on stride 
        unsigned      m_width;  // Width in pixels
        unsigned      m_height; // Height in pixels
        int           m_stride; // Number of bytes per row. Can be < 0
    };




    //==========================================================row_ptr_cache
    template<class T> class row_ptr_cache
    {
    public:
        typedef const_row_info<T> row_data;

        //-------------------------------------------------------------------
        row_ptr_cache() :
            m_buf(0),
            m_rows(),
            m_width(0),
            m_height(0),
            m_stride(0)
        {
        }

        //--------------------------------------------------------------------
        row_ptr_cache(T* buf, unsigned width, unsigned height, int stride) :
            m_buf(0),
            m_rows(),
            m_width(0),
            m_height(0),
            m_stride(0)
        {
            attach(buf, width, height, stride);
        }

        //--------------------------------------------------------------------
        void attach(T* buf, unsigned width, unsigned height, int stride)
        {
            m_buf = buf;
            m_width = width;
            m_height = height;
            m_stride = stride;
            if(height > m_rows.size())
            {
                m_rows.resize(height);
            }

            T* row_ptr = m_buf;

            if(stride < 0)
            {
                row_ptr = m_buf - int(height - 1) * stride;
            }

            T** rows = &m_rows[0];

            while(height--)
            {
                *rows++ = row_ptr;
                row_ptr += stride;
            }
        }

        //--------------------------------------------------------------------
        AGG_INLINE       T* buf()          { return m_buf;    }
        AGG_INLINE const T* buf()    const { return m_buf;    }
        AGG_INLINE unsigned width()  const { return m_width;  }
        AGG_INLINE unsigned height() const { return m_height; }
        AGG_INLINE int      stride() const { return m_stride; }
        AGG_INLINE unsigned stride_abs() const 
        {
            return (m_stride < 0) ? unsigned(-m_stride) : unsigned(m_stride); 
        }

        //--------------------------------------------------------------------
        AGG_INLINE       T* row_ptr(int, int y, unsigned) 
        { 
            return m_rows[y]; 
        }
        AGG_INLINE       T* row_ptr(int y)       { return m_rows[y]; }
        AGG_INLINE const T* row_ptr(int y) const { return m_rows[y]; }
        AGG_INLINE row_data row    (int y) const 
        { 
            return row_data(0, m_width-1, m_rows[y]); 
        }

        //--------------------------------------------------------------------
        T const* const* rows() const { return &m_rows[0]; }

        //--------------------------------------------------------------------
        template<class RenBuf>
        void copy_from(const RenBuf& src)
        {
            unsigned h = height();
            if(src.height() < h) h = src.height();
        
            unsigned l = stride_abs();
            if(src.stride_abs() < l) l = src.stride_abs();
        
            l *= sizeof(T);

            unsigned y;
            unsigned w = width();
            for (y = 0; y < h; y++)
            {
                memcpy(row_ptr(0, y, w), src.row_ptr(y), l);
            }
        }

        //--------------------------------------------------------------------
        void clear(T value)
        {
            unsigned y;
            unsigned w = width();
            unsigned stride = stride_abs();
            for(y = 0; y < height(); y++)
            {
                T* p = row_ptr(0, y, w);
                unsigned x;
                for(x = 0; x < stride; x++)
                {
                    *p++ = value;
                }
            }
        }

    private:
        //--------------------------------------------------------------------
        T*            m_buf;        // Pointer to renrdering buffer
        pod_array<T*> m_rows;       // Pointers to each row of the buffer
        unsigned      m_width;      // Width in pixels
        unsigned      m_height;     // Height in pixels
        int           m_stride;     // Number of bytes per row. Can be < 0
    };




    //========================================================rendering_buffer
    // 
    // The definition of the main type for accessing the rows in the frame 
    // buffer. It provides functionality to navigate to the rows in a 
    // rectangular matrix, from top to bottom or from bottom to top depending 
    // on stride.
    //
    // row_accessor is cheap to create/destroy, but performs one multiplication
    // when calling row_ptr().
    // 
    // row_ptr_cache creates an array of pointers to rows, so, the access 
    // via row_ptr() may be faster. But it requires memory allocation 
    // when creating. For example, on typical Intel Pentium hardware 
    // row_ptr_cache speeds span_image_filter_rgb_nn up to 10%
    //
    // It's used only in short hand typedefs like pixfmt_rgba32 and can be 
    // redefined in agg_config.h
    // In real applications you can use both, depending on your needs
    //------------------------------------------------------------------------
#ifdef AGG_RENDERING_BUFFER
    typedef AGG_RENDERING_BUFFER rendering_buffer;
#else
//  typedef row_ptr_cache<int8u> rendering_buffer;
    typedef row_accessor<int8u> rendering_buffer;
#endif

}


#endif
