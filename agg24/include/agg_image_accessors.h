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

#ifndef AGG_IMAGE_ACCESSORS_INCLUDED
#define AGG_IMAGE_ACCESSORS_INCLUDED

#include "agg_basics.h"

namespace agg
{

    //-----------------------------------------------------image_accessor_clip
    template<class PixFmt> class image_accessor_clip
    {
    public:
        typedef PixFmt   pixfmt_type;
        typedef typename pixfmt_type::color_type color_type;
        typedef typename pixfmt_type::order_type order_type;
        typedef typename pixfmt_type::value_type value_type;
        enum pix_width_e { pix_width = pixfmt_type::pix_width };

        image_accessor_clip() {}
        explicit image_accessor_clip(const pixfmt_type& pixf, 
                                     const color_type& bk) : 
            m_pixf(&pixf)
        {
            pixfmt_type::make_pix(m_bk_buf, bk);
        }

        void attach(const pixfmt_type& pixf)
        {
            m_pixf = &pixf;
        }

        void background_color(const color_type& bk)
        {
            pixfmt_type::make_pix(m_bk_buf, bk);
        }

    private:
        AGG_INLINE const int8u* pixel() const
        {
            if(m_y >= 0 && m_y < (int)m_pixf->height() &&
               m_x >= 0 && m_x < (int)m_pixf->width())
            {
                return m_pixf->pix_ptr(m_x, m_y);
            }
            return m_bk_buf;
        }

    public:
        AGG_INLINE const int8u* span(int x, int y, unsigned len)
        {
            m_x = m_x0 = x;
            m_y = y;
            if(y >= 0 && y < (int)m_pixf->height() &&
               x >= 0 && x+(int)len <= (int)m_pixf->width())
            {
                return m_pix_ptr = m_pixf->pix_ptr(x, y);
            }
            m_pix_ptr = 0;
            return pixel();
        }

        AGG_INLINE const int8u* next_x()
        {
            if(m_pix_ptr) return m_pix_ptr += pix_width;
            ++m_x;
            return pixel();
        }

        AGG_INLINE const int8u* next_y()
        {
            ++m_y;
            m_x = m_x0;
            if(m_pix_ptr && 
               m_y >= 0 && m_y < (int)m_pixf->height())
            {
                return m_pix_ptr = m_pixf->pix_ptr(m_x, m_y);
            }
            m_pix_ptr = 0;
            return pixel();
        }

    private:
        const pixfmt_type* m_pixf;
        int8u              m_bk_buf[4];
        int                m_x, m_x0, m_y;
        const int8u*       m_pix_ptr;
    };




    //--------------------------------------------------image_accessor_no_clip
    template<class PixFmt> class image_accessor_no_clip
    {
    public:
        typedef PixFmt   pixfmt_type;
        typedef typename pixfmt_type::color_type color_type;
        typedef typename pixfmt_type::order_type order_type;
        typedef typename pixfmt_type::value_type value_type;
        enum pix_width_e { pix_width = pixfmt_type::pix_width };

        image_accessor_no_clip() {}
        explicit image_accessor_no_clip(const pixfmt_type& pixf) : 
            m_pixf(&pixf) 
        {}

        void attach(const pixfmt_type& pixf)
        {
            m_pixf = &pixf;
        }

        AGG_INLINE const int8u* span(int x, int y, unsigned)
        {
            m_x = x;
            m_y = y;
            return m_pix_ptr = m_pixf->pix_ptr(x, y);
        }

        AGG_INLINE const int8u* next_x()
        {
            return m_pix_ptr += pix_width;
        }

        AGG_INLINE const int8u* next_y()
        {
            ++m_y;
            return m_pix_ptr = m_pixf->pix_ptr(m_x, m_y);
        }

    private:
        const pixfmt_type* m_pixf;
        int                m_x, m_y;
        const int8u*       m_pix_ptr;
    };




    //----------------------------------------------------image_accessor_clone
    template<class PixFmt> class image_accessor_clone
    {
    public:
        typedef PixFmt   pixfmt_type;
        typedef typename pixfmt_type::color_type color_type;
        typedef typename pixfmt_type::order_type order_type;
        typedef typename pixfmt_type::value_type value_type;
        enum pix_width_e { pix_width = pixfmt_type::pix_width };

        image_accessor_clone() {}
        explicit image_accessor_clone(const pixfmt_type& pixf) : 
            m_pixf(&pixf) 
        {}

        void attach(const pixfmt_type& pixf)
        {
            m_pixf = &pixf;
        }

    private:
        AGG_INLINE const int8u* pixel() const
        {
            register int x = m_x;
            register int y = m_y;
            if(x < 0) x = 0;
            if(y < 0) y = 0;
            if(x >= (int)m_pixf->width())  x = m_pixf->width() - 1;
            if(y >= (int)m_pixf->height()) y = m_pixf->height() - 1;
            return m_pixf->pix_ptr(x, y);
        }

    public:
        AGG_INLINE const int8u* span(int x, int y, unsigned len)
        {
            m_x = m_x0 = x;
            m_y = y;
            if(y >= 0 && y < (int)m_pixf->height() &&
               x >= 0 && x+len <= (int)m_pixf->width())
            {
                return m_pix_ptr = m_pixf->pix_ptr(x, y);
            }
            m_pix_ptr = 0;
            return pixel();
        }

        AGG_INLINE const int8u* next_x()
        {
            if(m_pix_ptr) return m_pix_ptr += pix_width;
            ++m_x;
            return pixel();
        }

        AGG_INLINE const int8u* next_y()
        {
            ++m_y;
            m_x = m_x0;
            if(m_pix_ptr && 
               m_y >= 0 && m_y < (int)m_pixf->height())
            {
                return m_pix_ptr = m_pixf->pix_ptr(m_x, m_y);
            }
            m_pix_ptr = 0;
            return pixel();
        }

    private:
        const pixfmt_type* m_pixf;
        int                m_x, m_x0, m_y;
        const int8u*       m_pix_ptr;
    };





    //-----------------------------------------------------image_accessor_wrap
    template<class PixFmt, class WrapX, class WrapY> class image_accessor_wrap
    {
    public:
        typedef PixFmt   pixfmt_type;
        typedef typename pixfmt_type::color_type color_type;
        typedef typename pixfmt_type::order_type order_type;
        typedef typename pixfmt_type::value_type value_type;
        enum pix_width_e { pix_width = pixfmt_type::pix_width };

        image_accessor_wrap() {}
        explicit image_accessor_wrap(const pixfmt_type& pixf) : 
            m_pixf(&pixf), 
            m_wrap_x(pixf.width()), 
            m_wrap_y(pixf.height())
        {}

        void attach(const pixfmt_type& pixf)
        {
            m_pixf = &pixf;
        }

        AGG_INLINE const int8u* span(int x, int y, unsigned)
        {
            m_x = x;
            m_row_ptr = m_pixf->row_ptr(m_wrap_y(y));
            return m_row_ptr + m_wrap_x(x) * pix_width;
        }

        AGG_INLINE const int8u* next_x()
        {
            int x = ++m_wrap_x;
            return m_row_ptr + x * pix_width;
        }

        AGG_INLINE const int8u* next_y()
        {
            m_row_ptr = m_pixf->row_ptr(++m_wrap_y);
            return m_row_ptr + m_wrap_x(m_x) * pix_width;
        }

    private:
        const pixfmt_type* m_pixf;
        const int8u*       m_row_ptr;
        int                m_x;
        WrapX              m_wrap_x;
        WrapY              m_wrap_y;
    };




    //--------------------------------------------------------wrap_mode_repeat
    class wrap_mode_repeat
    {
    public:
        wrap_mode_repeat() {}
        wrap_mode_repeat(unsigned size) : 
            m_size(size), 
            m_add(size * (0x3FFFFFFF / size)),
            m_value(0)
        {}

        AGG_INLINE unsigned operator() (int v)
        { 
            return m_value = (unsigned(v) + m_add) % m_size; 
        }

        AGG_INLINE unsigned operator++ ()
        {
            ++m_value;
            if(m_value >= m_size) m_value = 0;
            return m_value;
        }
    private:
        unsigned m_size;
        unsigned m_add;
        unsigned m_value;
    };


    //---------------------------------------------------wrap_mode_repeat_pow2
    class wrap_mode_repeat_pow2
    {
    public:
        wrap_mode_repeat_pow2() {}
        wrap_mode_repeat_pow2(unsigned size) : m_value(0)
        {
            m_mask = 1;
            while(m_mask < size) m_mask = (m_mask << 1) | 1;
            m_mask >>= 1;
        }
        AGG_INLINE unsigned operator() (int v)
        { 
            return m_value = unsigned(v) & m_mask;
        }
        AGG_INLINE unsigned operator++ ()
        {
            ++m_value;
            if(m_value > m_mask) m_value = 0;
            return m_value;
        }
    private:
        unsigned m_mask;
        unsigned m_value;
    };


    //----------------------------------------------wrap_mode_repeat_auto_pow2
    class wrap_mode_repeat_auto_pow2
    {
    public:
        wrap_mode_repeat_auto_pow2() {}
        wrap_mode_repeat_auto_pow2(unsigned size) :
            m_size(size),
            m_add(size * (0x3FFFFFFF / size)),
            m_mask((m_size & (m_size-1)) ? 0 : m_size-1),
            m_value(0)
        {}

        AGG_INLINE unsigned operator() (int v) 
        { 
            if(m_mask) return m_value = unsigned(v) & m_mask;
            return m_value = (unsigned(v) + m_add) % m_size;
        }
        AGG_INLINE unsigned operator++ ()
        {
            ++m_value;
            if(m_value >= m_size) m_value = 0;
            return m_value;
        }

    private:
        unsigned m_size;
        unsigned m_add;
        unsigned m_mask;
        unsigned m_value;
    };


    //-------------------------------------------------------wrap_mode_reflect
    class wrap_mode_reflect
    {
    public:
        wrap_mode_reflect() {}
        wrap_mode_reflect(unsigned size) : 
            m_size(size), 
            m_size2(size * 2),
            m_add(m_size2 * (0x3FFFFFFF / m_size2)),
            m_value(0)
        {}

        AGG_INLINE unsigned operator() (int v)
        { 
            m_value = (unsigned(v) + m_add) % m_size2;
            if(m_value >= m_size) return m_size2 - m_value - 1;
            return m_value;
        }

        AGG_INLINE unsigned operator++ ()
        {
            ++m_value;
            if(m_value >= m_size2) m_value = 0;
            if(m_value >= m_size) return m_size2 - m_value - 1;
            return m_value;
        }
    private:
        unsigned m_size;
        unsigned m_size2;
        unsigned m_add;
        unsigned m_value;
    };



    //--------------------------------------------------wrap_mode_reflect_pow2
    class wrap_mode_reflect_pow2
    {
    public:
        wrap_mode_reflect_pow2() {}
        wrap_mode_reflect_pow2(unsigned size) : m_value(0)
        {
            m_mask = 1;
            m_size = 1;
            while(m_mask < size) 
            {
                m_mask = (m_mask << 1) | 1;
                m_size <<= 1;
            }
        }
        AGG_INLINE unsigned operator() (int v)
        { 
            m_value = unsigned(v) & m_mask;
            if(m_value >= m_size) return m_mask - m_value;
            return m_value;
        }
        AGG_INLINE unsigned operator++ ()
        {
            ++m_value;
            m_value &= m_mask;
            if(m_value >= m_size) return m_mask - m_value;
            return m_value;
        }
    private:
        unsigned m_size;
        unsigned m_mask;
        unsigned m_value;
    };



    //---------------------------------------------wrap_mode_reflect_auto_pow2
    class wrap_mode_reflect_auto_pow2
    {
    public:
        wrap_mode_reflect_auto_pow2() {}
        wrap_mode_reflect_auto_pow2(unsigned size) :
            m_size(size),
            m_size2(size * 2),
            m_add(m_size2 * (0x3FFFFFFF / m_size2)),
            m_mask((m_size2 & (m_size2-1)) ? 0 : m_size2-1),
            m_value(0)
        {}

        AGG_INLINE unsigned operator() (int v) 
        { 
            m_value = m_mask ? unsigned(v) & m_mask : 
                              (unsigned(v) + m_add) % m_size2;
            if(m_value >= m_size) return m_size2 - m_value - 1;
            return m_value;            
        }
        AGG_INLINE unsigned operator++ ()
        {
            ++m_value;
            if(m_value >= m_size2) m_value = 0;
            if(m_value >= m_size) return m_size2 - m_value - 1;
            return m_value;
        }

    private:
        unsigned m_size;
        unsigned m_size2;
        unsigned m_add;
        unsigned m_mask;
        unsigned m_value;
    };


}


#endif
