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
// Adaptation for high precision colors has been sponsored by 
// Liberty Technology Systems, Inc., visit http://lib-sys.com
//
// Liberty Technology Systems, Inc. is the provider of
// PostScript and PDF technology for software developers.
// 
//----------------------------------------------------------------------------


#ifndef AGG_SPAN_PATTERN_INCLUDED
#define AGG_SPAN_PATTERN_INCLUDED

#include "agg_basics.h"
#include "agg_rendering_buffer.h"
#include "agg_span_generator.h"


namespace agg
{

    //---------------------------------------------------span_pattern_base
    template<class ColorT, class Allocator> 
    class span_pattern_base : public span_generator<ColorT, Allocator>
    {
    public:
        typedef ColorT color_type;
        typedef typename ColorT::value_type value_type;
        typedef Allocator alloc_type;
        enum { base_mask = color_type::base_mask };

        //----------------------------------------------------------------
        span_pattern_base(alloc_type& alloc) : 
            span_generator<color_type, alloc_type>(alloc) 
        {}

        //----------------------------------------------------------------
        span_pattern_base(alloc_type& alloc,
                     const rendering_buffer& src, 
                     unsigned offset_x, unsigned offset_y, 
                     double alpha) :
            span_generator<color_type, alloc_type>(alloc),
            m_src(&src),
            m_offset_x(offset_x),
            m_offset_y(offset_y),
            m_alpha(value_type(alpha * double(base_mask)))
        {}

        //----------------------------------------------------------------
        const rendering_buffer& source_image() const { return *m_src; }
        unsigned   offset_x()                  const { return m_offset_x; }
        unsigned   offset_y()                  const { return m_offset_y; }
        double     alpha()                     const { return m_alpha / double(base_mask); }
        value_type alpha_int()                 const { return m_alpha; }

        //----------------------------------------------------------------
        void source_image(const rendering_buffer& v)  { m_src = &v; }
        void offset_x(unsigned v) { m_offset_x = v; }
        void offset_y(unsigned v) { m_offset_y = v; }
        void alpha(double v)  { m_alpha = value_type(v * double(base_mask)); }

        //----------------------------------------------------------------
    private:
        const rendering_buffer* m_src;
        unsigned m_offset_x;
        unsigned m_offset_y;
        value_type m_alpha;
    };


    //---------------------------------------------------wrap_mode_repeat
    class wrap_mode_repeat
    {
    public:
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


    //---------------------------------------------wrap_mode_repeat_pow2
    class wrap_mode_repeat_pow2
    {
    public:
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


    //----------------------------------------wrap_mode_repeat_auto_pow2
    class wrap_mode_repeat_auto_pow2
    {
    public:
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


    //--------------------------------------------------wrap_mode_reflect
    class wrap_mode_reflect
    {
    public:
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



    //-------------------------------------------wrap_mode_reflect_pow2
    class wrap_mode_reflect_pow2
    {
    public:
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



    //---------------------------------------wrap_mode_reflect_auto_pow2
    class wrap_mode_reflect_auto_pow2
    {
    public:
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

