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


#ifndef AGG_SPAN_PATTERN_RGB_INCLUDED
#define AGG_SPAN_PATTERN_RGB_INCLUDED

#include "agg_basics.h"
#include "agg_pixfmt_rgb.h"
#include "agg_span_pattern.h"

namespace agg
{
    //=======================================================span_pattern_rgb
    template<class ColorT,
             class Order, 
             class WrapModeX,
             class WrapModeY,
             class Allocator = span_allocator<ColorT> > 
    class span_pattern_rgb : public span_pattern_base<ColorT, Allocator>
    {
    public:
        typedef ColorT color_type;
        typedef Order order_type;
        typedef Allocator alloc_type;
        typedef span_pattern_base<color_type, alloc_type> base_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        enum
        {
            base_shift = color_type::base_shift,
            base_mask  = color_type::base_mask
        };

        //--------------------------------------------------------------------
        span_pattern_rgb(alloc_type& alloc) : 
            base_type(alloc), 
            m_wrap_mode_x(1),
            m_wrap_mode_y(1)
        {}

        //----------------------------------------------------------------
        span_pattern_rgb(alloc_type& alloc,
                         const rendering_buffer& src, 
                         unsigned offset_x, 
                         unsigned offset_y,
                         value_type alpha = base_mask) :
            base_type(alloc, src, offset_x, offset_y, alpha),
            m_wrap_mode_x(src.width()),
            m_wrap_mode_y(src.height())
        {}

        //-------------------------------------------------------------------
        void source_image(const rendering_buffer& src) 
        { 
            base_type::source_image(src);
            m_wrap_mode_x = WrapModeX(src.width());
            m_wrap_mode_y = WrapModeY(src.height());
        }

        //--------------------------------------------------------------------
        color_type* generate(int x, int y, unsigned len)
        {   
            color_type* span = base_type::allocator().span();
            unsigned sx = m_wrap_mode_x(base_type::offset_x() + x);
            const value_type* row_ptr = 
                (const value_type*)base_type::source_image().row(
                    m_wrap_mode_y(
                        base_type::offset_y() + y));
            do
            {
                const value_type* p = row_ptr + sx + sx + sx;
                span->r = p[order_type::R];
                span->g = p[order_type::G];
                span->b = p[order_type::B];
                span->a = base_type::alpha_int();
                sx = ++m_wrap_mode_x;
                ++span;
            }
            while(--len);
            return base_type::allocator().span();
        }

    private:
        WrapModeX m_wrap_mode_x;
        WrapModeY m_wrap_mode_y;
    };



/*
    //=========================================================span_pattern_rgb
    template<class ColorT, class Order, class Allocator = span_allocator<ColorT> >
    class span_pattern_rgb : public span_pattern<rgba8, int8u, Allocator>
    {
    public:
        typedef Allocator alloc_type;
        typedef rgba8 color_type;
        typedef span_pattern<color_type, int8u, alloc_type> base_type;

        //--------------------------------------------------------------------
        span_pattern_rgb24(alloc_type& alloc) : base_type(alloc) {}

        //----------------------------------------------------------------
        span_pattern_rgb24(alloc_type& alloc,
                           const rendering_buffer& src, 
                           unsigned offset_x, unsigned offset_y, 
                           int8u alpha = 255) :
            base_type(alloc, src, offset_x, offset_y, alpha)
        {}


        //--------------------------------------------------------------------
        color_type* generate(int x, int y, unsigned len)
        {   
            color_type* span = base_type::allocator().span();
            unsigned sx = (base_type::offset_x() + x) % base_type::source_image().width();
            unsigned wp = base_type::source_image().width() * 3;
            const int8u* p = base_type::source_image().row((base_type::offset_y() + y) % base_type::source_image().height());
            p += sx * 3;
            do
            {
                span->r = p[Order::R];
                span->g = p[Order::G];
                span->b = p[Order::B];
                span->a = base_type::alpha();
                p += 3;
                ++sx;
                ++span;
                if(sx >= base_type::source_image().width())
                {
                    sx -= base_type::source_image().width();
                    p -= wp;
                }
            }
            while(--len);
            return base_type::allocator().span();
        }
    };
*/

}

#endif

