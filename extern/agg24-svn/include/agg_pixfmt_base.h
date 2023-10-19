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

#ifndef AGG_PIXFMT_BASE_INCLUDED
#define AGG_PIXFMT_BASE_INCLUDED

#include "agg_basics.h"
#include "agg_color_gray.h"
#include "agg_color_rgba.h"

namespace agg
{
    struct pixfmt_gray_tag
    {
    };

    struct pixfmt_rgb_tag
    {
    };

    struct pixfmt_rgba_tag
    {
    };

    //--------------------------------------------------------------blender_base
    template<class ColorT, class Order = void> 
    struct blender_base
    {
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;

        static rgba get(value_type r, value_type g, value_type b, value_type a, cover_type cover = cover_full)
        {
            if (cover > cover_none)
            {
                rgba c(
                    color_type::to_double(r), 
                    color_type::to_double(g), 
                    color_type::to_double(b), 
                    color_type::to_double(a));

                if (cover < cover_full)
                {
                    double x = double(cover) / cover_full;
                    c.r *= x;
                    c.g *= x;
                    c.b *= x;
                    c.a *= x;
                }

                return c;
            }
            else return rgba::no_color();
        }

        static rgba get(const value_type* p, cover_type cover = cover_full)
        {
            return get(
                p[order_type::R], 
                p[order_type::G], 
                p[order_type::B], 
                p[order_type::A], 
                cover);
        }

        static void set(value_type* p, value_type r, value_type g, value_type b, value_type a)
        {
            p[order_type::R] = r;
            p[order_type::G] = g;
            p[order_type::B] = b;
            p[order_type::A] = a;
        }

        static void set(value_type* p, const rgba& c)
        {
            p[order_type::R] = color_type::from_double(c.r);
            p[order_type::G] = color_type::from_double(c.g);
            p[order_type::B] = color_type::from_double(c.b);
            p[order_type::A] = color_type::from_double(c.a);
        }
    };
}

#endif
