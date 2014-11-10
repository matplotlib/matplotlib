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
// Adaptation for high precision colors has been sponsored by 
// Liberty Technology Systems, Inc., visit http://lib-sys.com
//
// Liberty Technology Systems, Inc. is the provider of
// PostScript and PDF technology for software developers.
// 
//----------------------------------------------------------------------------

#ifndef AGG_PIXFMT_RGBA_INCLUDED
#define AGG_PIXFMT_RGBA_INCLUDED

#include <string.h>
#include <math.h>
#include "agg_pixfmt_base.h"
#include "agg_rendering_buffer.h"

namespace agg
{
    template<class T> inline T sd_min(T a, T b) { return (a < b) ? a : b; }
    template<class T> inline T sd_max(T a, T b) { return (a > b) ? a : b; }

    inline rgba & clip(rgba & c)
    {
        if (c.a > 1) c.a = 1; else if (c.a < 0) c.a = 0;
        if (c.r > c.a) c.r = c.a; else if (c.r < 0) c.r = 0;
        if (c.g > c.a) c.g = c.a; else if (c.g < 0) c.g = 0;
        if (c.b > c.a) c.b = c.a; else if (c.b < 0) c.b = 0;
        return c;
    }

    //=========================================================multiplier_rgba
    template<class ColorT, class Order> 
    struct multiplier_rgba
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;

        //--------------------------------------------------------------------
        static AGG_INLINE void premultiply(value_type* p)
        {
            value_type a = p[Order::A];
            p[Order::R] = color_type::multiply(p[Order::R], a);
            p[Order::G] = color_type::multiply(p[Order::G], a);
            p[Order::B] = color_type::multiply(p[Order::B], a);
        }


        //--------------------------------------------------------------------
        static AGG_INLINE void demultiply(value_type* p)
        {
            value_type a = p[Order::A];
            p[Order::R] = color_type::demultiply(p[Order::R], a);
            p[Order::G] = color_type::demultiply(p[Order::G], a);
            p[Order::B] = color_type::demultiply(p[Order::B], a);
        }
    };

    //=====================================================apply_gamma_dir_rgba
    template<class ColorT, class Order, class GammaLut> 
    class apply_gamma_dir_rgba
    {
    public:
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;

        apply_gamma_dir_rgba(const GammaLut& gamma) : m_gamma(gamma) {}

        AGG_INLINE void operator () (value_type* p)
        {
            p[Order::R] = m_gamma.dir(p[Order::R]);
            p[Order::G] = m_gamma.dir(p[Order::G]);
            p[Order::B] = m_gamma.dir(p[Order::B]);
        }

    private:
        const GammaLut& m_gamma;
    };

    //=====================================================apply_gamma_inv_rgba
    template<class ColorT, class Order, class GammaLut> class apply_gamma_inv_rgba
    {
    public:
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;

        apply_gamma_inv_rgba(const GammaLut& gamma) : m_gamma(gamma) {}

        AGG_INLINE void operator () (value_type* p)
        {
            p[Order::R] = m_gamma.inv(p[Order::R]);
            p[Order::G] = m_gamma.inv(p[Order::G]);
            p[Order::B] = m_gamma.inv(p[Order::B]);
        }

    private:
        const GammaLut& m_gamma;
    };


    template<class ColorT, class Order> 
    struct conv_rgba_pre
    {
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;

        //--------------------------------------------------------------------
        static AGG_INLINE void set_plain_color(value_type* p, color_type c)
        {
            c.premultiply();
            p[Order::R] = c.r;
            p[Order::G] = c.g;
            p[Order::B] = c.b;
            p[Order::A] = c.a;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE color_type get_plain_color(const value_type* p)
        {
            return color_type(
                p[Order::R],
                p[Order::G],
                p[Order::B],
                p[Order::A]).demultiply();
        }
    };

    template<class ColorT, class Order> 
    struct conv_rgba_plain
    {
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;

        //--------------------------------------------------------------------
        static AGG_INLINE void set_plain_color(value_type* p, color_type c)
        {
            p[Order::R] = c.r;
            p[Order::G] = c.g;
            p[Order::B] = c.b;
            p[Order::A] = c.a;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE color_type get_plain_color(const value_type* p)
        {
            return color_type(
                p[Order::R],
                p[Order::G],
                p[Order::B],
                p[Order::A]);
        }
    };

    //=============================================================blender_rgba
    // Blends "plain" (i.e. non-premultiplied) colors into a premultiplied buffer.
    template<class ColorT, class Order> 
    struct blender_rgba : conv_rgba_pre<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;

        // Blend pixels using the non-premultiplied form of Alvy-Ray Smith's
        // compositing function. Since the render buffer is in fact premultiplied
        // we omit the initial premultiplication and final demultiplication.

        //--------------------------------------------------------------------
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cr, value_type cg, value_type cb, value_type alpha, cover_type cover)
        {
            blend_pix(p, cr, cg, cb, color_type::mult_cover(alpha, cover));
        }
        
        //--------------------------------------------------------------------
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cr, value_type cg, value_type cb, value_type alpha)
        {
            p[Order::R] = color_type::lerp(p[Order::R], cr, alpha);
            p[Order::G] = color_type::lerp(p[Order::G], cg, alpha);
            p[Order::B] = color_type::lerp(p[Order::B], cb, alpha);
            p[Order::A] = color_type::prelerp(p[Order::A], alpha, alpha);
        }
    };


    //========================================================blender_rgba_pre
    // Blends premultiplied colors into a premultiplied buffer.
    template<class ColorT, class Order> 
    struct blender_rgba_pre : conv_rgba_pre<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;

        // Blend pixels using the premultiplied form of Alvy-Ray Smith's
        // compositing function. 

        //--------------------------------------------------------------------
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cr, value_type cg, value_type cb, value_type alpha, cover_type cover)
        {
            blend_pix(p, 
                color_type::mult_cover(cr, cover), 
                color_type::mult_cover(cg, cover), 
                color_type::mult_cover(cb, cover), 
                color_type::mult_cover(alpha, cover));
        }
        
        //--------------------------------------------------------------------
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cr, value_type cg, value_type cb, value_type alpha)
        {
            p[Order::R] = color_type::prelerp(p[Order::R], cr, alpha);
            p[Order::G] = color_type::prelerp(p[Order::G], cg, alpha);
            p[Order::B] = color_type::prelerp(p[Order::B], cb, alpha);
            p[Order::A] = color_type::prelerp(p[Order::A], alpha, alpha);
        }
    };

    //======================================================blender_rgba_plain
    // Blends "plain" (non-premultiplied) colors into a plain (non-premultiplied) buffer.
    template<class ColorT, class Order> 
    struct blender_rgba_plain : conv_rgba_plain<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;

        // Blend pixels using the non-premultiplied form of Alvy-Ray Smith's
        // compositing function. 

        //--------------------------------------------------------------------
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cr, value_type cg, value_type cb, value_type alpha, cover_type cover)
        {
            blend_pix(p, cr, cg, cb, color_type::mult_cover(alpha, cover));
        }
        
        //--------------------------------------------------------------------
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cr, value_type cg, value_type cb, value_type alpha)
        {
            if (alpha > color_type::empty_value())
            {
                calc_type a = p[Order::A];
                calc_type r = color_type::multiply(p[Order::R], a);
                calc_type g = color_type::multiply(p[Order::G], a);
                calc_type b = color_type::multiply(p[Order::B], a);
                p[Order::R] = color_type::lerp(r, cr, alpha);
                p[Order::G] = color_type::lerp(g, cg, alpha);
                p[Order::B] = color_type::lerp(b, cb, alpha);
                p[Order::A] = color_type::prelerp(a, alpha, alpha);
                multiplier_rgba<ColorT, Order>::demultiply(p);
            }
        }
    };

    // SVG compositing operations.
    // For specifications, see http://www.w3.org/TR/SVGCompositing/

    //=========================================================comp_op_rgba_clear
    template<class ColorT, class Order> 
    struct comp_op_rgba_clear : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // Dca' = 0
        // Da'  = 0
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            if (cover >= cover_full)
            {
                p[0] = p[1] = p[2] = p[3] = color_type::empty_value(); 
            }
            else if (cover > cover_none)
            {
                set(p, get(p, cover_full - cover));
            }
        }
    };

    //===========================================================comp_op_rgba_src
    template<class ColorT, class Order> 
    struct comp_op_rgba_src : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // Dca' = Sca
        // Da'  = Sa
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            if (cover >= cover_full)
            {
                set(p, r, g, b, a);
            }
            else
            {
                rgba s = get(r, g, b, a, cover);
                rgba d = get(p, cover_full - cover);
                d.r += s.r;
                d.g += s.g;
                d.b += s.b;
                d.a += s.a;
                set(p, d);
            }
        }
    };

    //===========================================================comp_op_rgba_dst
    template<class ColorT, class Order> 
    struct comp_op_rgba_dst : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;

        // Dca' = Dca.Sa + Dca.(1 - Sa) = Dca
        // Da'  = Da.Sa + Da.(1 - Sa) = Da
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            // Well, that was easy!
        }
    };

    //======================================================comp_op_rgba_src_over
    template<class ColorT, class Order> 
    struct comp_op_rgba_src_over : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // Dca' = Sca + Dca.(1 - Sa) = Dca + Sca - Dca.Sa
        // Da'  = Sa + Da - Sa.Da 
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
#if 1
            blender_rgba_pre<ColorT, Order>::blend_pix(p, r, g, b, a, cover);
#else
            rgba s = get(r, g, b, a, cover);
            rgba d = get(p);
            d.r += s.r - d.r * s.a;
            d.g += s.g - d.g * s.a;
            d.b += s.b - d.b * s.a;
            d.a += s.a - d.a * s.a;
            set(p, d);
#endif
        }
    };

    //======================================================comp_op_rgba_dst_over
    template<class ColorT, class Order> 
    struct comp_op_rgba_dst_over : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // Dca' = Dca + Sca.(1 - Da)
        // Da'  = Sa + Da - Sa.Da = Da + Sa.(1 - Da)
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            rgba s = get(r, g, b, a, cover);
            rgba d = get(p);
            double d1a = 1 - d.a;
            d.r += s.r * d1a;
            d.g += s.g * d1a;
            d.b += s.b * d1a;
            d.a += s.a * d1a;
            set(p, d);
        }
    };

    //======================================================comp_op_rgba_src_in
    template<class ColorT, class Order> 
    struct comp_op_rgba_src_in : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // Dca' = Sca.Da
        // Da'  = Sa.Da 
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            double da = ColorT::to_double(p[Order::A]);
            if (da > 0)
            {
                rgba s = get(r, g, b, a, cover);
                rgba d = get(p, cover_full - cover);
                d.r += s.r * da;
                d.g += s.g * da;
                d.b += s.b * da;
                d.a += s.a * da;
                set(p, d);
            }
        }
    };

    //======================================================comp_op_rgba_dst_in
    template<class ColorT, class Order> 
    struct comp_op_rgba_dst_in : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // Dca' = Dca.Sa
        // Da'  = Sa.Da 
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            double sa = ColorT::to_double(a);
            rgba d = get(p, cover_full - cover);
            rgba d2 = get(p, cover);
            d.r += d2.r * sa;
            d.g += d2.g * sa;
            d.b += d2.b * sa;
            d.a += d2.a * sa;
            set(p, d);
        }
    };

    //======================================================comp_op_rgba_src_out
    template<class ColorT, class Order> 
    struct comp_op_rgba_src_out : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // Dca' = Sca.(1 - Da)
        // Da'  = Sa.(1 - Da) 
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            rgba s = get(r, g, b, a, cover);
            rgba d = get(p, cover_full - cover);
            double d1a = 1 - ColorT::to_double(p[Order::A]);
            d.r += s.r * d1a;
            d.g += s.g * d1a;
            d.b += s.b * d1a;
            d.a += s.a * d1a;
            set(p, d);
        }
    };

    //======================================================comp_op_rgba_dst_out
    template<class ColorT, class Order> 
    struct comp_op_rgba_dst_out : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // Dca' = Dca.(1 - Sa) 
        // Da'  = Da.(1 - Sa) 
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            rgba d = get(p, cover_full - cover);
            rgba dc = get(p, cover);
            double s1a = 1 - ColorT::to_double(a);
            d.r += dc.r * s1a;
            d.g += dc.g * s1a;
            d.b += dc.b * s1a;
            d.a += dc.a * s1a;
            set(p, d);
        }
    };

    //=====================================================comp_op_rgba_src_atop
    template<class ColorT, class Order> 
    struct comp_op_rgba_src_atop : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // Dca' = Sca.Da + Dca.(1 - Sa)
        // Da'  = Da
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            rgba s = get(r, g, b, a, cover);
            rgba d = get(p);
            double s1a = 1 - s.a;
            d.r = s.r * d.a + d.r * s1a;
            d.g = s.g * d.a + d.g * s1a;
            d.b = s.b * d.a + d.g * s1a;
            set(p, d);
        }
    };

    //=====================================================comp_op_rgba_dst_atop
    template<class ColorT, class Order> 
    struct comp_op_rgba_dst_atop : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // Dca' = Dca.Sa + Sca.(1 - Da)
        // Da'  = Sa 
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            rgba sc = get(r, g, b, a, cover);
            rgba dc = get(p, cover);
            rgba d = get(p, cover_full - cover);
            double sa = ColorT::to_double(a);
            double d1a = 1 - ColorT::to_double(p[Order::A]);
            d.r += dc.r * sa + sc.r * d1a;
            d.g += dc.g * sa + sc.g * d1a;
            d.b += dc.b * sa + sc.b * d1a;
            d.a += sc.a;
            set(p, d);
        }
    };

    //=========================================================comp_op_rgba_xor
    template<class ColorT, class Order> 
    struct comp_op_rgba_xor : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // Dca' = Sca.(1 - Da) + Dca.(1 - Sa)
        // Da'  = Sa + Da - 2.Sa.Da 
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            rgba s = get(r, g, b, a, cover);
            rgba d = get(p);
            double s1a = 1 - s.a;
            double d1a = 1 - ColorT::to_double(p[Order::A]);
            d.r = s.r * d1a + d.r * s1a;
            d.g = s.g * d1a + d.g * s1a;
            d.b = s.b * d1a + d.b * s1a;
            d.a = s.a + d.a - 2 * s.a * d.a;
            set(p, d);
        }
    };

    //=========================================================comp_op_rgba_plus
    template<class ColorT, class Order> 
    struct comp_op_rgba_plus : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // Dca' = Sca + Dca
        // Da'  = Sa + Da 
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            rgba s = get(r, g, b, a, cover);
            if (s.a > 0)
            {
                rgba d = get(p);
                d.a = sd_min(d.a + s.a, 1.0);
                d.r = sd_min(d.r + s.r, d.a);
                d.g = sd_min(d.g + s.g, d.a);
                d.b = sd_min(d.b + s.b, d.a);
                set(p, clip(d));
            }
        }
    };

    //========================================================comp_op_rgba_minus
    // Note: not included in SVG spec.
    template<class ColorT, class Order> 
    struct comp_op_rgba_minus : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // Dca' = Dca - Sca
        // Da' = 1 - (1 - Sa).(1 - Da) = Da + Sa - Sa.Da
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            rgba s = get(r, g, b, a, cover);
            if (s.a > 0)
            {
                rgba d = get(p);
                d.a += s.a - s.a * d.a;
                d.r = sd_max(d.r - s.r, 0.0);
                d.g = sd_max(d.g - s.g, 0.0);
                d.b = sd_max(d.b - s.b, 0.0);
                set(p, clip(d));
            }
        }
    };

    //=====================================================comp_op_rgba_multiply
    template<class ColorT, class Order> 
    struct comp_op_rgba_multiply : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // Dca' = Sca.Dca + Sca.(1 - Da) + Dca.(1 - Sa)
        // Da'  = Sa + Da - Sa.Da 
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            rgba s = get(r, g, b, a, cover);
            if (s.a > 0)
            {
                rgba d = get(p);
                double s1a = 1 - s.a;
                double d1a = 1 - d.a;
                d.r = s.r * d.r + s.r * d1a + d.r * s1a;
                d.g = s.g * d.g + s.g * d1a + d.g * s1a;
                d.b = s.b * d.b + s.b * d1a + d.b * s1a;
                d.a += s.a - s.a * d.a;
                set(p, clip(d));
            }
        }
    };

    //=====================================================comp_op_rgba_screen
    template<class ColorT, class Order> 
    struct comp_op_rgba_screen : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // Dca' = Sca + Dca - Sca.Dca
        // Da'  = Sa + Da - Sa.Da 
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            rgba s = get(r, g, b, a, cover);
            if (s.a > 0)
            {
                rgba d = get(p);
                d.r += s.r - s.r * d.r;
                d.g += s.g - s.g * d.g;
                d.b += s.b - s.b * d.b;
                d.a += s.a - s.a * d.a;
                set(p, clip(d));
            }
        }
    };

    //=====================================================comp_op_rgba_overlay
    template<class ColorT, class Order> 
    struct comp_op_rgba_overlay : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // if 2.Dca <= Da
        //   Dca' = 2.Sca.Dca + Sca.(1 - Da) + Dca.(1 - Sa)
        // otherwise
        //   Dca' = Sa.Da - 2.(Da - Dca).(Sa - Sca) + Sca.(1 - Da) + Dca.(1 - Sa)
        // 
        // Da' = Sa + Da - Sa.Da
        static AGG_INLINE double calc(double dca, double sca, double da, double sa, double sada, double d1a, double s1a)
        {
            return (2 * dca <= da) ? 
                2 * sca * dca + sca * d1a + dca * s1a : 
                sada - 2 * (da - dca) * (sa - sca) + sca * d1a + dca * s1a;
        }

        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            rgba s = get(r, g, b, a, cover);
            if (s.a > 0)
            {
                rgba d = get(p);
                double d1a = 1 - d.a;
                double s1a = 1 - s.a;
                double sada = s.a * d.a;
                d.r = calc(d.r, s.r, d.a, s.a, sada, d1a, s1a);
                d.g = calc(d.g, s.g, d.a, s.a, sada, d1a, s1a);
                d.b = calc(d.b, s.b, d.a, s.a, sada, d1a, s1a);
                d.a += s.a - s.a * d.a;
                set(p, clip(d));
            }
        }
    };

    //=====================================================comp_op_rgba_darken
    template<class ColorT, class Order> 
    struct comp_op_rgba_darken : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // Dca' = min(Sca.Da, Dca.Sa) + Sca.(1 - Da) + Dca.(1 - Sa)
        // Da'  = Sa + Da - Sa.Da 
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            rgba s = get(r, g, b, a, cover);
            if (s.a > 0)
            {
                rgba d = get(p);
                double d1a = 1 - d.a;
                double s1a = 1 - s.a;
                d.r = sd_min(s.r * d.a, d.r * s.a) + s.r * d1a + d.r * s1a;
                d.g = sd_min(s.g * d.a, d.g * s.a) + s.g * d1a + d.g * s1a;
                d.b = sd_min(s.b * d.a, d.b * s.a) + s.b * d1a + d.b * s1a;
                d.a += s.a - s.a * d.a;
                set(p, clip(d));
            }
        }
    };

    //=====================================================comp_op_rgba_lighten
    template<class ColorT, class Order> 
    struct comp_op_rgba_lighten : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // Dca' = max(Sca.Da, Dca.Sa) + Sca.(1 - Da) + Dca.(1 - Sa)
        // Da'  = Sa + Da - Sa.Da 
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            rgba s = get(r, g, b, a, cover);
            if (s.a > 0)
            {
                rgba d = get(p);
                double d1a = 1 - d.a;
                double s1a = 1 - s.a;
                d.r = sd_max(s.r * d.a, d.r * s.a) + s.r * d1a + d.r * s1a;
                d.g = sd_max(s.g * d.a, d.g * s.a) + s.g * d1a + d.g * s1a;
                d.b = sd_max(s.b * d.a, d.b * s.a) + s.b * d1a + d.b * s1a;
                d.a += s.a - s.a * d.a;
                set(p, clip(d));
            }
        }
    };

    //=====================================================comp_op_rgba_color_dodge
    template<class ColorT, class Order> 
    struct comp_op_rgba_color_dodge : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // if Sca == Sa and Dca == 0
        //     Dca' = Sca.(1 - Da) + Dca.(1 - Sa) = Sca.(1 - Da)
        // otherwise if Sca == Sa
        //     Dca' = Sa.Da + Sca.(1 - Da) + Dca.(1 - Sa)
        // otherwise if Sca < Sa
        //     Dca' = Sa.Da.min(1, Dca/Da.Sa/(Sa - Sca)) + Sca.(1 - Da) + Dca.(1 - Sa)
        //
        // Da'  = Sa + Da - Sa.Da
        static AGG_INLINE double calc(double dca, double sca, double da, double sa, double sada, double d1a, double s1a)
        {
            if (sca < sa) return sada * sd_min(1.0, (dca / da) * sa / (sa - sca)) + sca * d1a + dca * s1a;
            if (dca > 0) return sada + sca * d1a + dca * s1a;
            return sca * d1a;
        }

        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            rgba s = get(r, g, b, a, cover);
            if (s.a > 0)
            {
                rgba d = get(p);
                if (d.a > 0)
                {
                    double sada = s.a * d.a;
                    double s1a = 1 - s.a;
                    double d1a = 1 - d.a;
                    d.r = calc(d.r, s.r, d.a, s.a, sada, d1a, s1a);
                    d.g = calc(d.g, s.g, d.a, s.a, sada, d1a, s1a);
                    d.b = calc(d.b, s.b, d.a, s.a, sada, d1a, s1a);
                    d.a += s.a - s.a * d.a;
                    set(p, clip(d));
                }
                else set(p, s);
            }
        }
    };

    //=====================================================comp_op_rgba_color_burn
    template<class ColorT, class Order> 
    struct comp_op_rgba_color_burn : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // if Sca == 0 and Dca == Da
        //   Dca' = Sa.Da + Dca.(1 - Sa)
        // otherwise if Sca == 0
        //   Dca' = Dca.(1 - Sa)
        // otherwise if Sca > 0
        //   Dca' =  Sa.Da.(1 - min(1, (1 - Dca/Da).Sa/Sca)) + Sca.(1 - Da) + Dca.(1 - Sa)
        static AGG_INLINE double calc(double dca, double sca, double da, double sa, double sada, double d1a, double s1a)
        {
            if (sca > 0) return sada * (1 - sd_min(1.0, (1 - dca / da) * sa / sca)) + sca * d1a + dca * s1a;
            if (dca > da) return sada + dca * s1a;
            return dca * s1a;
        }

        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            rgba s = get(r, g, b, a, cover);
            if (s.a > 0)
            {
                rgba d = get(p);
                if (d.a > 0)
                {
                    double sada = s.a * d.a;
                    double s1a = 1 - s.a;
                    double d1a = 1 - d.a;
                    d.r = calc(d.r, s.r, d.a, s.a, sada, d1a, s1a);
                    d.g = calc(d.g, s.g, d.a, s.a, sada, d1a, s1a);
                    d.b = calc(d.b, s.b, d.a, s.a, sada, d1a, s1a);
                    d.a += s.a - sada;
                    set(p, clip(d));
                }
                else set(p, s);
            }
        }
    };

    //=====================================================comp_op_rgba_hard_light
    template<class ColorT, class Order> 
    struct comp_op_rgba_hard_light : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // if 2.Sca < Sa
        //    Dca' = 2.Sca.Dca + Sca.(1 - Da) + Dca.(1 - Sa)
        // otherwise
        //    Dca' = Sa.Da - 2.(Da - Dca).(Sa - Sca) + Sca.(1 - Da) + Dca.(1 - Sa)
        // 
        // Da'  = Sa + Da - Sa.Da
        static AGG_INLINE double calc(double dca, double sca, double da, double sa, double sada, double d1a, double s1a)
        {
            return (2 * sca < sa) ? 
                2 * sca * dca + sca * d1a + dca * s1a : 
                sada - 2 * (da - dca) * (sa - sca) + sca * d1a + dca * s1a;
        }

        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            rgba s = get(r, g, b, a, cover);
            if (s.a > 0)
            {
                rgba d = get(p);
                double d1a = 1 - d.a;
                double s1a = 1 - s.a;
                double sada = s.a * d.a;
                d.r = calc(d.r, s.r, d.a, s.a, sada, d1a, s1a);
                d.g = calc(d.g, s.g, d.a, s.a, sada, d1a, s1a);
                d.b = calc(d.b, s.b, d.a, s.a, sada, d1a, s1a);
                d.a += s.a - sada;
                set(p, clip(d));
            }
        }
    };

    //=====================================================comp_op_rgba_soft_light
    template<class ColorT, class Order> 
    struct comp_op_rgba_soft_light : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // if 2.Sca <= Sa
        //   Dca' = Dca.Sa - (Sa.Da - 2.Sca.Da).Dca.Sa.(Sa.Da - Dca.Sa) + Sca.(1 - Da) + Dca.(1 - Sa)
        // otherwise if 2.Sca > Sa and 4.Dca <= Da
        //   Dca' = Dca.Sa + (2.Sca.Da - Sa.Da).((((16.Dsa.Sa - 12).Dsa.Sa + 4).Dsa.Da) - Dsa.Da) + Sca.(1 - Da) + Dca.(1 - Sa)
        // otherwise if 2.Sca > Sa and 4.Dca > Da
        //   Dca' = Dca.Sa + (2.Sca.Da - Sa.Da).((Dca.Sa)^0.5 - Dca.Sa) + Sca.(1 - Da) + Dca.(1 - Sa)
        // 
        // Da'  = Sa + Da - Sa.Da 
        static AGG_INLINE double calc(double dca, double sca, double da, double sa, double sada, double d1a, double s1a)
        {
            double dcasa = dca * sa;
            if (2 * sca <= sa) return dcasa - (sada - 2 * sca * da) * dcasa * (sada - dcasa) + sca * d1a + dca * s1a;
            if (4 * dca <= da) return dcasa + (2 * sca * da - sada) * ((((16 * dcasa - 12) * dcasa + 4) * dca * da) - dca * da) + sca * d1a + dca * s1a;
            return dcasa + (2 * sca * da - sada) * (sqrt(dcasa) - dcasa) + sca * d1a + dca * s1a;
        }

        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            rgba s = get(r, g, b, a, cover);
            if (s.a > 0)
            {
                rgba d = get(p);
                if (d.a > 0)
                {
                    double sada = s.a * d.a;
                    double s1a = 1 - s.a;
                    double d1a = 1 - d.a;
                    d.r = calc(d.r, s.r, d.a, s.a, sada, d1a, s1a);
                    d.g = calc(d.g, s.g, d.a, s.a, sada, d1a, s1a);
                    d.b = calc(d.b, s.b, d.a, s.a, sada, d1a, s1a);
                    d.a += s.a - sada;
                    set(p, clip(d));
                }
                else set(p, s);
            }
        }
    };

    //=====================================================comp_op_rgba_difference
    template<class ColorT, class Order> 
    struct comp_op_rgba_difference : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // Dca' = Sca + Dca - 2.min(Sca.Da, Dca.Sa)
        // Da'  = Sa + Da - Sa.Da 
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            rgba s = get(r, g, b, a, cover);
            if (s.a > 0)
            {
                rgba d = get(p);
                d.r += s.r - 2 * sd_min(s.r * d.a, d.r * s.a);
                d.g += s.g - 2 * sd_min(s.g * d.a, d.g * s.a);
                d.b += s.b - 2 * sd_min(s.b * d.a, d.b * s.a);
                d.a += s.a - s.a * d.a;
                set(p, clip(d));
            }
        }
    };

    //=====================================================comp_op_rgba_exclusion
    template<class ColorT, class Order> 
    struct comp_op_rgba_exclusion : blender_base<ColorT, Order>
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        using blender_base<ColorT, Order>::get;
        using blender_base<ColorT, Order>::set;

        // Dca' = (Sca.Da + Dca.Sa - 2.Sca.Dca) + Sca.(1 - Da) + Dca.(1 - Sa)
        // Da'  = Sa + Da - Sa.Da 
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            rgba s = get(r, g, b, a, cover);
            if (s.a > 0)
            {
                rgba d = get(p);
                double d1a = 1 - d.a;
                double s1a = 1 - s.a;
                d.r = (s.r * d.a + d.r * s.a - 2 * s.r * d.r) + s.r * d1a + d.r * s1a;
                d.g = (s.g * d.a + d.g * s.a - 2 * s.g * d.g) + s.g * d1a + d.g * s1a;
                d.b = (s.b * d.a + d.b * s.a - 2 * s.b * d.b) + s.b * d1a + d.b * s1a;
                d.a += s.a - s.a * d.a;
                set(p, clip(d));
            }
        }
    };

#if 0
    //=====================================================comp_op_rgba_contrast
    template<class ColorT, class Order> struct comp_op_rgba_contrast
    {
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;
        enum base_scale_e
        { 
            base_shift = color_type::base_shift,
            base_mask  = color_type::base_mask
        };


        static AGG_INLINE void blend_pix(value_type* p, 
                                         unsigned sr, unsigned sg, unsigned sb, 
                                         unsigned sa, unsigned cover)
        {
            if (cover < 255)
            {
                sr = (sr * cover + 255) >> 8;
                sg = (sg * cover + 255) >> 8;
                sb = (sb * cover + 255) >> 8;
                sa = (sa * cover + 255) >> 8;
            }
            long_type dr = p[Order::R];
            long_type dg = p[Order::G];
            long_type db = p[Order::B];
            int       da = p[Order::A];
            long_type d2a = da >> 1;
            unsigned s2a = sa >> 1;

            int r = (int)((((dr - d2a) * int((sr - s2a)*2 + base_mask)) >> base_shift) + d2a); 
            int g = (int)((((dg - d2a) * int((sg - s2a)*2 + base_mask)) >> base_shift) + d2a); 
            int b = (int)((((db - d2a) * int((sb - s2a)*2 + base_mask)) >> base_shift) + d2a); 

            r = (r < 0) ? 0 : r;
            g = (g < 0) ? 0 : g;
            b = (b < 0) ? 0 : b;

            p[Order::R] = (value_type)((r > da) ? da : r);
            p[Order::G] = (value_type)((g > da) ? da : g);
            p[Order::B] = (value_type)((b > da) ? da : b);
        }
    };

    //=====================================================comp_op_rgba_invert
    template<class ColorT, class Order> struct comp_op_rgba_invert
    {
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;
        enum base_scale_e
        { 
            base_shift = color_type::base_shift,
            base_mask  = color_type::base_mask
        };

        // Dca' = (Da - Dca) * Sa + Dca.(1 - Sa)
        // Da'  = Sa + Da - Sa.Da 
        static AGG_INLINE void blend_pix(value_type* p, 
                                         unsigned sr, unsigned sg, unsigned sb, 
                                         unsigned sa, unsigned cover)
        {
            sa = (sa * cover + 255) >> 8;
            if (sa)
            {
                calc_type da = p[Order::A];
                calc_type dr = ((da - p[Order::R]) * sa + base_mask) >> base_shift;
                calc_type dg = ((da - p[Order::G]) * sa + base_mask) >> base_shift;
                calc_type db = ((da - p[Order::B]) * sa + base_mask) >> base_shift;
                calc_type s1a = base_mask - sa;
                p[Order::R] = (value_type)(dr + ((p[Order::R] * s1a + base_mask) >> base_shift));
                p[Order::G] = (value_type)(dg + ((p[Order::G] * s1a + base_mask) >> base_shift));
                p[Order::B] = (value_type)(db + ((p[Order::B] * s1a + base_mask) >> base_shift));
                p[Order::A] = (value_type)(sa + da - ((sa * da + base_mask) >> base_shift));
            }
        }
    };

    //=================================================comp_op_rgba_invert_rgb
    template<class ColorT, class Order> struct comp_op_rgba_invert_rgb
    {
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;
        enum base_scale_e
        { 
            base_shift = color_type::base_shift,
            base_mask  = color_type::base_mask
        };

        // Dca' = (Da - Dca) * Sca + Dca.(1 - Sa)
        // Da'  = Sa + Da - Sa.Da 
        static AGG_INLINE void blend_pix(value_type* p, 
                                         unsigned sr, unsigned sg, unsigned sb, 
                                         unsigned sa, unsigned cover)
        {
            if (cover < 255)
            {
                sr = (sr * cover + 255) >> 8;
                sg = (sg * cover + 255) >> 8;
                sb = (sb * cover + 255) >> 8;
                sa = (sa * cover + 255) >> 8;
            }
            if (sa)
            {
                calc_type da = p[Order::A];
                calc_type dr = ((da - p[Order::R]) * sr + base_mask) >> base_shift;
                calc_type dg = ((da - p[Order::G]) * sg + base_mask) >> base_shift;
                calc_type db = ((da - p[Order::B]) * sb + base_mask) >> base_shift;
                calc_type s1a = base_mask - sa;
                p[Order::R] = (value_type)(dr + ((p[Order::R] * s1a + base_mask) >> base_shift));
                p[Order::G] = (value_type)(dg + ((p[Order::G] * s1a + base_mask) >> base_shift));
                p[Order::B] = (value_type)(db + ((p[Order::B] * s1a + base_mask) >> base_shift));
                p[Order::A] = (value_type)(sa + da - ((sa * da + base_mask) >> base_shift));
            }
        }
    };
#endif


    //======================================================comp_op_table_rgba
    template<class ColorT, class Order> struct comp_op_table_rgba
    {
        typedef typename ColorT::value_type value_type;
        typedef typename ColorT::calc_type calc_type;
        typedef void (*comp_op_func_type)(value_type* p, 
                                          value_type cr, 
                                          value_type cg, 
                                          value_type cb,
                                          value_type ca,
                                          cover_type cover);
        static comp_op_func_type g_comp_op_func[];
    };

    //==========================================================g_comp_op_func
    template<class ColorT, class Order> 
    typename comp_op_table_rgba<ColorT, Order>::comp_op_func_type
    comp_op_table_rgba<ColorT, Order>::g_comp_op_func[] = 
    {
        comp_op_rgba_clear      <ColorT,Order>::blend_pix,
        comp_op_rgba_src        <ColorT,Order>::blend_pix,
        comp_op_rgba_dst        <ColorT,Order>::blend_pix,
        comp_op_rgba_src_over   <ColorT,Order>::blend_pix,
        comp_op_rgba_dst_over   <ColorT,Order>::blend_pix,
        comp_op_rgba_src_in     <ColorT,Order>::blend_pix,
        comp_op_rgba_dst_in     <ColorT,Order>::blend_pix,
        comp_op_rgba_src_out    <ColorT,Order>::blend_pix,
        comp_op_rgba_dst_out    <ColorT,Order>::blend_pix,
        comp_op_rgba_src_atop   <ColorT,Order>::blend_pix,
        comp_op_rgba_dst_atop   <ColorT,Order>::blend_pix,
        comp_op_rgba_xor        <ColorT,Order>::blend_pix,
        comp_op_rgba_plus       <ColorT,Order>::blend_pix,
        //comp_op_rgba_minus      <ColorT,Order>::blend_pix,
        comp_op_rgba_multiply   <ColorT,Order>::blend_pix,
        comp_op_rgba_screen     <ColorT,Order>::blend_pix,
        comp_op_rgba_overlay    <ColorT,Order>::blend_pix,
        comp_op_rgba_darken     <ColorT,Order>::blend_pix,
        comp_op_rgba_lighten    <ColorT,Order>::blend_pix,
        comp_op_rgba_color_dodge<ColorT,Order>::blend_pix,
        comp_op_rgba_color_burn <ColorT,Order>::blend_pix,
        comp_op_rgba_hard_light <ColorT,Order>::blend_pix,
        comp_op_rgba_soft_light <ColorT,Order>::blend_pix,
        comp_op_rgba_difference <ColorT,Order>::blend_pix,
        comp_op_rgba_exclusion  <ColorT,Order>::blend_pix,
        //comp_op_rgba_contrast   <ColorT,Order>::blend_pix,
        //comp_op_rgba_invert     <ColorT,Order>::blend_pix,
        //comp_op_rgba_invert_rgb <ColorT,Order>::blend_pix,
        0
    };


    //==============================================================comp_op_e
    enum comp_op_e
    {
        comp_op_clear,         //----comp_op_clear
        comp_op_src,           //----comp_op_src
        comp_op_dst,           //----comp_op_dst
        comp_op_src_over,      //----comp_op_src_over
        comp_op_dst_over,      //----comp_op_dst_over
        comp_op_src_in,        //----comp_op_src_in
        comp_op_dst_in,        //----comp_op_dst_in
        comp_op_src_out,       //----comp_op_src_out
        comp_op_dst_out,       //----comp_op_dst_out
        comp_op_src_atop,      //----comp_op_src_atop
        comp_op_dst_atop,      //----comp_op_dst_atop
        comp_op_xor,           //----comp_op_xor
        comp_op_plus,          //----comp_op_plus
        //comp_op_minus,         //----comp_op_minus
        comp_op_multiply,      //----comp_op_multiply
        comp_op_screen,        //----comp_op_screen
        comp_op_overlay,       //----comp_op_overlay
        comp_op_darken,        //----comp_op_darken
        comp_op_lighten,       //----comp_op_lighten
        comp_op_color_dodge,   //----comp_op_color_dodge
        comp_op_color_burn,    //----comp_op_color_burn
        comp_op_hard_light,    //----comp_op_hard_light
        comp_op_soft_light,    //----comp_op_soft_light
        comp_op_difference,    //----comp_op_difference
        comp_op_exclusion,     //----comp_op_exclusion
        //comp_op_contrast,      //----comp_op_contrast
        //comp_op_invert,        //----comp_op_invert
        //comp_op_invert_rgb,    //----comp_op_invert_rgb

        end_of_comp_op_e
    };







    //====================================================comp_op_adaptor_rgba
    template<class ColorT, class Order> 
    struct comp_op_adaptor_rgba
    {
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;

        static AGG_INLINE void blend_pix(unsigned op, value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            comp_op_table_rgba<ColorT, Order>::g_comp_op_func[op](p, 
                color_type::multiply(r, a), 
                color_type::multiply(g, a), 
                color_type::multiply(b, a), 
                a, cover);
        }
    };

    //=========================================comp_op_adaptor_clip_to_dst_rgba
    template<class ColorT, class Order> 
    struct comp_op_adaptor_clip_to_dst_rgba
    {
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;

        static AGG_INLINE void blend_pix(unsigned op, value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            r = color_type::multiply(r, a);
            g = color_type::multiply(g, a);
            b = color_type::multiply(b, a);
            value_type da = p[Order::A];
            comp_op_table_rgba<ColorT, Order>::g_comp_op_func[op](p, 
                color_type::multiply(r, da), 
                color_type::multiply(g, da), 
                color_type::multiply(b, da), 
                color_type::multiply(a, da), cover);
        }
    };

    //================================================comp_op_adaptor_rgba_pre
    template<class ColorT, class Order> 
    struct comp_op_adaptor_rgba_pre
    {
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;

        static AGG_INLINE void blend_pix(unsigned op, value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            comp_op_table_rgba<ColorT, Order>::g_comp_op_func[op](p, r, g, b, a, cover);
        }
    };

    //=====================================comp_op_adaptor_clip_to_dst_rgba_pre
    template<class ColorT, class Order> 
    struct comp_op_adaptor_clip_to_dst_rgba_pre
    {
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;

        static AGG_INLINE void blend_pix(unsigned op, value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            value_type da = p[Order::A];
            comp_op_table_rgba<ColorT, Order>::g_comp_op_func[op](p, 
                color_type::multiply(r, da), 
                color_type::multiply(g, da), 
                color_type::multiply(b, da), 
                color_type::multiply(a, da), cover);
        }
    };

    //====================================================comp_op_adaptor_rgba_plain
    template<class ColorT, class Order> 
    struct comp_op_adaptor_rgba_plain
    {
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;

        static AGG_INLINE void blend_pix(unsigned op, value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            multiplier_rgba<ColorT, Order>::premultiply(p);
            comp_op_adaptor_rgba<ColorT, Order>::blend_pix(op, p, r, g, b, a, cover);
            multiplier_rgba<ColorT, Order>::demultiply(p);
        }
    };

    //=========================================comp_op_adaptor_clip_to_dst_rgba_plain
    template<class ColorT, class Order> 
    struct comp_op_adaptor_clip_to_dst_rgba_plain
    {
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;

        static AGG_INLINE void blend_pix(unsigned op, value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            multiplier_rgba<ColorT, Order>::premultiply(p);
            comp_op_adaptor_clip_to_dst_rgba<ColorT, Order>::blend_pix(op, p, r, g, b, a, cover);
            multiplier_rgba<ColorT, Order>::demultiply(p);
        }
    };

    //=======================================================comp_adaptor_rgba
    template<class BlenderPre> 
    struct comp_adaptor_rgba
    {
        typedef typename BlenderPre::color_type color_type;
        typedef typename BlenderPre::order_type order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;

        static AGG_INLINE void blend_pix(unsigned op, value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            BlenderPre::blend_pix(p, 
                color_type::multiply(r, a), 
                color_type::multiply(g, a), 
                color_type::multiply(b, a), 
                a, cover);
        }
    };

    //==========================================comp_adaptor_clip_to_dst_rgba
    template<class BlenderPre> 
    struct comp_adaptor_clip_to_dst_rgba
    {
        typedef typename BlenderPre::color_type color_type;
        typedef typename BlenderPre::order_type order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;

        static AGG_INLINE void blend_pix(unsigned op, value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            r = color_type::multiply(r, a);
            g = color_type::multiply(g, a);
            b = color_type::multiply(b, a);
            value_type da = p[order_type::A];
            BlenderPre::blend_pix(p, 
                color_type::multiply(r, da), 
                color_type::multiply(g, da), 
                color_type::multiply(b, da), 
                color_type::multiply(a, da), cover);
        }
    };

    //=======================================================comp_adaptor_rgba_pre
    template<class BlenderPre> 
    struct comp_adaptor_rgba_pre
    {
        typedef typename BlenderPre::color_type color_type;
        typedef typename BlenderPre::order_type order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;

        static AGG_INLINE void blend_pix(unsigned op, value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            BlenderPre::blend_pix(p, r, g, b, a, cover);
        }
    };

    //======================================comp_adaptor_clip_to_dst_rgba_pre
    template<class BlenderPre> 
    struct comp_adaptor_clip_to_dst_rgba_pre
    {
        typedef typename BlenderPre::color_type color_type;
        typedef typename BlenderPre::order_type order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;

        static AGG_INLINE void blend_pix(unsigned op, value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            unsigned da = p[order_type::A];
            BlenderPre::blend_pix(p, 
                color_type::multiply(r, da), 
                color_type::multiply(g, da), 
                color_type::multiply(b, da), 
                color_type::multiply(a, da), 
                cover);
        }
    };

    //=======================================================comp_adaptor_rgba_plain
    template<class BlenderPre> 
    struct comp_adaptor_rgba_plain
    {
        typedef typename BlenderPre::color_type color_type;
        typedef typename BlenderPre::order_type order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;

        static AGG_INLINE void blend_pix(unsigned op, value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            multiplier_rgba<color_type, order_type>::premultiply(p);
            comp_adaptor_rgba<BlenderPre>::blend_pix(op, p, r, g, b, a, cover);
            multiplier_rgba<color_type, order_type>::demultiply(p);
        }
    };

    //==========================================comp_adaptor_clip_to_dst_rgba_plain
    template<class BlenderPre> 
    struct comp_adaptor_clip_to_dst_rgba_plain
    {
        typedef typename BlenderPre::color_type color_type;
        typedef typename BlenderPre::order_type order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;

        static AGG_INLINE void blend_pix(unsigned op, value_type* p, 
            value_type r, value_type g, value_type b, value_type a, cover_type cover)
        {
            multiplier_rgba<color_type, order_type>::premultiply(p);
            comp_adaptor_clip_to_dst_rgba<BlenderPre>::blend_pix(op, p, r, g, b, a, cover);
            multiplier_rgba<color_type, order_type>::demultiply(p);
        }
    };


    //=================================================pixfmt_alpha_blend_rgba
    template<class Blender, class RenBuf> 
    class pixfmt_alpha_blend_rgba
    {
    public:
        typedef pixfmt_rgba_tag pixfmt_category;
        typedef RenBuf   rbuf_type;
        typedef typename rbuf_type::row_data row_data;
        typedef Blender  blender_type;
        typedef typename blender_type::color_type color_type;
        typedef typename blender_type::order_type order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        enum 
        {
            num_components = 4,
            pix_step = 4,
            pix_width = sizeof(value_type) * pix_step,
        };
        struct pixel_type
        {
            value_type c[num_components];

            void set(value_type r, value_type g, value_type b, value_type a)
            {
                c[order_type::R] = r;
                c[order_type::G] = g;
                c[order_type::B] = b;
                c[order_type::A] = a;
            }

            void set(const color_type& color)
            {
                set(color.r, color.g, color.b, color.a);
            }

            void get(value_type& r, value_type& g, value_type& b, value_type& a) const
            {
                r = c[order_type::R];
                g = c[order_type::G];
                b = c[order_type::B];
                a = c[order_type::A];
            }

            color_type get() const
            {
                return color_type(
                    c[order_type::R], 
                    c[order_type::G], 
                    c[order_type::B],
                    c[order_type::A]);
            }

            pixel_type* next()
            {
                return (pixel_type*)(c + pix_step);
            }

            const pixel_type* next() const
            {
                return (const pixel_type*)(c + pix_step);
            }

            pixel_type* advance(int n)
            {
                return (pixel_type*)(c + n * pix_step);
            }

            const pixel_type* advance(int n) const
            {
                return (const pixel_type*)(c + n * pix_step);
            }
        };

    private:
        //--------------------------------------------------------------------
        AGG_INLINE void blend_pix(pixel_type* p, const color_type& c, unsigned cover)
        {
            m_blender.blend_pix(p->c, c.r, c.g, c.b, c.a, cover);
        }

        //--------------------------------------------------------------------
        AGG_INLINE void blend_pix(pixel_type* p, const color_type& c)
        {
            m_blender.blend_pix(p->c, c.r, c.g, c.b, c.a);
        }

        //--------------------------------------------------------------------
        AGG_INLINE void copy_or_blend_pix(pixel_type* p, const color_type& c, unsigned cover)
        {
            if (!c.is_transparent())
            {
                if (c.is_opaque() && cover == cover_mask)
                {
                    p->set(c.r, c.g, c.b, c.a);
                }
                else
                {
                    m_blender.blend_pix(p->c, c.r, c.g, c.b, c.a, cover);
                }
            }
        }

        //--------------------------------------------------------------------
        AGG_INLINE void copy_or_blend_pix(pixel_type* p, const color_type& c)
        {
            if (!c.is_transparent())
            {
                if (c.is_opaque())
                {
                    p->set(c.r, c.g, c.b, c.a);
                }
                else
                {
                    m_blender.blend_pix(p->c, c.r, c.g, c.b, c.a);
                }
            }
        }

    public:
        //--------------------------------------------------------------------
        pixfmt_alpha_blend_rgba() : m_rbuf(0) {}
        explicit pixfmt_alpha_blend_rgba(rbuf_type& rb) : m_rbuf(&rb) {}
        void attach(rbuf_type& rb) { m_rbuf = &rb; }

        //--------------------------------------------------------------------
        template<class PixFmt>
        bool attach(PixFmt& pixf, int x1, int y1, int x2, int y2)
        {
            rect_i r(x1, y1, x2, y2);
            if (r.clip(rect_i(0, 0, pixf.width()-1, pixf.height()-1)))
            {
                int stride = pixf.stride();
                m_rbuf->attach(pixf.pix_ptr(r.x1, stride < 0 ? r.y2 : r.y1), 
                               (r.x2 - r.x1) + 1,
                               (r.y2 - r.y1) + 1,
                               stride);
                return true;
            }
            return false;
        }

        //--------------------------------------------------------------------
        AGG_INLINE unsigned width()  const { return m_rbuf->width();  }
        AGG_INLINE unsigned height() const { return m_rbuf->height(); }
        AGG_INLINE int      stride() const { return m_rbuf->stride(); }

        //--------------------------------------------------------------------
        AGG_INLINE       int8u* row_ptr(int y)       { return m_rbuf->row_ptr(y); }
        AGG_INLINE const int8u* row_ptr(int y) const { return m_rbuf->row_ptr(y); }
        AGG_INLINE row_data     row(int y)     const { return m_rbuf->row(y); }

        //--------------------------------------------------------------------
        AGG_INLINE int8u* pix_ptr(int x, int y) 
        { 
            return m_rbuf->row_ptr(y) + sizeof(value_type) * (x * pix_step);
        }

        AGG_INLINE const int8u* pix_ptr(int x, int y) const 
        { 
            return m_rbuf->row_ptr(y) + sizeof(value_type) * (x * pix_step);
        }

        // Return pointer to pixel value, forcing row to be allocated.
        AGG_INLINE pixel_type* pix_value_ptr(int x, int y, unsigned len) 
        {
            return (pixel_type*)(m_rbuf->row_ptr(x, y, len) + sizeof(value_type) * (x * pix_step));
        }

        // Return pointer to pixel value, or null if row not allocated.
        AGG_INLINE const pixel_type* pix_value_ptr(int x, int y) const 
        {
            int8u* p = m_rbuf->row_ptr(y);
            return p ? (pixel_type*)(p + sizeof(value_type) * (x * pix_step)) : 0;
        }

        // Get pixel pointer from raw buffer pointer.
        AGG_INLINE static pixel_type* pix_value_ptr(void* p) 
        {
            return (pixel_type*)p;
        }

        // Get pixel pointer from raw buffer pointer.
        AGG_INLINE static const pixel_type* pix_value_ptr(const void* p) 
        {
            return (const pixel_type*)p;
        }

        //--------------------------------------------------------------------
        AGG_INLINE static void write_plain_color(void* p, color_type c)
        {
            blender_type::set_plain_color(pix_value_ptr(p)->c, c);
        }

        //--------------------------------------------------------------------
        AGG_INLINE static color_type read_plain_color(const void* p)
        {
            return blender_type::get_plain_color(pix_value_ptr(p)->c);
        }

        //--------------------------------------------------------------------
        AGG_INLINE static void make_pix(int8u* p, const color_type& c)
        {
            ((pixel_type*)p)->set(c);
        }

        //--------------------------------------------------------------------
        AGG_INLINE color_type pixel(int x, int y) const
        {
            if (const pixel_type* p = pix_value_ptr(x, y))
            {
                return p->get();
            }
            return color_type::no_color();
        }

        //--------------------------------------------------------------------
        AGG_INLINE void copy_pixel(int x, int y, const color_type& c)
        {
            pix_value_ptr(x, y, 1)->set(c);
        }

        //--------------------------------------------------------------------
        AGG_INLINE void blend_pixel(int x, int y, const color_type& c, int8u cover)
        {
            copy_or_blend_pix(pix_value_ptr(x, y, 1), c, cover);
        }

        //--------------------------------------------------------------------
        AGG_INLINE void copy_hline(int x, int y, 
                                   unsigned len, 
                                   const color_type& c)
        {
            pixel_type v;
            v.set(c);
            pixel_type* p = pix_value_ptr(x, y, len);
            do
            {
                *p = v;
                p = p->next();
            }
            while (--len);
        }


        //--------------------------------------------------------------------
        AGG_INLINE void copy_vline(int x, int y,
                                   unsigned len, 
                                   const color_type& c)
        {
            pixel_type v;
            v.set(c);
            do
            {
                *pix_value_ptr(x, y++, 1) = v;
            }
            while (--len);
        }

        //--------------------------------------------------------------------
        void blend_hline(int x, int y,
                         unsigned len, 
                         const color_type& c,
                         int8u cover)
        {
            if (!c.is_transparent())
            {
                pixel_type* p = pix_value_ptr(x, y, len);
                if (c.is_opaque() && cover == cover_mask)
                {
                    pixel_type v;
                    v.set(c);
                    do
                    {
                        *p = v;
                        p = p->next();
                    }
                    while (--len);
                }
                else
                {
                    if (cover == cover_mask)
                    {
                        do
                        {
                            blend_pix(p, c);
                            p = p->next();
                        }
                        while (--len);
                    }
                    else
                    {
                        do
                        {
                            blend_pix(p, c, cover);
                            p = p->next();
                        }
                        while (--len);
                    }
                }
            }
        }


        //--------------------------------------------------------------------
        void blend_vline(int x, int y,
                         unsigned len, 
                         const color_type& c,
                         int8u cover)
        {
            if (!c.is_transparent())
            {
                if (c.is_opaque() && cover == cover_mask)
                {
                    pixel_type v;
                    v.set(c);
                    do
                    {
                        *pix_value_ptr(x, y++, 1) = v;
                    }
                    while (--len);
                }
                else
                {
                    if (cover == cover_mask)
                    {
                        do
                        {
                            blend_pix(pix_value_ptr(x, y++, 1), c, c.a);
                        }
                        while (--len);
                    }
                    else
                    {
                        do
                        {
                            blend_pix(pix_value_ptr(x, y++, 1), c, cover);
                        }
                        while (--len);
                    }
                }
            }
        }


        //--------------------------------------------------------------------
        void blend_solid_hspan(int x, int y,
                               unsigned len, 
                               const color_type& c,
                               const int8u* covers)
        {
            if (!c.is_transparent())
            {
                pixel_type* p = pix_value_ptr(x, y, len);
                do 
                {
                    if (c.is_opaque() && *covers == cover_mask)
                    {
                        p->set(c);
                    }
                    else
                    {
                        blend_pix(p, c, *covers);
                    }
                    p = p->next();
                    ++covers;
                }
                while (--len);
            }
        }


        //--------------------------------------------------------------------
        void blend_solid_vspan(int x, int y,
                               unsigned len, 
                               const color_type& c,
                               const int8u* covers)
        {
            if (!c.is_transparent())
            {
                do 
                {
                    pixel_type* p = pix_value_ptr(x, y++, 1);
                    if (c.is_opaque() && *covers == cover_mask)
                    {
                        p->set(c);
                    }
                    else
                    {
                        blend_pix(p, c, *covers);
                    }
                    ++covers;
                }
                while (--len);
            }
        }

        //--------------------------------------------------------------------
        void copy_color_hspan(int x, int y,
                              unsigned len, 
                              const color_type* colors)
        {
            pixel_type* p = pix_value_ptr(x, y, len);
            do 
            {
                p->set(*colors++);
                p = p->next();
            }
            while (--len);
        }


        //--------------------------------------------------------------------
        void copy_color_vspan(int x, int y,
                              unsigned len, 
                              const color_type* colors)
        {
            do 
            {
                pix_value_ptr(x, y++, 1)->set(*colors++);
            }
            while (--len);
        }

        //--------------------------------------------------------------------
        void blend_color_hspan(int x, int y,
                               unsigned len, 
                               const color_type* colors,
                               const int8u* covers,
                               int8u cover)
        {
            pixel_type* p = pix_value_ptr(x, y, len);
            if (covers)
            {
                do 
                {
                    copy_or_blend_pix(p, *colors++, *covers++);
                    p = p->next();
                }
                while (--len);
            }
            else
            {
                if (cover == cover_mask)
                {
                    do 
                    {
                        copy_or_blend_pix(p, *colors++);
                        p = p->next();
                    }
                    while (--len);
                }
                else
                {
                    do 
                    {
                        copy_or_blend_pix(p, *colors++, cover);
                        p = p->next();
                    }
                    while (--len);
                }
            }
        }

        //--------------------------------------------------------------------
        void blend_color_vspan(int x, int y,
                               unsigned len, 
                               const color_type* colors,
                               const int8u* covers,
                               int8u cover)
        {
            if (covers)
            {
                do 
                {
                    copy_or_blend_pix(pix_value_ptr(x, y++, 1), *colors++, *covers++);
                }
                while (--len);
            }
            else
            {
                if (cover == cover_mask)
                {
                    do 
                    {
                        copy_or_blend_pix(pix_value_ptr(x, y++, 1), *colors++);
                    }
                    while (--len);
                }
                else
                {
                    do 
                    {
                        copy_or_blend_pix(pix_value_ptr(x, y++, 1), *colors++, cover);
                    }
                    while (--len);
                }
            }
        }

        //--------------------------------------------------------------------
        template<class Function> void for_each_pixel(Function f)
        {
            for (unsigned y = 0; y < height(); ++y)
            {
                row_data r = m_rbuf->row(y);
                if (r.ptr)
                {
                    unsigned len = r.x2 - r.x1 + 1;
                    pixel_type* p = pix_value_ptr(r.x1, y, len);
                    do
                    {
                        f(p->c);
                        p = p->next();
                    }
                    while (--len);
                }
            }
        }

        //--------------------------------------------------------------------
        void premultiply()
        {
            for_each_pixel(multiplier_rgba<color_type, order_type>::premultiply);
        }

        //--------------------------------------------------------------------
        void demultiply()
        {
            for_each_pixel(multiplier_rgba<color_type, order_type>::demultiply);
        }

        //--------------------------------------------------------------------
        template<class GammaLut> void apply_gamma_dir(const GammaLut& g)
        {
            for_each_pixel(apply_gamma_dir_rgba<color_type, order_type, GammaLut>(g));
        }

        //--------------------------------------------------------------------
        template<class GammaLut> void apply_gamma_inv(const GammaLut& g)
        {
            for_each_pixel(apply_gamma_inv_rgba<color_type, order_type, GammaLut>(g));
        }

        //--------------------------------------------------------------------
        template<class RenBuf2> void copy_from(const RenBuf2& from, 
                                               int xdst, int ydst,
                                               int xsrc, int ysrc,
                                               unsigned len)
        {
            if (const int8u* p = from.row_ptr(ysrc))
            {
                memmove(m_rbuf->row_ptr(xdst, ydst, len) + xdst * pix_width, 
                        p + xsrc * pix_width, 
                        len * pix_width);
            }
        }

        //--------------------------------------------------------------------
        // Blend from another RGBA surface.
        template<class SrcPixelFormatRenderer>
        void blend_from(const SrcPixelFormatRenderer& from, 
                        int xdst, int ydst,
                        int xsrc, int ysrc,
                        unsigned len,
                        int8u cover)
        {
            typedef typename SrcPixelFormatRenderer::pixel_type src_pixel_type;

            if (const src_pixel_type* psrc = from.pix_value_ptr(xsrc, ysrc))
            {
                pixel_type* pdst = pix_value_ptr(xdst, ydst, len);
                int srcinc = 1;
                int dstinc = 1;

                if (xdst > xsrc)
                {
                    psrc = psrc->advance(len - 1);
                    pdst = pdst->advance(len - 1);
                    srcinc = -1;
                    dstinc = -1;
                }

                if (cover == cover_mask)
                {
                    do 
                    {
                        copy_or_blend_pix(pdst, psrc->get());
                        psrc = psrc->advance(srcinc);
                        pdst = pdst->advance(dstinc);
                    }
                    while (--len);
                }
                else
                {
                    do 
                    {
                        copy_or_blend_pix(pdst, psrc->get(), cover);
                        psrc = psrc->advance(srcinc);
                        pdst = pdst->advance(dstinc);
                    }
                    while (--len);
                }
            }
        }

        //--------------------------------------------------------------------
        // Combine single color with grayscale surface and blend.
        template<class SrcPixelFormatRenderer>
        void blend_from_color(const SrcPixelFormatRenderer& from, 
                              const color_type& color,
                              int xdst, int ydst,
                              int xsrc, int ysrc,
                              unsigned len,
                              int8u cover)
        {
            typedef typename SrcPixelFormatRenderer::pixel_type src_pixel_type;
            typedef typename SrcPixelFormatRenderer::color_type src_color_type;

            if (const src_pixel_type* psrc = from.pix_value_ptr(xsrc, ysrc))
            {
                pixel_type* pdst = pix_value_ptr(xdst, ydst, len);

                do 
                {
                    copy_or_blend_pix(pdst, color, 
                        src_color_type::scale_cover(cover, psrc->c[0]));
                    psrc = psrc->next();
                    pdst = pdst->next();
                }
                while (--len);
            }
        }

        //--------------------------------------------------------------------
        // Blend from color table, using grayscale surface as indexes into table.
        // Obviously, this only works for integer value types.
        template<class SrcPixelFormatRenderer>
        void blend_from_lut(const SrcPixelFormatRenderer& from, 
                            const color_type* color_lut,
                            int xdst, int ydst,
                            int xsrc, int ysrc,
                            unsigned len,
                            int8u cover)
        {
            typedef typename SrcPixelFormatRenderer::pixel_type src_pixel_type;

            if (const src_pixel_type* psrc = from.pix_value_ptr(xsrc, ysrc))
            {
                pixel_type* pdst = pix_value_ptr(xdst, ydst, len);

                if (cover == cover_mask)
                {
                    do 
                    {
                        copy_or_blend_pix(pdst, color_lut[psrc->c[0]]);
                        psrc = psrc->next();
                        pdst = pdst->next();
                    }
                    while (--len);
                }
                else
                {
                    do 
                    {
                        copy_or_blend_pix(pdst, color_lut[psrc->c[0]], cover);
                        psrc = psrc->next();
                        pdst = pdst->next();
                    }
                    while (--len);
                }
            }
        }

    private:
        rbuf_type* m_rbuf;
        Blender    m_blender;
    };

    //================================================pixfmt_custom_blend_rgba
    template<class Blender, class RenBuf> class pixfmt_custom_blend_rgba
    {
    public:
        typedef pixfmt_rgba_tag pixfmt_category;
        typedef RenBuf   rbuf_type;
        typedef typename rbuf_type::row_data row_data;
        typedef Blender  blender_type;
        typedef typename blender_type::color_type color_type;
        typedef typename blender_type::order_type order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        enum 
        {
            num_components = 4,
            pix_step = 4,
            pix_width  = sizeof(value_type) * pix_step,
        };
        struct pixel_type
        {
            value_type c[num_components];

            void set(value_type r, value_type g, value_type b, value_type a)
            {
                c[order_type::R] = r;
                c[order_type::G] = g;
                c[order_type::B] = b;
                c[order_type::A] = a;
            }

            void set(const color_type& color)
            {
                set(color.r, color.g, color.b, color.a);
            }

            void get(value_type& r, value_type& g, value_type& b, value_type& a) const
            {
                r = c[order_type::R];
                g = c[order_type::G];
                b = c[order_type::B];
                a = c[order_type::A];
            }

            color_type get() const
            {
                return color_type(
                    c[order_type::R], 
                    c[order_type::G], 
                    c[order_type::B],
                    c[order_type::A]);
            }

            pixel_type* next()
            {
                return (pixel_type*)(c + pix_step);
            }

            const pixel_type* next() const
            {
                return (const pixel_type*)(c + pix_step);
            }

            pixel_type* advance(int n)
            {
                return (pixel_type*)(c + n * pix_step);
            }

            const pixel_type* advance(int n) const
            {
                return (const pixel_type*)(c + n * pix_step);
            }
        };


    private:
        //--------------------------------------------------------------------
        AGG_INLINE void blend_pix(pixel_type* p, const color_type& c, unsigned cover = cover_full)
        {
            m_blender.blend_pix(m_comp_op, p->c, c.r, c.g, c.b, c.a, cover);
        }

        //--------------------------------------------------------------------
        AGG_INLINE void copy_or_blend_pix(pixel_type* p, const color_type& c, unsigned cover = cover_full)
        {
            if (!c.is_transparent())
            {
                if (c.is_opaque() && cover == cover_mask)
                {
                    p->set(c.r, c.g, c.b, c.a);
                }
                else
                {
                    blend_pix(p, c, cover);
                }
            }
        }

    public:
        //--------------------------------------------------------------------
        pixfmt_custom_blend_rgba() : m_rbuf(0), m_comp_op(3) {}
        explicit pixfmt_custom_blend_rgba(rbuf_type& rb, unsigned comp_op=3) : 
            m_rbuf(&rb),
            m_comp_op(comp_op)
        {}
        void attach(rbuf_type& rb) { m_rbuf = &rb; }

        //--------------------------------------------------------------------
        template<class PixFmt>
        bool attach(PixFmt& pixf, int x1, int y1, int x2, int y2)
        {
            rect_i r(x1, y1, x2, y2);
            if (r.clip(rect_i(0, 0, pixf.width()-1, pixf.height()-1)))
            {
                int stride = pixf.stride();
                m_rbuf->attach(pixf.pix_ptr(r.x1, stride < 0 ? r.y2 : r.y1), 
                               (r.x2 - r.x1) + 1,
                               (r.y2 - r.y1) + 1,
                               stride);
                return true;
            }
            return false;
        }

        //--------------------------------------------------------------------
        void comp_op(unsigned op) { m_comp_op = op; }
        unsigned comp_op() const  { return m_comp_op; }

        //--------------------------------------------------------------------
        AGG_INLINE unsigned width()  const { return m_rbuf->width();  }
        AGG_INLINE unsigned height() const { return m_rbuf->height(); }
        AGG_INLINE int      stride() const { return m_rbuf->stride(); }

        //--------------------------------------------------------------------
        AGG_INLINE       int8u* row_ptr(int y)       { return m_rbuf->row_ptr(y); }
        AGG_INLINE const int8u* row_ptr(int y) const { return m_rbuf->row_ptr(y); }
        AGG_INLINE row_data     row(int y)     const { return m_rbuf->row(y); }

        //--------------------------------------------------------------------
        AGG_INLINE int8u* pix_ptr(int x, int y) 
        { 
            return m_rbuf->row_ptr(y) + sizeof(value_type) * (x * pix_step);
        }

        AGG_INLINE const int8u* pix_ptr(int x, int y) const 
        { 
            return m_rbuf->row_ptr(y) + sizeof(value_type) * (x * pix_step);
        }

        // Return pointer to pixel value, forcing row to be allocated.
        AGG_INLINE pixel_type* pix_value_ptr(int x, int y, unsigned len) 
        {
            return (pixel_type*)(m_rbuf->row_ptr(x, y, len) + sizeof(value_type) * (x * pix_step));
        }

        // Return pointer to pixel value, or null if row not allocated.
        AGG_INLINE const pixel_type* pix_value_ptr(int x, int y) const 
        {
            int8u* p = m_rbuf->row_ptr(y);
            return p ? (pixel_type*)(p + sizeof(value_type) * (x * pix_step)) : 0;
        }

        // Get pixel pointer from raw buffer pointer.
        AGG_INLINE static pixel_type* pix_value_ptr(void* p) 
        {
            return (pixel_type*)p;
        }

        // Get pixel pointer from raw buffer pointer.
        AGG_INLINE static const pixel_type* pix_value_ptr(const void* p) 
        {
            return (const pixel_type*)p;
        }

        //--------------------------------------------------------------------
        AGG_INLINE static void make_pix(int8u* p, const color_type& c)
        {
            ((pixel_type*)p)->set(c);
        }

        //--------------------------------------------------------------------
        AGG_INLINE color_type pixel(int x, int y) const
        {
            if (const pixel_type* p = pix_value_ptr(x, y))
            {
                return p->get();
            }
            return color_type::no_color();
        }

        //--------------------------------------------------------------------
        AGG_INLINE void copy_pixel(int x, int y, const color_type& c)
        {
            make_pix(pix_value_ptr(x, y, 1), c);
        }

        //--------------------------------------------------------------------
        AGG_INLINE void blend_pixel(int x, int y, const color_type& c, int8u cover)
        {
            blend_pix(pix_value_ptr(x, y, 1), c, cover);
        }

        //--------------------------------------------------------------------
        AGG_INLINE void copy_hline(int x, int y, 
                                   unsigned len, 
                                   const color_type& c)
        {
            pixel_type v;
            v.set(c);
            pixel_type* p = pix_value_ptr(x, y, len);
            do
            {
                *p = v;
                p = p->next();
            }
            while (--len);
        }


        //--------------------------------------------------------------------
        AGG_INLINE void copy_vline(int x, int y,
                                   unsigned len, 
                                   const color_type& c)
        {
            pixel_type v;
            v.set(c);
            do
            {
                *pix_value_ptr(x, y++, 1) = v;
            }
            while (--len);
        }

        //--------------------------------------------------------------------
        void blend_hline(int x, int y, unsigned len, 
                         const color_type& c, int8u cover)
        {

            pixel_type* p = pix_value_ptr(x, y, len);
            do
            {
                blend_pix(p, c, cover);
                p = p->next();
            }
            while (--len);
        }

        //--------------------------------------------------------------------
        void blend_vline(int x, int y, unsigned len, 
                         const color_type& c, int8u cover)
        {
            do
            {
                blend_pix(pix_value_ptr(x, y++, 1), c, cover);
            }
            while (--len);
        }

        //--------------------------------------------------------------------
        void blend_solid_hspan(int x, int y, unsigned len, 
                               const color_type& c, const int8u* covers)
        {
            pixel_type* p = pix_value_ptr(x, y, len);

            do 
            {
                blend_pix(p, c, *covers++);
                p = p->next();
            }
            while (--len);
        }

        //--------------------------------------------------------------------
        void blend_solid_vspan(int x, int y, unsigned len, 
                               const color_type& c, const int8u* covers)
        {
            do 
            {
                blend_pix(pix_value_ptr(x, y++, 1), c, *covers++);
            }
            while (--len);
        }

        //--------------------------------------------------------------------
        void copy_color_hspan(int x, int y,
                              unsigned len, 
                              const color_type* colors)
        {
            pixel_type* p = pix_value_ptr(x, y, len);

            do 
            {
                p->set(*colors++);
                p = p->next();
            }
            while (--len);
        }

        //--------------------------------------------------------------------
        void copy_color_vspan(int x, int y,
                              unsigned len, 
                              const color_type* colors)
        {
            do 
            {
                pix_value_ptr(x, y++, 1)->set(*colors++);
            }
            while (--len);
        }

        //--------------------------------------------------------------------
        void blend_color_hspan(int x, int y, unsigned len, 
                               const color_type* colors, 
                               const int8u* covers,
                               int8u cover)
        {
            pixel_type* p = pix_value_ptr(x, y, len);

            do 
            {
                blend_pix(p, *colors++, covers ? *covers++ : cover);
                p = p->next();
            }
            while (--len);
        }

        //--------------------------------------------------------------------
        void blend_color_vspan(int x, int y, unsigned len, 
                               const color_type* colors, 
                               const int8u* covers,
                               int8u cover)
        {
            do 
            {
                blend_pix(pix_value_ptr(x, y++, 1), *colors++, covers ? *covers++ : cover);
            }
            while (--len);

        }

        //--------------------------------------------------------------------
        template<class Function> void for_each_pixel(Function f)
        {
            unsigned y;
            for (y = 0; y < height(); ++y)
            {
                row_data r = m_rbuf->row(y);
                if (r.ptr)
                {
                    unsigned len = r.x2 - r.x1 + 1;
                    pixel_type* p = pix_value_ptr(r.x1, y, len);
                    do
                    {
                        f(p->c);
                        p = p->next();
                    }
                    while (--len);
                }
            }
        }

        //--------------------------------------------------------------------
        void premultiply()
        {
            for_each_pixel(multiplier_rgba<color_type, order_type>::premultiply);
        }

        //--------------------------------------------------------------------
        void demultiply()
        {
            for_each_pixel(multiplier_rgba<color_type, order_type>::demultiply);
        }

        //--------------------------------------------------------------------
        template<class GammaLut> void apply_gamma_dir(const GammaLut& g)
        {
            for_each_pixel(apply_gamma_dir_rgba<color_type, order_type, GammaLut>(g));
        }

        //--------------------------------------------------------------------
        template<class GammaLut> void apply_gamma_inv(const GammaLut& g)
        {
            for_each_pixel(apply_gamma_inv_rgba<color_type, order_type, GammaLut>(g));
        }

        //--------------------------------------------------------------------
        template<class RenBuf2> void copy_from(const RenBuf2& from, 
                                               int xdst, int ydst,
                                               int xsrc, int ysrc,
                                               unsigned len)
        {
            if (const int8u* p = from.row_ptr(ysrc))
            {
                memmove(m_rbuf->row_ptr(xdst, ydst, len) + xdst * pix_width, 
                        p + xsrc * pix_width, 
                        len * pix_width);
            }
        }

        //--------------------------------------------------------------------
        // Blend from another RGBA surface.
        template<class SrcPixelFormatRenderer> 
        void blend_from(const SrcPixelFormatRenderer& from, 
                        int xdst, int ydst,
                        int xsrc, int ysrc,
                        unsigned len,
                        int8u cover)
        {
            typedef typename SrcPixelFormatRenderer::pixel_type src_pixel_type;

            if (const src_pixel_type* psrc = from.pix_value_ptr(xsrc, ysrc))
            {
                pixel_type* pdst = pix_value_ptr(xdst, ydst, len);
                int srcinc = 1;
                int dstinc = 1;

                if (xdst > xsrc)
                {
                    psrc = psrc->advance(len - 1);
                    pdst = pdst->advance(len - 1);
                    srcinc = -1;
                    dstinc = -1;
                }

                do 
                {
                    blend_pix(pdst, psrc->get(), cover);
                    psrc = psrc->advance(srcinc);
                    pdst = pdst->advance(dstinc);
                }
                while (--len);
            }
        }

        //--------------------------------------------------------------------
        // Blend from single color, using grayscale surface as alpha channel.
        template<class SrcPixelFormatRenderer>
        void blend_from_color(const SrcPixelFormatRenderer& from, 
                              const color_type& color,
                              int xdst, int ydst,
                              int xsrc, int ysrc,
                              unsigned len,
                              int8u cover)
        {
            typedef typename SrcPixelFormatRenderer::pixel_type src_pixel_type;
            typedef typename SrcPixelFormatRenderer::color_type src_color_type;

            if (const src_pixel_type* psrc = from.pix_value_ptr(xsrc, ysrc))
            {
                pixel_type* pdst = pix_value_ptr(xdst, ydst, len);

                do 
                {
                    blend_pix(pdst, color,
                        src_color_type::scale_cover(cover, psrc->c[0]));
                    psrc = psrc->next();
                    pdst = pdst->next();
                }
                while (--len);
            }
        }

        //--------------------------------------------------------------------
        // Blend from color table, using grayscale surface as indexes into table.
        // Obviously, this only works for integer value types.
        template<class SrcPixelFormatRenderer>
        void blend_from_lut(const SrcPixelFormatRenderer& from, 
                            const color_type* color_lut,
                            int xdst, int ydst,
                            int xsrc, int ysrc,
                            unsigned len,
                            int8u cover)
        {
            typedef typename SrcPixelFormatRenderer::pixel_type src_pixel_type;

            if (const src_pixel_type* psrc = from.pix_value_ptr(xsrc, ysrc))
            {
                pixel_type* pdst = pix_value_ptr(xdst, ydst, len);

                do 
                {
                    blend_pix(pdst, color_lut[psrc->c[0]], cover);
                    psrc = psrc->next();
                    pdst = pdst->next();
                }
                while (--len);
            }
        }

    private:
        rbuf_type* m_rbuf;
        Blender m_blender;
        unsigned m_comp_op;
    };


    //-----------------------------------------------------------------------
    typedef blender_rgba<rgba8, order_rgba> blender_rgba32;
    typedef blender_rgba<rgba8, order_argb> blender_argb32;
    typedef blender_rgba<rgba8, order_abgr> blender_abgr32;
    typedef blender_rgba<rgba8, order_bgra> blender_bgra32;

    typedef blender_rgba<srgba8, order_rgba> blender_srgba32;
    typedef blender_rgba<srgba8, order_argb> blender_sargb32;
    typedef blender_rgba<srgba8, order_abgr> blender_sabgr32;
    typedef blender_rgba<srgba8, order_bgra> blender_sbgra32;

    typedef blender_rgba_pre<rgba8, order_rgba> blender_rgba32_pre;
    typedef blender_rgba_pre<rgba8, order_argb> blender_argb32_pre;
    typedef blender_rgba_pre<rgba8, order_abgr> blender_abgr32_pre;
    typedef blender_rgba_pre<rgba8, order_bgra> blender_bgra32_pre;

    typedef blender_rgba_pre<srgba8, order_rgba> blender_srgba32_pre;
    typedef blender_rgba_pre<srgba8, order_argb> blender_sargb32_pre;
    typedef blender_rgba_pre<srgba8, order_abgr> blender_sabgr32_pre;
    typedef blender_rgba_pre<srgba8, order_bgra> blender_sbgra32_pre;

    typedef blender_rgba_plain<rgba8, order_rgba> blender_rgba32_plain;
    typedef blender_rgba_plain<rgba8, order_argb> blender_argb32_plain;
    typedef blender_rgba_plain<rgba8, order_abgr> blender_abgr32_plain;
    typedef blender_rgba_plain<rgba8, order_bgra> blender_bgra32_plain;

    typedef blender_rgba_plain<srgba8, order_rgba> blender_srgba32_plain;
    typedef blender_rgba_plain<srgba8, order_argb> blender_sargb32_plain;
    typedef blender_rgba_plain<srgba8, order_abgr> blender_sabgr32_plain;
    typedef blender_rgba_plain<srgba8, order_bgra> blender_sbgra32_plain;

    typedef blender_rgba<rgba16, order_rgba> blender_rgba64;
    typedef blender_rgba<rgba16, order_argb> blender_argb64;
    typedef blender_rgba<rgba16, order_abgr> blender_abgr64;
    typedef blender_rgba<rgba16, order_bgra> blender_bgra64;

    typedef blender_rgba_pre<rgba16, order_rgba> blender_rgba64_pre;
    typedef blender_rgba_pre<rgba16, order_argb> blender_argb64_pre;
    typedef blender_rgba_pre<rgba16, order_abgr> blender_abgr64_pre;
    typedef blender_rgba_pre<rgba16, order_bgra> blender_bgra64_pre;

	typedef blender_rgba_plain<rgba16, order_rgba> blender_rgba64_plain;
	typedef blender_rgba_plain<rgba16, order_argb> blender_argb64_plain;
	typedef blender_rgba_plain<rgba16, order_abgr> blender_abgr64_plain;
	typedef blender_rgba_plain<rgba16, order_bgra> blender_bgra64_plain;

	typedef blender_rgba<rgba32, order_rgba> blender_rgba128;
    typedef blender_rgba<rgba32, order_argb> blender_argb128;
    typedef blender_rgba<rgba32, order_abgr> blender_abgr128;
    typedef blender_rgba<rgba32, order_bgra> blender_bgra128;

    typedef blender_rgba_pre<rgba32, order_rgba> blender_rgba128_pre;
    typedef blender_rgba_pre<rgba32, order_argb> blender_argb128_pre;
    typedef blender_rgba_pre<rgba32, order_abgr> blender_abgr128_pre;
    typedef blender_rgba_pre<rgba32, order_bgra> blender_bgra128_pre;

    typedef blender_rgba_plain<rgba32, order_rgba> blender_rgba128_plain;
    typedef blender_rgba_plain<rgba32, order_argb> blender_argb128_plain;
    typedef blender_rgba_plain<rgba32, order_abgr> blender_abgr128_plain;
    typedef blender_rgba_plain<rgba32, order_bgra> blender_bgra128_plain;


    //-----------------------------------------------------------------------
    typedef pixfmt_alpha_blend_rgba<blender_rgba32, rendering_buffer> pixfmt_rgba32;
    typedef pixfmt_alpha_blend_rgba<blender_argb32, rendering_buffer> pixfmt_argb32;
    typedef pixfmt_alpha_blend_rgba<blender_abgr32, rendering_buffer> pixfmt_abgr32;
    typedef pixfmt_alpha_blend_rgba<blender_bgra32, rendering_buffer> pixfmt_bgra32;

    typedef pixfmt_alpha_blend_rgba<blender_srgba32, rendering_buffer> pixfmt_srgba32;
    typedef pixfmt_alpha_blend_rgba<blender_sargb32, rendering_buffer> pixfmt_sargb32;
    typedef pixfmt_alpha_blend_rgba<blender_sabgr32, rendering_buffer> pixfmt_sabgr32;
    typedef pixfmt_alpha_blend_rgba<blender_sbgra32, rendering_buffer> pixfmt_sbgra32;

    typedef pixfmt_alpha_blend_rgba<blender_rgba32_pre, rendering_buffer> pixfmt_rgba32_pre;
    typedef pixfmt_alpha_blend_rgba<blender_argb32_pre, rendering_buffer> pixfmt_argb32_pre;
    typedef pixfmt_alpha_blend_rgba<blender_abgr32_pre, rendering_buffer> pixfmt_abgr32_pre;
    typedef pixfmt_alpha_blend_rgba<blender_bgra32_pre, rendering_buffer> pixfmt_bgra32_pre;

    typedef pixfmt_alpha_blend_rgba<blender_srgba32_pre, rendering_buffer> pixfmt_srgba32_pre;
    typedef pixfmt_alpha_blend_rgba<blender_sargb32_pre, rendering_buffer> pixfmt_sargb32_pre;
    typedef pixfmt_alpha_blend_rgba<blender_sabgr32_pre, rendering_buffer> pixfmt_sabgr32_pre;
    typedef pixfmt_alpha_blend_rgba<blender_sbgra32_pre, rendering_buffer> pixfmt_sbgra32_pre;

    typedef pixfmt_alpha_blend_rgba<blender_rgba32_plain, rendering_buffer> pixfmt_rgba32_plain;
    typedef pixfmt_alpha_blend_rgba<blender_argb32_plain, rendering_buffer> pixfmt_argb32_plain;
    typedef pixfmt_alpha_blend_rgba<blender_abgr32_plain, rendering_buffer> pixfmt_abgr32_plain;
    typedef pixfmt_alpha_blend_rgba<blender_bgra32_plain, rendering_buffer> pixfmt_bgra32_plain;

    typedef pixfmt_alpha_blend_rgba<blender_srgba32_plain, rendering_buffer> pixfmt_srgba32_plain;
    typedef pixfmt_alpha_blend_rgba<blender_sargb32_plain, rendering_buffer> pixfmt_sargb32_plain;
    typedef pixfmt_alpha_blend_rgba<blender_sabgr32_plain, rendering_buffer> pixfmt_sabgr32_plain;
    typedef pixfmt_alpha_blend_rgba<blender_sbgra32_plain, rendering_buffer> pixfmt_sbgra32_plain;

    typedef pixfmt_alpha_blend_rgba<blender_rgba64, rendering_buffer> pixfmt_rgba64;
    typedef pixfmt_alpha_blend_rgba<blender_argb64, rendering_buffer> pixfmt_argb64;
    typedef pixfmt_alpha_blend_rgba<blender_abgr64, rendering_buffer> pixfmt_abgr64;
    typedef pixfmt_alpha_blend_rgba<blender_bgra64, rendering_buffer> pixfmt_bgra64;

    typedef pixfmt_alpha_blend_rgba<blender_rgba64_pre, rendering_buffer> pixfmt_rgba64_pre;
    typedef pixfmt_alpha_blend_rgba<blender_argb64_pre, rendering_buffer> pixfmt_argb64_pre;
    typedef pixfmt_alpha_blend_rgba<blender_abgr64_pre, rendering_buffer> pixfmt_abgr64_pre;
    typedef pixfmt_alpha_blend_rgba<blender_bgra64_pre, rendering_buffer> pixfmt_bgra64_pre;

	typedef pixfmt_alpha_blend_rgba<blender_rgba64_plain, rendering_buffer> pixfmt_rgba64_plain;
	typedef pixfmt_alpha_blend_rgba<blender_argb64_plain, rendering_buffer> pixfmt_argb64_plain;
	typedef pixfmt_alpha_blend_rgba<blender_abgr64_plain, rendering_buffer> pixfmt_abgr64_plain;
	typedef pixfmt_alpha_blend_rgba<blender_bgra64_plain, rendering_buffer> pixfmt_bgra64_plain;

	typedef pixfmt_alpha_blend_rgba<blender_rgba128, rendering_buffer> pixfmt_rgba128;
    typedef pixfmt_alpha_blend_rgba<blender_argb128, rendering_buffer> pixfmt_argb128;
    typedef pixfmt_alpha_blend_rgba<blender_abgr128, rendering_buffer> pixfmt_abgr128;
    typedef pixfmt_alpha_blend_rgba<blender_bgra128, rendering_buffer> pixfmt_bgra128;

    typedef pixfmt_alpha_blend_rgba<blender_rgba128_pre, rendering_buffer> pixfmt_rgba128_pre;
    typedef pixfmt_alpha_blend_rgba<blender_argb128_pre, rendering_buffer> pixfmt_argb128_pre;
    typedef pixfmt_alpha_blend_rgba<blender_abgr128_pre, rendering_buffer> pixfmt_abgr128_pre;
    typedef pixfmt_alpha_blend_rgba<blender_bgra128_pre, rendering_buffer> pixfmt_bgra128_pre;

    typedef pixfmt_alpha_blend_rgba<blender_rgba128_plain, rendering_buffer> pixfmt_rgba128_plain;
    typedef pixfmt_alpha_blend_rgba<blender_argb128_plain, rendering_buffer> pixfmt_argb128_plain;
    typedef pixfmt_alpha_blend_rgba<blender_abgr128_plain, rendering_buffer> pixfmt_abgr128_plain;
    typedef pixfmt_alpha_blend_rgba<blender_bgra128_plain, rendering_buffer> pixfmt_bgra128_plain;

}

#endif

