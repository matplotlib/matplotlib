#ifndef __AGG_WORKAROUND_H__
#define __AGG_WORKAROUND_H__

#include "agg_pixfmt_rgba.h"

/**********************************************************************
 WORKAROUND: This class is to workaround a bug in Agg SVN where the
 blending of RGBA32 pixels does not preserve enough precision
*/

template<class ColorT, class Order>
struct fixed_blender_rgba_pre : agg::conv_rgba_pre<ColorT, Order>
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

    //--------------------------------------------------------------------
    static AGG_INLINE void blend_pix(value_type* p,
                                     value_type cr, value_type cg, value_type cb,
                                     value_type alpha, agg::cover_type cover)
    {
        blend_pix(p,
                  color_type::mult_cover(cr, cover),
                  color_type::mult_cover(cg, cover),
                  color_type::mult_cover(cb, cover),
                  color_type::mult_cover(alpha, cover));
    }

    //--------------------------------------------------------------------
    static AGG_INLINE void blend_pix(value_type* p,
                                     value_type cr, value_type cg, value_type cb,
                                     value_type alpha)
    {
        alpha = base_mask - alpha;
        p[Order::R] = (value_type)(((p[Order::R] * alpha) >> base_shift) + cr);
        p[Order::G] = (value_type)(((p[Order::G] * alpha) >> base_shift) + cg);
        p[Order::B] = (value_type)(((p[Order::B] * alpha) >> base_shift) + cb);
        p[Order::A] = (value_type)(base_mask - ((alpha * (base_mask - p[Order::A])) >> base_shift));
    }
};


template<class ColorT, class Order>
struct fixed_blender_rgba_plain : agg::conv_rgba_plain<ColorT, Order>
{
    typedef ColorT color_type;
    typedef Order order_type;
    typedef typename color_type::value_type value_type;
    typedef typename color_type::calc_type calc_type;
    typedef typename color_type::long_type long_type;
    enum base_scale_e { base_shift = color_type::base_shift };

    //--------------------------------------------------------------------
    static AGG_INLINE void blend_pix(value_type* p,
                                     value_type cr, value_type cg, value_type cb, value_type alpha, agg::cover_type cover)
    {
        blend_pix(p, cr, cg, cb, color_type::mult_cover(alpha, cover));
    }

    //--------------------------------------------------------------------
    static AGG_INLINE void blend_pix(value_type* p,
                                     value_type cr, value_type cg, value_type cb, value_type alpha)
    {
        if(alpha == 0) return;
        calc_type a = p[Order::A];
        calc_type r = p[Order::R] * a;
        calc_type g = p[Order::G] * a;
        calc_type b = p[Order::B] * a;
        a = ((alpha + a) << base_shift) - alpha * a;
        p[Order::A] = (value_type)(a >> base_shift);
        p[Order::R] = (value_type)((((cr << base_shift) - r) * alpha + (r << base_shift)) / a);
        p[Order::G] = (value_type)((((cg << base_shift) - g) * alpha + (g << base_shift)) / a);
        p[Order::B] = (value_type)((((cb << base_shift) - b) * alpha + (b << base_shift)) / a);
    }
};

#endif
