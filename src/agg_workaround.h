#ifndef MPL_AGG_WORKAROUND_H
#define MPL_AGG_WORKAROUND_H

#include "agg_pixfmt_rgba.h"

/**********************************************************************
 WORKAROUND: This class is to workaround a bug in Agg SVN where the
 blending of RGBA8 pixels does not preserve enough precision
*/

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

	// The following code does not accurately transform to higher bit depth
	// TODO: Improve the accuracy of this code when we are prepared to regenerate all baseline images
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


/**********************************************************************
 WORKAROUND: As above, this class is to workaround a bug in Agg where
 the plain->plain blending of RGBA8 pixels is forced back to 8-bit
 values between steps, which degrades the calculation.  This class
 supports compositing operators (aka blend modes), and falls back to
 the above class for normal alpha blending.
*/

template<class Order>
struct fixed_comp_op_adaptor_rgba8_plain
{
    typedef agg::rgba8 color_type;
    typedef Order order_type;
    typedef typename color_type::value_type value_type;
    typedef typename color_type::calc_type calc_type;
    typedef typename color_type::long_type long_type;

    static AGG_INLINE void blend_pix(unsigned op, value_type* p,
        value_type r, value_type g, value_type b, value_type a, agg::cover_type cover)
    {
        if(op == agg::comp_op_src_over)
        {
            // This operator is simply normal alpha blending, so fall back to the above class
	    // We need to special-case fully opaque to maintain past behavior
            if(a == 255 && cover == agg::cover_full) {
                p[Order::A] = a;
                p[Order::R] = r;
                p[Order::G] = g;
                p[Order::B] = b;
                return;
            }
            fixed_blender_rgba_plain<agg::rgba8, Order>::blend_pix(p, r, g, b, a, cover);
        }
        else
        {
            if(a == 0 || cover == 0) return;

            // Convert to RGBA32 (floats) and premultiply
            float d[4];
            for(int i = 0; i < 4; i++) d[i] = agg::rgba8::to_double(p[i]);
            agg::multiplier_rgba<agg::rgba32, Order>::premultiply(d);

            double sa = agg::rgba8::to_double(a);
            double sr = agg::rgba8::to_double(r) * sa;
            double sg = agg::rgba8::to_double(g) * sa;
            double sb = agg::rgba8::to_double(b) * sa;

            // Blend as floats (internally, doubles)
            agg::comp_op_table_rgba<agg::rgba32, Order>::g_comp_op_func[op](d, sr, sg, sb, sa, cover);

            // Demultiply and return to RGBA8 space
            agg::multiplier_rgba<agg::rgba32, Order>::demultiply(d);
            for(int i = 0; i < 4; i++) p[i] = agg::rgba8::from_double(d[i]);
        }
    }
};
#endif
