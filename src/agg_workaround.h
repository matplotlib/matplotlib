#ifndef MPL_AGG_WORKAROUND_H
#define MPL_AGG_WORKAROUND_H

#include "agg_pixfmt_rgba.h"
#include "agg_trans_affine.h"

/**********************************************************************
 WORKAROUND: This class is to workaround a bug in Agg SVN where the
 blending of RGBA32 pixels does not preserve enough precision
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
 This class provides higher-accuracy nearest-neighbor interpolation for
 affine transforms than span_interpolator_linear by using
 floating-point-based interpolation instead of integer-based
*/

template<class Transformer = agg::trans_affine, unsigned SubpixelShift = 8>
class accurate_interpolator_affine_nn
{
public:
    typedef Transformer trans_type;

    enum subpixel_scale_e
    {
        subpixel_shift = SubpixelShift,
        subpixel_scale  = 1 << subpixel_shift
    };

    //--------------------------------------------------------------------
    accurate_interpolator_affine_nn() {}
    accurate_interpolator_affine_nn(trans_type& trans) : m_trans(&trans) {}
    accurate_interpolator_affine_nn(trans_type& trans,
                                    double x, double y, unsigned len) :
        m_trans(&trans)
    {
        begin(x, y, len);
    }

    //----------------------------------------------------------------
    const trans_type& transformer() const { return *m_trans; }
    void transformer(trans_type& trans) { m_trans = &trans; }

    //----------------------------------------------------------------
    void begin(double x, double y, unsigned len)
    {
        m_len = len - 1;

        m_stx1 = x;
        m_sty1 = y;
        m_trans->transform(&m_stx1, &m_sty1);
        m_stx1 *= subpixel_scale;
        m_sty1 *= subpixel_scale;

        m_stx2 = x + m_len;
        m_sty2 = y;
        m_trans->transform(&m_stx2, &m_sty2);
        m_stx2 *= subpixel_scale;
        m_sty2 *= subpixel_scale;
    }

    //----------------------------------------------------------------
    void resynchronize(double xe, double ye, unsigned len)
    {
        m_len = len - 1;

        m_trans->transform(&xe, &ye);
        m_stx2 = xe * subpixel_scale;
        m_sty2 = ye * subpixel_scale;
    }

    //----------------------------------------------------------------
    void operator++()
    {
        m_stx1 += (m_stx2 - m_stx1) / m_len;
        m_sty1 += (m_sty2 - m_sty1) / m_len;
        m_len--;
    }

    //----------------------------------------------------------------
    void coordinates(int* x, int* y) const
    {
        // Truncate instead of round because this interpolator needs to
        // match the definitions for nearest-neighbor interpolation
        *x = int(m_stx1);
        *y = int(m_sty1);
    }

private:
    trans_type* m_trans;
    unsigned m_len;
    double m_stx1, m_sty1, m_stx2, m_sty2;
};
#endif
