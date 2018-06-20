/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_RESAMPLE_H
#define MPL_RESAMPLE_H

#include "agg_image_accessors.h"
#include "agg_path_storage.h"
#include "agg_pixfmt_gray.h"
#include "agg_pixfmt_rgb.h"
#include "agg_pixfmt_rgba.h"
#include "agg_renderer_base.h"
#include "agg_renderer_scanline.h"
#include "agg_rasterizer_scanline_aa.h"
#include "agg_scanline_u.h"
#include "agg_span_allocator.h"
#include "agg_span_converter.h"
#include "agg_span_image_filter_gray.h"
#include "agg_span_image_filter_rgba.h"
#include "agg_span_interpolator_adaptor.h"
#include "agg_span_interpolator_linear.h"

#include "agg_workaround.h"

// Based on:

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

//===================================================================gray64
namespace agg
{
    struct gray64
    {
        typedef double value_type;
        typedef double calc_type;
        typedef double long_type;
        typedef gray64 self_type;

        value_type v;
        value_type a;

        //--------------------------------------------------------------------
        gray64() {}

        //--------------------------------------------------------------------
        explicit gray64(value_type v_, value_type a_ = 1) :
        v(v_), a(a_) {}

        //--------------------------------------------------------------------
        gray64(const self_type& c, value_type a_) :
            v(c.v), a(a_) {}

        //--------------------------------------------------------------------
        gray64(const gray64& c) :
            v(c.v),
            a(c.a) {}

        //--------------------------------------------------------------------
        static AGG_INLINE double to_double(value_type a)
        {
            return a;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type from_double(double a)
        {
            return value_type(a);
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type empty_value()
        {
            return 0;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type full_value()
        {
            return 1;
        }

        //--------------------------------------------------------------------
        AGG_INLINE bool is_transparent() const
        {
            return a <= 0;
        }

        //--------------------------------------------------------------------
        AGG_INLINE bool is_opaque() const
        {
            return a >= 1;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type invert(value_type x)
        {
            return 1 - x;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type multiply(value_type a, value_type b)
        {
            return value_type(a * b);
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type demultiply(value_type a, value_type b)
        {
            return (b == 0) ? 0 : value_type(a / b);
        }

        //--------------------------------------------------------------------
        template<typename T>
        static AGG_INLINE T downscale(T a)
        {
            return a;
        }

        //--------------------------------------------------------------------
        template<typename T>
        static AGG_INLINE T downshift(T a, unsigned n)
        {
            return n > 0 ? a / (1 << n) : a;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type mult_cover(value_type a, cover_type b)
        {
            return value_type(a * b / cover_mask);
        }

        //--------------------------------------------------------------------
        static AGG_INLINE cover_type scale_cover(cover_type a, value_type b)
        {
            return cover_type(uround(a * b));
        }

        //--------------------------------------------------------------------
        // Interpolate p to q by a, assuming q is premultiplied by a.
        static AGG_INLINE value_type prelerp(value_type p, value_type q, value_type a)
        {
            return (1 - a) * p + q; // more accurate than "p + q - p * a"
        }

        //--------------------------------------------------------------------
        // Interpolate p to q by a.
        static AGG_INLINE value_type lerp(value_type p, value_type q, value_type a)
        {
            // The form "p + a * (q - p)" avoids a multiplication, but may produce an
            // inaccurate result. For example, "p + (q - p)" may not be exactly equal
            // to q. Therefore, stick to the basic expression, which at least produces
            // the correct result at either extreme.
            return (1 - a) * p + a * q;
        }

        //--------------------------------------------------------------------
        self_type& clear()
        {
            v = a = 0;
            return *this;
        }

        //--------------------------------------------------------------------
        self_type& transparent()
        {
            a = 0;
            return *this;
        }

        //--------------------------------------------------------------------
        self_type& opacity(double a_)
        {
            if (a_ < 0) a = 0;
            else if (a_ > 1) a = 1;
            else a = value_type(a_);
            return *this;
        }

        //--------------------------------------------------------------------
        double opacity() const
        {
            return a;
        }


        //--------------------------------------------------------------------
        self_type& premultiply()
        {
            if (a < 0) v = 0;
            else if(a < 1) v *= a;
            return *this;
        }

        //--------------------------------------------------------------------
        self_type& demultiply()
        {
            if (a < 0) v = 0;
            else if (a < 1) v /= a;
            return *this;
        }

        //--------------------------------------------------------------------
        self_type gradient(self_type c, double k) const
        {
            return self_type(
                             value_type(v + (c.v - v) * k),
                             value_type(a + (c.a - a) * k));
        }

        //--------------------------------------------------------------------
        static self_type no_color() { return self_type(0,0); }
    };


    //====================================================================rgba32
    struct rgba64
    {
        typedef double value_type;
        typedef double calc_type;
        typedef double long_type;
        typedef rgba64 self_type;

        value_type r;
        value_type g;
        value_type b;
        value_type a;

        //--------------------------------------------------------------------
        rgba64() {}

        //--------------------------------------------------------------------
        rgba64(value_type r_, value_type g_, value_type b_, value_type a_= 1) :
            r(r_), g(g_), b(b_), a(a_) {}

        //--------------------------------------------------------------------
        rgba64(const self_type& c, float a_) :
            r(c.r), g(c.g), b(c.b), a(a_) {}

        //--------------------------------------------------------------------
        rgba64(const rgba& c) :
            r(value_type(c.r)), g(value_type(c.g)), b(value_type(c.b)), a(value_type(c.a)) {}

        //--------------------------------------------------------------------
        operator rgba() const
        {
            return rgba(r, g, b, a);
        }

        //--------------------------------------------------------------------
        static AGG_INLINE double to_double(value_type a)
        {
            return a;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type from_double(double a)
        {
            return value_type(a);
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type empty_value()
        {
            return 0;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type full_value()
        {
            return 1;
        }

        //--------------------------------------------------------------------
        AGG_INLINE bool is_transparent() const
        {
            return a <= 0;
        }

        //--------------------------------------------------------------------
        AGG_INLINE bool is_opaque() const
        {
            return a >= 1;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type invert(value_type x)
        {
            return 1 - x;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type multiply(value_type a, value_type b)
        {
            return value_type(a * b);
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type demultiply(value_type a, value_type b)
        {
            return (b == 0) ? 0 : value_type(a / b);
        }

        //--------------------------------------------------------------------
        template<typename T>
        static AGG_INLINE T downscale(T a)
        {
            return a;
        }

        //--------------------------------------------------------------------
        template<typename T>
        static AGG_INLINE T downshift(T a, unsigned n)
        {
            return n > 0 ? a / (1 << n) : a;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type mult_cover(value_type a, cover_type b)
        {
            return value_type(a * b / cover_mask);
        }

        //--------------------------------------------------------------------
        static AGG_INLINE cover_type scale_cover(cover_type a, value_type b)
        {
            return cover_type(uround(a * b));
        }

        //--------------------------------------------------------------------
        // Interpolate p to q by a, assuming q is premultiplied by a.
        static AGG_INLINE value_type prelerp(value_type p, value_type q, value_type a)
        {
            return (1 - a) * p + q; // more accurate than "p + q - p * a"
        }

        //--------------------------------------------------------------------
        // Interpolate p to q by a.
        static AGG_INLINE value_type lerp(value_type p, value_type q, value_type a)
        {
            // The form "p + a * (q - p)" avoids a multiplication, but may produce an
            // inaccurate result. For example, "p + (q - p)" may not be exactly equal
            // to q. Therefore, stick to the basic expression, which at least produces
            // the correct result at either extreme.
            return (1 - a) * p + a * q;
        }

        //--------------------------------------------------------------------
        self_type& clear()
        {
            r = g = b = a = 0;
            return *this;
        }

        //--------------------------------------------------------------------
        self_type& transparent()
        {
            a = 0;
            return *this;
        }

        //--------------------------------------------------------------------
        AGG_INLINE self_type& opacity(double a_)
        {
            if (a_ < 0) a = 0;
            else if (a_ > 1) a = 1;
            else a = value_type(a_);
            return *this;
        }

        //--------------------------------------------------------------------
        double opacity() const
        {
            return a;
        }

        //--------------------------------------------------------------------
        AGG_INLINE self_type& premultiply()
        {
            if (a < 1)
            {
                if (a <= 0)
                {
                    r = g = b = 0;
                }
                else
                {
                    r *= a;
                    g *= a;
                    b *= a;
                }
            }
            return *this;
        }

        //--------------------------------------------------------------------
        AGG_INLINE self_type& demultiply()
        {
            if (a < 1)
            {
                if (a <= 0)
                {
                    r = g = b = 0;
                }
                else
                {
                    r /= a;
                    g /= a;
                    b /= a;
                }
            }
            return *this;
        }

        //--------------------------------------------------------------------
        AGG_INLINE self_type gradient(const self_type& c, double k) const
        {
            self_type ret;
            ret.r = value_type(r + (c.r - r) * k);
            ret.g = value_type(g + (c.g - g) * k);
            ret.b = value_type(b + (c.b - b) * k);
            ret.a = value_type(a + (c.a - a) * k);
            return ret;
        }

        //--------------------------------------------------------------------
        AGG_INLINE void add(const self_type& c, unsigned cover)
        {
            if (cover == cover_mask)
            {
                if (c.is_opaque())
                {
                    *this = c;
                    return;
                }
                else
                {
                    r += c.r;
                    g += c.g;
                    b += c.b;
                    a += c.a;
                }
            }
            else
            {
                r += mult_cover(c.r, cover);
                g += mult_cover(c.g, cover);
                b += mult_cover(c.b, cover);
                a += mult_cover(c.a, cover);
            }
            if (a > 1) a = 1;
            if (r > a) r = a;
            if (g > a) g = a;
            if (b > a) b = a;
        }

        //--------------------------------------------------------------------
        static self_type no_color() { return self_type(0,0,0,0); }
    };
}


typedef enum {
    NEAREST,
    BILINEAR,
    BICUBIC,
    SPLINE16,
    SPLINE36,
    HANNING,
    HAMMING,
    HERMITE,
    KAISER,
    QUADRIC,
    CATROM,
    GAUSSIAN,
    BESSEL,
    MITCHELL,
    SINC,
    LANCZOS,
    BLACKMAN,
    _n_interpolation
} interpolation_e;


template <typename T>
class type_mapping;


template <> class type_mapping<agg::rgba8>
{
 public:
    typedef agg::rgba8 color_type;
    typedef fixed_blender_rgba_plain<color_type, agg::order_rgba> blender_type;
    typedef fixed_blender_rgba_pre<color_type, agg::order_rgba> pre_blender_type;
    typedef agg::pixfmt_alpha_blend_rgba<blender_type, agg::rendering_buffer> pixfmt_type;
    typedef agg::pixfmt_alpha_blend_rgba<pre_blender_type, agg::rendering_buffer> pixfmt_pre_type;

    template <typename A>
    struct span_gen_affine_type
    {
        typedef agg::span_image_resample_rgba_affine<A> type;
    };

    template <typename A, typename B>
    struct span_gen_filter_type
    {
        typedef agg::span_image_filter_rgba<A, B> type;
    };

    template <typename A, typename B>
    struct span_gen_nn_type
    {
        typedef agg::span_image_filter_rgba_nn<A, B> type;
    };
};


template <> class type_mapping<agg::rgba16>
{
 public:
    typedef agg::rgba16 color_type;
    typedef fixed_blender_rgba_plain<color_type, agg::order_rgba> blender_type;
    typedef fixed_blender_rgba_pre<color_type, agg::order_rgba> pre_blender_type;
    typedef agg::pixfmt_alpha_blend_rgba<blender_type, agg::rendering_buffer> pixfmt_type;
    typedef agg::pixfmt_alpha_blend_rgba<pre_blender_type, agg::rendering_buffer> pixfmt_pre_type;

    template <typename A>
    struct span_gen_affine_type
    {
        typedef agg::span_image_resample_rgba_affine<A> type;
    };

    template <typename A, typename B>
    struct span_gen_filter_type
    {
        typedef agg::span_image_filter_rgba<A, B> type;
    };

    template <typename A, typename B>
    struct span_gen_nn_type
    {
        typedef agg::span_image_filter_rgba_nn<A, B> type;
    };
};


template <> class type_mapping<agg::rgba32>
{
 public:
    typedef agg::rgba32 color_type;
    typedef agg::blender_rgba_plain<color_type, agg::order_rgba> blender_type;
    typedef agg::blender_rgba_pre<color_type, agg::order_rgba> pre_blender_type;
    typedef agg::pixfmt_alpha_blend_rgba<blender_type, agg::rendering_buffer> pixfmt_type;
    typedef agg::pixfmt_alpha_blend_rgba<pre_blender_type, agg::rendering_buffer> pixfmt_pre_type;

    template <typename A>
    struct span_gen_affine_type
    {
        typedef agg::span_image_resample_rgba_affine<A> type;
    };

    template <typename A, typename B>
    struct span_gen_filter_type
    {
        typedef agg::span_image_filter_rgba<A, B> type;
    };

    template <typename A, typename B>
    struct span_gen_nn_type
    {
        typedef agg::span_image_filter_rgba_nn<A, B> type;
    };
};


template <> class type_mapping<agg::rgba64>
{
 public:
    typedef agg::rgba64 color_type;
    typedef agg::blender_rgba_plain<color_type, agg::order_rgba> blender_type;
    typedef agg::blender_rgba_pre<color_type, agg::order_rgba> pre_blender_type;
    typedef agg::pixfmt_alpha_blend_rgba<blender_type, agg::rendering_buffer> pixfmt_type;
    typedef agg::pixfmt_alpha_blend_rgba<pre_blender_type, agg::rendering_buffer> pixfmt_pre_type;

    template <typename A>
    struct span_gen_affine_type
    {
        typedef agg::span_image_resample_rgba_affine<A> type;
    };

    template <typename A, typename B>
    struct span_gen_filter_type
    {
        typedef agg::span_image_filter_rgba<A, B> type;
    };

    template <typename A, typename B>
    struct span_gen_nn_type
    {
        typedef agg::span_image_filter_rgba_nn<A, B> type;
    };
};


template <> class type_mapping<double>
{
 public:
    typedef agg::gray64 color_type;
    typedef agg::blender_gray<color_type> blender_type;
    typedef agg::pixfmt_alpha_blend_gray<blender_type, agg::rendering_buffer> pixfmt_type;
    typedef pixfmt_type pixfmt_pre_type;

    template <typename A>
    struct span_gen_affine_type
    {
        typedef agg::span_image_resample_gray_affine<A> type;
    };

    template <typename A, typename B>
    struct span_gen_filter_type
    {
        typedef agg::span_image_filter_gray<A, B> type;
    };

    template <typename A, typename B>
    struct span_gen_nn_type
    {
        typedef agg::span_image_filter_gray_nn<A, B> type;
    };
};


template <> class type_mapping<float>
{
 public:
    typedef agg::gray32 color_type;
    typedef agg::blender_gray<color_type> blender_type;
    typedef agg::pixfmt_alpha_blend_gray<blender_type, agg::rendering_buffer> pixfmt_type;
    typedef pixfmt_type pixfmt_pre_type;

    template <typename A>
    struct span_gen_affine_type
    {
        typedef agg::span_image_resample_gray_affine<A> type;
    };

    template <typename A, typename B>
    struct span_gen_filter_type
    {
        typedef agg::span_image_filter_gray<A, B> type;
    };

    template <typename A, typename B>
    struct span_gen_nn_type
    {
        typedef agg::span_image_filter_gray_nn<A, B> type;
    };
};


template <> class type_mapping<unsigned short>
{
 public:
    typedef agg::gray16 color_type;
    typedef agg::blender_gray<color_type> blender_type;
    typedef agg::pixfmt_alpha_blend_gray<blender_type, agg::rendering_buffer> pixfmt_type;
    typedef pixfmt_type pixfmt_pre_type;

    template <typename A>
    struct span_gen_affine_type
    {
        typedef agg::span_image_resample_gray_affine<A> type;
    };

    template <typename A, typename B>
    struct span_gen_filter_type
    {
        typedef agg::span_image_filter_gray<A, B> type;
    };

    template <typename A, typename B>
    struct span_gen_nn_type
    {
        typedef agg::span_image_filter_gray_nn<A, B> type;
    };
};


template <> class type_mapping<unsigned char>
{
 public:
    typedef agg::gray8 color_type;
    typedef agg::blender_gray<color_type> blender_type;
    typedef agg::pixfmt_alpha_blend_gray<blender_type, agg::rendering_buffer> pixfmt_type;
    typedef pixfmt_type pixfmt_pre_type;

    template <typename A>
    struct span_gen_affine_type
    {
        typedef agg::span_image_resample_gray_affine<A> type;
    };

    template <typename A, typename B>
    struct span_gen_filter_type
    {
        typedef agg::span_image_filter_gray<A, B> type;
    };

    template <typename A, typename B>
    struct span_gen_nn_type
    {
        typedef agg::span_image_filter_gray_nn<A, B> type;
    };
};



template<class color_type>
class span_conv_alpha
{
public:
    span_conv_alpha(const double alpha) :
        m_alpha(alpha)
    {
    }

    void prepare() {}

    void generate(color_type* span, int x, int y, unsigned len) const
    {
        if (m_alpha != 1.0) {
            do {
                span->a *= m_alpha;
                ++span;
            } while (--len);
        }
    }
private:

    const double m_alpha;
};


/* A class to use a lookup table for a transformation */
class lookup_distortion
{
public:
    lookup_distortion(const double *mesh, int in_width, int in_height,
                      int out_width, int out_height) :
        m_mesh(mesh),
        m_in_width(in_width),
        m_in_height(in_height),
        m_out_width(out_width),
        m_out_height(out_height)
    {}

    void calculate(int* x, int* y) {
        if (m_mesh) {
            double dx = double(*x) / agg::image_subpixel_scale;
            double dy = double(*y) / agg::image_subpixel_scale;
            if (dx >= 0 && dx < m_out_width &&
                dy >= 0 && dy < m_out_height) {
                const double *coord = m_mesh + (int(dy) * m_out_width + int(dx)) * 2;
                *x = int(coord[0] * agg::image_subpixel_scale);
                *y = int(coord[1] * agg::image_subpixel_scale);
            }
        }
    }

protected:
    const double *m_mesh;
    int m_in_width;
    int m_in_height;
    int m_out_width;
    int m_out_height;
};


struct resample_params_t {
    interpolation_e interpolation;
    bool is_affine;
    agg::trans_affine affine;
    const double *transform_mesh;
    bool resample;
    bool norm;
    double radius;
    double alpha;
};


static void get_filter(const resample_params_t &params,
                       agg::image_filter_lut &filter)
{
    switch (params.interpolation) {
    case NEAREST:
    case _n_interpolation:
        // Never should get here.  Here to silence compiler warnings.
        break;

    case HANNING:
        filter.calculate(agg::image_filter_hanning(), params.norm);
        break;

    case HAMMING:
        filter.calculate(agg::image_filter_hamming(), params.norm);
        break;

    case HERMITE:
        filter.calculate(agg::image_filter_hermite(), params.norm);
        break;

    case BILINEAR:
        filter.calculate(agg::image_filter_bilinear(), params.norm);
        break;

    case BICUBIC:
        filter.calculate(agg::image_filter_bicubic(), params.norm);
        break;

    case SPLINE16:
        filter.calculate(agg::image_filter_spline16(), params.norm);
        break;

    case SPLINE36:
        filter.calculate(agg::image_filter_spline36(), params.norm);
        break;

    case KAISER:
        filter.calculate(agg::image_filter_kaiser(), params.norm);
        break;

    case QUADRIC:
        filter.calculate(agg::image_filter_quadric(), params.norm);
        break;

    case CATROM:
        filter.calculate(agg::image_filter_catrom(), params.norm);
        break;

    case GAUSSIAN:
        filter.calculate(agg::image_filter_gaussian(), params.norm);
        break;

    case BESSEL:
        filter.calculate(agg::image_filter_bessel(), params.norm);
        break;

    case MITCHELL:
        filter.calculate(agg::image_filter_mitchell(), params.norm);
        break;

    case SINC:
        filter.calculate(agg::image_filter_sinc(params.radius), params.norm);
        break;

    case LANCZOS:
        filter.calculate(agg::image_filter_lanczos(params.radius), params.norm);
        break;

    case BLACKMAN:
        filter.calculate(agg::image_filter_blackman(params.radius), params.norm);
        break;
    }
}


template<class T>
void resample(
    const T *input, int in_width, int in_height,
    T *output, int out_width, int out_height,
    resample_params_t &params)
{
    typedef type_mapping<T> type_mapping_t;

    typedef typename type_mapping_t::pixfmt_type input_pixfmt_t;
    typedef typename type_mapping_t::pixfmt_type output_pixfmt_t;

    typedef agg::renderer_base<output_pixfmt_t> renderer_t;
    typedef agg::rasterizer_scanline_aa<agg::rasterizer_sl_clip_dbl> rasterizer_t;

    typedef agg::wrap_mode_reflect reflect_t;
    typedef agg::image_accessor_wrap<input_pixfmt_t, reflect_t, reflect_t> image_accessor_t;

    typedef agg::span_allocator<typename type_mapping_t::color_type> span_alloc_t;
    typedef span_conv_alpha<typename type_mapping_t::color_type> span_conv_alpha_t;

    typedef agg::span_interpolator_linear<> affine_interpolator_t;
    typedef agg::span_interpolator_adaptor<agg::span_interpolator_linear<>, lookup_distortion>
        arbitrary_interpolator_t;

    if (params.interpolation != NEAREST &&
        params.is_affine &&
        fabs(params.affine.sx) == 1.0 &&
        fabs(params.affine.sy) == 1.0 &&
        params.affine.shx == 0.0 &&
        params.affine.shy == 0.0) {
        params.interpolation = NEAREST;
    }

    span_alloc_t span_alloc;
    rasterizer_t rasterizer;
    agg::scanline_u8 scanline;

    span_conv_alpha_t conv_alpha(params.alpha);

    agg::rendering_buffer input_buffer;
    input_buffer.attach((unsigned char *)input, in_width, in_height,
                        in_width * sizeof(T));
    input_pixfmt_t input_pixfmt(input_buffer);
    image_accessor_t input_accessor(input_pixfmt);

    agg::rendering_buffer output_buffer;
    output_buffer.attach((unsigned char *)output, out_width, out_height,
                         out_width * sizeof(T));
    output_pixfmt_t output_pixfmt(output_buffer);
    renderer_t renderer(output_pixfmt);

    agg::trans_affine inverted = params.affine;
    inverted.invert();

    rasterizer.clip_box(0, 0, out_width, out_height);

    agg::path_storage path;
    if (params.is_affine) {
        path.move_to(0, 0);
        path.line_to(in_width, 0);
        path.line_to(in_width, in_height);
        path.line_to(0, in_height);
        path.close_polygon();
        agg::conv_transform<agg::path_storage> rectangle(path, params.affine);
        rasterizer.add_path(rectangle);
    } else {
        path.move_to(0, 0);
        path.line_to(out_width, 0);
        path.line_to(out_width, out_height);
        path.line_to(0, out_height);
        path.close_polygon();
        rasterizer.add_path(path);
    }

    if (params.interpolation == NEAREST) {
        if (params.is_affine) {
            typedef typename type_mapping_t::template span_gen_nn_type<image_accessor_t, affine_interpolator_t>::type span_gen_t;
            typedef agg::span_converter<span_gen_t, span_conv_alpha_t> span_conv_t;
            typedef agg::renderer_scanline_aa<renderer_t, span_alloc_t, span_conv_t> nn_renderer_t;

            affine_interpolator_t interpolator(inverted);
            span_gen_t span_gen(input_accessor, interpolator);
            span_conv_t span_conv(span_gen, conv_alpha);
            nn_renderer_t nn_renderer(renderer, span_alloc, span_conv);
            agg::render_scanlines(rasterizer, scanline, nn_renderer);
        } else {
            typedef typename type_mapping_t::template span_gen_nn_type<image_accessor_t, arbitrary_interpolator_t>::type span_gen_t;
            typedef agg::span_converter<span_gen_t, span_conv_alpha_t> span_conv_t;
            typedef agg::renderer_scanline_aa<renderer_t, span_alloc_t, span_conv_t> nn_renderer_t;

            lookup_distortion dist(
                params.transform_mesh, in_width, in_height, out_width, out_height);
            arbitrary_interpolator_t interpolator(inverted, dist);
            span_gen_t span_gen(input_accessor, interpolator);
            span_conv_t span_conv(span_gen, conv_alpha);
            nn_renderer_t nn_renderer(renderer, span_alloc, span_conv);
            agg::render_scanlines(rasterizer, scanline, nn_renderer);
        }
    } else {
        agg::image_filter_lut filter;
        get_filter(params, filter);

        if (params.is_affine && params.resample) {
            typedef typename type_mapping_t::template span_gen_affine_type<image_accessor_t>::type span_gen_t;
            typedef agg::span_converter<span_gen_t, span_conv_alpha_t> span_conv_t;
            typedef agg::renderer_scanline_aa<renderer_t, span_alloc_t, span_conv_t> int_renderer_t;

            affine_interpolator_t interpolator(inverted);
            span_gen_t span_gen(input_accessor, interpolator, filter);
            span_conv_t span_conv(span_gen, conv_alpha);
            int_renderer_t int_renderer(renderer, span_alloc, span_conv);
            agg::render_scanlines(rasterizer, scanline, int_renderer);
        } else {
            typedef typename type_mapping_t::template span_gen_filter_type<image_accessor_t, arbitrary_interpolator_t>::type span_gen_t;
            typedef agg::span_converter<span_gen_t, span_conv_alpha_t> span_conv_t;
            typedef agg::renderer_scanline_aa<renderer_t, span_alloc_t, span_conv_t> int_renderer_t;

            lookup_distortion dist(
                params.transform_mesh, in_width, in_height, out_width, out_height);
            arbitrary_interpolator_t interpolator(inverted, dist);
            span_gen_t span_gen(input_accessor, interpolator, filter);
            span_conv_t span_conv(span_gen, conv_alpha);
            int_renderer_t int_renderer(renderer, span_alloc, span_conv);
            agg::render_scanlines(rasterizer, scanline, int_renderer);
        }
    }
}

#endif /* MPL_RESAMPLE_H */
