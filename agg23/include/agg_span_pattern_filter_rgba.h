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
//
// classes span_pattern_filter_rgba*
//
//----------------------------------------------------------------------------
#ifndef AGG_SPAN_PATTERN_FILTER_RGBA_INCLUDED
#define AGG_SPAN_PATTERN_FILTER_RGBA_INCLUDED

#include "agg_basics.h"
#include "agg_color_rgba.h"
#include "agg_span_pattern.h"
#include "agg_span_image_filter.h"


namespace agg
{

    //===========================================span_pattern_filter_rgba
    template<class ColorT,
             class Order, 
             class Interpolator,
             class WrapModeX,
             class WrapModeY,
             class Allocator = span_allocator<ColorT> > 
    class span_pattern_filter_rgba_nn : 
    public span_image_filter<ColorT, Interpolator, Allocator>
    {
    public:
        typedef ColorT color_type;
        typedef Order order_type;
        typedef Interpolator interpolator_type;
        typedef Allocator alloc_type;
        typedef span_image_filter<color_type, interpolator_type, alloc_type> base_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        enum
        {
            base_shift = color_type::base_shift,
            base_mask  = color_type::base_mask
        };

        //--------------------------------------------------------------------
        span_pattern_filter_rgba_nn(alloc_type& alloc) : 
            base_type(alloc),
            m_wrap_mode_x(1),
            m_wrap_mode_y(1)
        {}

        //--------------------------------------------------------------------
        span_pattern_filter_rgba_nn(alloc_type& alloc,
                                    const rendering_buffer& src, 
                                    interpolator_type& inter) :
            base_type(alloc, src, color_type(0,0,0,0), inter, 0),
            m_wrap_mode_x(src.width()),
            m_wrap_mode_y(src.height())
        {}

        //--------------------------------------------------------------------
        void source_image(const rendering_buffer& src) 
        { 
            base_type::source_image(src);
            m_wrap_mode_x = WrapModeX(src.width());
            m_wrap_mode_y = WrapModeX(src.height());
        }

        //--------------------------------------------------------------------
        color_type* generate(int x, int y, unsigned len)
        {
            base_type::interpolator().begin(x + base_type::filter_dx_dbl(), 
                                            y + base_type::filter_dy_dbl(), len);
            const value_type *fg_ptr;
            color_type* span = base_type::allocator().span();
            do
            {
                base_type::interpolator().coordinates(&x, &y);

                x = m_wrap_mode_x(x >> image_subpixel_shift);
                y = m_wrap_mode_y(y >> image_subpixel_shift);

                fg_ptr = (value_type*)base_type::source_image().row(y) + (x << 2);
                span->r = fg_ptr[order_type::R];
                span->g = fg_ptr[order_type::G];
                span->b = fg_ptr[order_type::B];
                span->a = fg_ptr[order_type::A];
                ++span;
                ++base_type::interpolator();

            } while(--len);

            return base_type::allocator().span();
        }

    private:
        WrapModeX m_wrap_mode_x;
        WrapModeY m_wrap_mode_y;
    };










    //=====================================span_pattern_filter_rgba_bilinear
    template<class ColorT,
             class Order, 
             class Interpolator,
             class WrapModeX,
             class WrapModeY,
             class Allocator = span_allocator<ColorT> > 
    class span_pattern_filter_rgba_bilinear : 
    public span_image_filter<ColorT, Interpolator, Allocator>
    {
    public:
        typedef ColorT color_type;
        typedef Order order_type;
        typedef Interpolator interpolator_type;
        typedef Allocator alloc_type;
        typedef span_image_filter<color_type, interpolator_type, alloc_type> base_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        enum
        {
            base_shift = color_type::base_shift,
            base_mask  = color_type::base_mask
        };

        //--------------------------------------------------------------------
        span_pattern_filter_rgba_bilinear(alloc_type& alloc) : 
            base_type(alloc),
            m_wrap_mode_x(1),
            m_wrap_mode_y(1)
        {}

        //--------------------------------------------------------------------
        span_pattern_filter_rgba_bilinear(alloc_type& alloc,
                                          const rendering_buffer& src, 
                                          interpolator_type& inter) :
            base_type(alloc, src, color_type(0,0,0,0), inter, 0),
            m_wrap_mode_x(src.width()),
            m_wrap_mode_y(src.height())
        {}

        //-------------------------------------------------------------------
        void source_image(const rendering_buffer& src) 
        { 
            base_type::source_image(src);
            m_wrap_mode_x = WrapModeX(src.width());
            m_wrap_mode_y = WrapModeX(src.height());
        }

        //--------------------------------------------------------------------
        color_type* generate(int x, int y, unsigned len)
        {
            base_type::interpolator().begin(x + base_type::filter_dx_dbl(), 
                                            y + base_type::filter_dy_dbl(), len);
            calc_type fg[4];
            const value_type *fg_ptr;
            color_type* span = base_type::allocator().span();

            do
            {
                int x_hr;
                int y_hr;

                base_type::interpolator().coordinates(&x_hr, &y_hr);

                x_hr -= base_type::filter_dx_int();
                y_hr -= base_type::filter_dy_int();

                int x_lr = x_hr >> image_subpixel_shift;
                int y_lr = y_hr >> image_subpixel_shift;

                unsigned x1 = m_wrap_mode_x(x_lr);
                unsigned x2 = ++m_wrap_mode_x;
                x1 <<= 2;
                x2 <<= 2;

                unsigned y1 = m_wrap_mode_y(y_lr);
                unsigned y2 = ++m_wrap_mode_y;
                const value_type* ptr1 = (value_type*)base_type::source_image().row(y1);
                const value_type* ptr2 = (value_type*)base_type::source_image().row(y2);

                fg[0] = 
                fg[1] = 
                fg[2] = 
                fg[3] = image_subpixel_size * image_subpixel_size / 2;

                x_hr &= image_subpixel_mask;
                y_hr &= image_subpixel_mask;

                int weight;
                fg_ptr = ptr1 + x1;
                weight = (image_subpixel_size - x_hr) * 
                         (image_subpixel_size - y_hr);
                fg[0] += weight * fg_ptr[0];
                fg[1] += weight * fg_ptr[1];
                fg[2] += weight * fg_ptr[2];
                fg[3] += weight * fg_ptr[3];

                fg_ptr = ptr1 + x2;
                weight = x_hr * (image_subpixel_size - y_hr);
                fg[0] += weight * fg_ptr[0];
                fg[1] += weight * fg_ptr[1];
                fg[2] += weight * fg_ptr[2];
                fg[3] += weight * fg_ptr[3];

                fg_ptr = ptr2 + x1;
                weight = (image_subpixel_size - x_hr) * y_hr;
                fg[0] += weight * fg_ptr[0];
                fg[1] += weight * fg_ptr[1];
                fg[2] += weight * fg_ptr[2];
                fg[3] += weight * fg_ptr[3];

                fg_ptr = ptr2 + x2;
                weight = x_hr * y_hr;
                fg[0] += weight * fg_ptr[0];
                fg[1] += weight * fg_ptr[1];
                fg[2] += weight * fg_ptr[2];
                fg[3] += weight * fg_ptr[3];

                span->r = (value_type)(fg[order_type::R] >> image_subpixel_shift * 2);
                span->g = (value_type)(fg[order_type::G] >> image_subpixel_shift * 2);
                span->b = (value_type)(fg[order_type::B] >> image_subpixel_shift * 2);
                span->a = (value_type)(fg[order_type::A] >> image_subpixel_shift * 2);
                ++span;
                ++base_type::interpolator();

            } while(--len);

            return base_type::allocator().span();
        }
    private:
        WrapModeX m_wrap_mode_x;
        WrapModeY m_wrap_mode_y;
    };








    //=====================================span_pattern_filter_rgba_2x2
    template<class ColorT,
             class Order, 
             class Interpolator,
             class WrapModeX,
             class WrapModeY,
             class Allocator = span_allocator<ColorT> > 
    class span_pattern_filter_rgba_2x2 : 
    public span_image_filter<ColorT, Interpolator, Allocator>
    {
    public:
        typedef ColorT color_type;
        typedef Order order_type;
        typedef Interpolator interpolator_type;
        typedef Allocator alloc_type;
        typedef span_image_filter<color_type, interpolator_type, alloc_type> base_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        enum
        {
            base_shift = color_type::base_shift,
            base_mask  = color_type::base_mask
        };

        //--------------------------------------------------------------------
        span_pattern_filter_rgba_2x2(alloc_type& alloc) : 
            base_type(alloc),
            m_wrap_mode_x(1),
            m_wrap_mode_y(1)
        {}

        //--------------------------------------------------------------------
        span_pattern_filter_rgba_2x2(alloc_type& alloc,
                                     const rendering_buffer& src, 
                                     interpolator_type& inter,
                                     const image_filter_lut& filter) :
            base_type(alloc, src, color_type(0,0,0,0), inter, &filter),
            m_wrap_mode_x(src.width()),
            m_wrap_mode_y(src.height())
        {}

        //-------------------------------------------------------------------
        void source_image(const rendering_buffer& src) 
        { 
            base_type::source_image(src);
            m_wrap_mode_x = WrapModeX(src.width());
            m_wrap_mode_y = WrapModeX(src.height());
        }

        //--------------------------------------------------------------------
        color_type* generate(int x, int y, unsigned len)
        {
            base_type::interpolator().begin(x + base_type::filter_dx_dbl(), 
                                            y + base_type::filter_dy_dbl(), len);
            calc_type fg[4];
            const value_type *fg_ptr;
            color_type* span = base_type::allocator().span();
            const int16* weight_array = base_type::filter().weight_array() + 
                                        ((base_type::filter().diameter()/2 - 1) << 
                                          image_subpixel_shift);
            do
            {
                int x_hr;
                int y_hr;

                base_type::interpolator().coordinates(&x_hr, &y_hr);

                x_hr -= base_type::filter_dx_int();
                y_hr -= base_type::filter_dy_int();

                int x_lr = x_hr >> image_subpixel_shift;
                int y_lr = y_hr >> image_subpixel_shift;

                unsigned x1 = m_wrap_mode_x(x_lr);
                unsigned x2 = ++m_wrap_mode_x;
                x1 <<= 2;
                x2 <<= 2;

                unsigned y1 = m_wrap_mode_y(y_lr);
                unsigned y2 = ++m_wrap_mode_y;
                const value_type* ptr1 = (value_type*)base_type::source_image().row(y1);
                const value_type* ptr2 = (value_type*)base_type::source_image().row(y2);

                fg[0] = fg[1] = fg[2] = fg[3] = image_filter_size / 2;

                x_hr &= image_subpixel_mask;
                y_hr &= image_subpixel_mask;

                int weight;
                fg_ptr = ptr1 + x1;
                weight = (weight_array[x_hr + image_subpixel_size] * 
                          weight_array[y_hr + image_subpixel_size] + 
                          image_filter_size / 2) >> 
                          image_filter_shift;
                fg[0] += weight * fg_ptr[0];
                fg[1] += weight * fg_ptr[1];
                fg[2] += weight * fg_ptr[2];
                fg[3] += weight * fg_ptr[3];

                fg_ptr = ptr1 + x2;
                weight = (weight_array[x_hr] * 
                          weight_array[y_hr + image_subpixel_size] + 
                          image_filter_size / 2) >> 
                          image_filter_shift;
                fg[0] += weight * fg_ptr[0];
                fg[1] += weight * fg_ptr[1];
                fg[2] += weight * fg_ptr[2];
                fg[3] += weight * fg_ptr[3];

                fg_ptr = ptr2 + x1;
                weight = (weight_array[x_hr + image_subpixel_size] * 
                          weight_array[y_hr] + 
                          image_filter_size / 2) >> 
                          image_filter_shift;
                fg[0] += weight * fg_ptr[0];
                fg[1] += weight * fg_ptr[1];
                fg[2] += weight * fg_ptr[2];
                fg[3] += weight * fg_ptr[3];

                fg_ptr = ptr2 + x2;
                weight = (weight_array[x_hr] * 
                          weight_array[y_hr] + 
                          image_filter_size / 2) >> 
                          image_filter_shift;
                fg[0] += weight * fg_ptr[0];
                fg[1] += weight * fg_ptr[1];
                fg[2] += weight * fg_ptr[2];
                fg[3] += weight * fg_ptr[3];

                fg[0] >>= image_filter_shift;
                fg[1] >>= image_filter_shift;
                fg[2] >>= image_filter_shift;
                fg[3] >>= image_filter_shift;

                if(fg[order_type::A] > base_mask)         fg[order_type::A] = base_mask;
                if(fg[order_type::R] > fg[order_type::A]) fg[order_type::R] = fg[order_type::A];
                if(fg[order_type::G] > fg[order_type::A]) fg[order_type::G] = fg[order_type::A];
                if(fg[order_type::B] > fg[order_type::A]) fg[order_type::B] = fg[order_type::A];

                span->r = (value_type)fg[order_type::R];
                span->g = (value_type)fg[order_type::G];
                span->b = (value_type)fg[order_type::B];
                span->a = (value_type)fg[order_type::A];
                ++span;
                ++base_type::interpolator();

            } while(--len);

            return base_type::allocator().span();
        }
    private:
        WrapModeX m_wrap_mode_x;
        WrapModeY m_wrap_mode_y;
    };












    //==============================================span_pattern_filter_rgba
    template<class ColorT,
             class Order, 
             class Interpolator,
             class WrapModeX,
             class WrapModeY,
             class Allocator = span_allocator<ColorT> > 
    class span_pattern_filter_rgba : 
    public span_image_filter<ColorT, Interpolator, Allocator>
    {
    public:
        typedef ColorT color_type;
        typedef Order order_type;
        typedef Interpolator interpolator_type;
        typedef Allocator alloc_type;
        typedef span_image_filter<color_type, interpolator_type, alloc_type> base_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        enum
        {
            base_shift = color_type::base_shift,
            base_mask  = color_type::base_mask
        };

        //--------------------------------------------------------------------
        span_pattern_filter_rgba(alloc_type& alloc) : 
            base_type(alloc)
        {}

        //--------------------------------------------------------------------
        span_pattern_filter_rgba(alloc_type& alloc,
                                 const rendering_buffer& src, 
                                 interpolator_type& inter,
                                 const image_filter_lut& filter) :
            base_type(alloc, src, color_type(0,0,0,0), inter, &filter),
            m_wrap_mode_x(src.width()),
            m_wrap_mode_y(src.height())
        {}

        //--------------------------------------------------------------------
        void source_image(const rendering_buffer& src) 
        { 
            base_type::source_image(src);
            m_wrap_mode_x = WrapModeX(src.width());
            m_wrap_mode_y = WrapModeX(src.height());
        }

        //--------------------------------------------------------------------
        color_type* generate(int x, int y, unsigned len)
        {
            base_type::interpolator().begin(x + base_type::filter_dx_dbl(), 
                                            y + base_type::filter_dy_dbl(), len);
            int fg[4];

            unsigned   diameter     = base_type::filter().diameter();
            int        start        = base_type::filter().start();
            const int16* weight_array = base_type::filter().weight_array();

            color_type* span = base_type::allocator().span();

            int x_count; 
            int weight_y;

            do
            {
                base_type::interpolator().coordinates(&x, &y);

                x -= base_type::filter_dx_int();
                y -= base_type::filter_dy_int();

                int x_hr = x; 
                int y_hr = y; 

                int x_fract = x_hr & image_subpixel_mask;
                unsigned y_count = diameter;

                int y_lr  = m_wrap_mode_y((y >> image_subpixel_shift) + start);
                int x_int = (x >> image_subpixel_shift) + start;
                int x_lr;

                y_hr = image_subpixel_mask - (y_hr & image_subpixel_mask);
                fg[0] = fg[1] = fg[2] = fg[3] = image_filter_size / 2;

                do
                {
                    x_count = diameter;
                    weight_y = weight_array[y_hr];
                    x_hr = image_subpixel_mask - x_fract;
                    x_lr = m_wrap_mode_x(x_int);
                    const value_type* row_ptr = (value_type*)base_type::source_image().row(y_lr);
                    do
                    {
                        const value_type* fg_ptr = row_ptr + (x_lr << 2);
                        int weight = (weight_y * weight_array[x_hr] + 
                                     image_filter_size / 2) >> 
                                     image_filter_shift;
        
                        fg[0] += fg_ptr[0] * weight;
                        fg[1] += fg_ptr[1] * weight;
                        fg[2] += fg_ptr[2] * weight;
                        fg[3] += fg_ptr[3] * weight;

                        x_hr += image_subpixel_size;
                        x_lr = ++m_wrap_mode_x;
                    } while(--x_count);

                    y_hr += image_subpixel_size;
                    y_lr = ++m_wrap_mode_y;
                } while(--y_count);

                fg[0] >>= image_filter_shift;
                fg[1] >>= image_filter_shift;
                fg[2] >>= image_filter_shift;
                fg[3] >>= image_filter_shift;

                if(fg[0] < 0) fg[0] = 0;
                if(fg[1] < 0) fg[1] = 0;
                if(fg[2] < 0) fg[2] = 0;
                if(fg[3] < 0) fg[3] = 0;

                if(fg[order_type::A] > base_mask)         fg[order_type::A] = base_mask;
                if(fg[order_type::R] > fg[order_type::A]) fg[order_type::R] = fg[order_type::A];
                if(fg[order_type::G] > fg[order_type::A]) fg[order_type::G] = fg[order_type::A];
                if(fg[order_type::B] > fg[order_type::A]) fg[order_type::B] = fg[order_type::A];

                span->r = fg[order_type::R];
                span->g = fg[order_type::G];
                span->b = fg[order_type::B];
                span->a = fg[order_type::A];
                ++span;
                ++base_type::interpolator();

            } while(--len);

            return base_type::allocator().span();
        }

    private:
        WrapModeX m_wrap_mode_x;
        WrapModeY m_wrap_mode_y;
    };


}


#endif



