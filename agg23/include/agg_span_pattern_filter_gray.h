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
// classes span_pattern_filter_gray*
//
//----------------------------------------------------------------------------
#ifndef AGG_SPAN_PATTERN_FILTER_GRAY_INCLUDED
#define AGG_SPAN_PATTERN_FILTER_GRAY_INCLUDED

#include "agg_basics.h"
#include "agg_color_gray.h"
#include "agg_span_pattern.h"
#include "agg_span_image_filter.h"


namespace agg
{

    //===========================================span_pattern_filter_gray
    template<class ColorT,
             class Interpolator,
             class WrapModeX,
             class WrapModeY,
             class Allocator = span_allocator<ColorT> > 
    class span_pattern_filter_gray_nn : 
    public span_image_filter<ColorT, Interpolator, Allocator>
    {
    public:
        typedef ColorT color_type;
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
        span_pattern_filter_gray_nn(alloc_type& alloc) : 
            base_type(alloc),
            m_wrap_mode_x(1),
            m_wrap_mode_y(1)
        {}

        //--------------------------------------------------------------------
        span_pattern_filter_gray_nn(alloc_type& alloc,
                                    const rendering_buffer& src, 
                                    interpolator_type& inter) :
            base_type(alloc, src, color_type(0,0), inter, 0),
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
            color_type* span = base_type::allocator().span();
            do
            {
                base_type::interpolator().coordinates(&x, &y);

                x = m_wrap_mode_x(x >> image_subpixel_shift);
                y = m_wrap_mode_y(y >> image_subpixel_shift);

                span->v = *((value_type*)base_type::source_image().row(y) + x);
                span->a = base_mask;
                ++span;
                ++base_type::interpolator();

            } while(--len);

            return base_type::allocator().span();
        }

    private:
        WrapModeX m_wrap_mode_x;
        WrapModeY m_wrap_mode_y;
    };






    //====================================span_pattern_filter_gray_bilinear
    template<class ColorT,
             class Interpolator,
             class WrapModeX,
             class WrapModeY,
             class Allocator = span_allocator<ColorT> > 
    class span_pattern_filter_gray_bilinear : 
    public span_image_filter<ColorT, Interpolator, Allocator>
    {
    public:
        typedef ColorT color_type;
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
        span_pattern_filter_gray_bilinear(alloc_type& alloc) : 
            base_type(alloc),
            m_wrap_mode_x(1),
            m_wrap_mode_y(1)
        {}

        //--------------------------------------------------------------------
        span_pattern_filter_gray_bilinear(alloc_type& alloc,
                                          const rendering_buffer& src, 
                                          interpolator_type& inter) :
            base_type(alloc, src, color_type(0,0), inter, 0),
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
            calc_type fg;
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

                unsigned y1 = m_wrap_mode_y(y_lr);
                unsigned y2 = ++m_wrap_mode_y;
                const value_type* ptr1 = (value_type*)base_type::source_image().row(y1);
                const value_type* ptr2 = (value_type*)base_type::source_image().row(y2);

                fg = image_subpixel_size * image_subpixel_size / 2;

                x_hr &= image_subpixel_mask;
                y_hr &= image_subpixel_mask;

                fg += ptr1[x1] * (image_subpixel_size - x_hr) * (image_subpixel_size - y_hr);
                fg += ptr1[x2] * x_hr * (image_subpixel_size - y_hr);
                fg += ptr2[x1] * (image_subpixel_size - x_hr) * y_hr;
                fg += ptr2[x2] * x_hr * y_hr;

                span->v = (value_type)(fg >> image_subpixel_shift * 2);
                span->a = base_mask;
                ++span;
                ++base_type::interpolator();

            } while(--len);

            return base_type::allocator().span();
        }
    private:
        WrapModeX m_wrap_mode_x;
        WrapModeY m_wrap_mode_y;
    };









    //====================================span_pattern_filter_gray_2x2
    template<class ColorT,
             class Interpolator,
             class WrapModeX,
             class WrapModeY,
             class Allocator = span_allocator<ColorT> > 
    class span_pattern_filter_gray_2x2 : 
    public span_image_filter<ColorT, Interpolator, Allocator>
    {
    public:
        typedef ColorT color_type;
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
        span_pattern_filter_gray_2x2(alloc_type& alloc) : 
            base_type(alloc),
            m_wrap_mode_x(1),
            m_wrap_mode_y(1)
        {}

        //--------------------------------------------------------------------
        span_pattern_filter_gray_2x2(alloc_type& alloc,
                                     const rendering_buffer& src, 
                                     interpolator_type& inter,
                                     const image_filter_lut& filter) :
            base_type(alloc, src, color_type(0,0), inter, &filter),
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
            calc_type fg;
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

                unsigned y1 = m_wrap_mode_y(y_lr);
                unsigned y2 = ++m_wrap_mode_y;
                const value_type* ptr1 = (value_type*)base_type::source_image().row(y1);
                const value_type* ptr2 = (value_type*)base_type::source_image().row(y2);

                fg = image_filter_size / 2;

                x_hr &= image_subpixel_mask;
                y_hr &= image_subpixel_mask;

                fg += ptr1[x1] * ((weight_array[x_hr + image_subpixel_size] * 
                                   weight_array[y_hr + image_subpixel_size] + 
                                   image_filter_size / 2) >> 
                                   image_filter_shift);
                fg += ptr1[x2] * ((weight_array[x_hr] * 
                                   weight_array[y_hr + image_subpixel_size] + 
                                   image_filter_size / 2) >> 
                                   image_filter_shift);
                fg += ptr2[x1] * ((weight_array[x_hr + image_subpixel_size] * 
                                   weight_array[y_hr] + 
                                   image_filter_size / 2) >> 
                                   image_filter_shift);
                fg += ptr2[x2] * ((weight_array[x_hr] * 
                                   weight_array[y_hr] + 
                                   image_filter_size / 2) >> 
                                   image_filter_shift);

                fg >>= image_filter_shift;
                if(fg > base_mask) fg = base_mask;

                span->v = (value_type)fg;
                span->a = base_mask;
                ++span;
                ++base_type::interpolator();

            } while(--len);

            return base_type::allocator().span();
        }
    private:
        WrapModeX m_wrap_mode_x;
        WrapModeY m_wrap_mode_y;
    };












    //==============================================span_pattern_filter_gray
    template<class ColorT,
             class Interpolator,
             class WrapModeX,
             class WrapModeY,
             class Allocator = span_allocator<ColorT> > 
    class span_pattern_filter_gray : 
    public span_image_filter<ColorT, Interpolator, Allocator>
    {
    public:
        typedef ColorT color_type;
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
        span_pattern_filter_gray(alloc_type& alloc) : 
            base_type(alloc)
        {}

        //--------------------------------------------------------------------
        span_pattern_filter_gray(alloc_type& alloc,
                                 const rendering_buffer& src, 
                                 interpolator_type& inter,
                                 const image_filter_lut& filter) :
            base_type(alloc, src, color_type(0,0), inter, &filter),
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
            int fg;

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
                fg = image_filter_size / 2;

                do
                {
                    x_count = diameter;
                    weight_y = weight_array[y_hr];
                    x_hr = image_subpixel_mask - x_fract;
                    x_lr = m_wrap_mode_x(x_int);
                    const value_type* row_ptr = (value_type*)base_type::source_image().row(y_lr);
                    do
                    {
                        fg += row_ptr[x_lr] * ((weight_y * weight_array[x_hr] + 
                                               image_filter_size / 2) >> 
                                               image_filter_shift);
                        x_hr += image_subpixel_size;
                        x_lr = ++m_wrap_mode_x;
                    } while(--x_count);

                    y_hr += image_subpixel_size;
                    y_lr = ++m_wrap_mode_y;
                } while(--y_count);

                fg >>= image_filter_shift;

                if(fg < 0)         fg = 0;
                if(fg > base_mask) fg = base_mask;

                span->v = fg;
                span->a = base_mask;
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



