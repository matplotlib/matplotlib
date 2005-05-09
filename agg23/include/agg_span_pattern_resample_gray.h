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

#ifndef AGG_SPAN_PATTERN_RESAMPLE_GRAY_INCLUDED
#define AGG_SPAN_PATTERN_RESAMPLE_GRAY_INCLUDED

#include "agg_color_gray.h"
#include "agg_span_image_resample.h"

namespace agg
{

    //=======================================span_pattern_resample_gray_affine
    template<class ColorT,
             class WrapModeX,
             class WrapModeY,
             class Allocator = span_allocator<ColorT> > 
    class span_pattern_resample_gray_affine : 
    public span_image_resample_affine<ColorT, Allocator>
    {
    public:
        typedef ColorT color_type;
        typedef Allocator alloc_type;
        typedef span_image_resample_affine<color_type, alloc_type> base_type;
        typedef typename base_type::interpolator_type interpolator_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::long_type long_type;
        enum
        {
            base_shift      = color_type::base_shift,
            base_mask       = color_type::base_mask,
            downscale_shift = image_filter_shift
        };

        //--------------------------------------------------------------------
        span_pattern_resample_gray_affine(alloc_type& alloc) : 
            base_type(alloc),
            m_wrap_mode_x(1),
            m_wrap_mode_y(1) 
        {}

        //--------------------------------------------------------------------
        span_pattern_resample_gray_affine(alloc_type& alloc,
                                          const rendering_buffer& src, 
                                          interpolator_type& inter,
                                          const image_filter_lut& filter) :
            base_type(alloc, src, color_type(0,0), inter, filter),
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
            color_type* span = base_type::allocator().span();
            interpolator_type& intr = base_type::interpolator();
            intr.begin(x + base_type::filter_dx_dbl(), 
                       y + base_type::filter_dy_dbl(), len);
            long_type fg;

            int diameter = base_type::filter().diameter();
            int filter_size = diameter << image_subpixel_shift;
            int radius_x = (diameter * base_type::m_rx) >> 1;
            int radius_y = (diameter * base_type::m_ry) >> 1;
            int maxx = base_type::source_image().width() - 1;
            int maxy = base_type::source_image().height() - 1;
            const int16* weight_array = base_type::filter().weight_array();

            do
            {
                intr.coordinates(&x, &y);

                x += base_type::filter_dx_int() - radius_x;
                y += base_type::filter_dy_int() - radius_y;

                fg = image_filter_size / 2;

                int y_lr  = m_wrap_mode_y(y >> image_subpixel_shift);
                int y_hr = ((image_subpixel_mask - (y & image_subpixel_mask)) * 
                                base_type::m_ry_inv) >> 
                                    image_subpixel_shift;
                int total_weight = 0;
                int x_lr_ini = x >> image_subpixel_shift;
                int x_hr_ini = ((image_subpixel_mask - (x & image_subpixel_mask)) * 
                                   base_type::m_rx_inv) >> 
                                       image_subpixel_shift;
                do
                {
                    int weight_y = weight_array[y_hr];
                    int x_lr = m_wrap_mode_x(x_lr_ini);
                    int x_hr = x_hr_ini;
                    const value_type* row_ptr = (const value_type*)base_type::source_image().row(y_lr);
                    do
                    {
                        const value_type* fg_ptr = row_ptr + x_lr;
                        int weight = (weight_y * weight_array[x_hr] + 
                                     image_filter_size / 2) >> 
                                     downscale_shift;

                        fg += *fg_ptr * weight;
                        total_weight += weight;
                        x_hr   += base_type::m_rx_inv;
                        x_lr = ++m_wrap_mode_x;
                    }
                    while(x_hr < filter_size);

                    y_hr += base_type::m_ry_inv;
                    y_lr = ++m_wrap_mode_y;
                } while(y_hr < filter_size);

                fg /= total_weight;

                if(fg < 0)         fg = 0;
                if(fg > base_mask) fg = base_mask;

                span->v = (value_type)fg;
                span->a = (value_type)base_mask;

                ++span;
                ++intr;
            } while(--len);
            return base_type::allocator().span();
        }

    private:
        WrapModeX m_wrap_mode_x;
        WrapModeY m_wrap_mode_y;
    };







    //============================================span_pattern_resample_gray
    template<class ColorT,
             class Interpolator, 
             class WrapModeX,
             class WrapModeY,
             class Allocator = span_allocator<ColorT> >
    class span_pattern_resample_gray : 
    public span_image_resample<ColorT, Interpolator, Allocator>
    {
    public:
        typedef ColorT color_type;
        typedef Interpolator interpolator_type;
        typedef Allocator alloc_type;
        typedef span_image_resample<color_type, interpolator_type, alloc_type> base_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::long_type long_type;
        enum
        {
            base_shift = color_type::base_shift,
            base_mask  = color_type::base_mask,
            downscale_shift = image_filter_shift
        };

        //--------------------------------------------------------------------
        span_pattern_resample_gray(alloc_type& alloc) : 
            base_type(alloc),
            m_wrap_mode_x(1),
            m_wrap_mode_y(1)
        {}

        //--------------------------------------------------------------------
        span_pattern_resample_gray(alloc_type& alloc,
                                   const rendering_buffer& src, 
                                   interpolator_type& inter,
                                   const image_filter_lut& filter) :
            base_type(alloc, src, color_type(0,0), inter, filter),
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
            color_type* span = base_type::allocator().span();
            interpolator_type& intr = base_type::interpolator();
            intr.begin(x + base_type::filter_dx_dbl(), 
                       y + base_type::filter_dy_dbl(), len);
            long_type fg;

            int diameter = base_type::filter().diameter();
            int filter_size = diameter << image_subpixel_shift;
            const int16* weight_array = base_type::filter().weight_array();

            do
            {
                int rx;
                int ry;
                int rx_inv = image_subpixel_size;
                int ry_inv = image_subpixel_size;
                intr.coordinates(&x,  &y);
                intr.local_scale(&rx, &ry);

                rx = (rx * base_type::m_blur_x) >> image_subpixel_shift;
                ry = (ry * base_type::m_blur_y) >> image_subpixel_shift;

                if(rx < image_subpixel_size)
                {
                    rx = image_subpixel_size;
                }
                else
                {
                    if(rx > image_subpixel_size * base_type::m_scale_limit) 
                    {
                        rx = image_subpixel_size * base_type::m_scale_limit;
                    }
                    rx_inv = image_subpixel_size * image_subpixel_size / rx;
                }

                if(ry < image_subpixel_size)
                {
                    ry = image_subpixel_size;
                }
                else
                {
                    if(ry > image_subpixel_size * base_type::m_scale_limit) 
                    {
                        ry = image_subpixel_size * base_type::m_scale_limit;
                    }
                    ry_inv = image_subpixel_size * image_subpixel_size / ry;
                }

                int radius_x = (diameter * rx) >> 1;
                int radius_y = (diameter * ry) >> 1;
                int maxx = base_type::source_image().width() - 1;
                int maxy = base_type::source_image().height() - 1;

                x += base_type::filter_dx_int() - radius_x;
                y += base_type::filter_dy_int() - radius_y;

                fg = image_filter_size / 2;

                int y_lr  = m_wrap_mode_y(y >> image_subpixel_shift);
                int y_hr = ((image_subpixel_mask - (y & image_subpixel_mask)) * 
                               ry_inv) >> 
                                   image_subpixel_shift;
                int total_weight = 0;
                int x_lr_ini = x >> image_subpixel_shift;
                int x_hr_ini = ((image_subpixel_mask - (x & image_subpixel_mask)) * 
                                   rx_inv) >> 
                                       image_subpixel_shift;

                do
                {
                    int weight_y = weight_array[y_hr];
                    int x_lr = m_wrap_mode_x(x_lr_ini);
                    int x_hr = x_hr_ini;
                    const value_type* row_ptr = (const value_type*)base_type::source_image().row(y_lr);
                    do
                    {
                        const value_type* fg_ptr = row_ptr + x_lr;
                        int weight = (weight_y * weight_array[x_hr] + 
                                     image_filter_size / 2) >> 
                                     downscale_shift;
                        fg += *fg_ptr * weight;
                        total_weight += weight;
                        x_hr   += rx_inv;
                        x_lr = ++m_wrap_mode_x;
                    }
                    while(x_hr < filter_size);
                    y_hr += ry_inv;
                    y_lr = ++m_wrap_mode_y;
                }
                while(y_hr < filter_size);

                fg /= total_weight;

                if(fg < 0)         fg = 0;
                if(fg > base_mask) fg = base_mask;

                span->v = (value_type)fg;
                span->a = (value_type)base_mask;

                ++span;
                ++intr;
            } while(--len);
            return base_type::allocator().span();
        }

    private:
        WrapModeX m_wrap_mode_x;
        WrapModeY m_wrap_mode_y;
    };

}


#endif
