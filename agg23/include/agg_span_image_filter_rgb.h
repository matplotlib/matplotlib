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

#ifndef AGG_SPAN_IMAGE_FILTER_RGB_INCLUDED
#define AGG_SPAN_IMAGE_FILTER_RGB_INCLUDED

#include "agg_basics.h"
#include "agg_color_rgba.h"
#include "agg_span_image_filter.h"


namespace agg
{


    //==============================================span_image_filter_rgb_nn
    template<class ColorT,
             class Order, 
             class Interpolator,
             class Allocator = span_allocator<ColorT> > 
    class span_image_filter_rgb_nn : 
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
        span_image_filter_rgb_nn(alloc_type& alloc) : base_type(alloc) {}

        //--------------------------------------------------------------------
        span_image_filter_rgb_nn(alloc_type& alloc,
                                 const rendering_buffer& src, 
                                 const color_type& back_color,
                                 interpolator_type& inter) :
            base_type(alloc, src, back_color, inter, 0) 
        {}

        //--------------------------------------------------------------------
        color_type* generate(int x, int y, unsigned len)
        {
            base_type::interpolator().begin(x + base_type::filter_dx_dbl(), 
                                            y + base_type::filter_dy_dbl(), len);

            calc_type fg[3];
            calc_type src_alpha;

            const value_type *fg_ptr;
            color_type* span = base_type::allocator().span();

            int maxx = base_type::source_image().width() - 1;
            int maxy = base_type::source_image().height() - 1;

            do
            {
                base_type::interpolator().coordinates(&x, &y);

                x >>= image_subpixel_shift;
                y >>= image_subpixel_shift;

                if(x >= 0    && y >= 0 &&
                   x <= maxx && y <= maxy) 
                {
                    fg_ptr = (const value_type*)base_type::source_image().row(y) + x + x + x;
                    fg[0] = *fg_ptr++;
                    fg[1] = *fg_ptr++;
                    fg[2] = *fg_ptr++;
                    src_alpha = base_mask;
                }
                else
                {
                    fg[order_type::R] = base_type::background_color().r;
                    fg[order_type::G] = base_type::background_color().g;
                    fg[order_type::B] = base_type::background_color().b;
                    src_alpha         = base_type::background_color().a;
                }

                span->r = (value_type)fg[order_type::R];
                span->g = (value_type)fg[order_type::G];
                span->b = (value_type)fg[order_type::B];
                span->a = (value_type)src_alpha;
                ++span;
                ++base_type::interpolator();

            } while(--len);

            return base_type::allocator().span();
        }
    };




    //=========================================span_image_filter_rgb_bilinear
    template<class ColorT,
             class Order, 
             class Interpolator, 
             class Allocator = span_allocator<ColorT> > 
    class span_image_filter_rgb_bilinear : 
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
        span_image_filter_rgb_bilinear(alloc_type& alloc) : base_type(alloc) {}

        //--------------------------------------------------------------------
        span_image_filter_rgb_bilinear(alloc_type& alloc,
                                       const rendering_buffer& src, 
                                       const color_type& back_color,
                                       interpolator_type& inter) :
            base_type(alloc, src, back_color, inter, 0) 
        {}

        //--------------------------------------------------------------------
        color_type* generate(int x, int y, unsigned len)
        {
            base_type::interpolator().begin(x + base_type::filter_dx_dbl(), 
                                            y + base_type::filter_dy_dbl(), len);
            calc_type fg[3];
            calc_type src_alpha;
            value_type back_r = base_type::background_color().r;
            value_type back_g = base_type::background_color().g;
            value_type back_b = base_type::background_color().b;
            value_type back_a = base_type::background_color().a;

            const value_type *fg_ptr;

            color_type* span = base_type::allocator().span();

            int maxx = base_type::source_image().width() - 1;
            int maxy = base_type::source_image().height() - 1;

            do
            {
                int x_hr;
                int y_hr;
                
                base_type::interpolator().coordinates(&x_hr, &y_hr);

                x_hr -= base_type::filter_dx_int();
                y_hr -= base_type::filter_dy_int();

                int x_lr = x_hr >> image_subpixel_shift;
                int y_lr = y_hr >> image_subpixel_shift;
                unsigned weight;

                if(x_lr >= 0    && y_lr >= 0 &&
                   x_lr <  maxx && y_lr <  maxy) 
                {
                    fg[0] = 
                    fg[1] = 
                    fg[2] = image_subpixel_size * image_subpixel_size / 2;

                    x_hr &= image_subpixel_mask;
                    y_hr &= image_subpixel_mask;
                    fg_ptr = (const value_type*)base_type::source_image().row(y_lr) + x_lr + x_lr + x_lr;

                    weight = (image_subpixel_size - x_hr) * 
                             (image_subpixel_size - y_hr);
                    fg[0] += weight * *fg_ptr++;
                    fg[1] += weight * *fg_ptr++;
                    fg[2] += weight * *fg_ptr++;

                    weight = x_hr * (image_subpixel_size - y_hr);
                    fg[0] += weight * *fg_ptr++;
                    fg[1] += weight * *fg_ptr++;
                    fg[2] += weight * *fg_ptr++;

                    fg_ptr = (const value_type*)base_type::source_image().next_row(fg_ptr - 6);

                    weight = (image_subpixel_size - x_hr) * y_hr;
                    fg[0] += weight * *fg_ptr++;
                    fg[1] += weight * *fg_ptr++;
                    fg[2] += weight * *fg_ptr++;

                    weight = x_hr * y_hr;
                    fg[0] += weight * *fg_ptr++;
                    fg[1] += weight * *fg_ptr++;
                    fg[2] += weight * *fg_ptr++;

                    fg[0] >>= image_subpixel_shift * 2;
                    fg[1] >>= image_subpixel_shift * 2;
                    fg[2] >>= image_subpixel_shift * 2;
                    src_alpha = base_mask;
                }
                else
                {
                    if(x_lr < -1   || y_lr < -1 ||
                       x_lr > maxx || y_lr > maxy)
                    {
                        fg[order_type::R] = back_r;
                        fg[order_type::G] = back_g;
                        fg[order_type::B] = back_b;
                        src_alpha         = back_a;
                    }
                    else
                    {
                        fg[0] = 
                        fg[1] = 
                        fg[2] = 
                        src_alpha = image_subpixel_size * image_subpixel_size / 2;

                        x_hr &= image_subpixel_mask;
                        y_hr &= image_subpixel_mask;

                        weight = (image_subpixel_size - x_hr) * 
                                 (image_subpixel_size - y_hr);
                        if(x_lr >= 0    && y_lr >= 0 &&
                           x_lr <= maxx && y_lr <= maxy)
                        {
                            fg_ptr = (const value_type*)base_type::source_image().row(y_lr) + x_lr + x_lr + x_lr;
                            fg[0] += weight * *fg_ptr++;
                            fg[1] += weight * *fg_ptr++;
                            fg[2] += weight * *fg_ptr++;
                            src_alpha += weight * base_mask;
                        }
                        else
                        {
                            fg[order_type::R] += back_r * weight;
                            fg[order_type::G] += back_g * weight;
                            fg[order_type::B] += back_b * weight;
                            src_alpha         += back_a * weight;
                        }

                        x_lr++;

                        weight = x_hr * (image_subpixel_size - y_hr);
                        if(x_lr >= 0    && y_lr >= 0 &&
                           x_lr <= maxx && y_lr <= maxy)
                        {
                            fg_ptr = (const value_type*)base_type::source_image().row(y_lr) + x_lr + x_lr + x_lr;
                            fg[0]     += weight * *fg_ptr++;
                            fg[1]     += weight * *fg_ptr++;
                            fg[2]     += weight * *fg_ptr++;
                            src_alpha += weight * base_mask;
                        }
                        else
                        {
                            fg[order_type::R] += back_r * weight;
                            fg[order_type::G] += back_g * weight;
                            fg[order_type::B] += back_b * weight;
                            src_alpha         += back_a * weight;
                        }

                        x_lr--;
                        y_lr++;

                        weight = (image_subpixel_size - x_hr) * y_hr;
                        if(x_lr >= 0    && y_lr >= 0 &&
                           x_lr <= maxx && y_lr <= maxy)
                        {
                            fg_ptr = (const value_type*)base_type::source_image().row(y_lr) + x_lr + x_lr + x_lr;
                            fg[0] += weight * *fg_ptr++;
                            fg[1] += weight * *fg_ptr++;
                            fg[2] += weight * *fg_ptr++;
                            src_alpha += weight * base_mask;
                        }
                        else
                        {
                            fg[order_type::R] += back_r * weight;
                            fg[order_type::G] += back_g * weight;
                            fg[order_type::B] += back_b * weight;
                            src_alpha         += back_a * weight;
                        }

                        x_lr++;

                        weight = x_hr * y_hr;
                        if(x_lr >= 0    && y_lr >= 0 &&
                           x_lr <= maxx && y_lr <= maxy)
                        {
                            fg_ptr = (const value_type*)base_type::source_image().row(y_lr) + x_lr + x_lr + x_lr;
                            fg[0] += weight * *fg_ptr++;
                            fg[1] += weight * *fg_ptr++;
                            fg[2] += weight * *fg_ptr++;
                            src_alpha += weight * base_mask;
                        }
                        else
                        {
                            fg[order_type::R] += back_r * weight;
                            fg[order_type::G] += back_g * weight;
                            fg[order_type::B] += back_b * weight;
                            src_alpha         += back_a * weight;
                        }

                        fg[0] >>= image_subpixel_shift * 2;
                        fg[1] >>= image_subpixel_shift * 2;
                        fg[2] >>= image_subpixel_shift * 2;
                        src_alpha >>= image_subpixel_shift * 2;
                    }
                }

                span->r = (value_type)fg[order_type::R];
                span->g = (value_type)fg[order_type::G];
                span->b = (value_type)fg[order_type::B];
                span->a = (value_type)src_alpha;
                ++span;
                ++base_type::interpolator();

            } while(--len);

            return base_type::allocator().span();
        }
    };








    //=========================================span_image_filter_rgb_2x2
    template<class ColorT,
             class Order, 
             class Interpolator, 
             class Allocator = span_allocator<ColorT> > 
    class span_image_filter_rgb_2x2 : 
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
        span_image_filter_rgb_2x2(alloc_type& alloc) : base_type(alloc) {}

        //--------------------------------------------------------------------
        span_image_filter_rgb_2x2(alloc_type& alloc,
                                  const rendering_buffer& src, 
                                  const color_type& back_color,
                                  interpolator_type& inter,
                                  const image_filter_lut& filter) :
            base_type(alloc, src, back_color, inter, &filter) 
        {}

        //--------------------------------------------------------------------
        color_type* generate(int x, int y, unsigned len)
        {
            base_type::interpolator().begin(x + base_type::filter_dx_dbl(), 
                                            y + base_type::filter_dy_dbl(), len);
            calc_type fg[3];
            calc_type src_alpha;
            value_type back_r = base_type::background_color().r;
            value_type back_g = base_type::background_color().g;
            value_type back_b = base_type::background_color().b;
            value_type back_a = base_type::background_color().a;

            const value_type *fg_ptr;

            color_type* span = base_type::allocator().span();
            const int16* weight_array = base_type::filter().weight_array() + 
                                        ((base_type::filter().diameter()/2 - 1) << 
                                          image_subpixel_shift);

            int maxx = base_type::source_image().width() - 1;
            int maxy = base_type::source_image().height() - 1;

            do
            {
                int x_hr;
                int y_hr;
                
                base_type::interpolator().coordinates(&x_hr, &y_hr);

                x_hr -= base_type::filter_dx_int();
                y_hr -= base_type::filter_dy_int();

                int x_lr = x_hr >> image_subpixel_shift;
                int y_lr = y_hr >> image_subpixel_shift;
                unsigned weight;

                if(x_lr >= 0    && y_lr >= 0 &&
                   x_lr <  maxx && y_lr <  maxy) 
                {
                    fg[0] = fg[1] = fg[2] = image_filter_size / 2;

                    x_hr &= image_subpixel_mask;
                    y_hr &= image_subpixel_mask;
                    fg_ptr = (const value_type*)base_type::source_image().row(y_lr) + x_lr + x_lr + x_lr;

                    weight = (weight_array[x_hr + image_subpixel_size] * 
                              weight_array[y_hr + image_subpixel_size] + 
                              image_filter_size / 2) >> 
                              image_filter_shift;
                    fg[0] += weight * *fg_ptr++;
                    fg[1] += weight * *fg_ptr++;
                    fg[2] += weight * *fg_ptr++;

                    weight = (weight_array[x_hr] * 
                              weight_array[y_hr + image_subpixel_size] + 
                              image_filter_size / 2) >> 
                              image_filter_shift;
                    fg[0] += weight * *fg_ptr++;
                    fg[1] += weight * *fg_ptr++;
                    fg[2] += weight * *fg_ptr++;

                    fg_ptr = (const value_type*)base_type::source_image().next_row(fg_ptr - 6);

                    weight = (weight_array[x_hr + image_subpixel_size] * 
                              weight_array[y_hr] + 
                              image_filter_size / 2) >> 
                              image_filter_shift;
                    fg[0] += weight * *fg_ptr++;
                    fg[1] += weight * *fg_ptr++;
                    fg[2] += weight * *fg_ptr++;

                    weight = (weight_array[x_hr] * 
                              weight_array[y_hr] + 
                              image_filter_size / 2) >> 
                              image_filter_shift;
                    fg[0] += weight * *fg_ptr++;
                    fg[1] += weight * *fg_ptr++;
                    fg[2] += weight * *fg_ptr++;

                    fg[0] >>= image_filter_shift;
                    fg[1] >>= image_filter_shift;
                    fg[2] >>= image_filter_shift;
                    src_alpha = base_mask;

                    if(fg[0] > base_mask) fg[0] = base_mask;
                    if(fg[1] > base_mask) fg[1] = base_mask;
                    if(fg[2] > base_mask) fg[2] = base_mask;
                }
                else
                {
                    if(x_lr < -1   || y_lr < -1 ||
                       x_lr > maxx || y_lr > maxy)
                    {
                        fg[order_type::R] = back_r;
                        fg[order_type::G] = back_g;
                        fg[order_type::B] = back_b;
                        src_alpha         = back_a;
                    }
                    else
                    {
                        fg[0] = fg[1] = fg[2] = src_alpha = image_filter_size / 2;

                        x_hr &= image_subpixel_mask;
                        y_hr &= image_subpixel_mask;

                        weight = (weight_array[x_hr + image_subpixel_size] * 
                                  weight_array[y_hr + image_subpixel_size] + 
                                  image_filter_size / 2) >> 
                                  image_filter_shift;
                        if(x_lr >= 0    && y_lr >= 0 &&
                           x_lr <= maxx && y_lr <= maxy)
                        {
                            fg_ptr = (const value_type*)base_type::source_image().row(y_lr) + x_lr + x_lr + x_lr;
                            fg[0] += weight * *fg_ptr++;
                            fg[1] += weight * *fg_ptr++;
                            fg[2] += weight * *fg_ptr++;
                            src_alpha += weight * base_mask;
                        }
                        else
                        {
                            fg[order_type::R] += back_r * weight;
                            fg[order_type::G] += back_g * weight;
                            fg[order_type::B] += back_b * weight;
                            src_alpha         += back_a * weight;
                        }

                        x_lr++;

                        weight = (weight_array[x_hr] * 
                                  weight_array[y_hr + image_subpixel_size] + 
                                  image_filter_size / 2) >> 
                                  image_filter_shift;
                        if(x_lr >= 0    && y_lr >= 0 &&
                           x_lr <= maxx && y_lr <= maxy)
                        {
                            fg_ptr = (const value_type*)base_type::source_image().row(y_lr) + x_lr + x_lr + x_lr;
                            fg[0]     += weight * *fg_ptr++;
                            fg[1]     += weight * *fg_ptr++;
                            fg[2]     += weight * *fg_ptr++;
                            src_alpha += weight * base_mask;
                        }
                        else
                        {
                            fg[order_type::R] += back_r * weight;
                            fg[order_type::G] += back_g * weight;
                            fg[order_type::B] += back_b * weight;
                            src_alpha         += back_a * weight;
                        }

                        x_lr--;
                        y_lr++;

                        weight = (weight_array[x_hr + image_subpixel_size] * 
                                  weight_array[y_hr] + 
                                  image_filter_size / 2) >> 
                                  image_filter_shift;
                        if(x_lr >= 0    && y_lr >= 0 &&
                           x_lr <= maxx && y_lr <= maxy)
                        {
                            fg_ptr = (const value_type*)base_type::source_image().row(y_lr) + x_lr + x_lr + x_lr;
                            fg[0] += weight * *fg_ptr++;
                            fg[1] += weight * *fg_ptr++;
                            fg[2] += weight * *fg_ptr++;
                            src_alpha += weight * base_mask;
                        }
                        else
                        {
                            fg[order_type::R] += back_r * weight;
                            fg[order_type::G] += back_g * weight;
                            fg[order_type::B] += back_b * weight;
                            src_alpha         += back_a * weight;
                        }

                        x_lr++;

                        weight = (weight_array[x_hr] * 
                                  weight_array[y_hr] + 
                                  image_filter_size / 2) >> 
                                  image_filter_shift;
                        if(x_lr >= 0    && y_lr >= 0 &&
                           x_lr <= maxx && y_lr <= maxy)
                        {
                            fg_ptr = (const value_type*)base_type::source_image().row(y_lr) + x_lr + x_lr + x_lr;
                            fg[0] += weight * *fg_ptr++;
                            fg[1] += weight * *fg_ptr++;
                            fg[2] += weight * *fg_ptr++;
                            src_alpha += weight * base_mask;
                        }
                        else
                        {
                            fg[order_type::R] += back_r * weight;
                            fg[order_type::G] += back_g * weight;
                            fg[order_type::B] += back_b * weight;
                            src_alpha         += back_a * weight;
                        }

                        fg[0]     >>= image_filter_shift;
                        fg[1]     >>= image_filter_shift;
                        fg[2]     >>= image_filter_shift;
                        src_alpha >>= image_filter_shift;

                        if(src_alpha > base_mask) src_alpha = base_mask;
                        if(fg[0] > src_alpha) fg[0] = src_alpha;
                        if(fg[1] > src_alpha) fg[1] = src_alpha;
                        if(fg[2] > src_alpha) fg[2] = src_alpha;
                    }
                }

                span->r = (value_type)fg[order_type::R];
                span->g = (value_type)fg[order_type::G];
                span->b = (value_type)fg[order_type::B];
                span->a = (value_type)src_alpha;
                ++span;
                ++base_type::interpolator();

            } while(--len);

            return base_type::allocator().span();
        }
    };







    //=================================================span_image_filter_rgb
    template<class ColorT,
             class Order, 
             class Interpolator, 
             class Allocator = span_allocator<ColorT> > 
    class span_image_filter_rgb : 
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
        span_image_filter_rgb(alloc_type& alloc) : base_type(alloc) {}

        //--------------------------------------------------------------------
        span_image_filter_rgb(alloc_type& alloc,
                               const rendering_buffer& src, 
                               const color_type& back_color,
                               interpolator_type& inter,
                               const image_filter_lut& filter) :
            base_type(alloc, src, back_color, inter, &filter) 
        {}

        //--------------------------------------------------------------------
        color_type* generate(int x, int y, unsigned len)
        {
            base_type::interpolator().begin(x + base_type::filter_dx_dbl(), 
                                            y + base_type::filter_dy_dbl(), len);

            int fg[3];
            int src_alpha;
            value_type back_r = base_type::background_color().r;
            value_type back_g = base_type::background_color().g;
            value_type back_b = base_type::background_color().b;
            value_type back_a = base_type::background_color().a;

            const value_type* fg_ptr;

            unsigned   diameter     = base_type::filter().diameter();
            int        start        = base_type::filter().start();
            int        start1       = start - 1;
            const int16* weight_array = base_type::filter().weight_array();

            unsigned step_back = diameter * 3;
            color_type* span = base_type::allocator().span();

            int maxx = base_type::source_image().width() + start - 2;
            int maxy = base_type::source_image().height() + start - 2;

            int maxx2 = base_type::source_image().width() - start - 1;
            int maxy2 = base_type::source_image().height() - start - 1;

            int x_count; 
            int weight_y;

            do
            {
                base_type::interpolator().coordinates(&x, &y);

                x -= base_type::filter_dx_int();
                y -= base_type::filter_dy_int();

                int x_hr = x;
                int y_hr = y;
            
                int x_lr = x_hr >> image_subpixel_shift;
                int y_lr = y_hr >> image_subpixel_shift;

                fg[0] = fg[1] = fg[2] = image_filter_size / 2;

                int x_fract = x_hr & image_subpixel_mask;
                unsigned y_count = diameter;

                if(x_lr >= -start && y_lr >= -start &&
                   x_lr <= maxx   && y_lr <= maxy) 
                {
                    y_hr = image_subpixel_mask - (y_hr & image_subpixel_mask);
                    fg_ptr = (const value_type*)base_type::source_image().row(y_lr + start) + (x_lr + start) * 3;
                    do
                    {
                        x_count = diameter;
                        weight_y = weight_array[y_hr];
                        x_hr = image_subpixel_mask - x_fract;

                        do
                        {
                            int weight = (weight_y * weight_array[x_hr] + 
                                         image_filter_size / 2) >> 
                                         image_filter_shift;
            
                            fg[0] += *fg_ptr++ * weight;
                            fg[1] += *fg_ptr++ * weight;
                            fg[2] += *fg_ptr++ * weight;

                            x_hr += image_subpixel_size;

                        } while(--x_count);

                        y_hr += image_subpixel_size;
                        fg_ptr = (const value_type*)base_type::source_image().next_row(fg_ptr - step_back);

                    } while(--y_count);

                    fg[0] >>= image_filter_shift;
                    fg[1] >>= image_filter_shift;
                    fg[2] >>= image_filter_shift;

                    if(fg[0] < 0) fg[0] = 0;
                    if(fg[1] < 0) fg[1] = 0;
                    if(fg[2] < 0) fg[2] = 0;

                    if(fg[0] > base_mask) fg[0] = base_mask;
                    if(fg[1] > base_mask) fg[1] = base_mask;
                    if(fg[2] > base_mask) fg[2] = base_mask;
                    src_alpha = base_mask;
                }
                else
                {
                    if(x_lr < start1 || y_lr < start1 ||
                       x_lr > maxx2  || y_lr > maxy2) 
                    {
                        fg[order_type::R] = back_r;
                        fg[order_type::G] = back_g;
                        fg[order_type::B] = back_b;
                        src_alpha         = back_a;
                    }
                    else
                    {
                        src_alpha = image_filter_size / 2;
                        y_lr = (y >> image_subpixel_shift) + start;
                        y_hr = image_subpixel_mask - (y_hr & image_subpixel_mask);

                        do
                        {
                            x_count = diameter;
                            weight_y = weight_array[y_hr];
                            x_lr = (x >> image_subpixel_shift) + start;
                            x_hr = image_subpixel_mask - x_fract;

                            do
                            {
                                int weight = (weight_y * weight_array[x_hr] + 
                                             image_filter_size / 2) >> 
                                             image_filter_shift;

                                if(x_lr >= 0 && y_lr >= 0 && 
                                   x_lr < int(base_type::source_image().width()) && 
                                   y_lr < int(base_type::source_image().height()))
                                {
                                    fg_ptr = (const value_type*)base_type::source_image().row(y_lr) + x_lr * 3;
                                    fg[0] += *fg_ptr++ * weight;
                                    fg[1] += *fg_ptr++ * weight;
                                    fg[2] += *fg_ptr++ * weight;
                                    src_alpha += base_mask * weight;
                                }
                                else
                                {
                                    fg[order_type::R] += back_r * weight;
                                    fg[order_type::G] += back_g * weight;
                                    fg[order_type::B] += back_b * weight;
                                    src_alpha         += back_a * weight;
                                }
                                x_hr += image_subpixel_size;
                                x_lr++;

                            } while(--x_count);

                            y_hr += image_subpixel_size;
                            y_lr++;

                        } while(--y_count);


                        fg[0] >>= image_filter_shift;
                        fg[1] >>= image_filter_shift;
                        fg[2] >>= image_filter_shift;
                        src_alpha >>= image_filter_shift;

                        if(fg[0] < 0) fg[0] = 0;
                        if(fg[1] < 0) fg[1] = 0;
                        if(fg[2] < 0) fg[2] = 0;
                        if(src_alpha < 0) src_alpha = 0;

                        if(src_alpha > base_mask) src_alpha = base_mask;
                        if(fg[0] > src_alpha) fg[0] = src_alpha;
                        if(fg[1] > src_alpha) fg[1] = src_alpha;
                        if(fg[2] > src_alpha) fg[2] = src_alpha;
                    }
                }

                span->r = (value_type)fg[order_type::R];
                span->g = (value_type)fg[order_type::G];
                span->b = (value_type)fg[order_type::B];
                span->a = (value_type)src_alpha;

                ++span;
                ++base_type::interpolator();

            } while(--len);

            return base_type::allocator().span();
        }
    };



}


#endif



