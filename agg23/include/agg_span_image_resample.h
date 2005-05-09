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
#ifndef AGG_SPAN_IMAGE_RESAMPLE_INCLUDED
#define AGG_SPAN_IMAGE_RESAMPLE_INCLUDED

#include "agg_span_image_filter.h"
#include "agg_span_interpolator_linear.h"


namespace agg
{


    //=====================================================span_image_resample
    template<class ColorT, class Interpolator, class Allocator> 
    class span_image_resample : 
    public span_image_filter<ColorT, Interpolator, Allocator>
    {
    public:
        typedef ColorT color_type;
        typedef Interpolator interpolator_type;
        typedef Allocator alloc_type;
        typedef span_image_filter<color_type, interpolator_type, alloc_type> base_type;

        //--------------------------------------------------------------------
        span_image_resample(alloc_type& alloc) : 
            base_type(alloc),
            m_scale_limit(20),
            m_blur_x(image_subpixel_size),
            m_blur_y(image_subpixel_size)
        {}

        //--------------------------------------------------------------------
        span_image_resample(alloc_type& alloc,
                            const rendering_buffer& src, 
                            const color_type& back_color,
                            interpolator_type& inter,
                            const image_filter_lut& filter) :
            base_type(alloc, src, back_color, inter, &filter),
            m_scale_limit(20),
            m_blur_x(image_subpixel_size),
            m_blur_y(image_subpixel_size)
        {}


        //--------------------------------------------------------------------
        int  scale_limit() const { return m_scale_limit; }
        void scale_limit(int v)  { m_scale_limit = v; }

        //--------------------------------------------------------------------
        double blur_x() const { return double(m_blur_x) / double(image_subpixel_size); }
        double blur_y() const { return double(m_blur_y) / double(image_subpixel_size); }
        void blur_x(double v) { m_blur_x = int(v * double(image_subpixel_size) + 0.5); }
        void blur_y(double v) { m_blur_y = int(v * double(image_subpixel_size) + 0.5); }
        void blur(double v)   { m_blur_x = 
                                m_blur_y = int(v * double(image_subpixel_size) + 0.5); }

    protected:
        int m_scale_limit;
        int m_blur_x;
        int m_blur_y;
    };








    //==============================================span_image_resample_affine
    template<class ColorT, class Allocator> 
    class span_image_resample_affine : 
    public span_image_filter<ColorT, span_interpolator_linear<trans_affine>, Allocator>
    {
    public:
        typedef ColorT color_type;
        typedef span_interpolator_linear<trans_affine> interpolator_type;
        typedef Allocator alloc_type;
        typedef span_image_filter<color_type, interpolator_type, alloc_type> base_type;

        //--------------------------------------------------------------------
        span_image_resample_affine(alloc_type& alloc) : 
            base_type(alloc),
            m_scale_limit(200.0),
            m_blur_x(1.0),
            m_blur_y(1.0)
        {}

        //--------------------------------------------------------------------
        span_image_resample_affine(alloc_type& alloc,
                                   const rendering_buffer& src, 
                                   const color_type& back_color,
                                   interpolator_type& inter,
                                   const image_filter_lut& filter) :
            base_type(alloc, src, back_color, inter, &filter),
            m_scale_limit(200.0),
            m_blur_x(1.0),
            m_blur_y(1.0)
        {}


        //--------------------------------------------------------------------
        int  scale_limit() const { return int(m_scale_limit); }
        void scale_limit(int v)  { m_scale_limit = v; }

        //--------------------------------------------------------------------
        double blur_x() const { return m_blur_x; }
        double blur_y() const { return m_blur_y; }
        void blur_x(double v) { m_blur_x = v; }
        void blur_y(double v) { m_blur_y = v; }
        void blur(double v) { m_blur_x = m_blur_y = v; }


        //--------------------------------------------------------------------
        void prepare(unsigned max_span_len) 
        {
            base_type::prepare(max_span_len);

            double scale_x;
            double scale_y;

            base_type::interpolator().transformer().scaling_abs(&scale_x, &scale_y);

            m_rx     = image_subpixel_size;
            m_ry     = image_subpixel_size;
            m_rx_inv = image_subpixel_size;
            m_ry_inv = image_subpixel_size;

            scale_x *= m_blur_x;
            scale_y *= m_blur_y;

            if(scale_x * scale_y > m_scale_limit)
            {
                scale_x = scale_x * m_scale_limit / (scale_x * scale_y);
                scale_y = scale_y * m_scale_limit / (scale_x * scale_y);
            }

            if(scale_x > 1.0001)
            {
                if(scale_x > m_scale_limit) scale_x = m_scale_limit;
                m_rx     = int(    scale_x * double(image_subpixel_size) + 0.5);
                m_rx_inv = int(1.0/scale_x * double(image_subpixel_size) + 0.5);
            }

            if(scale_y > 1.0001)
            {
                if(scale_y > m_scale_limit) scale_y = m_scale_limit;
                m_ry     = int(    scale_y * double(image_subpixel_size) + 0.5);
                m_ry_inv = int(1.0/scale_y * double(image_subpixel_size) + 0.5);
            }
        }

    protected:
        int m_rx;
        int m_ry;
        int m_rx_inv;
        int m_ry_inv;

    private:
        double m_scale_limit;
        double m_blur_x;
        double m_blur_y;
    };

}

#endif
