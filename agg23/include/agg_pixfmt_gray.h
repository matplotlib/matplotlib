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

#ifndef AGG_PIXFMT_GRAY_INCLUDED
#define AGG_PIXFMT_GRAY_INCLUDED

#include <string.h>
#include "agg_basics.h"
#include "agg_color_gray.h"
#include "agg_rendering_buffer.h"

namespace agg
{
 
    //============================================================blender_gray
    template<class ColorT> struct blender_gray
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        enum { base_shift = color_type::base_shift };

        static AGG_INLINE void blend_pix(value_type* p, unsigned cv, 
                                         unsigned alpha, unsigned cover=0)
        {
            *p = (value_type)((((cv - calc_type(*p)) * alpha) + (calc_type(*p) << base_shift)) >> base_shift);
        }
    };


    //======================================================blender_gray_pre
    template<class ColorT> struct blender_gray_pre
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        enum { base_shift = color_type::base_shift };

        static AGG_INLINE void blend_pix(value_type* p, unsigned cv,
                                         unsigned alpha, unsigned cover)
        {
            alpha = color_type::base_mask - alpha;
            cover = (cover + 1) << (base_shift - 8);
            *p = (value_type)((*p * alpha + cv * cover) >> base_shift);
        }

        static AGG_INLINE void blend_pix(value_type* p, unsigned cv,
                                         unsigned alpha)
        {
            *p = (value_type)(((*p * (color_type::base_mask - alpha)) >> base_shift) + cv);
        }
    };
    


    //=====================================================apply_gamma_dir_gray
    template<class ColorT, class GammaLut> class apply_gamma_dir_gray
    {
    public:
        typedef typename ColorT::value_type value_type;

        apply_gamma_dir_gray(const GammaLut& gamma) : m_gamma(gamma) {}

        AGG_INLINE void operator () (value_type* p)
        {
            *p = m_gamma.dir(*p);
        }

    private:
        const GammaLut& m_gamma;
    };



    //=====================================================apply_gamma_inv_gray
    template<class ColorT, class GammaLut> class apply_gamma_inv_gray
    {
    public:
        typedef typename ColorT::value_type value_type;

        apply_gamma_inv_gray(const GammaLut& gamma) : m_gamma(gamma) {}

        AGG_INLINE void operator () (value_type* p)
        {
            *p = m_gamma.inv(*p);
        }

    private:
        const GammaLut& m_gamma;
    };



    //======================================================pixel_formats_gray
    template<class Blender, unsigned Step=1, unsigned Offset=0>
    class pixel_formats_gray
    {
    public:
        typedef rendering_buffer::row_data row_data;
        typedef rendering_buffer::span_data span_data;
        typedef typename Blender::color_type color_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        enum
        {
            base_shift = color_type::base_shift,
            base_size  = color_type::base_size,
            base_mask  = color_type::base_mask
        };

    private:
        //--------------------------------------------------------------------
        static AGG_INLINE void copy_or_blend_pix(value_type* p, 
                                                 const color_type& c, 
                                                 unsigned cover)
        {
            if (c.a)
            {
                calc_type alpha = (calc_type(c.a) * (cover + 1)) >> 8;
                if(alpha == base_mask)
                {
                    *p = c.v;
                }
                else
                {
                    Blender::blend_pix(p, c.v, alpha, cover);
                }
            }
        }


        static AGG_INLINE void copy_or_blend_pix(value_type* p, 
                                                 const color_type& c)
        {
            if (c.a)
            {
                if(c.a == base_mask)
                {
                    *p = c.v;
                }
                else
                {
                    Blender::blend_pix(p, c.v, c.a);
                }
            }
        }


    public:
        //--------------------------------------------------------------------
        pixel_formats_gray(rendering_buffer& rb) :
            m_rbuf(&rb)
        {}


        //--------------------------------------------------------------------
        AGG_INLINE unsigned width()  const { return m_rbuf->width();  }
        AGG_INLINE unsigned height() const { return m_rbuf->height(); }

        //--------------------------------------------------------------------
        AGG_INLINE color_type pixel(int x, int y) const
        {
            value_type* p = (value_type*)m_rbuf->row(y) + x * Step + Offset;
            return color_type(*p);
        }

        //--------------------------------------------------------------------
        row_data row(int x, int y) const
        {
            return row_data(x, 
                            width() - 1, 
                            m_rbuf->row(y) + 
                                x * Step * sizeof(value_type) + 
                                Offset * sizeof(value_type));
        }

        //--------------------------------------------------------------------
        span_data span(int x, int y, unsigned len)
        {
            return span_data(x, len,
                             m_rbuf->row(y) + 
                                x * Step * sizeof(value_type) + 
                                Offset * sizeof(value_type));
        }

        //--------------------------------------------------------------------
        AGG_INLINE void copy_pixel(int x, int y, const color_type& c)
        {
            *((value_type*)m_rbuf->row(y) + x * Step + Offset) = c.v;
        }

        //--------------------------------------------------------------------
        AGG_INLINE void blend_pixel(int x, int y, const color_type& c, int8u cover)
        {
            copy_or_blend_pix((value_type*)m_rbuf->row(y) + x * Step + Offset, c, cover);
        }


        //--------------------------------------------------------------------
        AGG_INLINE void copy_hline(int x, int y, 
                                   unsigned len, 
                                   const color_type& c)
        {
            value_type* p = (value_type*)m_rbuf->row(y) + x * Step + Offset;
            do
            {
                *p = c.v; 
                p += Step;
            }
            while(--len);
        }


        //--------------------------------------------------------------------
        AGG_INLINE void copy_vline(int x, int y,
                                   unsigned len, 
                                   const color_type& c)
        {
            value_type* p = (value_type*)m_rbuf->row(y) + x * Step + Offset;
            do
            {
                *p = c.v;
                p = (value_type*)m_rbuf->next_row(p);
            }
            while(--len);
        }


        //--------------------------------------------------------------------
        void blend_hline(int x, int y,
                         unsigned len, 
                         const color_type& c,
                         int8u cover)
        {
            if (c.a)
            {
                value_type* p = (value_type*)m_rbuf->row(y) + x * Step + Offset;
                calc_type alpha = (calc_type(c.a) * (cover + 1)) >> 8;
                if(alpha == base_mask)
                {
                    do
                    {
                        *p = c.v; 
                        p += Step;
                    }
                    while(--len);
                }
                else
                {
                    do
                    {
                        Blender::blend_pix(p, c.v, alpha, cover);
                        p += Step;
                    }
                    while(--len);
                }
            }
        }


        //--------------------------------------------------------------------
        void blend_vline(int x, int y,
                         unsigned len, 
                         const color_type& c,
                         int8u cover)
        {
            if (c.a)
            {
                value_type* p = (value_type*)m_rbuf->row(y) + x * Step + Offset;
                calc_type alpha = (calc_type(c.a) * (cover + 1)) >> 8;
                if(alpha == base_mask)
                {
                    do
                    {
                        *p = c.v; 
                        p = (value_type*)m_rbuf->next_row(p);
                    }
                    while(--len);
                }
                else
                {
                    do
                    {
                        Blender::blend_pix(p, c.v, alpha, cover);
                        p = (value_type*)m_rbuf->next_row(p);
                    }
                    while(--len);
                }
            }
        }


        //--------------------------------------------------------------------
        void blend_solid_hspan(int x, int y,
                               unsigned len, 
                               const color_type& c,
                               const int8u* covers)
        {
            if (c.a)
            {
                value_type* p = (value_type*)m_rbuf->row(y) + x * Step + Offset;
                do 
                {
                    calc_type alpha = (calc_type(c.a) * (calc_type(*covers) + 1)) >> 8;
                    if(alpha == base_mask)
                    {
                        *p = c.v;
                    }
                    else
                    {
                        Blender::blend_pix(p, c.v, alpha, *covers);
                    }
                    p += Step;
                    ++covers;
                }
                while(--len);
            }
        }


        //--------------------------------------------------------------------
        void blend_solid_vspan(int x, int y,
                               unsigned len, 
                               const color_type& c,
                               const int8u* covers)
        {
            if (c.a)
            {
                value_type* p = (value_type*)m_rbuf->row(y) + x * Step + Offset;
                do 
                {
                    calc_type alpha = (calc_type(c.a) * (calc_type(*covers) + 1)) >> 8;
                    if(alpha == base_mask)
                    {
                        *p = c.v;
                    }
                    else
                    {
                        Blender::blend_pix(p, c.v, alpha, *covers);
                    }
                    p = (value_type*)m_rbuf->next_row(p);
                    ++covers;
                }
                while(--len);
            }
        }


        //--------------------------------------------------------------------
        void copy_color_hspan(int x, int y,
                              unsigned len, 
                              const color_type* colors)
        {
            value_type* p = (value_type*)m_rbuf->row(y) + x * Step + Offset;
            do 
            {
                *p++ = colors->v;
                ++colors;
            }
            while(--len);
        }



        //--------------------------------------------------------------------
        void blend_color_hspan(int x, int y,
                               unsigned len, 
                               const color_type* colors,
                               const int8u* covers,
                               int8u cover)
        {
            value_type* p = (value_type*)m_rbuf->row(y) + x * Step + Offset;
            if(covers)
            {
                do 
                {
                    copy_or_blend_pix(p, *colors++, *covers++);
                    p += Step;
                }
                while(--len);
            }
            else
            {
                if(cover == 255)
                {
                    do 
                    {
                        if(colors->a == base_mask)
                        {
                            *p = colors->v;
                        }
                        else
                        {
                            copy_or_blend_pix(p, *colors);
                        }
                        p += Step;
                        ++colors;
                    }
                    while(--len);
                }
                else
                {
                    do 
                    {
                        copy_or_blend_pix(p, *colors++, cover);
                        p += Step;
                    }
                    while(--len);
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
            value_type* p = (value_type*)m_rbuf->row(y) + x * Step + Offset;
            if(covers)
            {
                do 
                {
                    copy_or_blend_pix(p, *colors++, *covers++);
                    p = (value_type*)m_rbuf->next_row(p);
                }
                while(--len);
            }
            else
            {
                if(cover == 255)
                {
                    do 
                    {
                        if(colors->a == base_mask)
                        {
                            *p = colors->v;
                        }
                        else
                        {
                            copy_or_blend_pix(p, *colors);
                        }
                        p = (value_type*)m_rbuf->next_row(p);
                        ++colors;
                    }
                    while(--len);
                }
                else
                {
                    do 
                    {
                        copy_or_blend_pix(p, *colors++, cover);
                        p = (value_type*)m_rbuf->next_row(p);
                    }
                    while(--len);
                }
            }
        }

        //--------------------------------------------------------------------
        template<class Function> void for_each_pixel(Function f)
        {
            unsigned y;
            for(y = 0; y < height(); ++y)
            {
                unsigned len = width();
                value_type* p = (value_type*)m_rbuf->row(y) + Offset;
                do
                {
                    f(p);
                    p += Step;
                }
                while(--len);
            }
        }

        //--------------------------------------------------------------------
        template<class GammaLut> void apply_gamma_dir(const GammaLut& g)
        {
            for_each_pixel(apply_gamma_dir_gray<color_type, GammaLut>(g));
        }

        //--------------------------------------------------------------------
        template<class GammaLut> void apply_gamma_inv(const GammaLut& g)
        {
            for_each_pixel(apply_gamma_inv_gray<color_type, GammaLut>(g));
        }

        //--------------------------------------------------------------------
        void copy_from(const rendering_buffer& from, 
                       int xdst, int ydst,
                       int xsrc, int ysrc,
                       unsigned len)
        {
            memmove((value_type*)m_rbuf->row(ydst) + xdst, 
                    (value_type*)from.row(ysrc) + xsrc, 
                    sizeof(value_type) * len);
        }

    private:
        rendering_buffer* m_rbuf;
    };

    typedef blender_gray<gray8>      blender_gray8;
    typedef blender_gray_pre<gray8>  blender_gray8_pre;
    typedef blender_gray<gray16>     blender_gray16;
    typedef blender_gray_pre<gray16> blender_gray16_pre;

    typedef pixel_formats_gray<blender_gray8, 1, 0> pixfmt_gray8;         //----pixfmt_gray8

    typedef pixel_formats_gray<blender_gray8, 3, 0> pixfmt_gray8_rgb24r;  //----pixfmt_gray8_rgb24r
    typedef pixel_formats_gray<blender_gray8, 3, 1> pixfmt_gray8_rgb24g;  //----pixfmt_gray8_rgb24g
    typedef pixel_formats_gray<blender_gray8, 3, 2> pixfmt_gray8_rgb24b;  //----pixfmt_gray8_rgb24b

    typedef pixel_formats_gray<blender_gray8, 3, 2> pixfmt_gray8_bgr24r;  //----pixfmt_gray8_bgr24r
    typedef pixel_formats_gray<blender_gray8, 3, 1> pixfmt_gray8_bgr24g;  //----pixfmt_gray8_bgr24g
    typedef pixel_formats_gray<blender_gray8, 3, 0> pixfmt_gray8_bgr24b;  //----pixfmt_gray8_bgr24b

    typedef pixel_formats_gray<blender_gray8, 4, 0> pixfmt_gray8_rgba32r; //----pixfmt_gray8_rgba32r
    typedef pixel_formats_gray<blender_gray8, 4, 1> pixfmt_gray8_rgba32g; //----pixfmt_gray8_rgba32g
    typedef pixel_formats_gray<blender_gray8, 4, 2> pixfmt_gray8_rgba32b; //----pixfmt_gray8_rgba32b
    typedef pixel_formats_gray<blender_gray8, 4, 3> pixfmt_gray8_rgba32a; //----pixfmt_gray8_rgba32a

    typedef pixel_formats_gray<blender_gray8, 4, 1> pixfmt_gray8_argb32r; //----pixfmt_gray8_argb32r
    typedef pixel_formats_gray<blender_gray8, 4, 2> pixfmt_gray8_argb32g; //----pixfmt_gray8_argb32g
    typedef pixel_formats_gray<blender_gray8, 4, 3> pixfmt_gray8_argb32b; //----pixfmt_gray8_argb32b
    typedef pixel_formats_gray<blender_gray8, 4, 0> pixfmt_gray8_argb32a; //----pixfmt_gray8_argb32a

    typedef pixel_formats_gray<blender_gray8, 4, 2> pixfmt_gray8_bgra32r; //----pixfmt_gray8_bgra32r
    typedef pixel_formats_gray<blender_gray8, 4, 1> pixfmt_gray8_bgra32g; //----pixfmt_gray8_bgra32g
    typedef pixel_formats_gray<blender_gray8, 4, 0> pixfmt_gray8_bgra32b; //----pixfmt_gray8_bgra32b
    typedef pixel_formats_gray<blender_gray8, 4, 3> pixfmt_gray8_bgra32a; //----pixfmt_gray8_bgra32a

    typedef pixel_formats_gray<blender_gray8, 4, 3> pixfmt_gray8_abgr32r; //----pixfmt_gray8_abgr32r
    typedef pixel_formats_gray<blender_gray8, 4, 2> pixfmt_gray8_abgr32g; //----pixfmt_gray8_abgr32g
    typedef pixel_formats_gray<blender_gray8, 4, 1> pixfmt_gray8_abgr32b; //----pixfmt_gray8_abgr32b
    typedef pixel_formats_gray<blender_gray8, 4, 0> pixfmt_gray8_abgr32a; //----pixfmt_gray8_abgr32a

    typedef pixel_formats_gray<blender_gray8_pre, 1, 0> pixfmt_gray8_pre;         //----pixfmt_gray8_pre

    typedef pixel_formats_gray<blender_gray8_pre, 3, 0> pixfmt_gray8_pre_rgb24r;  //----pixfmt_gray8_pre_rgb24r
    typedef pixel_formats_gray<blender_gray8_pre, 3, 1> pixfmt_gray8_pre_rgb24g;  //----pixfmt_gray8_pre_rgb24g
    typedef pixel_formats_gray<blender_gray8_pre, 3, 2> pixfmt_gray8_pre_rgb24b;  //----pixfmt_gray8_pre_rgb24b

    typedef pixel_formats_gray<blender_gray8_pre, 3, 2> pixfmt_gray8_pre_bgr24r;  //----pixfmt_gray8_pre_bgr24r
    typedef pixel_formats_gray<blender_gray8_pre, 3, 1> pixfmt_gray8_pre_bgr24g;  //----pixfmt_gray8_pre_bgr24g
    typedef pixel_formats_gray<blender_gray8_pre, 3, 0> pixfmt_gray8_pre_bgr24b;  //----pixfmt_gray8_pre_bgr24b

    typedef pixel_formats_gray<blender_gray8_pre, 4, 0> pixfmt_gray8_pre_rgba32r; //----pixfmt_gray8_pre_rgba32r
    typedef pixel_formats_gray<blender_gray8_pre, 4, 1> pixfmt_gray8_pre_rgba32g; //----pixfmt_gray8_pre_rgba32g
    typedef pixel_formats_gray<blender_gray8_pre, 4, 2> pixfmt_gray8_pre_rgba32b; //----pixfmt_gray8_pre_rgba32b
    typedef pixel_formats_gray<blender_gray8_pre, 4, 3> pixfmt_gray8_pre_rgba32a; //----pixfmt_gray8_pre_rgba32a

    typedef pixel_formats_gray<blender_gray8_pre, 4, 1> pixfmt_gray8_pre_argb32r; //----pixfmt_gray8_pre_argb32r
    typedef pixel_formats_gray<blender_gray8_pre, 4, 2> pixfmt_gray8_pre_argb32g; //----pixfmt_gray8_pre_argb32g
    typedef pixel_formats_gray<blender_gray8_pre, 4, 3> pixfmt_gray8_pre_argb32b; //----pixfmt_gray8_pre_argb32b
    typedef pixel_formats_gray<blender_gray8_pre, 4, 0> pixfmt_gray8_pre_argb32a; //----pixfmt_gray8_pre_argb32a

    typedef pixel_formats_gray<blender_gray8_pre, 4, 2> pixfmt_gray8_pre_bgra32r; //----pixfmt_gray8_pre_bgra32r
    typedef pixel_formats_gray<blender_gray8_pre, 4, 1> pixfmt_gray8_pre_bgra32g; //----pixfmt_gray8_pre_bgra32g
    typedef pixel_formats_gray<blender_gray8_pre, 4, 0> pixfmt_gray8_pre_bgra32b; //----pixfmt_gray8_pre_bgra32b
    typedef pixel_formats_gray<blender_gray8_pre, 4, 3> pixfmt_gray8_pre_bgra32a; //----pixfmt_gray8_pre_bgra32a

    typedef pixel_formats_gray<blender_gray8_pre, 4, 3> pixfmt_gray8_pre_abgr32r; //----pixfmt_gray8_pre_abgr32r
    typedef pixel_formats_gray<blender_gray8_pre, 4, 2> pixfmt_gray8_pre_abgr32g; //----pixfmt_gray8_pre_abgr32g
    typedef pixel_formats_gray<blender_gray8_pre, 4, 1> pixfmt_gray8_pre_abgr32b; //----pixfmt_gray8_pre_abgr32b
    typedef pixel_formats_gray<blender_gray8_pre, 4, 0> pixfmt_gray8_pre_abgr32a; //----pixfmt_gray8_pre_abgr32a

    typedef pixel_formats_gray<blender_gray16, 1, 0> pixfmt_gray16;         //----pixfmt_gray16

    typedef pixel_formats_gray<blender_gray16, 3, 0> pixfmt_gray16_rgb48r;  //----pixfmt_gray16_rgb48r
    typedef pixel_formats_gray<blender_gray16, 3, 1> pixfmt_gray16_rgb48g;  //----pixfmt_gray16_rgb48g
    typedef pixel_formats_gray<blender_gray16, 3, 2> pixfmt_gray16_rgb48b;  //----pixfmt_gray16_rgb48b

    typedef pixel_formats_gray<blender_gray16, 3, 2> pixfmt_gray16_bgr48r;  //----pixfmt_gray16_bgr48r
    typedef pixel_formats_gray<blender_gray16, 3, 1> pixfmt_gray16_bgr48g;  //----pixfmt_gray16_bgr48g
    typedef pixel_formats_gray<blender_gray16, 3, 0> pixfmt_gray16_bgr48b;  //----pixfmt_gray16_bgr48b

    typedef pixel_formats_gray<blender_gray16, 4, 0> pixfmt_gray16_rgba64r; //----pixfmt_gray16_rgba64r
    typedef pixel_formats_gray<blender_gray16, 4, 1> pixfmt_gray16_rgba64g; //----pixfmt_gray16_rgba64g
    typedef pixel_formats_gray<blender_gray16, 4, 2> pixfmt_gray16_rgba64b; //----pixfmt_gray16_rgba64b
    typedef pixel_formats_gray<blender_gray16, 4, 3> pixfmt_gray16_rgba64a; //----pixfmt_gray16_rgba64a

    typedef pixel_formats_gray<blender_gray16, 4, 1> pixfmt_gray16_argb64r; //----pixfmt_gray16_argb64r
    typedef pixel_formats_gray<blender_gray16, 4, 2> pixfmt_gray16_argb64g; //----pixfmt_gray16_argb64g
    typedef pixel_formats_gray<blender_gray16, 4, 3> pixfmt_gray16_argb64b; //----pixfmt_gray16_argb64b
    typedef pixel_formats_gray<blender_gray16, 4, 0> pixfmt_gray16_argb64a; //----pixfmt_gray16_argb64a

    typedef pixel_formats_gray<blender_gray16, 4, 2> pixfmt_gray16_bgra64r; //----pixfmt_gray16_bgra64r
    typedef pixel_formats_gray<blender_gray16, 4, 1> pixfmt_gray16_bgra64g; //----pixfmt_gray16_bgra64g
    typedef pixel_formats_gray<blender_gray16, 4, 0> pixfmt_gray16_bgra64b; //----pixfmt_gray16_bgra64b
    typedef pixel_formats_gray<blender_gray16, 4, 3> pixfmt_gray16_bgra64a; //----pixfmt_gray16_bgra64a

    typedef pixel_formats_gray<blender_gray16, 4, 3> pixfmt_gray16_abgr64r; //----pixfmt_gray16_abgr64r
    typedef pixel_formats_gray<blender_gray16, 4, 2> pixfmt_gray16_abgr64g; //----pixfmt_gray16_abgr64g
    typedef pixel_formats_gray<blender_gray16, 4, 1> pixfmt_gray16_abgr64b; //----pixfmt_gray16_abgr64b
    typedef pixel_formats_gray<blender_gray16, 4, 0> pixfmt_gray16_abgr64a; //----pixfmt_gray16_abgr64a

    typedef pixel_formats_gray<blender_gray16_pre, 1, 0> pixfmt_gray16_pre;         //----pixfmt_gray16_pre

    typedef pixel_formats_gray<blender_gray16_pre, 3, 0> pixfmt_gray16_pre_rgb48r;  //----pixfmt_gray16_pre_rgb48r
    typedef pixel_formats_gray<blender_gray16_pre, 3, 1> pixfmt_gray16_pre_rgb48g;  //----pixfmt_gray16_pre_rgb48g
    typedef pixel_formats_gray<blender_gray16_pre, 3, 2> pixfmt_gray16_pre_rgb48b;  //----pixfmt_gray16_pre_rgb48b

    typedef pixel_formats_gray<blender_gray16_pre, 3, 2> pixfmt_gray16_pre_bgr48r;  //----pixfmt_gray16_pre_bgr48r
    typedef pixel_formats_gray<blender_gray16_pre, 3, 1> pixfmt_gray16_pre_bgr48g;  //----pixfmt_gray16_pre_bgr48g
    typedef pixel_formats_gray<blender_gray16_pre, 3, 0> pixfmt_gray16_pre_bgr48b;  //----pixfmt_gray16_pre_bgr48b

    typedef pixel_formats_gray<blender_gray16_pre, 4, 0> pixfmt_gray16_pre_rgba64r; //----pixfmt_gray16_pre_rgba64r
    typedef pixel_formats_gray<blender_gray16_pre, 4, 1> pixfmt_gray16_pre_rgba64g; //----pixfmt_gray16_pre_rgba64g
    typedef pixel_formats_gray<blender_gray16_pre, 4, 2> pixfmt_gray16_pre_rgba64b; //----pixfmt_gray16_pre_rgba64b
    typedef pixel_formats_gray<blender_gray16_pre, 4, 3> pixfmt_gray16_pre_rgba64a; //----pixfmt_gray16_pre_rgba64a

    typedef pixel_formats_gray<blender_gray16_pre, 4, 1> pixfmt_gray16_pre_argb64r; //----pixfmt_gray16_pre_argb64r
    typedef pixel_formats_gray<blender_gray16_pre, 4, 2> pixfmt_gray16_pre_argb64g; //----pixfmt_gray16_pre_argb64g
    typedef pixel_formats_gray<blender_gray16_pre, 4, 3> pixfmt_gray16_pre_argb64b; //----pixfmt_gray16_pre_argb64b
    typedef pixel_formats_gray<blender_gray16_pre, 4, 0> pixfmt_gray16_pre_argb64a; //----pixfmt_gray16_pre_argb64a

    typedef pixel_formats_gray<blender_gray16_pre, 4, 2> pixfmt_gray16_pre_bgra64r; //----pixfmt_gray16_pre_bgra64r
    typedef pixel_formats_gray<blender_gray16_pre, 4, 1> pixfmt_gray16_pre_bgra64g; //----pixfmt_gray16_pre_bgra64g
    typedef pixel_formats_gray<blender_gray16_pre, 4, 0> pixfmt_gray16_pre_bgra64b; //----pixfmt_gray16_pre_bgra64b
    typedef pixel_formats_gray<blender_gray16_pre, 4, 3> pixfmt_gray16_pre_bgra64a; //----pixfmt_gray16_pre_bgra64a

    typedef pixel_formats_gray<blender_gray16_pre, 4, 3> pixfmt_gray16_pre_abgr64r; //----pixfmt_gray16_pre_abgr64r
    typedef pixel_formats_gray<blender_gray16_pre, 4, 2> pixfmt_gray16_pre_abgr64g; //----pixfmt_gray16_pre_abgr64g
    typedef pixel_formats_gray<blender_gray16_pre, 4, 1> pixfmt_gray16_pre_abgr64b; //----pixfmt_gray16_pre_abgr64b
    typedef pixel_formats_gray<blender_gray16_pre, 4, 0> pixfmt_gray16_pre_abgr64a; //----pixfmt_gray16_pre_abgr64a

}

#endif

