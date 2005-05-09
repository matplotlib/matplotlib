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

#ifndef AGG_PIXFMT_RGB_INCLUDED
#define AGG_PIXFMT_RGB_INCLUDED

#include <string.h>
#include "agg_basics.h"
#include "agg_color_rgba.h"
#include "agg_rendering_buffer.h"

namespace agg
{

    //=====================================================apply_gamma_dir_rgb
    template<class ColorT, class Order, class GammaLut> class apply_gamma_dir_rgb
    {
    public:
        typedef typename ColorT::value_type value_type;

        apply_gamma_dir_rgb(const GammaLut& gamma) : m_gamma(gamma) {}

        AGG_INLINE void operator () (value_type* p)
        {
            p[Order::R] = m_gamma.dir(p[Order::R]);
            p[Order::G] = m_gamma.dir(p[Order::G]);
            p[Order::B] = m_gamma.dir(p[Order::B]);
        }

    private:
        const GammaLut& m_gamma;
    };



    //=====================================================apply_gamma_inv_rgb
    template<class ColorT, class Order, class GammaLut> class apply_gamma_inv_rgb
    {
    public:
        typedef typename ColorT::value_type value_type;

        apply_gamma_inv_rgb(const GammaLut& gamma) : m_gamma(gamma) {}

        AGG_INLINE void operator () (value_type* p)
        {
            p[Order::R] = m_gamma.inv(p[Order::R]);
            p[Order::G] = m_gamma.inv(p[Order::G]);
            p[Order::B] = m_gamma.inv(p[Order::B]);
        }

    private:
        const GammaLut& m_gamma;
    };


    //=========================================================blender_rgb
    template<class ColorT, class Order> struct blender_rgb
    {
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        enum { base_shift = color_type::base_shift };

        //--------------------------------------------------------------------
        static AGG_INLINE void blend_pix(value_type* p, 
                                         unsigned cr, unsigned cg, unsigned cb, 
                                         unsigned alpha, 
                                         unsigned cover=0)
        {
            p[Order::R] += (value_type)(((cr - p[Order::R]) * alpha) >> base_shift);
            p[Order::G] += (value_type)(((cg - p[Order::G]) * alpha) >> base_shift);
            p[Order::B] += (value_type)(((cb - p[Order::B]) * alpha) >> base_shift);
        }
    };


    //======================================================blender_rgb_pre
    template<class ColorT, class Order> struct blender_rgb_pre
    {
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        enum { base_shift = color_type::base_shift };

        //--------------------------------------------------------------------
        static AGG_INLINE void blend_pix(value_type* p, 
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha,
                                         unsigned cover)
        {
            alpha = color_type::base_mask - alpha;
            cover = (cover + 1) << (base_shift - 8);
            p[Order::R] = (value_type)((p[Order::R] * alpha + cr * cover) >> base_shift);
            p[Order::G] = (value_type)((p[Order::G] * alpha + cg * cover) >> base_shift);
            p[Order::B] = (value_type)((p[Order::B] * alpha + cb * cover) >> base_shift);
        }

        //--------------------------------------------------------------------
        static AGG_INLINE void blend_pix(value_type* p, 
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha)
        {
            alpha = color_type::base_mask - alpha;
            p[Order::R] = (value_type)(((p[Order::R] * alpha) >> base_shift) + cr);
            p[Order::G] = (value_type)(((p[Order::G] * alpha) >> base_shift) + cg);
            p[Order::B] = (value_type)(((p[Order::B] * alpha) >> base_shift) + cb);
        }

    };



    //===================================================blender_rgb_gamma
    template<class ColorT, class Order, class Gamma> class blender_rgb_gamma
    {
    public:
        typedef ColorT color_type;
        typedef Order order_type;
        typedef Gamma gamma_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        enum { base_shift = color_type::base_shift };

        //--------------------------------------------------------------------
        blender_rgb_gamma() : m_gamma(0) {}
        void gamma(const gamma_type& g) { m_gamma = &g; }

        //--------------------------------------------------------------------
        AGG_INLINE void blend_pix(value_type* p, 
                                  unsigned cr, unsigned cg, unsigned cb,
                                  unsigned alpha, 
                                  unsigned cover=0)
        {
            calc_type r = m_gamma->dir(p[Order::R]);
            calc_type g = m_gamma->dir(p[Order::G]);
            calc_type b = m_gamma->dir(p[Order::B]);
            p[Order::R] = m_gamma->inv((((m_gamma->dir(cr) - r) * alpha) >> base_shift) + r);
            p[Order::G] = m_gamma->inv((((m_gamma->dir(cg) - g) * alpha) >> base_shift) + g);
            p[Order::B] = m_gamma->inv((((m_gamma->dir(cb) - b) * alpha) >> base_shift) + b);
        }

    private:
        const gamma_type* m_gamma;
    };



    
    //==================================================pixel_formats_rgb
    template<class Blender> class pixel_formats_rgb
    {
    public:
        typedef rendering_buffer::row_data row_data;
        typedef rendering_buffer::span_data span_data;
        typedef typename Blender::color_type color_type;
        typedef typename Blender::order_type order_type;
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
        AGG_INLINE void copy_or_blend_pix(value_type* p, 
                                          const color_type& c, 
                                          unsigned cover)
        {
            if (c.a)
            {
                calc_type alpha = (calc_type(c.a) * (cover + 1)) >> 8;
                if(alpha == base_mask)
                {
                    p[order_type::R] = c.r;
                    p[order_type::G] = c.g;
                    p[order_type::B] = c.b;
                }
                else
                {
                    m_blender.blend_pix(p, c.r, c.g, c.b, alpha, cover);
                }
            }
        }

        //--------------------------------------------------------------------
        AGG_INLINE void copy_or_blend_pix(value_type* p, 
                                          const color_type& c)
        {
            if (c.a)
            {
                if(c.a == base_mask)
                {
                    p[order_type::R] = c.r;
                    p[order_type::G] = c.g;
                    p[order_type::B] = c.b;
                }
                else
                {
                    m_blender.blend_pix(p, c.r, c.g, c.b, c.a);
                }
            }
        }


    public:
        //--------------------------------------------------------------------
        pixel_formats_rgb(rendering_buffer& rb) :
            m_rbuf(&rb)
        {}

        //--------------------------------------------------------------------
        Blender& blender() { return m_blender; }

        //--------------------------------------------------------------------
        AGG_INLINE unsigned width()  const { return m_rbuf->width();  }
        AGG_INLINE unsigned height() const { return m_rbuf->height(); }

        //--------------------------------------------------------------------
        AGG_INLINE color_type pixel(int x, int y) const
        {
            value_type* p = (value_type*)m_rbuf->row(y) + x + x + x;
            return color_type(p[order_type::R], 
                              p[order_type::G], 
                              p[order_type::B]);
        }

        //--------------------------------------------------------------------
        row_data row(int x, int y) const
        {
            return row_data(x, 
                            width() - 1, 
                            m_rbuf->row(y) + x * 3 * sizeof(value_type));
        }

        //--------------------------------------------------------------------
        span_data span(int x, int y, unsigned len)
        {
            return span_data(x, len, 
                             m_rbuf->row(y) + x * 3 * sizeof(value_type));
        }

        //--------------------------------------------------------------------
        AGG_INLINE void copy_pixel(int x, int y, const color_type& c)
        {
            value_type* p = (value_type*)m_rbuf->row(y) + x + x + x;
            p[order_type::R] = c.r;
            p[order_type::G] = c.g;
            p[order_type::B] = c.b;
        }

        //--------------------------------------------------------------------
        AGG_INLINE void blend_pixel(int x, int y, const color_type& c, int8u cover)
        {
            copy_or_blend_pix((value_type*)m_rbuf->row(y) + x + x + x, c, cover);
        }


        //--------------------------------------------------------------------
        AGG_INLINE void copy_hline(int x, int y, 
                                   unsigned len, 
                                   const color_type& c)
        {
            value_type* p = (value_type*)m_rbuf->row(y) + x + x + x;
            do
            {
                p[order_type::R] = c.r; 
                p[order_type::G] = c.g; 
                p[order_type::B] = c.b;
                p += 3;
            }
            while(--len);
        }


        //--------------------------------------------------------------------
        AGG_INLINE void copy_vline(int x, int y,
                                   unsigned len, 
                                   const color_type& c)
        {
            value_type* p = (value_type*)m_rbuf->row(y) + x + x + x;
            do
            {
                p[order_type::R] = c.r; 
                p[order_type::G] = c.g; 
                p[order_type::B] = c.b;
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
                value_type* p = (value_type*)m_rbuf->row(y) + x + x + x;
                calc_type alpha = (calc_type(c.a) * (calc_type(cover) + 1)) >> 8;
                if(alpha == base_mask)
                {
                    do
                    {
                        p[order_type::R] = c.r; 
                        p[order_type::G] = c.g; 
                        p[order_type::B] = c.b;
                        p += 3;
                    }
                    while(--len);
                }
                else
                {
                    do
                    {
                        m_blender.blend_pix(p, c.r, c.g, c.b, alpha, cover);
                        p += 3;
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
                value_type* p = (value_type*)m_rbuf->row(y) + x + x + x;
                calc_type alpha = (calc_type(c.a) * (cover + 1)) >> 8;
                if(alpha == base_mask)
                {
                    do
                    {
                        p[order_type::R] = c.r; 
                        p[order_type::G] = c.g; 
                        p[order_type::B] = c.b;
                        p = (value_type*)m_rbuf->next_row(p);
                    }
                    while(--len);
                }
                else
                {
                    do
                    {
                        m_blender.blend_pix(p, c.r, c.g, c.b, alpha, cover);
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
                value_type* p = (value_type*)m_rbuf->row(y) + x + x + x;
                do 
                {
                    calc_type alpha = (calc_type(c.a) * (calc_type(*covers) + 1)) >> 8;
                    if(alpha == base_mask)
                    {
                        p[order_type::R] = c.r;
                        p[order_type::G] = c.g;
                        p[order_type::B] = c.b;
                    }
                    else
                    {
                        m_blender.blend_pix(p, c.r, c.g, c.b, alpha, *covers);
                    }
                    p += 3;
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
                value_type* p = (value_type*)m_rbuf->row(y) + x + x + x;
                do 
                {
                    calc_type alpha = (calc_type(c.a) * (calc_type(*covers) + 1)) >> 8;
                    if(alpha == base_mask)
                    {
                        p[order_type::R] = c.r;
                        p[order_type::G] = c.g;
                        p[order_type::B] = c.b;
                    }
                    else
                    {
                        m_blender.blend_pix(p, c.r, c.g, c.b, alpha, *covers);
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
            value_type* p = (value_type*)m_rbuf->row(y) + x + x + x;
            do 
            {
                p[order_type::R] = colors->r;
                p[order_type::G] = colors->g;
                p[order_type::B] = colors->b;
                ++colors;
                p += 3;
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
            value_type* p = (value_type*)m_rbuf->row(y) + x + x + x;
            if(covers)
            {
                do 
                {
                    copy_or_blend_pix(p, *colors++, *covers++);
                    p += 3;
                }
                while(--len);
            }
            else
            {
                if(cover == 255)
                {
                    do 
                    {
                        copy_or_blend_pix(p, *colors++);
                        p += 3;
                    }
                    while(--len);
                }
                else
                {
                    do 
                    {
                        copy_or_blend_pix(p, *colors++, cover);
                        p += 3;
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
            value_type* p = (value_type*)m_rbuf->row(y) + x + x + x;
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
                        copy_or_blend_pix(p, *colors++);
                        p = (value_type*)m_rbuf->next_row(p);
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
                value_type* p = (value_type*)m_rbuf->row(y);
                do
                {
                    f(p);
                    p += 3;
                }
                while(--len);
            }
        }

        //--------------------------------------------------------------------
        template<class GammaLut> void apply_gamma_dir(const GammaLut& g)
        {
            for_each_pixel(apply_gamma_dir_rgb<color_type, order_type, GammaLut>(g));
        }

        //--------------------------------------------------------------------
        template<class GammaLut> void apply_gamma_inv(const GammaLut& g)
        {
            for_each_pixel(apply_gamma_inv_rgb<color_type, order_type, GammaLut>(g));
        }

        //--------------------------------------------------------------------
        void copy_from(const rendering_buffer& from, 
                       int xdst, int ydst,
                       int xsrc, int ysrc,
                       unsigned len)
        {
            memmove((value_type*)m_rbuf->row(ydst) + xdst * 3, 
                    (const value_type*)from.row(ysrc) + xsrc * 3, 
                    sizeof(value_type) * 3 * len);
        }


        //--------------------------------------------------------------------
        template<class SrcPixelFormatRenderer>
        void blend_from(const SrcPixelFormatRenderer& from, 
                        const int8u* psrc_,
                        int xdst, int ydst,
                        int xsrc, int ysrc,
                        unsigned len,
                        int8u cover)
        {
            typedef typename SrcPixelFormatRenderer::order_type src_order;

            const value_type* psrc = (const value_type*)psrc_;
            value_type* pdst = (value_type*)m_rbuf->row(ydst) + xdst * 3;


            if(cover == 255)
            {
                do 
                {
                    value_type alpha = psrc[src_order::A];
                    if(alpha)
                    {
                        if(alpha == base_mask)
                        {
                            pdst[order_type::R] = psrc[src_order::R];
                            pdst[order_type::G] = psrc[src_order::G];
                            pdst[order_type::B] = psrc[src_order::B];
                        }
                        else
                        {
                            m_blender.blend_pix(pdst, 
                                                psrc[src_order::R],
                                                psrc[src_order::G],
                                                psrc[src_order::B],
                                                alpha);
                        }
                    }
                    psrc += 4;
                    pdst += 3;
                }
                while(--len);
            }
            else
            {
                color_type color;
                do 
                {
                    color.r = psrc[src_order::R];
                    color.g = psrc[src_order::G];
                    color.b = psrc[src_order::B];
                    color.a = psrc[src_order::A];
                    copy_or_blend_pix(pdst, color, cover);
                    psrc += 4;
                    pdst += 3;
                }
                while(--len);
            }
        }

    private:
        rendering_buffer* m_rbuf;
        Blender           m_blender;
    };

    typedef pixel_formats_rgb<blender_rgb<rgba8,  order_rgb> > pixfmt_rgb24;    //----pixfmt_rgb24
    typedef pixel_formats_rgb<blender_rgb<rgba8,  order_bgr> > pixfmt_bgr24;    //----pixfmt_bgr24
    typedef pixel_formats_rgb<blender_rgb<rgba16, order_rgb> > pixfmt_rgb48;    //----pixfmt_rgb48
    typedef pixel_formats_rgb<blender_rgb<rgba16, order_bgr> > pixfmt_bgr48;    //----pixfmt_bgr48

    typedef pixel_formats_rgb<blender_rgb_pre<rgba8,  order_rgb> > pixfmt_rgb24_pre; //----pixfmt_rgb24_pre
    typedef pixel_formats_rgb<blender_rgb_pre<rgba8,  order_bgr> > pixfmt_bgr24_pre; //----pixfmt_bgr24_pre
    typedef pixel_formats_rgb<blender_rgb_pre<rgba16, order_rgb> > pixfmt_rgb48_pre; //----pixfmt_rgb48_pre
    typedef pixel_formats_rgb<blender_rgb_pre<rgba16, order_bgr> > pixfmt_bgr48_pre; //----pixfmt_bgr48_pre

    //-----------------------------------------------------pixfmt_rgb24_gamma
    template<class Gamma> class pixfmt_rgb24_gamma : 
    public pixel_formats_rgb<blender_rgb_gamma<rgba8, order_rgb, Gamma> >
    {
    public:
        pixfmt_rgb24_gamma(rendering_buffer& rb, const Gamma& g) :
            pixel_formats_rgb<blender_rgb_gamma<rgba8, order_rgb, Gamma> >(rb) 
        {
            this->blender().gamma(g);
        }
    };
        
    //-----------------------------------------------------pixfmt_bgr24_gamma
    template<class Gamma> class pixfmt_bgr24_gamma : 
    public pixel_formats_rgb<blender_rgb_gamma<rgba8, order_bgr, Gamma> >
    {
    public:
        pixfmt_bgr24_gamma(rendering_buffer& rb, const Gamma& g) :
            pixel_formats_rgb<blender_rgb_gamma<rgba8, order_bgr, Gamma> >(rb) 
        {
            this->blender().gamma(g);
        }
    };

    //-----------------------------------------------------pixfmt_rgb48_gamma
    template<class Gamma> class pixfmt_rgb48_gamma : 
    public pixel_formats_rgb<blender_rgb_gamma<rgba16, order_rgb, Gamma> >
    {
    public:
        pixfmt_rgb48_gamma(rendering_buffer& rb, const Gamma& g) :
            pixel_formats_rgb<blender_rgb_gamma<rgba16, order_rgb, Gamma> >(rb) 
        {
            this->blender().gamma(g);
        }
    };
        
    //-----------------------------------------------------pixfmt_bgr48_gamma
    template<class Gamma> class pixfmt_bgr48_gamma : 
    public pixel_formats_rgb<blender_rgb_gamma<rgba16, order_bgr, Gamma> >
    {
    public:
        pixfmt_bgr48_gamma(rendering_buffer& rb, const Gamma& g) :
            pixel_formats_rgb<blender_rgb_gamma<rgba16, order_bgr, Gamma> >(rb) 
        {
            this->blender().gamma(g);
        }
    };


}

#endif

