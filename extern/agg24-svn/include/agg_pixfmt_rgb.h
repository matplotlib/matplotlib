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
//----------------------------------------------------------------------------

#ifndef AGG_PIXFMT_RGB_INCLUDED
#define AGG_PIXFMT_RGB_INCLUDED

#include <string.h>
#include "agg_pixfmt_base.h"
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
    template<class ColorT, class Order> 
    struct blender_rgb
    {
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;

        // Blend pixels using the non-premultiplied form of Alvy-Ray Smith's
        // compositing function. Since the render buffer is opaque we skip the
        // initial premultiply and final demultiply.

        //--------------------------------------------------------------------
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cr, value_type cg, value_type cb, value_type alpha, cover_type cover)
        {
            blend_pix(p, cr, cg, cb, color_type::mult_cover(alpha, cover));
        }
        
        //--------------------------------------------------------------------
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cr, value_type cg, value_type cb, value_type alpha)
        {
            p[Order::R] = color_type::lerp(p[Order::R], cr, alpha);
            p[Order::G] = color_type::lerp(p[Order::G], cg, alpha);
            p[Order::B] = color_type::lerp(p[Order::B], cb, alpha);
        }
    };

    //======================================================blender_rgb_pre
    template<class ColorT, class Order> 
    struct blender_rgb_pre
    {
        typedef ColorT color_type;
        typedef Order order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;

        // Blend pixels using the premultiplied form of Alvy-Ray Smith's
        // compositing function. 

        //--------------------------------------------------------------------
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cr, value_type cg, value_type cb, value_type alpha, cover_type cover)
        {
            blend_pix(p, 
                color_type::mult_cover(cr, cover), 
                color_type::mult_cover(cg, cover), 
                color_type::mult_cover(cb, cover), 
                color_type::mult_cover(alpha, cover));
        }

        //--------------------------------------------------------------------
        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cr, value_type cg, value_type cb, value_type alpha)
        {
            p[Order::R] = color_type::prelerp(p[Order::R], cr, alpha);
            p[Order::G] = color_type::prelerp(p[Order::G], cg, alpha);
            p[Order::B] = color_type::prelerp(p[Order::B], cb, alpha);
        }
    };

    //===================================================blender_rgb_gamma
    template<class ColorT, class Order, class Gamma> 
    class blender_rgb_gamma : public blender_base<ColorT, Order>
    {
    public:
        typedef ColorT color_type;
        typedef Order order_type;
        typedef Gamma gamma_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;

        //--------------------------------------------------------------------
        blender_rgb_gamma() : m_gamma(0) {}
        void gamma(const gamma_type& g) { m_gamma = &g; }

        //--------------------------------------------------------------------
        AGG_INLINE void blend_pix(value_type* p, 
            value_type cr, value_type cg, value_type cb, value_type alpha, cover_type cover)
        {
            blend_pix(p, cr, cg, cb, color_type::mult_cover(alpha, cover));
        }
        
        //--------------------------------------------------------------------
        AGG_INLINE void blend_pix(value_type* p, 
            value_type cr, value_type cg, value_type cb, value_type alpha)
        {
            calc_type r = m_gamma->dir(p[Order::R]);
            calc_type g = m_gamma->dir(p[Order::G]);
            calc_type b = m_gamma->dir(p[Order::B]);
            p[Order::R] = m_gamma->inv(color_type::downscale((m_gamma->dir(cr) - r) * alpha) + r);
            p[Order::G] = m_gamma->inv(color_type::downscale((m_gamma->dir(cg) - g) * alpha) + g);
            p[Order::B] = m_gamma->inv(color_type::downscale((m_gamma->dir(cb) - b) * alpha) + b);
        }
        
    private:
        const gamma_type* m_gamma;
    };


    //==================================================pixfmt_alpha_blend_rgb
    template<class Blender, class RenBuf, unsigned Step, unsigned Offset = 0> 
    class pixfmt_alpha_blend_rgb
    {
    public:
        typedef pixfmt_rgb_tag pixfmt_category;
        typedef RenBuf   rbuf_type;
        typedef Blender  blender_type;
        typedef typename rbuf_type::row_data row_data;
        typedef typename blender_type::color_type color_type;
        typedef typename blender_type::order_type order_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        enum 
        {
            num_components = 3,
            pix_step = Step,
            pix_offset = Offset,
            pix_width = sizeof(value_type) * pix_step
        };
        struct pixel_type
        {
            value_type c[num_components];

            void set(value_type r, value_type g, value_type b)
            {
                c[order_type::R] = r;
                c[order_type::G] = g;
                c[order_type::B] = b;
            }

            void set(const color_type& color)
            {
                set(color.r, color.g, color.b);
            }

            void get(value_type& r, value_type& g, value_type& b) const
            {
                r = c[order_type::R];
                g = c[order_type::G];
                b = c[order_type::B];
            }

            color_type get() const
            {
                return color_type(
                    c[order_type::R], 
                    c[order_type::G], 
                    c[order_type::B]);
            }

            pixel_type* next()
            {
                return (pixel_type*)(c + pix_step);
            }

            const pixel_type* next() const
            {
                return (const pixel_type*)(c + pix_step);
            }

            pixel_type* advance(int n)
            {
                return (pixel_type*)(c + n * pix_step);
            }

            const pixel_type* advance(int n) const
            {
                return (const pixel_type*)(c + n * pix_step);
            }
        };

    private:
        //--------------------------------------------------------------------
        AGG_INLINE void blend_pix(pixel_type* p, 
            value_type r, value_type g, value_type b, value_type a, 
            unsigned cover)
        {
            m_blender.blend_pix(p->c, r, g, b, a, cover);
        }

        //--------------------------------------------------------------------
        AGG_INLINE void blend_pix(pixel_type* p, 
            value_type r, value_type g, value_type b, value_type a)
        {
            m_blender.blend_pix(p->c, r, g, b, a);
        }

        //--------------------------------------------------------------------
        AGG_INLINE void blend_pix(pixel_type* p, const color_type& c, unsigned cover)
        {
            m_blender.blend_pix(p->c, c.r, c.g, c.b, c.a, cover);
        }

        //--------------------------------------------------------------------
        AGG_INLINE void blend_pix(pixel_type* p, const color_type& c)
        {
            m_blender.blend_pix(p->c, c.r, c.g, c.b, c.a);
        }

        //--------------------------------------------------------------------
        AGG_INLINE void copy_or_blend_pix(pixel_type* p, const color_type& c, unsigned cover)
        {
            if (!c.is_transparent())
            {
                if (c.is_opaque() && cover == cover_mask)
                {
                    p->set(c);
                }
                else
                {
                    blend_pix(p, c, cover);
                }
            }
        }

        //--------------------------------------------------------------------
        AGG_INLINE void copy_or_blend_pix(pixel_type* p, const color_type& c)
        {
            if (!c.is_transparent())
            {
                if (c.is_opaque())
                {
                    p->set(c);
                }
                else
                {
                    blend_pix(p, c);
                }
            }
        }

    public:
        //--------------------------------------------------------------------
        explicit pixfmt_alpha_blend_rgb(rbuf_type& rb) :
            m_rbuf(&rb)
        {}
        void attach(rbuf_type& rb) { m_rbuf = &rb; }

        //--------------------------------------------------------------------
        template<class PixFmt>
        bool attach(PixFmt& pixf, int x1, int y1, int x2, int y2)
        {
            rect_i r(x1, y1, x2, y2);
            if (r.clip(rect_i(0, 0, pixf.width()-1, pixf.height()-1)))
            {
                int stride = pixf.stride();
                m_rbuf->attach(pixf.pix_ptr(r.x1, stride < 0 ? r.y2 : r.y1), 
                               (r.x2 - r.x1) + 1,
                               (r.y2 - r.y1) + 1,
                               stride);
                return true;
            }
            return false;
        }

        //--------------------------------------------------------------------
        Blender& blender() { return m_blender; }

        //--------------------------------------------------------------------
        AGG_INLINE unsigned width()  const { return m_rbuf->width();  }
        AGG_INLINE unsigned height() const { return m_rbuf->height(); }
        AGG_INLINE int      stride() const { return m_rbuf->stride(); }

        //--------------------------------------------------------------------
        AGG_INLINE       int8u* row_ptr(int y)       { return m_rbuf->row_ptr(y); }
        AGG_INLINE const int8u* row_ptr(int y) const { return m_rbuf->row_ptr(y); }
        AGG_INLINE row_data     row(int y)     const { return m_rbuf->row(y); }

        //--------------------------------------------------------------------
        AGG_INLINE int8u* pix_ptr(int x, int y) 
        { 
            return m_rbuf->row_ptr(y) + sizeof(value_type) * (x * pix_step + pix_offset);
        }

        AGG_INLINE const int8u* pix_ptr(int x, int y) const 
        { 
            return m_rbuf->row_ptr(y) + sizeof(value_type) * (x * pix_step + pix_offset);
        }

        // Return pointer to pixel value, forcing row to be allocated.
        AGG_INLINE pixel_type* pix_value_ptr(int x, int y, unsigned len) 
        {
            return (pixel_type*)(m_rbuf->row_ptr(x, y, len) + sizeof(value_type) * (x * pix_step + pix_offset));
        }

        // Return pointer to pixel value, or null if row not allocated.
        AGG_INLINE const pixel_type* pix_value_ptr(int x, int y) const 
        {
            int8u* p = m_rbuf->row_ptr(y);
            return p ? (pixel_type*)(p + sizeof(value_type) * (x * pix_step + pix_offset)) : 0;
        }

        // Get pixel pointer from raw buffer pointer.
        AGG_INLINE static pixel_type* pix_value_ptr(void* p) 
        {
            return (pixel_type*)((value_type*)p + pix_offset);
        }

        // Get pixel pointer from raw buffer pointer.
        AGG_INLINE static const pixel_type* pix_value_ptr(const void* p) 
        {
            return (const pixel_type*)((const value_type*)p + pix_offset);
        }

        //--------------------------------------------------------------------
        AGG_INLINE static void write_plain_color(void* p, color_type c)
        {
            // RGB formats are implicitly premultiplied.
            c.premultiply();
            pix_value_ptr(p)->set(c);
        }

        //--------------------------------------------------------------------
        AGG_INLINE static color_type read_plain_color(const void* p)
        {
            return pix_value_ptr(p)->get();
        }

        //--------------------------------------------------------------------
        AGG_INLINE static void make_pix(int8u* p, const color_type& c)
        {
            ((pixel_type*)p)->set(c);
        }

        //--------------------------------------------------------------------
        AGG_INLINE color_type pixel(int x, int y) const
        {
            if (const pixel_type* p = pix_value_ptr(x, y))
            {
                return p->get();
            }
            return color_type::no_color();
        }

        //--------------------------------------------------------------------
        AGG_INLINE void copy_pixel(int x, int y, const color_type& c)
        {
            pix_value_ptr(x, y, 1)->set(c);
        }

        //--------------------------------------------------------------------
        AGG_INLINE void blend_pixel(int x, int y, const color_type& c, int8u cover)
        {
            copy_or_blend_pix(pix_value_ptr(x, y, 1), c, cover);
        }

        //--------------------------------------------------------------------
        AGG_INLINE void copy_hline(int x, int y, 
                                   unsigned len, 
                                   const color_type& c)
        {
            pixel_type* p = pix_value_ptr(x, y, len);
            do
            {
                p->set(c);
                p = p->next();
            }
            while(--len);
        }


        //--------------------------------------------------------------------
        AGG_INLINE void copy_vline(int x, int y,
                                   unsigned len, 
                                   const color_type& c)
        {
            do
            {
                pix_value_ptr(x, y++, 1)->set(c);
            }
            while (--len);
        }

        //--------------------------------------------------------------------
        void blend_hline(int x, int y,
                         unsigned len, 
                         const color_type& c,
                         int8u cover)
        {
            if (!c.is_transparent())
            {
                pixel_type* p = pix_value_ptr(x, y, len);

                if (c.is_opaque() && cover == cover_mask)
                {
                    do
                    {
                        p->set(c);
                        p = p->next();
                    }
                    while (--len);
                }
                else
                {
                    do
                    {
                        blend_pix(p, c, cover);
                        p = p->next();
                    }
                    while (--len);
                }
            }
        }


        //--------------------------------------------------------------------
        void blend_vline(int x, int y,
                         unsigned len, 
                         const color_type& c,
                         int8u cover)
        {
            if (!c.is_transparent())
            {
                if (c.is_opaque() && cover == cover_mask)
                {
                    do
                    {
                        pix_value_ptr(x, y++, 1)->set(c);
                    }
                    while (--len);
                }
                else
                {
                    do
                    {
                        blend_pix(pix_value_ptr(x, y++, 1), c, cover);
                    }
                    while (--len);
                }
            }
        }

        //--------------------------------------------------------------------
        void blend_solid_hspan(int x, int y,
                               unsigned len, 
                               const color_type& c,
                               const int8u* covers)
        {
            if (!c.is_transparent())
            {
                pixel_type* p = pix_value_ptr(x, y, len);

                do 
                {
                    if (c.is_opaque() && *covers == cover_mask)
                    {
                        p->set(c);
                    }
                    else
                    {
                        blend_pix(p, c, *covers);
                    }
                    p = p->next();
                    ++covers;
                }
                while (--len);
            }
        }


        //--------------------------------------------------------------------
        void blend_solid_vspan(int x, int y,
                               unsigned len, 
                               const color_type& c,
                               const int8u* covers)
        {
            if (!c.is_transparent())
            {
                do 
                {
                    pixel_type* p = pix_value_ptr(x, y++, 1);

                    if (c.is_opaque() && *covers == cover_mask)
                    {
                        p->set(c);
                    }
                    else
                    {
                        blend_pix(p, c, *covers);
                    }
                    ++covers;
                }
                while (--len);
            }
        }

        //--------------------------------------------------------------------
        void copy_color_hspan(int x, int y,
                              unsigned len, 
                              const color_type* colors)
        {
            pixel_type* p = pix_value_ptr(x, y, len);

            do 
            {
                p->set(*colors++);
                p = p->next();
            }
            while (--len);
        }


        //--------------------------------------------------------------------
        void copy_color_vspan(int x, int y,
                              unsigned len, 
                              const color_type* colors)
        {
            do 
            {
                pix_value_ptr(x, y++, 1)->set(*colors++);
            }
            while (--len);
        }

        //--------------------------------------------------------------------
        void blend_color_hspan(int x, int y,
                               unsigned len, 
                               const color_type* colors,
                               const int8u* covers,
                               int8u cover)
        {
            pixel_type* p = pix_value_ptr(x, y, len);

            if (covers)
            {
                do 
                {
                    copy_or_blend_pix(p, *colors++, *covers++);
                    p = p->next();
                }
                while (--len);
            }
            else
            {
                if (cover == cover_mask)
                {
                    do 
                    {
                        copy_or_blend_pix(p, *colors++);
                        p = p->next();
                    }
                    while (--len);
                }
                else
                {
                    do 
                    {
                        copy_or_blend_pix(p, *colors++, cover);
                        p = p->next();
                    }
                    while (--len);
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
            if (covers)
            {
                do 
                {
                    copy_or_blend_pix(pix_value_ptr(x, y++, 1), *colors++, *covers++);
                }
                while (--len);
            }
            else
            {
                if (cover == cover_mask)
                {
                    do 
                    {
                        copy_or_blend_pix(pix_value_ptr(x, y++, 1), *colors++);
                    }
                    while (--len);
                }
                else
                {
                    do 
                    {
                        copy_or_blend_pix(pix_value_ptr(x, y++, 1), *colors++, cover);
                    }
                    while (--len);
                }
            }
        }

        //--------------------------------------------------------------------
        template<class Function> void for_each_pixel(Function f)
        {
            for (unsigned y = 0; y < height(); ++y)
            {
                row_data r = m_rbuf->row(y);
                if (r.ptr)
                {
                    unsigned len = r.x2 - r.x1 + 1;
                    pixel_type* p = pix_value_ptr(r.x1, y, len);
                    do
                    {
                        f(p->c);
                        p = p->next();
                    }
                    while (--len);
                }
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
        template<class RenBuf2>
        void copy_from(const RenBuf2& from, 
                       int xdst, int ydst,
                       int xsrc, int ysrc,
                       unsigned len)
        {
            if (const int8u* p = from.row_ptr(ysrc))
            {
                memmove(m_rbuf->row_ptr(xdst, ydst, len) + xdst * pix_width, 
                        p + xsrc * pix_width, 
                        len * pix_width);
            }
        }

        //--------------------------------------------------------------------
        // Blend from an RGBA surface.
        template<class SrcPixelFormatRenderer>
        void blend_from(const SrcPixelFormatRenderer& from, 
                        int xdst, int ydst,
                        int xsrc, int ysrc,
                        unsigned len,
                        int8u cover)
        {
            typedef typename SrcPixelFormatRenderer::pixel_type src_pixel_type;
            typedef typename SrcPixelFormatRenderer::order_type src_order;

            if (const src_pixel_type* psrc = from.pix_value_ptr(xsrc, ysrc))
            {
                pixel_type* pdst = pix_value_ptr(xdst, ydst, len);

                if (cover == cover_mask)
                {
                    do 
                    {
                        value_type alpha = psrc->c[src_order::A];
                        if (alpha <= color_type::empty_value())
                        {
                            if (alpha >= color_type::full_value())
                            {
                                pdst->c[order_type::R] = psrc->c[src_order::R];
                                pdst->c[order_type::G] = psrc->c[src_order::G];
                                pdst->c[order_type::B] = psrc->c[src_order::B];
                            }
                            else
                            {
                                blend_pix(pdst, 
                                    psrc->c[src_order::R],
                                    psrc->c[src_order::G],
                                    psrc->c[src_order::B],
                                    alpha);
                            }
                        }
                        psrc = psrc->next();
                        pdst = pdst->next();
                    }
                    while(--len);
                }
                else
                {
                    do 
                    {
                        copy_or_blend_pix(pdst, psrc->get(), cover);
                        psrc = psrc->next();
                        pdst = pdst->next();
                    }
                    while (--len);
                }
            }
        }

        //--------------------------------------------------------------------
        // Blend from single color, using grayscale surface as alpha channel.
        template<class SrcPixelFormatRenderer>
        void blend_from_color(const SrcPixelFormatRenderer& from, 
                              const color_type& color,
                              int xdst, int ydst,
                              int xsrc, int ysrc,
                              unsigned len,
                              int8u cover)
        {
            typedef typename SrcPixelFormatRenderer::pixel_type src_pixel_type;
            typedef typename SrcPixelFormatRenderer::color_type src_color_type;

            if (const src_pixel_type* psrc = from.pix_value_ptr(xsrc, ysrc))
            {
                pixel_type* pdst = pix_value_ptr(xdst, ydst, len);

                do 
                {
                    copy_or_blend_pix(pdst, color, src_color_type::scale_cover(cover, psrc->c[0]));
                    psrc = psrc->next();
                    pdst = pdst->next();
                }
                while (--len);
            }
        }

        //--------------------------------------------------------------------
        // Blend from color table, using grayscale surface as indexes into table. 
        // Obviously, this only works for integer value types.
        template<class SrcPixelFormatRenderer>
        void blend_from_lut(const SrcPixelFormatRenderer& from, 
                            const color_type* color_lut,
                            int xdst, int ydst,
                            int xsrc, int ysrc,
                            unsigned len,
                            int8u cover)
        {
            typedef typename SrcPixelFormatRenderer::pixel_type src_pixel_type;

            if (const src_pixel_type* psrc = from.pix_value_ptr(xsrc, ysrc))
            {
                pixel_type* pdst = pix_value_ptr(xdst, ydst, len);

                if (cover == cover_mask)
                {
                    do 
                    {
                        const color_type& color = color_lut[psrc->c[0]];
                        blend_pix(pdst, color);
                        psrc = psrc->next();
                        pdst = pdst->next();
                    }
                    while(--len);
                }
                else
                {
                    do 
                    {
                        copy_or_blend_pix(pdst, color_lut[psrc->c[0]], cover);
                        psrc = psrc->next();
                        pdst = pdst->next();
                    }
                    while(--len);
                }
            }
        }

    private:
        rbuf_type* m_rbuf;
        Blender    m_blender;
    };
    
    //-----------------------------------------------------------------------
    typedef blender_rgb<rgba8, order_rgb> blender_rgb24;
    typedef blender_rgb<rgba8, order_bgr> blender_bgr24;
    typedef blender_rgb<srgba8, order_rgb> blender_srgb24;
    typedef blender_rgb<srgba8, order_bgr> blender_sbgr24;
    typedef blender_rgb<rgba16, order_rgb> blender_rgb48;
    typedef blender_rgb<rgba16, order_bgr> blender_bgr48;
    typedef blender_rgb<rgba32, order_rgb> blender_rgb96;
    typedef blender_rgb<rgba32, order_bgr> blender_bgr96;

    typedef blender_rgb_pre<rgba8, order_rgb> blender_rgb24_pre;
    typedef blender_rgb_pre<rgba8, order_bgr> blender_bgr24_pre;
    typedef blender_rgb_pre<srgba8, order_rgb> blender_srgb24_pre;
    typedef blender_rgb_pre<srgba8, order_bgr> blender_sbgr24_pre;
    typedef blender_rgb_pre<rgba16, order_rgb> blender_rgb48_pre;
    typedef blender_rgb_pre<rgba16, order_bgr> blender_bgr48_pre;
    typedef blender_rgb_pre<rgba32, order_rgb> blender_rgb96_pre;
    typedef blender_rgb_pre<rgba32, order_bgr> blender_bgr96_pre;

    typedef pixfmt_alpha_blend_rgb<blender_rgb24, rendering_buffer, 3> pixfmt_rgb24;
    typedef pixfmt_alpha_blend_rgb<blender_bgr24, rendering_buffer, 3> pixfmt_bgr24;
    typedef pixfmt_alpha_blend_rgb<blender_srgb24, rendering_buffer, 3> pixfmt_srgb24;
    typedef pixfmt_alpha_blend_rgb<blender_sbgr24, rendering_buffer, 3> pixfmt_sbgr24;
    typedef pixfmt_alpha_blend_rgb<blender_rgb48, rendering_buffer, 3> pixfmt_rgb48;
    typedef pixfmt_alpha_blend_rgb<blender_bgr48, rendering_buffer, 3> pixfmt_bgr48;
    typedef pixfmt_alpha_blend_rgb<blender_rgb96, rendering_buffer, 3> pixfmt_rgb96;
    typedef pixfmt_alpha_blend_rgb<blender_bgr96, rendering_buffer, 3> pixfmt_bgr96;

    typedef pixfmt_alpha_blend_rgb<blender_rgb24_pre, rendering_buffer, 3> pixfmt_rgb24_pre;
    typedef pixfmt_alpha_blend_rgb<blender_bgr24_pre, rendering_buffer, 3> pixfmt_bgr24_pre;
    typedef pixfmt_alpha_blend_rgb<blender_srgb24_pre, rendering_buffer, 3> pixfmt_srgb24_pre;
    typedef pixfmt_alpha_blend_rgb<blender_sbgr24_pre, rendering_buffer, 3> pixfmt_sbgr24_pre;
    typedef pixfmt_alpha_blend_rgb<blender_rgb48_pre, rendering_buffer, 3> pixfmt_rgb48_pre;
    typedef pixfmt_alpha_blend_rgb<blender_bgr48_pre, rendering_buffer, 3> pixfmt_bgr48_pre;
    typedef pixfmt_alpha_blend_rgb<blender_rgb96_pre, rendering_buffer, 3> pixfmt_rgb96_pre;
    typedef pixfmt_alpha_blend_rgb<blender_bgr96_pre, rendering_buffer, 3> pixfmt_bgr96_pre;

    typedef pixfmt_alpha_blend_rgb<blender_rgb24, rendering_buffer, 4, 0> pixfmt_rgbx32;
    typedef pixfmt_alpha_blend_rgb<blender_rgb24, rendering_buffer, 4, 1> pixfmt_xrgb32;
    typedef pixfmt_alpha_blend_rgb<blender_bgr24, rendering_buffer, 4, 1> pixfmt_xbgr32;
    typedef pixfmt_alpha_blend_rgb<blender_bgr24, rendering_buffer, 4, 0> pixfmt_bgrx32;
    typedef pixfmt_alpha_blend_rgb<blender_srgb24, rendering_buffer, 4, 0> pixfmt_srgbx32;
    typedef pixfmt_alpha_blend_rgb<blender_srgb24, rendering_buffer, 4, 1> pixfmt_sxrgb32;
    typedef pixfmt_alpha_blend_rgb<blender_sbgr24, rendering_buffer, 4, 1> pixfmt_sxbgr32;
    typedef pixfmt_alpha_blend_rgb<blender_sbgr24, rendering_buffer, 4, 0> pixfmt_sbgrx32;
    typedef pixfmt_alpha_blend_rgb<blender_rgb48, rendering_buffer, 4, 0> pixfmt_rgbx64;
    typedef pixfmt_alpha_blend_rgb<blender_rgb48, rendering_buffer, 4, 1> pixfmt_xrgb64;
    typedef pixfmt_alpha_blend_rgb<blender_bgr48, rendering_buffer, 4, 1> pixfmt_xbgr64;
    typedef pixfmt_alpha_blend_rgb<blender_bgr48, rendering_buffer, 4, 0> pixfmt_bgrx64;
    typedef pixfmt_alpha_blend_rgb<blender_rgb96, rendering_buffer, 4, 0> pixfmt_rgbx128;
    typedef pixfmt_alpha_blend_rgb<blender_rgb96, rendering_buffer, 4, 1> pixfmt_xrgb128;
    typedef pixfmt_alpha_blend_rgb<blender_bgr96, rendering_buffer, 4, 1> pixfmt_xbgr128;
    typedef pixfmt_alpha_blend_rgb<blender_bgr96, rendering_buffer, 4, 0> pixfmt_bgrx128;

    typedef pixfmt_alpha_blend_rgb<blender_rgb24_pre, rendering_buffer, 4, 0> pixfmt_rgbx32_pre;
    typedef pixfmt_alpha_blend_rgb<blender_rgb24_pre, rendering_buffer, 4, 1> pixfmt_xrgb32_pre;
    typedef pixfmt_alpha_blend_rgb<blender_bgr24_pre, rendering_buffer, 4, 1> pixfmt_xbgr32_pre;
    typedef pixfmt_alpha_blend_rgb<blender_bgr24_pre, rendering_buffer, 4, 0> pixfmt_bgrx32_pre;
    typedef pixfmt_alpha_blend_rgb<blender_srgb24_pre, rendering_buffer, 4, 0> pixfmt_srgbx32_pre;
    typedef pixfmt_alpha_blend_rgb<blender_srgb24_pre, rendering_buffer, 4, 1> pixfmt_sxrgb32_pre;
    typedef pixfmt_alpha_blend_rgb<blender_sbgr24_pre, rendering_buffer, 4, 1> pixfmt_sxbgr32_pre;
    typedef pixfmt_alpha_blend_rgb<blender_sbgr24_pre, rendering_buffer, 4, 0> pixfmt_sbgrx32_pre;
    typedef pixfmt_alpha_blend_rgb<blender_rgb48_pre, rendering_buffer, 4, 0> pixfmt_rgbx64_pre;
    typedef pixfmt_alpha_blend_rgb<blender_rgb48_pre, rendering_buffer, 4, 1> pixfmt_xrgb64_pre;
    typedef pixfmt_alpha_blend_rgb<blender_bgr48_pre, rendering_buffer, 4, 1> pixfmt_xbgr64_pre;
    typedef pixfmt_alpha_blend_rgb<blender_bgr48_pre, rendering_buffer, 4, 0> pixfmt_bgrx64_pre;
    typedef pixfmt_alpha_blend_rgb<blender_rgb96_pre, rendering_buffer, 4, 0> pixfmt_rgbx128_pre;
    typedef pixfmt_alpha_blend_rgb<blender_rgb96_pre, rendering_buffer, 4, 1> pixfmt_xrgb128_pre;
    typedef pixfmt_alpha_blend_rgb<blender_bgr96_pre, rendering_buffer, 4, 1> pixfmt_xbgr128_pre;
    typedef pixfmt_alpha_blend_rgb<blender_bgr96_pre, rendering_buffer, 4, 0> pixfmt_bgrx128_pre;
    

    //-----------------------------------------------------pixfmt_rgb24_gamma
    template<class Gamma> class pixfmt_rgb24_gamma : 
    public pixfmt_alpha_blend_rgb<blender_rgb_gamma<rgba8, order_rgb, Gamma>, rendering_buffer, 3>
    {
    public:
        pixfmt_rgb24_gamma(rendering_buffer& rb, const Gamma& g) :
            pixfmt_alpha_blend_rgb<blender_rgb_gamma<rgba8, order_rgb, Gamma>, rendering_buffer, 3>(rb) 
        {
            this->blender().gamma(g);
        }
    };
        
    //-----------------------------------------------------pixfmt_srgb24_gamma
    template<class Gamma> class pixfmt_srgb24_gamma : 
    public pixfmt_alpha_blend_rgb<blender_rgb_gamma<srgba8, order_rgb, Gamma>, rendering_buffer, 3>
    {
    public:
        pixfmt_srgb24_gamma(rendering_buffer& rb, const Gamma& g) :
            pixfmt_alpha_blend_rgb<blender_rgb_gamma<srgba8, order_rgb, Gamma>, rendering_buffer, 3>(rb) 
        {
            this->blender().gamma(g);
        }
    };
        
    //-----------------------------------------------------pixfmt_bgr24_gamma
    template<class Gamma> class pixfmt_bgr24_gamma : 
    public pixfmt_alpha_blend_rgb<blender_rgb_gamma<rgba8, order_bgr, Gamma>, rendering_buffer, 3>
    {
    public:
        pixfmt_bgr24_gamma(rendering_buffer& rb, const Gamma& g) :
            pixfmt_alpha_blend_rgb<blender_rgb_gamma<rgba8, order_bgr, Gamma>, rendering_buffer, 3>(rb) 
        {
            this->blender().gamma(g);
        }
    };

    //-----------------------------------------------------pixfmt_sbgr24_gamma
    template<class Gamma> class pixfmt_sbgr24_gamma : 
    public pixfmt_alpha_blend_rgb<blender_rgb_gamma<srgba8, order_bgr, Gamma>, rendering_buffer, 3>
    {
    public:
        pixfmt_sbgr24_gamma(rendering_buffer& rb, const Gamma& g) :
            pixfmt_alpha_blend_rgb<blender_rgb_gamma<srgba8, order_bgr, Gamma>, rendering_buffer, 3>(rb) 
        {
            this->blender().gamma(g);
        }
    };

    //-----------------------------------------------------pixfmt_rgb48_gamma
    template<class Gamma> class pixfmt_rgb48_gamma : 
    public pixfmt_alpha_blend_rgb<blender_rgb_gamma<rgba16, order_rgb, Gamma>, rendering_buffer, 3>
    {
    public:
        pixfmt_rgb48_gamma(rendering_buffer& rb, const Gamma& g) :
            pixfmt_alpha_blend_rgb<blender_rgb_gamma<rgba16, order_rgb, Gamma>, rendering_buffer, 3>(rb) 
        {
            this->blender().gamma(g);
        }
    };
        
    //-----------------------------------------------------pixfmt_bgr48_gamma
    template<class Gamma> class pixfmt_bgr48_gamma : 
    public pixfmt_alpha_blend_rgb<blender_rgb_gamma<rgba16, order_bgr, Gamma>, rendering_buffer, 3>
    {
    public:
        pixfmt_bgr48_gamma(rendering_buffer& rb, const Gamma& g) :
            pixfmt_alpha_blend_rgb<blender_rgb_gamma<rgba16, order_bgr, Gamma>, rendering_buffer, 3>(rb) 
        {
            this->blender().gamma(g);
        }
    };
    
}

#endif

