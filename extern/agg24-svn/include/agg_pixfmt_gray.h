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

#ifndef AGG_PIXFMT_GRAY_INCLUDED
#define AGG_PIXFMT_GRAY_INCLUDED

#include <string.h>
#include "agg_pixfmt_base.h"
#include "agg_rendering_buffer.h"

namespace agg
{
 
    //============================================================blender_gray
    template<class ColorT> struct blender_gray
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;

        // Blend pixels using the non-premultiplied form of Alvy-Ray Smith's
        // compositing function. Since the render buffer is opaque we skip the
        // initial premultiply and final demultiply.

        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cv, value_type alpha, cover_type cover)
        {
            blend_pix(p, cv, color_type::mult_cover(alpha, cover));
        }

        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cv, value_type alpha)
        {
            *p = color_type::lerp(*p, cv, alpha);
        }
    };


    //======================================================blender_gray_pre
    template<class ColorT> struct blender_gray_pre
    {
        typedef ColorT color_type;
        typedef typename color_type::value_type value_type;
        typedef typename color_type::calc_type calc_type;
        typedef typename color_type::long_type long_type;

        // Blend pixels using the premultiplied form of Alvy-Ray Smith's
        // compositing function. 

        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cv, value_type alpha, cover_type cover)
        {
            blend_pix(p, color_type::mult_cover(cv, cover), color_type::mult_cover(alpha, cover));
        }

        static AGG_INLINE void blend_pix(value_type* p, 
            value_type cv, value_type alpha)
        {
            *p = color_type::prelerp(*p, cv, alpha);
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



    //=================================================pixfmt_alpha_blend_gray
    template<class Blender, class RenBuf, unsigned Step = 1, unsigned Offset = 0>
    class pixfmt_alpha_blend_gray
    {
    public:
        typedef pixfmt_gray_tag pixfmt_category;
        typedef RenBuf   rbuf_type;
        typedef typename rbuf_type::row_data row_data;
        typedef Blender  blender_type;
        typedef typename blender_type::color_type color_type;
        typedef int                               order_type; // A fake one
        typedef typename color_type::value_type   value_type;
        typedef typename color_type::calc_type    calc_type;
        enum 
        {
            num_components = 1,
            pix_width = sizeof(value_type) * Step,
            pix_step = Step,
            pix_offset = Offset,
        };
        struct pixel_type
        {
            value_type c[num_components];

            void set(value_type v)
            {
                c[0] = v;
            }

            void set(const color_type& color)
            {
                set(color.v);
            }

            void get(value_type& v) const
            {
                v = c[0];
            }

            color_type get() const
            {
                return color_type(c[0]);
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
            value_type v, value_type a, 
            unsigned cover)
        {
            blender_type::blend_pix(p->c, v, a, cover);
        }

        //--------------------------------------------------------------------
        AGG_INLINE void blend_pix(pixel_type* p, value_type v, value_type a)
        {
            blender_type::blend_pix(p->c, v, a);
        }

        //--------------------------------------------------------------------
        AGG_INLINE void blend_pix(pixel_type* p, const color_type& c, unsigned cover)
        {
            blender_type::blend_pix(p->c, c.v, c.a, cover);
        }

        //--------------------------------------------------------------------
        AGG_INLINE void blend_pix(pixel_type* p, const color_type& c)
        {
            blender_type::blend_pix(p->c, c.v, c.a);
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
        explicit pixfmt_alpha_blend_gray(rbuf_type& rb) :
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
        AGG_INLINE unsigned width()  const { return m_rbuf->width();  }
        AGG_INLINE unsigned height() const { return m_rbuf->height(); }
        AGG_INLINE int      stride() const { return m_rbuf->stride(); }

        //--------------------------------------------------------------------
        int8u* row_ptr(int y)       { return m_rbuf->row_ptr(y); }
        const int8u* row_ptr(int y) const { return m_rbuf->row_ptr(y); }
        row_data     row(int y)     const { return m_rbuf->row(y); }

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
            // Grayscale formats are implicitly premultiplied.
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
            unsigned y;
            for (y = 0; y < height(); ++y)
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
            for_each_pixel(apply_gamma_dir_gray<color_type, GammaLut>(g));
        }

        //--------------------------------------------------------------------
        template<class GammaLut> void apply_gamma_inv(const GammaLut& g)
        {
            for_each_pixel(apply_gamma_inv_gray<color_type, GammaLut>(g));
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

                do 
                {
                    copy_or_blend_pix(pdst, color_lut[psrc->c[0]], cover);
                    psrc = psrc->next();
                    pdst = pdst->next();
                }
                while (--len);
            }
        }

    private:
        rbuf_type* m_rbuf;
    };

    typedef blender_gray<gray8> blender_gray8;
    typedef blender_gray<sgray8> blender_sgray8;
    typedef blender_gray<gray16> blender_gray16;
    typedef blender_gray<gray32> blender_gray32;

    typedef blender_gray_pre<gray8> blender_gray8_pre;
    typedef blender_gray_pre<sgray8> blender_sgray8_pre;
    typedef blender_gray_pre<gray16> blender_gray16_pre;
    typedef blender_gray_pre<gray32> blender_gray32_pre;

    typedef pixfmt_alpha_blend_gray<blender_gray8, rendering_buffer> pixfmt_gray8;
    typedef pixfmt_alpha_blend_gray<blender_sgray8, rendering_buffer> pixfmt_sgray8;
    typedef pixfmt_alpha_blend_gray<blender_gray16, rendering_buffer> pixfmt_gray16;
    typedef pixfmt_alpha_blend_gray<blender_gray32, rendering_buffer> pixfmt_gray32;

    typedef pixfmt_alpha_blend_gray<blender_gray8_pre, rendering_buffer> pixfmt_gray8_pre;
    typedef pixfmt_alpha_blend_gray<blender_sgray8_pre, rendering_buffer> pixfmt_sgray8_pre;
    typedef pixfmt_alpha_blend_gray<blender_gray16_pre, rendering_buffer> pixfmt_gray16_pre;
    typedef pixfmt_alpha_blend_gray<blender_gray32_pre, rendering_buffer> pixfmt_gray32_pre;
}

#endif

