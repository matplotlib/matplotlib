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

#ifndef AGG_PIXFMT_RGB_PACKED_INCLUDED
#define AGG_PIXFMT_RGB_PACKED_INCLUDED

#include <string.h>
#include "agg_basics.h"
#include "agg_color_rgba.h"
#include "agg_rendering_buffer.h"

namespace agg
{
    //=========================================================blender_rgb555
    struct blender_rgb555
    {
        typedef rgba8 color_type;
        typedef color_type::value_type value_type;
        typedef color_type::calc_type calc_type;
        typedef int16u pixel_type;

        static AGG_INLINE void blend_pix(pixel_type* p, 
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha, 
                                         unsigned)
        {
            pixel_type rgb = *p;
            calc_type r = (rgb >> 7) & 0xF8;
            calc_type g = (rgb >> 2) & 0xF8;
            calc_type b = (rgb << 3) & 0xF8;
            *p = (pixel_type)
               (((((cr - r) * alpha + (r << 8)) >> 1)  & 0x7C00) |
                ((((cg - g) * alpha + (g << 8)) >> 6)  & 0x03E0) |
                 (((cb - b) * alpha + (b << 8)) >> 11) | 0x8000);
        }

        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((r & 0xF8) << 7) | 
                                ((g & 0xF8) << 2) | 
                                 (b >> 3) | 0x8000);
        }

        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p >> 7) & 0xF8, 
                              (p >> 2) & 0xF8, 
                              (p << 3) & 0xF8);
        }
    };


    //=====================================================blender_rgb555_pre
    struct blender_rgb555_pre
    {
        typedef rgba8 color_type;
        typedef color_type::value_type value_type;
        typedef color_type::calc_type calc_type;
        typedef int16u pixel_type;

        static AGG_INLINE void blend_pix(pixel_type* p, 
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha, 
                                         unsigned cover)
        {
            alpha = color_type::base_mask - alpha;
            pixel_type rgb = *p;
            calc_type r = (rgb >> 7) & 0xF8;
            calc_type g = (rgb >> 2) & 0xF8;
            calc_type b = (rgb << 3) & 0xF8;
            *p = (pixel_type)
               ((((r * alpha + cr * cover) >> 1)  & 0x7C00) |
                (((g * alpha + cg * cover) >> 6)  & 0x03E0) |
                 ((b * alpha + cb * cover) >> 11) | 0x8000);
        }

        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((r & 0xF8) << 7) | 
                                ((g & 0xF8) << 2) | 
                                 (b >> 3) | 0x8000);
        }

        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p >> 7) & 0xF8, 
                              (p >> 2) & 0xF8, 
                              (p << 3) & 0xF8);
        }
    };




    //=====================================================blender_rgb555_gamma
    template<class Gamma> class blender_rgb555_gamma
    {
    public:
        typedef rgba8 color_type;
        typedef color_type::value_type value_type;
        typedef color_type::calc_type calc_type;
        typedef int16u pixel_type;
        typedef Gamma gamma_type;

        blender_rgb555_gamma() : m_gamma(0) {}
        void gamma(const gamma_type& g) { m_gamma = &g; }

        AGG_INLINE void blend_pix(pixel_type* p, 
                                  unsigned cr, unsigned cg, unsigned cb,
                                  unsigned alpha, 
                                  unsigned)
        {
            pixel_type rgb = *p;
            calc_type r = m_gamma->dir((rgb >> 7) & 0xF8);
            calc_type g = m_gamma->dir((rgb >> 2) & 0xF8);
            calc_type b = m_gamma->dir((rgb << 3) & 0xF8);
            *p = (pixel_type)
               (((m_gamma->inv(((m_gamma->dir(cr) - r) * alpha + (r << 8)) >> 8) << 7) & 0x7C00) |
                ((m_gamma->inv(((m_gamma->dir(cg) - g) * alpha + (g << 8)) >> 8) << 2) & 0x03E0) |
                 (m_gamma->inv(((m_gamma->dir(cb) - b) * alpha + (b << 8)) >> 8) >> 3) | 0x8000);
        }

        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((r & 0xF8) << 7) | 
                                ((g & 0xF8) << 2) | 
                                 (b >> 3) | 0x8000);
        }

        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p >> 7) & 0xF8, 
                              (p >> 2) & 0xF8, 
                              (p << 3) & 0xF8);
        }

    private:
        const Gamma* m_gamma;
    };





    //=========================================================blender_rgb565
    struct blender_rgb565
    {
        typedef rgba8 color_type;
        typedef color_type::value_type value_type;
        typedef color_type::calc_type calc_type;
        typedef int16u pixel_type;

        static AGG_INLINE void blend_pix(pixel_type* p, 
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha, 
                                         unsigned)
        {
            pixel_type rgb = *p;
            calc_type r = (rgb >> 8) & 0xF8;
            calc_type g = (rgb >> 3) & 0xFC;
            calc_type b = (rgb << 3) & 0xF8;
            *p = (pixel_type)
               (((((cr - r) * alpha + (r << 8))     ) & 0xF800) |
                ((((cg - g) * alpha + (g << 8)) >> 5) & 0x07E0) |
                 (((cb - b) * alpha + (b << 8)) >> 11));
        }

        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
        }

        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p >> 8) & 0xF8, 
                              (p >> 3) & 0xFC, 
                              (p << 3) & 0xF8);
        }
    };



    //=====================================================blender_rgb565_pre
    struct blender_rgb565_pre
    {
        typedef rgba8 color_type;
        typedef color_type::value_type value_type;
        typedef color_type::calc_type calc_type;
        typedef int16u pixel_type;

        static AGG_INLINE void blend_pix(pixel_type* p, 
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha, 
                                         unsigned cover)
        {
            alpha = color_type::base_mask - alpha;
            pixel_type rgb = *p;
            calc_type r = (rgb >> 8) & 0xF8;
            calc_type g = (rgb >> 3) & 0xFC;
            calc_type b = (rgb << 3) & 0xF8;
            *p = (pixel_type)
               ((((r * alpha + cr * cover)      ) & 0xF800) |
                (((g * alpha + cg * cover) >> 5 ) & 0x07E0) |
                 ((b * alpha + cb * cover) >> 11));
        }

        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
        }

        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p >> 8) & 0xF8, 
                              (p >> 3) & 0xFC, 
                              (p << 3) & 0xF8);
        }
    };



    //=====================================================blender_rgb565_gamma
    template<class Gamma> class blender_rgb565_gamma
    {
    public:
        typedef rgba8 color_type;
        typedef color_type::value_type value_type;
        typedef color_type::calc_type calc_type;
        typedef int16u pixel_type;
        typedef Gamma gamma_type;

        blender_rgb565_gamma() : m_gamma(0) {}
        void gamma(const gamma_type& g) { m_gamma = &g; }

        AGG_INLINE void blend_pix(pixel_type* p, 
                                  unsigned cr, unsigned cg, unsigned cb,
                                  unsigned alpha, 
                                  unsigned)
        {
            pixel_type rgb = *p;
            calc_type r = m_gamma->dir((rgb >> 8) & 0xF8);
            calc_type g = m_gamma->dir((rgb >> 3) & 0xFC);
            calc_type b = m_gamma->dir((rgb << 3) & 0xF8);
            *p = (pixel_type)
               (((m_gamma->inv(((m_gamma->dir(cr) - r) * alpha + (r << 8)) >> 8) << 8) & 0xF800) |
                ((m_gamma->inv(((m_gamma->dir(cg) - g) * alpha + (g << 8)) >> 8) << 3) & 0x07E0) |
                 (m_gamma->inv(((m_gamma->dir(cb) - b) * alpha + (b << 8)) >> 8) >> 3));
        }

        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
        }

        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p >> 8) & 0xF8, 
                              (p >> 3) & 0xFC, 
                              (p << 3) & 0xF8);
        }

    private:
        const Gamma* m_gamma;
    };



    //=====================================================blender_rgbAAA
    struct blender_rgbAAA
    {
        typedef rgba16 color_type;
        typedef color_type::value_type value_type;
        typedef color_type::calc_type calc_type;
        typedef int32u pixel_type;

        static AGG_INLINE void blend_pix(pixel_type* p, 
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha, 
                                         unsigned)
        {
            pixel_type rgb = *p;
            calc_type r = (rgb >> 14) & 0xFFC0;
            calc_type g = (rgb >> 4)  & 0xFFC0;
            calc_type b = (rgb << 6)  & 0xFFC0;
            *p = (pixel_type)
               (((((cr - r) * alpha + (r << 16)) >> 2)  & 0x3FF00000) |
                ((((cg - g) * alpha + (g << 16)) >> 12) & 0x000FFC00) |
                 (((cb - b) * alpha + (b << 16)) >> 22) | 0xC0000000);
        }

        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((r & 0xFFC0) << 14) | 
                                ((g & 0xFFC0) << 4) | 
                                 (b >> 6) | 0xC0000000);
        }

        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p >> 14) & 0xFFC0, 
                              (p >> 4)  & 0xFFC0, 
                              (p << 6)  & 0xFFC0);
        }
    };



    //==================================================blender_rgbAAA_pre
    struct blender_rgbAAA_pre
    {
        typedef rgba16 color_type;
        typedef color_type::value_type value_type;
        typedef color_type::calc_type calc_type;
        typedef int32u pixel_type;

        static AGG_INLINE void blend_pix(pixel_type* p, 
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha, 
                                         unsigned cover)
        {
            alpha = color_type::base_mask - alpha;
            cover = (cover + 1) << (color_type::base_shift - 8);
            pixel_type rgb = *p;
            calc_type r = (rgb >> 14) & 0xFFC0;
            calc_type g = (rgb >> 4)  & 0xFFC0;
            calc_type b = (rgb << 6)  & 0xFFC0;
            *p = (pixel_type)
               ((((r * alpha + cr * cover) >> 2)  & 0x3FF00000) |
                (((g * alpha + cg * cover) >> 12) & 0x000FFC00) |
                 ((b * alpha + cb * cover) >> 22) | 0xC0000000);
        }

        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((r & 0xFFC0) << 14) | 
                                ((g & 0xFFC0) << 4) | 
                                 (b >> 6) | 0xC0000000);
        }

        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p >> 14) & 0xFFC0, 
                              (p >> 4)  & 0xFFC0, 
                              (p << 6)  & 0xFFC0);
        }
    };



    //=================================================blender_rgbAAA_gamma
    template<class Gamma> class blender_rgbAAA_gamma
    {
    public:
        typedef rgba16 color_type;
        typedef color_type::value_type value_type;
        typedef color_type::calc_type calc_type;
        typedef int32u pixel_type;
        typedef Gamma gamma_type;

        blender_rgbAAA_gamma() : m_gamma(0) {}
        void gamma(const gamma_type& g) { m_gamma = &g; }

        AGG_INLINE void blend_pix(pixel_type* p, 
                                  unsigned cr, unsigned cg, unsigned cb,
                                  unsigned alpha, 
                                  unsigned)
        {
            pixel_type rgb = *p;
            calc_type r = m_gamma->dir((rgb >> 14) & 0xFFC0);
            calc_type g = m_gamma->dir((rgb >> 4)  & 0xFFC0);
            calc_type b = m_gamma->dir((rgb << 6)  & 0xFFC0);
            *p = (pixel_type)
               (((m_gamma->inv(((m_gamma->dir(cr) - r) * alpha + (r << 16)) >> 16) << 14) & 0x3FF00000) |
                ((m_gamma->inv(((m_gamma->dir(cg) - g) * alpha + (g << 16)) >> 16) << 4 ) & 0x000FFC00) |
                 (m_gamma->inv(((m_gamma->dir(cb) - b) * alpha + (b << 16)) >> 16) >> 6 ) | 0xC0000000);
        }

        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((r & 0xFFC0) << 14) | 
                                ((g & 0xFFC0) << 4) | 
                                 (b >> 6) | 0xC0000000);
        }

        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p >> 14) & 0xFFC0, 
                              (p >> 4)  & 0xFFC0, 
                              (p << 6)  & 0xFFC0);
        }
    private:
        const Gamma* m_gamma;
    };


    //=====================================================blender_bgrAAA
    struct blender_bgrAAA
    {
        typedef rgba16 color_type;
        typedef color_type::value_type value_type;
        typedef color_type::calc_type calc_type;
        typedef int32u pixel_type;

        static AGG_INLINE void blend_pix(pixel_type* p, 
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha, 
                                         unsigned)
        {
            pixel_type bgr = *p;
            calc_type b = (bgr >> 14) & 0xFFC0;
            calc_type g = (bgr >> 4)  & 0xFFC0;
            calc_type r = (bgr << 6)  & 0xFFC0;
            *p = (pixel_type)
               (((((cb - b) * alpha + (b << 16)) >> 2)  & 0x3FF00000) |
                ((((cg - g) * alpha + (g << 16)) >> 12) & 0x000FFC00) |
                 (((cr - r) * alpha + (r << 16)) >> 22) | 0xC0000000);
        }

        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((b & 0xFFC0) << 14) | 
                                ((g & 0xFFC0) << 4) | 
                                 (r >> 6) | 0xC0000000);
        }

        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p << 6)  & 0xFFC0, 
                              (p >> 4)  & 0xFFC0, 
                              (p >> 14) & 0xFFC0);
        }
    };



    //=================================================blender_bgrAAA_pre
    struct blender_bgrAAA_pre
    {
        typedef rgba16 color_type;
        typedef color_type::value_type value_type;
        typedef color_type::calc_type calc_type;
        typedef int32u pixel_type;

        static AGG_INLINE void blend_pix(pixel_type* p, 
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha, 
                                         unsigned cover)
        {
            alpha = color_type::base_mask - alpha;
            cover = (cover + 1) << (color_type::base_shift - 8);
            pixel_type bgr = *p;
            calc_type b = (bgr >> 14) & 0xFFC0;
            calc_type g = (bgr >> 4)  & 0xFFC0;
            calc_type r = (bgr << 6)  & 0xFFC0;
            *p = (pixel_type)
               ((((b * alpha + cb * cover) >> 2)  & 0x3FF00000) |
                (((g * alpha + cg * cover) >> 12) & 0x000FFC00) |
                 ((r * alpha + cr * cover) >> 22) | 0xC0000000);
        }

        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((b & 0xFFC0) << 14) | 
                                ((g & 0xFFC0) << 4) | 
                                 (r >> 6) | 0xC0000000);
        }

        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p << 6)  & 0xFFC0, 
                              (p >> 4)  & 0xFFC0, 
                              (p >> 14) & 0xFFC0);
        }
    };



    //=================================================blender_bgrAAA_gamma
    template<class Gamma> class blender_bgrAAA_gamma
    {
    public:
        typedef rgba16 color_type;
        typedef color_type::value_type value_type;
        typedef color_type::calc_type calc_type;
        typedef int32u pixel_type;
        typedef Gamma gamma_type;

        blender_bgrAAA_gamma() : m_gamma(0) {}
        void gamma(const gamma_type& g) { m_gamma = &g; }

        AGG_INLINE void blend_pix(pixel_type* p, 
                                  unsigned cr, unsigned cg, unsigned cb,
                                  unsigned alpha, 
                                  unsigned)
        {
            pixel_type bgr = *p;
            calc_type b = m_gamma->dir((bgr >> 14) & 0xFFC0);
            calc_type g = m_gamma->dir((bgr >> 4)  & 0xFFC0);
            calc_type r = m_gamma->dir((bgr << 6)  & 0xFFC0);
            *p = (pixel_type)
               (((m_gamma->inv(((m_gamma->dir(cb) - b) * alpha + (b << 16)) >> 16) << 14) & 0x3FF00000) |
                ((m_gamma->inv(((m_gamma->dir(cg) - g) * alpha + (g << 16)) >> 16) << 4 ) & 0x000FFC00) |
                 (m_gamma->inv(((m_gamma->dir(cr) - r) * alpha + (r << 16)) >> 16) >> 6 ) | 0xC0000000);
        }

        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((b & 0xFFC0) << 14) | 
                                ((g & 0xFFC0) << 4) | 
                                 (r >> 6) | 0xC0000000);
        }

        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p << 6)  & 0xFFC0, 
                              (p >> 4)  & 0xFFC0, 
                              (p >> 14) & 0xFFC0);
        }

    private:
        const Gamma* m_gamma;
    };



    //=====================================================blender_rgbBBA
    struct blender_rgbBBA
    {
        typedef rgba16 color_type;
        typedef color_type::value_type value_type;
        typedef color_type::calc_type calc_type;
        typedef int32u pixel_type;

        static AGG_INLINE void blend_pix(pixel_type* p, 
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha, 
                                         unsigned)
        {
            pixel_type rgb = *p;
            calc_type r = (rgb >> 16) & 0xFFE0;
            calc_type g = (rgb >> 5)  & 0xFFE0;
            calc_type b = (rgb << 6)  & 0xFFC0;
            *p = (pixel_type)
               (((((cr - r) * alpha + (r << 16))      ) & 0xFFE00000) |
                ((((cg - g) * alpha + (g << 16)) >> 11) & 0x001FFC00) |
                 (((cb - b) * alpha + (b << 16)) >> 22));
        }

        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((r & 0xFFE0) << 16) | ((g & 0xFFE0) << 5) | (b >> 6));
        }

        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p >> 16) & 0xFFE0, 
                              (p >> 5)  & 0xFFE0, 
                              (p << 6)  & 0xFFC0);
        }
    };


    //=================================================blender_rgbBBA_pre
    struct blender_rgbBBA_pre
    {
        typedef rgba16 color_type;
        typedef color_type::value_type value_type;
        typedef color_type::calc_type calc_type;
        typedef int32u pixel_type;

        static AGG_INLINE void blend_pix(pixel_type* p, 
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha, 
                                         unsigned cover)
        {
            alpha = color_type::base_mask - alpha;
            cover = (cover + 1) << (color_type::base_shift - 8);
            pixel_type rgb = *p;
            calc_type r = (rgb >> 16) & 0xFFE0;
            calc_type g = (rgb >> 5)  & 0xFFE0;
            calc_type b = (rgb << 6)  & 0xFFC0;
            *p = (pixel_type)
               ((((r * alpha + cr * cover)      ) & 0xFFE00000) |
                (((g * alpha + cg * cover) >> 11) & 0x001FFC00) |
                 ((b * alpha + cb * cover) >> 22));
        }

        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((r & 0xFFE0) << 16) | ((g & 0xFFE0) << 5) | (b >> 6));
        }

        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p >> 16) & 0xFFE0, 
                              (p >> 5)  & 0xFFE0, 
                              (p << 6)  & 0xFFC0);
        }
    };



    //=================================================blender_rgbBBA_gamma
    template<class Gamma> class blender_rgbBBA_gamma
    {
    public:
        typedef rgba16 color_type;
        typedef color_type::value_type value_type;
        typedef color_type::calc_type calc_type;
        typedef int32u pixel_type;
        typedef Gamma gamma_type;

        blender_rgbBBA_gamma() : m_gamma(0) {}
        void gamma(const gamma_type& g) { m_gamma = &g; }

        AGG_INLINE void blend_pix(pixel_type* p, 
                                  unsigned cr, unsigned cg, unsigned cb,
                                  unsigned alpha, 
                                  unsigned)
        {
            pixel_type rgb = *p;
            calc_type r = m_gamma->dir((rgb >> 16) & 0xFFE0);
            calc_type g = m_gamma->dir((rgb >> 5)  & 0xFFE0);
            calc_type b = m_gamma->dir((rgb << 6)  & 0xFFC0);
            *p = (pixel_type)
               (((m_gamma->inv(((m_gamma->dir(cr) - r) * alpha + (r << 16)) >> 16) << 16) & 0xFFE00000) |
                ((m_gamma->inv(((m_gamma->dir(cg) - g) * alpha + (g << 16)) >> 16) << 5 ) & 0x001FFC00) |
                 (m_gamma->inv(((m_gamma->dir(cb) - b) * alpha + (b << 16)) >> 16) >> 6 ));
        }

        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((r & 0xFFE0) << 16) | ((g & 0xFFE0) << 5) | (b >> 6));
        }

        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p >> 16) & 0xFFE0, 
                              (p >> 5)  & 0xFFE0, 
                              (p << 6)  & 0xFFC0);
        }

    private:
        const Gamma* m_gamma;
    };


    //=====================================================blender_bgrABB
    struct blender_bgrABB
    {
        typedef rgba16 color_type;
        typedef color_type::value_type value_type;
        typedef color_type::calc_type calc_type;
        typedef int32u pixel_type;

        static AGG_INLINE void blend_pix(pixel_type* p, 
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha, 
                                         unsigned)
        {
            pixel_type bgr = *p;
            calc_type b = (bgr >> 16) & 0xFFC0;
            calc_type g = (bgr >> 6)  & 0xFFE0;
            calc_type r = (bgr << 5)  & 0xFFE0;
            *p = (pixel_type)
               (((((cb - b) * alpha + (b << 16))      ) & 0xFFC00000) |
                ((((cg - g) * alpha + (g << 16)) >> 10) & 0x003FF800) |
                 (((cr - r) * alpha + (r << 16)) >> 21));
        }

        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((b & 0xFFC0) << 16) | ((g & 0xFFE0) << 6) | (r >> 5));
        }

        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p << 5)  & 0xFFE0,
                              (p >> 6)  & 0xFFE0, 
                              (p >> 16) & 0xFFC0);
        }
    };


    //=================================================blender_bgrABB_pre
    struct blender_bgrABB_pre
    {
        typedef rgba16 color_type;
        typedef color_type::value_type value_type;
        typedef color_type::calc_type calc_type;
        typedef int32u pixel_type;

        static AGG_INLINE void blend_pix(pixel_type* p, 
                                         unsigned cr, unsigned cg, unsigned cb,
                                         unsigned alpha, 
                                         unsigned cover)
        {
            alpha = color_type::base_mask - alpha;
            cover = (cover + 1) << (color_type::base_shift - 8);
            pixel_type bgr = *p;
            calc_type b = (bgr >> 16) & 0xFFC0;
            calc_type g = (bgr >> 6)  & 0xFFE0;
            calc_type r = (bgr << 5)  & 0xFFE0;
            *p = (pixel_type)
               ((((b * alpha + cb * cover)      ) & 0xFFC00000) |
                (((g * alpha + cg * cover) >> 10) & 0x003FF800) |
                 ((r * alpha + cr * cover) >> 21));
        }

        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((b & 0xFFC0) << 16) | ((g & 0xFFE0) << 6) | (r >> 5));
        }

        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p << 5)  & 0xFFE0,
                              (p >> 6)  & 0xFFE0, 
                              (p >> 16) & 0xFFC0);
        }
    };



    //=================================================blender_bgrABB_gamma
    template<class Gamma> class blender_bgrABB_gamma
    {
    public:
        typedef rgba16 color_type;
        typedef color_type::value_type value_type;
        typedef color_type::calc_type calc_type;
        typedef int32u pixel_type;
        typedef Gamma gamma_type;

        blender_bgrABB_gamma() : m_gamma(0) {}
        void gamma(const gamma_type& g) { m_gamma = &g; }

        AGG_INLINE void blend_pix(pixel_type* p, 
                                  unsigned cr, unsigned cg, unsigned cb,
                                  unsigned alpha, 
                                  unsigned)
        {
            pixel_type bgr = *p;
            calc_type b = m_gamma->dir((bgr >> 16) & 0xFFC0);
            calc_type g = m_gamma->dir((bgr >> 6)  & 0xFFE0);
            calc_type r = m_gamma->dir((bgr << 5)  & 0xFFE0);
            *p = (pixel_type)
               (((m_gamma->inv(((m_gamma->dir(cb) - b) * alpha + (b << 16)) >> 16) << 16) & 0xFFC00000) |
                ((m_gamma->inv(((m_gamma->dir(cg) - g) * alpha + (g << 16)) >> 16) << 6 ) & 0x003FF800) |
                 (m_gamma->inv(((m_gamma->dir(cr) - r) * alpha + (r << 16)) >> 16) >> 5 ));
        }

        static AGG_INLINE pixel_type make_pix(unsigned r, unsigned g, unsigned b)
        {
            return (pixel_type)(((b & 0xFFC0) << 16) | ((g & 0xFFE0) << 6) | (r >> 5));
        }

        static AGG_INLINE color_type make_color(pixel_type p)
        {
            return color_type((p << 5)  & 0xFFE0,
                              (p >> 6)  & 0xFFE0, 
                              (p >> 16) & 0xFFC0);
        }

    private:
        const Gamma* m_gamma;
    };


    
    //===========================================pixfmt_alpha_blend_rgb_packed
    template<class Blender,  class RenBuf> class pixfmt_alpha_blend_rgb_packed
    {
    public:
        typedef RenBuf   rbuf_type;
        typedef typename rbuf_type::row_data row_data;
        typedef Blender  blender_type;
        typedef typename blender_type::color_type color_type;
        typedef typename blender_type::pixel_type pixel_type;
        typedef int                               order_type; // A fake one
        typedef typename color_type::value_type   value_type;
        typedef typename color_type::calc_type    calc_type;
        enum base_scale_e 
        {
            base_shift = color_type::base_shift,
            base_scale = color_type::base_scale,
            base_mask  = color_type::base_mask,
            pix_width  = sizeof(pixel_type),
        };

    private:
        //--------------------------------------------------------------------
        AGG_INLINE void copy_or_blend_pix(pixel_type* p, const color_type& c, unsigned cover)
        {
            if (c.a)
            {
                calc_type alpha = (calc_type(c.a) * (cover + 1)) >> 8;
                if(alpha == base_mask)
                {
                    *p = m_blender.make_pix(c.r, c.g, c.b);
                }
                else
                {
                    m_blender.blend_pix(p, c.r, c.g, c.b, alpha, cover);
                }
            }
        }

    public:
        //--------------------------------------------------------------------
        explicit pixfmt_alpha_blend_rgb_packed(rbuf_type& rb) : m_rbuf(&rb) {}
        void attach(rbuf_type& rb) { m_rbuf = &rb; }

        //--------------------------------------------------------------------
        template<class PixFmt>
        bool attach(PixFmt& pixf, int x1, int y1, int x2, int y2)
        {
            rect_i r(x1, y1, x2, y2);
            if(r.clip(rect_i(0, 0, pixf.width()-1, pixf.height()-1)))
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
            return m_rbuf->row_ptr(y) + x * pix_width;
        }

        AGG_INLINE const int8u* pix_ptr(int x, int y) const
        {
            return m_rbuf->row_ptr(y) + x * pix_width;
        }

        //--------------------------------------------------------------------
        AGG_INLINE void make_pix(int8u* p, const color_type& c)
        {
            *(pixel_type*)p = m_blender.make_pix(c.r, c.g, c.b);
        }

        //--------------------------------------------------------------------
        AGG_INLINE color_type pixel(int x, int y) const
        {
            return m_blender.make_color(((pixel_type*)m_rbuf->row_ptr(y))[x]);
        }

        //--------------------------------------------------------------------
        AGG_INLINE void copy_pixel(int x, int y, const color_type& c)
        {
            ((pixel_type*)
                m_rbuf->row_ptr(x, y, 1))[x] = 
                    m_blender.make_pix(c.r, c.g, c.b);
        }

        //--------------------------------------------------------------------
        AGG_INLINE void blend_pixel(int x, int y, const color_type& c, int8u cover)
        {
            copy_or_blend_pix((pixel_type*)m_rbuf->row_ptr(x, y, 1) + x, c, cover);
        }

        //--------------------------------------------------------------------
        AGG_INLINE void copy_hline(int x, int y, 
                                   unsigned len, 
                                   const color_type& c)
        {
            pixel_type* p = (pixel_type*)m_rbuf->row_ptr(x, y, len) + x;
            pixel_type v = m_blender.make_pix(c.r, c.g, c.b);
            do
            {
                *p++ = v;
            }
            while(--len);
        }

        //--------------------------------------------------------------------
        AGG_INLINE void copy_vline(int x, int y,
                                   unsigned len, 
                                   const color_type& c)
        {
            pixel_type v = m_blender.make_pix(c.r, c.g, c.b);
            do
            {
                pixel_type* p = (pixel_type*)m_rbuf->row_ptr(x, y++, 1) + x;
                *p = v;
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
                pixel_type* p = (pixel_type*)m_rbuf->row_ptr(x, y, len) + x;
                calc_type alpha = (calc_type(c.a) * (cover + 1)) >> 8;
                if(alpha == base_mask)
                {
                    pixel_type v = m_blender.make_pix(c.r, c.g, c.b);
                    do
                    {
                        *p++ = v;
                    }
                    while(--len);
                }
                else
                {
                    do
                    {
                        m_blender.blend_pix(p, c.r, c.g, c.b, alpha, cover);
                        ++p;
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
                calc_type alpha = (calc_type(c.a) * (cover + 1)) >> 8;
                if(alpha == base_mask)
                {
                    pixel_type v = m_blender.make_pix(c.r, c.g, c.b);
                    do
                    {
                        ((pixel_type*)m_rbuf->row_ptr(x, y++, 1))[x] = v;
                    }
                    while(--len);
                }
                else
                {
                    do
                    {
                        m_blender.blend_pix(
                            (pixel_type*)m_rbuf->row_ptr(x, y++, 1), 
                            c.r, c.g, c.b, alpha, cover);
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
            pixel_type* p = (pixel_type*)m_rbuf->row_ptr(x, y, len) + x;
            do 
            {
                copy_or_blend_pix(p, c, *covers++);
                ++p;
            }
            while(--len);
        }

        //--------------------------------------------------------------------
        void blend_solid_vspan(int x, int y,
                               unsigned len, 
                               const color_type& c,
                               const int8u* covers)
        {
            do 
            {
                copy_or_blend_pix((pixel_type*)m_rbuf->row_ptr(x, y++, 1) + x, 
                                  c, *covers++);
            }
            while(--len);
        }

        //--------------------------------------------------------------------
        void copy_color_hspan(int x, int y,
                              unsigned len, 
                              const color_type* colors)
        {
            pixel_type* p = (pixel_type*)m_rbuf->row_ptr(x, y, len) + x;
            do 
            {
                *p++ = m_blender.make_pix(colors->r, colors->g, colors->b);
                ++colors;
            }
            while(--len);
        }

        //--------------------------------------------------------------------
        void copy_color_vspan(int x, int y,
                              unsigned len, 
                              const color_type* colors)
        {
            do 
            {
                pixel_type* p = (pixel_type*)m_rbuf->row_ptr(x, y++, 1) + x;
                *p = m_blender.make_pix(colors->r, colors->g, colors->b);
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
            pixel_type* p = (pixel_type*)m_rbuf->row_ptr(x, y, len) + x;
            do 
            {
                copy_or_blend_pix(p++, *colors++, covers ? *covers++ : cover);
            }
            while(--len);
        }

        //--------------------------------------------------------------------
        void blend_color_vspan(int x, int y,
                               unsigned len, 
                               const color_type* colors,
                               const int8u* covers,
                               int8u cover)
        {
            do 
            {
                copy_or_blend_pix((pixel_type*)m_rbuf->row_ptr(x, y++, 1) + x, 
                                  *colors++, covers ? *covers++ : cover);
            }
            while(--len);
        }
        
        //--------------------------------------------------------------------
        template<class RenBuf2>
        void copy_from(const RenBuf2& from, 
                       int xdst, int ydst,
                       int xsrc, int ysrc,
                       unsigned len)
        {
            const int8u* p = from.row_ptr(ysrc);
            if(p)
            {
                memmove(m_rbuf->row_ptr(xdst, ydst, len) + xdst * pix_width, 
                        p + xsrc * pix_width, 
                        len * pix_width);
            }
        }

        //--------------------------------------------------------------------
        template<class SrcPixelFormatRenderer>
        void blend_from(const SrcPixelFormatRenderer& from, 
                        int xdst, int ydst,
                        int xsrc, int ysrc,
                        unsigned len,
                        int8u cover)
        {
            typedef typename SrcPixelFormatRenderer::order_type src_order;

            const value_type* psrc = (const value_type*)from.row_ptr(ysrc);
            if(psrc)
            {
                psrc += xsrc * 4;
                pixel_type* pdst = 
                    (pixel_type*)m_rbuf->row_ptr(xdst, ydst, len) + xdst;
                do 
                {
                    value_type alpha = psrc[src_order::A];
                    if(alpha)
                    {
                        if(alpha == base_mask && cover == 255)
                        {
                            *pdst = m_blender.make_pix(psrc[src_order::R], 
                                                       psrc[src_order::G],
                                                       psrc[src_order::B]);
                        }
                        else
                        {
                            m_blender.blend_pix(pdst, 
                                                psrc[src_order::R],
                                                psrc[src_order::G],
                                                psrc[src_order::B],
                                                alpha,
                                                cover);
                        }
                    }
                    psrc += 4;
                    ++pdst;
                }
                while(--len);
            }
        }

        //--------------------------------------------------------------------
        template<class SrcPixelFormatRenderer>
        void blend_from_color(const SrcPixelFormatRenderer& from, 
                              const color_type& color,
                              int xdst, int ydst,
                              int xsrc, int ysrc,
                              unsigned len,
                              int8u cover)
        {
            typedef typename SrcPixelFormatRenderer::value_type src_value_type;
            typedef typename SrcPixelFormatRenderer::color_type src_color_type;
            const src_value_type* psrc = (src_value_type*)from.row_ptr(ysrc);
            if(psrc)
            {
                psrc += xsrc * SrcPixelFormatRenderer::pix_step + SrcPixelFormatRenderer::pix_offset;
                pixel_type* pdst = 
                    (pixel_type*)m_rbuf->row_ptr(xdst, ydst, len) + xdst;

                do 
                {
                    m_blender.blend_pix(pdst, 
                                        color.r, color.g, color.b, color.a,
                                        cover);
                    psrc += SrcPixelFormatRenderer::pix_step;
                    ++pdst;
                }
                while(--len);
            }
        }

        //--------------------------------------------------------------------
        template<class SrcPixelFormatRenderer>
        void blend_from_lut(const SrcPixelFormatRenderer& from, 
                            const color_type* color_lut,
                            int xdst, int ydst,
                            int xsrc, int ysrc,
                            unsigned len,
                            int8u cover)
        {
            typedef typename SrcPixelFormatRenderer::value_type src_value_type;
            const src_value_type* psrc = (src_value_type*)from.row_ptr(ysrc);
            if(psrc)
            {
                psrc += xsrc * SrcPixelFormatRenderer::pix_step + SrcPixelFormatRenderer::pix_offset;
                pixel_type* pdst = 
                    (pixel_type*)m_rbuf->row_ptr(xdst, ydst, len) + xdst;

                do 
                {
                    const color_type& color = color_lut[*psrc];
                    m_blender.blend_pix(pdst, 
                                        color.r, color.g, color.b, color.a,
                                        cover);
                    psrc += SrcPixelFormatRenderer::pix_step;
                    ++pdst;
                }
                while(--len);
            }
        }



    private:
        rbuf_type* m_rbuf;
        Blender    m_blender;
    };

    typedef pixfmt_alpha_blend_rgb_packed<blender_rgb555, rendering_buffer> pixfmt_rgb555; //----pixfmt_rgb555
    typedef pixfmt_alpha_blend_rgb_packed<blender_rgb565, rendering_buffer> pixfmt_rgb565; //----pixfmt_rgb565

    typedef pixfmt_alpha_blend_rgb_packed<blender_rgb555_pre, rendering_buffer> pixfmt_rgb555_pre; //----pixfmt_rgb555_pre
    typedef pixfmt_alpha_blend_rgb_packed<blender_rgb565_pre, rendering_buffer> pixfmt_rgb565_pre; //----pixfmt_rgb565_pre

    typedef pixfmt_alpha_blend_rgb_packed<blender_rgbAAA, rendering_buffer> pixfmt_rgbAAA; //----pixfmt_rgbAAA
    typedef pixfmt_alpha_blend_rgb_packed<blender_bgrAAA, rendering_buffer> pixfmt_bgrAAA; //----pixfmt_bgrAAA
    typedef pixfmt_alpha_blend_rgb_packed<blender_rgbBBA, rendering_buffer> pixfmt_rgbBBA; //----pixfmt_rgbBBA
    typedef pixfmt_alpha_blend_rgb_packed<blender_bgrABB, rendering_buffer> pixfmt_bgrABB; //----pixfmt_bgrABB

    typedef pixfmt_alpha_blend_rgb_packed<blender_rgbAAA_pre, rendering_buffer> pixfmt_rgbAAA_pre; //----pixfmt_rgbAAA_pre
    typedef pixfmt_alpha_blend_rgb_packed<blender_bgrAAA_pre, rendering_buffer> pixfmt_bgrAAA_pre; //----pixfmt_bgrAAA_pre
    typedef pixfmt_alpha_blend_rgb_packed<blender_rgbBBA_pre, rendering_buffer> pixfmt_rgbBBA_pre; //----pixfmt_rgbBBA_pre
    typedef pixfmt_alpha_blend_rgb_packed<blender_bgrABB_pre, rendering_buffer> pixfmt_bgrABB_pre; //----pixfmt_bgrABB_pre


    //-----------------------------------------------------pixfmt_rgb555_gamma
    template<class Gamma> class pixfmt_rgb555_gamma : 
    public pixfmt_alpha_blend_rgb_packed<blender_rgb555_gamma<Gamma>, 
                                         rendering_buffer>
    {
    public:
        pixfmt_rgb555_gamma(rendering_buffer& rb, const Gamma& g) :
            pixfmt_alpha_blend_rgb_packed<blender_rgb555_gamma<Gamma>, 
                                          rendering_buffer>(rb) 
        {
            this->blender().gamma(g);
        }
    };


    //-----------------------------------------------------pixfmt_rgb565_gamma
    template<class Gamma> class pixfmt_rgb565_gamma : 
    public pixfmt_alpha_blend_rgb_packed<blender_rgb565_gamma<Gamma>, rendering_buffer>
    {
    public:
        pixfmt_rgb565_gamma(rendering_buffer& rb, const Gamma& g) :
            pixfmt_alpha_blend_rgb_packed<blender_rgb565_gamma<Gamma>, rendering_buffer>(rb) 
        {
            this->blender().gamma(g);
        }
    };


    //-----------------------------------------------------pixfmt_rgbAAA_gamma
    template<class Gamma> class pixfmt_rgbAAA_gamma : 
    public pixfmt_alpha_blend_rgb_packed<blender_rgbAAA_gamma<Gamma>, 
                                         rendering_buffer>
    {
    public:
        pixfmt_rgbAAA_gamma(rendering_buffer& rb, const Gamma& g) :
            pixfmt_alpha_blend_rgb_packed<blender_rgbAAA_gamma<Gamma>, 
                                          rendering_buffer>(rb) 
        {
            this->blender().gamma(g);
        }
    };


    //-----------------------------------------------------pixfmt_bgrAAA_gamma
    template<class Gamma> class pixfmt_bgrAAA_gamma : 
    public pixfmt_alpha_blend_rgb_packed<blender_bgrAAA_gamma<Gamma>, 
                                         rendering_buffer>
    {
    public:
        pixfmt_bgrAAA_gamma(rendering_buffer& rb, const Gamma& g) :
            pixfmt_alpha_blend_rgb_packed<blender_bgrAAA_gamma<Gamma>, 
                                          rendering_buffer>(rb) 
        {
            this->blender().gamma(g);
        }
    };


    //-----------------------------------------------------pixfmt_rgbBBA_gamma
    template<class Gamma> class pixfmt_rgbBBA_gamma : 
    public pixfmt_alpha_blend_rgb_packed<blender_rgbBBA_gamma<Gamma>, 
                                         rendering_buffer>
    {
    public:
        pixfmt_rgbBBA_gamma(rendering_buffer& rb, const Gamma& g) :
            pixfmt_alpha_blend_rgb_packed<blender_rgbBBA_gamma<Gamma>, 
                                          rendering_buffer>(rb) 
        {
            this->blender().gamma(g);
        }
    };


    //-----------------------------------------------------pixfmt_bgrABB_gamma
    template<class Gamma> class pixfmt_bgrABB_gamma : 
    public pixfmt_alpha_blend_rgb_packed<blender_bgrABB_gamma<Gamma>, 
                                         rendering_buffer>
    {
    public:
        pixfmt_bgrABB_gamma(rendering_buffer& rb, const Gamma& g) :
            pixfmt_alpha_blend_rgb_packed<blender_bgrABB_gamma<Gamma>, 
                                          rendering_buffer>(rb) 
        {
            this->blender().gamma(g);
        }
    };


}

#endif

