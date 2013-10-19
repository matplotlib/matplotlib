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
// scanline_u8 class
//
//----------------------------------------------------------------------------
#ifndef AGG_ALPHA_MASK_U8_INCLUDED
#define AGG_ALPHA_MASK_U8_INCLUDED

#include <string.h>
#include "agg_basics.h"
#include "agg_rendering_buffer.h"

namespace agg
{
    //===================================================one_component_mask_u8
    struct one_component_mask_u8
    {
        static unsigned calculate(const int8u* p) { return *p; }
    };


    //=====================================================rgb_to_gray_mask_u8
    template<unsigned R, unsigned G, unsigned B>
    struct rgb_to_gray_mask_u8
    {
        static unsigned calculate(const int8u* p)
        {
            return (p[R]*77 + p[G]*150 + p[B]*29) >> 8;
        }
    };

    //==========================================================alpha_mask_u8
    template<unsigned Step=1, unsigned Offset=0, class MaskF=one_component_mask_u8>
    class alpha_mask_u8
    {
    public:
        typedef int8u cover_type;
        typedef alpha_mask_u8<Step, Offset, MaskF> self_type;
        enum cover_scale_e
        {
            cover_shift = 8,
            cover_none  = 0,
            cover_full  = 255
        };

        alpha_mask_u8() : m_rbuf(0) {}
        explicit alpha_mask_u8(rendering_buffer& rbuf) : m_rbuf(&rbuf) {}

        void attach(rendering_buffer& rbuf) { m_rbuf = &rbuf; }

        MaskF& mask_function() { return m_mask_function; }
        const MaskF& mask_function() const { return m_mask_function; }


        //--------------------------------------------------------------------
        cover_type pixel(int x, int y) const
        {
            if(x >= 0 && y >= 0 &&
               x < (int)m_rbuf->width() &&
               y < (int)m_rbuf->height())
            {
                return (cover_type)m_mask_function.calculate(
                                        m_rbuf->row_ptr(y) + x * Step + Offset);
            }
            return 0;
        }

        //--------------------------------------------------------------------
        cover_type combine_pixel(int x, int y, cover_type val) const
        {
            if(x >= 0 && y >= 0 &&
               x < (int)m_rbuf->width() &&
               y < (int)m_rbuf->height())
            {
                return (cover_type)((cover_full + val *
                                     m_mask_function.calculate(
                                        m_rbuf->row_ptr(y) + x * Step + Offset)) >>
                                     cover_shift);
            }
            return 0;
        }


        //--------------------------------------------------------------------
        void fill_hspan(int x, int y, cover_type* dst, int num_pix) const
        {
            int xmax = m_rbuf->width() - 1;
            int ymax = m_rbuf->height() - 1;

            int count = num_pix;
            cover_type* covers = dst;

            if(y < 0 || y > ymax)
            {
                memset(dst, 0, num_pix * sizeof(cover_type));
                return;
            }

            if(x < 0)
            {
                count += x;
                if(count <= 0)
                {
                    memset(dst, 0, num_pix * sizeof(cover_type));
                    return;
                }
                memset(covers, 0, -x * sizeof(cover_type));
                covers -= x;
                x = 0;
            }

            if(x + count > xmax)
            {
                int rest = x + count - xmax - 1;
                count -= rest;
                if(count <= 0)
                {
                    memset(dst, 0, num_pix * sizeof(cover_type));
                    return;
                }
                memset(covers + count, 0, rest * sizeof(cover_type));
            }

            const int8u* mask = m_rbuf->row_ptr(y) + x * Step + Offset;
            do
            {
                *covers++ = (cover_type)m_mask_function.calculate(mask);
                mask += Step;
            }
            while(--count);
        }


        //--------------------------------------------------------------------
        void combine_hspan(int x, int y, cover_type* dst, int num_pix) const
        {
            int xmax = m_rbuf->width() - 1;
            int ymax = m_rbuf->height() - 1;

            int count = num_pix;
            cover_type* covers = dst;

            if(y < 0 || y > ymax)
            {
                memset(dst, 0, num_pix * sizeof(cover_type));
                return;
            }

            if(x < 0)
            {
                count += x;
                if(count <= 0)
                {
                    memset(dst, 0, num_pix * sizeof(cover_type));
                    return;
                }
                memset(covers, 0, -x * sizeof(cover_type));
                covers -= x;
                x = 0;
            }

            if(x + count > xmax)
            {
                int rest = x + count - xmax - 1;
                count -= rest;
                if(count <= 0)
                {
                    memset(dst, 0, num_pix * sizeof(cover_type));
                    return;
                }
                memset(covers + count, 0, rest * sizeof(cover_type));
            }

            const int8u* mask = m_rbuf->row_ptr(y) + x * Step + Offset;
            do
            {
                *covers = (cover_type)((cover_full + (*covers) *
                                       m_mask_function.calculate(mask)) >>
                                       cover_shift);
                ++covers;
                mask += Step;
            }
            while(--count);
        }

        //--------------------------------------------------------------------
        void fill_vspan(int x, int y, cover_type* dst, int num_pix) const
        {
            int xmax = m_rbuf->width() - 1;
            int ymax = m_rbuf->height() - 1;

            int count = num_pix;
            cover_type* covers = dst;

            if(x < 0 || x > xmax)
            {
                memset(dst, 0, num_pix * sizeof(cover_type));
                return;
            }

            if(y < 0)
            {
                count += y;
                if(count <= 0)
                {
                    memset(dst, 0, num_pix * sizeof(cover_type));
                    return;
                }
                memset(covers, 0, -y * sizeof(cover_type));
                covers -= y;
                y = 0;
            }

            if(y + count > ymax)
            {
                int rest = y + count - ymax - 1;
                count -= rest;
                if(count <= 0)
                {
                    memset(dst, 0, num_pix * sizeof(cover_type));
                    return;
                }
                memset(covers + count, 0, rest * sizeof(cover_type));
            }

            const int8u* mask = m_rbuf->row_ptr(y) + x * Step + Offset;
            do
            {
                *covers++ = (cover_type)m_mask_function.calculate(mask);
                mask += m_rbuf->stride();
            }
            while(--count);
        }

        //--------------------------------------------------------------------
        void combine_vspan(int x, int y, cover_type* dst, int num_pix) const
        {
            int xmax = m_rbuf->width() - 1;
            int ymax = m_rbuf->height() - 1;

            int count = num_pix;
            cover_type* covers = dst;

            if(x < 0 || x > xmax)
            {
                memset(dst, 0, num_pix * sizeof(cover_type));
                return;
            }

            if(y < 0)
            {
                count += y;
                if(count <= 0)
                {
                    memset(dst, 0, num_pix * sizeof(cover_type));
                    return;
                }
                memset(covers, 0, -y * sizeof(cover_type));
                covers -= y;
                y = 0;
            }

            if(y + count > ymax)
            {
                int rest = y + count - ymax - 1;
                count -= rest;
                if(count <= 0)
                {
                    memset(dst, 0, num_pix * sizeof(cover_type));
                    return;
                }
                memset(covers + count, 0, rest * sizeof(cover_type));
            }

            const int8u* mask = m_rbuf->row_ptr(y) + x * Step + Offset;
            do
            {
                *covers = (cover_type)((cover_full + (*covers) *
                                       m_mask_function.calculate(mask)) >>
                                       cover_shift);
                ++covers;
                mask += m_rbuf->stride();
            }
            while(--count);
        }


    private:
        alpha_mask_u8(const self_type&);
        const self_type& operator = (const self_type&);

        agg::rendering_buffer* m_rbuf;
        MaskF             m_mask_function;
    };


    typedef alpha_mask_u8<1, 0> alpha_mask_gray8;   //----alpha_mask_gray8

    typedef alpha_mask_u8<3, 0> alpha_mask_rgb24r;  //----alpha_mask_rgb24r
    typedef alpha_mask_u8<3, 1> alpha_mask_rgb24g;  //----alpha_mask_rgb24g
    typedef alpha_mask_u8<3, 2> alpha_mask_rgb24b;  //----alpha_mask_rgb24b

    typedef alpha_mask_u8<3, 2> alpha_mask_bgr24r;  //----alpha_mask_bgr24r
    typedef alpha_mask_u8<3, 1> alpha_mask_bgr24g;  //----alpha_mask_bgr24g
    typedef alpha_mask_u8<3, 0> alpha_mask_bgr24b;  //----alpha_mask_bgr24b

    typedef alpha_mask_u8<4, 0> alpha_mask_rgba32r; //----alpha_mask_rgba32r
    typedef alpha_mask_u8<4, 1> alpha_mask_rgba32g; //----alpha_mask_rgba32g
    typedef alpha_mask_u8<4, 2> alpha_mask_rgba32b; //----alpha_mask_rgba32b
    typedef alpha_mask_u8<4, 3> alpha_mask_rgba32a; //----alpha_mask_rgba32a

    typedef alpha_mask_u8<4, 1> alpha_mask_argb32r; //----alpha_mask_argb32r
    typedef alpha_mask_u8<4, 2> alpha_mask_argb32g; //----alpha_mask_argb32g
    typedef alpha_mask_u8<4, 3> alpha_mask_argb32b; //----alpha_mask_argb32b
    typedef alpha_mask_u8<4, 0> alpha_mask_argb32a; //----alpha_mask_argb32a

    typedef alpha_mask_u8<4, 2> alpha_mask_bgra32r; //----alpha_mask_bgra32r
    typedef alpha_mask_u8<4, 1> alpha_mask_bgra32g; //----alpha_mask_bgra32g
    typedef alpha_mask_u8<4, 0> alpha_mask_bgra32b; //----alpha_mask_bgra32b
    typedef alpha_mask_u8<4, 3> alpha_mask_bgra32a; //----alpha_mask_bgra32a

    typedef alpha_mask_u8<4, 3> alpha_mask_abgr32r; //----alpha_mask_abgr32r
    typedef alpha_mask_u8<4, 2> alpha_mask_abgr32g; //----alpha_mask_abgr32g
    typedef alpha_mask_u8<4, 1> alpha_mask_abgr32b; //----alpha_mask_abgr32b
    typedef alpha_mask_u8<4, 0> alpha_mask_abgr32a; //----alpha_mask_abgr32a

    typedef alpha_mask_u8<3, 0, rgb_to_gray_mask_u8<0, 1, 2> > alpha_mask_rgb24gray;  //----alpha_mask_rgb24gray
    typedef alpha_mask_u8<3, 0, rgb_to_gray_mask_u8<2, 1, 0> > alpha_mask_bgr24gray;  //----alpha_mask_bgr24gray
    typedef alpha_mask_u8<4, 0, rgb_to_gray_mask_u8<0, 1, 2> > alpha_mask_rgba32gray; //----alpha_mask_rgba32gray
    typedef alpha_mask_u8<4, 1, rgb_to_gray_mask_u8<0, 1, 2> > alpha_mask_argb32gray; //----alpha_mask_argb32gray
    typedef alpha_mask_u8<4, 0, rgb_to_gray_mask_u8<2, 1, 0> > alpha_mask_bgra32gray; //----alpha_mask_bgra32gray
    typedef alpha_mask_u8<4, 1, rgb_to_gray_mask_u8<2, 1, 0> > alpha_mask_abgr32gray; //----alpha_mask_abgr32gray



    //==========================================================amask_no_clip_u8
    template<unsigned Step=1, unsigned Offset=0, class MaskF=one_component_mask_u8>
    class amask_no_clip_u8
    {
    public:
        typedef int8u cover_type;
        typedef amask_no_clip_u8<Step, Offset, MaskF> self_type;
        enum cover_scale_e
        {
            cover_shift = 8,
            cover_none  = 0,
            cover_full  = 255
        };

        amask_no_clip_u8() : m_rbuf(0) {}
        explicit amask_no_clip_u8(rendering_buffer& rbuf) : m_rbuf(&rbuf) {}

        void attach(rendering_buffer& rbuf) { m_rbuf = &rbuf; }

        MaskF& mask_function() { return m_mask_function; }
        const MaskF& mask_function() const { return m_mask_function; }


        //--------------------------------------------------------------------
        cover_type pixel(int x, int y) const
        {
            return (cover_type)m_mask_function.calculate(
                                   m_rbuf->row_ptr(y) + x * Step + Offset);
        }


        //--------------------------------------------------------------------
        cover_type combine_pixel(int x, int y, cover_type val) const
        {
            return (cover_type)((cover_full + val *
                                 m_mask_function.calculate(
                                    m_rbuf->row_ptr(y) + x * Step + Offset)) >>
                                 cover_shift);
        }


        //--------------------------------------------------------------------
        void fill_hspan(int x, int y, cover_type* dst, int num_pix) const
        {
            const int8u* mask = m_rbuf->row_ptr(y) + x * Step + Offset;
            do
            {
                *dst++ = (cover_type)m_mask_function.calculate(mask);
                mask += Step;
            }
            while(--num_pix);
        }



        //--------------------------------------------------------------------
        void combine_hspan(int x, int y, cover_type* dst, int num_pix) const
        {
            const int8u* mask = m_rbuf->row_ptr(y) + x * Step + Offset;
            do
            {
                *dst = (cover_type)((cover_full + (*dst) *
                                    m_mask_function.calculate(mask)) >>
                                    cover_shift);
                ++dst;
                mask += Step;
            }
            while(--num_pix);
        }


        //--------------------------------------------------------------------
        void fill_vspan(int x, int y, cover_type* dst, int num_pix) const
        {
            const int8u* mask = m_rbuf->row_ptr(y) + x * Step + Offset;
            do
            {
                *dst++ = (cover_type)m_mask_function.calculate(mask);
                mask += m_rbuf->stride();
            }
            while(--num_pix);
        }


        //--------------------------------------------------------------------
        void combine_vspan(int x, int y, cover_type* dst, int num_pix) const
        {
            const int8u* mask = m_rbuf->row_ptr(y) + x * Step + Offset;
            do
            {
                *dst = (cover_type)((cover_full + (*dst) *
                                    m_mask_function.calculate(mask)) >>
                                    cover_shift);
                ++dst;
                mask += m_rbuf->stride();
            }
            while(--num_pix);
        }

    private:
        amask_no_clip_u8(const self_type&);
        const self_type& operator = (const self_type&);

        agg::rendering_buffer* m_rbuf;
        MaskF             m_mask_function;
    };


    typedef amask_no_clip_u8<1, 0> amask_no_clip_gray8;   //----amask_no_clip_gray8

    typedef amask_no_clip_u8<3, 0> amask_no_clip_rgb24r;  //----amask_no_clip_rgb24r
    typedef amask_no_clip_u8<3, 1> amask_no_clip_rgb24g;  //----amask_no_clip_rgb24g
    typedef amask_no_clip_u8<3, 2> amask_no_clip_rgb24b;  //----amask_no_clip_rgb24b

    typedef amask_no_clip_u8<3, 2> amask_no_clip_bgr24r;  //----amask_no_clip_bgr24r
    typedef amask_no_clip_u8<3, 1> amask_no_clip_bgr24g;  //----amask_no_clip_bgr24g
    typedef amask_no_clip_u8<3, 0> amask_no_clip_bgr24b;  //----amask_no_clip_bgr24b

    typedef amask_no_clip_u8<4, 0> amask_no_clip_rgba32r; //----amask_no_clip_rgba32r
    typedef amask_no_clip_u8<4, 1> amask_no_clip_rgba32g; //----amask_no_clip_rgba32g
    typedef amask_no_clip_u8<4, 2> amask_no_clip_rgba32b; //----amask_no_clip_rgba32b
    typedef amask_no_clip_u8<4, 3> amask_no_clip_rgba32a; //----amask_no_clip_rgba32a

    typedef amask_no_clip_u8<4, 1> amask_no_clip_argb32r; //----amask_no_clip_argb32r
    typedef amask_no_clip_u8<4, 2> amask_no_clip_argb32g; //----amask_no_clip_argb32g
    typedef amask_no_clip_u8<4, 3> amask_no_clip_argb32b; //----amask_no_clip_argb32b
    typedef amask_no_clip_u8<4, 0> amask_no_clip_argb32a; //----amask_no_clip_argb32a

    typedef amask_no_clip_u8<4, 2> amask_no_clip_bgra32r; //----amask_no_clip_bgra32r
    typedef amask_no_clip_u8<4, 1> amask_no_clip_bgra32g; //----amask_no_clip_bgra32g
    typedef amask_no_clip_u8<4, 0> amask_no_clip_bgra32b; //----amask_no_clip_bgra32b
    typedef amask_no_clip_u8<4, 3> amask_no_clip_bgra32a; //----amask_no_clip_bgra32a

    typedef amask_no_clip_u8<4, 3> amask_no_clip_abgr32r; //----amask_no_clip_abgr32r
    typedef amask_no_clip_u8<4, 2> amask_no_clip_abgr32g; //----amask_no_clip_abgr32g
    typedef amask_no_clip_u8<4, 1> amask_no_clip_abgr32b; //----amask_no_clip_abgr32b
    typedef amask_no_clip_u8<4, 0> amask_no_clip_abgr32a; //----amask_no_clip_abgr32a

    typedef amask_no_clip_u8<3, 0, rgb_to_gray_mask_u8<0, 1, 2> > amask_no_clip_rgb24gray;  //----amask_no_clip_rgb24gray
    typedef amask_no_clip_u8<3, 0, rgb_to_gray_mask_u8<2, 1, 0> > amask_no_clip_bgr24gray;  //----amask_no_clip_bgr24gray
    typedef amask_no_clip_u8<4, 0, rgb_to_gray_mask_u8<0, 1, 2> > amask_no_clip_rgba32gray; //----amask_no_clip_rgba32gray
    typedef amask_no_clip_u8<4, 1, rgb_to_gray_mask_u8<0, 1, 2> > amask_no_clip_argb32gray; //----amask_no_clip_argb32gray
    typedef amask_no_clip_u8<4, 0, rgb_to_gray_mask_u8<2, 1, 0> > amask_no_clip_bgra32gray; //----amask_no_clip_bgra32gray
    typedef amask_no_clip_u8<4, 1, rgb_to_gray_mask_u8<2, 1, 0> > amask_no_clip_abgr32gray; //----amask_no_clip_abgr32gray


}



#endif
