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
// A set of functors used with color_conv(). See file agg_color_conv.h
// These functors can convert images with up to 8 bits per component.
// Use convertors in the following way:
//
// agg::color_conv(dst, src, agg::color_conv_XXXX_to_YYYY());
// whare XXXX and YYYY can be any of:
//  rgb24
//  bgr24
//  rgba32
//  abgr32
//  argb32
//  bgra32
//  rgb555
//  rgb565
//----------------------------------------------------------------------------

#ifndef AGG_COLOR_CONV_RGB8_INCLUDED
#define AGG_COLOR_CONV_RGB8_INCLUDED

#include "agg_basics.h"
#include "agg_color_conv.h"

namespace agg
{

    //-----------------------------------------------------color_conv_rgb24
    class color_conv_rgb24
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                *dst++ = src[2];
                *dst++ = src[1];
                *dst++ = src[0];
                src += 3;
            }
            while(--width);
        }
    };

    typedef color_conv_rgb24 color_conv_rgb24_to_bgr24;
    typedef color_conv_rgb24 color_conv_bgr24_to_rgb24;

    typedef color_conv_same<3> color_conv_bgr24_to_bgr24;
    typedef color_conv_same<3> color_conv_rgb24_to_rgb24;



    //------------------------------------------------------color_conv_rgba32
    template<int I1, int I2, int I3, int I4> class color_conv_rgba32
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                *dst++ = src[I1];
                *dst++ = src[I2];
                *dst++ = src[I3];
                *dst++ = src[I4]; 
                src += 4;
            }
            while(--width);
        }
    };


    //------------------------------------------------------------------------
    typedef color_conv_rgba32<0,3,2,1> color_conv_argb32_to_abgr32; //----color_conv_argb32_to_abgr32
    typedef color_conv_rgba32<3,2,1,0> color_conv_argb32_to_bgra32; //----color_conv_argb32_to_bgra32
    typedef color_conv_rgba32<1,2,3,0> color_conv_argb32_to_rgba32; //----color_conv_argb32_to_rgba32
    typedef color_conv_rgba32<3,0,1,2> color_conv_bgra32_to_abgr32; //----color_conv_bgra32_to_abgr32
    typedef color_conv_rgba32<3,2,1,0> color_conv_bgra32_to_argb32; //----color_conv_bgra32_to_argb32
    typedef color_conv_rgba32<2,1,0,3> color_conv_bgra32_to_rgba32; //----color_conv_bgra32_to_rgba32
    typedef color_conv_rgba32<3,2,1,0> color_conv_rgba32_to_abgr32; //----color_conv_rgba32_to_abgr32
    typedef color_conv_rgba32<3,0,1,2> color_conv_rgba32_to_argb32; //----color_conv_rgba32_to_argb32
    typedef color_conv_rgba32<2,1,0,3> color_conv_rgba32_to_bgra32; //----color_conv_rgba32_to_bgra32
    typedef color_conv_rgba32<0,3,2,1> color_conv_abgr32_to_argb32; //----color_conv_abgr32_to_argb32
    typedef color_conv_rgba32<1,2,3,0> color_conv_abgr32_to_bgra32; //----color_conv_abgr32_to_bgra32
    typedef color_conv_rgba32<3,2,1,0> color_conv_abgr32_to_rgba32; //----color_conv_abgr32_to_rgba32

    //------------------------------------------------------------------------
    typedef color_conv_same<4> color_conv_rgba32_to_rgba32; //----color_conv_rgba32_to_rgba32
    typedef color_conv_same<4> color_conv_argb32_to_argb32; //----color_conv_argb32_to_argb32
    typedef color_conv_same<4> color_conv_bgra32_to_bgra32; //----color_conv_bgra32_to_bgra32
    typedef color_conv_same<4> color_conv_abgr32_to_abgr32; //----color_conv_abgr32_to_abgr32


    //--------------------------------------------color_conv_rgb24_rgba32
    template<int I1, int I2, int I3, int A> class color_conv_rgb24_rgba32
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                dst[I1] = *src++;
                dst[I2] = *src++;
                dst[I3] = *src++;
                dst[A]  = 255; 
                dst += 4;
            }
            while(--width);
        }
    };


    //------------------------------------------------------------------------
    typedef color_conv_rgb24_rgba32<1,2,3,0> color_conv_rgb24_to_argb32; //----color_conv_rgb24_to_argb32
    typedef color_conv_rgb24_rgba32<3,2,1,0> color_conv_rgb24_to_abgr32; //----color_conv_rgb24_to_abgr32
    typedef color_conv_rgb24_rgba32<2,1,0,3> color_conv_rgb24_to_bgra32; //----color_conv_rgb24_to_bgra32
    typedef color_conv_rgb24_rgba32<0,1,2,3> color_conv_rgb24_to_rgba32; //----color_conv_rgb24_to_rgba32
    typedef color_conv_rgb24_rgba32<3,2,1,0> color_conv_bgr24_to_argb32; //----color_conv_bgr24_to_argb32
    typedef color_conv_rgb24_rgba32<1,2,3,0> color_conv_bgr24_to_abgr32; //----color_conv_bgr24_to_abgr32
    typedef color_conv_rgb24_rgba32<0,1,2,3> color_conv_bgr24_to_bgra32; //----color_conv_bgr24_to_bgra32
    typedef color_conv_rgb24_rgba32<2,1,0,3> color_conv_bgr24_to_rgba32; //----color_conv_bgr24_to_rgba32

    

    //-------------------------------------------------color_conv_rgba32_rgb24
    template<int I1, int I2, int I3> class color_conv_rgba32_rgb24
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                *dst++ = src[I1];
                *dst++ = src[I2];
                *dst++ = src[I3];
                src += 4;
            }
            while(--width);
        }
    };



    //------------------------------------------------------------------------
    typedef color_conv_rgba32_rgb24<1,2,3> color_conv_argb32_to_rgb24; //----color_conv_argb32_to_rgb24
    typedef color_conv_rgba32_rgb24<3,2,1> color_conv_abgr32_to_rgb24; //----color_conv_abgr32_to_rgb24
    typedef color_conv_rgba32_rgb24<2,1,0> color_conv_bgra32_to_rgb24; //----color_conv_bgra32_to_rgb24
    typedef color_conv_rgba32_rgb24<0,1,2> color_conv_rgba32_to_rgb24; //----color_conv_rgba32_to_rgb24
    typedef color_conv_rgba32_rgb24<3,2,1> color_conv_argb32_to_bgr24; //----color_conv_argb32_to_bgr24
    typedef color_conv_rgba32_rgb24<1,2,3> color_conv_abgr32_to_bgr24; //----color_conv_abgr32_to_bgr24
    typedef color_conv_rgba32_rgb24<0,1,2> color_conv_bgra32_to_bgr24; //----color_conv_bgra32_to_bgr24
    typedef color_conv_rgba32_rgb24<2,1,0> color_conv_rgba32_to_bgr24; //----color_conv_rgba32_to_bgr24


    //------------------------------------------------color_conv_rgb555_rgb24
    template<int R, int B> class color_conv_rgb555_rgb24
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                unsigned rgb = *(int16u*)src;
                dst[R] = (int8u)((rgb >> 7) & 0xF8);
                dst[1] = (int8u)((rgb >> 2) & 0xF8);
                dst[B] = (int8u)((rgb << 3) & 0xF8);
                src += 2;
                dst += 3;
            }
            while(--width);
        }
    };


    //------------------------------------------------------------------------
    typedef color_conv_rgb555_rgb24<2,0> color_conv_rgb555_to_bgr24; //----color_conv_rgb555_to_bgr24
    typedef color_conv_rgb555_rgb24<0,2> color_conv_rgb555_to_rgb24; //----color_conv_rgb555_to_rgb24


    //-------------------------------------------------color_conv_rgb24_rgb555
    template<int R, int B> class color_conv_rgb24_rgb555
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                *(int16u*)dst = (int16u)(((unsigned(src[R]) << 7) & 0x7C00) | 
                                         ((unsigned(src[1]) << 2) & 0x3E0)  |
                                         ((unsigned(src[B]) >> 3)));
                src += 3;
                dst += 2;
            }
            while(--width);
        }
    };


    //------------------------------------------------------------------------
    typedef color_conv_rgb24_rgb555<2,0> color_conv_bgr24_to_rgb555; //----color_conv_bgr24_to_rgb555
    typedef color_conv_rgb24_rgb555<0,2> color_conv_rgb24_to_rgb555; //----color_conv_rgb24_to_rgb555


    //-------------------------------------------------color_conv_rgb565_rgb24
    template<int R, int B> class color_conv_rgb565_rgb24
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                unsigned rgb = *(int16u*)src;
                dst[R] = (rgb >> 8) & 0xF8;
                dst[1] = (rgb >> 3) & 0xFC;
                dst[B] = (rgb << 3) & 0xF8;
                src += 2;
                dst += 3;
            }
            while(--width);
        }
    };


    //------------------------------------------------------------------------
    typedef color_conv_rgb565_rgb24<2,0> color_conv_rgb565_to_bgr24; //----color_conv_rgb565_to_bgr24
    typedef color_conv_rgb565_rgb24<0,2> color_conv_rgb565_to_rgb24; //----color_conv_rgb565_to_rgb24


    //-------------------------------------------------color_conv_rgb24_rgb565
    template<int R, int B> class color_conv_rgb24_rgb565
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                *(int16u*)dst = (int16u)(((unsigned(src[R]) << 8) & 0xF800) | 
                                         ((unsigned(src[1]) << 3) & 0x7E0)  |
                                         ((unsigned(src[B]) >> 3)));
                src += 3;
                dst += 2;
            }
            while(--width);
        }
    };


    //------------------------------------------------------------------------
    typedef color_conv_rgb24_rgb565<2,0> color_conv_bgr24_to_rgb565; //----color_conv_bgr24_to_rgb565
    typedef color_conv_rgb24_rgb565<0,2> color_conv_rgb24_to_rgb565; //----color_conv_rgb24_to_rgb565



    //-------------------------------------------------color_conv_rgb555_rgba32
    template<int R, int G, int B, int A> class color_conv_rgb555_rgba32
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                int rgb = *(int16*)src;
                dst[R] = (int8u)((rgb >> 7) & 0xF8);
                dst[G] = (int8u)((rgb >> 2) & 0xF8);
                dst[B] = (int8u)((rgb << 3) & 0xF8);
                dst[A] = (int8u)(rgb >> 15);
                src += 2;
                dst += 4;
            }
            while(--width);
        }
    };


    //------------------------------------------------------------------------
    typedef color_conv_rgb555_rgba32<1,2,3,0> color_conv_rgb555_to_argb32; //----color_conv_rgb555_to_argb32
    typedef color_conv_rgb555_rgba32<3,2,1,0> color_conv_rgb555_to_abgr32; //----color_conv_rgb555_to_abgr32
    typedef color_conv_rgb555_rgba32<2,1,0,3> color_conv_rgb555_to_bgra32; //----color_conv_rgb555_to_bgra32
    typedef color_conv_rgb555_rgba32<0,1,2,3> color_conv_rgb555_to_rgba32; //----color_conv_rgb555_to_rgba32


    //------------------------------------------------color_conv_rgba32_rgb555
    template<int R, int G, int B, int A> class color_conv_rgba32_rgb555
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                *(int16u*)dst = (int16u)(((unsigned(src[R]) << 7) & 0x7C00) | 
                                         ((unsigned(src[G]) << 2) & 0x3E0)  |
                                         ((unsigned(src[B]) >> 3)) |
                                         ((unsigned(src[A]) << 8) & 0x8000));
                src += 4;
                dst += 2;
            }
            while(--width);
        }
    };


    //------------------------------------------------------------------------
    typedef color_conv_rgba32_rgb555<1,2,3,0> color_conv_argb32_to_rgb555; //----color_conv_argb32_to_rgb555
    typedef color_conv_rgba32_rgb555<3,2,1,0> color_conv_abgr32_to_rgb555; //----color_conv_abgr32_to_rgb555
    typedef color_conv_rgba32_rgb555<2,1,0,3> color_conv_bgra32_to_rgb555; //----color_conv_bgra32_to_rgb555
    typedef color_conv_rgba32_rgb555<0,1,2,3> color_conv_rgba32_to_rgb555; //----color_conv_rgba32_to_rgb555



    //------------------------------------------------color_conv_rgb565_rgba32
    template<int R, int G, int B, int A> class color_conv_rgb565_rgba32
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                int rgb = *(int16*)src;
                dst[R] = (rgb >> 8) & 0xF8;
                dst[G] = (rgb >> 3) & 0xFC;
                dst[B] = (rgb << 3) & 0xF8;
                dst[A] = 255;
                src += 2;
                dst += 4;
            }
            while(--width);
        }
    };


    //------------------------------------------------------------------------
    typedef color_conv_rgb565_rgba32<1,2,3,0> color_conv_rgb565_to_argb32; //----color_conv_rgb565_to_argb32
    typedef color_conv_rgb565_rgba32<3,2,1,0> color_conv_rgb565_to_abgr32; //----color_conv_rgb565_to_abgr32
    typedef color_conv_rgb565_rgba32<2,1,0,3> color_conv_rgb565_to_bgra32; //----color_conv_rgb565_to_bgra32
    typedef color_conv_rgb565_rgba32<0,1,2,3> color_conv_rgb565_to_rgba32; //----color_conv_rgb565_to_rgba32


    //------------------------------------------------color_conv_rgba32_rgb565
    template<int R, int G, int B> class color_conv_rgba32_rgb565
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                *(int16u*)dst = (int16u)(((unsigned(src[R]) << 8) & 0xF800) | 
                                         ((unsigned(src[G]) << 3) & 0x7E0)  |
                                         ((unsigned(src[B]) >> 3)));
                src += 4;
                dst += 2;
            }
            while(--width);
        }
    };


    //------------------------------------------------------------------------
    typedef color_conv_rgba32_rgb565<1,2,3> color_conv_argb32_to_rgb565; //----color_conv_argb32_to_rgb565
    typedef color_conv_rgba32_rgb565<3,2,1> color_conv_abgr32_to_rgb565; //----color_conv_abgr32_to_rgb565
    typedef color_conv_rgba32_rgb565<2,1,0> color_conv_bgra32_to_rgb565; //----color_conv_bgra32_to_rgb565
    typedef color_conv_rgba32_rgb565<0,1,2> color_conv_rgba32_to_rgb565; //----color_conv_rgba32_to_rgb565


    //---------------------------------------------color_conv_rgb555_to_rgb565
    class color_conv_rgb555_to_rgb565
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                unsigned rgb = *(int16u*)src;
                *(int16u*)dst = (int16u)(((rgb << 1) & 0xFFC0) | (rgb & 0x1F));
                src += 2;
                dst += 2;
            }
            while(--width);
        }
    };


    //----------------------------------------------color_conv_rgb565_to_rgb555
    class color_conv_rgb565_to_rgb555
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                unsigned rgb = *(int16u*)src;
                *(int16u*)dst = (int16u)(((rgb >> 1) & 0x7FE0) | (rgb & 0x1F));
                src += 2;
                dst += 2;
            }
            while(--width);
        }
    };


    //------------------------------------------------------------------------
    typedef color_conv_same<2> color_conv_rgb555_to_rgb555; //----color_conv_rgb555_to_rgb555
    typedef color_conv_same<2> color_conv_rgb565_to_rgb565; //----color_conv_rgb565_to_rgb565

    
    template<int R, int B> class color_conv_rgb24_gray8
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                *dst++ = (src[R]*77 + src[1]*150 + src[B]*29) >> 8;
                src += 3;
            }
            while(--width);
        }
    };

    typedef color_conv_rgb24_gray8<0,2> color_conv_rgb24_to_gray8; //----color_conv_rgb24_to_gray8
    typedef color_conv_rgb24_gray8<2,0> color_conv_bgr24_to_gray8; //----color_conv_bgr24_to_gray8


}



#endif
