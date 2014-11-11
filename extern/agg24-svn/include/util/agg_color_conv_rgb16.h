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
// This part of the library has been sponsored by 
// Liberty Technology Systems, Inc., visit http://lib-sys.com
//
// Liberty Technology Systems, Inc. is the provider of
// PostScript and PDF technology for software developers.
// 
//----------------------------------------------------------------------------
//
// A set of functors used with color_conv(). See file agg_color_conv.h
// These functors can convert images with up to 8 bits per component.
// Use convertors in the following way:
//
// agg::color_conv(dst, src, agg::color_conv_XXXX_to_YYYY());
//----------------------------------------------------------------------------

#ifndef AGG_COLOR_CONV_RGB16_INCLUDED
#define AGG_COLOR_CONV_RGB16_INCLUDED

#include "agg_basics.h"
#include "agg_color_conv.h"

namespace agg
{

    //-------------------------------------------------color_conv_gray16_to_gray8
    class color_conv_gray16_to_gray8
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            int16u* s = (int16u*)src;
            do
            {
                *dst++ = *s++ >> 8;
            }
            while(--width);
        }
    };


    //-----------------------------------------------------color_conv_rgb24_rgb48
    template<int I1, int I3> class color_conv_rgb24_rgb48
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            int16u* d = (int16u*)dst;
            do
            {
                *d++ = (src[I1] << 8) | src[I1];
                *d++ = (src[1]  << 8) | src[1] ;
                *d++ = (src[I3] << 8) | src[I3];
                src += 3;
            }
            while(--width);
        }
    };

    typedef color_conv_rgb24_rgb48<0,2> color_conv_rgb24_to_rgb48;
    typedef color_conv_rgb24_rgb48<0,2> color_conv_bgr24_to_bgr48;
    typedef color_conv_rgb24_rgb48<2,0> color_conv_rgb24_to_bgr48;
    typedef color_conv_rgb24_rgb48<2,0> color_conv_bgr24_to_rgb48;


    //-----------------------------------------------------color_conv_rgb24_rgb48
    template<int I1, int I3> class color_conv_rgb48_rgb24
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            const int16u* s = (const int16u*)src;
            do
            {
                *dst++ = s[I1] >> 8;
                *dst++ = s[1]  >> 8;
                *dst++ = s[I3] >> 8;
                s += 3;
            }
            while(--width);
        }
    };

    typedef color_conv_rgb48_rgb24<0,2> color_conv_rgb48_to_rgb24;
    typedef color_conv_rgb48_rgb24<0,2> color_conv_bgr48_to_bgr24;
    typedef color_conv_rgb48_rgb24<2,0> color_conv_rgb48_to_bgr24;
    typedef color_conv_rgb48_rgb24<2,0> color_conv_bgr48_to_rgb24;


    //----------------------------------------------color_conv_rgbAAA_rgb24
    template<int R, int B> class color_conv_rgbAAA_rgb24
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                int32u rgb = *(int32u*)src;
                dst[R] = int8u(rgb >> 22);
                dst[1] = int8u(rgb >> 12);
                dst[B] = int8u(rgb >> 2);
                src += 4;
                dst += 3;
            }
            while(--width);
        }
    };

    typedef color_conv_rgbAAA_rgb24<0,2> color_conv_rgbAAA_to_rgb24;
    typedef color_conv_rgbAAA_rgb24<2,0> color_conv_rgbAAA_to_bgr24;
    typedef color_conv_rgbAAA_rgb24<2,0> color_conv_bgrAAA_to_rgb24;
    typedef color_conv_rgbAAA_rgb24<0,2> color_conv_bgrAAA_to_bgr24;


    //----------------------------------------------color_conv_rgbBBA_rgb24
    template<int R, int B> class color_conv_rgbBBA_rgb24
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                int32u rgb = *(int32u*)src;
                dst[R] = int8u(rgb >> 24);
                dst[1] = int8u(rgb >> 13);
                dst[B] = int8u(rgb >> 2);
                src += 4;
                dst += 3;
            }
            while(--width);
        }
    };

    typedef color_conv_rgbBBA_rgb24<0,2> color_conv_rgbBBA_to_rgb24;
    typedef color_conv_rgbBBA_rgb24<2,0> color_conv_rgbBBA_to_bgr24;


    //----------------------------------------------color_conv_bgrABB_rgb24
    template<int B, int R> class color_conv_bgrABB_rgb24
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                int32u bgr = *(int32u*)src;
                dst[R] = int8u(bgr >> 3);
                dst[1] = int8u(bgr >> 14);
                dst[B] = int8u(bgr >> 24);
                src += 4;
                dst += 3;
            }
            while(--width);
        }
    };

    typedef color_conv_bgrABB_rgb24<2,0> color_conv_bgrABB_to_rgb24;
    typedef color_conv_bgrABB_rgb24<0,2> color_conv_bgrABB_to_bgr24;


    //-------------------------------------------------color_conv_rgba64_rgba32
    template<int I1, int I2, int I3, int I4> class color_conv_rgba64_rgba32
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            do
            {
                *dst++ = int8u(((int16u*)src)[I1] >> 8);
                *dst++ = int8u(((int16u*)src)[I2] >> 8);
                *dst++ = int8u(((int16u*)src)[I3] >> 8);
                *dst++ = int8u(((int16u*)src)[I4] >> 8); 
                src += 8;
            }
            while(--width);
        }
    };

    //------------------------------------------------------------------------
    typedef color_conv_rgba64_rgba32<0,1,2,3> color_conv_rgba64_to_rgba32; //----color_conv_rgba64_to_rgba32
    typedef color_conv_rgba64_rgba32<0,1,2,3> color_conv_argb64_to_argb32; //----color_conv_argb64_to_argb32
    typedef color_conv_rgba64_rgba32<0,1,2,3> color_conv_bgra64_to_bgra32; //----color_conv_bgra64_to_bgra32
    typedef color_conv_rgba64_rgba32<0,1,2,3> color_conv_abgr64_to_abgr32; //----color_conv_abgr64_to_abgr32
    typedef color_conv_rgba64_rgba32<0,3,2,1> color_conv_argb64_to_abgr32; //----color_conv_argb64_to_abgr32
    typedef color_conv_rgba64_rgba32<3,2,1,0> color_conv_argb64_to_bgra32; //----color_conv_argb64_to_bgra32
    typedef color_conv_rgba64_rgba32<1,2,3,0> color_conv_argb64_to_rgba32; //----color_conv_argb64_to_rgba32
    typedef color_conv_rgba64_rgba32<3,0,1,2> color_conv_bgra64_to_abgr32; //----color_conv_bgra64_to_abgr32
    typedef color_conv_rgba64_rgba32<3,2,1,0> color_conv_bgra64_to_argb32; //----color_conv_bgra64_to_argb32
    typedef color_conv_rgba64_rgba32<2,1,0,3> color_conv_bgra64_to_rgba32; //----color_conv_bgra64_to_rgba32
    typedef color_conv_rgba64_rgba32<3,2,1,0> color_conv_rgba64_to_abgr32; //----color_conv_rgba64_to_abgr32
    typedef color_conv_rgba64_rgba32<3,0,1,2> color_conv_rgba64_to_argb32; //----color_conv_rgba64_to_argb32
    typedef color_conv_rgba64_rgba32<2,1,0,3> color_conv_rgba64_to_bgra32; //----color_conv_rgba64_to_bgra32
    typedef color_conv_rgba64_rgba32<0,3,2,1> color_conv_abgr64_to_argb32; //----color_conv_abgr64_to_argb32
    typedef color_conv_rgba64_rgba32<1,2,3,0> color_conv_abgr64_to_bgra32; //----color_conv_abgr64_to_bgra32
    typedef color_conv_rgba64_rgba32<3,2,1,0> color_conv_abgr64_to_rgba32; //----color_conv_abgr64_to_rgba32



    //--------------------------------------------color_conv_rgb24_rgba64
    template<int I1, int I2, int I3, int A> class color_conv_rgb24_rgba64
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            int16u* d = (int16u*)dst;
            do
            {
                d[I1] = (src[0] << 8) | src[0];
                d[I2] = (src[1] << 8) | src[1];
                d[I3] = (src[2] << 8) | src[2];
                d[A]  = 65535; 
                d   += 4;
                src += 3;
            }
            while(--width);
        }
    };


    //------------------------------------------------------------------------
    typedef color_conv_rgb24_rgba64<1,2,3,0> color_conv_rgb24_to_argb64; //----color_conv_rgb24_to_argb64
    typedef color_conv_rgb24_rgba64<3,2,1,0> color_conv_rgb24_to_abgr64; //----color_conv_rgb24_to_abgr64
    typedef color_conv_rgb24_rgba64<2,1,0,3> color_conv_rgb24_to_bgra64; //----color_conv_rgb24_to_bgra64
    typedef color_conv_rgb24_rgba64<0,1,2,3> color_conv_rgb24_to_rgba64; //----color_conv_rgb24_to_rgba64
    typedef color_conv_rgb24_rgba64<3,2,1,0> color_conv_bgr24_to_argb64; //----color_conv_bgr24_to_argb64
    typedef color_conv_rgb24_rgba64<1,2,3,0> color_conv_bgr24_to_abgr64; //----color_conv_bgr24_to_abgr64
    typedef color_conv_rgb24_rgba64<0,1,2,3> color_conv_bgr24_to_bgra64; //----color_conv_bgr24_to_bgra64
    typedef color_conv_rgb24_rgba64<2,1,0,3> color_conv_bgr24_to_rgba64; //----color_conv_bgr24_to_rgba64


    template<int R, int B> class color_conv_rgb24_gray16
    {
    public:
        void operator () (int8u* dst, 
                          const int8u* src,
                          unsigned width) const
        {
            int16u* d = (int16u*)dst;
            do
            {
                *d++ = src[R]*77 + src[1]*150 + src[B]*29;
                src += 3;
            }
            while(--width);
        }
    };

    typedef color_conv_rgb24_gray16<0,2> color_conv_rgb24_to_gray16;
    typedef color_conv_rgb24_gray16<2,0> color_conv_bgr24_to_gray16;


}


#endif
