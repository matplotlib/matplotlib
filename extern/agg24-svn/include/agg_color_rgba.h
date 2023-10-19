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
//
// Adaptation for high precision colors has been sponsored by
// Liberty Technology Systems, Inc., visit http://lib-sys.com
//
// Liberty Technology Systems, Inc. is the provider of
// PostScript and PDF technology for software developers.
//
//----------------------------------------------------------------------------
// Contact: mcseem@antigrain.com
//          mcseemagg@yahoo.com
//          http://www.antigrain.com
//----------------------------------------------------------------------------

#ifndef AGG_COLOR_RGBA_INCLUDED
#define AGG_COLOR_RGBA_INCLUDED

#include <math.h>
#include "agg_basics.h"
#include "agg_gamma_lut.h"

namespace agg
{
    // Supported component orders for RGB and RGBA pixel formats
    //=======================================================================
    struct order_rgb  { enum rgb_e  { R=0, G=1, B=2, N=3 }; };
    struct order_bgr  { enum bgr_e  { B=0, G=1, R=2, N=3 }; };
    struct order_rgba { enum rgba_e { R=0, G=1, B=2, A=3, N=4 }; };
    struct order_argb { enum argb_e { A=0, R=1, G=2, B=3, N=4 }; };
    struct order_abgr { enum abgr_e { A=0, B=1, G=2, R=3, N=4 }; };
    struct order_bgra { enum bgra_e { B=0, G=1, R=2, A=3, N=4 }; };

    // Colorspace tag types.
    struct linear {};
    struct sRGB {};

    //====================================================================rgba
    struct rgba
    {
        typedef double value_type;

        double r;
        double g;
        double b;
        double a;

        //--------------------------------------------------------------------
        rgba() {}

        //--------------------------------------------------------------------
        rgba(double r_, double g_, double b_, double a_=1.0) :
            r(r_), g(g_), b(b_), a(a_) {}

        //--------------------------------------------------------------------
        rgba(const rgba& c, double a_) : r(c.r), g(c.g), b(c.b), a(a_) {}

        //--------------------------------------------------------------------
        rgba& clear()
        {
            r = g = b = a = 0;
			return *this;
        }

        //--------------------------------------------------------------------
        rgba& transparent()
        {
            a = 0;
            return *this;
        }

        //--------------------------------------------------------------------
        rgba& opacity(double a_)
        {
            if (a_ < 0) a = 0;
            else if (a_ > 1) a = 1;
            else a = a_;
            return *this;
        }

        //--------------------------------------------------------------------
        double opacity() const
        {
            return a;
        }

        //--------------------------------------------------------------------
        rgba& premultiply()
        {
            r *= a;
            g *= a;
            b *= a;
            return *this;
        }

        //--------------------------------------------------------------------
        rgba& premultiply(double a_)
        {
            if (a <= 0 || a_ <= 0)
            {
                r = g = b = a = 0;
            }
            else
            {
                a_ /= a;
                r *= a_;
                g *= a_;
                b *= a_;
                a  = a_;
            }
            return *this;
        }

        //--------------------------------------------------------------------
        rgba& demultiply()
        {
            if (a == 0)
            {
                r = g = b = 0;
            }
            else
            {
                double a_ = 1.0 / a;
                r *= a_;
                g *= a_;
                b *= a_;
            }
            return *this;
        }


        //--------------------------------------------------------------------
        rgba gradient(rgba c, double k) const
        {
            rgba ret;
            ret.r = r + (c.r - r) * k;
            ret.g = g + (c.g - g) * k;
            ret.b = b + (c.b - b) * k;
            ret.a = a + (c.a - a) * k;
            return ret;
        }

        rgba& operator+=(const rgba& c)
        {
            r += c.r;
            g += c.g;
            b += c.b;
            a += c.a;
            return *this;
        }

        rgba& operator*=(double k)
        {
            r *= k;
            g *= k;
            b *= k;
            a *= k;
            return *this;
        }

        //--------------------------------------------------------------------
        static rgba no_color() { return rgba(0,0,0,0); }

        //--------------------------------------------------------------------
        static rgba from_wavelength(double wl, double gamma = 1.0);

        //--------------------------------------------------------------------
        explicit rgba(double wavelen, double gamma=1.0)
        {
            *this = from_wavelength(wavelen, gamma);
        }

    };

    inline rgba operator+(const rgba& a, const rgba& b)
    {
        return rgba(a) += b;
    }

    inline rgba operator*(const rgba& a, double b)
    {
        return rgba(a) *= b;
    }

    //------------------------------------------------------------------------
    inline rgba rgba::from_wavelength(double wl, double gamma)
    {
        rgba t(0.0, 0.0, 0.0);

        if (wl >= 380.0 && wl <= 440.0)
        {
            t.r = -1.0 * (wl - 440.0) / (440.0 - 380.0);
            t.b = 1.0;
        }
        else if (wl >= 440.0 && wl <= 490.0)
        {
            t.g = (wl - 440.0) / (490.0 - 440.0);
            t.b = 1.0;
        }
        else if (wl >= 490.0 && wl <= 510.0)
        {
            t.g = 1.0;
            t.b = -1.0 * (wl - 510.0) / (510.0 - 490.0);
        }
        else if (wl >= 510.0 && wl <= 580.0)
        {
            t.r = (wl - 510.0) / (580.0 - 510.0);
            t.g = 1.0;
        }
        else if (wl >= 580.0 && wl <= 645.0)
        {
            t.r = 1.0;
            t.g = -1.0 * (wl - 645.0) / (645.0 - 580.0);
        }
        else if (wl >= 645.0 && wl <= 780.0)
        {
            t.r = 1.0;
        }

        double s = 1.0;
        if (wl > 700.0)       s = 0.3 + 0.7 * (780.0 - wl) / (780.0 - 700.0);
        else if (wl <  420.0) s = 0.3 + 0.7 * (wl - 380.0) / (420.0 - 380.0);

        t.r = pow(t.r * s, gamma);
        t.g = pow(t.g * s, gamma);
        t.b = pow(t.b * s, gamma);
        return t;
    }

    inline rgba rgba_pre(double r, double g, double b, double a)
    {
        return rgba(r, g, b, a).premultiply();
    }


    //===================================================================rgba8
    template<class Colorspace>
    struct rgba8T
    {
        typedef int8u  value_type;
        typedef int32u calc_type;
        typedef int32  long_type;
        enum base_scale_e
        {
            base_shift = 8,
            base_scale = 1 << base_shift,
            base_mask  = base_scale - 1,
            base_MSB = 1 << (base_shift - 1)
        };
        typedef rgba8T self_type;


        value_type r;
        value_type g;
        value_type b;
        value_type a;

        static void convert(rgba8T<linear>& dst, const rgba8T<sRGB>& src)
        {
            dst.r = sRGB_conv<value_type>::rgb_from_sRGB(src.r);
            dst.g = sRGB_conv<value_type>::rgb_from_sRGB(src.g);
            dst.b = sRGB_conv<value_type>::rgb_from_sRGB(src.b);
            dst.a = src.a;
        }

        static void convert(rgba8T<sRGB>& dst, const rgba8T<linear>& src)
        {
            dst.r = sRGB_conv<value_type>::rgb_to_sRGB(src.r);
            dst.g = sRGB_conv<value_type>::rgb_to_sRGB(src.g);
            dst.b = sRGB_conv<value_type>::rgb_to_sRGB(src.b);
            dst.a = src.a;
        }

        static void convert(rgba8T<linear>& dst, const rgba& src)
        {
            dst.r = value_type(uround(src.r * base_mask));
            dst.g = value_type(uround(src.g * base_mask));
            dst.b = value_type(uround(src.b * base_mask));
            dst.a = value_type(uround(src.a * base_mask));
        }

        static void convert(rgba8T<sRGB>& dst, const rgba& src)
        {
            // Use the "float" table.
            dst.r = sRGB_conv<float>::rgb_to_sRGB(float(src.r));
            dst.g = sRGB_conv<float>::rgb_to_sRGB(float(src.g));
            dst.b = sRGB_conv<float>::rgb_to_sRGB(float(src.b));
            dst.a = sRGB_conv<float>::alpha_to_sRGB(float(src.a));
        }

        static void convert(rgba& dst, const rgba8T<linear>& src)
        {
            dst.r = src.r / 255.0;
            dst.g = src.g / 255.0;
            dst.b = src.b / 255.0;
            dst.a = src.a / 255.0;
        }

        static void convert(rgba& dst, const rgba8T<sRGB>& src)
        {
            // Use the "float" table.
            dst.r = sRGB_conv<float>::rgb_from_sRGB(src.r);
            dst.g = sRGB_conv<float>::rgb_from_sRGB(src.g);
            dst.b = sRGB_conv<float>::rgb_from_sRGB(src.b);
            dst.a = sRGB_conv<float>::alpha_from_sRGB(src.a);
        }

        //--------------------------------------------------------------------
        rgba8T() {}

        //--------------------------------------------------------------------
        rgba8T(unsigned r_, unsigned g_, unsigned b_, unsigned a_ = base_mask) :
            r(value_type(r_)),
            g(value_type(g_)),
            b(value_type(b_)),
            a(value_type(a_)) {}

        //--------------------------------------------------------------------
        rgba8T(const rgba& c)
        {
            convert(*this, c);
        }

        //--------------------------------------------------------------------
        rgba8T(const self_type& c, unsigned a_) :
            r(c.r), g(c.g), b(c.b), a(value_type(a_)) {}

        //--------------------------------------------------------------------
        template<class T>
        rgba8T(const rgba8T<T>& c)
        {
            convert(*this, c);
        }

        //--------------------------------------------------------------------
        operator rgba() const
        {
            rgba c;
            convert(c, *this);
            return c;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE double to_double(value_type a)
        {
            return double(a) / base_mask;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type from_double(double a)
        {
            return value_type(uround(a * base_mask));
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type empty_value()
        {
            return 0;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type full_value()
        {
            return base_mask;
        }

        //--------------------------------------------------------------------
        AGG_INLINE bool is_transparent() const
        {
            return a == 0;
        }

        //--------------------------------------------------------------------
        AGG_INLINE bool is_opaque() const
        {
            return a == base_mask;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type invert(value_type x)
        {
            return base_mask - x;
        }

        //--------------------------------------------------------------------
        // Fixed-point multiply, exact over int8u.
        static AGG_INLINE value_type multiply(value_type a, value_type b)
        {
            calc_type t = a * b + base_MSB;
            return value_type(((t >> base_shift) + t) >> base_shift);
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type demultiply(value_type a, value_type b)
        {
            if (a * b == 0)
            {
                return 0;
            }
            else if (a >= b)
            {
                return base_mask;
            }
            else return value_type((a * base_mask + (b >> 1)) / b);
        }

        //--------------------------------------------------------------------
        template<typename T>
        static AGG_INLINE T downscale(T a)
        {
            return a >> base_shift;
        }

        //--------------------------------------------------------------------
        template<typename T>
        static AGG_INLINE T downshift(T a, unsigned n)
        {
            return a >> n;
        }

        //--------------------------------------------------------------------
        // Fixed-point multiply, exact over int8u.
        // Specifically for multiplying a color component by a cover.
        static AGG_INLINE value_type mult_cover(value_type a, cover_type b)
        {
            return multiply(a, b);
        }

        //--------------------------------------------------------------------
        static AGG_INLINE cover_type scale_cover(cover_type a, value_type b)
        {
            return multiply(b, a);
        }

        //--------------------------------------------------------------------
        // Interpolate p to q by a, assuming q is premultiplied by a.
        static AGG_INLINE value_type prelerp(value_type p, value_type q, value_type a)
        {
            return p + q - multiply(p, a);
        }

        //--------------------------------------------------------------------
        // Interpolate p to q by a.
        static AGG_INLINE value_type lerp(value_type p, value_type q, value_type a)
        {
            int t = (q - p) * a + base_MSB - (p > q);
            return value_type(p + (((t >> base_shift) + t) >> base_shift));
        }

        //--------------------------------------------------------------------
        self_type& clear()
        {
            r = g = b = a = 0;
			return *this;
        }

        //--------------------------------------------------------------------
        self_type& transparent()
        {
            a = 0;
            return *this;
        }

        //--------------------------------------------------------------------
        self_type& opacity(double a_)
        {
            if (a_ < 0) a = 0;
            else if (a_ > 1) a = 1;
            else a = (value_type)uround(a_ * double(base_mask));
            return *this;
        }

        //--------------------------------------------------------------------
        double opacity() const
        {
            return double(a) / double(base_mask);
        }

        //--------------------------------------------------------------------
        AGG_INLINE self_type& premultiply()
        {
            if (a != base_mask)
            {
                if (a == 0)
                {
                    r = g = b = 0;
                }
                else
                {
                    r = multiply(r, a);
                    g = multiply(g, a);
                    b = multiply(b, a);
                }
            }
            return *this;
        }

        //--------------------------------------------------------------------
        AGG_INLINE self_type& premultiply(unsigned a_)
        {
            if (a != base_mask || a_ < base_mask)
            {
                if (a == 0 || a_ == 0)
                {
                    r = g = b = a = 0;
                }
                else
                {
                    calc_type r_ = (calc_type(r) * a_) / a;
                    calc_type g_ = (calc_type(g) * a_) / a;
                    calc_type b_ = (calc_type(b) * a_) / a;
                    r = value_type((r_ > a_) ? a_ : r_);
                    g = value_type((g_ > a_) ? a_ : g_);
                    b = value_type((b_ > a_) ? a_ : b_);
                    a = value_type(a_);
                }
            }
            return *this;
        }

        //--------------------------------------------------------------------
        AGG_INLINE self_type& demultiply()
        {
            if (a < base_mask)
            {
                if (a == 0)
                {
                    r = g = b = 0;
                }
                else
                {
                    calc_type r_ = (calc_type(r) * base_mask) / a;
                    calc_type g_ = (calc_type(g) * base_mask) / a;
                    calc_type b_ = (calc_type(b) * base_mask) / a;
                    r = value_type((r_ > calc_type(base_mask)) ? calc_type(base_mask) : r_);
                    g = value_type((g_ > calc_type(base_mask)) ? calc_type(base_mask) : g_);
                    b = value_type((b_ > calc_type(base_mask)) ? calc_type(base_mask) : b_);
                }
            }
            return *this;
        }

        //--------------------------------------------------------------------
        AGG_INLINE self_type gradient(const self_type& c, double k) const
        {
            self_type ret;
            calc_type ik = uround(k * base_mask);
            ret.r = lerp(r, c.r, ik);
            ret.g = lerp(g, c.g, ik);
            ret.b = lerp(b, c.b, ik);
            ret.a = lerp(a, c.a, ik);
            return ret;
        }

        //--------------------------------------------------------------------
        AGG_INLINE void add(const self_type& c, unsigned cover)
        {
            calc_type cr, cg, cb, ca;
            if (cover == cover_mask)
            {
                if (c.a == base_mask)
                {
                    *this = c;
                    return;
                }
                else
                {
                    cr = r + c.r;
                    cg = g + c.g;
                    cb = b + c.b;
                    ca = a + c.a;
                }
            }
            else
            {
                cr = r + mult_cover(c.r, cover);
                cg = g + mult_cover(c.g, cover);
                cb = b + mult_cover(c.b, cover);
                ca = a + mult_cover(c.a, cover);
            }
            r = (value_type)((cr > calc_type(base_mask)) ? calc_type(base_mask) : cr);
            g = (value_type)((cg > calc_type(base_mask)) ? calc_type(base_mask) : cg);
            b = (value_type)((cb > calc_type(base_mask)) ? calc_type(base_mask) : cb);
            a = (value_type)((ca > calc_type(base_mask)) ? calc_type(base_mask) : ca);
        }

        //--------------------------------------------------------------------
        template<class GammaLUT>
        AGG_INLINE void apply_gamma_dir(const GammaLUT& gamma)
        {
            r = gamma.dir(r);
            g = gamma.dir(g);
            b = gamma.dir(b);
        }

        //--------------------------------------------------------------------
        template<class GammaLUT>
        AGG_INLINE void apply_gamma_inv(const GammaLUT& gamma)
        {
            r = gamma.inv(r);
            g = gamma.inv(g);
            b = gamma.inv(b);
        }

        //--------------------------------------------------------------------
        static self_type no_color() { return self_type(0,0,0,0); }

        //--------------------------------------------------------------------
        static self_type from_wavelength(double wl, double gamma = 1.0)
        {
            return self_type(rgba::from_wavelength(wl, gamma));
        }
    };

    typedef rgba8T<linear> rgba8;
    typedef rgba8T<sRGB> srgba8;


    //-------------------------------------------------------------rgb8_packed
    inline rgba8 rgb8_packed(unsigned v)
    {
        return rgba8((v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF);
    }

    //-------------------------------------------------------------bgr8_packed
    inline rgba8 bgr8_packed(unsigned v)
    {
        return rgba8(v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF);
    }

    //------------------------------------------------------------argb8_packed
    inline rgba8 argb8_packed(unsigned v)
    {
        return rgba8((v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF, v >> 24);
    }

    //---------------------------------------------------------rgba8_gamma_dir
    template<class GammaLUT>
    rgba8 rgba8_gamma_dir(rgba8 c, const GammaLUT& gamma)
    {
        return rgba8(gamma.dir(c.r), gamma.dir(c.g), gamma.dir(c.b), c.a);
    }

    //---------------------------------------------------------rgba8_gamma_inv
    template<class GammaLUT>
    rgba8 rgba8_gamma_inv(rgba8 c, const GammaLUT& gamma)
    {
        return rgba8(gamma.inv(c.r), gamma.inv(c.g), gamma.inv(c.b), c.a);
    }



    //==================================================================rgba16
    struct rgba16
    {
        typedef int16u value_type;
        typedef int32u calc_type;
        typedef int64  long_type;
        enum base_scale_e
        {
            base_shift = 16,
            base_scale = 1 << base_shift,
            base_mask  = base_scale - 1,
            base_MSB = 1 << (base_shift - 1)
        };
        typedef rgba16 self_type;

        value_type r;
        value_type g;
        value_type b;
        value_type a;

        //--------------------------------------------------------------------
        rgba16() {}

        //--------------------------------------------------------------------
        rgba16(unsigned r_, unsigned g_, unsigned b_, unsigned a_=base_mask) :
            r(value_type(r_)),
            g(value_type(g_)),
            b(value_type(b_)),
            a(value_type(a_)) {}

        //--------------------------------------------------------------------
        rgba16(const self_type& c, unsigned a_) :
            r(c.r), g(c.g), b(c.b), a(value_type(a_)) {}

        //--------------------------------------------------------------------
        rgba16(const rgba& c) :
            r((value_type)uround(c.r * double(base_mask))),
            g((value_type)uround(c.g * double(base_mask))),
            b((value_type)uround(c.b * double(base_mask))),
            a((value_type)uround(c.a * double(base_mask))) {}

        //--------------------------------------------------------------------
        rgba16(const rgba8& c) :
            r(value_type((value_type(c.r) << 8) | c.r)),
            g(value_type((value_type(c.g) << 8) | c.g)),
            b(value_type((value_type(c.b) << 8) | c.b)),
            a(value_type((value_type(c.a) << 8) | c.a)) {}

        //--------------------------------------------------------------------
        rgba16(const srgba8& c) :
            r(sRGB_conv<value_type>::rgb_from_sRGB(c.r)),
            g(sRGB_conv<value_type>::rgb_from_sRGB(c.g)),
            b(sRGB_conv<value_type>::rgb_from_sRGB(c.b)),
            a(sRGB_conv<value_type>::alpha_from_sRGB(c.a)) {}

        //--------------------------------------------------------------------
        operator rgba() const
        {
            return rgba(
                r / 65535.0,
                g / 65535.0,
                b / 65535.0,
                a / 65535.0);
        }

        //--------------------------------------------------------------------
        operator rgba8() const
        {
            return rgba8(r >> 8, g >> 8, b >> 8, a >> 8);
        }

        //--------------------------------------------------------------------
        operator srgba8() const
        {
            // Return (non-premultiplied) sRGB values.
            return srgba8(
                sRGB_conv<value_type>::rgb_to_sRGB(r),
                sRGB_conv<value_type>::rgb_to_sRGB(g),
                sRGB_conv<value_type>::rgb_to_sRGB(b),
                sRGB_conv<value_type>::alpha_to_sRGB(a));
        }

        //--------------------------------------------------------------------
        static AGG_INLINE double to_double(value_type a)
        {
            return double(a) / base_mask;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type from_double(double a)
        {
            return value_type(uround(a * base_mask));
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type empty_value()
        {
            return 0;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type full_value()
        {
            return base_mask;
        }

        //--------------------------------------------------------------------
        AGG_INLINE bool is_transparent() const
        {
            return a == 0;
        }

        //--------------------------------------------------------------------
        AGG_INLINE bool is_opaque() const
        {
            return a == base_mask;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type invert(value_type x)
        {
            return base_mask - x;
        }

        //--------------------------------------------------------------------
        // Fixed-point multiply, exact over int16u.
        static AGG_INLINE value_type multiply(value_type a, value_type b)
        {
            calc_type t = a * b + base_MSB;
            return value_type(((t >> base_shift) + t) >> base_shift);
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type demultiply(value_type a, value_type b)
        {
            if (a * b == 0)
            {
                return 0;
            }
            else if (a >= b)
            {
                return base_mask;
            }
            else return value_type((a * base_mask + (b >> 1)) / b);
        }

        //--------------------------------------------------------------------
        template<typename T>
        static AGG_INLINE T downscale(T a)
        {
            return a >> base_shift;
        }

        //--------------------------------------------------------------------
        template<typename T>
        static AGG_INLINE T downshift(T a, unsigned n)
        {
            return a >> n;
        }

        //--------------------------------------------------------------------
        // Fixed-point multiply, almost exact over int16u.
        // Specifically for multiplying a color component by a cover.
        static AGG_INLINE value_type mult_cover(value_type a, cover_type b)
        {
            return multiply(a, (b << 8) | b);
        }

        //--------------------------------------------------------------------
        static AGG_INLINE cover_type scale_cover(cover_type a, value_type b)
        {
            return multiply((a << 8) | a, b) >> 8;
        }

        //--------------------------------------------------------------------
        // Interpolate p to q by a, assuming q is premultiplied by a.
        static AGG_INLINE value_type prelerp(value_type p, value_type q, value_type a)
        {
            return p + q - multiply(p, a);
        }

        //--------------------------------------------------------------------
        // Interpolate p to q by a.
        static AGG_INLINE value_type lerp(value_type p, value_type q, value_type a)
        {
            int t = (q - p) * a + base_MSB - (p > q);
            return value_type(p + (((t >> base_shift) + t) >> base_shift));
        }

        //--------------------------------------------------------------------
        self_type& clear()
        {
            r = g = b = a = 0;
			return *this;
        }

        //--------------------------------------------------------------------
        self_type& transparent()
        {
            a = 0;
            return *this;
        }

        //--------------------------------------------------------------------
        AGG_INLINE self_type& opacity(double a_)
        {
            if (a_ < 0) a = 0;
            if (a_ > 1) a = 1;
            a = value_type(uround(a_ * double(base_mask)));
            return *this;
        }

        //--------------------------------------------------------------------
        double opacity() const
        {
            return double(a) / double(base_mask);
        }

        //--------------------------------------------------------------------
        AGG_INLINE self_type& premultiply()
        {
            if (a != base_mask)
            {
                if (a == 0)
                {
                    r = g = b = 0;
                }
                else
                {
                    r = multiply(r, a);
                    g = multiply(g, a);
                    b = multiply(b, a);
                }
            }
            return *this;
        }

        //--------------------------------------------------------------------
        AGG_INLINE self_type& premultiply(unsigned a_)
        {
            if (a < base_mask || a_ < base_mask)
            {
                if (a == 0 || a_ == 0)
                {
                    r = g = b = a = 0;
                }
                else
                {
                    calc_type r_ = (calc_type(r) * a_) / a;
                    calc_type g_ = (calc_type(g) * a_) / a;
                    calc_type b_ = (calc_type(b) * a_) / a;
                    r = value_type((r_ > a_) ? a_ : r_);
                    g = value_type((g_ > a_) ? a_ : g_);
                    b = value_type((b_ > a_) ? a_ : b_);
                    a = value_type(a_);
                }
            }
            return *this;
        }

        //--------------------------------------------------------------------
        AGG_INLINE self_type& demultiply()
        {
            if (a < base_mask)
            {
                if (a == 0)
                {
                    r = g = b = 0;
                }
                else
                {
                    calc_type r_ = (calc_type(r) * base_mask) / a;
                    calc_type g_ = (calc_type(g) * base_mask) / a;
                    calc_type b_ = (calc_type(b) * base_mask) / a;
                    r = value_type((r_ > calc_type(base_mask)) ? calc_type(base_mask) : r_);
                    g = value_type((g_ > calc_type(base_mask)) ? calc_type(base_mask) : g_);
                    b = value_type((b_ > calc_type(base_mask)) ? calc_type(base_mask) : b_);
                }
            }
            return *this;
        }

        //--------------------------------------------------------------------
        AGG_INLINE self_type gradient(const self_type& c, double k) const
        {
            self_type ret;
            calc_type ik = uround(k * base_mask);
            ret.r = lerp(r, c.r, ik);
            ret.g = lerp(g, c.g, ik);
            ret.b = lerp(b, c.b, ik);
            ret.a = lerp(a, c.a, ik);
            return ret;
        }

        //--------------------------------------------------------------------
        AGG_INLINE void add(const self_type& c, unsigned cover)
        {
            calc_type cr, cg, cb, ca;
            if (cover == cover_mask)
            {
                if (c.a == base_mask)
                {
                    *this = c;
                    return;
                }
                else
                {
                    cr = r + c.r;
                    cg = g + c.g;
                    cb = b + c.b;
                    ca = a + c.a;
                }
            }
            else
            {
                cr = r + mult_cover(c.r, cover);
                cg = g + mult_cover(c.g, cover);
                cb = b + mult_cover(c.b, cover);
                ca = a + mult_cover(c.a, cover);
            }
            r = (value_type)((cr > calc_type(base_mask)) ? calc_type(base_mask) : cr);
            g = (value_type)((cg > calc_type(base_mask)) ? calc_type(base_mask) : cg);
            b = (value_type)((cb > calc_type(base_mask)) ? calc_type(base_mask) : cb);
            a = (value_type)((ca > calc_type(base_mask)) ? calc_type(base_mask) : ca);
        }

        //--------------------------------------------------------------------
        template<class GammaLUT>
        AGG_INLINE void apply_gamma_dir(const GammaLUT& gamma)
        {
            r = gamma.dir(r);
            g = gamma.dir(g);
            b = gamma.dir(b);
        }

        //--------------------------------------------------------------------
        template<class GammaLUT>
        AGG_INLINE void apply_gamma_inv(const GammaLUT& gamma)
        {
            r = gamma.inv(r);
            g = gamma.inv(g);
            b = gamma.inv(b);
        }

        //--------------------------------------------------------------------
        static self_type no_color() { return self_type(0,0,0,0); }

        //--------------------------------------------------------------------
        static self_type from_wavelength(double wl, double gamma = 1.0)
        {
            return self_type(rgba::from_wavelength(wl, gamma));
        }
    };


    //------------------------------------------------------rgba16_gamma_dir
    template<class GammaLUT>
    rgba16 rgba16_gamma_dir(rgba16 c, const GammaLUT& gamma)
    {
        return rgba16(gamma.dir(c.r), gamma.dir(c.g), gamma.dir(c.b), c.a);
    }

    //------------------------------------------------------rgba16_gamma_inv
    template<class GammaLUT>
    rgba16 rgba16_gamma_inv(rgba16 c, const GammaLUT& gamma)
    {
        return rgba16(gamma.inv(c.r), gamma.inv(c.g), gamma.inv(c.b), c.a);
    }

    //====================================================================rgba32
    struct rgba32
    {
        typedef float value_type;
        typedef double calc_type;
        typedef double long_type;
        typedef rgba32 self_type;

        value_type r;
        value_type g;
        value_type b;
        value_type a;

        //--------------------------------------------------------------------
        rgba32() {}

        //--------------------------------------------------------------------
        rgba32(value_type r_, value_type g_, value_type b_, value_type a_= 1) :
            r(r_), g(g_), b(b_), a(a_) {}

        //--------------------------------------------------------------------
        rgba32(const self_type& c, float a_) :
            r(c.r), g(c.g), b(c.b), a(a_) {}

        //--------------------------------------------------------------------
        rgba32(const rgba& c) :
            r(value_type(c.r)), g(value_type(c.g)), b(value_type(c.b)), a(value_type(c.a)) {}

        //--------------------------------------------------------------------
        rgba32(const rgba8& c) :
            r(value_type(c.r / 255.0)),
            g(value_type(c.g / 255.0)),
            b(value_type(c.b / 255.0)),
            a(value_type(c.a / 255.0)) {}

        //--------------------------------------------------------------------
        rgba32(const srgba8& c) :
            r(sRGB_conv<value_type>::rgb_from_sRGB(c.r)),
            g(sRGB_conv<value_type>::rgb_from_sRGB(c.g)),
            b(sRGB_conv<value_type>::rgb_from_sRGB(c.b)),
            a(sRGB_conv<value_type>::alpha_from_sRGB(c.a)) {}

        //--------------------------------------------------------------------
        rgba32(const rgba16& c) :
            r(value_type(c.r / 65535.0)),
            g(value_type(c.g / 65535.0)),
            b(value_type(c.b / 65535.0)),
            a(value_type(c.a / 65535.0)) {}

        //--------------------------------------------------------------------
        operator rgba() const
        {
            return rgba(r, g, b, a);
        }

        //--------------------------------------------------------------------
        operator rgba8() const
        {
            return rgba8(
                uround(r * 255.0),
                uround(g * 255.0),
                uround(b * 255.0),
                uround(a * 255.0));
        }

        //--------------------------------------------------------------------
        operator srgba8() const
        {
            return srgba8(
                sRGB_conv<value_type>::rgb_to_sRGB(r),
                sRGB_conv<value_type>::rgb_to_sRGB(g),
                sRGB_conv<value_type>::rgb_to_sRGB(b),
                sRGB_conv<value_type>::alpha_to_sRGB(a));
        }

        //--------------------------------------------------------------------
        operator rgba16() const
        {
            return rgba8(
                uround(r * 65535.0),
                uround(g * 65535.0),
                uround(b * 65535.0),
                uround(a * 65535.0));
        }

        //--------------------------------------------------------------------
        static AGG_INLINE double to_double(value_type a)
        {
            return a;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type from_double(double a)
        {
            return value_type(a);
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type empty_value()
        {
            return 0;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type full_value()
        {
            return 1;
        }

        //--------------------------------------------------------------------
        AGG_INLINE bool is_transparent() const
        {
            return a <= 0;
        }

        //--------------------------------------------------------------------
        AGG_INLINE bool is_opaque() const
        {
            return a >= 1;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type invert(value_type x)
        {
            return 1 - x;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type multiply(value_type a, value_type b)
        {
            return value_type(a * b);
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type demultiply(value_type a, value_type b)
        {
            return (b == 0) ? 0 : value_type(a / b);
        }

        //--------------------------------------------------------------------
        template<typename T>
        static AGG_INLINE T downscale(T a)
        {
            return a;
        }

        //--------------------------------------------------------------------
        template<typename T>
        static AGG_INLINE T downshift(T a, unsigned n)
        {
            return n > 0 ? a / (1 << n) : a;
        }

        //--------------------------------------------------------------------
        static AGG_INLINE value_type mult_cover(value_type a, cover_type b)
        {
            return value_type(a * b / cover_mask);
        }

        //--------------------------------------------------------------------
        static AGG_INLINE cover_type scale_cover(cover_type a, value_type b)
        {
            return cover_type(uround(a * b));
        }

        //--------------------------------------------------------------------
        // Interpolate p to q by a, assuming q is premultiplied by a.
        static AGG_INLINE value_type prelerp(value_type p, value_type q, value_type a)
        {
            return (1 - a) * p + q; // more accurate than "p + q - p * a"
        }

        //--------------------------------------------------------------------
        // Interpolate p to q by a.
        static AGG_INLINE value_type lerp(value_type p, value_type q, value_type a)
        {
			// The form "p + a * (q - p)" avoids a multiplication, but may produce an
			// inaccurate result. For example, "p + (q - p)" may not be exactly equal
			// to q. Therefore, stick to the basic expression, which at least produces
			// the correct result at either extreme.
			return (1 - a) * p + a * q;
        }

        //--------------------------------------------------------------------
        self_type& clear()
        {
            r = g = b = a = 0;
			return *this;
        }

        //--------------------------------------------------------------------
        self_type& transparent()
        {
            a = 0;
            return *this;
        }

        //--------------------------------------------------------------------
        AGG_INLINE self_type& opacity(double a_)
        {
            if (a_ < 0) a = 0;
            else if (a_ > 1) a = 1;
            else a = value_type(a_);
            return *this;
        }

        //--------------------------------------------------------------------
        double opacity() const
        {
            return a;
        }

        //--------------------------------------------------------------------
        AGG_INLINE self_type& premultiply()
        {
            if (a < 1)
            {
                if (a <= 0)
                {
                    r = g = b = 0;
                }
                else
                {
                    r *= a;
                    g *= a;
                    b *= a;
                }
            }
            return *this;
        }

        //--------------------------------------------------------------------
        AGG_INLINE self_type& demultiply()
        {
            if (a < 1)
            {
                if (a <= 0)
                {
                    r = g = b = 0;
                }
                else
                {
                    r /= a;
                    g /= a;
                    b /= a;
                }
            }
            return *this;
        }

        //--------------------------------------------------------------------
        AGG_INLINE self_type gradient(const self_type& c, double k) const
        {
            self_type ret;
            ret.r = value_type(r + (c.r - r) * k);
            ret.g = value_type(g + (c.g - g) * k);
            ret.b = value_type(b + (c.b - b) * k);
            ret.a = value_type(a + (c.a - a) * k);
            return ret;
        }

        //--------------------------------------------------------------------
        AGG_INLINE void add(const self_type& c, unsigned cover)
        {
            if (cover == cover_mask)
            {
                if (c.is_opaque())
                {
                    *this = c;
                    return;
                }
                else
                {
                    r += c.r;
                    g += c.g;
                    b += c.b;
                    a += c.a;
                }
            }
            else
            {
                r += mult_cover(c.r, cover);
                g += mult_cover(c.g, cover);
                b += mult_cover(c.b, cover);
                a += mult_cover(c.a, cover);
            }
            if (a > 1) a = 1;
            if (r > a) r = a;
            if (g > a) g = a;
            if (b > a) b = a;
        }

        //--------------------------------------------------------------------
        template<class GammaLUT>
        AGG_INLINE void apply_gamma_dir(const GammaLUT& gamma)
        {
            r = gamma.dir(r);
            g = gamma.dir(g);
            b = gamma.dir(b);
        }

        //--------------------------------------------------------------------
        template<class GammaLUT>
        AGG_INLINE void apply_gamma_inv(const GammaLUT& gamma)
        {
            r = gamma.inv(r);
            g = gamma.inv(g);
            b = gamma.inv(b);
        }

        //--------------------------------------------------------------------
        static self_type no_color() { return self_type(0,0,0,0); }

        //--------------------------------------------------------------------
        static self_type from_wavelength(double wl, double gamma = 1)
        {
            return self_type(rgba::from_wavelength(wl, gamma));
        }
    };
}



#endif
