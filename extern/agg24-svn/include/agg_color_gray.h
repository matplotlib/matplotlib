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
//
// color types gray8, gray16
//
//----------------------------------------------------------------------------

#ifndef AGG_COLOR_GRAY_INCLUDED
#define AGG_COLOR_GRAY_INCLUDED

#include "agg_basics.h"
#include "agg_color_rgba.h"

namespace agg
{

    //===================================================================gray8
    template<class Colorspace>
    struct gray8T
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
        typedef gray8T self_type;

        value_type v;
        value_type a;

        static value_type luminance(const rgba& c)
        {
            // Calculate grayscale value as per ITU-R BT.709.
            return value_type(uround((0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b) * base_mask));
        }

        static value_type luminance(const rgba8& c)
        {
            // Calculate grayscale value as per ITU-R BT.709.
            return value_type((55u * c.r + 184u * c.g + 18u * c.b) >> 8);
        }

        static void convert(gray8T<linear>& dst, const gray8T<sRGB>& src)
        {
            dst.v = sRGB_conv<value_type>::rgb_from_sRGB(src.v);
            dst.a = src.a;
        }

        static void convert(gray8T<sRGB>& dst, const gray8T<linear>& src)
        {
            dst.v = sRGB_conv<value_type>::rgb_to_sRGB(src.v);
            dst.a = src.a;
        }

        static void convert(gray8T<linear>& dst, const rgba8& src)
        {
            dst.v = luminance(src);
            dst.a = src.a;
        }

        static void convert(gray8T<linear>& dst, const srgba8& src)
        {
            // The RGB weights are only valid for linear values.
            convert(dst, rgba8(src));
        }

        static void convert(gray8T<sRGB>& dst, const rgba8& src)
        {
            dst.v = sRGB_conv<value_type>::rgb_to_sRGB(luminance(src));
            dst.a = src.a;
        }

        static void convert(gray8T<sRGB>& dst, const srgba8& src)
        {
            // The RGB weights are only valid for linear values.
            convert(dst, rgba8(src));
        }

        //--------------------------------------------------------------------
        gray8T() {}

        //--------------------------------------------------------------------
        explicit gray8T(unsigned v_, unsigned a_ = base_mask) :
            v(int8u(v_)), a(int8u(a_)) {}

        //--------------------------------------------------------------------
        gray8T(const self_type& c, unsigned a_) :
            v(c.v), a(value_type(a_)) {}

        //--------------------------------------------------------------------
        gray8T(const rgba& c) :
            v(luminance(c)),
            a(value_type(uround(c.a * base_mask))) {}

        //--------------------------------------------------------------------
        template<class T>
        gray8T(const gray8T<T>& c)
        {
            convert(*this, c);
        }

        //--------------------------------------------------------------------
        template<class T>
        gray8T(const rgba8T<T>& c)
        {
            convert(*this, c);
        }

        //--------------------------------------------------------------------
        template<class T>
        T convert_from_sRGB() const
        {
            typename T::value_type y = sRGB_conv<typename T::value_type>::rgb_from_sRGB(v);
            return T(y, y, y, sRGB_conv<typename T::value_type>::alpha_from_sRGB(a));
        }

        template<class T>
        T convert_to_sRGB() const
        {
            typename T::value_type y = sRGB_conv<typename T::value_type>::rgb_to_sRGB(v);
            return T(y, y, y, sRGB_conv<typename T::value_type>::alpha_to_sRGB(a));
        }

        //--------------------------------------------------------------------
        rgba8 make_rgba8(const linear&) const
        {
            return rgba8(v, v, v, a);
        }

        rgba8 make_rgba8(const sRGB&) const
        {
            return convert_from_sRGB<srgba8>();
        }

        operator rgba8() const
        {
            return make_rgba8(Colorspace());
        }

        //--------------------------------------------------------------------
        srgba8 make_srgba8(const linear&) const
        {
            return convert_to_sRGB<rgba8>();
        }

        srgba8 make_srgba8(const sRGB&) const
        {
            return srgba8(v, v, v, a);
        }

        operator srgba8() const
        {
            return make_rgba8(Colorspace());
        }

        //--------------------------------------------------------------------
        rgba16 make_rgba16(const linear&) const
        {
            rgba16::value_type rgb = (v << 8) | v;
            return rgba16(rgb, rgb, rgb, (a << 8) | a);
        }

        rgba16 make_rgba16(const sRGB&) const
        {
            return convert_from_sRGB<rgba16>();
        }

        operator rgba16() const
        {
            return make_rgba16(Colorspace());
        }

        //--------------------------------------------------------------------
        rgba32 make_rgba32(const linear&) const
        {
            rgba32::value_type v32 = v / 255.0f;
            return rgba32(v32, v32, v32, a / 255.0f);
        }

        rgba32 make_rgba32(const sRGB&) const
        {
            return convert_from_sRGB<rgba32>();
        }

        operator rgba32() const
        {
            return make_rgba32(Colorspace());
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
        static AGG_INLINE value_type mult_cover(value_type a, value_type b)
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
            v = a = 0;
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
        self_type& premultiply()
        {
            if (a < base_mask)
            {
                if (a == 0) v = 0;
                else v = multiply(v, a);
            }
            return *this;
        }

        //--------------------------------------------------------------------
        self_type& demultiply()
        {
            if (a < base_mask)
            {
                if (a == 0)
                {
                    v = 0;
                }
                else
                {
                    calc_type v_ = (calc_type(v) * base_mask) / a;
                    v = value_type((v_ > base_mask) ? (value_type)base_mask : v_);
                }
            }
            return *this;
        }

        //--------------------------------------------------------------------
        self_type gradient(self_type c, double k) const
        {
            self_type ret;
            calc_type ik = uround(k * base_scale);
            ret.v = lerp(v, c.v, ik);
            ret.a = lerp(a, c.a, ik);
            return ret;
        }

        //--------------------------------------------------------------------
        AGG_INLINE void add(const self_type& c, unsigned cover)
        {
            calc_type cv, ca;
            if (cover == cover_mask)
            {
                if (c.a == base_mask)
                {
                    *this = c;
                    return;
                }
                else
                {
                    cv = v + c.v;
                    ca = a + c.a;
                }
            }
            else
            {
                cv = v + mult_cover(c.v, cover);
                ca = a + mult_cover(c.a, cover);
            }
            v = (value_type)((cv > calc_type(base_mask)) ? calc_type(base_mask) : cv);
            a = (value_type)((ca > calc_type(base_mask)) ? calc_type(base_mask) : ca);
        }

        //--------------------------------------------------------------------
        static self_type no_color() { return self_type(0,0); }
    };

    typedef gray8T<linear> gray8;
    typedef gray8T<sRGB> sgray8;


    //==================================================================gray16
    struct gray16
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
        typedef gray16 self_type;

        value_type v;
        value_type a;

        static value_type luminance(const rgba& c)
        {
            // Calculate grayscale value as per ITU-R BT.709.
            return value_type(uround((0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b) * base_mask));
        }

        static value_type luminance(const rgba16& c)
        {
            // Calculate grayscale value as per ITU-R BT.709.
            return value_type((13933u * c.r + 46872u * c.g + 4732u * c.b) >> 16);
        }

        static value_type luminance(const rgba8& c)
        {
            return luminance(rgba16(c));
        }

        static value_type luminance(const srgba8& c)
        {
            return luminance(rgba16(c));
        }

        static value_type luminance(const rgba32& c)
        {
            return luminance(rgba(c));
        }

        //--------------------------------------------------------------------
        gray16() {}

        //--------------------------------------------------------------------
        explicit gray16(unsigned v_, unsigned a_ = base_mask) :
            v(int16u(v_)), a(int16u(a_)) {}

        //--------------------------------------------------------------------
        gray16(const self_type& c, unsigned a_) :
            v(c.v), a(value_type(a_)) {}

        //--------------------------------------------------------------------
        gray16(const rgba& c) :
            v(luminance(c)),
            a((value_type)uround(c.a * double(base_mask))) {}

        //--------------------------------------------------------------------
        gray16(const rgba8& c) :
            v(luminance(c)),
            a((value_type(c.a) << 8) | c.a) {}

        //--------------------------------------------------------------------
        gray16(const srgba8& c) :
            v(luminance(c)),
            a((value_type(c.a) << 8) | c.a) {}

        //--------------------------------------------------------------------
        gray16(const rgba16& c) :
            v(luminance(c)),
            a(c.a) {}

        //--------------------------------------------------------------------
        gray16(const gray8& c) :
            v((value_type(c.v) << 8) | c.v),
            a((value_type(c.a) << 8) | c.a) {}

        //--------------------------------------------------------------------
        gray16(const sgray8& c) :
            v(sRGB_conv<value_type>::rgb_from_sRGB(c.v)),
            a(sRGB_conv<value_type>::alpha_from_sRGB(c.a)) {}

        //--------------------------------------------------------------------
        operator rgba8() const
        {
            return rgba8(v >> 8, v >> 8, v >> 8, a >> 8);
        }

        //--------------------------------------------------------------------
        operator srgba8() const
        {
            value_type y = sRGB_conv<value_type>::rgb_to_sRGB(v);
            return srgba8(y, y, y, sRGB_conv<value_type>::alpha_to_sRGB(a));
        }

        //--------------------------------------------------------------------
        operator rgba16() const
        {
            return rgba16(v, v, v, a);
        }

		//--------------------------------------------------------------------
		operator rgba32() const
		{
			rgba32::value_type v32 = v / 65535.0f;
			return rgba32(v32, v32, v32, a / 65535.0f);
		}

		//--------------------------------------------------------------------
        operator gray8() const
        {
            return gray8(v >> 8, a >> 8);
        }

        //--------------------------------------------------------------------
        operator sgray8() const
        {
            return sgray8(
                sRGB_conv<value_type>::rgb_to_sRGB(v),
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
            return multiply(a, b << 8 | b);
        }

        //--------------------------------------------------------------------
        static AGG_INLINE cover_type scale_cover(cover_type a, value_type b)
        {
            return mult_cover(b, a) >> 8;
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
            v = a = 0;
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
            else if(a_ > 1) a = 1;
            else a = (value_type)uround(a_ * double(base_mask));
			return *this;
        }

        //--------------------------------------------------------------------
        double opacity() const
        {
            return double(a) / double(base_mask);
        }


        //--------------------------------------------------------------------
        self_type& premultiply()
        {
            if (a < base_mask)
            {
                if(a == 0) v = 0;
                else v = multiply(v, a);
            }
            return *this;
        }

        //--------------------------------------------------------------------
        self_type& demultiply()
        {
            if (a < base_mask)
            {
                if (a == 0)
                {
                    v = 0;
                }
                else
                {
                    calc_type v_ = (calc_type(v) * base_mask) / a;
                    v = value_type((v_ > base_mask) ? base_mask : v_);
                }
            }
            return *this;
        }

        //--------------------------------------------------------------------
        self_type gradient(self_type c, double k) const
        {
            self_type ret;
            calc_type ik = uround(k * base_scale);
            ret.v = lerp(v, c.v, ik);
            ret.a = lerp(a, c.a, ik);
            return ret;
        }

        //--------------------------------------------------------------------
        AGG_INLINE void add(const self_type& c, unsigned cover)
        {
            calc_type cv, ca;
            if (cover == cover_mask)
            {
                if (c.a == base_mask)
                {
                    *this = c;
                    return;
                }
                else
                {
                    cv = v + c.v;
                    ca = a + c.a;
                }
            }
            else
            {
                cv = v + mult_cover(c.v, cover);
                ca = a + mult_cover(c.a, cover);
            }
            v = (value_type)((cv > calc_type(base_mask)) ? calc_type(base_mask) : cv);
            a = (value_type)((ca > calc_type(base_mask)) ? calc_type(base_mask) : ca);
        }

        //--------------------------------------------------------------------
        static self_type no_color() { return self_type(0,0); }
    };


    //===================================================================gray32
    struct gray32
    {
        typedef float value_type;
        typedef double calc_type;
        typedef double long_type;
        typedef gray32 self_type;

        value_type v;
        value_type a;

        // Calculate grayscale value as per ITU-R BT.709.
        static value_type luminance(double r, double g, double b)
        {
            return value_type(0.2126 * r + 0.7152 * g + 0.0722 * b);
        }

        static value_type luminance(const rgba& c)
        {
            return luminance(c.r, c.g, c.b);
        }

        static value_type luminance(const rgba32& c)
        {
            return luminance(c.r, c.g, c.b);
        }

        static value_type luminance(const rgba8& c)
        {
            return luminance(c.r / 255.0, c.g / 255.0, c.g / 255.0);
        }

        static value_type luminance(const rgba16& c)
        {
            return luminance(c.r / 65535.0, c.g / 65535.0, c.g / 65535.0);
        }

        //--------------------------------------------------------------------
        gray32() {}

        //--------------------------------------------------------------------
        explicit gray32(value_type v_, value_type a_ = 1) :
            v(v_), a(a_) {}

        //--------------------------------------------------------------------
        gray32(const self_type& c, value_type a_) :
            v(c.v), a(a_) {}

        //--------------------------------------------------------------------
        gray32(const rgba& c) :
            v(luminance(c)),
            a(value_type(c.a)) {}

        //--------------------------------------------------------------------
        gray32(const rgba8& c) :
            v(luminance(c)),
            a(value_type(c.a / 255.0)) {}

        //--------------------------------------------------------------------
        gray32(const srgba8& c) :
            v(luminance(rgba32(c))),
            a(value_type(c.a / 255.0)) {}

        //--------------------------------------------------------------------
        gray32(const rgba16& c) :
            v(luminance(c)),
            a(value_type(c.a / 65535.0)) {}

        //--------------------------------------------------------------------
        gray32(const rgba32& c) :
            v(luminance(c)),
            a(value_type(c.a)) {}

        //--------------------------------------------------------------------
        gray32(const gray8& c) :
            v(value_type(c.v / 255.0)),
            a(value_type(c.a / 255.0)) {}

        //--------------------------------------------------------------------
        gray32(const sgray8& c) :
            v(sRGB_conv<value_type>::rgb_from_sRGB(c.v)),
            a(sRGB_conv<value_type>::alpha_from_sRGB(c.a)) {}

        //--------------------------------------------------------------------
        gray32(const gray16& c) :
            v(value_type(c.v / 65535.0)),
            a(value_type(c.a / 65535.0)) {}

        //--------------------------------------------------------------------
        operator rgba() const
        {
            return rgba(v, v, v, a);
        }

        //--------------------------------------------------------------------
        operator gray8() const
        {
            return gray8(uround(v * 255.0), uround(a * 255.0));
        }

        //--------------------------------------------------------------------
        operator sgray8() const
        {
            // Return (non-premultiplied) sRGB values.
            return sgray8(
                sRGB_conv<value_type>::rgb_to_sRGB(v),
                sRGB_conv<value_type>::alpha_to_sRGB(a));
        }

        //--------------------------------------------------------------------
        operator gray16() const
        {
            return gray16(uround(v * 65535.0), uround(a * 65535.0));
        }

        //--------------------------------------------------------------------
        operator rgba8() const
        {
            rgba8::value_type y = uround(v * 255.0);
            return rgba8(y, y, y, uround(a * 255.0));
        }

        //--------------------------------------------------------------------
        operator srgba8() const
        {
            srgba8::value_type y = sRGB_conv<value_type>::rgb_to_sRGB(v);
            return srgba8(y, y, y, sRGB_conv<value_type>::alpha_to_sRGB(a));
        }

		//--------------------------------------------------------------------
		operator rgba16() const
		{
			rgba16::value_type y = uround(v * 65535.0);
			return rgba16(y, y, y, uround(a * 65535.0));
		}

		//--------------------------------------------------------------------
		operator rgba32() const
		{
            return rgba32(v, v, v, a);
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
            v = a = 0;
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
            else a = value_type(a_);
			return *this;
        }

        //--------------------------------------------------------------------
        double opacity() const
        {
            return a;
        }


        //--------------------------------------------------------------------
        self_type& premultiply()
        {
            if (a < 0) v = 0;
            else if(a < 1) v *= a;
            return *this;
        }

        //--------------------------------------------------------------------
        self_type& demultiply()
        {
            if (a < 0) v = 0;
            else if (a < 1) v /= a;
            return *this;
        }

        //--------------------------------------------------------------------
        self_type gradient(self_type c, double k) const
        {
            return self_type(
                value_type(v + (c.v - v) * k),
                value_type(a + (c.a - a) * k));
        }

        //--------------------------------------------------------------------
        static self_type no_color() { return self_type(0,0); }
    };
}




#endif
