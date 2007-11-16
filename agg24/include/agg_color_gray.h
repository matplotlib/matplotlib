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
    struct gray8
    {
        typedef int8u  value_type;
        typedef int32u calc_type;
        typedef int32  long_type;
        enum base_scale_e
        {
            base_shift = 8,
            base_scale = 1 << base_shift,
            base_mask  = base_scale - 1
        };
        typedef gray8 self_type;

        value_type v;
        value_type a;

        //--------------------------------------------------------------------
        gray8() {}

        //--------------------------------------------------------------------
        gray8(unsigned v_, unsigned a_=base_mask) :
            v(int8u(v_)), a(int8u(a_)) {}

        //--------------------------------------------------------------------
        gray8(const self_type& c, unsigned a_) :
            v(c.v), a(value_type(a_)) {}

        //--------------------------------------------------------------------
        gray8(const rgba& c) :
            v((value_type)uround((0.299*c.r + 0.587*c.g + 0.114*c.b) * double(base_mask))),
            a((value_type)uround(c.a * double(base_mask))) {}

        //--------------------------------------------------------------------
        gray8(const rgba& c, double a_) :
            v((value_type)uround((0.299*c.r + 0.587*c.g + 0.114*c.b) * double(base_mask))),
            a((value_type)uround(a_ * double(base_mask))) {}

        //--------------------------------------------------------------------
        gray8(const rgba8& c) :
            v((c.r*77 + c.g*150 + c.b*29) >> 8),
            a(c.a) {}

        //--------------------------------------------------------------------
        gray8(const rgba8& c, unsigned a_) :
            v((c.r*77 + c.g*150 + c.b*29) >> 8),
            a(a_) {}

        //--------------------------------------------------------------------
        void clear()
        {
            v = a = 0;
        }

        //--------------------------------------------------------------------
        const self_type& transparent()
        {
            a = 0;
            return *this;
        }

        //--------------------------------------------------------------------
        void opacity(double a_)
        {
            if(a_ < 0.0) a_ = 0.0;
            if(a_ > 1.0) a_ = 1.0;
            a = (value_type)uround(a_ * double(base_mask));
        }

        //--------------------------------------------------------------------
        double opacity() const
        {
            return double(a) / double(base_mask);
        }


        //--------------------------------------------------------------------
        const self_type& premultiply()
        {
            if(a == base_mask) return *this;
            if(a == 0)
            {
                v = 0;
                return *this;
            }
            v = value_type((calc_type(v) * a) >> base_shift);
            return *this;
        }

        //--------------------------------------------------------------------
        const self_type& premultiply(unsigned a_)
        {
            if(a == base_mask && a_ >= base_mask) return *this;
            if(a == 0 || a_ == 0)
            {
                v = a = 0;
                return *this;
            }
            calc_type v_ = (calc_type(v) * a_) / a;
            v = value_type((v_ > a_) ? a_ : v_);
            a = value_type(a_);
            return *this;
        }

        //--------------------------------------------------------------------
        const self_type& demultiply()
        {
            if(a == base_mask) return *this;
            if(a == 0)
            {
                v = 0;
                return *this;
            }
            calc_type v_ = (calc_type(v) * base_mask) / a;
            v = value_type((v_ > base_mask) ? (value_type)base_mask : v_);
            return *this;
        }

        //--------------------------------------------------------------------
        self_type gradient(self_type c, double k) const
        {
            self_type ret;
            calc_type ik = uround(k * base_scale);
            ret.v = value_type(calc_type(v) + (((calc_type(c.v) - v) * ik) >> base_shift));
            ret.a = value_type(calc_type(a) + (((calc_type(c.a) - a) * ik) >> base_shift));
            return ret;
        }

        //--------------------------------------------------------------------
        AGG_INLINE void add(const self_type& c, unsigned cover)
        {
            calc_type cv, ca;
            if(cover == cover_mask)
            {
                if(c.a == base_mask) 
                {
                    *this = c;
                }
                else
                {
                    cv = v + c.v; v = (cv > calc_type(base_mask)) ? calc_type(base_mask) : cv;
                    ca = a + c.a; a = (ca > calc_type(base_mask)) ? calc_type(base_mask) : ca;
                }
            }
            else
            {
                cv = v + ((c.v * cover + cover_mask/2) >> cover_shift);
                ca = a + ((c.a * cover + cover_mask/2) >> cover_shift);
                v = (cv > calc_type(base_mask)) ? calc_type(base_mask) : cv;
                a = (ca > calc_type(base_mask)) ? calc_type(base_mask) : ca;
            }
        }

        //--------------------------------------------------------------------
        static self_type no_color() { return self_type(0,0); }
    };


    //-------------------------------------------------------------gray8_pre
    inline gray8 gray8_pre(unsigned v, unsigned a = gray8::base_mask)
    {
        return gray8(v,a).premultiply();
    }
    inline gray8 gray8_pre(const gray8& c, unsigned a)
    {
        return gray8(c,a).premultiply();
    }
    inline gray8 gray8_pre(const rgba& c)
    {
        return gray8(c).premultiply();
    }
    inline gray8 gray8_pre(const rgba& c, double a)
    {
        return gray8(c,a).premultiply();
    }
    inline gray8 gray8_pre(const rgba8& c)
    {
        return gray8(c).premultiply();
    }
    inline gray8 gray8_pre(const rgba8& c, unsigned a)
    {
        return gray8(c,a).premultiply();
    }




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
            base_mask  = base_scale - 1
        };
        typedef gray16 self_type;

        value_type v;
        value_type a;

        //--------------------------------------------------------------------
        gray16() {}

        //--------------------------------------------------------------------
        gray16(unsigned v_, unsigned a_=base_mask) :
            v(int16u(v_)), a(int16u(a_)) {}

        //--------------------------------------------------------------------
        gray16(const self_type& c, unsigned a_) :
            v(c.v), a(value_type(a_)) {}

        //--------------------------------------------------------------------
        gray16(const rgba& c) :
            v((value_type)uround((0.299*c.r + 0.587*c.g + 0.114*c.b) * double(base_mask))),
            a((value_type)uround(c.a * double(base_mask))) {}

        //--------------------------------------------------------------------
        gray16(const rgba& c, double a_) :
            v((value_type)uround((0.299*c.r + 0.587*c.g + 0.114*c.b) * double(base_mask))),
            a((value_type)uround(a_ * double(base_mask))) {}

        //--------------------------------------------------------------------
        gray16(const rgba8& c) :
            v(c.r*77 + c.g*150 + c.b*29),
            a((value_type(c.a) << 8) | c.a) {}

        //--------------------------------------------------------------------
        gray16(const rgba8& c, unsigned a_) :
            v(c.r*77 + c.g*150 + c.b*29),
            a((value_type(a_) << 8) | c.a) {}

        //--------------------------------------------------------------------
        void clear()
        {
            v = a = 0;
        }

        //--------------------------------------------------------------------
        const self_type& transparent()
        {
            a = 0;
            return *this;
        }

        //--------------------------------------------------------------------
        void opacity(double a_)
        {
            if(a_ < 0.0) a_ = 0.0;
            if(a_ > 1.0) a_ = 1.0;
            a = (value_type)uround(a_ * double(base_mask));
        }

        //--------------------------------------------------------------------
        double opacity() const
        {
            return double(a) / double(base_mask);
        }


        //--------------------------------------------------------------------
        const self_type& premultiply()
        {
            if(a == base_mask) return *this;
            if(a == 0)
            {
                v = 0;
                return *this;
            }
            v = value_type((calc_type(v) * a) >> base_shift);
            return *this;
        }

        //--------------------------------------------------------------------
        const self_type& premultiply(unsigned a_)
        {
            if(a == base_mask && a_ >= base_mask) return *this;
            if(a == 0 || a_ == 0)
            {
                v = a = 0;
                return *this;
            }
            calc_type v_ = (calc_type(v) * a_) / a;
            v = value_type((v_ > a_) ? a_ : v_);
            a = value_type(a_);
            return *this;
        }

        //--------------------------------------------------------------------
        const self_type& demultiply()
        {
            if(a == base_mask) return *this;
            if(a == 0)
            {
                v = 0;
                return *this;
            }
            calc_type v_ = (calc_type(v) * base_mask) / a;
            v = value_type((v_ > base_mask) ? base_mask : v_);
            return *this;
        }

        //--------------------------------------------------------------------
        self_type gradient(self_type c, double k) const
        {
            self_type ret;
            calc_type ik = uround(k * base_scale);
            ret.v = value_type(calc_type(v) + (((calc_type(c.v) - v) * ik) >> base_shift));
            ret.a = value_type(calc_type(a) + (((calc_type(c.a) - a) * ik) >> base_shift));
            return ret;
        }

        //--------------------------------------------------------------------
        AGG_INLINE void add(const self_type& c, unsigned cover)
        {
            calc_type cv, ca;
            if(cover == cover_mask)
            {
                if(c.a == base_mask) 
                {
                    *this = c;
                }
                else
                {
                    cv = v + c.v; v = (cv > calc_type(base_mask)) ? calc_type(base_mask) : cv;
                    ca = a + c.a; a = (ca > calc_type(base_mask)) ? calc_type(base_mask) : ca;
                }
            }
            else
            {
                cv = v + ((c.v * cover + cover_mask/2) >> cover_shift);
                ca = a + ((c.a * cover + cover_mask/2) >> cover_shift);
                v = (cv > calc_type(base_mask)) ? calc_type(base_mask) : cv;
                a = (ca > calc_type(base_mask)) ? calc_type(base_mask) : ca;
            }
        }

        //--------------------------------------------------------------------
        static self_type no_color() { return self_type(0,0); }
    };


    //------------------------------------------------------------gray16_pre
    inline gray16 gray16_pre(unsigned v, unsigned a = gray16::base_mask)
    {
        return gray16(v,a).premultiply();
    }
    inline gray16 gray16_pre(const gray16& c, unsigned a)
    {
        return gray16(c,a).premultiply();
    }
    inline gray16 gray16_pre(const rgba& c)
    {
        return gray16(c).premultiply();
    }
    inline gray16 gray16_pre(const rgba& c, double a)
    {
        return gray16(c,a).premultiply();
    }
    inline gray16 gray16_pre(const rgba8& c)
    {
        return gray16(c).premultiply();
    }
    inline gray16 gray16_pre(const rgba8& c, unsigned a)
    {
        return gray16(c,a).premultiply();
    }


}




#endif
