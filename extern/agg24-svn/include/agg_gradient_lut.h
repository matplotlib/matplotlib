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

#ifndef AGG_GRADIENT_LUT_INCLUDED
#define AGG_GRADIENT_LUT_INCLUDED

#include "agg_array.h"
#include "agg_dda_line.h"
#include "agg_color_rgba.h"
#include "agg_color_gray.h"

namespace agg
{

    //======================================================color_interpolator
    template<class ColorT> struct color_interpolator
    {
    public:
        typedef ColorT color_type;

        color_interpolator(const color_type& c1, 
                           const color_type& c2, 
                           unsigned len) :
            m_c1(c1),
            m_c2(c2),
            m_len(len),
            m_count(0)
        {}

        void operator ++ ()
        {
            ++m_count;
        }

        color_type color() const
        {
            return m_c1.gradient(m_c2, double(m_count) / m_len);
        }

    private:
        color_type m_c1;
        color_type m_c2;
        unsigned   m_len;
        unsigned   m_count;
    };

    //========================================================================
    // Fast specialization for rgba8
    template<> struct color_interpolator<rgba8>
    {
    public:
        typedef rgba8 color_type;

        color_interpolator(const color_type& c1, 
                           const color_type& c2, 
                           unsigned len) :
            r(c1.r, c2.r, len),
            g(c1.g, c2.g, len),
            b(c1.b, c2.b, len),
            a(c1.a, c2.a, len)
        {}

        void operator ++ ()
        {
            ++r; ++g; ++b; ++a;
        }

        color_type color() const
        {
            return color_type(r.y(), g.y(), b.y(), a.y());
        }

    private:
        agg::dda_line_interpolator<14> r, g, b, a;
    };

    //========================================================================
    // Fast specialization for gray8
    template<> struct color_interpolator<gray8>
    {
    public:
        typedef gray8 color_type;

        color_interpolator(const color_type& c1, 
                           const color_type& c2, 
                           unsigned len) :
            v(c1.v, c2.v, len),
            a(c1.a, c2.a, len)
        {}

        void operator ++ ()
        {
            ++v; ++a;
        }

        color_type color() const
        {
            return color_type(v.y(), a.y());
        }

    private:
        agg::dda_line_interpolator<14> v,a;
    };

    //============================================================gradient_lut
    template<class ColorInterpolator, 
             unsigned ColorLutSize=256> class gradient_lut
    {
    public:
        typedef ColorInterpolator interpolator_type;
        typedef typename interpolator_type::color_type color_type;
        enum { color_lut_size = ColorLutSize };

        //--------------------------------------------------------------------
        gradient_lut() : m_color_lut(color_lut_size) {}

        // Build Gradient Lut
        // First, call remove_all(), then add_color() at least twice, 
        // then build_lut(). Argument "offset" in add_color must be 
        // in range [0...1] and defines a color stop as it is described 
        // in SVG specification, section Gradients and Patterns. 
        // The simplest linear gradient is:
        //    gradient_lut.add_color(0.0, start_color);
        //    gradient_lut.add_color(1.0, end_color);
        //--------------------------------------------------------------------
        void remove_all();
        void add_color(double offset, const color_type& color);
        void build_lut();

        // Size-index Interface. This class can be used directly as the 
        // ColorF in span_gradient. All it needs is two access methods 
        // size() and operator [].
        //--------------------------------------------------------------------
        static unsigned size() 
        { 
            return color_lut_size; 
        }
        const color_type& operator [] (unsigned i) const 
        { 
            return m_color_lut[i]; 
        }

    private:
        //--------------------------------------------------------------------
        struct color_point
        {
            double     offset;
            color_type color;

            color_point() {}
            color_point(double off, const color_type& c) : 
                offset(off), color(c)
            {
                if(offset < 0.0) offset = 0.0;
                if(offset > 1.0) offset = 1.0;
            }
        };
        typedef agg::pod_bvector<color_point, 4> color_profile_type;
        typedef agg::pod_array<color_type>       color_lut_type;

        static bool offset_less(const color_point& a, const color_point& b)
        {
            return a.offset < b.offset;
        }
        static bool offset_equal(const color_point& a, const color_point& b)
        {
            return a.offset == b.offset;
        }

        //--------------------------------------------------------------------
        color_profile_type  m_color_profile;
        color_lut_type      m_color_lut;
    };



    //------------------------------------------------------------------------
    template<class T, unsigned S>
    void gradient_lut<T,S>::remove_all()
    { 
        m_color_profile.remove_all(); 
    }

    //------------------------------------------------------------------------
    template<class T, unsigned S>
    void gradient_lut<T,S>::add_color(double offset, const color_type& color)
    {
        m_color_profile.add(color_point(offset, color));
    }

    //------------------------------------------------------------------------
    template<class T, unsigned S>
    void gradient_lut<T,S>::build_lut()
    {
        quick_sort(m_color_profile, offset_less);
        m_color_profile.cut_at(remove_duplicates(m_color_profile, offset_equal));
        if(m_color_profile.size() >= 2)
        {
            unsigned i;
            unsigned start = uround(m_color_profile[0].offset * color_lut_size);
            unsigned end;
            color_type c = m_color_profile[0].color;
            for(i = 0; i < start; i++) 
            {
                m_color_lut[i] = c;
            }
            for(i = 1; i < m_color_profile.size(); i++)
            {
                end  = uround(m_color_profile[i].offset * color_lut_size);
                interpolator_type ci(m_color_profile[i-1].color, 
                                     m_color_profile[i  ].color, 
                                     end - start + 1);
                while(start < end)
                {
                    m_color_lut[start] = ci.color();
                    ++ci;
                    ++start;
                }
            }
            c = m_color_profile.last().color;
            for(; end < m_color_lut.size(); end++)
            {
                m_color_lut[end] = c;
            }
        }
    }
}




#endif
