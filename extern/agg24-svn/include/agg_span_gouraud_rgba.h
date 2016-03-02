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

#ifndef AGG_SPAN_GOURAUD_RGBA_INCLUDED
#define AGG_SPAN_GOURAUD_RGBA_INCLUDED

#include "agg_basics.h"
#include "agg_color_rgba.h"
#include "agg_dda_line.h"
#include "agg_span_gouraud.h"

namespace agg
{

    //=======================================================span_gouraud_rgba
    template<class ColorT> class span_gouraud_rgba : public span_gouraud<ColorT>
    {
    public:
        typedef ColorT color_type;
        typedef typename ColorT::value_type value_type;
        typedef span_gouraud<color_type> base_type;
        typedef typename base_type::coord_type coord_type;
        enum subpixel_scale_e
        { 
            subpixel_shift = 4, 
            subpixel_scale = 1 << subpixel_shift
        };

    private:
        //--------------------------------------------------------------------
        struct rgba_calc
        {
            void init(const coord_type& c1, const coord_type& c2)
            {
                m_x1  = c1.x - 0.5; 
                m_y1  = c1.y - 0.5;
                m_dx  = c2.x - c1.x;
                double dy = c2.y - c1.y;
                m_1dy = (dy < 1e-5) ? 1e5 : 1.0 / dy;
                m_r1  = c1.color.r;
                m_g1  = c1.color.g;
                m_b1  = c1.color.b;
                m_a1  = c1.color.a;
                m_dr  = c2.color.r - m_r1;
                m_dg  = c2.color.g - m_g1;
                m_db  = c2.color.b - m_b1;
                m_da  = c2.color.a - m_a1;
            }

            void calc(double y)
            {
                double k = (y - m_y1) * m_1dy;
                if(k < 0.0) k = 0.0;
                if(k > 1.0) k = 1.0;
                m_r = m_r1 + iround(m_dr * k);
                m_g = m_g1 + iround(m_dg * k);
                m_b = m_b1 + iround(m_db * k);
                m_a = m_a1 + iround(m_da * k);
                m_x = iround((m_x1 + m_dx * k) * subpixel_scale);
            }

            double m_x1;
            double m_y1;
            double m_dx;
            double m_1dy;
            int    m_r1;
            int    m_g1;
            int    m_b1;
            int    m_a1;
            int    m_dr;
            int    m_dg;
            int    m_db;
            int    m_da;
            int    m_r;
            int    m_g;
            int    m_b;
            int    m_a;
            int    m_x;
        };

    public:

        //--------------------------------------------------------------------
        span_gouraud_rgba() {}
        span_gouraud_rgba(const color_type& c1, 
                          const color_type& c2, 
                          const color_type& c3,
                          double x1, double y1, 
                          double x2, double y2,
                          double x3, double y3, 
                          double d = 0) : 
            base_type(c1, c2, c3, x1, y1, x2, y2, x3, y3, d)
        {}

        //--------------------------------------------------------------------
        void prepare()
        {
            coord_type coord[3];
            base_type::arrange_vertices(coord);

            m_y2 = int(coord[1].y);

            m_swap = cross_product(coord[0].x, coord[0].y, 
                                   coord[2].x, coord[2].y,
                                   coord[1].x, coord[1].y) < 0.0;

            m_rgba1.init(coord[0], coord[2]);
            m_rgba2.init(coord[0], coord[1]);
            m_rgba3.init(coord[1], coord[2]);
        }

        //--------------------------------------------------------------------
        void generate(color_type* span, int x, int y, unsigned len)
        {
            m_rgba1.calc(y);//(m_rgba1.m_1dy > 2) ? m_rgba1.m_y1 : y);
            const rgba_calc* pc1 = &m_rgba1;
            const rgba_calc* pc2 = &m_rgba2;

            if(y <= m_y2)
            {          
                // Bottom part of the triangle (first subtriangle)
                //-------------------------
                m_rgba2.calc(y + m_rgba2.m_1dy);
            }
            else
            {
                // Upper part (second subtriangle)
                m_rgba3.calc(y - m_rgba3.m_1dy);
                //-------------------------
                pc2 = &m_rgba3;
            }

            if(m_swap)
            {
                // It means that the triangle is oriented clockwise, 
                // so that we need to swap the controlling structures
                //-------------------------
                const rgba_calc* t = pc2;
                pc2 = pc1;
                pc1 = t;
            }

            // Get the horizontal length with subpixel accuracy
            // and protect it from division by zero
            //-------------------------
            int nlen = abs(pc2->m_x - pc1->m_x);
            if(nlen <= 0) nlen = 1;

            dda_line_interpolator<14> r(pc1->m_r, pc2->m_r, nlen);
            dda_line_interpolator<14> g(pc1->m_g, pc2->m_g, nlen);
            dda_line_interpolator<14> b(pc1->m_b, pc2->m_b, nlen);
            dda_line_interpolator<14> a(pc1->m_a, pc2->m_a, nlen);

            // Calculate the starting point of the gradient with subpixel 
            // accuracy and correct (roll back) the interpolators.
            // This operation will also clip the beginning of the span
            // if necessary.
            //-------------------------
            int start = pc1->m_x - (x << subpixel_shift);
            r    -= start; 
            g    -= start; 
            b    -= start; 
            a    -= start;
            nlen += start;

            int vr, vg, vb, va;
            enum lim_e { lim = color_type::base_mask };

            // Beginning part of the span. Since we rolled back the 
            // interpolators, the color values may have overflow.
            // So that, we render the beginning part with checking 
            // for overflow. It lasts until "start" is positive;
            // typically it's 1-2 pixels, but may be more in some cases.
            //-------------------------
            while(len && start > 0)
            {
                vr = r.y();
                vg = g.y();
                vb = b.y();
                va = a.y();
                if(vr < 0) vr = 0; if(vr > lim) vr = lim;
                if(vg < 0) vg = 0; if(vg > lim) vg = lim;
                if(vb < 0) vb = 0; if(vb > lim) vb = lim;
                if(va < 0) va = 0; if(va > lim) va = lim;
                span->r = (value_type)vr;
                span->g = (value_type)vg;
                span->b = (value_type)vb;
                span->a = (value_type)va;
                r     += subpixel_scale; 
                g     += subpixel_scale; 
                b     += subpixel_scale; 
                a     += subpixel_scale;
                nlen  -= subpixel_scale;
                start -= subpixel_scale;
                ++span;
                --len;
            }

            // Middle part, no checking for overflow.
            // Actual spans can be longer than the calculated length
            // because of anti-aliasing, thus, the interpolators can 
            // overflow. But while "nlen" is positive we are safe.
            //-------------------------
            while(len && nlen > 0)
            {
                span->r = (value_type)r.y();
                span->g = (value_type)g.y();
                span->b = (value_type)b.y();
                span->a = (value_type)a.y();
                r    += subpixel_scale; 
                g    += subpixel_scale; 
                b    += subpixel_scale; 
                a    += subpixel_scale;
                nlen -= subpixel_scale;
                ++span;
                --len;
            }

            // Ending part; checking for overflow.
            // Typically it's 1-2 pixels, but may be more in some cases.
            //-------------------------
            while(len)
            {
                vr = r.y();
                vg = g.y();
                vb = b.y();
                va = a.y();
                if(vr < 0) vr = 0; if(vr > lim) vr = lim;
                if(vg < 0) vg = 0; if(vg > lim) vg = lim;
                if(vb < 0) vb = 0; if(vb > lim) vb = lim;
                if(va < 0) va = 0; if(va > lim) va = lim;
                span->r = (value_type)vr;
                span->g = (value_type)vg;
                span->b = (value_type)vb;
                span->a = (value_type)va;
                r += subpixel_scale; 
                g += subpixel_scale; 
                b += subpixel_scale; 
                a += subpixel_scale;
                ++span;
                --len;
            }
        }

    private:
        bool      m_swap;
        int       m_y2;
        rgba_calc m_rgba1;
        rgba_calc m_rgba2;
        rgba_calc m_rgba3;
    };



}

#endif
