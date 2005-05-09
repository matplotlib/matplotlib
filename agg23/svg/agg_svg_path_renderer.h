//----------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.3
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
// SVG path renderer.
//
//----------------------------------------------------------------------------
#ifndef AGG_SVG_PATH_RENDERER_INCLUDED
#define AGG_SVG_PATH_RENDERER_INCLUDED

#include "agg_path_storage.h"
#include "agg_conv_transform.h"
#include "agg_conv_stroke.h"
#include "agg_conv_contour.h"
#include "agg_conv_curve.h"
#include "agg_color_rgba.h"
#include "agg_array.h"
#include "agg_bounding_rect.h"
#include "agg_rasterizer_scanline_aa.h"
#include "agg_svg_path_tokenizer.h"

namespace agg
{
namespace svg
{

    //============================================================================
    // Basic path attributes
    struct path_attributes
    {
        unsigned     index;
        rgba8        fill_color;
        rgba8        stroke_color;
        bool         fill_flag;
        bool         stroke_flag;
        bool         even_odd_flag;
        line_join_e  line_join;
        line_cap_e   line_cap;
        double       miter_limit;
        double       stroke_width;
        trans_affine transform;

        // Empty constructor
        path_attributes() :
            index(0),
            fill_color(rgba(0,0,0)),
            stroke_color(rgba(0,0,0)),
            fill_flag(true),
            stroke_flag(false),
            even_odd_flag(false),
            line_join(miter_join),
            line_cap(butt_cap),
            miter_limit(4.0),
            stroke_width(1.0),
            transform()
        {
        }

        // Copy constructor
        path_attributes(const path_attributes& attr) :
            index(attr.index),
            fill_color(attr.fill_color),
            stroke_color(attr.stroke_color),
            fill_flag(attr.fill_flag),
            stroke_flag(attr.stroke_flag),
            even_odd_flag(attr.even_odd_flag),
            line_join(attr.line_join),
            line_cap(attr.line_cap),
            miter_limit(attr.miter_limit),
            stroke_width(attr.stroke_width),
            transform(attr.transform)
        {
        }

        // Copy constructor with new index value
        path_attributes(const path_attributes& attr, unsigned idx) :
            index(idx),
            fill_color(attr.fill_color),
            stroke_color(attr.stroke_color),
            fill_flag(attr.fill_flag),
            stroke_flag(attr.stroke_flag),
            even_odd_flag(attr.even_odd_flag),
            line_join(attr.line_join),
            line_cap(attr.line_cap),
            miter_limit(attr.miter_limit),
            stroke_width(attr.stroke_width),
            transform(attr.transform)
        {
        }
    };


    //============================================================================
    // Path container and renderer. 
    class path_renderer
    {
    public:
        typedef pod_deque<path_attributes>              attr_storage;

        typedef conv_curve<path_storage>                curved;

        typedef conv_stroke<curved>                     curved_stroked;
        typedef conv_transform<curved_stroked>          curved_stroked_trans;

        typedef conv_transform<curved>                  curved_trans;
        typedef conv_contour<curved_trans>              curved_trans_contour;

        path_renderer();

        void remove_all();

        // Use these functions as follows:
        // begin_path() when the XML tag <path> comes ("start_element" handler)
        // parse_path() on "d=" tag attribute
        // end_path() when parsing of the entire tag is done.
        void begin_path();
        void parse_path(path_tokenizer& tok);
        void end_path();

        // The following functions are essentially a "reflection" of
        // the respective SVG path commands.
        void move_to(double x, double y, bool rel=false);   // M, m
        void line_to(double x,  double y, bool rel=false);  // L, l
        void hline_to(double x, bool rel=false);            // H, h
        void vline_to(double y, bool rel=false);            // V, v
        void curve3(double x1, double y1,                   // Q, q
                    double x,  double y, bool rel=false);
        void curve3(double x, double y, bool rel=false);    // T, t
        void curve4(double x1, double y1,                   // C, c
                    double x2, double y2, 
                    double x,  double y, bool rel=false);
        void curve4(double x2, double y2,                   // S, s
                    double x,  double y, bool rel=false);
        void close_subpath();                               // Z, z

        template<class VertexSource> 
        void add_path(VertexSource& vs, 
                      unsigned path_id = 0, 
                      bool solid_path = true)
        {
            m_storage.add_path(vs, path_id, solid_path);
        }


        

        // Call these functions on <g> tag (start_element, end_element respectively)
        void push_attr();
        void pop_attr();

        // Attribute setting functions.
        void fill(const rgba8& f);
        void stroke(const rgba8& s);
        void even_odd(bool flag);
        void stroke_width(double w);
        void fill_none();
        void stroke_none();
        void fill_opacity(double op);
        void stroke_opacity(double op);
        void line_join(line_join_e join);
        void line_cap(line_cap_e cap);
        void miter_limit(double ml);
        trans_affine& transform();

        // Make all polygons CCW-oriented
        void arrange_orientations()
        {
            m_storage.arrange_orientations_all_paths(path_flags_ccw);
        }

        // Expand all polygons 
        void expand(double value)
        {
            m_curved_trans_contour.width(value);
        }

        unsigned operator [](unsigned idx)
        {
            m_transform = m_attr_storage[idx].transform;
            return m_attr_storage[idx].index;
        }

        void bounding_rect(double* x1, double* y1, double* x2, double* y2)
        {
            agg::bounding_rect(m_curved_trans, *this, 0, m_attr_storage.size(), x1, y1, x2, y2);
        }

        // Rendering. One can specify two additional parameters: 
        // trans_affine and opacity. They can be used to transform the whole
        // image and/or to make it translucent.
        template<class Rasterizer, class Scanline, class Renderer> 
        void render(Rasterizer& ras, 
                    Scanline& sl,
                    Renderer& ren, 
                    const trans_affine& mtx, 
                    const rect& cb,
                    double opacity=1.0)
        {
            unsigned i;

            ras.clip_box(cb.x1, cb.y1, cb.x2, cb.y2);

            for(i = 0; i < m_attr_storage.size(); i++)
            {
                const path_attributes& attr = m_attr_storage[i];
                m_transform = attr.transform;
                m_transform *= mtx;
                m_curved.approximation_scale(pow(m_transform.scale(), 0.75));

                rgba8 color;

                if(attr.fill_flag)
                {
                    ras.reset();
                    ras.filling_rule(attr.even_odd_flag ? fill_even_odd : fill_non_zero);
                    if(fabs(m_curved_trans_contour.width()) < 0.0001)
                    {
                        ras.add_path(m_curved_trans, attr.index);
                    }
                    else
                    {
                        m_curved_trans_contour.miter_limit(attr.miter_limit);
                        ras.add_path(m_curved_trans_contour, attr.index);
                    }

                    color = attr.fill_color;
                    color.opacity(color.opacity() * opacity);
                    ren.color(color);
                    agg::render_scanlines(ras, sl, ren);
                }

                if(attr.stroke_flag)
                {
                    m_curved_stroked.width(attr.stroke_width);
                    m_curved_stroked.line_join(attr.line_join);
                    m_curved_stroked.line_cap(attr.line_cap);
                    m_curved_stroked.miter_limit(attr.miter_limit);
                    ras.reset();
                    ras.filling_rule(fill_non_zero);
                    ras.add_path(m_curved_stroked_trans, attr.index);
                    color = attr.stroke_color;
                    color.opacity(color.opacity() * opacity);
                    ren.color(color);
                    agg::render_scanlines(ras, sl, ren);
                }
            }
        }

    private:
        path_attributes& cur_attr();

        path_storage   m_storage;
        attr_storage   m_attr_storage;
        attr_storage   m_attr_stack;
        trans_affine   m_transform;

        curved                       m_curved;

        curved_stroked               m_curved_stroked;
        curved_stroked_trans         m_curved_stroked_trans;

        curved_trans                 m_curved_trans;
        curved_trans_contour         m_curved_trans_contour;
    };

}
}

#endif
