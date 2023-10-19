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
// classes slider_ctrl_impl, slider_ctrl
//
//----------------------------------------------------------------------------

#ifndef AGG_SLIDER_CTRL_INCLUDED
#define AGG_SLIDER_CTRL_INCLUDED

#include "agg_basics.h"
#include "agg_math.h"
#include "agg_ellipse.h"
#include "agg_trans_affine.h"
#include "agg_color_rgba.h"
#include "agg_gsv_text.h"
#include "agg_conv_stroke.h"
#include "agg_path_storage.h"
#include "agg_ctrl.h"


namespace agg
{

    //--------------------------------------------------------slider_ctrl_impl
    class slider_ctrl_impl : public ctrl
    {
    public:
        slider_ctrl_impl(double x1, double y1, double x2, double y2, bool flip_y=false);

        void border_width(double t, double extra=0.0);

        void range(double min, double max) { m_min = min; m_max = max; }
        void num_steps(unsigned num) { m_num_steps = num; }
        void label(const char* fmt);
        void text_thickness(double t) { m_text_thickness = t; }

        bool descending() const { return m_descending; }
        void descending(bool v) { m_descending = v; }

        double value() const { return m_value * (m_max - m_min) + m_min; }
        void value(double value);

        virtual bool in_rect(double x, double y) const;
        virtual bool on_mouse_button_down(double x, double y);
        virtual bool on_mouse_button_up(double x, double y);
        virtual bool on_mouse_move(double x, double y, bool button_flag);
        virtual bool on_arrow_keys(bool left, bool right, bool down, bool up);

        // Vertex source interface
        unsigned num_paths() { return 6; };
        void     rewind(unsigned path_id);
        unsigned vertex(double* x, double* y);

    private:
        void calc_box();
        bool normalize_value(bool preview_value_flag);

        double   m_border_width;
        double   m_border_extra;
        double   m_text_thickness;
        double   m_value;
        double   m_preview_value;
        double   m_min;
        double   m_max;
        unsigned m_num_steps;
        bool     m_descending;
        char     m_label[64];
        double   m_xs1;
        double   m_ys1;
        double   m_xs2;
        double   m_ys2;
        double   m_pdx;
        bool     m_mouse_move;
        double   m_vx[32];
        double   m_vy[32];

        ellipse  m_ellipse;

        unsigned m_idx;
        unsigned m_vertex;

        gsv_text              m_text;
        conv_stroke<gsv_text> m_text_poly;
        path_storage          m_storage;

    };



    //----------------------------------------------------------slider_ctrl
    template<class ColorT> class slider_ctrl : public slider_ctrl_impl
    {
    public:
        slider_ctrl(double x1, double y1, double x2, double y2, bool flip_y=false) :
            slider_ctrl_impl(x1, y1, x2, y2, flip_y),
            m_background_color(rgba(1.0, 0.9, 0.8)),
            m_triangle_color(rgba(0.7, 0.6, 0.6)),
            m_text_color(rgba(0.0, 0.0, 0.0)),
            m_pointer_preview_color(rgba(0.6, 0.4, 0.4, 0.4)),
            m_pointer_color(rgba(0.8, 0.0, 0.0, 0.6))
        {
            m_colors[0] = &m_background_color;
            m_colors[1] = &m_triangle_color;
            m_colors[2] = &m_text_color;
            m_colors[3] = &m_pointer_preview_color;
            m_colors[4] = &m_pointer_color;
            m_colors[5] = &m_text_color;
        }
          

        void background_color(const ColorT& c) { m_background_color = c; }
        void pointer_color(const ColorT& c) { m_pointer_color = c; }

        const ColorT& color(unsigned i) const { return *m_colors[i]; } 

    private:
        slider_ctrl(const slider_ctrl<ColorT>&);
        const slider_ctrl<ColorT>& operator = (const slider_ctrl<ColorT>&);

        ColorT m_background_color;
        ColorT m_triangle_color;
        ColorT m_text_color;
        ColorT m_pointer_preview_color;
        ColorT m_pointer_color;
        ColorT* m_colors[6];
    };





}



#endif

