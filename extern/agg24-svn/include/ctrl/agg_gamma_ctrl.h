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
// class gamma_ctrl
//
//----------------------------------------------------------------------------

#ifndef AGG_GAMMA_CTRL_INCLUDED
#define AGG_GAMMA_CTRL_INCLUDED

#include "agg_basics.h"
#include "agg_gamma_spline.h"
#include "agg_ellipse.h"
#include "agg_conv_stroke.h"
#include "agg_gsv_text.h"
#include "agg_trans_affine.h"
#include "agg_color_rgba.h"
#include "agg_ctrl.h"

namespace agg
{
    //------------------------------------------------------------------------
    // Class that can be used to create an interactive control to set up 
    // gamma arrays.
    //------------------------------------------------------------------------
    class gamma_ctrl_impl : public ctrl
    {
    public:
        gamma_ctrl_impl(double x1, double y1, double x2, double y2, bool flip_y=false);

        // Set other parameters
        void border_width(double t, double extra=0.0);
        void curve_width(double t)         { m_curve_width = t; }
        void grid_width(double t)          { m_grid_width = t; }
        void text_thickness(double t)      { m_text_thickness = t; }
        void text_size(double h, double w=0.0);
        void point_size(double s)              { m_point_size = s; }

        // Event handlers. Just call them if the respective events
        // in your system occure. The functions return true if redrawing
        // is required.
        virtual bool in_rect(double x, double y) const;
        virtual bool on_mouse_button_down(double x, double y);
        virtual bool on_mouse_button_up(double x, double y);
        virtual bool on_mouse_move(double x, double y, bool button_flag);
        virtual bool on_arrow_keys(bool left, bool right, bool down, bool up);

        void change_active_point();

        // A copy of agg::gamma_spline interface
        void  values(double kx1, double ky1, double kx2, double ky2);
        void  values(double* kx1, double* ky1, double* kx2, double* ky2) const;
        const unsigned char* gamma() const { return m_gamma_spline.gamma(); }
        double y(double x) const { return m_gamma_spline.y(x); }
        double operator() (double x) const { return m_gamma_spline.y(x); }
        const gamma_spline& get_gamma_spline() const { return m_gamma_spline; }

        // Vertex soutce interface
        unsigned num_paths() { return 7; }
        void     rewind(unsigned idx);
        unsigned vertex(double* x, double* y);

    private:
        void calc_spline_box();
        void calc_points();
        void calc_values();

        gamma_spline  m_gamma_spline;
        double m_border_width;
        double m_border_extra;
        double m_curve_width;
        double m_grid_width;
        double m_text_thickness;
        double m_point_size;
        double m_text_height;
        double m_text_width;
        double m_xc1;
        double m_yc1;
        double m_xc2;
        double m_yc2;
        double m_xs1;
        double m_ys1;
        double m_xs2;
        double m_ys2;
        double m_xt1;
        double m_yt1;
        double m_xt2;
        double m_yt2;
        conv_stroke<gamma_spline> m_curve_poly;
        ellipse                   m_ellipse;
        gsv_text                  m_text;
        conv_stroke<gsv_text>     m_text_poly;
        unsigned m_idx;
        unsigned m_vertex;
        double   m_vx[32];
        double   m_vy[32];
        double   m_xp1;
        double   m_yp1;
        double   m_xp2;
        double   m_yp2;
        bool     m_p1_active;
        unsigned m_mouse_point;
        double   m_pdx;
        double   m_pdy;
    };



    template<class ColorT> class gamma_ctrl : public gamma_ctrl_impl
    {
    public:
        gamma_ctrl(double x1, double y1, double x2, double y2, bool flip_y=false) :
            gamma_ctrl_impl(x1, y1, x2, y2, flip_y),
            m_background_color(rgba(1.0, 1.0, 0.9)),
            m_border_color(rgba(0.0, 0.0, 0.0)),
            m_curve_color(rgba(0.0, 0.0, 0.0)),
            m_grid_color(rgba(0.2, 0.2, 0.0)),
            m_inactive_pnt_color(rgba(0.0, 0.0, 0.0)),
            m_active_pnt_color(rgba(1.0, 0.0, 0.0)),
            m_text_color(rgba(0.0, 0.0, 0.0))
        {
            m_colors[0] = &m_background_color;
            m_colors[1] = &m_border_color;
            m_colors[2] = &m_curve_color;
            m_colors[3] = &m_grid_color;
            m_colors[4] = &m_inactive_pnt_color;
            m_colors[5] = &m_active_pnt_color;
            m_colors[6] = &m_text_color;
        }

        // Set colors
        void background_color(const ColorT& c)   { m_background_color = c; }
        void border_color(const ColorT& c)       { m_border_color = c; }
        void curve_color(const ColorT& c)        { m_curve_color = c; }
        void grid_color(const ColorT& c)         { m_grid_color = c; }
        void inactive_pnt_color(const ColorT& c) { m_inactive_pnt_color = c; }
        void active_pnt_color(const ColorT& c)   { m_active_pnt_color = c; }
        void text_color(const ColorT& c)         { m_text_color = c; }
        const ColorT& color(unsigned i) const { return *m_colors[i]; } 

    private:
        gamma_ctrl(const gamma_ctrl<ColorT>&);
        const gamma_ctrl<ColorT>& operator = (const gamma_ctrl<ColorT>&);

        ColorT  m_background_color;
        ColorT  m_border_color;
        ColorT  m_curve_color;
        ColorT  m_grid_color;
        ColorT  m_inactive_pnt_color;
        ColorT  m_active_pnt_color;
        ColorT  m_text_color;
        ColorT* m_colors[7];
    };


}

#endif
