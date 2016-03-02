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
// classes rbox_ctrl_impl, rbox_ctrl
//
//----------------------------------------------------------------------------

#ifndef AGG_RBOX_CTRL_INCLUDED
#define AGG_RBOX_CTRL_INCLUDED

#include "agg_array.h"
#include "agg_ellipse.h"
#include "agg_conv_stroke.h"
#include "agg_gsv_text.h"
#include "agg_trans_affine.h"
#include "agg_color_rgba.h"
#include "agg_ctrl.h"



namespace agg
{

    //------------------------------------------------------------------------
    class rbox_ctrl_impl : public ctrl
    {
    public:
        rbox_ctrl_impl(double x1, double y1, double x2, double y2, bool flip_y=false);

        void border_width(double t, double extra=0.0);
        void text_thickness(double t)  { m_text_thickness = t; }
        void text_size(double h, double w=0.0);

        void add_item(const char* text);
        int  cur_item() const { return m_cur_item; }
        void cur_item(int i) { m_cur_item = i; }

        virtual bool in_rect(double x, double y) const;
        virtual bool on_mouse_button_down(double x, double y);
        virtual bool on_mouse_button_up(double x, double y);
        virtual bool on_mouse_move(double x, double y, bool button_flag);
        virtual bool on_arrow_keys(bool left, bool right, bool down, bool up);

        // Vertex soutce interface
        unsigned num_paths() { return 5; };
        void     rewind(unsigned path_id);
        unsigned vertex(double* x, double* y);

    private:
        void calc_rbox();

        double          m_border_width;
        double          m_border_extra;
        double          m_text_thickness;
        double          m_text_height;
        double          m_text_width;
        pod_array<char> m_items[32];
        unsigned        m_num_items;
        int             m_cur_item;

        double   m_xs1;
        double   m_ys1;
        double   m_xs2;
        double   m_ys2;

        double   m_vx[32];
        double   m_vy[32];
        unsigned m_draw_item;
        double   m_dy;

        ellipse               m_ellipse;
        conv_stroke<ellipse>  m_ellipse_poly;
        gsv_text              m_text;
        conv_stroke<gsv_text> m_text_poly;

        unsigned m_idx;
        unsigned m_vertex;
    };



    //------------------------------------------------------------------------
    template<class ColorT> class rbox_ctrl : public rbox_ctrl_impl
    {
    public:
        rbox_ctrl(double x1, double y1, double x2, double y2, bool flip_y=false) :
            rbox_ctrl_impl(x1, y1, x2, y2, flip_y),
            m_background_color(rgba(1.0, 1.0, 0.9)),
            m_border_color(rgba(0.0, 0.0, 0.0)),
            m_text_color(rgba(0.0, 0.0, 0.0)),
            m_inactive_color(rgba(0.0, 0.0, 0.0)),
            m_active_color(rgba(0.4, 0.0, 0.0))
        {
            m_colors[0] = &m_background_color;
            m_colors[1] = &m_border_color;
            m_colors[2] = &m_text_color;
            m_colors[3] = &m_inactive_color;
            m_colors[4] = &m_active_color;
        }
          

        void background_color(const ColorT& c) { m_background_color = c; }
        void border_color(const ColorT& c) { m_border_color = c; }
        void text_color(const ColorT& c) { m_text_color = c; }
        void inactive_color(const ColorT& c) { m_inactive_color = c; }
        void active_color(const ColorT& c) { m_active_color = c; }

        const ColorT& color(unsigned i) const { return *m_colors[i]; } 

    private:
        rbox_ctrl(const rbox_ctrl<ColorT>&);
        const rbox_ctrl<ColorT>& operator = (const rbox_ctrl<ColorT>&);
       
        ColorT m_background_color;
        ColorT m_border_color;
        ColorT m_text_color;
        ColorT m_inactive_color;
        ColorT m_active_color;
        ColorT* m_colors[5];
    };



}



#endif

