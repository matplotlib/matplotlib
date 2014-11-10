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
// class gamma_ctrl_impl
//
//----------------------------------------------------------------------------

#include <stdio.h>
#include "agg_math.h"
#include "ctrl/agg_gamma_ctrl.h"

namespace agg
{

    //------------------------------------------------------------------------
    gamma_ctrl_impl::gamma_ctrl_impl(double x1, double y1, double x2, double y2, bool flip_y) :
        ctrl(x1, y1, x2, y2, flip_y),
        m_border_width(2.0),
        m_border_extra(0.0),
        m_curve_width(2.0),
        m_grid_width(0.2),
        m_text_thickness(1.5),
        m_point_size(5.0),
        m_text_height(9.0),
        m_text_width(0.0),
        m_xc1(x1),
        m_yc1(y1),
        m_xc2(x2),
        m_yc2(y2 - m_text_height * 2.0),
        m_xt1(x1),
        m_yt1(y2 - m_text_height * 2.0),
        m_xt2(x2),
        m_yt2(y2),
        m_curve_poly(m_gamma_spline),
        m_text_poly(m_text),
        m_idx(0),
        m_vertex(0),
        m_p1_active(true),
        m_mouse_point(0),
        m_pdx(0.0),
        m_pdy(0.0)
    {
        calc_spline_box();
    }


    //------------------------------------------------------------------------
    void gamma_ctrl_impl::calc_spline_box()
    {
        m_xs1 = m_xc1 + m_border_width;
        m_ys1 = m_yc1 + m_border_width;
        m_xs2 = m_xc2 - m_border_width;
        m_ys2 = m_yc2 - m_border_width * 0.5;
    }


    //------------------------------------------------------------------------
    void gamma_ctrl_impl::calc_points()
    {
        double kx1, ky1, kx2, ky2;
        m_gamma_spline.values(&kx1, &ky1, &kx2, &ky2);
        m_xp1 = m_xs1 + (m_xs2 - m_xs1) * kx1 * 0.25;
        m_yp1 = m_ys1 + (m_ys2 - m_ys1) * ky1 * 0.25;
        m_xp2 = m_xs2 - (m_xs2 - m_xs1) * kx2 * 0.25;
        m_yp2 = m_ys2 - (m_ys2 - m_ys1) * ky2 * 0.25;
    }


    //------------------------------------------------------------------------
    void gamma_ctrl_impl::calc_values()
    {
        double kx1, ky1, kx2, ky2;

        kx1 = (m_xp1 - m_xs1) * 4.0 / (m_xs2 - m_xs1);
        ky1 = (m_yp1 - m_ys1) * 4.0 / (m_ys2 - m_ys1);
        kx2 = (m_xs2 - m_xp2) * 4.0 / (m_xs2 - m_xs1);
        ky2 = (m_ys2 - m_yp2) * 4.0 / (m_ys2 - m_ys1);
        m_gamma_spline.values(kx1, ky1, kx2, ky2);
    }


    //------------------------------------------------------------------------
    void gamma_ctrl_impl::text_size(double h, double w) 
    { 
        m_text_width = w; 
        m_text_height = h; 
        m_yc2 = m_y2 - m_text_height * 2.0;
        m_yt1 = m_y2 - m_text_height * 2.0;
        calc_spline_box();
    }


    //------------------------------------------------------------------------
    void gamma_ctrl_impl::border_width(double t, double extra)
    { 
        m_border_width = t; 
        m_border_extra = extra;
        calc_spline_box(); 
    }

    //------------------------------------------------------------------------
    void gamma_ctrl_impl::values(double kx1, double ky1, double kx2, double ky2)
    {
        m_gamma_spline.values(kx1, ky1, kx2, ky2);
    }


    //------------------------------------------------------------------------
    void gamma_ctrl_impl::values(double* kx1, double* ky1, double* kx2, double* ky2) const
    {
        m_gamma_spline.values(kx1, ky1, kx2, ky2);
    }

    //------------------------------------------------------------------------
    void  gamma_ctrl_impl::rewind(unsigned idx)
    {
        double kx1, ky1, kx2, ky2;
        char tbuf[32];

        m_idx = idx;

        switch(idx)
        {
        default:

        case 0:                 // Background
            m_vertex = 0;
            m_vx[0] = m_x1 - m_border_extra; 
            m_vy[0] = m_y1 - m_border_extra;
            m_vx[1] = m_x2 + m_border_extra; 
            m_vy[1] = m_y1 - m_border_extra;
            m_vx[2] = m_x2 + m_border_extra; 
            m_vy[2] = m_y2 + m_border_extra;
            m_vx[3] = m_x1 - m_border_extra; 
            m_vy[3] = m_y2 + m_border_extra;
            break;

        case 1:                 // Border
            m_vertex = 0;
            m_vx[0] = m_x1; 
            m_vy[0] = m_y1;
            m_vx[1] = m_x2; 
            m_vy[1] = m_y1;
            m_vx[2] = m_x2; 
            m_vy[2] = m_y2;
            m_vx[3] = m_x1; 
            m_vy[3] = m_y2;
            m_vx[4] = m_x1 + m_border_width; 
            m_vy[4] = m_y1 + m_border_width; 
            m_vx[5] = m_x1 + m_border_width; 
            m_vy[5] = m_y2 - m_border_width; 
            m_vx[6] = m_x2 - m_border_width; 
            m_vy[6] = m_y2 - m_border_width; 
            m_vx[7] = m_x2 - m_border_width; 
            m_vy[7] = m_y1 + m_border_width; 
            m_vx[8] = m_xc1 + m_border_width;
            m_vy[8] = m_yc2 - m_border_width * 0.5;
            m_vx[9] = m_xc2 - m_border_width;
            m_vy[9] = m_yc2 - m_border_width * 0.5;
            m_vx[10] = m_xc2 - m_border_width;
            m_vy[10] = m_yc2 + m_border_width * 0.5;
            m_vx[11] = m_xc1 + m_border_width;
            m_vy[11] = m_yc2 + m_border_width * 0.5;
            break;

        case 2:                 // Curve
            m_gamma_spline.box(m_xs1, m_ys1, m_xs2, m_ys2);
            m_curve_poly.width(m_curve_width);
            m_curve_poly.rewind(0);
            break;

        case 3:                 // Grid
            m_vertex = 0;
            m_vx[0] = m_xs1;
            m_vy[0] = (m_ys1 + m_ys2) * 0.5 - m_grid_width * 0.5;
            m_vx[1] = m_xs2;
            m_vy[1] = (m_ys1 + m_ys2) * 0.5 - m_grid_width * 0.5;
            m_vx[2] = m_xs2;
            m_vy[2] = (m_ys1 + m_ys2) * 0.5 + m_grid_width * 0.5;
            m_vx[3] = m_xs1;
            m_vy[3] = (m_ys1 + m_ys2) * 0.5 + m_grid_width * 0.5;
            m_vx[4] = (m_xs1 + m_xs2) * 0.5 - m_grid_width * 0.5;
            m_vy[4] = m_ys1;
            m_vx[5] = (m_xs1 + m_xs2) * 0.5 - m_grid_width * 0.5;
            m_vy[5] = m_ys2;
            m_vx[6] = (m_xs1 + m_xs2) * 0.5 + m_grid_width * 0.5;
            m_vy[6] = m_ys2;
            m_vx[7] = (m_xs1 + m_xs2) * 0.5 + m_grid_width * 0.5;
            m_vy[7] = m_ys1;
            calc_points();
            m_vx[8] = m_xs1;
            m_vy[8] = m_yp1 - m_grid_width * 0.5;
            m_vx[9] = m_xp1 - m_grid_width * 0.5;
            m_vy[9] = m_yp1 - m_grid_width * 0.5;
            m_vx[10] = m_xp1 - m_grid_width * 0.5;
            m_vy[10] = m_ys1;
            m_vx[11] = m_xp1 + m_grid_width * 0.5;
            m_vy[11] = m_ys1;
            m_vx[12] = m_xp1 + m_grid_width * 0.5;
            m_vy[12] = m_yp1 + m_grid_width * 0.5;
            m_vx[13] = m_xs1;
            m_vy[13] = m_yp1 + m_grid_width * 0.5;
            m_vx[14] = m_xs2;
            m_vy[14] = m_yp2 + m_grid_width * 0.5;
            m_vx[15] = m_xp2 + m_grid_width * 0.5;
            m_vy[15] = m_yp2 + m_grid_width * 0.5;
            m_vx[16] = m_xp2 + m_grid_width * 0.5;
            m_vy[16] = m_ys2;
            m_vx[17] = m_xp2 - m_grid_width * 0.5;
            m_vy[17] = m_ys2;
            m_vx[18] = m_xp2 - m_grid_width * 0.5;
            m_vy[18] = m_yp2 - m_grid_width * 0.5;
            m_vx[19] = m_xs2;
            m_vy[19] = m_yp2 - m_grid_width * 0.5;
            break;

        case 4:                 // Point1
            calc_points();
            if(m_p1_active) m_ellipse.init(m_xp2, m_yp2, m_point_size, m_point_size, 32);
            else            m_ellipse.init(m_xp1, m_yp1, m_point_size, m_point_size, 32);
            break;

        case 5:                 // Point2
            calc_points();
            if(m_p1_active) m_ellipse.init(m_xp1, m_yp1, m_point_size, m_point_size, 32);
            else            m_ellipse.init(m_xp2, m_yp2, m_point_size, m_point_size, 32);
            break;

        case 6:                 // Text
            m_gamma_spline.values(&kx1, &ky1, &kx2, &ky2);
            sprintf(tbuf, "%5.3f %5.3f %5.3f %5.3f", kx1, ky1, kx2, ky2);
            m_text.text(tbuf);
            m_text.size(m_text_height, m_text_width);
            m_text.start_point(m_xt1 + m_border_width * 2.0, (m_yt1 + m_yt2) * 0.5 - m_text_height * 0.5);
            m_text_poly.width(m_text_thickness);
            m_text_poly.line_join(round_join);
            m_text_poly.line_cap(round_cap);
            m_text_poly.rewind(0);
            break;
        }
    }


    //------------------------------------------------------------------------
    unsigned gamma_ctrl_impl::vertex(double* x, double* y)
    {
        unsigned cmd = path_cmd_line_to;
        switch(m_idx)
        {
        case 0:
            if(m_vertex == 0) cmd = path_cmd_move_to;
            if(m_vertex >= 4) cmd = path_cmd_stop;
            *x = m_vx[m_vertex];
            *y = m_vy[m_vertex];
            m_vertex++;
            break;

        case 1:
            if(m_vertex == 0 || m_vertex == 4 || m_vertex == 8) cmd = path_cmd_move_to;
            if(m_vertex >= 12) cmd = path_cmd_stop;
            *x = m_vx[m_vertex];
            *y = m_vy[m_vertex];
            m_vertex++;
            break;

        case 2:
            cmd = m_curve_poly.vertex(x, y);
            break;

        case 3:
            if(m_vertex == 0  || 
               m_vertex == 4  || 
               m_vertex == 8  ||
               m_vertex == 14) cmd = path_cmd_move_to;

            if(m_vertex >= 20) cmd = path_cmd_stop;
            *x = m_vx[m_vertex];
            *y = m_vy[m_vertex];
            m_vertex++;
            break;

        case 4:                 // Point1
        case 5:                 // Point2
            cmd = m_ellipse.vertex(x, y);
            break;

        case 6:
            cmd = m_text_poly.vertex(x, y);
            break;

        default:
            cmd = path_cmd_stop;
            break;
        }

        if(!is_stop(cmd))
        {
            transform_xy(x, y);
        }

        return cmd;
    }



    //------------------------------------------------------------------------
    bool gamma_ctrl_impl::on_arrow_keys(bool left, bool right, bool down, bool up)
    {
        double kx1, ky1, kx2, ky2;
        bool ret = false;
        m_gamma_spline.values(&kx1, &ky1, &kx2, &ky2);
        if(m_p1_active)
        {
            if(left)  { kx1 -= 0.005; ret = true; }
            if(right) { kx1 += 0.005; ret = true; }
            if(down)  { ky1 -= 0.005; ret = true; }
            if(up)    { ky1 += 0.005; ret = true; }
        }
        else
        {
            if(left)  { kx2 += 0.005; ret = true; }
            if(right) { kx2 -= 0.005; ret = true; }
            if(down)  { ky2 += 0.005; ret = true; }
            if(up)    { ky2 -= 0.005; ret = true; }
        }
        if(ret)
        {
            m_gamma_spline.values(kx1, ky1, kx2, ky2);
        }
        return ret;
    }


    
    //------------------------------------------------------------------------
    void gamma_ctrl_impl::change_active_point()
    {
        m_p1_active = m_p1_active ? false : true;
    }




    //------------------------------------------------------------------------
    bool gamma_ctrl_impl::in_rect(double x, double y) const
    {
        inverse_transform_xy(&x, &y);
        return x >= m_x1 && x <= m_x2 && y >= m_y1 && y <= m_y2;
    }


    //------------------------------------------------------------------------
    bool gamma_ctrl_impl::on_mouse_button_down(double x, double y)
    {
        inverse_transform_xy(&x, &y);
        calc_points();

        if(calc_distance(x, y, m_xp1, m_yp1) <= m_point_size + 1)
        {
            m_mouse_point = 1;
            m_pdx = m_xp1 - x;
            m_pdy = m_yp1 - y;
            m_p1_active = true;
            return true;
        }

        if(calc_distance(x, y, m_xp2, m_yp2) <= m_point_size + 1)
        {
            m_mouse_point = 2;
            m_pdx = m_xp2 - x;
            m_pdy = m_yp2 - y;
            m_p1_active = false;
            return true;
        }

        return false;
    }


    //------------------------------------------------------------------------
    bool gamma_ctrl_impl::on_mouse_button_up(double, double)
    {
        if(m_mouse_point)
        {
            m_mouse_point = 0;
            return true;
        }
        return false;
    }


    //------------------------------------------------------------------------
    bool gamma_ctrl_impl::on_mouse_move(double x, double y, bool button_flag)
    {
        inverse_transform_xy(&x, &y);
        if(!button_flag)
        {
            return on_mouse_button_up(x, y);
        }

        if(m_mouse_point == 1)
        {
            m_xp1 = x + m_pdx;
            m_yp1 = y + m_pdy;
            calc_values();
            return true;
        }
        if(m_mouse_point == 2)
        {
            m_xp2 = x + m_pdx;
            m_yp2 = y + m_pdy;
            calc_values();
            return true;
        }
        return false;
    }



}

