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
// classes spline_ctrl_impl, spline_ctrl
//
//----------------------------------------------------------------------------

#include "ctrl/agg_spline_ctrl.h"


namespace agg
{

    //------------------------------------------------------------------------
    spline_ctrl_impl::spline_ctrl_impl(double x1, double y1, double x2, double y2, 
                                       unsigned num_pnt, bool flip_y) :
        ctrl(x1, y1, x2, y2, flip_y),
        m_num_pnt(num_pnt),
        m_border_width(1.0),
        m_border_extra(0.0),
        m_curve_width(1.0),
        m_point_size(3.0),
        m_curve_poly(m_curve_pnt),
        m_idx(0),
        m_vertex(0),
        m_active_pnt(-1),
        m_move_pnt(-1),
        m_pdx(0.0),
        m_pdy(0.0)
    {
        if(m_num_pnt < 4)  m_num_pnt = 4;
        if(m_num_pnt > 32) m_num_pnt = 32;

        unsigned i;
        for(i = 0; i < m_num_pnt; i++)
        {
            m_xp[i] = double(i) / double(m_num_pnt - 1);
            m_yp[i] = 0.5;
        }
        calc_spline_box();
        update_spline();
    }


    //------------------------------------------------------------------------
    void spline_ctrl_impl::border_width(double t, double extra)
    { 
        m_border_width = t; 
        m_border_extra = extra;
        calc_spline_box(); 
    }


    //------------------------------------------------------------------------
    void spline_ctrl_impl::calc_spline_box()
    {
        m_xs1 = m_x1 + m_border_width;
        m_ys1 = m_y1 + m_border_width;
        m_xs2 = m_x2 - m_border_width;
        m_ys2 = m_y2 - m_border_width;
    }


    //------------------------------------------------------------------------
    void spline_ctrl_impl::update_spline()
    {
        int i;
        m_spline.init(m_num_pnt, m_xp, m_yp);
        for(i = 0; i < 256; i++)
        {
            m_spline_values[i] = m_spline.get(double(i) / 255.0);
            if(m_spline_values[i] < 0.0) m_spline_values[i] = 0.0;
            if(m_spline_values[i] > 1.0) m_spline_values[i] = 1.0;
            m_spline_values8[i] = (int8u)(m_spline_values[i] * 255.0);
        }
    }


    //------------------------------------------------------------------------
    void spline_ctrl_impl::calc_curve()
    {
        int i;
        m_curve_pnt.remove_all();
        m_curve_pnt.move_to(m_xs1, m_ys1 + (m_ys2 - m_ys1) * m_spline_values[0]);
        for(i = 1; i < 256; i++)
        {
            m_curve_pnt.line_to(m_xs1 + (m_xs2 - m_xs1) * double(i) / 255.0, 
                                m_ys1 + (m_ys2 - m_ys1) * m_spline_values[i]);
        }
    }


    //------------------------------------------------------------------------
    double spline_ctrl_impl::calc_xp(unsigned idx)
    {
        return m_xs1 + (m_xs2 - m_xs1) * m_xp[idx];
    }


    //------------------------------------------------------------------------
    double spline_ctrl_impl::calc_yp(unsigned idx)
    {
        return m_ys1 + (m_ys2 - m_ys1) * m_yp[idx];
    }


    //------------------------------------------------------------------------
    void spline_ctrl_impl::set_xp(unsigned idx, double val)
    {
        if(val < 0.0) val = 0.0;
        if(val > 1.0) val = 1.0;

        if(idx == 0)
        {
            val = 0.0;
        }
        else if(idx == m_num_pnt - 1)
        {
            val = 1.0;
        }
        else
        {
            if(val < m_xp[idx - 1] + 0.001) val = m_xp[idx - 1] + 0.001;
            if(val > m_xp[idx + 1] - 0.001) val = m_xp[idx + 1] - 0.001;
        }
        m_xp[idx] = val;
    }

    //------------------------------------------------------------------------
    void spline_ctrl_impl::set_yp(unsigned idx, double val)
    {
        if(val < 0.0) val = 0.0;
        if(val > 1.0) val = 1.0;
        m_yp[idx] = val;
    }


    //------------------------------------------------------------------------
    void spline_ctrl_impl::point(unsigned idx, double x, double y)
    {
        if(idx < m_num_pnt) 
        {
            set_xp(idx, x);
            set_yp(idx, y);
        }
    }


    //------------------------------------------------------------------------
    void spline_ctrl_impl::value(unsigned idx, double y)
    {
        if(idx < m_num_pnt) 
        {
            set_yp(idx, y);
        }
    }

    //------------------------------------------------------------------------
    double spline_ctrl_impl::value(double x) const
    { 
        x = m_spline.get(x);
        if(x < 0.0) x = 0.0;
        if(x > 1.0) x = 1.0;
        return x;
    }


    //------------------------------------------------------------------------
    void spline_ctrl_impl::rewind(unsigned idx)
    {
        unsigned i;

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
            break;

        case 2:                 // Curve
            calc_curve();
            m_curve_poly.width(m_curve_width);
            m_curve_poly.rewind(0);
            break;


        case 3:                 // Inactive points
            m_curve_pnt.remove_all();
            for(i = 0; i < m_num_pnt; i++)
            {
                if(int(i) != m_active_pnt)
                {
                    m_ellipse.init(calc_xp(i), calc_yp(i), 
                                   m_point_size, m_point_size, 32);
                    m_curve_pnt.concat_path(m_ellipse);
                }
            }
            m_curve_poly.rewind(0);
            break;


        case 4:                 // Active point
            m_curve_pnt.remove_all();
            if(m_active_pnt >= 0)
            {
                m_ellipse.init(calc_xp(m_active_pnt), calc_yp(m_active_pnt), 
                               m_point_size, m_point_size, 32);

                m_curve_pnt.concat_path(m_ellipse);
            }
            m_curve_poly.rewind(0);
            break;

        }
    }


    //------------------------------------------------------------------------
    unsigned spline_ctrl_impl::vertex(double* x, double* y)
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
            if(m_vertex == 0 || m_vertex == 4) cmd = path_cmd_move_to;
            if(m_vertex >= 8) cmd = path_cmd_stop;
            *x = m_vx[m_vertex];
            *y = m_vy[m_vertex];
            m_vertex++;
            break;

        case 2:
            cmd = m_curve_poly.vertex(x, y);
            break;

        case 3:
        case 4:
            cmd = m_curve_pnt.vertex(x, y);
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
    void spline_ctrl_impl::active_point(int i)
    {
        m_active_pnt = i;
    }


    //------------------------------------------------------------------------
    bool spline_ctrl_impl::in_rect(double x, double y) const
    {
        inverse_transform_xy(&x, &y);
        return x >= m_x1 && x <= m_x2 && y >= m_y1 && y <= m_y2;
    }


    //------------------------------------------------------------------------
    bool spline_ctrl_impl::on_mouse_button_down(double x, double y)
    {
        inverse_transform_xy(&x, &y);
        unsigned i;
        for(i = 0; i < m_num_pnt; i++)  
        {
            double xp = calc_xp(i);
            double yp = calc_yp(i);
            if(calc_distance(x, y, xp, yp) <= m_point_size + 1)
            {
                m_pdx = xp - x;
                m_pdy = yp - y;
                m_active_pnt = m_move_pnt = int(i);
                return true;
            }
        }
        return false;
    }


    //------------------------------------------------------------------------
    bool spline_ctrl_impl::on_mouse_button_up(double, double)
    {
        if(m_move_pnt >= 0)
        {
            m_move_pnt = -1;
            return true;
        }
        return false;
    }


    //------------------------------------------------------------------------
    bool spline_ctrl_impl::on_mouse_move(double x, double y, bool button_flag)
    {
        inverse_transform_xy(&x, &y);
        if(!button_flag)
        {
            return on_mouse_button_up(x, y);
        }

        if(m_move_pnt >= 0)
        {
            double xp = x + m_pdx;
            double yp = y + m_pdy;

            set_xp(m_move_pnt, (xp - m_xs1) / (m_xs2 - m_xs1));
            set_yp(m_move_pnt, (yp - m_ys1) / (m_ys2 - m_ys1));

            update_spline();
            return true;
        }
        return false;
    }


    //------------------------------------------------------------------------
    bool spline_ctrl_impl::on_arrow_keys(bool left, bool right, bool down, bool up)
    {
        double kx = 0.0;
        double ky = 0.0;
        bool ret = false;
        if(m_active_pnt >= 0)
        {
            kx = m_xp[m_active_pnt];
            ky = m_yp[m_active_pnt];
            if(left)  { kx -= 0.001; ret = true; }
            if(right) { kx += 0.001; ret = true; }
            if(down)  { ky -= 0.001; ret = true; }
            if(up)    { ky += 0.001; ret = true; }
        }
        if(ret)
        {
            set_xp(m_active_pnt, kx);
            set_yp(m_active_pnt, ky);
            update_spline();
        }
        return ret;
    }




}

