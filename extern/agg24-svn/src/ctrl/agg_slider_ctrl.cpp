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

#include <string.h>
#include <stdio.h>
#include "ctrl/agg_slider_ctrl.h"

namespace agg
{

    //------------------------------------------------------------------------
    slider_ctrl_impl::slider_ctrl_impl(double x1, double y1, 
                                       double x2, double y2, bool flip_y) :
        ctrl(x1, y1, x2, y2, flip_y),
        m_border_width(1.0),
        m_border_extra((y2 - y1) / 2),
        m_text_thickness(1.0),
        m_pdx(0.0),
        m_mouse_move(false),
        m_value(0.5),
        m_preview_value(0.5),
        m_min(0.0),
        m_max(1.0),
        m_num_steps(0),
        m_descending(false),
        m_text_poly(m_text)
    {
        m_label[0] = 0;
        calc_box();
    }


    //------------------------------------------------------------------------
    void slider_ctrl_impl::calc_box()
    {
        m_xs1 = m_x1 + m_border_width;
        m_ys1 = m_y1 + m_border_width;
        m_xs2 = m_x2 - m_border_width;
        m_ys2 = m_y2 - m_border_width;
    }


    //------------------------------------------------------------------------
    bool slider_ctrl_impl::normalize_value(bool preview_value_flag)
    {
        bool ret = true;
        if(m_num_steps)
        {
            int step = int(m_preview_value * m_num_steps + 0.5);
            ret = m_value != step / double(m_num_steps);
            m_value = step / double(m_num_steps);
        }
        else
        {
            m_value = m_preview_value;
        }

        if(preview_value_flag)
        {
            m_preview_value = m_value;
        }
        return ret;
    }


    //------------------------------------------------------------------------
    void slider_ctrl_impl::border_width(double t, double extra)
    { 
        m_border_width = t; 
        m_border_extra = extra;
        calc_box(); 
    }


    //------------------------------------------------------------------------
    void slider_ctrl_impl::value(double value) 
    { 
        m_preview_value = (value - m_min) / (m_max - m_min); 
        if(m_preview_value > 1.0) m_preview_value = 1.0;
        if(m_preview_value < 0.0) m_preview_value = 0.0;
        normalize_value(true);
    }

    //------------------------------------------------------------------------
    void slider_ctrl_impl::label(const char* fmt)
    {
        m_label[0] = 0;
        if(fmt)
        {
            unsigned len = strlen(fmt);
            if(len > 63) len = 63;
            memcpy(m_label, fmt, len);
            m_label[len] = 0;
        }
    }

    //------------------------------------------------------------------------
    void slider_ctrl_impl::rewind(unsigned idx)
    {
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

        case 1:                 // Triangle
            m_vertex = 0;
            if(m_descending)
            {
                m_vx[0] = m_x1; 
                m_vy[0] = m_y1;
                m_vx[1] = m_x2; 
                m_vy[1] = m_y1;
                m_vx[2] = m_x1; 
                m_vy[2] = m_y2;
                m_vx[3] = m_x1; 
                m_vy[3] = m_y1;
            }
            else
            {
                m_vx[0] = m_x1; 
                m_vy[0] = m_y1;
                m_vx[1] = m_x2; 
                m_vy[1] = m_y1;
                m_vx[2] = m_x2; 
                m_vy[2] = m_y2;
                m_vx[3] = m_x1; 
                m_vy[3] = m_y1;
            }
            break;

        case 2:
            m_text.text(m_label);
            if(m_label[0])
            {
                char buf[256];
                sprintf(buf, m_label, value());
                m_text.text(buf);
            }
            m_text.start_point(m_x1, m_y1);
            m_text.size((m_y2 - m_y1) * 1.2, m_y2 - m_y1);
            m_text_poly.width(m_text_thickness);
            m_text_poly.line_join(round_join);
            m_text_poly.line_cap(round_cap);
            m_text_poly.rewind(0);
            break;

        case 3:                 // pointer preview
            m_ellipse.init(m_xs1 + (m_xs2 - m_xs1) * m_preview_value,
                           (m_ys1 + m_ys2) / 2.0,
                           m_y2 - m_y1,
                           m_y2 - m_y1, 
                           32);
            break;


        case 4:                 // pointer
            normalize_value(false);
            m_ellipse.init(m_xs1 + (m_xs2 - m_xs1) * m_value,
                           (m_ys1 + m_ys2) / 2.0,
                           m_y2 - m_y1,
                           m_y2 - m_y1, 
                           32);
            m_ellipse.rewind(0);
            break;

        case 5:
            m_storage.remove_all();
            if(m_num_steps)
            {
                unsigned i;
                double d = (m_xs2 - m_xs1) / m_num_steps;
                if(d > 0.004) d = 0.004;
                for(i = 0; i < m_num_steps + 1; i++)
                {
                    double x = m_xs1 + (m_xs2 - m_xs1) * i / m_num_steps;
                    m_storage.move_to(x, m_y1);
                    m_storage.line_to(x - d * (m_x2 - m_x1), m_y1 - m_border_extra);
                    m_storage.line_to(x + d * (m_x2 - m_x1), m_y1 - m_border_extra);
                }
            }
        }
    }


    //------------------------------------------------------------------------
    unsigned slider_ctrl_impl::vertex(double* x, double* y)
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
            if(m_vertex == 0) cmd = path_cmd_move_to;
            if(m_vertex >= 4) cmd = path_cmd_stop;
            *x = m_vx[m_vertex];
            *y = m_vy[m_vertex];
             m_vertex++;
            break;

        case 2:
            cmd = m_text_poly.vertex(x, y);
            break;

        case 3:
        case 4:
            cmd = m_ellipse.vertex(x, y);
            break;

        case 5:
            cmd = m_storage.vertex(x, y);
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
    bool slider_ctrl_impl::in_rect(double x, double y) const
    {
        inverse_transform_xy(&x, &y);
        return x >= m_x1 && x <= m_x2 && y >= m_y1 && y <= m_y2;
    }


    //------------------------------------------------------------------------
    bool slider_ctrl_impl::on_mouse_button_down(double x, double y)
    {
        inverse_transform_xy(&x, &y);

        double xp = m_xs1 + (m_xs2 - m_xs1) * m_value;
        double yp = (m_ys1 + m_ys2) / 2.0;

        if(calc_distance(x, y, xp, yp) <= m_y2 - m_y1)
        {
            m_pdx = xp - x;
            m_mouse_move = true;
            return true;
        }
        return false;
    }


    //------------------------------------------------------------------------
    bool slider_ctrl_impl::on_mouse_move(double x, double y, bool button_flag)
    {
        inverse_transform_xy(&x, &y);
        if(!button_flag)
        {
            on_mouse_button_up(x, y);
            return false;
        }

        if(m_mouse_move)
        {
            double xp = x + m_pdx;
            m_preview_value = (xp - m_xs1) / (m_xs2 - m_xs1);
            if(m_preview_value < 0.0) m_preview_value = 0.0;
            if(m_preview_value > 1.0) m_preview_value = 1.0;
            return true;
        }
        return false;
    }


    //------------------------------------------------------------------------
    bool slider_ctrl_impl::on_mouse_button_up(double, double)
    {
        m_mouse_move = false;
        normalize_value(true);
        return true;
    }


    //------------------------------------------------------------------------
    bool slider_ctrl_impl::on_arrow_keys(bool left, bool right, bool down, bool up)
    {
        double d = 0.005;
        if(m_num_steps)
        {
            d = 1.0 / m_num_steps;
        }
        
        if(right || up)
        {
            m_preview_value += d;
            if(m_preview_value > 1.0) m_preview_value = 1.0;
            normalize_value(true);
            return true;
        }

        if(left || down)
        {
            m_preview_value -= d;
            if(m_preview_value < 0.0) m_preview_value = 0.0;
            normalize_value(true);
            return true;
        }
        return false;
    }

}

