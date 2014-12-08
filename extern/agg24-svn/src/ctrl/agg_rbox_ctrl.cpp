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

#include <string.h>
#include "ctrl/agg_rbox_ctrl.h"

namespace agg
{
  
    //------------------------------------------------------------------------
    rbox_ctrl_impl::rbox_ctrl_impl(double x1, double y1, 
                                   double x2, double y2, bool flip_y) :
        ctrl(x1, y1, x2, y2, flip_y),
        m_border_width(1.0),
        m_border_extra(0.0),
        m_text_thickness(1.5),
        m_text_height(9.0),
        m_text_width(0.0),
        m_num_items(0),
        m_cur_item(-1),
        m_ellipse_poly(m_ellipse),
        m_text_poly(m_text),
        m_idx(0),
        m_vertex(0)
    {
        calc_rbox();
    }


    //------------------------------------------------------------------------
    void rbox_ctrl_impl::calc_rbox()
    {
        m_xs1 = m_x1 + m_border_width;
        m_ys1 = m_y1 + m_border_width;
        m_xs2 = m_x2 - m_border_width;
        m_ys2 = m_y2 - m_border_width;
    }


    //------------------------------------------------------------------------
    void rbox_ctrl_impl::add_item(const char* text)
    {
        if(m_num_items < 32)
        {
            m_items[m_num_items].resize(strlen(text) + 1);
            strcpy(&m_items[m_num_items][0], text);
            m_num_items++;
        }
    }


    //------------------------------------------------------------------------
    void rbox_ctrl_impl::border_width(double t, double extra)
    { 
        m_border_width = t; 
        m_border_extra = extra;
        calc_rbox(); 
    }


    //------------------------------------------------------------------------
    void rbox_ctrl_impl::text_size(double h, double w) 
    { 
        m_text_width = w; 
        m_text_height = h; 
    }



    //------------------------------------------------------------------------
    void rbox_ctrl_impl::rewind(unsigned idx)
    {
        m_idx = idx;
        m_dy = m_text_height * 2.0;
        m_draw_item = 0;

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

        case 2:                 // Text
            m_text.text(&m_items[0][0]);
            m_text.start_point(m_xs1 + m_dy * 1.5, m_ys1 + m_dy / 2.0);
            m_text.size(m_text_height, m_text_width);
            m_text_poly.width(m_text_thickness);
            m_text_poly.line_join(round_join);
            m_text_poly.line_cap(round_cap);
            m_text_poly.rewind(0);
            break;

        case 3:                 // Inactive items
            m_ellipse.init(m_xs1 + m_dy / 1.3, 
                           m_ys1 + m_dy / 1.3,
                           m_text_height / 1.5, 
                           m_text_height / 1.5, 32);
            m_ellipse_poly.width(m_text_thickness);
            m_ellipse_poly.rewind(0);
            break;


        case 4:                 // Active Item
            if(m_cur_item >= 0)
            {
                m_ellipse.init(m_xs1 + m_dy / 1.3, 
                               m_ys1 + m_dy * m_cur_item + m_dy / 1.3,
                               m_text_height / 2.0, 
                               m_text_height / 2.0, 32);
                m_ellipse.rewind(0);
            }
            break;

        }
    }


    //------------------------------------------------------------------------
    unsigned rbox_ctrl_impl::vertex(double* x, double* y)
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
            cmd = m_text_poly.vertex(x, y);
            if(is_stop(cmd))
            {
                m_draw_item++;
                if(m_draw_item >= m_num_items)
                {
                    break;
                }
                else
                {
                    m_text.text(&m_items[m_draw_item][0]);
                    m_text.start_point(m_xs1 + m_dy * 1.5, 
                                       m_ys1 + m_dy * (m_draw_item + 1) - m_dy / 2.0);

                    m_text_poly.rewind(0);
                    cmd = m_text_poly.vertex(x, y);
                }
            }
            break;

        case 3:
            cmd = m_ellipse_poly.vertex(x, y);
            if(is_stop(cmd))
            {
                m_draw_item++;
                if(m_draw_item >= m_num_items)
                {
                    break;
                }
                else
                {
                    m_ellipse.init(m_xs1 + m_dy / 1.3, 
                                   m_ys1 + m_dy * m_draw_item + m_dy / 1.3,
                                   m_text_height / 1.5, 
                                   m_text_height / 1.5, 32);
                    m_ellipse_poly.rewind(0);
                    cmd = m_ellipse_poly.vertex(x, y);
                }
            }
            break;


        case 4:
            if(m_cur_item >= 0)
            {
                cmd = m_ellipse.vertex(x, y);
            }
            else
            {
                cmd = path_cmd_stop;
            }
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
    bool rbox_ctrl_impl::in_rect(double x, double y) const
    {
        inverse_transform_xy(&x, &y);
        return x >= m_x1 && x <= m_x2 && y >= m_y1 && y <= m_y2;
    }



    //------------------------------------------------------------------------
    bool rbox_ctrl_impl::on_mouse_button_down(double x, double y)
    {
        inverse_transform_xy(&x, &y);
        unsigned i;
        for(i = 0; i < m_num_items; i++)  
        {
            double xp = m_xs1 + m_dy / 1.3;
            double yp = m_ys1 + m_dy * i + m_dy / 1.3;
            if(calc_distance(x, y, xp, yp) <= m_text_height / 1.5)
            {
                m_cur_item = int(i);
                return true;
            }
        }
        return false;
    }


    //------------------------------------------------------------------------
    bool rbox_ctrl_impl::on_mouse_move(double, double, bool)
    {
        return false;
    }

    //------------------------------------------------------------------------
    bool rbox_ctrl_impl::on_mouse_button_up(double, double)
    {
        return false;
    }

    //------------------------------------------------------------------------
    bool rbox_ctrl_impl::on_arrow_keys(bool left, bool right, bool down, bool up)
    {
        if(m_cur_item >= 0)
        {
            if(up || right) 
            {
                m_cur_item++;
                if(m_cur_item >= int(m_num_items))
                {
                    m_cur_item = 0;
                }
                return true;
            }

            if(down || left) 
            {
                m_cur_item--;
                if(m_cur_item < 0)
                {
                    m_cur_item = m_num_items - 1;
                }
                return true;
            }
        }
        return false;
    }


}


