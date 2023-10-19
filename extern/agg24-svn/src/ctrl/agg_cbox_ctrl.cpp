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
#include "ctrl/agg_cbox_ctrl.h"


namespace agg
{

    //------------------------------------------------------------------------
    cbox_ctrl_impl::cbox_ctrl_impl(double x, double y, 
                                   const char* l, 
                                   bool flip_y) :
        ctrl(x, y, x + 9.0 * 1.5, y + 9.0 * 1.5, flip_y),
        m_text_thickness(1.5),
        m_text_height(9.0),
        m_text_width(0.0),
        m_status(false),
        m_text_poly(m_text)
    {
        label(l);
    }


    //------------------------------------------------------------------------
    void cbox_ctrl_impl::text_size(double h, double w)
    {
        m_text_width = w; 
        m_text_height = h; 
    }

    //------------------------------------------------------------------------
    void cbox_ctrl_impl::label(const char* l)
    {
        unsigned len = strlen(l);
        if(len > 127) len = 127;
        memcpy(m_label, l, len);
        m_label[len] = 0;
    }


    //------------------------------------------------------------------------
    bool cbox_ctrl_impl::on_mouse_button_down(double x, double y)
    {
        inverse_transform_xy(&x, &y);
        if(x >= m_x1 && y >= m_y1 && x <= m_x2 && y <= m_y2)
        {
            m_status = !m_status;
            return true;
        }
        return false;
    }


    //------------------------------------------------------------------------
    bool cbox_ctrl_impl::on_mouse_move(double, double, bool)
    {
        return false;
    }

    //------------------------------------------------------------------------
    bool cbox_ctrl_impl::in_rect(double x, double y) const
    {
        inverse_transform_xy(&x, &y);
        return x >= m_x1 && y >= m_y1 && x <= m_x2 && y <= m_y2;
    }

    //------------------------------------------------------------------------
    bool cbox_ctrl_impl::on_mouse_button_up(double, double)
    {
        return false;
    }

    //------------------------------------------------------------------------
    bool cbox_ctrl_impl::on_arrow_keys(bool, bool, bool, bool)
    {
        return false;
    }


    //------------------------------------------------------------------------
    void cbox_ctrl_impl::rewind(unsigned idx)
    {
        m_idx = idx;

        double d2;
        double t;

        switch(idx)
        {
        default:
        case 0:                 // Border
            m_vertex = 0;
            m_vx[0] = m_x1; 
            m_vy[0] = m_y1;
            m_vx[1] = m_x2;
            m_vy[1] = m_y1;
            m_vx[2] = m_x2;
            m_vy[2] = m_y2;
            m_vx[3] = m_x1; 
            m_vy[3] = m_y2;
            m_vx[4] = m_x1 + m_text_thickness; 
            m_vy[4] = m_y1 + m_text_thickness; 
            m_vx[5] = m_x1 + m_text_thickness; 
            m_vy[5] = m_y2 - m_text_thickness;
            m_vx[6] = m_x2 - m_text_thickness;
            m_vy[6] = m_y2 - m_text_thickness;
            m_vx[7] = m_x2 - m_text_thickness;
            m_vy[7] = m_y1 + m_text_thickness; 
            break;

        case 1:                 // Text
            m_text.text(m_label);
            m_text.start_point(m_x1 + m_text_height * 2.0, m_y1 + m_text_height / 5.0);
            m_text.size(m_text_height, m_text_width);
            m_text_poly.width(m_text_thickness);
            m_text_poly.line_join(round_join);
            m_text_poly.line_cap(round_cap);
            m_text_poly.rewind(0);
            break;

        case 2:                 // Active item
            m_vertex = 0;
            d2 = (m_y2 - m_y1) / 2.0;
            t = m_text_thickness * 1.5;
            m_vx[0] = m_x1 + m_text_thickness;
            m_vy[0] = m_y1 + m_text_thickness;
            m_vx[1] = m_x1 + d2;
            m_vy[1] = m_y1 + d2 - t;
            m_vx[2] = m_x2 - m_text_thickness;
            m_vy[2] = m_y1 + m_text_thickness;
            m_vx[3] = m_x1 + d2 + t;
            m_vy[3] = m_y1 + d2;
            m_vx[4] = m_x2 - m_text_thickness;
            m_vy[4] = m_y2 - m_text_thickness;
            m_vx[5] = m_x1 + d2;
            m_vy[5] = m_y1 + d2 + t;
            m_vx[6] = m_x1 + m_text_thickness;
            m_vy[6] = m_y2 - m_text_thickness;
            m_vx[7] = m_x1 + d2 - t;
            m_vy[7] = m_y1 + d2;
            break;

        }
    }




    //------------------------------------------------------------------------
    unsigned cbox_ctrl_impl::vertex(double* x, double* y)
    {
        unsigned cmd = path_cmd_line_to;
        switch(m_idx)
        {
        case 0:
            if(m_vertex == 0 || m_vertex == 4) cmd = path_cmd_move_to;
            if(m_vertex >= 8) cmd = path_cmd_stop;
            *x = m_vx[m_vertex];
            *y = m_vy[m_vertex];
            m_vertex++;
            break;

        case 1:
            cmd = m_text_poly.vertex(x, y);
            break;

        case 2:
            if(m_status)
            {
                if(m_vertex == 0) cmd = path_cmd_move_to;
                if(m_vertex >= 8) cmd = path_cmd_stop;
                *x = m_vx[m_vertex];
                *y = m_vy[m_vertex];
                m_vertex++;
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
}



