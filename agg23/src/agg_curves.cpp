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
// classes curve3 and curve4
//
//----------------------------------------------------------------------------

#include <math.h>
#include "agg_curves.h"

namespace agg
{

    //------------------------------------------------------------------------
    void curve3::init(double x1, double y1, 
                      double x2, double y2, 
                      double x3, double y3)
    {
        m_start_x = x1;
        m_start_y = y1;
        m_end_x   = x3;
        m_end_y   = y3;

        double dx1 = x2 - x1;
        double dy1 = y2 - y1;
        double dx2 = x3 - x2;
        double dy2 = y3 - y2;

        double len = sqrt(dx1 * dx1 + dy1 * dy1) + sqrt(dx2 * dx2 + dy2 * dy2); 

        m_num_steps = int(len * 0.25 * m_scale);

        if(m_num_steps < 2)
        {
            m_num_steps = 2;   
        }

        double subdivide_step  = 1.0 / m_num_steps;
        double subdivide_step2 = subdivide_step * subdivide_step;

        double tmpx = (x1 - x2 * 2.0 + x3) * subdivide_step2;
        double tmpy = (y1 - y2 * 2.0 + y3) * subdivide_step2;

        m_saved_fx = m_fx = x1;
        m_saved_fy = m_fy = y1;
        
        m_saved_dfx = m_dfx = tmpx + (x2 - x1) * (2.0 * subdivide_step);
        m_saved_dfy = m_dfy = tmpy + (y2 - y1) * (2.0 * subdivide_step);

        m_ddfx = tmpx * 2.0;
        m_ddfy = tmpy * 2.0;

        m_step = m_num_steps;
    }




    //------------------------------------------------------------------------
    void curve3::rewind(unsigned)
    {
        if(m_num_steps == 0)
        {
            m_step = -1;
            return;
        }
        m_step = m_num_steps;
        m_fx   = m_saved_fx;
        m_fy   = m_saved_fy;
        m_dfx  = m_saved_dfx;
        m_dfy  = m_saved_dfy;
    }




    //------------------------------------------------------------------------
    unsigned curve3::vertex(double* x, double* y)
    {
        if(m_step < 0) return path_cmd_stop;
        if(m_step == m_num_steps)
        {
            *x = m_start_x;
            *y = m_start_y;
            --m_step;
            return path_cmd_move_to;
        }
        if(m_step == 0)
        {
            *x = m_end_x;
            *y = m_end_y;
            --m_step;
            return path_cmd_line_to;
        }
        m_fx  += m_dfx; 
        m_fy  += m_dfy;
        m_dfx += m_ddfx; 
        m_dfy += m_ddfy; 
        *x = m_fx;
        *y = m_fy;
        --m_step;
        return path_cmd_line_to;
    }













    //------------------------------------------------------------------------
    void curve4::init(double x1, double y1, 
                      double x2, double y2, 
                      double x3, double y3,
                      double x4, double y4)
    {
        m_start_x = x1;
        m_start_y = y1;
        m_end_x   = x4;
        m_end_y   = y4;

        double dx1 = x2 - x1;
        double dy1 = y2 - y1;
        double dx2 = x3 - x2;
        double dy2 = y3 - y2;
        double dx3 = x4 - x3;
        double dy3 = y4 - y3;

        double len = sqrt(dx1 * dx1 + dy1 * dy1) + 
                     sqrt(dx2 * dx2 + dy2 * dy2) + 
                     sqrt(dx3 * dx3 + dy3 * dy3);

        m_num_steps = int(len * 0.25 * m_scale);

        if(m_num_steps < 2)
        {
            m_num_steps = 2;   
        }

        double subdivide_step  = 1.0 / m_num_steps;
        double subdivide_step2 = subdivide_step * subdivide_step;
        double subdivide_step3 = subdivide_step * subdivide_step * subdivide_step;

	    double pre1 = 3.0 * subdivide_step;
	    double pre2 = 3.0 * subdivide_step2;
	    double pre4 = 6.0 * subdivide_step2;
	    double pre5 = 6.0 * subdivide_step3;
	
        double tmp1x = x1 - x2 * 2.0 + x3;
        double tmp1y = y1 - y2 * 2.0 + y3;

        double tmp2x = (x2 - x3) * 3.0 - x1 + x4;
        double tmp2y = (y2 - y3) * 3.0 - y1 + y4;

        m_saved_fx = m_fx = x1;
        m_saved_fy = m_fy = y1;

        m_saved_dfx = m_dfx = (x2 - x1) * pre1 + tmp1x * pre2 + tmp2x * subdivide_step3;
        m_saved_dfy = m_dfy = (y2 - y1) * pre1 + tmp1y * pre2 + tmp2y * subdivide_step3;

        m_saved_ddfx = m_ddfx = tmp1x * pre4 + tmp2x * pre5;
        m_saved_ddfy = m_ddfy = tmp1y * pre4 + tmp2y * pre5;

        m_dddfx = tmp2x * pre5;
        m_dddfy = tmp2y * pre5;

        m_step = m_num_steps;
    }




    //------------------------------------------------------------------------
    void curve4::rewind(unsigned)
    {
        if(m_num_steps == 0)
        {
            m_step = -1;
            return;
        }
        m_step = m_num_steps;
        m_fx   = m_saved_fx;
        m_fy   = m_saved_fy;
        m_dfx  = m_saved_dfx;
        m_dfy  = m_saved_dfy;
        m_ddfx = m_saved_ddfx;
        m_ddfy = m_saved_ddfy;
    }





    //------------------------------------------------------------------------
    unsigned curve4::vertex(double* x, double* y)
    {
        if(m_step < 0) return path_cmd_stop;
        if(m_step == m_num_steps)
        {
            *x = m_start_x;
            *y = m_start_y;
            --m_step;
            return path_cmd_move_to;
        }

        if(m_step == 0)
        {
            *x = m_end_x;
            *y = m_end_y;
            --m_step;
            return path_cmd_line_to;
        }

        m_fx   += m_dfx; 
        m_fy   += m_dfy;
        m_dfx  += m_ddfx; 
        m_dfy  += m_ddfy; 
        m_ddfx += m_dddfx; 
        m_ddfy += m_dddfy; 
        *x = m_fx;
        *y = m_fy;
        --m_step;
        return path_cmd_line_to;
    }




}

