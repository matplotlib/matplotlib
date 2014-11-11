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

#include "agg_renderer_outline_aa.h"

namespace agg
{

    //---------------------------------------------------------------------
    void line_profile_aa::width(double w)
    {
        if(w < 0.0) w = 0.0;

        if(w < m_smoother_width) w += w;
        else                     w += m_smoother_width;

        w *= 0.5;

        w -= m_smoother_width;
        double s = m_smoother_width;
        if(w < 0.0) 
        {
            s += w;
            w = 0.0;
        }
        set(w, s);
    }


    //---------------------------------------------------------------------
    line_profile_aa::value_type* line_profile_aa::profile(double w)
    {
        m_subpixel_width = uround(w * subpixel_scale);
        unsigned size = m_subpixel_width + subpixel_scale * 6;
        if(size > m_profile.size())
        {
            m_profile.resize(size);
        }
        return &m_profile[0];
    }


    //---------------------------------------------------------------------
    void line_profile_aa::set(double center_width, double smoother_width)
    {
        double base_val = 1.0;
        if(center_width == 0.0)   center_width = 1.0 / subpixel_scale;
        if(smoother_width == 0.0) smoother_width = 1.0 / subpixel_scale;

        double width = center_width + smoother_width;
        if(width < m_min_width)
        {
            double k = width / m_min_width;
            base_val *= k;
            center_width /= k;
            smoother_width /= k;
        }

        value_type* ch = profile(center_width + smoother_width);

        unsigned subpixel_center_width = unsigned(center_width * subpixel_scale);
        unsigned subpixel_smoother_width = unsigned(smoother_width * subpixel_scale);

        value_type* ch_center   = ch + subpixel_scale*2;
        value_type* ch_smoother = ch_center + subpixel_center_width;

        unsigned i;

        unsigned val = m_gamma[unsigned(base_val * aa_mask)];
        ch = ch_center;
        for(i = 0; i < subpixel_center_width; i++)
        {
            *ch++ = (value_type)val;
        }

        for(i = 0; i < subpixel_smoother_width; i++)
        {
            *ch_smoother++ = 
                m_gamma[unsigned((base_val - 
                                  base_val * 
                                  (double(i) / subpixel_smoother_width)) * aa_mask)];
        }

        unsigned n_smoother = profile_size() - 
                              subpixel_smoother_width - 
                              subpixel_center_width - 
                              subpixel_scale*2;

        val = m_gamma[0];
        for(i = 0; i < n_smoother; i++)
        {
            *ch_smoother++ = (value_type)val;
        }

        ch = ch_center;
        for(i = 0; i < subpixel_scale*2; i++)
        {
            *--ch = *ch_center++;
        }
    }


}

