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
// Viewport transformer - simple orthogonal conversions from world coordinates
//                        to screen (device) ones.
//
//----------------------------------------------------------------------------

#ifndef AGG_TRANS_VIEWPORT_INCLUDED
#define AGG_TRANS_VIEWPORT_INCLUDED

#include <string.h>
#include "agg_trans_affine.h"


namespace agg
{

    enum aspect_ratio_e
    {
        aspect_ratio_stretch,
        aspect_ratio_meet,
        aspect_ratio_slice
    };


    //----------------------------------------------------------trans_viewport
    class trans_viewport
    {
    public:
        //-------------------------------------------------------------------
        trans_viewport() : 
            m_world_x1(0.0),
            m_world_y1(0.0),
            m_world_x2(1.0),
            m_world_y2(1.0),
            m_device_x1(0.0),
            m_device_y1(0.0),
            m_device_x2(1.0),
            m_device_y2(1.0),
            m_aspect(aspect_ratio_stretch),
            m_is_valid(true),
            m_align_x(0.5),
            m_align_y(0.5),
            m_wx1(0.0),
            m_wy1(0.0),
            m_wx2(1.0),
            m_wy2(1.0),
            m_dx1(0.0),
            m_dy1(0.0),
            m_kx(1.0),
            m_ky(1.0)
        {}

        //-------------------------------------------------------------------
        void preserve_aspect_ratio(double alignx, 
                                   double aligny, 
                                   aspect_ratio_e aspect)
        {
            m_align_x = alignx;
            m_align_y = aligny;
            m_aspect  = aspect;
            update();
        }

        //-------------------------------------------------------------------
        void device_viewport(double x1, double y1, double x2, double y2)
        {
            m_device_x1 = x1;
            m_device_y1 = y1;
            m_device_x2 = x2;
            m_device_y2 = y2;
            update();
        }

        //-------------------------------------------------------------------
        void world_viewport(double x1, double y1, double x2, double y2)
        {
            m_world_x1 = x1;
            m_world_y1 = y1;
            m_world_x2 = x2;
            m_world_y2 = y2;
            update();
        }

        //-------------------------------------------------------------------
        void device_viewport(double* x1, double* y1, double* x2, double* y2) const
        {
            *x1 = m_device_x1;
            *y1 = m_device_y1;
            *x2 = m_device_x2;
            *y2 = m_device_y2;
        }

        //-------------------------------------------------------------------
        void world_viewport(double* x1, double* y1, double* x2, double* y2) const
        {
            *x1 = m_world_x1;
            *y1 = m_world_y1;
            *x2 = m_world_x2;
            *y2 = m_world_y2;
        }

        //-------------------------------------------------------------------
        void world_viewport_actual(double* x1, double* y1, 
                                   double* x2, double* y2) const
        {
            *x1 = m_wx1;
            *y1 = m_wy1;
            *x2 = m_wx2;
            *y2 = m_wy2;
        }

        //-------------------------------------------------------------------
        bool   is_valid()             const { return m_is_valid; }
        double align_x()              const { return m_align_x; }
        double align_y()              const { return m_align_y; }
        aspect_ratio_e aspect_ratio() const { return m_aspect; }

        //-------------------------------------------------------------------
        void transform(double* x, double* y) const
        {
            *x = (*x - m_wx1) * m_kx + m_dx1;
            *y = (*y - m_wy1) * m_ky + m_dy1;
        }

        //-------------------------------------------------------------------
        void transform_scale_only(double* x, double* y) const
        {
            *x *= m_kx;
            *y *= m_ky;
        }

        //-------------------------------------------------------------------
        void inverse_transform(double* x, double* y) const
        {
            *x = (*x - m_dx1) / m_kx + m_wx1;
            *y = (*y - m_dy1) / m_ky + m_wy1;
        }

        //-------------------------------------------------------------------
        void inverse_transform_scale_only(double* x, double* y) const
        {
            *x /= m_kx;
            *y /= m_ky;
        }

        //-------------------------------------------------------------------
        double device_dx() const { return m_dx1 - m_wx1 * m_kx; }
        double device_dy() const { return m_dy1 - m_wy1 * m_ky; }

        //-------------------------------------------------------------------
        double scale_x() const
        {
            return m_kx;
        }

        //-------------------------------------------------------------------
        double scale_y() const
        {
            return m_ky;
        }

        //-------------------------------------------------------------------
        double scale() const
        {
            return (m_kx + m_ky) * 0.5;
        }

        //-------------------------------------------------------------------
        trans_affine to_affine() const
        {
            trans_affine mtx = trans_affine_translation(-m_wx1, -m_wy1);
            mtx *= trans_affine_scaling(m_kx, m_ky);
            mtx *= trans_affine_translation(m_dx1, m_dy1);
            return mtx;
        }

        //-------------------------------------------------------------------
        trans_affine to_affine_scale_only() const
        {
            return trans_affine_scaling(m_kx, m_ky);
        }

        //-------------------------------------------------------------------
        unsigned byte_size() const
        {
            return sizeof(*this);
        }

        void serialize(int8u* ptr) const
        {
            memcpy(ptr, this, sizeof(*this)); 
        }

        void deserialize(const int8u* ptr)
        {
            memcpy(this,  ptr, sizeof(*this));
        }

    private:
        void update();

        double         m_world_x1;
        double         m_world_y1;
        double         m_world_x2;
        double         m_world_y2;
        double         m_device_x1;
        double         m_device_y1;
        double         m_device_x2;
        double         m_device_y2;
        aspect_ratio_e m_aspect;
        bool           m_is_valid;
        double         m_align_x;
        double         m_align_y;
        double         m_wx1;
        double         m_wy1;
        double         m_wx2;
        double         m_wy2;
        double         m_dx1;
        double         m_dy1;
        double         m_kx;
        double         m_ky;
    };



    //-----------------------------------------------------------------------
    inline void trans_viewport::update()
    {
        const double epsilon = 1e-30;
        if(fabs(m_world_x1  - m_world_x2)  < epsilon ||
           fabs(m_world_y1  - m_world_y2)  < epsilon ||
           fabs(m_device_x1 - m_device_x2) < epsilon ||
           fabs(m_device_y1 - m_device_y2) < epsilon)
        {
            m_wx1 = m_world_x1;
            m_wy1 = m_world_y1;
            m_wx2 = m_world_x1 + 1.0;
            m_wy2 = m_world_y2 + 1.0;
            m_dx1 = m_device_x1;
            m_dy1 = m_device_y1;
            m_kx  = 1.0;
            m_ky  = 1.0;
            m_is_valid = false;
            return;
        }

        double world_x1  = m_world_x1;
        double world_y1  = m_world_y1;
        double world_x2  = m_world_x2;
        double world_y2  = m_world_y2;
        double device_x1 = m_device_x1;
        double device_y1 = m_device_y1;
        double device_x2 = m_device_x2;
        double device_y2 = m_device_y2;
        if(m_aspect != aspect_ratio_stretch)
        {
            double d;
            m_kx = (device_x2 - device_x1) / (world_x2 - world_x1);
            m_ky = (device_y2 - device_y1) / (world_y2 - world_y1);

            if((m_aspect == aspect_ratio_meet) == (m_kx < m_ky))
            {
                d         = (world_y2 - world_y1) * m_ky / m_kx;
                world_y1 += (world_y2 - world_y1 - d) * m_align_y;
                world_y2  =  world_y1 + d;
            }
            else
            {
                d         = (world_x2 - world_x1) * m_kx / m_ky;
                world_x1 += (world_x2 - world_x1 - d) * m_align_x;
                world_x2  =  world_x1 + d;
            }
        }
        m_wx1 = world_x1;
        m_wy1 = world_y1;
        m_wx2 = world_x2;
        m_wy2 = world_y2;
        m_dx1 = device_x1;
        m_dy1 = device_y1;
        m_kx  = (device_x2 - device_x1) / (world_x2 - world_x1);
        m_ky  = (device_y2 - device_y1) / (world_y2 - world_y1);
        m_is_valid = true;
    }


}


#endif
