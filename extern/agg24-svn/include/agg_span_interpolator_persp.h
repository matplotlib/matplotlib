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
#ifndef AGG_SPAN_INTERPOLATOR_PERSP_INCLUDED
#define AGG_SPAN_INTERPOLATOR_PERSP_INCLUDED

#include "agg_trans_perspective.h"
#include "agg_dda_line.h"

namespace agg
{



    //===========================================span_interpolator_persp_exact
    template<unsigned SubpixelShift = 8> 
    class span_interpolator_persp_exact
    {
    public:
        typedef trans_perspective trans_type;
        typedef trans_perspective::iterator_x iterator_type;
        enum subpixel_scale_e
        {
            subpixel_shift = SubpixelShift,
            subpixel_scale = 1 << subpixel_shift
        };

        //--------------------------------------------------------------------
        span_interpolator_persp_exact() {}

        //--------------------------------------------------------------------
        // Arbitrary quadrangle transformations
        span_interpolator_persp_exact(const double* src, const double* dst) 
        {
            quad_to_quad(src, dst);
        }

        //--------------------------------------------------------------------
        // Direct transformations 
        span_interpolator_persp_exact(double x1, double y1, 
                                      double x2, double y2, 
                                      const double* quad)
        {
            rect_to_quad(x1, y1, x2, y2, quad);
        }

        //--------------------------------------------------------------------
        // Reverse transformations 
        span_interpolator_persp_exact(const double* quad, 
                                      double x1, double y1, 
                                      double x2, double y2)
        {
            quad_to_rect(quad, x1, y1, x2, y2);
        }

        //--------------------------------------------------------------------
        // Set the transformations using two arbitrary quadrangles.
        void quad_to_quad(const double* src, const double* dst)
        {
            m_trans_dir.quad_to_quad(src, dst);
            m_trans_inv.quad_to_quad(dst, src);
        }

        //--------------------------------------------------------------------
        // Set the direct transformations, i.e., rectangle -> quadrangle
        void rect_to_quad(double x1, double y1, double x2, double y2, 
                          const double* quad)
        {
            double src[8];
            src[0] = src[6] = x1;
            src[2] = src[4] = x2;
            src[1] = src[3] = y1;
            src[5] = src[7] = y2;
            quad_to_quad(src, quad);
        }


        //--------------------------------------------------------------------
        // Set the reverse transformations, i.e., quadrangle -> rectangle
        void quad_to_rect(const double* quad, 
                          double x1, double y1, double x2, double y2)
        {
            double dst[8];
            dst[0] = dst[6] = x1;
            dst[2] = dst[4] = x2;
            dst[1] = dst[3] = y1;
            dst[5] = dst[7] = y2;
            quad_to_quad(quad, dst);
        }

        //--------------------------------------------------------------------
        // Check if the equations were solved successfully
        bool is_valid() const { return m_trans_dir.is_valid(); }

        //----------------------------------------------------------------
        void begin(double x, double y, unsigned len)
        {
            m_iterator = m_trans_dir.begin(x, y, 1.0);
            double xt = m_iterator.x;
            double yt = m_iterator.y;

            double dx;
            double dy;
            const double delta = 1/double(subpixel_scale);
            dx = xt + delta;
            dy = yt;
            m_trans_inv.transform(&dx, &dy);
            dx -= x;
            dy -= y;
            int sx1 = uround(subpixel_scale/sqrt(dx*dx + dy*dy)) >> subpixel_shift;
            dx = xt;
            dy = yt + delta;
            m_trans_inv.transform(&dx, &dy);
            dx -= x;
            dy -= y;
            int sy1 = uround(subpixel_scale/sqrt(dx*dx + dy*dy)) >> subpixel_shift;

            x += len;
            xt = x;
            yt = y;
            m_trans_dir.transform(&xt, &yt);

            dx = xt + delta;
            dy = yt;
            m_trans_inv.transform(&dx, &dy);
            dx -= x;
            dy -= y;
            int sx2 = uround(subpixel_scale/sqrt(dx*dx + dy*dy)) >> subpixel_shift;
            dx = xt;
            dy = yt + delta;
            m_trans_inv.transform(&dx, &dy);
            dx -= x;
            dy -= y;
            int sy2 = uround(subpixel_scale/sqrt(dx*dx + dy*dy)) >> subpixel_shift;

            m_scale_x = dda2_line_interpolator(sx1, sx2, len);
            m_scale_y = dda2_line_interpolator(sy1, sy2, len);
        }


        //----------------------------------------------------------------
        void resynchronize(double xe, double ye, unsigned len)
        {
            // Assume x1,y1 are equal to the ones at the previous end point 
            int sx1 = m_scale_x.y();
            int sy1 = m_scale_y.y();

            // Calculate transformed coordinates at x2,y2 
            double xt = xe;
            double yt = ye;
            m_trans_dir.transform(&xt, &yt);

            const double delta = 1/double(subpixel_scale);
            double dx;
            double dy;

            // Calculate scale by X at x2,y2
            dx = xt + delta;
            dy = yt;
            m_trans_inv.transform(&dx, &dy);
            dx -= xe;
            dy -= ye;
            int sx2 = uround(subpixel_scale/sqrt(dx*dx + dy*dy)) >> subpixel_shift;

            // Calculate scale by Y at x2,y2
            dx = xt;
            dy = yt + delta;
            m_trans_inv.transform(&dx, &dy);
            dx -= xe;
            dy -= ye;
            int sy2 = uround(subpixel_scale/sqrt(dx*dx + dy*dy)) >> subpixel_shift;

            // Initialize the interpolators
            m_scale_x = dda2_line_interpolator(sx1, sx2, len);
            m_scale_y = dda2_line_interpolator(sy1, sy2, len);
        }



        //----------------------------------------------------------------
        void operator++()
        {
            ++m_iterator;
            ++m_scale_x;
            ++m_scale_y;
        }

        //----------------------------------------------------------------
        void coordinates(int* x, int* y) const
        {
            *x = iround(m_iterator.x * subpixel_scale);
            *y = iround(m_iterator.y * subpixel_scale);
        }

        //----------------------------------------------------------------
        void local_scale(int* x, int* y)
        {
            *x = m_scale_x.y();
            *y = m_scale_y.y();
        }

        //----------------------------------------------------------------
        void transform(double* x, double* y) const
        {
            m_trans_dir.transform(x, y);
        }
        
    private:
        trans_type             m_trans_dir;
        trans_type             m_trans_inv;
        iterator_type          m_iterator;
        dda2_line_interpolator m_scale_x;
        dda2_line_interpolator m_scale_y;
    };











    //============================================span_interpolator_persp_lerp
    template<unsigned SubpixelShift = 8> 
    class span_interpolator_persp_lerp
    {
    public:
        typedef trans_perspective trans_type;
        enum subpixel_scale_e
        {
            subpixel_shift = SubpixelShift,
            subpixel_scale = 1 << subpixel_shift
        };

        //--------------------------------------------------------------------
        span_interpolator_persp_lerp() {}

        //--------------------------------------------------------------------
        // Arbitrary quadrangle transformations
        span_interpolator_persp_lerp(const double* src, const double* dst) 
        {
            quad_to_quad(src, dst);
        }

        //--------------------------------------------------------------------
        // Direct transformations 
        span_interpolator_persp_lerp(double x1, double y1, 
                                     double x2, double y2, 
                                     const double* quad)
        {
            rect_to_quad(x1, y1, x2, y2, quad);
        }

        //--------------------------------------------------------------------
        // Reverse transformations 
        span_interpolator_persp_lerp(const double* quad, 
                                     double x1, double y1, 
                                     double x2, double y2)
        {
            quad_to_rect(quad, x1, y1, x2, y2);
        }

        //--------------------------------------------------------------------
        // Set the transformations using two arbitrary quadrangles.
        void quad_to_quad(const double* src, const double* dst)
        {
            m_trans_dir.quad_to_quad(src, dst);
            m_trans_inv.quad_to_quad(dst, src);
        }

        //--------------------------------------------------------------------
        // Set the direct transformations, i.e., rectangle -> quadrangle
        void rect_to_quad(double x1, double y1, double x2, double y2, 
                          const double* quad)
        {
            double src[8];
            src[0] = src[6] = x1;
            src[2] = src[4] = x2;
            src[1] = src[3] = y1;
            src[5] = src[7] = y2;
            quad_to_quad(src, quad);
        }


        //--------------------------------------------------------------------
        // Set the reverse transformations, i.e., quadrangle -> rectangle
        void quad_to_rect(const double* quad, 
                          double x1, double y1, double x2, double y2)
        {
            double dst[8];
            dst[0] = dst[6] = x1;
            dst[2] = dst[4] = x2;
            dst[1] = dst[3] = y1;
            dst[5] = dst[7] = y2;
            quad_to_quad(quad, dst);
        }

        //--------------------------------------------------------------------
        // Check if the equations were solved successfully
        bool is_valid() const { return m_trans_dir.is_valid(); }

        //----------------------------------------------------------------
        void begin(double x, double y, unsigned len)
        {
            // Calculate transformed coordinates at x1,y1 
            double xt = x;
            double yt = y;
            m_trans_dir.transform(&xt, &yt);
            int x1 = iround(xt * subpixel_scale);
            int y1 = iround(yt * subpixel_scale);

            double dx;
            double dy;
            const double delta = 1/double(subpixel_scale);

            // Calculate scale by X at x1,y1
            dx = xt + delta;
            dy = yt;
            m_trans_inv.transform(&dx, &dy);
            dx -= x;
            dy -= y;
            int sx1 = uround(subpixel_scale/sqrt(dx*dx + dy*dy)) >> subpixel_shift;

            // Calculate scale by Y at x1,y1
            dx = xt;
            dy = yt + delta;
            m_trans_inv.transform(&dx, &dy);
            dx -= x;
            dy -= y;
            int sy1 = uround(subpixel_scale/sqrt(dx*dx + dy*dy)) >> subpixel_shift;

            // Calculate transformed coordinates at x2,y2 
            x += len;
            xt = x;
            yt = y;
            m_trans_dir.transform(&xt, &yt);
            int x2 = iround(xt * subpixel_scale);
            int y2 = iround(yt * subpixel_scale);

            // Calculate scale by X at x2,y2
            dx = xt + delta;
            dy = yt;
            m_trans_inv.transform(&dx, &dy);
            dx -= x;
            dy -= y;
            int sx2 = uround(subpixel_scale/sqrt(dx*dx + dy*dy)) >> subpixel_shift;

            // Calculate scale by Y at x2,y2
            dx = xt;
            dy = yt + delta;
            m_trans_inv.transform(&dx, &dy);
            dx -= x;
            dy -= y;
            int sy2 = uround(subpixel_scale/sqrt(dx*dx + dy*dy)) >> subpixel_shift;

            // Initialize the interpolators
            m_coord_x = dda2_line_interpolator(x1,  x2,  len);
            m_coord_y = dda2_line_interpolator(y1,  y2,  len);
            m_scale_x = dda2_line_interpolator(sx1, sx2, len);
            m_scale_y = dda2_line_interpolator(sy1, sy2, len);
        }


        //----------------------------------------------------------------
        void resynchronize(double xe, double ye, unsigned len)
        {
            // Assume x1,y1 are equal to the ones at the previous end point 
            int x1  = m_coord_x.y();
            int y1  = m_coord_y.y();
            int sx1 = m_scale_x.y();
            int sy1 = m_scale_y.y();

            // Calculate transformed coordinates at x2,y2 
            double xt = xe;
            double yt = ye;
            m_trans_dir.transform(&xt, &yt);
            int x2 = iround(xt * subpixel_scale);
            int y2 = iround(yt * subpixel_scale);

            const double delta = 1/double(subpixel_scale);
            double dx;
            double dy;

            // Calculate scale by X at x2,y2
            dx = xt + delta;
            dy = yt;
            m_trans_inv.transform(&dx, &dy);
            dx -= xe;
            dy -= ye;
            int sx2 = uround(subpixel_scale/sqrt(dx*dx + dy*dy)) >> subpixel_shift;

            // Calculate scale by Y at x2,y2
            dx = xt;
            dy = yt + delta;
            m_trans_inv.transform(&dx, &dy);
            dx -= xe;
            dy -= ye;
            int sy2 = uround(subpixel_scale/sqrt(dx*dx + dy*dy)) >> subpixel_shift;

            // Initialize the interpolators
            m_coord_x = dda2_line_interpolator(x1,  x2,  len);
            m_coord_y = dda2_line_interpolator(y1,  y2,  len);
            m_scale_x = dda2_line_interpolator(sx1, sx2, len);
            m_scale_y = dda2_line_interpolator(sy1, sy2, len);
        }


        //----------------------------------------------------------------
        void operator++()
        {
            ++m_coord_x;
            ++m_coord_y;
            ++m_scale_x;
            ++m_scale_y;
        }

        //----------------------------------------------------------------
        void coordinates(int* x, int* y) const
        {
            *x = m_coord_x.y();
            *y = m_coord_y.y();
        }

        //----------------------------------------------------------------
        void local_scale(int* x, int* y)
        {
            *x = m_scale_x.y();
            *y = m_scale_y.y();
        }

        //----------------------------------------------------------------
        void transform(double* x, double* y) const
        {
            m_trans_dir.transform(x, y);
        }
        
    private:
        trans_type             m_trans_dir;
        trans_type             m_trans_inv;
        dda2_line_interpolator m_coord_x;
        dda2_line_interpolator m_coord_y;
        dda2_line_interpolator m_scale_x;
        dda2_line_interpolator m_scale_y;
    };

}

#endif
