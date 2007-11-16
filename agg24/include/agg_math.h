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
// Bessel function (besj) was adapted for use in AGG library by Andy Wilk 
// Contact: castor.vulgaris@gmail.com
//----------------------------------------------------------------------------

#ifndef AGG_MATH_INCLUDED
#define AGG_MATH_INCLUDED

#include <math.h>
#include "agg_basics.h"

namespace agg
{

    //------------------------------------------------------vertex_dist_epsilon
    // Coinciding points maximal distance (Epsilon)
    const double vertex_dist_epsilon = 1e-14;

    //-----------------------------------------------------intersection_epsilon
    // See calc_intersection
    const double intersection_epsilon = 1.0e-30;

    //------------------------------------------------------------cross_product
    AGG_INLINE double cross_product(double x1, double y1, 
                                    double x2, double y2, 
                                    double x,  double y)
    {
        return (x - x2) * (y2 - y1) - (y - y2) * (x2 - x1);
    }

    //--------------------------------------------------------point_in_triangle
    AGG_INLINE bool point_in_triangle(double x1, double y1, 
                                      double x2, double y2, 
                                      double x3, double y3, 
                                      double x,  double y)
    {
        bool cp1 = cross_product(x1, y1, x2, y2, x, y) < 0.0;
        bool cp2 = cross_product(x2, y2, x3, y3, x, y) < 0.0;
        bool cp3 = cross_product(x3, y3, x1, y1, x, y) < 0.0;
        return cp1 == cp2 && cp2 == cp3 && cp3 == cp1;
    }

    //-----------------------------------------------------------calc_distance
    AGG_INLINE double calc_distance(double x1, double y1, double x2, double y2)
    {
        double dx = x2-x1;
        double dy = y2-y1;
        return sqrt(dx * dx + dy * dy);
    }

    //--------------------------------------------------------calc_sq_distance
    AGG_INLINE double calc_sq_distance(double x1, double y1, double x2, double y2)
    {
        double dx = x2-x1;
        double dy = y2-y1;
        return dx * dx + dy * dy;
    }

    //------------------------------------------------calc_line_point_distance
    AGG_INLINE double calc_line_point_distance(double x1, double y1, 
                                               double x2, double y2, 
                                               double x,  double y)
    {
        double dx = x2-x1;
        double dy = y2-y1;
        double d = sqrt(dx * dx + dy * dy);
        if(d < vertex_dist_epsilon)
        {
            return calc_distance(x1, y1, x, y);
        }
        return ((x - x2) * dy - (y - y2) * dx) / d;
    }

    //-------------------------------------------------------calc_line_point_u
    AGG_INLINE double calc_segment_point_u(double x1, double y1, 
                                           double x2, double y2, 
                                           double x,  double y)
    {
        double dx = x2 - x1;
        double dy = y2 - y1;

        if(dx == 0 && dy == 0)
        {
	        return 0;
        }

        double pdx = x - x1;
        double pdy = y - y1;

        return (pdx * dx + pdy * dy) / (dx * dx + dy * dy);
    }

    //---------------------------------------------calc_line_point_sq_distance
    AGG_INLINE double calc_segment_point_sq_distance(double x1, double y1, 
                                                     double x2, double y2, 
                                                     double x,  double y,
                                                     double u)
    {
        if(u <= 0)
        {
	        return calc_sq_distance(x, y, x1, y1);
        }
        else 
        if(u >= 1)
        {
	        return calc_sq_distance(x, y, x2, y2);
        }
        return calc_sq_distance(x, y, x1 + u * (x2 - x1), y1 + u * (y2 - y1));
    }

    //---------------------------------------------calc_line_point_sq_distance
    AGG_INLINE double calc_segment_point_sq_distance(double x1, double y1, 
                                                     double x2, double y2, 
                                                     double x,  double y)
    {
        return 
            calc_segment_point_sq_distance(
                x1, y1, x2, y2, x, y,
                calc_segment_point_u(x1, y1, x2, y2, x, y));
    }

    //-------------------------------------------------------calc_intersection
    AGG_INLINE bool calc_intersection(double ax, double ay, double bx, double by,
                                      double cx, double cy, double dx, double dy,
                                      double* x, double* y)
    {
        double num = (ay-cy) * (dx-cx) - (ax-cx) * (dy-cy);
        double den = (bx-ax) * (dy-cy) - (by-ay) * (dx-cx);
        if(fabs(den) < intersection_epsilon) return false;
        double r = num / den;
        *x = ax + r * (bx-ax);
        *y = ay + r * (by-ay);
        return true;
    }

    //-----------------------------------------------------intersection_exists
    AGG_INLINE bool intersection_exists(double x1, double y1, double x2, double y2,
                                        double x3, double y3, double x4, double y4)
    {
        // It's less expensive but you can't control the 
        // boundary conditions: Less or LessEqual
        double dx1 = x2 - x1;
        double dy1 = y2 - y1;
        double dx2 = x4 - x3;
        double dy2 = y4 - y3;
        return ((x3 - x2) * dy1 - (y3 - y2) * dx1 < 0.0) != 
               ((x4 - x2) * dy1 - (y4 - y2) * dx1 < 0.0) &&
               ((x1 - x4) * dy2 - (y1 - y4) * dx2 < 0.0) !=
               ((x2 - x4) * dy2 - (y2 - y4) * dx2 < 0.0);

        // It's is more expensive but more flexible 
        // in terms of boundary conditions.
        //--------------------
        //double den  = (x2-x1) * (y4-y3) - (y2-y1) * (x4-x3);
        //if(fabs(den) < intersection_epsilon) return false;
        //double nom1 = (x4-x3) * (y1-y3) - (y4-y3) * (x1-x3);
        //double nom2 = (x2-x1) * (y1-y3) - (y2-y1) * (x1-x3);
        //double ua = nom1 / den;
        //double ub = nom2 / den;
        //return ua >= 0.0 && ua <= 1.0 && ub >= 0.0 && ub <= 1.0;
    }

    //--------------------------------------------------------calc_orthogonal
    AGG_INLINE void calc_orthogonal(double thickness,
                                    double x1, double y1,
                                    double x2, double y2,
                                    double* x, double* y)
    {
        double dx = x2 - x1;
        double dy = y2 - y1;
        double d = sqrt(dx*dx + dy*dy); 
        *x =  thickness * dy / d;
        *y = -thickness * dx / d;
    }

    //--------------------------------------------------------dilate_triangle
    AGG_INLINE void dilate_triangle(double x1, double y1,
                                    double x2, double y2,
                                    double x3, double y3,
                                    double *x, double* y,
                                    double d)
    {
        double dx1=0.0;
        double dy1=0.0; 
        double dx2=0.0;
        double dy2=0.0; 
        double dx3=0.0;
        double dy3=0.0; 
        double loc = cross_product(x1, y1, x2, y2, x3, y3);
        if(fabs(loc) > intersection_epsilon)
        {
            if(cross_product(x1, y1, x2, y2, x3, y3) > 0.0) 
            {
                d = -d;
            }
            calc_orthogonal(d, x1, y1, x2, y2, &dx1, &dy1);
            calc_orthogonal(d, x2, y2, x3, y3, &dx2, &dy2);
            calc_orthogonal(d, x3, y3, x1, y1, &dx3, &dy3);
        }
        *x++ = x1 + dx1;  *y++ = y1 + dy1;
        *x++ = x2 + dx1;  *y++ = y2 + dy1;
        *x++ = x2 + dx2;  *y++ = y2 + dy2;
        *x++ = x3 + dx2;  *y++ = y3 + dy2;
        *x++ = x3 + dx3;  *y++ = y3 + dy3;
        *x++ = x1 + dx3;  *y++ = y1 + dy3;
    }

    //------------------------------------------------------calc_triangle_area
    AGG_INLINE double calc_triangle_area(double x1, double y1,
                                         double x2, double y2,
                                         double x3, double y3)
    {
        return (x1*y2 - x2*y1 + x2*y3 - x3*y2 + x3*y1 - x1*y3) * 0.5;
    }

    //-------------------------------------------------------calc_polygon_area
    template<class Storage> double calc_polygon_area(const Storage& st)
    {
        unsigned i;
        double sum = 0.0;
        double x  = st[0].x;
        double y  = st[0].y;
        double xs = x;
        double ys = y;

        for(i = 1; i < st.size(); i++)
        {
            const typename Storage::value_type& v = st[i];
            sum += x * v.y - y * v.x;
            x = v.x;
            y = v.y;
        }
        return (sum + x * ys - y * xs) * 0.5;
    }

    //------------------------------------------------------------------------
    // Tables for fast sqrt
    extern int16u g_sqrt_table[1024];
    extern int8   g_elder_bit_table[256];


    //---------------------------------------------------------------fast_sqrt
    //Fast integer Sqrt - really fast: no cycles, divisions or multiplications
    #if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable : 4035) //Disable warning "no return value"
    #endif
    AGG_INLINE unsigned fast_sqrt(unsigned val)
    {
    #if defined(_M_IX86) && defined(_MSC_VER) && !defined(AGG_NO_ASM)
        //For Ix86 family processors this assembler code is used. 
        //The key command here is bsr - determination the number of the most 
        //significant bit of the value. For other processors
        //(and maybe compilers) the pure C "#else" section is used.
        __asm
        {
            mov ebx, val
            mov edx, 11
            bsr ecx, ebx
            sub ecx, 9
            jle less_than_9_bits
            shr ecx, 1
            adc ecx, 0
            sub edx, ecx
            shl ecx, 1
            shr ebx, cl
    less_than_9_bits:
            xor eax, eax
            mov  ax, g_sqrt_table[ebx*2]
            mov ecx, edx
            shr eax, cl
        }
    #else

        //This code is actually pure C and portable to most 
        //arcitectures including 64bit ones. 
        unsigned t = val;
        int bit=0;
        unsigned shift = 11;

        //The following piece of code is just an emulation of the
        //Ix86 assembler command "bsr" (see above). However on old
        //Intels (like Intel MMX 233MHz) this code is about twice 
        //faster (sic!) then just one "bsr". On PIII and PIV the
        //bsr is optimized quite well.
        bit = t >> 24;
        if(bit)
        {
            bit = g_elder_bit_table[bit] + 24;
        }
        else
        {
            bit = (t >> 16) & 0xFF;
            if(bit)
            {
                bit = g_elder_bit_table[bit] + 16;
            }
            else
            {
                bit = (t >> 8) & 0xFF;
                if(bit)
                {
                    bit = g_elder_bit_table[bit] + 8;
                }
                else
                {
                    bit = g_elder_bit_table[t];
                }
            }
        }

        //This code calculates the sqrt.
        bit -= 9;
        if(bit > 0)
        {
            bit = (bit >> 1) + (bit & 1);
            shift -= bit;
            val >>= (bit << 1);
        }
        return g_sqrt_table[val] >> shift;
    #endif
    }
    #if defined(_MSC_VER)
    #pragma warning(pop)
    #endif




    //--------------------------------------------------------------------besj
    // Function BESJ calculates Bessel function of first kind of order n
    // Arguments:
    //     n - an integer (>=0), the order
    //     x - value at which the Bessel function is required
    //--------------------
    // C++ Mathematical Library
    // Convereted from equivalent FORTRAN library
    // Converetd by Gareth Walker for use by course 392 computational project
    // All functions tested and yield the same results as the corresponding
    // FORTRAN versions.
    //
    // If you have any problems using these functions please report them to
    // M.Muldoon@UMIST.ac.uk
    //
    // Documentation available on the web
    // http://www.ma.umist.ac.uk/mrm/Teaching/392/libs/392.html
    // Version 1.0   8/98
    // 29 October, 1999
    //--------------------
    // Adapted for use in AGG library by Andy Wilk (castor.vulgaris@gmail.com)
    //------------------------------------------------------------------------
    inline double besj(double x, int n)
    {
        if(n < 0)
        {
            return 0;
        }
        double d = 1E-6;
        double b = 0;
        if(fabs(x) <= d) 
        {
            if(n != 0) return 0;
            return 1;
        }
        double b1 = 0; // b1 is the value from the previous iteration
        // Set up a starting order for recurrence
        int m1 = (int)fabs(x) + 6;
        if(fabs(x) > 5) 
        {
            m1 = (int)(fabs(1.4 * x + 60 / x));
        }
        int m2 = (int)(n + 2 + fabs(x) / 4);
        if (m1 > m2) 
        {
            m2 = m1;
        }
    
        // Apply recurrence down from curent max order
        for(;;) 
        {
            double c3 = 0;
            double c2 = 1E-30;
            double c4 = 0;
            int m8 = 1;
            if (m2 / 2 * 2 == m2) 
            {
                m8 = -1;
            }
            int imax = m2 - 2;
            for (int i = 1; i <= imax; i++) 
            {
                double c6 = 2 * (m2 - i) * c2 / x - c3;
                c3 = c2;
                c2 = c6;
                if(m2 - i - 1 == n)
                {
                    b = c6;
                }
                m8 = -1 * m8;
                if (m8 > 0)
                {
                    c4 = c4 + 2 * c6;
                }
            }
            double c6 = 2 * c2 / x - c3;
            if(n == 0)
            {
                b = c6;
            }
            c4 += c6;
            b /= c4;
            if(fabs(b - b1) < d)
            {
                return b;
            }
            b1 = b;
            m2 += 3;
        }
    }

}


#endif
