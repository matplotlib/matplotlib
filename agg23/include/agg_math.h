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

#ifndef AGG_MATH_INCLUDED
#define AGG_MATH_INCLUDED

#include <math.h>
#include "agg_basics.h"

namespace agg
{

    const double intersection_epsilon = 1.0e-30;

    //------------------------------------------------------calc_point_location
    AGG_INLINE double calc_point_location(double x1, double y1, 
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
        bool cp1 = calc_point_location(x1, y1, x2, y2, x, y) < 0.0;
        bool cp2 = calc_point_location(x2, y2, x3, y3, x, y) < 0.0;
        bool cp3 = calc_point_location(x3, y3, x1, y1, x, y) < 0.0;
        return cp1 == cp2 && cp2 == cp3 && cp3 == cp1;
    }


    //-----------------------------------------------------------calc_distance
    AGG_INLINE double calc_distance(double x1, double y1, double x2, double y2)
    {
        double dx = x2-x1;
        double dy = y2-y1;
        return sqrt(dx * dx + dy * dy);
    }


    //------------------------------------------------calc_point_line_distance
    AGG_INLINE double calc_point_line_distance(double x1, double y1, 
                                               double x2, double y2, 
                                               double x,  double y)
    {
        double dx = x2-x1;
        double dy = y2-y1;
        return ((x - x2) * dy - (y - y2) * dx) / sqrt(dx * dx + dy * dy);
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


    //--------------------------------------------------------calc_orthogonal
    AGG_INLINE void calc_orthogonal(double thickness,
                                    double x1, double y1,
                                    double x2, double y2,
                                    double* x, double* y)
    {
        double dx = x2 - x1;
        double dy = y2 - y1;
        double d = sqrt(dx*dx + dy*dy); 
        *x = thickness * dy / d;
        *y = thickness * dx / d;
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
        double loc = calc_point_location(x1, y1, x2, y2, x3, y3);
        if(fabs(loc) > intersection_epsilon)
        {
            if(calc_point_location(x1, y1, x2, y2, x3, y3) > 0.0) 
            {
                d = -d;
            }
            calc_orthogonal(d, x1, y1, x2, y2, &dx1, &dy1);
            calc_orthogonal(d, x2, y2, x3, y3, &dx2, &dy2);
            calc_orthogonal(d, x3, y3, x1, y1, &dx3, &dy3);
        }
        *x++ = x1 + dx1;  *y++ = y1 - dy1;
        *x++ = x2 + dx1;  *y++ = y2 - dy1;
        *x++ = x2 + dx2;  *y++ = y2 - dy2;
        *x++ = x3 + dx2;  *y++ = y3 - dy2;
        *x++ = x3 + dx3;  *y++ = y3 - dy3;
        *x++ = x1 + dx3;  *y++ = y1 - dy3;
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

        //This is calculation sqrt itself.
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




}


#endif
