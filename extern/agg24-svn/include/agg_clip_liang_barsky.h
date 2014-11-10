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
// Liang-Barsky clipping 
//
//----------------------------------------------------------------------------
#ifndef AGG_CLIP_LIANG_BARSKY_INCLUDED
#define AGG_CLIP_LIANG_BARSKY_INCLUDED

#include "agg_basics.h"

namespace agg
{

    //------------------------------------------------------------------------
    enum clipping_flags_e
    {
        clipping_flags_x1_clipped = 4,
        clipping_flags_x2_clipped = 1,
        clipping_flags_y1_clipped = 8,
        clipping_flags_y2_clipped = 2,
        clipping_flags_x_clipped = clipping_flags_x1_clipped | clipping_flags_x2_clipped,
        clipping_flags_y_clipped = clipping_flags_y1_clipped | clipping_flags_y2_clipped
    };

    //----------------------------------------------------------clipping_flags
    // Determine the clipping code of the vertex according to the 
    // Cyrus-Beck line clipping algorithm
    //
    //        |        |
    //  0110  |  0010  | 0011
    //        |        |
    // -------+--------+-------- clip_box.y2
    //        |        |
    //  0100  |  0000  | 0001
    //        |        |
    // -------+--------+-------- clip_box.y1
    //        |        |
    //  1100  |  1000  | 1001
    //        |        |
    //  clip_box.x1  clip_box.x2
    //
    // 
    template<class T>
    inline unsigned clipping_flags(T x, T y, const rect_base<T>& clip_box)
    {
        return  (x > clip_box.x2) |
               ((y > clip_box.y2) << 1) |
               ((x < clip_box.x1) << 2) |
               ((y < clip_box.y1) << 3);
    }

    //--------------------------------------------------------clipping_flags_x
    template<class T>
    inline unsigned clipping_flags_x(T x, const rect_base<T>& clip_box)
    {
        return  (x > clip_box.x2) | ((x < clip_box.x1) << 2);
    }


    //--------------------------------------------------------clipping_flags_y
    template<class T>
    inline unsigned clipping_flags_y(T y, const rect_base<T>& clip_box)
    {
        return ((y > clip_box.y2) << 1) | ((y < clip_box.y1) << 3);
    }


    //-------------------------------------------------------clip_liang_barsky
    template<class T>
    inline unsigned clip_liang_barsky(T x1, T y1, T x2, T y2,
                                      const rect_base<T>& clip_box,
                                      T* x, T* y)
    {
        const double nearzero = 1e-30;

        double deltax = x2 - x1;
        double deltay = y2 - y1; 
        double xin;
        double xout;
        double yin;
        double yout;
        double tinx;
        double tiny;
        double toutx;
        double touty;  
        double tin1;
        double tin2;
        double tout1;
        unsigned np = 0;

        if(deltax == 0.0) 
        {   
            // bump off of the vertical
            deltax = (x1 > clip_box.x1) ? -nearzero : nearzero;
        }

        if(deltay == 0.0) 
        { 
            // bump off of the horizontal 
            deltay = (y1 > clip_box.y1) ? -nearzero : nearzero;
        }
        
        if(deltax > 0.0) 
        {                
            // points to right
            xin  = clip_box.x1;
            xout = clip_box.x2;
        }
        else 
        {
            xin  = clip_box.x2;
            xout = clip_box.x1;
        }

        if(deltay > 0.0) 
        {
            // points up
            yin  = clip_box.y1;
            yout = clip_box.y2;
        }
        else 
        {
            yin  = clip_box.y2;
            yout = clip_box.y1;
        }
        
        tinx = (xin - x1) / deltax;
        tiny = (yin - y1) / deltay;
        
        if (tinx < tiny) 
        {
            // hits x first
            tin1 = tinx;
            tin2 = tiny;
        }
        else
        {
            // hits y first
            tin1 = tiny;
            tin2 = tinx;
        }
        
        if(tin1 <= 1.0) 
        {
            if(0.0 < tin1) 
            {
                *x++ = (T)xin;
                *y++ = (T)yin;
                ++np;
            }

            if(tin2 <= 1.0)
            {
                toutx = (xout - x1) / deltax;
                touty = (yout - y1) / deltay;
                
                tout1 = (toutx < touty) ? toutx : touty;
                
                if(tin2 > 0.0 || tout1 > 0.0) 
                {
                    if(tin2 <= tout1) 
                    {
                        if(tin2 > 0.0) 
                        {
                            if(tinx > tiny) 
                            {
                                *x++ = (T)xin;
                                *y++ = (T)(y1 + tinx * deltay);
                            }
                            else 
                            {
                                *x++ = (T)(x1 + tiny * deltax);
                                *y++ = (T)yin;
                            }
                            ++np;
                        }

                        if(tout1 < 1.0) 
                        {
                            if(toutx < touty) 
                            {
                                *x++ = (T)xout;
                                *y++ = (T)(y1 + toutx * deltay);
                            }
                            else 
                            {
                                *x++ = (T)(x1 + touty * deltax);
                                *y++ = (T)yout;
                            }
                        }
                        else 
                        {
                            *x++ = x2;
                            *y++ = y2;
                        }
                        ++np;
                    }
                    else 
                    {
                        if(tinx > tiny) 
                        {
                            *x++ = (T)xin;
                            *y++ = (T)yout;
                        }
                        else 
                        {
                            *x++ = (T)xout;
                            *y++ = (T)yin;
                        }
                        ++np;
                    }
                }
            }
        }
        return np;
    }


    //----------------------------------------------------------------------------
    template<class T>
    bool clip_move_point(T x1, T y1, T x2, T y2, 
                         const rect_base<T>& clip_box, 
                         T* x, T* y, unsigned flags)
    {
       T bound;

       if(flags & clipping_flags_x_clipped)
       {
           if(x1 == x2)
           {
               return false;
           }
           bound = (flags & clipping_flags_x1_clipped) ? clip_box.x1 : clip_box.x2;
           *y = (T)(double(bound - x1) * (y2 - y1) / (x2 - x1) + y1);
           *x = bound;
       }

       flags = clipping_flags_y(*y, clip_box);
       if(flags & clipping_flags_y_clipped)
       {
           if(y1 == y2)
           {
               return false;
           }
           bound = (flags & clipping_flags_y1_clipped) ? clip_box.y1 : clip_box.y2;
           *x = (T)(double(bound - y1) * (x2 - x1) / (y2 - y1) + x1);
           *y = bound;
       }
       return true;
    }

    //-------------------------------------------------------clip_line_segment
    // Returns: ret >= 4        - Fully clipped
    //          (ret & 1) != 0  - First point has been moved
    //          (ret & 2) != 0  - Second point has been moved
    //
    template<class T>
    unsigned clip_line_segment(T* x1, T* y1, T* x2, T* y2,
                               const rect_base<T>& clip_box)
    {
        unsigned f1 = clipping_flags(*x1, *y1, clip_box);
        unsigned f2 = clipping_flags(*x2, *y2, clip_box);
        unsigned ret = 0;

        if((f2 | f1) == 0)
        {
            // Fully visible
            return 0;
        }

        if((f1 & clipping_flags_x_clipped) != 0 && 
           (f1 & clipping_flags_x_clipped) == (f2 & clipping_flags_x_clipped))
        {
            // Fully clipped
            return 4;
        }

        if((f1 & clipping_flags_y_clipped) != 0 && 
           (f1 & clipping_flags_y_clipped) == (f2 & clipping_flags_y_clipped))
        {
            // Fully clipped
            return 4;
        }

        T tx1 = *x1;
        T ty1 = *y1;
        T tx2 = *x2;
        T ty2 = *y2;
        if(f1) 
        {   
            if(!clip_move_point(tx1, ty1, tx2, ty2, clip_box, x1, y1, f1)) 
            {
                return 4;
            }
            if(*x1 == *x2 && *y1 == *y2) 
            {
                return 4;
            }
            ret |= 1;
        }
        if(f2) 
        {
            if(!clip_move_point(tx1, ty1, tx2, ty2, clip_box, x2, y2, f2))
            {
                return 4;
            }
            if(*x1 == *x2 && *y1 == *y2) 
            {
                return 4;
            }
            ret |= 2;
        }
        return ret;
    }


}


#endif
