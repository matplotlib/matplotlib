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
// Liang-Barsky clipping 
//
//----------------------------------------------------------------------------
#ifndef AGG_CLIP_LIANG_BARSKY_INCLUDED
#define AGG_CLIP_LIANG_BARSKY_INCLUDED

#include "agg_basics.h"

namespace agg
{

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
    

}

#endif
