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
#ifndef AGG_RASTERIZER_SL_CLIP_INCLUDED
#define AGG_RASTERIZER_SL_CLIP_INCLUDED

#include "agg_clip_liang_barsky.h"

namespace agg
{
    //--------------------------------------------------------poly_max_coord_e
    enum poly_max_coord_e
    {
        poly_max_coord = (1 << 30) - 1 //----poly_max_coord
    };
    
    //------------------------------------------------------------ras_conv_int
    struct ras_conv_int
    {
        typedef int coord_type;
        static AGG_INLINE int mul_div(double a, double b, double c)
        {
            return iround(a * b / c);
        }
        static int xi(int v) { return v; }
        static int yi(int v) { return v; }
        static int upscale(double v) { return iround(v * poly_subpixel_scale); }
        static int downscale(int v)  { return v; }
    };

    //--------------------------------------------------------ras_conv_int_sat
    struct ras_conv_int_sat
    {
        typedef int coord_type;
        static AGG_INLINE int mul_div(double a, double b, double c)
        {
            return saturation<poly_max_coord>::iround(a * b / c);
        }
        static int xi(int v) { return v; }
        static int yi(int v) { return v; }
        static int upscale(double v) 
        { 
            return saturation<poly_max_coord>::iround(v * poly_subpixel_scale); 
        }
        static int downscale(int v) { return v; }
    };

    //---------------------------------------------------------ras_conv_int_3x
    struct ras_conv_int_3x
    {
        typedef int coord_type;
        static AGG_INLINE int mul_div(double a, double b, double c)
        {
            return iround(a * b / c);
        }
        static int xi(int v) { return v * 3; }
        static int yi(int v) { return v; }
        static int upscale(double v) { return iround(v * poly_subpixel_scale); }
        static int downscale(int v)  { return v; }
    };

    //-----------------------------------------------------------ras_conv_dbl
    struct ras_conv_dbl
    {
        typedef double coord_type;
        static AGG_INLINE double mul_div(double a, double b, double c)
        {
            return a * b / c;
        }
        static int xi(double v) { return iround(v * poly_subpixel_scale); }
        static int yi(double v) { return iround(v * poly_subpixel_scale); }
        static double upscale(double v) { return v; }
        static double downscale(int v)  { return v / double(poly_subpixel_scale); }
    };

    //--------------------------------------------------------ras_conv_dbl_3x
    struct ras_conv_dbl_3x
    {
        typedef double coord_type;
        static AGG_INLINE double mul_div(double a, double b, double c)
        {
            return a * b / c;
        }
        static int xi(double v) { return iround(v * poly_subpixel_scale * 3); }
        static int yi(double v) { return iround(v * poly_subpixel_scale); }
        static double upscale(double v) { return v; }
        static double downscale(int v)  { return v / double(poly_subpixel_scale); }
    };





    //------------------------------------------------------rasterizer_sl_clip
    template<class Conv> class rasterizer_sl_clip
    {
    public:
        typedef Conv                      conv_type;
        typedef typename Conv::coord_type coord_type;
        typedef rect_base<coord_type>     rect_type;

        //--------------------------------------------------------------------
        rasterizer_sl_clip() :  
            m_clip_box(0,0,0,0),
            m_x1(0),
            m_y1(0),
            m_f1(0),
            m_clipping(false) 
        {}

        //--------------------------------------------------------------------
        void reset_clipping()
        {
            m_clipping = false;
        }

        //--------------------------------------------------------------------
        void clip_box(coord_type x1, coord_type y1, coord_type x2, coord_type y2)
        {
            m_clip_box = rect_type(x1, y1, x2, y2);
            m_clip_box.normalize();
            m_clipping = true;
        }

        //--------------------------------------------------------------------
        void move_to(coord_type x1, coord_type y1)
        {
            m_x1 = x1;
            m_y1 = y1;
            if(m_clipping) m_f1 = clipping_flags(x1, y1, m_clip_box);
        }

    private:
        //------------------------------------------------------------------------
        template<class Rasterizer>
        AGG_INLINE void line_clip_y(Rasterizer& ras,
                                    coord_type x1, coord_type y1, 
                                    coord_type x2, coord_type y2, 
                                    unsigned   f1, unsigned   f2) const
        {
            f1 &= 10;
            f2 &= 10;
            if((f1 | f2) == 0)
            {
                // Fully visible
                ras.line(Conv::xi(x1), Conv::yi(y1), Conv::xi(x2), Conv::yi(y2)); 
            }
            else
            {
                if(f1 == f2)
                {
                    // Invisible by Y
                    return;
                }

                coord_type tx1 = x1;
                coord_type ty1 = y1;
                coord_type tx2 = x2;
                coord_type ty2 = y2;

                if(f1 & 8) // y1 < clip.y1
                {
                    tx1 = x1 + Conv::mul_div(m_clip_box.y1-y1, x2-x1, y2-y1);
                    ty1 = m_clip_box.y1;
                }

                if(f1 & 2) // y1 > clip.y2
                {
                    tx1 = x1 + Conv::mul_div(m_clip_box.y2-y1, x2-x1, y2-y1);
                    ty1 = m_clip_box.y2;
                }

                if(f2 & 8) // y2 < clip.y1
                {
                    tx2 = x1 + Conv::mul_div(m_clip_box.y1-y1, x2-x1, y2-y1);
                    ty2 = m_clip_box.y1;
                }

                if(f2 & 2) // y2 > clip.y2
                {
                    tx2 = x1 + Conv::mul_div(m_clip_box.y2-y1, x2-x1, y2-y1);
                    ty2 = m_clip_box.y2;
                }
                ras.line(Conv::xi(tx1), Conv::yi(ty1), 
                         Conv::xi(tx2), Conv::yi(ty2)); 
            }
        }


    public:
        //--------------------------------------------------------------------
        template<class Rasterizer>
        void line_to(Rasterizer& ras, coord_type x2, coord_type y2)
        {
            if(m_clipping)
            {
                unsigned f2 = clipping_flags(x2, y2, m_clip_box);

                if((m_f1 & 10) == (f2 & 10) && (m_f1 & 10) != 0)
                {
                    // Invisible by Y
                    m_x1 = x2;
                    m_y1 = y2;
                    m_f1 = f2;
                    return;
                }

                coord_type x1 = m_x1;
                coord_type y1 = m_y1;
                unsigned   f1 = m_f1;
                coord_type y3, y4;
                unsigned   f3, f4;

                switch(((f1 & 5) << 1) | (f2 & 5))
                {
                case 0: // Visible by X
                    line_clip_y(ras, x1, y1, x2, y2, f1, f2);
                    break;

                case 1: // x2 > clip.x2
                    y3 = y1 + Conv::mul_div(m_clip_box.x2-x1, y2-y1, x2-x1);
                    f3 = clipping_flags_y(y3, m_clip_box);
                    line_clip_y(ras, x1, y1, m_clip_box.x2, y3, f1, f3);
                    line_clip_y(ras, m_clip_box.x2, y3, m_clip_box.x2, y2, f3, f2);
                    break;

                case 2: // x1 > clip.x2
                    y3 = y1 + Conv::mul_div(m_clip_box.x2-x1, y2-y1, x2-x1);
                    f3 = clipping_flags_y(y3, m_clip_box);
                    line_clip_y(ras, m_clip_box.x2, y1, m_clip_box.x2, y3, f1, f3);
                    line_clip_y(ras, m_clip_box.x2, y3, x2, y2, f3, f2);
                    break;

                case 3: // x1 > clip.x2 && x2 > clip.x2
                    line_clip_y(ras, m_clip_box.x2, y1, m_clip_box.x2, y2, f1, f2);
                    break;

                case 4: // x2 < clip.x1
                    y3 = y1 + Conv::mul_div(m_clip_box.x1-x1, y2-y1, x2-x1);
                    f3 = clipping_flags_y(y3, m_clip_box);
                    line_clip_y(ras, x1, y1, m_clip_box.x1, y3, f1, f3);
                    line_clip_y(ras, m_clip_box.x1, y3, m_clip_box.x1, y2, f3, f2);
                    break;

                case 6: // x1 > clip.x2 && x2 < clip.x1
                    y3 = y1 + Conv::mul_div(m_clip_box.x2-x1, y2-y1, x2-x1);
                    y4 = y1 + Conv::mul_div(m_clip_box.x1-x1, y2-y1, x2-x1);
                    f3 = clipping_flags_y(y3, m_clip_box);
                    f4 = clipping_flags_y(y4, m_clip_box);
                    line_clip_y(ras, m_clip_box.x2, y1, m_clip_box.x2, y3, f1, f3);
                    line_clip_y(ras, m_clip_box.x2, y3, m_clip_box.x1, y4, f3, f4);
                    line_clip_y(ras, m_clip_box.x1, y4, m_clip_box.x1, y2, f4, f2);
                    break;

                case 8: // x1 < clip.x1
                    y3 = y1 + Conv::mul_div(m_clip_box.x1-x1, y2-y1, x2-x1);
                    f3 = clipping_flags_y(y3, m_clip_box);
                    line_clip_y(ras, m_clip_box.x1, y1, m_clip_box.x1, y3, f1, f3);
                    line_clip_y(ras, m_clip_box.x1, y3, x2, y2, f3, f2);
                    break;

                case 9:  // x1 < clip.x1 && x2 > clip.x2
                    y3 = y1 + Conv::mul_div(m_clip_box.x1-x1, y2-y1, x2-x1);
                    y4 = y1 + Conv::mul_div(m_clip_box.x2-x1, y2-y1, x2-x1);
                    f3 = clipping_flags_y(y3, m_clip_box);
                    f4 = clipping_flags_y(y4, m_clip_box);
                    line_clip_y(ras, m_clip_box.x1, y1, m_clip_box.x1, y3, f1, f3);
                    line_clip_y(ras, m_clip_box.x1, y3, m_clip_box.x2, y4, f3, f4);
                    line_clip_y(ras, m_clip_box.x2, y4, m_clip_box.x2, y2, f4, f2);
                    break;

                case 12: // x1 < clip.x1 && x2 < clip.x1
                    line_clip_y(ras, m_clip_box.x1, y1, m_clip_box.x1, y2, f1, f2);
                    break;
                }
                m_f1 = f2;
            }
            else
            {
                ras.line(Conv::xi(m_x1), Conv::yi(m_y1), 
                         Conv::xi(x2),   Conv::yi(y2)); 
            }
            m_x1 = x2;
            m_y1 = y2;
        }


    private:
        rect_type        m_clip_box;
        coord_type       m_x1;
        coord_type       m_y1;
        unsigned         m_f1;
        bool             m_clipping;
    };




    //---------------------------------------------------rasterizer_sl_no_clip
    class rasterizer_sl_no_clip
    {
    public:
        typedef ras_conv_int conv_type;
        typedef int          coord_type;

        rasterizer_sl_no_clip() : m_x1(0), m_y1(0) {}

        void reset_clipping() {}
        void clip_box(coord_type x1, coord_type y1, coord_type x2, coord_type y2) {}
        void move_to(coord_type x1, coord_type y1) { m_x1 = x1; m_y1 = y1; }

        template<class Rasterizer>
        void line_to(Rasterizer& ras, coord_type x2, coord_type y2) 
        { 
            ras.line(m_x1, m_y1, x2, y2); 
            m_x1 = x2; 
            m_y1 = y2;
        }

    private:
        int m_x1, m_y1;
    };


    //                                         -----rasterizer_sl_clip_int
    //                                         -----rasterizer_sl_clip_int_sat
    //                                         -----rasterizer_sl_clip_int_3x
    //                                         -----rasterizer_sl_clip_dbl
    //                                         -----rasterizer_sl_clip_dbl_3x
    //------------------------------------------------------------------------
    typedef rasterizer_sl_clip<ras_conv_int>     rasterizer_sl_clip_int;
    typedef rasterizer_sl_clip<ras_conv_int_sat> rasterizer_sl_clip_int_sat;
    typedef rasterizer_sl_clip<ras_conv_int_3x>  rasterizer_sl_clip_int_3x;
    typedef rasterizer_sl_clip<ras_conv_dbl>     rasterizer_sl_clip_dbl;
    typedef rasterizer_sl_clip<ras_conv_dbl_3x>  rasterizer_sl_clip_dbl_3x;


}

#endif
