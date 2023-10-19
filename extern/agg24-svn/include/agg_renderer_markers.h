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
// class renderer_markers
//
//----------------------------------------------------------------------------

#ifndef AGG_RENDERER_MARKERS_INCLUDED
#define AGG_RENDERER_MARKERS_INCLUDED

#include "agg_basics.h"
#include "agg_renderer_primitives.h"

namespace agg
{

    //---------------------------------------------------------------marker_e
    enum marker_e
    {
        marker_square,
        marker_diamond,
        marker_circle,
        marker_crossed_circle,
        marker_semiellipse_left,
        marker_semiellipse_right,
        marker_semiellipse_up,
        marker_semiellipse_down,
        marker_triangle_left,
        marker_triangle_right,
        marker_triangle_up,
        marker_triangle_down,
        marker_four_rays,
        marker_cross,
        marker_x,
        marker_dash,
        marker_dot,
        marker_pixel,
        
        end_of_markers
    };



    //--------------------------------------------------------renderer_markers
    template<class BaseRenderer> class renderer_markers :
    public renderer_primitives<BaseRenderer>
    {
    public:
        typedef renderer_primitives<BaseRenderer> base_type;
        typedef BaseRenderer base_ren_type;
        typedef typename base_ren_type::color_type color_type;

        //--------------------------------------------------------------------
        renderer_markers(base_ren_type& rbuf) :
            base_type(rbuf)
        {}

        //--------------------------------------------------------------------
        bool visible(int x, int y, int r) const
        {
            rect_i rc(x-r, y-r, x+y, y+r);
            return rc.clip(base_type::ren().bounding_clip_box());  
        }

        //--------------------------------------------------------------------
        void square(int x, int y, int r)
        {
            if(visible(x, y, r)) 
            {  
                if(r) base_type::outlined_rectangle(x-r, y-r, x+r, y+r);
                else  base_type::ren().blend_pixel(x, y, base_type::fill_color(), cover_full);
            }
        }

        //--------------------------------------------------------------------
        void diamond(int x, int y, int r)
        {
            if(visible(x, y, r))
            {
                if(r)
                {
                    int dy = -r;
                    int dx = 0;
                    do
                    {
                        base_type::ren().blend_pixel(x - dx, y + dy, base_type::line_color(), cover_full);
                        base_type::ren().blend_pixel(x + dx, y + dy, base_type::line_color(), cover_full);
                        base_type::ren().blend_pixel(x - dx, y - dy, base_type::line_color(), cover_full);
                        base_type::ren().blend_pixel(x + dx, y - dy, base_type::line_color(), cover_full);
                        
                        if(dx)
                        {
                            base_type::ren().blend_hline(x-dx+1, y+dy, x+dx-1, base_type::fill_color(), cover_full);
                            base_type::ren().blend_hline(x-dx+1, y-dy, x+dx-1, base_type::fill_color(), cover_full);
                        }
                        ++dy;
                        ++dx;
                    }
                    while(dy <= 0);
                }
                else
                {
                    base_type::ren().blend_pixel(x, y, base_type::fill_color(), cover_full);
                }
            }
        }

        //--------------------------------------------------------------------
        void circle(int x, int y, int r)
        {
            if(visible(x, y, r))
            {
                if(r) base_type::outlined_ellipse(x, y, r, r);
                else  base_type::ren().blend_pixel(x, y, base_type::fill_color(), cover_full);
            }
        }



        //--------------------------------------------------------------------
        void crossed_circle(int x, int y, int r)
        {
            if(visible(x, y, r))
            {
                if(r)
                {
                    base_type::outlined_ellipse(x, y, r, r);
                    int r6 = r + (r >> 1);
                    if(r <= 2) r6++;
                    r >>= 1;
                    base_type::ren().blend_hline(x-r6, y, x-r,  base_type::line_color(), cover_full);
                    base_type::ren().blend_hline(x+r,  y, x+r6, base_type::line_color(), cover_full);
                    base_type::ren().blend_vline(x, y-r6, y-r,  base_type::line_color(), cover_full);
                    base_type::ren().blend_vline(x, y+r,  y+r6, base_type::line_color(), cover_full);
                }
                else
                {
                    base_type::ren().blend_pixel(x, y, base_type::fill_color(), cover_full);
                }
            }
        }


        //------------------------------------------------------------------------
        void semiellipse_left(int x, int y, int r)
        {
            if(visible(x, y, r))
            {
                if(r)
                {
                    int r8 = r * 4 / 5;
                    int dy = -r;
                    int dx = 0;
                    ellipse_bresenham_interpolator ei(r * 3 / 5, r+r8);
                    do
                    {
                        dx += ei.dx();
                        dy += ei.dy();
                        
                        base_type::ren().blend_pixel(x + dy, y + dx, base_type::line_color(), cover_full);
                        base_type::ren().blend_pixel(x + dy, y - dx, base_type::line_color(), cover_full);
                        
                        if(ei.dy() && dx)
                        {
                            base_type::ren().blend_vline(x+dy, y-dx+1, y+dx-1, base_type::fill_color(), cover_full);
                        }
                        ++ei;
                    }
                    while(dy < r8);
                    base_type::ren().blend_vline(x+dy, y-dx, y+dx, base_type::line_color(), cover_full);
                }
                else
                {
                    base_type::ren().blend_pixel(x, y, base_type::fill_color(), cover_full);
                }
            }
        }


        //--------------------------------------------------------------------
        void semiellipse_right(int x, int y, int r)
        {
            if(visible(x, y, r))
            {
                if(r)
                {
                    int r8 = r * 4 / 5;
                    int dy = -r;
                    int dx = 0;
                    ellipse_bresenham_interpolator ei(r * 3 / 5, r+r8);
                    do
                    {
                        dx += ei.dx();
                        dy += ei.dy();
                        
                        base_type::ren().blend_pixel(x - dy, y + dx, base_type::line_color(), cover_full);
                        base_type::ren().blend_pixel(x - dy, y - dx, base_type::line_color(), cover_full);
                        
                        if(ei.dy() && dx)
                        {
                            base_type::ren().blend_vline(x-dy, y-dx+1, y+dx-1, base_type::fill_color(), cover_full);
                        }
                        ++ei;
                    }
                    while(dy < r8);
                    base_type::ren().blend_vline(x-dy, y-dx, y+dx, base_type::line_color(), cover_full);
                }
                else
                {
                    base_type::ren().blend_pixel(x, y, base_type::fill_color(), cover_full);
                }
            }
        }


        //--------------------------------------------------------------------
        void semiellipse_up(int x, int y, int r)
        {
            if(visible(x, y, r))
            {
                if(r)
                {
                    int r8 = r * 4 / 5;
                    int dy = -r;
                    int dx = 0;
                    ellipse_bresenham_interpolator ei(r * 3 / 5, r+r8);
                    do
                    {
                        dx += ei.dx();
                        dy += ei.dy();
                        
                        base_type::ren().blend_pixel(x + dx, y - dy, base_type::line_color(), cover_full);
                        base_type::ren().blend_pixel(x - dx, y - dy, base_type::line_color(), cover_full);
                        
                        if(ei.dy() && dx)
                        {
                            base_type::ren().blend_hline(x-dx+1, y-dy, x+dx-1, base_type::fill_color(), cover_full);
                        }
                        ++ei;
                    }
                    while(dy < r8);
                    base_type::ren().blend_hline(x-dx, y-dy-1, x+dx, base_type::line_color(), cover_full);
                }
                else
                {
                    base_type::ren().blend_pixel(x, y, base_type::fill_color(), cover_full);
                }
            }
        }


        //--------------------------------------------------------------------
        void semiellipse_down(int x, int y, int r)
        {
            if(visible(x, y, r))
            {
                if(r)
                {
                    int r8 = r * 4 / 5;
                    int dy = -r;
                    int dx = 0;
                    ellipse_bresenham_interpolator ei(r * 3 / 5, r+r8);
                    do
                    {
                        dx += ei.dx();
                        dy += ei.dy();
                        
                        base_type::ren().blend_pixel(x + dx, y + dy, base_type::line_color(), cover_full);
                        base_type::ren().blend_pixel(x - dx, y + dy, base_type::line_color(), cover_full);
                        
                        if(ei.dy() && dx)
                        {
                            base_type::ren().blend_hline(x-dx+1, y+dy, x+dx-1, base_type::fill_color(), cover_full);
                        }
                        ++ei;
                    }
                    while(dy < r8);
                    base_type::ren().blend_hline(x-dx, y+dy+1, x+dx, base_type::line_color(), cover_full);
                }
                else
                {
                    base_type::ren().blend_pixel(x, y, base_type::fill_color(), cover_full);
                }
            }
        }


        //--------------------------------------------------------------------
        void triangle_left(int x, int y, int r)
        {
            if(visible(x, y, r))
            {
                if(r)
                {
                    int dy = -r;
                    int dx = 0;
                    int flip = 0;
                    int r6 = r * 3 / 5;
                    do
                    {
                        base_type::ren().blend_pixel(x + dy, y - dx, base_type::line_color(), cover_full);
                        base_type::ren().blend_pixel(x + dy, y + dx, base_type::line_color(), cover_full);
                        
                        if(dx)
                        {
                            base_type::ren().blend_vline(x+dy, y-dx+1, y+dx-1, base_type::fill_color(), cover_full);
                        }
                        ++dy;
                        dx += flip;
                        flip ^= 1;
                    }
                    while(dy < r6);
                    base_type::ren().blend_vline(x+dy, y-dx, y+dx, base_type::line_color(), cover_full);
                }
                else
                {
                    base_type::ren().blend_pixel(x, y, base_type::fill_color(), cover_full);
                }
            }
        }


        //--------------------------------------------------------------------
        void triangle_right(int x, int y, int r)
        {
            if(visible(x, y, r))
            {
                if(r)
                {
                    int dy = -r;
                    int dx = 0;
                    int flip = 0;
                    int r6 = r * 3 / 5;
                    do
                    {
                        base_type::ren().blend_pixel(x - dy, y - dx, base_type::line_color(), cover_full);
                        base_type::ren().blend_pixel(x - dy, y + dx, base_type::line_color(), cover_full);
                        
                        if(dx)
                        {
                            base_type::ren().blend_vline(x-dy, y-dx+1, y+dx-1, base_type::fill_color(), cover_full);
                        }
                        ++dy;
                        dx += flip;
                        flip ^= 1;
                    }
                    while(dy < r6);
                    base_type::ren().blend_vline(x-dy, y-dx, y+dx, base_type::line_color(), cover_full);
                }
                else
                {
                    base_type::ren().blend_pixel(x, y, base_type::fill_color(), cover_full);
                }
            }
        }


        //--------------------------------------------------------------------
        void triangle_up(int x, int y, int r)
        {
            if(visible(x, y, r))
            {
                if(r)
                {
                    int dy = -r;
                    int dx = 0;
                    int flip = 0;
                    int r6 = r * 3 / 5;
                    do
                    {
                        base_type::ren().blend_pixel(x - dx, y - dy, base_type::line_color(), cover_full);
                        base_type::ren().blend_pixel(x + dx, y - dy, base_type::line_color(), cover_full);
                        
                        if(dx)
                        {
                            base_type::ren().blend_hline(x-dx+1, y-dy, x+dx-1, base_type::fill_color(), cover_full);
                        }
                        ++dy;
                        dx += flip;
                        flip ^= 1;
                    }
                    while(dy < r6);
                    base_type::ren().blend_hline(x-dx, y-dy, x+dx, base_type::line_color(), cover_full);
                }
                else
                {
                    base_type::ren().blend_pixel(x, y, base_type::fill_color(), cover_full);
                }
            }
        }


        //--------------------------------------------------------------------
        void triangle_down(int x, int y, int r)
        {
            if(visible(x, y, r))
            {
                if(r)
                {
                    int dy = -r;
                    int dx = 0;
                    int flip = 0;
                    int r6 = r * 3 / 5;
                    do
                    {
                        base_type::ren().blend_pixel(x - dx, y + dy, base_type::line_color(), cover_full);
                        base_type::ren().blend_pixel(x + dx, y + dy, base_type::line_color(), cover_full);
                        
                        if(dx)
                        {
                            base_type::ren().blend_hline(x-dx+1, y+dy, x+dx-1, base_type::fill_color(), cover_full);
                        }
                        ++dy;
                        dx += flip;
                        flip ^= 1;
                    }
                    while(dy < r6);
                    base_type::ren().blend_hline(x-dx, y+dy, x+dx, base_type::line_color(), cover_full);
                }
                else
                {
                    base_type::ren().blend_pixel(x, y, base_type::fill_color(), cover_full);
                }
            }
        }


        //--------------------------------------------------------------------
        void four_rays(int x, int y, int r)
        {
            if(visible(x, y, r))
            {
                if(r)
                {
                    int dy = -r;
                    int dx = 0;
                    int flip = 0;
                    int r3 = -(r / 3);
                    do
                    {
                        base_type::ren().blend_pixel(x - dx, y + dy, base_type::line_color(), cover_full);
                        base_type::ren().blend_pixel(x + dx, y + dy, base_type::line_color(), cover_full);
                        base_type::ren().blend_pixel(x - dx, y - dy, base_type::line_color(), cover_full);
                        base_type::ren().blend_pixel(x + dx, y - dy, base_type::line_color(), cover_full);
                        base_type::ren().blend_pixel(x + dy, y - dx, base_type::line_color(), cover_full);
                        base_type::ren().blend_pixel(x + dy, y + dx, base_type::line_color(), cover_full);
                        base_type::ren().blend_pixel(x - dy, y - dx, base_type::line_color(), cover_full);
                        base_type::ren().blend_pixel(x - dy, y + dx, base_type::line_color(), cover_full);
                        
                        if(dx)
                        {
                            base_type::ren().blend_hline(x-dx+1, y+dy,   x+dx-1, base_type::fill_color(), cover_full);
                            base_type::ren().blend_hline(x-dx+1, y-dy,   x+dx-1, base_type::fill_color(), cover_full);
                            base_type::ren().blend_vline(x+dy,   y-dx+1, y+dx-1, base_type::fill_color(), cover_full);
                            base_type::ren().blend_vline(x-dy,   y-dx+1, y+dx-1, base_type::fill_color(), cover_full);
                        }
                        ++dy;
                        dx += flip;
                        flip ^= 1;
                    }
                    while(dy <= r3);
                    base_type::solid_rectangle(x+r3+1, y+r3+1, x-r3-1, y-r3-1);
                }
                else
                {
                    base_type::ren().blend_pixel(x, y, base_type::fill_color(), cover_full);
                }
            }
        }


        //--------------------------------------------------------------------
        void cross(int x, int y, int r)
        {
            if(visible(x, y, r))
            {
                if(r)
                {
                    base_type::ren().blend_vline(x, y-r, y+r, base_type::line_color(), cover_full);
                    base_type::ren().blend_hline(x-r, y, x+r, base_type::line_color(), cover_full);
                }
                else
                {
                    base_type::ren().blend_pixel(x, y, base_type::fill_color(), cover_full);
                }
            }
        }
        
        
        //--------------------------------------------------------------------
        void xing(int x, int y, int r)
        {
            if(visible(x, y, r))
            {
                if(r)
                {
                    int dy = -r * 7 / 10;
                    do
                    {
                        base_type::ren().blend_pixel(x + dy, y + dy, base_type::line_color(), cover_full);
                        base_type::ren().blend_pixel(x - dy, y + dy, base_type::line_color(), cover_full);
                        base_type::ren().blend_pixel(x + dy, y - dy, base_type::line_color(), cover_full);
                        base_type::ren().blend_pixel(x - dy, y - dy, base_type::line_color(), cover_full);
                        ++dy;
                    }
                    while(dy < 0);
                }
                base_type::ren().blend_pixel(x, y, base_type::fill_color(), cover_full);
            }
        }
        
        
        //--------------------------------------------------------------------
        void dash(int x, int y, int r)
        {
            if(visible(x, y, r)) 
            {
                if(r) base_type::ren().blend_hline(x-r, y, x+r, base_type::line_color(), cover_full);
                else  base_type::ren().blend_pixel(x, y, base_type::fill_color(), cover_full);
            }
        }
        
        
        //--------------------------------------------------------------------
        void dot(int x, int y, int r)
        {
            if(visible(x, y, r)) 
            {
                if(r) base_type::solid_ellipse(x, y, r, r);
                else  base_type::ren().blend_pixel(x, y, base_type::fill_color(), cover_full);
            }
        }
        
        //--------------------------------------------------------------------
        void pixel(int x, int y, int)
        {
            base_type::ren().blend_pixel(x, y, base_type::fill_color(), cover_full);
        }
        
        //--------------------------------------------------------------------
        void marker(int x, int y, int r, marker_e type)
        {
            switch(type)
            {
                case marker_square:            square(x, y, r);            break;
                case marker_diamond:           diamond(x, y, r);           break;
                case marker_circle:            circle(x, y, r);            break;
                case marker_crossed_circle:    crossed_circle(x, y, r);    break;
                case marker_semiellipse_left:  semiellipse_left(x, y, r);  break;
                case marker_semiellipse_right: semiellipse_right(x, y, r); break;
                case marker_semiellipse_up:    semiellipse_up(x, y, r);    break;
                case marker_semiellipse_down:  semiellipse_down(x, y, r);  break;
                case marker_triangle_left:     triangle_left(x, y, r);     break;
                case marker_triangle_right:    triangle_right(x, y, r);    break;
                case marker_triangle_up:       triangle_up(x, y, r);       break;
                case marker_triangle_down:     triangle_down(x, y, r);     break;
                case marker_four_rays:         four_rays(x, y, r);         break;
                case marker_cross:             cross(x, y, r);             break;
                case marker_x:                 xing(x, y, r);              break;
                case marker_dash:              dash(x, y, r);              break;
                case marker_dot:               dot(x, y, r);               break;
                case marker_pixel:             pixel(x, y, r);             break;
            }
        }


        //--------------------------------------------------------------------
        template<class T>
        void markers(int n, const T* x, const T* y, T r, marker_e type)
        {
            if(n <= 0) return;
            if(r == 0)
            {
                do
                {
                    base_type::ren().blend_pixel(int(*x), int(*y), base_type::fill_color(), cover_full);
                    ++x;
                    ++y;
                }
                while(--n);
                return;
            }
            
            switch(type)
            {
                case marker_square:            do { square           (int(*x), int(*y), int(r)); ++x; ++y; } while(--n); break;
                case marker_diamond:           do { diamond          (int(*x), int(*y), int(r)); ++x; ++y; } while(--n); break;
                case marker_circle:            do { circle           (int(*x), int(*y), int(r)); ++x; ++y; } while(--n); break;
                case marker_crossed_circle:    do { crossed_circle   (int(*x), int(*y), int(r)); ++x; ++y; } while(--n); break;
                case marker_semiellipse_left:  do { semiellipse_left (int(*x), int(*y), int(r)); ++x; ++y; } while(--n); break;
                case marker_semiellipse_right: do { semiellipse_right(int(*x), int(*y), int(r)); ++x; ++y; } while(--n); break;
                case marker_semiellipse_up:    do { semiellipse_up   (int(*x), int(*y), int(r)); ++x; ++y; } while(--n); break;
                case marker_semiellipse_down:  do { semiellipse_down (int(*x), int(*y), int(r)); ++x; ++y; } while(--n); break;
                case marker_triangle_left:     do { triangle_left    (int(*x), int(*y), int(r)); ++x; ++y; } while(--n); break;
                case marker_triangle_right:    do { triangle_right   (int(*x), int(*y), int(r)); ++x; ++y; } while(--n); break;
                case marker_triangle_up:       do { triangle_up      (int(*x), int(*y), int(r)); ++x; ++y; } while(--n); break;
                case marker_triangle_down:     do { triangle_down    (int(*x), int(*y), int(r)); ++x; ++y; } while(--n); break;
                case marker_four_rays:         do { four_rays        (int(*x), int(*y), int(r)); ++x; ++y; } while(--n); break;
                case marker_cross:             do { cross            (int(*x), int(*y), int(r)); ++x; ++y; } while(--n); break;
                case marker_x:                 do { xing             (int(*x), int(*y), int(r)); ++x; ++y; } while(--n); break;
                case marker_dash:              do { dash             (int(*x), int(*y), int(r)); ++x; ++y; } while(--n); break;
                case marker_dot:               do { dot              (int(*x), int(*y), int(r)); ++x; ++y; } while(--n); break;
                case marker_pixel:             do { pixel            (int(*x), int(*y), int(r)); ++x; ++y; } while(--n); break;
            }                                                                                  
        }
        
        //--------------------------------------------------------------------
        template<class T>
        void markers(int n, const T* x, const T* y, const T* r, marker_e type)
        {
            if(n <= 0) return;
            switch(type)
            {
                case marker_square:            do { square           (int(*x), int(*y), int(*r)); ++x; ++y; ++r; } while(--n); break;
                case marker_diamond:           do { diamond          (int(*x), int(*y), int(*r)); ++x; ++y; ++r; } while(--n); break;
                case marker_circle:            do { circle           (int(*x), int(*y), int(*r)); ++x; ++y; ++r; } while(--n); break;
                case marker_crossed_circle:    do { crossed_circle   (int(*x), int(*y), int(*r)); ++x; ++y; ++r; } while(--n); break;
                case marker_semiellipse_left:  do { semiellipse_left (int(*x), int(*y), int(*r)); ++x; ++y; ++r; } while(--n); break;
                case marker_semiellipse_right: do { semiellipse_right(int(*x), int(*y), int(*r)); ++x; ++y; ++r; } while(--n); break;
                case marker_semiellipse_up:    do { semiellipse_up   (int(*x), int(*y), int(*r)); ++x; ++y; ++r; } while(--n); break;
                case marker_semiellipse_down:  do { semiellipse_down (int(*x), int(*y), int(*r)); ++x; ++y; ++r; } while(--n); break;
                case marker_triangle_left:     do { triangle_left    (int(*x), int(*y), int(*r)); ++x; ++y; ++r; } while(--n); break;
                case marker_triangle_right:    do { triangle_right   (int(*x), int(*y), int(*r)); ++x; ++y; ++r; } while(--n); break;
                case marker_triangle_up:       do { triangle_up      (int(*x), int(*y), int(*r)); ++x; ++y; ++r; } while(--n); break;
                case marker_triangle_down:     do { triangle_down    (int(*x), int(*y), int(*r)); ++x; ++y; ++r; } while(--n); break;
                case marker_four_rays:         do { four_rays        (int(*x), int(*y), int(*r)); ++x; ++y; ++r; } while(--n); break;
                case marker_cross:             do { cross            (int(*x), int(*y), int(*r)); ++x; ++y; ++r; } while(--n); break;
                case marker_x:                 do { xing             (int(*x), int(*y), int(*r)); ++x; ++y; ++r; } while(--n); break;
                case marker_dash:              do { dash             (int(*x), int(*y), int(*r)); ++x; ++y; ++r; } while(--n); break;
                case marker_dot:               do { dot              (int(*x), int(*y), int(*r)); ++x; ++y; ++r; } while(--n); break;
                case marker_pixel:             do { pixel            (int(*x), int(*y), int(*r)); ++x; ++y; ++r; } while(--n); break;
            }                                                                                  
        }
        
        //--------------------------------------------------------------------
        template<class T>
        void markers(int n, const T* x, const T* y, const T* r, const color_type* fc, marker_e type)
        {
            if(n <= 0) return;
            switch(type)
            {
                case marker_square:            do { base_type::fill_color(*fc); square           (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; } while(--n); break;
                case marker_diamond:           do { base_type::fill_color(*fc); diamond          (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; } while(--n); break;
                case marker_circle:            do { base_type::fill_color(*fc); circle           (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; } while(--n); break;
                case marker_crossed_circle:    do { base_type::fill_color(*fc); crossed_circle   (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; } while(--n); break;
                case marker_semiellipse_left:  do { base_type::fill_color(*fc); semiellipse_left (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; } while(--n); break;
                case marker_semiellipse_right: do { base_type::fill_color(*fc); semiellipse_right(int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; } while(--n); break;
                case marker_semiellipse_up:    do { base_type::fill_color(*fc); semiellipse_up   (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; } while(--n); break;
                case marker_semiellipse_down:  do { base_type::fill_color(*fc); semiellipse_down (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; } while(--n); break;
                case marker_triangle_left:     do { base_type::fill_color(*fc); triangle_left    (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; } while(--n); break;
                case marker_triangle_right:    do { base_type::fill_color(*fc); triangle_right   (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; } while(--n); break;
                case marker_triangle_up:       do { base_type::fill_color(*fc); triangle_up      (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; } while(--n); break;
                case marker_triangle_down:     do { base_type::fill_color(*fc); triangle_down    (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; } while(--n); break;
                case marker_four_rays:         do { base_type::fill_color(*fc); four_rays        (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; } while(--n); break;
                case marker_cross:             do { base_type::fill_color(*fc); cross            (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; } while(--n); break;
                case marker_x:                 do { base_type::fill_color(*fc); xing             (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; } while(--n); break;
                case marker_dash:              do { base_type::fill_color(*fc); dash             (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; } while(--n); break;
                case marker_dot:               do { base_type::fill_color(*fc); dot              (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; } while(--n); break;
                case marker_pixel:             do { base_type::fill_color(*fc); pixel            (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; } while(--n); break;
            }
        }
        
        //--------------------------------------------------------------------
        template<class T>
        void markers(int n, const T* x, const T* y, const T* r, const color_type* fc, const color_type* lc, marker_e type)
        {
            if(n <= 0) return;
            switch(type)
            {
                case marker_square:            do { base_type::fill_color(*fc); base_type::line_color(*lc); square           (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; ++lc; } while(--n); break;
                case marker_diamond:           do { base_type::fill_color(*fc); base_type::line_color(*lc); diamond          (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; ++lc; } while(--n); break;
                case marker_circle:            do { base_type::fill_color(*fc); base_type::line_color(*lc); circle           (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; ++lc; } while(--n); break;
                case marker_crossed_circle:    do { base_type::fill_color(*fc); base_type::line_color(*lc); crossed_circle   (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; ++lc; } while(--n); break;
                case marker_semiellipse_left:  do { base_type::fill_color(*fc); base_type::line_color(*lc); semiellipse_left (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; ++lc; } while(--n); break;
                case marker_semiellipse_right: do { base_type::fill_color(*fc); base_type::line_color(*lc); semiellipse_right(int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; ++lc; } while(--n); break;
                case marker_semiellipse_up:    do { base_type::fill_color(*fc); base_type::line_color(*lc); semiellipse_up   (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; ++lc; } while(--n); break;
                case marker_semiellipse_down:  do { base_type::fill_color(*fc); base_type::line_color(*lc); semiellipse_down (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; ++lc; } while(--n); break;
                case marker_triangle_left:     do { base_type::fill_color(*fc); base_type::line_color(*lc); triangle_left    (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; ++lc; } while(--n); break;
                case marker_triangle_right:    do { base_type::fill_color(*fc); base_type::line_color(*lc); triangle_right   (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; ++lc; } while(--n); break;
                case marker_triangle_up:       do { base_type::fill_color(*fc); base_type::line_color(*lc); triangle_up      (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; ++lc; } while(--n); break;
                case marker_triangle_down:     do { base_type::fill_color(*fc); base_type::line_color(*lc); triangle_down    (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; ++lc; } while(--n); break;
                case marker_four_rays:         do { base_type::fill_color(*fc); base_type::line_color(*lc); four_rays        (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; ++lc; } while(--n); break;
                case marker_cross:             do { base_type::fill_color(*fc); base_type::line_color(*lc); cross            (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; ++lc; } while(--n); break;
                case marker_x:                 do { base_type::fill_color(*fc); base_type::line_color(*lc); xing             (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; ++lc; } while(--n); break;
                case marker_dash:              do { base_type::fill_color(*fc); base_type::line_color(*lc); dash             (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; ++lc; } while(--n); break;
                case marker_dot:               do { base_type::fill_color(*fc); base_type::line_color(*lc); dot              (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; ++lc; } while(--n); break;
                case marker_pixel:             do { base_type::fill_color(*fc); base_type::line_color(*lc); pixel            (int(*x), int(*y), int(*r)); ++x; ++y; ++r; ++fc; ++lc; } while(--n); break;
            }
        }
    };

}

#endif
