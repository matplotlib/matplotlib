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

#ifndef AGG_BASICS_INCLUDED
#define AGG_BASICS_INCLUDED

#include "agg_config.h"

//-------------------------------------------------------- Default basic types
//
// If the compiler has different capacity of the basic types you can redefine
// them via the compiler command line or by generating agg_config.h that is
// empty by default.
//
#ifndef AGG_INT8
#define AGG_INT8 signed char
#endif

#ifndef AGG_INT8U
#define AGG_INT8U unsigned char
#endif

#ifndef AGG_INT16
#define AGG_INT16 short
#endif

#ifndef AGG_INT16U
#define AGG_INT16U unsigned short
#endif

#ifndef AGG_INT32
#define AGG_INT32 int
#endif

#ifndef AGG_INT32U
#define AGG_INT32U unsigned
#endif

#ifndef AGG_INT64
#if defined(_MSC_VER) || defined(__BORLANDC__)
#define AGG_INT64 signed __int64
#else
#define AGG_INT64 signed long long
#endif
#endif

#ifndef AGG_INT64U
#if defined(_MSC_VER) || defined(__BORLANDC__)
#define AGG_INT64U unsigned __int64
#else
#define AGG_INT64U unsigned long long
#endif
#endif

//------------------------------------------------ Some fixes for MS Visual C++
#if defined(_MSC_VER)
#pragma warning(disable:4786) // Identifier was truncated...
#endif

#if defined(_MSC_VER)
#define AGG_INLINE __forceinline
#else
#define AGG_INLINE inline
#endif

namespace agg
{
    //-------------------------------------------------------------------------
    typedef AGG_INT8   int8;         //----int8
    typedef AGG_INT8U  int8u;        //----int8u
    typedef AGG_INT16  int16;        //----int16
    typedef AGG_INT16U int16u;       //----int16u
    typedef AGG_INT32  int32;        //----int32
    typedef AGG_INT32U int32u;       //----int32u
    typedef AGG_INT64  int64;        //----int64
    typedef AGG_INT64U int64u;       //----int64u

    //-------------------------------------------------------------------------
    typedef unsigned char cover_type;    //----cover_type
    enum
    {
        cover_shift = 8,                 //----cover_shift
        cover_size  = 1 << cover_shift,  //----cover_size 
        cover_mask  = cover_size - 1,    //----cover_mask 
        cover_none  = 0,                 //----cover_none 
        cover_full  = cover_mask         //----cover_full 
    };


    //-----------------------------------------------------------------------pi
    const double pi = 3.14159265358979323846;

    //------------------------------------------------------------------deg2rad
    inline double deg2rad(double deg)
    {
        return deg * pi / 180.0;
    }

    //------------------------------------------------------------------rad2deg
    inline double rad2deg(double rad)
    {
        return rad * 180.0 / pi;
    }
 
    //----------------------------------------------------------------rect_base
    template<class T> struct rect_base
    {
        typedef rect_base<T> self_type;
        T x1;
        T y1;
        T x2;
        T y2;

        rect_base() {}
        rect_base(T x1_, T y1_, T x2_, T y2_) :
            x1(x1_), y1(y1_), x2(x2_), y2(y2_) {}

        const self_type& normalize()
        {
            T t;
            if(x1 > x2) { t = x1; x1 = x2; x2 = t; }
            if(y1 > y2) { t = y1; y1 = y2; y2 = t; }
            return *this;
        }

        bool clip(const self_type& r)
        {
            if(x2 > r.x2) x2 = r.x2;
            if(y2 > r.y2) y2 = r.y2;
            if(x1 < r.x1) x1 = r.x1;
            if(y1 < r.y1) y1 = r.y1;
            return x1 <= x2 && y1 <= y2;
        }

        bool is_valid() const
        {
            return x1 <= x2 && y1 <= y2;
        }
    };

    //-----------------------------------------------------intersect_rectangles
    template<class Rect> 
    inline Rect intersect_rectangles(const Rect& r1, const Rect& r2)
    {
        Rect r = r1;

        // First process x2,y2 because the other order 
        // results in Internal Compiler Error under 
        // Microsoft Visual C++ .NET 2003 69462-335-0000007-18038 in 
        // case of "Maximize Speed" optimization option.
        //-----------------
        if(r.x2 > r2.x2) r.x2 = r2.x2; 
        if(r.y2 > r2.y2) r.y2 = r2.y2;
        if(r.x1 < r2.x1) r.x1 = r2.x1;
        if(r.y1 < r2.y1) r.y1 = r2.y1;
        return r;
    }


    //---------------------------------------------------------unite_rectangles
    template<class Rect> 
    inline Rect unite_rectangles(const Rect& r1, const Rect& r2)
    {
        Rect r = r1;
        if(r.x2 < r2.x2) r.x2 = r2.x2;
        if(r.y2 < r2.y2) r.y2 = r2.y2;
        if(r.x1 > r2.x1) r.x1 = r2.x1;
        if(r.y1 > r2.y1) r.y1 = r2.y1;
        return r;
    }

    typedef rect_base<int>    rect;   //----rect
    typedef rect_base<double> rect_d; //----rect_d

    //---------------------------------------------------------path_commands_e
    enum path_commands_e
    {
        path_cmd_stop     = 0,        //----path_cmd_stop    
        path_cmd_move_to  = 1,        //----path_cmd_move_to 
        path_cmd_line_to  = 2,        //----path_cmd_line_to 
        path_cmd_curve3   = 3,        //----path_cmd_curve3  
        path_cmd_curve4   = 4,        //----path_cmd_curve4  
        path_cmd_curveN   = 5,        //----path_cmd_curveN
        path_cmd_catrom   = 6,        //----path_cmd_catrom
        path_cmd_ubspline = 7,        //----path_cmd_ubspline
        path_cmd_end_poly = 0x0F,     //----path_cmd_end_poly
        path_cmd_mask     = 0x0F      //----path_cmd_mask    
    };

    //------------------------------------------------------------path_flags_e
    enum path_flags_e
    {
        path_flags_none  = 0,         //----path_flags_none 
        path_flags_ccw   = 0x10,      //----path_flags_ccw  
        path_flags_cw    = 0x20,      //----path_flags_cw   
        path_flags_close = 0x40,      //----path_flags_close
        path_flags_mask  = 0xF0       //----path_flags_mask 
    };

    //---------------------------------------------------------------is_vertex
    inline bool is_vertex(unsigned c)
    {
        return c >= path_cmd_move_to && c < path_cmd_end_poly;
    }

    //-----------------------------------------------------------------is_stop
    inline bool is_stop(unsigned c)
    { 
        return c == path_cmd_stop;
    }

    //--------------------------------------------------------------is_move_to
    inline bool is_move_to(unsigned c)
    {
        return c == path_cmd_move_to;
    }

    //--------------------------------------------------------------is_line_to
    inline bool is_line_to(unsigned c)
    {
        return c == path_cmd_line_to;
    }

    //----------------------------------------------------------------is_curve
    inline bool is_curve(unsigned c)
    {
        return c == path_cmd_curve3 || c == path_cmd_curve4;
    }

    //---------------------------------------------------------------is_curve3
    inline bool is_curve3(unsigned c)
    {
        return c == path_cmd_curve3;
    }

    //---------------------------------------------------------------is_curve4
    inline bool is_curve4(unsigned c)
    {
        return c == path_cmd_curve4;
    }

    //-------------------------------------------------------------is_end_poly
    inline bool is_end_poly(unsigned c)
    {
        return (c & path_cmd_mask) == path_cmd_end_poly;
    }

    //----------------------------------------------------------------is_close
    inline bool is_close(unsigned c)
    {
        return (c & ~(path_flags_cw | path_flags_ccw)) ==
               (path_cmd_end_poly | path_flags_close); 
    }

    //------------------------------------------------------------is_next_poly
    inline bool is_next_poly(unsigned c)
    {
        return is_stop(c) || is_move_to(c) || is_end_poly(c);
    }

    //-------------------------------------------------------------------is_cw
    inline bool is_cw(unsigned c)
    {
        return (c & path_flags_cw) != 0;
    }

    //------------------------------------------------------------------is_ccw
    inline bool is_ccw(unsigned c)
    {
        return (c & path_flags_ccw) != 0;
    }

    //-------------------------------------------------------------is_oriented
    inline bool is_oriented(unsigned c)
    {
        return (c & (path_flags_cw | path_flags_ccw)) != 0; 
    }

    //---------------------------------------------------------------is_closed
    inline bool is_closed(unsigned c)
    {
        return (c & path_flags_close) != 0; 
    }

    //----------------------------------------------------------get_close_flag
    inline unsigned get_close_flag(unsigned c)
    {
        return c & path_flags_close; 
    }

    //-------------------------------------------------------clear_orientation
    inline unsigned clear_orientation(unsigned c)
    {
        return c & ~(path_flags_cw | path_flags_ccw);
    }

    //---------------------------------------------------------get_orientation
    inline unsigned get_orientation(unsigned c)
    {
        return c & (path_flags_cw | path_flags_ccw);
    }

    //---------------------------------------------------------set_orientation
    inline unsigned set_orientation(unsigned c, unsigned o)
    {
        return clear_orientation(c) | o;
    }

    //--------------------------------------------------------------point_type
    struct point_type
    {
        double x, y;

        point_type() {}
        point_type(double x_, double y_) : x(x_), y(y_) {}
    };

    //-------------------------------------------------------------vertex_type
    struct vertex_type
    {
        double   x, y;
        unsigned cmd;

        vertex_type() {}
        vertex_type(double x_, double y_, unsigned cmd_) : 
            x(x_), y(y_), cmd(cmd_) {}
    };


}


#endif

