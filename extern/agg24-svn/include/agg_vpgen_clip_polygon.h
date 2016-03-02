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

#ifndef AGG_VPGEN_CLIP_POLYGON_INCLUDED
#define AGG_VPGEN_CLIP_POLYGON_INCLUDED

#include "agg_basics.h"

namespace agg
{

    //======================================================vpgen_clip_polygon
    //
    // See Implementation agg_vpgen_clip_polygon.cpp
    //
    class vpgen_clip_polygon
    {
    public:
        vpgen_clip_polygon() : 
            m_clip_box(0, 0, 1, 1),
            m_x1(0),
            m_y1(0),
            m_clip_flags(0),
            m_num_vertices(0),
            m_vertex(0),
            m_cmd(path_cmd_move_to)
        {
        }

        void clip_box(double x1, double y1, double x2, double y2)
        {
            m_clip_box.x1 = x1;
            m_clip_box.y1 = y1;
            m_clip_box.x2 = x2;
            m_clip_box.y2 = y2;
            m_clip_box.normalize();
        }


        double x1() const { return m_clip_box.x1; }
        double y1() const { return m_clip_box.y1; }
        double x2() const { return m_clip_box.x2; }
        double y2() const { return m_clip_box.y2; }

        static bool auto_close()   { return true;  }
        static bool auto_unclose() { return false; }

        void     reset();
        void     move_to(double x, double y);
        void     line_to(double x, double y);
        unsigned vertex(double* x, double* y);

    private:
        unsigned clipping_flags(double x, double y);

    private:
        rect_d        m_clip_box;
        double        m_x1;
        double        m_y1;
        unsigned      m_clip_flags;
        double        m_x[4];
        double        m_y[4];
        unsigned      m_num_vertices;
        unsigned      m_vertex;
        unsigned      m_cmd;
    };

}


#endif
