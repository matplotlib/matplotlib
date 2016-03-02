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
// bounding_rect function template
//
//----------------------------------------------------------------------------
#ifndef AGG_BOUNDING_RECT_INCLUDED
#define AGG_BOUNDING_RECT_INCLUDED

#include "agg_basics.h"

namespace agg
{

    //-----------------------------------------------------------bounding_rect
    template<class VertexSource, class GetId, class CoordT>
    bool bounding_rect(VertexSource& vs, GetId& gi, 
                       unsigned start, unsigned num, 
                       CoordT* x1, CoordT* y1, CoordT* x2, CoordT* y2)
    {
        unsigned i;
        double x;
        double y;
        bool first = true;

        *x1 = CoordT(1);
        *y1 = CoordT(1);
        *x2 = CoordT(0);
        *y2 = CoordT(0);

        for(i = 0; i < num; i++)
        {
            vs.rewind(gi[start + i]);
            unsigned cmd;
            while(!is_stop(cmd = vs.vertex(&x, &y)))
            {
                if(is_vertex(cmd))
                {
                    if(first)
                    {
                        *x1 = CoordT(x);
                        *y1 = CoordT(y);
                        *x2 = CoordT(x);
                        *y2 = CoordT(y);
                        first = false;
                    }
                    else
                    {
                        if(CoordT(x) < *x1) *x1 = CoordT(x);
                        if(CoordT(y) < *y1) *y1 = CoordT(y);
                        if(CoordT(x) > *x2) *x2 = CoordT(x);
                        if(CoordT(y) > *y2) *y2 = CoordT(y);
                    }
                }
            }
        }
        return *x1 <= *x2 && *y1 <= *y2;
    }


    //-----------------------------------------------------bounding_rect_single
    template<class VertexSource, class CoordT> 
    bool bounding_rect_single(VertexSource& vs, unsigned path_id,
                              CoordT* x1, CoordT* y1, CoordT* x2, CoordT* y2)
    {
        double x;
        double y;
        bool first = true;

        *x1 = CoordT(1);
        *y1 = CoordT(1);
        *x2 = CoordT(0);
        *y2 = CoordT(0);

        vs.rewind(path_id);
        unsigned cmd;
        while(!is_stop(cmd = vs.vertex(&x, &y)))
        {
            if(is_vertex(cmd))
            {
                if(first)
                {
                    *x1 = CoordT(x);
                    *y1 = CoordT(y);
                    *x2 = CoordT(x);
                    *y2 = CoordT(y);
                    first = false;
                }
                else
                {
                    if(CoordT(x) < *x1) *x1 = CoordT(x);
                    if(CoordT(y) < *y1) *y1 = CoordT(y);
                    if(CoordT(x) > *x2) *x2 = CoordT(x);
                    if(CoordT(y) > *y2) *y2 = CoordT(y);
                }
            }
        }
        return *x1 <= *x2 && *y1 <= *y2;
    }


}

#endif
