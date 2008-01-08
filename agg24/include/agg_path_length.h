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
#ifndef AGG_PATH_LENGTH_INCLUDED
#define AGG_PATH_LENGTH_INCLUDED

#include "agg_math.h"

namespace agg
{
    template<class VertexSource> 
    double path_length(VertexSource& vs, unsigned path_id = 0)
    {
        double len = 0.0;
        double start_x = 0.0;
        double start_y = 0.0;
        double x1 = 0.0;
        double y1 = 0.0;
        double x2 = 0.0;
        double y2 = 0.0;
        bool first = true;

        unsigned cmd;
        vs.rewind(path_id);
        while(!is_stop(cmd = vs.vertex(&x2, &y2)))
        {
            if(is_vertex(cmd))
            {
                if(first || is_move_to(cmd))
                {
                    start_x = x2;
                    start_y = y2;
                }
                else
                {
                    len += calc_distance(x1, y1, x2, y2);
                }
                x1 = x2;
                y1 = y2;
                first = false;
            }
            else
            {
                if(is_close(cmd) && !first)
                {
                    len += calc_distance(x1, y1, start_x, start_y);
                }
            }
        }
        return len;
    }
}

#endif
