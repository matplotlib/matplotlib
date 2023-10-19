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

#ifndef AGG_SHORTEN_PATH_INCLUDED
#define AGG_SHORTEN_PATH_INCLUDED

#include "agg_basics.h"
#include "agg_vertex_sequence.h"

namespace agg
{

    //===========================================================shorten_path
    template<class VertexSequence> 
    void shorten_path(VertexSequence& vs, double s, unsigned closed = 0)
    {
        typedef typename VertexSequence::value_type vertex_type;

        if(s > 0.0 && vs.size() > 1)
        {
            double d;
            int n = int(vs.size() - 2);
            while(n)
            {
                d = vs[n].dist;
                if(d > s) break;
                vs.remove_last();
                s -= d;
                --n;
            }
            if(vs.size() < 2)
            {
                vs.remove_all();
            }
            else
            {
                n = vs.size() - 1;
                vertex_type& prev = vs[n-1];
                vertex_type& last = vs[n];
                d = (prev.dist - s) / prev.dist;
                double x = prev.x + (last.x - prev.x) * d;
                double y = prev.y + (last.y - prev.y) * d;
                last.x = x;
                last.y = y;
                if(!prev(last)) vs.remove_last();
                vs.close(closed != 0);
            }
        }
    }


}

#endif
