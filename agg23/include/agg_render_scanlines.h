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

#ifndef AGG_RENDER_SCANLINES_INCLUDED
#define AGG_RENDER_SCANLINES_INCLUDED

#include "agg_basics.h"

namespace agg
{
    //========================================================render_scanlines
    template<class Rasterizer, class Scanline, class Renderer>
    void render_scanlines(Rasterizer& ras, Scanline& sl, Renderer& ren)
    {
        if(ras.rewind_scanlines())
        {
            sl.reset(ras.min_x(), ras.max_x());
            ren.prepare(unsigned(ras.max_x() - ras.min_x() + 2));

            while(ras.sweep_scanline(sl))
            {
                ren.render(sl);
            }
        }
    }


    //========================================================render_all_paths
    template<class Rasterizer, class Scanline, class Renderer, 
             class VertexSource, class ColorStorage, class PathId>
    void render_all_paths(Rasterizer& ras, 
                          Scanline& sl,
                          Renderer& r, 
                          VertexSource& vs, 
                          const ColorStorage& as, 
                          const PathId& path_id,
                          unsigned num_paths)
    {
        for(unsigned i = 0; i < num_paths; i++)
        {
            ras.reset();
            ras.add_path(vs, path_id[i]);
            r.color(as[i]);
            render_scanlines(ras, sl, r);
        }
    }


}

#endif



