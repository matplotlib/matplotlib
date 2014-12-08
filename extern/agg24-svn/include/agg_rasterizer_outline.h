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
#ifndef AGG_RASTERIZER_OUTLINE_INCLUDED
#define AGG_RASTERIZER_OUTLINE_INCLUDED

#include "agg_basics.h"

namespace agg
{
    //======================================================rasterizer_outline
    template<class Renderer> class rasterizer_outline
    {
    public:
        explicit rasterizer_outline(Renderer& ren) : 
            m_ren(&ren), 
            m_start_x(0), 
            m_start_y(0), 
            m_vertices(0)
        {}
        void attach(Renderer& ren) { m_ren = &ren; }


        //--------------------------------------------------------------------
        void move_to(int x, int y)
        {
            m_vertices = 1;
            m_ren->move_to(m_start_x = x, m_start_y = y);
        }

        //--------------------------------------------------------------------
        void line_to(int x, int y)
        {
            ++m_vertices;
            m_ren->line_to(x, y);
        }

        //--------------------------------------------------------------------
        void move_to_d(double x, double y)
        {
            move_to(m_ren->coord(x), m_ren->coord(y));
        }

        //--------------------------------------------------------------------
        void line_to_d(double x, double y)
        {
            line_to(m_ren->coord(x), m_ren->coord(y));
        }

        //--------------------------------------------------------------------
        void close()
        {
            if(m_vertices > 2)
            {
                line_to(m_start_x, m_start_y);
            }
            m_vertices = 0;
        }

        //--------------------------------------------------------------------
        void add_vertex(double x, double y, unsigned cmd)
        {
            if(is_move_to(cmd)) 
            {
                move_to_d(x, y);
            }
            else 
            {
                if(is_end_poly(cmd))
                {
                    if(is_closed(cmd)) close();
                }
                else
                {
                    line_to_d(x, y);
                }
            }
        }


        //--------------------------------------------------------------------
        template<class VertexSource>
        void add_path(VertexSource& vs, unsigned path_id=0)
        {
            double x;
            double y;

            unsigned cmd;
            vs.rewind(path_id);
            while(!is_stop(cmd = vs.vertex(&x, &y)))
            {
                add_vertex(x, y, cmd);
            }
        }


        //--------------------------------------------------------------------
        template<class VertexSource, class ColorStorage, class PathId>
        void render_all_paths(VertexSource& vs, 
                              const ColorStorage& colors, 
                              const PathId& path_id,
                              unsigned num_paths)
        {
            for(unsigned i = 0; i < num_paths; i++)
            {
                m_ren->line_color(colors[i]);
                add_path(vs, path_id[i]);
            }
        }


        //--------------------------------------------------------------------
        template<class Ctrl> void render_ctrl(Ctrl& c)
        {
            unsigned i;
            for(i = 0; i < c.num_paths(); i++)
            {
                m_ren->line_color(c.color(i));
                add_path(c, i);
            }
        }


    private:
        Renderer* m_ren;
        int       m_start_x;
        int       m_start_y;
        unsigned  m_vertices;
    };


}


#endif

