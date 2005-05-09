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
#ifndef AGG_RASTERIZER_OUTLINE_AA_INCLUDED
#define AGG_RASTERIZER_OUTLINE_AA_INCLUDED

#include "agg_basics.h"
#include "agg_line_aa_basics.h"
#include "agg_vertex_sequence.h"

namespace agg
{

    //-------------------------------------------------------------------------
    inline bool cmp_dist_start(int d) { return d > 0;  }
    inline bool cmp_dist_end(int d)   { return d <= 0; }



    //-----------------------------------------------------------line_aa_vertex
    // Vertex (x, y) with the distance to the next one. The last vertex has 
    // the distance between the last and the first points
    struct line_aa_vertex
    {
        int x;
        int y;
        int len;

        line_aa_vertex() {}
        line_aa_vertex(int x_, int y_) :
            x(x_),
            y(y_),
            len(0)
        {
        }

        bool operator () (const line_aa_vertex& val)
        {
            double dx = val.x - x;
            double dy = val.y - y;
            return (len = int(sqrt(dx * dx + dy * dy))) > 
                   (line_subpixel_size + line_subpixel_size / 2);
        }
    };




    //=======================================================rasterizer_outline_aa
    template<class Renderer> class rasterizer_outline_aa
    {
    private:
        //------------------------------------------------------------------------
        struct draw_vars
        {
            unsigned idx;
            int x1, y1, x2, y2;
            line_parameters curr, next;
            int lcurr, lnext;
            int xb1, yb1, xb2, yb2;
            unsigned flags;
        };

        void draw(draw_vars& dv, unsigned start, unsigned end);

    public:
        typedef line_aa_vertex                  vertex_type;
        typedef vertex_sequence<vertex_type, 6> vertex_storage_type;

        rasterizer_outline_aa(Renderer& ren) : 
            m_ren(ren), 
            m_accurate_join(m_ren.accurate_join_only()),
            m_round_cap(false),
            m_start_x(0),
            m_start_y(0)
        {
        }

        //------------------------------------------------------------------------
        void accurate_join(bool v) 
        { 
            m_accurate_join = m_ren.accurate_join_only() ? true : v; 
        }
        bool accurate_join() const { return m_accurate_join; }

        //------------------------------------------------------------------------
        void round_cap(bool v) { m_round_cap = v; }
        bool round_cap() const { return m_round_cap; }

        //------------------------------------------------------------------------
        void move_to(int x, int y)
        {
            m_src_vertices.modify_last(vertex_type(m_start_x = x, m_start_y = y));
        }

        //------------------------------------------------------------------------
        void line_to(int x, int y)
        {
            m_src_vertices.add(vertex_type(x, y));
        }

        //------------------------------------------------------------------------
        void move_to_d(double x, double y)
        {
            move_to(line_coord(x), line_coord(y));
        }

        //------------------------------------------------------------------------
        void line_to_d(double x, double y)
        {
            line_to(line_coord(x), line_coord(y));
        }

        //------------------------------------------------------------------------
        void render(bool close_polygon);

        //------------------------------------------------------------------------
        void add_vertex(double x, double y, unsigned cmd)
        {
            if(is_move_to(cmd)) 
            {
                render(false);
                move_to_d(x, y);
            }
            else 
            {
                if(is_end_poly(cmd))
                {
                    render(is_closed(cmd));
                    if(is_closed(cmd)) 
                    {
                        move_to(m_start_x, m_start_y);
                    }
                }
                else
                {
                    line_to_d(x, y);
                }
            }
        }

        //------------------------------------------------------------------------
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
            render(false);
        }


        //------------------------------------------------------------------------
        template<class VertexSource, class ColorStorage, class PathId>
        void render_all_paths(VertexSource& vs, 
                              const ColorStorage& colors, 
                              const PathId& path_id,
                              unsigned num_paths)
        {
            for(unsigned i = 0; i < num_paths; i++)
            {
                m_ren.color(colors[i]);
                add_path(vs, path_id[i]);
            }
        }


        //------------------------------------------------------------------------
        template<class Ctrl> void render_ctrl(Ctrl& c)
        {
            unsigned i;
            for(i = 0; i < c.num_paths(); i++)
            {
                m_ren.color(c.color(i));
                add_path(c, i);
            }
        }

    private:
        rasterizer_outline_aa(const rasterizer_outline_aa<Renderer>&);
        const rasterizer_outline_aa<Renderer>& operator = 
            (const rasterizer_outline_aa<Renderer>&);

        Renderer& m_ren;
        vertex_storage_type m_src_vertices;
        bool m_accurate_join;
        bool m_round_cap;
        int  m_start_x;
        int  m_start_y;
    };








    //----------------------------------------------------------------------------
    template<class Renderer> 
    void rasterizer_outline_aa<Renderer>::draw(draw_vars& dv, unsigned start, unsigned end)
    {
        unsigned i;
        const vertex_storage_type::value_type* v;

        for(i = start; i < end; i++)
        {
            switch(dv.flags)
            {
            case 0: m_ren.line3(dv.curr, dv.xb1, dv.yb1, dv.xb2, dv.yb2); break;
            case 1: m_ren.line2(dv.curr, dv.xb2, dv.yb2); break;
            case 2: m_ren.line1(dv.curr, dv.xb1, dv.yb1); break;
            case 3: m_ren.line0(dv.curr); break;
            }

            dv.x1 = dv.x2;
            dv.y1 = dv.y2;
            dv.lcurr = dv.lnext;
            dv.lnext = m_src_vertices[dv.idx].len;

            ++dv.idx;
            if(dv.idx >= m_src_vertices.size()) dv.idx = 0; 

            v = &m_src_vertices[dv.idx];
            dv.x2 = v->x;
            dv.y2 = v->y;

            dv.curr = dv.next;
            dv.next = line_parameters(dv.x1, dv.y1, dv.x2, dv.y2, dv.lnext);
            dv.xb1 = dv.xb2;
            dv.yb1 = dv.yb2;

            if(m_accurate_join)
            {
                dv.flags = 0;
            }
            else
            {
                dv.flags >>= 1;
                dv.flags |= ((dv.curr.diagonal_quadrant() == 
                              dv.next.diagonal_quadrant()) << 1);
            }

            if((dv.flags & 2) == 0)
            {
                bisectrix(dv.curr, dv.next, &dv.xb2, &dv.yb2);
            }
        }
    }




    //----------------------------------------------------------------------------
    template<class Renderer> 
    void rasterizer_outline_aa<Renderer>::render(bool close_polygon)
    {
        m_src_vertices.close(close_polygon);
        draw_vars dv;
        const vertex_storage_type::value_type* v;
        int x1;
        int y1;
        int x2;
        int y2;
        int lprev;

        if(close_polygon)
        {
            if(m_src_vertices.size() >= 3)
            {
                dv.idx = 2;

                v     = &m_src_vertices[m_src_vertices.size() - 1];
                x1    = v->x;
                y1    = v->y;
                lprev = v->len;

                v  = &m_src_vertices[0];
                x2 = v->x;
                y2 = v->y;
                dv.lcurr = v->len;
                line_parameters prev(x1, y1, x2, y2, lprev);

                v = &m_src_vertices[1];
                dv.x1    = v->x;
                dv.y1    = v->y;
                dv.lnext = v->len;
                dv.curr = line_parameters(x2, y2, dv.x1, dv.y1, dv.lcurr);

                v = &m_src_vertices[dv.idx];
                dv.x2 = v->x;
                dv.y2 = v->y;
                dv.next = line_parameters(dv.x1, dv.y1, dv.x2, dv.y2, dv.lnext);

                dv.xb1 = 0;
                dv.yb1 = 0;
                dv.xb2 = 0;
                dv.yb2 = 0;

                if(m_accurate_join)
                {
                    dv.flags = 0;
                }
                else
                {
                    dv.flags = 
                            (prev.diagonal_quadrant() == dv.curr.diagonal_quadrant()) |
                        ((dv.curr.diagonal_quadrant() == dv.next.diagonal_quadrant()) << 1);
                }

                if((dv.flags & 1) == 0)
                {
                    bisectrix(prev, dv.curr, &dv.xb1, &dv.yb1);
                }

                if((dv.flags & 2) == 0)
                {
                    bisectrix(dv.curr, dv.next, &dv.xb2, &dv.yb2);
                }
                draw(dv, 0, m_src_vertices.size());
            }
        }
        else
        {
            switch(m_src_vertices.size())
            {
                case 0:
                case 1:
                    break;

                case 2:
                {
                    v     = &m_src_vertices[0];
                    x1    = v->x;
                    y1    = v->y;
                    lprev = v->len;
                    v     = &m_src_vertices[1];
                    x2    = v->x;
                    y2    = v->y;
                    line_parameters lp(x1, y1, x2, y2, lprev);
                    if(m_round_cap) 
                    {
                        m_ren.semidot(cmp_dist_start, x1, y1, x1 + (y2 - y1), y1 - (x2 - x1));
                    }
                    m_ren.line3(lp, 
                                x1 + (y2 - y1), 
                                y1 - (x2 - x1),
                                x2 + (y2 - y1), 
                                y2 - (x2 - x1));
                    if(m_round_cap) 
                    {
                        m_ren.semidot(cmp_dist_end, x2, y2, x2 + (y2 - y1), y2 - (x2 - x1));
                    }
                }
                break;

                case 3:
                {
                    int x3, y3;
                    int lnext;
                    v     = &m_src_vertices[0];
                    x1    = v->x;
                    y1    = v->y;
                    lprev = v->len;
                    v     = &m_src_vertices[1];
                    x2    = v->x;
                    y2    = v->y;
                    lnext = v->len;
                    v     = &m_src_vertices[2];
                    x3    = v->x;
                    y3    = v->y;
                    line_parameters lp1(x1, y1, x2, y2, lprev);
                    line_parameters lp2(x2, y2, x3, y3, lnext);
                    bisectrix(lp1, lp2, &dv.xb1, &dv.yb1);

                    if(m_round_cap) 
                    {
                        m_ren.semidot(cmp_dist_start, x1, y1, x1 + (y2 - y1), y1 - (x2 - x1));
                    }
                    m_ren.line3(lp1, 
                                x1 + (y2 - y1), 
                                y1 - (x2 - x1),
                                dv.xb1, 
                                dv.yb1);

                    m_ren.line3(lp2, 
                                dv.xb1,
                                dv.yb1,
                                x3 + (y3 - y2), 
                                y3 - (x3 - x2));
                    if(m_round_cap) 
                    {
                        m_ren.semidot(cmp_dist_end, x3, y3, x3 + (y3 - y2), y3 - (x3 - x2));
                    }
                }
                break;

                default:
                {
                    dv.idx = 3;

                    v     = &m_src_vertices[0];
                    x1    = v->x;
                    y1    = v->y;
                    lprev = v->len;

                    v  = &m_src_vertices[1];
                    x2 = v->x;
                    y2 = v->y;
                    dv.lcurr = v->len;
                    line_parameters prev(x1, y1, x2, y2, lprev);

                    v = &m_src_vertices[2];
                    dv.x1    = v->x;
                    dv.y1    = v->y;
                    dv.lnext = v->len;
                    dv.curr = line_parameters(x2, y2, dv.x1, dv.y1, dv.lcurr);

                    v = &m_src_vertices[dv.idx];
                    dv.x2 = v->x;
                    dv.y2 = v->y;
                    dv.next = line_parameters(dv.x1, dv.y1, dv.x2, dv.y2, dv.lnext);

                    dv.xb1 = 0;
                    dv.yb1 = 0;
                    dv.xb2 = 0;
                    dv.yb2 = 0;

                    if(m_accurate_join)
                    {
                        dv.flags = 0;
                    }
                    else
                    {
                        dv.flags = 
                                (prev.diagonal_quadrant() == dv.curr.diagonal_quadrant()) |
                            ((dv.curr.diagonal_quadrant() == dv.next.diagonal_quadrant()) << 1);
                    }

                    if((dv.flags & 1) == 0)
                    {
                        bisectrix(prev, dv.curr, &dv.xb1, &dv.yb1);
                        m_ren.line3(prev, 
                                    x1 + (y2 - y1), 
                                    y1 - (x2 - x1),
                                    dv.xb1, 
                                    dv.yb1);
                    }
                    else
                    {
                        m_ren.line1(prev, 
                                    x1 + (y2 - y1), 
                                    y1 - (x2 - x1));
                    }
                    if(m_round_cap) 
                    {
                        m_ren.semidot(cmp_dist_start, x1, y1, x1 + (y2 - y1), y1 - (x2 - x1));
                    }
                    if((dv.flags & 2) == 0)
                    {
                        bisectrix(dv.curr, dv.next, &dv.xb2, &dv.yb2);
                    }

                    draw(dv, 1, m_src_vertices.size() - 2);

                    if((dv.flags & 1) == 0)
                    {
                        m_ren.line3(dv.curr, 
                                    dv.xb1, 
                                    dv.yb1,
                                    dv.curr.x2 + (dv.curr.y2 - dv.curr.y1), 
                                    dv.curr.y2 - (dv.curr.x2 - dv.curr.x1));
                    }
                    else
                    {
                        m_ren.line2(dv.curr, 
                                    dv.curr.x2 + (dv.curr.y2 - dv.curr.y1), 
                                    dv.curr.y2 - (dv.curr.x2 - dv.curr.x1));
                    }
                    if(m_round_cap) 
                    {
                        m_ren.semidot(cmp_dist_end, dv.curr.x2, dv.curr.y2,
                                      dv.curr.x2 + (dv.curr.y2 - dv.curr.y1),
                                      dv.curr.y2 - (dv.curr.x2 - dv.curr.x1));
                    }

                }
                break;
            }
        }
        m_src_vertices.remove_all();
    }


}


#endif

