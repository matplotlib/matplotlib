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
// General Polygon Clipper based on the GPC library by Alan Murta 
// Union, Intersection, XOR, A-B, B-A
// Contact the author if you intend to use it in commercial applications!
// http://www.cs.man.ac.uk/aig/staff/alan/software/
// Alan Murta (email: gpc@cs.man.ac.uk)
//
//----------------------------------------------------------------------------

#ifndef AGG_CONV_GPC_INCLUDED
#define AGG_CONV_GPC_INCLUDED

#include <math.h>
#include "agg_basics.h"
#include "agg_array.h"

extern "C" 
{ 
#include "gpc.h" 
}

namespace agg
{
    enum gpc_op_e
    {
        gpc_or,
        gpc_and,
        gpc_xor,
        gpc_a_minus_b,
        gpc_b_minus_a
    };


    //================================================================conv_gpc
    template<class VSA, class VSB> class conv_gpc
    {
        enum status
        {
            status_move_to,
            status_line_to,
            status_stop
        };

        struct contour_header_type
        {
            int num_vertices;
            int hole_flag;
            gpc_vertex* vertices;
        };

        typedef pod_bvector<gpc_vertex, 8>          vertex_array_type;
        typedef pod_bvector<contour_header_type, 6> contour_header_array_type;


    public:
        typedef VSA source_a_type;
        typedef VSB source_b_type;
        typedef conv_gpc<source_a_type, source_b_type> self_type;

        ~conv_gpc()
        {
            free_gpc_data();
        }

        conv_gpc(source_a_type& a, source_b_type& b, gpc_op_e op = gpc_or) :
            m_src_a(&a),
            m_src_b(&b),
            m_status(status_move_to),
            m_vertex(-1),
            m_contour(-1),
            m_operation(op)
        {
            memset(&m_poly_a, 0, sizeof(m_poly_a));
            memset(&m_poly_b, 0, sizeof(m_poly_b));
            memset(&m_result, 0, sizeof(m_result));
        }

        void attach1(VSA& source) { m_src_a = &source; }
        void attach2(VSB& source) { m_src_b = &source; }

        void operation(gpc_op_e v) { m_operation = v; }

        // Vertex Source Interface
        void     rewind(unsigned path_id);
        unsigned vertex(double* x, double* y);

    private:
        conv_gpc(const conv_gpc<VSA, VSB>&);
        const conv_gpc<VSA, VSB>& operator = (const conv_gpc<VSA, VSB>&);

        //--------------------------------------------------------------------
        void free_polygon(gpc_polygon& p);
        void free_result();
        void free_gpc_data();
        void start_contour();
        void add_vertex(double x, double y);
        void end_contour(unsigned orientation);
        void make_polygon(gpc_polygon& p);
        void start_extracting();
        bool next_contour();
        bool next_vertex(double* x, double* y);


        //--------------------------------------------------------------------
        template<class VS> void add(VS& src, gpc_polygon& p)
        {
            unsigned cmd;
            double x, y;
            double start_x = 0.0;
            double start_y = 0.0;
            bool line_to = false;
            unsigned orientation = 0;

            m_contour_accumulator.remove_all();

            while(!is_stop(cmd = src.vertex(&x, &y)))
            {
                if(is_vertex(cmd))
                {
                    if(is_move_to(cmd))
                    {
                        if(line_to)
                        {
                            end_contour(orientation);
                            orientation = 0;
                        }
                        start_contour();
                        start_x = x;
                        start_y = y;
                    }
                    add_vertex(x, y);
                    line_to = true;
                }
                else
                {
                    if(is_end_poly(cmd))
                    {
                        orientation = get_orientation(cmd);
                        if(line_to && is_closed(cmd))
                        {
                            add_vertex(start_x, start_y);
                        }
                    }
                }
            }
            if(line_to)
            {
                end_contour(orientation);
            }
            make_polygon(p);
        }


    private:
        //--------------------------------------------------------------------
        source_a_type*             m_src_a;
        source_b_type*             m_src_b;
        status                     m_status;
        int                        m_vertex;
        int                        m_contour;
        gpc_op_e                   m_operation;
        vertex_array_type          m_vertex_accumulator;
        contour_header_array_type  m_contour_accumulator;
        gpc_polygon                m_poly_a;
        gpc_polygon                m_poly_b;
        gpc_polygon                m_result;
    };





    //------------------------------------------------------------------------
    template<class VSA, class VSB> 
    void conv_gpc<VSA, VSB>::free_polygon(gpc_polygon& p)
    {
        int i;
        for(i = 0; i < p.num_contours; i++)
        {
            pod_allocator<gpc_vertex>::deallocate(p.contour[i].vertex, 
                                                  p.contour[i].num_vertices);
        }
        pod_allocator<gpc_vertex_list>::deallocate(p.contour, p.num_contours);
        memset(&p, 0, sizeof(gpc_polygon));
    }


    //------------------------------------------------------------------------
    template<class VSA, class VSB> 
    void conv_gpc<VSA, VSB>::free_result()
    {
        if(m_result.contour)
        {
            gpc_free_polygon(&m_result);
        }
        memset(&m_result, 0, sizeof(m_result));
    }


    //------------------------------------------------------------------------
    template<class VSA, class VSB> 
    void conv_gpc<VSA, VSB>::free_gpc_data()
    {
        free_polygon(m_poly_a);
        free_polygon(m_poly_b);
        free_result();
    }


    //------------------------------------------------------------------------
    template<class VSA, class VSB> 
    void conv_gpc<VSA, VSB>::start_contour()
    {
        contour_header_type h;
        memset(&h, 0, sizeof(h));
        m_contour_accumulator.add(h);
        m_vertex_accumulator.remove_all();
    }


    //------------------------------------------------------------------------
    template<class VSA, class VSB> 
    inline void conv_gpc<VSA, VSB>::add_vertex(double x, double y)
    {
        gpc_vertex v;
        v.x = x;
        v.y = y;
        m_vertex_accumulator.add(v);
    }


    //------------------------------------------------------------------------
    template<class VSA, class VSB> 
    void conv_gpc<VSA, VSB>::end_contour(unsigned orientation)
    {
        if(m_contour_accumulator.size())
        {
            if(m_vertex_accumulator.size() > 2)
            {
                contour_header_type& h = 
                    m_contour_accumulator[m_contour_accumulator.size() - 1];

                h.num_vertices = m_vertex_accumulator.size();
                h.hole_flag = 0;

                // TO DO: Clarify the "holes"
                //if(is_cw(orientation)) h.hole_flag = 1;

                h.vertices = pod_allocator<gpc_vertex>::allocate(h.num_vertices);
                gpc_vertex* d = h.vertices;
                int i;
                for(i = 0; i < h.num_vertices; i++)
                {
                    const gpc_vertex& s = m_vertex_accumulator[i];
                    d->x = s.x;
                    d->y = s.y;
                    ++d;
                }
            }
            else
            {
                m_vertex_accumulator.remove_last();
            }
        }
    }


    //------------------------------------------------------------------------
    template<class VSA, class VSB> 
    void conv_gpc<VSA, VSB>::make_polygon(gpc_polygon& p)
    {
        free_polygon(p);
        if(m_contour_accumulator.size())
        {
            p.num_contours = m_contour_accumulator.size();

            p.hole = 0;
            p.contour = pod_allocator<gpc_vertex_list>::allocate(p.num_contours);

            int i;
            gpc_vertex_list* pv = p.contour;
            for(i = 0; i < p.num_contours; i++)
            {
                const contour_header_type& h = m_contour_accumulator[i];
                pv->num_vertices = h.num_vertices;
                pv->vertex = h.vertices;
                ++pv;
            }
        }
    }


    //------------------------------------------------------------------------
    template<class VSA, class VSB> 
    void conv_gpc<VSA, VSB>::start_extracting()
    {
        m_status = status_move_to;
        m_contour = -1;
        m_vertex = -1;
    }


    //------------------------------------------------------------------------
    template<class VSA, class VSB> 
    bool conv_gpc<VSA, VSB>::next_contour()
    {
        if(++m_contour < m_result.num_contours)
        {
            m_vertex = -1;
            return true;
        }
        return false;
    }


    //------------------------------------------------------------------------
    template<class VSA, class VSB> 
    inline bool conv_gpc<VSA, VSB>::next_vertex(double* x, double* y)
    {
        const gpc_vertex_list& vlist = m_result.contour[m_contour];
        if(++m_vertex < vlist.num_vertices)
        {
            const gpc_vertex& v = vlist.vertex[m_vertex];
            *x = v.x;
            *y = v.y;
            return true;
        }
        return false;
    }


    //------------------------------------------------------------------------
    template<class VSA, class VSB> 
    void conv_gpc<VSA, VSB>::rewind(unsigned path_id)
    {
        free_result();
        m_src_a->rewind(path_id);
        m_src_b->rewind(path_id);
        add(*m_src_a, m_poly_a);
        add(*m_src_b, m_poly_b);
        switch(m_operation)
        {
           case gpc_or:
                gpc_polygon_clip(GPC_UNION,
                                 &m_poly_a,
                                 &m_poly_b,
                                 &m_result);
               break;

           case gpc_and:
                gpc_polygon_clip(GPC_INT,
                                 &m_poly_a,
                                 &m_poly_b,
                                 &m_result);
               break;

           case gpc_xor:
                gpc_polygon_clip(GPC_XOR,
                                 &m_poly_a,
                                 &m_poly_b,
                                 &m_result);
               break;

           case gpc_a_minus_b:
                gpc_polygon_clip(GPC_DIFF,
                                 &m_poly_a,
                                 &m_poly_b,
                                 &m_result);
               break;

           case gpc_b_minus_a:
                gpc_polygon_clip(GPC_DIFF,
                                 &m_poly_b,
                                 &m_poly_a,
                                 &m_result);
               break;
        }
        start_extracting();
    }


    //------------------------------------------------------------------------
    template<class VSA, class VSB> 
    unsigned conv_gpc<VSA, VSB>::vertex(double* x, double* y)
    {
        if(m_status == status_move_to)
        {
            if(next_contour()) 
            {
                if(next_vertex(x, y))
                {
                    m_status = status_line_to;
                    return path_cmd_move_to;
                }
                m_status = status_stop;
                return path_cmd_end_poly | path_flags_close;
            }
        }
        else
        {
            if(next_vertex(x, y))
            {
                return path_cmd_line_to;
            }
            else
            {
                m_status = status_move_to;
            }
            return path_cmd_end_poly | path_flags_close;
        }
        return path_cmd_stop;
    }

   
}


#endif
