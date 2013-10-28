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

#ifndef AGG_PATH_STORAGE_INCLUDED
#define AGG_PATH_STORAGE_INCLUDED

#include <string.h>
#include <math.h>
#include "agg_math.h"
#include "agg_array.h"
#include "agg_bezier_arc.h"

namespace agg
{


    //----------------------------------------------------vertex_block_storage
    template<class T, unsigned BlockShift=8, unsigned BlockPool=256>
    class vertex_block_storage
    {
    public:
        // Allocation parameters
        enum block_scale_e
        {
            block_shift = BlockShift,
            block_size  = 1 << block_shift,
            block_mask  = block_size - 1,
            block_pool  = BlockPool
        };

        typedef T value_type;
        typedef vertex_block_storage<T, BlockShift, BlockPool> self_type;

        ~vertex_block_storage();
        vertex_block_storage();
        vertex_block_storage(const self_type& v);
        const self_type& operator = (const self_type& ps);

        void remove_all();
        void free_all();

        void add_vertex(double x, double y, unsigned cmd);
        void modify_vertex(unsigned idx, double x, double y);
        void modify_vertex(unsigned idx, double x, double y, unsigned cmd);
        void modify_command(unsigned idx, unsigned cmd);
        void swap_vertices(unsigned v1, unsigned v2);

        unsigned last_command() const;
        unsigned last_vertex(double* x, double* y) const;
        unsigned prev_vertex(double* x, double* y) const;

        double last_x() const;
        double last_y() const;

        unsigned total_vertices() const;
        unsigned vertex(unsigned idx, double* x, double* y) const;
        unsigned command(unsigned idx) const;

    private:
        void   allocate_block(unsigned nb);
        int8u* storage_ptrs(T** xy_ptr);

    private:
        unsigned m_total_vertices;
        unsigned m_total_blocks;
        unsigned m_max_blocks;
        T**      m_coord_blocks;
        int8u**  m_cmd_blocks;
    };


    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    void vertex_block_storage<T,S,P>::free_all()
    {
        if(m_total_blocks)
        {
            T** coord_blk = m_coord_blocks + m_total_blocks - 1;
            while(m_total_blocks--)
            {
                pod_allocator<T>::deallocate(
                    *coord_blk,
                    block_size * 2 + 
                    block_size / (sizeof(T) / sizeof(unsigned char)));
                --coord_blk;
            }
            pod_allocator<T*>::deallocate(m_coord_blocks, m_max_blocks * 2);
            m_total_blocks   = 0;
            m_max_blocks     = 0;
            m_coord_blocks   = 0;
            m_cmd_blocks     = 0;
            m_total_vertices = 0;
        }
    }

    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    vertex_block_storage<T,S,P>::~vertex_block_storage()
    {
        free_all();
    }

    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    vertex_block_storage<T,S,P>::vertex_block_storage() :
        m_total_vertices(0),
        m_total_blocks(0),
        m_max_blocks(0),
        m_coord_blocks(0),
        m_cmd_blocks(0)
    {
    }

    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    vertex_block_storage<T,S,P>::vertex_block_storage(const vertex_block_storage<T,S,P>& v) :
        m_total_vertices(0),
        m_total_blocks(0),
        m_max_blocks(0),
        m_coord_blocks(0),
        m_cmd_blocks(0)
    {
        *this = v;
    }

    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    const vertex_block_storage<T,S,P>& 
    vertex_block_storage<T,S,P>::operator = (const vertex_block_storage<T,S,P>& v)
    {
        remove_all();
        unsigned i;
        for(i = 0; i < v.total_vertices(); i++)
        {
            double x, y;
            unsigned cmd = v.vertex(i, &x, &y);
            add_vertex(x, y, cmd);
        }
	    return *this;
    }

    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    inline void vertex_block_storage<T,S,P>::remove_all()
    {
        m_total_vertices = 0;
    }

    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    inline void vertex_block_storage<T,S,P>::add_vertex(double x, double y, 
                                                        unsigned cmd)
    {
        T* coord_ptr = 0;
        *storage_ptrs(&coord_ptr) = (int8u)cmd;
        coord_ptr[0] = T(x);
        coord_ptr[1] = T(y);
        m_total_vertices++;
    }

    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    inline void vertex_block_storage<T,S,P>::modify_vertex(unsigned idx, 
                                                           double x, double y)
    {
        T* pv = m_coord_blocks[idx >> block_shift] + ((idx & block_mask) << 1);
        pv[0] = T(x);
        pv[1] = T(y);
    }

    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    inline void vertex_block_storage<T,S,P>::modify_vertex(unsigned idx, 
                                                           double x, double y, 
                                                           unsigned cmd)
    {
        unsigned block = idx >> block_shift;
        unsigned offset = idx & block_mask;
        T* pv = m_coord_blocks[block] + (offset << 1);
        pv[0] = T(x);
        pv[1] = T(y);
        m_cmd_blocks[block][offset] = (int8u)cmd;
    }

    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    inline void vertex_block_storage<T,S,P>::modify_command(unsigned idx, 
                                                            unsigned cmd)
    {
        m_cmd_blocks[idx >> block_shift][idx & block_mask] = (int8u)cmd;
    }

    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    inline void vertex_block_storage<T,S,P>::swap_vertices(unsigned v1, unsigned v2)
    {
        unsigned b1 = v1 >> block_shift;
        unsigned b2 = v2 >> block_shift;
        unsigned o1 = v1 & block_mask;
        unsigned o2 = v2 & block_mask;
        T* pv1 = m_coord_blocks[b1] + (o1 << 1);
        T* pv2 = m_coord_blocks[b2] + (o2 << 1);
        T  val;
        val = pv1[0]; pv1[0] = pv2[0]; pv2[0] = val;
        val = pv1[1]; pv1[1] = pv2[1]; pv2[1] = val;
        int8u cmd = m_cmd_blocks[b1][o1];
        m_cmd_blocks[b1][o1] = m_cmd_blocks[b2][o2];
        m_cmd_blocks[b2][o2] = cmd;
    }

    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    inline unsigned vertex_block_storage<T,S,P>::last_command() const
    {
        if(m_total_vertices) return command(m_total_vertices - 1);
        return path_cmd_stop;
    }

    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    inline unsigned vertex_block_storage<T,S,P>::last_vertex(double* x, double* y) const
    {
        if(m_total_vertices) return vertex(m_total_vertices - 1, x, y);
        return path_cmd_stop;
    }

    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    inline unsigned vertex_block_storage<T,S,P>::prev_vertex(double* x, double* y) const
    {
        if(m_total_vertices > 1) return vertex(m_total_vertices - 2, x, y);
        return path_cmd_stop;
    }

    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    inline double vertex_block_storage<T,S,P>::last_x() const
    {
        if(m_total_vertices)
        {
            unsigned idx = m_total_vertices - 1;
            return m_coord_blocks[idx >> block_shift][(idx & block_mask) << 1];
        }
        return 0.0;
    }

    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    inline double vertex_block_storage<T,S,P>::last_y() const
    {
        if(m_total_vertices)
        {
            unsigned idx = m_total_vertices - 1;
            return m_coord_blocks[idx >> block_shift][((idx & block_mask) << 1) + 1];
        }
        return 0.0;
    }

    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    inline unsigned vertex_block_storage<T,S,P>::total_vertices() const
    {
        return m_total_vertices;
    }

    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    inline unsigned vertex_block_storage<T,S,P>::vertex(unsigned idx, 
                                                        double* x, double* y) const
    {
        unsigned nb = idx >> block_shift;
        const T* pv = m_coord_blocks[nb] + ((idx & block_mask) << 1);
        *x = pv[0];
        *y = pv[1];
        return m_cmd_blocks[nb][idx & block_mask];
    }

    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    inline unsigned vertex_block_storage<T,S,P>::command(unsigned idx) const
    {
        return m_cmd_blocks[idx >> block_shift][idx & block_mask];
    }

    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    void vertex_block_storage<T,S,P>::allocate_block(unsigned nb)
    {
        if(nb >= m_max_blocks) 
        {
            T** new_coords = 
                pod_allocator<T*>::allocate((m_max_blocks + block_pool) * 2);

            unsigned char** new_cmds = 
                (unsigned char**)(new_coords + m_max_blocks + block_pool);

            if(m_coord_blocks)
            {
                memcpy(new_coords, 
                       m_coord_blocks, 
                       m_max_blocks * sizeof(T*));

                memcpy(new_cmds, 
                       m_cmd_blocks, 
                       m_max_blocks * sizeof(unsigned char*));

                pod_allocator<T*>::deallocate(m_coord_blocks, m_max_blocks * 2);
            }
            m_coord_blocks = new_coords;
            m_cmd_blocks   = new_cmds;
            m_max_blocks  += block_pool;
        }
        m_coord_blocks[nb] = 
            pod_allocator<T>::allocate(block_size * 2 + 
                   block_size / (sizeof(T) / sizeof(unsigned char)));

        m_cmd_blocks[nb]  = 
            (unsigned char*)(m_coord_blocks[nb] + block_size * 2);

        m_total_blocks++;
    }

    //------------------------------------------------------------------------
    template<class T, unsigned S, unsigned P>
    int8u* vertex_block_storage<T,S,P>::storage_ptrs(T** xy_ptr)
    {
        unsigned nb = m_total_vertices >> block_shift;
        if(nb >= m_total_blocks)
        {
            allocate_block(nb);
        }
        *xy_ptr = m_coord_blocks[nb] + ((m_total_vertices & block_mask) << 1);
        return m_cmd_blocks[nb] + (m_total_vertices & block_mask);
    }




    //-----------------------------------------------------poly_plain_adaptor
    template<class T> class poly_plain_adaptor
    {
    public:
        typedef T value_type;

        poly_plain_adaptor() : 
            m_data(0), 
            m_ptr(0),
            m_end(0),
            m_closed(false),
            m_stop(false)
        {}

        poly_plain_adaptor(const T* data, unsigned num_points, bool closed) :
            m_data(data), 
            m_ptr(data),
            m_end(data + num_points * 2),
            m_closed(closed),
            m_stop(false)
        {}

        void init(const T* data, unsigned num_points, bool closed)
        {
            m_data = data;
            m_ptr = data;
            m_end = data + num_points * 2;
            m_closed = closed;
            m_stop = false;
        }

        void rewind(unsigned)
        {
            m_ptr = m_data;
            m_stop = false;
        }

        unsigned vertex(double* x, double* y)
        {
            if(m_ptr < m_end)
            {
                bool first = m_ptr == m_data;
                *x = *m_ptr++;
                *y = *m_ptr++;
                return first ? path_cmd_move_to : path_cmd_line_to;
            }
            *x = *y = 0.0;
            if(m_closed && !m_stop)
            {
                m_stop = true;
                return path_cmd_end_poly | path_flags_close;
            }
            return path_cmd_stop;
        }

    private:
        const T* m_data;
        const T* m_ptr;
        const T* m_end;
        bool     m_closed;
        bool     m_stop;
    };





    //-------------------------------------------------poly_container_adaptor
    template<class Container> class poly_container_adaptor
    {
    public:
        typedef typename Container::value_type vertex_type;

        poly_container_adaptor() : 
            m_container(0), 
            m_index(0),
            m_closed(false),
            m_stop(false)
        {}

        poly_container_adaptor(const Container& data, bool closed) :
            m_container(&data), 
            m_index(0),
            m_closed(closed),
            m_stop(false)
        {}

        void init(const Container& data, bool closed)
        {
            m_container = &data;
            m_index = 0;
            m_closed = closed;
            m_stop = false;
        }

        void rewind(unsigned)
        {
            m_index = 0;
            m_stop = false;
        }

        unsigned vertex(double* x, double* y)
        {
            if(m_index < m_container->size())
            {
                bool first = m_index == 0;
                const vertex_type& v = (*m_container)[m_index++];
                *x = v.x;
                *y = v.y;
                return first ? path_cmd_move_to : path_cmd_line_to;
            }
            *x = *y = 0.0;
            if(m_closed && !m_stop)
            {
                m_stop = true;
                return path_cmd_end_poly | path_flags_close;
            }
            return path_cmd_stop;
        }

    private:
        const Container* m_container;
        unsigned m_index;
        bool     m_closed;
        bool     m_stop;
    };



    //-----------------------------------------poly_container_reverse_adaptor
    template<class Container> class poly_container_reverse_adaptor
    {
    public:
        typedef typename Container::value_type vertex_type;

        poly_container_reverse_adaptor() : 
            m_container(0), 
            m_index(-1),
            m_closed(false),
            m_stop(false)
        {}

        poly_container_reverse_adaptor(const Container& data, bool closed) :
            m_container(&data), 
            m_index(-1),
            m_closed(closed),
            m_stop(false)
        {}

        void init(const Container& data, bool closed)
        {
            m_container = &data;
            m_index = m_container->size() - 1;
            m_closed = closed;
            m_stop = false;
        }

        void rewind(unsigned)
        {
            m_index = m_container->size() - 1;
            m_stop = false;
        }

        unsigned vertex(double* x, double* y)
        {
            if(m_index >= 0)
            {
                bool first = m_index == int(m_container->size() - 1);
                const vertex_type& v = (*m_container)[m_index--];
                *x = v.x;
                *y = v.y;
                return first ? path_cmd_move_to : path_cmd_line_to;
            }
            *x = *y = 0.0;
            if(m_closed && !m_stop)
            {
                m_stop = true;
                return path_cmd_end_poly | path_flags_close;
            }
            return path_cmd_stop;
        }

    private:
        const Container* m_container;
        int  m_index;
        bool m_closed;
        bool m_stop;
    };





    //--------------------------------------------------------line_adaptor
    class line_adaptor
    {
    public:
        typedef double value_type;

        line_adaptor() : m_line(m_coord, 2, false) {}
        line_adaptor(double x1, double y1, double x2, double y2) :
            m_line(m_coord, 2, false)
        {
            m_coord[0] = x1;
            m_coord[1] = y1;
            m_coord[2] = x2;
            m_coord[3] = y2;
        }
        
        void init(double x1, double y1, double x2, double y2)
        {
            m_coord[0] = x1;
            m_coord[1] = y1;
            m_coord[2] = x2;
            m_coord[3] = y2;
            m_line.rewind(0);
        }

        void rewind(unsigned)
        {
            m_line.rewind(0);
        }

        unsigned vertex(double* x, double* y)
        {
            return m_line.vertex(x, y);
        }

    private:
        double                     m_coord[4];
        poly_plain_adaptor<double> m_line;
    };













    //---------------------------------------------------------------path_base
    // A container to store vertices with their flags. 
    // A path consists of a number of contours separated with "move_to" 
    // commands. The path storage can keep and maintain more than one
    // path. 
    // To navigate to the beginning of a particular path, use rewind(path_id);
    // Where path_id is what start_new_path() returns. So, when you call
    // start_new_path() you need to store its return value somewhere else
    // to navigate to the path afterwards.
    //
    // See also: vertex_source concept
    //------------------------------------------------------------------------
    template<class VertexContainer> class path_base
    {
    public:
        typedef VertexContainer            container_type;
        typedef path_base<VertexContainer> self_type;

        //--------------------------------------------------------------------
        path_base() : m_vertices(), m_iterator(0) {}
        void remove_all() { m_vertices.remove_all(); m_iterator = 0; }
        void free_all()   { m_vertices.free_all();   m_iterator = 0; }

        // Make path functions
        //--------------------------------------------------------------------
        unsigned start_new_path();

        void move_to(double x, double y);
        void move_rel(double dx, double dy);

        void line_to(double x, double y);
        void line_rel(double dx, double dy);

        void hline_to(double x);
        void hline_rel(double dx);

        void vline_to(double y);
        void vline_rel(double dy);

        void arc_to(double rx, double ry,
                    double angle,
                    bool large_arc_flag,
                    bool sweep_flag,
                    double x, double y);

        void arc_rel(double rx, double ry,
                     double angle,
                     bool large_arc_flag,
                     bool sweep_flag,
                     double dx, double dy);

        void curve3(double x_ctrl, double y_ctrl, 
                    double x_to,   double y_to);

        void curve3_rel(double dx_ctrl, double dy_ctrl, 
                        double dx_to,   double dy_to);

        void curve3(double x_to, double y_to);

        void curve3_rel(double dx_to, double dy_to);

        void curve4(double x_ctrl1, double y_ctrl1, 
                    double x_ctrl2, double y_ctrl2, 
                    double x_to,    double y_to);

        void curve4_rel(double dx_ctrl1, double dy_ctrl1, 
                        double dx_ctrl2, double dy_ctrl2, 
                        double dx_to,    double dy_to);

        void curve4(double x_ctrl2, double y_ctrl2, 
                    double x_to,    double y_to);

        void curve4_rel(double x_ctrl2, double y_ctrl2, 
                        double x_to,    double y_to);


        void end_poly(unsigned flags = path_flags_close);
        void close_polygon(unsigned flags = path_flags_none);

        // Accessors
        //--------------------------------------------------------------------
        const container_type& vertices() const { return m_vertices; } 
              container_type& vertices()       { return m_vertices; } 

        unsigned total_vertices() const;

        void rel_to_abs(double* x, double* y) const;

        unsigned last_vertex(double* x, double* y) const;
        unsigned prev_vertex(double* x, double* y) const;

        double last_x() const;
        double last_y() const;

        unsigned vertex(unsigned idx, double* x, double* y) const;
        unsigned command(unsigned idx) const;

        void modify_vertex(unsigned idx, double x, double y);
        void modify_vertex(unsigned idx, double x, double y, unsigned cmd);
        void modify_command(unsigned idx, unsigned cmd);

        // VertexSource interface
        //--------------------------------------------------------------------
        void     rewind(unsigned path_id);
        unsigned vertex(double* x, double* y);

        // Arrange the orientation of a polygon, all polygons in a path, 
        // or in all paths. After calling arrange_orientations() or 
        // arrange_orientations_all_paths(), all the polygons will have 
        // the same orientation, i.e. path_flags_cw or path_flags_ccw
        //--------------------------------------------------------------------
        unsigned arrange_polygon_orientation(unsigned start, path_flags_e orientation);
        unsigned arrange_orientations(unsigned path_id, path_flags_e orientation);
        void     arrange_orientations_all_paths(path_flags_e orientation);
        void     invert_polygon(unsigned start);

        // Flip all vertices horizontally or vertically, 
        // between x1 and x2, or between y1 and y2 respectively
        //--------------------------------------------------------------------
        void flip_x(double x1, double x2);
        void flip_y(double y1, double y2);

        // Concatenate path. The path is added as is.
        //--------------------------------------------------------------------
        template<class VertexSource> 
        void concat_path(VertexSource& vs, unsigned path_id = 0)
        {
            double x, y;
            unsigned cmd;
            vs.rewind(path_id);
            while(!is_stop(cmd = vs.vertex(&x, &y)))
            {
                m_vertices.add_vertex(x, y, cmd);
            }
        }

        //--------------------------------------------------------------------
        // Join path. The path is joined with the existing one, that is, 
        // it behaves as if the pen of a plotter was always down (drawing)
        template<class VertexSource> 
        void join_path(VertexSource& vs, unsigned path_id = 0)
        {
            double x, y;
            unsigned cmd;
            vs.rewind(path_id);
            cmd = vs.vertex(&x, &y);
            if(!is_stop(cmd))
            {
                if(is_vertex(cmd))
                {
                    double x0, y0;
                    unsigned cmd0 = last_vertex(&x0, &y0);
                    if(is_vertex(cmd0))
                    {
                        if(calc_distance(x, y, x0, y0) > vertex_dist_epsilon)
                        {
                            if(is_move_to(cmd)) cmd = path_cmd_line_to;
                            m_vertices.add_vertex(x, y, cmd);
                        }
                    }
                    else
                    {
                        if(is_stop(cmd0))
                        {
                            cmd = path_cmd_move_to;
                        }
                        else
                        {
                            if(is_move_to(cmd)) cmd = path_cmd_line_to;
                        }
                        m_vertices.add_vertex(x, y, cmd);
                    }
                }
                while(!is_stop(cmd = vs.vertex(&x, &y)))
                {
                    m_vertices.add_vertex(x, y, is_move_to(cmd) ? 
                                                    unsigned(path_cmd_line_to) : 
                                                    cmd);
                }
            }
        }

        // Concatenate polygon/polyline. 
        //--------------------------------------------------------------------
        template<class T> void concat_poly(const T* data, 
                                           unsigned num_points,
                                           bool closed)
        {
            poly_plain_adaptor<T> poly(data, num_points, closed);
            concat_path(poly);
        }

        // Join polygon/polyline continuously.
        //--------------------------------------------------------------------
        template<class T> void join_poly(const T* data, 
                                         unsigned num_points,
                                         bool closed)
        {
            poly_plain_adaptor<T> poly(data, num_points, closed);
            join_path(poly);
        }

        //--------------------------------------------------------------------
        void translate(double dx, double dy, unsigned path_id=0);
        void translate_all_paths(double dx, double dy);

        //--------------------------------------------------------------------
        template<class Trans>
        void transform(const Trans& trans, unsigned path_id=0)
        {
            unsigned num_ver = m_vertices.total_vertices();
            for(; path_id < num_ver; path_id++)
            {
                double x, y;
                unsigned cmd = m_vertices.vertex(path_id, &x, &y);
                if(is_stop(cmd)) break;
                if(is_vertex(cmd))
                {
                    trans.transform(&x, &y);
                    m_vertices.modify_vertex(path_id, x, y);
                }
            }
        }

        //--------------------------------------------------------------------
        template<class Trans>
        void transform_all_paths(const Trans& trans)
        {
            unsigned idx;
            unsigned num_ver = m_vertices.total_vertices();
            for(idx = 0; idx < num_ver; idx++)
            {
                double x, y;
                if(is_vertex(m_vertices.vertex(idx, &x, &y)))
                {
                    trans.transform(&x, &y);
                    m_vertices.modify_vertex(idx, x, y);
                }
            }
        }



    private:
        unsigned perceive_polygon_orientation(unsigned start, unsigned end);
        void     invert_polygon(unsigned start, unsigned end);

        VertexContainer m_vertices;
        unsigned        m_iterator;
    };

    //------------------------------------------------------------------------
    template<class VC> 
    unsigned path_base<VC>::start_new_path()
    {
        if(!is_stop(m_vertices.last_command()))
        {
            m_vertices.add_vertex(0.0, 0.0, path_cmd_stop);
        }
        return m_vertices.total_vertices();
    }


    //------------------------------------------------------------------------
    template<class VC> 
    inline void path_base<VC>::rel_to_abs(double* x, double* y) const
    {
        if(m_vertices.total_vertices())
        {
            double x2;
            double y2;
            if(is_vertex(m_vertices.last_vertex(&x2, &y2)))
            {
                *x += x2;
                *y += y2;
            }
        }
    }

    //------------------------------------------------------------------------
    template<class VC> 
    inline void path_base<VC>::move_to(double x, double y)
    {
        m_vertices.add_vertex(x, y, path_cmd_move_to);
    }

    //------------------------------------------------------------------------
    template<class VC> 
    inline void path_base<VC>::move_rel(double dx, double dy)
    {
        rel_to_abs(&dx, &dy);
        m_vertices.add_vertex(dx, dy, path_cmd_move_to);
    }

    //------------------------------------------------------------------------
    template<class VC> 
    inline void path_base<VC>::line_to(double x, double y)
    {
        m_vertices.add_vertex(x, y, path_cmd_line_to);
    }

    //------------------------------------------------------------------------
    template<class VC> 
    inline void path_base<VC>::line_rel(double dx, double dy)
    {
        rel_to_abs(&dx, &dy);
        m_vertices.add_vertex(dx, dy, path_cmd_line_to);
    }

    //------------------------------------------------------------------------
    template<class VC> 
    inline void path_base<VC>::hline_to(double x)
    {
        m_vertices.add_vertex(x, last_y(), path_cmd_line_to);
    }

    //------------------------------------------------------------------------
    template<class VC> 
    inline void path_base<VC>::hline_rel(double dx)
    {
        double dy = 0;
        rel_to_abs(&dx, &dy);
        m_vertices.add_vertex(dx, dy, path_cmd_line_to);
    }

    //------------------------------------------------------------------------
    template<class VC> 
    inline void path_base<VC>::vline_to(double y)
    {
        m_vertices.add_vertex(last_x(), y, path_cmd_line_to);
    }

    //------------------------------------------------------------------------
    template<class VC> 
    inline void path_base<VC>::vline_rel(double dy)
    {
        double dx = 0;
        rel_to_abs(&dx, &dy);
        m_vertices.add_vertex(dx, dy, path_cmd_line_to);
    }

    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::arc_to(double rx, double ry,
                               double angle,
                               bool large_arc_flag,
                               bool sweep_flag,
                               double x, double y)
    {
        if(m_vertices.total_vertices() && is_vertex(m_vertices.last_command()))
        {
            const double epsilon = 1e-30;
            double x0 = 0.0;
            double y0 = 0.0;
            m_vertices.last_vertex(&x0, &y0);

            rx = fabs(rx);
            ry = fabs(ry);

            // Ensure radii are valid
            //-------------------------
            if(rx < epsilon || ry < epsilon) 
            {
                line_to(x, y);
                return;
            }

            if(calc_distance(x0, y0, x, y) < epsilon)
            {
                // If the endpoints (x, y) and (x0, y0) are identical, then this
                // is equivalent to omitting the elliptical arc segment entirely.
                return;
            }
            bezier_arc_svg a(x0, y0, rx, ry, angle, large_arc_flag, sweep_flag, x, y);
            if(a.radii_ok())
            {
                join_path(a);
            }
            else
            {
                line_to(x, y);
            }
        }
        else
        {
            move_to(x, y);
        }
    }

    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::arc_rel(double rx, double ry,
                                double angle,
                                bool large_arc_flag,
                                bool sweep_flag,
                                double dx, double dy)
    {
        rel_to_abs(&dx, &dy);
        arc_to(rx, ry, angle, large_arc_flag, sweep_flag, dx, dy);
    }

    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::curve3(double x_ctrl, double y_ctrl, 
                               double x_to,   double y_to)
    {
        m_vertices.add_vertex(x_ctrl, y_ctrl, path_cmd_curve3);
        m_vertices.add_vertex(x_to,   y_to,   path_cmd_curve3);
    }

    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::curve3_rel(double dx_ctrl, double dy_ctrl, 
                                   double dx_to,   double dy_to)
    {
        rel_to_abs(&dx_ctrl, &dy_ctrl);
        rel_to_abs(&dx_to,   &dy_to);
        m_vertices.add_vertex(dx_ctrl, dy_ctrl, path_cmd_curve3);
        m_vertices.add_vertex(dx_to,   dy_to,   path_cmd_curve3);
    }

    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::curve3(double x_to, double y_to)
    {
        double x0;
        double y0;
        if(is_vertex(m_vertices.last_vertex(&x0, &y0)))
        {
            double x_ctrl;
            double y_ctrl; 
            unsigned cmd = m_vertices.prev_vertex(&x_ctrl, &y_ctrl);
            if(is_curve(cmd))
            {
                x_ctrl = x0 + x0 - x_ctrl;
                y_ctrl = y0 + y0 - y_ctrl;
            }
            else
            {
                x_ctrl = x0;
                y_ctrl = y0;
            }
            curve3(x_ctrl, y_ctrl, x_to, y_to);
        }
    }

    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::curve3_rel(double dx_to, double dy_to)
    {
        rel_to_abs(&dx_to, &dy_to);
        curve3(dx_to, dy_to);
    }

    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::curve4(double x_ctrl1, double y_ctrl1, 
                               double x_ctrl2, double y_ctrl2, 
                               double x_to,    double y_to)
    {
        m_vertices.add_vertex(x_ctrl1, y_ctrl1, path_cmd_curve4);
        m_vertices.add_vertex(x_ctrl2, y_ctrl2, path_cmd_curve4);
        m_vertices.add_vertex(x_to,    y_to,    path_cmd_curve4);
    }

    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::curve4_rel(double dx_ctrl1, double dy_ctrl1, 
                                   double dx_ctrl2, double dy_ctrl2, 
                                   double dx_to,    double dy_to)
    {
        rel_to_abs(&dx_ctrl1, &dy_ctrl1);
        rel_to_abs(&dx_ctrl2, &dy_ctrl2);
        rel_to_abs(&dx_to,    &dy_to);
        m_vertices.add_vertex(dx_ctrl1, dy_ctrl1, path_cmd_curve4);
        m_vertices.add_vertex(dx_ctrl2, dy_ctrl2, path_cmd_curve4);
        m_vertices.add_vertex(dx_to,    dy_to,    path_cmd_curve4);
    }

    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::curve4(double x_ctrl2, double y_ctrl2, 
                               double x_to,    double y_to)
    {
        double x0;
        double y0;
        if(is_vertex(last_vertex(&x0, &y0)))
        {
            double x_ctrl1;
            double y_ctrl1; 
            unsigned cmd = prev_vertex(&x_ctrl1, &y_ctrl1);
            if(is_curve(cmd))
            {
                x_ctrl1 = x0 + x0 - x_ctrl1;
                y_ctrl1 = y0 + y0 - y_ctrl1;
            }
            else
            {
                x_ctrl1 = x0;
                y_ctrl1 = y0;
            }
            curve4(x_ctrl1, y_ctrl1, x_ctrl2, y_ctrl2, x_to, y_to);
        }
    }

    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::curve4_rel(double dx_ctrl2, double dy_ctrl2, 
                                   double dx_to,    double dy_to)
    {
        rel_to_abs(&dx_ctrl2, &dy_ctrl2);
        rel_to_abs(&dx_to,    &dy_to);
        curve4(dx_ctrl2, dy_ctrl2, dx_to, dy_to);
    }

    //------------------------------------------------------------------------
    template<class VC> 
    inline void path_base<VC>::end_poly(unsigned flags)
    {
        if(is_vertex(m_vertices.last_command()))
        {
            m_vertices.add_vertex(0.0, 0.0, path_cmd_end_poly | flags);
        }
    }

    //------------------------------------------------------------------------
    template<class VC> 
    inline void path_base<VC>::close_polygon(unsigned flags)
    {
        end_poly(path_flags_close | flags);
    }

    //------------------------------------------------------------------------
    template<class VC> 
    inline unsigned path_base<VC>::total_vertices() const
    {
        return m_vertices.total_vertices();
    }

    //------------------------------------------------------------------------
    template<class VC> 
    inline unsigned path_base<VC>::last_vertex(double* x, double* y) const
    {
        return m_vertices.last_vertex(x, y);
    }

    //------------------------------------------------------------------------
    template<class VC> 
    inline unsigned path_base<VC>::prev_vertex(double* x, double* y) const
    {
        return m_vertices.prev_vertex(x, y);
    }

    //------------------------------------------------------------------------
    template<class VC> 
    inline double path_base<VC>::last_x() const
    {
        return m_vertices.last_x();
    }

    //------------------------------------------------------------------------
    template<class VC> 
    inline double path_base<VC>::last_y() const
    {
        return m_vertices.last_y();
    }

    //------------------------------------------------------------------------
    template<class VC> 
    inline unsigned path_base<VC>::vertex(unsigned idx, double* x, double* y) const
    {
        return m_vertices.vertex(idx, x, y);
    }
 
    //------------------------------------------------------------------------
    template<class VC> 
    inline unsigned path_base<VC>::command(unsigned idx) const
    {
        return m_vertices.command(idx);
    }

    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::modify_vertex(unsigned idx, double x, double y)
    {
        m_vertices.modify_vertex(idx, x, y);
    }

    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::modify_vertex(unsigned idx, double x, double y, unsigned cmd)
    {
        m_vertices.modify_vertex(idx, x, y, cmd);
    }

    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::modify_command(unsigned idx, unsigned cmd)
    {
        m_vertices.modify_command(idx, cmd);
    }

    //------------------------------------------------------------------------
    template<class VC> 
    inline void path_base<VC>::rewind(unsigned path_id)
    {
        m_iterator = path_id;
    }

    //------------------------------------------------------------------------
    template<class VC> 
    inline unsigned path_base<VC>::vertex(double* x, double* y)
    {
        if(m_iterator >= m_vertices.total_vertices()) return path_cmd_stop;
        return m_vertices.vertex(m_iterator++, x, y);
    }

    //------------------------------------------------------------------------
    template<class VC> 
    unsigned path_base<VC>::perceive_polygon_orientation(unsigned start,
                                                         unsigned end)
    {
        // Calculate signed area (double area to be exact)
        //---------------------
        unsigned np = end - start;
        double area = 0.0;
        unsigned i;
        for(i = 0; i < np; i++)
        {
            double x1, y1, x2, y2;
            m_vertices.vertex(start + i,            &x1, &y1);
            m_vertices.vertex(start + (i + 1) % np, &x2, &y2);
            area += x1 * y2 - y1 * x2;
        }
        return (area < 0.0) ? path_flags_cw : path_flags_ccw;
    }

    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::invert_polygon(unsigned start, unsigned end)
    {
        unsigned i;
        unsigned tmp_cmd = m_vertices.command(start);
        
        --end; // Make "end" inclusive

        // Shift all commands to one position
        for(i = start; i < end; i++)
        {
            m_vertices.modify_command(i, m_vertices.command(i + 1));
        }

        // Assign starting command to the ending command
        m_vertices.modify_command(end, tmp_cmd);

        // Reverse the polygon
        while(end > start)
        {
            m_vertices.swap_vertices(start++, end--);
        }
    }

    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::invert_polygon(unsigned start)
    {
        // Skip all non-vertices at the beginning
        while(start < m_vertices.total_vertices() && 
              !is_vertex(m_vertices.command(start))) ++start;

        // Skip all insignificant move_to
        while(start+1 < m_vertices.total_vertices() && 
              is_move_to(m_vertices.command(start)) &&
              is_move_to(m_vertices.command(start+1))) ++start;

        // Find the last vertex
        unsigned end = start + 1;
        while(end < m_vertices.total_vertices() && 
              !is_next_poly(m_vertices.command(end))) ++end;

        invert_polygon(start, end);
    }

    //------------------------------------------------------------------------
    template<class VC> 
    unsigned path_base<VC>::arrange_polygon_orientation(unsigned start, 
                                                        path_flags_e orientation)
    {
        if(orientation == path_flags_none) return start;
        
        // Skip all non-vertices at the beginning
        while(start < m_vertices.total_vertices() && 
              !is_vertex(m_vertices.command(start))) ++start;

        // Skip all insignificant move_to
        while(start+1 < m_vertices.total_vertices() && 
              is_move_to(m_vertices.command(start)) &&
              is_move_to(m_vertices.command(start+1))) ++start;

        // Find the last vertex
        unsigned end = start + 1;
        while(end < m_vertices.total_vertices() && 
              !is_next_poly(m_vertices.command(end))) ++end;

        if(end - start > 2)
        {
            if(perceive_polygon_orientation(start, end) != unsigned(orientation))
            {
                // Invert polygon, set orientation flag, and skip all end_poly
                invert_polygon(start, end);
                unsigned cmd;
                while(end < m_vertices.total_vertices() && 
                      is_end_poly(cmd = m_vertices.command(end)))
                {
                    m_vertices.modify_command(end++, set_orientation(cmd, orientation));
                }
            }
        }
        return end;
    }

    //------------------------------------------------------------------------
    template<class VC> 
    unsigned path_base<VC>::arrange_orientations(unsigned start, 
                                                 path_flags_e orientation)
    {
        if(orientation != path_flags_none)
        {
            while(start < m_vertices.total_vertices())
            {
                start = arrange_polygon_orientation(start, orientation);
                if(is_stop(m_vertices.command(start)))
                {
                    ++start;
                    break;
                }
            }
        }
        return start;
    }

    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::arrange_orientations_all_paths(path_flags_e orientation)
    {
        if(orientation != path_flags_none)
        {
            unsigned start = 0;
            while(start < m_vertices.total_vertices())
            {
                start = arrange_orientations(start, orientation);
            }
        }
    }

    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::flip_x(double x1, double x2)
    {
        unsigned i;
        double x, y;
        for(i = 0; i < m_vertices.total_vertices(); i++)
        {
            unsigned cmd = m_vertices.vertex(i, &x, &y);
            if(is_vertex(cmd))
            {
                m_vertices.modify_vertex(i, x2 - x + x1, y);
            }
        }
    }

    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::flip_y(double y1, double y2)
    {
        unsigned i;
        double x, y;
        for(i = 0; i < m_vertices.total_vertices(); i++)
        {
            unsigned cmd = m_vertices.vertex(i, &x, &y);
            if(is_vertex(cmd))
            {
                m_vertices.modify_vertex(i, x, y2 - y + y1);
            }
        }
    }

    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::translate(double dx, double dy, unsigned path_id)
    {
        unsigned num_ver = m_vertices.total_vertices();
        for(; path_id < num_ver; path_id++)
        {
            double x, y;
            unsigned cmd = m_vertices.vertex(path_id, &x, &y);
            if(is_stop(cmd)) break;
            if(is_vertex(cmd))
            {
                x += dx;
                y += dy;
                m_vertices.modify_vertex(path_id, x, y);
            }
        }
    }

    //------------------------------------------------------------------------
    template<class VC> 
    void path_base<VC>::translate_all_paths(double dx, double dy)
    {
        unsigned idx;
        unsigned num_ver = m_vertices.total_vertices();
        for(idx = 0; idx < num_ver; idx++)
        {
            double x, y;
            if(is_vertex(m_vertices.vertex(idx, &x, &y)))
            {
                x += dx;
                y += dy;
                m_vertices.modify_vertex(idx, x, y);
            }
        }
    }

    //-----------------------------------------------------vertex_stl_storage
    template<class Container> class vertex_stl_storage
    {
    public:
        typedef typename Container::value_type vertex_type;
        typedef typename vertex_type::value_type value_type;

        void remove_all() { m_vertices.clear(); }
        void free_all()   { m_vertices.clear(); }

        void add_vertex(double x, double y, unsigned cmd)
        {
            m_vertices.push_back(vertex_type(value_type(x), 
                                             value_type(y), 
                                             int8u(cmd)));
        }

        void modify_vertex(unsigned idx, double x, double y)
        {
            vertex_type& v = m_vertices[idx];
            v.x = value_type(x);
            v.y = value_type(y);
        }

        void modify_vertex(unsigned idx, double x, double y, unsigned cmd)
        {
            vertex_type& v = m_vertices[idx];
            v.x   = value_type(x);
            v.y   = value_type(y);
            v.cmd = int8u(cmd);
        }

        void modify_command(unsigned idx, unsigned cmd)
        {
            m_vertices[idx].cmd = int8u(cmd);
        }

        void swap_vertices(unsigned v1, unsigned v2)
        {
            vertex_type t = m_vertices[v1];
            m_vertices[v1] = m_vertices[v2];
            m_vertices[v2] = t;
        }

        unsigned last_command() const
        {
            return m_vertices.size() ? 
                m_vertices[m_vertices.size() - 1].cmd : 
                path_cmd_stop;
        }

        unsigned last_vertex(double* x, double* y) const
        {
            if(m_vertices.size() == 0)
            {
                *x = *y = 0.0;
                return path_cmd_stop;
            }
            return vertex(m_vertices.size() - 1, x, y);
        }

        unsigned prev_vertex(double* x, double* y) const
        {
            if(m_vertices.size() < 2)
            {
                *x = *y = 0.0;
                return path_cmd_stop;
            }
            return vertex(m_vertices.size() - 2, x, y);
        }

        double last_x() const
        {
            return m_vertices.size() ? m_vertices[m_vertices.size() - 1].x : 0.0;
        }

        double last_y() const
        {
            return m_vertices.size() ? m_vertices[m_vertices.size() - 1].y : 0.0;
        }

        unsigned total_vertices() const
        {
            return m_vertices.size();
        }

        unsigned vertex(unsigned idx, double* x, double* y) const
        {
            const vertex_type& v = m_vertices[idx];
            *x = v.x;
            *y = v.y;
            return v.cmd;
        }

        unsigned command(unsigned idx) const
        {
            return m_vertices[idx].cmd;
        }

    private:
        Container m_vertices;
    };

    //-----------------------------------------------------------path_storage
    typedef path_base<vertex_block_storage<double> > path_storage;

    // Example of declarations path_storage with pod_bvector as a container
    //-----------------------------------------------------------------------
    //typedef path_base<vertex_stl_storage<pod_bvector<vertex_d> > > path_storage;

}



// Example of declarations path_storage with std::vector as a container
//---------------------------------------------------------------------------
//#include <vector>
//namespace agg
//{
//    typedef path_base<vertex_stl_storage<std::vector<vertex_d> > > stl_path_storage; 
//}




#endif
