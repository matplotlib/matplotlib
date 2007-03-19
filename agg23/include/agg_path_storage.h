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

#ifndef AGG_PATH_STORAGE_INCLUDED
#define AGG_PATH_STORAGE_INCLUDED

#include "agg_basics.h"

namespace agg
{

    //------------------------------------------------------------path_storage
    // A container to store vertices with their flags. 
    // A path consists of a number of contours separated with "move_to" 
    // commands. The path storage can keep and maintain more than one
    // path. 
    // To navigate to the beginning of a particular path, use rewind(path_id);
    // Where path_id is what start_new_path() returns. So, when you call
    // start_new_path() you need to store its return value somewhere else
    // to navigate to the path afterwards.
    //
    // See Implementation: agg_path_storage.cpp 
    // See also: vertex_source concept
    //------------------------------------------------------------------------
    class path_storage
    {
        // Allocation parameters
        enum
        {
            block_shift = 8,
            block_size  = 1 << block_shift,
            block_mask  = block_size - 1,
            block_pool  = 256
        };

    public:
        //--------------------------------------------------------------------
        class vertex_source
        {
        public:
            vertex_source() {}
            vertex_source(const path_storage& p) : m_path(&p), m_vertex_idx(0) {}

            void rewind(unsigned path_id)
            {
                m_vertex_idx = path_id;
            }

            unsigned vertex(double* x, double* y)
            {
                return (m_vertex_idx < m_path->total_vertices())? 
                    m_path->vertex(m_vertex_idx++, x, y):
                    path_cmd_stop;
            }

        private:
            const path_storage* m_path;
            unsigned            m_vertex_idx;
        };


        ~path_storage();
        path_storage();
        path_storage(const path_storage& ps);

        void remove_all();

        unsigned last_vertex(double* x, double* y) const;
        unsigned prev_vertex(double* x, double* y) const;

        double last_x() const;
        double last_y() const;

        void rel_to_abs(double* x, double* y) const;

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

        void close_polygon(unsigned flags = path_flags_none)
        {
            end_poly(path_flags_close | flags);
        }

        void add_poly(const double* vertices, unsigned num, 
                      bool solid_path = false,
                      unsigned end_flags = path_flags_none);

        template<class VertexSource> 
        void add_path(VertexSource& vs, 
                      unsigned path_id = 0, 
                      bool solid_path = true)
        {
            double x, y;
            unsigned cmd;
            vs.rewind(path_id);
            while(!is_stop(cmd = vs.vertex(&x, &y)))
            {
                if(is_move_to(cmd) && solid_path && m_total_vertices) 
                {
                    cmd = path_cmd_line_to;
                }
                add_vertex(x, y, cmd);
            }
        }

        unsigned start_new_path();

        void copy_from(const path_storage& ps);
        const path_storage& operator = (const path_storage& ps)
        {
            copy_from(ps);
            return *this;
        }


        unsigned total_vertices() const { return m_total_vertices; }
        unsigned vertex(unsigned idx, double* x, double* y) const
        {
            unsigned nb = idx >> block_shift;
            const double* pv = m_coord_blocks[nb] + ((idx & block_mask) << 1);
            *x = *pv++;
            *y = *pv;
            return m_cmd_blocks[nb][idx & block_mask];
        }
        unsigned command(unsigned idx) const
        {
            return m_cmd_blocks[idx >> block_shift][idx & block_mask];
        }

        void     rewind(unsigned path_id);
        unsigned vertex(double* x, double* y);

        // Arrange the orientation of all the polygons. After calling this
        // method all the polygons will have the same orientation
        // determined by the new_orientation flag, i.e., 
        // path_flags_cw or path_flags_ccw
        unsigned arrange_orientations(unsigned path_id, path_flags_e new_orientation);
        void arrange_orientations_all_paths(path_flags_e new_orientation);

        // Flip all the vertices horizontally or vertically
        void flip_x(double x1, double x2);
        void flip_y(double y1, double y2);
        
        // This function adds a vertex with its flags directly. Since there's no 
        // checking for errors, keeping proper path integrity is the responsibility
        // of the caller. It can be said the function is "not very public". 
        void add_vertex(double x, double y, unsigned cmd);

        // Allows you to modify vertex coordinates. The caller must know 
        // the index of the vertex. 
        void modify_vertex(unsigned idx, double x, double y)
        {
            double* pv = m_coord_blocks[idx >> block_shift] + ((idx & block_mask) << 1);
            *pv++ = x;
            *pv   = y;
        }

        // Allows you to modify vertex command. The caller must know 
        // the index of the vertex. 
        void modify_command(unsigned idx, unsigned cmd)
        {
            m_cmd_blocks[idx >> block_shift][idx & block_mask] = (unsigned char)cmd;
        }


    private:
        void allocate_block(unsigned nb);
        unsigned char* storage_ptrs(double** xy_ptr);
        unsigned perceive_polygon_orientation(unsigned idx, 
                                              double xs, double ys,
                                              unsigned* orientation);
        void reverse_polygon(unsigned start, unsigned end);

    private:
        unsigned        m_total_vertices;
        unsigned        m_total_blocks;
        unsigned        m_max_blocks;
        double**        m_coord_blocks;
        unsigned char** m_cmd_blocks;
        unsigned        m_iterator;
    };


    //------------------------------------------------------------------------
    inline unsigned path_storage::vertex(double* x, double* y)
    {
        if(m_iterator >= m_total_vertices) return path_cmd_stop;
        return vertex(m_iterator++, x, y);
    }

    //------------------------------------------------------------------------
    inline unsigned path_storage::prev_vertex(double* x, double* y) const
    {
        if(m_total_vertices > 1)
        {
            return vertex(m_total_vertices - 2, x, y);
        }
        return path_cmd_stop;
    }

    //------------------------------------------------------------------------
    inline unsigned path_storage::last_vertex(double* x, double* y) const
    {
        if(m_total_vertices)
        {
            return vertex(m_total_vertices - 1, x, y);
        }
        return path_cmd_stop;
    }

    //------------------------------------------------------------------------
    inline double path_storage::last_x() const
    {
        if(m_total_vertices)
        {
            unsigned idx = m_total_vertices - 1;
            return m_coord_blocks[idx >> block_shift][(idx & block_mask) << 1];
        }
        return 0.0;
    }

    //------------------------------------------------------------------------
    inline double path_storage::last_y() const
    {
        if(m_total_vertices)
        {
            unsigned idx = m_total_vertices - 1;
            return m_coord_blocks[idx >> block_shift][((idx & block_mask) << 1) + 1];
        }
        return 0.0;
    }

    //------------------------------------------------------------------------
    inline void path_storage::rel_to_abs(double* x, double* y) const
    {
        if(m_total_vertices)
        {
            double x2;
            double y2;
            if(is_vertex(vertex(m_total_vertices - 1, &x2, &y2)))
            {
                *x += x2;
                *y += y2;
            }
        }
    }

    //------------------------------------------------------------------------
    inline unsigned char* path_storage::storage_ptrs(double** xy_ptr)
    {
        unsigned nb = m_total_vertices >> block_shift;
        if(nb >= m_total_blocks)
        {
            allocate_block(nb);
        }
        *xy_ptr = m_coord_blocks[nb] + ((m_total_vertices & block_mask) << 1);
        return m_cmd_blocks[nb] + (m_total_vertices & block_mask);
    }


    //------------------------------------------------------------------------
    inline void path_storage::add_vertex(double x, double y, unsigned cmd)
    {
        double* coord_ptr = 0;
        unsigned char* cmd_ptr = storage_ptrs(&coord_ptr);
        *cmd_ptr = (unsigned char)cmd;
        *coord_ptr++ = x;
        *coord_ptr   = y;
        m_total_vertices++;
    }

    //------------------------------------------------------------------------
    inline void path_storage::move_to(double x, double y)
    {
        add_vertex(x, y, path_cmd_move_to);
    }

    //------------------------------------------------------------------------
    inline void path_storage::move_rel(double dx, double dy)
    {
        rel_to_abs(&dx, &dy);
        add_vertex(dx, dy, path_cmd_move_to);
    }

    //------------------------------------------------------------------------
    inline void path_storage::line_to(double x, double y)
    {
        add_vertex(x, y, path_cmd_line_to);
    }

    //------------------------------------------------------------------------
    inline void path_storage::line_rel(double dx, double dy)
    {
        rel_to_abs(&dx, &dy);
        add_vertex(dx, dy, path_cmd_line_to);
    }

    //------------------------------------------------------------------------
    inline void path_storage::hline_to(double x)
    {
        add_vertex(x, last_y(), path_cmd_line_to);
    }

    //------------------------------------------------------------------------
    inline void path_storage::hline_rel(double dx)
    {
        double dy = 0;
        rel_to_abs(&dx, &dy);
        add_vertex(dx, dy, path_cmd_line_to);
    }

    //------------------------------------------------------------------------
    inline void path_storage::vline_to(double y)
    {
        add_vertex(last_x(), y, path_cmd_line_to);
    }

    //------------------------------------------------------------------------
    inline void path_storage::vline_rel(double dy)
    {
        double dx = 0;
        rel_to_abs(&dx, &dy);
        add_vertex(dx, dy, path_cmd_line_to);
    }

}



#endif
