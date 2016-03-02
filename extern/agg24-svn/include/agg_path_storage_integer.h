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

#ifndef AGG_PATH_STORAGE_INTEGER_INCLUDED
#define AGG_PATH_STORAGE_INTEGER_INCLUDED

#include <string.h>
#include "agg_array.h"

namespace agg
{
    //---------------------------------------------------------vertex_integer
    template<class T, unsigned CoordShift=6> struct vertex_integer
    {
        enum path_cmd
        {
            cmd_move_to = 0,
            cmd_line_to = 1,
            cmd_curve3  = 2,
            cmd_curve4  = 3
        };

        enum coord_scale_e
        {
            coord_shift = CoordShift,
            coord_scale  = 1 << coord_shift
        };

        T x,y;
        vertex_integer() {}
        vertex_integer(T x_, T y_, unsigned flag) :
            x(((x_ << 1) & ~1) | (flag &  1)),
            y(((y_ << 1) & ~1) | (flag >> 1)) {}

        unsigned vertex(double* x_, double* y_, 
                        double dx=0, double dy=0,
                        double scale=1.0) const
        {
            *x_ = dx + (double(x >> 1) / coord_scale) * scale;
            *y_ = dy + (double(y >> 1) / coord_scale) * scale;
            switch(((y & 1) << 1) | (x & 1))
            {
                case cmd_move_to: return path_cmd_move_to;
                case cmd_line_to: return path_cmd_line_to;
                case cmd_curve3:  return path_cmd_curve3;
                case cmd_curve4:  return path_cmd_curve4;
            }
            return path_cmd_stop;
        }
    };


    //---------------------------------------------------path_storage_integer
    template<class T, unsigned CoordShift=6> class path_storage_integer
    {
    public:
        typedef T value_type;
        typedef vertex_integer<T, CoordShift> vertex_integer_type;

        //--------------------------------------------------------------------
        path_storage_integer() : m_storage(), m_vertex_idx(0), m_closed(true) {}

        //--------------------------------------------------------------------
        void remove_all() { m_storage.remove_all(); }

        //--------------------------------------------------------------------
        void move_to(T x, T y)
        {
            m_storage.add(vertex_integer_type(x, y, vertex_integer_type::cmd_move_to));
        }

        //--------------------------------------------------------------------
        void line_to(T x, T y)
        {
            m_storage.add(vertex_integer_type(x, y, vertex_integer_type::cmd_line_to));
        }

        //--------------------------------------------------------------------
        void curve3(T x_ctrl,  T y_ctrl, 
                    T x_to,    T y_to)
        {
            m_storage.add(vertex_integer_type(x_ctrl, y_ctrl, vertex_integer_type::cmd_curve3));
            m_storage.add(vertex_integer_type(x_to,   y_to,   vertex_integer_type::cmd_curve3));
        }

        //--------------------------------------------------------------------
        void curve4(T x_ctrl1, T y_ctrl1, 
                    T x_ctrl2, T y_ctrl2, 
                    T x_to,    T y_to)
        {
            m_storage.add(vertex_integer_type(x_ctrl1, y_ctrl1, vertex_integer_type::cmd_curve4));
            m_storage.add(vertex_integer_type(x_ctrl2, y_ctrl2, vertex_integer_type::cmd_curve4));
            m_storage.add(vertex_integer_type(x_to,    y_to,    vertex_integer_type::cmd_curve4));
        }

        //--------------------------------------------------------------------
        void close_polygon() {}

        //--------------------------------------------------------------------
        unsigned size() const { return m_storage.size(); }
        unsigned vertex(unsigned idx, double* x, double* y) const
        {
            return m_storage[idx].vertex(x, y);
        }

        //--------------------------------------------------------------------
        unsigned byte_size() const { return m_storage.size() * sizeof(vertex_integer_type); }
        void serialize(int8u* ptr) const
        {
            unsigned i;
            for(i = 0; i < m_storage.size(); i++)
            {
                memcpy(ptr, &m_storage[i], sizeof(vertex_integer_type));
                ptr += sizeof(vertex_integer_type);
            }
        }

        //--------------------------------------------------------------------
        void rewind(unsigned) 
        { 
            m_vertex_idx = 0; 
            m_closed = true;
        }

        //--------------------------------------------------------------------
        unsigned vertex(double* x, double* y)
        {
            if(m_storage.size() < 2 || m_vertex_idx > m_storage.size()) 
            {
                *x = 0;
                *y = 0;
                return path_cmd_stop;
            }
            if(m_vertex_idx == m_storage.size())
            {
                *x = 0;
                *y = 0;
                ++m_vertex_idx;
                return path_cmd_end_poly | path_flags_close;
            }
            unsigned cmd = m_storage[m_vertex_idx].vertex(x, y);
            if(is_move_to(cmd) && !m_closed)
            {
                *x = 0;
                *y = 0;
                m_closed = true;
                return path_cmd_end_poly | path_flags_close;
            }
            m_closed = false;
            ++m_vertex_idx;
            return cmd;
        }

        //--------------------------------------------------------------------
        rect_d bounding_rect() const
        {
            rect_d bounds(1e100, 1e100, -1e100, -1e100);
            if(m_storage.size() == 0)
            {
                bounds.x1 = bounds.y1 = bounds.x2 = bounds.y2 = 0.0;
            }
            else
            {
                unsigned i;
                for(i = 0; i < m_storage.size(); i++)
                {
                    double x, y;
                    m_storage[i].vertex(&x, &y);
                    if(x < bounds.x1) bounds.x1 = x;
                    if(y < bounds.y1) bounds.y1 = y;
                    if(x > bounds.x2) bounds.x2 = x;
                    if(y > bounds.y2) bounds.y2 = y;
                }
            }
            return bounds;
        }

    private:
        pod_bvector<vertex_integer_type, 6> m_storage;
        unsigned                            m_vertex_idx;
        bool                                m_closed;
    };




    //-----------------------------------------serialized_integer_path_adaptor
    template<class T, unsigned CoordShift=6> class serialized_integer_path_adaptor
    {
    public:
        typedef vertex_integer<T, CoordShift> vertex_integer_type;

        //--------------------------------------------------------------------
        serialized_integer_path_adaptor() :
            m_data(0),
            m_end(0),
            m_ptr(0),
            m_dx(0.0),
            m_dy(0.0),
            m_scale(1.0),
            m_vertices(0)
        {}

        //--------------------------------------------------------------------
        serialized_integer_path_adaptor(const int8u* data, unsigned size,
                                        double dx, double dy) :
            m_data(data),
            m_end(data + size),
            m_ptr(data),
            m_dx(dx),
            m_dy(dy),
            m_vertices(0)
        {}

        //--------------------------------------------------------------------
        void init(const int8u* data, unsigned size, 
                  double dx, double dy, double scale=1.0)
        {
            m_data     = data;
            m_end      = data + size;
            m_ptr      = data;
            m_dx       = dx;
            m_dy       = dy;
            m_scale    = scale;
            m_vertices = 0;
        }


        //--------------------------------------------------------------------
        void rewind(unsigned) 
        { 
            m_ptr      = m_data; 
            m_vertices = 0;
        }

        //--------------------------------------------------------------------
        unsigned vertex(double* x, double* y)
        {
            if(m_data == 0 || m_ptr > m_end) 
            {
                *x = 0;
                *y = 0;
                return path_cmd_stop;
            }

            if(m_ptr == m_end)
            {
                *x = 0;
                *y = 0;
                m_ptr += sizeof(vertex_integer_type);
                return path_cmd_end_poly | path_flags_close;
            }

            vertex_integer_type v;
            memcpy(&v, m_ptr, sizeof(vertex_integer_type));
            unsigned cmd = v.vertex(x, y, m_dx, m_dy, m_scale);
            if(is_move_to(cmd) && m_vertices > 2)
            {
                *x = 0;
                *y = 0;
                m_vertices = 0;
                return path_cmd_end_poly | path_flags_close;
            }
            ++m_vertices;
            m_ptr += sizeof(vertex_integer_type);
            return cmd;
        }

    private:
        const int8u* m_data;
        const int8u* m_end;
        const int8u* m_ptr;
        double       m_dx;
        double       m_dy;
        double       m_scale;
        unsigned     m_vertices;
    };

}


#endif

