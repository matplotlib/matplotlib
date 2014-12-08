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

#ifndef AGG_TRANS_SINGLE_PATH_INCLUDED
#define AGG_TRANS_SINGLE_PATH_INCLUDED

#include "agg_basics.h"
#include "agg_vertex_sequence.h"

namespace agg
{

    // See also: agg_trans_single_path.cpp
    //
    //-------------------------------------------------------trans_single_path
    class trans_single_path
    {
        enum status_e
        {
            initial,
            making_path,
            ready
        };

    public:
        typedef vertex_sequence<vertex_dist, 6> vertex_storage;

        trans_single_path();

        //--------------------------------------------------------------------
        void   base_length(double v)  { m_base_length = v; }
        double base_length() const { return m_base_length; }

        //--------------------------------------------------------------------
        void preserve_x_scale(bool f) { m_preserve_x_scale = f;    }
        bool preserve_x_scale() const { return m_preserve_x_scale; }

        //--------------------------------------------------------------------
        void reset();
        void move_to(double x, double y);
        void line_to(double x, double y);
        void finalize_path();

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
                if(is_move_to(cmd)) 
                {
                    move_to(x, y);
                }
                else 
                {
                    if(is_vertex(cmd))
                    {
                        line_to(x, y);
                    }
                }
            }
            finalize_path();
        }

        //--------------------------------------------------------------------
        double total_length() const;
        void transform(double *x, double *y) const;

    private:
        vertex_storage m_src_vertices;
        double         m_base_length;
        double         m_kindex;
        status_e       m_status;
        bool           m_preserve_x_scale;
    };


}

#endif
