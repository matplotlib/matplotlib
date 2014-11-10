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
// classes conv_curve
//
//----------------------------------------------------------------------------

#ifndef AGG_CONV_CURVE_INCLUDED
#define AGG_CONV_CURVE_INCLUDED

#include "agg_basics.h"
#include "agg_curves.h"

namespace agg
{


    //---------------------------------------------------------------conv_curve
    // Curve converter class. Any path storage can have Bezier curves defined 
    // by their control points. There're two types of curves supported: curve3 
    // and curve4. Curve3 is a conic Bezier curve with 2 endpoints and 1 control
    // point. Curve4 has 2 control points (4 points in total) and can be used
    // to interpolate more complicated curves. Curve4, unlike curve3 can be used 
    // to approximate arcs, both circular and elliptical. Curves are approximated 
    // with straight lines and one of the approaches is just to store the whole 
    // sequence of vertices that approximate our curve. It takes additional 
    // memory, and at the same time the consecutive vertices can be calculated 
    // on demand. 
    //
    // Initially, path storages are not suppose to keep all the vertices of the
    // curves (although, nothing prevents us from doing so). Instead, path_storage
    // keeps only vertices, needed to calculate a curve on demand. Those vertices
    // are marked with special commands. So, if the path_storage contains curves 
    // (which are not real curves yet), and we render this storage directly, 
    // all we will see is only 2 or 3 straight line segments (for curve3 and 
    // curve4 respectively). If we need to see real curves drawn we need to 
    // include this class into the conversion pipeline. 
    //
    // Class conv_curve recognizes commands path_cmd_curve3 and path_cmd_curve4 
    // and converts these vertices into a move_to/line_to sequence. 
    //-----------------------------------------------------------------------
    template<class VertexSource, 
             class Curve3=curve3, 
             class Curve4=curve4> class conv_curve
    {
    public:
        typedef Curve3 curve3_type;
        typedef Curve4 curve4_type;
        typedef conv_curve<VertexSource, Curve3, Curve4> self_type;

        explicit conv_curve(VertexSource& source) :
          m_source(&source), m_last_x(0.0), m_last_y(0.0) {}
        void attach(VertexSource& source) { m_source = &source; }

        void approximation_method(curve_approximation_method_e v) 
        { 
            m_curve3.approximation_method(v);
            m_curve4.approximation_method(v);
        }

        curve_approximation_method_e approximation_method() const 
        { 
            return m_curve4.approximation_method();
        }

        void approximation_scale(double s) 
        { 
            m_curve3.approximation_scale(s); 
            m_curve4.approximation_scale(s); 
        }

        double approximation_scale() const 
        { 
            return m_curve4.approximation_scale();  
        }

        void angle_tolerance(double v) 
        { 
            m_curve3.angle_tolerance(v); 
            m_curve4.angle_tolerance(v); 
        }

        double angle_tolerance() const 
        { 
            return m_curve4.angle_tolerance();  
        }

        void cusp_limit(double v) 
        { 
            m_curve3.cusp_limit(v); 
            m_curve4.cusp_limit(v); 
        }

        double cusp_limit() const 
        { 
            return m_curve4.cusp_limit();  
        }

        void     rewind(unsigned path_id); 
        unsigned vertex(double* x, double* y);

    private:
        conv_curve(const self_type&);
        const self_type& operator = (const self_type&);

        VertexSource* m_source;
        double        m_last_x;
        double        m_last_y;
        curve3_type   m_curve3;
        curve4_type   m_curve4;
    };



    //------------------------------------------------------------------------
    template<class VertexSource, class Curve3, class Curve4>
    void conv_curve<VertexSource, Curve3, Curve4>::rewind(unsigned path_id)
    {
        m_source->rewind(path_id);
        m_last_x = 0.0;
        m_last_y = 0.0;
        m_curve3.reset();
        m_curve4.reset();
    }


    //------------------------------------------------------------------------
    template<class VertexSource, class Curve3, class Curve4>
    unsigned conv_curve<VertexSource, Curve3, Curve4>::vertex(double* x, double* y)
    {
        if(!is_stop(m_curve3.vertex(x, y)))
        {
            m_last_x = *x;
            m_last_y = *y;
            return path_cmd_line_to;
        }

        if(!is_stop(m_curve4.vertex(x, y)))
        {
            m_last_x = *x;
            m_last_y = *y;
            return path_cmd_line_to;
        }

        double ct2_x;
        double ct2_y;
        double end_x;
        double end_y;

        unsigned cmd = m_source->vertex(x, y);
        switch(cmd)
        {
        case path_cmd_curve3:
            m_source->vertex(&end_x, &end_y);

            m_curve3.init(m_last_x, m_last_y, 
                          *x,       *y, 
                          end_x,     end_y);

            m_curve3.vertex(x, y);    // First call returns path_cmd_move_to
            m_curve3.vertex(x, y);    // This is the first vertex of the curve
            cmd = path_cmd_line_to;
            break;

        case path_cmd_curve4:
            m_source->vertex(&ct2_x, &ct2_y);
            m_source->vertex(&end_x, &end_y);

            m_curve4.init(m_last_x, m_last_y, 
                          *x,       *y, 
                          ct2_x,    ct2_y, 
                          end_x,    end_y);

            m_curve4.vertex(x, y);    // First call returns path_cmd_move_to
            m_curve4.vertex(x, y);    // This is the first vertex of the curve
            cmd = path_cmd_line_to;
            break;
        }
        m_last_x = *x;
        m_last_y = *y;
        return cmd;
    }


}



#endif
