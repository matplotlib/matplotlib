//----------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.3
// Copyright (C) 2002-2005 Maxim Shemanarev (http://www.antigrain.com)
// Copyright (C) 2005 Tony Juricic (tonygeek@yahoo.com)
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
// classes curve3 and curve4
//
//----------------------------------------------------------------------------

#ifndef AGG_CURVES_INCLUDED
#define AGG_CURVES_INCLUDED

#include "agg_basics.h"

namespace agg
{

    // See Implemantation agg_curves.cpp

    
    //------------------------------------------------------------------curve3
    class curve3
    {
    public:
        curve3() :
          m_num_steps(0), m_step(0), m_scale(1.0) { }

        curve3(double x1, double y1, 
               double x2, double y2, 
               double x3, double y3) :
          m_num_steps(0), m_step(0), m_scale(1.0) 
        { 
            init(x1, y1, x2, y2, x3, y3);
        }

        void reset() { m_num_steps = 0; m_step = -1; }
        void init(double x1, double y1, 
                  double x2, double y2, 
                  double x3, double y3);
        void approximation_scale(double s) { m_scale = s; }
        double approximation_scale() const { return m_scale;  }

        void     rewind(unsigned path_id);
        unsigned vertex(double* x, double* y);

    private:
        int      m_num_steps;
        int      m_step;
        double   m_scale;
        double   m_start_x; 
        double   m_start_y;
        double   m_end_x; 
        double   m_end_y;
        double   m_fx; 
        double   m_fy;
        double   m_dfx; 
        double   m_dfy;
        double   m_ddfx; 
        double   m_ddfy;
        double   m_saved_fx; 
        double   m_saved_fy;
        double   m_saved_dfx; 
        double   m_saved_dfy;
    };







    //-----------------------------------------------------------------curve4
    class curve4
    {
    public:
        struct points
        {
            double cp[8];
            points() {}
            points(double x1, double y1, 
                   double x2, double y2,
                   double x3, double y3, 
                   double x4, double y4)
            {
                cp[0] = x1; cp[1] = y1; cp[2] = x2; cp[3] = y2;
                cp[4] = x3; cp[5] = y3; cp[6] = x4; cp[7] = y4;
            }
            double  operator [] (unsigned i) const { return cp[i]; }
            double& operator [] (unsigned i)       { return cp[i]; }
        };

        curve4() :
            m_num_steps(0), m_step(0), m_scale(1.0) { }

        curve4(double x1, double y1, 
               double x2, double y2, 
               double x3, double y3,
               double x4, double y4) :
            m_num_steps(0), m_step(0), m_scale(1.0) 
        { 
            init(x1, y1, x2, y2, x3, y3, x4, y4);
        }

        curve4(const points& cp) :
            m_num_steps(0), m_step(0), m_scale(1.0) 
        { 
            init(cp[0], cp[1], cp[2], cp[3], cp[4], cp[5], cp[6], cp[7]);
        }

        void reset() { m_num_steps = 0; m_step = -1; }
        void init(double x1, double y1, 
                  double x2, double y2, 
                  double x3, double y3,
                  double x4, double y4);

        void init(const points& cp)
        {
            init(cp[0], cp[1], cp[2], cp[3], cp[4], cp[5], cp[6], cp[7]);
        }

        void approximation_scale(double s) { m_scale = s; }
        double approximation_scale() const { return m_scale;  }

        void     rewind(unsigned path_id);
        unsigned vertex(double* x, double* y);

    private:
        int      m_num_steps;
        int      m_step;
        double   m_scale;
        double   m_start_x; 
        double   m_start_y;
        double   m_end_x; 
        double   m_end_y;
        double   m_fx; 
        double   m_fy;
        double   m_dfx; 
        double   m_dfy;
        double   m_ddfx; 
        double   m_ddfy;
        double   m_dddfx; 
        double   m_dddfy;
        double   m_saved_fx; 
        double   m_saved_fy;
        double   m_saved_dfx; 
        double   m_saved_dfy;
        double   m_saved_ddfx; 
        double   m_saved_ddfy;
    };



    //-------------------------------------------------------catrom_to_bezier
    inline curve4::points catrom_to_bezier(double x1, double y1, 
                                           double x2, double y2, 
                                           double x3, double y3,
                                           double x4, double y4)
    {
        // Trans. matrix Catmull-Rom to Bezier
        //
        //  0       1       0       0
        //  -1/6    1       1/6     0
        //  0       1/6     1       -1/6
        //  0       0       1       0
        //
        return curve4::points(
            x2,
            y2,
            (-x1 + 6*x2 + x3) / 6,
            (-y1 + 6*y2 + y3) / 6,
            ( x2 + 6*x3 - x4) / 6,
            ( y2 + 6*y3 - y4) / 6,
            x3,
            y3);
    }


    //-----------------------------------------------------------------------
    inline curve4::points 
    catrom_to_bezier(const curve4::points& cp)
    {
        return catrom_to_bezier(cp[0], cp[1], cp[2], cp[3], 
                                cp[4], cp[5], cp[6], cp[7]);
    }



    //-----------------------------------------------------ubspline_to_bezier
    inline curve4::points ubspline_to_bezier(double x1, double y1, 
                                             double x2, double y2, 
                                             double x3, double y3,
                                             double x4, double y4)
    {
        // Trans. matrix Uniform BSpline to Bezier
        //
        //  1/6     4/6     1/6     0
        //  0       4/6     2/6     0
        //  0       2/6     4/6     0
        //  0       1/6     4/6     1/6
        //
        return curve4::points(
            (x1 + 4*x2 + x3) / 6,
            (y1 + 4*y2 + y3) / 6,
            (4*x2 + 2*x3) / 6,
            (4*y2 + 2*y3) / 6,
            (2*x2 + 4*x3) / 6,
            (2*y2 + 4*y3) / 6,
            (x2 + 4*x3 + x4) / 6,
            (y2 + 4*y3 + y4) / 6);
    }


    //-----------------------------------------------------------------------
    inline curve4::points 
    ubspline_to_bezier(const curve4::points& cp)
    {
        return ubspline_to_bezier(cp[0], cp[1], cp[2], cp[3], 
                                  cp[4], cp[5], cp[6], cp[7]);
    }




    //------------------------------------------------------hermite_to_bezier
    inline curve4::points hermite_to_bezier(double x1, double y1, 
                                            double x2, double y2, 
                                            double x3, double y3,
                                            double x4, double y4)
    {
        // Trans. matrix Hermite to Bezier
        //
        //  1       0       0       0
        //  1       0       1/3     0
        //  0       1       0       -1/3
        //  0       1       0       0
        //
        return curve4::points(
            x1,
            y1,
            (3*x1 + x3) / 3,
            (3*y1 + y3) / 3,
            (3*x2 - x4) / 3,
            (3*y2 - y4) / 3,
            x2,
            y2);
    }



    //-----------------------------------------------------------------------
    inline curve4::points 
    hermite_to_bezier(const curve4::points& cp)
    {
        return hermite_to_bezier(cp[0], cp[1], cp[2], cp[3], 
                                 cp[4], cp[5], cp[6], cp[7]);
    }




}

#endif
