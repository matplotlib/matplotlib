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

#ifndef AGG_GAMMA_FUNCTIONS_INCLUDED
#define AGG_GAMMA_FUNCTIONS_INCLUDED

#include <math.h>
#include "agg_basics.h"

namespace agg
{
    //===============================================================gamma_none
    struct gamma_none
    {
        double operator()(double x) const { return x; }
    };


    //==============================================================gamma_power
    class gamma_power
    {
    public:
        gamma_power() : m_gamma(1.0) {}
        gamma_power(double g) : m_gamma(g) {}

        void gamma(double g) { m_gamma = g; }
        double gamma() const { return m_gamma; }

        double operator() (double x) const
        {
            return pow(x, m_gamma);
        }

    private:
        double m_gamma;
    };


    //==========================================================gamma_threshold
    class gamma_threshold
    {
    public:
        gamma_threshold() : m_threshold(0.5) {}
        gamma_threshold(double t) : m_threshold(t) {}

        void threshold(double t) { m_threshold = t; }
        double threshold() const { return m_threshold; }

        double operator() (double x) const
        {
            return (x < m_threshold) ? 0.0 : 1.0;
        }

    private:
        double m_threshold;
    };


    //============================================================gamma_linear
    class gamma_linear
    {
    public:
        gamma_linear() : m_start(0.0), m_end(1.0) {}
        gamma_linear(double s, double e) : m_start(s), m_end(e) {}

        void set(double s, double e) { m_start = s; m_end = e; }
        void start(double s) { m_start = s; }
        void end(double e) { m_end = e; }
        double start() const { return m_start; }
        double end() const { return m_end; }

        double operator() (double x) const
        {
            if(x < m_start) return 0.0;
            if(x > m_end) return 1.0;
            return (x - m_start) / (m_end - m_start);
        }

    private:
        double m_start;
        double m_end;
    };


    //==========================================================gamma_multiply
    class gamma_multiply
    {
    public:
        gamma_multiply() : m_mul(1.0) {}
        gamma_multiply(double v) : m_mul(v) {}

        void value(double v) { m_mul = v; }
        double value() const { return m_mul; }

        double operator() (double x) const
        {
            double y = x * m_mul;
            if(y > 1.0) y = 1.0;
            return y;
        }

    private:
        double m_mul;
    };

}

#endif



