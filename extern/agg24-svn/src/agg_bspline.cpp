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
// class bspline
//
//----------------------------------------------------------------------------

#include "agg_bspline.h"

namespace agg
{
    //------------------------------------------------------------------------
    bspline::bspline() :
        m_max(0),
        m_num(0),
        m_x(0),
        m_y(0),
        m_last_idx(-1)
    {
    }

    //------------------------------------------------------------------------
    bspline::bspline(int num) :
        m_max(0),
        m_num(0),
        m_x(0),
        m_y(0),
        m_last_idx(-1)
    {
        init(num);
    }

    //------------------------------------------------------------------------
    bspline::bspline(int num, const double* x, const double* y) :
        m_max(0),
        m_num(0),
        m_x(0),
        m_y(0),
        m_last_idx(-1)
    {
        init(num, x, y);
    }

    
    //------------------------------------------------------------------------
    void bspline::init(int max)
    {
        if(max > 2 && max > m_max)
        {
            m_am.resize(max * 3);
            m_max = max;
            m_x   = &m_am[m_max];
            m_y   = &m_am[m_max * 2];
        }
        m_num = 0;
        m_last_idx = -1;
    }


    //------------------------------------------------------------------------
    void bspline::add_point(double x, double y)
    {
        if(m_num < m_max)
        {
            m_x[m_num] = x;
            m_y[m_num] = y;
            ++m_num;
        }
    }


    //------------------------------------------------------------------------
    void bspline::prepare()
    {
        if(m_num > 2)
        {
            int i, k, n1;
            double* temp; 
            double* r; 
            double* s;
            double h, p, d, f, e;
    
            for(k = 0; k < m_num; k++) 
            {
                m_am[k] = 0.0;
            }

            n1 = 3 * m_num;

            pod_array<double> al(n1);
            temp = &al[0];

            for(k = 0; k < n1; k++) 
            {
                temp[k] = 0.0;
            }

            r = temp + m_num;
            s = temp + m_num * 2;

            n1 = m_num - 1;
            d = m_x[1] - m_x[0];
            e = (m_y[1] - m_y[0]) / d;

            for(k = 1; k < n1; k++) 
            {
                h     = d;
                d     = m_x[k + 1] - m_x[k];
                f     = e;
                e     = (m_y[k + 1] - m_y[k]) / d;
                al[k] = d / (d + h);
                r[k]  = 1.0 - al[k];
                s[k]  = 6.0 * (e - f) / (h + d);
            }

            for(k = 1; k < n1; k++) 
            {
                p = 1.0 / (r[k] * al[k - 1] + 2.0);
                al[k] *= -p;
                s[k] = (s[k] - r[k] * s[k - 1]) * p; 
            }

            m_am[n1]     = 0.0;
            al[n1 - 1]   = s[n1 - 1];
            m_am[n1 - 1] = al[n1 - 1];

            for(k = n1 - 2, i = 0; i < m_num - 2; i++, k--) 
            {
                al[k]   = al[k] * al[k + 1] + s[k];
                m_am[k] = al[k];
            }
        }
        m_last_idx = -1;
    }



    //------------------------------------------------------------------------
    void bspline::init(int num, const double* x, const double* y)
    {
        if(num > 2)
        {
            init(num);
            int i;
            for(i = 0; i < num; i++)
            {
                add_point(*x++, *y++);
            }
            prepare();
        }
        m_last_idx = -1;
    }


    //------------------------------------------------------------------------
    void bspline::bsearch(int n, const double *x, double x0, int *i) 
    {
        int j = n - 1;
        int k;
          
        for(*i = 0; (j - *i) > 1; ) 
        {
            if(x0 < x[k = (*i + j) >> 1]) j = k; 
            else                         *i = k;
        }
    }



    //------------------------------------------------------------------------
    double bspline::interpolation(double x, int i) const
    {
        int j = i + 1;
        double d = m_x[i] - m_x[j];
        double h = x - m_x[j];
        double r = m_x[i] - x;
        double p = d * d / 6.0;
        return (m_am[j] * r * r * r + m_am[i] * h * h * h) / 6.0 / d +
               ((m_y[j] - m_am[j] * p) * r + (m_y[i] - m_am[i] * p) * h) / d;
    }


    //------------------------------------------------------------------------
    double bspline::extrapolation_left(double x) const
    {
        double d = m_x[1] - m_x[0];
        return (-d * m_am[1] / 6 + (m_y[1] - m_y[0]) / d) * 
               (x - m_x[0]) + 
               m_y[0];
    }

    //------------------------------------------------------------------------
    double bspline::extrapolation_right(double x) const
    {
        double d = m_x[m_num - 1] - m_x[m_num - 2];
        return (d * m_am[m_num - 2] / 6 + (m_y[m_num - 1] - m_y[m_num - 2]) / d) * 
               (x - m_x[m_num - 1]) + 
               m_y[m_num - 1];
    }

    //------------------------------------------------------------------------
    double bspline::get(double x) const
    {
        if(m_num > 2)
        {
            int i;

            // Extrapolation on the left
            if(x < m_x[0]) return extrapolation_left(x);

            // Extrapolation on the right
            if(x >= m_x[m_num - 1]) return extrapolation_right(x);

            // Interpolation
            bsearch(m_num, m_x, x, &i);
            return interpolation(x, i);
        }
        return 0.0;
    }


    //------------------------------------------------------------------------
    double bspline::get_stateful(double x) const
    {
        if(m_num > 2)
        {
            // Extrapolation on the left
            if(x < m_x[0]) return extrapolation_left(x);

            // Extrapolation on the right
            if(x >= m_x[m_num - 1]) return extrapolation_right(x);

            if(m_last_idx >= 0)
            {
                // Check if x is not in current range
                if(x < m_x[m_last_idx] || x > m_x[m_last_idx + 1])
                {
                    // Check if x between next points (most probably)
                    if(m_last_idx < m_num - 2 && 
                       x >= m_x[m_last_idx + 1] &&
                       x <= m_x[m_last_idx + 2])
                    {
                        ++m_last_idx;
                    }
                    else
                    if(m_last_idx > 0 && 
                       x >= m_x[m_last_idx - 1] && 
                       x <= m_x[m_last_idx])
                    {
                        // x is between pevious points
                        --m_last_idx;
                    }
                    else
                    {
                        // Else perform full search
                        bsearch(m_num, m_x, x, &m_last_idx);
                    }
                }
                return interpolation(x, m_last_idx);
            }
            else
            {
                // Interpolation
                bsearch(m_num, m_x, x, &m_last_idx);
                return interpolation(x, m_last_idx);
            }
        }
        return 0.0;
    }

}

