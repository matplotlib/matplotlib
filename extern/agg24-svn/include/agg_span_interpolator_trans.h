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
// Horizontal span interpolator for use with an arbitrary transformer
// The efficiency highly depends on the operations done in the transformer
//
//----------------------------------------------------------------------------

#ifndef AGG_SPAN_INTERPOLATOR_TRANS_INCLUDED
#define AGG_SPAN_INTERPOLATOR_TRANS_INCLUDED

#include "agg_basics.h"

namespace agg
{
    //=================================================span_interpolator_trans
    template<class Transformer, unsigned SubpixelShift = 8> 
    class span_interpolator_trans
    {
    public:
        typedef Transformer trans_type;
        enum subpixel_scale_e
        {
            subpixel_shift = SubpixelShift,
            subpixel_scale = 1 << subpixel_shift
        };

        //--------------------------------------------------------------------
        span_interpolator_trans() {}
        span_interpolator_trans(trans_type& trans) : m_trans(&trans) {}
        span_interpolator_trans(trans_type& trans,
                                double x, double y, unsigned) :
            m_trans(&trans)
        {
            begin(x, y, 0);
        }

        //----------------------------------------------------------------
        const trans_type& transformer() const { return *m_trans; }
        void transformer(const trans_type& trans) { m_trans = &trans; }

        //----------------------------------------------------------------
        void begin(double x, double y, unsigned)
        {
            m_x = x;
            m_y = y;
            m_trans->transform(&x, &y);
            m_ix = iround(x * subpixel_scale);
            m_iy = iround(y * subpixel_scale);
        }

        //----------------------------------------------------------------
        void operator++()
        {
            m_x += 1.0;
            double x = m_x;
            double y = m_y;
            m_trans->transform(&x, &y);
            m_ix = iround(x * subpixel_scale);
            m_iy = iround(y * subpixel_scale);
        }

        //----------------------------------------------------------------
        void coordinates(int* x, int* y) const
        {
            *x = m_ix;
            *y = m_iy;
        }

    private:
        trans_type*       m_trans;
        double            m_x;
        double            m_y;
        int               m_ix;
        int               m_iy;
    };

}

#endif
