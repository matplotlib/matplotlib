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

#ifndef AGG_SPAN_INTERPOLATOR_ADAPTOR_INCLUDED
#define AGG_SPAN_INTERPOLATOR_ADAPTOR_INCLUDED

#include "agg_basics.h"

namespace agg
{

    //===============================================span_interpolator_adaptor
    template<class Interpolator, class Distortion>
    class span_interpolator_adaptor : public Interpolator
    {
    public:
        typedef Interpolator base_type;
        typedef typename base_type::trans_type trans_type;
        typedef Distortion distortion_type;

        //--------------------------------------------------------------------
        span_interpolator_adaptor() {}
        span_interpolator_adaptor(trans_type& trans, 
                                  distortion_type& dist) :
            base_type(trans),
            m_distortion(&dist)
        {   
        }

        //--------------------------------------------------------------------
        span_interpolator_adaptor(trans_type& trans,
                                  distortion_type& dist,
                                  double x, double y, unsigned len) :
            base_type(trans, x, y, len),
            m_distortion(&dist)
        {
        }

        //--------------------------------------------------------------------
        distortion_type& distortion() const
        {
            return *m_distortion;
        }

        //--------------------------------------------------------------------
        void distortion(distortion_type& dist)
        {
            m_distortion = dist;
        }

        //--------------------------------------------------------------------
        void coordinates(int* x, int* y) const
        {
            base_type::coordinates(x, y);
            m_distortion->calculate(x, y);
        }

    private:
        //--------------------------------------------------------------------
        distortion_type* m_distortion;
    };
}


#endif
