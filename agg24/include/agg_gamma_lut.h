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

#ifndef AGG_GAMMA_LUT_INCLUDED
#define AGG_GAMMA_LUT_INCLUDED

#include <math.h>
#include "agg_basics.h"

namespace agg
{
    template<class LoResT=int8u, 
             class HiResT=int8u, 
             unsigned GammaShift=8, 
             unsigned HiResShift=8> class gamma_lut
    {
    public:
        typedef gamma_lut<LoResT, HiResT, GammaShift, HiResShift> self_type;

        enum gamma_scale_e
        {
            gamma_shift = GammaShift,
            gamma_size  = 1 << gamma_shift,
            gamma_mask  = gamma_size - 1
        };

        enum hi_res_scale_e
        {
            hi_res_shift = HiResShift,
            hi_res_size  = 1 << hi_res_shift,
            hi_res_mask  = hi_res_size - 1
        };

        ~gamma_lut()
        {
            pod_allocator<LoResT>::deallocate(m_inv_gamma, hi_res_size);
            pod_allocator<HiResT>::deallocate(m_dir_gamma, gamma_size);
        }

        gamma_lut() : 
            m_gamma(1.0), 
            m_dir_gamma(pod_allocator<HiResT>::allocate(gamma_size)),
            m_inv_gamma(pod_allocator<LoResT>::allocate(hi_res_size))
        {
            unsigned i;
            for(i = 0; i < gamma_size; i++)
            {
                m_dir_gamma[i] = HiResT(i << (hi_res_shift - gamma_shift));
            }

            for(i = 0; i < hi_res_size; i++)
            {
                m_inv_gamma[i] = LoResT(i >> (hi_res_shift - gamma_shift));
            }
        }

        gamma_lut(double g) :
            m_gamma(1.0), 
            m_dir_gamma(pod_allocator<HiResT>::allocate(gamma_size)),
            m_inv_gamma(pod_allocator<LoResT>::allocate(hi_res_size))
        {
            gamma(g);
        }

        void gamma(double g) 
        {
            m_gamma = g;

            unsigned i;
            for(i = 0; i < gamma_size; i++)
            {
                m_dir_gamma[i] = (HiResT)
                    uround(pow(i / double(gamma_mask), m_gamma) * double(hi_res_mask));
            }

            double inv_g = 1.0 / g;
            for(i = 0; i < hi_res_size; i++)
            {
                m_inv_gamma[i] = (LoResT)
                    uround(pow(i / double(hi_res_mask), inv_g) * double(gamma_mask));
            }
        }

        double gamma() const
        {
            return m_gamma;
        }

        HiResT dir(LoResT v) const 
        { 
            return m_dir_gamma[unsigned(v)]; 
        }

        LoResT inv(HiResT v) const 
        { 
            return m_inv_gamma[unsigned(v)];
        }

    private:
        gamma_lut(const self_type&);
        const self_type& operator = (const self_type&);

        double m_gamma;
        HiResT* m_dir_gamma;
        LoResT* m_inv_gamma;
    };
}

#endif
