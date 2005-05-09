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

#ifndef AGG_SPAN_INTERPOLATOR_LINEAR_INCLUDED
#define AGG_SPAN_INTERPOLATOR_LINEAR_INCLUDED

#include "agg_basics.h"
#include "agg_dda_line.h"
#include "agg_trans_affine.h"

namespace agg
{

    //================================================span_interpolator_linear
    template<class Transformer = trans_affine, unsigned SubpixelShift = 8> 
    class span_interpolator_linear
    {
    public:
        typedef Transformer trans_type;

        enum
        {
            subpixel_shift = SubpixelShift,
            subpixel_size  = 1 << subpixel_shift
        };

        //--------------------------------------------------------------------
        span_interpolator_linear() {}
        span_interpolator_linear(const trans_type& trans) : m_trans(&trans) {}
        span_interpolator_linear(const trans_type& trans,
                                 double x, double y, unsigned len) :
            m_trans(&trans)
        {
            begin(x, y, len);
        }

        //----------------------------------------------------------------
        const trans_type& transformer() const { return *m_trans; }
        void transformer(const trans_type& trans) { m_trans = &trans; }

        //----------------------------------------------------------------
        void begin(double x, double y, unsigned len)
        {
            double tx;
            double ty;

            tx = x;
            ty = y;
            m_trans->transform(&tx, &ty);
            int x1 = int(tx * subpixel_size);
            int y1 = int(ty * subpixel_size);

            tx = x + len;
            ty = y;
            m_trans->transform(&tx, &ty);
            int x2 = int(tx * subpixel_size);
            int y2 = int(ty * subpixel_size);

            m_li_x = dda2_line_interpolator(x1, x2, len);
            m_li_y = dda2_line_interpolator(y1, y2, len);
        }

        //----------------------------------------------------------------
        void resynchronize(double xe, double ye, unsigned len)
        {
            m_trans->transform(&xe, &ye);
            m_li_x = dda2_line_interpolator(m_li_x.y(), int(xe * subpixel_size), len);
            m_li_y = dda2_line_interpolator(m_li_y.y(), int(ye * subpixel_size), len);
        }
    
        //----------------------------------------------------------------
        void operator++()
        {
            ++m_li_x;
            ++m_li_y;
        }

        //----------------------------------------------------------------
        void coordinates(int* x, int* y) const
        {
            *x = m_li_x.y();
            *y = m_li_y.y();
        }

    private:
        const trans_type* m_trans;
        dda2_line_interpolator m_li_x;
        dda2_line_interpolator m_li_y;
    };






    //=====================================span_interpolator_linear_subdiv
    template<class Transformer = trans_affine, unsigned SubpixelShift = 8> 
    class span_interpolator_linear_subdiv
    {
    public:
        typedef Transformer trans_type;

        enum
        {
            subpixel_shift = SubpixelShift,
            subpixel_size  = 1 << subpixel_shift
        };


        //----------------------------------------------------------------
        span_interpolator_linear_subdiv() :
            m_subdiv_shift(4),
            m_subdiv_size(1 << m_subdiv_shift),
            m_subdiv_mask(m_subdiv_size - 1) {}

        span_interpolator_linear_subdiv(const trans_type& trans, 
                                        unsigned subdiv_shift = 4) : 
            m_subdiv_shift(subdiv_shift),
            m_subdiv_size(1 << m_subdiv_shift),
            m_subdiv_mask(m_subdiv_size - 1),
            m_trans(&trans) {}

        span_interpolator_linear_subdiv(const trans_type& trans,
                                        double x, double y, unsigned len,
                                        unsigned subdiv_shift = 4) :
            m_subdiv_shift(subdiv_shift),
            m_subdiv_size(1 << m_subdiv_shift),
            m_subdiv_mask(m_subdiv_size - 1),
            m_trans(&trans)
        {
            begin(x, y, len);
        }

        //----------------------------------------------------------------
        const trans_type& transformer() const { return *m_trans; }
        void transformer(const trans_type& trans) { m_trans = &trans; }

        //----------------------------------------------------------------
        unsigned subdiv_shift() const { return m_subdiv_shift; }
        void subdiv_shift(unsigned shift) 
        {
            m_subdiv_shift = shift;
            m_subdiv_size = 1 << m_subdiv_shift;
            m_subdiv_mask = m_subdiv_size - 1;
        }

        //----------------------------------------------------------------
        void begin(double x, double y, unsigned len)
        {
            double tx;
            double ty;
            m_pos   = 1;
            m_src_x = int(x * subpixel_size) + subpixel_size;
            m_src_y = y;
            m_len   = len;

            if(len > m_subdiv_size) len = m_subdiv_size;
            tx = x;
            ty = y;
            m_trans->transform(&tx, &ty);
            int x1 = int(tx * subpixel_size);
            int y1 = int(ty * subpixel_size);

            tx = x + len;
            ty = y;
            m_trans->transform(&tx, &ty);

            m_li_x = dda2_line_interpolator(x1, int(tx * subpixel_size), len);
            m_li_y = dda2_line_interpolator(y1, int(ty * subpixel_size), len);
        }

        //----------------------------------------------------------------
        void operator++()
        {
            ++m_li_x;
            ++m_li_y;
            if(m_pos >= m_subdiv_size)
            {
                unsigned len = m_len;
                if(len > m_subdiv_size) len = m_subdiv_size;
                double tx = double(m_src_x) / double(subpixel_size) + len;
                double ty = m_src_y;
                m_trans->transform(&tx, &ty);
                m_li_x = dda2_line_interpolator(m_li_x.y(), int(tx * subpixel_size), len);
                m_li_y = dda2_line_interpolator(m_li_y.y(), int(ty * subpixel_size), len);
                m_pos = 0;
            }
            m_src_x += subpixel_size;
            ++m_pos;
            --m_len;
        }

        //----------------------------------------------------------------
        void coordinates(int* x, int* y) const
        {
            *x = m_li_x.y();
            *y = m_li_y.y();
        }

    private:
        unsigned m_subdiv_shift;
        unsigned m_subdiv_size;
        unsigned m_subdiv_mask;
        const trans_type* m_trans;
        dda2_line_interpolator m_li_x;
        dda2_line_interpolator m_li_y;
        int      m_src_x;
        double   m_src_y;
        unsigned m_pos;
        unsigned m_len;
    };


}



#endif


