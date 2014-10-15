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

#ifndef AGG_PIXFMT_AMASK_ADAPTOR_INCLUDED
#define AGG_PIXFMT_AMASK_ADAPTOR_INCLUDED


#include <string.h>
#include "agg_array.h"
#include "agg_rendering_buffer.h"


namespace agg
{
    //==================================================pixfmt_amask_adaptor
    template<class PixFmt, class AlphaMask> class pixfmt_amask_adaptor
    {
    public:
        typedef PixFmt pixfmt_type;
        typedef typename pixfmt_type::color_type color_type;
        typedef typename pixfmt_type::row_data row_data;
        typedef AlphaMask amask_type;
        typedef typename amask_type::cover_type cover_type;

    private:
        enum span_extra_tail_e { span_extra_tail = 256 };

        void realloc_span(unsigned len)
        {
            if(len > m_span.size())
            {
                m_span.resize(len + span_extra_tail);
            }
        }

        void init_span(unsigned len)
        {
            realloc_span(len);
            memset(&m_span[0], amask_type::cover_full, len * sizeof(cover_type));
        }

        void init_span(unsigned len, const cover_type* covers)
        {
            realloc_span(len);
            memcpy(&m_span[0], covers, len * sizeof(cover_type));
        }


    public:
        pixfmt_amask_adaptor(pixfmt_type& pixf, amask_type& mask) :
            m_pixf(&pixf), m_mask(&mask), m_span()
        {}

        void attach_pixfmt(pixfmt_type& pixf) { m_pixf = &pixf; }
        void attach_alpha_mask(amask_type& mask) { m_mask = &mask; }

        //--------------------------------------------------------------------
        template<class PixFmt2>
        bool attach_pixfmt(PixFmt2& pixf, int x1, int y1, int x2, int y2)
        {
            return m_pixf->attach(pixf, x1, y1, x2, y2);
        }

        //--------------------------------------------------------------------
        unsigned width()  const { return m_pixf->width();  }
        unsigned height() const { return m_pixf->height(); }

        //--------------------------------------------------------------------
        color_type pixel(int x, int y)
        {
            return m_pixf->pixel(x, y);
        }

        //--------------------------------------------------------------------
        void copy_pixel(int x, int y, const color_type& c)
        {
            m_pixf->blend_pixel(x, y, c, m_mask->pixel(x, y));
        }

        //--------------------------------------------------------------------
        void blend_pixel(int x, int y, const color_type& c, cover_type cover)
        {
            m_pixf->blend_pixel(x, y, c, m_mask->combine_pixel(x, y, cover));
        }

        //--------------------------------------------------------------------
        void copy_hline(int x, int y, 
                        unsigned len, 
                        const color_type& c)
        {
            realloc_span(len);
            m_mask->fill_hspan(x, y, &m_span[0], len);
            m_pixf->blend_solid_hspan(x, y, len, c, &m_span[0]);
        }

        //--------------------------------------------------------------------
        void blend_hline(int x, int y,
                         unsigned len, 
                         const color_type& c,
                         cover_type cover)
        {
            init_span(len);
            m_mask->combine_hspan(x, y, &m_span[0], len);
            m_pixf->blend_solid_hspan(x, y, len, c, &m_span[0]);
        }

        //--------------------------------------------------------------------
        void copy_vline(int x, int y,
                        unsigned len, 
                        const color_type& c)
        {
            realloc_span(len);
            m_mask->fill_vspan(x, y, &m_span[0], len);
            m_pixf->blend_solid_vspan(x, y, len, c, &m_span[0]);
        }

        //--------------------------------------------------------------------
        void blend_vline(int x, int y,
                         unsigned len, 
                         const color_type& c,
                         cover_type cover)
        {
            init_span(len);
            m_mask->combine_vspan(x, y, &m_span[0], len);
            m_pixf->blend_solid_vspan(x, y, len, c, &m_span[0]);
        }

        //--------------------------------------------------------------------
        void copy_from(const rendering_buffer& from, 
                       int xdst, int ydst,
                       int xsrc, int ysrc,
                       unsigned len)
        {
            m_pixf->copy_from(from, xdst, ydst, xsrc, ysrc, len);
        }


        //--------------------------------------------------------------------
        void blend_solid_hspan(int x, int y,
                               unsigned len, 
                               const color_type& c,
                               const cover_type* covers)
        {
            init_span(len, covers);
            m_mask->combine_hspan(x, y, &m_span[0], len);
            m_pixf->blend_solid_hspan(x, y, len, c, &m_span[0]);
        }


        //--------------------------------------------------------------------
        void blend_solid_vspan(int x, int y,
                               unsigned len, 
                               const color_type& c,
                               const cover_type* covers)
        {
            init_span(len, covers);
            m_mask->combine_vspan(x, y, &m_span[0], len);
            m_pixf->blend_solid_vspan(x, y, len, c, &m_span[0]);
        }


        //--------------------------------------------------------------------
        void copy_color_hspan(int x, int y, unsigned len, const color_type* colors)
        {
            realloc_span(len);
            m_mask->fill_hspan(x, y, &m_span[0], len);
            m_pixf->blend_color_hspan(x, y, len, colors, &m_span[0], cover_full);
        }

        //--------------------------------------------------------------------
        void copy_color_vspan(int x, int y, unsigned len, const color_type* colors)
        {
            realloc_span(len);
            m_mask->fill_vspan(x, y, &m_span[0], len);
            m_pixf->blend_color_vspan(x, y, len, colors, &m_span[0], cover_full);
        }

        //--------------------------------------------------------------------
        void blend_color_hspan(int x, int y,
                               unsigned len, 
                               const color_type* colors,
                               const cover_type* covers,
                               cover_type cover = cover_full)
        {
            if(covers) 
            {
                init_span(len, covers);
                m_mask->combine_hspan(x, y, &m_span[0], len);
            }
            else
            {
                realloc_span(len);
                m_mask->fill_hspan(x, y, &m_span[0], len);
            }
            m_pixf->blend_color_hspan(x, y, len, colors, &m_span[0], cover);
        }


        //--------------------------------------------------------------------
        void blend_color_vspan(int x, int y,
                               unsigned len, 
                               const color_type* colors,
                               const cover_type* covers,
                               cover_type cover = cover_full)
        {
            if(covers) 
            {
                init_span(len, covers);
                m_mask->combine_vspan(x, y, &m_span[0], len);
            }
            else
            {
                realloc_span(len);
                m_mask->fill_vspan(x, y, &m_span[0], len);
            }
            m_pixf->blend_color_vspan(x, y, len, colors, &m_span[0], cover);
        }

    private:
        pixfmt_type*          m_pixf;
        const amask_type*     m_mask;
        pod_array<cover_type> m_span;
    };

}

#endif

