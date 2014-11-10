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
// class renderer_mclip
//
//----------------------------------------------------------------------------

#ifndef AGG_RENDERER_MCLIP_INCLUDED
#define AGG_RENDERER_MCLIP_INCLUDED

#include "agg_basics.h"
#include "agg_array.h"
#include "agg_renderer_base.h"

namespace agg
{

    //----------------------------------------------------------renderer_mclip
    template<class PixelFormat> class renderer_mclip
    {
    public:
        typedef PixelFormat pixfmt_type;
        typedef typename pixfmt_type::color_type color_type;
        typedef typename pixfmt_type::row_data row_data;
        typedef renderer_base<pixfmt_type> base_ren_type;

        //--------------------------------------------------------------------
        explicit renderer_mclip(pixfmt_type& pixf) :
            m_ren(pixf),
            m_curr_cb(0),
            m_bounds(m_ren.xmin(), m_ren.ymin(), m_ren.xmax(), m_ren.ymax())
        {}
        void attach(pixfmt_type& pixf)
        {
            m_ren.attach(pixf);
            reset_clipping(true);
        }
          
        //--------------------------------------------------------------------
        const pixfmt_type& ren() const { return m_ren.ren();  }
        pixfmt_type& ren() { return m_ren.ren();  }

        //--------------------------------------------------------------------
        unsigned width()  const { return m_ren.width();  }
        unsigned height() const { return m_ren.height(); }

        //--------------------------------------------------------------------
        const rect_i& clip_box() const { return m_ren.clip_box(); }
        int           xmin()     const { return m_ren.xmin(); }
        int           ymin()     const { return m_ren.ymin(); }
        int           xmax()     const { return m_ren.xmax(); }
        int           ymax()     const { return m_ren.ymax(); }

        //--------------------------------------------------------------------
        const rect_i& bounding_clip_box() const { return m_bounds;    }
        int           bounding_xmin()     const { return m_bounds.x1; }
        int           bounding_ymin()     const { return m_bounds.y1; }
        int           bounding_xmax()     const { return m_bounds.x2; }
        int           bounding_ymax()     const { return m_bounds.y2; }

        //--------------------------------------------------------------------
        void first_clip_box() 
        {
            m_curr_cb = 0;
            if(m_clip.size())
            {
                const rect_i& cb = m_clip[0];
                m_ren.clip_box_naked(cb.x1, cb.y1, cb.x2, cb.y2);
            }
        }

        //--------------------------------------------------------------------
        bool next_clip_box() 
        { 
            if(++m_curr_cb < m_clip.size())
            {
                const rect_i& cb = m_clip[m_curr_cb];
                m_ren.clip_box_naked(cb.x1, cb.y1, cb.x2, cb.y2);
                return true;
            }
            return false; 
        }

        //--------------------------------------------------------------------
        void reset_clipping(bool visibility)
        {
            m_ren.reset_clipping(visibility);
            m_clip.remove_all();
            m_curr_cb = 0;
            m_bounds = m_ren.clip_box();
        }
        
        //--------------------------------------------------------------------
        void add_clip_box(int x1, int y1, int x2, int y2)
        {
            rect_i cb(x1, y1, x2, y2); 
            cb.normalize();
            if(cb.clip(rect_i(0, 0, width() - 1, height() - 1)))
            {
                m_clip.add(cb);
                if(cb.x1 < m_bounds.x1) m_bounds.x1 = cb.x1;
                if(cb.y1 < m_bounds.y1) m_bounds.y1 = cb.y1;
                if(cb.x2 > m_bounds.x2) m_bounds.x2 = cb.x2;
                if(cb.y2 > m_bounds.y2) m_bounds.y2 = cb.y2;
            }
        }

        //--------------------------------------------------------------------
        void clear(const color_type& c)
        {
            m_ren.clear(c);
        }
          
        //--------------------------------------------------------------------
        void copy_pixel(int x, int y, const color_type& c)
        {
            first_clip_box();
            do
            {
                if(m_ren.inbox(x, y))
                {
                    m_ren.ren().copy_pixel(x, y, c);
                    break;
                }
            }
            while(next_clip_box());
        }

        //--------------------------------------------------------------------
        void blend_pixel(int x, int y, const color_type& c, cover_type cover)
        {
            first_clip_box();
            do
            {
                if(m_ren.inbox(x, y))
                {
                    m_ren.ren().blend_pixel(x, y, c, cover);
                    break;
                }
            }
            while(next_clip_box());
        }

        //--------------------------------------------------------------------
        color_type pixel(int x, int y) const
        {
            first_clip_box();
            do
            {
                if(m_ren.inbox(x, y))
                {
                    return m_ren.ren().pixel(x, y);
                }
            }
            while(next_clip_box());
            return color_type::no_color();
        }

        //--------------------------------------------------------------------
        void copy_hline(int x1, int y, int x2, const color_type& c)
        {
            first_clip_box();
            do
            {
                m_ren.copy_hline(x1, y, x2, c);
            }
            while(next_clip_box());
        }

        //--------------------------------------------------------------------
        void copy_vline(int x, int y1, int y2, const color_type& c)
        {
            first_clip_box();
            do
            {
                m_ren.copy_vline(x, y1, y2, c);
            }
            while(next_clip_box());
        }

        //--------------------------------------------------------------------
        void blend_hline(int x1, int y, int x2, 
                         const color_type& c, cover_type cover)
        {
            first_clip_box();
            do
            {
                m_ren.blend_hline(x1, y, x2, c, cover);
            }
            while(next_clip_box());
        }

        //--------------------------------------------------------------------
        void blend_vline(int x, int y1, int y2, 
                         const color_type& c, cover_type cover)
        {
            first_clip_box();
            do
            {
                m_ren.blend_vline(x, y1, y2, c, cover);
            }
            while(next_clip_box());
        }

        //--------------------------------------------------------------------
        void copy_bar(int x1, int y1, int x2, int y2, const color_type& c)
        {
            first_clip_box();
            do
            {
                m_ren.copy_bar(x1, y1, x2, y2, c);
            }
            while(next_clip_box());
        }

        //--------------------------------------------------------------------
        void blend_bar(int x1, int y1, int x2, int y2, 
                       const color_type& c, cover_type cover)
        {
            first_clip_box();
            do
            {
                m_ren.blend_bar(x1, y1, x2, y2, c, cover);
            }
            while(next_clip_box());
        }

        //--------------------------------------------------------------------
        void blend_solid_hspan(int x, int y, int len, 
                               const color_type& c, const cover_type* covers)
        {
            first_clip_box();
            do
            {
                m_ren.blend_solid_hspan(x, y, len, c, covers);
            }
            while(next_clip_box());
        }

        //--------------------------------------------------------------------
        void blend_solid_vspan(int x, int y, int len, 
                               const color_type& c, const cover_type* covers)
        {
            first_clip_box();
            do
            {
                m_ren.blend_solid_vspan(x, y, len, c, covers);
            }
            while(next_clip_box());
        }


        //--------------------------------------------------------------------
        void copy_color_hspan(int x, int y, int len, const color_type* colors)
        {
            first_clip_box();
            do
            {
                m_ren.copy_color_hspan(x, y, len, colors);
            }
            while(next_clip_box());
        }

        //--------------------------------------------------------------------
        void blend_color_hspan(int x, int y, int len, 
                               const color_type* colors, 
                               const cover_type* covers,
                               cover_type cover = cover_full)
        {
            first_clip_box();
            do
            {
                m_ren.blend_color_hspan(x, y, len, colors, covers, cover);
            }
            while(next_clip_box());
        }

        //--------------------------------------------------------------------
        void blend_color_vspan(int x, int y, int len, 
                               const color_type* colors, 
                               const cover_type* covers,
                               cover_type cover = cover_full)
        {
            first_clip_box();
            do
            {
                m_ren.blend_color_vspan(x, y, len, colors, covers, cover);
            }
            while(next_clip_box());
        }

        //--------------------------------------------------------------------
        void copy_from(const rendering_buffer& from, 
                       const rect_i* rc=0, 
                       int x_to=0, 
                       int y_to=0)
        {
            first_clip_box();
            do
            {
                m_ren.copy_from(from, rc, x_to, y_to);
            }
            while(next_clip_box());
        }

        //--------------------------------------------------------------------
        template<class SrcPixelFormatRenderer>
        void blend_from(const SrcPixelFormatRenderer& src, 
                        const rect_i* rect_src_ptr = 0, 
                        int dx = 0, 
                        int dy = 0,
                        cover_type cover = cover_full)
        {
            first_clip_box();
            do
            {
                m_ren.blend_from(src, rect_src_ptr, dx, dy, cover);
            }
            while(next_clip_box());
        }

        
    private:
        renderer_mclip(const renderer_mclip<PixelFormat>&);
        const renderer_mclip<PixelFormat>& 
            operator = (const renderer_mclip<PixelFormat>&);

        base_ren_type          m_ren;
        pod_bvector<rect_i, 4> m_clip;
        unsigned               m_curr_cb;
        rect_i                 m_bounds;
    };


}

#endif
