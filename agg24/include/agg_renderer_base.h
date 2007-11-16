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
// class renderer_base
//
//----------------------------------------------------------------------------

#ifndef AGG_RENDERER_BASE_INCLUDED
#define AGG_RENDERER_BASE_INCLUDED

#include "agg_basics.h"
#include "agg_rendering_buffer.h"

namespace agg
{

    //-----------------------------------------------------------renderer_base
    template<class PixelFormat> class renderer_base
    {
    public:
        typedef PixelFormat pixfmt_type;
        typedef typename pixfmt_type::color_type color_type;
        typedef typename pixfmt_type::row_data row_data;

        //--------------------------------------------------------------------
        renderer_base() : m_ren(0), m_clip_box(1, 1, 0, 0) {}
        explicit renderer_base(pixfmt_type& ren) :
            m_ren(&ren),
            m_clip_box(0, 0, ren.width() - 1, ren.height() - 1)
        {}
        void attach(pixfmt_type& ren)
        {
            m_ren = &ren;
            m_clip_box = rect_i(0, 0, ren.width() - 1, ren.height() - 1);
        }

        //--------------------------------------------------------------------
        const pixfmt_type& ren() const { return *m_ren;  }
        pixfmt_type& ren() { return *m_ren;  }
          
        //--------------------------------------------------------------------
        unsigned width()  const { return m_ren->width();  }
        unsigned height() const { return m_ren->height(); }

        //--------------------------------------------------------------------
        bool clip_box(int x1, int y1, int x2, int y2)
        {
            rect_i cb(x1, y1, x2, y2);
            cb.normalize();
            if(cb.clip(rect_i(0, 0, width() - 1, height() - 1)))
            {
                m_clip_box = cb;
                return true;
            }
            m_clip_box.x1 = 1;
            m_clip_box.y1 = 1;
            m_clip_box.x2 = 0;
            m_clip_box.y2 = 0;
            return false;
        }

        //--------------------------------------------------------------------
        void reset_clipping(bool visibility)
        {
            if(visibility)
            {
                m_clip_box.x1 = 0;
                m_clip_box.y1 = 0;
                m_clip_box.x2 = width() - 1;
                m_clip_box.y2 = height() - 1;
            }
            else
            {
                m_clip_box.x1 = 1;
                m_clip_box.y1 = 1;
                m_clip_box.x2 = 0;
                m_clip_box.y2 = 0;
            }
        }

        //--------------------------------------------------------------------
        void clip_box_naked(int x1, int y1, int x2, int y2)
        {
            m_clip_box.x1 = x1;
            m_clip_box.y1 = y1;
            m_clip_box.x2 = x2;
            m_clip_box.y2 = y2;
        }

        //--------------------------------------------------------------------
        bool inbox(int x, int y) const
        {
            return x >= m_clip_box.x1 && y >= m_clip_box.y1 &&
                   x <= m_clip_box.x2 && y <= m_clip_box.y2;
        }

        //--------------------------------------------------------------------
        const rect_i& clip_box() const { return m_clip_box;    }
        int           xmin()     const { return m_clip_box.x1; }
        int           ymin()     const { return m_clip_box.y1; }
        int           xmax()     const { return m_clip_box.x2; }
        int           ymax()     const { return m_clip_box.y2; }

        //--------------------------------------------------------------------
        const rect_i& bounding_clip_box() const { return m_clip_box;    }
        int           bounding_xmin()     const { return m_clip_box.x1; }
        int           bounding_ymin()     const { return m_clip_box.y1; }
        int           bounding_xmax()     const { return m_clip_box.x2; }
        int           bounding_ymax()     const { return m_clip_box.y2; }

        //--------------------------------------------------------------------
        void clear(const color_type& c)
        {
            unsigned y;
            if(width())
            {
                for(y = 0; y < height(); y++)
                {
                    m_ren->copy_hline(0, y, width(), c);
                }
            }
        }
          

        //--------------------------------------------------------------------
        void copy_pixel(int x, int y, const color_type& c)
        {
            if(inbox(x, y))
            {
                m_ren->copy_pixel(x, y, c);
            }
        }

        //--------------------------------------------------------------------
        void blend_pixel(int x, int y, const color_type& c, cover_type cover)
        {
            if(inbox(x, y))
            {
                m_ren->blend_pixel(x, y, c, cover);
            }
        }

        //--------------------------------------------------------------------
        color_type pixel(int x, int y) const
        {
            return inbox(x, y) ? 
                   m_ren->pixel(x, y) :
                   color_type::no_color();
        }

        //--------------------------------------------------------------------
        void copy_hline(int x1, int y, int x2, const color_type& c)
        {
            if(x1 > x2) { int t = x2; x2 = x1; x1 = t; }
            if(y  > ymax()) return;
            if(y  < ymin()) return;
            if(x1 > xmax()) return;
            if(x2 < xmin()) return;

            if(x1 < xmin()) x1 = xmin();
            if(x2 > xmax()) x2 = xmax();

            m_ren->copy_hline(x1, y, x2 - x1 + 1, c);
        }

        //--------------------------------------------------------------------
        void copy_vline(int x, int y1, int y2, const color_type& c)
        {
            if(y1 > y2) { int t = y2; y2 = y1; y1 = t; }
            if(x  > xmax()) return;
            if(x  < xmin()) return;
            if(y1 > ymax()) return;
            if(y2 < ymin()) return;

            if(y1 < ymin()) y1 = ymin();
            if(y2 > ymax()) y2 = ymax();

            m_ren->copy_vline(x, y1, y2 - y1 + 1, c);
        }

        //--------------------------------------------------------------------
        void blend_hline(int x1, int y, int x2, 
                         const color_type& c, cover_type cover)
        {
            if(x1 > x2) { int t = x2; x2 = x1; x1 = t; }
            if(y  > ymax()) return;
            if(y  < ymin()) return;
            if(x1 > xmax()) return;
            if(x2 < xmin()) return;

            if(x1 < xmin()) x1 = xmin();
            if(x2 > xmax()) x2 = xmax();

            m_ren->blend_hline(x1, y, x2 - x1 + 1, c, cover);
        }

        //--------------------------------------------------------------------
        void blend_vline(int x, int y1, int y2, 
                         const color_type& c, cover_type cover)
        {
            if(y1 > y2) { int t = y2; y2 = y1; y1 = t; }
            if(x  > xmax()) return;
            if(x  < xmin()) return;
            if(y1 > ymax()) return;
            if(y2 < ymin()) return;

            if(y1 < ymin()) y1 = ymin();
            if(y2 > ymax()) y2 = ymax();

            m_ren->blend_vline(x, y1, y2 - y1 + 1, c, cover);
        }


        //--------------------------------------------------------------------
        void copy_bar(int x1, int y1, int x2, int y2, const color_type& c)
        {
            rect_i rc(x1, y1, x2, y2);
            rc.normalize();
            if(rc.clip(clip_box()))
            {
                int y;
                for(y = rc.y1; y <= rc.y2; y++)
                {
                    m_ren->copy_hline(rc.x1, y, unsigned(rc.x2 - rc.x1 + 1), c);
                }
            }
        }

        //--------------------------------------------------------------------
        void blend_bar(int x1, int y1, int x2, int y2, 
                       const color_type& c, cover_type cover)
        {
            rect_i rc(x1, y1, x2, y2);
            rc.normalize();
            if(rc.clip(clip_box()))
            {
                int y;
                for(y = rc.y1; y <= rc.y2; y++)
                {
                    m_ren->blend_hline(rc.x1,
                                       y,
                                       unsigned(rc.x2 - rc.x1 + 1), 
                                       c, 
                                       cover);
                }
            }
        }

        //--------------------------------------------------------------------
        void blend_solid_hspan(int x, int y, int len, 
                               const color_type& c, 
                               const cover_type* covers)
        {
            if(y > ymax()) return;
            if(y < ymin()) return;

            if(x < xmin())
            {
                len -= xmin() - x;
                if(len <= 0) return;
                covers += xmin() - x;
                x = xmin();
            }
            if(x + len > xmax())
            {
                len = xmax() - x + 1;
                if(len <= 0) return;
            }
            m_ren->blend_solid_hspan(x, y, len, c, covers);
        }

        //--------------------------------------------------------------------
        void blend_solid_vspan(int x, int y, int len, 
                               const color_type& c, 
                               const cover_type* covers)
        {
            if(x > xmax()) return;
            if(x < xmin()) return;

            if(y < ymin())
            {
                len -= ymin() - y;
                if(len <= 0) return;
                covers += ymin() - y;
                y = ymin();
            }
            if(y + len > ymax())
            {
                len = ymax() - y + 1;
                if(len <= 0) return;
            }
            m_ren->blend_solid_vspan(x, y, len, c, covers);
        }


        //--------------------------------------------------------------------
        void copy_color_hspan(int x, int y, int len, const color_type* colors)
        {
            if(y > ymax()) return;
            if(y < ymin()) return;

            if(x < xmin())
            {
                int d = xmin() - x;
                len -= d;
                if(len <= 0) return;
                colors += d;
                x = xmin();
            }
            if(x + len > xmax())
            {
                len = xmax() - x + 1;
                if(len <= 0) return;
            }
            m_ren->copy_color_hspan(x, y, len, colors);
        }


        //--------------------------------------------------------------------
        void copy_color_vspan(int x, int y, int len, const color_type* colors)
        {
            if(x > xmax()) return;
            if(x < xmin()) return;

            if(y < ymin())
            {
                int d = ymin() - y;
                len -= d;
                if(len <= 0) return;
                colors += d;
                y = ymin();
            }
            if(y + len > ymax())
            {
                len = ymax() - y + 1;
                if(len <= 0) return;
            }
            m_ren->copy_color_vspan(x, y, len, colors);
        }


        //--------------------------------------------------------------------
        void blend_color_hspan(int x, int y, int len, 
                               const color_type* colors, 
                               const cover_type* covers,
                               cover_type cover = agg::cover_full)
        {
            if(y > ymax()) return;
            if(y < ymin()) return;

            if(x < xmin())
            {
                int d = xmin() - x;
                len -= d;
                if(len <= 0) return;
                if(covers) covers += d;
                colors += d;
                x = xmin();
            }
            if(x + len > xmax())
            {
                len = xmax() - x + 1;
                if(len <= 0) return;
            }
            m_ren->blend_color_hspan(x, y, len, colors, covers, cover);
        }

        //--------------------------------------------------------------------
        void blend_color_vspan(int x, int y, int len, 
                               const color_type* colors, 
                               const cover_type* covers,
                               cover_type cover = agg::cover_full)
        {
            if(x > xmax()) return;
            if(x < xmin()) return;

            if(y < ymin())
            {
                int d = ymin() - y;
                len -= d;
                if(len <= 0) return;
                if(covers) covers += d;
                colors += d;
                y = ymin();
            }
            if(y + len > ymax())
            {
                len = ymax() - y + 1;
                if(len <= 0) return;
            }
            m_ren->blend_color_vspan(x, y, len, colors, covers, cover);
        }

        //--------------------------------------------------------------------
        rect_i clip_rect_area(rect_i& dst, rect_i& src, int wsrc, int hsrc) const
        {
            rect_i rc(0,0,0,0);
            rect_i cb = clip_box();
            ++cb.x2;
            ++cb.y2;

            if(src.x1 < 0)
            {
                dst.x1 -= src.x1;
                src.x1 = 0;
            }
            if(src.y1 < 0)
            {
                dst.y1 -= src.y1;
                src.y1 = 0;
            }

            if(src.x2 > wsrc) src.x2 = wsrc;
            if(src.y2 > hsrc) src.y2 = hsrc;

            if(dst.x1 < cb.x1)
            {
                src.x1 += cb.x1 - dst.x1;
                dst.x1 = cb.x1;
            }
            if(dst.y1 < cb.y1)
            {
                src.y1 += cb.y1 - dst.y1;
                dst.y1 = cb.y1;
            }

            if(dst.x2 > cb.x2) dst.x2 = cb.x2;
            if(dst.y2 > cb.y2) dst.y2 = cb.y2;

            rc.x2 = dst.x2 - dst.x1;
            rc.y2 = dst.y2 - dst.y1;

            if(rc.x2 > src.x2 - src.x1) rc.x2 = src.x2 - src.x1;
            if(rc.y2 > src.y2 - src.y1) rc.y2 = src.y2 - src.y1;
            return rc;
        }

        //--------------------------------------------------------------------
        template<class RenBuf>
        void copy_from(const RenBuf& src, 
                       const rect_i* rect_src_ptr = 0, 
                       int dx = 0, 
                       int dy = 0)
        {
            rect_i rsrc(0, 0, src.width(), src.height());
            if(rect_src_ptr)
            {
                rsrc.x1 = rect_src_ptr->x1; 
                rsrc.y1 = rect_src_ptr->y1;
                rsrc.x2 = rect_src_ptr->x2 + 1;
                rsrc.y2 = rect_src_ptr->y2 + 1;
            }

            // Version with xdst, ydst (absolute positioning)
            //rect_i rdst(xdst, ydst, xdst + rsrc.x2 - rsrc.x1, ydst + rsrc.y2 - rsrc.y1);

            // Version with dx, dy (relative positioning)
            rect_i rdst(rsrc.x1 + dx, rsrc.y1 + dy, rsrc.x2 + dx, rsrc.y2 + dy);

            rect_i rc = clip_rect_area(rdst, rsrc, src.width(), src.height());

            if(rc.x2 > 0)
            {
                int incy = 1;
                if(rdst.y1 > rsrc.y1)
                {
                    rsrc.y1 += rc.y2 - 1;
                    rdst.y1 += rc.y2 - 1;
                    incy = -1;
                }
                while(rc.y2 > 0)
                {
                    m_ren->copy_from(src, 
                                     rdst.x1, rdst.y1,
                                     rsrc.x1, rsrc.y1,
                                     rc.x2);
                    rdst.y1 += incy;
                    rsrc.y1 += incy;
                    --rc.y2;
                }
            }
        }

        //--------------------------------------------------------------------
        template<class SrcPixelFormatRenderer>
        void blend_from(const SrcPixelFormatRenderer& src, 
                        const rect_i* rect_src_ptr = 0, 
                        int dx = 0, 
                        int dy = 0,
                        cover_type cover = agg::cover_full)
        {
            rect_i rsrc(0, 0, src.width(), src.height());
            if(rect_src_ptr)
            {
                rsrc.x1 = rect_src_ptr->x1; 
                rsrc.y1 = rect_src_ptr->y1;
                rsrc.x2 = rect_src_ptr->x2 + 1;
                rsrc.y2 = rect_src_ptr->y2 + 1;
            }

            // Version with xdst, ydst (absolute positioning)
            //rect_i rdst(xdst, ydst, xdst + rsrc.x2 - rsrc.x1, ydst + rsrc.y2 - rsrc.y1);

            // Version with dx, dy (relative positioning)
            rect_i rdst(rsrc.x1 + dx, rsrc.y1 + dy, rsrc.x2 + dx, rsrc.y2 + dy);
            rect_i rc = clip_rect_area(rdst, rsrc, src.width(), src.height());

            if(rc.x2 > 0)
            {
                int incy = 1;
                if(rdst.y1 > rsrc.y1)
                {
                    rsrc.y1 += rc.y2 - 1;
                    rdst.y1 += rc.y2 - 1;
                    incy = -1;
                }
                while(rc.y2 > 0)
                {
                    typename SrcPixelFormatRenderer::row_data rw = src.row(rsrc.y1);
                    if(rw.ptr)
                    {
                        int x1src = rsrc.x1;
                        int x1dst = rdst.x1;
                        int len   = rc.x2;
                        if(rw.x1 > x1src)
                        {
                            x1dst += rw.x1 - x1src;
                            len   -= rw.x1 - x1src;
                            x1src  = rw.x1;
                        }
                        if(len > 0)
                        {
                            if(x1src + len-1 > rw.x2)
                            {
                                len -= x1src + len - rw.x2 - 1;
                            }
                            if(len > 0)
                            {
                                m_ren->blend_from(src,
                                                  x1dst, rdst.y1,
                                                  x1src, rsrc.y1,
                                                  len,
                                                  cover);
                            }
                        }
                    }
                    rdst.y1 += incy;
                    rsrc.y1 += incy;
                    --rc.y2;
                }
            }
        }

        //--------------------------------------------------------------------
        template<class SrcPixelFormatRenderer>
        void blend_from_color(const SrcPixelFormatRenderer& src, 
                              const color_type& color,
                              const rect_i* rect_src_ptr = 0, 
                              int dx = 0, 
                              int dy = 0,
                              cover_type cover = agg::cover_full)
        {
            rect_i rsrc(0, 0, src.width(), src.height());
            if(rect_src_ptr)
            {
                rsrc.x1 = rect_src_ptr->x1; 
                rsrc.y1 = rect_src_ptr->y1;
                rsrc.x2 = rect_src_ptr->x2 + 1;
                rsrc.y2 = rect_src_ptr->y2 + 1;
            }

            // Version with xdst, ydst (absolute positioning)
            //rect_i rdst(xdst, ydst, xdst + rsrc.x2 - rsrc.x1, ydst + rsrc.y2 - rsrc.y1);

            // Version with dx, dy (relative positioning)
            rect_i rdst(rsrc.x1 + dx, rsrc.y1 + dy, rsrc.x2 + dx, rsrc.y2 + dy);
            rect_i rc = clip_rect_area(rdst, rsrc, src.width(), src.height());

            if(rc.x2 > 0)
            {
                int incy = 1;
                if(rdst.y1 > rsrc.y1)
                {
                    rsrc.y1 += rc.y2 - 1;
                    rdst.y1 += rc.y2 - 1;
                    incy = -1;
                }
                while(rc.y2 > 0)
                {
                    typename SrcPixelFormatRenderer::row_data rw = src.row(rsrc.y1);
                    if(rw.ptr)
                    {
                        int x1src = rsrc.x1;
                        int x1dst = rdst.x1;
                        int len   = rc.x2;
                        if(rw.x1 > x1src)
                        {
                            x1dst += rw.x1 - x1src;
                            len   -= rw.x1 - x1src;
                            x1src  = rw.x1;
                        }
                        if(len > 0)
                        {
                            if(x1src + len-1 > rw.x2)
                            {
                                len -= x1src + len - rw.x2 - 1;
                            }
                            if(len > 0)
                            {
                                m_ren->blend_from_color(src,
                                                        color,
                                                        x1dst, rdst.y1,
                                                        x1src, rsrc.y1,
                                                        len,
                                                        cover);
                            }
                        }
                    }
                    rdst.y1 += incy;
                    rsrc.y1 += incy;
                    --rc.y2;
                }
            }
        }

        //--------------------------------------------------------------------
        template<class SrcPixelFormatRenderer>
        void blend_from_lut(const SrcPixelFormatRenderer& src, 
                            const color_type* color_lut,
                            const rect_i* rect_src_ptr = 0, 
                            int dx = 0, 
                            int dy = 0,
                            cover_type cover = agg::cover_full)
        {
            rect_i rsrc(0, 0, src.width(), src.height());
            if(rect_src_ptr)
            {
                rsrc.x1 = rect_src_ptr->x1; 
                rsrc.y1 = rect_src_ptr->y1;
                rsrc.x2 = rect_src_ptr->x2 + 1;
                rsrc.y2 = rect_src_ptr->y2 + 1;
            }

            // Version with xdst, ydst (absolute positioning)
            //rect_i rdst(xdst, ydst, xdst + rsrc.x2 - rsrc.x1, ydst + rsrc.y2 - rsrc.y1);

            // Version with dx, dy (relative positioning)
            rect_i rdst(rsrc.x1 + dx, rsrc.y1 + dy, rsrc.x2 + dx, rsrc.y2 + dy);
            rect_i rc = clip_rect_area(rdst, rsrc, src.width(), src.height());

            if(rc.x2 > 0)
            {
                int incy = 1;
                if(rdst.y1 > rsrc.y1)
                {
                    rsrc.y1 += rc.y2 - 1;
                    rdst.y1 += rc.y2 - 1;
                    incy = -1;
                }
                while(rc.y2 > 0)
                {
                    typename SrcPixelFormatRenderer::row_data rw = src.row(rsrc.y1);
                    if(rw.ptr)
                    {
                        int x1src = rsrc.x1;
                        int x1dst = rdst.x1;
                        int len   = rc.x2;
                        if(rw.x1 > x1src)
                        {
                            x1dst += rw.x1 - x1src;
                            len   -= rw.x1 - x1src;
                            x1src  = rw.x1;
                        }
                        if(len > 0)
                        {
                            if(x1src + len-1 > rw.x2)
                            {
                                len -= x1src + len - rw.x2 - 1;
                            }
                            if(len > 0)
                            {
                                m_ren->blend_from_lut(src,
                                                      color_lut,
                                                      x1dst, rdst.y1,
                                                      x1src, rsrc.y1,
                                                      len,
                                                      cover);
                            }
                        }
                    }
                    rdst.y1 += incy;
                    rsrc.y1 += incy;
                    --rc.y2;
                }
            }
        }

    private:
        pixfmt_type* m_ren;
        rect_i       m_clip_box;
    };


}

#endif
