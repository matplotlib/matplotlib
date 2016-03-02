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
#ifndef AGG_RENDERER_OUTLINE_IMAGE_INCLUDED
#define AGG_RENDERER_OUTLINE_IMAGE_INCLUDED

#include "agg_array.h"
#include "agg_math.h"
#include "agg_line_aa_basics.h"
#include "agg_dda_line.h"
#include "agg_rendering_buffer.h"
#include "agg_clip_liang_barsky.h"


namespace agg
{
    //========================================================line_image_scale
    template<class Source> class line_image_scale
    {
    public:
        typedef typename Source::color_type color_type;

        line_image_scale(const Source& src, double height) :
            m_source(src), 
            m_height(height),
            m_scale(src.height() / height),
            m_scale_inv(height / src.height())
        {
        }

        double width()  const { return m_source.width(); }
        double height() const { return m_height; }

        color_type pixel(int x, int y) const 
        { 
            if (m_scale < 1.0)
            {
                // Interpolate between nearest source pixels.
                double src_y = (y + 0.5) * m_scale - 0.5;
                int h  = m_source.height() - 1;
                int y1 = ifloor(src_y);
                int y2 = y1 + 1;
                rgba pix1 = (y1 < 0) ? rgba::no_color() : m_source.pixel(x, y1);
                rgba pix2 = (y2 > h) ? rgba::no_color() : m_source.pixel(x, y2);
                return pix1.gradient(pix2, src_y - y1);
            }
            else
            {
                // Average source pixels between y and y+1.
                double src_y1 = (y + 0.5) * m_scale - 0.5;
                double src_y2 = src_y1 + m_scale;
                int h  = m_source.height() - 1;
                int y1 = ifloor(src_y1);
                int y2 = ifloor(src_y2);
                rgba c = rgba::no_color();
                if (y1 >= 0) c += rgba(m_source.pixel(x, y1)) *= y1 + 1 - src_y1;
                while (++y1 < y2)
                {
                    if (y1 <= h) c += m_source.pixel(x, y1);
                }
                if (y2 <= h) c += rgba(m_source.pixel(x, y2)) *= src_y2 - y2;
                return c *= m_scale_inv;
            }
        }

    private:
        line_image_scale(const line_image_scale<Source>&);
        const line_image_scale<Source>& operator = (const line_image_scale<Source>&);

        const Source& m_source;
        double        m_height;
        double        m_scale;
        double        m_scale_inv;
    };



    //======================================================line_image_pattern
    template<class Filter> class line_image_pattern
    {
    public:
        typedef Filter filter_type;
        typedef typename filter_type::color_type color_type;

        //--------------------------------------------------------------------
        line_image_pattern(Filter& filter) :
            m_filter(&filter),
            m_dilation(filter.dilation() + 1),
            m_dilation_hr(m_dilation << line_subpixel_shift),
            m_data(),
            m_width(0),
            m_height(0),
            m_width_hr(0),
            m_half_height_hr(0),
            m_offset_y_hr(0)
        {
        }

        // Create
        //--------------------------------------------------------------------
        template<class Source> 
        line_image_pattern(Filter& filter, const Source& src) :
            m_filter(&filter),
            m_dilation(filter.dilation() + 1),
            m_dilation_hr(m_dilation << line_subpixel_shift),
            m_data(),
            m_width(0),
            m_height(0),
            m_width_hr(0),
            m_half_height_hr(0),
            m_offset_y_hr(0)
        {
            create(src);
        }

        // Create
        //--------------------------------------------------------------------
        template<class Source> void create(const Source& src)
        {
            m_height = uceil(src.height());
            m_width  = uceil(src.width());
            m_width_hr = uround(src.width() * line_subpixel_scale);
            m_half_height_hr = uround(src.height() * line_subpixel_scale/2);
            m_offset_y_hr = m_dilation_hr + m_half_height_hr - line_subpixel_scale/2;
            m_half_height_hr += line_subpixel_scale/2;

            m_data.resize((m_width + m_dilation * 2) * (m_height + m_dilation * 2));

            m_buf.attach(&m_data[0], m_width  + m_dilation * 2, 
                                     m_height + m_dilation * 2, 
                                     m_width  + m_dilation * 2);
            unsigned x, y;
            color_type* d1;
            color_type* d2;
            for(y = 0; y < m_height; y++)
            {
                d1 = m_buf.row_ptr(y + m_dilation) + m_dilation;
                for(x = 0; x < m_width; x++)
                {
                    *d1++ = src.pixel(x, y);
                }
            }

            const color_type* s1;
            const color_type* s2;
            for(y = 0; y < m_dilation; y++)
            {
                //s1 = m_buf.row_ptr(m_height + m_dilation - 1) + m_dilation;
                //s2 = m_buf.row_ptr(m_dilation) + m_dilation;
                d1 = m_buf.row_ptr(m_dilation + m_height + y) + m_dilation;
                d2 = m_buf.row_ptr(m_dilation - y - 1) + m_dilation;
                for(x = 0; x < m_width; x++)
                {
                    //*d1++ = color_type(*s1++, 0);
                    //*d2++ = color_type(*s2++, 0);
                    *d1++ = color_type::no_color();
                    *d2++ = color_type::no_color();
                }
            }

            unsigned h = m_height + m_dilation * 2;
            for(y = 0; y < h; y++)
            {
                s1 = m_buf.row_ptr(y) + m_dilation;
                s2 = m_buf.row_ptr(y) + m_dilation + m_width;
                d1 = m_buf.row_ptr(y) + m_dilation + m_width;
                d2 = m_buf.row_ptr(y) + m_dilation;

                for(x = 0; x < m_dilation; x++)
                {
                    *d1++ = *s1++;
                    *--d2 = *--s2;
                }
            }
        }

        //--------------------------------------------------------------------
        int pattern_width() const { return m_width_hr; }
        int line_width()    const { return m_half_height_hr; }
        double width()      const { return m_height; }

        //--------------------------------------------------------------------
        void pixel(color_type* p, int x, int y) const
        {
            m_filter->pixel_high_res(m_buf.rows(), 
                                     p, 
                                     x % m_width_hr + m_dilation_hr,
                                     y + m_offset_y_hr);
        }

        //--------------------------------------------------------------------
        const filter_type& filter() const { return *m_filter; }

    private:
        line_image_pattern(const line_image_pattern<filter_type>&);
        const line_image_pattern<filter_type>& 
            operator = (const line_image_pattern<filter_type>&);

    protected:
        row_ptr_cache<color_type> m_buf;
        const filter_type*        m_filter;
        unsigned                  m_dilation;
        int                       m_dilation_hr;
        pod_array<color_type>     m_data;
        unsigned                  m_width;
        unsigned                  m_height;
        int                       m_width_hr;
        int                       m_half_height_hr;
        int                       m_offset_y_hr;
    };






    //=================================================line_image_pattern_pow2
    template<class Filter> class line_image_pattern_pow2 : 
    public line_image_pattern<Filter>
    {
    public:
        typedef Filter filter_type;
        typedef typename filter_type::color_type color_type;
        typedef line_image_pattern<Filter> base_type;

        //--------------------------------------------------------------------
        line_image_pattern_pow2(Filter& filter) :
            line_image_pattern<Filter>(filter), m_mask(line_subpixel_mask) {}

        //--------------------------------------------------------------------
        template<class Source> 
        line_image_pattern_pow2(Filter& filter, const Source& src) :
            line_image_pattern<Filter>(filter), m_mask(line_subpixel_mask)
        {
            create(src);
        }
            
        //--------------------------------------------------------------------
        template<class Source> void create(const Source& src)
        {
            line_image_pattern<Filter>::create(src);
            m_mask = 1;
            while(m_mask < base_type::m_width) 
            {
                m_mask <<= 1;
                m_mask |= 1;
            }
            m_mask <<= line_subpixel_shift - 1;
            m_mask |=  line_subpixel_mask;
            base_type::m_width_hr = m_mask + 1;
        }

        //--------------------------------------------------------------------
        void pixel(color_type* p, int x, int y) const
        {
            base_type::m_filter->pixel_high_res(
                    base_type::m_buf.rows(), 
                    p,
                    (x & m_mask) + base_type::m_dilation_hr,
                    y + base_type::m_offset_y_hr);
        }
    private:
        unsigned m_mask;
    };
    
    
    
    
    
    
    
    //===================================================distance_interpolator4
    class distance_interpolator4
    {
    public:
        //---------------------------------------------------------------------
        distance_interpolator4() {}
        distance_interpolator4(int x1,  int y1, int x2, int y2,
                               int sx,  int sy, int ex, int ey, 
                               int len, double scale, int x, int y) :
            m_dx(x2 - x1),
            m_dy(y2 - y1),
            m_dx_start(line_mr(sx) - line_mr(x1)),
            m_dy_start(line_mr(sy) - line_mr(y1)),
            m_dx_end(line_mr(ex) - line_mr(x2)),
            m_dy_end(line_mr(ey) - line_mr(y2)),

            m_dist(iround(double(x + line_subpixel_scale/2 - x2) * double(m_dy) - 
                          double(y + line_subpixel_scale/2 - y2) * double(m_dx))),

            m_dist_start((line_mr(x + line_subpixel_scale/2) - line_mr(sx)) * m_dy_start - 
                         (line_mr(y + line_subpixel_scale/2) - line_mr(sy)) * m_dx_start),

            m_dist_end((line_mr(x + line_subpixel_scale/2) - line_mr(ex)) * m_dy_end - 
                       (line_mr(y + line_subpixel_scale/2) - line_mr(ey)) * m_dx_end),
            m_len(uround(len / scale))
        {
            double d = len * scale;
            int dx = iround(((x2 - x1) << line_subpixel_shift) / d);
            int dy = iround(((y2 - y1) << line_subpixel_shift) / d);
            m_dx_pict   = -dy;
            m_dy_pict   =  dx;
            m_dist_pict =  ((x + line_subpixel_scale/2 - (x1 - dy)) * m_dy_pict - 
                            (y + line_subpixel_scale/2 - (y1 + dx)) * m_dx_pict) >> 
                           line_subpixel_shift;

            m_dx       <<= line_subpixel_shift;
            m_dy       <<= line_subpixel_shift;
            m_dx_start <<= line_mr_subpixel_shift;
            m_dy_start <<= line_mr_subpixel_shift;
            m_dx_end   <<= line_mr_subpixel_shift;
            m_dy_end   <<= line_mr_subpixel_shift;
        }

        //---------------------------------------------------------------------
        void inc_x() 
        { 
            m_dist += m_dy; 
            m_dist_start += m_dy_start; 
            m_dist_pict += m_dy_pict; 
            m_dist_end += m_dy_end; 
        }

        //---------------------------------------------------------------------
        void dec_x() 
        { 
            m_dist -= m_dy; 
            m_dist_start -= m_dy_start; 
            m_dist_pict -= m_dy_pict; 
            m_dist_end -= m_dy_end; 
        }

        //---------------------------------------------------------------------
        void inc_y() 
        { 
            m_dist -= m_dx; 
            m_dist_start -= m_dx_start; 
            m_dist_pict -= m_dx_pict; 
            m_dist_end -= m_dx_end; 
        }

        //---------------------------------------------------------------------
        void dec_y() 
        { 
            m_dist += m_dx; 
            m_dist_start += m_dx_start; 
            m_dist_pict += m_dx_pict; 
            m_dist_end += m_dx_end; 
        }

        //---------------------------------------------------------------------
        void inc_x(int dy)
        {
            m_dist       += m_dy; 
            m_dist_start += m_dy_start; 
            m_dist_pict  += m_dy_pict; 
            m_dist_end   += m_dy_end;
            if(dy > 0)
            {
                m_dist       -= m_dx; 
                m_dist_start -= m_dx_start; 
                m_dist_pict  -= m_dx_pict; 
                m_dist_end   -= m_dx_end;
            }
            if(dy < 0)
            {
                m_dist       += m_dx; 
                m_dist_start += m_dx_start; 
                m_dist_pict  += m_dx_pict; 
                m_dist_end   += m_dx_end;
            }
        }

        //---------------------------------------------------------------------
        void dec_x(int dy)
        {
            m_dist       -= m_dy; 
            m_dist_start -= m_dy_start; 
            m_dist_pict  -= m_dy_pict; 
            m_dist_end   -= m_dy_end;
            if(dy > 0)
            {
                m_dist       -= m_dx; 
                m_dist_start -= m_dx_start; 
                m_dist_pict  -= m_dx_pict; 
                m_dist_end   -= m_dx_end;
            }
            if(dy < 0)
            {
                m_dist       += m_dx; 
                m_dist_start += m_dx_start; 
                m_dist_pict  += m_dx_pict; 
                m_dist_end   += m_dx_end;
            }
        }

        //---------------------------------------------------------------------
        void inc_y(int dx)
        {
            m_dist       -= m_dx; 
            m_dist_start -= m_dx_start; 
            m_dist_pict  -= m_dx_pict; 
            m_dist_end   -= m_dx_end;
            if(dx > 0)
            {
                m_dist       += m_dy; 
                m_dist_start += m_dy_start; 
                m_dist_pict  += m_dy_pict; 
                m_dist_end   += m_dy_end;
            }
            if(dx < 0)
            {
                m_dist       -= m_dy; 
                m_dist_start -= m_dy_start; 
                m_dist_pict  -= m_dy_pict; 
                m_dist_end   -= m_dy_end;
            }
        }

        //---------------------------------------------------------------------
        void dec_y(int dx)
        {
            m_dist       += m_dx; 
            m_dist_start += m_dx_start; 
            m_dist_pict  += m_dx_pict; 
            m_dist_end   += m_dx_end;
            if(dx > 0)
            {
                m_dist       += m_dy; 
                m_dist_start += m_dy_start; 
                m_dist_pict  += m_dy_pict; 
                m_dist_end   += m_dy_end;
            }
            if(dx < 0)
            {
                m_dist       -= m_dy; 
                m_dist_start -= m_dy_start; 
                m_dist_pict  -= m_dy_pict; 
                m_dist_end   -= m_dy_end;
            }
        }

        //---------------------------------------------------------------------
        int dist()       const { return m_dist;       }
        int dist_start() const { return m_dist_start; }
        int dist_pict()  const { return m_dist_pict;  }
        int dist_end()   const { return m_dist_end;   }

        //---------------------------------------------------------------------
        int dx()       const { return m_dx;       }
        int dy()       const { return m_dy;       }
        int dx_start() const { return m_dx_start; }
        int dy_start() const { return m_dy_start; }
        int dx_pict()  const { return m_dx_pict;  }
        int dy_pict()  const { return m_dy_pict;  }
        int dx_end()   const { return m_dx_end;   }
        int dy_end()   const { return m_dy_end;   }
        int len()      const { return m_len;      }

    private:
        //---------------------------------------------------------------------
        int m_dx;
        int m_dy;
        int m_dx_start;
        int m_dy_start;
        int m_dx_pict;
        int m_dy_pict;
        int m_dx_end;
        int m_dy_end;

        int m_dist;
        int m_dist_start;
        int m_dist_pict;
        int m_dist_end;
        int m_len;
    };





    //==================================================line_interpolator_image
    template<class Renderer> class line_interpolator_image
    {
    public:
        typedef Renderer renderer_type;
        typedef typename Renderer::color_type color_type;

        //---------------------------------------------------------------------
        enum max_half_width_e
        { 
            max_half_width = 64
        };

        //---------------------------------------------------------------------
        line_interpolator_image(renderer_type& ren, const line_parameters& lp,
                                int sx, int sy, int ex, int ey, 
                                int pattern_start,
                                double scale_x) :
            m_lp(lp),
            m_li(lp.vertical ? line_dbl_hr(lp.x2 - lp.x1) :
                               line_dbl_hr(lp.y2 - lp.y1),
                 lp.vertical ? abs(lp.y2 - lp.y1) : 
                               abs(lp.x2 - lp.x1) + 1),
            m_di(lp.x1, lp.y1, lp.x2, lp.y2, sx, sy, ex, ey, lp.len, scale_x,
                 lp.x1 & ~line_subpixel_mask, lp.y1 & ~line_subpixel_mask),
            m_ren(ren),
            m_x(lp.x1 >> line_subpixel_shift),
            m_y(lp.y1 >> line_subpixel_shift),
            m_old_x(m_x),
            m_old_y(m_y),
            m_count((lp.vertical ? abs((lp.y2 >> line_subpixel_shift) - m_y) :
                                   abs((lp.x2 >> line_subpixel_shift) - m_x))),
            m_width(ren.subpixel_width()),
            //m_max_extent(m_width >> (line_subpixel_shift - 2)),
            m_max_extent((m_width + line_subpixel_scale) >> line_subpixel_shift),
            m_start(pattern_start + (m_max_extent + 2) * ren.pattern_width()),
            m_step(0)
        {
            agg::dda2_line_interpolator li(0, lp.vertical ? 
                                              (lp.dy << agg::line_subpixel_shift) :
                                              (lp.dx << agg::line_subpixel_shift),
                                           lp.len);

            unsigned i;
            int stop = m_width + line_subpixel_scale * 2;
            for(i = 0; i < max_half_width; ++i)
            {
                m_dist_pos[i] = li.y();
                if(m_dist_pos[i] >= stop) break;
                ++li;
            }
            m_dist_pos[i] = 0x7FFF0000;

            int dist1_start;
            int dist2_start;
            int npix = 1;

            if(lp.vertical)
            {
                do
                {
                    --m_li;
                    m_y -= lp.inc;
                    m_x = (m_lp.x1 + m_li.y()) >> line_subpixel_shift;

                    if(lp.inc > 0) m_di.dec_y(m_x - m_old_x);
                    else           m_di.inc_y(m_x - m_old_x);

                    m_old_x = m_x;

                    dist1_start = dist2_start = m_di.dist_start(); 

                    int dx = 0;
                    if(dist1_start < 0) ++npix;
                    do
                    {
                        dist1_start += m_di.dy_start();
                        dist2_start -= m_di.dy_start();
                        if(dist1_start < 0) ++npix;
                        if(dist2_start < 0) ++npix;
                        ++dx;
                    }
                    while(m_dist_pos[dx] <= m_width);
                    if(npix == 0) break;

                    npix = 0;
                }
                while(--m_step >= -m_max_extent);
            }
            else
            {
                do
                {
                    --m_li;

                    m_x -= lp.inc;
                    m_y = (m_lp.y1 + m_li.y()) >> line_subpixel_shift;

                    if(lp.inc > 0) m_di.dec_x(m_y - m_old_y);
                    else           m_di.inc_x(m_y - m_old_y);

                    m_old_y = m_y;

                    dist1_start = dist2_start = m_di.dist_start(); 

                    int dy = 0;
                    if(dist1_start < 0) ++npix;
                    do
                    {
                        dist1_start -= m_di.dx_start();
                        dist2_start += m_di.dx_start();
                        if(dist1_start < 0) ++npix;
                        if(dist2_start < 0) ++npix;
                        ++dy;
                    }
                    while(m_dist_pos[dy] <= m_width);
                    if(npix == 0) break;

                    npix = 0;
                }
                while(--m_step >= -m_max_extent);
            }
            m_li.adjust_forward();
            m_step -= m_max_extent;
        }

        //---------------------------------------------------------------------
        bool step_hor()
        {
            ++m_li;
            m_x += m_lp.inc;
            m_y = (m_lp.y1 + m_li.y()) >> line_subpixel_shift;

            if(m_lp.inc > 0) m_di.inc_x(m_y - m_old_y);
            else             m_di.dec_x(m_y - m_old_y);

            m_old_y = m_y;

            int s1 = m_di.dist() / m_lp.len;
            int s2 = -s1;

            if(m_lp.inc < 0) s1 = -s1;

            int dist_start;
            int dist_pict;
            int dist_end;
            int dy;
            int dist;

            dist_start = m_di.dist_start();
            dist_pict  = m_di.dist_pict() + m_start;
            dist_end   = m_di.dist_end();
            color_type* p0 = m_colors + max_half_width + 2;
            color_type* p1 = p0;

            int npix = 0;
            p1->clear();
            if(dist_end > 0)
            {
                if(dist_start <= 0)
                {
                    m_ren.pixel(p1, dist_pict, s2);
                }
                ++npix;
            }
            ++p1;

            dy = 1;
            while((dist = m_dist_pos[dy]) - s1 <= m_width)
            {
                dist_start -= m_di.dx_start();
                dist_pict  -= m_di.dx_pict();
                dist_end   -= m_di.dx_end();
                p1->clear();
                if(dist_end > 0 && dist_start <= 0)
                {   
                    if(m_lp.inc > 0) dist = -dist;
                    m_ren.pixel(p1, dist_pict, s2 - dist);
                    ++npix;
                }
                ++p1;
                ++dy;
            }

            dy = 1;
            dist_start = m_di.dist_start();
            dist_pict  = m_di.dist_pict() + m_start;
            dist_end   = m_di.dist_end();
            while((dist = m_dist_pos[dy]) + s1 <= m_width)
            {
                dist_start += m_di.dx_start();
                dist_pict  += m_di.dx_pict();
                dist_end   += m_di.dx_end();
                --p0;
                p0->clear();
                if(dist_end > 0 && dist_start <= 0)
                {   
                    if(m_lp.inc > 0) dist = -dist;
                    m_ren.pixel(p0, dist_pict, s2 + dist);
                    ++npix;
                }
                ++dy;
            }
            m_ren.blend_color_vspan(m_x, 
                                    m_y - dy + 1, 
                                    unsigned(p1 - p0), 
                                    p0); 
            return npix && ++m_step < m_count;
        }



        //---------------------------------------------------------------------
        bool step_ver()
        {
            ++m_li;
            m_y += m_lp.inc;
            m_x = (m_lp.x1 + m_li.y()) >> line_subpixel_shift;

            if(m_lp.inc > 0) m_di.inc_y(m_x - m_old_x);
            else             m_di.dec_y(m_x - m_old_x);

            m_old_x = m_x;

            int s1 = m_di.dist() / m_lp.len;
            int s2 = -s1;

            if(m_lp.inc > 0) s1 = -s1;

            int dist_start;
            int dist_pict;
            int dist_end;
            int dist;
            int dx;

            dist_start = m_di.dist_start();
            dist_pict  = m_di.dist_pict() + m_start;
            dist_end   = m_di.dist_end();
            color_type* p0 = m_colors + max_half_width + 2;
            color_type* p1 = p0;

            int npix = 0;
            p1->clear();
            if(dist_end > 0)
            {
                if(dist_start <= 0)
                {
                    m_ren.pixel(p1, dist_pict, s2);
                }
                ++npix;
            }
            ++p1;

            dx = 1;
            while((dist = m_dist_pos[dx]) - s1 <= m_width)
            {
                dist_start += m_di.dy_start();
                dist_pict  += m_di.dy_pict();
                dist_end   += m_di.dy_end();
                p1->clear();
                if(dist_end > 0 && dist_start <= 0)
                {   
                    if(m_lp.inc > 0) dist = -dist;
                    m_ren.pixel(p1, dist_pict, s2 + dist);
                    ++npix;
                }
                ++p1;
                ++dx;
            }

            dx = 1;
            dist_start = m_di.dist_start();
            dist_pict  = m_di.dist_pict() + m_start;
            dist_end   = m_di.dist_end();
            while((dist = m_dist_pos[dx]) + s1 <= m_width)
            {
                dist_start -= m_di.dy_start();
                dist_pict  -= m_di.dy_pict();
                dist_end   -= m_di.dy_end();
                --p0;
                p0->clear();
                if(dist_end > 0 && dist_start <= 0)
                {   
                    if(m_lp.inc > 0) dist = -dist;
                    m_ren.pixel(p0, dist_pict, s2 - dist);
                    ++npix;
                }
                ++dx;
            }
            m_ren.blend_color_hspan(m_x - dx + 1, 
                                    m_y, 
                                    unsigned(p1 - p0), 
                                    p0);
            return npix && ++m_step < m_count;
        }


        //---------------------------------------------------------------------
        int  pattern_end() const { return m_start + m_di.len(); }

        //---------------------------------------------------------------------
        bool vertical() const { return m_lp.vertical; }
        int  width() const { return m_width; }
        int  count() const { return m_count; }

    private:
        line_interpolator_image(const line_interpolator_image<Renderer>&);
        const line_interpolator_image<Renderer>&
            operator = (const line_interpolator_image<Renderer>&);

    protected:
        const line_parameters& m_lp;
        dda2_line_interpolator m_li;
        distance_interpolator4 m_di; 
        renderer_type&         m_ren;
        int m_plen;
        int m_x;
        int m_y;
        int m_old_x;
        int m_old_y;
        int m_count;
        int m_width;
        int m_max_extent;
        int m_start;
        int m_step;
        int m_dist_pos[max_half_width + 1];
        color_type m_colors[max_half_width * 2 + 4];
    };








    //===================================================renderer_outline_image
    template<class BaseRenderer, class ImagePattern> 
    class renderer_outline_image
    {
    public:
        //---------------------------------------------------------------------
        typedef BaseRenderer base_ren_type;
        typedef renderer_outline_image<BaseRenderer, ImagePattern> self_type;
        typedef typename base_ren_type::color_type color_type;
        typedef ImagePattern pattern_type;


        //---------------------------------------------------------------------
        renderer_outline_image(base_ren_type& ren, pattern_type& patt) :
            m_ren(&ren),
            m_pattern(&patt),
            m_start(0),
            m_scale_x(1.0),
            m_clip_box(0,0,0,0),
            m_clipping(false)
        {}
        void attach(base_ren_type& ren) { m_ren = &ren; }

        //---------------------------------------------------------------------
        void pattern(pattern_type& p) { m_pattern = &p; }
        pattern_type& pattern() const { return *m_pattern; }

        //---------------------------------------------------------------------
        void reset_clipping() { m_clipping = false; }
        void clip_box(double x1, double y1, double x2, double y2)
        {
            m_clip_box.x1 = line_coord_sat::conv(x1);
            m_clip_box.y1 = line_coord_sat::conv(y1);
            m_clip_box.x2 = line_coord_sat::conv(x2);
            m_clip_box.y2 = line_coord_sat::conv(y2);
            m_clipping = true;
        }

        //---------------------------------------------------------------------
        void   scale_x(double s) { m_scale_x = s; }
        double scale_x() const   { return m_scale_x; }

        //---------------------------------------------------------------------
        void   start_x(double s) { m_start = iround(s * line_subpixel_scale); }
        double start_x() const   { return double(m_start) / line_subpixel_scale; }

        //---------------------------------------------------------------------
        int subpixel_width() const { return m_pattern->line_width(); }
        int pattern_width() const { return m_pattern->pattern_width(); }
        double width() const { return double(subpixel_width()) / line_subpixel_scale; }

        //-------------------------------------------------------------------------
        void pixel(color_type* p, int x, int y) const
        {
            m_pattern->pixel(p, x, y);
        }

        //-------------------------------------------------------------------------
        void blend_color_hspan(int x, int y, unsigned len, const color_type* colors)
        {
            m_ren->blend_color_hspan(x, y, len, colors, 0);
        }

        //-------------------------------------------------------------------------
        void blend_color_vspan(int x, int y, unsigned len, const color_type* colors)
        {
            m_ren->blend_color_vspan(x, y, len, colors, 0);
        }

        //-------------------------------------------------------------------------
        static bool accurate_join_only() { return true; }

        //-------------------------------------------------------------------------
        template<class Cmp> 
        void semidot(Cmp, int, int, int, int)
        {
        }

        //-------------------------------------------------------------------------
        void pie(int, int, int, int, int, int)
        {
        }

        //-------------------------------------------------------------------------
        void line0(const line_parameters&)
        {
        }

        //-------------------------------------------------------------------------
        void line1(const line_parameters&, int, int)
        {
        }

        //-------------------------------------------------------------------------
        void line2(const line_parameters&, int, int)
        {
        }

        //-------------------------------------------------------------------------
        void line3_no_clip(const line_parameters& lp, 
                           int sx, int sy, int ex, int ey)
        {
            if(lp.len > line_max_length)
            {
                line_parameters lp1, lp2;
                lp.divide(lp1, lp2);
                int mx = lp1.x2 + (lp1.y2 - lp1.y1);
                int my = lp1.y2 - (lp1.x2 - lp1.x1);
                line3_no_clip(lp1, (lp.x1 + sx) >> 1, (lp.y1 + sy) >> 1, mx, my);
                line3_no_clip(lp2, mx, my, (lp.x2 + ex) >> 1, (lp.y2 + ey) >> 1);
                return;
            }
            
            fix_degenerate_bisectrix_start(lp, &sx, &sy);
            fix_degenerate_bisectrix_end(lp, &ex, &ey);
            line_interpolator_image<self_type> li(*this, lp, 
                                                  sx, sy, 
                                                  ex, ey, 
                                                  m_start, m_scale_x);
            if(li.vertical())
            {
                while(li.step_ver());
            }
            else
            {
                while(li.step_hor());
            }
            m_start += uround(lp.len / m_scale_x);
        }

        //-------------------------------------------------------------------------
        void line3(const line_parameters& lp, 
                   int sx, int sy, int ex, int ey)
        {
            if(m_clipping)
            {
                int x1 = lp.x1;
                int y1 = lp.y1;
                int x2 = lp.x2;
                int y2 = lp.y2;
                unsigned flags = clip_line_segment(&x1, &y1, &x2, &y2, m_clip_box);
                int start = m_start;
                if((flags & 4) == 0)
                {
                    if(flags)
                    {
                        line_parameters lp2(x1, y1, x2, y2, 
                                           uround(calc_distance(x1, y1, x2, y2)));
                        if(flags & 1)
                        {
                            m_start += uround(calc_distance(lp.x1, lp.y1, x1, y1) / m_scale_x);
                            sx = x1 + (y2 - y1); 
                            sy = y1 - (x2 - x1);
                        }
                        else
                        {
                            while(abs(sx - lp.x1) + abs(sy - lp.y1) > lp2.len)
                            {
                                sx = (lp.x1 + sx) >> 1;
                                sy = (lp.y1 + sy) >> 1;
                            }
                        }
                        if(flags & 2)
                        {
                            ex = x2 + (y2 - y1); 
                            ey = y2 - (x2 - x1);
                        }
                        else
                        {
                            while(abs(ex - lp.x2) + abs(ey - lp.y2) > lp2.len)
                            {
                                ex = (lp.x2 + ex) >> 1;
                                ey = (lp.y2 + ey) >> 1;
                            }
                        }
                        line3_no_clip(lp2, sx, sy, ex, ey);
                    }
                    else
                    {
                        line3_no_clip(lp, sx, sy, ex, ey);
                    }
                }
                m_start = start + uround(lp.len / m_scale_x);
            }
            else
            {
                line3_no_clip(lp, sx, sy, ex, ey);
            }
        }

    private:
        base_ren_type*      m_ren;
        pattern_type* m_pattern;
        int                 m_start;
        double              m_scale_x;
        rect_i              m_clip_box;
        bool                m_clipping;
    };





}



#endif
