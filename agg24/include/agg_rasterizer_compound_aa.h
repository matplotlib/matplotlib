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
//
// The author gratefully acknowleges the support of David Turner, 
// Robert Wilhelm, and Werner Lemberg - the authors of the FreeType 
// libray - in producing this work. See http://www.freetype.org for details.
//
//----------------------------------------------------------------------------
// Contact: mcseem@antigrain.com
//          mcseemagg@yahoo.com
//          http://www.antigrain.com
//----------------------------------------------------------------------------
//
// Adaptation for 32-bit screen coordinates has been sponsored by 
// Liberty Technology Systems, Inc., visit http://lib-sys.com
//
// Liberty Technology Systems, Inc. is the provider of
// PostScript and PDF technology for software developers.
// 
//----------------------------------------------------------------------------
#ifndef AGG_RASTERIZER_COMPOUND_AA_INCLUDED
#define AGG_RASTERIZER_COMPOUND_AA_INCLUDED

#include "agg_rasterizer_cells_aa.h"
#include "agg_rasterizer_sl_clip.h"

namespace agg
{

    //-----------------------------------------------------------cell_style_aa
    // A pixel cell. There're no constructors defined and it was done 
    // intentionally in order to avoid extra overhead when allocating an 
    // array of cells.
    struct cell_style_aa
    {
        int   x;
        int   y;
        int   cover;
        int   area;
        int16 left, right;

        void initial()
        {
            x     = 0x7FFFFFFF;
            y     = 0x7FFFFFFF;
            cover = 0;
            area  = 0;
            left  = -1;
            right = -1;
        }

        void style(const cell_style_aa& c)
        {
            left  = c.left;
            right = c.right;
        }

        int not_equal(int ex, int ey, const cell_style_aa& c) const
        {
            return (ex - x) | (ey - y) | (left - c.left) | (right - c.right);
        }
    };


    //===========================================================layer_order_e
    enum layer_order_e
    {
        layer_unsorted, //------layer_unsorted
        layer_direct,   //------layer_direct
        layer_inverse   //------layer_inverse
    };


    //==================================================rasterizer_compound_aa
    template<class Clip=rasterizer_sl_clip_int> class rasterizer_compound_aa
    {
        struct style_info 
        { 
            unsigned start_cell;
            unsigned num_cells;
            int      last_x;
        };

        struct cell_info
        {
            int x, area, cover; 
        };

    public:
        typedef Clip                      clip_type;
        typedef typename Clip::conv_type  conv_type;
        typedef typename Clip::coord_type coord_type;

        enum aa_scale_e
        {
            aa_shift  = 8,
            aa_scale  = 1 << aa_shift,
            aa_mask   = aa_scale - 1,
            aa_scale2 = aa_scale * 2,
            aa_mask2  = aa_scale2 - 1
        };

        //--------------------------------------------------------------------
        rasterizer_compound_aa() : 
            m_outline(),
            m_clipper(),
            m_filling_rule(fill_non_zero),
            m_layer_order(layer_direct),
            m_styles(),  // Active Styles
            m_ast(),     // Active Style Table (unique values)
            m_asm(),     // Active Style Mask 
            m_cells(),
            m_cover_buf(),
            m_master_alpha(),
            m_min_style(0x7FFFFFFF),
            m_max_style(-0x7FFFFFFF),
            m_start_x(0),
            m_start_y(0),
            m_scan_y(0x7FFFFFFF),
            m_sl_start(0),
            m_sl_len(0)
        {}

        //--------------------------------------------------------------------
        void reset(); 
        void reset_clipping();
        void clip_box(double x1, double y1, double x2, double y2);
        void filling_rule(filling_rule_e filling_rule);
        void layer_order(layer_order_e order);
        void master_alpha(int style, double alpha);

        //--------------------------------------------------------------------
        void styles(int left, int right);
        void move_to(int x, int y);
        void line_to(int x, int y);
        void move_to_d(double x, double y);
        void line_to_d(double x, double y);
        void add_vertex(double x, double y, unsigned cmd);

        void edge(int x1, int y1, int x2, int y2);
        void edge_d(double x1, double y1, double x2, double y2);

        //-------------------------------------------------------------------
        template<class VertexSource>
        void add_path(VertexSource& vs, unsigned path_id=0)
        {
            double x;
            double y;

            unsigned cmd;
            vs.rewind(path_id);
            if(m_outline.sorted()) reset();
            while(!is_stop(cmd = vs.vertex(&x, &y)))
            {
                add_vertex(x, y, cmd);
            }
        }

        
        //--------------------------------------------------------------------
        int min_x()     const { return m_outline.min_x(); }
        int min_y()     const { return m_outline.min_y(); }
        int max_x()     const { return m_outline.max_x(); }
        int max_y()     const { return m_outline.max_y(); }
        int min_style() const { return m_min_style; }
        int max_style() const { return m_max_style; }

        //--------------------------------------------------------------------
        void sort();
        bool rewind_scanlines();
        unsigned sweep_styles();
        int      scanline_start()  const { return m_sl_start; }
        unsigned scanline_length() const { return m_sl_len;   }
        unsigned style(unsigned style_idx) const;

        cover_type* allocate_cover_buffer(unsigned len);

        //--------------------------------------------------------------------
        bool navigate_scanline(int y); 
        bool hit_test(int tx, int ty);

        //--------------------------------------------------------------------
        AGG_INLINE unsigned calculate_alpha(int area, unsigned master_alpha) const
        {
            int cover = area >> (poly_subpixel_shift*2 + 1 - aa_shift);
            if(cover < 0) cover = -cover;
            if(m_filling_rule == fill_even_odd)
            {
                cover &= aa_mask2;
                if(cover > aa_scale)
                {
                    cover = aa_scale2 - cover;
                }
            }
            if(cover > aa_mask) cover = aa_mask;
            return (cover * master_alpha + aa_mask) >> aa_shift;
        }

        //--------------------------------------------------------------------
        // Sweeps one scanline with one style index. The style ID can be 
        // determined by calling style(). 
        template<class Scanline> bool sweep_scanline(Scanline& sl, int style_idx)
        {
            int scan_y = m_scan_y - 1;
            if(scan_y > m_outline.max_y()) return false;

            sl.reset_spans();

            unsigned master_alpha = aa_mask;

            if(style_idx < 0) 
            {
                style_idx = 0;
            }
            else 
            {
                style_idx++;
                master_alpha = m_master_alpha[m_ast[style_idx] + m_min_style - 1];
            }

            const style_info& st = m_styles[m_ast[style_idx]];

            unsigned num_cells = st.num_cells;
            cell_info* cell = &m_cells[st.start_cell];

            int cover = 0;
            while(num_cells--)
            {
                unsigned alpha;
                int x = cell->x;
                int area = cell->area;

                cover += cell->cover;

                ++cell;

                if(area)
                {
                    alpha = calculate_alpha((cover << (poly_subpixel_shift + 1)) - area,
                                            master_alpha);
                    sl.add_cell(x, alpha);
                    x++;
                }

                if(num_cells && cell->x > x)
                {
                    alpha = calculate_alpha(cover << (poly_subpixel_shift + 1),
                                            master_alpha);
                    if(alpha)
                    {
                        sl.add_span(x, cell->x - x, alpha);
                    }
                }
            }

            if(sl.num_spans() == 0) return false;
            sl.finalize(scan_y);
            return true;
        }

    private:
        void add_style(int style_id);
        void allocate_master_alpha();

        //--------------------------------------------------------------------
        // Disable copying
        rasterizer_compound_aa(const rasterizer_compound_aa<Clip>&);
        const rasterizer_compound_aa<Clip>& 
        operator = (const rasterizer_compound_aa<Clip>&);

    private:
        rasterizer_cells_aa<cell_style_aa> m_outline;
        clip_type              m_clipper;
        filling_rule_e         m_filling_rule;
        layer_order_e          m_layer_order;
        pod_vector<style_info> m_styles;  // Active Styles
        pod_vector<unsigned>   m_ast;     // Active Style Table (unique values)
        pod_vector<int8u>      m_asm;     // Active Style Mask 
        pod_vector<cell_info>  m_cells;
        pod_vector<cover_type> m_cover_buf;
        pod_bvector<unsigned>  m_master_alpha;

        int        m_min_style;
        int        m_max_style;
        coord_type m_start_x;
        coord_type m_start_y;
        int        m_scan_y;
        int        m_sl_start;
        unsigned   m_sl_len;
    };










    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::reset() 
    { 
        m_outline.reset(); 
        m_min_style =  0x7FFFFFFF;
        m_max_style = -0x7FFFFFFF;
        m_scan_y    =  0x7FFFFFFF;
        m_sl_start  =  0;
        m_sl_len    = 0;
    }

    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::filling_rule(filling_rule_e filling_rule) 
    { 
        m_filling_rule = filling_rule; 
    }

    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::layer_order(layer_order_e order)
    {
        m_layer_order = order;
    }

    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::clip_box(double x1, double y1, 
                                                double x2, double y2)
    {
        reset();
        m_clipper.clip_box(conv_type::upscale(x1), conv_type::upscale(y1), 
                           conv_type::upscale(x2), conv_type::upscale(y2));
    }

    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::reset_clipping()
    {
        reset();
        m_clipper.reset_clipping();
    }

    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::styles(int left, int right)
    {
        cell_style_aa cell;
        cell.initial();
        cell.left = (int16)left;
        cell.right = (int16)right;
        m_outline.style(cell);
        if(left  >= 0 && left  < m_min_style) m_min_style = left;
        if(left  >= 0 && left  > m_max_style) m_max_style = left;
        if(right >= 0 && right < m_min_style) m_min_style = right;
        if(right >= 0 && right > m_max_style) m_max_style = right;
    }

    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::move_to(int x, int y)
    {
        if(m_outline.sorted()) reset();
        m_clipper.move_to(m_start_x = conv_type::downscale(x), 
                          m_start_y = conv_type::downscale(y));
    }

    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::line_to(int x, int y)
    {
        m_clipper.line_to(m_outline, 
                          conv_type::downscale(x), 
                          conv_type::downscale(y));
    }

    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::move_to_d(double x, double y) 
    { 
        if(m_outline.sorted()) reset();
        m_clipper.move_to(m_start_x = conv_type::upscale(x), 
                          m_start_y = conv_type::upscale(y)); 
    }

    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::line_to_d(double x, double y) 
    { 
        m_clipper.line_to(m_outline, 
                          conv_type::upscale(x), 
                          conv_type::upscale(y)); 
    }

    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::add_vertex(double x, double y, unsigned cmd)
    {
        if(is_move_to(cmd)) 
        {
            move_to_d(x, y);
        }
        else 
        if(is_vertex(cmd))
        {
            line_to_d(x, y);
        }
        else
        if(is_close(cmd))
        {
            m_clipper.line_to(m_outline, m_start_x, m_start_y);
        }
    }

    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::edge(int x1, int y1, int x2, int y2)
    {
        if(m_outline.sorted()) reset();
        m_clipper.move_to(conv_type::downscale(x1), conv_type::downscale(y1));
        m_clipper.line_to(m_outline, 
                          conv_type::downscale(x2), 
                          conv_type::downscale(y2));
    }
    
    //------------------------------------------------------------------------
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::edge_d(double x1, double y1, 
                                              double x2, double y2)
    {
        if(m_outline.sorted()) reset();
        m_clipper.move_to(conv_type::upscale(x1), conv_type::upscale(y1)); 
        m_clipper.line_to(m_outline, 
                          conv_type::upscale(x2), 
                          conv_type::upscale(y2)); 
    }

    //------------------------------------------------------------------------
    template<class Clip> 
    AGG_INLINE void rasterizer_compound_aa<Clip>::sort()
    {
        m_outline.sort_cells();
    }

    //------------------------------------------------------------------------
    template<class Clip> 
    AGG_INLINE bool rasterizer_compound_aa<Clip>::rewind_scanlines()
    {
        m_outline.sort_cells();
        if(m_outline.total_cells() == 0) 
        {
            return false;
        }
        if(m_max_style < m_min_style)
        {
            return false;
        }
        m_scan_y = m_outline.min_y();
        m_styles.allocate(m_max_style - m_min_style + 2, 128);
        allocate_master_alpha();
        return true;
    }

    //------------------------------------------------------------------------
    template<class Clip> 
    AGG_INLINE void rasterizer_compound_aa<Clip>::add_style(int style_id)
    {
        if(style_id < 0) style_id  = 0;
        else             style_id -= m_min_style - 1;

        unsigned nbyte = style_id >> 3;
        unsigned mask = 1 << (style_id & 7);

        style_info* style = &m_styles[style_id];
        if((m_asm[nbyte] & mask) == 0)
        {
            m_ast.add(style_id);
            m_asm[nbyte] |= mask;
            style->start_cell = 0;
            style->num_cells = 0;
            style->last_x = -0x7FFFFFFF;
        }
        ++style->start_cell;
    }

    //------------------------------------------------------------------------
    // Returns the number of styles
    template<class Clip> 
    unsigned rasterizer_compound_aa<Clip>::sweep_styles()
    {
        for(;;)
        {
            if(m_scan_y > m_outline.max_y()) return 0;
            unsigned num_cells = m_outline.scanline_num_cells(m_scan_y);
            const cell_style_aa* const* cells = m_outline.scanline_cells(m_scan_y);
            unsigned num_styles = m_max_style - m_min_style + 2;
            const cell_style_aa* curr_cell;
            unsigned style_id;
            style_info* style;
            cell_info* cell;

            m_cells.allocate(num_cells * 2, 256); // Each cell can have two styles
            m_ast.capacity(num_styles, 64);
            m_asm.allocate((num_styles + 7) >> 3, 8);
            m_asm.zero();

            if(num_cells)
            {
                // Pre-add zero (for no-fill style, that is, -1).
                // We need that to ensure that the "-1 style" would go first.
                m_asm[0] |= 1; 
                m_ast.add(0);
                style = &m_styles[0];
                style->start_cell = 0;
                style->num_cells = 0;
                style->last_x = -0x7FFFFFFF;

                m_sl_start = cells[0]->x;
                m_sl_len   = cells[num_cells-1]->x - m_sl_start + 1;
                while(num_cells--)
                {
                    curr_cell = *cells++;
                    add_style(curr_cell->left);
                    add_style(curr_cell->right);
                }

                // Convert the Y-histogram into the array of starting indexes
                unsigned i;
                unsigned start_cell = 0;
                for(i = 0; i < m_ast.size(); i++)
                {
                    style_info& st = m_styles[m_ast[i]];
                    unsigned v = st.start_cell;
                    st.start_cell = start_cell;
                    start_cell += v;
                }

                cells = m_outline.scanline_cells(m_scan_y);
                num_cells = m_outline.scanline_num_cells(m_scan_y);

                while(num_cells--)
                {
                    curr_cell = *cells++;
                    style_id = (curr_cell->left < 0) ? 0 : 
                                curr_cell->left - m_min_style + 1;

                    style = &m_styles[style_id];
                    if(curr_cell->x == style->last_x)
                    {
                        cell = &m_cells[style->start_cell + style->num_cells - 1];
                        cell->area  += curr_cell->area;
                        cell->cover += curr_cell->cover;
                    }
                    else
                    {
                        cell = &m_cells[style->start_cell + style->num_cells];
                        cell->x       = curr_cell->x;
                        cell->area    = curr_cell->area;
                        cell->cover   = curr_cell->cover;
                        style->last_x = curr_cell->x;
                        style->num_cells++;
                    }

                    style_id = (curr_cell->right < 0) ? 0 : 
                                curr_cell->right - m_min_style + 1;

                    style = &m_styles[style_id];
                    if(curr_cell->x == style->last_x)
                    {
                        cell = &m_cells[style->start_cell + style->num_cells - 1];
                        cell->area  -= curr_cell->area;
                        cell->cover -= curr_cell->cover;
                    }
                    else
                    {
                        cell = &m_cells[style->start_cell + style->num_cells];
                        cell->x       =  curr_cell->x;
                        cell->area    = -curr_cell->area;
                        cell->cover   = -curr_cell->cover;
                        style->last_x =  curr_cell->x;
                        style->num_cells++;
                    }
                }
            }
            if(m_ast.size() > 1) break;
            ++m_scan_y;
        }
        ++m_scan_y;

        if(m_layer_order != layer_unsorted)
        {
            range_adaptor<pod_vector<unsigned> > ra(m_ast, 1, m_ast.size() - 1);
            if(m_layer_order == layer_direct) quick_sort(ra, unsigned_greater);
            else                              quick_sort(ra, unsigned_less);
        }

        return m_ast.size() - 1;
    }

    //------------------------------------------------------------------------
    // Returns style ID depending of the existing style index
    template<class Clip> 
    AGG_INLINE 
    unsigned rasterizer_compound_aa<Clip>::style(unsigned style_idx) const
    {
        return m_ast[style_idx + 1] + m_min_style - 1;
    }

    //------------------------------------------------------------------------ 
    template<class Clip> 
    AGG_INLINE bool rasterizer_compound_aa<Clip>::navigate_scanline(int y)
    {
        m_outline.sort_cells();
        if(m_outline.total_cells() == 0) 
        {
            return false;
        }
        if(m_max_style < m_min_style)
        {
            return false;
        }
        if(y < m_outline.min_y() || y > m_outline.max_y()) 
        {
            return false;
        }
        m_scan_y = y;
        m_styles.allocate(m_max_style - m_min_style + 2, 128);
        allocate_master_alpha();
        return true;
    }
    
    //------------------------------------------------------------------------ 
    template<class Clip> 
    bool rasterizer_compound_aa<Clip>::hit_test(int tx, int ty)
    {
        if(!navigate_scanline(ty)) 
        {
            return false;
        }

        unsigned num_styles = sweep_styles(); 
        if(num_styles <= 0)
        {
            return false;
        }

        scanline_hit_test sl(tx);
        sweep_scanline(sl, -1);
        return sl.hit();
    }

    //------------------------------------------------------------------------ 
    template<class Clip> 
    cover_type* rasterizer_compound_aa<Clip>::allocate_cover_buffer(unsigned len)
    {
        m_cover_buf.allocate(len, 256);
        return &m_cover_buf[0];
    }

    //------------------------------------------------------------------------ 
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::allocate_master_alpha()
    {
        while((int)m_master_alpha.size() <= m_max_style)
        {
            m_master_alpha.add(aa_mask);
        }
    }

    //------------------------------------------------------------------------ 
    template<class Clip> 
    void rasterizer_compound_aa<Clip>::master_alpha(int style, double alpha)
    {
        if(style >= 0)
        {
            while((int)m_master_alpha.size() <= style)
            {
                m_master_alpha.add(aa_mask);
            }
            m_master_alpha[style] = uround(alpha * aa_mask);
        }
    }

}



#endif

