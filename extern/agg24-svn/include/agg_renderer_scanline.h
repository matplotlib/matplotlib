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

#ifndef AGG_RENDERER_SCANLINE_INCLUDED
#define AGG_RENDERER_SCANLINE_INCLUDED

#include "agg_basics.h"
#include "agg_renderer_base.h"

namespace agg
{

    //================================================render_scanline_aa_solid
    template<class Scanline, class BaseRenderer, class ColorT> 
    void render_scanline_aa_solid(const Scanline& sl, 
                                  BaseRenderer& ren, 
                                  const ColorT& color)
    {
        int y = sl.y();
        unsigned num_spans = sl.num_spans();
        typename Scanline::const_iterator span = sl.begin();

        for(;;)
        {
            int x = span->x;
            if(span->len > 0)
            {
                ren.blend_solid_hspan(x, y, (unsigned)span->len, 
                                      color, 
                                      span->covers);
            }
            else
            {
                ren.blend_hline(x, y, (unsigned)(x - span->len - 1), 
                                color, 
                                *(span->covers));
            }
            if(--num_spans == 0) break;
            ++span;
        }
    }

    //===============================================render_scanlines_aa_solid
    template<class Rasterizer, class Scanline, 
             class BaseRenderer, class ColorT>
    void render_scanlines_aa_solid(Rasterizer& ras, Scanline& sl, 
                                   BaseRenderer& ren, const ColorT& color)
    {
        if(ras.rewind_scanlines())
        {
            // Explicitly convert "color" to the BaseRenderer color type.
            // For example, it can be called with color type "rgba", while
            // "rgba8" is needed. Otherwise it will be implicitly 
            // converted in the loop many times.
            //----------------------
            typename BaseRenderer::color_type ren_color(color);

            sl.reset(ras.min_x(), ras.max_x());
            while(ras.sweep_scanline(sl))
            {
                //render_scanline_aa_solid(sl, ren, ren_color);

                // This code is equivalent to the above call (copy/paste). 
                // It's just a "manual" optimization for old compilers,
                // like Microsoft Visual C++ v6.0
                //-------------------------------
                int y = sl.y();
                unsigned num_spans = sl.num_spans();
                typename Scanline::const_iterator span = sl.begin();

                for(;;)
                {
                    int x = span->x;
                    if(span->len > 0)
                    {
                        ren.blend_solid_hspan(x, y, (unsigned)span->len, 
                                              ren_color, 
                                              span->covers);
                    }
                    else
                    {
                        ren.blend_hline(x, y, (unsigned)(x - span->len - 1), 
                                        ren_color, 
                                        *(span->covers));
                    }
                    if(--num_spans == 0) break;
                    ++span;
                }
            }
        }
    }

    //==============================================renderer_scanline_aa_solid
    template<class BaseRenderer> class renderer_scanline_aa_solid
    {
    public:
        typedef BaseRenderer base_ren_type;
        typedef typename base_ren_type::color_type color_type;

        //--------------------------------------------------------------------
        renderer_scanline_aa_solid() : m_ren(0) {}
        explicit renderer_scanline_aa_solid(base_ren_type& ren) : m_ren(&ren) {}
        void attach(base_ren_type& ren)
        {
            m_ren = &ren;
        }
        
        //--------------------------------------------------------------------
        void color(const color_type& c) { m_color = c; }
        const color_type& color() const { return m_color; }

        //--------------------------------------------------------------------
        void prepare() {}

        //--------------------------------------------------------------------
        template<class Scanline> void render(const Scanline& sl)
        {
            render_scanline_aa_solid(sl, *m_ren, m_color);
        }
        
    private:
        base_ren_type* m_ren;
        color_type m_color;
    };













    //======================================================render_scanline_aa
    template<class Scanline, class BaseRenderer, 
             class SpanAllocator, class SpanGenerator> 
    void render_scanline_aa(const Scanline& sl, BaseRenderer& ren, 
                            SpanAllocator& alloc, SpanGenerator& span_gen)
    {
        int y = sl.y();

        unsigned num_spans = sl.num_spans();
        typename Scanline::const_iterator span = sl.begin();
        for(;;)
        {
            int x = span->x;
            int len = span->len;
            const typename Scanline::cover_type* covers = span->covers;

            if(len < 0) len = -len;
            typename BaseRenderer::color_type* colors = alloc.allocate(len);
            span_gen.generate(colors, x, y, len);
            ren.blend_color_hspan(x, y, len, colors, 
                                  (span->len < 0) ? 0 : covers, *covers);

            if(--num_spans == 0) break;
            ++span;
        }
    }

    //=====================================================render_scanlines_aa
    template<class Rasterizer, class Scanline, class BaseRenderer, 
             class SpanAllocator, class SpanGenerator>
    void render_scanlines_aa(Rasterizer& ras, Scanline& sl, BaseRenderer& ren, 
                             SpanAllocator& alloc, SpanGenerator& span_gen)
    {
        if(ras.rewind_scanlines())
        {
            sl.reset(ras.min_x(), ras.max_x());
            span_gen.prepare();
            while(ras.sweep_scanline(sl))
            {
                render_scanline_aa(sl, ren, alloc, span_gen);
            }
        }
    }

    //====================================================renderer_scanline_aa
    template<class BaseRenderer, class SpanAllocator, class SpanGenerator> 
    class renderer_scanline_aa
    {
    public:
        typedef BaseRenderer  base_ren_type;
        typedef SpanAllocator alloc_type;
        typedef SpanGenerator span_gen_type;

        //--------------------------------------------------------------------
        renderer_scanline_aa() : m_ren(0), m_alloc(0), m_span_gen(0) {}
        renderer_scanline_aa(base_ren_type& ren, 
                             alloc_type& alloc, 
                             span_gen_type& span_gen) :
            m_ren(&ren),
            m_alloc(&alloc),
            m_span_gen(&span_gen)
        {}
        void attach(base_ren_type& ren, 
                    alloc_type& alloc, 
                    span_gen_type& span_gen)
        {
            m_ren = &ren;
            m_alloc = &alloc;
            m_span_gen = &span_gen;
        }
        
        //--------------------------------------------------------------------
        void prepare() { m_span_gen->prepare(); }

        //--------------------------------------------------------------------
        template<class Scanline> void render(const Scanline& sl)
        {
            render_scanline_aa(sl, *m_ren, *m_alloc, *m_span_gen);
        }

    private:
        base_ren_type* m_ren;
        alloc_type*    m_alloc;
        span_gen_type* m_span_gen;
    };






    //===============================================render_scanline_bin_solid
    template<class Scanline, class BaseRenderer, class ColorT> 
    void render_scanline_bin_solid(const Scanline& sl, 
                                   BaseRenderer& ren, 
                                   const ColorT& color)
    {
        unsigned num_spans = sl.num_spans();
        typename Scanline::const_iterator span = sl.begin();
        for(;;)
        {
            ren.blend_hline(span->x, 
                            sl.y(), 
                            span->x - 1 + ((span->len < 0) ? 
                                              -span->len : 
                                               span->len), 
                               color, 
                               cover_full);
            if(--num_spans == 0) break;
            ++span;
        }
    }

    //==============================================render_scanlines_bin_solid
    template<class Rasterizer, class Scanline, 
             class BaseRenderer, class ColorT>
    void render_scanlines_bin_solid(Rasterizer& ras, Scanline& sl, 
                                    BaseRenderer& ren, const ColorT& color)
    {
        if(ras.rewind_scanlines())
        {
            // Explicitly convert "color" to the BaseRenderer color type.
            // For example, it can be called with color type "rgba", while
            // "rgba8" is needed. Otherwise it will be implicitly 
            // converted in the loop many times.
            //----------------------
            typename BaseRenderer::color_type ren_color(color);

            sl.reset(ras.min_x(), ras.max_x());
            while(ras.sweep_scanline(sl))
            {
                //render_scanline_bin_solid(sl, ren, ren_color);

                // This code is equivalent to the above call (copy/paste). 
                // It's just a "manual" optimization for old compilers,
                // like Microsoft Visual C++ v6.0
                //-------------------------------
                unsigned num_spans = sl.num_spans();
                typename Scanline::const_iterator span = sl.begin();
                for(;;)
                {
                    ren.blend_hline(span->x, 
                                    sl.y(), 
                                    span->x - 1 + ((span->len < 0) ? 
                                                      -span->len : 
                                                       span->len), 
                                       ren_color, 
                                       cover_full);
                    if(--num_spans == 0) break;
                    ++span;
                }
            }
        }
    }

    //=============================================renderer_scanline_bin_solid
    template<class BaseRenderer> class renderer_scanline_bin_solid
    {
    public:
        typedef BaseRenderer base_ren_type;
        typedef typename base_ren_type::color_type color_type;

        //--------------------------------------------------------------------
        renderer_scanline_bin_solid() : m_ren(0) {}
        explicit renderer_scanline_bin_solid(base_ren_type& ren) : m_ren(&ren) {}
        void attach(base_ren_type& ren)
        {
            m_ren = &ren;
        }
        
        //--------------------------------------------------------------------
        void color(const color_type& c) { m_color = c; }
        const color_type& color() const { return m_color; }

        //--------------------------------------------------------------------
        void prepare() {}

        //--------------------------------------------------------------------
        template<class Scanline> void render(const Scanline& sl)
        {
            render_scanline_bin_solid(sl, *m_ren, m_color);
        }
        
    private:
        base_ren_type* m_ren;
        color_type m_color;
    };








    //======================================================render_scanline_bin
    template<class Scanline, class BaseRenderer, 
             class SpanAllocator, class SpanGenerator> 
    void render_scanline_bin(const Scanline& sl, BaseRenderer& ren, 
                             SpanAllocator& alloc, SpanGenerator& span_gen)
    {
        int y = sl.y();

        unsigned num_spans = sl.num_spans();
        typename Scanline::const_iterator span = sl.begin();
        for(;;)
        {
            int x = span->x;
            int len = span->len;
            if(len < 0) len = -len;
            typename BaseRenderer::color_type* colors = alloc.allocate(len);
            span_gen.generate(colors, x, y, len);
            ren.blend_color_hspan(x, y, len, colors, 0, cover_full); 
            if(--num_spans == 0) break;
            ++span;
        }
    }

    //=====================================================render_scanlines_bin
    template<class Rasterizer, class Scanline, class BaseRenderer, 
             class SpanAllocator, class SpanGenerator>
    void render_scanlines_bin(Rasterizer& ras, Scanline& sl, BaseRenderer& ren, 
                              SpanAllocator& alloc, SpanGenerator& span_gen)
    {
        if(ras.rewind_scanlines())
        {
            sl.reset(ras.min_x(), ras.max_x());
            span_gen.prepare();
            while(ras.sweep_scanline(sl))
            {
                render_scanline_bin(sl, ren, alloc, span_gen);
            }
        }
    }

    //====================================================renderer_scanline_bin
    template<class BaseRenderer, class SpanAllocator, class SpanGenerator> 
    class renderer_scanline_bin
    {
    public:
        typedef BaseRenderer  base_ren_type;
        typedef SpanAllocator alloc_type;
        typedef SpanGenerator span_gen_type;

        //--------------------------------------------------------------------
        renderer_scanline_bin() : m_ren(0), m_alloc(0), m_span_gen(0) {}
        renderer_scanline_bin(base_ren_type& ren, 
                              alloc_type& alloc, 
                              span_gen_type& span_gen) :
            m_ren(&ren),
            m_alloc(&alloc),
            m_span_gen(&span_gen)
        {}
        void attach(base_ren_type& ren, 
                    alloc_type& alloc, 
                    span_gen_type& span_gen)
        {
            m_ren = &ren;
            m_alloc = &alloc;
            m_span_gen = &span_gen;
        }
        
        //--------------------------------------------------------------------
        void prepare() { m_span_gen->prepare(); }

        //--------------------------------------------------------------------
        template<class Scanline> void render(const Scanline& sl)
        {
            render_scanline_bin(sl, *m_ren, *m_alloc, *m_span_gen);
        }

    private:
        base_ren_type* m_ren;
        alloc_type*    m_alloc;
        span_gen_type* m_span_gen;
    };










    //========================================================render_scanlines
    template<class Rasterizer, class Scanline, class Renderer>
    void render_scanlines(Rasterizer& ras, Scanline& sl, Renderer& ren)
    {
        if(ras.rewind_scanlines())
        {
            sl.reset(ras.min_x(), ras.max_x());
            ren.prepare();
            while(ras.sweep_scanline(sl))
            {
                ren.render(sl);
            }
        }
    }

    //========================================================render_all_paths
    template<class Rasterizer, class Scanline, class Renderer, 
             class VertexSource, class ColorStorage, class PathId>
    void render_all_paths(Rasterizer& ras, 
                          Scanline& sl,
                          Renderer& r, 
                          VertexSource& vs, 
                          const ColorStorage& as, 
                          const PathId& path_id,
                          unsigned num_paths)
    {
        for(unsigned i = 0; i < num_paths; i++)
        {
            ras.reset();
            ras.add_path(vs, path_id[i]);
            r.color(as[i]);
            render_scanlines(ras, sl, r);
        }
    }






    //=============================================render_scanlines_compound
    template<class Rasterizer, 
             class ScanlineAA, 
             class ScanlineBin, 
             class BaseRenderer, 
             class SpanAllocator,
             class StyleHandler>
    void render_scanlines_compound(Rasterizer& ras, 
                                   ScanlineAA& sl_aa,
                                   ScanlineBin& sl_bin,
                                   BaseRenderer& ren,
                                   SpanAllocator& alloc,
                                   StyleHandler& sh)
    {
        if(ras.rewind_scanlines())
        {
            int min_x = ras.min_x();
            int len = ras.max_x() - min_x + 2;
            sl_aa.reset(min_x, ras.max_x());
            sl_bin.reset(min_x, ras.max_x());

            typedef typename BaseRenderer::color_type color_type;
            color_type* color_span = alloc.allocate(len * 2);
            color_type* mix_buffer = color_span + len;
            unsigned num_spans;

            unsigned num_styles;
            unsigned style;
            bool     solid;
            while((num_styles = ras.sweep_styles()) > 0)
            {
                typename ScanlineAA::const_iterator span_aa;
                if(num_styles == 1)
                {
                    // Optimization for a single style. Happens often
                    //-------------------------
                    if(ras.sweep_scanline(sl_aa, 0))
                    {
                        style = ras.style(0);
                        if(sh.is_solid(style))
                        {
                            // Just solid fill
                            //-----------------------
                            render_scanline_aa_solid(sl_aa, ren, sh.color(style));
                        }
                        else
                        {
                            // Arbitrary span generator
                            //-----------------------
                            span_aa   = sl_aa.begin();
                            num_spans = sl_aa.num_spans();
                            for(;;)
                            {
                                len = span_aa->len;
                                sh.generate_span(color_span, 
                                                 span_aa->x, 
                                                 sl_aa.y(), 
                                                 len, 
                                                 style);

                                ren.blend_color_hspan(span_aa->x, 
                                                      sl_aa.y(), 
                                                      span_aa->len,
                                                      color_span,
                                                      span_aa->covers);
                                if(--num_spans == 0) break;
                                ++span_aa;
                            }
                        }
                    }
                }
                else
                {
                    if(ras.sweep_scanline(sl_bin, -1))
                    {
                        // Clear the spans of the mix_buffer
                        //--------------------
                        typename ScanlineBin::const_iterator span_bin = sl_bin.begin();
                        num_spans = sl_bin.num_spans();
                        for(;;)
                        {
                            memset(mix_buffer + span_bin->x - min_x, 
                                   0, 
                                   span_bin->len * sizeof(color_type));

                            if(--num_spans == 0) break;
                            ++span_bin;
                        }

                        unsigned i;
                        for(i = 0; i < num_styles; i++)
                        {
                            style = ras.style(i);
                            solid = sh.is_solid(style);

                            if(ras.sweep_scanline(sl_aa, i))
                            {
                                color_type* colors;
                                color_type* cspan;
                                typename ScanlineAA::cover_type* covers;
                                span_aa   = sl_aa.begin();
                                num_spans = sl_aa.num_spans();
                                if(solid)
                                {
                                    // Just solid fill
                                    //-----------------------
                                    for(;;)
                                    {
                                        color_type c = sh.color(style);
                                        len    = span_aa->len;
                                        colors = mix_buffer + span_aa->x - min_x;
                                        covers = span_aa->covers;
                                        do
                                        {
                                            if(*covers == cover_full) 
                                            {
                                                *colors = c;
                                            }
                                            else
                                            {
                                                colors->add(c, *covers);
                                            }
                                            ++colors;
                                            ++covers;
                                        }
                                        while(--len);
                                        if(--num_spans == 0) break;
                                        ++span_aa;
                                    }
                                }
                                else
                                {
                                    // Arbitrary span generator
                                    //-----------------------
                                    for(;;)
                                    {
                                        len = span_aa->len;
                                        colors = mix_buffer + span_aa->x - min_x;
                                        cspan  = color_span;
                                        sh.generate_span(cspan, 
                                                         span_aa->x, 
                                                         sl_aa.y(), 
                                                         len, 
                                                         style);
                                        covers = span_aa->covers;
                                        do
                                        {
                                            if(*covers == cover_full) 
                                            {
                                                *colors = *cspan;
                                            }
                                            else
                                            {
                                                colors->add(*cspan, *covers);
                                            }
                                            ++cspan;
                                            ++colors;
                                            ++covers;
                                        }
                                        while(--len);
                                        if(--num_spans == 0) break;
                                        ++span_aa;
                                    }
                                }
                            }
                        }

                        // Emit the blended result as a color hspan
                        //-------------------------
                        span_bin = sl_bin.begin();
                        num_spans = sl_bin.num_spans();
                        for(;;)
                        {
                            ren.blend_color_hspan(span_bin->x, 
                                                  sl_bin.y(), 
                                                  span_bin->len,
                                                  mix_buffer + span_bin->x - min_x,
                                                  0,
                                                  cover_full);
                            if(--num_spans == 0) break;
                            ++span_bin;
                        }
                    } // if(ras.sweep_scanline(sl_bin, -1))
                } // if(num_styles == 1) ... else
            } // while((num_styles = ras.sweep_styles()) > 0)
        } // if(ras.rewind_scanlines())
    }

    //=======================================render_scanlines_compound_layered
    template<class Rasterizer, 
             class ScanlineAA, 
             class BaseRenderer, 
             class SpanAllocator,
             class StyleHandler>
    void render_scanlines_compound_layered(Rasterizer& ras, 
                                           ScanlineAA& sl_aa,
                                           BaseRenderer& ren,
                                           SpanAllocator& alloc,
                                           StyleHandler& sh)
    {
        if(ras.rewind_scanlines())
        {
            int min_x = ras.min_x();
            int len = ras.max_x() - min_x + 2;
            sl_aa.reset(min_x, ras.max_x());

            typedef typename BaseRenderer::color_type color_type;
            color_type* color_span   = alloc.allocate(len * 2);
            color_type* mix_buffer   = color_span + len;
            cover_type* cover_buffer = ras.allocate_cover_buffer(len);
            unsigned num_spans;

            unsigned num_styles;
            unsigned style;
            bool     solid;
            while((num_styles = ras.sweep_styles()) > 0)
            {
                typename ScanlineAA::const_iterator span_aa;
                if(num_styles == 1)
                {
                    // Optimization for a single style. Happens often
                    //-------------------------
                    if(ras.sweep_scanline(sl_aa, 0))
                    {
                        style = ras.style(0);
                        if(sh.is_solid(style))
                        {
                            // Just solid fill
                            //-----------------------
                            render_scanline_aa_solid(sl_aa, ren, sh.color(style));
                        }
                        else
                        {
                            // Arbitrary span generator
                            //-----------------------
                            span_aa   = sl_aa.begin();
                            num_spans = sl_aa.num_spans();
                            for(;;)
                            {
                                len = span_aa->len;
                                sh.generate_span(color_span, 
                                                 span_aa->x, 
                                                 sl_aa.y(), 
                                                 len, 
                                                 style);

                                ren.blend_color_hspan(span_aa->x, 
                                                      sl_aa.y(), 
                                                      span_aa->len,
                                                      color_span,
                                                      span_aa->covers);
                                if(--num_spans == 0) break;
                                ++span_aa;
                            }
                        }
                    }
                }
                else
                {
                    int      sl_start = ras.scanline_start();
                    unsigned sl_len   = ras.scanline_length();

                    if(sl_len)
                    {
                        memset(mix_buffer + sl_start - min_x, 
                               0, 
                               sl_len * sizeof(color_type));

                        memset(cover_buffer + sl_start - min_x, 
                               0, 
                               sl_len * sizeof(cover_type));

                        int sl_y = 0x7FFFFFFF;
                        unsigned i;
                        for(i = 0; i < num_styles; i++)
                        {
                            style = ras.style(i);
                            solid = sh.is_solid(style);

                            if(ras.sweep_scanline(sl_aa, i))
                            {
                                unsigned    cover;
                                color_type* colors;
                                color_type* cspan;
                                cover_type* src_covers;
                                cover_type* dst_covers;
                                span_aa   = sl_aa.begin();
                                num_spans = sl_aa.num_spans();
                                sl_y      = sl_aa.y();
                                if(solid)
                                {
                                    // Just solid fill
                                    //-----------------------
                                    for(;;)
                                    {
                                        color_type c = sh.color(style);
                                        len    = span_aa->len;
                                        colors = mix_buffer + span_aa->x - min_x;
                                        src_covers = span_aa->covers;
                                        dst_covers = cover_buffer + span_aa->x - min_x;
                                        do
                                        {
                                            cover = *src_covers;
                                            if(*dst_covers + cover > cover_full)
                                            {
                                                cover = cover_full - *dst_covers;
                                            }
                                            if(cover)
                                            {
                                                colors->add(c, cover);
                                                *dst_covers += cover;
                                            }
                                            ++colors;
                                            ++src_covers;
                                            ++dst_covers;
                                        }
                                        while(--len);
                                        if(--num_spans == 0) break;
                                        ++span_aa;
                                    }
                                }
                                else
                                {
                                    // Arbitrary span generator
                                    //-----------------------
                                    for(;;)
                                    {
                                        len = span_aa->len;
                                        colors = mix_buffer + span_aa->x - min_x;
                                        cspan  = color_span;
                                        sh.generate_span(cspan, 
                                                         span_aa->x, 
                                                         sl_aa.y(), 
                                                         len, 
                                                         style);
                                        src_covers = span_aa->covers;
                                        dst_covers = cover_buffer + span_aa->x - min_x;
                                        do
                                        {
                                            cover = *src_covers;
                                            if(*dst_covers + cover > cover_full)
                                            {
                                                cover = cover_full - *dst_covers;
                                            }
                                            if(cover)
                                            {
                                                colors->add(*cspan, cover);
                                                *dst_covers += cover;
                                            }
                                            ++cspan;
                                            ++colors;
                                            ++src_covers;
                                            ++dst_covers;
                                        }
                                        while(--len);
                                        if(--num_spans == 0) break;
                                        ++span_aa;
                                    }
                                }
                            }
                        }
                        ren.blend_color_hspan(sl_start, 
                                              sl_y, 
                                              sl_len,
                                              mix_buffer + sl_start - min_x,
                                              0,
                                              cover_full);
                    } //if(sl_len)
                } //if(num_styles == 1) ... else
            } //while((num_styles = ras.sweep_styles()) > 0)
        } //if(ras.rewind_scanlines())
    }


}

#endif
