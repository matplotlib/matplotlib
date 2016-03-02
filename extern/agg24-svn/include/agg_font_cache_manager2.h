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

#ifndef AGG_FONT_CACHE_MANAGER2_INCLUDED
#define AGG_FONT_CACHE_MANAGER2_INCLUDED

#include <cassert>
#include <exception>
#include <string.h>
#include "agg_array.h"

namespace agg {

namespace fman {
  //---------------------------------------------------------glyph_data_type
  enum glyph_data_type
  {
    glyph_data_invalid = 0,
    glyph_data_mono    = 1,
    glyph_data_gray8   = 2,
    glyph_data_outline = 3
  };


  //-------------------------------------------------------------cached_glyph
  struct cached_glyph
  {
    void *			cached_font;
    unsigned		glyph_code;
    unsigned        glyph_index;
    int8u*          data;
    unsigned        data_size;
    glyph_data_type data_type;
    rect_i          bounds;
    double          advance_x;
    double          advance_y;
  };


  //--------------------------------------------------------------cached_glyphs
  class cached_glyphs
  {
  public:
    enum block_size_e { block_size = 16384-16 };

    //--------------------------------------------------------------------
    cached_glyphs()
      : m_allocator(block_size)
    { memset(m_glyphs, 0, sizeof(m_glyphs)); }

    //--------------------------------------------------------------------
    const cached_glyph* find_glyph(unsigned glyph_code) const
    {
      unsigned msb = (glyph_code >> 8) & 0xFF;
      if(m_glyphs[msb])
      {
        return m_glyphs[msb][glyph_code & 0xFF];
      }
      return 0;
    }

    //--------------------------------------------------------------------
    cached_glyph* cache_glyph(
      void *			cached_font,
      unsigned        glyph_code,
      unsigned        glyph_index,
      unsigned        data_size,
      glyph_data_type data_type,
      const rect_i&   bounds,
      double          advance_x,
      double          advance_y)
    {
      unsigned msb = (glyph_code >> 8) & 0xFF;
      if(m_glyphs[msb] == 0)
      {
        m_glyphs[msb] =
          (cached_glyph**)m_allocator.allocate(sizeof(cached_glyph*) * 256,
          sizeof(cached_glyph*));
        memset(m_glyphs[msb], 0, sizeof(cached_glyph*) * 256);
      }

      unsigned lsb = glyph_code & 0xFF;
      if(m_glyphs[msb][lsb]) return 0; // Already exists, do not overwrite

      cached_glyph* glyph =
        (cached_glyph*)m_allocator.allocate(sizeof(cached_glyph),
        sizeof(double));

      glyph->cached_font		  = cached_font;
      glyph->glyph_code		  = glyph_code;
      glyph->glyph_index        = glyph_index;
      glyph->data               = m_allocator.allocate(data_size);
      glyph->data_size          = data_size;
      glyph->data_type          = data_type;
      glyph->bounds             = bounds;
      glyph->advance_x          = advance_x;
      glyph->advance_y          = advance_y;
      return m_glyphs[msb][lsb] = glyph;
    }

  private:
    block_allocator m_allocator;
    cached_glyph**   m_glyphs[256];
  };



  //------------------------------------------------------------------------
  enum glyph_rendering
  {
    glyph_ren_native_mono,
    glyph_ren_native_gray8,
    glyph_ren_outline,
    glyph_ren_agg_mono,
    glyph_ren_agg_gray8
  };




  //------------------------------------------------------font_cache_manager
  template<class FontEngine> class font_cache_manager
  {
  public:
    typedef FontEngine font_engine_type;
    typedef font_cache_manager<FontEngine> self_type;
    typedef typename font_engine_type::path_adaptor_type   path_adaptor_type;
    typedef typename font_engine_type::gray8_adaptor_type  gray8_adaptor_type;
    typedef typename gray8_adaptor_type::embedded_scanline gray8_scanline_type;
    typedef typename font_engine_type::mono_adaptor_type   mono_adaptor_type;
    typedef typename mono_adaptor_type::embedded_scanline  mono_scanline_type;

    struct cached_font
    {
      cached_font(
        font_engine_type& engine,
        typename FontEngine::loaded_face *face,
        double height,
        double width,
        bool hinting,
        glyph_rendering rendering )
        : m_engine( engine )
        , m_face( face )
        , m_height( height )
        , m_width( width )
        , m_hinting( hinting )
        , m_rendering( rendering )
      {
        select_face();
        m_face_height=m_face->height();
        m_face_width=m_face->width();
        m_face_ascent=m_face->ascent();
        m_face_descent=m_face->descent();
        m_face_ascent_b=m_face->ascent_b();
        m_face_descent_b=m_face->descent_b();
      }

      double height() const
      {
        return m_face_height;
      }

      double width() const
      {
        return m_face_width;
      }

      double ascent() const
      {
        return m_face_ascent;
      }

      double descent() const
      {
        return m_face_descent;
      }

      double ascent_b() const
      {
        return m_face_ascent_b;
      }

      double descent_b() const
      {
        return m_face_descent_b;
      }

      bool add_kerning( const cached_glyph *first, const cached_glyph *second, double* x, double* y)
      {
        if( !first || !second )
          return false;
        select_face();
        return m_face->add_kerning(
          first->glyph_index, second->glyph_index, x, y );
      }

      void select_face()
      {
        m_face->select_instance( m_height, m_width, m_hinting, m_rendering );
      }

      const cached_glyph *get_glyph(unsigned cp)
      {
        const cached_glyph *glyph=m_glyphs.find_glyph(cp);
        if( glyph==0 )
        {
          typename FontEngine::prepared_glyph prepared;
          select_face();
          bool success=m_face->prepare_glyph(cp, &prepared);
          if( success )
          {
            glyph=m_glyphs.cache_glyph(
              this,
              prepared.glyph_code,
              prepared.glyph_index,
              prepared.data_size,
              prepared.data_type,
              prepared.bounds,
              prepared.advance_x,
              prepared.advance_y );
            assert( glyph!=0 );
            m_face->write_glyph_to(&prepared,glyph->data);
          }
        }
        return glyph;
      }

      font_engine_type&   m_engine;
      typename FontEngine::loaded_face *m_face;
      double				m_height;
      double				m_width;
      bool				m_hinting;
      glyph_rendering		m_rendering;
      double				m_face_height;
      double				m_face_width;
      double				m_face_ascent;
      double				m_face_descent;
      double				m_face_ascent_b;
      double				m_face_descent_b;
      cached_glyphs		m_glyphs;
    };

    //--------------------------------------------------------------------
    font_cache_manager(font_engine_type& engine, unsigned max_fonts=32)
      :m_engine(engine)
    { }

    //--------------------------------------------------------------------
    void init_embedded_adaptors(const cached_glyph* gl,
      double x, double y,
      double scale=1.0)
    {
      if(gl)
      {
        switch(gl->data_type)
        {
        default: return;
        case glyph_data_mono:
          m_mono_adaptor.init(gl->data, gl->data_size, x, y);
          break;

        case glyph_data_gray8:
          m_gray8_adaptor.init(gl->data, gl->data_size, x, y);
          break;

        case glyph_data_outline:
          m_path_adaptor.init(gl->data, gl->data_size, x, y, scale);
          break;
        }
      }
    }


    //--------------------------------------------------------------------
    path_adaptor_type&   path_adaptor()   { return m_path_adaptor;   }
    gray8_adaptor_type&  gray8_adaptor()  { return m_gray8_adaptor;  }
    gray8_scanline_type& gray8_scanline() { return m_gray8_scanline; }
    mono_adaptor_type&   mono_adaptor()   { return m_mono_adaptor;   }
    mono_scanline_type&  mono_scanline()  { return m_mono_scanline;  }


  private:
    //--------------------------------------------------------------------
    font_cache_manager(const self_type&);
    const self_type& operator = (const self_type&);

    font_engine_type&   m_engine;
    path_adaptor_type   m_path_adaptor;
    gray8_adaptor_type  m_gray8_adaptor;
    gray8_scanline_type m_gray8_scanline;
    mono_adaptor_type   m_mono_adaptor;
    mono_scanline_type  m_mono_scanline;
  };

}
}

#endif

