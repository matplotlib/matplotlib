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

#ifndef AGG_FONT_WIN32_TT_INCLUDED
#define AGG_FONT_WIN32_TT_INCLUDED

#include <windows.h>
#include "agg_scanline_storage_aa.h"
#include "agg_scanline_storage_bin.h"
#include "agg_scanline_u.h"
#include "agg_scanline_bin.h"
#include "agg_path_storage_integer.h"
#include "agg_rasterizer_scanline_aa.h"
#include "agg_conv_curve.h"
#include "agg_trans_affine.h"
#include "agg_font_cache_manager.h"

namespace agg
{

    //-----------------------------------------------font_engine_win32_tt_base
    class font_engine_win32_tt_base
    {
       enum { buf_size = 32768-32 };

    public:
        //--------------------------------------------------------------------
        typedef serialized_scanlines_adaptor_aa<int8u>    gray8_adaptor_type;
        typedef serialized_scanlines_adaptor_bin          mono_adaptor_type;
        typedef scanline_storage_aa8                      scanlines_aa_type;
        typedef scanline_storage_bin                      scanlines_bin_type;

        //--------------------------------------------------------------------
        ~font_engine_win32_tt_base();
        font_engine_win32_tt_base(bool flag32, HDC dc, unsigned max_fonts = 32);

        // Set font parameters
        //--------------------------------------------------------------------
        void resolution(unsigned dpi) { m_resolution = unsigned(dpi); }
        void height(double h)         { m_height = unsigned(h);  }
        void width(double w)          { m_width = unsigned(w);   }
        void weight(int w)            { m_weight = w;            }
        void italic(bool it)          { m_italic = it;           }
        void char_set(DWORD c)        { m_char_set = c;          }
        void pitch_and_family(DWORD p){ m_pitch_and_family = p; }
        void flip_y(bool flip)        { m_flip_y = flip;         }
        void hinting(bool h)          { m_hinting = h;           }
        bool create_font(const char* typeface_, glyph_rendering ren_type);

        bool create_font(const char* typeface_, 
                         glyph_rendering ren_type,
                         double height_,
                         double width_=0.0,
                         int weight_=FW_REGULAR,
                         bool italic_=false,
                         DWORD char_set_=ANSI_CHARSET,
                         DWORD pitch_and_family_=FF_DONTCARE);

        // Set Gamma
        //--------------------------------------------------------------------
        template<class GammaF> void gamma(const GammaF& f)
        {
            m_rasterizer.gamma(f);
        }

        // Accessors
        //--------------------------------------------------------------------
        unsigned    resolution()   const { return m_resolution; }
        const char* typeface()     const { return m_typeface;   }
        double      height()       const { return m_height;     }
        double      width()        const { return m_width;      }
        int         weight()       const { return m_weight;     }
        bool        italic()       const { return m_italic;     }
        DWORD       char_set()     const { return m_char_set;   }
        DWORD       pitch_and_family() const { return m_pitch_and_family; }
        bool        hinting()      const { return m_hinting;    }
        bool        flip_y()       const { return m_flip_y;     }


        // Interface mandatory to implement for font_cache_manager
        //--------------------------------------------------------------------
        const char*     font_signature() const { return m_signature;    }
        int             change_stamp()   const { return m_change_stamp; }

        bool            prepare_glyph(unsigned glyph_code);
        unsigned        glyph_index() const { return m_glyph_index; }
        unsigned        data_size()   const { return m_data_size;   }
        glyph_data_type data_type()   const { return m_data_type;   }
        const rect&     bounds()      const { return m_bounds;      }
        double          advance_x()   const { return m_advance_x;   }
        double          advance_y()   const { return m_advance_y;   }
        void            write_glyph_to(int8u* data) const;
        bool            add_kerning(unsigned first, unsigned second,
                                    double* x, double* y);

    private:
        font_engine_win32_tt_base(const font_engine_win32_tt_base&);
        const font_engine_win32_tt_base& operator = (const font_engine_win32_tt_base&);

        void update_signature();
        static bool pair_less(const KERNINGPAIR v1, const KERNINGPAIR v2);
        void load_kerning_pairs();
        void sort_kerning_pairs();
        int  find_font(const char* name) const;

        bool            m_flag32;
        HDC             m_dc;
        HFONT           m_old_font;
        HFONT*          m_fonts;
        unsigned        m_num_fonts;
        unsigned        m_max_fonts;
        char**          m_font_names;
        HFONT           m_cur_font;

        int             m_change_stamp;
        char*           m_typeface;
        unsigned        m_typeface_len;
        char*           m_signature;
        unsigned        m_height;
        unsigned        m_width;
        int             m_weight;
        bool            m_italic;
        DWORD           m_char_set;
        DWORD           m_pitch_and_family;
        bool            m_hinting;
        bool            m_flip_y;

        bool            m_font_created;
        unsigned        m_resolution;
        glyph_rendering m_glyph_rendering;
        unsigned        m_glyph_index;
        unsigned        m_data_size;
        glyph_data_type m_data_type;
        rect            m_bounds;
        double          m_advance_x;
        double          m_advance_y;
        MAT2            m_matrix;
        char*           m_gbuf;
        KERNINGPAIR*    m_kerning_pairs;
        unsigned        m_num_kerning_pairs;
        unsigned        m_max_kerning_pairs;


        path_storage_integer<int16, 6>              m_path16;
        path_storage_integer<int32, 6>              m_path32;
        conv_curve<path_storage_integer<int16, 6> > m_curves16;
        conv_curve<path_storage_integer<int32, 6> > m_curves32;
        scanline_u8              m_scanline_aa;
        scanline_bin             m_scanline_bin;
        scanlines_aa_type        m_scanlines_aa;
        scanlines_bin_type       m_scanlines_bin;
        rasterizer_scanline_aa<> m_rasterizer;
    };




    //------------------------------------------------font_engine_win32_tt_int16
    // This class uses values of type int16 (10.6 format) for the vector cache. 
    // The vector cache is compact, but when rendering glyphs of height
    // more that 200 there integer overflow can occur.
    //
    class font_engine_win32_tt_int16 : public font_engine_win32_tt_base
    {
    public:
        typedef serialized_integer_path_adaptor<int16, 6>     path_adaptor_type;
        typedef font_engine_win32_tt_base::gray8_adaptor_type gray8_adaptor_type;
        typedef font_engine_win32_tt_base::mono_adaptor_type  mono_adaptor_type;
        typedef font_engine_win32_tt_base::scanlines_aa_type  scanlines_aa_type;
        typedef font_engine_win32_tt_base::scanlines_bin_type scanlines_bin_type;

        font_engine_win32_tt_int16(HDC dc, unsigned max_fonts = 32) : 
            font_engine_win32_tt_base(false, dc, max_fonts) {}
    };

    //------------------------------------------------font_engine_win32_tt_int32
    // This class uses values of type int32 (26.6 format) for the vector cache. 
    // The vector cache is twice larger than in font_engine_win32_tt_int16, 
    // but it allows you to render glyphs of very large sizes.
    //
    class font_engine_win32_tt_int32 : public font_engine_win32_tt_base
    {
    public:
        typedef serialized_integer_path_adaptor<int32, 6>     path_adaptor_type;
        typedef font_engine_win32_tt_base::gray8_adaptor_type gray8_adaptor_type;
        typedef font_engine_win32_tt_base::mono_adaptor_type  mono_adaptor_type;
        typedef font_engine_win32_tt_base::scanlines_aa_type  scanlines_aa_type;
        typedef font_engine_win32_tt_base::scanlines_bin_type scanlines_bin_type;

        font_engine_win32_tt_int32(HDC dc, unsigned max_fonts = 32) : 
            font_engine_win32_tt_base(true, dc, max_fonts) {}
    };


}

#endif
