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

#include <stdio.h>
#include "agg_font_win32_tt.h"
#include "agg_bitset_iterator.h"
#include "agg_renderer_scanline.h"

#ifdef AGG_WIN9X_COMPLIANT
#define GetGlyphOutlineX GetGlyphOutline
#else
#define GetGlyphOutlineX GetGlyphOutlineW
#endif

namespace agg
{

    //------------------------------------------------------------------------------
    //
    // This code implements the AUTODIN II polynomial
    // The variable corresponding to the macro argument "crc" should
    // be an unsigned long.
    // Oroginal code  by Spencer Garrett <srg@quick.com>
    //

    // generated using the AUTODIN II polynomial
    //   x^32 + x^26 + x^23 + x^22 + x^16 +
    //   x^12 + x^11 + x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + x^1 + 1
    //
    //------------------------------------------------------------------------------

    static const unsigned crc32tab[256] = 
    {
       0x00000000, 0x77073096, 0xee0e612c, 0x990951ba,
       0x076dc419, 0x706af48f, 0xe963a535, 0x9e6495a3,
       0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
       0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91,
       0x1db71064, 0x6ab020f2, 0xf3b97148, 0x84be41de,
       0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
       0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec,
       0x14015c4f, 0x63066cd9, 0xfa0f3d63, 0x8d080df5,
       0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
       0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b,
       0x35b5a8fa, 0x42b2986c, 0xdbbbc9d6, 0xacbcf940,
       0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
       0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116,
       0x21b4f4b5, 0x56b3c423, 0xcfba9599, 0xb8bda50f,
       0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
       0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d,
       0x76dc4190, 0x01db7106, 0x98d220bc, 0xefd5102a,
       0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
       0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818,
       0x7f6a0dbb, 0x086d3d2d, 0x91646c97, 0xe6635c01,
       0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
       0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457,
       0x65b0d9c6, 0x12b7e950, 0x8bbeb8ea, 0xfcb9887c,
       0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
       0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2,
       0x4adfa541, 0x3dd895d7, 0xa4d1c46d, 0xd3d6f4fb,
       0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
       0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9,
       0x5005713c, 0x270241aa, 0xbe0b1010, 0xc90c2086,
       0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
       0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4,
       0x59b33d17, 0x2eb40d81, 0xb7bd5c3b, 0xc0ba6cad,
       0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
       0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683,
       0xe3630b12, 0x94643b84, 0x0d6d6a3e, 0x7a6a5aa8,
       0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
       0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe,
       0xf762575d, 0x806567cb, 0x196c3671, 0x6e6b06e7,
       0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
       0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5,
       0xd6d6a3e8, 0xa1d1937e, 0x38d8c2c4, 0x4fdff252,
       0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
       0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60,
       0xdf60efc3, 0xa867df55, 0x316e8eef, 0x4669be79,
       0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
       0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f,
       0xc5ba3bbe, 0xb2bd0b28, 0x2bb45a92, 0x5cb36a04,
       0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
       0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a,
       0x9c0906a9, 0xeb0e363f, 0x72076785, 0x05005713,
       0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
       0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21,
       0x86d3d2d4, 0xf1d4e242, 0x68ddb3f8, 0x1fda836e,
       0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
       0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c,
       0x8f659eff, 0xf862ae69, 0x616bffd3, 0x166ccf45,
       0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
       0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db,
       0xaed16a4a, 0xd9d65adc, 0x40df0b66, 0x37d83bf0,
       0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
       0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6,
       0xbad03605, 0xcdd70693, 0x54de5729, 0x23d967bf,
       0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
       0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d,
    };


    //------------------------------------------------------------------------------

    static unsigned calc_crc32(const unsigned char* buf, unsigned size)
    {
        unsigned crc = (unsigned)~0;
        const unsigned char* p;
        unsigned len = 0; 
        unsigned nr = size;

        for (len += nr, p = buf; nr--; ++p) 
        {
            crc = (crc >> 8) ^ crc32tab[(crc ^ *p) & 0xff];
        }
        return ~crc;
    }


    //------------------------------------------------------------------------
    template<class Scanline, class ScanlineStorage>
    void decompose_win32_glyph_bitmap_mono(const char* gbuf, 
                                           int w, int h,
                                           int x, int y,
                                           bool flip_y,
                                           Scanline& sl,
                                           ScanlineStorage& storage)
    {
        int i;
        int pitch = ((w + 31) >> 5) << 2;
        const int8u* buf = (const int8u*)gbuf;
        sl.reset(x, x + w);
        storage.prepare(w + 2);
        if(flip_y)
        {
            buf += pitch * (h - 1);
            y += h;
            pitch = -pitch;
        }
        for(i = 0; i < h; i++)
        {
            sl.reset_spans();
            bitset_iterator bits(buf, 0);
            int j;
            for(j = 0; j < w; j++)
            {
                if(bits.bit()) sl.add_cell(x + j, cover_full);
                ++bits;
            }
            buf += pitch;
            if(sl.num_spans())
            {
                sl.finalize(y - i - 1);
                storage.render(sl);
            }
        }
    }



    //------------------------------------------------------------------------
    template<class Rasterizer, class Scanline, class ScanlineStorage>
    void decompose_win32_glyph_bitmap_gray8(const char* gbuf, 
                                            int w, int h,
                                            int x, int y,
                                            bool flip_y,
                                            Rasterizer& ras,
                                            Scanline& sl,
                                            ScanlineStorage& storage)
    {
        int i, j;
        int pitch = ((w + 3) >> 2) << 2;
        const int8u* buf = (const int8u*)gbuf;
        sl.reset(x, x + w);
        storage.prepare(w + 2);
        if(flip_y)
        {
            buf += pitch * (h - 1);
            y += h;
            pitch = -pitch;
        }
        for(i = 0; i < h; i++)
        {
            sl.reset_spans();
            const int8u* p = buf;
            for(j = 0; j < w; j++)
            {
                if(*p) 
                {
                    unsigned v = *p;
                    if(v == 64) v = 255;
                    else v <<= 2;
                    sl.add_cell(x + j, ras.apply_gamma(v));
                }
                ++p;
            }
            buf += pitch;
            if(sl.num_spans())
            {
                sl.finalize(y - i - 1);
                storage.render(sl);
            }
        }
    }



    //------------------------------------------------------------------------
    template<class PathStorage, class ConvCoord>
    bool decompose_win32_glyph_outline(const char* gbuf,
                                       unsigned total_size,
                                       bool flip_y, 
                                       PathStorage& path, 
                                       ConvCoord conv)
    {
        const char* cur_glyph = gbuf;
        const char* end_glyph = gbuf + total_size;
        
        while(cur_glyph < end_glyph)
        {
            const TTPOLYGONHEADER* th = (TTPOLYGONHEADER*)cur_glyph;
            
            const char* end_poly = cur_glyph + th->cb;
            const char* cur_poly = cur_glyph + sizeof(TTPOLYGONHEADER);

            path.move_to(conv(th->pfxStart.x), 
                         flip_y ? -conv(th->pfxStart.y) :
                                   conv(th->pfxStart.y));
           
            while(cur_poly < end_poly)
            {
                const TTPOLYCURVE* pc = (const TTPOLYCURVE*)cur_poly;
                
                if (pc->wType == TT_PRIM_LINE)
                {
                    int i;
                    for (i = 0; i < pc->cpfx; i++)
                    {
                        path.line_to(conv(pc->apfx[i].x),
                                     flip_y ? -conv(pc->apfx[i].y) :
                                               conv(pc->apfx[i].y));
                    }
                }
                
                if (pc->wType == TT_PRIM_QSPLINE)
                {
                    int u;
                    for (u = 0; u < pc->cpfx - 1; u++)  // Walk through points in spline
                    {
                        POINTFX pnt_b = pc->apfx[u];    // B is always the current point
                        POINTFX pnt_c = pc->apfx[u+1];
                        
                        if (u < pc->cpfx - 2)           // If not on last spline, compute C
                        {
                            // midpoint (x,y)
                            *(int*)&pnt_c.x = (*(int*)&pnt_b.x + *(int*)&pnt_c.x) / 2;
                            *(int*)&pnt_c.y = (*(int*)&pnt_b.y + *(int*)&pnt_c.y) / 2;
                        }
                        
                        path.curve3(conv(pnt_b.x),
                                    flip_y ? -conv(pnt_b.y) : 
                                              conv(pnt_b.y),
                                    conv(pnt_c.x),
                                    flip_y ? -conv(pnt_c.y) : 
                                              conv(pnt_c.y));
                    }
                }
                cur_poly += sizeof(WORD) * 2 + sizeof(POINTFX) * pc->cpfx;
            }
            cur_glyph += th->cb;
        }
        return true;
    }




    //------------------------------------------------------------------------
    font_engine_win32_tt_base::~font_engine_win32_tt_base()
    {
        delete [] m_kerning_pairs;
        delete [] m_gbuf;
        delete [] m_signature;
        delete [] m_typeface;
        if(m_dc && m_old_font) ::SelectObject(m_dc, m_old_font);
        unsigned i;
        for(i = 0; i < m_num_fonts; ++i)
        {
            delete [] m_font_names[i];
            ::DeleteObject(m_fonts[i]);
        }
        delete [] m_font_names;
        delete [] m_fonts;
    }



    //------------------------------------------------------------------------
    font_engine_win32_tt_base::font_engine_win32_tt_base(bool flag32, 
                                                         HDC dc, 
                                                         unsigned max_fonts) :
        m_flag32(flag32),
        m_dc(dc),
        m_old_font(m_dc ? (HFONT)::GetCurrentObject(m_dc, OBJ_FONT) : 0),
        m_fonts(new HFONT [max_fonts]),
        m_num_fonts(0),
        m_max_fonts(max_fonts),
        m_font_names(new char* [max_fonts]),
        m_cur_font(0),

        m_change_stamp(0),
        m_typeface(new char [256-16]),
        m_typeface_len(256-16-1),
        m_signature(new char [256+256-16]),
        m_height(0),
        m_width(0),
        m_weight(FW_REGULAR),
        m_italic(false),
        m_char_set(ANSI_CHARSET),
        m_pitch_and_family(FF_DONTCARE),
        m_hinting(true),
        m_flip_y(false),
        m_font_created(false),
        m_resolution(0),
        m_glyph_rendering(glyph_ren_native_gray8),
        m_glyph_index(0),
        m_data_size(0),
        m_data_type(glyph_data_invalid),
        m_bounds(1,1,0,0),
        m_advance_x(0.0),
        m_advance_y(0.0),
        m_gbuf(new char [buf_size]),
        m_kerning_pairs(0),
        m_num_kerning_pairs(0),
        m_max_kerning_pairs(0),

        m_path16(),
        m_path32(),
        m_curves16(m_path16),
        m_curves32(m_path32),
        m_scanline_aa(),
        m_scanline_bin(),
        m_scanlines_aa(),
        m_scanlines_bin(),
        m_rasterizer()
    {
        m_curves16.approximation_scale(4.0);
        m_curves32.approximation_scale(4.0);
        memset(&m_matrix, 0, sizeof(m_matrix));
        m_matrix.eM11.value = 1;
        m_matrix.eM22.value = 1;
    }



    //------------------------------------------------------------------------
    int font_engine_win32_tt_base::find_font(const char* name) const
    {
        unsigned i;
        for(i = 0; i < m_num_fonts; ++i)
        {
            if(strcmp(name, m_font_names[i]) == 0) return i;
        }
        return -1;
    }

    //------------------------------------------------------------------------
    static inline FIXED dbl_to_fx(double d)
    {
        int l;
        l = int(d * 65536.0);
        return *(FIXED*)&l;
    }

    //------------------------------------------------------------------------
    static inline FIXED negate_fx(const FIXED& fx)
    {
        int l = -(*(int*)(&fx));
        return *(FIXED*)&l;
    }

    //------------------------------------------------------------------------
    static inline double fx_to_dbl(const FIXED& p)
    {
        return double(p.value) + double(p.fract) * (1.0 / 65536.0);
    }

    //------------------------------------------------------------------------
    static inline int fx_to_plain_int(const FIXED& fx)
    {
        return *(int*)(&fx);
    }


    //------------------------------------------------------------------------
    bool font_engine_win32_tt_base::create_font(const char* typeface_, 
                                                glyph_rendering ren_type)
    {
        if(m_dc)
        {
            unsigned len = strlen(typeface_);
            if(len > m_typeface_len)
            {
                delete [] m_signature;
                delete [] m_typeface;
                m_typeface  = new char [len + 32];
                m_signature = new char [len + 32 + 256];
                m_typeface_len = len + 32 - 1;
            }

            strcpy(m_typeface, typeface_);

            int h = m_height;
            int w = m_width;

            if(m_resolution)
            {
                h = ::MulDiv(m_height, m_resolution, 72);
                w = ::MulDiv(m_width,  m_resolution, 72);
            }

            m_glyph_rendering = ren_type;
            update_signature();
            int idx = find_font(m_signature);
            if(idx >= 0)
            {
                m_cur_font = m_fonts[idx];
                ::SelectObject(m_dc, m_cur_font);
                m_num_kerning_pairs = 0;
                return true;
            }
            else
            {
                m_cur_font = ::CreateFont(-h,                     // height of font
                                          w,                      // average character width
                                          0,                      // angle of escapement
                                          0,                      // base-line orientation angle
                                          m_weight,               // font weight
                                          m_italic,               // italic attribute option
                                          0,                      // underline attribute option
                                          0,                      // strikeout attribute option
                                          m_char_set,             // character set identifier
                                          OUT_DEFAULT_PRECIS,     // output precision
                                          CLIP_DEFAULT_PRECIS,    // clipping precision
                                          ANTIALIASED_QUALITY,    // output quality
                                          m_pitch_and_family,     // pitch and family
                                          m_typeface);            // typeface name
                if(m_cur_font)
                {
                    if(m_num_fonts >= m_max_fonts)
                    {
                        delete [] m_font_names[0];
                        if(m_old_font) ::SelectObject(m_dc, m_old_font);
                        ::DeleteObject(m_fonts[0]);
                        memcpy(m_fonts, 
                               m_fonts + 1, 
                               (m_max_fonts - 1) * sizeof(HFONT));
                        memcpy(m_font_names, 
                               m_font_names + 1, 
                               (m_max_fonts - 1) * sizeof(char*));
                        m_num_fonts = m_max_fonts - 1;
                    }

                    m_font_names[m_num_fonts] = new char[strlen(m_signature) + 1];
                    strcpy(m_font_names[m_num_fonts], m_signature);
                    m_fonts[m_num_fonts] = m_cur_font;
                    ++m_num_fonts;
                    ::SelectObject(m_dc, m_cur_font);
                    m_num_kerning_pairs = 0;
                    update_signature();
                    return true;
                }
            }
        }
        return false;
    }





    //------------------------------------------------------------------------
    bool font_engine_win32_tt_base::create_font(const char* typeface_, 
                                                glyph_rendering ren_type,
                                                double height_,
                                                double width_,
                                                int weight_,
                                                bool italic_,
                                                DWORD char_set_,
                                                DWORD pitch_and_family_)
    {
        height(height_);
        width(width_);
        weight(weight_);
        italic(italic_);
        char_set(char_set_);
        pitch_and_family(pitch_and_family_);
        return create_font(typeface_, ren_type);
    }




    //------------------------------------------------------------------------
    void font_engine_win32_tt_base::update_signature()
    {
        if(m_dc && m_cur_font)
        {
            unsigned gamma_hash = 0;
            if(m_glyph_rendering == glyph_ren_native_gray8 ||
               m_glyph_rendering == glyph_ren_agg_mono || 
               m_glyph_rendering == glyph_ren_agg_gray8)
            {
                unsigned char gamma_table[rasterizer_scanline_aa<>::aa_num];
                unsigned i;
                for(i = 0; i < rasterizer_scanline_aa<>::aa_num; ++i)
                {
                    gamma_table[i] = m_rasterizer.apply_gamma(i);
                }
                gamma_hash = calc_crc32(gamma_table, sizeof(gamma_table));
            }
      
            sprintf(m_signature, 
                    "%s,%u,%d,%d:%dx%d,%d,%d,%d,%d,%d,%08X", 
                    m_typeface,
                    m_char_set,
                    int(m_glyph_rendering),
                    m_resolution,
                    m_height,
                    m_width,
                    m_weight,
                    int(m_italic),
                    int(m_hinting),
                    int(m_flip_y),
                    int(m_pitch_and_family),
                    gamma_hash);
            ++m_change_stamp;
        }
    }



    //------------------------------------------------------------------------
    static inline int fx_to_int(const FIXED& p)
    {
        return (int(p.value) << 6) + (int(p.fract) >> 10);
    }


    //------------------------------------------------------------------------
    bool font_engine_win32_tt_base::prepare_glyph(unsigned glyph_code)
    {
        if(m_dc && m_cur_font)
        {
            int format = GGO_BITMAP;

            switch(m_glyph_rendering)
            {
            case glyph_ren_native_gray8: 
                format = GGO_GRAY8_BITMAP;
                break;

            case glyph_ren_outline:
            case glyph_ren_agg_mono:
            case glyph_ren_agg_gray8:
                format = GGO_NATIVE;
                break;
            }

#ifndef GGO_UNHINTED         // For compatibility with old SDKs.
#define GGO_UNHINTED 0x0100
#endif
            if(!m_hinting) format |= GGO_UNHINTED;
        
            GLYPHMETRICS gm;
            int total_size = GetGlyphOutlineX(m_dc,
                                              glyph_code,
                                              format,
                                              &gm,
                                              buf_size,
                                              (void*)m_gbuf,
                                              &m_matrix);

            if(total_size < 0) 
            {
                // GetGlyphOutline() fails when being called for
                // GGO_GRAY8_BITMAP and white space (stupid Microsoft).
                // It doesn't even initialize the glyph metrics
                // structure. So, we have to query the metrics
                // separately (basically we need gmCellIncX).
                int total_size = GetGlyphOutlineX(m_dc,
                                                  glyph_code,
                                                  GGO_METRICS,
                                                  &gm,
                                                  buf_size,
                                                  (void*)m_gbuf,
                                                  &m_matrix);

                if(total_size < 0) return false;
                gm.gmBlackBoxX = gm.gmBlackBoxY = 0;
                total_size = 0;
            }

            m_glyph_index = glyph_code;
            m_advance_x =  gm.gmCellIncX;
            m_advance_y = -gm.gmCellIncY;

            switch(m_glyph_rendering)
            {
            case glyph_ren_native_mono: 
                decompose_win32_glyph_bitmap_mono(m_gbuf, 
                                                  gm.gmBlackBoxX,
                                                  gm.gmBlackBoxY,
                                                  gm.gmptGlyphOrigin.x,
                                                  m_flip_y ? -gm.gmptGlyphOrigin.y : 
                                                              gm.gmptGlyphOrigin.y,
                                                  m_flip_y,
                                                  m_scanline_bin,
                                                  m_scanlines_bin);
                m_bounds.x1 = m_scanlines_bin.min_x();
                m_bounds.y1 = m_scanlines_bin.min_y();
                m_bounds.x2 = m_scanlines_bin.max_x();
                m_bounds.y2 = m_scanlines_bin.max_y();
                m_data_size = m_scanlines_bin.byte_size(); 
                m_data_type = glyph_data_mono;
                return true;

            case glyph_ren_native_gray8:
                decompose_win32_glyph_bitmap_gray8(m_gbuf, 
                                                   gm.gmBlackBoxX,
                                                   gm.gmBlackBoxY,
                                                   gm.gmptGlyphOrigin.x,
                                                   m_flip_y ? -gm.gmptGlyphOrigin.y : 
                                                               gm.gmptGlyphOrigin.y,
                                                   m_flip_y,
                                                   m_rasterizer,
                                                   m_scanline_aa,
                                                   m_scanlines_aa);
                m_bounds.x1 = m_scanlines_aa.min_x();
                m_bounds.y1 = m_scanlines_aa.min_y();
                m_bounds.x2 = m_scanlines_aa.max_x();
                m_bounds.y2 = m_scanlines_aa.max_y();
                m_data_size = m_scanlines_aa.byte_size(); 
                m_data_type = glyph_data_gray8;
                return true;

            case glyph_ren_outline:
                if(m_flag32)
                {
                    m_path32.remove_all();
                    if(decompose_win32_glyph_outline(m_gbuf,
                                                     total_size,
                                                     m_flip_y, 
                                                     m_path32,
                                                     fx_to_int))
                    {
                        rect_d bnd  = m_path32.bounding_rect();
                        m_data_size = m_path32.byte_size();
                        m_data_type = glyph_data_outline;
                        m_bounds.x1 = int(floor(bnd.x1));
                        m_bounds.y1 = int(floor(bnd.y1));
                        m_bounds.x2 = int(ceil(bnd.x2));
                        m_bounds.y2 = int(ceil(bnd.y2));
                        return true;
                    }
                }
                else
                {
                    m_path16.remove_all();
                    if(decompose_win32_glyph_outline(m_gbuf,
                                                     total_size,
                                                     m_flip_y, 
                                                     m_path16,
                                                     fx_to_int))
                    {
                        rect_d bnd  = m_path16.bounding_rect();
                        m_data_size = m_path16.byte_size();
                        m_data_type = glyph_data_outline;
                        m_bounds.x1 = int(floor(bnd.x1));
                        m_bounds.y1 = int(floor(bnd.y1));
                        m_bounds.x2 = int(ceil(bnd.x2));
                        m_bounds.y2 = int(ceil(bnd.y2));
                        return true;
                    }
                }
                break;

            case glyph_ren_agg_mono:
                m_rasterizer.reset();
                if(m_flag32)
                {
                    m_path32.remove_all();
                    decompose_win32_glyph_outline(m_gbuf,
                                                  total_size,
                                                  m_flip_y, 
                                                  m_path32,
                                                  fx_to_int);
                    m_rasterizer.add_path(m_curves32);
                }
                else
                {
                    m_path16.remove_all();
                    decompose_win32_glyph_outline(m_gbuf,
                                                  total_size,
                                                  m_flip_y, 
                                                  m_path16,
                                                  fx_to_int);
                    m_rasterizer.add_path(m_curves16);
                }
                m_scanlines_bin.prepare(1); // Remove all 
                render_scanlines(m_rasterizer, m_scanline_bin, m_scanlines_bin);
                m_bounds.x1 = m_scanlines_bin.min_x();
                m_bounds.y1 = m_scanlines_bin.min_y();
                m_bounds.x2 = m_scanlines_bin.max_x();
                m_bounds.y2 = m_scanlines_bin.max_y();
                m_data_size = m_scanlines_bin.byte_size(); 
                m_data_type = glyph_data_mono;
                return true;

            case glyph_ren_agg_gray8:
                m_rasterizer.reset();
                if(m_flag32)
                {
                    m_path32.remove_all();
                    decompose_win32_glyph_outline(m_gbuf,
                                                  total_size,
                                                  m_flip_y, 
                                                  m_path32,
                                                  fx_to_int);
                    m_rasterizer.add_path(m_curves32);
                }
                else
                {
                    m_path16.remove_all();
                    decompose_win32_glyph_outline(m_gbuf,
                                                  total_size,
                                                  m_flip_y, 
                                                  m_path16,
                                                  fx_to_int);
                    m_rasterizer.add_path(m_curves16);
                }
                m_scanlines_aa.prepare(1); // Remove all 
                render_scanlines(m_rasterizer, m_scanline_aa, m_scanlines_aa);
                m_bounds.x1 = m_scanlines_aa.min_x();
                m_bounds.y1 = m_scanlines_aa.min_y();
                m_bounds.x2 = m_scanlines_aa.max_x();
                m_bounds.y2 = m_scanlines_aa.max_y();
                m_data_size = m_scanlines_aa.byte_size(); 
                m_data_type = glyph_data_gray8;
                return true;
            }
        }
        return false;
    }



    //------------------------------------------------------------------------
    void font_engine_win32_tt_base::write_glyph_to(int8u* data) const
    {
        if(data && m_data_size)
        {
            switch(m_data_type)
            {
            case glyph_data_mono:    m_scanlines_bin.serialize(data); break;
            case glyph_data_gray8:   m_scanlines_aa.serialize(data);  break;
            case glyph_data_outline:
                if(m_flag32)
                {
                    m_path32.serialize(data);
                }
                else
                {
                    m_path16.serialize(data);
                }
                break;
            }
        }
    }



    //------------------------------------------------------------------------
    bool font_engine_win32_tt_base::pair_less(const KERNINGPAIR v1, 
                                              const KERNINGPAIR v2)
    {
        if(v1.wFirst != v2.wFirst) return v1.wFirst < v2.wFirst;
        return v1.wSecond < v2.wSecond;
    }


    //------------------------------------------------------------------------
    void font_engine_win32_tt_base::sort_kerning_pairs()
    {
        pod_array_adaptor<KERNINGPAIR> pairs(m_kerning_pairs, m_num_kerning_pairs);
        quick_sort(pairs, pair_less);
    }



    //------------------------------------------------------------------------
    void font_engine_win32_tt_base::load_kerning_pairs()
    {
        if(m_dc && m_cur_font)
        {
            if(m_kerning_pairs == 0)
            {
                m_kerning_pairs = new KERNINGPAIR [16384-16];
                m_max_kerning_pairs = 16384-16;
            }
            m_num_kerning_pairs = ::GetKerningPairs(m_dc, 
                                                    m_max_kerning_pairs,
                                                    m_kerning_pairs);

            if(m_num_kerning_pairs)
            {
                // Check to see if the kerning pairs are sorted and
                // sort them if they are not.
                //----------------
                unsigned i;
                for(i = 1; i < m_num_kerning_pairs; ++i)
                {
                    if(!pair_less(m_kerning_pairs[i - 1], m_kerning_pairs[i]))
                    {
                        sort_kerning_pairs();
                        break;
                    }
                }
            }
        }
    }


    //------------------------------------------------------------------------
    bool font_engine_win32_tt_base::add_kerning(unsigned first, unsigned second,
                                                double* x, double* y)
    {
        if(m_dc && m_cur_font)
        {
            if(m_num_kerning_pairs == 0)
            {
                load_kerning_pairs();
            }

            int end = m_num_kerning_pairs - 1;
            int beg = 0;
            KERNINGPAIR t;
            t.wFirst = (WORD)first;
            t.wSecond = (WORD)second;
            while(beg <= end)
            {
                int mid = (end + beg) / 2;
                if(m_kerning_pairs[mid].wFirst  == t.wFirst &&
                   m_kerning_pairs[mid].wSecond == t.wSecond)
                {
                    double dx = m_kerning_pairs[mid].iKernAmount;
                    double dy = 0.0;
                    *x += dx * fx_to_dbl(m_matrix.eM11) + dy * fx_to_dbl(m_matrix.eM21);
                    *y += dx * fx_to_dbl(m_matrix.eM12) + dy * fx_to_dbl(m_matrix.eM22);
                    return true;
                }
                else
                if(pair_less(t, m_kerning_pairs[mid]))
                {
                    end = mid - 1;
                }
                else
                {
                    beg = mid + 1;
                }
            }
            return false;
        }
        return false;
    }



}

