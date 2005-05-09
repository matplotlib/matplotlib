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
#include "agg_font_freetype.h"
#include "agg_bitset_iterator.h"
#include "agg_renderer_scanline.h"


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
    inline double conv_coord_64(int v)
    {
        return double(v) / 64.0;
    }


    //------------------------------------------------------------------------
    inline int conv_coord_none(int v)
    {
        return v;
    }


    //------------------------------------------------------------------------
    template<class PathStorage, class ConvCoord>
    bool decompose_ft_outline(const FT_Outline& outline,
                              bool flip_y,
                              PathStorage& path, 
                              ConvCoord conv)
    {   
        FT_Vector   v_last;
        FT_Vector   v_control;
        FT_Vector   v_start;

        FT_Vector*  point;
        FT_Vector*  limit;
        char*       tags;

        int   n;         // index of contour in outline
        int   first;     // index of first point in contour
        char  tag;       // current point's state

        first = 0;

        for(n = 0; n < outline.n_contours; n++)
        {
            int  last;  // index of last point in contour

            last  = outline.contours[n];
            limit = outline.points + last;

            v_start = outline.points[first];
            v_last  = outline.points[last];

            v_control = v_start;

            point = outline.points + first;
            tags  = outline.tags  + first;
            tag   = FT_CURVE_TAG(tags[0]);

            // A contour cannot start with a cubic control point!
            if(tag == FT_CURVE_TAG_CUBIC) return false;

            // check first point to determine origin
            if( tag == FT_CURVE_TAG_CONIC)
            {
                // first point is conic control.  Yes, this happens.
                if(FT_CURVE_TAG(outline.tags[last]) == FT_CURVE_TAG_ON)
                {
                    // start at last point if it is on the curve
                    v_start = v_last;
                    limit--;
                }
                else
                {
                    // if both first and last points are conic,
                    // start at their middle and record its position
                    // for closure
                    v_start.x = (v_start.x + v_last.x) / 2;
                    v_start.y = (v_start.y + v_last.y) / 2;

                    v_last = v_start;
                }
                point--;
                tags--;
            }

            path.move_to(conv(v_start.x), flip_y ? -conv(v_start.y) : conv(v_start.y));

            while(point < limit)
            {
                point++;
                tags++;

                tag = FT_CURVE_TAG(tags[0]);
                switch(tag)
                {
                    case FT_CURVE_TAG_ON:  // emit a single line_to
                    {
                        path.line_to(conv(point->x), flip_y ? -conv(point->y) : conv(point->y));
                        continue;
                    }

                    case FT_CURVE_TAG_CONIC:  // consume conic arcs
                    {
                        v_control.x = point->x;
                        v_control.y = point->y;

                    Do_Conic:
                        if(point < limit)
                        {
                            FT_Vector vec;
                            FT_Vector v_middle;

                            point++;
                            tags++;
                            tag = FT_CURVE_TAG(tags[0]);

                            vec.x = point->x;
                            vec.y = point->y;

                            if(tag == FT_CURVE_TAG_ON)
                            {
                                path.curve3(conv(v_control.x), 
                                            flip_y ? -conv(v_control.y) : conv(v_control.y), 
                                            conv(vec.x), 
                                            flip_y ? -conv(vec.y) : conv(vec.y));
                                continue;
                            }

                            if(tag != FT_CURVE_TAG_CONIC) return false;

                            v_middle.x = (v_control.x + vec.x) / 2;
                            v_middle.y = (v_control.y + vec.y) / 2;

                            path.curve3(conv(v_control.x), 
                                        flip_y ? -conv(v_control.y) : conv(v_control.y), 
                                        conv(v_middle.x), 
                                        flip_y ? -conv(v_middle.y) : conv(v_middle.y));

                            v_control = vec;
                            goto Do_Conic;
                        }
                        path.curve3(conv(v_control.x), 
                                    flip_y ? -conv(v_control.y) : conv(v_control.y), 
                                    conv(v_start.x), 
                                    flip_y ? -conv(v_start.y) : conv(v_start.y));
                        goto Close;
                    }

                    default:  // FT_CURVE_TAG_CUBIC
                    {
                        FT_Vector vec1, vec2;

                        if(point + 1 > limit || FT_CURVE_TAG(tags[1]) != FT_CURVE_TAG_CUBIC)
                        {
                            return false;
                        }

                        vec1.x = point[0].x; 
                        vec1.y = point[0].y;
                        vec2.x = point[1].x; 
                        vec2.y = point[1].y;

                        point += 2;
                        tags  += 2;

                        if(point <= limit)
                        {
                            FT_Vector vec;

                            vec.x = point->x;
                            vec.y = point->y;

                            path.curve4(conv(vec1.x), 
                                        flip_y ? -conv(vec1.y) : conv(vec1.y), 
                                        conv(vec2.x), 
                                        flip_y ? -conv(vec2.y) : conv(vec2.y),
                                        conv(vec.x), 
                                        flip_y ? -conv(vec.y) : conv(vec.y));
                            continue;
                        }

                        path.curve4(conv(vec1.x), 
                                    flip_y ? -conv(vec1.y) : conv(vec1.y), 
                                    conv(vec2.x), 
                                    flip_y ? -conv(vec2.y) : conv(vec2.y),
                                    conv(v_start.x), 
                                    flip_y ? -conv(v_start.y) : conv(v_start.y));
                        goto Close;
                    }
                }
            }

            path.close_polygon();

       Close:
            first = last + 1; 
        }

        return true;
    }



    //------------------------------------------------------------------------
    template<class Scanline, class ScanlineStorage>
    void decompose_ft_bitmap_mono(const FT_Bitmap& bitmap,
                                  int x, int y,
                                  bool flip_y,
                                  Scanline& sl,
                                  ScanlineStorage& storage)
    {
        int i;
        const int8u* buf = (const int8u*)bitmap.buffer;
        int pitch = bitmap.pitch;
        sl.reset(x, x + bitmap.width);
        storage.prepare(bitmap.width + 2);
        if(flip_y)
        {
            buf += bitmap.pitch * (bitmap.rows - 1);
            y += bitmap.rows;
            pitch = -pitch;
        }
        for(i = 0; i < bitmap.rows; i++)
        {
            sl.reset_spans();
            bitset_iterator bits(buf, 0);
            int j;
            for(j = 0; j < bitmap.width; j++)
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
    void decompose_ft_bitmap_gray8(const FT_Bitmap& bitmap,
                                   int x, int y,
                                   bool flip_y,
                                   Rasterizer& ras,
                                   Scanline& sl,
                                   ScanlineStorage& storage)
    {
        int i, j;
        const int8u* buf = (const int8u*)bitmap.buffer;
        int pitch = bitmap.pitch;
        sl.reset(x, x + bitmap.width);
        storage.prepare(bitmap.width + 2);
        if(flip_y)
        {
            buf += bitmap.pitch * (bitmap.rows - 1);
            y += bitmap.rows;
            pitch = -pitch;
        }
        for(i = 0; i < bitmap.rows; i++)
        {
            sl.reset_spans();
            const int8u* p = buf;
            for(j = 0; j < bitmap.width; j++)
            {
                if(*p) sl.add_cell(x + j, ras.apply_gamma(*p));
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
    font_engine_freetype_base::~font_engine_freetype_base()
    {
        unsigned i;
        for(i = 0; i < m_num_faces; ++i)
        {
            delete [] m_face_names[i];
            FT_Done_Face(m_faces[i]);
        }
        delete [] m_face_names;
        delete [] m_faces;
        delete [] m_signature;
        if(m_library_initialized) FT_Done_FreeType(m_library);
    }


    //------------------------------------------------------------------------
    font_engine_freetype_base::font_engine_freetype_base(bool flag32, 
                                                         unsigned max_faces) :
        m_flag32(flag32),
        m_change_stamp(0),
        m_last_error(0),
        m_name(0),
        m_name_len(256-16-1),
        m_face_index(0),
        m_char_map(FT_ENCODING_NONE),
        m_signature(new char [256+256-16]),
        m_height(0),
        m_width(0),
        m_hinting(true),
        m_flip_y(false),
        m_library_initialized(false),
        m_library(0),
        m_faces(new FT_Face [max_faces]),
        m_face_names(new char* [max_faces]),
        m_num_faces(0),
        m_max_faces(max_faces),
        m_cur_face(0),
        m_resolution(0),
        m_glyph_rendering(glyph_ren_native_gray8),
        m_glyph_index(0),
        m_data_size(0),
        m_data_type(glyph_data_invalid),
        m_bounds(1,1,0,0),
        m_advance_x(0.0),
        m_advance_y(0.0),

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
        m_last_error = FT_Init_FreeType(&m_library);
        if(m_last_error == 0) m_library_initialized = true;
    }



    //------------------------------------------------------------------------
    void font_engine_freetype_base::resolution(unsigned dpi) 
    { 
        m_resolution = dpi;
        update_char_size();
    }


    //------------------------------------------------------------------------
    int font_engine_freetype_base::find_face(const char* face_name) const
    {
        unsigned i;
        for(i = 0; i < m_num_faces; ++i)
        {
            if(strcmp(face_name, m_face_names[i]) == 0) return i;
        }
        return -1;
    }


    //------------------------------------------------------------------------
    double font_engine_freetype_base::ascender() const
    {
        if(m_cur_face)
        {
            return m_cur_face->ascender * height() / m_cur_face->height;
        }
        return 0.0;
    }

    //------------------------------------------------------------------------
    double font_engine_freetype_base::descender() const
    {
        if(m_cur_face)
        {
            return m_cur_face->descender * height() / m_cur_face->height;
        }
        return 0.0;
    }


    //------------------------------------------------------------------------
    bool font_engine_freetype_base::load_font(const char* font_name, 
                                              unsigned face_index,
                                              glyph_rendering ren_type,
                                              const char* font_mem, 
                                              const long font_mem_size)
    {
        bool ret = false;

        if(m_library_initialized)
        {
            m_last_error = 0;

            int idx = find_face(font_name);
            if(idx >= 0)
            {
                m_cur_face = m_faces[idx];
                m_name     = m_face_names[idx];
            }
            else
            {
                if(m_num_faces >= m_max_faces)
                {
                    delete [] m_face_names[0];
                    FT_Done_Face(m_faces[0]);
                    memcpy(m_faces, 
                           m_faces + 1, 
                           (m_max_faces - 1) * sizeof(FT_Face));
                    memcpy(m_face_names, 
                           m_face_names + 1, 
                           (m_max_faces - 1) * sizeof(char*));
                    m_num_faces = m_max_faces - 1;
                }

                if (font_mem && font_mem_size)
                {
                    m_last_error = FT_New_Memory_Face(m_library, 
                                                      (const FT_Byte*)font_mem, 
                                                      font_mem_size, 
                                                      face_index, 
                                                      &m_faces[m_num_faces]);
                }
                else
                {
                    m_last_error = FT_New_Face(m_library,
                                               font_name,
                                               face_index,
                                               &m_faces[m_num_faces]);
                }

                if(m_last_error == 0)
                {
                    m_face_names[m_num_faces] = new char [strlen(font_name) + 1];
                    strcpy(m_face_names[m_num_faces], font_name);
                    m_cur_face = m_faces[m_num_faces];
                    m_name     = m_face_names[m_num_faces];
                    ++m_num_faces;
                }
                else
                {
                    m_face_names[m_num_faces] = 0;
                    m_cur_face = 0;
                    m_name = 0;
                }
            }


            if(m_last_error == 0)
            {
                ret = true;
                
                switch(ren_type)
                {
                case glyph_ren_native_mono:
                    m_glyph_rendering = glyph_ren_native_mono;
                    break;

                case glyph_ren_native_gray8:
                    m_glyph_rendering = glyph_ren_native_gray8;
                    break;

                case glyph_ren_outline:
                    if(FT_IS_SCALABLE(m_cur_face))
                    {
                        m_glyph_rendering = glyph_ren_outline;
                    }
                    else
                    {
                        m_glyph_rendering = glyph_ren_native_gray8;
                    }
                    break;

                case glyph_ren_agg_mono:
                    if(FT_IS_SCALABLE(m_cur_face))
                    {
                        m_glyph_rendering = glyph_ren_agg_mono;
                    }
                    else
                    {
                        m_glyph_rendering = glyph_ren_native_mono;
                    }
                    break;

                case glyph_ren_agg_gray8:
                    if(FT_IS_SCALABLE(m_cur_face))
                    {
                        m_glyph_rendering = glyph_ren_agg_gray8;
                    }
                    else
                    {
                        m_glyph_rendering = glyph_ren_native_gray8;
                    }
                    break;
                }
                update_signature();
            }
        }
        return ret;
    }


    //------------------------------------------------------------------------
    bool font_engine_freetype_base::attach(const char* file_name)
    {
        if(m_cur_face)
        {
            m_last_error = FT_Attach_File(m_cur_face, file_name);
            return m_last_error == 0;
        }
        return false;
    }

    //------------------------------------------------------------------------
    unsigned font_engine_freetype_base::num_faces() const
    {
        if(m_cur_face)
        {
            return m_cur_face->num_faces;
        }
        return 0;
    }

    //------------------------------------------------------------------------
    bool font_engine_freetype_base::char_map(FT_Encoding char_map)
    {
        if(m_cur_face)
        {
            m_last_error = FT_Select_Charmap(m_cur_face, m_char_map);
            if(m_last_error == 0)
            {
                update_signature();
                return true;
            }
        }
        return false;
    }

    //------------------------------------------------------------------------
    bool font_engine_freetype_base::height(double h)
    {
        m_height = int(h * 64.0);
        if(m_cur_face)
        {
            update_char_size();
            return true;
        }
        return false;
    }

    //------------------------------------------------------------------------
    bool font_engine_freetype_base::width(double w)
    {
        m_width = int(w * 64.0);
        if(m_cur_face)
        {
            update_char_size();
            return true;
        }
        return false;
    }

    //------------------------------------------------------------------------
    void font_engine_freetype_base::hinting(bool h)
    { 
        m_hinting = h; 
        if(m_cur_face)
        {
            update_signature();
        }
    }

    //------------------------------------------------------------------------
    void font_engine_freetype_base::flip_y(bool f)
    { 
        m_flip_y = f; 
        if(m_cur_face)
        {
            update_signature();
        }
    }

    //------------------------------------------------------------------------
    void font_engine_freetype_base::update_signature()
    {
        if(m_cur_face && m_name)
        {
            unsigned name_len = strlen(m_name);
            if(name_len > m_name_len)
            {
                delete [] m_signature;
                m_signature = new char [name_len + 32 + 256];
                m_name_len = name_len + 32 - 1;
            }

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
                    "%s,%u,%d,%d,%d:%dx%d,%d,%d,%08X", 
                    m_name,
                    m_char_map,
                    m_face_index,
                    int(m_glyph_rendering),
                    m_resolution,
                    m_height,
                    m_width,
                    int(m_hinting),
                    int(m_flip_y),
                    gamma_hash);
            ++m_change_stamp;
        }
    }


    //------------------------------------------------------------------------
    void font_engine_freetype_base::update_char_size()
    {
        if(m_cur_face)
        {
            if(m_resolution)
            {
                FT_Set_Char_Size(m_cur_face, 
                                 m_width,       // char_width in 1/64th of points
                                 m_height,      // char_height in 1/64th of points
                                 m_resolution,  // horizontal device resolution 
                                 m_resolution); // vertical device resolution 
            }
            else
            {
                FT_Set_Pixel_Sizes(m_cur_face,    
                                   m_width >> 6,    // pixel_width
                                   m_height >> 6);  // pixel_height
            }
            update_signature();
        }
    }





    //------------------------------------------------------------------------
    bool font_engine_freetype_base::prepare_glyph(unsigned glyph_code)
    {
        m_glyph_index = FT_Get_Char_Index(m_cur_face, glyph_code);
        m_last_error = FT_Load_Glyph(m_cur_face, 
                                     m_glyph_index, 
                                     m_hinting ? FT_LOAD_DEFAULT : FT_LOAD_NO_HINTING);
//                                     m_hinting ? FT_LOAD_FORCE_AUTOHINT : FT_LOAD_NO_HINTING);
        if(m_last_error == 0)
        {
            switch(m_glyph_rendering)
            {
            case glyph_ren_native_mono:
                m_last_error = FT_Render_Glyph(m_cur_face->glyph, FT_RENDER_MODE_MONO);
                if(m_last_error == 0)
                {
                    decompose_ft_bitmap_mono(m_cur_face->glyph->bitmap, 
                                             m_cur_face->glyph->bitmap_left,
                                             m_flip_y ? -m_cur_face->glyph->bitmap_top : 
                                                         m_cur_face->glyph->bitmap_top,
                                             m_flip_y,
                                             m_scanline_bin,
                                             m_scanlines_bin);
                    m_bounds.x1 = m_scanlines_bin.min_x();
                    m_bounds.y1 = m_scanlines_bin.min_y();
                    m_bounds.x2 = m_scanlines_bin.max_x();
                    m_bounds.y2 = m_scanlines_bin.max_y();
                    m_data_size = m_scanlines_bin.byte_size(); 
                    m_data_type = glyph_data_mono;
                    m_advance_x = double(m_cur_face->glyph->advance.x) / 64.0;
                    m_advance_y = double(m_cur_face->glyph->advance.y) / 64.0;
                    return true;
                }
                break;


            case glyph_ren_native_gray8:
                m_last_error = FT_Render_Glyph(m_cur_face->glyph, FT_RENDER_MODE_NORMAL);
                if(m_last_error == 0)
                {
                    decompose_ft_bitmap_gray8(m_cur_face->glyph->bitmap, 
                                              m_cur_face->glyph->bitmap_left,
                                              m_flip_y ? -m_cur_face->glyph->bitmap_top : 
                                                          m_cur_face->glyph->bitmap_top,
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
                    m_advance_x = double(m_cur_face->glyph->advance.x) / 64.0;
                    m_advance_y = double(m_cur_face->glyph->advance.y) / 64.0;
                    return true;
                }
                break;


            case glyph_ren_outline:
                if(m_last_error == 0)
                {
                    if(m_flag32)
                    {
                        m_path32.remove_all();
                        if(decompose_ft_outline(m_cur_face->glyph->outline,
                                                m_flip_y, 
                                                m_path32, 
                                                conv_coord_none))
                        {
                            rect_d bnd  = m_path32.bounding_rect();
                            m_data_size = m_path32.byte_size();
                            m_data_type = glyph_data_outline;
                            m_bounds.x1 = int(floor(bnd.x1));
                            m_bounds.y1 = int(floor(bnd.y1));
                            m_bounds.x2 = int(ceil(bnd.x2));
                            m_bounds.y2 = int(ceil(bnd.y2));
                            m_advance_x = double(m_cur_face->glyph->advance.x) / 64.0;
                            m_advance_y = double(m_cur_face->glyph->advance.y) / 64.0;
                            return true;
                        }
                    }
                    else
                    {
                        m_path16.remove_all();
                        if(decompose_ft_outline(m_cur_face->glyph->outline,
                                                m_flip_y, 
                                                m_path16, 
                                                conv_coord_none))
                        {
                            rect_d bnd  = m_path16.bounding_rect();
                            m_data_size = m_path16.byte_size();
                            m_data_type = glyph_data_outline;
                            m_bounds.x1 = int(floor(bnd.x1));
                            m_bounds.y1 = int(floor(bnd.y1));
                            m_bounds.x2 = int(ceil(bnd.x2));
                            m_bounds.y2 = int(ceil(bnd.y2));
                            m_advance_x = double(m_cur_face->glyph->advance.x) / 64.0;
                            m_advance_y = double(m_cur_face->glyph->advance.y) / 64.0;
                            return true;
                        }
                    }
                }
                return false;

            case glyph_ren_agg_mono:
                if(m_last_error == 0)
                {
                    m_rasterizer.reset();
                    if(m_flag32)
                    {
                        m_path32.remove_all();
                        decompose_ft_outline(m_cur_face->glyph->outline,
                                             m_flip_y, 
                                             m_path32, 
                                             conv_coord_none);
                        m_rasterizer.add_path(m_curves32);
                    }
                    else
                    {
                        m_path16.remove_all();
                        decompose_ft_outline(m_cur_face->glyph->outline,
                                             m_flip_y, 
                                             m_path16, 
                                             conv_coord_none);
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
                    m_advance_x = double(m_cur_face->glyph->advance.x) / 64.0;
                    m_advance_y = double(m_cur_face->glyph->advance.y) / 64.0;
                    return true;
                }
                return false;


            case glyph_ren_agg_gray8:
                if(m_last_error == 0)
                {
                    m_rasterizer.reset();
                    if(m_flag32)
                    {
                        m_path32.remove_all();
                        decompose_ft_outline(m_cur_face->glyph->outline,
                                             m_flip_y, 
                                             m_path32, 
                                             conv_coord_none);
                        m_rasterizer.add_path(m_curves32);
                    }
                    else
                    {
                        m_path16.remove_all();
                        decompose_ft_outline(m_cur_face->glyph->outline,
                                             m_flip_y, 
                                             m_path16, 
                                             conv_coord_none);
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
                    m_advance_x = double(m_cur_face->glyph->advance.x) / 64.0;
                    m_advance_y = double(m_cur_face->glyph->advance.y) / 64.0;
                    return true;
                }
                return false;
            }
        }
        return false;
    }




    //------------------------------------------------------------------------
    void font_engine_freetype_base::write_glyph_to(int8u* data) const
    {
        if(data && m_data_size)
        {
            switch(m_data_type)
            {
            default: return;
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
            case glyph_data_invalid: break;
            }
        }
    }



    //------------------------------------------------------------------------
    bool font_engine_freetype_base::add_kerning(unsigned first, unsigned second,
                                           double* x, double* y)
    {
        if(m_cur_face && first && second && FT_HAS_KERNING(m_cur_face))
        {
            FT_Vector delta;
            FT_Get_Kerning(m_cur_face, first, second,
                           FT_KERNING_DEFAULT, &delta);
            *x += double(delta.x) / 64.0;
            *y += double(delta.y) / 64.0;
            return true;
        }
        return false;
    }



}


