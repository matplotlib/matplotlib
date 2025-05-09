/* -*- mode: c++; c-basic-offset: 4 -*- */

#include <algorithm>
#include <cstdio>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ft2font.h"
#include "mplutils.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338328
#endif

/**
 To improve the hinting of the fonts, this code uses a hack
 presented here:

 http://agg.sourceforge.net/antigrain.com/research/font_rasterization/index.html

 The idea is to limit the effect of hinting in the x-direction, while
 preserving hinting in the y-direction.  Since freetype does not
 support this directly, the dpi in the x-direction is set higher than
 in the y-direction, which affects the hinting grid.  Then, a global
 transform is placed on the font to shrink it back to the desired
 size.  While it is a bit surprising that the dpi setting affects
 hinting, whereas the global transform does not, this is documented
 behavior of FreeType, and therefore hopefully unlikely to change.
 The FreeType 2 tutorial says:

      NOTE: The transformation is applied to every glyph that is
      loaded through FT_Load_Glyph and is completely independent of
      any hinting process. This means that you won't get the same
      results if you load a glyph at the size of 24 pixels, or a glyph
      at the size at 12 pixels scaled by 2 through a transform,
      because the hints will have been computed differently (except
      you have disabled hints).
 */

FT_Library _ft2Library;

// FreeType error codes; loaded as per fterror.h.
static char const* ft_error_string(FT_Error error) {
#undef __FTERRORS_H__
#define FT_ERROR_START_LIST     switch (error) {
#define FT_ERRORDEF( e, v, s )    case v: return s;
#define FT_ERROR_END_LIST         default: return NULL; }
#include FT_ERRORS_H
}

void throw_ft_error(std::string message, FT_Error error) {
    char const* s = ft_error_string(error);
    std::ostringstream os("");
    if (s) {
        os << message << " (" << s << "; error code 0x" << std::hex << error << ")";
    } else {  // Should not occur, but don't add another error from failed lookup.
        os << message << " (error code 0x" << std::hex << error << ")";
    }
    throw std::runtime_error(os.str());
}

FT2Image::FT2Image() : m_buffer(nullptr), m_width(0), m_height(0)
{
}

FT2Image::FT2Image(unsigned long width, unsigned long height)
    : m_buffer(nullptr), m_width(0), m_height(0)
{
    resize(width, height);
}

FT2Image::~FT2Image()
{
    delete[] m_buffer;
}

void FT2Image::resize(long width, long height)
{
    if (width <= 0) {
        width = 1;
    }
    if (height <= 0) {
        height = 1;
    }
    size_t numBytes = width * height;

    if ((unsigned long)width != m_width || (unsigned long)height != m_height) {
        if (numBytes > m_width * m_height) {
            delete[] m_buffer;
            m_buffer = nullptr;
            m_buffer = new unsigned char[numBytes];
        }

        m_width = (unsigned long)width;
        m_height = (unsigned long)height;
    }

    if (numBytes && m_buffer) {
        memset(m_buffer, 0, numBytes);
    }
}

void FT2Image::draw_bitmap(FT_Bitmap *bitmap, FT_Int x, FT_Int y)
{
    FT_Int image_width = (FT_Int)m_width;
    FT_Int image_height = (FT_Int)m_height;
    FT_Int char_width = bitmap->width;
    FT_Int char_height = bitmap->rows;

    FT_Int x1 = std::min(std::max(x, 0), image_width);
    FT_Int y1 = std::min(std::max(y, 0), image_height);
    FT_Int x2 = std::min(std::max(x + char_width, 0), image_width);
    FT_Int y2 = std::min(std::max(y + char_height, 0), image_height);

    FT_Int x_start = std::max(0, -x);
    FT_Int y_offset = y1 - std::max(0, -y);

    if (bitmap->pixel_mode == FT_PIXEL_MODE_GRAY) {
        for (FT_Int i = y1; i < y2; ++i) {
            unsigned char *dst = m_buffer + (i * image_width + x1);
            unsigned char *src = bitmap->buffer + (((i - y_offset) * bitmap->pitch) + x_start);
            for (FT_Int j = x1; j < x2; ++j, ++dst, ++src)
                *dst |= *src;
        }
    } else if (bitmap->pixel_mode == FT_PIXEL_MODE_MONO) {
        for (FT_Int i = y1; i < y2; ++i) {
            unsigned char *dst = m_buffer + (i * image_width + x1);
            unsigned char *src = bitmap->buffer + ((i - y_offset) * bitmap->pitch);
            for (FT_Int j = x1; j < x2; ++j, ++dst) {
                int x = (j - x1 + x_start);
                int val = *(src + (x >> 3)) & (1 << (7 - (x & 0x7)));
                *dst = val ? 255 : *dst;
            }
        }
    } else {
        throw std::runtime_error("Unknown pixel mode");
    }
}

void
FT2Image::draw_rect_filled(unsigned long x0, unsigned long y0, unsigned long x1, unsigned long y1)
{
    x0 = std::min(x0, m_width);
    y0 = std::min(y0, m_height);
    x1 = std::min(x1 + 1, m_width);
    y1 = std::min(y1 + 1, m_height);

    for (size_t j = y0; j < y1; j++) {
        for (size_t i = x0; i < x1; i++) {
            m_buffer[i + j * m_width] = 255;
        }
    }
}

// ft_outline_decomposer should be passed to FT_Outline_Decompose.
struct ft_outline_decomposer
{
    std::vector<double> &vertices;
    std::vector<unsigned char> &codes;
};

static int
ft_outline_move_to(FT_Vector const* to, void* user)
{
    ft_outline_decomposer* d = reinterpret_cast<ft_outline_decomposer*>(user);
    if (!d->vertices.empty()) {
        // Appending CLOSEPOLY is important to make patheffects work.
        d->vertices.push_back(0);
        d->vertices.push_back(0);
        d->codes.push_back(CLOSEPOLY);
    }
    d->vertices.push_back(to->x * (1. / 64.));
    d->vertices.push_back(to->y * (1. / 64.));
    d->codes.push_back(MOVETO);
    return 0;
}

static int
ft_outline_line_to(FT_Vector const* to, void* user)
{
    ft_outline_decomposer* d = reinterpret_cast<ft_outline_decomposer*>(user);
    d->vertices.push_back(to->x * (1. / 64.));
    d->vertices.push_back(to->y * (1. / 64.));
    d->codes.push_back(LINETO);
    return 0;
}

static int
ft_outline_conic_to(FT_Vector const* control, FT_Vector const* to, void* user)
{
    ft_outline_decomposer* d = reinterpret_cast<ft_outline_decomposer*>(user);
    d->vertices.push_back(control->x * (1. / 64.));
    d->vertices.push_back(control->y * (1. / 64.));
    d->vertices.push_back(to->x * (1. / 64.));
    d->vertices.push_back(to->y * (1. / 64.));
    d->codes.push_back(CURVE3);
    d->codes.push_back(CURVE3);
    return 0;
}

static int
ft_outline_cubic_to(
  FT_Vector const* c1, FT_Vector const* c2, FT_Vector const* to, void* user)
{
    ft_outline_decomposer* d = reinterpret_cast<ft_outline_decomposer*>(user);
    d->vertices.push_back(c1->x * (1. / 64.));
    d->vertices.push_back(c1->y * (1. / 64.));
    d->vertices.push_back(c2->x * (1. / 64.));
    d->vertices.push_back(c2->y * (1. / 64.));
    d->vertices.push_back(to->x * (1. / 64.));
    d->vertices.push_back(to->y * (1. / 64.));
    d->codes.push_back(CURVE4);
    d->codes.push_back(CURVE4);
    d->codes.push_back(CURVE4);
    return 0;
}

static FT_Outline_Funcs ft_outline_funcs = {
    ft_outline_move_to,
    ft_outline_line_to,
    ft_outline_conic_to,
    ft_outline_cubic_to};

void
FT2Font::get_path(std::vector<double> &vertices, std::vector<unsigned char> &codes)
{
    if (!face->glyph) {
        throw std::runtime_error("No glyph loaded");
    }
    ft_outline_decomposer decomposer = {
        vertices,
        codes,
    };
    // We can make a close-enough estimate based on number of points and number of
    // contours (which produce a MOVETO each), though it's slightly underestimating due
    // to higher-order curves.
    size_t estimated_points = static_cast<size_t>(face->glyph->outline.n_contours) +
                              static_cast<size_t>(face->glyph->outline.n_points);
    vertices.reserve(2 * estimated_points);
    codes.reserve(estimated_points);
    if (FT_Error error = FT_Outline_Decompose(
            &face->glyph->outline, &ft_outline_funcs, &decomposer)) {
        throw std::runtime_error("FT_Outline_Decompose failed with error " +
                                 std::to_string(error));
    }
    if (vertices.empty()) {  // Don't append CLOSEPOLY to null glyphs.
        return;
    }
    vertices.push_back(0);
    vertices.push_back(0);
    codes.push_back(CLOSEPOLY);
}

FT2Font::FT2Font(FT_Open_Args &open_args,
                 long hinting_factor_,
                 std::vector<FT2Font *> &fallback_list,
                 FT2Font::WarnFunc warn, bool warn_if_used)
    : ft_glyph_warn(warn), warn_if_used(warn_if_used), image(), face(nullptr),
      hinting_factor(hinting_factor_),
      // set default kerning factor to 0, i.e., no kerning manipulation
      kerning_factor(0)
{
    clear();

    FT_Error error = FT_Open_Face(_ft2Library, &open_args, 0, &face);
    if (error) {
        throw_ft_error("Can not load face", error);
    }

    // set a default fontsize 12 pt at 72dpi
    error = FT_Set_Char_Size(face, 12 * 64, 0, 72 * (unsigned int)hinting_factor, 72);
    if (error) {
        FT_Done_Face(face);
        throw_ft_error("Could not set the fontsize", error);
    }

    if (open_args.stream != nullptr) {
        face->face_flags |= FT_FACE_FLAG_EXTERNAL_STREAM;
    }

    FT_Matrix transform = { 65536 / hinting_factor, 0, 0, 65536 };
    FT_Set_Transform(face, &transform, nullptr);

    // Set fallbacks
    std::copy(fallback_list.begin(), fallback_list.end(), std::back_inserter(fallbacks));
}

FT2Font::~FT2Font()
{
    for (auto & glyph : glyphs) {
        FT_Done_Glyph(glyph);
    }

    if (face) {
        FT_Done_Face(face);
    }
}

void FT2Font::clear()
{
    pen.x = pen.y = 0;
    bbox.xMin = bbox.yMin = bbox.xMax = bbox.yMax = 0;
    advance = 0;

    for (auto & glyph : glyphs) {
        FT_Done_Glyph(glyph);
    }

    glyphs.clear();
    glyph_to_font.clear();
    char_to_font.clear();

    for (auto & fallback : fallbacks) {
        fallback->clear();
    }
}

void FT2Font::set_size(double ptsize, double dpi)
{
    FT_Error error = FT_Set_Char_Size(
        face, (FT_F26Dot6)(ptsize * 64), 0, (FT_UInt)(dpi * hinting_factor), (FT_UInt)dpi);
    if (error) {
        throw_ft_error("Could not set the fontsize", error);
    }
    FT_Matrix transform = { 65536 / hinting_factor, 0, 0, 65536 };
    FT_Set_Transform(face, &transform, nullptr);

    for (auto & fallback : fallbacks) {
        fallback->set_size(ptsize, dpi);
    }
}

void FT2Font::set_charmap(int i)
{
    if (i >= face->num_charmaps) {
        throw std::runtime_error("i exceeds the available number of char maps");
    }
    FT_CharMap charmap = face->charmaps[i];
    if (FT_Error error = FT_Set_Charmap(face, charmap)) {
        throw_ft_error("Could not set the charmap", error);
    }
}

void FT2Font::select_charmap(unsigned long i)
{
    if (FT_Error error = FT_Select_Charmap(face, (FT_Encoding)i)) {
        throw_ft_error("Could not set the charmap", error);
    }
}

int FT2Font::get_kerning(FT_UInt left, FT_UInt right, FT_Kerning_Mode mode,
                         bool fallback = false)
{
    if (fallback && glyph_to_font.find(left) != glyph_to_font.end() &&
        glyph_to_font.find(right) != glyph_to_font.end()) {
        FT2Font *left_ft_object = glyph_to_font[left];
        FT2Font *right_ft_object = glyph_to_font[right];
        if (left_ft_object != right_ft_object) {
            // we do not know how to do kerning between different fonts
            return 0;
        }
        // if left_ft_object is the same as right_ft_object,
        // do the exact same thing which set_text does.
        return right_ft_object->get_kerning(left, right, mode, false);
    }
    else
    {
        FT_Vector delta;
        return get_kerning(left, right, mode, delta);
    }
}

int FT2Font::get_kerning(FT_UInt left, FT_UInt right, FT_Kerning_Mode mode,
                         FT_Vector &delta)
{
    if (!FT_HAS_KERNING(face)) {
        return 0;
    }

    if (!FT_Get_Kerning(face, left, right, mode, &delta)) {
        return (int)(delta.x) / (hinting_factor << kerning_factor);
    } else {
        return 0;
    }
}

void FT2Font::set_kerning_factor(int factor)
{
    kerning_factor = factor;
    for (auto & fallback : fallbacks) {
        fallback->set_kerning_factor(factor);
    }
}

void FT2Font::set_text(
    std::u32string_view text, double angle, FT_Int32 flags, std::vector<double> &xys)
{
    FT_Matrix matrix; /* transformation matrix */

    angle = angle * (2 * M_PI / 360.0);

    // this computes width and height in subpixels so we have to multiply by 64
    double cosangle = cos(angle) * 0x10000L;
    double sinangle = sin(angle) * 0x10000L;

    matrix.xx = (FT_Fixed)cosangle;
    matrix.xy = (FT_Fixed)-sinangle;
    matrix.yx = (FT_Fixed)sinangle;
    matrix.yy = (FT_Fixed)cosangle;

    clear();

    bbox.xMin = bbox.yMin = 32000;
    bbox.xMax = bbox.yMax = -32000;

    FT_UInt previous = 0;
    FT2Font *previous_ft_object = nullptr;

    for (auto codepoint : text) {
        FT_UInt glyph_index = 0;
        FT_BBox glyph_bbox;
        FT_Pos last_advance;

        FT_Error charcode_error, glyph_error;
        std::set<FT_String*> glyph_seen_fonts;
        FT2Font *ft_object_with_glyph = this;
        bool was_found = load_char_with_fallback(ft_object_with_glyph, glyph_index, glyphs,
                                                 char_to_font, glyph_to_font, codepoint, flags,
                                                 charcode_error, glyph_error, glyph_seen_fonts, false);
        if (!was_found) {
            ft_glyph_warn((FT_ULong)codepoint, glyph_seen_fonts);
            // render missing glyph tofu
            // come back to top-most font
            ft_object_with_glyph = this;
            char_to_font[codepoint] = ft_object_with_glyph;
            glyph_to_font[glyph_index] = ft_object_with_glyph;
            ft_object_with_glyph->load_glyph(glyph_index, flags, ft_object_with_glyph, false);
        } else if (ft_object_with_glyph->warn_if_used) {
            ft_glyph_warn((FT_ULong)codepoint, glyph_seen_fonts);
        }

        // retrieve kerning distance and move pen position
        if ((ft_object_with_glyph == previous_ft_object) &&  // if both fonts are the same
            ft_object_with_glyph->has_kerning() &&           // if the font knows how to kern
            previous && glyph_index                          // and we really have 2 glyphs
            ) {
            FT_Vector delta;
            pen.x += ft_object_with_glyph->get_kerning(previous, glyph_index, FT_KERNING_DEFAULT, delta);
        }

        // extract glyph image and store it in our table
        FT_Glyph &thisGlyph = glyphs[glyphs.size() - 1];

        last_advance = ft_object_with_glyph->get_face()->glyph->advance.x;
        FT_Glyph_Transform(thisGlyph, nullptr, &pen);
        FT_Glyph_Transform(thisGlyph, &matrix, nullptr);
        xys.push_back(pen.x);
        xys.push_back(pen.y);

        FT_Glyph_Get_CBox(thisGlyph, FT_GLYPH_BBOX_SUBPIXELS, &glyph_bbox);

        bbox.xMin = std::min(bbox.xMin, glyph_bbox.xMin);
        bbox.xMax = std::max(bbox.xMax, glyph_bbox.xMax);
        bbox.yMin = std::min(bbox.yMin, glyph_bbox.yMin);
        bbox.yMax = std::max(bbox.yMax, glyph_bbox.yMax);

        pen.x += last_advance;

        previous = glyph_index;
        previous_ft_object = ft_object_with_glyph;

    }

    FT_Vector_Transform(&pen, &matrix);
    advance = pen.x;

    if (bbox.xMin > bbox.xMax) {
        bbox.xMin = bbox.yMin = bbox.xMax = bbox.yMax = 0;
    }
}

void FT2Font::load_char(long charcode, FT_Int32 flags, FT2Font *&ft_object, bool fallback = false)
{
    // if this is parent FT2Font, cache will be filled in 2 ways:
    // 1. set_text was previously called
    // 2. set_text was not called and fallback was enabled
    std::set <FT_String *> glyph_seen_fonts;
    if (fallback && char_to_font.find(charcode) != char_to_font.end()) {
        ft_object = char_to_font[charcode];
        // since it will be assigned to ft_object anyway
        FT2Font *throwaway = nullptr;
        ft_object->load_char(charcode, flags, throwaway, false);
    } else if (fallback) {
        FT_UInt final_glyph_index;
        FT_Error charcode_error, glyph_error;
        FT2Font *ft_object_with_glyph = this;
        bool was_found = load_char_with_fallback(ft_object_with_glyph, final_glyph_index,
                                                 glyphs, char_to_font, glyph_to_font,
                                                 charcode, flags, charcode_error, glyph_error,
                                                 glyph_seen_fonts, true);
        if (!was_found) {
            ft_glyph_warn(charcode, glyph_seen_fonts);
            if (charcode_error) {
                throw_ft_error("Could not load charcode", charcode_error);
            }
            else if (glyph_error) {
                throw_ft_error("Could not load charcode", glyph_error);
            }
        } else if (ft_object_with_glyph->warn_if_used) {
            ft_glyph_warn(charcode, glyph_seen_fonts);
        }
        ft_object = ft_object_with_glyph;
    } else {
        //no fallback case
        ft_object = this;
        FT_UInt glyph_index = FT_Get_Char_Index(face, (FT_ULong) charcode);
        if (!glyph_index){
            glyph_seen_fonts.insert((face != nullptr)?face->family_name: nullptr);
            ft_glyph_warn((FT_ULong)charcode, glyph_seen_fonts);
        }
        if (FT_Error error = FT_Load_Glyph(face, glyph_index, flags)) {
            throw_ft_error("Could not load charcode", error);
        }
        FT_Glyph thisGlyph;
        if (FT_Error error = FT_Get_Glyph(face->glyph, &thisGlyph)) {
            throw_ft_error("Could not get glyph", error);
        }
        glyphs.push_back(thisGlyph);
    }
}


bool FT2Font::get_char_fallback_index(FT_ULong charcode, int& index) const
{
    FT_UInt glyph_index = FT_Get_Char_Index(face, charcode);
    if (glyph_index) {
        // -1 means the host has the char and we do not need to fallback
        index = -1;
        return true;
    } else {
        int inner_index = 0;
        bool was_found;

        for (size_t i = 0; i < fallbacks.size(); ++i) {
            // TODO handle recursion somehow!
            was_found = fallbacks[i]->get_char_fallback_index(charcode, inner_index);
            if (was_found) {
                index = i;
                return true;
            }
        }
    }
    return false;
}


bool FT2Font::load_char_with_fallback(FT2Font *&ft_object_with_glyph,
                                      FT_UInt &final_glyph_index,
                                      std::vector<FT_Glyph> &parent_glyphs,
                                      std::unordered_map<long, FT2Font *> &parent_char_to_font,
                                      std::unordered_map<FT_UInt, FT2Font *> &parent_glyph_to_font,
                                      long charcode,
                                      FT_Int32 flags,
                                      FT_Error &charcode_error,
                                      FT_Error &glyph_error,
                                      std::set<FT_String*> &glyph_seen_fonts,
                                      bool override = false)
{
    FT_UInt glyph_index = FT_Get_Char_Index(face, charcode);
    if (!warn_if_used) {
        glyph_seen_fonts.insert(face->family_name);
    }

    if (glyph_index || override) {
        charcode_error = FT_Load_Glyph(face, glyph_index, flags);
        if (charcode_error) {
            return false;
        }
        FT_Glyph thisGlyph;
        glyph_error = FT_Get_Glyph(face->glyph, &thisGlyph);
        if (glyph_error) {
            return false;
        }

        final_glyph_index = glyph_index;

        // cache the result for future
        // need to store this for anytime a character is loaded from a parent
        // FT2Font object or to generate a mapping of individual characters to fonts
        ft_object_with_glyph = this;
        parent_glyph_to_font[final_glyph_index] = this;
        parent_char_to_font[charcode] = this;
        parent_glyphs.push_back(thisGlyph);
        return true;
    }
    else {
        for (auto & fallback : fallbacks) {
            bool was_found = fallback->load_char_with_fallback(
                ft_object_with_glyph, final_glyph_index, parent_glyphs,
                parent_char_to_font, parent_glyph_to_font, charcode, flags,
                charcode_error, glyph_error, glyph_seen_fonts, override);
            if (was_found) {
                return true;
            }
        }
        return false;
    }
}

void FT2Font::load_glyph(FT_UInt glyph_index,
                         FT_Int32 flags,
                         FT2Font *&ft_object,
                         bool fallback = false)
{
    // cache is only for parent FT2Font
    if (fallback && glyph_to_font.find(glyph_index) != glyph_to_font.end()) {
        ft_object = glyph_to_font[glyph_index];
    } else {
        ft_object = this;
    }

    ft_object->load_glyph(glyph_index, flags);
}

void FT2Font::load_glyph(FT_UInt glyph_index, FT_Int32 flags)
{
    if (FT_Error error = FT_Load_Glyph(face, glyph_index, flags)) {
        throw_ft_error("Could not load glyph", error);
    }
    FT_Glyph thisGlyph;
    if (FT_Error error = FT_Get_Glyph(face->glyph, &thisGlyph)) {
        throw_ft_error("Could not get glyph", error);
    }
    glyphs.push_back(thisGlyph);
}

FT_UInt FT2Font::get_char_index(FT_ULong charcode, bool fallback = false)
{
    FT2Font *ft_object = nullptr;
    if (fallback && char_to_font.find(charcode) != char_to_font.end()) {
        // fallback denotes whether we want to search fallback list.
        // should call set_text/load_char_with_fallback to parent FT2Font before
        // wanting to use fallback list here. (since that populates the cache)
        ft_object = char_to_font[charcode];
    } else {
        // set as self
        ft_object = this;
    }

    return FT_Get_Char_Index(ft_object->get_face(), charcode);
}

void FT2Font::get_width_height(long *width, long *height)
{
    *width = advance;
    *height = bbox.yMax - bbox.yMin;
}

long FT2Font::get_descent()
{
    return -bbox.yMin;
}

void FT2Font::get_bitmap_offset(long *x, long *y)
{
    *x = bbox.xMin;
    *y = 0;
}

void FT2Font::draw_glyphs_to_bitmap(bool antialiased)
{
    long width = (bbox.xMax - bbox.xMin) / 64 + 2;
    long height = (bbox.yMax - bbox.yMin) / 64 + 2;

    image.resize(width, height);

    for (auto & glyph : glyphs) {
        FT_Error error = FT_Glyph_To_Bitmap(
            &glyph, antialiased ? FT_RENDER_MODE_NORMAL : FT_RENDER_MODE_MONO, nullptr, 1);
        if (error) {
            throw_ft_error("Could not convert glyph to bitmap", error);
        }

        FT_BitmapGlyph bitmap = (FT_BitmapGlyph)glyph;
        // now, draw to our target surface (convert position)

        // bitmap left and top in pixel, string bbox in subpixel
        FT_Int x = (FT_Int)(bitmap->left - (bbox.xMin * (1. / 64.)));
        FT_Int y = (FT_Int)((bbox.yMax * (1. / 64.)) - bitmap->top + 1);

        image.draw_bitmap(&bitmap->bitmap, x, y);
    }
}

void FT2Font::draw_glyph_to_bitmap(
    FT2Image &im, double x, double y, size_t glyphInd, bool antialiased)
{
    FT_Vector sub_offset = {
        int((x - std::floor(x)) * 64), int((y - std::floor(y)) * 64),
    };

    if (glyphInd >= glyphs.size()) {
        throw std::runtime_error("glyph num is out of range");
    }

    FT_Error error = FT_Glyph_To_Bitmap(
      &glyphs[glyphInd],
      antialiased ? FT_RENDER_MODE_NORMAL : FT_RENDER_MODE_MONO,
      &sub_offset, // additional translation
      1 // destroy image
      );
    if (error) {
        throw_ft_error("Could not convert glyph to bitmap", error);
    }

    FT_BitmapGlyph bitmap = (FT_BitmapGlyph)glyphs[glyphInd];

    im.draw_bitmap(&bitmap->bitmap, std::floor(x) + bitmap->left, std::floor(y) - bitmap->top);
}

void FT2Font::get_glyph_name(unsigned int glyph_number, std::string &buffer,
                             bool fallback = false)
{
    if (fallback && glyph_to_font.find(glyph_number) != glyph_to_font.end()) {
        // cache is only for parent FT2Font
        FT2Font *ft_object = glyph_to_font[glyph_number];
        ft_object->get_glyph_name(glyph_number, buffer, false);
        return;
    }
    if (!FT_HAS_GLYPH_NAMES(face)) {
        /* Note that this generated name must match the name that
           is generated by ttconv in ttfont_CharStrings_getname. */
        auto len = snprintf(buffer.data(), buffer.size(), "uni%08x", glyph_number);
        if (len >= 0) {
            buffer.resize(len);
        } else {
            throw std::runtime_error("Failed to convert glyph to standard name");
        }
    } else {
        if (FT_Error error = FT_Get_Glyph_Name(face, glyph_number, buffer.data(), buffer.size())) {
            throw_ft_error("Could not get glyph names", error);
        }
        auto len = buffer.find('\0');
        if (len != buffer.npos) {
            buffer.resize(len);
        }
    }
}

long FT2Font::get_name_index(char *name)
{
    return FT_Get_Name_Index(face, (FT_String *)name);
}
