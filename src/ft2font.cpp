/* -*- mode: c++; c-basic-offset: 4 -*- */

#include "ft2font.h"
#include "mplutils.h"

#include <algorithm>
#include <cstdio>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

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

FT2Image::FT2Image(unsigned long width, unsigned long height)
    : m_buffer((unsigned char *)calloc(width * height, 1)), m_width(width), m_height(height)
{
}

FT2Image::~FT2Image()
{
    free(m_buffer);
}

void draw_bitmap(
    py::array_t<uint8_t, py::array::c_style> im, FT_Bitmap *bitmap, FT_Int x, FT_Int y)
{
    auto buf = im.mutable_data(0);

    FT_Int image_width = (FT_Int)im.shape(1);
    FT_Int image_height = (FT_Int)im.shape(0);
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
            unsigned char *dst = buf + (i * image_width + x1);
            unsigned char *src = bitmap->buffer + (((i - y_offset) * bitmap->pitch) + x_start);
            for (FT_Int j = x1; j < x2; ++j, ++dst, ++src)
                *dst |= *src;
        }
    } else if (bitmap->pixel_mode == FT_PIXEL_MODE_MONO) {
        for (FT_Int i = y1; i < y2; ++i) {
            unsigned char *dst = buf + (i * image_width + x1);
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

FT2Font::FT2Font(long hinting_factor_, std::vector<FT2Font *> &fallback_list,
                 bool warn_if_used)
    : warn_if_used(warn_if_used), image({1, 1}), face(nullptr), fallbacks(fallback_list),
      hinting_factor(hinting_factor_),
      // set default kerning factor to 0, i.e., no kerning manipulation
      kerning_factor(0)
{
    clear();
}

FT2Font::~FT2Font()
{
    close();
}

void FT2Font::open(FT_Open_Args &open_args)
{
    FT_CHECK(FT_Open_Face, _ft2Library, &open_args, 0, &face);
    if (open_args.stream != nullptr) {
        face->face_flags |= FT_FACE_FLAG_EXTERNAL_STREAM;
    }

    // This allows us to get back to our data if we need it, though it makes a pointer
    // loop, so don't set a free-function for it.
    face->generic.data = this;
    face->generic.finalizer = nullptr;
}

void FT2Font::close()
{
    // This should be idempotent, in case a user manually calls close before the
    // destructor does. Note for example, that PyFT2Font _does_ call this before the
    // base destructor to ensure internal pointers are cleared early enough.

    for (auto & glyph : glyphs) {
        FT_Done_Glyph(glyph);
    }
    glyphs.clear();

    if (face) {
        FT_Done_Face(face);
        face = nullptr;
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
    char_to_font.clear();

    for (auto & fallback : fallbacks) {
        fallback->clear();
    }
}

void FT2Font::set_size(double ptsize, double dpi)
{
    FT_CHECK(
        FT_Set_Char_Size,
        face, (FT_F26Dot6)(ptsize * 64), 0, (FT_UInt)(dpi * hinting_factor), (FT_UInt)dpi);
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
    FT_CHECK(FT_Set_Charmap, face, face->charmaps[i]);
}

void FT2Font::select_charmap(unsigned long i)
{
    FT_CHECK(FT_Select_Charmap, face, (FT_Encoding)i);
}

int FT2Font::get_kerning(FT_UInt left, FT_UInt right, FT_Kerning_Mode mode)
{
    if (!FT_HAS_KERNING(face)) {
        return 0;
    }

    FT_Vector delta;
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

std::vector<raqm_glyph_t> FT2Font::layout(
    std::u32string_view text, FT_Int32 flags,
    std::optional<std::vector<std::string>> features, LanguageType languages,
    std::set<FT_String*>& glyph_seen_fonts)
{
    clear();

    auto rq = raqm_create();
    if (!rq) {
        throw std::runtime_error("failed to compute text layout");
    }
    [[maybe_unused]] auto const& rq_cleanup =
        std::unique_ptr<std::remove_pointer_t<raqm_t>, decltype(&raqm_destroy)>(
            rq, raqm_destroy);

    if (!raqm_set_text(rq, reinterpret_cast<const uint32_t *>(text.data()),
                       text.size()))
    {
        throw std::runtime_error("failed to set text for layout");
    }
    if (!raqm_set_freetype_face(rq, face)) {
        throw std::runtime_error("failed to set text face for layout");
    }
    if (!raqm_set_freetype_load_flags(rq, flags)) {
        throw std::runtime_error("failed to set text flags for layout");
    }
    if (features) {
        for (auto const& feature : *features) {
            if (!raqm_add_font_feature(rq, feature.c_str(), feature.size())) {
                throw std::runtime_error("failed to set font feature {}"_s.format(feature));
            }
        }
    }
    if (languages) {
        for (auto & [lang_str, start, end] : *languages) {
            if (!raqm_set_language(rq, lang_str.c_str(), start, end - start)) {
                throw std::runtime_error(
                    "failed to set language between {} and {} characters "_s
                    "to {!r} for layout"_s.format(
                        start, end, lang_str));
            }
        }
    }
    if (!raqm_layout(rq)) {
        throw std::runtime_error("failed to layout text");
    }

    std::vector<std::pair<size_t, const FT_Face&>> face_substitutions;
    glyph_seen_fonts.insert(face->family_name);

    // Attempt to use fallback fonts if necessary.
    for (auto const& fallback : fallbacks) {
        size_t num_glyphs = 0;
        auto const& rq_glyphs = raqm_get_glyphs(rq, &num_glyphs);
        bool new_fallback_used = false;

        // Sort clusters (n.b. std::map is ordered), as RTL text will be returned in
        // display, not source, order.
        std::map<decltype(raqm_glyph_t::cluster), bool> cluster_missing;
        for (size_t i = 0; i < num_glyphs; i++) {
            auto const& rglyph = rq_glyphs[i];

            // Sometimes multiple glyphs are necessary for a single cluster; if any are
            // not found, we want to "poison" the whole set and keep them missing.
            cluster_missing[rglyph.cluster] |= (rglyph.index == 0);
        }

        for (auto it = cluster_missing.cbegin(); it != cluster_missing.cend(); ) {
            auto [cluster, missing] = *it;
            ++it;  // Early change so we can access the next cluster below.
            if (missing) {
                auto next = (it != cluster_missing.cend()) ? it->first : text.size();
                for (auto i = cluster; i < next; i++) {
                    face_substitutions.emplace_back(i, fallback->face);
                }
                new_fallback_used = true;
            }
        }

        if (!new_fallback_used) {
            // If we never used a fallback, then we're good to go with the existing
            // layout we have already made.
            break;
        }

        // If a fallback was used, then re-attempt the layout with the new fonts.
        if (!fallback->warn_if_used) {
            glyph_seen_fonts.insert(fallback->face->family_name);
        }

        raqm_clear_contents(rq);
        if (!raqm_set_text(rq,
                           reinterpret_cast<const uint32_t *>(text.data()),
                           text.size()))
        {
            throw std::runtime_error("failed to set text for layout");
        }
        if (!raqm_set_freetype_face(rq, face)) {
            throw std::runtime_error("failed to set text face for layout");
        }
        for (auto [cluster, fallback] : face_substitutions) {
            raqm_set_freetype_face_range(rq, fallback, cluster, 1);
        }
        if (!raqm_set_freetype_load_flags(rq, flags)) {
            throw std::runtime_error("failed to set text flags for layout");
        }
        if (features) {
            for (auto const& feature : *features) {
                if (!raqm_add_font_feature(rq, feature.c_str(), feature.size())) {
                    throw std::runtime_error(
                        "failed to set font feature {}"_s.format(feature));
                }
            }
        }
        if (languages) {
            for (auto & [lang_str, start, end] : *languages) {
                if (!raqm_set_language(rq, lang_str.c_str(), start, end - start)) {
                    throw std::runtime_error(
                        "failed to set language between {} and {} characters "_s
                        "to {!r} for layout"_s.format(
                            start, end, lang_str));
                }
            }
        }
        if (!raqm_layout(rq)) {
            throw std::runtime_error("failed to layout text");
        }
    }

    size_t num_glyphs = 0;
    auto const& rq_glyphs = raqm_get_glyphs(rq, &num_glyphs);

    return std::vector<raqm_glyph_t>(rq_glyphs, rq_glyphs + num_glyphs);
}

void FT2Font::set_text(
    std::u32string_view text, double angle, FT_Int32 flags,
    std::optional<std::vector<std::string>> features, LanguageType languages,
    std::vector<double> &xys)
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

    std::set<FT_String*> glyph_seen_fonts;
    auto rq_glyphs = layout(text, flags, features, languages, glyph_seen_fonts);

    bbox.xMin = bbox.yMin = 32000;
    bbox.xMax = bbox.yMax = -32000;

    for (auto const& rglyph : rq_glyphs) {
        // Warn for missing glyphs.
        if (rglyph.index == 0) {
            ft_glyph_warn(text[rglyph.cluster], glyph_seen_fonts);
            continue;
        }
        FT2Font *wrapped_font = static_cast<FT2Font *>(rglyph.ftface->generic.data);
        if (wrapped_font->warn_if_used) {
            ft_glyph_warn(text[rglyph.cluster], glyph_seen_fonts);
        }

        // extract glyph image and store it in our table
        FT_Error error;
        error = FT_Load_Glyph(rglyph.ftface, rglyph.index, flags);
        if (error) {
            throw std::runtime_error("failed to load glyph");
        }
        FT_Glyph thisGlyph;
        error = FT_Get_Glyph(rglyph.ftface->glyph, &thisGlyph);
        if (error) {
            throw std::runtime_error("failed to get glyph");
        }

        pen.x += rglyph.x_offset;
        pen.y += rglyph.y_offset;

        FT_Glyph_Transform(thisGlyph, nullptr, &pen);
        FT_Glyph_Transform(thisGlyph, &matrix, nullptr);
        xys.push_back(pen.x);
        xys.push_back(pen.y);

        FT_BBox glyph_bbox;
        FT_Glyph_Get_CBox(thisGlyph, FT_GLYPH_BBOX_SUBPIXELS, &glyph_bbox);

        bbox.xMin = std::min(bbox.xMin, glyph_bbox.xMin);
        bbox.xMax = std::max(bbox.xMax, glyph_bbox.xMax);
        bbox.yMin = std::min(bbox.yMin, glyph_bbox.yMin);
        bbox.yMax = std::max(bbox.yMax, glyph_bbox.yMax);

        if ((flags & FT_LOAD_NO_HINTING) != 0) {
            pen.x += rglyph.x_advance - rglyph.x_offset;
        } else {
            pen.x += hinting_factor * rglyph.x_advance - rglyph.x_offset;
        }
        pen.y += rglyph.y_advance - rglyph.y_offset;

        glyphs.push_back(thisGlyph);
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
                                                 glyphs, char_to_font,
                                                 charcode, flags, charcode_error, glyph_error,
                                                 glyph_seen_fonts);
        if (!was_found) {
            ft_glyph_warn(charcode, glyph_seen_fonts);
            if (charcode_error) {
                THROW_FT_ERROR("charcode loading", charcode_error);
            }
            else if (glyph_error) {
                THROW_FT_ERROR("charcode loading", glyph_error);
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
        FT_CHECK(FT_Load_Glyph, face, glyph_index, flags);
        FT_Glyph thisGlyph;
        FT_CHECK(FT_Get_Glyph, face->glyph, &thisGlyph);
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
                                      long charcode,
                                      FT_Int32 flags,
                                      FT_Error &charcode_error,
                                      FT_Error &glyph_error,
                                      std::set<FT_String*> &glyph_seen_fonts)
{
    FT_UInt glyph_index = FT_Get_Char_Index(face, charcode);
    if (!warn_if_used) {
        glyph_seen_fonts.insert(face->family_name);
    }

    if (glyph_index) {
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
        parent_char_to_font[charcode] = this;
        parent_glyphs.push_back(thisGlyph);
        return true;
    }
    else {
        for (auto & fallback : fallbacks) {
            bool was_found = fallback->load_char_with_fallback(
                ft_object_with_glyph, final_glyph_index, parent_glyphs,
                parent_char_to_font, charcode, flags,
                charcode_error, glyph_error, glyph_seen_fonts);
            if (was_found) {
                return true;
            }
        }
        return false;
    }
}

void FT2Font::load_glyph(FT_UInt glyph_index, FT_Int32 flags)
{
    FT_CHECK(FT_Load_Glyph, face, glyph_index, flags);
    FT_Glyph thisGlyph;
    FT_CHECK(FT_Get_Glyph, face->glyph, &thisGlyph);
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

std::tuple<long, long> FT2Font::get_width_height()
{
    return {advance, bbox.yMax - bbox.yMin};
}

long FT2Font::get_descent()
{
    return -bbox.yMin;
}

std::tuple<long, long> FT2Font::get_bitmap_offset()
{
    return {bbox.xMin, 0};
}

void FT2Font::draw_glyphs_to_bitmap(bool antialiased)
{
    long width = (bbox.xMax - bbox.xMin) / 64 + 2;
    long height = (bbox.yMax - bbox.yMin) / 64 + 2;

    image = py::array_t<uint8_t>{{height, width}};
    std::memset(image.mutable_data(0), 0, image.nbytes());

    for (auto & glyph: glyphs) {
        FT_CHECK(
            FT_Glyph_To_Bitmap,
            &glyph, antialiased ? FT_RENDER_MODE_NORMAL : FT_RENDER_MODE_MONO, nullptr, 1);
        FT_BitmapGlyph bitmap = (FT_BitmapGlyph)glyph;
        // now, draw to our target surface (convert position)

        // bitmap left and top in pixel, string bbox in subpixel
        FT_Int x = (FT_Int)(bitmap->left - (bbox.xMin * (1. / 64.)));
        FT_Int y = (FT_Int)((bbox.yMax * (1. / 64.)) - bitmap->top + 1);

        draw_bitmap(image, &bitmap->bitmap, x, y);
    }
}

void FT2Font::draw_glyph_to_bitmap(
    py::array_t<uint8_t, py::array::c_style> im,
    int x, int y, size_t glyphInd, bool antialiased)
{
    FT_Vector sub_offset;
    sub_offset.x = 0; // int((xd - (double)x) * 64.0);
    sub_offset.y = 0; // int((yd - (double)y) * 64.0);

    if (glyphInd >= glyphs.size()) {
        throw std::runtime_error("glyph num is out of range");
    }

    FT_CHECK(
        FT_Glyph_To_Bitmap,
        &glyphs[glyphInd],
        antialiased ? FT_RENDER_MODE_NORMAL : FT_RENDER_MODE_MONO,
        &sub_offset, // additional translation
        1); // destroy image
    FT_BitmapGlyph bitmap = (FT_BitmapGlyph)glyphs[glyphInd];

    draw_bitmap(im, &bitmap->bitmap, x + bitmap->left, y);
}

std::string FT2Font::get_glyph_name(unsigned int glyph_number)
{
    std::string buffer;
    buffer.resize(128);

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
        FT_CHECK(FT_Get_Glyph_Name, face, glyph_number, buffer.data(), buffer.size());
        auto len = buffer.find('\0');
        if (len != buffer.npos) {
            buffer.resize(len);
        }
    }

    return buffer;
}

long FT2Font::get_name_index(char *name)
{
    return FT_Get_Name_Index(face, (FT_String *)name);
}
