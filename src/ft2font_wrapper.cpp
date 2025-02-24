#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "ft2font.h"
#include "_enums.h"

#include <set>
#include <sstream>
#include <unordered_map>

namespace py = pybind11;
using namespace pybind11::literals;

template <typename T>
using double_or_ = std::variant<double, T>;

template <typename T>
static T
_double_to_(const char *name, double_or_<T> &var)
{
    if (auto value = std::get_if<double>(&var)) {
        auto api = py::module_::import("matplotlib._api");
        auto warn = api.attr("warn_deprecated");
        warn("since"_a="3.10", "name"_a=name, "obj_type"_a="parameter as float",
             "alternative"_a="int({})"_s.format(name));
        return static_cast<T>(*value);
    } else if (auto value = std::get_if<T>(&var)) {
        return *value;
    } else {
        // pybind11 will have only allowed types that match the variant, so this `else`
        // can't happen. We only have this case because older macOS doesn't support
        // `std::get` and using the conditional `std::get_if` means an `else` to silence
        // compiler warnings about "unhandled" cases.
        throw std::runtime_error("Should not happen");
    }
}

/**********************************************************************
 * Enumerations
 * */

const char *Kerning__doc__ = R"""(
    Kerning modes for `.FT2Font.get_kerning`.

    For more information, see `the FreeType documentation
    <https://freetype.org/freetype2/docs/reference/ft2-glyph_retrieval.html#ft_kerning_mode>`_.

    .. versionadded:: 3.10
)""";

P11X_DECLARE_ENUM(
    "Kerning", "Enum",
    {"DEFAULT", FT_KERNING_DEFAULT},
    {"UNFITTED", FT_KERNING_UNFITTED},
    {"UNSCALED", FT_KERNING_UNSCALED},
);

const char *FaceFlags__doc__ = R"""(
    Flags returned by `FT2Font.face_flags`.

    For more information, see `the FreeType documentation
    <https://freetype.org/freetype2/docs/reference/ft2-face_creation.html#ft_face_flag_xxx>`_.

    .. versionadded:: 3.10
)""";

#ifndef FT_FACE_FLAG_VARIATION  // backcompat: ft 2.9.0.
#define FT_FACE_FLAG_VARIATION (1L << 15)
#endif
#ifndef FT_FACE_FLAG_SVG  // backcompat: ft 2.12.0.
#define FT_FACE_FLAG_SVG (1L << 16)
#endif
#ifndef FT_FACE_FLAG_SBIX  // backcompat: ft 2.12.0.
#define FT_FACE_FLAG_SBIX (1L << 17)
#endif
#ifndef FT_FACE_FLAG_SBIX_OVERLAY  // backcompat: ft 2.12.0.
#define FT_FACE_FLAG_SBIX_OVERLAY (1L << 18)
#endif

enum class FaceFlags : FT_Long {
#define DECLARE_FLAG(name) name = FT_FACE_FLAG_##name
    DECLARE_FLAG(SCALABLE),
    DECLARE_FLAG(FIXED_SIZES),
    DECLARE_FLAG(FIXED_WIDTH),
    DECLARE_FLAG(SFNT),
    DECLARE_FLAG(HORIZONTAL),
    DECLARE_FLAG(VERTICAL),
    DECLARE_FLAG(KERNING),
    DECLARE_FLAG(FAST_GLYPHS),
    DECLARE_FLAG(MULTIPLE_MASTERS),
    DECLARE_FLAG(GLYPH_NAMES),
    DECLARE_FLAG(EXTERNAL_STREAM),
    DECLARE_FLAG(HINTER),
    DECLARE_FLAG(CID_KEYED),
    DECLARE_FLAG(TRICKY),
    DECLARE_FLAG(COLOR),
    DECLARE_FLAG(VARIATION),
    DECLARE_FLAG(SVG),
    DECLARE_FLAG(SBIX),
    DECLARE_FLAG(SBIX_OVERLAY),
#undef DECLARE_FLAG
};

P11X_DECLARE_ENUM(
    "FaceFlags", "Flag",
    {"SCALABLE", FaceFlags::SCALABLE},
    {"FIXED_SIZES", FaceFlags::FIXED_SIZES},
    {"FIXED_WIDTH", FaceFlags::FIXED_WIDTH},
    {"SFNT", FaceFlags::SFNT},
    {"HORIZONTAL", FaceFlags::HORIZONTAL},
    {"VERTICAL", FaceFlags::VERTICAL},
    {"KERNING", FaceFlags::KERNING},
    {"FAST_GLYPHS", FaceFlags::FAST_GLYPHS},
    {"MULTIPLE_MASTERS", FaceFlags::MULTIPLE_MASTERS},
    {"GLYPH_NAMES", FaceFlags::GLYPH_NAMES},
    {"EXTERNAL_STREAM", FaceFlags::EXTERNAL_STREAM},
    {"HINTER", FaceFlags::HINTER},
    {"CID_KEYED", FaceFlags::CID_KEYED},
    {"TRICKY", FaceFlags::TRICKY},
    {"COLOR", FaceFlags::COLOR},
    {"VARIATION", FaceFlags::VARIATION},
    {"SVG", FaceFlags::SVG},
    {"SBIX", FaceFlags::SBIX},
    {"SBIX_OVERLAY", FaceFlags::SBIX_OVERLAY},
);

const char *LoadFlags__doc__ = R"""(
    Flags for `FT2Font.load_char`, `FT2Font.load_glyph`, and `FT2Font.set_text`.

    For more information, see `the FreeType documentation
    <https://freetype.org/freetype2/docs/reference/ft2-glyph_retrieval.html#ft_load_xxx>`_.

    .. versionadded:: 3.10
)""";

#ifndef FT_LOAD_COMPUTE_METRICS  // backcompat: ft 2.6.1.
#define FT_LOAD_COMPUTE_METRICS (1L << 21)
#endif
#ifndef FT_LOAD_BITMAP_METRICS_ONLY  // backcompat: ft 2.7.1.
#define FT_LOAD_BITMAP_METRICS_ONLY (1L << 22)
#endif
#ifndef FT_LOAD_NO_SVG  // backcompat: ft 2.13.1.
#define FT_LOAD_NO_SVG (1L << 24)
#endif

enum class LoadFlags : FT_Int32 {
#define DECLARE_FLAG(name) name = FT_LOAD_##name
    DECLARE_FLAG(DEFAULT),
    DECLARE_FLAG(NO_SCALE),
    DECLARE_FLAG(NO_HINTING),
    DECLARE_FLAG(RENDER),
    DECLARE_FLAG(NO_BITMAP),
    DECLARE_FLAG(VERTICAL_LAYOUT),
    DECLARE_FLAG(FORCE_AUTOHINT),
    DECLARE_FLAG(CROP_BITMAP),
    DECLARE_FLAG(PEDANTIC),
    DECLARE_FLAG(IGNORE_GLOBAL_ADVANCE_WIDTH),
    DECLARE_FLAG(NO_RECURSE),
    DECLARE_FLAG(IGNORE_TRANSFORM),
    DECLARE_FLAG(MONOCHROME),
    DECLARE_FLAG(LINEAR_DESIGN),
    DECLARE_FLAG(NO_AUTOHINT),
    DECLARE_FLAG(COLOR),
    DECLARE_FLAG(COMPUTE_METRICS),
    DECLARE_FLAG(BITMAP_METRICS_ONLY),
    DECLARE_FLAG(NO_SVG),
    DECLARE_FLAG(TARGET_NORMAL),
    DECLARE_FLAG(TARGET_LIGHT),
    DECLARE_FLAG(TARGET_MONO),
    DECLARE_FLAG(TARGET_LCD),
    DECLARE_FLAG(TARGET_LCD_V),
#undef DECLARE_FLAG
};

P11X_DECLARE_ENUM(
    "LoadFlags", "Flag",
    {"DEFAULT", LoadFlags::DEFAULT},
    {"NO_SCALE", LoadFlags::NO_SCALE},
    {"NO_HINTING", LoadFlags::NO_HINTING},
    {"RENDER", LoadFlags::RENDER},
    {"NO_BITMAP", LoadFlags::NO_BITMAP},
    {"VERTICAL_LAYOUT", LoadFlags::VERTICAL_LAYOUT},
    {"FORCE_AUTOHINT", LoadFlags::FORCE_AUTOHINT},
    {"CROP_BITMAP", LoadFlags::CROP_BITMAP},
    {"PEDANTIC", LoadFlags::PEDANTIC},
    {"IGNORE_GLOBAL_ADVANCE_WIDTH", LoadFlags::IGNORE_GLOBAL_ADVANCE_WIDTH},
    {"NO_RECURSE", LoadFlags::NO_RECURSE},
    {"IGNORE_TRANSFORM", LoadFlags::IGNORE_TRANSFORM},
    {"MONOCHROME", LoadFlags::MONOCHROME},
    {"LINEAR_DESIGN", LoadFlags::LINEAR_DESIGN},
    {"NO_AUTOHINT", LoadFlags::NO_AUTOHINT},
    {"COLOR", LoadFlags::COLOR},
    {"COMPUTE_METRICS", LoadFlags::COMPUTE_METRICS},
    {"BITMAP_METRICS_ONLY", LoadFlags::BITMAP_METRICS_ONLY},
    {"NO_SVG", LoadFlags::NO_SVG},
    // These must be unique, but the others can be OR'd together; I don't know if
    // there's any way to really enforce that.
    {"TARGET_NORMAL", LoadFlags::TARGET_NORMAL},
    {"TARGET_LIGHT", LoadFlags::TARGET_LIGHT},
    {"TARGET_MONO", LoadFlags::TARGET_MONO},
    {"TARGET_LCD", LoadFlags::TARGET_LCD},
    {"TARGET_LCD_V", LoadFlags::TARGET_LCD_V},
);

const char *StyleFlags__doc__ = R"""(
    Flags returned by `FT2Font.style_flags`.

    For more information, see `the FreeType documentation
    <https://freetype.org/freetype2/docs/reference/ft2-face_creation.html#ft_style_flag_xxx>`_.

    .. versionadded:: 3.10
)""";

enum class StyleFlags : FT_Long {
#define DECLARE_FLAG(name) name = FT_STYLE_FLAG_##name
    NORMAL = 0,
    DECLARE_FLAG(ITALIC),
    DECLARE_FLAG(BOLD),
#undef DECLARE_FLAG
};

P11X_DECLARE_ENUM(
    "StyleFlags", "Flag",
    {"NORMAL", StyleFlags::NORMAL},
    {"ITALIC", StyleFlags::ITALIC},
    {"BOLD", StyleFlags::BOLD},
);

/**********************************************************************
 * FT2Image
 * */

const char *PyFT2Image__doc__ = R"""(
    An image buffer for drawing glyphs.
)""";

const char *PyFT2Image_init__doc__ = R"""(
    Parameters
    ----------
    width, height : int
        The dimensions of the image buffer.
)""";

const char *PyFT2Image_draw_rect_filled__doc__ = R"""(
    Draw a filled rectangle to the image.

    Parameters
    ----------
    x0, y0, x1, y1 : float
        The bounds of the rectangle from (x0, y0) to (x1, y1).
)""";

static void
PyFT2Image_draw_rect_filled(FT2Image *self,
                            double_or_<long> vx0, double_or_<long> vy0,
                            double_or_<long> vx1, double_or_<long> vy1)
{
    auto x0 = _double_to_<long>("x0", vx0);
    auto y0 = _double_to_<long>("y0", vy0);
    auto x1 = _double_to_<long>("x1", vx1);
    auto y1 = _double_to_<long>("y1", vy1);

    self->draw_rect_filled(x0, y0, x1, y1);
}

/**********************************************************************
 * Glyph
 * */

typedef struct
{
    size_t glyphInd;
    long width;
    long height;
    long horiBearingX;
    long horiBearingY;
    long horiAdvance;
    long linearHoriAdvance;
    long vertBearingX;
    long vertBearingY;
    long vertAdvance;
    FT_BBox bbox;
} PyGlyph;

const char *PyGlyph__doc__ = R"""(
    Information about a single glyph.

    You cannot create instances of this object yourself, but must use
    `.FT2Font.load_char` or `.FT2Font.load_glyph` to generate one. This object may be
    used in a call to `.FT2Font.draw_glyph_to_bitmap`.

    For more information on the various metrics, see `the FreeType documentation
    <https://freetype.org/freetype2/docs/glyphs/glyphs-3.html>`_.
)""";

static PyGlyph *
PyGlyph_from_FT2Font(const FT2Font *font)
{
    const FT_Face &face = font->get_face();
    const long hinting_factor = font->get_hinting_factor();
    const FT_Glyph &glyph = font->get_last_glyph();

    PyGlyph *self = new PyGlyph();

    self->glyphInd = font->get_last_glyph_index();
    FT_Glyph_Get_CBox(glyph, ft_glyph_bbox_subpixels, &self->bbox);

    self->width = face->glyph->metrics.width / hinting_factor;
    self->height = face->glyph->metrics.height;
    self->horiBearingX = face->glyph->metrics.horiBearingX / hinting_factor;
    self->horiBearingY = face->glyph->metrics.horiBearingY;
    self->horiAdvance = face->glyph->metrics.horiAdvance;
    self->linearHoriAdvance = face->glyph->linearHoriAdvance / hinting_factor;
    self->vertBearingX = face->glyph->metrics.vertBearingX;
    self->vertBearingY = face->glyph->metrics.vertBearingY;
    self->vertAdvance = face->glyph->metrics.vertAdvance;

    return self;
}

static py::tuple
PyGlyph_get_bbox(PyGlyph *self)
{
    return py::make_tuple(self->bbox.xMin, self->bbox.yMin,
                          self->bbox.xMax, self->bbox.yMax);
}

/**********************************************************************
 * FT2Font
 * */

struct PyFT2Font
{
    FT2Font *x;
    py::object py_file;
    FT_StreamRec stream;
    py::list fallbacks;

    ~PyFT2Font()
    {
        delete this->x;
    }
};

const char *PyFT2Font__doc__ = R"""(
    An object representing a single font face.

    Outside of the font itself and querying its properties, this object provides methods
    for processing text strings into glyph shapes.

    Commonly, one will use `FT2Font.set_text` to load some glyph metrics and outlines.
    Then `FT2Font.draw_glyphs_to_bitmap` and `FT2Font.get_image` may be used to get a
    rendered form of the loaded string.

    For single characters, `FT2Font.load_char` or `FT2Font.load_glyph` may be used,
    either directly for their return values, or to use `FT2Font.draw_glyph_to_bitmap` or
    `FT2Font.get_path`.

    Useful metrics may be examined via the `Glyph` return values or
    `FT2Font.get_kerning`. Most dimensions are given in 26.6 or 16.6 fixed-point
    integers representing subpixels. Divide these values by 64 to produce floating-point
    pixels.
)""";

static unsigned long
read_from_file_callback(FT_Stream stream, unsigned long offset, unsigned char *buffer,
                        unsigned long count)
{
    PyFT2Font *self = (PyFT2Font *)stream->descriptor.pointer;
    Py_ssize_t n_read = 0;
    try {
        char *tmpbuf;
        auto seek_result = self->py_file.attr("seek")(offset);
        auto read_result = self->py_file.attr("read")(count);
        if (PyBytes_AsStringAndSize(read_result.ptr(), &tmpbuf, &n_read) == -1) {
            throw py::error_already_set();
        }
        memcpy(buffer, tmpbuf, n_read);
    } catch (py::error_already_set &eas) {
        eas.discard_as_unraisable(__func__);
        if (!count) {
            return 1;  // Non-zero signals error, when count == 0.
        }
    }
    return (unsigned long)n_read;
}

static void
close_file_callback(FT_Stream stream)
{
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    PyFT2Font *self = (PyFT2Font *)stream->descriptor.pointer;
    try {
        self->py_file.attr("close")();
    } catch (py::error_already_set &eas) {
        eas.discard_as_unraisable(__func__);
    }
    self->py_file = py::object();
    PyErr_Restore(type, value, traceback);
}

static void
ft_glyph_warn(FT_ULong charcode, std::set<FT_String*> family_names)
{
    std::set<FT_String*>::iterator it = family_names.begin();
    std::stringstream ss;
    ss<<*it;
    while(++it != family_names.end()){
        ss<<", "<<*it;
    }

    auto text_helpers = py::module_::import("matplotlib._text_helpers");
    auto warn_on_missing_glyph = text_helpers.attr("warn_on_missing_glyph");
    warn_on_missing_glyph(charcode, ss.str());
}

const char *PyFT2Font_init__doc__ = R"""(
    Parameters
    ----------
    filename : str or file-like
        The source of the font data in a format (ttf or ttc) that FreeType can read.

    hinting_factor : int, optional
        Must be positive. Used to scale the hinting in the x-direction.

    _fallback_list : list of FT2Font, optional
        A list of FT2Font objects used to find missing glyphs.

        .. warning::
            This API is both private and provisional: do not use it directly.

    _kerning_factor : int, optional
        Used to adjust the degree of kerning.

        .. warning::
            This API is private: do not use it directly.
)""";

static PyFT2Font *
PyFT2Font_init(py::object filename, long hinting_factor = 8,
               std::optional<std::vector<PyFT2Font *>> fallback_list = std::nullopt,
               int kerning_factor = 0)
{
    if (hinting_factor <= 0) {
        throw py::value_error("hinting_factor must be greater than 0");
    }

    PyFT2Font *self = new PyFT2Font();
    self->x = nullptr;
    memset(&self->stream, 0, sizeof(FT_StreamRec));
    self->stream.base = nullptr;
    self->stream.size = 0x7fffffff;  // Unknown size.
    self->stream.pos = 0;
    self->stream.descriptor.pointer = self;
    self->stream.read = &read_from_file_callback;
    FT_Open_Args open_args;
    memset((void *)&open_args, 0, sizeof(FT_Open_Args));
    open_args.flags = FT_OPEN_STREAM;
    open_args.stream = &self->stream;

    std::vector<FT2Font *> fallback_fonts;
    if (fallback_list) {
        // go through fallbacks to add them to our lists
        for (auto item : *fallback_list) {
            self->fallbacks.append(item);
            // Also (locally) cache the underlying FT2Font objects. As long as
            // the Python objects are kept alive, these pointer are good.
            FT2Font *fback = item->x;
            fallback_fonts.push_back(fback);
        }
    }

    if (py::isinstance<py::bytes>(filename) || py::isinstance<py::str>(filename)) {
        self->py_file = py::module_::import("io").attr("open")(filename, "rb");
        self->stream.close = &close_file_callback;
    } else {
        try {
            // This will catch various issues:
            // 1. `read` not being an attribute.
            // 2. `read` raising an error.
            // 3. `read` returning something other than `bytes`.
            auto data = filename.attr("read")(0).cast<py::bytes>();
        } catch (const std::exception&) {
            throw py::type_error(
                "First argument must be a path to a font file or a binary-mode file object");
        }
        self->py_file = filename;
        self->stream.close = nullptr;
    }

    self->x = new FT2Font(open_args, hinting_factor, fallback_fonts, ft_glyph_warn);

    self->x->set_kerning_factor(kerning_factor);

    return self;
}

const char *PyFT2Font_clear__doc__ =
    "Clear all the glyphs, reset for a new call to `.set_text`.";

static void
PyFT2Font_clear(PyFT2Font *self)
{
    self->x->clear();
}

const char *PyFT2Font_set_size__doc__ = R"""(
    Set the size of the text.

    Parameters
    ----------
    ptsize : float
        The size of the text in points.
    dpi : float
        The DPI used for rendering the text.
)""";

static void
PyFT2Font_set_size(PyFT2Font *self, double ptsize, double dpi)
{
    self->x->set_size(ptsize, dpi);
}

const char *PyFT2Font_set_charmap__doc__ = R"""(
    Make the i-th charmap current.

    For more details on character mapping, see the `FreeType documentation
    <https://freetype.org/freetype2/docs/reference/ft2-character_mapping.html>`_.

    Parameters
    ----------
    i : int
        The charmap number in the range [0, `.num_charmaps`).

    See Also
    --------
    .num_charmaps
    .select_charmap
    .get_charmap
)""";

static void
PyFT2Font_set_charmap(PyFT2Font *self, int i)
{
    self->x->set_charmap(i);
}

const char *PyFT2Font_select_charmap__doc__ = R"""(
    Select a charmap by its FT_Encoding number.

    For more details on character mapping, see the `FreeType documentation
    <https://freetype.org/freetype2/docs/reference/ft2-character_mapping.html>`_.

    Parameters
    ----------
    i : int
        The charmap in the form defined by FreeType:
        https://freetype.org/freetype2/docs/reference/ft2-character_mapping.html#ft_encoding

    See Also
    --------
    .set_charmap
    .get_charmap
)""";

static void
PyFT2Font_select_charmap(PyFT2Font *self, unsigned long i)
{
    self->x->select_charmap(i);
}

const char *PyFT2Font_get_kerning__doc__ = R"""(
    Get the kerning between two glyphs.

    Parameters
    ----------
    left, right : int
        The glyph indices. Note these are not characters nor character codes.
        Use `.get_char_index` to convert character codes to glyph indices.

    mode : Kerning
        A kerning mode constant:

        - ``DEFAULT``  - Return scaled and grid-fitted kerning distances.
        - ``UNFITTED`` - Return scaled but un-grid-fitted kerning distances.
        - ``UNSCALED`` - Return the kerning vector in original font units.

        .. versionchanged:: 3.10
            This now takes a `.ft2font.Kerning` value instead of an `int`.

    Returns
    -------
    int
        The kerning adjustment between the two glyphs.
)""";

static int
PyFT2Font_get_kerning(PyFT2Font *self, FT_UInt left, FT_UInt right,
                      std::variant<FT_Kerning_Mode, FT_UInt> mode_or_int)
{
    bool fallback = true;
    FT_Kerning_Mode mode;

    if (auto value = std::get_if<FT_UInt>(&mode_or_int)) {
        auto api = py::module_::import("matplotlib._api");
        auto warn = api.attr("warn_deprecated");
        warn("since"_a="3.10", "name"_a="mode", "obj_type"_a="parameter as int",
             "alternative"_a="Kerning enum values");
        mode = static_cast<FT_Kerning_Mode>(*value);
    } else if (auto value = std::get_if<FT_Kerning_Mode>(&mode_or_int)) {
        mode = *value;
    } else {
        // NOTE: this can never happen as pybind11 would have checked the type in the
        // Python wrapper before calling this function, but we need to keep the
        // std::get_if instead of std::get for macOS 10.12 compatibility.
        throw py::type_error("mode must be Kerning or int");
    }

    return self->x->get_kerning(left, right, mode, fallback);
}

const char *PyFT2Font_get_fontmap__doc__ = R"""(
    Get a mapping between characters and the font that includes them.

    .. warning::
        This API uses the fallback list and is both private and provisional: do not use
        it directly.

    Parameters
    ----------
    text : str
        The characters for which to find fonts.

    Returns
    -------
    dict[str, FT2Font]
        A dictionary mapping unicode characters to `.FT2Font` objects.
)""";

static py::dict
PyFT2Font_get_fontmap(PyFT2Font *self, std::u32string text)
{
    std::set<FT_ULong> codepoints;

    py::dict char_to_font;
    for (auto code : text) {
        if (!codepoints.insert(code).second) {
            continue;
        }

        py::object target_font;
        int index;
        if (self->x->get_char_fallback_index(code, index)) {
            if (index >= 0) {
                target_font = self->fallbacks[index];
            } else {
                target_font = py::cast(self);
            }
        } else {
            // TODO Handle recursion!
            target_font = py::cast(self);
        }

        auto key = py::cast(std::u32string(1, code));
        char_to_font[key] = target_font;
    }
    return char_to_font;
}

const char *PyFT2Font_set_text__doc__ = R"""(
    Set the text *string* and *angle*.

    You must call this before `.draw_glyphs_to_bitmap`.

    Parameters
    ----------
    string : str
        The text to prepare rendering information for.
    angle : float
        The angle at which to render the supplied text.
    flags : LoadFlags, default: `.LoadFlags.FORCE_AUTOHINT`
        Any bitwise-OR combination of the `.LoadFlags` flags.

        .. versionchanged:: 3.10
            This now takes an `.ft2font.LoadFlags` instead of an int.

    Returns
    -------
    np.ndarray[double]
        A sequence of x,y glyph positions in 26.6 subpixels; divide by 64 for pixels.
)""";

static py::array_t<double>
PyFT2Font_set_text(PyFT2Font *self, std::u32string_view text, double angle = 0.0,
                   std::variant<LoadFlags, FT_Int32> flags_or_int = LoadFlags::FORCE_AUTOHINT)
{
    std::vector<double> xys;
    LoadFlags flags;

    if (auto value = std::get_if<FT_Int32>(&flags_or_int)) {
        auto api = py::module_::import("matplotlib._api");
        auto warn = api.attr("warn_deprecated");
        warn("since"_a="3.10", "name"_a="flags", "obj_type"_a="parameter as int",
             "alternative"_a="LoadFlags enum values");
        flags = static_cast<LoadFlags>(*value);
    } else if (auto value = std::get_if<LoadFlags>(&flags_or_int)) {
        flags = *value;
    } else {
        // NOTE: this can never happen as pybind11 would have checked the type in the
        // Python wrapper before calling this function, but we need to keep the
        // std::get_if instead of std::get for macOS 10.12 compatibility.
        throw py::type_error("flags must be LoadFlags or int");
    }

    self->x->set_text(text, angle, static_cast<FT_Int32>(flags), xys);

    py::ssize_t dims[] = { static_cast<py::ssize_t>(xys.size()) / 2, 2 };
    py::array_t<double> result(dims);
    if (xys.size() > 0) {
        memcpy(result.mutable_data(), xys.data(), result.nbytes());
    }
    return result;
}

const char *PyFT2Font_get_num_glyphs__doc__ = "Return the number of loaded glyphs.";

static size_t
PyFT2Font_get_num_glyphs(PyFT2Font *self)
{
    return self->x->get_num_glyphs();
}

const char *PyFT2Font_load_char__doc__ = R"""(
    Load character in current fontfile and set glyph.

    Parameters
    ----------
    charcode : int
        The character code to prepare rendering information for. This code must be in
        the charmap, or else a ``.notdef`` glyph may be returned instead.
    flags : LoadFlags, default: `.LoadFlags.FORCE_AUTOHINT`
        Any bitwise-OR combination of the `.LoadFlags` flags.

        .. versionchanged:: 3.10
            This now takes an `.ft2font.LoadFlags` instead of an int.

    Returns
    -------
    Glyph
        The glyph information corresponding to the specified character.

    See Also
    --------
    .load_glyph
    .select_charmap
    .set_charmap
)""";

static PyGlyph *
PyFT2Font_load_char(PyFT2Font *self, long charcode,
                    std::variant<LoadFlags, FT_Int32> flags_or_int = LoadFlags::FORCE_AUTOHINT)
{
    bool fallback = true;
    FT2Font *ft_object = nullptr;
    LoadFlags flags;

    if (auto value = std::get_if<FT_Int32>(&flags_or_int)) {
        auto api = py::module_::import("matplotlib._api");
        auto warn = api.attr("warn_deprecated");
        warn("since"_a="3.10", "name"_a="flags", "obj_type"_a="parameter as int",
             "alternative"_a="LoadFlags enum values");
        flags = static_cast<LoadFlags>(*value);
    } else if (auto value = std::get_if<LoadFlags>(&flags_or_int)) {
        flags = *value;
    } else {
        // NOTE: this can never happen as pybind11 would have checked the type in the
        // Python wrapper before calling this function, but we need to keep the
        // std::get_if instead of std::get for macOS 10.12 compatibility.
        throw py::type_error("flags must be LoadFlags or int");
    }

    self->x->load_char(charcode, static_cast<FT_Int32>(flags), ft_object, fallback);

    return PyGlyph_from_FT2Font(ft_object);
}

const char *PyFT2Font_load_glyph__doc__ = R"""(
    Load glyph index in current fontfile and set glyph.

    Note that the glyph index is specific to a font, and not universal like a Unicode
    code point.

    Parameters
    ----------
    glyph_index : int
        The glyph index to prepare rendering information for.
    flags : LoadFlags, default: `.LoadFlags.FORCE_AUTOHINT`
        Any bitwise-OR combination of the `.LoadFlags` flags.

        .. versionchanged:: 3.10
            This now takes an `.ft2font.LoadFlags` instead of an int.

    Returns
    -------
    Glyph
        The glyph information corresponding to the specified index.

    See Also
    --------
    .load_char
)""";

static PyGlyph *
PyFT2Font_load_glyph(PyFT2Font *self, FT_UInt glyph_index,
                     std::variant<LoadFlags, FT_Int32> flags_or_int = LoadFlags::FORCE_AUTOHINT)
{
    bool fallback = true;
    FT2Font *ft_object = nullptr;
    LoadFlags flags;

    if (auto value = std::get_if<FT_Int32>(&flags_or_int)) {
        auto api = py::module_::import("matplotlib._api");
        auto warn = api.attr("warn_deprecated");
        warn("since"_a="3.10", "name"_a="flags", "obj_type"_a="parameter as int",
             "alternative"_a="LoadFlags enum values");
        flags = static_cast<LoadFlags>(*value);
    } else if (auto value = std::get_if<LoadFlags>(&flags_or_int)) {
        flags = *value;
    } else {
        // NOTE: this can never happen as pybind11 would have checked the type in the
        // Python wrapper before calling this function, but we need to keep the
        // std::get_if instead of std::get for macOS 10.12 compatibility.
        throw py::type_error("flags must be LoadFlags or int");
    }

    self->x->load_glyph(glyph_index, static_cast<FT_Int32>(flags), ft_object, fallback);

    return PyGlyph_from_FT2Font(ft_object);
}

const char *PyFT2Font_get_width_height__doc__ = R"""(
    Get the dimensions of the current string set by `.set_text`.

    The rotation of the string is accounted for.

    Returns
    -------
    width, height : float
        The width and height in 26.6 subpixels of the current string. To get width and
        height in pixels, divide these values by 64.

    See Also
    --------
    .get_bitmap_offset
    .get_descent
)""";

static py::tuple
PyFT2Font_get_width_height(PyFT2Font *self)
{
    long width, height;

    self->x->get_width_height(&width, &height);

    return py::make_tuple(width, height);
}

const char *PyFT2Font_get_bitmap_offset__doc__ = R"""(
    Get the (x, y) offset for the bitmap if ink hangs left or below (0, 0).

    Since Matplotlib only supports left-to-right text, y is always 0.

    Returns
    -------
    x, y : float
        The x and y offset in 26.6 subpixels of the bitmap. To get x and y in pixels,
        divide these values by 64.

    See Also
    --------
    .get_width_height
    .get_descent
)""";

static py::tuple
PyFT2Font_get_bitmap_offset(PyFT2Font *self)
{
    long x, y;

    self->x->get_bitmap_offset(&x, &y);

    return py::make_tuple(x, y);
}

const char *PyFT2Font_get_descent__doc__ = R"""(
    Get the descent of the current string set by `.set_text`.

    The rotation of the string is accounted for.

    Returns
    -------
    int
        The descent in 26.6 subpixels of the bitmap. To get the descent in pixels,
        divide these values by 64.

    See Also
    --------
    .get_bitmap_offset
    .get_width_height
)""";

static long
PyFT2Font_get_descent(PyFT2Font *self)
{
    return self->x->get_descent();
}

const char *PyFT2Font_draw_glyphs_to_bitmap__doc__ = R"""(
    Draw the glyphs that were loaded by `.set_text` to the bitmap.

    The bitmap size will be automatically set to include the glyphs.

    Parameters
    ----------
    antialiased : bool, default: True
        Whether to render glyphs 8-bit antialiased or in pure black-and-white.

    See Also
    --------
    .draw_glyph_to_bitmap
)""";

static void
PyFT2Font_draw_glyphs_to_bitmap(PyFT2Font *self, bool antialiased = true)
{
    self->x->draw_glyphs_to_bitmap(antialiased);
}

const char *PyFT2Font_draw_glyph_to_bitmap__doc__ = R"""(
    Draw a single glyph to the bitmap at pixel locations x, y.

    Note it is your responsibility to create the image manually with the correct size
    before this call is made.

    If you want automatic layout, use `.set_text` in combinations with
    `.draw_glyphs_to_bitmap`. This function is instead intended for people who want to
    render individual glyphs (e.g., returned by `.load_char`) at precise locations.

    Parameters
    ----------
    image : FT2Image
        The image buffer on which to draw the glyph.
    x, y : int
        The pixel location at which to draw the glyph.
    glyph : Glyph
        The glyph to draw.
    antialiased : bool, default: True
        Whether to render glyphs 8-bit antialiased or in pure black-and-white.

    See Also
    --------
    .draw_glyphs_to_bitmap
)""";

static void
PyFT2Font_draw_glyph_to_bitmap(PyFT2Font *self, FT2Image &image,
                               double_or_<int> vxd, double_or_<int> vyd,
                               PyGlyph *glyph, bool antialiased = true)
{
    auto xd = _double_to_<int>("x", vxd);
    auto yd = _double_to_<int>("y", vyd);

    self->x->draw_glyph_to_bitmap(image, xd, yd, glyph->glyphInd, antialiased);
}

const char *PyFT2Font_get_glyph_name__doc__ = R"""(
    Retrieve the ASCII name of a given glyph *index* in a face.

    Due to Matplotlib's internal design, for fonts that do not contain glyph names (per
    ``FT_FACE_FLAG_GLYPH_NAMES``), this returns a made-up name which does *not*
    roundtrip through `.get_name_index`.

    Parameters
    ----------
    index : int
        The glyph number to query.

    Returns
    -------
    str
        The name of the glyph, or if the font does not contain names, a name synthesized
        by Matplotlib.

    See Also
    --------
    .get_name_index
)""";

static py::str
PyFT2Font_get_glyph_name(PyFT2Font *self, unsigned int glyph_number)
{
    std::string buffer;
    bool fallback = true;

    buffer.resize(128);
    self->x->get_glyph_name(glyph_number, buffer, fallback);
    return buffer;
}

const char *PyFT2Font_get_charmap__doc__ = R"""(
    Return a mapping of character codes to glyph indices in the font.

    The charmap is Unicode by default, but may be changed by `.set_charmap` or
    `.select_charmap`.

    Returns
    -------
    dict[int, int]
        A dictionary of the selected charmap mapping character codes to their
        corresponding glyph indices.
)""";

static py::dict
PyFT2Font_get_charmap(PyFT2Font *self)
{
    py::dict charmap;
    FT_UInt index;
    FT_ULong code = FT_Get_First_Char(self->x->get_face(), &index);
    while (index != 0) {
        charmap[py::cast(code)] = py::cast(index);
        code = FT_Get_Next_Char(self->x->get_face(), code, &index);
    }
    return charmap;
}

const char *PyFT2Font_get_char_index__doc__ = R"""(
    Return the glyph index corresponding to a character code point.

    Parameters
    ----------
    codepoint : int
        A character code point in the current charmap (which defaults to Unicode.)

    Returns
    -------
    int
        The corresponding glyph index.

    See Also
    --------
    .set_charmap
    .select_charmap
    .get_glyph_name
    .get_name_index
)""";

static FT_UInt
PyFT2Font_get_char_index(PyFT2Font *self, FT_ULong ccode)
{
    bool fallback = true;

    return self->x->get_char_index(ccode, fallback);
}

const char *PyFT2Font_get_sfnt__doc__ = R"""(
    Load the entire SFNT names table.

    Returns
    -------
    dict[tuple[int, int, int, int], bytes]
        The SFNT names table; the dictionary keys are tuples of:

            (platform-ID, ISO-encoding-scheme, language-code, description)

        and the values are the direct information from the font table.
)""";

static py::dict
PyFT2Font_get_sfnt(PyFT2Font *self)
{
    if (!(self->x->get_face()->face_flags & FT_FACE_FLAG_SFNT)) {
        throw py::value_error("No SFNT name table");
    }

    size_t count = FT_Get_Sfnt_Name_Count(self->x->get_face());

    py::dict names;

    for (FT_UInt j = 0; j < count; ++j) {
        FT_SfntName sfnt;
        FT_Error error = FT_Get_Sfnt_Name(self->x->get_face(), j, &sfnt);

        if (error) {
            throw py::value_error("Could not get SFNT name");
        }

        auto key = py::make_tuple(
            sfnt.platform_id, sfnt.encoding_id, sfnt.language_id, sfnt.name_id);
        auto val = py::bytes(reinterpret_cast<const char *>(sfnt.string),
                             sfnt.string_len);
        names[key] = val;
    }

    return names;
}

const char *PyFT2Font_get_name_index__doc__ = R"""(
    Return the glyph index of a given glyph *name*.

    Parameters
    ----------
    name : str
        The name of the glyph to query.

    Returns
    -------
    int
        The corresponding glyph index; 0 means 'undefined character code'.

    See Also
    --------
    .get_char_index
    .get_glyph_name
)""";

static long
PyFT2Font_get_name_index(PyFT2Font *self, char *glyphname)
{
    return self->x->get_name_index(glyphname);
}

const char *PyFT2Font_get_ps_font_info__doc__ = R"""(
    Return the information in the PS Font Info structure.

    For more information, see the `FreeType documentation on this structure
    <https://freetype.org/freetype2/docs/reference/ft2-type1_tables.html#ps_fontinforec>`_.

    Returns
    -------
    version : str
    notice : str
    full_name : str
    family_name : str
    weight : str
    italic_angle : int
    is_fixed_pitch : bool
    underline_position : int
    underline_thickness : int
)""";

static py::tuple
PyFT2Font_get_ps_font_info(PyFT2Font *self)
{
    PS_FontInfoRec fontinfo;

    FT_Error error = FT_Get_PS_Font_Info(self->x->get_face(), &fontinfo);
    if (error) {
        throw py::value_error("Could not get PS font info");
    }

    return py::make_tuple(
        fontinfo.version ? fontinfo.version : "",
        fontinfo.notice ? fontinfo.notice : "",
        fontinfo.full_name ? fontinfo.full_name : "",
        fontinfo.family_name ? fontinfo.family_name : "",
        fontinfo.weight ? fontinfo.weight : "",
        fontinfo.italic_angle,
        fontinfo.is_fixed_pitch,
        fontinfo.underline_position,
        fontinfo.underline_thickness);
}

const char *PyFT2Font_get_sfnt_table__doc__ = R"""(
    Return one of the SFNT tables.

    Parameters
    ----------
    name : {"head", "maxp", "OS/2", "hhea", "vhea", "post", "pclt"}
        Which table to return.

    Returns
    -------
    dict[str, Any]
        The corresponding table; for more information, see `the FreeType documentation
        <https://freetype.org/freetype2/docs/reference/ft2-truetype_tables.html>`_.
)""";

static std::optional<py::dict>
PyFT2Font_get_sfnt_table(PyFT2Font *self, std::string tagname)
{
    FT_Sfnt_Tag tag;
    const std::unordered_map<std::string, FT_Sfnt_Tag> names = {
        {"head", FT_SFNT_HEAD},
        {"maxp", FT_SFNT_MAXP},
        {"OS/2", FT_SFNT_OS2},
        {"hhea", FT_SFNT_HHEA},
        {"vhea", FT_SFNT_VHEA},
        {"post", FT_SFNT_POST},
        {"pclt", FT_SFNT_PCLT},
    };

    try {
        tag = names.at(tagname);
    } catch (const std::out_of_range&) {
        return std::nullopt;
    }

    void *table = FT_Get_Sfnt_Table(self->x->get_face(), tag);
    if (!table) {
        return std::nullopt;
    }

    switch (tag) {
    case FT_SFNT_HEAD: {
        auto t = static_cast<TT_Header *>(table);
        return py::dict(
            "version"_a=py::make_tuple(FIXED_MAJOR(t->Table_Version),
                                       FIXED_MINOR(t->Table_Version)),
            "fontRevision"_a=py::make_tuple(FIXED_MAJOR(t->Font_Revision),
                                            FIXED_MINOR(t->Font_Revision)),
            "checkSumAdjustment"_a=t->CheckSum_Adjust,
            "magicNumber"_a=t->Magic_Number,
            "flags"_a=t->Flags,
            "unitsPerEm"_a=t->Units_Per_EM,
            // FreeType 2.6.1 defines these two timestamps as FT_Long, but they should
            // be unsigned (fixed in 2.10.0):
            // https://gitlab.freedesktop.org/freetype/freetype/-/commit/3e8ec291ffcfa03c8ecba1cdbfaa55f5577f5612
            // It's actually read from the file structure as two 32-bit values, so we
            // need to cast down in size to prevent sign extension from producing huge
            // 64-bit values.
            "created"_a=py::make_tuple(static_cast<unsigned int>(t->Created[0]),
                                       static_cast<unsigned int>(t->Created[1])),
            "modified"_a=py::make_tuple(static_cast<unsigned int>(t->Modified[0]),
                                        static_cast<unsigned int>(t->Modified[1])),
            "xMin"_a=t->xMin,
            "yMin"_a=t->yMin,
            "xMax"_a=t->xMax,
            "yMax"_a=t->yMax,
            "macStyle"_a=t->Mac_Style,
            "lowestRecPPEM"_a=t->Lowest_Rec_PPEM,
            "fontDirectionHint"_a=t->Font_Direction,
            "indexToLocFormat"_a=t->Index_To_Loc_Format,
            "glyphDataFormat"_a=t->Glyph_Data_Format);
    }
    case FT_SFNT_MAXP: {
        auto t = static_cast<TT_MaxProfile *>(table);
        return py::dict(
            "version"_a=py::make_tuple(FIXED_MAJOR(t->version),
                                       FIXED_MINOR(t->version)),
            "numGlyphs"_a=t->numGlyphs,
            "maxPoints"_a=t->maxPoints,
            "maxContours"_a=t->maxContours,
            "maxComponentPoints"_a=t->maxCompositePoints,
            "maxComponentContours"_a=t->maxCompositeContours,
            "maxZones"_a=t->maxZones,
            "maxTwilightPoints"_a=t->maxTwilightPoints,
            "maxStorage"_a=t->maxStorage,
            "maxFunctionDefs"_a=t->maxFunctionDefs,
            "maxInstructionDefs"_a=t->maxInstructionDefs,
            "maxStackElements"_a=t->maxStackElements,
            "maxSizeOfInstructions"_a=t->maxSizeOfInstructions,
            "maxComponentElements"_a=t->maxComponentElements,
            "maxComponentDepth"_a=t->maxComponentDepth);
    }
    case FT_SFNT_OS2: {
        auto t = static_cast<TT_OS2 *>(table);
        return py::dict(
            "version"_a=t->version,
            "xAvgCharWidth"_a=t->xAvgCharWidth,
            "usWeightClass"_a=t->usWeightClass,
            "usWidthClass"_a=t->usWidthClass,
            "fsType"_a=t->fsType,
            "ySubscriptXSize"_a=t->ySubscriptXSize,
            "ySubscriptYSize"_a=t->ySubscriptYSize,
            "ySubscriptXOffset"_a=t->ySubscriptXOffset,
            "ySubscriptYOffset"_a=t->ySubscriptYOffset,
            "ySuperscriptXSize"_a=t->ySuperscriptXSize,
            "ySuperscriptYSize"_a=t->ySuperscriptYSize,
            "ySuperscriptXOffset"_a=t->ySuperscriptXOffset,
            "ySuperscriptYOffset"_a=t->ySuperscriptYOffset,
            "yStrikeoutSize"_a=t->yStrikeoutSize,
            "yStrikeoutPosition"_a=t->yStrikeoutPosition,
            "sFamilyClass"_a=t->sFamilyClass,
            "panose"_a=py::bytes(reinterpret_cast<const char *>(t->panose), 10),
            "ulCharRange"_a=py::make_tuple(t->ulUnicodeRange1, t->ulUnicodeRange2,
                                           t->ulUnicodeRange3, t->ulUnicodeRange4),
            "achVendID"_a=py::bytes(reinterpret_cast<const char *>(t->achVendID), 4),
            "fsSelection"_a=t->fsSelection,
            "fsFirstCharIndex"_a=t->usFirstCharIndex,
            "fsLastCharIndex"_a=t->usLastCharIndex);
    }
    case FT_SFNT_HHEA: {
        auto t = static_cast<TT_HoriHeader *>(table);
        return py::dict(
            "version"_a=py::make_tuple(FIXED_MAJOR(t->Version),
                                       FIXED_MINOR(t->Version)),
            "ascent"_a=t->Ascender,
            "descent"_a=t->Descender,
            "lineGap"_a=t->Line_Gap,
            "advanceWidthMax"_a=t->advance_Width_Max,
            "minLeftBearing"_a=t->min_Left_Side_Bearing,
            "minRightBearing"_a=t->min_Right_Side_Bearing,
            "xMaxExtent"_a=t->xMax_Extent,
            "caretSlopeRise"_a=t->caret_Slope_Rise,
            "caretSlopeRun"_a=t->caret_Slope_Run,
            "caretOffset"_a=t->caret_Offset,
            "metricDataFormat"_a=t->metric_Data_Format,
            "numOfLongHorMetrics"_a=t->number_Of_HMetrics);
    }
    case FT_SFNT_VHEA: {
        auto t = static_cast<TT_VertHeader *>(table);
        return py::dict(
            "version"_a=py::make_tuple(FIXED_MAJOR(t->Version),
                                       FIXED_MINOR(t->Version)),
            "vertTypoAscender"_a=t->Ascender,
            "vertTypoDescender"_a=t->Descender,
            "vertTypoLineGap"_a=t->Line_Gap,
            "advanceHeightMax"_a=t->advance_Height_Max,
            "minTopSideBearing"_a=t->min_Top_Side_Bearing,
            "minBottomSizeBearing"_a=t->min_Bottom_Side_Bearing,
            "yMaxExtent"_a=t->yMax_Extent,
            "caretSlopeRise"_a=t->caret_Slope_Rise,
            "caretSlopeRun"_a=t->caret_Slope_Run,
            "caretOffset"_a=t->caret_Offset,
            "metricDataFormat"_a=t->metric_Data_Format,
            "numOfLongVerMetrics"_a=t->number_Of_VMetrics);
    }
    case FT_SFNT_POST: {
        auto t = static_cast<TT_Postscript *>(table);
        return py::dict(
            "format"_a=py::make_tuple(FIXED_MAJOR(t->FormatType),
                                      FIXED_MINOR(t->FormatType)),
            "italicAngle"_a=py::make_tuple(FIXED_MAJOR(t->italicAngle),
                                           FIXED_MINOR(t->italicAngle)),
            "underlinePosition"_a=t->underlinePosition,
            "underlineThickness"_a=t->underlineThickness,
            "isFixedPitch"_a=t->isFixedPitch,
            "minMemType42"_a=t->minMemType42,
            "maxMemType42"_a=t->maxMemType42,
            "minMemType1"_a=t->minMemType1,
            "maxMemType1"_a=t->maxMemType1);
    }
    case FT_SFNT_PCLT: {
        auto t = static_cast<TT_PCLT *>(table);
        return py::dict(
            "version"_a=py::make_tuple(FIXED_MAJOR(t->Version),
                                       FIXED_MINOR(t->Version)),
            "fontNumber"_a=t->FontNumber,
            "pitch"_a=t->Pitch,
            "xHeight"_a=t->xHeight,
            "style"_a=t->Style,
            "typeFamily"_a=t->TypeFamily,
            "capHeight"_a=t->CapHeight,
            "symbolSet"_a=t->SymbolSet,
            "typeFace"_a=py::bytes(reinterpret_cast<const char *>(t->TypeFace), 16),
            "characterComplement"_a=py::bytes(
                reinterpret_cast<const char *>(t->CharacterComplement), 8),
            "strokeWeight"_a=t->StrokeWeight,
            "widthType"_a=t->WidthType,
            "serifStyle"_a=t->SerifStyle);
    }
    default:
        return std::nullopt;
    }
}

const char *PyFT2Font_get_path__doc__ = R"""(
    Get the path data from the currently loaded glyph.

    Returns
    -------
    vertices : np.ndarray[double]
        The (N, 2) array of vertices describing the current glyph.
    codes : np.ndarray[np.uint8]
        The (N, ) array of codes corresponding to the vertices.

    See Also
    --------
    .get_image
    .load_char
    .load_glyph
    .set_text
)""";

static py::tuple
PyFT2Font_get_path(PyFT2Font *self)
{
    std::vector<double> vertices;
    std::vector<unsigned char> codes;

    self->x->get_path(vertices, codes);

    py::ssize_t length = codes.size();
    py::ssize_t vertices_dims[2] = { length, 2 };
    py::array_t<double> vertices_arr(vertices_dims);
    if (length > 0) {
        memcpy(vertices_arr.mutable_data(), vertices.data(), vertices_arr.nbytes());
    }
    py::ssize_t codes_dims[1] = { length };
    py::array_t<unsigned char> codes_arr(codes_dims);
    if (length > 0) {
        memcpy(codes_arr.mutable_data(), codes.data(), codes_arr.nbytes());
    }

    return py::make_tuple(vertices_arr, codes_arr);
}

const char *PyFT2Font_get_image__doc__ = R"""(
    Return the underlying image buffer for this font object.

    Returns
    -------
    np.ndarray[int]

    See Also
    --------
    .get_path
)""";

static py::array
PyFT2Font_get_image(PyFT2Font *self)
{
    FT2Image &im = self->x->get_image();
    py::ssize_t dims[] = {
        static_cast<py::ssize_t>(im.get_height()),
        static_cast<py::ssize_t>(im.get_width())
    };
    return py::array_t<unsigned char>(dims, im.get_buffer());
}

static const char *
PyFT2Font_postscript_name(PyFT2Font *self)
{
    const char *ps_name = FT_Get_Postscript_Name(self->x->get_face());
    if (ps_name == nullptr) {
        ps_name = "UNAVAILABLE";
    }

    return ps_name;
}

static FT_Long
PyFT2Font_num_faces(PyFT2Font *self)
{
    return self->x->get_face()->num_faces;
}

static const char *
PyFT2Font_family_name(PyFT2Font *self)
{
    const char *name = self->x->get_face()->family_name;
    if (name == nullptr) {
        name = "UNAVAILABLE";
    }
    return name;
}

static const char *
PyFT2Font_style_name(PyFT2Font *self)
{
    const char *name = self->x->get_face()->style_name;
    if (name == nullptr) {
        name = "UNAVAILABLE";
    }
    return name;
}

static FaceFlags
PyFT2Font_face_flags(PyFT2Font *self)
{
    return static_cast<FaceFlags>(self->x->get_face()->face_flags);
}

static StyleFlags
PyFT2Font_style_flags(PyFT2Font *self)
{
    return static_cast<StyleFlags>(self->x->get_face()->style_flags & 0xffff);
}

static FT_Long
PyFT2Font_num_named_instances(PyFT2Font *self)
{
    return (self->x->get_face()->style_flags & 0x7fff0000) >> 16;
}

static FT_Long
PyFT2Font_num_glyphs(PyFT2Font *self)
{
    return self->x->get_face()->num_glyphs;
}

static FT_Int
PyFT2Font_num_fixed_sizes(PyFT2Font *self)
{
    return self->x->get_face()->num_fixed_sizes;
}

static FT_Int
PyFT2Font_num_charmaps(PyFT2Font *self)
{
    return self->x->get_face()->num_charmaps;
}

static bool
PyFT2Font_scalable(PyFT2Font *self)
{
    if (FT_IS_SCALABLE(self->x->get_face())) {
        return true;
    }
    return false;
}

static FT_UShort
PyFT2Font_units_per_EM(PyFT2Font *self)
{
    return self->x->get_face()->units_per_EM;
}

static py::tuple
PyFT2Font_get_bbox(PyFT2Font *self)
{
    FT_BBox *bbox = &(self->x->get_face()->bbox);

    return py::make_tuple(bbox->xMin, bbox->yMin, bbox->xMax, bbox->yMax);
}

static FT_Short
PyFT2Font_ascender(PyFT2Font *self)
{
    return self->x->get_face()->ascender;
}

static FT_Short
PyFT2Font_descender(PyFT2Font *self)
{
    return self->x->get_face()->descender;
}

static FT_Short
PyFT2Font_height(PyFT2Font *self)
{
    return self->x->get_face()->height;
}

static FT_Short
PyFT2Font_max_advance_width(PyFT2Font *self)
{
    return self->x->get_face()->max_advance_width;
}

static FT_Short
PyFT2Font_max_advance_height(PyFT2Font *self)
{
    return self->x->get_face()->max_advance_height;
}

static FT_Short
PyFT2Font_underline_position(PyFT2Font *self)
{
    return self->x->get_face()->underline_position;
}

static FT_Short
PyFT2Font_underline_thickness(PyFT2Font *self)
{
    return self->x->get_face()->underline_thickness;
}

static py::str
PyFT2Font_fname(PyFT2Font *self)
{
    if (self->stream.close) {  // Called passed a filename to the constructor.
        return self->py_file.attr("name");
    } else {
        return py::cast<py::str>(self->py_file);
    }
}

static py::object
ft2font__getattr__(std::string name) {
    auto api = py::module_::import("matplotlib._api");
    auto warn = api.attr("warn_deprecated");

#define DEPRECATE_ATTR_FROM_ENUM(attr_, alternative_, real_value_) \
    do { \
        if (name == #attr_) { \
            warn("since"_a="3.10", "name"_a=#attr_, "obj_type"_a="attribute", \
                 "alternative"_a=#alternative_); \
            return py::cast(static_cast<int>(real_value_)); \
        } \
    } while(0)
    DEPRECATE_ATTR_FROM_ENUM(KERNING_DEFAULT, Kerning.DEFAULT, FT_KERNING_DEFAULT);
    DEPRECATE_ATTR_FROM_ENUM(KERNING_UNFITTED, Kerning.UNFITTED, FT_KERNING_UNFITTED);
    DEPRECATE_ATTR_FROM_ENUM(KERNING_UNSCALED, Kerning.UNSCALED, FT_KERNING_UNSCALED);

#undef DEPRECATE_ATTR_FROM_ENUM

#define DEPRECATE_ATTR_FROM_FLAG(attr_, enum_, value_) \
    do { \
        if (name == #attr_) { \
            warn("since"_a="3.10", "name"_a=#attr_, "obj_type"_a="attribute", \
                 "alternative"_a=#enum_ "." #value_); \
            return py::cast(enum_::value_); \
        } \
    } while(0)

    DEPRECATE_ATTR_FROM_FLAG(LOAD_DEFAULT, LoadFlags, DEFAULT);
    DEPRECATE_ATTR_FROM_FLAG(LOAD_NO_SCALE, LoadFlags, NO_SCALE);
    DEPRECATE_ATTR_FROM_FLAG(LOAD_NO_HINTING, LoadFlags, NO_HINTING);
    DEPRECATE_ATTR_FROM_FLAG(LOAD_RENDER, LoadFlags, RENDER);
    DEPRECATE_ATTR_FROM_FLAG(LOAD_NO_BITMAP, LoadFlags, NO_BITMAP);
    DEPRECATE_ATTR_FROM_FLAG(LOAD_VERTICAL_LAYOUT, LoadFlags, VERTICAL_LAYOUT);
    DEPRECATE_ATTR_FROM_FLAG(LOAD_FORCE_AUTOHINT, LoadFlags, FORCE_AUTOHINT);
    DEPRECATE_ATTR_FROM_FLAG(LOAD_CROP_BITMAP, LoadFlags, CROP_BITMAP);
    DEPRECATE_ATTR_FROM_FLAG(LOAD_PEDANTIC, LoadFlags, PEDANTIC);
    DEPRECATE_ATTR_FROM_FLAG(LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH, LoadFlags,
                             IGNORE_GLOBAL_ADVANCE_WIDTH);
    DEPRECATE_ATTR_FROM_FLAG(LOAD_NO_RECURSE, LoadFlags, NO_RECURSE);
    DEPRECATE_ATTR_FROM_FLAG(LOAD_IGNORE_TRANSFORM, LoadFlags, IGNORE_TRANSFORM);
    DEPRECATE_ATTR_FROM_FLAG(LOAD_MONOCHROME, LoadFlags, MONOCHROME);
    DEPRECATE_ATTR_FROM_FLAG(LOAD_LINEAR_DESIGN, LoadFlags, LINEAR_DESIGN);
    DEPRECATE_ATTR_FROM_FLAG(LOAD_NO_AUTOHINT, LoadFlags, NO_AUTOHINT);

    DEPRECATE_ATTR_FROM_FLAG(LOAD_TARGET_NORMAL, LoadFlags, TARGET_NORMAL);
    DEPRECATE_ATTR_FROM_FLAG(LOAD_TARGET_LIGHT, LoadFlags, TARGET_LIGHT);
    DEPRECATE_ATTR_FROM_FLAG(LOAD_TARGET_MONO, LoadFlags, TARGET_MONO);
    DEPRECATE_ATTR_FROM_FLAG(LOAD_TARGET_LCD, LoadFlags, TARGET_LCD);
    DEPRECATE_ATTR_FROM_FLAG(LOAD_TARGET_LCD_V, LoadFlags, TARGET_LCD_V);

    DEPRECATE_ATTR_FROM_FLAG(SCALABLE, FaceFlags, SCALABLE);
    DEPRECATE_ATTR_FROM_FLAG(FIXED_SIZES, FaceFlags, FIXED_SIZES);
    DEPRECATE_ATTR_FROM_FLAG(FIXED_WIDTH, FaceFlags, FIXED_WIDTH);
    DEPRECATE_ATTR_FROM_FLAG(SFNT, FaceFlags, SFNT);
    DEPRECATE_ATTR_FROM_FLAG(HORIZONTAL, FaceFlags, HORIZONTAL);
    DEPRECATE_ATTR_FROM_FLAG(VERTICAL, FaceFlags, VERTICAL);
    DEPRECATE_ATTR_FROM_FLAG(KERNING, FaceFlags, KERNING);
    DEPRECATE_ATTR_FROM_FLAG(FAST_GLYPHS, FaceFlags, FAST_GLYPHS);
    DEPRECATE_ATTR_FROM_FLAG(MULTIPLE_MASTERS, FaceFlags, MULTIPLE_MASTERS);
    DEPRECATE_ATTR_FROM_FLAG(GLYPH_NAMES, FaceFlags, GLYPH_NAMES);
    DEPRECATE_ATTR_FROM_FLAG(EXTERNAL_STREAM, FaceFlags, EXTERNAL_STREAM);

    DEPRECATE_ATTR_FROM_FLAG(ITALIC, StyleFlags, ITALIC);
    DEPRECATE_ATTR_FROM_FLAG(BOLD, StyleFlags, BOLD);
#undef DEPRECATE_ATTR_FROM_FLAG

    throw py::attribute_error(
        "module 'matplotlib.ft2font' has no attribute {!r}"_s.format(name));
}

PYBIND11_MODULE(ft2font, m, py::mod_gil_not_used())
{
    if (FT_Init_FreeType(&_ft2Library)) {  // initialize library
        throw std::runtime_error("Could not initialize the freetype2 library");
    }
    FT_Int major, minor, patch;
    char version_string[64];
    FT_Library_Version(_ft2Library, &major, &minor, &patch);
    snprintf(version_string, sizeof(version_string), "%d.%d.%d", major, minor, patch);

    p11x::bind_enums(m);
    p11x::enums["Kerning"].attr("__doc__") = Kerning__doc__;
    p11x::enums["LoadFlags"].attr("__doc__") = LoadFlags__doc__;
    p11x::enums["FaceFlags"].attr("__doc__") = FaceFlags__doc__;
    p11x::enums["StyleFlags"].attr("__doc__") = StyleFlags__doc__;

    py::class_<FT2Image>(m, "FT2Image", py::is_final(), py::buffer_protocol(),
                         PyFT2Image__doc__)
        .def(py::init(
                [](double_or_<long> width, double_or_<long> height) {
                    return new FT2Image(
                        _double_to_<long>("width", width),
                        _double_to_<long>("height", height)
                    );
                }),
             "width"_a, "height"_a, PyFT2Image_init__doc__)
        .def("draw_rect_filled", &PyFT2Image_draw_rect_filled,
             "x0"_a, "y0"_a, "x1"_a, "y1"_a,
             PyFT2Image_draw_rect_filled__doc__)
        .def_buffer([](FT2Image &self) -> py::buffer_info {
            std::vector<py::size_t> shape { self.get_height(), self.get_width() };
            std::vector<py::size_t> strides { self.get_width(), 1 };
            return py::buffer_info(self.get_buffer(), shape, strides);
        });

    py::class_<PyGlyph>(m, "Glyph", py::is_final(), PyGlyph__doc__)
        .def(py::init<>([]() -> PyGlyph {
            // Glyph is not useful from Python, so mark it as not constructible.
            throw std::runtime_error("Glyph is not constructible");
        }))
        .def_readonly("width", &PyGlyph::width, "The glyph's width.")
        .def_readonly("height", &PyGlyph::height, "The glyph's height.")
        .def_readonly("horiBearingX", &PyGlyph::horiBearingX,
                      "Left side bearing for horizontal layout.")
        .def_readonly("horiBearingY", &PyGlyph::horiBearingY,
                      "Top side bearing for horizontal layout.")
        .def_readonly("horiAdvance", &PyGlyph::horiAdvance,
                      "Advance width for horizontal layout.")
        .def_readonly("linearHoriAdvance", &PyGlyph::linearHoriAdvance,
                      "The advance width of the unhinted glyph.")
        .def_readonly("vertBearingX", &PyGlyph::vertBearingX,
                      "Left side bearing for vertical layout.")
        .def_readonly("vertBearingY", &PyGlyph::vertBearingY,
                      "Top side bearing for vertical layout.")
        .def_readonly("vertAdvance", &PyGlyph::vertAdvance,
                      "Advance height for vertical layout.")
        .def_property_readonly("bbox", &PyGlyph_get_bbox,
                               "The control box of the glyph.");

    py::class_<PyFT2Font>(m, "FT2Font", py::is_final(), py::buffer_protocol(),
                          PyFT2Font__doc__)
        .def(py::init(&PyFT2Font_init),
             "filename"_a, "hinting_factor"_a=8, py::kw_only(),
             "_fallback_list"_a=py::none(), "_kerning_factor"_a=0,
             PyFT2Font_init__doc__)
        .def("clear", &PyFT2Font_clear, PyFT2Font_clear__doc__)
        .def("set_size", &PyFT2Font_set_size, "ptsize"_a, "dpi"_a,
             PyFT2Font_set_size__doc__)
        .def("set_charmap", &PyFT2Font_set_charmap, "i"_a,
             PyFT2Font_set_charmap__doc__)
        .def("select_charmap", &PyFT2Font_select_charmap, "i"_a,
             PyFT2Font_select_charmap__doc__)
        .def("get_kerning", &PyFT2Font_get_kerning, "left"_a, "right"_a, "mode"_a,
             PyFT2Font_get_kerning__doc__)
        .def("set_text", &PyFT2Font_set_text,
             "string"_a, "angle"_a=0.0, "flags"_a=LoadFlags::FORCE_AUTOHINT,
             PyFT2Font_set_text__doc__)
        .def("_get_fontmap", &PyFT2Font_get_fontmap, "string"_a,
             PyFT2Font_get_fontmap__doc__)
        .def("get_num_glyphs", &PyFT2Font_get_num_glyphs, PyFT2Font_get_num_glyphs__doc__)
        .def("load_char", &PyFT2Font_load_char,
             "charcode"_a, "flags"_a=LoadFlags::FORCE_AUTOHINT,
             PyFT2Font_load_char__doc__)
        .def("load_glyph", &PyFT2Font_load_glyph,
             "glyph_index"_a, "flags"_a=LoadFlags::FORCE_AUTOHINT,
             PyFT2Font_load_glyph__doc__)
        .def("get_width_height", &PyFT2Font_get_width_height,
             PyFT2Font_get_width_height__doc__)
        .def("get_bitmap_offset", &PyFT2Font_get_bitmap_offset,
             PyFT2Font_get_bitmap_offset__doc__)
        .def("get_descent", &PyFT2Font_get_descent, PyFT2Font_get_descent__doc__)
        .def("draw_glyphs_to_bitmap", &PyFT2Font_draw_glyphs_to_bitmap,
             py::kw_only(), "antialiased"_a=true,
             PyFT2Font_draw_glyphs_to_bitmap__doc__)
        .def("draw_glyph_to_bitmap", &PyFT2Font_draw_glyph_to_bitmap,
             "image"_a, "x"_a, "y"_a, "glyph"_a, py::kw_only(), "antialiased"_a=true,
             PyFT2Font_draw_glyph_to_bitmap__doc__)
        .def("get_glyph_name", &PyFT2Font_get_glyph_name, "index"_a,
             PyFT2Font_get_glyph_name__doc__)
        .def("get_charmap", &PyFT2Font_get_charmap, PyFT2Font_get_charmap__doc__)
        .def("get_char_index", &PyFT2Font_get_char_index, "codepoint"_a,
             PyFT2Font_get_char_index__doc__)
        .def("get_sfnt", &PyFT2Font_get_sfnt, PyFT2Font_get_sfnt__doc__)
        .def("get_name_index", &PyFT2Font_get_name_index, "name"_a,
             PyFT2Font_get_name_index__doc__)
        .def("get_ps_font_info", &PyFT2Font_get_ps_font_info,
             PyFT2Font_get_ps_font_info__doc__)
        .def("get_sfnt_table", &PyFT2Font_get_sfnt_table, "name"_a,
             PyFT2Font_get_sfnt_table__doc__)
        .def("get_path", &PyFT2Font_get_path, PyFT2Font_get_path__doc__)
        .def("get_image", &PyFT2Font_get_image, PyFT2Font_get_image__doc__)

        .def_property_readonly("postscript_name", &PyFT2Font_postscript_name,
                               "PostScript name of the font.")
        .def_property_readonly("num_faces", &PyFT2Font_num_faces,
                               "Number of faces in file.")
        .def_property_readonly("family_name", &PyFT2Font_family_name,
                               "Face family name.")
        .def_property_readonly("style_name", &PyFT2Font_style_name,
                               "Style name.")
        .def_property_readonly("face_flags", &PyFT2Font_face_flags,
                               "Face flags; see `.FaceFlags`.")
        .def_property_readonly("style_flags", &PyFT2Font_style_flags,
                               "Style flags; see `.StyleFlags`.")
        .def_property_readonly("num_named_instances", &PyFT2Font_num_named_instances,
                               "Number of named instances in the face.")
        .def_property_readonly("num_glyphs", &PyFT2Font_num_glyphs,
                               "Number of glyphs in the face.")
        .def_property_readonly("num_fixed_sizes", &PyFT2Font_num_fixed_sizes,
                               "Number of bitmap in the face.")
        .def_property_readonly("num_charmaps", &PyFT2Font_num_charmaps,
                               "Number of charmaps in the face.")
        .def_property_readonly("scalable", &PyFT2Font_scalable,
                               "Whether face is scalable; attributes after this one "
                               "are only defined for scalable faces.")
        .def_property_readonly("units_per_EM", &PyFT2Font_units_per_EM,
                               "Number of font units covered by the EM.")
        .def_property_readonly("bbox", &PyFT2Font_get_bbox,
                               "Face global bounding box (xmin, ymin, xmax, ymax).")
        .def_property_readonly("ascender", &PyFT2Font_ascender,
                               "Ascender in 26.6 units.")
        .def_property_readonly("descender", &PyFT2Font_descender,
                               "Descender in 26.6 units.")
        .def_property_readonly("height", &PyFT2Font_height,
                               "Height in 26.6 units; used to compute a default line "
                               "spacing (baseline-to-baseline distance).")
        .def_property_readonly("max_advance_width", &PyFT2Font_max_advance_width,
                               "Maximum horizontal cursor advance for all glyphs.")
        .def_property_readonly("max_advance_height", &PyFT2Font_max_advance_height,
                               "Maximum vertical cursor advance for all glyphs.")
        .def_property_readonly("underline_position", &PyFT2Font_underline_position,
                               "Vertical position of the underline bar.")
        .def_property_readonly("underline_thickness", &PyFT2Font_underline_thickness,
                               "Thickness of the underline bar.")
        .def_property_readonly("fname", &PyFT2Font_fname,
                               "The original filename for this object.")

        .def_buffer([](PyFT2Font &self) -> py::buffer_info {
            FT2Image &im = self.x->get_image();
            std::vector<py::size_t> shape { im.get_height(), im.get_width() };
            std::vector<py::size_t> strides { im.get_width(), 1 };
            return py::buffer_info(im.get_buffer(), shape, strides);
        });

    m.attr("__freetype_version__") = version_string;
    m.attr("__freetype_build_type__") = FREETYPE_BUILD_TYPE;
    m.def("__getattr__", ft2font__getattr__);
}
