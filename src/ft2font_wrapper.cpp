#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "ft2font.h"
#include "numpy/arrayobject.h"

#include <set>
#include <sstream>
#include <unordered_map>

namespace py = pybind11;
using namespace pybind11::literals;

/**********************************************************************
 * FT2Image
 * */

const char *PyFT2Image_draw_rect_filled__doc__ =
    "Draw a filled rectangle to the image.";

static void
PyFT2Image_draw_rect_filled(FT2Image *self, double x0, double y0, double x1, double y1)
{
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

const char *PyFT2Font_init__doc__ =
    "Create a new FT2Font object.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "filename : str or file-like\n"
    "    The source of the font data in a format (ttf or ttc) that FreeType can read\n"
    "\n"
    "hinting_factor : int, optional\n"
    "    Must be positive. Used to scale the hinting in the x-direction\n"
    "_fallback_list : list of FT2Font, optional\n"
    "    A list of FT2Font objects used to find missing glyphs.\n"
    "\n"
    "    .. warning::\n"
    "        This API is both private and provisional: do not use it directly\n"
    "\n"
    "_kerning_factor : int, optional\n"
    "    Used to adjust the degree of kerning.\n"
    "\n"
    "    .. warning::\n"
    "        This API is private: do not use it directly\n"
;

static PyFT2Font *
PyFT2Font_init(py::object filename, long hinting_factor = 8,
               std::optional<std::vector<PyFT2Font *>> fallback_list = std::nullopt,
               int kerning_factor = 0)
{
    if (hinting_factor <= 0) {
        throw py::value_error("hinting_factor must be greater than 0");
    }

    PyFT2Font *self = new PyFT2Font();
    self->x = NULL;
    memset(&self->stream, 0, sizeof(FT_StreamRec));
    self->stream.base = NULL;
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
        self->stream.close = NULL;
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

const char *PyFT2Font_set_size__doc__ =
    "Set the point size and dpi of the text.";

static void
PyFT2Font_set_size(PyFT2Font *self, double ptsize, double dpi)
{
    self->x->set_size(ptsize, dpi);
}

const char *PyFT2Font_set_charmap__doc__ =
    "Make the i-th charmap current.";

static void
PyFT2Font_set_charmap(PyFT2Font *self, int i)
{
    self->x->set_charmap(i);
}

const char *PyFT2Font_select_charmap__doc__ =
    "Select a charmap by its FT_Encoding number.";

static void
PyFT2Font_select_charmap(PyFT2Font *self, unsigned long i)
{
    self->x->select_charmap(i);
}

const char *PyFT2Font_get_kerning__doc__ =
    "Get the kerning between *left* and *right* glyph indices.\n"
    "\n"
    "*mode* is a kerning mode constant:\n"
    "\n"
    "- KERNING_DEFAULT  - Return scaled and grid-fitted kerning distances\n"
    "- KERNING_UNFITTED - Return scaled but un-grid-fitted kerning distances\n"
    "- KERNING_UNSCALED - Return the kerning vector in original font units\n";

static int
PyFT2Font_get_kerning(PyFT2Font *self, FT_UInt left, FT_UInt right, FT_UInt mode)
{
    bool fallback = true;

    return self->x->get_kerning(left, right, mode, fallback);
}

const char *PyFT2Font_get_fontmap__doc__ =
    "Get a mapping between characters and the font that includes them.\n"
    "A dictionary mapping unicode characters to PyFT2Font objects.";

static py::dict
PyFT2Font_get_fontmap(PyFT2Font *self, std::u32string text)
{
    std::set<FT_ULong> codepoints;

    for (auto code : text) {
        codepoints.insert(code);
    }

    py::dict char_to_font;
    for (auto code : codepoints) {
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

const char *PyFT2Font_set_text__doc__ =
    "Set the text *string* and *angle*.\n"
    "*flags* can be a bitwise-or of the LOAD_XXX constants;\n"
    "the default value is LOAD_FORCE_AUTOHINT.\n"
    "You must call this before `.draw_glyphs_to_bitmap`.\n"
    "A sequence of x,y positions in 26.6 subpixels is returned; divide by 64 for pixels.\n";

static py::array_t<double>
PyFT2Font_set_text(PyFT2Font *self, std::u32string_view text, double angle = 0.0,
                   FT_Int32 flags = FT_LOAD_FORCE_AUTOHINT)
{
    std::vector<double> xys;

    self->x->set_text(text, angle, flags, xys);

    py::ssize_t dims[] = { static_cast<py::ssize_t>(xys.size()) / 2, 2 };
    py::array_t<double> result(dims);
    if (xys.size() > 0) {
        memcpy(result.mutable_data(), xys.data(), result.nbytes());
    }
    return result;
}

const char *PyFT2Font_get_num_glyphs__doc__ =
    "Return the number of loaded glyphs.";

static size_t
PyFT2Font_get_num_glyphs(PyFT2Font *self)
{
    return self->x->get_num_glyphs();
}

const char *PyFT2Font_load_char__doc__ =
    "Load character with *charcode* in current fontfile and set glyph.\n"
    "*flags* can be a bitwise-or of the LOAD_XXX constants;\n"
    "the default value is LOAD_FORCE_AUTOHINT.\n"
    "Return value is a Glyph object, with attributes\n\n"
    "- width: glyph width\n"
    "- height: glyph height\n"
    "- bbox: the glyph bbox (xmin, ymin, xmax, ymax)\n"
    "- horiBearingX: left side bearing in horizontal layouts\n"
    "- horiBearingY: top side bearing in horizontal layouts\n"
    "- horiAdvance: advance width for horizontal layout\n"
    "- vertBearingX: left side bearing in vertical layouts\n"
    "- vertBearingY: top side bearing in vertical layouts\n"
    "- vertAdvance: advance height for vertical layout\n";

static PyGlyph *
PyFT2Font_load_char(PyFT2Font *self, long charcode,
                    FT_Int32 flags = FT_LOAD_FORCE_AUTOHINT)
{
    bool fallback = true;
    FT2Font *ft_object = NULL;

    self->x->load_char(charcode, flags, ft_object, fallback);

    return PyGlyph_from_FT2Font(ft_object);
}

const char *PyFT2Font_load_glyph__doc__ =
    "Load character with *glyphindex* in current fontfile and set glyph.\n"
    "*flags* can be a bitwise-or of the LOAD_XXX constants;\n"
    "the default value is LOAD_FORCE_AUTOHINT.\n"
    "Return value is a Glyph object, with attributes\n\n"
    "- width: glyph width\n"
    "- height: glyph height\n"
    "- bbox: the glyph bbox (xmin, ymin, xmax, ymax)\n"
    "- horiBearingX: left side bearing in horizontal layouts\n"
    "- horiBearingY: top side bearing in horizontal layouts\n"
    "- horiAdvance: advance width for horizontal layout\n"
    "- vertBearingX: left side bearing in vertical layouts\n"
    "- vertBearingY: top side bearing in vertical layouts\n"
    "- vertAdvance: advance height for vertical layout\n";

static PyGlyph *
PyFT2Font_load_glyph(PyFT2Font *self, FT_UInt glyph_index,
                     FT_Int32 flags = FT_LOAD_FORCE_AUTOHINT)
{
    bool fallback = true;
    FT2Font *ft_object = NULL;

    self->x->load_glyph(glyph_index, flags, ft_object, fallback);

    return PyGlyph_from_FT2Font(ft_object);
}

const char *PyFT2Font_get_width_height__doc__ =
    "Get the width and height in 26.6 subpixels of the current string set by `.set_text`.\n"
    "The rotation of the string is accounted for.  To get width and height\n"
    "in pixels, divide these values by 64.\n";

static py::tuple
PyFT2Font_get_width_height(PyFT2Font *self)
{
    long width, height;

    self->x->get_width_height(&width, &height);

    return py::make_tuple(width, height);
}

const char *PyFT2Font_get_bitmap_offset__doc__ =
    "Get the (x, y) offset in 26.6 subpixels for the bitmap if ink hangs left or below (0, 0).\n"
    "Since Matplotlib only supports left-to-right text, y is always 0.\n";

static py::tuple
PyFT2Font_get_bitmap_offset(PyFT2Font *self)
{
    long x, y;

    self->x->get_bitmap_offset(&x, &y);

    return py::make_tuple(x, y);
}

const char *PyFT2Font_get_descent__doc__ =
    "Get the descent in 26.6 subpixels of the current string set by `.set_text`.\n"
    "The rotation of the string is accounted for.  To get the descent\n"
    "in pixels, divide this value by 64.\n";

static long
PyFT2Font_get_descent(PyFT2Font *self)
{
    return self->x->get_descent();
}

const char *PyFT2Font_draw_glyphs_to_bitmap__doc__ =
    "Draw the glyphs that were loaded by `.set_text` to the bitmap.\n"
    "\n"
    "The bitmap size will be automatically set to include the glyphs.\n";

static void
PyFT2Font_draw_glyphs_to_bitmap(PyFT2Font *self, bool antialiased = true)
{
    self->x->draw_glyphs_to_bitmap(antialiased);
}

const char *PyFT2Font_draw_glyph_to_bitmap__doc__ =
    "Draw a single glyph to the bitmap at pixel locations x, y.\n"
    "\n"
    "Note it is your responsibility to create the image manually\n"
    "with the correct size before this call is made.\n"
    "\n"
    "If you want automatic layout, use `.set_text` in combinations with\n"
    "`.draw_glyphs_to_bitmap`.  This function is instead intended for people\n"
    "who want to render individual glyphs (e.g., returned by `.load_char`)\n"
    "at precise locations.\n";

static void
PyFT2Font_draw_glyph_to_bitmap(PyFT2Font *self, FT2Image &image, double xd, double yd,
                               PyGlyph *glyph, bool antialiased = true)
{
    self->x->draw_glyph_to_bitmap(image, xd, yd, glyph->glyphInd, antialiased);
}

const char *PyFT2Font_get_glyph_name__doc__ =
    "Retrieve the ASCII name of a given glyph *index* in a face.\n"
    "\n"
    "Due to Matplotlib's internal design, for fonts that do not contain glyph\n"
    "names (per FT_FACE_FLAG_GLYPH_NAMES), this returns a made-up name which\n"
    "does *not* roundtrip through `.get_name_index`.\n";

static py::str
PyFT2Font_get_glyph_name(PyFT2Font *self, unsigned int glyph_number)
{
    std::string buffer;
    bool fallback = true;

    buffer.resize(128);
    self->x->get_glyph_name(glyph_number, buffer, fallback);
    return buffer;
}

const char *PyFT2Font_get_charmap__doc__ =
    "Return a dict that maps the character codes of the selected charmap\n"
    "(Unicode by default) to their corresponding glyph indices.\n";

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


const char *PyFT2Font_get_char_index__doc__ =
    "Return the glyph index corresponding to a character *codepoint*.";

static FT_UInt
PyFT2Font_get_char_index(PyFT2Font *self, FT_ULong ccode)
{
    bool fallback = true;

    return self->x->get_char_index(ccode, fallback);
}


const char *PyFT2Font_get_sfnt__doc__ =
    "Load the entire SFNT names table, as a dict whose keys are\n"
    "(platform-ID, ISO-encoding-scheme, language-code, and description)\n"
    "tuples.\n";

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

const char *PyFT2Font_get_name_index__doc__ =
    "Return the glyph index of a given glyph *name*.\n"
    "The glyph index 0 means 'undefined character code'.\n";

static long
PyFT2Font_get_name_index(PyFT2Font *self, char *glyphname)
{
    return self->x->get_name_index(glyphname);
}

const char *PyFT2Font_get_ps_font_info__doc__ =
    "Return the information in the PS Font Info structure.";

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

const char *PyFT2Font_get_sfnt_table__doc__ =
    "Return one of the following SFNT tables: head, maxp, OS/2, hhea, "
    "vhea, post, or pclt.";

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

const char *PyFT2Font_get_path__doc__ =
    "Get the path data from the currently loaded glyph as a tuple of vertices, codes.";

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

const char *PyFT2Font_get_image__doc__ =
    "Return the underlying image buffer for this font object.";

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
    if (ps_name == NULL) {
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
    if (name == NULL) {
        name = "UNAVAILABLE";
    }
    return name;
}

static const char *
PyFT2Font_style_name(PyFT2Font *self)
{
    const char *name = self->x->get_face()->style_name;
    if (name == NULL) {
        name = "UNAVAILABLE";
    }
    return name;
}

static FT_Long
PyFT2Font_face_flags(PyFT2Font *self)
{
    return self->x->get_face()->face_flags;
}

static FT_Long
PyFT2Font_style_flags(PyFT2Font *self)
{
    return self->x->get_face()->style_flags;
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

PYBIND11_MODULE(ft2font, m)
{
    auto ia = [m]() -> const void* {
        import_array();
        return &m;
    };
    if (ia() == NULL) {
        throw py::error_already_set();
    }

    if (FT_Init_FreeType(&_ft2Library)) {  // initialize library
        throw std::runtime_error("Could not initialize the freetype2 library");
    }
    FT_Int major, minor, patch;
    char version_string[64];
    FT_Library_Version(_ft2Library, &major, &minor, &patch);
    snprintf(version_string, sizeof(version_string), "%d.%d.%d", major, minor, patch);

    py::class_<FT2Image>(m, "FT2Image", py::is_final(), py::buffer_protocol())
        .def(py::init<double, double>(), "width"_a, "height"_a)
        .def("draw_rect_filled", &PyFT2Image_draw_rect_filled,
             "x0"_a, "y0"_a, "x1"_a, "y1"_a,
             PyFT2Image_draw_rect_filled__doc__)
        .def_buffer([](FT2Image &self) -> py::buffer_info {
            std::vector<py::size_t> shape { self.get_height(), self.get_width() };
            std::vector<py::size_t> strides { self.get_width(), 1 };
            return py::buffer_info(self.get_buffer(), shape, strides);
        });

    py::class_<PyGlyph>(m, "Glyph", py::is_final())
        .def(py::init<>([]() -> PyGlyph {
            // Glyph is not useful from Python, so mark it as not constructible.
            throw std::runtime_error("Glyph is not constructible");
        }))
        .def_readonly("width", &PyGlyph::width)
        .def_readonly("height", &PyGlyph::height)
        .def_readonly("horiBearingX", &PyGlyph::horiBearingX)
        .def_readonly("horiBearingY", &PyGlyph::horiBearingY)
        .def_readonly("horiAdvance", &PyGlyph::horiAdvance)
        .def_readonly("linearHoriAdvance", &PyGlyph::linearHoriAdvance)
        .def_readonly("vertBearingX", &PyGlyph::vertBearingX)
        .def_readonly("vertBearingY", &PyGlyph::vertBearingY)
        .def_readonly("vertAdvance", &PyGlyph::vertAdvance)
        .def_property_readonly("bbox", &PyGlyph_get_bbox);

    py::class_<PyFT2Font>(m, "FT2Font", py::is_final(), py::buffer_protocol())
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
             "string"_a, "angle"_a=0.0, "flags"_a=FT_LOAD_FORCE_AUTOHINT,
             PyFT2Font_set_text__doc__)
        .def("_get_fontmap", &PyFT2Font_get_fontmap, "string"_a,
             PyFT2Font_get_fontmap__doc__)
        .def("get_num_glyphs", &PyFT2Font_get_num_glyphs, PyFT2Font_get_num_glyphs__doc__)
        .def("load_char", &PyFT2Font_load_char,
             "charcode"_a, "flags"_a=FT_LOAD_FORCE_AUTOHINT,
             PyFT2Font_load_char__doc__)
        .def("load_glyph", &PyFT2Font_load_glyph,
             "glyph_index"_a, "flags"_a=FT_LOAD_FORCE_AUTOHINT,
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
                               "Face flags; see the ft2font constants.")
        .def_property_readonly("style_flags", &PyFT2Font_style_flags,
                               "Style flags; see the ft2font constants.")
        .def_property_readonly("num_glyphs", &PyFT2Font_num_glyphs,
                               "Number of glyphs in the face.")
        .def_property_readonly("num_fixed_sizes", &PyFT2Font_num_fixed_sizes,
                               "Number of bitmap in the face.")
        .def_property_readonly("num_charmaps", &PyFT2Font_num_charmaps)
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
        .def_property_readonly("fname", &PyFT2Font_fname)

        .def_buffer([](PyFT2Font &self) -> py::buffer_info {
            FT2Image &im = self.x->get_image();
            std::vector<py::size_t> shape { im.get_height(), im.get_width() };
            std::vector<py::size_t> strides { im.get_width(), 1 };
            return py::buffer_info(im.get_buffer(), shape, strides);
        });

    m.attr("__freetype_version__") = version_string;
    m.attr("__freetype_build_type__") = FREETYPE_BUILD_TYPE;
    m.attr("SCALABLE") = FT_FACE_FLAG_SCALABLE;
    m.attr("FIXED_SIZES") = FT_FACE_FLAG_FIXED_SIZES;
    m.attr("FIXED_WIDTH") = FT_FACE_FLAG_FIXED_WIDTH;
    m.attr("SFNT") = FT_FACE_FLAG_SFNT;
    m.attr("HORIZONTAL") = FT_FACE_FLAG_HORIZONTAL;
    m.attr("VERTICAL") = FT_FACE_FLAG_VERTICAL;
    m.attr("KERNING") = FT_FACE_FLAG_KERNING;
    m.attr("FAST_GLYPHS") = FT_FACE_FLAG_FAST_GLYPHS;
    m.attr("MULTIPLE_MASTERS") = FT_FACE_FLAG_MULTIPLE_MASTERS;
    m.attr("GLYPH_NAMES") = FT_FACE_FLAG_GLYPH_NAMES;
    m.attr("EXTERNAL_STREAM") = FT_FACE_FLAG_EXTERNAL_STREAM;
    m.attr("ITALIC") = FT_STYLE_FLAG_ITALIC;
    m.attr("BOLD") = FT_STYLE_FLAG_BOLD;
    m.attr("KERNING_DEFAULT") = (int)FT_KERNING_DEFAULT;
    m.attr("KERNING_UNFITTED") = (int)FT_KERNING_UNFITTED;
    m.attr("KERNING_UNSCALED") = (int)FT_KERNING_UNSCALED;
    m.attr("LOAD_DEFAULT") = FT_LOAD_DEFAULT;
    m.attr("LOAD_NO_SCALE") = FT_LOAD_NO_SCALE;
    m.attr("LOAD_NO_HINTING") = FT_LOAD_NO_HINTING;
    m.attr("LOAD_RENDER") = FT_LOAD_RENDER;
    m.attr("LOAD_NO_BITMAP") = FT_LOAD_NO_BITMAP;
    m.attr("LOAD_VERTICAL_LAYOUT") = FT_LOAD_VERTICAL_LAYOUT;
    m.attr("LOAD_FORCE_AUTOHINT") = FT_LOAD_FORCE_AUTOHINT;
    m.attr("LOAD_CROP_BITMAP") = FT_LOAD_CROP_BITMAP;
    m.attr("LOAD_PEDANTIC") = FT_LOAD_PEDANTIC;
    m.attr("LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH") = FT_LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH;
    m.attr("LOAD_NO_RECURSE") = FT_LOAD_NO_RECURSE;
    m.attr("LOAD_IGNORE_TRANSFORM") = FT_LOAD_IGNORE_TRANSFORM;
    m.attr("LOAD_MONOCHROME") = FT_LOAD_MONOCHROME;
    m.attr("LOAD_LINEAR_DESIGN") = FT_LOAD_LINEAR_DESIGN;
    m.attr("LOAD_NO_AUTOHINT") = (unsigned long)FT_LOAD_NO_AUTOHINT;
    m.attr("LOAD_TARGET_NORMAL") = (unsigned long)FT_LOAD_TARGET_NORMAL;
    m.attr("LOAD_TARGET_LIGHT") = (unsigned long)FT_LOAD_TARGET_LIGHT;
    m.attr("LOAD_TARGET_MONO") = (unsigned long)FT_LOAD_TARGET_MONO;
    m.attr("LOAD_TARGET_LCD") = (unsigned long)FT_LOAD_TARGET_LCD;
    m.attr("LOAD_TARGET_LCD_V") = (unsigned long)FT_LOAD_TARGET_LCD_V;
}
