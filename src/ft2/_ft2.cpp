#include "_ft2.h"
#include "_layout.h"

namespace matplotlib::ft2 {

Face::Face(std::string const& path, FT_Long index, double hinting_factor) :
  ptr{
    [&]() -> std::shared_ptr<FT_FaceRec_> {
      auto face = FT_Face{};
      FT_CHECK(FT_New_Face, library, path.data(), index, &face);
      auto transform = FT_Matrix{FT_Fixed(65536 / hinting_factor), 0, 0, 65536};
      FT_Set_Transform(face, &transform, nullptr);
      return {face, FT_Done_Face};
    }()
  },
  path{path},
  hinting_factor{hinting_factor}
{}

Glyph::Glyph(FT_Face const& face, double hinting_factor) :
  ptr{
    [&]() -> std::shared_ptr<FT_GlyphRec_> {
      auto glyph = FT_Glyph{};
      FT_CHECK(FT_Get_Glyph, face->glyph, &glyph);
      return {glyph, FT_Done_Glyph};
    }()
  },
  metrics{face->glyph->metrics},
  linearHoriAdvance{face->glyph->linearHoriAdvance},
  linearVertAdvance{face->glyph->linearVertAdvance},
  hinting_factor{hinting_factor}
{}

PYBIND11_MODULE(_ft2, m)
{
  using namespace pybind11::literals;

  m.doc() = R"__doc__(
A wrapper extension module for FreeType2.

Unless stated otherwise, all methods directly wrap a corresponding FreeType
function, and all lengths of this API are either in pixels (if FreeType
intenally uses 26.6 or 16.16 fixed point -- conversion is handled by this
module) or in font units (if FreeType internally uses font units).  The latter
case is explicitly mentioned where applicable.
)__doc__";

  FT_CHECK(FT_Init_FreeType, &library);

  auto major = FT_Int{}, minor = FT_Int{}, patch = FT_Int{};
  FT_Library_Version(library, &major, &minor, &patch);
  m.attr("__freetype_version__")
    = std::to_string(major) + "."
      + std::to_string(minor) + "."
      + std::to_string(patch);

#define DECLARE_FLAG(name) m.attr(#name) = FT_##name
  DECLARE_FLAG(FACE_FLAG_SCALABLE);
  DECLARE_FLAG(FACE_FLAG_FIXED_SIZES);
  DECLARE_FLAG(FACE_FLAG_FIXED_WIDTH);
  DECLARE_FLAG(FACE_FLAG_SFNT);
  DECLARE_FLAG(FACE_FLAG_HORIZONTAL);
  DECLARE_FLAG(FACE_FLAG_VERTICAL);
  DECLARE_FLAG(FACE_FLAG_KERNING);
  DECLARE_FLAG(FACE_FLAG_FAST_GLYPHS);
  DECLARE_FLAG(FACE_FLAG_MULTIPLE_MASTERS);
  DECLARE_FLAG(FACE_FLAG_GLYPH_NAMES);
  DECLARE_FLAG(FACE_FLAG_EXTERNAL_STREAM);
  DECLARE_FLAG(FACE_FLAG_HINTER);
  DECLARE_FLAG(FACE_FLAG_CID_KEYED);
  DECLARE_FLAG(FACE_FLAG_TRICKY);
  DECLARE_FLAG(FACE_FLAG_COLOR);

  DECLARE_FLAG(LOAD_DEFAULT);
  DECLARE_FLAG(LOAD_NO_SCALE);
  DECLARE_FLAG(LOAD_NO_HINTING);
  DECLARE_FLAG(LOAD_RENDER);
  DECLARE_FLAG(LOAD_NO_BITMAP);
  DECLARE_FLAG(LOAD_VERTICAL_LAYOUT);
  DECLARE_FLAG(LOAD_FORCE_AUTOHINT);
  DECLARE_FLAG(LOAD_CROP_BITMAP);
  DECLARE_FLAG(LOAD_PEDANTIC);
  DECLARE_FLAG(LOAD_IGNORE_GLOBAL_ADVANCE_WIDTH);
  DECLARE_FLAG(LOAD_NO_RECURSE);
  DECLARE_FLAG(LOAD_IGNORE_TRANSFORM);
  DECLARE_FLAG(LOAD_MONOCHROME);
  DECLARE_FLAG(LOAD_LINEAR_DESIGN);
  DECLARE_FLAG(LOAD_NO_AUTOHINT);
  DECLARE_FLAG(LOAD_COLOR);
#ifdef FT_LOAD_COMPUTE_METRICS  // backcompat: ft 2.6.1.
  DECLARE_FLAG(LOAD_COMPUTE_METRICS);
#endif
#ifdef FT_LOAD_BITMAP_METRICS_ONLY  // backcompat: ft 2.7.1.
  DECLARE_FLAG(LOAD_BITMAP_METRICS_ONLY);
#endif

  DECLARE_FLAG(LOAD_TARGET_NORMAL);
  DECLARE_FLAG(LOAD_TARGET_LIGHT);
  DECLARE_FLAG(LOAD_TARGET_MONO);
  DECLARE_FLAG(LOAD_TARGET_LCD);
  DECLARE_FLAG(LOAD_TARGET_LCD_V);

  DECLARE_FLAG(STYLE_FLAG_ITALIC);
  DECLARE_FLAG(STYLE_FLAG_BOLD);
#undef DECLARE_FLAG

  py::enum_<FT_Encoding>(m, "Encoding")
#define DECLARE_ENUM(name) .value(#name, FT_ENCODING_##name)
  DECLARE_ENUM(NONE)
  DECLARE_ENUM(UNICODE)
  DECLARE_ENUM(MS_SYMBOL)
  DECLARE_ENUM(ADOBE_LATIN_1)
  DECLARE_ENUM(OLD_LATIN_2)
  DECLARE_ENUM(SJIS)
#ifdef FT_ENCODING_PRC  // backcompat: ft 2.8.0.1.
  DECLARE_ENUM(PRC)
#endif
  DECLARE_ENUM(BIG5)
  DECLARE_ENUM(WANSUNG)
  DECLARE_ENUM(JOHAB)
  DECLARE_ENUM(ADOBE_STANDARD)
  DECLARE_ENUM(ADOBE_EXPERT)
  DECLARE_ENUM(ADOBE_CUSTOM)
  DECLARE_ENUM(APPLE_ROMAN);
#undef DECLARE_ENUM

  py::enum_<FT_Glyph_BBox_Mode>(m, "GlyphBbox")
#define DECLARE_ENUM(name) .value(#name, FT_GLYPH_BBOX_##name)
  DECLARE_ENUM(UNSCALED)
  DECLARE_ENUM(SUBPIXELS)
  DECLARE_ENUM(GRIDFIT)
  DECLARE_ENUM(TRUNCATE)
  DECLARE_ENUM(PIXELS);
#undef DECLARE_ENUM

  py::enum_<FT_Kerning_Mode>(m, "Kerning")
#define DECLARE_ENUM(name) .value(#name, FT_KERNING_##name)
  DECLARE_ENUM(DEFAULT)
  DECLARE_ENUM(UNFITTED)
  DECLARE_ENUM(UNSCALED);
#undef DECLARE_ENUM

  py::class_<Face>(m, "Face", R"__doc__(
A lightweight wrapper around a ``FT_Face``.

Length attributes of this class (`ascender`, `descender`, etc.) are in font
units.
)__doc__")
    .def(
      py::init(
        [](py::object path, FT_Long index, double hinting_factor) -> Face* {
          PyBytesObject* bytes = nullptr;
          char* buffer = nullptr;
          auto length = std::string::size_type{};  // i.e. size_t.
          PyUnicode_FSConverter(path.ptr(), &bytes)
            && PyBytes_AsStringAndSize(
                 reinterpret_cast<PyObject*>(bytes),
                 &buffer,
                 reinterpret_cast<Py_ssize_t*>(&length));
          Py_XDECREF(bytes);
          if (PyErr_Occurred()) {
            throw py::error_already_set();
          }
          return new Face{std::string{buffer, length}, index, hinting_factor};
        }),
      "path"_a, "index"_a, "hinting_factor"_a=1,
      R"__doc__(
Load a `Face` from a path and an index.
)__doc__")

    .def_property_readonly(
      "pathname",
      [](Face const& pyface) -> py::handle {
        return
          PyUnicode_DecodeFSDefaultAndSize(
            pyface.path.c_str(), pyface.path.size());
      },
      R"__doc__(
The path from which the `Face` was loaded, as a string.  This is the only
attribute that does not come from FreeType itself.
)__doc__")

#define DECLARE_FIELD(prop) \
  .def_property_readonly( \
    #prop, [](Face const& pyface) { return pyface.ptr->prop; })
    DECLARE_FIELD(num_faces)
    DECLARE_FIELD(face_flags)
    DECLARE_FIELD(style_flags)
    DECLARE_FIELD(num_glyphs)
    DECLARE_FIELD(family_name)
    DECLARE_FIELD(style_name)
    DECLARE_FIELD(num_fixed_sizes)
    // available_sizes -> not supported.
    DECLARE_FIELD(num_charmaps)
    // charmaps -> not supported.
    // generic -> not supported.
    .def_property_readonly(
      "bbox",
      [](Face const& pyface) -> std::tuple<FT_Pos, FT_Pos, FT_Pos, FT_Pos> {
        auto [xmin, ymin, xmax, ymax] = pyface.ptr->bbox;
        return {xmin, ymin, xmax, ymax};
      })
    DECLARE_FIELD(units_per_EM)
    DECLARE_FIELD(ascender)
    DECLARE_FIELD(descender)
    DECLARE_FIELD(height)
    DECLARE_FIELD(max_advance_width)
    DECLARE_FIELD(max_advance_height)
    DECLARE_FIELD(underline_position)
    DECLARE_FIELD(underline_thickness)
    .def_property_readonly(
      "glyph",
      [](Face const& pyface) -> Glyph {
        return {pyface.ptr.get(), pyface.hinting_factor};
      })
    // size -> set_char_size.
    // charmap -> not supported.
#undef DECLARE_FIELD
    .def(
      "get_char_index",
      [](Face const& pyface, FT_ULong codepoint) -> FT_UInt {
        return FT_Get_Char_Index(pyface.ptr.get(), codepoint);
      },
      "codepoint"_a)
    .def(
      "get_font_format",
      [](Face const& pyface) -> std::string {
        // backcompat: FT_Get_Font_Format in ft 2.6.
        return FT_Get_X11_Font_Format(pyface.ptr.get());
      })
    .def(
      "get_glyph_name",
      [](Face const& pyface, FT_UInt index) -> std::string {
        auto face = pyface.ptr.get();
        char buf[128];  // Limit to PS identifier size.
        // This branch is copied from ft2font.cpp.
        if (!FT_HAS_GLYPH_NAMES(face)) {
          /* Note that this generated name must match the name that
            is generated by ttconv in ttfont_CharStrings_getname. */
          PyOS_snprintf(buf, 128, "uni%08x", index);
        } else {
          FT_CHECK(FT_Get_Glyph_Name, face, index, buf, 128);
        }
        return buf;
      },
      "index"_a)
    .def(
      "get_kerning",
      [](Face const& pyface, FT_UInt left, FT_UInt right, FT_Kerning_Mode mode)
      -> std::tuple<double, double> {
        if (FT_HAS_KERNING(pyface.ptr.get())) {
          if (mode == FT_KERNING_UNSCALED) {
            // So that whoever *actually* needs this can implement the correct
            // conversion factor.
            throw
              std::runtime_error("Unknown conversion for FT_KERNING_UNSCALED");
          }
          auto delta = FT_Vector{};
          FT_CHECK(
            FT_Get_Kerning, pyface.ptr.get(), left, right, mode, &delta);
          return {delta.x / 64., delta.y / 64.};
        } else {
          return {0, 0};
        }
      },
      "left"_a, "right"_a, "mode"_a)
    .def(
      "get_postscript_name",
      [](Face const& pyface) -> char const* {
        return FT_Get_Postscript_Name(pyface.ptr.get());
      })
    .def(
      "get_ps_font_info",
      [](Face const& pyface) -> py::dict {
        auto info = PS_FontInfoRec{};
        FT_CHECK(FT_Get_PS_Font_Info, pyface.ptr.get(), &info);
        auto pyinfo = py::dict{};
#define COPY_FIELD(field) pyinfo[#field] = info.field
        COPY_FIELD(version);
        COPY_FIELD(notice);
        COPY_FIELD(full_name);
        COPY_FIELD(family_name);
        COPY_FIELD(weight);
        COPY_FIELD(italic_angle);
        COPY_FIELD(is_fixed_pitch);
        COPY_FIELD(underline_position);
        COPY_FIELD(underline_thickness);
#undef COPY_FIELD
        return pyinfo;
      })
    .def(
      "get_sfnt_name_table",
      // Don't bother returning a
      //   std::unordered_map<
      //     std::tuple<FT_UShort, FT_UShort, FT_UShort, FT_UShort>, py::bytes>
      // because std::hash is not specialized for tuples...
      [](Face const& pyface) -> py::dict {
        auto face = pyface.ptr.get();
        if (!FT_IS_SFNT(face)) {
          throw std::runtime_error("Font not using the SFNT storage scheme");
        }
        auto table = py::dict{};
        auto name_count = FT_Get_Sfnt_Name_Count(face);
        for (auto i = 0u; i < name_count; ++i) {
          auto sfnt_name = FT_SfntName{};
          FT_CHECK(FT_Get_Sfnt_Name, face, i, &sfnt_name);
          table[py::make_tuple(sfnt_name.platform_id, sfnt_name.encoding_id,
                               sfnt_name.language_id, sfnt_name.name_id)]
            = py::bytes((char*)(sfnt_name.string), sfnt_name.string_len);
        }
        return table;
      }, R"__doc__(
Return the SFNT names table.

The returned dict maps ``(platform_id, encoding_id, language_id, name_id)``
keys to the corresponding 'name' bytestrings.
)__doc__")
    .def(
      "get_sfnt_table",
      [](Face const& pyface, std::string tag) -> std::optional<py::dict> {
        auto face = pyface.ptr.get();
        if (!FT_IS_SFNT(face)) {
          throw std::runtime_error("Font not using the SFNT storage scheme");
        }
        auto table = py::dict{};
        auto copy_field = [&](char const* name, auto value) -> void {
          if constexpr (std::is_array_v<decltype(value)>) {
            table[name] =
              py::bytes(
                static_cast<char*>(value), std::extent_v<decltype(value)>);
          } else {
            table[name] = value;
          }
        };
#define COPY_FIELD(field) copy_field(#field, ptr->field)
        if (tag == "head") {
          auto ptr =
            // backcompat: FT_SFNT_HEAD in ft 2.5.4.  Same for other ft_sfnt's.
            static_cast<TT_Header*>(FT_Get_Sfnt_Table(face, ft_sfnt_head));
          if (!ptr) {
            return {};
          }
          COPY_FIELD(Table_Version);
          COPY_FIELD(Font_Revision);
          COPY_FIELD(CheckSum_Adjust);
          COPY_FIELD(Magic_Number);
          COPY_FIELD(Flags);
          COPY_FIELD(Units_Per_EM);
          table["Created"] = py::make_tuple(ptr->Created[0], ptr->Created[1]);
          table["Modified"] = py::make_tuple(ptr->Modified[0], ptr->Modified[1]);
          COPY_FIELD(xMin);
          COPY_FIELD(yMin);
          COPY_FIELD(xMax);
          COPY_FIELD(yMax);
          COPY_FIELD(Mac_Style);
          COPY_FIELD(Lowest_Rec_PPEM);
          COPY_FIELD(Font_Direction);
          COPY_FIELD(Index_To_Loc_Format);
          COPY_FIELD(Glyph_Data_Format);
        } else if (tag == "maxp") {
          auto ptr =
            static_cast<TT_MaxProfile*>(FT_Get_Sfnt_Table(face, ft_sfnt_maxp));
          COPY_FIELD(version);
          COPY_FIELD(numGlyphs);
          COPY_FIELD(maxPoints);
          COPY_FIELD(maxContours);
          COPY_FIELD(maxCompositePoints);
          COPY_FIELD(maxCompositeContours);
          COPY_FIELD(maxZones);
          COPY_FIELD(maxTwilightPoints);
          COPY_FIELD(maxStorage);
          COPY_FIELD(maxFunctionDefs);
          COPY_FIELD(maxInstructionDefs);
          COPY_FIELD(maxStackElements);
          COPY_FIELD(maxSizeOfInstructions);
          COPY_FIELD(maxComponentElements);
          COPY_FIELD(maxComponentDepth);
        } else if (tag == "OS/2") {
          auto ptr =
            static_cast<TT_OS2*>(FT_Get_Sfnt_Table(face, ft_sfnt_os2));
          if (!ptr) {
            return {};
          }
          // NOTE: Don't bother with optional tags.
          COPY_FIELD(version);
          COPY_FIELD(xAvgCharWidth);
          COPY_FIELD(usWeightClass);
          COPY_FIELD(usWidthClass);
          COPY_FIELD(fsType);
          COPY_FIELD(ySubscriptXSize);
          COPY_FIELD(ySubscriptYSize);
          COPY_FIELD(ySubscriptXOffset);
          COPY_FIELD(ySubscriptYOffset);
          COPY_FIELD(ySuperscriptXSize);
          COPY_FIELD(ySuperscriptYSize);
          COPY_FIELD(ySuperscriptXOffset);
          COPY_FIELD(ySuperscriptYOffset);
          COPY_FIELD(yStrikeoutSize);
          COPY_FIELD(yStrikeoutPosition);
          COPY_FIELD(sFamilyClass);
          COPY_FIELD(panose);
          COPY_FIELD(ulUnicodeRange1);
          COPY_FIELD(ulUnicodeRange2);
          COPY_FIELD(ulUnicodeRange3);
          COPY_FIELD(ulUnicodeRange4);
          COPY_FIELD(achVendID);
          COPY_FIELD(fsSelection);
          COPY_FIELD(usFirstCharIndex);
          COPY_FIELD(usLastCharIndex);
          COPY_FIELD(sTypoAscender);
          COPY_FIELD(sTypoDescender);
          COPY_FIELD(sTypoLineGap);
          COPY_FIELD(usWinAscent);
          COPY_FIELD(usWinDescent);
        } else if (tag == "hhea") {
          auto ptr =
            static_cast<TT_HoriHeader*>(FT_Get_Sfnt_Table(face, ft_sfnt_hhea));
          if (!ptr) {
            return {};
          }
          // NOTE: Skip reserved, {long,short}_metrics.
          COPY_FIELD(Version);
          COPY_FIELD(Ascender);
          COPY_FIELD(Descender);
          COPY_FIELD(Line_Gap);
          COPY_FIELD(advance_Width_Max);
          COPY_FIELD(min_Left_Side_Bearing);
          COPY_FIELD(min_Right_Side_Bearing);
          COPY_FIELD(xMax_Extent);
          COPY_FIELD(caret_Slope_Rise);
          COPY_FIELD(caret_Slope_Run);
          COPY_FIELD(caret_Offset);
          COPY_FIELD(metric_Data_Format);
          COPY_FIELD(number_Of_HMetrics);
        } else if (tag == "vhea") {
          auto ptr =
            static_cast<TT_VertHeader*>(FT_Get_Sfnt_Table(face, ft_sfnt_vhea));
          if (!ptr) {
            throw std::runtime_error("No \"vhea\" table");
          }
          // NOTE: Skip reserved, {long,short}_metrics.
          COPY_FIELD(Version);
          COPY_FIELD(Ascender);
          COPY_FIELD(Descender);
          COPY_FIELD(Line_Gap);
          COPY_FIELD(advance_Height_Max);
          COPY_FIELD(min_Top_Side_Bearing);
          COPY_FIELD(min_Bottom_Side_Bearing);
          COPY_FIELD(yMax_Extent);
          COPY_FIELD(caret_Slope_Rise);
          COPY_FIELD(caret_Slope_Run);
          COPY_FIELD(caret_Offset);
          COPY_FIELD(metric_Data_Format);
          COPY_FIELD(number_Of_VMetrics);
        } else if (tag == "post") {
          auto ptr =
            static_cast<TT_Postscript*>(FT_Get_Sfnt_Table(face, ft_sfnt_post));
          if (!ptr) {
            return {};
          }
          COPY_FIELD(FormatType);
          // NOTE: ft2font splits italicAngle into two uint16s but this seems wrong.
          COPY_FIELD(italicAngle);
          COPY_FIELD(underlinePosition);
          COPY_FIELD(underlineThickness);
          COPY_FIELD(isFixedPitch);
          COPY_FIELD(minMemType42);
          COPY_FIELD(maxMemType42);
          COPY_FIELD(minMemType1);
          COPY_FIELD(maxMemType1);
        } else if (tag == "pclt") {
          auto ptr =
            static_cast<TT_PCLT*>(FT_Get_Sfnt_Table(face, ft_sfnt_pclt));
          if (!ptr) {
            return {};
          }
          // NOTE: Skip reserved.
          COPY_FIELD(Version);
          COPY_FIELD(FontNumber);
          COPY_FIELD(Pitch);
          COPY_FIELD(xHeight);
          COPY_FIELD(Style);
          COPY_FIELD(TypeFamily);
          COPY_FIELD(CapHeight);
          COPY_FIELD(SymbolSet);
          COPY_FIELD(TypeFace);
          COPY_FIELD(CharacterComplement);
          COPY_FIELD(FileName);
          COPY_FIELD(StrokeWeight);
          COPY_FIELD(WidthType);
          COPY_FIELD(SerifStyle);
        } else {
          throw std::runtime_error("Invalid SFNT table");
        }
#undef COPY_FIELD
        return table;
      })
    .def(
      "load_char",
      [](Face const& pyface, FT_ULong codepoint, FT_Int32 flags) -> void {
        FT_CHECK(FT_Load_Char, pyface.ptr.get(), codepoint, flags);
      },
      "codepoint"_a, "flags"_a)
    .def(
      "select_charmap",
      [](Face const& pyface, FT_Encoding encoding) -> void {
        FT_CHECK(FT_Select_Charmap, pyface.ptr.get(), encoding);
      },
      "encoding"_a)
    .def(
      "set_char_size",
      [](Face const& pyface, double pt_size, double dpi) -> void {
        auto face = pyface.ptr.get();
        FT_CHECK(FT_Set_Char_Size,
                 face, pt_size * 64, 0, dpi * pyface.hinting_factor, dpi);
        auto transform =
          FT_Matrix{FT_Fixed(65536 / pyface.hinting_factor), 0, 0, 65536};
        FT_Set_Transform(face, &transform, nullptr);
      },
      "pt_size"_a, "dpi"_a)
    ;

  py::class_<Glyph>(m, "Glyph", R"__doc__(
A lightweight wrapper around a ``FT_Glyph``.

This object cannot be constructed directly.  Instead, load a glyph in a face's
glyph slot with `Face.load_char`, then access the face's `glyph` property
(which calls ``FT_Get_Glyph``).

This class exposes the attributes of the original glyph slot and its glyph
metrics.  Length attibutes are in pixels (this module handles conversion from
26.6 and 16.6 fixed point formats internally).
)__doc__")
    .def_property_readonly(
      "width",
      [](Glyph& pyglyph) -> double {
        return pyglyph.metrics.width / 64. / pyglyph.hinting_factor;
      })
    .def_property_readonly(
      "height",
      [](Glyph& pyglyph) -> double {
        return pyglyph.metrics.height / 64.;
      })
    .def_property_readonly(
      "horiBearingX",
      [](Glyph& pyglyph) -> double {
        return pyglyph.metrics.horiBearingX / 64. / pyglyph.hinting_factor;
      })
    .def_property_readonly(
      "horiBearingY",
      [](Glyph& pyglyph) -> double {
        return pyglyph.metrics.horiBearingY / 64.;
      })
    .def_property_readonly(
      "horiAdvance",
      [](Glyph& pyglyph) -> double {
        return pyglyph.metrics.horiAdvance / 64.;
      })
    .def_property_readonly(
      "vertBearingX",
      [](Glyph& pyglyph) -> double {
        return pyglyph.metrics.vertBearingX / 64.;
      })
    .def_property_readonly(
      "vertBearingY",
      [](Glyph& pyglyph) -> double {
        return pyglyph.metrics.vertBearingY / 64.;
      })
    .def_property_readonly(
      "vertAdvance",
      [](Glyph& pyglyph) -> double {
        return pyglyph.metrics.vertAdvance / 64.;
      })
    .def_property_readonly(
      "linearHoriAdvance",
      [](Glyph& pyglyph) -> double {
        return pyglyph.linearHoriAdvance / 65536. / pyglyph.hinting_factor;
      })
    .def_property_readonly(
      "linearVertAdvance",
      [](Glyph& pyglyph) -> double {
        return pyglyph.linearVertAdvance / 65536.;
      })
    .def(
      "get_cbox",
      [](Glyph& pyglyph, FT_UInt mode)
      -> std::tuple<double, double, double, double> {
        auto bbox = FT_BBox{};
        FT_Glyph_Get_CBox(pyglyph.ptr.get(), mode, &bbox);
        auto conv = ((mode == FT_GLYPH_BBOX_SUBPIXELS)
                     || mode == FT_GLYPH_BBOX_GRIDFIT) ? 1. / 64 : 1.;
        return {bbox.xMin * conv, bbox.xMax * conv,
                bbox.yMin * conv, bbox.yMax * conv};
      });

  py::class_<Layout>(m, "Layout", R"__doc__(
A text (and rectangles) layout engine.

Use as follows::

   # Construct a layout.
   layout = Layout.simple(...)
   # or
   layout = Layout.manual(...)
   # Access the layout's bounds:
   layout.xMin, layout.xMax, layout.yMin, layout.yMax
   # Render the text into a numpy array:
   result = layout.render()
)__doc__")
    .def("render", &Layout::render, "antialiased"_a, R"__doc__(
Render the laid out text into a numpy array.
)__doc__")
    .def_property_readonly(
      "xMin",
      [](Layout& layout) -> double { return layout.bbox.xMin / 64.; })
    .def_property_readonly(
      "xMax",
      [](Layout& layout) -> double { return layout.bbox.xMax / 64.; })
    .def_property_readonly(
      "yMin",
      [](Layout& layout) -> double { return layout.bbox.yMin / 64.; })
    .def_property_readonly(
      "yMax",
      [](Layout& layout) -> double { return layout.bbox.yMax / 64.; })
    .def_static(
      "simple",
      [](std::u32string const& string, Face const& pyface, FT_Int32 flags)
      -> Layout {
        return Layout::simple(string, pyface.ptr.get(), flags);
      },
      "string"_a, "face"_a, "flags"_a, R"__doc__(
Layout a string by positioning individual glyphs one after the other.

After each glyph, the current position is advanced by the glyph's advance and
the kerning between this glyph and the next one.
)__doc__")
    .def_static(
      "manual",
      [](std::vector<std::tuple<Glyph, double, double>> const& positioned_pyglyphs,
         std::vector<rect_t> const& rectangles,
         FT_Int32 flags)
      -> Layout {
        auto positioned_glyphs
          = std::vector<std::tuple<FT_Glyph, double, double>>{};
        std::transform(
          positioned_pyglyphs.begin(), positioned_pyglyphs.end(),
          std::back_inserter(positioned_glyphs),
          [](std::tuple<Glyph, double, double> const& positioned_pyglyph)
          -> std::tuple<FT_Glyph, double, double> {
            auto& [pyglyph, x, y] = positioned_pyglyph;
            return {pyglyph.ptr.get(), x, y};
          });
        return Layout::manual(positioned_glyphs, rectangles, flags);
      },
      "positioned_glyphs"_a, "rectangles"_a, "flags"_a, R"__doc__(
Prepare a list of manually laid out glyphs and rectangles for rendering.
)__doc__");

  m.def(
    "glyph_to_path",
    [](Glyph const& glyph)
    -> std::tuple<std::vector<std::tuple<double, double>>, std::vector<int>> {
      if (glyph.ptr->format != FT_GLYPH_FORMAT_OUTLINE) {
        throw std::runtime_error("Not an outline glyph");
      }
      // outline.n_points seems invalid :-(
      auto path = std::tuple<std::vector<std::tuple<double, double>>,
                             std::vector<int>>{};
      auto& [vertices, codes] = path;
      auto move_to =
        [](FT_Vector const* to, void* raw) -> int {
        auto& [vertices, codes] = *static_cast<decltype(&path)>(raw);
        if (vertices.size()) {
          vertices.push_back({NAN, NAN});
          codes.push_back(79);
        }
        vertices.push_back({to->x / 64., to->y / 64.});
        codes.push_back(1);
        return 0;
      };
      auto line_to =
        [](FT_Vector const* to, void* raw) -> int {
        auto& [vertices, codes] = *static_cast<decltype(&path)>(raw);
        vertices.push_back({to->x / 64., to->y / 64.});
        codes.push_back(2);
        return 0;
      };
      auto conic_to =
        [](FT_Vector const* control, FT_Vector const* to, void* raw) -> int {
        auto& [vertices, codes] = *static_cast<decltype(&path)>(raw);
        vertices.push_back({control->x / 64., control->y / 64.});
        vertices.push_back({to->x / 64., to->y / 64.});
        codes.push_back(3);
        codes.push_back(3);
        return 0;
      };
      auto cubic_to =
        [](FT_Vector const* control1, FT_Vector const* control2,
           FT_Vector const* to, void* raw) -> int {
        auto& [vertices, codes] = *static_cast<decltype(&path)>(raw);
        vertices.push_back({control1->x / 64., control1->y / 64.});
        vertices.push_back({control2->x / 64., control2->y / 64.});
        vertices.push_back({to->x / 64., to->y / 64.});
        codes.push_back(4);
        codes.push_back(4);
        codes.push_back(4);
        return 0;
      };
      auto outline_funcs =
        FT_Outline_Funcs{move_to, line_to, conic_to, cubic_to, 0, 0};
      FT_Outline_Decompose(
        &reinterpret_cast<FT_OutlineGlyph>(glyph.ptr.get())->outline,
        &outline_funcs,
        &path);
      if (vertices.size()) {
        vertices.push_back({NAN, NAN});
        codes.push_back(79);
      } else {
        vertices.push_back({0, 0});
        codes.push_back(1);
      }
      return path;
    },
    "glyph"_a, R"__doc__(
Extract a glyph's outline in Matplotlib's ``(vertices, codes)`` format.
)__doc__");
}

}
