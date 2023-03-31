/* -*- mode: c++; c-basic-offset: 4 -*- */

/*
  _ttconv.c

  Python wrapper for TrueType conversion library in ../ttconv.
 */
#include "mplutils.h"

#include <pybind11/pybind11.h>
#include "ttconv/pprdrv.h"
#include <vector>

namespace py = pybind11;

/**
 * An implementation of TTStreamWriter that writes to a Python
 * file-like object.
 */
class PythonFileWriter : public TTStreamWriter
{
  py::function _write_method;

  public:
    PythonFileWriter(py::object& file_object)
    : _write_method(file_object.attr("write")) {}

    virtual void write(const char *a)
    {
        PyObject* decoded = PyUnicode_DecodeLatin1(a, strlen(a), "");
        if (decoded == NULL) {
            throw py::error_already_set();
        }
        _write_method(py::handle(decoded));
        Py_DECREF(decoded);
    }
};

static void convert_ttf_to_ps(
    const char *filename,
    py::object &output,
    int fonttype,
    py::iterable* glyph_ids)
{
    PythonFileWriter output_(output);

    std::vector<int> glyph_ids_;
    if (glyph_ids) {
        for (py::handle glyph_id: *glyph_ids) {
            glyph_ids_.push_back(glyph_id.cast<int>());
        }
    }

    if (fonttype != 3 && fonttype != 42) {
        throw py::value_error(
            "fonttype must be either 3 (raw Postscript) or 42 (embedded Truetype)");
    }

    try
    {
        insert_ttfont(filename, output_, static_cast<font_type_enum>(fonttype), glyph_ids_);
    }
    catch (TTException &e)
    {
        throw std::runtime_error(e.getMessage());
    }
    catch (...)
    {
        throw std::runtime_error("Unknown C++ exception");
    }
}

PYBIND11_MODULE(_ttconv, m) {
    m.doc() = "Module to handle converting and subsetting TrueType "
              "fonts to Postscript Type 3, Postscript Type 42 and "
              "Pdf Type 3 fonts.";
    m.def("convert_ttf_to_ps", &convert_ttf_to_ps,
        py::arg("filename"),
        py::arg("output"),
        py::arg("fonttype"),
        py::arg("glyph_ids") = py::none(),
        "Converts the Truetype font into a Type 3 or Type 42 Postscript font, "
        "optionally subsetting the font to only the desired set of characters.\n"
        "\n"
        "filename is the path to a TTF font file.\n"
        "output is a Python file-like object with a write method that the Postscript "
        "font data will be written to.\n"
        "fonttype may be either 3 or 42.  Type 3 is a \"raw Postscript\" font. "
        "Type 42 is an embedded Truetype font.  Glyph subsetting is not supported "
        "for Type 42 fonts within this module (needs to be done externally).\n"
        "glyph_ids (optional) is a list of glyph ids (integers) to keep when "
        "subsetting to a Type 3 font.  If glyph_ids is not provided or is None, "
        "then all glyphs will be included.  If any of the glyphs specified are "
        "composite glyphs, then the component glyphs will also be included."
    );
}
