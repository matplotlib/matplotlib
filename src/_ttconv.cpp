/* -*- mode: c++; c-basic-offset: 4 -*- */

/*
  _ttconv.c

  Python wrapper for TrueType conversion library in ../ttconv.
 */
#define PY_SSIZE_T_CLEAN
#include "mplutils.h"

#include <Python.h>
#include "ttconv/pprdrv.h"
#include "py_exceptions.h"
#include <vector>
#include <cassert>

/**
 * An implementation of TTStreamWriter that writes to a Python
 * file-like object.
 */
class PythonFileWriter : public TTStreamWriter
{
    PyObject *_write_method;

  public:
    PythonFileWriter()
    {
        _write_method = NULL;
    }

    ~PythonFileWriter()
    {
        Py_XDECREF(_write_method);
    }

    void set(PyObject *write_method)
    {
        Py_XDECREF(_write_method);
        _write_method = write_method;
        Py_XINCREF(_write_method);
    }

    virtual void write(const char *a)
    {
        PyObject *result = NULL;
        if (_write_method) {
            PyObject *decoded = NULL;
            decoded = PyUnicode_DecodeLatin1(a, strlen(a), "");
            if (decoded == NULL) {
                throw py::exception();
            }
            result = PyObject_CallFunctionObjArgs(_write_method, decoded, NULL);
            Py_DECREF(decoded);
            if (!result) {
                throw py::exception();
            }
            Py_DECREF(result);
        }
    }
};

int fileobject_to_PythonFileWriter(PyObject *object, void *address)
{
    PythonFileWriter *file_writer = (PythonFileWriter *)address;

    PyObject *write_method = PyObject_GetAttrString(object, "write");
    if (write_method == NULL || !PyCallable_Check(write_method)) {
        PyErr_SetString(PyExc_TypeError, "Expected a file-like object with a write method.");
        return 0;
    }

    file_writer->set(write_method);
    Py_DECREF(write_method);

    return 1;
}

int pyiterable_to_vector_int(PyObject *object, void *address)
{
    std::vector<int> *result = (std::vector<int> *)address;

    PyObject *iterator = PyObject_GetIter(object);
    if (!iterator) {
        return 0;
    }

    PyObject *item;
    while ((item = PyIter_Next(iterator))) {
        long value = PyLong_AsLong(item);
        Py_DECREF(item);
        if (value == -1 && PyErr_Occurred()) {
            return 0;
        }
        result->push_back((int)value);
    }

    Py_DECREF(iterator);

    return 1;
}

static PyObject *convert_ttf_to_ps(PyObject *self, PyObject *args, PyObject *kwds)
{
    const char *filename;
    PythonFileWriter output;
    int fonttype;
    std::vector<int> glyph_ids;

    static const char *kwlist[] = { "filename", "output", "fonttype", "glyph_ids", NULL };
    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwds,
                                     "yO&i|O&:convert_ttf_to_ps",
                                     (char **)kwlist,
                                     &filename,
                                     fileobject_to_PythonFileWriter,
                                     &output,
                                     &fonttype,
                                     pyiterable_to_vector_int,
                                     &glyph_ids)) {
        return NULL;
    }

    if (fonttype != 3 && fonttype != 42) {
        PyErr_SetString(PyExc_ValueError,
                        "fonttype must be either 3 (raw Postscript) or 42 "
                        "(embedded Truetype)");
        return NULL;
    }

    try
    {
        insert_ttfont(filename, output, (font_type_enum)fonttype, glyph_ids);
    }
    catch (TTException &e)
    {
        PyErr_SetString(PyExc_RuntimeError, e.getMessage());
        return NULL;
    }
    catch (const py::exception &)
    {
        return NULL;
    }
    catch (...)
    {
        PyErr_SetString(PyExc_RuntimeError, "Unknown C++ exception");
        return NULL;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef ttconv_methods[] =
{
    {
        "convert_ttf_to_ps", (PyCFunction)convert_ttf_to_ps, METH_VARARGS | METH_KEYWORDS,
        "convert_ttf_to_ps(filename, output, fonttype, glyph_ids)\n"
        "\n"
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
    },
    {0, 0, 0, 0}  /* Sentinel */
};

static const char *module_docstring =
    "Module to handle converting and subsetting TrueType "
    "fonts to Postscript Type 3, Postscript Type 42 and "
    "Pdf Type 3 fonts.";

static PyModuleDef ttconv_module = {
    PyModuleDef_HEAD_INIT,
    "ttconv",
    module_docstring,
    -1,
    ttconv_methods,
};

#pragma GCC visibility push(default)

PyMODINIT_FUNC
PyInit__ttconv(void)
{
    return PyModule_Create(&ttconv_module);
}

#pragma GCC visibility pop
