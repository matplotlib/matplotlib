/*
  _ttconv.c

  Python wrapper for TrueType conversion library in ../ttconv.
 */

#include <Python.h>
#include "ttconv/pprdrv.h"
#include <vector>

class PythonExceptionOccurred {

};

/**
 * An implementation of TTStreamWriter that writes to a Python
 * file-like object.
 */
class PythonFileWriter : public TTStreamWriter {
  PyObject* _write_method;

public:
  PythonFileWriter() {
    _write_method = NULL;
  }

  ~PythonFileWriter() {
    if (_write_method)
      Py_DECREF(_write_method);
  }

  void set(PyObject* write_method) {
    if (_write_method)
      Py_DECREF(_write_method);
    _write_method = write_method;
    if (_write_method)
      Py_INCREF(_write_method);
  }

  virtual void write(const char* a) {
    if (_write_method)
      if (! PyObject_CallFunction(_write_method, (char *)"s", a))
	throw PythonExceptionOccurred();
  }
};

int fileobject_to_PythonFileWriter(PyObject* object, void* address) {
  PythonFileWriter* file_writer = (PythonFileWriter*)address;

  PyObject* write_method = PyObject_GetAttrString(object, "write");
  if (write_method == NULL || ! PyCallable_Check(write_method)) {
    PyErr_SetString(PyExc_TypeError, "Expected a file-like object with a write method.");
    return 0;
  }

  file_writer->set(write_method);

  return 1;
}

int pyiterable_to_vector_int(PyObject* object, void* address) {
  std::vector<int>* result = (std::vector<int>*)address;

  PyObject* iterator = PyObject_GetIter(object);
  if (! iterator)
    return 0;

  PyObject* item;
  while ( (item = PyIter_Next(iterator)) ) {
    long value = PyInt_AsLong(item);
    if (value == -1 && PyErr_Occurred())
      return 0;
    result->push_back(value);
  }

  return 1;
}

static PyObject*
convert_ttf_to_ps(PyObject* self, PyObject* args, PyObject* kwds) {
  const char*		filename;
  PythonFileWriter	output;
  int			fonttype;
  std::vector<int>	glyph_ids;

  static const char *kwlist[] = { 
    "filename", "output", "fonttype", "glyph_ids", NULL };
  if (! PyArg_ParseTupleAndKeywords
      (args, kwds, 
       "sO&i|O&:convert_ttf_to_ps", 
       (char**)kwlist,
       &filename,
       fileobject_to_PythonFileWriter,
       &output,
       &fonttype,
       pyiterable_to_vector_int,
       &glyph_ids))
    return NULL;
  
  if (fonttype != 3 && fonttype != 42) {
    PyErr_SetString(PyExc_ValueError, 
		    "fonttype must be either 3 (raw Postscript) or 42 "
		    "(embedded Truetype)");
    return NULL;
  }

  try {
    insert_ttfont( filename, output, (font_type_enum)fonttype, glyph_ids );
  } catch (TTException& e) {
    PyErr_SetString(PyExc_RuntimeError, e.getMessage());
    return NULL;
  } catch (PythonExceptionOccurred& e) {
    return NULL;
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "Unknown C++ exception");
    return NULL;
  }

  Py_INCREF(Py_None);
  return Py_None;
}

class PythonDictionaryCallback : public TTDictionaryCallback {
  PyObject* _dict;

public:
  PythonDictionaryCallback(PyObject* dict) {
    _dict = dict;
  }

  virtual void add_pair(const char* a, const char* b) {
    PyObject* value = PyString_FromString(b);
    if (value)
      if (PyDict_SetItemString(_dict, a, value))
	throw PythonExceptionOccurred();
  }
};

static PyObject*
py_get_pdf_charprocs(PyObject* self, PyObject* args, PyObject* kwds) {
  const char*		filename;
  std::vector<int>	glyph_ids;
  PyObject*             result;

  static const char *kwlist[] = { "filename", "glyph_ids", NULL };
  if (! PyArg_ParseTupleAndKeywords
      (args, kwds, 
       "s|O&:convert_ttf_to_ps", 
       (char **)kwlist,
       &filename,
       pyiterable_to_vector_int,
       &glyph_ids))
    return NULL;

  result = PyDict_New();
  if (!result)
    return NULL;

  PythonDictionaryCallback dict(result);

  try {
    ::get_pdf_charprocs( filename, glyph_ids, dict );
  } catch (TTException& e) {
    PyErr_SetString(PyExc_RuntimeError, e.getMessage());
    return NULL;
  } catch (PythonExceptionOccurred& e) {
    return NULL;
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "Unknown C++ exception");
    return NULL;
  }

  return result;
}

static PyMethodDef ttconv_methods[] = {
  {"convert_ttf_to_ps", (PyCFunction)convert_ttf_to_ps, METH_KEYWORDS, 
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
   "for Type 42 fonts.\n"
   "glyph_ids (optional) is a list of glyph ids (integers) to keep when "
   "subsetting to a Type 3 font.  If glyph_ids is not provided or is None, "
   "then all glyphs will be included.  If any of the glyphs specified are "
   "composite glyphs, then the component glyphs will also be included."
  },
  {"get_pdf_charprocs", (PyCFunction)py_get_pdf_charprocs, METH_KEYWORDS, 
   "get_pdf_charprocs(filename, glyph_ids)\n"
   "\n"
   "Given a Truetype font file, returns a dictionary containing the PDF Type 3\n"
   "representation of its paths.  Useful for subsetting a Truetype font inside\n"
   "of a PDF file.\n"
   "\n"
   "filename is the path to a TTF font file.\n"
   "glyph_ids is a list of the numeric glyph ids to include.\n"
   "The return value is a dictionary where the keys are glyph names and\n"
   "the values are the stream content needed to render that glyph.  This\n"
   "is useful to generate the CharProcs dictionary in a PDF Type 3 font.\n"
  },
  {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC	/* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
initttconv(void) 
{
    PyObject* m;

    m = Py_InitModule3("ttconv", ttconv_methods,
                       "Module to handle converting and subsetting TrueType "
		       "fonts to Postscript Type 3, Postscript Type 42 and "
		       "Pdf Type 3 fonts.");
}

