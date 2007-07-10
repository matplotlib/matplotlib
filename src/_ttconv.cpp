#include "CXX/Extensions.hxx"
#include "CXX/Objects.hxx"
#include "ttconv/pprdrv.h"

class ttconv_module : public Py::ExtensionModule<ttconv_module>
{
public:
  ttconv_module()
    : Py::ExtensionModule<ttconv_module>( "ttconv" )
  {
    add_varargs_method("convert_ttf_to_ps", 
		       &ttconv_module::convert_ttf_to_ps,
		       ttconv_module::convert_ttf_to_ps__doc__);
    add_varargs_method("get_pdf_charprocs", 
		       &ttconv_module::get_pdf_charprocs,
		       ttconv_module::get_pdf_charprocs__doc__);

    initialize( "The ttconv module" );
  }

  Py::Object
  convert_ttf_to_ps(const Py::Tuple& args);
  static char convert_ttf_to_ps__doc__[];

  Py::Object
  get_pdf_charprocs(const Py::Tuple& args);
  static char get_pdf_charprocs__doc__[];
};

char  ttconv_module::convert_ttf_to_ps__doc__[] =
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
;

/**
 * An implementation of TTStreamWriter that writes to a Python
 * file-like object.
 */
class PythonFileWriter : public TTStreamWriter {
  Py::Callable _write_method;

public:
  PythonFileWriter(const Py::Object& file_like_object) {
    _write_method = file_like_object.getAttr( "write" );
  }

  virtual void write(const char* a) {
    Py::Tuple args(1);
    args[0] = Py::String(a);
    _write_method.apply(args);
  }
};

Py::Object
ttconv_module::convert_ttf_to_ps(const Py::Tuple & args) {
  args.verify_length(3, 4);

  std::string fname = Py::String(args[0]).as_std_string();

  PythonFileWriter python_file_writer(args[1]);

  long font_type = (long)Py::Int(args[2]);
  if ( font_type != 3 && font_type != 42 ) {
    throw Py::ValueError("Font type must be either 3 (raw Postscript) or 42 (embedded Truetype)");
  }

  std::vector<int> glyph_ids;
  if ( args.size() == 4 ) {
    if ( args[3] != Py::None() ) {
      Py::SeqBase< Py::Int > py_glyph_ids = args[3];
      size_t num_glyphs = py_glyph_ids.size();
      // If there are no included glyphs, just return
      if (num_glyphs == 0) {
	return Py::Object();
      }
      glyph_ids.reserve(num_glyphs);
      for (size_t i = 0; i < num_glyphs; ++i) {
	glyph_ids.push_back( (long) py_glyph_ids.getItem(i) );
      }
    }
  }

  try {
    insert_ttfont( fname.c_str(), python_file_writer, (font_type_enum) font_type, glyph_ids );
  } catch (TTException& e) {
    throw Py::RuntimeError(e.getMessage());
  }

  return Py::Object();
}

char  ttconv_module::get_pdf_charprocs__doc__[] =
"get_pdf_charprocs(filename, glyph_ids)\n"
"\n"
"Given a Truetype font file, returns a dictionary containing the PDF Type 3\n"
"representation of its path.  Useful for subsetting a Truetype font inside\n"
"of a PDF file.\n"
"\n"
"filename is the path to a TTF font file.\n"
"glyph_ids is a list of the numeric glyph ids to include.\n"
"The return value is a dictionary where the keys are glyph names and \n"
"the values are the stream content needed to render that glyph.  This\n"
"is useful to generate the CharProcs dictionary in a PDF Type 3 font.\n"
;

/**
 * An implementation of TTStreamWriter that writes to a Python
 * file-like object.
 */
class PythonDictionaryCallback : public TTDictionaryCallback {
  Py::Dict _dict;

public:
  PythonDictionaryCallback(const Py::Dict& dict) : _dict(dict) {

  }

  virtual void add_pair(const char* a, const char* b) {
    _dict.setItem(a, Py::String(b));
  }
};

Py::Object
ttconv_module::get_pdf_charprocs(const Py::Tuple & args) {
  args.verify_length(1, 2);

  Py::Dict result;

  std::string fname = Py::String(args[0]).as_std_string();

  std::vector<int> glyph_ids;
  if ( args.size() == 2 ) {
    if ( args[1] != Py::None() ) {
      Py::SeqBase< Py::Int > py_glyph_ids = args[1];
      size_t num_glyphs = py_glyph_ids.size();
      // If there are no included glyphs, just return
      if (num_glyphs == 0) {
	return result;
      }
      glyph_ids.reserve(num_glyphs);
      for (size_t i = 0; i < num_glyphs; ++i) {
	glyph_ids.push_back( (long) py_glyph_ids.getItem(i) );
      }
    }
  }

  PythonDictionaryCallback dictCallback(result);

  try {
    ::get_pdf_charprocs( fname.c_str(), glyph_ids, dictCallback );
  } catch (TTException& e) {
    throw Py::RuntimeError(e.getMessage());
  }

  return result;
}

#if defined(_MSC_VER)
DL_EXPORT(void)
#elif defined(__cplusplus)
  extern "C" void
#else
void
#endif
initttconv(void)
{
  static ttconv_module* ttconv = new ttconv_module;
}
