#include "CXX/Extensions.hxx"
#include "CXX/Objects.hxx"
#include "ttconv/pprdrv.h"

class ttf2ps_module : public Py::ExtensionModule<ttf2ps_module>
{
public:
  ttf2ps_module()
    : Py::ExtensionModule<ttf2ps_module>( "ttf2ps" )
  {
    add_varargs_method("convert_ttf_to_ps", 
		       &ttf2ps_module::convert_ttf_to_ps,
		       ttf2ps_module::convert_ttf_to_ps__doc__);

    initialize( "The ttf2ps module" );
  }

  Py::Object
  convert_ttf_to_ps(const Py::Tuple & args);

  static char convert_ttf_to_ps__doc__[];
};

char  ttf2ps_module::convert_ttf_to_ps__doc__[] =
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
ttf2ps_module::convert_ttf_to_ps(const Py::Tuple & args) {
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
    insert_ttfont( fname.c_str(), python_file_writer, font_type, glyph_ids );
  } catch (TTException& e) {
    throw Py::RuntimeError(e.getMessage());
  }

  return Py::Object();
}

#if defined(_MSC_VER)
DL_EXPORT(void)
#elif defined(__cplusplus)
  extern "C" void
#else
void
#endif
initttf2ps(void)
{
  static ttf2ps_module* ttf2ps = new ttf2ps_module;
}
