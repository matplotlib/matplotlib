/* A python interface to freetype2 */
#ifndef _FT2FONT_H
#define _FT2FONT_H
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <utility>

extern "C" {
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_GLYPH_H
#include FT_SFNT_NAMES_H
#include FT_TYPE1_TABLES_H
#include FT_TRUETYPE_TABLES_H
}
#include "CXX/Extensions.hxx"
#include "CXX/Objects.hxx"


// the freetype string rendered into a width, height buffer
class FT2Image : public Py::PythonExtension<FT2Image> {
public:
  // FT2Image();
  FT2Image(unsigned long width, unsigned long height);
  ~FT2Image();

  static void init_type();

  void draw_bitmap(FT_Bitmap* bitmap, FT_Int x, FT_Int y);
  void write_bitmap(const char* filename) const;
  void draw_rect(unsigned long x0, unsigned long y0, 
		 unsigned long x1, unsigned long y1);
  void draw_rect_filled(unsigned long x0, unsigned long y0, 
			unsigned long x1, unsigned long y1);
  
  unsigned int get_width() const { return _width; };
  unsigned int get_height() const { return _height; };
  const unsigned char *const get_buffer() const { return _buffer; };

  static char write_bitmap__doc__ [];
  Py::Object py_write_bitmap(const Py::Tuple & args);
  static char draw_rect__doc__ [];
  Py::Object py_draw_rect(const Py::Tuple & args);
  static char draw_rect_filled__doc__ [];
  Py::Object py_draw_rect_filled(const Py::Tuple & args);
  static char as_str__doc__ [];
  Py::Object py_as_str(const Py::Tuple & args);
  static char as_rgb_str__doc__ [];
  Py::Object py_as_rgb_str(const Py::Tuple & args);
  static char as_rgba_str__doc__ [];
  Py::Object py_as_rgba_str(const Py::Tuple & args);

  Py::Object py_get_width(const Py::Tuple & args);
  Py::Object py_get_height(const Py::Tuple & args);

 private:
  bool _isDirty;
  unsigned char *_buffer;
  unsigned long _width;
  unsigned long _height;
  FT2Image* _rgbCopy;
  FT2Image* _rgbaCopy;

  void makeRgbCopy();
  void makeRgbaCopy();

  void resize(unsigned long width, unsigned long height);
};


class Glyph : public Py::PythonExtension<Glyph> {
public:
  Glyph( const FT_Face&, const FT_Glyph&, size_t);
  ~Glyph();
  int setattr( const char *_name, const Py::Object &value );
  Py::Object getattr( const char *_name );
  static void init_type(void);
  size_t glyphInd;
  Py::Object get_path( const FT_Face& face );
private:
  Py::Dict __dict__;
  static char get_path__doc__[];
};

class FT2Font : public Py::PythonExtension<FT2Font> {

public:
  FT2Font(std::string);
  ~FT2Font();
  static void init_type(void);
  Py::Object clear(const Py::Tuple & args);
  Py::Object set_size(const Py::Tuple & args);
  Py::Object set_charmap(const Py::Tuple & args);
  Py::Object set_text(const Py::Tuple & args, const Py::Dict & kwargs);
  Py::Object get_glyph(const Py::Tuple & args);
  Py::Object get_kerning(const Py::Tuple & args);
  Py::Object get_num_glyphs(const Py::Tuple & args);
  Py::Object load_char(const Py::Tuple & args, const Py::Dict & kws);
  Py::Object get_width_height(const Py::Tuple & args);
  Py::Object get_descent(const Py::Tuple & args);
  Py::Object draw_rect_filled(const Py::Tuple & args);
  Py::Object get_xys(const Py::Tuple & args);
  Py::Object draw_glyphs_to_bitmap(const Py::Tuple & args);
  Py::Object draw_glyph_to_bitmap(const Py::Tuple & args);
  Py::Object get_glyph_name(const Py::Tuple & args);
  Py::Object get_charmap(const Py::Tuple & args);
  Py::Object get_sfnt(const Py::Tuple & args);
  Py::Object get_name_index(const Py::Tuple & args);
  Py::Object get_ps_font_info(const Py::Tuple & args);
  Py::Object get_sfnt_table(const Py::Tuple & args);
  Py::Object get_image(const Py::Tuple & args);
  Py::Object attach_file(const Py::Tuple & args);
  int setattr( const char *_name, const Py::Object &value );
  Py::Object getattr( const char *_name );
  FT2Image* image;

private:
  Py::Dict __dict__;
  FT_Face       face;
  FT_Matrix     matrix;                 /* transformation matrix */
  FT_Vector     pen;                    /* untransformed origin  */
  FT_Error      error;
  std::vector<FT_Glyph> glyphs;
  std::vector<FT_Vector> pos;
  std::vector<Glyph*> gms;
  double angle;
  double ptsize;
  double dpi;


  FT_BBox compute_string_bbox();
  void set_scalable_attributes();

  static char clear__doc__ [];
  static char set_size__doc__ [];
  static char set_charmap__doc__ [];
  static char set_text__doc__ [];
  static char get_glyph__doc__ [];
  static char get_num_glyphs__doc__ [];
  static char load_char__doc__ [];
  static char get_width_height__doc__ [];
  static char get_descent__doc__ [];
  static char get_kerning__doc__ [];
  static char draw_glyphs_to_bitmap__doc__ [];
  static char get_xys__doc__ [];
  static char draw_glyph_to_bitmap__doc__ [];
  static char get_glyph_name__doc__[];
  static char get_charmap__doc__[];
  static char get_sfnt__doc__ [];
  static char get_name_index__doc__[];
  static char get_ps_font_info__doc__[];
  static char get_sfnt_table__doc__[];
  static char get_image__doc__[];
  static char attach_file__doc__[];
};

// the extension module
class ft2font_module : public Py::ExtensionModule<ft2font_module>

{
public:
  ft2font_module()
    : Py::ExtensionModule<ft2font_module>( "ft2font" )
  {
    FT2Image::init_type();
    Glyph::init_type();
    FT2Font::init_type();

    add_varargs_method("FT2Font", &ft2font_module::new_ft2font, 
		       "FT2Font");
    add_varargs_method("FT2Image", &ft2font_module::new_ft2image, 
		       "FT2Image");
    initialize( "The ft2font module" );
  }
  
  ~ft2font_module(); 
  //static FT_Library ft2Library;
  
private:

  Py::Object new_ft2font (const Py::Tuple &args);
  Py::Object new_ft2image (const Py::Tuple &args);
};



#endif
