/* image.h	
 *
 */

#ifndef _IMAGE_H
#define _IMAGE_H

#include "Python.h"

#include "agg_trans_affine.h"
#include "agg_rendering_buffer.h"
#include "agg_color_rgba8.h"
#include "CXX/Extensions.hxx"



class Image : public Py::PythonExtension<Image> {
public:
  Image();
  virtual ~Image();

  static void init_type(void);
  int setattr( const char*, const Py::Object & );
  Py::Object getattr( const char * name );

  Py::Object apply_rotation(const Py::Tuple& args);
  Py::Object apply_scaling(const Py::Tuple& args);
  Py::Object apply_translation(const Py::Tuple& args);
  Py::Object as_str(const Py::Tuple& args);
  Py::Object reset_matrix(const Py::Tuple& args);
  Py::Object resize(const Py::Tuple& args);
  Py::Object get_aspect(const Py::Tuple& args);
  Py::Object get_size(const Py::Tuple& args);
  Py::Object get_interpolation(const Py::Tuple& args);
  Py::Object set_interpolation(const Py::Tuple& args);
  Py::Object set_aspect(const Py::Tuple& args);
  Py::Object write_png(const Py::Tuple& args);
  Py::Object set_bg(const Py::Tuple& args);

  enum { BICUBIC=0, BILINEAR, BLACKMAN100, BLACKMAN256, BLACKMAN64, 
	 NEAREST, SINC144, SINC256, SINC64, SPLINE16, SPLINE36};
  enum { ASPECT_PRESERVE=0, ASPECT_FREE};

  agg::int8u *bufferIn;
  agg::rendering_buffer *rbufIn;
  size_t colsIn, rowsIn;             

  agg::int8u *bufferOut;
  agg::rendering_buffer *rbufOut;
  size_t colsOut, rowsOut;             
  unsigned BPP;

  unsigned interpolation, aspect;
  agg::rgba bg;  
private:
  Py::Dict __dict__;
  agg::trans_affine srcMatrix, imageMatrix;


  static char apply_rotation__doc__[];
  static char apply_scaling__doc__[];
  static char apply_translation__doc__[];
  static char as_str__doc__[];
  static char reset_matrix__doc__[];
  static char resize__doc__[];
  static char get_aspect__doc__[];
  static char get_size__doc__[];
  static char get_interpolation__doc__[];
  static char set_interpolation__doc__[];
  static char set_aspect__doc__[];
  static char write_png__doc__[];
  static char set_bg__doc__[];

};


/*
class ImageComposite : public Py::PythonExtension<ImageComposite> {

}
*/


// the extension module
class _image_module : public Py::ExtensionModule<_image_module>
{
public:
  _image_module() : Py::ExtensionModule<_image_module>( "_image" )
  {
    Image::init_type();

    add_varargs_method("fromarray", &_image_module::fromarray, 
		       "fromarray");
    add_varargs_method("readpng", &_image_module::readpng, 
		       "readpng");
    add_varargs_method("from_images", &_image_module::from_images, 
		       "from_images");
    initialize( "The _image module" );
  }
  
  ~_image_module() {} 
  
private:

  Py::Object fromarray (const Py::Tuple &args);
  Py::Object readpng (const Py::Tuple &args);
  Py::Object from_images (const Py::Tuple &args);
  static char _image_module_fromarray__doc__[];

};



#endif

