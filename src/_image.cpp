#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <png.h>
#include "Python.h"

#ifdef NUMARRAY
#include "numarray/arrayobject.h" 
#else
#include "Numeric/arrayobject.h" 
#endif   

#include "agg_pixfmt_rgb24.h"
#include "agg_pixfmt_rgba32.h"
#include "agg_rendering_buffer.h"
#include "agg_rasterizer_scanline_aa.h"
#include "agg_path_storage.h"
#include "agg_conv_transform.h"
#include "agg_span_image_filter_rgb24.h"
#include "agg_span_image_filter_rgba32.h"
#include "agg_span_interpolator_linear.h"
#include "agg_scanline_u8.h"
#include "agg_scanline_p32.h"
#include "agg_renderer_scanline.h"

#include "_image.h"
#include "mplutils.h"



typedef agg::pixel_formats_rgba32<agg::order_rgba32> pixfmt;
typedef agg::renderer_base<pixfmt> renderer_base;
typedef agg::span_interpolator_linear<> interpolator_type;
typedef agg::span_image_filter_rgba32_bilinear<agg::order_rgba32, interpolator_type> span_gen_type;
typedef agg::renderer_scanline_u<renderer_base, span_gen_type> renderer_type;
typedef agg::rasterizer_scanline_aa<> rasterizer;


Image::Image() :
  bufferIn(NULL), rbufIn(NULL), colsIn(0), rowsIn(0),
  bufferOut(NULL), rbufOut(NULL), colsOut(0), rowsOut(0),  BPP(4),
  interpolation(BILINEAR), aspect(ASPECT_FREE), bg(1,1,1,0) {
  _VERBOSE("Image::Image");
}

Image::~Image() {  
  _VERBOSE("Image::~Image");
  
  delete [] bufferIn; bufferIn = NULL;
  delete rbufIn; rbufIn=NULL;
  
  delete rbufOut; rbufOut = NULL;
  delete [] bufferOut; bufferOut=NULL;
}

int
Image::setattr( const char * name, const Py::Object & value ) {
  _VERBOSE("Image::setattr");
  __dict__[name] = value;
  return 0;
}

Py::Object 
Image::getattr( const char * name ) {
  _VERBOSE("Image::getattro");
  if ( __dict__.hasKey(name) ) return __dict__[name];
  else return getattr_default( name );
  
}

char Image::apply_rotation__doc__[] = 
"apply_rotation(angle)\n"
"\n"
"Apply the rotation (degrees) to image"
;
Py::Object
Image::apply_rotation(const Py::Tuple& args) {
  _VERBOSE("Image::apply_rotation");
  
  args.verify_length(1);  
  double r = Py::Float(args[0]);
  
  
  agg::affine_matrix M = agg::rotation_matrix( r * agg::pi / 180.0);	      
  srcMatrix *= M;
  imageMatrix *= M;
  return Py::Object();  
}


char Image::set_bg__doc__[] = 
"set_bg(r,g,b,a)\n"
"\n"
"Set the background color"
;

Py::Object
Image::set_bg(const Py::Tuple& args) {
  _VERBOSE("Image::set_bg");
  
  args.verify_length(4);
  bg.r = Py::Float(args[0]);
  bg.g = Py::Float(args[1]);
  bg.b = Py::Float(args[2]);
  bg.a = Py::Float(args[3]);
  return Py::Object();
}

char Image::apply_scaling__doc__[] = 
"apply_scaling(sx, sy)\n"
"\n"
"Apply the scale factors sx, sy to the transform matrix"
;

Py::Object
Image::apply_scaling(const Py::Tuple& args) {
  _VERBOSE("Image::apply_scaling");
  
  args.verify_length(2);
  double sx = Py::Float(args[0]);
  double sy = Py::Float(args[1]);
  
  //printf("applying scaling %1.2f, %1.2f\n", sx, sy);
  agg::affine_matrix M = agg::scaling_matrix(sx, sy);	      
  srcMatrix *= M;
  imageMatrix *= M;
  
  return Py::Object();
  
  
}

char Image::apply_translation__doc__[] = 
"apply_translation(tx, ty)\n"
"\n"
"Apply the translation tx, ty to the transform matrix"
;

Py::Object
Image::apply_translation(const Py::Tuple& args) {
  _VERBOSE("Image::apply_translation");
  
  args.verify_length(2);
  double tx = Py::Float(args[0]);
  double ty = Py::Float(args[1]);
  
  //printf("applying translation %1.2f, %1.2f\n", tx, ty);
  agg::affine_matrix M = agg::translation_matrix(tx, ty);	      
  srcMatrix *= M;
  imageMatrix *= M;
  
  return Py::Object();
  
  
}


char Image::as_str__doc__[] = 
"numrows, numcols, s = as_str(flipud)"
"\n"
"Call this function after resize to get the data as string\n"
"The string is a numrows by numcols x 4 (RGBA) unsigned char buffer\n"
"if flipud==1, flip the rows upside down\n"
;

Py::Object
Image::as_str(const Py::Tuple& args) {
  _VERBOSE("Image::as_str");
  
  args.verify_length(1);
  int flipud = Py::Int(args[0]);
  if (!flipud) {
    return Py::asObject(Py_BuildValue("lls#", rowsOut, colsOut, 
				      bufferOut, colsOut*rowsOut*4));
  }
  
  const size_t NUMBYTES(rowsOut * colsOut * BPP);
  const size_t BPR = colsOut * BPP; // bytes per row
  
  agg::int8u* buffer = new agg::int8u[NUMBYTES];    
  if (buffer ==NULL) //todo: also handle allocation throw
    throw Py::MemoryError("Image::as_str could not allocate memory");
  
  size_t ind=0;
  for (long rowNum=rowsOut-1; rowNum>=0; rowNum--) { //not unsigned!
    size_t start = rowNum*BPR;
    for (size_t j=0; j<BPR; j++) {
      buffer[ind++] = *(bufferOut + start + j);
    }
  }
  PyObject* o = Py_BuildValue("lls#", rowsOut, colsOut, 
			      buffer, NUMBYTES);
  delete [] buffer;
  return Py::asObject(o);
  
}


char Image::reset_matrix__doc__[] = 
"reset_matrix()"
"\n"
"Reset the transformation matrix"
;

Py::Object
Image::reset_matrix(const Py::Tuple& args) {
  _VERBOSE("Image::reset_matrix");
  
  args.verify_length(0);
  srcMatrix.reset();
  imageMatrix.reset();
  
  return Py::Object();
  
  
}

char Image::resize__doc__[] = 
"resize(width, height)\n"
"\n"
"Resize the image to width, height using interpolation"
;

Py::Object
Image::resize(const Py::Tuple& args) {
  _VERBOSE("Image::resize");
  
  args.verify_length(2);
  
  if (bufferIn ==NULL) 
    throw Py::RuntimeError("You must first load the image"); 
  
  int numcols = Py::Int(args[0]);
  int numrows = Py::Int(args[1]);
  
  colsOut = numcols;
  rowsOut = numrows;
  
  
  size_t NUMBYTES(numrows * numcols * BPP);
  agg::int8u *buffer = new agg::int8u[NUMBYTES];  
  if (buffer ==NULL) //todo: also handle allocation throw
    throw Py::MemoryError("Image::resize could not allocate memory");
  
  rbufOut = new agg::rendering_buffer;
  rbufOut->attach(buffer, numcols, numrows, numcols * BPP);
  
  // init the output rendering/rasterizing stuff
  pixfmt pixf(*rbufOut);
  renderer_base rb(pixf);
  rb.clear(bg);
  agg::rasterizer_scanline_aa<> ras;
  agg::scanline_u8 sl;
  
  
  //srcMatrix *= resizingMatrix;
  //imageMatrix *= resizingMatrix;
  imageMatrix.invert();
  interpolator_type interpolator(imageMatrix); 
  
  // the image path
  agg::path_storage path;
  path.move_to(0, 0);
  path.line_to(colsIn, 0);
  path.line_to(colsIn, rowsIn);
  path.line_to(0, rowsIn);
  path.close_polygon();
  agg::conv_transform<agg::path_storage> imageBox(path, srcMatrix);
  ras.add_path(imageBox);
  
  
  
  agg::image_filter_base* filter = 0;  
  agg::span_allocator<agg::rgba8> sa;	
  switch(interpolation)
    {
    case NEAREST:
      {
	typedef agg::span_image_filter_rgba32_nn<agg::order_rgba32,
	  interpolator_type> span_gen_type;
	typedef agg::renderer_scanline_u<renderer_base, span_gen_type> renderer_type;
	
	span_gen_type sg(sa, *rbufIn, agg::rgba(1,1,1,0), interpolator);
	renderer_type ri(rb, sg);
	ras.add_path(imageBox);
	ras.render(sl, ri);
      }
      break;
      
    case BILINEAR:
      {
	typedef agg::span_image_filter_rgba32_bilinear<agg::order_rgba32,
	  interpolator_type> span_gen_type;
	typedef agg::renderer_scanline_u<renderer_base, span_gen_type> renderer_type;
	
	span_gen_type sg(sa, *rbufIn, agg::rgba(1,1,1,0), interpolator);
	renderer_type ri(rb, sg);
	ras.add_path(imageBox);
	ras.render(sl, ri);
      }
      break;
      
    case BICUBIC:     filter = new agg::image_filter<agg::image_filter_bicubic>;
    case SPLINE16:    filter = new agg::image_filter<agg::image_filter_spline16>;
    case SPLINE36:    filter = new agg::image_filter<agg::image_filter_spline36>;   
    case SINC64:      filter = new agg::image_filter<agg::image_filter_sinc64>;     
    case SINC144:     filter = new agg::image_filter<agg::image_filter_sinc144>;    
    case SINC256:     filter = new agg::image_filter<agg::image_filter_sinc256>;    
    case BLACKMAN64:  filter = new agg::image_filter<agg::image_filter_blackman64>; 
    case BLACKMAN100: filter = new agg::image_filter<agg::image_filter_blackman100>;
    case BLACKMAN256: filter = new agg::image_filter<agg::image_filter_blackman256>;
      
      typedef agg::span_image_filter_rgba32<agg::order_rgba32,
	interpolator_type> span_gen_type;
      typedef agg::renderer_scanline_u<renderer_base, span_gen_type> renderer_type;
      
      span_gen_type sg(sa, *rbufIn, agg::rgba(1,1,1,0), interpolator, *filter);
      renderer_type ri(rb, sg);
      ras.render(sl, ri);
      
    }
  
  /*
    std::ofstream of2( "image_out.raw", std::ios::binary|std::ios::out);
    for (size_t i=0; i<NUMBYTES; ++i)
    of2.write((char*)&buffer[i], sizeof(char));
  */
  
  
  bufferOut = buffer;
  
  
  return Py::Object();
  
}





char Image::get_aspect__doc__[] = 
"get_aspect()\n"
"\n"
"Get the aspect constraint constants"
;

Py::Object
Image::get_aspect(const Py::Tuple& args) {
  _VERBOSE("Image::get_aspect");
  
  args.verify_length(0);
  return Py::Int((int)aspect);   
}

char Image::get_size__doc__[] = 
"numrows, numcols = get_size()\n"
"\n"
"Get the number or rows and columns of the input image"
;

Py::Object
Image::get_size(const Py::Tuple& args) {
  _VERBOSE("Image::get_size");
  
  args.verify_length(0);
  
  Py::Tuple ret(2);
  ret[0] = Py::Int((long)rowsIn);
  ret[1] = Py::Int((long)colsIn);
  return ret;
  
}


char Image::get_interpolation__doc__[] = 
"get_interpolation()\n"
"\n"
"Get the interpolation scheme to one of the module constants, "
"one of image.NEAREST, image.BILINEAR, etc..."
;

Py::Object
Image::get_interpolation(const Py::Tuple& args) {
  _VERBOSE("Image::get_interpolation");
  
  args.verify_length(0);
  return Py::Int((int)interpolation);
}


char Image::set_interpolation__doc__[] = 
"set_interpolation(scheme)\n"
"\n"
"Set the interpolation scheme to one of the module constants, "
"eg, image.NEAREST, image.BILINEAR, etc..."
;

Py::Object
Image::set_interpolation(const Py::Tuple& args) {
  _VERBOSE("Image::set_interpolation");
  
  args.verify_length(1);
  
  size_t method = Py::Int(args[0]);
  interpolation = (unsigned)method;  
  return Py::Object();
  
}



// this code is heavily adapted from the paint license, which is in
// the file paint.license (BSD compatible) included in this
// distribution.  TODO, add license file to MANIFEST.in and CVS
char Image::write_png__doc__[] = 
"write_png(fname)\n"
"\n"
"Write the image to filename fname as png";
Py::Object 
Image::write_png(const Py::Tuple& args)
{
  //small memory leak in this function - JDH 2004-06-08
  _VERBOSE("Image::write_png");
  
  args.verify_length(1);
  
  std::string fileName = Py::String(args[0]);
  const char *file_name = fileName.c_str();
  FILE *fp;
  png_structp png_ptr;
  png_infop info_ptr;
  struct        png_color_8_struct sig_bit;
  png_uint_32 row;
  
  //todo: allocate on heap
  png_bytep row_pointers[rowsOut];
  for (row = 0; row < rowsOut; ++row) {
    row_pointers[row] = bufferOut + row * colsOut * 4;
  }
  
  fp = fopen(file_name, "wb");
  if (fp == NULL) 
    throw Py::RuntimeError("could not open file");
  
  
  png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (png_ptr == NULL) {
    fclose(fp);
    throw Py::RuntimeError("could not create write struct");
  }
  
  info_ptr = png_create_info_struct(png_ptr);
  if (info_ptr == NULL) {
    fclose(fp);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    throw Py::RuntimeError("could not create info struct");
  }
  
  if (setjmp(png_ptr->jmpbuf)) {
    fclose(fp);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    throw Py::RuntimeError("error building image");
  }
  
  png_init_io(png_ptr, fp);
  png_set_IHDR(png_ptr, info_ptr,
	       colsOut, rowsOut, 8,
	       PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE,
	       PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
  
  // this a a color image!
  sig_bit.gray = 0;
  sig_bit.red = 8;
  sig_bit.green = 8;
  sig_bit.blue = 8;
  /* if the image has an alpha channel then */
  sig_bit.alpha = 8;
  png_set_sBIT(png_ptr, info_ptr, &sig_bit);
  
  png_write_info(png_ptr, info_ptr);
  png_write_image(png_ptr, row_pointers);
  png_write_end(png_ptr, info_ptr);
  png_destroy_write_struct(&png_ptr, &info_ptr);
  fclose(fp);
  
  return Py::Object();
}



char Image::set_aspect__doc__[] = 
"set_aspect(scheme)\n"
"\n"
"Set the aspect ration to one of the image module constant."
"eg, one of image.ASPECT_PRESERVE, image.ASPECT_FREE"
;
Py::Object
Image::set_aspect(const Py::Tuple& args) {
  _VERBOSE("Image::set_aspect");
  
  args.verify_length(1);
  size_t method = Py::Int(args[0]);
  aspect = (unsigned)method;  
  return Py::Object();
  
}

void 
Image::init_type() {
  _VERBOSE("Image::init_type");
  
  behaviors().name("Image");
  behaviors().doc("Image");
  behaviors().supportGetattr();
  behaviors().supportSetattr();
  
  add_varargs_method( "apply_rotation", &Image::apply_rotation, Image::apply_rotation__doc__);
  add_varargs_method( "apply_scaling",	&Image::apply_scaling, Image::apply_scaling__doc__);
  add_varargs_method( "apply_translation", &Image::apply_translation, Image::apply_translation__doc__);
  add_varargs_method( "as_str", &Image::as_str, Image::as_str__doc__);
  add_varargs_method( "get_aspect", &Image::get_aspect, Image::get_aspect__doc__);
  add_varargs_method( "get_interpolation", &Image::get_interpolation, Image::get_interpolation__doc__);
  add_varargs_method( "get_size", &Image::get_size, Image::get_size__doc__);
  add_varargs_method( "reset_matrix", &Image::reset_matrix, Image::reset_matrix__doc__);
  add_varargs_method( "resize", &Image::resize, Image::resize__doc__);
  add_varargs_method( "set_interpolation", &Image::set_interpolation, Image::set_interpolation__doc__);
  add_varargs_method( "set_aspect", &Image::set_aspect, Image::set_aspect__doc__);
  add_varargs_method( "write_png", &Image::write_png, Image::write_png__doc__);
  add_varargs_method( "set_bg", &Image::set_bg, Image::set_bg__doc__);
  
  
}




char _image_module_from_images__doc__[] = 
"from_images(numrows, numcols, seq)\n"
"\n"
"return an image instance with numrows, numcols from a seq of image\n"
"instances using alpha blending.  seq is a list of (Image, ox, oy)"
;
Py::Object
_image_module::from_images(const Py::Tuple& args) {
  _VERBOSE("_image_module::from_images");
  
  args.verify_length(3);
  
  size_t numrows = Py::Int(args[0]);
  size_t numcols = Py::Int(args[1]);
  
  Py::SeqBase<Py::Object> tups = args[2];
  size_t N = tups.length();
  
  if (N==0)
    throw Py::RuntimeError("Empty list of images");
  
  Py::Tuple tup;
  
  size_t ox(0), oy(0), thisx(0), thisy(0);
  
  //copy image 0 output buffer into return images output buffer
  Image* imo = new Image;
  imo->rowsOut  = numrows;
  imo->colsOut  = numcols;
  
  size_t NUMBYTES(numrows * numcols * imo->BPP);    
  imo->bufferOut = new agg::int8u[NUMBYTES];  
  if (imo->bufferOut==NULL) //todo: also handle allocation throw
    throw Py::MemoryError("_image_module::from_images could not allocate memory");
  
  imo->rbufOut = new agg::rendering_buffer;
  imo->rbufOut->attach(imo->bufferOut, imo->colsOut, imo->rowsOut, imo->colsOut * imo->BPP);
  
  pixfmt pixf(*imo->rbufOut);
  renderer_base rb(pixf);
  
  
  for (size_t imnum=0; imnum< N; imnum++) {
    tup = Py::Tuple(tups[imnum]);
    Image* thisim = static_cast<Image*>(tup[0].ptr());    
    if (imnum==0) 
      rb.clear(thisim->bg);
    ox = Py::Int(tup[1]);
    oy = Py::Int(tup[2]);
    
    size_t ind=0;
    for (size_t j=0; j<thisim->rowsOut; j++) {
      for (size_t i=0; i<thisim->colsOut; i++) {
	thisx = i+ox;  
	thisy = j+oy; 
	if (thisx<0 || thisx>=numcols || thisy<0 || thisy>=numrows) {
	  ind +=4;
	  continue;
	}
	
	pixfmt::color_type p;
	p.r = *(thisim->bufferOut+ind++);
	p.g = *(thisim->bufferOut+ind++);
	p.b = *(thisim->bufferOut+ind++);
	p.a = *(thisim->bufferOut+ind++);
	pixf.blend_pixel(thisx, thisy, p, 255);
      }
    }
  }
  
  return Py::asObject(imo);
  
  
  
}



char _image_module_readpng__doc__[] = 
"readpng(fname)\n"
"\n"
"Load an image from png file into a numerix array of MxNx4 uint8";
Py::Object
_image_module::readpng(const Py::Tuple& args) {
  
  args.verify_length(1);
  std::string fname = Py::String(args[0]);
  
  png_byte header[8];	// 8 is the maximum size that can be checked

  FILE *fp = fopen(fname.c_str(), "rb");
  if (!fp)
    throw Py::RuntimeError("_image_module::readpng could not open PNG file for reading");

  fread(header, 1, 8, fp);
  if (png_sig_cmp(header, 0, 8))
    throw Py::RuntimeError("_image_module::readpng: file not recognized as a PNG file");
  
  
  /* initialize stuff */
  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  
  if (!png_ptr)
    throw Py::RuntimeError("_image_module::readpng:  png_create_read_struct failed");
  
  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
    throw Py::RuntimeError("_image_module::readpng:  png_create_info_struct failed");
  
  if (setjmp(png_jmpbuf(png_ptr)))
    throw Py::RuntimeError("_image_module::readpng:  error during init_io");
  
  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);
  
  png_read_info(png_ptr, info_ptr);
  
  png_uint_32 width = info_ptr->width;
  png_uint_32 height = info_ptr->height;

  // convert misc color types to rgb for simplicity
  if (info_ptr->color_type == PNG_COLOR_TYPE_GRAY ||
      info_ptr->color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
    png_set_gray_to_rgb(png_ptr);  
  else if (info_ptr->color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_palette_to_rgb(png_ptr);


  int bit_depth = info_ptr->bit_depth;
  if (bit_depth == 16)  png_set_strip_16(png_ptr);


  png_set_interlace_handling(png_ptr);
  png_read_update_info(png_ptr, info_ptr);
  
  bool rgba = info_ptr->color_type == PNG_COLOR_TYPE_RGBA;
  if ( (info_ptr->color_type != PNG_COLOR_TYPE_RGB) || rgba) {
    std::cerr << "Found color type " << (int)info_ptr->color_type << std::endl;
    throw Py::RuntimeError("_image_module::readpng: cannot handle color_type");
  }
  
  /* read file */
  if (setjmp(png_jmpbuf(png_ptr)))
    throw Py::RuntimeError("_image_module::readpng: error during read_image");
  
   png_bytep row_pointers[height];

   for (png_uint_32 row = 0; row < height; row++)
     row_pointers[row] = new png_byte[png_get_rowbytes(png_ptr,info_ptr)];
  
  png_read_image(png_ptr, row_pointers);
  

  
  int dimensions[3];
  dimensions[0] = height;  //numrows
  dimensions[1] = width;   //numcols
  dimensions[2] = 4;

  PyArrayObject *A = (PyArrayObject *) PyArray_FromDims(3, dimensions, PyArray_FLOAT);

  
  for (png_uint_32 y = 0; y < height; y++) {
    png_byte* row = row_pointers[y];
    for (png_uint_32 x = 0; x < width; x++) {
      
      png_byte* ptr = (rgba) ? &(row[x*4]) : &(row[x*3]);
      size_t offset = y*A->strides[0] + x*A->strides[1];
      //if ((y<10)&&(x==10)) std::cout << "r = " << ptr[0] << " " << ptr[0]/255.0 << std::endl;
      *(float*)(A->data + offset + 0*A->strides[2]) = ptr[0]/255.0;
      *(float*)(A->data + offset + 1*A->strides[2]) = ptr[1]/255.0;
      *(float*)(A->data + offset + 2*A->strides[2]) = ptr[2]/255.0;
      *(float*)(A->data + offset + 3*A->strides[2]) = rgba ? ptr[3]/255.0 : 1.0;
    }
  }
  
  //free the png memory
  png_read_end(png_ptr, info_ptr);
  png_destroy_read_struct(&png_ptr, &info_ptr, png_infopp_NULL);
  fclose(fp);
  for (png_uint_32 row = 0; row < height; row++)
    delete [] row_pointers[row];
  return Py::asObject((PyObject*)A);
}


char _image_module_fromarray__doc__[] = 
"fromarray(A, isoutput)\n"
"\n"
"Load the image from a Numeric or numarray array\n"
"By default this function fills the input buffer, which can subsequently\n"
"be resampled using resize.  If isoutput=1, fill the output buffer.\n"
"This is used to support raw pixel images w/o resampling"
;
Py::Object
_image_module::fromarray(const Py::Tuple& args) {
  _VERBOSE("_image_module::fromarray");
  
  args.verify_length(2);
  
  Py::Object x = args[0];
  int isoutput = Py::Int(args[1]);
  PyArrayObject *A = (PyArrayObject *) PyArray_ContiguousFromObject(x.ptr(), PyArray_DOUBLE, 2, 3); 
  
  if (A==NULL) 
    throw Py::ValueError("Array must be rank 2 or 3 of doubles"); 
  
  
  Image* imo = new Image;
  
  imo->rowsIn  = A->dimensions[0];
  imo->colsIn  = A->dimensions[1];
  
  
  size_t NUMBYTES(imo->colsIn * imo->rowsIn * imo->BPP);
  agg::int8u *buffer = new agg::int8u[NUMBYTES];  
  if (buffer==NULL) //todo: also handle allocation throw
    throw Py::MemoryError("_image_module::fromarray could not allocate memory");
  
  imo->bufferIn = buffer;
  imo->rbufIn = new agg::rendering_buffer;
  imo->rbufIn->attach(buffer, imo->colsIn, imo->rowsIn, imo->colsIn*imo->BPP);
  
  
  if   (A->nd == 2) { //assume luminance for now; 
    
    agg::int8u gray;
    int start = 0;    
    for (size_t rownum=0; rownum<imo->rowsIn; rownum++) 
      for (size_t colnum=0; colnum<imo->colsIn; colnum++) {
	
	double val = *(double *)(A->data + rownum*A->strides[0] + colnum*A->strides[1]);
	
	gray = int(255 * val);
	*(buffer+start++) = gray;       // red
	*(buffer+start++) = gray;       // green
	*(buffer+start++) = gray;       // blue
	*(buffer+start++)   = 255;        // alpha
      }
    
  }
  else if   (A->nd == 3) { // assume RGB
    
    if (A->dimensions[2] != 3 && A->dimensions[2] != 4 ) {
      Py_XDECREF(A);  
      throw Py::ValueError("3rd dimension must be length 3 (RGB) or 4 (RGBA)"); 
      
    }
    
    int rgba = A->dimensions[2]==4;
    
    int start = 0;    
    double r,g,b,alpha;
    int offset =0;
    
    for (size_t rownum=0; rownum<imo->rowsIn; rownum++) 
      for (size_t colnum=0; colnum<imo->colsIn; colnum++) {
	offset = rownum*A->strides[0] + colnum*A->strides[1];
	r = *(double *)(A->data + offset);
	g = *(double *)(A->data + offset + A->strides[2] );
	b = *(double *)(A->data + offset + 2*A->strides[2] );
 	
	if (rgba) 
	  alpha = *(double *)(A->data + offset + 3*A->strides[2] );
	else
	  alpha = 1.0;
	
	*(buffer+start++) = int(255*r);         // red
	*(buffer+start++) = int(255*g);         // green
	*(buffer+start++) = int(255*b);         // blue
	*(buffer+start++) = int(255*alpha);     // alpha
	
      }
  } 
  else   { // error
    Py_XDECREF(A);  
    throw Py::ValueError("Illegal array rank; must be rank; must 2 or 3"); 
  }
  Py_XDECREF(A);  
  
  if (isoutput) {
    // make the output buffer point to the input buffer
    
    imo->rowsOut  = imo->rowsIn;
    imo->colsOut  = imo->colsIn;
    
    imo->bufferOut = new agg::int8u[NUMBYTES];  
    if (buffer == imo->bufferOut) //todo: also handle allocation throw
      throw Py::MemoryError("_image_module::fromarray could not allocate memory");
    
    imo->rbufOut = new agg::rendering_buffer;
    imo->rbufOut->attach(imo->bufferOut, imo->colsOut, imo->rowsOut, imo->colsOut * imo->BPP);
    
    for (size_t i=0; i<NUMBYTES; i++)
      *(imo->bufferOut +i) = *(imo->bufferIn +i);
  }
  return Py::asObject( imo );
}




#if defined(_MSC_VER)
DL_EXPORT(void)
#elif defined(__cplusplus)
  extern "C" void
#else
void
#endif
init_image(void) {
  _VERBOSE("init_image");
  
  static _image_module* _image = new _image_module;
  
  import_array();  
  Py::Dict d = _image->moduleDictionary();
  d["BICUBIC"] = Py::Int(Image::BICUBIC);
  d["BILINEAR"] = Py::Int(Image::BILINEAR);
  d["BLACKMAN100"] = Py::Int(Image::BLACKMAN100);
  d["BLACKMAN256"] = Py::Int(Image::BLACKMAN256);
  d["BLACKMAN64"] = Py::Int(Image::BLACKMAN64);
  d["NEAREST"] = Py::Int(Image::NEAREST);
  d["SINC144"] = Py::Int(Image::SINC144);
  d["SINC256"] = Py::Int(Image::SINC256);
  d["SINC64"] = Py::Int(Image::SINC64);
  d["SPLINE16"] = Py::Int(Image::SPLINE16);
  d["SPLINE36"] = Py::Int(Image::SPLINE36);
  
  d["ASPECT_FREE"] = Py::Int(Image::ASPECT_FREE);
  d["ASPECT_PRESERVE"] = Py::Int(Image::ASPECT_PRESERVE);
  
  
}

