/* A rewrite of _backend_agg using PyCXX to handle ref counting, etc..
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdio>

#include <png.h>
#include "util/agg_color_conv_rgb8.h"

#include "ft2font.h"
#include "_image.h"
#include "_backend_agg.h"


/* ------------ RendererAgg methods ------------- */

const size_t 
RendererAgg::PIXELS_PER_INCH(96);

RendererAgg::RendererAgg(unsigned int width, unsigned int height, double dpi, 
			 int debug) :
  width(width), 
  height(height), 
  dpi(dpi), 
  NUMBYTES(width*height*4),
  debug(debug)
{
  if (debug)
    std::cout << "RendererAgg::RendererAgg" << std::endl;
  unsigned stride(width*4);    
  
  
  pixBuffer = new agg::int8u[NUMBYTES];  
  
  renderingBuffer = new agg::rendering_buffer;
  renderingBuffer->attach(pixBuffer, width, height, stride);
  slineP8 = new scanline_p8;
  slineBin = new scanline_bin;
  
  
  pixFmt = new pixfmt(*renderingBuffer);
  rendererBase = new renderer_base(*pixFmt);
  rendererBase->clear(agg::rgba(1, 1, 1));
  
  theRenderer = new renderer(*rendererBase);
  rendererBin = new renderer_bin(*rendererBase);
  theRasterizer = new rasterizer(); 
  
};

Py::Object
RendererAgg::draw_rectangle(const Py::Tuple & args) {
  if (debug)
    std::cout << "RendererAgg::draw_rectangle" << std::endl;
  args.verify_length(6);
  
  Py::Object gcEdge( args[0] );
  Py::Object rgbFaceMaybeNone( args[1] );
  
  double l = Py::Float( args[2] ); 
  double b = Py::Float( args[3] ); 
  double w = Py::Float( args[4] ); 
  double h = Py::Float( args[5] ); 
  
  set_clip_rectangle(gcEdge);
  
  double lw = points_to_pixels ( gcEdge.getAttr("_linewidth") ) ;
  
  agg::path_storage path;
  
  b = height - (b+h);
  path.move_to(l, b+h);
  path.line_to(l+w, b+h);
  path.line_to(l+w, b);
  path.line_to(l, b);
  path.close_polygon();
  
  agg::rgba edgecolor = get_color(gcEdge);
  
  
  if (rgbFaceMaybeNone.ptr() != Py_None) {
    //fill the face
    Py::SeqBase<Py::Object> rgbFace = rgbFaceMaybeNone;
    agg::rgba facecolor = rgb_to_color(rgbFace, edgecolor.a);
    
    theRenderer->color(facecolor);
    theRasterizer->add_path(path);    
    theRasterizer->render(*slineP8, *theRenderer);  
    
  }
  
  //now fill the edge
  agg::conv_stroke<agg::path_storage> stroke(path);
  stroke.width(lw);
  theRenderer->color(edgecolor);
  //self->theRasterizer->gamma(agg::gamma_power(gamma));
  theRasterizer->add_path(stroke);
  theRasterizer->render(*slineP8, *theRenderer);  
  
  
  return Py::Object();
  
}

Py::Object
RendererAgg::draw_ellipse(const Py::Tuple& args) {
  if (debug)
    std::cout << "RendererAgg::draw_ellipse" << std::endl;
  
  args.verify_length(6);  
  Py::Object gcEdge = args[0];
  Py::Object rgbFaceMaybeNone = args[1];
  
  double x = Py::Float( args[2] ); 
  double y = Py::Float( args[3] ); 
  double w = Py::Float( args[4] ); 
  double h = Py::Float( args[5] ); 
  
  set_clip_rectangle(gcEdge);
  
  //last arg is num steps
  agg::ellipse path(x, height-y, w, h, 100); 
  agg::rgba edgecolor = get_color(gcEdge);  
  
  
  if (rgbFaceMaybeNone.ptr() != Py_None) {
    Py::SeqBase<Py::Object> rgbFace = rgbFaceMaybeNone;
    agg::rgba facecolor = rgb_to_color(rgbFace, edgecolor.a);
    theRenderer->color(facecolor);
    theRasterizer->add_path(path);    
    theRasterizer->render(*slineP8, *theRenderer);  
    
  }
  
  
  //now fill the edge
  
  double lw = points_to_pixels ( gcEdge.getAttr("_linewidth") ) ;
  
  agg::conv_stroke<agg::ellipse> stroke(path);
  stroke.width(lw);
  theRenderer->color(edgecolor);
  //self->theRasterizer->gamma(agg::gamma_power(gamma));
  theRasterizer->add_path(stroke);
  theRasterizer->render(*slineP8, *theRenderer);  
  
  return Py::Object();
  
}

Py::Object
RendererAgg::draw_polygon(const Py::Tuple& args) {
  if (debug)
    std::cout << "RendererAgg::draw_polygon" << std::endl;
  
  args.verify_length(3);  
  
  Py::Object gcEdge( args[0] );
  Py::Object rgbFaceMaybeNone( args[1] );
  Py::SeqBase<Py::Object> points( args[2] );
  
  
  set_clip_rectangle(gcEdge);
  
  double lw = points_to_pixels ( gcEdge.getAttr("_linewidth") ) ;
  
  size_t Npoints = points.length();
  if (Npoints<=0)
    return Py::Object();
  
  
  Py::SeqBase<Py::Object> xy = Py::SeqBase<Py::Object>( points[0] );
  double x = Py::Float( xy[0] );
  double y = Py::Float( xy[1] );
  
  y = height - y;
  
  agg::path_storage path;
  path.move_to(x, y);
  
  for (size_t i=1; i<Npoints; ++i) {
    
    xy = points[i];
    x = Py::Float( xy[0] );
    y = Py::Float( xy[1] );
    
    y = height - y;
    path.line_to(x, y);
    
  }
  path.close_polygon();
  
  agg::rgba edgecolor = get_color(gcEdge);
  
  if (rgbFaceMaybeNone.ptr() != Py_None) {
    //fill the face
    Py::SeqBase<Py::Object> rgbFace = rgbFaceMaybeNone;
    agg::rgba facecolor = rgb_to_color(rgbFace, edgecolor.a);
    theRenderer->color(facecolor);
    theRasterizer->add_path(path);    
    theRasterizer->render(*slineP8, *theRenderer);  
  }
  
  //now fill the edge
  agg::conv_stroke<agg::path_storage> stroke(path);
  stroke.width(lw);
  
  theRenderer->color(edgecolor);
  //self->theRasterizer->gamma(agg::gamma_power(gamma));
  theRasterizer->add_path(stroke);
  theRasterizer->render(*slineP8, *theRenderer);  
  return Py::Object();
  
}

Py::Object
RendererAgg::draw_lines(const Py::Tuple& args) {
  
  
  if (debug)
    std::cout << "RendererAgg::draw_lines" << std::endl;
  args.verify_length(3);  
  Py::Object gc = args[0];
  Py::SeqBase<Py::Object> x = args[1];  //todo: use numerix for efficiency
  Py::SeqBase<Py::Object> y = args[2];  //todo: use numerix for efficiency
  
  
  set_clip_rectangle(gc);
  size_t Nx = x.length();
  size_t Ny = y.length();
  
  if (Nx!=Ny) 
    throw Py::ValueError("x and y must be equal length sequences");
  
  
  if (Nx<2) 
    throw Py::ValueError("x and y must have length >= 2");
  
  
  agg::gen_stroke::line_cap_e cap = get_linecap(gc);
  agg::gen_stroke::line_join_e join = get_joinstyle(gc);
  
  
  
  double lw = points_to_pixels ( gc.getAttr("_linewidth") ) ;
  
  agg::rgba color = get_color(gc);
  
  
  // process the dashes
  Py::Tuple dashes = get_dashes(gc);
  
  bool useDashes = dashes[0].ptr() != Py_None;
  double offset = 0;
  Py::SeqBase<Py::Object> dashSeq;
  
  if ( dashes[0].ptr() != Py_None ) { // use dashes
    //TODO: use offset
    offset = points_to_pixels_snapto(dashes[0]);
    dashSeq = dashes[1]; 
  };
  
  
  agg::path_storage path;
  
  int isaa = antialiased(gc);
  double thisX(0), thisY(0);
  int ix(0), iy(0);
  if (Nx==2) { 
    // this is a little hack - len(2) lines are probably grid and
    // ticks so I'm going to snap to pixel
    //printf("snapto %d\n", Nx);
    ix = Py::Float(x[0]);
    iy = Py::Float(y[0]);
    thisX = ix+0.5;
    thisY = height - iy + 0.5;  //flipy
    path.move_to(thisX, thisY);
    for (size_t i=1; i<Nx; ++i) {
      ix = Py::Float(x[i]);
      iy = Py::Float(y[i]);
      thisX = ix+0.5;
      thisY = height - iy + 0.5;  //flipy
      path.line_to(thisX, thisY);
    }
    
  }
  else {
    thisX = Py::Float( x[0] );
    thisY = Py::Float( y[0] );
    thisY = height - thisY; //flipy
    path.move_to(thisX, thisY);
    for (size_t i=1; i<Nx; ++i) {
      thisX = Py::Float( x[i] );
      thisY = Py::Float( y[i] );
      thisY = height - thisY;  //flipy
      path.line_to(thisX, thisY);
    }
  }  
 
  if (! useDashes ) {
    agg::conv_stroke<agg::path_storage> stroke(path);
    stroke.line_cap(cap);
    stroke.line_join(join);
    stroke.width(lw);
    theRasterizer->add_path(stroke);
  }
  else {
    // set the dashes //TODO: scale for DPI
    size_t N = dashSeq.length();
    if (N%2 != 0  ) 
      throw Py::ValueError("dashes must be an even length sequence");     
    
    typedef agg::conv_dash<agg::path_storage> dash_t;
    dash_t dash(path);
    agg::conv_stroke<dash_t> stroke(dash);
    double on, off;
    for (size_t i=0; i<N/2; i+=1) {
      
      on = points_to_pixels_snapto(dashSeq[2*i]);
      off = points_to_pixels_snapto(dashSeq[2*i+1]);
      //std::cout << "agg setting" << on << " " << off << std::endl;
      dash.add_dash(on, off);
    }
    stroke.line_cap(cap);
    stroke.line_join(join);
    stroke.width(lw);
    theRasterizer->add_path(stroke);
    
  }
  
  
  if ( isaa ) {
    theRenderer->color(color);    
    theRasterizer->render(*slineP8, *theRenderer); 
  }
  else {
    rendererBin->color(color);    
    theRasterizer->render(*slineBin, *rendererBin); 
  }
  
  return Py::Object();
  
}

Py::Object
RendererAgg::draw_text(const Py::Tuple& args) {
  if (debug)
    std::cout << "RendererAgg::draw_text" << std::endl;
  
  args.verify_length(4);
  
  
  FT2FontObject *font = (FT2FontObject *)args[0].ptr();
  
  int x = Py::Int( args[1] );
  int y = Py::Int( args[2] );
  Py::SeqBase<Py::Object> rgba = Py::SeqBase<Py::Object>( args[3] );
  
  double r = Py::Float( rgba[0] );
  double g = Py::Float( rgba[1] );
  double b = Py::Float( rgba[2] );
  double a = Py::Float( rgba[3] );
  
  
  pixfmt::color_type p;
  p.r = int(255*r); p.b = int(255*b); p.g = int(255*g); p.a = int(255*a);
  
  for (size_t i=0; i<font->image.width; ++i) {
    for (size_t j=0; j<font->image.height; ++j) {
      if (i+x>=width)  continue;
      if (j+y>=height) continue;
      
      pixFmt->blend_pixel
	(i+x, y+j, p, font->image.buffer[i + j*font->image.width]);
    }
  }
  return Py::Object();
  
}

Py::Object 
RendererAgg::draw_image(const Py::Tuple& args) {
  if (debug)
    std::cout << "RendererAgg::draw_image" << std::endl;
  args.verify_length(3);
  
  int x = Py::Int(args[0]);
  int y = Py::Int(args[1]);
  ImageObject *image = (ImageObject *)args[2].ptr();
  
  
  //todo: handle x and y
  agg::rect r(0, 0, image->rowsOut, image->colsOut);
  
  rendererBase->copy_from(*image->rbufOut, &r, x, y);
  
  return Py::Object();
  
}


Py::Object 
RendererAgg::write_rgba(const Py::Tuple& args) {
  if (debug)
    std::cout << "RendererAgg::write_rgba" << std::endl;
  
  args.verify_length(1);  
  std::string fname = Py::String( args[0]);
  
  std::ofstream of2( fname.c_str(), std::ios::binary|std::ios::out);
  for (size_t i=0; i<NUMBYTES; ++i) {
    of2.write((char*)&(pixBuffer[i]), sizeof(char));
  }
  return Py::Object();
  
}


// this code is heavily adapted from the paint license, which is in
// the file paint.license (BSD compatible) included in this
// distribution.  TODO, add license file to MANIFEST.in and CVS
Py::Object 
RendererAgg::write_png(const Py::Tuple& args)
{
  if (debug)
    std::cout << "RendererAgg::write_png" << std::endl;
  
  args.verify_length(1);
  
  std::string fileName = Py::String(args[0]);
  const char *file_name = fileName.c_str();
  FILE *fp;
  png_structp png_ptr;
  png_infop info_ptr;
  struct        png_color_8_struct sig_bit;
  png_uint_32 row;
  
  png_bytep row_pointers[height];
  for (row = 0; row < height; ++row) {
    row_pointers[row] = pixBuffer + row * width * 4;
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
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    throw Py::RuntimeError("could not create info struct");
  }
  
  if (setjmp(png_ptr->jmpbuf)) {
    fclose(fp);
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    throw Py::RuntimeError("error building image");
  }
  
  png_init_io(png_ptr, fp);
  png_set_IHDR(png_ptr, info_ptr,
	       width, height, 8,
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
  png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
  fclose(fp);
  
  return Py::Object();
}

Py::Object 
RendererAgg::tostring_rgb(const Py::Tuple& args) {
  //"Return the rendered buffer as an RGB string";
  
  if (debug)
    std::cout << "RendererAgg::tostring_rgb" << std::endl;
  
  args.verify_length(0);    
  int row_len = width*3;
  unsigned char* buf_tmp = 
    new unsigned char[row_len * height];
  agg::rendering_buffer renderingBufferTmp;
  renderingBufferTmp.attach(buf_tmp, 
			    width, 
			    height, 
			    row_len);
  
  color_conv(&renderingBufferTmp, renderingBuffer, agg::color_conv_rgba32_to_rgb24());
  
  
  //todo: how to do this with native CXX
  return Py::Object(Py_BuildValue("s#", 
				  buf_tmp, 
				  row_len * height));
  //len = row_len * height
  //std::string s(buf_tmp);
  //return Py::String(buf_tmp, row_len * height);
}

agg::rgba
RendererAgg::get_color(const Py::Object& gc) {
  
  if (debug)
    std::cout << "RendererAgg::get_color" << std::endl;
  
  Py::Tuple rgb = Py::Tuple( gc.getAttr("_rgb") );
  
  double alpha = Py::Float( gc.getAttr("_alpha") );
  
  double r = Py::Float(rgb[0]);
  double g = Py::Float(rgb[1]);
  double b = Py::Float(rgb[2]);
  return agg::rgba(r, g, b, alpha); 
  
}

agg::gen_stroke::line_cap_e
RendererAgg::get_linecap(const Py::Object& gc) {
  if (debug)
    std::cout << "RendererAgg::get_linecap" << std::endl;
  
  std::string capstyle = Py::String( gc.getAttr( "_capstyle" ) );
  
  if (capstyle=="butt") 
    return agg::gen_stroke::butt_cap;
  else if (capstyle=="round") 
    return agg::gen_stroke::round_cap;
  else if(capstyle=="projecting") 
    return agg::gen_stroke::square_cap;
  else 
    throw Py::ValueError("GC _capstyle attribute must be one of butt, round, projecting");
  
}

agg::gen_stroke::line_join_e
RendererAgg::get_joinstyle(const Py::Object& gc) {
  if (debug)
    std::cout << "RendererAgg::get_joinstyle" << std::endl;
  
  std::string joinstyle = Py::String( gc.getAttr("_joinstyle") );
  
  if (joinstyle=="miter") 
    return agg::gen_stroke::miter_join;
  else if (joinstyle=="round") 
    return agg::gen_stroke::round_join;
  else if(joinstyle=="bevel") 
    return agg::gen_stroke::bevel_join;
  else 
    throw Py::ValueError("GC _joinstyle attribute must be one of butt, round, projecting");
  
}

Py::Tuple
RendererAgg::get_dashes(const Py::Object& gc) {
  //return the dashOffset, dashes sequence tuple.  
  if (debug)
    std::cout << "RendererAgg::get_dashes" << std::endl;
  
  Py::Tuple _dashes = gc.getAttr("_dashes");
  
  size_t N = _dashes.length();
  
  if (N!=2) 
    throw Py::ValueError("GC _dashes must be a length 2 tuple");    
  
  return _dashes;
}


agg::rgba
RendererAgg::rgb_to_color(const Py::SeqBase<Py::Object>& rgb, double alpha) {
  if (debug)
    std::cout << "RendererAgg::rgb_to_color" << std::endl;
  
  double r = Py::Float(rgb[0]);
  double g = Py::Float(rgb[1]);
  double b = Py::Float(rgb[2]);
  return agg::rgba(r, g, b, alpha); 
  
}

void
RendererAgg::set_clip_rectangle( const Py::Object& gc) {
  //set the clip rectangle from the gc
  
  if (debug)
    std::cout << "RendererAgg::set_clip_rectangle" << std::endl;
  
  Py::Object o ( gc.getAttr( "_cliprect" ) );
  
  if (o.ptr()==Py_None) {
    // set clipping to false and return success
    theRasterizer->reset_clipping();
    return;
  }
  
  Py::SeqBase<Py::Object> rect( o );
  
  double l = Py::Float(rect[0]) ; 
  double b = Py::Float(rect[1]) ; 
  double w = Py::Float(rect[2]) ; 
  double h = Py::Float(rect[3]) ; 
  
  theRasterizer->clip_box(l, height-(b+h),
			  l+w, height-b);
}




int
RendererAgg::antialiased(const Py::Object& gc) {
  //return 1 if gc is antialiased
  if (debug)
    std::cout << "RendererAgg::antialiased" << std::endl;
  int isaa = Py::Int( gc.getAttr( "_antialiased") );
  return isaa;
}

double
RendererAgg::points_to_pixels_snapto(const Py::Object& points) {
  // convert a value in points to pixels depending on renderer dpi and
  // screen pixels per inch
  // snap return pixels to grid
  if (debug)
    std::cout << "RendererAgg::points_to_pixels_snapto" << std::endl;
  double p = Py::Float( points ) ;
  return (int)(p*PIXELS_PER_INCH/72.0*dpi/72.0)+0.5;
  
  
}

double
RendererAgg::points_to_pixels( const Py::Object& points) {
  if (debug)
    std::cout << "RendererAgg::points_to_pixels" << std::endl;
  double p = Py::Float( points ) ;
  return p * PIXELS_PER_INCH/72.0*dpi/72.0;
}

RendererAgg::~RendererAgg() {
  
  if (debug)
    std::cout << "RendererAgg::~RendererAgg" << std::endl;
  
  
  delete slineP8;
  delete slineBin;
  delete theRasterizer;
  delete rendererBin;
  delete rendererBase;
  delete pixFmt;
  delete renderingBuffer;
  delete [] pixBuffer;
  
}

/* ------------ module methods ------------- */
Py::Object _backend_agg_module::new_renderer (const Py::Tuple &args, 
					      const Py::Dict &kws)
{
  
  if (args.length() != 3 )
    {
      throw Py::RuntimeError("Incorrect # of args to RendererAgg(width, height, dpi).");
    }
  
  int debug;
  if ( kws.hasKey("debug") ) debug = Py::Int( kws["debug"] );
  else debug=0;
  
  int width = Py::Int(args[0]);
  int height = Py::Int(args[1]);
  double dpi = Py::Float(args[2]);
  return Py::asObject(new RendererAgg(width, height, dpi, debug));
}

void RendererAgg::init_type()
{
  behaviors().name("RendererAgg");
  behaviors().doc("The agg backend extension module");
  
  add_varargs_method("draw_rectangle", &RendererAgg::draw_rectangle, 
		     "draw_rectangle(gc, rgbFace, l, b, w, h)\n");
  add_varargs_method("draw_ellipse", &RendererAgg::draw_ellipse, 
		     "draw_ellipse(gc, rgbFace, x, y, w, h)\n");
  add_varargs_method("draw_polygon", &RendererAgg::draw_polygon, 
		     "draw_polygon(gc, rgbFace, points)\n");
  add_varargs_method("draw_lines", &RendererAgg::draw_lines, 
		     "draw_lines(gc, x, y,)\n");
  add_varargs_method("draw_text", &RendererAgg::draw_text, 
		     "draw_text(font, x, y, r, g, b, a)\n");
  add_varargs_method("draw_image", &RendererAgg::draw_image, 
		     "draw_image(x, y, im)");
  add_varargs_method("write_rgba", &RendererAgg::write_rgba, 
		     "write_rgba(fname)");
  add_varargs_method("write_png", &RendererAgg::write_png, 
		     "write_png(fname)");
  add_varargs_method("tostring_rgb", &RendererAgg::tostring_rgb, 
		     "s = tostring_rgb()");
  
}

extern "C"
DL_EXPORT(void)
  init_backend_agg(void)
{
  static _backend_agg_module* _backend_agg = new _backend_agg_module;
  
};





