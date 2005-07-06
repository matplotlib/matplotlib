/* _backend_agg.h - A rewrite of _backend_agg using PyCXX to handle
   ref counting, etc..
*/

#ifndef __BACKEND_AGG_H
#define __BACKEND_AGG_H

#include "CXX/Extensions.hxx"
#include "agg_buffer.h"  // a swig wrapper

#include "agg_arrowhead.h"
#include "agg_basics.h"
#include "agg_color_rgba.h"
#include "agg_conv_concat.h"
#include "agg_conv_contour.h"
#include "agg_conv_curve.h"
#include "agg_conv_dash.h"
#include "agg_conv_marker.h"
#include "agg_conv_marker_adaptor.h"
#include "agg_math_stroke.h"
#include "agg_conv_stroke.h"
#include "agg_ellipse.h"
#include "agg_embedded_raster_fonts.h"
#include "agg_path_storage.h"
#include "agg_pixfmt_rgb.h"
#include "agg_pixfmt_rgba.h"
#include "agg_rasterizer_outline.h"
#include "agg_rasterizer_scanline_aa.h"
#include "agg_renderer_outline_aa.h"
#include "agg_renderer_raster_text.h"
#include "agg_renderer_scanline.h"
#include "agg_rendering_buffer.h"
#include "agg_scanline_bin.h"
#include "agg_scanline_p.h"
#include "agg_vcgen_markers_term.h"

typedef agg::pixfmt_rgba32 pixfmt;
typedef agg::renderer_base<pixfmt> renderer_base;
typedef agg::renderer_scanline_aa_solid<renderer_base> renderer_aa;
typedef agg::renderer_scanline_bin_solid<renderer_base> renderer_bin;
typedef agg::rasterizer_scanline_aa<> rasterizer;
typedef agg::scanline_p8 scanline_p8;
typedef agg::scanline_bin scanline_bin;

// a helper class to pass agg::buffer objects around.  agg::buffer is
// a class in the swig wrapper
class BufferRegion : public Py::PythonExtension<BufferRegion> {
public:
  BufferRegion( agg::buffer& aggbuf, const agg::rect &r) : aggbuf(aggbuf), rect(r) {}
  agg::buffer aggbuf;
  agg::rect rect;
  static void init_type(void) {
    behaviors().name("BufferRegion");
    behaviors().doc("A wrapper to pass agg buffer objects to and from the python level");
  }

  virtual ~BufferRegion() {};
};

class GCAgg {
public:
  GCAgg(const Py::Object& gc, double dpi);

  ~GCAgg() {
    delete [] dasha;
    delete [] cliprect;
  }

  double dpi;
  bool isaa;
  
  agg::line_cap_e cap; 
  agg::line_join_e join;

  
  double linewidth;
  double alpha;
  agg::rgba color;

  double *cliprect;
  
  //dashes
  size_t Ndash;
  double dashOffset;
  double *dasha;

protected:
  agg::rgba get_color(const Py::Object& gc);
  double points_to_pixels( const Py::Object& points);
  void _set_linecap(const Py::Object& gc) ;
  void _set_joinstyle(const Py::Object& gc) ;
  void _set_dashes(const Py::Object& gc) ;
  void _set_clip_rectangle( const Py::Object& gc);
  void _set_antialiased( const Py::Object& gc);
};

// th renderer
class RendererAgg: public Py::PythonExtension<RendererAgg> {
  typedef std::pair<bool, agg::rgba> facepair_t;
public:
  RendererAgg(unsigned int width, unsigned int height, double dpi, int debug);
  static void init_type(void);

  unsigned int get_width() { return width;}
  unsigned int get_height() { return height;}
  // the drawing methods
  Py::Object draw_rectangle(const Py::Tuple & args);
  Py::Object draw_ellipse(const Py::Tuple & args);
  Py::Object draw_polygon(const Py::Tuple & args);
  Py::Object draw_line_collection(const Py::Tuple& args);
  Py::Object draw_poly_collection(const Py::Tuple& args);
  Py::Object draw_regpoly_collection(const Py::Tuple& args);
  Py::Object draw_lines(const Py::Tuple & args);
  Py::Object draw_path(const Py::Tuple & args);
  //Py::Object _draw_markers_nocache(const Py::Tuple & args);
  //Py::Object _draw_markers_cache(const Py::Tuple & args);
  Py::Object draw_markers(const Py::Tuple & args);
  Py::Object draw_text(const Py::Tuple & args);
  Py::Object draw_image(const Py::Tuple & args);

  Py::Object write_rgba(const Py::Tuple & args);
  Py::Object write_png(const Py::Tuple & args);
  Py::Object tostring_rgb(const Py::Tuple & args);
  Py::Object tostring_argb(const Py::Tuple & args);
  Py::Object tostring_bgra(const Py::Tuple & args);
  Py::Object buffer_rgba(const Py::Tuple & args);
  Py::Object clear(const Py::Tuple & args);
  Py::Object cache(const Py::Tuple & args);
  Py::Object blit(const Py::Tuple & args);

  Py::Object copy_from_bbox(const Py::Tuple & args);
  Py::Object restore_region(const Py::Tuple & args);

  
  


  virtual ~RendererAgg(); 

  static const size_t PIXELS_PER_INCH;
  unsigned int width, height;
  double dpi;
  size_t NUMBYTES;  //the number of bytes in buffer
  size_t CACHEBYTES;  //the number of bytes in cache buffer

  agg::int8u *pixBuffer;
  agg::int8u *cacheBuffer;
  agg::rendering_buffer *renderingBuffer;

  scanline_p8* slineP8;
  scanline_bin* slineBin;
  pixfmt *pixFmt;
  renderer_base *rendererBase;
  renderer_aa *rendererAA;
  renderer_bin *rendererBin;
  rasterizer *theRasterizer;


  const int debug;

protected:
  agg::rect bbox_to_rect( const Py::Object& o);
  double points_to_pixels( const Py::Object& points);
  double points_to_pixels_snapto( const Py::Object& points);

  void set_clip_from_bbox(const Py::Object& o);
  agg::rgba rgb_to_color(const Py::SeqBase<Py::Object>& rgb, double alpha);
  facepair_t _get_rgba_face(const Py::Object& rgbFace, double alpha);
  void set_clipbox_rasterizer( double *cliprect);
  template <class VS> void _fill_and_stroke(VS&, const GCAgg&, const facepair_t&, bool curvy=true);  

  template<class PathSource>
  void _render_lines_path(PathSource &ps, const GCAgg& gc);
};


// the extension module
class _backend_agg_module : public Py::ExtensionModule<_backend_agg_module>
{
public:
  _backend_agg_module()
    : Py::ExtensionModule<_backend_agg_module>( "_backend_agg" )
  {

    BufferRegion::init_type();
    RendererAgg::init_type();

    add_keyword_method("RendererAgg", &_backend_agg_module::new_renderer, 
		       "RendererAgg(width, height, dpi)");
    initialize( "The agg rendering backend" );
  }
  
  virtual ~_backend_agg_module() {}
  
private:
  
  Py::Object new_renderer (const Py::Tuple &args, const Py::Dict &kws);


  
};



#endif

