/* _backend_agg.h - A rewrite of _backend_agg using PyCXX to handle
   ref counting, etc..
*/

#ifndef __BACKEND_AGG_H
#define __BACKEND_AGG_H
#include <utility>
#include "CXX/Extensions.hxx"
#include "agg_buffer.h"  // a swig wrapper

#include "agg_arrowhead.h"
#include "agg_basics.h"
#include "agg_bezier_arc.h"
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
#include "agg_pixfmt_gray.h"
#include "agg_alpha_mask_u8.h"
#include "agg_pixfmt_amask_adaptor.h"
#include "agg_rasterizer_outline.h"
#include "agg_rasterizer_scanline_aa.h"
#include "agg_renderer_outline_aa.h"
#include "agg_renderer_raster_text.h"
#include "agg_renderer_scanline.h"
#include "agg_rendering_buffer.h"
#include "agg_scanline_bin.h"
#include "agg_scanline_u.h"
#include "agg_scanline_p.h"
#include "agg_vcgen_markers_term.h"

// These are copied directly from path.py, and must be kept in sync
#define STOP   0
#define MOVETO 1
#define LINETO 2
#define CURVE3 3
#define CURVE4 4
#define CLOSEPOLY 5

const size_t NUM_VERTICES[] = { 1, 1, 1, 2, 3, 1 };

typedef agg::pixfmt_rgba32 pixfmt;
typedef agg::renderer_base<pixfmt> renderer_base;
typedef agg::renderer_scanline_aa_solid<renderer_base> renderer_aa;
typedef agg::renderer_scanline_bin_solid<renderer_base> renderer_bin;
typedef agg::rasterizer_scanline_aa<> rasterizer;

typedef agg::scanline_p8 scanline_p8;
typedef agg::scanline_bin scanline_bin;
//yypedef agg::scanline_u8_am<agg::alpha_mask_gray8> scanline_alphamask;
typedef agg::amask_no_clip_gray8 alpha_mask_type;


typedef agg::renderer_base<agg::pixfmt_gray8> renderer_base_alpha_mask_type;
typedef agg::renderer_scanline_aa_solid<renderer_base_alpha_mask_type> renderer_alpha_mask_type;

struct SnapData {
  SnapData(const bool& newpoint, const float& xsnap, const float& ysnap) :
    newpoint(newpoint), xsnap(xsnap), ysnap(ysnap) {}
  bool newpoint;
  float xsnap, ysnap;
};

class SafeSnap {
  // snap to pixel center, avoiding 0 path length rounding errors.
public:
  SafeSnap() : first(true), xsnap(0.0), lastx(0.0), lastxsnap(0.0),
	       ysnap(0.0), lasty(0.0), lastysnap(0.0)  {}
  SnapData snap (const float& x, const float& y);
      
private:
  bool first;
  float xsnap, lastx, lastxsnap, ysnap, lasty, lastysnap;
};

// a helper class to pass agg::buffer objects around.  agg::buffer is
// a class in the swig wrapper
class BufferRegion : public Py::PythonExtension<BufferRegion> {
public:
  BufferRegion( agg::buffer& aggbuf, const agg::rect &r, bool freemem=true) : aggbuf(aggbuf), rect(r), freemem(freemem) {
    //std::cout << "buffer region" << std::endl;
  }
  agg::buffer aggbuf;
  agg::rect rect;
  bool freemem;
  Py::Object to_string(const Py::Tuple &args);

  static void init_type(void);
  virtual ~BufferRegion() {
    //std::cout << "buffer region bye bye" << std::endl;
    if (freemem) {
      delete [] aggbuf.data;
      aggbuf.data = NULL;
    }
  };
};

class GCAgg {
public:
  GCAgg(const Py::Object& gc, double dpi, bool snapto=false);

  ~GCAgg() {
    delete [] dasha;
    delete [] cliprect;
  }

  double dpi;
  bool snapto;
  bool isaa;

  agg::line_cap_e cap;
  agg::line_join_e join;


  double linewidth;
  double alpha;
  agg::rgba color;

  double *cliprect;
  Py::Object clippath;
  agg::trans_affine clippath_trans;

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
  void _set_clip_path( const Py::Object& gc);
  void _set_antialiased( const Py::Object& gc);


};


//struct AMRenderer {
//  
//}

// the renderer
class RendererAgg: public Py::PythonExtension<RendererAgg> {
  typedef std::pair<bool, agg::rgba> facepair_t;
public:
  RendererAgg(unsigned int width, unsigned int height, double dpi, int debug);
  static void init_type(void);

  unsigned int get_width() { return width;}
  unsigned int get_height() { return height;}
  // the drawing methods
  //Py::Object _draw_markers_nocache(const Py::Tuple & args);
  //Py::Object _draw_markers_cache(const Py::Tuple & args);
  Py::Object draw_markers(const Py::Tuple & args);
  Py::Object draw_text_image(const Py::Tuple & args);
  Py::Object draw_image(const Py::Tuple & args);
  Py::Object draw_path(const Py::Tuple & args);

  Py::Object write_rgba(const Py::Tuple & args);
  Py::Object write_png(const Py::Tuple & args);
  Py::Object tostring_rgb(const Py::Tuple & args);
  Py::Object tostring_argb(const Py::Tuple & args);
  Py::Object tostring_bgra(const Py::Tuple & args);
  Py::Object buffer_rgba(const Py::Tuple & args);
  Py::Object clear(const Py::Tuple & args);

  Py::Object copy_from_bbox(const Py::Tuple & args);
  Py::Object restore_region(const Py::Tuple & args);

  virtual ~RendererAgg();

  static const size_t PIXELS_PER_INCH;
  unsigned int width, height;
  double dpi;
  size_t NUMBYTES;  //the number of bytes in buffer

  agg::int8u *pixBuffer;
  agg::rendering_buffer *renderingBuffer;

  agg::int8u *alphaBuffer;
  agg::rendering_buffer *alphaMaskRenderingBuffer;
  alpha_mask_type *alphaMask;
  agg::pixfmt_gray8 *pixfmtAlphaMask;
  renderer_base_alpha_mask_type *rendererBaseAlphaMask;
  renderer_alpha_mask_type *rendererAlphaMask;
  agg::scanline_p8 *scanlineAlphaMask;



  scanline_p8* slineP8;
  scanline_bin* slineBin;
  pixfmt *pixFmt;
  renderer_base *rendererBase;
  renderer_aa *rendererAA;
  renderer_bin *rendererBin;
  rasterizer *theRasterizer;


  const int debug;

protected:
  template<class T>
  agg::rect_base<T> bbox_to_rect( const Py::Object& o);
  double points_to_pixels( const Py::Object& points);
  double points_to_pixels_snapto( const Py::Object& points);
  int intersectCheck(double, double, double, double, double, int*);
  void set_clip_from_bbox(const Py::Object& o);
  agg::rgba rgb_to_color(const Py::SeqBase<Py::Object>& rgb, double alpha);
  facepair_t _get_rgba_face(const Py::Object& rgbFace, double alpha);
  template<class R>
  void set_clipbox(double *cliprect, R rasterizer);
  bool render_clippath(const GCAgg& gc);

private:
  Py::Object lastclippath;
  agg::trans_affine lastclippath_transform;
};

// the extension module
class _backend_agg_module : public Py::ExtensionModule<_backend_agg_module>
{
public:
  _backend_agg_module()
    : Py::ExtensionModule<_backend_agg_module>( "_backend_agg" )
  {
    RendererAgg::init_type();

    add_keyword_method("RendererAgg", &_backend_agg_module::new_renderer,
		       "RendererAgg(width, height, dpi)");
    add_varargs_method("point_in_path", &_backend_agg_module::point_in_path,
		       "point_in_path(x, y, path, trans)");
    add_varargs_method("point_on_path", &_backend_agg_module::point_on_path,
		       "point_on_path(x, y, r, path, trans)");
    add_varargs_method("get_path_extents", &_backend_agg_module::get_path_extents,
		       "get_path_extents(path, trans)");
    initialize( "The agg rendering backend" );
  }

  virtual ~_backend_agg_module() {}

private:

  Py::Object new_renderer (const Py::Tuple &args, const Py::Dict &kws);
  Py::Object point_in_path(const Py::Tuple& args);
  Py::Object point_on_path(const Py::Tuple& args);
  Py::Object get_path_extents(const Py::Tuple& args);
};



#endif

