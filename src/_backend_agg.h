/* _backend_agg.h - A rewrite of _backend_agg using PyCXX to handle
   ref counting, etc..
*/

#ifndef __BACKEND_AGG_H
#define __BACKEND_AGG_H

#include "CXX/Extensions.hxx"

#include "agg_arrowhead.h"
#include "agg_basics.h"
#include "agg_conv_concat.h"
#include "agg_conv_contour.h"
#include "agg_conv_curve.h"
#include "agg_conv_dash.h"
#include "agg_conv_marker.h"
#include "agg_conv_marker_adaptor.h"
#include "agg_conv_stroke.h"
#include "agg_ellipse.h"
#include "agg_embedded_raster_fonts.h"
#include "agg_gen_markers_term.h"
#include "agg_path_storage.h"
#include "agg_pixfmt_rgb24.h"
#include "agg_pixfmt_rgba32.h"
#include "agg_rasterizer_outline.h"
#include "agg_rasterizer_scanline_aa.h"
#include "agg_scanline_bin.h"
#include "agg_renderer_outline_aa.h"
#include "agg_renderer_raster_text.h"
#include "agg_renderer_scanline.h"
#include "agg_rendering_buffer.h"
#include "agg_scanline_p32.h"


typedef agg::pixel_formats_rgba32<agg::order_rgba32> pixfmt;
//typedef agg::pixel_formats_rgb24<agg::order_bgr24> pixfmt;
typedef agg::renderer_base<pixfmt> renderer_base;
typedef agg::renderer_scanline_p_solid<renderer_base> renderer;
typedef agg::renderer_scanline_bin_solid<renderer_base> renderer_bin;
typedef agg::rasterizer_scanline_aa<> rasterizer;
typedef agg::scanline_p8 scanline_p8;
typedef agg::scanline_bin scanline_bin;

// the renderer
class RendererAgg: public Py::PythonExtension<RendererAgg> {
public:
  RendererAgg(unsigned int width, unsigned int height, double dpi, int debug);
  static void init_type(void);

  unsigned int get_width() { return width;}
  unsigned int get_height() { return height;}
  // the drawing methods
  Py::Object draw_rectangle(const Py::Tuple & args);
  Py::Object draw_ellipse(const Py::Tuple & args);
  Py::Object draw_polygon(const Py::Tuple & args);
  Py::Object draw_lines(const Py::Tuple & args);
  Py::Object draw_text(const Py::Tuple & args);
  Py::Object draw_image(const Py::Tuple & args);

  Py::Object write_rgba(const Py::Tuple & args);
  Py::Object write_png(const Py::Tuple & args);
  Py::Object tostring_rgb(const Py::Tuple & args);


  virtual ~RendererAgg(); 

  static const size_t PIXELS_PER_INCH;
  unsigned int width, height;
  double dpi;
  size_t NUMBYTES;  //the number of bytes in buffer

  agg::int8u *pixBuffer;
  agg::rendering_buffer *renderingBuffer;

  scanline_p8* slineP8;
  scanline_bin* slineBin;
  pixfmt *pixFmt;
  renderer_base *rendererBase;
  renderer *theRenderer;
  renderer_bin *rendererBin;
  rasterizer *theRasterizer;


  const int debug;

protected:

  // helper methods to process gc
  void set_clip_rectangle( const Py::Object& gc);
  Py::Tuple get_dashes( const Py::Object& gc);
  int antialiased( const Py::Object& gc);
  agg::rgba get_color(const Py::Object& gc);
  agg::gen_stroke::line_cap_e get_linecap(const Py::Object& gc);
  agg::gen_stroke::line_join_e get_joinstyle(const Py::Object& gc);

  double points_to_pixels( const Py::Object& points);
  double points_to_pixels_snapto( const Py::Object& points);
  agg::rgba rgb_to_color(const Py::SeqBase<Py::Object>& rgb, double alpha);
  
  
} ;


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
    initialize( "The agg rendering backend" );
  }
  
  virtual ~_backend_agg_module() {}
  
private:
  
  Py::Object new_renderer (const Py::Tuple &args, const Py::Dict &kws);


  
};



#endif

