//SWIG interface to agg_basics
%module agg
%{
#include "agg_basics.h"
#include "agg_color_rgba.h"
#include "agg_pixfmt_rgba.h"
#include "agg_trans_affine.h"
#include "agg_path_storage.h"  
#include "agg_buffer.h"   // my own buffer wrapper
#include "agg_rendering_buffer.h"  
#include "agg_renderer_base.h"
#include "agg_math_stroke.h"
#include "agg_conv_stroke.h"
#include "agg_vcgen_stroke.h"  
#include "agg_rasterizer_scanline_aa.h"
#include "agg_renderer_scanline.h"
#include "agg_render_scanlines.h"
#include "agg_scanline_bin.h"
#include "agg_scanline_p.h"

using namespace agg;

typedef agg::rgba8 color_type;
typedef agg::pixfmt_rgba32 pixel_format;  

 %}	

typedef agg::rgba8 color_type;
typedef agg::pixfmt_rgba32 pixel_format;  
//typedef pixel_formats_rgba<blender_rgba32, pixel32_type> pixel_format;

%include "agg_basics.i"


%typemap(argout) double *array6 {

  // Append output value $1 to $result
  $1 = PyString_AsString($input);   /* char *str */
  $2 = PyString_Size($input);       /* int len   */
  PyObject *ret = PyTuple_New(6);
  for (unsigned i=0; i<6; i++)
    PyTuple_SetItem(ret,i,PyFloat_FromDouble($1[i]));
  $result = ret;
}


%typemap(python,in) (unsigned char *bytes, int len)
{
  if (!PyString_Check($input)) {
    PyErr_SetString(PyExc_ValueError,"Expected a string");
    return NULL;
  }
  $1 = PyString_AsString($input);
  $2 = PyString_Size($input);
}

%typemap(out) agg::binary_data {
    $result = PyString_FromStringAndSize((const char*)$1.data,$1.size);
}

%include "agg_buffer.h"

%include "agg_color_rgba.h"
%include "agg_trans_affine.i"
%include "agg_path_storage.i"
%include "agg_math_stroke.h"
%include "agg_vcgen_stroke.h"
#include "agg_conv_adaptor_vcgen.h"

//%template(conv_adaptor_vcgen) agg::conv_adaptor_vcgen<agg::path_storage,agg::vcgen_stroke,agg::null_markers>;

%rename(null_markers) agg::null_markers;
struct agg::null_markers {};

%rename(conv_adaptor_vcgen) conv_adaptor_vcgen<agg::path_storage,agg::vcgen_stroke,agg::null_markers>;
class conv_adaptor_vcgen<agg::path_storage,agg::vcgen_stroke,agg::null_markers>
{
public:
  conv_adaptor_vcgen<agg::path_storage,agg::vcgen_stroke,agg::null_markers>(agg::path_storage& source);
};


%include "agg_rendering_buffer.h"
%template(rendering_buffer) agg::row_ptr_cache<agg::int8u>;
%extend agg::row_ptr_cache<agg::int8u> { 
  void attachb(buffer *buf) {
    self->attach(buf->data, buf->width, buf->height, buf->stride);
  }
}

%include "agg_pixfmt_rgba.h"
%template(pixel_format) agg::pixel_formats_rgba<agg::blender_rgba32, agg::pixel32_type>;



#include "agg_renderer_base.h"
%rename(renderer_base) renderer_base<pixel_format>;
class renderer_base<pixel_format>
{
public:
  renderer_base<pixel_format>(pixel_format& ren);
  const pixel_format& ren();
  unsigned width()  const;
  unsigned height() const;
  bool clip_box(int x1, int y1, int x2, int y2);
  void reset_clipping(bool visibility);
  void clip_box_naked(int x1, int y1, int x2, int y2);
  bool inbox(int x, int y) const;
  void first_clip_box();
  bool next_clip_box();
  const agg::rect& clip_box() const;
  int         xmin()     const;
  int         ymin()     const;
  int         xmax()     const;
  int         ymax()     const;
  
  const agg::rect& bounding_clip_box() const;
  int         bounding_xmin()     const;
  int         bounding_ymin()     const;
  int         bounding_xmax()     const;
  int         bounding_ymax()     const;
  void clear(const color_type& c);
  void copy_pixel(int x, int y, const color_type& c);
  void blend_pixel(int x, int y, const color_type& c, agg::cover_type cover);
  color_type pixel(int x, int y) const;
  void copy_hline(int x1, int y, int x2, const color_type& c);
  void copy_vline(int x, int y1, int y2, const color_type& c);
  void blend_hline(int x1, int y, int x2, 
		   const color_type& c, agg::cover_type cover);
  void blend_vline(int x, int y1, int y2, 
		   const color_type& c, agg::cover_type cover);
  void copy_bar(int x1, int y1, int x2, int y2, const color_type& c);
  void blend_bar(int x1, int y1, int x2, int y2, 
		 const color_type& c, agg::cover_type cover);
  void blend_solid_hspan(int x, int y, int len, 
			 const color_type& c, 
			 const agg::cover_type* covers);
  void blend_solid_vspan(int x, int y, int len, 
			 const color_type& c, 
			 const agg::cover_type* covers);

  void blend_color_hspan(int x, int y, int len, 
			 const color_type* colors, 
			 const agg::cover_type* covers,
			 agg::cover_type cover = cover_full);
  void blend_color_vspan(int x, int y, int len, 
			 const color_type* colors, 
			 const agg::cover_type* covers,
			 agg::cover_type cover = cover_full);
  void blend_color_hspan_no_clip(int x, int y, int len, 
				 const color_type* colors, 
				 const agg::cover_type* covers,
				 agg::cover_type cover = cover_full);
  void blend_color_vspan_no_clip(int x, int y, int len, 
				 const color_type* colors, 
				 const agg::cover_type* covers,
				 agg::cover_type cover = cover_full);
  rect clip_rect_area(agg::rect& dst, agg::rect& src, int wsrc, int hsrc) const;
  void copy_from(const agg::rendering_buffer& src, 
		 const agg::rect* rect_src_ptr = 0, 
		 int dx = 0, 
		 int dy = 0);
  /* todo fixme
  template<class SrcPixelFormatRenderer>
  void blend_from(const SrcPixelFormatRenderer& src, 
		  const rect* rect_src_ptr = 0, 
		  int dx = 0, 
		  int dy = 0);
  */
};


#include "agg_conv_stroke.h"
%rename(conv_stroke) conv_stroke<agg::path_storage>;
struct conv_stroke<agg::path_storage> {
  conv_stroke<agg::path_storage>(agg::path_storage& vs);
  void line_cap(agg::line_cap_e lc);
  void line_join(agg::line_join_e lj);
  agg::line_cap_e  line_cap();
  agg::line_join_e line_join();
  void width(double w);
  void miter_limit(double ml);
  void miter_limit_theta(double t);
  void approximation_scale(double as);
  double width() const;
  double miter_limit() const;
  double approximation_scale();
  void shorten(double s);
  double shorten() const;
};


#include "agg_rasterizer_scanline_aa.h"
%rename(rasterizer_scanline_aa) rasterizer_scanline_aa<>;
class rasterizer_scanline_aa<>
{
public:
  rasterizer_scanline_aa<>();
  void reset(); 
  void filling_rule(agg::filling_rule_e filling_rule);
  void clip_box(double x1, double y1, double x2, double y2);
  void reset_clipping();
  //template<class GammaF> void gamma(const GammaF& gamma_function)
  unsigned apply_gamma(unsigned cover) const; 
  void add_vertex(double x, double y, unsigned cmd);
  void move_to(int x, int y);
  void line_to(int x, int y);
  void close_polygon();
  void move_to_d(double x, double y);
  void line_to_d(double x, double y);
  int min_x() const;
  int min_y() const;
  int max_x() const;
  int max_y() const;
  
  unsigned calculate_alpha(int area);
  void sort();
  bool rewind_scanlines();
  //template<class Scanline> bool sweep_scanline(Scanline& sl);
  bool hit_test(int tx, int ty);
  //fixme void add_xy(const double* x, const double* y, unsigned n);
  //template<class VertexSource>
  void add_path(agg::path_storage& vs, unsigned id=0);
  void add_path(agg::conv_stroke<agg::path_storage>& vs, unsigned id=0);
  
};






#include "agg_renderer_scanline.h"
//todo fixme color type
%rename(renderer_scanline_aa_solid) renderer_scanline_aa_solid<renderer_base<pixel_format> >;
class renderer_scanline_aa_solid<renderer_base<pixel_format> >
{
public:
  renderer_scanline_aa_solid<renderer_base<pixel_format> >(renderer_base<pixel_format>& ren);
  void color(const color_type& c);
  const color_type& color() const;
  void prepare(unsigned);
};

%rename(renderer_scanline_bin_solid) renderer_scanline_bin_solid<renderer_base<pixel_format> >;
class renderer_scanline_bin_solid<renderer_base<pixel_format> >
{
public:
  renderer_scanline_bin_solid<renderer_base<pixel_format> >(renderer_base<pixel_format>& ren);
  void color(const color_type& c);
  const color_type& color() const;
  void prepare(unsigned);
};


%include "agg_scanline_p.h"
%template(scanline_p8) agg::scanline_p<agg::int8u>;

%include "agg_scanline_bin.h"

%include "agg_render_scanlines.h"
%template(render_scanlines) agg::render_scanlines<
  rasterizer_scanline_aa<>,
  agg::scanline_p<agg::int8u>,
  renderer_scanline_aa_solid<renderer_base<pixel_format> > >;
						  
					    

