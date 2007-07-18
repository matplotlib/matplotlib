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
#include "agg_conv_transform.h"
#include "agg_conv_curve.h"
#include "agg_vcgen_stroke.h"  
#include "agg_rasterizer_scanline_aa.h"
#include "agg_renderer_scanline.h"
#include "agg_render_scanlines.h"
#include "agg_scanline_bin.h"
#include "agg_scanline_p.h"
#include "agg_span_interpolator_linear.h"


using namespace agg;

#include "agg_typedefs.h"
 %}	


%include "agg_typedefs.h"
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



%include "agg_rendering_buffer.h"
%template(rendering_buffer) agg::row_ptr_cache<agg::int8u>;
%extend agg::row_ptr_cache<agg::int8u> { 
  void attachb(buffer *buf) {
    self->attach(buf->data, buf->width, buf->height, buf->stride);
  }
}

%include "agg_pixfmt_rgba.h"
%template(pixel_format_rgba) agg::pixel_formats_rgba<agg::blender_rgba32, agg::pixel32_type>;


%include "agg_renderer_base.i"
%include "agg_conv_curve.i"
%include "agg_conv_transform.i"
%include "agg_conv_stroke.i"

%include "agg_rasterizer_scanline_aa.i"


//%include "agg_span_interpolator_linear.h"
//%template(span_interpolator_linear_affine) agg::span_interpolator_linear<agg::trans_affine>;
//%include "agg_span_image_filter.i"

%include "agg_renderer_scanline.i"


%include "agg_scanline_p.h"
%template(scanline_p8) agg::scanline_p<agg::int8u>;

%include "agg_scanline_bin.i"


%include "agg_render_scanlines.h"
%template(render_scanlines_rgba) agg::render_scanlines<
  agg::rasterizer_scanline_aa<>,
  agg::scanline_p<agg::int8u>,
  agg::renderer_scanline_aa_solid<renderer_base_rgba_t> >;



%template(render_scanlines_bin_rgba) agg::render_scanlines<
  agg::rasterizer_scanline_aa<>,
  agg::scanline_bin,
  agg::renderer_scanline_bin_solid<renderer_base_rgba_t> >;

