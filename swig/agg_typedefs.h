/* typedefs.h	-- John Hunter
 *
 * $Header$
 * $Log$
 * Revision 1.2  2005/06/23 22:25:10  jdh2358
 * some work on agg swig wrapper
 *
 * Revision 1.1  2005/05/12 19:16:35  jdh2358
 * cleaned up agg wrapper
 *
 */

#ifndef _TYPEDEFS_H
#define _TYPEDEFS_H
//#include "agg_span_interpolator_linear.h"
//#include "agg_span_image_filter_rgba.h"


typedef agg::path_storage path_t;
typedef agg::conv_stroke<path_t> stroke_t;
typedef agg::conv_transform<path_t, agg::trans_affine> transpath_t;
typedef agg::conv_stroke<transpath_t> stroketrans_t;
typedef agg::conv_curve<path_t> curve_t;
typedef agg::conv_stroke<curve_t> strokecurve_t;
typedef agg::conv_transform<curve_t, agg::trans_affine> transcurve_t;
typedef agg::conv_stroke<transcurve_t> stroketranscurve_t;
typedef agg::conv_curve<transpath_t> curvetrans_t;
typedef agg::conv_stroke<curvetrans_t> strokecurvetrans_t;
typedef agg::pixel_formats_rgba<agg::blender_rgba32, agg::pixel32_type> pixfmt_rgba_t;
typedef agg::renderer_base<pixfmt_rgba_t> renderer_base_rgba_t;

//typedef agg::span_interpolator_linear<agg::trans_affine>   interpolator_linear_t;

//typedef agg::span_image_filter_rgba_nn<agg::rgba8, agg::order_rgba, interpolator_linear_t> span_imfilt_rgba_nn_interplinear_t;

#endif
