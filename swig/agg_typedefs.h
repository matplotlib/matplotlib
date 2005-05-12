/* typedefs.h	-- John Hunter
 *
 * $Header$
 * $Log$
 * Revision 1.1  2005/05/12 19:16:35  jdh2358
 * cleaned up agg wrapper
 *
 */

#ifndef _TYPEDEFS_H
#define _TYPEDEFS_H
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

#endif
