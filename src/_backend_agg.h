/* _backend_agg.h	-- 
 *
 * $Header$
 * $Log$
 * Revision 1.3  2004/03/02 20:47:52  jdh2358
 * update htdocs - lots of small fixes
 *
 * Revision 1.2  2004/02/23 16:20:57  jdh2358
 * resynced cvs to freetype2
 *
 * Revision 1.1  2004/02/12 18:30:52  jdh2358
 * finished agg backend
 *
 */

#ifndef __BACKEND_AGG_H
#define __BACKEND_AGG_H

#include <fstream>
#include <cmath>
#include <cstdio>
#include <freetype/freetype.h>

#include "agg_arrowhead.h"
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

#include "Python.h"

/* PIXEL FORMAT, if you change the pixel format, you will need to make
     a few changes to _backend_agg.cpp

  renderer->buffer allocation - the size of the buffer for rgb is 2
      and rgba is 4; see newRendererAggObject

  stride : ditto; see newRendererAggObject

  write_png : the write png code needs to know the pixel size both for
     determing the row pointers and for setting the image type flag

  Also, if you switch from RGB to BGR you will probably want to take a
    look at how color are created in gc_get_color

  Q: are there any methods to help write pixel neutral code, something
     like sizeof(pixfmt)

*/

typedef agg::pixel_formats_rgba32<agg::order_rgba32> pixfmt;
//typedef agg::pixel_formats_rgb24<agg::order_bgr24> pixfmt;
typedef agg::renderer_base<pixfmt> renderer_base;
typedef agg::renderer_scanline_p_solid<renderer_base> renderer;
typedef agg::renderer_scanline_bin_solid<renderer_base> renderer_bin;
typedef agg::rasterizer_scanline_aa<> rasterizer;
typedef agg::scanline_p8 scanline_p8;
typedef agg::scanline_bin scanline_bin;





/*----------------------------------
 *
 *   The Renderer
 *
 *----------------------------------- */
 
typedef struct {
  PyObject_HEAD
  PyObject	*x_attr;	/* Attributes dictionary */
  agg::rendering_buffer *rbuf;
  pixfmt *pixf;
  renderer_base *rbase;
  renderer *ren;
  renderer_bin *ren_bin;
  rasterizer *ras;
  agg::int8u *buffer;
  scanline_p8 *sline_p8;
  scanline_bin *sline_bin;
  size_t NUMBYTES;  //the number of bytes in buffer
  double dpi;
} RendererAggObject;



#endif

