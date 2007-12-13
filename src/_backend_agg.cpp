/* A rewrite of _backend_agg using PyCXX to handle ref counting, etc..
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <png.h>
#include <time.h>
#include <algorithm>


#include "agg_conv_transform.h"
#include "agg_conv_curve.h"
#include "agg_scanline_storage_aa.h"
#include "agg_scanline_storage_bin.h"
#include "agg_renderer_primitives.h"
#include "agg_span_image_filter_gray.h"
#include "agg_span_interpolator_linear.h"
#include "agg_span_allocator.h"
#include "util/agg_color_conv_rgb8.h"

#include "ft2font.h"
#include "_image.h"
#include "_backend_agg.h"
#include "_transforms.h"
#include "mplutils.h"

#include "swig_runtime.h"
#include "MPL_isnan.h"

#define PY_ARRAY_TYPES_PREFIX NumPy
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"

#ifndef M_PI
#define M_PI       3.14159265358979323846
#endif
#ifndef M_PI_4
#define M_PI_4     0.785398163397448309616
#endif
#ifndef M_PI_2
#define M_PI_2     1.57079632679489661923
#endif

GCAgg::GCAgg(const Py::Object &gc, double dpi, bool snapto) :
  dpi(dpi), snapto(snapto), isaa(true), linewidth(1.0), alpha(1.0),
  cliprect(NULL), clippath(NULL),
  Ndash(0), dashOffset(0.0), dasha(NULL)
{

  _VERBOSE("GCAgg::GCAgg");
  linewidth = points_to_pixels ( gc.getAttr("_linewidth") ) ;
  alpha = Py::Float( gc.getAttr("_alpha") );
  color = get_color(gc);
  _set_antialiased(gc);
  _set_linecap(gc);
  _set_joinstyle(gc);
  _set_dashes(gc);
  _set_clip_rectangle(gc);
  _set_clip_path(gc);
}

void
GCAgg::_set_antialiased(const Py::Object& gc) {
  _VERBOSE("GCAgg::antialiased");
  isaa = Py::Int( gc.getAttr( "_antialiased") );

}

agg::rgba
GCAgg::get_color(const Py::Object& gc) {
  _VERBOSE("GCAgg::get_color");
  Py::Tuple rgb = Py::Tuple( gc.getAttr("_rgb") );

  double alpha = Py::Float( gc.getAttr("_alpha") );

  double r = Py::Float(rgb[0]);
  double g = Py::Float(rgb[1]);
  double b = Py::Float(rgb[2]);
  return agg::rgba(r, g, b, alpha);
}

double
GCAgg::points_to_pixels( const Py::Object& points) {
  _VERBOSE("GCAgg::points_to_pixels");
  double p = Py::Float( points ) ;
  return p * dpi/72.0;
}


void
GCAgg::_set_linecap(const Py::Object& gc) {
  _VERBOSE("GCAgg::_set_linecap");

  std::string capstyle = Py::String( gc.getAttr( "_capstyle" ) );

  if (capstyle=="butt")
    cap = agg::butt_cap;
  else if (capstyle=="round")
    cap = agg::round_cap;
  else if(capstyle=="projecting")
    cap = agg::square_cap;
  else
    throw Py::ValueError(Printf("GC _capstyle attribute must be one of butt, round, projecting; found %s", capstyle.c_str()).str());

}

void
GCAgg::_set_joinstyle(const Py::Object& gc) {
  _VERBOSE("GCAgg::_set_joinstyle");

  std::string joinstyle = Py::String( gc.getAttr("_joinstyle") );

  if (joinstyle=="miter")
    join =  agg::miter_join;
  else if (joinstyle=="round")
    join = agg::round_join;
  else if(joinstyle=="bevel")
    join = agg::bevel_join;
  else
    throw Py::ValueError(Printf("GC _joinstyle attribute must be one of butt, round, projecting; found %s", joinstyle.c_str()).str());

}

void
GCAgg::_set_dashes(const Py::Object& gc) {
  //return the dashOffset, dashes sequence tuple.
  _VERBOSE("GCAgg::_set_dashes");

  delete [] dasha;
  dasha = NULL;

  Py::Tuple dashtup = gc.getAttr("_dashes");

  if (dashtup.length()!=2)
    throw Py::ValueError(Printf("GC dashtup must be a length 2 tuple; found %d", dashtup.length()).str());


  bool useDashes = dashtup[0].ptr() != Py_None;

  if ( !useDashes ) return;

  dashOffset = points_to_pixels(dashtup[0]);
  Py::SeqBase<Py::Object> dashSeq;
  dashSeq = dashtup[1];

  Ndash = dashSeq.length();
  if (Ndash%2 != 0  )
    throw Py::ValueError(Printf("dash sequence must be an even length sequence; found %d", Ndash).str());

  dasha = new double[Ndash];
  double val;
  for (size_t i=0; i<Ndash; i++) {
    val = points_to_pixels(dashSeq[i]);
    if (this->snapto) val = (int)val +0.5;
    dasha[i] = val;
  }
}


void
GCAgg::_set_clip_rectangle( const Py::Object& gc) {
  //set the clip rectangle from the gc

  _VERBOSE("GCAgg::_set_clip_rectangle");

  delete [] cliprect;
  cliprect = NULL;

  Py::Object o ( gc.getAttr( "_cliprect" ) );
  if (o.ptr()==Py_None) {
    return;
  }

  Py::SeqBase<Py::Object> rect( o );

  double l = Py::Float(rect[0]) ;
  double b = Py::Float(rect[1]) ;
  double w = Py::Float(rect[2]) ;
  double h = Py::Float(rect[3]) ;

  cliprect = new double[4];
  //todo check for memory alloc failure
  cliprect[0] = l;
  cliprect[1] = b;
  cliprect[2] = w;
  cliprect[3] = h;
}

void
GCAgg::_set_clip_path( const Py::Object& gc) {
  //set the clip path from the gc

  _VERBOSE("GCAgg::_set_clip_path");

  delete clippath;
  clippath = NULL;

  Py::Object o  = gc.getAttr( "_clippath" );
  if (o.ptr()==Py_None) {
    return;
  }

  agg::path_storage *tmppath;
  swig_type_info * descr = SWIG_TypeQuery("agg::path_storage *");
  assert(descr);
  if (SWIG_ConvertPtr(o.ptr(),(void **)(&tmppath), descr, 0) == -1) {
    throw Py::TypeError("Could not convert gc path_storage");
  }

  tmppath->rewind(0);
  clippath = new agg::path_storage();
  clippath->copy_from(*tmppath);
  clippath->rewind(0);
  tmppath->rewind(0);
}


Py::Object BufferRegion::to_string(const Py::Tuple &args) {

  // owned=true to prevent memory leak
  return Py::String(PyString_FromStringAndSize((const char*)aggbuf.data,aggbuf.height*aggbuf.stride), true);
}




const size_t
RendererAgg::PIXELS_PER_INCH(96);

RendererAgg::RendererAgg(unsigned int width, unsigned int height, double dpi,
			 int debug) :
  width(width),
  height(height),
  dpi(dpi),
  NUMBYTES(width*height*4),
  debug(debug),
  lastclippath(NULL)
{
  _VERBOSE("RendererAgg::RendererAgg");
  unsigned stride(width*4);


  pixBuffer = new agg::int8u[NUMBYTES];
  renderingBuffer = new agg::rendering_buffer;
  renderingBuffer->attach(pixBuffer, width, height, stride);

  alphaBuffer = new agg::int8u[NUMBYTES];
  alphaMaskRenderingBuffer = new agg::rendering_buffer;
  alphaMaskRenderingBuffer->attach(alphaBuffer, width, height, stride);
  alphaMask = new alpha_mask_type(*alphaMaskRenderingBuffer);

  pixfmtAlphaMask = new agg::pixfmt_gray8(*alphaMaskRenderingBuffer);
  rendererBaseAlphaMask = new renderer_base_alpha_mask_type(*pixfmtAlphaMask);
  rendererAlphaMask = new renderer_alpha_mask_type(*rendererBaseAlphaMask);
  scanlineAlphaMask = new agg::scanline_p8();


  slineP8 = new scanline_p8;
  slineBin = new scanline_bin;


  pixFmt = new pixfmt(*renderingBuffer);
  rendererBase = new renderer_base(*pixFmt);
  rendererBase->clear(agg::rgba(1, 1, 1, 0));

  rendererAA = new renderer_aa(*rendererBase);
  rendererBin = new renderer_bin(*rendererBase);
  theRasterizer = new rasterizer();
  //theRasterizer->filling_rule(agg::fill_even_odd);
  //theRasterizer->filling_rule(agg::fill_non_zero);

};



void
RendererAgg::set_clipbox_rasterizer( double *cliprect) {
  //set the clip rectangle from the gc

  _VERBOSE("RendererAgg::set_clipbox_rasterizer");


  theRasterizer->reset_clipping();
  rendererBase->reset_clipping(true);

  //if (cliprect==NULL) {
  //  theRasterizer->reset_clipping();
  //  rendererBase->reset_clipping(true);
  //}
  if (cliprect!=NULL) {

    double l = cliprect[0] ;
    double b = cliprect[1] ;
    double w = cliprect[2] ;
    double h = cliprect[3] ;

    theRasterizer->clip_box(l, height-(b+h),
			    l+w, height-b);
  }
  _VERBOSE("RendererAgg::set_clipbox_rasterizer done");

}

std::pair<bool, agg::rgba>
RendererAgg::_get_rgba_face(const Py::Object& rgbFace, double alpha) {
  _VERBOSE("RendererAgg::_get_rgba_face");
  std::pair<bool, agg::rgba> face;

  if (rgbFace.ptr() == Py_None) {
    face.first = false;
  }
  else {
    face.first = true;
    Py::Tuple rgb = Py::Tuple(rgbFace);
    face.second = rgb_to_color(rgb, alpha);
  }
  return face;

}

template <class VS>
void
RendererAgg::_fill_and_stroke(VS& path,
			      const GCAgg& gc,
			      const facepair_t& face,
			      bool curvy) {
  typedef agg::conv_curve<VS> curve_t;

  //bool isclippath(gc.clippath!=NULL);
  //if (isclippath) _process_alpha_mask(gc);

  if (face.first) {
    rendererAA->color(face.second);
    if (curvy) {
      curve_t curve(path);
      theRasterizer->add_path(curve);
    }
    else
      theRasterizer->add_path(path);

    /*
    if (isclippath) {
	typedef agg::pixfmt_amask_adaptor<pixfmt, alpha_mask_type> pixfmt_amask_type;
	typedef agg::renderer_base<pixfmt_amask_type>              amask_ren_type;
	pixfmt_amask_type pfa(*pixFmt, *alphaMask);
	amask_ren_type r(pfa);
	typedef agg::renderer_scanline_aa_solid<amask_ren_type> renderer_type;
	renderer_type ren(r);
	ren.color(gc.color);
	//std::cout << "render clippath" << std::endl;

	agg::render_scanlines(*theRasterizer, *slineP8, ren);
      }
      else {
	rendererAA->color(gc.color);
	agg::render_scanlines(*theRasterizer, *slineP8, *rendererAA);
      }
    */
    agg::render_scanlines(*theRasterizer, *slineP8, *rendererAA);
  }

  //now stroke the edge
  if (gc.linewidth) {
    if (curvy) {
      curve_t curve(path);
      agg::conv_stroke<curve_t> stroke(curve);
      stroke.width(gc.linewidth);
      stroke.line_cap(gc.cap);
      stroke.line_join(gc.join);
      theRasterizer->add_path(stroke);
    }
    else {
      agg::conv_stroke<VS> stroke(path);
      stroke.width(gc.linewidth);
      stroke.line_cap(gc.cap);
      stroke.line_join(gc.join);
      theRasterizer->add_path(stroke);
    }


    /*
    if ( gc.isaa ) {
      if (isclippath) {
	typedef agg::pixfmt_amask_adaptor<pixfmt, alpha_mask_type> pixfmt_amask_type;
	typedef agg::renderer_base<pixfmt_amask_type>              amask_ren_type;
	pixfmt_amask_type pfa(*pixFmt, *alphaMask);
	amask_ren_type r(pfa);
	typedef agg::renderer_scanline_aa_solid<amask_ren_type> renderer_type;
	renderer_type ren(r);
	ren.color(gc.color);
	//std::cout << "render clippath" << std::endl;

	agg::render_scanlines(*theRasterizer, *slineP8, ren);
      }
      else {
	rendererAA->color(gc.color);
	agg::render_scanlines(*theRasterizer, *slineP8, *rendererAA);
      }
    }
    else {
      if (isclippath) {
	typedef agg::pixfmt_amask_adaptor<pixfmt, alpha_mask_type> pixfmt_amask_type;
	typedef agg::renderer_base<pixfmt_amask_type>              amask_ren_type;
	pixfmt_amask_type pfa(*pixFmt, *alphaMask);
	amask_ren_type r(pfa);
	typedef agg::renderer_scanline_bin_solid<amask_ren_type> renderer_type;
	renderer_type ren(r);
	ren.color(gc.color);
	agg::render_scanlines(*theRasterizer, *slineP8, ren);
      }
      else{
	rendererBin->color(gc.color);
	agg::render_scanlines(*theRasterizer, *slineBin, *rendererBin);
      }
    }

    */

    if ( gc.isaa ) {
      rendererAA->color(gc.color);
      agg::render_scanlines(*theRasterizer, *slineP8, *rendererAA);
    }
    else {
      rendererBin->color(gc.color);
      agg::render_scanlines(*theRasterizer, *slineBin, *rendererBin);
    }
  }


}

Py::Object
RendererAgg::draw_rectangle(const Py::Tuple & args) {
  _VERBOSE("RendererAgg::draw_rectangle");
  args.verify_length(6);


  GCAgg gc = GCAgg(args[0], dpi);
  facepair_t face = _get_rgba_face(args[1], gc.alpha);

  double l = Py::Float( args[2] );
  double b = Py::Float( args[3] );
  double w = Py::Float( args[4] );
  double h = Py::Float( args[5] );

  b = height - (b+h);
  double r = l + w;
  double t = b + h;

  //snapto pixel centers
  l = (int)l + 0.5;
  b = (int)b + 0.5;
  r = (int)r + 0.5;
  t = (int)t + 0.5;


  set_clipbox_rasterizer(gc.cliprect);

  agg::path_storage path;


  path.move_to(l, t);
  path.line_to(r, t);
  path.line_to(r, b);
  path.line_to(l, b);
  path.close_polygon();

  _fill_and_stroke(path, gc, face, false);

  return Py::Object();

}

Py::Object
RendererAgg::draw_ellipse(const Py::Tuple& args) {
  _VERBOSE("RendererAgg::draw_ellipse");
  args.verify_length(7);

  GCAgg gc = GCAgg(args[0], dpi);
  facepair_t face = _get_rgba_face(args[1], gc.alpha);

  double x = Py::Float( args[2] );
  double y = Py::Float( args[3] );
  double w = Py::Float( args[4] );
  double h = Py::Float( args[5] );
  double rot = Py::Float( args[6] );

  double r; // rot in radians

  set_clipbox_rasterizer(gc.cliprect);

  // Approximate the ellipse with 4 bezier paths
  agg::path_storage path;
  if (rot == 0.0) // simple case
    {
      path.move_to(x, height-(y+h));
      path.arc_to(w, h, 0.0, false, true, x+w, height-y);
      path.arc_to(w, h, 0.0, false, true, x,   height-(y-h));
      path.arc_to(w, h, 0.0, false, true, x-w, height-y);
      path.arc_to(w, h, 0.0, false, true, x,   height-(y+h));
      path.close_polygon();
    }
  else // rotate by hand :(
    {
      // deg to rad
      r = rot * (M_PI/180.0);
      path.move_to(                      x+(cos(r)*w),          height-(y+(sin(r)*w)));
      path.arc_to(w, h, -r, false, true, x+(cos(r+M_PI_2*3)*h), height-(y+(sin(r+M_PI_2*3)*h)));
      path.arc_to(w, h, -r, false, true, x+(cos(r+M_PI)*w),     height-(y+(sin(r+M_PI)*w)));
      path.arc_to(w, h, -r, false, true, x+(cos(r+M_PI_2)*h),   height-(y+(sin(r+M_PI_2)*h)));
      path.arc_to(w, h, -r, false, true, x+(cos(r)*w),          height-(y+(sin(r)*w)));
      path.close_polygon();
    }

  _fill_and_stroke(path, gc, face);
  return Py::Object();

}

Py::Object
RendererAgg::draw_polygon(const Py::Tuple& args) {
  _VERBOSE("RendererAgg::draw_polygon");

  args.verify_length(3);

  GCAgg gc = GCAgg(args[0], dpi);
  facepair_t face = _get_rgba_face(args[1], gc.alpha);

  Py::SeqBase<Py::Object> points( args[2] );

  set_clipbox_rasterizer(gc.cliprect);

  size_t Npoints = points.length();
  if (Npoints<=0)
    return Py::Object();


  // dump the x.y vertices into a double array for faster look ahead
  // and behind access
  double *xs = new double[Npoints];
  double *ys = new double[Npoints];

  for (size_t i=0; i<Npoints; i++) {
    Py::SeqBase<Py::Object> xy(points[i]);
    xy = Py::Tuple(points[i]);
    xs[i] = Py::Float(xy[0]);
    ys[i] = Py::Float(xy[1]);
    ys[i] = height - ys[i];
  }



  agg::path_storage path;
  for (size_t j=0; j<Npoints; j++) {

    double x = xs[j];
    double y = ys[j];

    //snapto pixel centers
    x = (int)x + 0.5;
    y = (int)y + 0.5;

    if (j==0) path.move_to(x,y);
    else path.line_to(x,y);
  }
  path.close_polygon();

  _fill_and_stroke(path, gc, face, false);

  delete [] xs;
  delete [] ys;

  _VERBOSE("RendererAgg::draw_polygon DONE");
  return Py::Object();

}



SnapData
SafeSnap::snap (const float& x, const float& y) {
  xsnap = (int)x + 0.5;
  ysnap = (int)y + 0.5;



  if ( first || ( (xsnap!=lastxsnap) || (ysnap!=lastysnap) ) ) {
    lastxsnap = xsnap;
    lastysnap = ysnap;
    lastx = x;
    lasty = y;
    first = false;
    return SnapData(true, xsnap, ysnap);
  }

  // ok both are equal and we need to do an offset
  if ( (x==lastx) && (y==lasty) ) {
    // no choice but to return equal coords; set newpoint = false
    lastxsnap = xsnap;
    lastysnap = ysnap;
    lastx = x;
    lasty = y;
    return SnapData(false, xsnap, ysnap);
  }

  // ok the real points are not identical but the rounded ones, so do
  // a one pixel offset
  if (x>lastx) xsnap += 1.;
  else if (x<lastx) xsnap -= 1.;

  if (y>lasty) ysnap += 1.;
  else if (y<lasty) ysnap -= 1.;

  lastxsnap = xsnap;
  lastysnap = ysnap;
  lastx = x;
  lasty = y;
  return SnapData(true, xsnap, ysnap);
}





Py::Object
RendererAgg::draw_line_collection(const Py::Tuple& args) {

  _VERBOSE("RendererAgg::draw_line_collection");

  args.verify_length(9);
  theRasterizer->reset_clipping();


  //segments, trans, clipbox, colors, linewidths, antialiaseds
  Py::SeqBase<Py::Object> segments = args[0];

  Transformation* transform = static_cast<Transformation*>(args[1].ptr());

  set_clip_from_bbox(args[2]);

  Py::SeqBase<Py::Object> colors = args[3];
  Py::SeqBase<Py::Object> linewidths = args[4];
  Py::SeqBase<Py::Object> linestyle = args[5];
  Py::SeqBase<Py::Object> antialiaseds = args[6];

  bool usingOffsets = args[7].ptr()!=Py_None;
  Py::SeqBase<Py::Object> offsets;
  Transformation* transOffset=NULL;
  if  (usingOffsets) {
    offsets = Py::SeqBase<Py::Object>(args[7]);
    transOffset = static_cast<Transformation*>(args[8].ptr());
  }

  size_t Nsegments = segments.length();
  size_t Nc = colors.length();
  size_t Nlw = linewidths.length();
  size_t Naa = antialiaseds.length();
  size_t Noffsets = 0;
  size_t N = Nsegments;
  size_t Ndash = 0;

  Py::SeqBase<Py::Object> dashtup(linestyle);
  bool useDashes = dashtup[0].ptr() != Py_None;

  double offset = 0;
  Py::SeqBase<Py::Object> dashSeq;
  typedef agg::conv_dash<agg::path_storage> dash_t;
  double *dasha = NULL;

  if ( useDashes ) {

    //TODO: use offset
    offset = points_to_pixels_snapto(dashtup[0]);
    dashSeq = dashtup[1];

    Ndash = dashSeq.length();
    if (Ndash%2 != 0  )
      throw Py::ValueError(Printf("dashes must be an even length sequence; found %d", N).str());

    dasha = new double[Ndash];

    for (size_t i=0; i<Ndash; i++)
      dasha[i] = points_to_pixels(dashSeq[i]);
  }


  if (usingOffsets) {
    Noffsets = offsets.length();
    if (Noffsets>Nsegments) N = Noffsets;
  }

  double xo(0.0), yo(0.0), thisx(0.0), thisy(0.0);
  std::pair<double, double> xy;
  Py::SeqBase<Py::Object> xyo;
  Py::SeqBase<Py::Object> xys;
  for (size_t i=0; i<N; i++) {
    if (usingOffsets) {
      xyo = Py::SeqBase<Py::Object>(offsets[i%Noffsets]);
      xo = Py::Float(xyo[0]);
      yo = Py::Float(xyo[1]);
      try {
	xy = transOffset->operator()(xo,yo);
      }
      catch (...) {
	throw Py::ValueError("Domain error on transOffset->operator in draw_line_collection");
      }

      xo = xy.first;
      yo = xy.second;
    }

    xys = segments[i%Nsegments];
    size_t numtups = xys.length();
    if (numtups<2) continue;


    bool snapto=numtups==2;
    agg::path_storage path;

    //std::cout << "trying snapto " << numtups << " " << snapto << std::endl;

    SafeSnap snap;


    for (size_t j=0; j<numtups; j++) {
      xyo = xys[j];
      thisx = Py::Float(xyo[0]);
      thisy = Py::Float(xyo[1]);
      try {
	xy = transform->operator()(thisx,thisy);
      }

      catch (...) {
	throw Py::ValueError("Domain error on transOffset->operator in draw_line_collection");
      }

      thisx = xy.first;
      thisy = xy.second;

      if (usingOffsets) {
	thisx += xo;
	thisy += yo;
      }

      if (snapto) { // snap to pixel for len(2) lines
	SnapData snapdata(snap.snap(thisx, thisy));
	// TODO: process newpoint
	//if (!snapdata.newpoint) {
	//  std::cout << "newpoint warning " << thisx << " " << thisy << std::endl;
	//}
	//std::cout << "snapto" << thisx << " " << thisy << std::endl;
	thisx = snapdata.xsnap;
	thisy = snapdata.ysnap;

	//thisx = (int)thisx + 0.5;
	//thisy = (int)thisy + 0.5;
      }

      if (j==0)  path.move_to(thisx, height-thisy);
      else       path.line_to(thisx, height-thisy);
    }



    double lw = points_to_pixels ( Py::Float( linewidths[i%Nlw] ) );

    if (! useDashes ) {

      agg::conv_stroke<agg::path_storage> stroke(path);
      //stroke.line_cap(cap);
      //stroke.line_join(join);
      stroke.width(lw);
      theRasterizer->add_path(stroke);
    }
    else {

      dash_t dash(path);
      //dash.dash_start(offset);
      for (size_t idash=0; idash<Ndash/2; idash++)
	dash.add_dash(dasha[2*idash], dasha[2*idash+1]);

      agg::conv_stroke<dash_t> stroke(dash);
      //stroke.line_cap(cap);
      //stroke.line_join(join);
      stroke.width(lw);
      theRasterizer->add_path(stroke);
    }

    // get the color and render
    Py::SeqBase<Py::Object> rgba(colors[ i%Nc]);
    double r = Py::Float(rgba[0]);
    double g = Py::Float(rgba[1]);
    double b = Py::Float(rgba[2]);
    double a = Py::Float(rgba[3]);
    agg::rgba color(r, g, b, a);

    // render antialiased or not
    int isaa = Py::Int(antialiaseds[i%Naa]);
    if ( isaa ) {
      rendererAA->color(color);
      agg::render_scanlines(*theRasterizer, *slineP8, *rendererAA);
    }
    else {
      rendererBin->color(color);
      agg::render_scanlines(*theRasterizer, *slineBin, *rendererBin);
    }
  } //for every segment
  if (useDashes) delete [] dasha;
  return Py::Object();
}



Py::Object
RendererAgg::copy_from_bbox(const Py::Tuple& args) {
  //copy region in bbox to buffer and return swig/agg buffer object
  args.verify_length(1);


  agg::rect r = bbox_to_rect(args[0]);
  /*
    r.x1 -=5;
    r.y1 -=5;
    r.x2 +=5;
    r.y2 +=5;
  */
  int boxwidth = r.x2-r.x1;
  int boxheight = r.y2-r.y1;
  int boxstride = boxwidth*4;
  agg::buffer buf(boxwidth, boxheight, boxstride, false);
  if (buf.data ==NULL) {
    throw Py::MemoryError("RendererAgg::copy_from_bbox could not allocate memory for buffer");
  }

  agg::rendering_buffer rbuf;
  rbuf.attach(buf.data, boxwidth, boxheight, boxstride);

  pixfmt pf(rbuf);
  renderer_base rb(pf);
  //rb.clear(agg::rgba(1, 0, 0)); //todo remove me
  rb.copy_from(*renderingBuffer, &r, -r.x1, -r.y1);
  BufferRegion* reg = new BufferRegion(buf, r, true);
  return Py::asObject(reg);



}

Py::Object
RendererAgg::restore_region(const Py::Tuple& args) {
  //copy BufferRegion to buffer
  args.verify_length(1);
  BufferRegion* region  = static_cast<BufferRegion*>(args[0].ptr());

  if (region->aggbuf.data==NULL)
    return Py::Object();
  //throw Py::ValueError("Cannot restore_region from NULL data");


  agg::rendering_buffer rbuf;
  rbuf.attach(region->aggbuf.data,
	      region->aggbuf.width,
	      region->aggbuf.height,
	      region->aggbuf.stride);

  rendererBase->copy_from(rbuf, 0, region->rect.x1, region->rect.y1);

  return Py::Object();



}


agg::rect_base<int>
RendererAgg::bbox_to_rect(const Py::Object& o) {
  //return the agg::rect for bbox, flipping y

  Bbox* clipbox = static_cast<Bbox*>(o.ptr());
  double l = clipbox->ll_api()->x_api()->val() ;
  double b = clipbox->ll_api()->y_api()->val();
  double r = clipbox->ur_api()->x_api()->val() ;
  double t = clipbox->ur_api()->y_api()->val() ; ;

  agg::rect rect( (int)l, height-(int)t, (int)r, height-(int)b ) ;
  if (!rect.is_valid())
    throw Py::ValueError("Invalid rectangle in bbox_to_rect");
  return rect;

}

void
RendererAgg::set_clip_from_bbox(const Py::Object& o) {

  // do not puut this in the else below.  We want to unconditionally
  // clear the clip
  theRasterizer->reset_clipping();
  rendererBase->reset_clipping(true);

  if (o.ptr() != Py_None) {  //using clip
    // Bbox::check(args[0]) failing; something about cross module?
    // set the clip rectangle
    // flipy

    Bbox* clipbox = static_cast<Bbox*>(o.ptr());
    double l = clipbox->ll_api()->x_api()->val() ;
    double b = clipbox->ll_api()->y_api()->val();
    double r = clipbox->ur_api()->x_api()->val() ;
    double t = clipbox->ur_api()->y_api()->val() ; ;
    theRasterizer->clip_box(l, height-t, r, height-b);
    rendererBase->clip_box((int)l, (int)(height-t), (int)r, (int)(height-b));
  }

}

/****************************/

int RendererAgg::intersectCheck(double yCoord, double x1, double y1, double x2, double y2, int* intersectPoint)
{
  /* Returns 0 if no intersection or 1 if yes */
  /* If yes, changes intersectPoint to the x coordinate of the point of intersection */
  if ((y1>=yCoord) != (y2>=yCoord)) {
    /* Don't need to check for y1==y2 because the above condition rejects it automatically */
    *intersectPoint = (int)( ( x1 * (y2 - yCoord) + x2 * (yCoord - y1) ) / (y2 - y1) + 0.5);
    return 1;
  }
  return 0;
}

int RendererAgg::inPolygon(int row, const double xs[4], const double ys[4], int col[4])
{
  int numIntersect = 0;
  int i;
  /* Determines the boundaries of the row of pixels that is in the polygon */
  /* A pixel (x, y) is in the polygon if its center (x+0.5, y+0.5) is */
  double ycoord = (double(row) + 0.5);
  for(i=0; i<=3; i++)
    numIntersect += intersectCheck(ycoord, xs[i], ys[i], xs[(i+1)%4], ys[(i+1)%4], col+numIntersect);

  /* reorder if necessary */
  if (numIntersect == 2 && col[0] > col[1]) std::swap(col[0],col[1]);
  if (numIntersect == 4) {
    // Inline bubble sort on array of size 4
    if (col[0] > col[1]) std::swap(col[0],col[1]);
    if (col[1] > col[2]) std::swap(col[1],col[2]);
    if (col[2] > col[3]) std::swap(col[2],col[3]);
    if (col[0] > col[1]) std::swap(col[0],col[1]);
    if (col[1] > col[2]) std::swap(col[1],col[2]);
    if (col[0] > col[1]) std::swap(col[0],col[1]);
  }
  // numIntersect must be 0, 2 or 4
  return numIntersect;
}

void RendererAgg::DrawQuadMesh(int meshWidth, int meshHeight, void* colors_void, const double xCoords[], const double yCoords[])
{
  PyArrayObject* colors = (PyArrayObject*)colors_void;

  /* draw each quadrilateral */
  //	agg::renderer_primitives<agg::renderer_base<agg::pixfmt_rgba32> > lineRen(*rendererBase);
  int i = 0;
  int j = 0;
  int k = 0;
  double xs[4];
  double ys[4];
  int col[4];
  int numCol;
  double ymin;
  int firstRow;
  double ymax;
  int lastRow;
  for(i=0; i < meshHeight; i++)
    {
      for(j=0; j < meshWidth; j++)
	{
	  //currTime = clock();
	  xs[0] = xCoords[(i * (meshWidth + 1)) + j];
	  ys[0] = yCoords[(i * (meshWidth + 1)) + j];
	  xs[1] = xCoords[(i * (meshWidth + 1)) + j+1];
	  ys[1] = yCoords[(i * (meshWidth + 1)) + j+1];
	  xs[3] = xCoords[((i+1) * (meshWidth + 1)) + j];
	  ys[3] = yCoords[((i+1) * (meshWidth + 1)) + j];
	  xs[2] = xCoords[((i+1) * (meshWidth + 1)) + j+1];
	  ys[2] = yCoords[((i+1) * (meshWidth + 1)) + j+1];
	  ymin = min(min(min(ys[0], ys[1]), ys[2]), ys[3]);
	  ymax = max(max(max(ys[0], ys[1]), ys[2]), ys[3]);
	  firstRow = (int)(ymin);
	  lastRow = (int)(ymax);
	  //timer1 += (clock() - currTime);
	  //currTime = clock();
	  //timer2 += (clock() - currTime);
	  //currTime = clock();
	  size_t color_index = (i * meshWidth) + j;
	  agg::rgba color(*(double*)PyArray_GETPTR2(colors, color_index, 0),
			  *(double*)PyArray_GETPTR2(colors, color_index, 1),
			  *(double*)PyArray_GETPTR2(colors, color_index, 2),
			  *(double*)PyArray_GETPTR2(colors, color_index, 3));

	  for(k = firstRow; k <= lastRow; k++)
	    {
	      numCol = inPolygon(k, xs, ys, col);

	      if (numCol >= 2) rendererBase->copy_hline(col[0], k, col[1] - 1, color);
	      if (numCol == 4) rendererBase->copy_hline(col[2], k, col[3] - 1, color);
	    }
	}
    }
  return;
}

void RendererAgg::DrawQuadMeshEdges(int meshWidth, int meshHeight, const double xCoords[], const double yCoords[])
{
  int i, j;
  agg::renderer_primitives<agg::renderer_base<agg::pixfmt_rgba32> > lineRen(*rendererBase);
  agg::rgba8 lc(0, 0, 0, 32);
  lineRen.line_color(lc);
  /* show the vertical edges */
  for(i=0; i <= meshWidth; i++)
    {
      lineRen.move_to((int)(256.0 * (xCoords[i])), (int)(256.0 * (yCoords[i])));
      for(j=1; j <= meshHeight; j++)
	lineRen.line_to((int)(256.0 *(xCoords[(j * (meshWidth + 1))+i])), (int)(256.0 * (yCoords[(j * (meshWidth + 1))+i])));
    }
  /* show the horizontal edges */
  for(i=0; i <= meshHeight; i++)
    {
      lineRen.move_to((int)(256.0 * (xCoords[i * (meshWidth + 1)])), (int)(256.0 * (yCoords[i * (meshWidth + 1)])));
      for(j=1; j <= meshWidth; j++)
	lineRen.line_to((int)(256.0 * (xCoords[(i * (meshWidth + 1))+j])), (int)(256.0 * (yCoords[(i * (meshWidth + 1))+j])));
    }
}

Py::Object
RendererAgg::draw_quad_mesh(const Py::Tuple& args){
  //printf("#1: %d\n", clock());
  Py::Object colorsi = args[2];
  Py::Object xCoordsi = args[3];
  Py::Object yCoordsi = args[4];
  int meshWidth = Py::Int(args[0]);
  int meshHeight = Py::Int(args[1]);
  int showedges = Py::Int(args[9]);
  PyArrayObject *colors = (PyArrayObject *) PyArray_ContiguousFromObject(colorsi.ptr(), PyArray_DOUBLE, 2, 2);
  PyArrayObject *xCoords = (PyArrayObject *) PyArray_ContiguousFromObject(xCoordsi.ptr(), PyArray_DOUBLE, 1, 1);
  PyArrayObject *yCoords = (PyArrayObject *) PyArray_ContiguousFromObject(yCoordsi.ptr(), PyArray_DOUBLE, 1, 1);
  /*****transformations****/
  /* do transformations */
  //todo: fix transformation check
  Transformation* transform = static_cast<Transformation*>(args[6].ptr());

  try {
    transform->eval_scalars();
  }
  catch(...) {
    throw Py::ValueError("Domain error on eval_scalars in RendererAgg::draw_quad_mesh");
  }

  set_clip_from_bbox(args[5]);
  // Does it make sense to support offsets in QuadMesh?
  // When would they be used?
  Py::SeqBase<Py::Object> offsets;
  Transformation* transOffset = NULL;
  bool usingOffsets = args[7].ptr() != Py_None;
  if (usingOffsets) {
    offsets = args[7];
    //todo: fix transformation check
    transOffset = static_cast<Transformation*>(args[8].ptr());
    try {
      transOffset->eval_scalars();
    }
    catch(...) {
      throw Py::ValueError("Domain error on transOffset eval_scalars in RendererAgg::draw_quad_mesh");
    }

  }
  size_t Noffsets;
  if(usingOffsets)
    Noffsets = offsets.length();
  else
    Noffsets = 0;
  size_t Nverts = xCoords->dimensions[0];
  /*  size_t N = (Noffsets>Nverts) ? Noffsets : Nverts; */

  std::pair<double, double> xyo, xy;

  //do non-offset transformations
  double* newXCoords = new double[Nverts];
  double* newYCoords = new double[Nverts];
  size_t k, q;
  transform->arrayOperator(Nverts, (const double *)xCoords->data,
                                    (const double *)yCoords->data,
                                    newXCoords, newYCoords);
  if(usingOffsets)
    {
      double* xOffsets = new double[Noffsets];
      double* yOffsets = new double[Noffsets];
      double* newXOffsets = new double[Noffsets];
      double* newYOffsets = new double[Noffsets];
      for(k=0; k < Noffsets; k++)
        {
          Py::SeqBase<Py::Object> pos = Py::SeqBase<Py::Object>(offsets[k]);
          xOffsets[k] = Py::Float(pos[0]);
          yOffsets[k] = Py::Float(pos[1]);
        }
      transOffset->arrayOperator(Noffsets, xOffsets, yOffsets, newXOffsets, newYOffsets);
      for(k=0; k < Nverts; k++)
        {
          newXCoords[k] += newXOffsets[k];
          newYCoords[k] += newYOffsets[k];
        }
      delete xOffsets;
      delete yOffsets;
      delete newXOffsets;
      delete newYOffsets;
    }

  for(q=0; q < Nverts; q++)
    {
      newYCoords[q] = height - newYCoords[q];
    }

  /**** End of transformations ****/

  DrawQuadMesh(meshWidth, meshHeight, colors, &(newXCoords[0]), &(newYCoords[0]));
  if(showedges)
    DrawQuadMeshEdges(meshWidth, meshHeight, &(newXCoords[0]), &(newYCoords[0]));
  Py_XDECREF(xCoords);
  Py_XDECREF(yCoords);
  Py_XDECREF(colors);
  delete newXCoords;
  delete newYCoords;
  //printf("#2: %d\n", clock());
  return Py::Object();
}

/****************************/
Py::Object
RendererAgg::draw_poly_collection(const Py::Tuple& args) {
  theRasterizer->reset_clipping();

  _VERBOSE("RendererAgg::draw_poly_collection");

  args.verify_length(9);


  Py::SeqBase<Py::Object> verts = args[0];

  //todo: fix transformation check
  Transformation* transform = static_cast<Transformation*>(args[1].ptr());

  try {
    transform->eval_scalars();
  }
  catch(...) {
    throw Py::ValueError("Domain error on eval_scalars in RendererAgg::draw_poly_collection");
  }


  set_clip_from_bbox(args[2]);

  Py::SeqBase<Py::Object> facecolors = args[3];
  Py::SeqBase<Py::Object> edgecolors = args[4];
  Py::SeqBase<Py::Object> linewidths = args[5];
  Py::SeqBase<Py::Object> antialiaseds = args[6];


  Py::SeqBase<Py::Object> offsets;
  Transformation* transOffset = NULL;
  bool usingOffsets = args[7].ptr() != Py_None;
  if (usingOffsets) {
    offsets = args[7];
    //todo: fix transformation check
    transOffset = static_cast<Transformation*>(args[8].ptr());
    try {
      transOffset->eval_scalars();
    }
    catch(...) {
      throw Py::ValueError("Domain error on transoffset eval_scalars in RendererAgg::draw_poly_collection");
    }

  }

  size_t Noffsets = offsets.length();
  size_t Nverts = verts.length();
  size_t Nface = facecolors.length();
  size_t Nedge = edgecolors.length();
  size_t Nlw = linewidths.length();
  size_t Naa = antialiaseds.length();

  size_t N = (Noffsets>Nverts) ? Noffsets : Nverts;

  std::pair<double, double> xyo, xy;
  Py::SeqBase<Py::Object> thisverts;
  size_t i, j;
  for (i=0; i<N; i++) {

    thisverts = verts[i % Nverts];

    if (usingOffsets) {
      Py::SeqBase<Py::Object> pos = Py::SeqBase<Py::Object>(offsets[i]);
      double xo = Py::Float(pos[0]);
      double yo = Py::Float(pos[1]);
      try {
	xyo = transOffset->operator()(xo, yo);
      }
      catch (...) {
	throw Py::ValueError("Domain error on transOffset->operator in draw_line_collection");
      }

    }

    size_t Nverts = thisverts.length();
    agg::path_storage path;

    Py::SeqBase<Py::Object>  thisvert;


    // dump the verts to double arrays so we can do more efficient
    // look aheads and behinds when doing snapto pixels
    double *xs = new double[Nverts];
    double *ys = new double[Nverts];
    for (j=0; j<Nverts; j++) {
      thisvert = thisverts[j];
      double x = Py::Float(thisvert[0]);
      double y = Py::Float(thisvert[1]);
      try {
	xy = transform->operator()(x, y);
      }
      catch(...) {
	delete [] xs;
	delete [] ys;
	throw Py::ValueError("Domain error on eval_scalars in RendererAgg::draw_poly_collection");
      }


      if (usingOffsets) {
	xy.first  += xyo.first;
	xy.second += xyo.second;
      }

      xy.second = height - xy.second;
      xs[j] = xy.first;
      ys[j] = xy.second;

    }

    for (j=0; j<Nverts; j++) {

      double x = xs[j];
      double y = ys[j];

      if (j==0) {
	if (xs[j] == xs[Nverts-1]) x = (int)xs[j] + 0.5;
	if (ys[j] == ys[Nverts-1]) y = (int)ys[j] + 0.5;
      }
      else if (j==Nverts-1) {
	if (xs[j] == xs[0]) x = (int)xs[j] + 0.5;
	if (ys[j] == ys[0]) y = (int)ys[j] + 0.5;
      }

      if (j < Nverts-1) {
	if (xs[j] == xs[j+1]) x = (int)xs[j] + 0.5;
	if (ys[j] == ys[j+1]) y = (int)ys[j] + 0.5;
      }
      if (j>0) {
	if (xs[j] == xs[j-1]) x = (int)xs[j] + 0.5;
	if (ys[j] == ys[j-1]) y = (int)ys[j] + 0.5;
      }

      if (j==0) path.move_to(x,y);
      else path.line_to(x,y);
    }

    path.close_polygon();
    int isaa = Py::Int(antialiaseds[i%Naa]);
    // get the facecolor and render
    Py::SeqBase<Py::Object>  rgba = Py::SeqBase<Py::Object>(facecolors[ i%Nface]);
    double r = Py::Float(rgba[0]);
    double g = Py::Float(rgba[1]);
    double b = Py::Float(rgba[2]);
    double a = Py::Float(rgba[3]);
    if (a>0) { //only render if alpha>0
      agg::rgba facecolor(r, g, b, a);

      theRasterizer->add_path(path);

      if (isaa) {
	rendererAA->color(facecolor);
	agg::render_scanlines(*theRasterizer, *slineP8, *rendererAA);
      }
      else {
	rendererBin->color(facecolor);
	agg::render_scanlines(*theRasterizer, *slineBin, *rendererBin);
      }
    } //renderer face

      // get the edgecolor and render
    rgba = Py::SeqBase<Py::Object>(edgecolors[ i%Nedge]);
    r = Py::Float(rgba[0]);
    g = Py::Float(rgba[1]);
    b = Py::Float(rgba[2]);
    a = Py::Float(rgba[3]);

    double lw = points_to_pixels ( Py::Float( linewidths[i%Nlw] ) );
    if ((a>0) && lw) { //only render if alpha>0 and linewidth !=0
      agg::rgba edgecolor(r, g, b, a);

      agg::conv_stroke<agg::path_storage> stroke(path);
      //stroke.line_cap(cap);
      //stroke.line_join(join);
      stroke.width(lw);
      theRasterizer->add_path(stroke);

      // render antialiased or not
      if ( isaa ) {
	rendererAA->color(edgecolor);
	agg::render_scanlines(*theRasterizer, *slineP8, *rendererAA);
      }
      else {
	rendererBin->color(edgecolor);
	agg::render_scanlines(*theRasterizer, *slineBin, *rendererBin);
      }
    } //rendered edge

    delete [] xs;
    delete [] ys;

  } // for every poly
  return Py::Object();
}

Py::Object
RendererAgg::draw_regpoly_collection(const Py::Tuple& args) {
  theRasterizer->reset_clipping();

  _VERBOSE("RendererAgg::draw_regpoly_collection");
  args.verify_length(9);


  set_clip_from_bbox(args[0]);

  Py::SeqBase<Py::Object> offsets = args[1];

  // this is throwing even though the instance is a Transformation!
  //if (!Transformation::check(args[2]))
  // throw Py::TypeError("RendererAgg::draw_regpoly_collection(clipbox, offsets, transOffset, verts, ...) expected a Transformation instance for transOffset");

  Transformation* transOffset = static_cast<Transformation*>(args[2].ptr());


  try {
    transOffset->eval_scalars();
  }
  catch(...) {
    throw Py::ValueError("Domain error on eval_scalars in RendererAgg::draw_regpoly_collection");
  }


  Py::SeqBase<Py::Object> verts = args[3];
  Py::SeqBase<Py::Object> sizes = args[4];
  Py::SeqBase<Py::Object> facecolors = args[5];
  Py::SeqBase<Py::Object> edgecolors = args[6];
  Py::SeqBase<Py::Object> linewidths = args[7];
  Py::SeqBase<Py::Object> antialiaseds = args[8];

  size_t Noffsets = offsets.length();
  size_t Nverts = verts.length();
  size_t Nsizes = sizes.length();
  size_t Nface = facecolors.length();
  size_t Nedge = edgecolors.length();
  size_t Nlw = linewidths.length();
  size_t Naa = antialiaseds.length();

  double thisx, thisy;

  // dump the x.y vertices into a double array for faster access
  double *xverts = new double[Nverts];
  double *yverts = new double[Nverts];
  Py::SeqBase<Py::Object> xy;
  size_t i, j;
  for (i=0; i<Nverts; i++) {
    xy = Py::SeqBase<Py::Object>(verts[i]);
    xverts[i] = Py::Float(xy[0]);
    yverts[i] = Py::Float(xy[1]);
  }

  std::pair<double, double> offsetPair;
  for (i=0; i<Noffsets; i++) {
    Py::SeqBase<Py::Object> pos = Py::SeqBase<Py::Object>(offsets[i]);
    double xo = Py::Float(pos[0]);
    double yo = Py::Float(pos[1]);
    try {
      offsetPair = transOffset->operator()(xo, yo);
    }
    catch(...) {
      delete [] xverts;
      delete [] yverts;
      throw Py::ValueError("Domain error on eval_scalars in RendererAgg::draw_regpoly_collection");
    }



    double scale = Py::Float(sizes[i%Nsizes]);


    agg::path_storage path;

    for (j=0; j<Nverts; j++) {
      thisx = scale*xverts[j] + offsetPair.first;
      thisy = scale*yverts[j] + offsetPair.second;
      thisy = height - thisy;
      if (j==0) path.move_to(thisx, thisy);
      else path.line_to(thisx, thisy);


    }
    path.close_polygon();
    int isaa = Py::Int(antialiaseds[i%Naa]);
    // get the facecolor and render
    Py::SeqBase<Py::Object> rgba = Py::SeqBase<Py::Object>(facecolors[ i%Nface]);
    double r = Py::Float(rgba[0]);
    double g = Py::Float(rgba[1]);
    double b = Py::Float(rgba[2]);
    double a = Py::Float(rgba[3]);
    if (a>0) { //only render if alpha>0
      agg::rgba facecolor(r, g, b, a);

      theRasterizer->add_path(path);

      if (isaa) {
	rendererAA->color(facecolor);
	agg::render_scanlines(*theRasterizer, *slineP8, *rendererAA);
      }
      else {
	rendererBin->color(facecolor);
	agg::render_scanlines(*theRasterizer, *slineBin, *rendererBin);
      }
    } //renderer face

      // get the edgecolor and render
    rgba = Py::SeqBase<Py::Object>(edgecolors[ i%Nedge]);
    r = Py::Float(rgba[0]);
    g = Py::Float(rgba[1]);
    b = Py::Float(rgba[2]);
    a = Py::Float(rgba[3]);
    double lw = points_to_pixels ( Py::Float( linewidths[i%Nlw] ) );
    if ((a>0) && lw) { //only render if alpha>0
      agg::rgba edgecolor(r, g, b, a);

      agg::conv_stroke<agg::path_storage> stroke(path);
      //stroke.line_cap(cap);
      //stroke.line_join(join);
      stroke.width(lw);
      theRasterizer->add_path(stroke);

      // render antialiased or not
      if ( isaa ) {
	rendererAA->color(edgecolor);
	agg::render_scanlines(*theRasterizer, *slineP8, *rendererAA);
      }
      else {
	rendererBin->color(edgecolor);
	agg::render_scanlines(*theRasterizer, *slineBin, *rendererBin);
      }
    } //rendered edge

  } // for every poly
  delete [] xverts;
  delete [] yverts;
  return Py::Object();
}

Py::Object
RendererAgg::draw_lines(const Py::Tuple& args) {

 _VERBOSE("RendererAgg::draw_lines");
  args.verify_length(4);

  Py::Object xo = args[1];
  Py::Object yo = args[2];

  PyArrayObject *xa = (PyArrayObject *) PyArray_ContiguousFromObject(xo.ptr(), PyArray_DOUBLE, 1, 1);

  if (xa==NULL)
    throw Py::TypeError("RendererAgg::draw_lines expected numerix array");


  PyArrayObject *ya = (PyArrayObject *) PyArray_ContiguousFromObject(yo.ptr(), PyArray_DOUBLE, 1, 1);

  if (ya==NULL)
    throw Py::TypeError("RendererAgg::draw_lines expected numerix array");


  size_t Nx = xa->dimensions[0];
  size_t Ny = ya->dimensions[0];

  if (Nx!=Ny)
    throw Py::ValueError(Printf("x and y must be equal length arrays; found %d and %d", Nx, Ny).str());

  // call gc with snapto==True if line len is 2 to fix grid line
  // problem
  bool snapto = false;
  if (Nx==2) {
    // disable subpiel rendering for len(2) horizontal or vertical
    // lines
    double x0 = *(double *)(xa->data + 0*xa->strides[0]);
    double x1 = *(double *)(xa->data + 1*xa->strides[0]);
    double y0 = *(double *)(ya->data + 0*ya->strides[0]);
    double y1 = *(double *)(ya->data + 1*ya->strides[0]);
    snapto = (x0==x1) || (y0==y1);

  }
  GCAgg gc = GCAgg(args[0], dpi, snapto);

  set_clipbox_rasterizer(gc.cliprect);
  //path_t transpath(path, xytrans);
  _process_alpha_mask(gc);

  Transformation* mpltransform = static_cast<Transformation*>(args[3].ptr());

  double a, b, c, d, tx, ty;
  try {
    mpltransform->affine_params_api(&a, &b, &c, &d, &tx, &ty);
  }
  catch(...) {
    throw Py::ValueError("Domain error on affine_params_api in RendererAgg::draw_lines");
  }

  agg::trans_affine xytrans = agg::trans_affine(a,b,c,d,tx,ty);


  agg::path_storage path;


  bool needNonlinear = mpltransform->need_nonlinear_api();

  double thisx(0.0), thisy(0.0);
  double origdx(0.0), origdy(0.0), origdNorm2(0);
  bool moveto = true;
  double heightd = height;

  double lastx(0), lasty(0);
  double lastWrittenx(0), lastWritteny(0);
  bool clipped = false;

  bool haveMin = false, lastMax = true;
  double dnorm2Min(0), dnorm2Max(0);
  double maxX(0), maxY(0), minX(0), minY(0);

  double totdx, totdy, totdot;
  double paradx, parady, paradNorm2;
  double perpdx, perpdy, perpdNorm2;

  int counter = 0;
  //idea: we can skip drawing many lines: lines < 1 pixel in length, lines
  //outside of the drawing area, and we can combine sequential parallel lines
  //into a single line instead of redrawing lines over the same points.
  //The loop below works a bit like a state machine, where what it does depends
  //on what it did in the last looping. To test whether sequential lines
  //are close to parallel, I calculate the distance moved perpendicular to the
  //last line. Once it gets too big, the lines cannot be combined.
  for (size_t i=0; i<Nx; i++) {

    thisx = *(double *)(xa->data + i*xa->strides[0]);
    thisy = *(double *)(ya->data + i*ya->strides[0]);

    if (needNonlinear)
      try {
        mpltransform->nonlinear_only_api(&thisx, &thisy);
      }
      catch (...) {
        moveto = true;
        continue;
      }
      if (MPL_isnan64(thisx) || MPL_isnan64(thisy)) {
        moveto = true;
        continue;
      }

    //use agg's transformer?
    xytrans.transform(&thisx, &thisy);
    thisy = heightd - thisy; //flipy

    if (snapto) {
      //disable subpixel rendering for horizontal or vertical lines of len=2
      //because it causes irregular line widths for grids and ticks
      thisx = (int)thisx + 0.5;
      thisy = (int)thisy + 0.5;
    }

    //if we are starting a new path segment, move to the first point + init
    if(moveto){
      path.move_to(thisx, thisy);
      lastx = thisx;
      lasty = thisy;
      origdNorm2 = 0; //resets the orig-vector variables (see if-statement below)
      moveto = false;
      continue;
    }

    //don't render line segments less that on pixel long!
    if (fabs(thisx-lastx) < 1.0 && fabs(thisy-lasty) < 1.0 ){
      continue; //don't update lastx this time!
    }

    //skip any lines that are outside the drawing area. Note: More lines
    //could be clipped, but a more involved calculation would be needed
    if( (thisx < 0      && lastx < 0     ) ||
        (thisx > width  && lastx > width ) ||
        (thisy < 0      && lasty < 0     ) ||
        (thisy > height && lasty > height) ){
      lastx = thisx;
      lasty = thisy;
      clipped = true;
      continue;
    }

    //if we have no orig vector, set it to this vector and continue.
    //this orig vector is the reference vector we will build up the line to
    if(origdNorm2 == 0){
      //if we clipped after the moveto but before we got here, redo the moveto
      if(clipped){
        path.move_to(lastx, lasty);
        clipped = false;
      }

      origdx = thisx - lastx;
      origdy = thisy - lasty;
      origdNorm2 = origdx*origdx + origdy*origdy;

      //set all the variables to reflect this new orig vecor
      dnorm2Max = origdNorm2;
      dnorm2Min = 0;
      haveMin = false;
      lastMax = true;
      maxX = thisx;
      maxY = thisy;
      minX = lastx;
      minY = lasty;

      lastWrittenx = lastx;
      lastWritteny = lasty;

      //set the last point seen
      lastx = thisx;
      lasty = thisy;
      continue;
    }

    //if got to here, then we have an orig vector and we just got
    //a vector in the sequence.

    //check that the perpendicular distance we have moved from the
    //last written point compared to the line we are building is not too
    //much. If o is the orig vector (we are building on), and v is the vector
    //from the last written point to the current point, then the perpendicular
    //vector is  p = v - (o.v)o,  and we normalize o  (by dividing the
    //second term by o.o).

    //get the v vector
    totdx = thisx - lastWrittenx;
    totdy = thisy - lastWritteny;
    totdot = origdx*totdx + origdy*totdy;

    //get the para vector ( = (o.v)o/(o.o) )
    paradx = totdot*origdx/origdNorm2;
    parady = totdot*origdy/origdNorm2;
    paradNorm2 = paradx*paradx + parady*parady;

    //get the perp vector ( = v - para )
    perpdx = totdx - paradx;
    perpdy = totdy - parady;
    perpdNorm2 = perpdx*perpdx + perpdy*perpdy;

    //if the perp vector is less than some number of (squared) pixels in size,
    //then merge the current vector
    if(perpdNorm2 < 0.25 ){
      //check if the current vector is parallel or
      //anti-parallel to the orig vector. If it is parallel, test
      //if it is the longest of the vectors we are merging in that direction.
      //If anti-p, test if it is the longest in the opposite direction (the
      //min of our final line)

      lastMax = false;
      if(totdot >= 0){
        if(paradNorm2 > dnorm2Max){
          lastMax = true;
          dnorm2Max = paradNorm2;
          maxX = lastWrittenx + paradx;
          maxY = lastWritteny + parady;
        }
      }
      else{

        haveMin = true;
        if(paradNorm2 > dnorm2Min){
          dnorm2Min = paradNorm2;
          minX = lastWrittenx + paradx;
          minY = lastWritteny + parady;
        }
      }

      lastx = thisx;
      lasty = thisy;
      continue;
    }

    //if we get here, then this vector was not similar enough to the line
    //we are building, so we need to draw that line and start the next one.

    //if the line needs to extend in the opposite direction from the direction
    //we are drawing in, move back to we start drawing from back there.
    if(haveMin){
      path.line_to(minX, minY); //would be move_to if not for artifacts
    }

    path.line_to(maxX, maxY);

    //if we clipped some segments between this line and the next line
    //we are starting, we also need to move to the last point.
    if(clipped){
      path.move_to(lastx, lasty);
    }
    else if(!lastMax){
    	//if the last line was not the longest line, then move back to the end
      //point of the last line in the sequence. Only do this if not clipped,
      //since in that case lastx,lasty is not part of the line just drawn.
      path.line_to(lastx, lasty); //would be move_to if not for artifacts
    }

    //std::cout << "draw lines (" << lastx << ", " << lasty << ")" << std::endl;

    //now reset all the variables to get ready for the next line

    origdx = thisx - lastx;
    origdy = thisy - lasty;
    origdNorm2 = origdx*origdx + origdy*origdy;

    dnorm2Max = origdNorm2;
    dnorm2Min = 0;
    haveMin = false;
    lastMax = true;
    maxX = thisx;
    maxY = thisy;
    minX = lastx;
    minY = lasty;

    lastWrittenx = lastx;
    lastWritteny = lasty;

    clipped = false;

    lastx = thisx;
    lasty = thisy;

    counter++;
  }

  //draw the last line, which is usually not drawn in the loop
  if(origdNorm2 != 0){
    if(haveMin){
      path.line_to(minX, minY); //would be move_to if not for artifacts
    }

    path.line_to(maxX, maxY);
  }

  //std::cout << "drew " << counter+1 << " lines" << std::endl;

  Py_XDECREF(xa);
  Py_XDECREF(ya);

  //typedef agg::conv_transform<agg::path_storage, agg::trans_affine> path_t;
  //path_t transpath(path, xytrans);
  _VERBOSE("RendererAgg::draw_lines rendering lines path");
  _render_lines_path(path, gc);

  _VERBOSE("RendererAgg::draw_lines DONE");
  return Py::Object();

}

bool
RendererAgg::_process_alpha_mask(const GCAgg& gc)
  //if gc has a clippath set, process the alpha mask and return True,
  //else return False
{
  if (gc.clippath==NULL) {
    return false;
  }
  if (0 &(gc.clippath==lastclippath)) {
    //std::cout << "seen it" << std::endl;
    return true;
  }
  rendererBaseAlphaMask->clear(agg::gray8(0, 0));
  gc.clippath->rewind(0);
  theRasterizer->add_path(*(gc.clippath));
  rendererAlphaMask->color(agg::gray8(255,255));
  agg::render_scanlines(*theRasterizer, *scanlineAlphaMask, *rendererAlphaMask);
  lastclippath = gc.clippath;
  return true;
}

template<class PathSource>
void
RendererAgg::_render_lines_path(PathSource &path, const GCAgg& gc) {
  _VERBOSE("RendererAgg::_render_lines_path");
  typedef PathSource path_t;
  //typedef agg::conv_transform<agg::path_storage, agg::trans_affine> path_t;
  typedef agg::conv_stroke<path_t> stroke_t;
  typedef agg::conv_dash<path_t> dash_t;

  bool isclippath(gc.clippath!=NULL);

  if (gc.dasha==NULL ) { //no dashes
    stroke_t stroke(path);
    stroke.width(gc.linewidth);
    stroke.line_cap(gc.cap);
    stroke.line_join(gc.join);
    theRasterizer->add_path(stroke);
  }
  else {
    dash_t dash(path);

    //todo: dash.dash_start(gc.dashOffset);
    for (size_t i=0; i<gc.Ndash/2; i+=1)
      dash.add_dash(gc.dasha[2*i], gc.dasha[2*i+1]);

    agg::conv_stroke<dash_t> stroke(dash);
    stroke.line_cap(gc.cap);
    stroke.line_join(gc.join);
    stroke.width(gc.linewidth);
    theRasterizer->add_path(stroke); //boyle freeze is herre
  }


  if ( gc.isaa ) {
    if (isclippath) {
      typedef agg::pixfmt_amask_adaptor<pixfmt, alpha_mask_type> pixfmt_amask_type;
      typedef agg::renderer_base<pixfmt_amask_type>              amask_ren_type;
      pixfmt_amask_type pfa(*pixFmt, *alphaMask);
      amask_ren_type r(pfa);
      typedef agg::renderer_scanline_aa_solid<amask_ren_type> renderer_type;
      renderer_type ren(r);
      ren.color(gc.color);
      //std::cout << "render clippath" << std::endl;

      agg::render_scanlines(*theRasterizer, *slineP8, ren);
    }
    else {
      rendererAA->color(gc.color);
      agg::render_scanlines(*theRasterizer, *slineP8, *rendererAA);
    }
  }
  else {
    if (isclippath) {
      typedef agg::pixfmt_amask_adaptor<pixfmt, alpha_mask_type> pixfmt_amask_type;
      typedef agg::renderer_base<pixfmt_amask_type>              amask_ren_type;
      pixfmt_amask_type pfa(*pixFmt, *alphaMask);
      amask_ren_type r(pfa);
      typedef agg::renderer_scanline_bin_solid<amask_ren_type> renderer_type;
      renderer_type ren(r);
      ren.color(gc.color);
      agg::render_scanlines(*theRasterizer, *slineP8, ren);
    }
    else{
      rendererBin->color(gc.color);
      agg::render_scanlines(*theRasterizer, *slineBin, *rendererBin);
    }
  }
}

Py::Object
RendererAgg::draw_markers(const Py::Tuple& args) {
  theRasterizer->reset_clipping();

  _VERBOSE("RendererAgg::_draw_markers_cache");
  args.verify_length(6);

  _VERBOSE("RendererAgg::_draw_markers_cache setting gc");
  GCAgg gc = GCAgg(args[0], dpi);


  agg::path_storage *ppath;

  swig_type_info * descr = SWIG_TypeQuery("agg::path_storage *");
  assert(descr);
  if (SWIG_ConvertPtr(args[1].ptr(),(void **)(&ppath), descr, 0) == -1) {
    throw Py::TypeError("Could not convert path_storage");
  }
  facepair_t face = _get_rgba_face(args[2], gc.alpha);

  Py::Object xo = args[3];
  Py::Object yo = args[4];

  PyArrayObject *xa = (PyArrayObject *) PyArray_ContiguousFromObject(xo.ptr(), PyArray_DOUBLE, 1, 1);

  if (xa==NULL)
    throw Py::TypeError("RendererAgg::_draw_markers_cache expected numerix array");


  PyArrayObject *ya = (PyArrayObject *) PyArray_ContiguousFromObject(yo.ptr(), PyArray_DOUBLE, 1, 1);

  if (ya==NULL)
    throw Py::TypeError("RendererAgg::_draw_markers_cache expected numerix array");

  Transformation* mpltransform = static_cast<Transformation*>(args[5].ptr());

  double a, b, c, d, tx, ty;
  try {
    mpltransform->affine_params_api(&a, &b, &c, &d, &tx, &ty);
  }
  catch(...) {
    throw Py::ValueError("Domain error on affine_params_api in RendererAgg::_draw_markers_cache");
  }

  agg::trans_affine xytrans = agg::trans_affine(a,b,c,d,tx,ty);

  size_t Nx = xa->dimensions[0];
  size_t Ny = ya->dimensions[0];

  if (Nx!=Ny)
    throw Py::ValueError(Printf("x and y must be equal length arrays; found %d and %d", Nx, Ny).str());


  double heightd = double(height);


  ppath->rewind(0);
  ppath->flip_y(0,0);
  typedef agg::conv_curve<agg::path_storage> curve_t;
  curve_t curve(*ppath);

  //maxim's suggestions for cached scanlines
  agg::scanline_storage_aa8 scanlines;
  theRasterizer->reset();

  agg::int8u* fillCache = NULL;
  unsigned fillSize = 0;
  if (face.first) {
    theRasterizer->add_path(curve);
    agg::render_scanlines(*theRasterizer, *slineP8, scanlines);
    fillSize = scanlines.byte_size();
    fillCache = new agg::int8u[fillSize]; // or any container
    scanlines.serialize(fillCache);
  }

  agg::conv_stroke<curve_t> stroke(curve);
  stroke.width(gc.linewidth);
  stroke.line_cap(gc.cap);
  stroke.line_join(gc.join);
  theRasterizer->reset();
  theRasterizer->add_path(stroke);
  agg::render_scanlines(*theRasterizer, *slineP8, scanlines);
  unsigned strokeSize = scanlines.byte_size();
  agg::int8u* strokeCache = new agg::int8u[strokeSize]; // or any container
  scanlines.serialize(strokeCache);

  theRasterizer->reset_clipping();


  if (gc.cliprect==NULL) {
    rendererBase->reset_clipping(true);
  }
  else {
    int l = (int)(gc.cliprect[0]) ;
    int b = (int)(gc.cliprect[1]) ;
    int w = (int)(gc.cliprect[2]) ;
    int h = (int)(gc.cliprect[3]) ;
    rendererBase->clip_box(l, height-(b+h),l+w, height-b);
  }


  double thisx, thisy;
  for (size_t i=0; i<Nx; i++) {
    thisx = *(double *)(xa->data + i*xa->strides[0]);
    thisy = *(double *)(ya->data + i*ya->strides[0]);

    if (mpltransform->need_nonlinear_api())
      try {
	mpltransform->nonlinear_only_api(&thisx, &thisy);
      }
      catch(...) {
	continue;
      }

    xytrans.transform(&thisx, &thisy);

    thisy = heightd - thisy;  //flipy

    thisx = (int)thisx + 0.5;
    thisy = (int)thisy + 0.5;
    if (thisx<0) continue;
    if (thisy<0) continue;
    if (thisx>width) continue;
    if (thisy>height) continue;

    agg::serialized_scanlines_adaptor_aa8 sa;
    agg::serialized_scanlines_adaptor_aa8::embedded_scanline sl;

    if (face.first) {
      //render the fill
      sa.init(fillCache, fillSize, thisx, thisy);
      rendererAA->color(face.second);
      agg::render_scanlines(sa, sl, *rendererAA);
    }

    //render the stroke
    sa.init(strokeCache, strokeSize, thisx, thisy);
    rendererAA->color(gc.color);
    agg::render_scanlines(sa, sl, *rendererAA);

  } //for each marker

  Py_XDECREF(xa);
  Py_XDECREF(ya);

  if (face.first)
    delete [] fillCache;
  delete [] strokeCache;

  _VERBOSE("RendererAgg::_draw_markers_cache done");
  return Py::Object();

}




Py::Object
RendererAgg::draw_path(const Py::Tuple& args) {
  //draw_path(gc, rgbFace, path, transform)
  theRasterizer->reset_clipping();

  _VERBOSE("RendererAgg::draw_path");
  args.verify_length(3);

  GCAgg gc = GCAgg(args[0], dpi);
  facepair_t face = _get_rgba_face(args[1], gc.alpha);

  agg::path_storage *path;
  swig_type_info * descr = SWIG_TypeQuery("agg::path_storage *");
  assert(descr);
  if (SWIG_ConvertPtr(args[2].ptr(),(void **)(&path), descr, 0) == -1)
    throw Py::TypeError("Could not convert path_storage");



  double heightd = double(height);
  agg::path_storage tpath;  // the flipped path
  size_t Nx = path->total_vertices();
  double x, y;
  unsigned cmd;
  bool curvy = false;
  for (size_t i=0; i<Nx; i++) {

    if (cmd==agg::path_cmd_curve3 || cmd==agg::path_cmd_curve4) curvy=true;
    cmd = path->vertex(i, &x, &y);
    tpath.add_vertex(x, heightd-y, cmd);
  }
  set_clipbox_rasterizer(gc.cliprect);
  _fill_and_stroke(tpath, gc, face, curvy);
  return Py::Object();

}

/**
 * This is a custom span generator that converts spans in the
 * 8-bit inverted greyscale font buffer to rgba that agg can use.
 */
template<
  class ColorT,
  class ChildGenerator>
class font_to_rgba :
  public agg::span_generator<ColorT,
			     agg::span_allocator<ColorT> >
{
public:
  typedef ChildGenerator child_type;
  typedef ColorT color_type;
  typedef agg::span_allocator<color_type> allocator_type;
  typedef agg::span_generator<
    ColorT,
    agg::span_allocator<ColorT> > base_type;

private:
  child_type* _gen;
  allocator_type _alloc;
  color_type _color;

public:
  font_to_rgba(child_type* gen, color_type color) :
    base_type(_alloc),
    _gen(gen),
    _color(color) {
  }

  color_type* generate(int x, int y, unsigned len)
  {
    color_type* dst = base_type::allocator().span();

    typename child_type::color_type* src = _gen->generate(x, y, len);

    do {
      *dst = _color;
      dst->a = ((unsigned int)_color.a * (unsigned int)src->v) >> 8;
      ++src;
      ++dst;
    } while (--len);

    return base_type::allocator().span();
  }

  void prepare(unsigned max_span_len)
  {
    _alloc.allocate(max_span_len);
    _gen->prepare(max_span_len);
  }

};

Py::Object
RendererAgg::draw_text_image(const Py::Tuple& args) {
  _VERBOSE("RendererAgg::draw_text");

  typedef agg::span_interpolator_linear<> interpolator_type;
  typedef agg::span_image_filter_gray<agg::gray8, interpolator_type>
    image_span_gen_type;
  typedef font_to_rgba<pixfmt::color_type, image_span_gen_type>
    span_gen_type;
  typedef agg::renderer_scanline_aa<renderer_base, span_gen_type>
    renderer_type;

  args.verify_length(5);

  const unsigned char* buffer = NULL;
  int width, height;
  Py::Object image_obj = args[0];
  PyArrayObject* image_array = NULL;
  if (PyArray_Check(image_obj.ptr())) {
    image_array = (PyArrayObject*)PyArray_FromObject(image_obj.ptr(), PyArray_UBYTE, 2, 2);
    if (!image_array)
      throw Py::ValueError("First argument to draw_text_image must be a FT2Font.Image object or a Nx2 uint8 numpy array.");
    buffer = (unsigned char *)PyArray_DATA(image_array);
    width = PyArray_DIM(image_array, 1);
    height = PyArray_DIM(image_array, 0);
  } else {
    FT2Image *image = static_cast<FT2Image*>(args[0].ptr());
    if (!image->get_buffer())
      throw Py::ValueError("First argument to draw_text_image must be a FT2Font.Image object or a Nx2 uint8 numpy array.");
    buffer = image->get_buffer();
    width = image->get_width();
    height = image->get_height();
  }

  int x(0),y(0);
  try {
    x = Py::Int( args[1] );
    y = Py::Int( args[2] );
  }
  catch (Py::TypeError) {
    //x,y out of range; todo issue warning?
    if (image_array)
      Py_XDECREF(image_array);
    return Py::Object();
  }

  double angle = Py::Float( args[3] );

  GCAgg gc = GCAgg(args[4], dpi);

  set_clipbox_rasterizer(gc.cliprect);

  agg::rendering_buffer srcbuf((agg::int8u*)buffer, width, height, width);
  agg::pixfmt_gray8 pixf_img(srcbuf);

  agg::trans_affine mtx;
  mtx *= agg::trans_affine_translation(0, -(int)height);
  mtx *= agg::trans_affine_rotation(-angle * agg::pi / 180.0);
  mtx *= agg::trans_affine_translation(x, y);

  agg::path_storage rect;
  rect.move_to(0, 0);
  rect.line_to(width, 0);
  rect.line_to(width, height);
  rect.line_to(0, height);
  rect.line_to(0, 0);
  agg::conv_transform<agg::path_storage> rect2(rect, mtx);

  agg::trans_affine inv_mtx(mtx);
  inv_mtx.invert();

  agg::image_filter_lut filter;
  filter.calculate(agg::image_filter_spline36());
  interpolator_type interpolator(inv_mtx);
  agg::span_allocator<agg::gray8> gray_span_allocator;
  image_span_gen_type image_span_generator(gray_span_allocator,
					   srcbuf, 0, interpolator, filter);

  span_gen_type output_span_generator(&image_span_generator, gc.color);
  renderer_type ri(*rendererBase, output_span_generator);
  //agg::rasterizer_scanline_aa<> rasterizer;
  //agg::scanline_p8 scanline;
  //rasterizer.add_path(rect2);
  //agg::render_scanlines(rasterizer, scanline, ri);


  theRasterizer->add_path(rect2);
  agg::render_scanlines(*theRasterizer, *slineP8, ri);

  if (image_array)
    Py_XDECREF(image_array);

  return Py::Object();
}


Py::Object
RendererAgg::draw_image(const Py::Tuple& args) {
  _VERBOSE("RendererAgg::draw_image");
  args.verify_length(4);

  float x = Py::Float(args[0]);
  float y = Py::Float(args[1]);
  Image *image = static_cast<Image*>(args[2].ptr());

  set_clip_from_bbox(args[3]);

  pixfmt pixf(*(image->rbufOut));


  Py::Tuple empty;
  image->flipud_out(empty);
  rendererBase->blend_from(pixf, 0, (int)x, (int)(height-(y+image->rowsOut)));
  image->flipud_out(empty);


  return Py::Object();

}


Py::Object
RendererAgg::write_rgba(const Py::Tuple& args) {
  _VERBOSE("RendererAgg::write_rgba");

  args.verify_length(1);
  std::string fname = Py::String( args[0]);

  std::ofstream of2( fname.c_str(), std::ios::binary|std::ios::out);
  for (size_t i=0; i<NUMBYTES; i++) {
    of2.write((char*)&(pixBuffer[i]), sizeof(char));
  }
  return Py::Object();

}

static void write_png_data(png_structp png_ptr, png_bytep data, png_size_t length) {
  PyObject* py_file_obj = (PyObject*)png_get_io_ptr(png_ptr);
  PyObject* write_method = PyObject_GetAttrString(py_file_obj, "write");
  PyObject* result = NULL;
  if (write_method)
    result = PyObject_CallFunction(write_method, "s#", data, length);
  Py_XDECREF(write_method);
  Py_XDECREF(result);
}

static void flush_png_data(png_structp png_ptr) {
  PyObject* py_file_obj = (PyObject*)png_get_io_ptr(png_ptr);
  PyObject* flush_method = PyObject_GetAttrString(py_file_obj, "flush");
  PyObject* result = NULL;
  if (flush_method)
    result = PyObject_CallFunction(flush_method, "");
  Py_XDECREF(flush_method);
  Py_XDECREF(result);
}

// this code is heavily adapted from the paint license, which is in
// the file paint.license (BSD compatible) included in this
// distribution.  TODO, add license file to MANIFEST.in and CVS
Py::Object
RendererAgg::write_png(const Py::Tuple& args)
{
  _VERBOSE("RendererAgg::write_png");

  args.verify_length(1, 2);

  FILE *fp = NULL;
  Py::Object py_fileobj = Py::Object(args[0]);
  if (py_fileobj.isString()) {
    std::string fileName = Py::String(py_fileobj);
    const char *file_name = fileName.c_str();
    if ((fp = fopen(file_name, "wb")) == NULL)
      throw Py::RuntimeError( Printf("Could not open file %s", file_name).str() );
  }
  else {
    PyObject* write_method = PyObject_GetAttrString(py_fileobj.ptr(), "write");
    if (!(write_method && PyCallable_Check(write_method))) {
      Py_XDECREF(write_method);
      throw Py::TypeError("Object does not appear to be a path or a Python file-like object");
    }
    Py_XDECREF(write_method);
  }

  png_bytep *row_pointers = NULL;
  png_structp png_ptr = NULL;
  png_infop info_ptr = NULL;

  try {
    struct        png_color_8_struct sig_bit;
    png_uint_32 row;

    row_pointers = new png_bytep[height];
    for (row = 0; row < height; ++row) {
      row_pointers[row] = pixBuffer + row * width * 4;
    }

    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png_ptr == NULL) {
      throw Py::RuntimeError("Could not create write struct");
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL) {
      throw Py::RuntimeError("Could not create info struct");
    }

    if (setjmp(png_ptr->jmpbuf)) {
      throw Py::RuntimeError("Error building image");
    }

    if (fp) {
      png_init_io(png_ptr, fp);
    } else {
      png_set_write_fn(png_ptr, (void*)py_fileobj.ptr(),
		       &write_png_data, &flush_png_data);
    }
    png_set_IHDR(png_ptr, info_ptr,
		 width, height, 8,
		 PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE,
		 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    // Save the dpi of the image in the file
    if (args.size() == 2) {
      double dpi = Py::Float(args[1]);
      size_t dots_per_meter = (size_t)(dpi / (2.54 / 100.0));
      png_set_pHYs(png_ptr, info_ptr, dots_per_meter, dots_per_meter, PNG_RESOLUTION_METER);
    }

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

    /* Changed calls to png_destroy_write_struct to follow
       http://www.libpng.org/pub/png/libpng-manual.txt.
       This ensures the info_ptr memory is released.
    */

  } catch (...) {
      if (fp) fclose(fp);
      delete [] row_pointers;
      if (png_ptr && info_ptr) png_destroy_write_struct(&png_ptr, &info_ptr);
      throw;
  }

  png_destroy_write_struct(&png_ptr, &info_ptr);
  delete [] row_pointers;
  if (fp) fclose(fp);

  return Py::Object();
}


Py::Object
RendererAgg::tostring_rgb(const Py::Tuple& args) {
  //"Return the rendered buffer as an RGB string";

  _VERBOSE("RendererAgg::tostring_rgb");

  args.verify_length(0);
  int row_len = width*3;
  unsigned char* buf_tmp = new unsigned char[row_len * height];
  if (buf_tmp ==NULL) {
    //todo: also handle allocation throw
    throw Py::MemoryError("RendererAgg::tostring_rgb could not allocate memory");
  }
  agg::rendering_buffer renderingBufferTmp;
  renderingBufferTmp.attach(buf_tmp,
			    width,
			    height,
			    row_len);

  agg::color_conv(&renderingBufferTmp, renderingBuffer, agg::color_conv_rgba32_to_rgb24());


  //todo: how to do this with native CXX
  PyObject* o = Py_BuildValue("s#",
			      buf_tmp,
			      row_len * height);
  delete [] buf_tmp;
  return Py::asObject(o);
}


Py::Object
RendererAgg::tostring_argb(const Py::Tuple& args) {
  //"Return the rendered buffer as an RGB string";

  _VERBOSE("RendererAgg::tostring_argb");

  args.verify_length(0);
  int row_len = width*4;
  unsigned char* buf_tmp = new unsigned char[row_len * height];
  if (buf_tmp ==NULL) {
    //todo: also handle allocation throw
    throw Py::MemoryError("RendererAgg::tostring_argb could not allocate memory");
  }
  agg::rendering_buffer renderingBufferTmp;
  renderingBufferTmp.attach(buf_tmp,
			    width,
			    height,
			    row_len);

  agg::color_conv(&renderingBufferTmp, renderingBuffer, agg::color_conv_rgba32_to_argb32());


  //todo: how to do this with native CXX
  PyObject* o = Py_BuildValue("s#",
			      buf_tmp,
			      row_len * height);
  delete [] buf_tmp;
  return Py::asObject(o);
}

Py::Object
RendererAgg::tostring_bgra(const Py::Tuple& args) {
  //"Return the rendered buffer as an RGB string";

  _VERBOSE("RendererAgg::tostring_bgra");

  args.verify_length(0);
  int row_len = width*4;
  unsigned char* buf_tmp = new unsigned char[row_len * height];
  if (buf_tmp ==NULL) {
    //todo: also handle allocation throw
    throw Py::MemoryError("RendererAgg::tostring_bgra could not allocate memory");
  }
  agg::rendering_buffer renderingBufferTmp;
  renderingBufferTmp.attach(buf_tmp,
			    width,
			    height,
			    row_len);

  agg::color_conv(&renderingBufferTmp, renderingBuffer, agg::color_conv_rgba32_to_bgra32());


  //todo: how to do this with native CXX
  PyObject* o = Py_BuildValue("s#",
			      buf_tmp,
			      row_len * height);
  delete [] buf_tmp;
  return Py::asObject(o);
}

Py::Object
RendererAgg::buffer_rgba(const Py::Tuple& args) {
  //"expose the rendered buffer as Python buffer object, starting from postion x,y";

  _VERBOSE("RendererAgg::buffer_rgba");

  args.verify_length(2);
  int startw = Py::Int(args[0]);
  int starth = Py::Int(args[1]);
  int row_len = width*4;
  int start=row_len*starth+startw*4;
  return Py::asObject(PyBuffer_FromMemory( pixBuffer+start, row_len*height-start));
}



Py::Object
RendererAgg::clear(const Py::Tuple& args) {
  //"clear the rendered buffer";

  _VERBOSE("RendererAgg::clear");

  args.verify_length(0);
  rendererBase->clear(agg::rgba(1, 1, 1, 0));

  return Py::Object();
}


agg::rgba
RendererAgg::rgb_to_color(const Py::SeqBase<Py::Object>& rgb, double alpha) {
  _VERBOSE("RendererAgg::rgb_to_color");

  double r = Py::Float(rgb[0]);
  double g = Py::Float(rgb[1]);
  double b = Py::Float(rgb[2]);
  return agg::rgba(r, g, b, alpha);

}


double
RendererAgg::points_to_pixels_snapto(const Py::Object& points) {
  // convert a value in points to pixels depending on renderer dpi and
  // screen pixels per inch
  // snap return pixels to grid
  _VERBOSE("RendererAgg::points_to_pixels_snapto");
  double p = Py::Float( points ) ;
  //return (int)(p*PIXELS_PER_INCH/72.0*dpi/72.0)+0.5;
  return (int)(p*dpi/72.0)+0.5;


}

double
RendererAgg::points_to_pixels( const Py::Object& points) {
  _VERBOSE("RendererAgg::points_to_pixels");
  double p = Py::Float( points ) ;
  //return p * PIXELS_PER_INCH/72.0*dpi/72.0;
  return p * dpi/72.0;
}


RendererAgg::~RendererAgg() {

  _VERBOSE("RendererAgg::~RendererAgg");


  delete slineP8;
  delete slineBin;
  delete theRasterizer;
  delete rendererAA;
  delete rendererBin;
  delete rendererBase;
  delete pixFmt;
  delete renderingBuffer;

  delete alphaMask;
  delete alphaMaskRenderingBuffer;
  delete [] alphaBuffer;
  delete [] pixBuffer;
  delete pixfmtAlphaMask;
  delete rendererBaseAlphaMask;
  delete rendererAlphaMask;
  delete scanlineAlphaMask;

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


void BufferRegion::init_type() {
  behaviors().name("BufferRegion");
  behaviors().doc("A wrapper to pass agg buffer objects to and from the python level");

  add_varargs_method("to_string", &BufferRegion::to_string,
		     "to_string()");

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
  add_varargs_method("draw_line_collection",
		     &RendererAgg::draw_line_collection,
		     "draw_line_collection(segments, trans, clipbox, colors, linewidths, antialiaseds)\n");
  add_varargs_method("draw_poly_collection",
		     &RendererAgg::draw_poly_collection,
		     "draw_poly_collection()\n");
  add_varargs_method("draw_regpoly_collection",
		     &RendererAgg::draw_regpoly_collection,
		     "draw_regpoly_collection()\n");
  add_varargs_method("draw_quad_mesh",
		     &RendererAgg::draw_quad_mesh,
		     "draw_quad_mesh()\n");
  add_varargs_method("draw_lines", &RendererAgg::draw_lines,
		     "draw_lines(gc, x, y,)\n");
  add_varargs_method("draw_markers", &RendererAgg::draw_markers,
		     "draw_markers(gc, path, x, y)\n");
  add_varargs_method("draw_path", &RendererAgg::draw_path,
		     "draw_path(gc, rgbFace, path, transform)\n");
  add_varargs_method("draw_text_image", &RendererAgg::draw_text_image,
		     "draw_text_image(font_image, x, y, r, g, b, a)\n");
  add_varargs_method("draw_image", &RendererAgg::draw_image,
		     "draw_image(x, y, im)");
  add_varargs_method("write_rgba", &RendererAgg::write_rgba,
		     "write_rgba(fname)");
  add_varargs_method("write_png", &RendererAgg::write_png,
		     "write_png(fname, dpi=None)");
  add_varargs_method("tostring_rgb", &RendererAgg::tostring_rgb,
		     "s = tostring_rgb()");
  add_varargs_method("tostring_argb", &RendererAgg::tostring_argb,
		     "s = tostring_argb()");
  add_varargs_method("tostring_bgra", &RendererAgg::tostring_bgra,
		     "s = tostring_bgra()");
  add_varargs_method("buffer_rgba", &RendererAgg::buffer_rgba,
		     "buffer = buffer_rgba()");
  add_varargs_method("clear", &RendererAgg::clear,
		     "clear()");
  add_varargs_method("copy_from_bbox", &RendererAgg::copy_from_bbox,
		     "copy_from_bbox(bbox)");

  add_varargs_method("restore_region", &RendererAgg::restore_region,
		     "restore_region(region)");


}


extern "C"
DL_EXPORT(void)
  init_backend_agg(void)
{
  //static _backend_agg_module* _backend_agg = new _backend_agg_module;

  _VERBOSE("init_backend_agg");

  import_array();

  static _backend_agg_module* _backend_agg = NULL;
  _backend_agg = new _backend_agg_module;

};
