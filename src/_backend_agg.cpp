/* A rewrite of _backend_agg using PyCXX to handle ref counting, etc..
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <png.h>


#include "agg_conv_transform.h"
#include "agg_conv_curve.h"
#include "agg_scanline_storage_aa.h"
#include "agg_scanline_storage_bin.h"
#include "util/agg_color_conv_rgb8.h"

#include "ft2font.h"
#include "_image.h"
#include "_backend_agg.h"
#include "_transforms.h"
#include "mplutils.h"

#include "swig_runtime.h"


#ifdef NUMARRAY
#include "numarray/arrayobject.h"
#else
#ifdef NUMERIC
#include "Numeric/arrayobject.h"
#else
#include "scipy/arrayobject.h"
#endif
#endif

/* ------------ RendererAgg methods ------------- */


GCAgg::GCAgg(const Py::Object &gc, double dpi) :
  dpi(dpi), isaa(true), linewidth(1.0), alpha(1.0), cliprect(NULL),
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

  for (size_t i=0; i<Ndash; i++)
    dasha[i] = points_to_pixels(dashSeq[i]);

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
  _VERBOSE("RendererAgg::RendererAgg");
  unsigned stride(width*4);


  pixBuffer = new agg::int8u[NUMBYTES];
  cacheBuffer = NULL;


  renderingBuffer = new agg::rendering_buffer;
  renderingBuffer->attach(pixBuffer, width, height, stride);
  slineP8 = new scanline_p8;
  slineBin = new scanline_bin;


  pixFmt = new pixfmt(*renderingBuffer);
  rendererBase = new renderer_base(*pixFmt);
  rendererBase->clear(agg::rgba(1, 1, 1, 0));

  rendererAA = new renderer_aa(*rendererBase);
  rendererBin = new renderer_bin(*rendererBase);
  theRasterizer = new rasterizer();

};



void
RendererAgg::set_clipbox_rasterizer( double *cliprect) {
  //set the clip rectangle from the gc

  _VERBOSE("RendererAgg::set_clipbox_rasterizer");

  if (cliprect==NULL) {
    theRasterizer->reset_clipping();
    rendererBase->reset_clipping(true);
  }
  else {

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

  if (face.first) {
    rendererAA->color(face.second);
    if (curvy) {
      curve_t curve(path);
      theRasterizer->add_path(curve);
    }
    else
      theRasterizer->add_path(path);
    agg::render_scanlines(*theRasterizer, *slineP8, *rendererAA);
  }

  //now stroke the edge
  if (curvy) {
    curve_t curve(path);
    agg::conv_stroke<curve_t> stroke(curve);
    stroke.width(gc.linewidth);
    stroke.line_cap(gc.cap);
    stroke.line_join(gc.join);
    rendererAA->color(gc.color);
    theRasterizer->add_path(stroke);
  }
  else {
    agg::conv_stroke<VS> stroke(path);
    stroke.width(gc.linewidth);
    stroke.line_cap(gc.cap);
    stroke.line_join(gc.join);
    rendererAA->color(gc.color);
    theRasterizer->add_path(stroke);
  }
  agg::render_scanlines(*theRasterizer, *slineP8, *rendererAA);

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

  set_clipbox_rasterizer(gc.cliprect);

  agg::path_storage path;

  b = height - (b+h);
  path.move_to(l, b+h);
  path.line_to(l+w, b+h);
  path.line_to(l+w, b);
  path.line_to(l, b);
  path.close_polygon();

  _fill_and_stroke(path, gc, face, false);

  return Py::Object();

}

Py::Object
RendererAgg::draw_ellipse(const Py::Tuple& args) {
  _VERBOSE("RendererAgg::draw_ellipse");
  args.verify_length(6);

  GCAgg gc = GCAgg(args[0], dpi);
  facepair_t face = _get_rgba_face(args[1], gc.alpha);


  double x = Py::Float( args[2] );
  double y = Py::Float( args[3] );
  double w = Py::Float( args[4] );
  double h = Py::Float( args[5] );

  set_clipbox_rasterizer(gc.cliprect);

  //last arg is num steps
  agg::ellipse path(x, height-y, w, h, 100);

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


  // dump the x.y vertices into a double array for faster look ahread
  // and behind access
  double xs[Npoints];
  double ys[Npoints];

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

    if (j==0) path.move_to(x,y);
    else path.line_to(x,y);
  }
  path.close_polygon();

  _fill_and_stroke(path, gc, face, false);
  _VERBOSE("RendererAgg::draw_polygon DONE");
  return Py::Object();

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
	thisx = (int)thisx + 0.5;
	thisy = (int)thisy + 0.5;
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
  BufferRegion* reg = new BufferRegion(buf, r);

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
  for (size_t i=0; i<N; i++) {

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
    double xs[Nverts], ys[Nverts];
    for (size_t j=0; j<Nverts; j++) {
      thisvert = thisverts[j];
      double x = Py::Float(thisvert[0]);
      double y = Py::Float(thisvert[1]);
      try {
	xy = transform->operator()(x, y);
      }
      catch(...) {
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

    for (size_t j=0; j<Nverts; j++) {

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

    if (a>0) { //only render if alpha>0
      agg::rgba edgecolor(r, g, b, a);

      agg::conv_stroke<agg::path_storage> stroke(path);
      //stroke.line_cap(cap);
      //stroke.line_join(join);
      double lw = points_to_pixels ( Py::Float( linewidths[i%Nlw] ) );
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
  double xverts[Nverts];
  double yverts[Nverts];
  Py::SeqBase<Py::Object> xy;
  for (size_t i=0; i<Nverts; i++) {
    xy = Py::SeqBase<Py::Object>(verts[i]);
    xverts[i] = Py::Float(xy[0]);
    yverts[i] = Py::Float(xy[1]);
  }

  std::pair<double, double> offsetPair;
  for (size_t i=0; i<Noffsets; i++) {
    Py::SeqBase<Py::Object> pos = Py::SeqBase<Py::Object>(offsets[i]);
    double xo = Py::Float(pos[0]);
    double yo = Py::Float(pos[1]);
    try {
      offsetPair = transOffset->operator()(xo, yo);
    }
    catch(...) {
      throw Py::ValueError("Domain error on eval_scalars in RendererAgg::draw_regpoly_collection");
    }



    double scale = Py::Float(sizes[i%Nsizes]);


    agg::path_storage path;

    for (size_t j=0; j<Nverts; j++) {
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
    if (a>0) { //only render if alpha>0
      agg::rgba edgecolor(r, g, b, a);

      agg::conv_stroke<agg::path_storage> stroke(path);
      //stroke.line_cap(cap);
      //stroke.line_join(join);
      double lw = points_to_pixels ( Py::Float( linewidths[i%Nlw] ) );
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
  return Py::Object();
}

Py::Object
RendererAgg::draw_lines(const Py::Tuple& args) {


  _VERBOSE("RendererAgg::draw_lines");
  args.verify_length(4);
  GCAgg gc = GCAgg(args[0], dpi);

  set_clipbox_rasterizer(gc.cliprect);

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

  double thisx, thisy;
  bool moveto = true;
  double heightd = height;

  double lastx(-2.0), lasty(-2.0);

  bool snapto = Nx==2;

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

    //use agg's transformer?
    xytrans.transform(&thisx, &thisy);
    thisy = heightd - thisy; //flipy

    //don't render line segments less that on pixel long!
    if (!moveto && (i>0) && fabs(thisx-lastx)<1.0 && fabs(thisy-lasty)<1.0) {
      continue;
    }

    lastx = thisx;
    lasty = thisy;
    if (snapto) {
      thisx = (int)thisx + 0.5;
      thisy = (int)thisy + 0.5;
    }

    if (moveto)
      path.move_to(thisx, thisy);
    else
      path.line_to(thisx, thisy);

    moveto = false;

  }

  Py_XDECREF(xa);
  Py_XDECREF(ya);

  //typedef agg::conv_transform<agg::path_storage, agg::trans_affine> path_t;
  //path_t transpath(path, xytrans);
  _render_lines_path(path, gc);

  _VERBOSE("RendererAgg::draw_lines DONE");
  return Py::Object();

}

template<class PathSource>
void
RendererAgg::_render_lines_path(PathSource &path, const GCAgg& gc) {

  typedef PathSource path_t;
  //typedef agg::conv_transform<agg::path_storage, agg::trans_affine> path_t;
  typedef agg::conv_stroke<path_t> stroke_t;
  typedef agg::conv_dash<path_t> dash_t;

  //path_t transpath(path, xytrans);

  if (gc.dasha==NULL ) { //no dashes
    stroke_t stroke(path);
    stroke.width(gc.linewidth);
    stroke.line_cap(gc.cap);
    stroke.line_join(gc.join);
    rendererAA->color(gc.color);
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
    theRasterizer->add_path(stroke);

  }

  if ( gc.isaa ) {
    rendererAA->color(gc.color);
    agg::render_scanlines(*theRasterizer, *slineP8, *rendererAA);
  }
  else {
    rendererBin->color(gc.color);
    agg::render_scanlines(*theRasterizer, *slineBin, *rendererBin);
  }

}

/*
Py::Object
RendererAgg::draw_markers(const Py::Tuple& args) {
  // there is a win32 specific segfault that happens when using the
  // cacheing code.  The segfault always happens on an agg call but I
  // can't reproduce a standalone case.  Until this gets sorted out,
  // I'm using two different versions of draw_markers, one which uses
  // raster caches of the markers (much faster) for non win32
  // platforms, and the older, slower, no cache version for win32.
#if defined(MS_WIN32)
  return _draw_markers_nocache(args);
#else
  return _draw_markers_cache(args);
#endif
}


Py::Object
RendererAgg::_draw_markers_nocache(const Py::Tuple& args) {
  //draw_markers(gc, path, rgbFace, xo, yo, transform)
  theRasterizer->reset_clipping();

  _VERBOSE("RendererAgg::_draw_markers_nocache");
  args.verify_length(6);

  GCAgg gc = GCAgg(args[0], dpi);

  agg::path_storage *ppath;
  swig_type_info * descr = SWIG_TypeQuery("agg::path_storage *");
  assert(descr);
  if (SWIG_ConvertPtr(args[1].ptr(),(void **)(&ppath), descr, 0) == -1)
    throw Py::TypeError("Could not convert path_storage");


  facepair_t face = _get_rgba_face(args[2], gc.alpha);
  Py::Object xo = args[3];
  Py::Object yo = args[4];

  PyArrayObject *xa = (PyArrayObject *) PyArray_ContiguousFromObject(xo.ptr(), PyArray_DOUBLE, 1, 1);

  if (xa==NULL)
    throw Py::TypeError("RendererAgg::_draw_markers_nocache expected numerix array");


  PyArrayObject *ya = (PyArrayObject *) PyArray_ContiguousFromObject(yo.ptr(), PyArray_DOUBLE, 1, 1);

  if (ya==NULL)
    throw Py::TypeError("RendererAgg::_draw_markers_nocache expected numerix array");

  Transformation* mpltransform = static_cast<Transformation*>(args[5].ptr());

  double a, b, c, d, tx, ty;
  try {
    mpltransform->affine_params_api(&a, &b, &c, &d, &tx, &ty);
  }
  catch(...) {
    throw Py::ValueError("Domain error on affine_params_api in RendererAgg::_draw_markers_nocache");
  }

  agg::trans_affine xytrans = agg::trans_affine(a,b,c,d,tx,ty);


  size_t Nx = xa->dimensions[0];
  size_t Ny = ya->dimensions[0];

  if (Nx!=Ny)
    throw Py::ValueError(Printf("x and y must be equal length arrays; found %d and %d", Nx, Ny).str());


  double heightd = double(height);

  bool curvy = false;
  size_t Npath = ppath->total_vertices();
  for (size_t i=0; i<Npath; i++) {
    double x, y;
    unsigned code = ppath->vertex(&x, &y);

    if (code==3||code==4||code==5)
      curvy = true;
  }





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

  agg::path_storage markers;
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
    markers.move_to(thisx, thisy);
    //agg::path_storage marker;
    double x, y;
    unsigned cmd;
    ppath->rewind(0);
    while(!agg::is_stop(cmd = ppath->vertex(&x, &y)))
      markers.add_vertex(x+thisx, thisy-y, cmd);

    _fill_and_stroke(markers, gc, face, curvy);
  } //for each marker


  Py_XDECREF(xa);
  Py_XDECREF(ya);

  return Py::Object();

}

*/
Py::Object
RendererAgg::draw_markers(const Py::Tuple& args) {
  //_draw_markers_cache(gc, path, rgbFace, xo, yo, transform)
  theRasterizer->reset_clipping();

  _VERBOSE("RendererAgg::_draw_markers_cache");
  args.verify_length(6);

  _VERBOSE("RendererAgg::_draw_markers_cache setting gc");
  GCAgg gc = GCAgg(args[0], dpi);


  agg::path_storage *ppath;
  _VERBOSE("RendererAgg::_draw_markers_cache get path storage");
  swig_type_info * descr = SWIG_TypeQuery("agg::path_storage *");
  _VERBOSE("RendererAgg::_draw_markers_cache got path storage");
  assert(descr);
  _VERBOSE("RendererAgg::_draw_markers_cache asserted");
  if (SWIG_ConvertPtr(args[1].ptr(),(void **)(&ppath), descr, 0) == -1) {
    _VERBOSE("RendererAgg::_draw_markers_cache throwing");
    throw Py::TypeError("Could not convert path_storage");
  }
  _VERBOSE("RendererAgg::_draw_markers_cache getface");
  facepair_t face = _get_rgba_face(args[2], gc.alpha);

  _VERBOSE("RendererAgg::_draw_markers_cache 1");
  Py::Object xo = args[3];
  Py::Object yo = args[4];

  PyArrayObject *xa = (PyArrayObject *) PyArray_ContiguousFromObject(xo.ptr(), PyArray_DOUBLE, 1, 1);

  if (xa==NULL)
    throw Py::TypeError("RendererAgg::_draw_markers_cache expected numerix array");


  PyArrayObject *ya = (PyArrayObject *) PyArray_ContiguousFromObject(yo.ptr(), PyArray_DOUBLE, 1, 1);

  _VERBOSE("RendererAgg::_draw_markers_cache 2");
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

  _VERBOSE("RendererAgg::_draw_markers_cache 3");
  size_t Nx = xa->dimensions[0];
  size_t Ny = ya->dimensions[0];

  if (Nx!=Ny)
    throw Py::ValueError(Printf("x and y must be equal length arrays; found %d and %d", Nx, Ny).str());


  double heightd = double(height);


  ppath->rewind(0);
  ppath->flip_y(0,0);
  typedef agg::conv_curve<agg::path_storage> curve_t;
  curve_t curve(*ppath);

  _VERBOSE("RendererAgg::_draw_markers_cache 4");
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

  _VERBOSE("RendererAgg::_draw_markers_cache 5");


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


  _VERBOSE("RendererAgg::_draw_markers_cache 6");
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

    _VERBOSE("RendererAgg::_draw_markers_cache 7");
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

  _VERBOSE("RendererAgg::_draw_markers_cache 8");
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
  args.verify_length(4);

  GCAgg gc = GCAgg(args[0], dpi);
  facepair_t face = _get_rgba_face(args[1], gc.alpha);

  agg::path_storage *path;
  swig_type_info * descr = SWIG_TypeQuery("agg::path_storage *");
  assert(descr);
  if (SWIG_ConvertPtr(args[2].ptr(),(void **)(&path), descr, 0) == -1)
    throw Py::TypeError("Could not convert path_storage");


  Transformation* mpltransform = static_cast<Transformation*>(args[3].ptr());

  double a, b, c, d, tx, ty;
  try {
    mpltransform->affine_params_api(&a, &b, &c, &d, &tx, &ty);
  }
  catch(...) {
    throw Py::ValueError("Domain error on affine_params_api in RendererAgg::draw_path");
  }

  agg::trans_affine xytrans = agg::trans_affine(a,b,c,d,tx,ty);

  double heightd = double(height);
  agg::path_storage tpath;  // the mpl transformed path
  bool needNonlinear = mpltransform->need_nonlinear_api();
  size_t Nx = path->total_vertices();
  double x, y;
  unsigned cmd;
  bool curvy = false;
  for (size_t i=0; i<Nx; i++) {
    cmd = path->vertex(i, &x, &y);
    if (cmd==agg::path_cmd_curve3 || cmd==agg::path_cmd_curve4) curvy=true;
    if (needNonlinear)
      try {
	mpltransform->nonlinear_only_api(&x, &y);
      }
      catch (...) {
	throw Py::ValueError("Domain error on nonlinear_only_api in RendererAgg::draw_path");

      }

    //use agg's transformer?
    xytrans.transform(&x, &y);
    y = heightd - y; //flipy
    tpath.add_vertex(x,y,cmd);
  }

  _fill_and_stroke(tpath, gc, face, curvy);
  return Py::Object();

}



Py::Object
RendererAgg::draw_text(const Py::Tuple& args) {
  _VERBOSE("RendererAgg::draw_text");

  args.verify_length(4);


  FT2Font *font = static_cast<FT2Font*>(args[0].ptr());

  int x(0),y(0);
  try {
    x = Py::Int( args[1] );
    y = Py::Int( args[2] );
  }
  catch (Py::TypeError) {
    //x,y out of range; todo issue warning?
    return Py::Object();
  }

  GCAgg gc = GCAgg(args[3], dpi);

  set_clipbox_rasterizer( gc.cliprect);


  pixfmt::color_type p;
  p.r = int(255*gc.color.r);
  p.b = int(255*gc.color.b);
  p.g = int(255*gc.color.g);
  p.a = int(255*gc.color.a);

  //y = y-font->image.height;
  unsigned thisx, thisy;

  double l = 0;
  double b = 0;
  double r = width;
  double t = height;
  if (gc.cliprect!=NULL) {
    l = gc.cliprect[0] ;
    b = gc.cliprect[1] ;
    double w = gc.cliprect[2];
    double h = gc.cliprect[3];
    r = l+w;
    t = b+h;
  }


  for (size_t i=0; i<font->image.width; i++) {
    for (size_t j=0; j<font->image.height; j++) {
      thisx = i+x+font->image.offsetx;
      thisy = j+y+font->image.offsety;
      if (thisx<l || thisx>=r)  continue;
      if (thisy<height-t || thisy>=height-b) continue;
      pixFmt->blend_pixel
	(thisx, thisy, p, font->image.buffer[i + j*font->image.width]);
    }
  }

  /*  bbox the text for debug purposes

  agg::path_storage path;

  path.move_to(x, y);
  path.line_to(x, y+font->image.height);
  path.line_to(x+font->image.width, y+font->image.height);
  path.line_to(x+font->image.width, y);
  path.close_polygon();

  agg::rgba edgecolor(1,0,0,1);

  //now fill the edge
  agg::conv_stroke<agg::path_storage> stroke(path);
  stroke.width(1.0);
  rendererAA->color(edgecolor);
  //self->theRasterizer->gamma(agg::gamma_power(gamma));
  theRasterizer->add_path(stroke);
  agg::render_scanlines(*theRasterizer, *slineP8, *rendererAA);

  */

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


// this code is heavily adapted from the paint license, which is in
// the file paint.license (BSD compatible) included in this
// distribution.  TODO, add license file to MANIFEST.in and CVS
Py::Object
RendererAgg::write_png(const Py::Tuple& args)
{
  _VERBOSE("RendererAgg::write_png");

  args.verify_length(1);

  FILE *fp;
  Py::Object o = Py::Object(args[0]);
  bool fpclose = true;
  if (o.isString()) {
    std::string fileName = Py::String(o);
    const char *file_name = fileName.c_str();
    if ((fp = fopen(file_name, "wb")) == NULL)
      throw Py::RuntimeError( Printf("Could not open file %s", file_name).str() );
  }
  else {
    if ((fp = PyFile_AsFile(o.ptr())) == NULL)
      throw Py::TypeError("Could not convert object to file pointer");
    fpclose = false;
  }

  png_structp png_ptr;
  png_infop info_ptr;
  struct        png_color_8_struct sig_bit;
  png_uint_32 row;

  png_bytep row_pointers[height];
  for (row = 0; row < height; ++row) {
    row_pointers[row] = pixBuffer + row * width * 4;
  }


  if (fp == NULL)
    throw Py::RuntimeError("Could not open file");


  png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (png_ptr == NULL) {
    if (fpclose) fclose(fp);
    throw Py::RuntimeError("Could not create write struct");
  }

  info_ptr = png_create_info_struct(png_ptr);
  if (info_ptr == NULL) {
    if (fpclose) fclose(fp);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    throw Py::RuntimeError("Could not create info struct");
  }

  if (setjmp(png_ptr->jmpbuf)) {
    if (fpclose) fclose(fp);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    throw Py::RuntimeError("Error building image");
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

  /* Changed calls to png_destroy_write_struct to follow
     http://www.libpng.org/pub/png/libpng-manual.txt.
     This ensures the info_ptr memory is released.
  */

  png_destroy_write_struct(&png_ptr, &info_ptr);


  if (fpclose) fclose(fp);

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

  color_conv(&renderingBufferTmp, renderingBuffer, agg::color_conv_rgba32_to_rgb24());


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

  color_conv(&renderingBufferTmp, renderingBuffer, agg::color_conv_rgba32_to_argb32());


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

  color_conv(&renderingBufferTmp, renderingBuffer, agg::color_conv_rgba32_to_bgra32());


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
  delete [] pixBuffer;
  delete [] cacheBuffer;

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
  add_varargs_method("draw_line_collection",
		     &RendererAgg::draw_line_collection,
		     "draw_line_collection(segments, trans, clipbox, colors, linewidths, antialiaseds)\n");
  add_varargs_method("draw_poly_collection",
		     &RendererAgg::draw_poly_collection,
		     "draw_poly_collection()\n");
  add_varargs_method("draw_regpoly_collection",
		     &RendererAgg::draw_regpoly_collection,
		     "draw_regpoly_collection()\n");
  add_varargs_method("draw_lines", &RendererAgg::draw_lines,
		     "draw_lines(gc, x, y,)\n");
  add_varargs_method("draw_markers", &RendererAgg::draw_markers,
		     "draw_markers(gc, path, x, y)\n");
  add_varargs_method("draw_path", &RendererAgg::draw_path,
		     "draw_path(gc, rgbFace, path, transform)\n");
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
#ifdef NUMARRAY
  init_na_backend_agg(void)
#else
#   ifdef NUMERIC
  init_nc_backend_agg(void)
#   else
  init_ns_backend_agg(void)
#   endif
#endif
{
  //static _backend_agg_module* _backend_agg = new _backend_agg_module;

#ifdef NUMARRAY
  _VERBOSE("init_na_backend_agg");
#else
#   ifdef NUMERIC
  _VERBOSE("init_nc_backend_agg");
#   else
  _VERBOSE("init_ns_backend_agg");
#   endif
#endif

  import_array();

  static _backend_agg_module* _backend_agg = NULL;
  _backend_agg = new _backend_agg_module;

};
