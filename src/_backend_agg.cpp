/* A rewrite of _backend_agg using PyCXX to handle ref counting, etc..
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdio>

#include <png.h>
#include "util/agg_color_conv_rgb8.h"

#include "ft2font.h"
#include "_image.h"
#include "_backend_agg.h"

#include "_transforms.h"

/* ------------ RendererAgg methods ------------- */

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
  if (debug)
    std::cout << "RendererAgg::RendererAgg" << std::endl;
  unsigned stride(width*4);    
  
  
  pixBuffer = new agg::int8u[NUMBYTES];  
  
  renderingBuffer = new agg::rendering_buffer;
  renderingBuffer->attach(pixBuffer, width, height, stride);
  slineP8 = new scanline_p8;
  slineBin = new scanline_bin;
  
  
  pixFmt = new pixfmt(*renderingBuffer);
  rendererBase = new renderer_base(*pixFmt);
  rendererBase->clear(agg::rgba(1, 1, 1, 0));
  
  theRenderer = new renderer(*rendererBase);
  rendererBin = new renderer_bin(*rendererBase);
  theRasterizer = new rasterizer(); 
  
};

Py::Object
RendererAgg::draw_rectangle(const Py::Tuple & args) {
  if (debug)
    std::cout << "RendererAgg::draw_rectangle" << std::endl;
  args.verify_length(6);
  
  Py::Object gcEdge( args[0] );
  Py::Object rgbFaceMaybeNone( args[1] );
  
  double l = Py::Float( args[2] ); 
  double b = Py::Float( args[3] ); 
  double w = Py::Float( args[4] ); 
  double h = Py::Float( args[5] ); 
  
  set_clip_rectangle(gcEdge);
  
  double lw = points_to_pixels ( gcEdge.getAttr("_linewidth") ) ;
  
  agg::path_storage path;
  
  b = height - (b+h);
  path.move_to(l, b+h);
  path.line_to(l+w, b+h);
  path.line_to(l+w, b);
  path.line_to(l, b);
  path.close_polygon();
  
  agg::rgba edgecolor = get_color(gcEdge);
  
  
  if (rgbFaceMaybeNone.ptr() != Py_None) {
    //fill the face
    Py::SeqBase<Py::Object> rgbFace = rgbFaceMaybeNone;
    agg::rgba facecolor = rgb_to_color(rgbFace, edgecolor.a);
    
    theRenderer->color(facecolor);
    theRasterizer->add_path(path);    
    theRasterizer->render(*slineP8, *theRenderer);  
    
  }
  
  //now fill the edge
  agg::conv_stroke<agg::path_storage> stroke(path);
  stroke.width(lw);
  theRenderer->color(edgecolor);
  //self->theRasterizer->gamma(agg::gamma_power(gamma));
  theRasterizer->add_path(stroke);
  theRasterizer->render(*slineP8, *theRenderer);  
  
  
  return Py::Object();
  
}

Py::Object
RendererAgg::draw_ellipse(const Py::Tuple& args) {
  if (debug)
    std::cout << "RendererAgg::draw_ellipse" << std::endl;
  
  args.verify_length(6);  
  Py::Object gcEdge = args[0];
  Py::Object rgbFaceMaybeNone = args[1];
  
  double x = Py::Float( args[2] ); 
  double y = Py::Float( args[3] ); 
  double w = Py::Float( args[4] ); 
  double h = Py::Float( args[5] ); 
  
  set_clip_rectangle(gcEdge);
  
  //last arg is num steps
  agg::ellipse path(x, height-y, w, h, 100); 
  agg::rgba edgecolor = get_color(gcEdge);  
  
  
  if (rgbFaceMaybeNone.ptr() != Py_None) {
    Py::SeqBase<Py::Object> rgbFace = rgbFaceMaybeNone;
    agg::rgba facecolor = rgb_to_color(rgbFace, edgecolor.a);
    theRenderer->color(facecolor);
    theRasterizer->add_path(path);    
    theRasterizer->render(*slineP8, *theRenderer);  
    
  }
  
  
  //now fill the edge
  
  double lw = points_to_pixels ( gcEdge.getAttr("_linewidth") ) ;
  
  agg::conv_stroke<agg::ellipse> stroke(path);
  stroke.width(lw);
  theRenderer->color(edgecolor);
  //self->theRasterizer->gamma(agg::gamma_power(gamma));
  theRasterizer->add_path(stroke);
  theRasterizer->render(*slineP8, *theRenderer);  
  
  return Py::Object();
  
}

Py::Object
RendererAgg::draw_polygon(const Py::Tuple& args) {
  if (debug)
    std::cout << "RendererAgg::draw_polygon" << std::endl;
  
  args.verify_length(3);  
  
  Py::Object gcEdge( args[0] );
  Py::Object rgbFaceMaybeNone( args[1] );
  Py::SeqBase<Py::Object> points( args[2] );
  
  
  set_clip_rectangle(gcEdge);
  agg::gen_stroke::line_cap_e cap = get_linecap(gcEdge);
  agg::gen_stroke::line_join_e join = get_joinstyle(gcEdge);
  
  double lw = points_to_pixels ( gcEdge.getAttr("_linewidth") ) ;
  
  size_t Npoints = points.length();
  if (Npoints<=0)
    return Py::Object();
  
  
  // dump the x.y vertices into a double array for faster look ahread
  // and behind access
  double xs[Npoints];
  double ys[Npoints];
  Py::Tuple xy;
  for (size_t i=0; i<Npoints; ++i) {
    xy = Py::Tuple(points[i]);
    xs[i] = Py::Float(xy[0]);
    ys[i] = Py::Float(xy[1]);
    ys[i] = height - ys[i];

  }
  
  agg::path_storage path;  
  for (size_t j=0; j<Npoints; ++j) {

    double x = xs[j];
    double y = ys[j];
     
    if (j==0) path.move_to(x,y);
    else path.line_to(x,y); 
  }
  path.close_polygon();
  
  agg::rgba edgecolor = get_color(gcEdge);

  
  if (rgbFaceMaybeNone.ptr() != Py_None) {
    //fill the face
    Py::SeqBase<Py::Object> rgbFace = rgbFaceMaybeNone;
    agg::rgba facecolor = rgb_to_color(rgbFace, edgecolor.a);
    theRenderer->color(facecolor);
    theRasterizer->add_path(path);    
    theRasterizer->render(*slineP8, *theRenderer);  
  }
  
  //now fill the edge
  agg::conv_stroke<agg::path_storage> stroke(path);
  stroke.width(lw);
  stroke.line_cap(cap);
  stroke.line_join(join);
  
  theRenderer->color(edgecolor);
  //self->theRasterizer->gamma(agg::gamma_power(gamma));
  theRasterizer->add_path(stroke);
  theRasterizer->render(*slineP8, *theRenderer);  
  return Py::Object();
  
}

Py::Object
RendererAgg::draw_line_collection(const Py::Tuple& args) {
  
  
  if (debug)
    std::cout << "RendererAgg::draw_line_collection" << std::endl;
  args.verify_length(8);  
  
  
  //segments, trans, clipbox, colors, linewidths, antialiaseds
  Py::SeqBase<Py::Object> segments = args[0];  


  /* this line is broken, mysteriously
  if (!Transformation::check(args[1])) 
    throw Py::TypeError("RendererAgg::draw_line_collection(segments, transform, ...) expected a Transformation instance for transform");
  
  */

  Transformation* transform = static_cast<Transformation*>(args[1].ptr());

  set_clip_from_bbox(args[2]);

  Py::SeqBase<Py::Object> colors = args[3];  
  Py::SeqBase<Py::Object> linewidths = args[4];  
  Py::SeqBase<Py::Object> antialiaseds = args[5];  

  bool usingOffsets = args[6].ptr()!=Py_None;
  Py::SeqBase<Py::Object> offsets;
  Transformation* transOffset=NULL;
  if  (usingOffsets) {
    /* this line is broken, mysteriously
    if (!Transformation::check(args[7])) 
      throw Py::TypeError("RendererAgg::draw_line_collection expected a Transformation instance for transOffset");
    */
    offsets = Py::SeqBase<Py::Object>(args[6]);        
    transOffset = static_cast<Transformation*>(args[7].ptr());
  }

  size_t Nsegments = segments.length();
  size_t Nc = colors.length();
  size_t Nlw = linewidths.length();
  size_t Naa = antialiaseds.length();
  size_t Noffsets = 0;
  size_t N = Nsegments;

  if (usingOffsets) {
    Noffsets = offsets.length();
    if (Noffsets>Nsegments) N = Noffsets;
  }
  

  Py::Tuple xyo, pos;
  for (size_t i=0; i<N; ++i) {
    
    pos = Py::Tuple(segments[i%Nsegments]);
    double x0 = Py::Float(pos[0]);
    double y0 = Py::Float(pos[1]);
    double x1 = Py::Float(pos[2]);
    double y1 = Py::Float(pos[3]);


    std::pair<double, double> xy = transform->operator()(x0,y0);
    x0 = xy.first;
    y0 = xy.second;

    xy = transform->operator()(x1,y1);
    x1 = xy.first;
    y1 = xy.second;

    if (usingOffsets) {
      xyo = Py::Tuple(offsets[i%Noffsets]);
      double xo = Py::Float(xyo[0]);
      double yo = Py::Float(xyo[1]);
      std::pair<double, double> xy = transOffset->operator()(xo,yo);
      x0 += xy.first;
      y0 += xy.second;
      x1 += xy.first;
      y1 += xy.second;
      
    }
    //snap x to pixel for verical lines
    if (x0==x1) {
      x0 = (int)x0 + 0.5;
      x1 = (int)x1 + 0.5;
    }
    
    //snap y to pixel for horizontal lines
    if (y0==y1) {
      y0 = (int)y0 + 0.5;
      y1 = (int)y1 + 0.5;
    }

      
    agg::path_storage path;
    path.move_to(x0, height-y0);
    path.line_to(x1, height-y1);

    agg::conv_stroke<agg::path_storage> stroke(path);
    //stroke.line_cap(cap);
    //stroke.line_join(join);
    double lw = points_to_pixels ( Py::Float( linewidths[i%Nlw] ) );
    
    stroke.width(lw);
    theRasterizer->add_path(stroke);

    // get the color and render
    Py::Tuple rgba = Py::Tuple(colors[ i%Nc]);
    double r = Py::Float(rgba[0]);
    double g = Py::Float(rgba[1]);
    double b = Py::Float(rgba[2]); 
    double a = Py::Float(rgba[3]);
    agg::rgba color(r, g, b, a); 

    // render antialiased or not
    int isaa = Py::Int(antialiaseds[i%Naa]);
    if ( isaa ) {
      theRenderer->color(color);    
      theRasterizer->render(*slineP8, *theRenderer); 
    }
    else {
      rendererBin->color(color);    
      theRasterizer->render(*slineBin, *rendererBin); 
    }
  } //for every segment
  return Py::Object();
}


void
RendererAgg::set_clip_from_bbox(const Py::Object& o) {
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
  }

  
}


Py::Object
RendererAgg::draw_poly_collection(const Py::Tuple& args) {
  
  
  if (debug)
    std::cout << "RendererAgg::draw_poly_collection" << std::endl;
  args.verify_length(9);  
  

  Py::SeqBase<Py::Object> verts = args[0];    

  //todo: fix transformation check
  Transformation* transform = static_cast<Transformation*>(args[1].ptr());
  transform->eval_scalars();

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
    transOffset->eval_scalars();
  }

  size_t Noffsets = offsets.length();
  size_t Nverts = verts.length();
  size_t Nface = facecolors.length();
  size_t Nedge = edgecolors.length();
  size_t Nlw = linewidths.length();
  size_t Naa = antialiaseds.length();

  size_t N = (Noffsets>Nverts) ? Noffsets : Nverts;
   
  std::pair<double, double> xyo, xy;
  Py::Tuple thisverts;
  for (size_t i=0; i<N; ++i) {

    thisverts = verts[i % Nverts];

    if (usingOffsets) {
      Py::Tuple pos = Py::Tuple(offsets[i]);
      double xo = Py::Float(pos[0]);
      double yo = Py::Float(pos[1]);
      xyo = transOffset->operator()(xo, yo);
    }

    size_t Nverts = thisverts.length();
    agg::path_storage path;
    
    Py::Tuple thisvert;

    
    // dump the verts to double arrays so we can do more efficient
    // look aheads and behinds when doing snapto pixels
    double xs[Nverts], ys[Nverts];    
    for (size_t j=0; j<Nverts; ++j) {
      thisvert = Py::Tuple(thisverts[j]);
      double x = Py::Float(thisvert[0]);
      double y = Py::Float(thisvert[1]);
      xy = transform->operator()(x, y);      

      if (usingOffsets) {
	xy.first  += xyo.first;
	xy.second += xyo.second;
      }

      xy.second = height - xy.second;
      xs[j] = xy.first;
      ys[j] = xy.second;

    }

    for (size_t j=0; j<Nverts; ++j) {

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
    Py::Tuple rgba = Py::Tuple(facecolors[ i%Nface]);
    double r = Py::Float(rgba[0]);
    double g = Py::Float(rgba[1]);
    double b = Py::Float(rgba[2]);
    double a = Py::Float(rgba[3]);
    if (a>0) { //only render if alpha>0
      agg::rgba facecolor(r, g, b, a); 

      theRasterizer->add_path(path);          

      if (isaa) {
	theRenderer->color(facecolor);    
	theRasterizer->render(*slineP8, *theRenderer); 
      }
      else {
	rendererBin->color(facecolor);    
	theRasterizer->render(*slineBin, *rendererBin); 
      }
    } //renderer face
    
    // get the edgecolor and render
    rgba = Py::Tuple(edgecolors[ i%Nedge]);
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
	theRenderer->color(edgecolor);    
	theRasterizer->render(*slineP8, *theRenderer); 
      }
      else {
	rendererBin->color(edgecolor);    
	theRasterizer->render(*slineBin, *rendererBin); 
      }
    } //rendered edge
    
  } // for every poly
  return Py::Object();
}

Py::Object
RendererAgg::draw_regpoly_collection(const Py::Tuple& args) {
  
  
  if (debug)
    std::cout << "RendererAgg::draw_regpoly_collection" << std::endl;
  args.verify_length(9);  
  
  
  set_clip_from_bbox(args[0]);

  Py::SeqBase<Py::Object> offsets = args[1];  

  // this is throwing even though the instance is a Transformation!
  //if (!Transformation::check(args[2])) 
  // throw Py::TypeError("RendererAgg::draw_regpoly_collection(clipbox, offsets, transOffset, verts, ...) expected a Transformation instance for transOffset");
  
  Transformation* transOffset = static_cast<Transformation*>(args[2].ptr());


  transOffset->eval_scalars();

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
  
  // dump the x.y vertices into a double array for paster access
  double xverts[Nverts];
  double yverts[Nverts];
  Py::Tuple xy;
  for (size_t i=0; i<Nverts; ++i) {
    xy = Py::Tuple(verts[i]);
    xverts[i] = Py::Float(xy[0]);
    yverts[i] = Py::Float(xy[1]);
  }
  
  std::pair<double, double> offsetPair;
  for (size_t i=0; i<Noffsets; ++i) {
    Py::Tuple pos = Py::Tuple(offsets[i]);
    double xo = Py::Float(pos[0]);
    double yo = Py::Float(pos[1]);
    offsetPair = transOffset->operator()(xo, yo);
    
    
    double scale = Py::Float(sizes[i%Nsizes]);
    
    
    agg::path_storage path;
    
    for (size_t j=0; j<Nverts; ++j) {
      thisx = scale*xverts[j] + offsetPair.first;
      thisy = scale*yverts[j] + offsetPair.second;
      thisy = height - thisy;
      if (j==0) path.move_to(thisx, thisy);
      else path.line_to(thisx, thisy);

      
    }
    path.close_polygon();
    int isaa = Py::Int(antialiaseds[i%Naa]);     
    // get the facecolor and render
    Py::Tuple rgba = Py::Tuple(facecolors[ i%Nface]);
    double r = Py::Float(rgba[0]);
    double g = Py::Float(rgba[1]);
    double b = Py::Float(rgba[2]);
    double a = Py::Float(rgba[3]);
    if (a>0) { //only render if alpha>0
      agg::rgba facecolor(r, g, b, a); 

      theRasterizer->add_path(path);          

      if (isaa) {
	theRenderer->color(facecolor);    
	theRasterizer->render(*slineP8, *theRenderer); 
      }
      else {
	rendererBin->color(facecolor);    
	theRasterizer->render(*slineBin, *rendererBin); 
      }
    } //renderer face
    
    // get the edgecolor and render
    rgba = Py::Tuple(edgecolors[ i%Nedge]);
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
	theRenderer->color(edgecolor);    
	theRasterizer->render(*slineP8, *theRenderer); 
      }
      else {
	rendererBin->color(edgecolor);    
	theRasterizer->render(*slineBin, *rendererBin); 
      }
    } //rendered edge
    
  } // for every poly
  return Py::Object();
}

Py::Object
RendererAgg::draw_lines(const Py::Tuple& args) {
  
  
  if (debug)
    std::cout << "RendererAgg::draw_lines" << std::endl;
  args.verify_length(3);  
  Py::Object gc = args[0];
  Py::SeqBase<Py::Object> x = args[1];  //todo: use numerix for efficiency
  Py::SeqBase<Py::Object> y = args[2];  //todo: use numerix for efficiency
  
  set_clip_rectangle(gc);
  size_t Nx = x.length();
  size_t Ny = y.length();
  
  if (Nx!=Ny) 
    throw Py::ValueError("x and y must be equal length sequences");
  
  
  if (Nx<2) 
    throw Py::ValueError("x and y must have length >= 2");
  
  
  agg::gen_stroke::line_cap_e cap = get_linecap(gc);
  agg::gen_stroke::line_join_e join = get_joinstyle(gc);
  
  
  double lw = points_to_pixels ( gc.getAttr("_linewidth") ) ;
  std::cout << "agg lw " << lw << std::endl;
  agg::rgba color = get_color(gc);
  
  
  // process the dashes
  Py::Tuple dashes = get_dashes(gc);
  
  bool useDashes = dashes[0].ptr() != Py_None;
  double offset = 0;
  Py::SeqBase<Py::Object> dashSeq;
  
  if ( dashes[0].ptr() != Py_None ) { // use dashes
    //TODO: use offset
    offset = points_to_pixels_snapto(dashes[0]);
    dashSeq = dashes[1]; 
  };
  
  
  agg::path_storage path;
  
  int isaa = antialiased(gc);


  if (Nx==2) { 
    // this is a little hack - len(2) lines are probably grid and
    // ticks so I'm going to snap to pixel
    //printf("snapto %d\n", Nx);
    double x0 = Py::Float(x[0]);
    double y0 = Py::Float(y[0]);
    double x1 = Py::Float(x[1]);
    double y1 = Py::Float(y[1]);

    if (x0==x1) {
      x0 = (int)x0 + 0.5;
      x1 = (int)x1 + 0.5;
    }

    if (y0==y1) {
      y0 = (int)y0 + 0.5;
      y1 = (int)y1 + 0.5;
    }

    y0 = height-y0;
    y1 = height-y1;

    path.move_to(x0, y0);
    path.line_to(x1, y1);
    
  }
  else {
    double thisX = Py::Float( x[0] );
    double thisY = Py::Float( y[0] );
    thisY = height - thisY; //flipy
    path.move_to(thisX, thisY);
    for (size_t i=1; i<Nx; ++i) {
      thisX = Py::Float( x[i] );
      thisY = Py::Float( y[i] );
      thisY = height - thisY;  //flipy
      path.line_to(thisX, thisY);
    }
  }  
  
  
  if (! useDashes ) {
    
    agg::conv_stroke<agg::path_storage> stroke(path);
    stroke.line_cap(cap);
    stroke.line_join(join);
    stroke.width(lw);
    //freeze was here std::cout << "\t adding path!" << std::endl;         
    theRasterizer->add_path(stroke);
  }
  else {
    // set the dashes //TODO: scale for DPI
    
    size_t N = dashSeq.length();
    if (N%2 != 0  ) 
      throw Py::ValueError("dashes must be an even length sequence");     
    
    typedef agg::conv_dash<agg::path_storage> dash_t;
    dash_t dash(path);
    
    double on, off;
    
    for (size_t i=0; i<N/2; i+=1) {
      on = points_to_pixels_snapto(dashSeq[2*i]);
      off = points_to_pixels_snapto(dashSeq[2*i+1]);
      dash.add_dash(on, off);
    }
    agg::conv_stroke<dash_t> stroke(dash);
    stroke.line_cap(cap);
    stroke.line_join(join);
    stroke.width(lw);
    theRasterizer->add_path(stroke);
    
  }
  
  if ( isaa ) {
    theRenderer->color(color);    
    theRasterizer->render(*slineP8, *theRenderer); 
  }
  else {
    rendererBin->color(color);    
    theRasterizer->render(*slineBin, *rendererBin); 
  }
  
  return Py::Object();
  
}

Py::Object
RendererAgg::draw_text(const Py::Tuple& args) {
  if (debug)
    std::cout << "RendererAgg::draw_text" << std::endl;
  
  args.verify_length(4);
  
  
  FT2FontObject *font = (FT2FontObject *)args[0].ptr();
  
  int x = Py::Int( args[1] );
  int y = Py::Int( args[2] );
  Py::Object gc = args[3];

  Py::Object o ( gc.getAttr( "_cliprect" ) );

  bool useClip = o.ptr()!=Py_None;
  double l = 0;
  double b = 0;
  double r = width;
  double t = height;
  if (useClip) {
    Py::SeqBase<Py::Object> rect( o );
  
    l = Py::Float(rect[0]) ; 
    b = Py::Float(rect[1]) ; 
    double w = Py::Float(rect[2]) ; 
    double h = Py::Float(rect[3]) ; 
    r = l+w;
    t = b+h;
    //std::cout << b << " " << h << " " << " " << t << std::endl;
  }

  agg::rgba color = get_color(gc);  
  pixfmt::color_type p;
  p.r = int(255*color.r); p.b = int(255*color.b); 
  p.g = int(255*color.g); p.a = int(255*color.a);
   
  //y = y-font->image.height;
  unsigned thisx, thisy;
  
  for (size_t i=0; i<font->image.width; ++i) {
    for (size_t j=0; j<font->image.height; ++j) {
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
  theRenderer->color(edgecolor);
  //self->theRasterizer->gamma(agg::gamma_power(gamma));
  theRasterizer->add_path(stroke);
  theRasterizer->render(*slineP8, *theRenderer);  
  
  */

  return Py::Object();
  
}

Py::Object 
RendererAgg::draw_image(const Py::Tuple& args) {
  if (debug)
    std::cout << "RendererAgg::draw_image" << std::endl;
  args.verify_length(3);
  
  int x = Py::Int(args[0]);
  int y = Py::Int(args[1]);
  ImageObject *image = (ImageObject *)args[2].ptr();
  
  
  //todo: handle x and y
  agg::rect r(0, 0, image->rowsOut, image->colsOut);
  
  rendererBase->copy_from(*image->rbufOut, &r, x, y);
  
  return Py::Object();
  
}


Py::Object 
RendererAgg::write_rgba(const Py::Tuple& args) {
  if (debug)
    std::cout << "RendererAgg::write_rgba" << std::endl;
  
  args.verify_length(1);  
  std::string fname = Py::String( args[0]);
  
  std::ofstream of2( fname.c_str(), std::ios::binary|std::ios::out);
  for (size_t i=0; i<NUMBYTES; ++i) {
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
  if (debug)
    std::cout << "RendererAgg::write_png" << std::endl;
  
  args.verify_length(1);
  
  std::string fileName = Py::String(args[0]);
  const char *file_name = fileName.c_str();
  FILE *fp;
  png_structp png_ptr;
  png_infop info_ptr;
  struct        png_color_8_struct sig_bit;
  png_uint_32 row;
  
  png_bytep row_pointers[height];
  for (row = 0; row < height; ++row) {
    row_pointers[row] = pixBuffer + row * width * 4;
  }
  
  fp = fopen(file_name, "wb");
  if (fp == NULL) 
    throw Py::RuntimeError("could not open file");
  
  
  png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (png_ptr == NULL) {
    fclose(fp);
    throw Py::RuntimeError("could not create write struct");
  }
  
  info_ptr = png_create_info_struct(png_ptr);
  if (info_ptr == NULL) {
    fclose(fp);
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    throw Py::RuntimeError("could not create info struct");
  }
  
  if (setjmp(png_ptr->jmpbuf)) {
    fclose(fp);
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
    throw Py::RuntimeError("error building image");
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
  png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
  fclose(fp);
  
  return Py::Object();
}

Py::Object 
RendererAgg::tostring_rgb(const Py::Tuple& args) {
  //"Return the rendered buffer as an RGB string";
  
  if (debug)
    std::cout << "RendererAgg::tostring_rgb" << std::endl;
  
  args.verify_length(0);    
  int row_len = width*3;
  unsigned char* buf_tmp = 
    new unsigned char[row_len * height];
  agg::rendering_buffer renderingBufferTmp;
  renderingBufferTmp.attach(buf_tmp, 
			    width, 
			    height, 
			    row_len);
  
  color_conv(&renderingBufferTmp, renderingBuffer, agg::color_conv_rgba32_to_rgb24());
  
  
  //todo: how to do this with native CXX
  return Py::Object(Py_BuildValue("s#", 
				  buf_tmp, 
				  row_len * height));
  //len = row_len * height
  //std::string s(buf_tmp);
  //return Py::String(buf_tmp, row_len * height);
}

agg::rgba
RendererAgg::get_color(const Py::Object& gc) {
  
  if (debug)
    std::cout << "RendererAgg::get_color" << std::endl;
  
  Py::Tuple rgb = Py::Tuple( gc.getAttr("_rgb") );
  
  double alpha = Py::Float( gc.getAttr("_alpha") );
  
  double r = Py::Float(rgb[0]);
  double g = Py::Float(rgb[1]);
  double b = Py::Float(rgb[2]);
  return agg::rgba(r, g, b, alpha); 
  
}

agg::gen_stroke::line_cap_e
RendererAgg::get_linecap(const Py::Object& gc) {
  if (debug)
    std::cout << "RendererAgg::get_linecap" << std::endl;
  
  std::string capstyle = Py::String( gc.getAttr( "_capstyle" ) );
  
  if (capstyle=="butt") 
    return agg::gen_stroke::butt_cap;
  else if (capstyle=="round") 
    return agg::gen_stroke::round_cap;
  else if(capstyle=="projecting") 
    return agg::gen_stroke::square_cap;
  else 
    throw Py::ValueError("GC _capstyle attribute must be one of butt, round, projecting");
  
}

agg::gen_stroke::line_join_e
RendererAgg::get_joinstyle(const Py::Object& gc) {
  if (debug)
    std::cout << "RendererAgg::get_joinstyle" << std::endl;
  
  std::string joinstyle = Py::String( gc.getAttr("_joinstyle") );
  
  if (joinstyle=="miter") 
    return agg::gen_stroke::miter_join;
  else if (joinstyle=="round") 
    return agg::gen_stroke::round_join;
  else if(joinstyle=="bevel") 
    return agg::gen_stroke::bevel_join;
  else 
    throw Py::ValueError("GC _joinstyle attribute must be one of butt, round, projecting");
  
}

Py::Tuple
RendererAgg::get_dashes(const Py::Object& gc) {
  //return the dashOffset, dashes sequence tuple.  
  if (debug)
    std::cout << "RendererAgg::get_dashes" << std::endl;
  
  Py::Tuple _dashes = gc.getAttr("_dashes");
  
  size_t N = _dashes.length();
  
  if (N!=2) 
    throw Py::ValueError("GC _dashes must be a length 2 tuple");    
  
  return _dashes;
}


agg::rgba
RendererAgg::rgb_to_color(const Py::SeqBase<Py::Object>& rgb, double alpha) {
  if (debug)
    std::cout << "RendererAgg::rgb_to_color" << std::endl;
  
  double r = Py::Float(rgb[0]);
  double g = Py::Float(rgb[1]);
  double b = Py::Float(rgb[2]);
  return agg::rgba(r, g, b, alpha); 
  
}

void
RendererAgg::set_clip_rectangle( const Py::Object& gc) {
  //set the clip rectangle from the gc
  
  if (debug)
    std::cout << "RendererAgg::set_clip_rectangle" << std::endl;
  
  Py::Object o ( gc.getAttr( "_cliprect" ) );
  
  if (o.ptr()==Py_None) {
    // set clipping to false and return success
    theRasterizer->reset_clipping();
    return;
  }
  
  Py::SeqBase<Py::Object> rect( o );
  
  double l = Py::Float(rect[0]) ; 
  double b = Py::Float(rect[1]) ; 
  double w = Py::Float(rect[2]) ; 
  double h = Py::Float(rect[3]) ; 
  
  theRasterizer->clip_box(l, height-(b+h),
			  l+w, height-b);
}




int
RendererAgg::antialiased(const Py::Object& gc) {
  //return 1 if gc is antialiased
  if (debug)
    std::cout << "RendererAgg::antialiased" << std::endl;
  int isaa = Py::Int( gc.getAttr( "_antialiased") );
  return isaa;
}

double
RendererAgg::points_to_pixels_snapto(const Py::Object& points) {
  // convert a value in points to pixels depending on renderer dpi and
  // screen pixels per inch
  // snap return pixels to grid
  if (debug)
    std::cout << "RendererAgg::points_to_pixels_snapto" << std::endl;
  double p = Py::Float( points ) ;
  //return (int)(p*PIXELS_PER_INCH/72.0*dpi/72.0)+0.5;
  return (int)(p*dpi/72.0)+0.5;
  
  
}

double
RendererAgg::points_to_pixels( const Py::Object& points) {
  if (debug)
    std::cout << "RendererAgg::points_to_pixels" << std::endl;
  double p = Py::Float( points ) ;
  //return p * PIXELS_PER_INCH/72.0*dpi/72.0;
  return p * dpi/72.0;
}

RendererAgg::~RendererAgg() {
  
  if (debug)
    std::cout << "RendererAgg::~RendererAgg" << std::endl;
  
  
  delete slineP8;
  delete slineBin;
  delete theRasterizer;
  delete rendererBin;
  delete rendererBase;
  delete pixFmt;
  delete renderingBuffer;
  delete [] pixBuffer;
  
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
  
}

extern "C"
DL_EXPORT(void)
  init_backend_agg(void)
{
  //suppress unused warning by creating in two lines
  static _backend_agg_module* _backend_agg = NULL;
  _backend_agg = new _backend_agg_module;
  
};





