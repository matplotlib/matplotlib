// font and text handling was lifted almost verbatim from module
// paint.  The paint license is in LICENSE_PAINT that accompanies the
// matplotlib src distribution

#include <cstring>
#include "_backend_agg.h"

static char agg_font__doc__[] =
"Font(filename, size = 12, rotate = 0)\n"
"\n"
"Load the named font and return the font object";

static PyObject *agg_font(PyObject *module, PyObject *args)
{
    return font_new(args);
}



// image renderers that also have to deal with points (72/inch) make
// an assumption about how many pixels represent 1 inch.  GD and paint
// use 96.  What's good for the goose ...
#define PIXELS_PER_INCH 96

double _seqitem_as_double(PyObject *seq, size_t i) {
  //give a py sequence, return the ith element as a double; all memory
  //handling is done internally so caller does not need to delete any
  //memory
  PyObject *o1, *o2; 
  double val;
  o1 = PySequence_GetItem( seq, i);
  o2 = PyNumber_Float( o1 );
  Py_XDECREF(o1);
  
  val = PyFloat_AsDouble( o2);
  Py_XDECREF(o2);
  
  return val;
}

double* _pyobject_as_double(PyObject *o) {
  // convert a pyobect to a double.  Return NULL on error but do not
  // set err string; caller must delete the memory on non null
  PyObject *tmp; 
  double val;

  tmp = PyNumber_Float( o );

  if (tmp==NULL) return NULL;

  val = PyFloat_AsDouble( tmp );
  Py_XDECREF(tmp);
  
  return new double(val);
}

double 
_points_to_pixels(RendererAggObject* renderer, double pt) {
  // convert a value in points to pixels depending on renderer dpi and
  // scrren pixels per inch
  return pt*PIXELS_PER_INCH/72.0*renderer->dpi/72.0;

}

agg::gen_stroke::line_cap_e
_gc_get_linecap(PyObject *gc) {

  PyObject *capstyle;
  capstyle = PyObject_GetAttrString( gc, "_capstyle");

  if (capstyle==NULL) {
    PyErr_SetString(PyExc_TypeError, 
		    "Could not find the GC _capstyle attribute");
    return agg::gen_stroke::butt_cap;
  }

  if (! PyString_Check(capstyle)) {
    PyErr_SetString(PyExc_TypeError, 
		    " GC _capstyle attribute must be string");
    return agg::gen_stroke::butt_cap;
  }

  char *s = PyString_AsString(capstyle);
  if (strcmp(s, "butt")==0) {
    return agg::gen_stroke::butt_cap;
  }
  else if (strcmp(s, "round")==0) {
    return agg::gen_stroke::round_cap;
  }
  else if(strcmp(s, "projecting")==0) {
    return agg::gen_stroke::square_cap;
  }
  else {
    PyErr_SetString(PyExc_ValueError, 
		    " GC _capstyle attribute must be one of butt, round, projecting");
    return agg::gen_stroke::butt_cap;
  }
  return agg::gen_stroke::butt_cap;

}



agg::gen_stroke::line_join_e
_gc_get_joinstyle(PyObject *gc) {

  PyObject *joinstyle;
  joinstyle = PyObject_GetAttrString( gc, "_joinstyle");
  
  
  if (joinstyle==NULL) {
    PyErr_SetString(PyExc_TypeError, 
		    "Could not find the GC _joinstyle attribute");
    return agg::gen_stroke::miter_join;
  }

  if (! PyString_Check(joinstyle)) {
    PyErr_SetString(PyExc_TypeError, 
		    " GC _joinstyle attribute must be string");
    return agg::gen_stroke::miter_join;
  }

  char *s = PyString_AsString(joinstyle);
  if (strcmp(s, "miter")==0) {
    return agg::gen_stroke::miter_join;
  }
  else if (strcmp(s, "round")==0) {
    return agg::gen_stroke::round_join;
  }
  else if(strcmp(s, "bevel")==0) {
    return agg::gen_stroke::bevel_join;
  }
  else {
    PyErr_SetString(PyExc_ValueError, 
		    " GC _joinstyle attribute must be one of butt, round, projecting");
    return agg::gen_stroke::miter_join;
  }

    return agg::gen_stroke::miter_join;
}

agg::rgba* 
_gc_get_color(PyObject *gc) {
  //get a pointer to an agg color arg, return NULL and set error string on error
  PyObject *rgb;
  PyObject *alphao;
  rgb = PyObject_GetAttrString( gc, "_rgb");
  if (rgb==NULL) {
    PyErr_SetString(PyExc_TypeError, 
		    "Could not find the GC _rgb attribute");
    return NULL;    
  }

  alphao = PyObject_GetAttrString( gc, "_alpha");
  if (alphao==NULL) {
    PyErr_SetString(PyExc_TypeError, 
		    "Could not find the GC _alpha attribute");
    return NULL;    
  }

  double r, g, b, *palpha;

  int N;
  N = PySequence_Length(rgb);

  if (N==-1) {
    PyErr_SetString(PyExc_ValueError, 
		    "GC _rgb must be a sequence type");    
    return NULL;
  }
  
  if (N!=3) {
    PyErr_SetString(PyExc_ValueError, 
		    "GC _rgb must be a length 3 tuple");    
    return NULL;
  }

  r = _seqitem_as_double(rgb, 0);
  g = _seqitem_as_double(rgb, 1);
  b = _seqitem_as_double(rgb, 2);
  palpha = _pyobject_as_double(alphao);
  if (palpha==NULL) {
    PyErr_SetString(PyExc_TypeError, 
		    "Could not convert alpha to float");
    delete palpha;
    return NULL;
  }
    
  //printf("setting alpha %1.2f\n", *alpha);
  agg::rgba* color = new agg::rgba(r, g, b, *palpha); 
  delete palpha;
  return color;

}

PyObject *
_gc_get_dashes(PyObject *gc) {
  //return the dashOffset, dashes sequence tuple.  
  
  PyObject *_dashes;

  _dashes = PyObject_GetAttrString( gc, "_dashes");
  if (_dashes==NULL) {
    PyErr_SetString(PyExc_TypeError, 
		    "Could not find the GC _dashes attribute");
    return NULL;    
  }
  

  int N;
  N = PySequence_Length(_dashes);

  if (N==-1) {
    PyErr_SetString(PyExc_ValueError, 
		    "GC _dashes must be a sequence type");    
    return NULL;
  }
  
  if (N!=2) {
    PyErr_SetString(PyExc_ValueError, 
		    "GC _dashes must be a length 2 tuple");    
    return NULL;
  }

  return _dashes;
}

int 
_gc_set_clip_rect(PyObject *gc, RendererAggObject* renderer) {
  //set the clip rect.  If return is False, this function will set the
  //error string and the caller should return NULL to python
  PyObject *rect;

  rect = PyObject_GetAttrString( gc, "_cliprect");
  if (rect==NULL) {
    PyErr_SetString(PyExc_TypeError, 
		    "Could not find the GC _cliprect attribute");
    return 0;    
  }

  if (rect==Py_None) {
    // set clipping to false and return success
    renderer->ras->reset_clipping();
    return 1;
  }

  int N = PySequence_Length(rect);

  if (N==-1) {
    PyErr_SetString(PyExc_ValueError, 
		    "GC _rect must be a sequence type");    
    return 0;
  }


  
  if (N!=4) {
    PyErr_SetString(PyExc_ValueError, 
		    "GC _cliprect must be a length 4 sequence");    
    return 0;
  }


  float l, b, w, h;
  l = _seqitem_as_double(rect, 0);
  b = _seqitem_as_double(rect, 1);
  w = _seqitem_as_double(rect, 2);
  h = _seqitem_as_double(rect, 3);
  renderer->ras->clip_box(l, renderer->rbase->height()-(b+h),
			  l+w, renderer->rbase->height()-b);
  return 1;

}

double *
_gc_get_linewidth(PyObject *gc) {
  //get the linewdith.  If return is <0, this function will set the
  //error string and the caller should return NULL to python.  Caller
  //must delete memory on non null return
  PyObject *lwo;

  lwo = PyObject_GetAttrString( gc, "_linewidth");
  if (lwo==NULL) {
    PyErr_SetString(PyExc_TypeError, 
		    "Could not find the GC _linewidth attribute");
    return NULL;    
  }

  return _pyobject_as_double(lwo);

}



extern "C" staticforward PyTypeObject RendererAgg_Type;

#define RendererAggObject_Check(v)	((v)->ob_type == &RendererAgg_Type)

static RendererAggObject *
newRendererAggObject(PyObject *args)
{
  //printf("newRendererAggObject start\n");
  RendererAggObject *self;
  int width, height;
  double dpi;
  if (!PyArg_ParseTuple(args, "iid:RendererAgg", &width, &height, &dpi))
    return NULL;

  
  self = PyObject_New(RendererAggObject, &RendererAgg_Type);
  if (self == NULL)
    return NULL;

   
  unsigned stride(width*4);    //TODO, pixfmt call to make rgba type independet
  size_t NUMBYTES(width*height*4);
  agg::int8u *buffer = new agg::int8u[NUMBYTES];  
  
  self->rbuf = new agg::rendering_buffer;
  self->rbuf->attach(buffer, width, height, stride);
  self->sline = new scanline;
 
  self->pixf = new pixfmt(*self->rbuf);
  self->rbase = new renderer_base(*self->pixf);
  self->rbase->clear(agg::rgba(1, 1, 1));
  
  self->ren = new renderer(*self->rbase);
  self->ras = new rasterizer(); 
  self->buffer = buffer; 
  self->dpi = dpi; 
  self->NUMBYTES = NUMBYTES; 
  self->x_attr = NULL;
  
  return self;
}

static PyObject *
_backend_agg_new_renderer(PyObject *self, PyObject *args)
{
  RendererAggObject *rv;
  
  rv = newRendererAggObject(args);
  if ( rv == NULL )
    return NULL;
  return (PyObject *)rv;
}

static void
RendererAgg_dealloc(RendererAggObject *self)
{

  PyObject_Del(self);
  delete self->rbuf;
  delete self->pixf;
  delete self->rbase;
  delete self->ren;

  delete self->ras;
  delete self->buffer;

}


static PyObject *
RendererAgg_draw_ellipse(RendererAggObject *renderer, PyObject* args) {

  PyObject *gcEdge, *gcFace;
  float x,y,w,h;
  if (!PyArg_ParseTuple(args, "OOffff", &gcEdge, &gcFace, &x, &y, &w, &h))
    return NULL;

  if (! _gc_set_clip_rect(gcEdge, renderer)) return NULL;

  //last arg is num steps
  agg::ellipse path(x, renderer->rbase->height()-y, w, h, 100); 
  

  if (gcFace != Py_None) {
    //printf("Filling the face\n");
    //fill the face
    agg::rgba* color = _gc_get_color(gcFace);
    if (color==NULL) return NULL;
    renderer->ren->color(*color);
    renderer->ras->add_path(path);    
    renderer->ras->render(*renderer->sline, *renderer->ren);  
    delete color;
  }
  
  //now fill the edge

  double* plw = _gc_get_linewidth(gcEdge);
  if (plw==NULL) return NULL;
  double lw = _points_to_pixels(renderer, *plw);

  agg::conv_stroke<agg::ellipse> stroke(path);
  stroke.width(lw);
  agg::rgba* color = _gc_get_color(gcEdge);
  if (color==NULL) return NULL;
  renderer->ren->color(*color);
  //self->ras->gamma(agg::gamma_power(gamma));
  renderer->ras->add_path(stroke);
  renderer->ras->render(*renderer->sline, *renderer->ren);  
  delete color;
  Py_INCREF(Py_None);
  return Py_None;

}

char RendererAgg_draw_polygon__doc__[] = 
"draw_polygon(gcEdge, gdFace, points)\n"
"\n"
"Draw a polygon using the gd edge and face.  point is a sequence of x,y tuples";

static PyObject *
RendererAgg_draw_polygon(RendererAggObject *renderer, PyObject* args) {

  PyObject *gcEdge, *gcFace, *points;

  if (!PyArg_ParseTuple(args, "OOO", &gcEdge, &gcFace, &points))
    return NULL;
  if (! _gc_set_clip_rect(gcEdge, renderer)) return NULL;
  double* plw = _gc_get_linewidth(gcEdge);
  if (plw==NULL) return NULL;
  double lw = _points_to_pixels(renderer, *plw);
  delete plw;
  //printf("draw_rectangle inited\n");
  agg::path_storage path;

  PyObject *tup;      // the x,y tup
  PyObject *xo, *yo;  // xi, yi in tup
  PyObject *xf, *yf;  // xi, yi in tup as Py_Float
  double x, y;        // finally, the damned numbers

  int Npoints = PySequence_Length(points);

  if (Npoints==-1) {
    PyErr_SetString(PyExc_ValueError, 
		    "points must be a sequence type");    
    return NULL;
  }
  
  tup = PySequence_GetItem( points, 0);
  int N = PySequence_Length(tup);

  if (N!=2) {
    PyErr_SetString(PyExc_ValueError, 
		    "seq must be a sequence of length 2 tuples");    
    Py_XDECREF(tup);
    return NULL;
  }

  xo = PySequence_GetItem( tup, 0);
  yo = PySequence_GetItem( tup, 1);
  Py_XDECREF(tup);

  xf = PyNumber_Float( xo );
  yf = PyNumber_Float( yo );
  Py_XDECREF(xo); Py_XDECREF(yo);

  
  x = PyFloat_AsDouble(xf);
  y = PyFloat_AsDouble(yf);
  Py_XDECREF(xf);   Py_XDECREF(yf);
  
  y = renderer->rbase->height() - y;
  path.move_to(x, y);

  for (int i=1; i<Npoints; ++i) {

    tup = PySequence_GetItem(points, i);
    int N = PySequence_Length(tup);

    if (N!=2) {
      PyErr_SetString(PyExc_ValueError, 
		      "points must be a sequence of length 2 tuples");    
      Py_XDECREF(tup);
      return NULL;
    }

    xo = PySequence_GetItem( tup, 0);
    yo = PySequence_GetItem( tup, 1);
    Py_XDECREF(tup);

    xf = PyNumber_Float( xo );
    yf = PyNumber_Float( yo );
    Py_XDECREF(xo); Py_XDECREF(yo);

    x = PyFloat_AsDouble(xf);
    y = PyFloat_AsDouble(yf);
    Py_XDECREF(xf);   Py_XDECREF(yf);
  
    y = renderer->rbase->height() - y;
    path.line_to(x, y);
    
  }
  path.close_polygon();

  
  if (gcFace != Py_None) {
    //fill the face
    agg::rgba* color = _gc_get_color(gcFace);
    if (color==NULL) return NULL;
    renderer->ren->color(*color);
    renderer->ras->add_path(path);    
    renderer->ras->render(*renderer->sline, *renderer->ren);  
    delete color;
  }
  
  //now fill the edge
  agg::conv_stroke<agg::path_storage> stroke(path);
  stroke.width(lw);
  //printf("setting edge wid %1.2f\n", lw);
  agg::rgba* color = _gc_get_color(gcEdge);
  if (color==NULL) return NULL;
  renderer->ren->color(*color);
  //self->ras->gamma(agg::gamma_power(gamma));
  renderer->ras->add_path(stroke);
  renderer->ras->render(*renderer->sline, *renderer->ren);  
  delete color;
  Py_INCREF(Py_None);
  return Py_None;

}

static PyObject *
RendererAgg_draw_rectangle(RendererAggObject *renderer, PyObject* args) {

  PyObject *gcEdge, *gcFace;
  float l,b,w,h;
  if (!PyArg_ParseTuple(args, "OOffff", &gcEdge, &gcFace, &l, &b, &w, &h))
    return NULL;
  if (! _gc_set_clip_rect(gcEdge, renderer)) return NULL;
  double* plw = _gc_get_linewidth(gcEdge);
  if (plw==NULL) return NULL;
  double lw = _points_to_pixels(renderer, *plw);
  delete plw;
  //printf("draw_rectangle inited\n");
  agg::path_storage path;

  //printf("draw_rectangle path created: %f, %f, %f, %f\n", l, b, w, h);
  b = renderer->rbase->height() - (b+h);
  path.move_to(l, b+h);
  path.line_to(l+w, b+h);
  path.line_to(l+w, b);
  path.line_to(l, b);
  path.close_polygon();
  //printf("draw_rectangle path built\n");

  
  if (gcFace != Py_None) {
    //fill the face
    agg::rgba* color = _gc_get_color(gcFace);
    if (color==NULL) return NULL;
    renderer->ren->color(*color);
    renderer->ras->add_path(path);    
    renderer->ras->render(*renderer->sline, *renderer->ren);  
    delete color;
  }
  
  //now fill the edge
  agg::conv_stroke<agg::path_storage> stroke(path);
  stroke.width(lw);
  //printf("setting edge wid %1.2f\n", lw);
  agg::rgba* color = _gc_get_color(gcEdge);
  if (color==NULL) return NULL;
  renderer->ren->color(*color);
  //self->ras->gamma(agg::gamma_power(gamma));
  renderer->ras->add_path(stroke);
  renderer->ras->render(*renderer->sline, *renderer->ren);  
  delete color;
  Py_INCREF(Py_None);
  return Py_None;

}


static PyObject *
RendererAgg_draw_lines(RendererAggObject *renderer, PyObject* args) {

  PyObject *gc;
  PyObject *x, *y;


  if (!PyArg_ParseTuple(args, "OOO", &gc, &x, &y))
    return NULL;

  if (! _gc_set_clip_rect(gc, renderer)) return NULL;
  int Nx, Ny;
  Nx = PySequence_Length(x);
  if (Nx==-1) {
    PyErr_SetString(PyExc_ValueError, 
		    "x must be a sequence type");    
    return NULL;
  }

  Ny = PySequence_Length(y);
  if (Ny==-1) {
    PyErr_SetString(PyExc_ValueError, 
		    "y must be a sequence type");    
    return NULL;
  }


  if (Nx!=Ny) {
    PyErr_SetString(PyExc_ValueError, 
		    "x and y must be equal length sequences");
  }

  if (Nx<2) {
    PyErr_SetString(PyExc_ValueError, 
		    "x and y must have length >= 2");
  }

  //printf("RendererAgg_draw_lines looks ok\n");  

  agg::gen_stroke::line_cap_e cap = _gc_get_linecap(gc);
  agg::gen_stroke::line_join_e join = _gc_get_joinstyle(gc);

  double *plw = _gc_get_linewidth(gc);
  if (plw==NULL) return NULL;
  double lw = _points_to_pixels(renderer, *plw);
  delete plw;

  agg::rgba* color = _gc_get_color(gc);
  if (color==NULL) return NULL;


  // process the dashes
  PyObject *dashes = _gc_get_dashes(gc);
  if (dashes==NULL) return NULL;

  int useDashes;
  PyObject *val0 = PySequence_GetItem(dashes, 0);
  if (val0==Py_None) useDashes=0;
  else useDashes=1;
  Py_XDECREF(val0);

  double offset = 0;
  PyObject *dashSeq = NULL;
  if (useDashes) {
    //TODO: use offset
    offset = _points_to_pixels(renderer, _seqitem_as_double(dashes, 0));
    //note, you must decref this later if useDashes
    dashSeq = PySequence_GetItem(dashes, 1); 
  };
    
    

  agg::path_storage path;

  double thisX, thisY;	
  unsigned winHeight = renderer->rbase->height();
  thisX = _seqitem_as_double(x, 0);
  thisY = winHeight - _seqitem_as_double(y, 0);
  path.move_to(thisX, thisY);
  for (int i=1; i<Nx; ++i) {
    thisX = _seqitem_as_double(x, i);
    thisY = winHeight - _seqitem_as_double(y, i);
    path.line_to(thisX, thisY);
  }

  renderer->ren->color(*color);


  if (! useDashes ) {
    //printf("no dashes\n");
    agg::conv_stroke<agg::path_storage> stroke(path);
    stroke.width(1.0);
    stroke.line_cap(cap);
    stroke.line_join(join);
    stroke.width(lw);
    renderer->ras->add_path(stroke);
  }
  else {
    // set the dashes //TODO: scale for DPI
    //printf("dashes\n");
    int N(PySequence_Length(dashSeq));
    if (N==-1) {
      PyErr_SetString(PyExc_TypeError, 
		      "dashes must be None or a sequence");     
      Py_XDECREF(dashSeq);
      return NULL;      
    }
    if (N%2 != 0  ) {
      PyErr_SetString(PyExc_ValueError, 
		      "dashes must be an even length sequence");     
      Py_XDECREF(dashSeq);
      return NULL;      
    }



    typedef agg::conv_dash<agg::path_storage> dash_t;
    dash_t dash(path);
    agg::conv_stroke<dash_t> stroke(dash);
    double on, off;
    for (int i=0; i<N/2; i+=2) {
      on = _points_to_pixels(renderer,  _seqitem_as_double(dashSeq, 2*i));
      off = _points_to_pixels(renderer, _seqitem_as_double(dashSeq, 2*i+1));
      dash.add_dash(on, off);
      //printf("adding dashes %1.2f, %1.2f\n", on, off);
    }
    stroke.width(1.0);
    stroke.line_cap(cap);
    stroke.line_join(join);
    stroke.width(lw);
    renderer->ras->add_path(stroke);
    Py_XDECREF(dashSeq);
  }
    



  renderer->ras->render(*renderer->sline, *renderer->ren);  
  delete color;
  //printf("RendererAgg_draw_lines done\n");  

  Py_INCREF(Py_None);
  return Py_None;

}



static char RendererAgg_rgb__doc__[] =
"rgb(r, g, b)\n"
"\n"
"Create an rgba color value with a = 0xff.";

static PyObject *RendererAgg_rgb(PyObject *self, PyObject *args)
{
    int r, g, b;

    if (!PyArg_ParseTuple(args, "iii", &r, &g, &b))
	return NULL;

    return (PyObject*)PyInt_FromLong((r << 24) + (g << 16) + (b << 8) + 0xff);
}

static char RendererAgg_rgba__doc__[] =
"rgba(r, g, b, a)\n"
"\n"
"Create an rgba color value.";

static PyObject *RendererAgg_rgba(PyObject *self, PyObject *args)
{
    int r, g, b, a;

    if (!PyArg_ParseTuple(args, "iiii", &r, &g, &b, &a))
	return NULL;

    return (PyObject*)PyInt_FromLong((r << 24) + (g << 16) + (b << 8) + a);
}

char RendererAgg_draw_text__doc__[] = 
"draw_text(font, x, y, textcolor, text)\n"
"\n"
"Render the text in the supplied font at the specified location";

PyObject *
RendererAgg_draw_text(RendererAggObject *renderer, PyObject* args) {

  return font_draw_text(renderer, args);
}

static PyObject *
RendererAgg_write_rgba(RendererAggObject *renderer, PyObject* args) {
  //printf("save buffer called\n");

  PyObject *fnameo = NULL;
  char *fname = NULL;
  if (!PyArg_ParseTuple(args, "O", &fnameo))
    return NULL;


  fname = PyString_AsString(fnameo);

  std::ofstream of2( fname, std::ios::binary|std::ios::out);
  //printf("About to write %d bytes\n", renderer->NUMBYTES);
  for (size_t i=0; i<renderer->NUMBYTES; ++i) {
    of2.write((char*)&(renderer->buffer[i]), sizeof(char));
  }
  //printf("save buffer wrote\n");
  Py_INCREF(Py_None);
  return Py_None;

}


// this code is heavily adapted from the paint license, which is in
// the file paint.license (BSD compatible) included in this
// distribution.  TODO, add license file to MANIFEST.in and CVS
static PyObject *
RendererAgg_write_png(RendererAggObject *renderer, PyObject *args)
{
    char *file_name;
    FILE *fp;
    png_structp png_ptr;
    png_infop info_ptr;
    struct        png_color_8_struct sig_bit;
    png_uint_32 row, height, width;

    if (!PyArg_ParseTuple(args, "s", &file_name))
	return NULL;

    height = renderer->rbase->height();
    width = renderer->rbase->width();
    png_bytep row_pointers[height];
    for (row = 0; row < height; ++row) {
      row_pointers[row] = renderer->buffer + row * width * 4;
    }

    fp = fopen(file_name, "wb");
    if (fp == NULL) {
	PyErr_SetString(PyExc_IOError, "could not open file");
	return NULL;
    }


    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png_ptr == NULL) {
	fclose(fp);
	PyErr_SetString(PyExc_RuntimeError, "could not create write struct");
	return NULL;
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL) {
	fclose(fp);
	png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
	PyErr_SetString(PyExc_RuntimeError, "could not create info struct");
	return NULL;
    }

    if (setjmp(png_ptr->jmpbuf)) {
	fclose(fp);
	png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
	PyErr_SetString(PyExc_RuntimeError, "error building image");
	return NULL;
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

    Py_INCREF(Py_None);
    return Py_None;
}


// must be defined before getattr
static PyMethodDef RendererAgg_methods[] = {

  { "draw_ellipse",	(PyCFunction)RendererAgg_draw_ellipse,	 METH_VARARGS},
  { "draw_rectangle",	(PyCFunction)RendererAgg_draw_rectangle, METH_VARARGS},
  { "draw_polygon",	(PyCFunction)RendererAgg_draw_polygon, METH_VARARGS},
  { "draw_lines",	(PyCFunction)RendererAgg_draw_lines,	 METH_VARARGS},
  { "draw_text",	(PyCFunction)RendererAgg_draw_text,	 METH_VARARGS, RendererAgg_draw_text__doc__},
  { "write_rgba",	(PyCFunction)RendererAgg_write_rgba,	 METH_VARARGS},
  { "write_png",	(PyCFunction)RendererAgg_write_png,	 METH_VARARGS},
  { "rgb",              (PyCFunction)RendererAgg_rgb, METH_VARARGS, RendererAgg_rgb__doc__ },
  { "rgba",             (PyCFunction)RendererAgg_rgba, METH_VARARGS, RendererAgg_rgba__doc__ },

  {NULL,		NULL}		/* sentinel */
};




static PyObject *
RendererAgg_getattr(RendererAggObject *self, char *name)
{
  if (self->x_attr != NULL) {
    PyObject *v = PyDict_GetItemString(self->x_attr, name);
    if (v != NULL) {
      Py_INCREF(v);
      return v;
    }
  }
  return Py_FindMethod(RendererAgg_methods, (PyObject *)self, name);
}


static int
RendererAgg_setattr(RendererAggObject *self, char *name, PyObject *v)
{
  if (self->x_attr == NULL) {
    self->x_attr = PyDict_New();
    if (self->x_attr == NULL)
      return -1;
  }
  if (v == NULL) {
    int rv = PyDict_DelItemString(self->x_attr, name);
    if (rv < 0)
      PyErr_SetString(PyExc_AttributeError,
		      "delete non-existing RendererAgg attribute");
    return rv;
  }
  else
    return PyDict_SetItemString(self->x_attr, name, v);
}


static PyTypeObject RendererAgg_Type = {
  /* The ob_type field must be initialized in the module init function
   * to be portable to Windows without using C++. */
  PyObject_HEAD_INIT(NULL)
  0,			/*ob_size*/
  "_backend_agg.RendererAgg",		/*tp_name*/
  sizeof(RendererAggObject),	/*tp_basicsize*/
  0,			/*tp_itemsize*/
  /* methods */
  (destructor)RendererAgg_dealloc, /*tp_dealloc*/
  0,			/*tp_print*/
  (getattrfunc)RendererAgg_getattr, /*tp_getattr*/
  (setattrfunc)RendererAgg_setattr, /*tp_setattr*/
  0,			/*tp_compare*/
  0,			/*tp_repr*/
  0,			/*tp_as_number*/
  0,			/*tp_as_sequence*/
  0,			/*tp_as_mapping*/
  0,			/*tp_hash*/
  0,                      /*tp_call*/
  0,                      /*tp_str*/
  0,                      /*tp_getattro*/
  0,                      /*tp_setattro*/
  0,                      /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT,     /*tp_flags*/
  0,                      /*tp_doc*/
  0,                      /*tp_traverse*/
  0,                      /*tp_clear*/
  0,                      /*tp_richcompare*/
  0,                      /*tp_weaklistoffset*/
  0,                      /*tp_iter*/
  0,                      /*tp_iternext*/
  0,                      /*tp_methods*/
  0,                      /*tp_members*/
  0,                      /*tp_getset*/
  0,                      /*tp_base*/
  0,                      /*tp_dict*/
  0,                      /*tp_descr_get*/
  0,                      /*tp_descr_set*/
  0,                      /*tp_dictoffset*/
  0,                      /*tp_init*/
  0,                      /*tp_alloc*/
  0,                      /*tp_new*/
  0,                      /*tp_free*/
  0,                      /*tp_is_gc*/
};




/* --------------------------------------------------------------------- */





static PyMethodDef _backend_agg_methods[] = {
  {"RendererAgg",	_backend_agg_new_renderer,      METH_VARARGS},
  { "Font",             (PyCFunction)agg_font,          METH_VARARGS, agg_font__doc__ },
  {NULL,		NULL}		/* sentinel */
};


extern "C"
DL_EXPORT(void)
  init_backend_agg(void)
{
  PyObject *module, *d;
  
  /* Initialize the type of the new type object here; doing it here
   * is required for portability to Windows without requiring C++. */
  RendererAgg_Type.ob_type = &PyType_Type;
  Font_Type.ob_type = &PyType_Type;
  
  /* Create the module and add the functions */
  module = Py_InitModule("_backend_agg", _backend_agg_methods);
  
  /* Add some symbolic constants to the module */
  d = PyModule_GetDict(module);
  ErrorObject = PyErr_NewException("_backend_agg.error", NULL, NULL);
  PyDict_SetItemString(d, "error", ErrorObject);
}
