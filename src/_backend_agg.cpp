#include <fstream>
#include <cmath>
#include <cstdio>

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
#include "agg_rasterizer_outline.h"
#include "agg_rasterizer_scanline_aa.h"
#include "agg_renderer_outline_aa.h"
#include "agg_renderer_raster_text.h"
#include "agg_renderer_scanline.h"
#include "agg_rendering_buffer.h"
#include "agg_scanline_p32.h"

#include "Python.h"

typedef agg::pixel_formats_rgb24<agg::order_bgr24> pixfmt;
typedef agg::renderer_base<pixfmt> renderer_base;
typedef agg::renderer_scanline_p_solid<renderer_base> renderer;
typedef agg::rasterizer_scanline_aa<> rasterizer;
typedef agg::scanline_p8 scanline;


static PyObject *ErrorObject;

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
  rasterizer *ras;
  agg::int8u *buffer;
  scanline *sline;
  size_t NUMBYTES;  //the number of bytes in buffer

} RendererAggObject;

double _seqitem_as_double(PyObject *seq, size_t i) {
  //give a py sequence, return the ith element as a double
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
  // set err string
  PyObject *tmp; 
  double val;

  tmp = PyNumber_Float( o );

  if (tmp==NULL) return NULL;

  val = PyFloat_AsDouble( tmp );
  Py_XDECREF(tmp);
  
  return new double(val);
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

  double r, g, b, *alpha;

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
  alpha = _pyobject_as_double(alphao);
  if (alpha==NULL) {
    PyErr_SetString(PyExc_TypeError, 
		    "Could not convert alpha to float");
    delete alpha;
    return NULL;

  }
  printf("setting alpha %1.2f\n", *alpha);
  agg::rgba* color = new agg::rgba(b, g, r, *alpha); 
  delete alpha;
  return color;

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
  printf("setting clip rectangle %1.1f, %1.1f, %1.1f, %1.1f\n",
	 l,renderer->rbase->height()-(b+h),w,h);
  renderer->ras->clip_box(l, renderer->rbase->height()-(b+h),
			  l+w, renderer->rbase->height()-b);
  return 1;

}



extern "C" staticforward PyTypeObject RendererAgg_Type;


#define RendererAggObject_Check(v)	((v)->ob_type == &RendererAgg_Type)

static RendererAggObject *
newRendererAggObject(PyObject *args)
{
  RendererAggObject *self;
  int width, height;
  if (!PyArg_ParseTuple(args, "ii:RendererAgg", &width, &height))
    return NULL;
  
  self = PyObject_New(RendererAggObject, &RendererAgg_Type);
  if (self == NULL)
    return NULL;
   
  unsigned stride(width*3);
  size_t NUMBYTES(width*height*3);
  agg::int8u *buffer = new agg::int8u[NUMBYTES];  //TODO: heap vs stack?
  
  self->rbuf = new agg::rendering_buffer;
  self->rbuf->attach(buffer, width, height, stride);
  self->sline = new scanline;
 
  self->pixf = new pixfmt(*self->rbuf);
  self->rbase = new renderer_base(*self->pixf);
  self->rbase->clear(agg::rgba(1, 1, 1));
  
  self->ren = new renderer(*self->rbase);
  self->ras = new rasterizer(); 
  self->buffer = buffer; 
  self->NUMBYTES = NUMBYTES; 
  self->x_attr = NULL;

  //draw(self);  
  //print(self); //calling this makes agg.raw work ok.  Is is path?  Something 

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
  printf("deallocating renderer\n");
  PyObject_Del(self);
  delete self->rbuf;
  delete self->pixf;
  delete self->rbase;
  delete self->ren;

  delete self->ras;
  delete self->buffer;
  printf("deallocating renderer done\n");

}


static PyObject *
RendererAgg_draw_ellipse(RendererAggObject *renderer, PyObject* args) {

  PyObject *gcEdge, *gcFace;
  float x,y,w,h;
  if (!PyArg_ParseTuple(args, "OOffff", &gcEdge, &gcFace, &x, &y, &w, &h))
    return NULL;

  if (! _gc_set_clip_rect(gcEdge, renderer)) return NULL;

  agg::ellipse path(x, renderer->rbase->height()-(y+h), w, h, 100); //last arg is num steps
  

  if (gcFace != Py_None) {
    printf("Filling the face\n");
    //fill the face
    agg::rgba* color = _gc_get_color(gcFace);
    if (color==NULL) return NULL;
    renderer->ren->color(*color);
    renderer->ras->add_path(path);    
    renderer->ras->render(*renderer->sline, *renderer->ren);  
    delete color;
  }
  
  //now fill the edge
  agg::conv_stroke<agg::ellipse> stroke(path);
  stroke.width(1.0);
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
  stroke.width(1.0);
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
  PyObject *dashes;

  if (!PyArg_ParseTuple(args, "OOOO", &gc, &x, &y, &dashes))
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

  agg::rgba* color = _gc_get_color(gc);
  if (color==NULL) return NULL;
  renderer->ren->color(*color);


  if (dashes == Py_None) {
    //printf("no dashes\n");
    agg::conv_stroke<agg::path_storage> stroke(path);
    stroke.width(1.0);
    renderer->ras->add_path(stroke);
  }
  else {
    // set the dashes
    //printf("dashes\n");
    int N(PySequence_Length(dashes));
    if (N==-1) {
      PyErr_SetString(PyExc_TypeError, 
		      "dashes must be None or a sequence");     
      return NULL;      
    }
    if (N%2 != 0  ) {
      PyErr_SetString(PyExc_ValueError, 
		      "dashes must be an even length sequence");     
      return NULL;      
    }



    typedef agg::conv_dash<agg::path_storage> dash_t;
    dash_t dash(path);
    agg::conv_stroke<dash_t> stroke(dash);
    double on, off;
    for (int i=0; i<N/2; i+=2) {
      on = _seqitem_as_double(dashes, 2*i);
      off = _seqitem_as_double(dashes, 2*i+1);
      dash.add_dash(on, off);
      //printf("adding dashes %1.2f, %1.2f\n", on, off);
    }
    stroke.width(1.0);
    renderer->ras->add_path(stroke);
  }
    



  renderer->ras->render(*renderer->sline, *renderer->ren);  
  delete color;

  //printf("RendererAgg_draw_lines done\n");  

  Py_INCREF(Py_None);
  return Py_None;

}


static PyObject *
RendererAgg_save_buffer(RendererAggObject *renderer, PyObject* args) {
  //printf("save buffer called\n");

  PyObject *fnameo = NULL;
  char *fname = NULL;
  if (!PyArg_ParseTuple(args, "O", &fnameo))
    return NULL;


  fname = PyString_AsString(fnameo);

  //printf("save buffer ready\n");
  //scanline sline;
  //renderer->ras->render(renderer->sline, *renderer->ren);
  //printf("save buffer rendered\n");

  std::ofstream of2( fname, std::ios::binary|std::ios::out);
  printf("About to write %d bytes\n", renderer->NUMBYTES);
  for (size_t i=0; i<renderer->NUMBYTES; ++i) {
    of2.write((char*)&(renderer->buffer[i]), sizeof(char));
  }
  //printf("save buffer wrote\n");
  Py_INCREF(Py_None);
  return Py_None;

}

// must be defined before getattr
static PyMethodDef RendererAgg_methods[] = {
  {"draw_ellipse",	(PyCFunction)RendererAgg_draw_ellipse,	METH_VARARGS},
  {"draw_rectangle",	(PyCFunction)RendererAgg_draw_rectangle,	METH_VARARGS},
  {"draw_lines",	(PyCFunction)RendererAgg_draw_lines,	METH_VARARGS},
  {"_save_buffer",	(PyCFunction)RendererAgg_save_buffer,	METH_VARARGS},
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
  
  /* Create the module and add the functions */
  module = Py_InitModule("_backend_agg", _backend_agg_methods);
  
  /* Add some symbolic constants to the module */
  d = PyModule_GetDict(module);
  ErrorObject = PyErr_NewException("_backend_agg.error", NULL, NULL);
  PyDict_SetItemString(d, "error", ErrorObject);
}
