#include <fstream>
#include <cmath>
#include <cstdio>
#include "Python.h"
#include "Numeric/arrayobject.h"  //after Python.h

#include "agg_pixfmt_rgb24.h"
#include "agg_pixfmt_rgba32.h"
#include "agg_rendering_buffer.h"
#include "agg_rasterizer_scanline_aa.h"
#include "agg_path_storage.h"
#include "agg_affine_matrix.h"
#include "agg_conv_transform.h"
#include "agg_span_image_filter_rgb24.h"
#include "agg_span_image_filter_rgba32.h"
#include "agg_span_interpolator_linear.h"
#include "agg_scanline_u8.h"
#include "agg_scanline_p32.h"
#include "agg_renderer_scanline.h"

#include "image.h"





#define MODULE_CONSTANT(d, name, val) \
{ \
PyObject *tmp = PyInt_FromLong((long)val);\
PyDict_SetItemString(d, name, tmp);\
Py_DECREF(tmp);\
}

static PyObject *ErrorObject;

typedef agg::pixel_formats_rgba32<agg::order_rgba32> pixfmt;
typedef agg::renderer_base<pixfmt> renderer_base;
typedef agg::span_interpolator_linear<> interpolator_type;
typedef agg::span_image_filter_rgba32_bilinear<agg::order_rgba32, interpolator_type> span_gen_type;
typedef agg::renderer_scanline_u<renderer_base, span_gen_type> renderer_type;
typedef agg::rasterizer_scanline_aa<> rasterizer;


extern "C" static PyTypeObject Image_Type;
#define ImageObject_Check(v)	((v)->ob_type == &Image_Type)

enum { BICUBIC=0, BILINEAR, BLACKMAN100, BLACKMAN256, BLACKMAN64, 
       NEAREST, SINC144, SINC256, SINC64, SPLINE16, SPLINE36};
enum { ASPECT_PRESERVE=0, ASPECT_FREE};




char Image_resize__doc__[] = 
"resize(width, height)\n"
"\n"
"Resize the image to width, height using interpolation"
;

static PyObject *
Image_resize(ImageObject *image, PyObject* args) {

  if (image->bufferIn ==NULL) {
    PyErr_SetString(PyExc_RuntimeError, "You must first load the image"); 
    return NULL;
  }

  int width, height;

  if (!PyArg_ParseTuple(args, "ii", &width, &height))
    return NULL;

  image->widthOut  = width;
  image->heightOut = height;

  size_t NUMBYTES(width * height * image->BPP);
  agg::int8u *buffer = new agg::int8u[NUMBYTES];  
  image->rbufOut = new agg::rendering_buffer;
  image->rbufOut->attach(buffer, width, height, width * image->BPP);
  
  // init the output rendering/rasterizing stuff
  pixfmt pixf(*image->rbufOut);
  renderer_base rb(pixf);
  rb.clear(agg::rgba(1, 1, 1));
  agg::rasterizer_scanline_aa<> ras;
  agg::scanline_u8 sl;

  // compute the resizing matrix, with or w/o constrained aspect
  // ratio
  agg::affine_matrix resizingMatrix;

  double sx = float(image->widthOut)/image->widthIn;
  double sy = float(image->heightOut)/image->heightIn;
  
  switch(image->aspect) {
  case ASPECT_PRESERVE:
      {
	if(sy < sx) sx = sy;
	resizingMatrix = agg::scaling_matrix(sx, sx);	      
	break;
      }
  case ASPECT_FREE:     
    {
      resizingMatrix = agg::scaling_matrix(sx, sy); 
      break;
    }
  default:
    PyErr_SetString(PyExc_ValueError, "Aspect constant not recognized"); 
    return NULL;
  }
  
  agg::affine_matrix srcMatrix;
  srcMatrix *= resizingMatrix;
  agg::affine_matrix imageMatrix;
  imageMatrix *= resizingMatrix;
  imageMatrix.invert();
  interpolator_type interpolator(imageMatrix); 

  // the image path
  agg::path_storage path;
  path.move_to(0, 0);
  path.line_to(image->widthIn, 0);
  path.line_to(image->widthIn, image->heightIn);
  path.line_to(0, image->heightIn);
  path.close_polygon();
  agg::conv_transform<agg::path_storage> imageBox(path, srcMatrix);
  ras.add_path(imageBox);
  
  

  agg::image_filter_base* filter = 0;  
  agg::span_allocator<agg::rgba8> sa;	
  switch(image->interpolation)
    {
    case NEAREST:
      {
	typedef agg::span_image_filter_rgba32_nn<agg::order_rgba32,
	  interpolator_type> span_gen_type;
	typedef agg::renderer_scanline_u<renderer_base, span_gen_type> renderer_type;
	
	span_gen_type sg(sa, *image->rbufIn, agg::rgba(1,1,1,0), interpolator);
	renderer_type ri(rb, sg);
	ras.add_path(imageBox);
	ras.render(sl, ri);
      }
      break;
      
    case BILINEAR:
      {
	typedef agg::span_image_filter_rgba32_bilinear<agg::order_rgba32,
	  interpolator_type> span_gen_type;
	typedef agg::renderer_scanline_u<renderer_base, span_gen_type> renderer_type;
	
	span_gen_type sg(sa, *image->rbufIn, agg::rgba(1,1,1,0), interpolator);
	renderer_type ri(rb, sg);
	ras.add_path(imageBox);
	ras.render(sl, ri);
      }
      break;
      
    case BICUBIC:     filter = new agg::image_filter<agg::image_filter_bicubic>;
    case SPLINE16:    filter = new agg::image_filter<agg::image_filter_spline16>;
    case SPLINE36:    filter = new agg::image_filter<agg::image_filter_spline36>;   
    case SINC64:      filter = new agg::image_filter<agg::image_filter_sinc64>;     
    case SINC144:     filter = new agg::image_filter<agg::image_filter_sinc144>;    
    case SINC256:     filter = new agg::image_filter<agg::image_filter_sinc256>;    
    case BLACKMAN64:  filter = new agg::image_filter<agg::image_filter_blackman64>; 
    case BLACKMAN100: filter = new agg::image_filter<agg::image_filter_blackman100>;
    case BLACKMAN256: filter = new agg::image_filter<agg::image_filter_blackman256>;
      
      typedef agg::span_image_filter_rgba32<agg::order_rgba32,
	interpolator_type> span_gen_type;
      typedef agg::renderer_scanline_u<renderer_base, span_gen_type> renderer_type;
	
      span_gen_type sg(sa, *image->rbufIn, agg::rgba(1,1,1,0), interpolator, *filter);
      renderer_type ri(rb, sg);
      ras.render(sl, ri);

    }
	
	
  std::ofstream of2( "image_out.raw", std::ios::binary|std::ios::out);
  for (size_t i=0; i<NUMBYTES; ++i)
    of2.write((char*)&buffer[i], sizeof(char));


  
  image->bufferOut = buffer;
  
  

  Py_INCREF(Py_None);
  return Py_None;

}


char Image_set_interpolation__doc__[] = 
"set_interpolation(scheme)\n"
"\n"
"Set the interpolation scheme to one of the module constants, "
"eg, image.NEAREST, image.BILINEAR, etc..."
;

static PyObject *
Image_set_interpolation(ImageObject *image, PyObject* args) {


  size_t method;
  if (!PyArg_ParseTuple(args,"i", &method))
    return NULL;

  image->interpolation = (unsigned)method;

  Py_INCREF(Py_None);
  return Py_None;

}

char Image_set_preserve_aspect__doc__[] = 
"set_preserve_aspect(scheme)\n"
"\n"
"Set the aspect ration to one of the image module constant."
"eg, one of image.ASPECT_PRESERVE, image.ASPECT_FREE"
;
static PyObject *
Image_set_preserve_aspect(ImageObject *image, PyObject* args) {


  size_t method;
  if (!PyArg_ParseTuple(args,"i", &method))
    return NULL;

  image->aspect = (unsigned)method;

  Py_INCREF(Py_None);
  return Py_None;

}




static void
Image_dealloc(ImageObject *self)
{
  PyObject_Del(self);

  delete [] self->bufferIn;
  delete self->rbufOut;
  delete [] self->bufferOut;
  delete self->rbufIn;

}



// must be defined before getattr
static PyMethodDef Image_methods[] = {
  { "resize",	(PyCFunction)Image_resize,	 METH_VARARGS, Image_resize__doc__},
  { "set_interpolation",	(PyCFunction)Image_set_interpolation,	 METH_VARARGS, Image_set_interpolation__doc__},
  { "set_preserve_aspect",	(PyCFunction)Image_set_preserve_aspect,	 METH_VARARGS, Image_set_preserve_aspect__doc__},
  {NULL,		NULL}		/* sentinel */
};




static PyObject *
Image_getattr(ImageObject *self, char *name)
{
  if (self->x_attr != NULL) {
    PyObject *v = PyDict_GetItemString(self->x_attr, name);
    if (v != NULL) {
      Py_INCREF(v);
      return v;
    }
  }
  return Py_FindMethod(Image_methods, (PyObject *)self, name);
}


static int
Image_setattr(ImageObject *self, char *name, PyObject *v)
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
		      "delete non-existing Image attribute");
    return rv;
  }
  else
    return PyDict_SetItemString(self->x_attr, name, v);
}


static PyTypeObject Image_Type = {
  /* The ob_type field must be initialized in the module init function
   * to be portable to Windows without using C++. */
  PyObject_HEAD_INIT(NULL)
  0,			/*ob_size*/
  "image.Image",		/*tp_name*/
  sizeof(ImageObject),	/*tp_basicsize*/
  0,			/*tp_itemsize*/
  /* methods */
  (destructor)Image_dealloc, /*tp_dealloc*/
  0,			/*tp_print*/
  (getattrfunc)Image_getattr, /*tp_getattr*/
  (setattrfunc)Image_setattr, /*tp_setattr*/
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



static ImageObject *
newImageObject(PyObject *args)
{
  ImageObject *self;

  self = PyObject_New(ImageObject, &Image_Type);
  if (self == NULL)
    return NULL;
   
  self->bufferIn = NULL;
  self->rbufIn = NULL;
  self->widthIn = 0; 
  self->heightIn = 0; 
  self->BPP = 4; 

  self->bufferOut = NULL;
  self->rbufOut = NULL;
  self->widthOut = 0; 
  self->heightOut = 0; 

  self->interpolation = BILINEAR;
  self->aspect = ASPECT_FREE;
  self->x_attr = NULL;

  
  
  return self;
}


char image_fromfile__doc__[] = 
"fromfile(filename)\n"
"\n"
"Load the image from file filename\n"
;
static PyObject *
image_fromfile(PyObject *self, PyObject *args) {

  char *fname;
  if (!PyArg_ParseTuple(args, "s", &fname)) {
    return NULL;
  }


  ImageObject *imo;
  
  imo = newImageObject(args);
  if ( imo == NULL )
    return NULL;
  
  // load the image object here
  return (PyObject *)imo;
}

char image_fromarray__doc__[] = 
"fromarray(A)\n"
"\n"
"Load the image from Numeric array\n"
;
static PyObject *
image_fromarray(PyObject *self, PyObject *args) {

  PyArrayObject *A; 


  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &A)) 
    return NULL; 

  if (A->nd != 2 && A->nd != 3) {
    PyErr_SetString(PyExc_ValueError, 
		    "array must be rank-2 or 3"); 
    return NULL; 

  }
  //printf("num dim %u, type_num %u\n", A->nd, A->descr->type_num);
 
  if (A->descr->type_num != PyArray_DOUBLE) {
    PyErr_SetString(PyExc_ValueError, 
		    "array must be type float"); 
    return NULL; 
  } 

  ImageObject *imo;

  imo = newImageObject(args);
  if ( imo == NULL )
    return NULL;

  imo->widthIn  = A->dimensions[0];
  imo->heightIn = A->dimensions[1];
  size_t NUMBYTES(imo->widthIn * imo->heightIn * imo->BPP);
  agg::int8u *buffer = new agg::int8u[NUMBYTES];  

  imo->rbufIn = new agg::rendering_buffer;
  imo->rbufIn->attach(buffer, imo->widthIn, imo->heightIn, imo->widthIn*imo->BPP);
  
  if   (A->nd == 2) { //assume luminance for now; 
    unsigned N = imo->widthIn * imo->heightIn;

    double *raw =  (double *)A->data;

    agg::int8u gray;
    int start = 0;
    for (size_t i=0; i<N; ++i) {
      start = imo->BPP*i;                      
      gray = int(255* (*raw++));      // convert from double to uint8
      *(buffer+start++) = gray;       // red
      *(buffer+start++) = gray;       // green
      *(buffer+start++) = gray;       // blue
      *(buffer+start)   = 255;        // alpha
    }
  }
  else {
    // todo rank 3 arrays
  }
    
  imo->bufferIn = buffer;
  return (PyObject *)imo;
}




static PyMethodDef image_methods[] = {
  { "fromfile",	 (PyCFunction)image_fromfile,	 METH_VARARGS, image_fromfile__doc__},
  { "fromarray", (PyCFunction)image_fromarray,	 METH_VARARGS, image_fromarray__doc__},

  {NULL,		NULL}		/* sentinel */
};



extern "C"
DL_EXPORT(void)
  initimage(void)
{
  PyObject *module, *d;
  
  /* Initialize the type of the new type object here; doing it here
   * is required for portability to Windows without requiring C++. */
  Image_Type.ob_type = &PyType_Type;
  
  /* Create the module and add the functions */
  module = Py_InitModule("image", image_methods);

  /* Import the array object */
  import_array();

  
  /* Add some symbolic constants to the module */
  d = PyModule_GetDict(module);

  ErrorObject = PyErr_NewException("image.error", NULL, NULL);
  PyDict_SetItemString(d, "error", ErrorObject);

  MODULE_CONSTANT(d, "BICUBIC", BICUBIC);
  MODULE_CONSTANT(d, "BILINEAR", BILINEAR);
  MODULE_CONSTANT(d, "BLACKMAN100", BLACKMAN100);
  MODULE_CONSTANT(d, "BLACKMAN256", BLACKMAN256);
  MODULE_CONSTANT(d, "BLACKMAN64", BLACKMAN64);
  MODULE_CONSTANT(d, "NEAREST", NEAREST);
  MODULE_CONSTANT(d, "SINC144", SINC144);
  MODULE_CONSTANT(d, "SINC256", SINC256);
  MODULE_CONSTANT(d, "SINC64", SINC64);
  MODULE_CONSTANT(d, "SPLINE16", SPLINE16);
  MODULE_CONSTANT(d, "SPLINE36", SPLINE36);

  MODULE_CONSTANT(d, "ASPECT_FREE", ASPECT_FREE);
  MODULE_CONSTANT(d, "ASPECT_PRESERVE", ASPECT_PRESERVE);


}
