
#include <fstream>
#include <cmath>
#include <cstdio>
#include "Python.h"
#include "Numeric/arrayobject.h" 

#include "agg_pixfmt_rgb24.h"
#include "agg_pixfmt_rgba32.h"
#include "agg_rendering_buffer.h"
#include "agg_rasterizer_scanline_aa.h"
#include "agg_path_storage.h"
#include "agg_conv_transform.h"
#include "agg_span_image_filter_rgb24.h"
#include "agg_span_image_filter_rgba32.h"
#include "agg_span_interpolator_linear.h"
#include "agg_scanline_u8.h"
#include "agg_scanline_p32.h"
#include "agg_renderer_scanline.h"

#include "_image.h"






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


extern PyTypeObject Image_Type;
#define ImageObject_Check(v)	((v)->ob_type == &Image_Type)

enum { BICUBIC=0, BILINEAR, BLACKMAN100, BLACKMAN256, BLACKMAN64, 
       NEAREST, SINC144, SINC256, SINC64, SPLINE16, SPLINE36};
enum { ASPECT_PRESERVE=0, ASPECT_FREE};


char Image_apply_rotation__doc__[] = 
"apply_rotation(angle)\n"
"\n"
"Apply the rotation (degrees) to image"
;

static PyObject *
Image_apply_rotation(ImageObject *image, PyObject* args) {
  
  double r;
  if (!PyArg_ParseTuple(args, "d", &r))
    return NULL;
  
  
  agg::affine_matrix M = agg::rotation_matrix( r * agg::pi / 180.0);	      
  image->srcMatrix *= M;
  image->imageMatrix *= M;
  Py_INCREF(Py_None);
  return Py_None;
  
  
}


char Image_apply_scaling__doc__[] = 
"apply_scaling(sx, sy)\n"
"\n"
"Apply the scale factors sx, sy to the transform matrix"
;

static PyObject *
Image_apply_scaling(ImageObject *image, PyObject* args) {
  
  double sx, sy;
  if (!PyArg_ParseTuple(args, "dd", &sx, &sy))
    return NULL;
  
  //printf("applying scaling %1.2f, %1.2f\n", sx, sy);
  agg::affine_matrix M = agg::scaling_matrix(sx, sy);	      
  image->srcMatrix *= M;
  image->imageMatrix *= M;
  Py_INCREF(Py_None);
  return Py_None;
  
  
}

char Image_apply_translation__doc__[] = 
"apply_translation(tx, ty)\n"
"\n"
"Apply the translation tx, ty to the transform matrix"
;

static PyObject *
Image_apply_translation(ImageObject *image, PyObject* args) {
  
  double tx, ty;
  if (!PyArg_ParseTuple(args, "dd", &tx, &ty))
    return NULL;
  
  //printf("applying translation %1.2f, %1.2f\n", tx, ty);
  agg::affine_matrix M = agg::translation_matrix(tx, ty);	      
  image->srcMatrix *= M;
  image->imageMatrix *= M;
  Py_INCREF(Py_None);
  return Py_None;
  
  
}


char Image_as_str__doc__[] = 
"numrows, numcols, s = as_str()"
"\n"
"Call this function after resize to get the data as string"
"The string is a numrows by numcols x 4 (RGBA) unsigned char buffer"
;

static PyObject *
Image_as_str(ImageObject *image, PyObject* args) {
  
  if (!PyArg_ParseTuple(args, ":as_str"))
    return NULL;
  
  return Py_BuildValue("lls#", image->rowsOut, image->colsOut, 
		       image->bufferOut, image->colsOut*image->rowsOut*4);
  
  
}


char Image_reset_matrix__doc__[] = 
"reset_matrix()"
"\n"
"Reset the transformation matrix"
;

static PyObject *
Image_reset_matrix(ImageObject *image, PyObject* args) {
  image->srcMatrix.reset();
  image->imageMatrix.reset();
  Py_INCREF(Py_None);
  return Py_None;
  
  
}

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
  
  int numcols, numrows;
  
  if (!PyArg_ParseTuple(args, "ii", &numcols, &numrows))
    return NULL;
  
  image->colsOut  = numcols;
  image->rowsOut = numrows;
  
  size_t NUMBYTES(numrows * numcols * image->BPP);
  agg::int8u *buffer = new agg::int8u[NUMBYTES];  
  image->rbufOut = new agg::rendering_buffer;
  image->rbufOut->attach(buffer, numrows, numcols, numrows * image->BPP);
  
  // init the output rendering/rasterizing stuff
  pixfmt pixf(*image->rbufOut);
  renderer_base rb(pixf);
  rb.clear(agg::rgba(1, 1, 1));
  agg::rasterizer_scanline_aa<> ras;
  agg::scanline_u8 sl;
  
  
  //image->srcMatrix *= resizingMatrix;
  //image->imageMatrix *= resizingMatrix;
  image->imageMatrix.invert();
  interpolator_type interpolator(image->imageMatrix); 
  
  // the image path
  agg::path_storage path;
  path.move_to(0, 0);
  path.line_to(image->colsIn, 0);
  path.line_to(image->colsIn, image->rowsIn);
  path.line_to(0, image->rowsIn);
  path.close_polygon();
  agg::conv_transform<agg::path_storage> imageBox(path, image->srcMatrix);
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
  
  /*
    std::ofstream of2( "image_out.raw", std::ios::binary|std::ios::out);
    for (size_t i=0; i<NUMBYTES; ++i)
    of2.write((char*)&buffer[i], sizeof(char));
  */
  
  
  image->bufferOut = buffer;
  
  
  
  Py_INCREF(Py_None);
  return Py_None;
  
}

char Image_get_aspect__doc__[] = 
"get_aspect()\n"
"\n"
"Get the aspeect constraint constants"
;

static PyObject *
Image_get_aspect(ImageObject *image, PyObject* args) {
  
  
  return Py_BuildValue("l", (long)image->aspect);
  
}

char Image_get_size__doc__[] = 
"numrows, numcols = get_size()\n"
"\n"
"Get the number or rows and columns of the input image"
;

static PyObject *
Image_get_size(ImageObject *image, PyObject* args) {
  
  
  return Py_BuildValue("(ll)", (long)image->rowsIn, (long)image->colsIn);
  
}


char Image_get_interpolation__doc__[] = 
"get_interpolation()\n"
"\n"
"Get the interpolation scheme to one of the module constants, "
"one of image.NEAREST, image.BILINEAR, etc..."
;

static PyObject *
Image_get_interpolation(ImageObject *image, PyObject* args) {
  
  
  return Py_BuildValue("l", (long)image->interpolation);
  
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



char Image_set_aspect__doc__[] = 
"set_aspect(scheme)\n"
"\n"
"Set the aspect ration to one of the image module constant."
"eg, one of image.ASPECT_PRESERVE, image.ASPECT_FREE"
;
static PyObject *
Image_set_aspect(ImageObject *image, PyObject* args) {
  
  
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
  { "apply_rotation",	(PyCFunction)Image_apply_rotation,	 METH_VARARGS, Image_apply_rotation__doc__},
  { "apply_scaling",	(PyCFunction)Image_apply_scaling,	 METH_VARARGS, Image_apply_scaling__doc__},
  { "apply_translation",	(PyCFunction)Image_apply_translation,	 METH_VARARGS, Image_apply_translation__doc__},
  { "as_str",	(PyCFunction)Image_as_str,	 METH_VARARGS, Image_as_str__doc__},
  { "get_aspect",	        (PyCFunction)Image_get_aspect,	 METH_VARARGS, Image_get_aspect__doc__},
  { "get_interpolation",	(PyCFunction)Image_get_interpolation,	 METH_VARARGS, Image_get_interpolation__doc__},
  { "get_size",	        (PyCFunction)Image_get_size,	 METH_VARARGS, Image_get_size__doc__},
  { "reset_matrix",	(PyCFunction)Image_reset_matrix,	 METH_VARARGS, Image_reset_matrix__doc__},
  { "resize",	(PyCFunction)Image_resize,	 METH_VARARGS, Image_resize__doc__},
  { "set_interpolation",	(PyCFunction)Image_set_interpolation,	 METH_VARARGS, Image_set_interpolation__doc__},
  { "set_aspect",	(PyCFunction)Image_set_aspect,	 METH_VARARGS, Image_set_aspect__doc__},
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


PyTypeObject Image_Type = {
  /* The ob_type field must be initialized in the module init function
   * to be portable to Windows without using C++. */
  PyObject_HEAD_INIT(NULL)
  0,			/*ob_size*/
  "_image.Image",		/*tp_name*/
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
  self->colsIn = 0; 
  self->rowsIn = 0; 
  self->BPP = 4; 
  
  self->bufferOut = NULL;
  self->rbufOut = NULL;
  self->colsOut = 0; 
  self->rowsOut = 0; 
  
  self->interpolation = BILINEAR;
  self->aspect = ASPECT_FREE;
  self->x_attr = NULL;
  
  
  
  return self;
}


char _image_fromfile__doc__[] = 
"fromfile(filename)\n"
"\n"
"Load the image from file filename\n"
;
static PyObject *
_image_fromfile(PyObject *self, PyObject *args) {
  
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

char _image_fromarray__doc__[] = 
"fromarray(A)\n"
"\n"
"Load the image from a Numeric or numarray array\n"
;
static PyObject *
_image_fromarray(PyObject *self, PyObject *args) {
  
  PyObject *x; 
  PyArrayObject *A;
  
  if (!PyArg_ParseTuple(args, "O", &x))
    return NULL; 
  
  //rank 2 or 3
  A = (PyArrayObject *) PyArray_ContiguousFromObject(x, PyArray_DOUBLE, 2, 3); 

  if (!A) {
    PyErr_SetString(PyExc_ValueError, 
		    "Array must be rank 2 or 3 of doubles"); 
    return NULL; 
  }
  
  ImageObject *imo;
  
  imo = newImageObject(args);
  if ( imo == NULL )
    return NULL;
  
  imo->rowsIn  = A->dimensions[0];
  imo->colsIn  = A->dimensions[1];
  
  size_t NUMBYTES(imo->colsIn * imo->rowsIn * imo->BPP);
  agg::int8u *buffer = new agg::int8u[NUMBYTES];  
  imo->bufferIn = buffer;
  imo->rbufIn = new agg::rendering_buffer;
  imo->rbufIn->attach(buffer, imo->colsIn, imo->rowsIn, imo->colsIn*imo->BPP);
  
  if   (A->nd == 2) { //assume luminance for now; 
    
    agg::int8u gray;
    int start = 0;    
    for (size_t rownum=0; rownum<imo->rowsIn; rownum++) 
      for (size_t colnum=0; colnum<imo->colsIn; colnum++) {
	
	double val = *(double *)(A->data + rownum*A->strides[0] + colnum*A->strides[1]);
	
	gray = int(255 * val);
	*(buffer+start++) = gray;       // red
	*(buffer+start++) = gray;       // green
	*(buffer+start++) = gray;       // blue
	*(buffer+start++)   = 255;        // alpha
      }
    
  }
  else if   (A->nd == 3) { // assume RGB

    
    if (A->dimensions[2] != 3 && A->dimensions[2] != 4 ) {
      PyErr_SetString(PyExc_ValueError, 
		      "3rd dimension must be length 3 (RGB) or 4 (RGBA)"); 
      return NULL;
      
    }
    
    int rgba = A->dimensions[2]==4;
    
    int start = 0;    
    double r,g,b,alpha;
    int offset =0;
    for (size_t rownum=0; rownum<imo->rowsIn; rownum++) 
      for (size_t colnum=0; colnum<imo->colsIn; colnum++) {
	offset = rownum*A->strides[0] + colnum*A->strides[1];
	r = *(double *)(A->data + offset);
	g = *(double *)(A->data + offset + A->strides[2] );
	b = *(double *)(A->data + offset + 2*A->strides[2] );
	
	if (rgba) 
	  alpha = *(double *)(A->data + offset + 3*A->strides[2] );
	else
	  alpha = 1.0;

	*(buffer+start++) = int(255*r);         // red
	*(buffer+start++) = int(255*g);         // green
	*(buffer+start++) = int(255*b);         // blue
	*(buffer+start++) = int(255*alpha);     // alpha

      }
  } 
  else   { // error
    PyErr_SetString(PyExc_ValueError, 
		    "Illegal array rank; must be rank; must 2 or 3"); 
    return NULL;
  }
  
  return (PyObject *)imo;
}





static PyMethodDef _image_methods[] = {
  { "fromfile",	 (PyCFunction)_image_fromfile,	 METH_VARARGS, _image_fromfile__doc__},
  { "fromarray", (PyCFunction)_image_fromarray,	 METH_VARARGS, _image_fromarray__doc__},
  
  {NULL,		NULL}		/* sentinel */
};



extern "C"
DL_EXPORT(void)
  init_image(void)
{
  PyObject *module, *d;
  
  /* Initialize the type of the new type object here; doing it here
   * is required for portability to Windows without requiring C++. */
  Image_Type.ob_type = &PyType_Type;
  
  /* Create the module and add the functions */
  module = Py_InitModule("_image", _image_methods);
  
  /* Import the array object */
  import_array();
  
  
  /* Add some symbolic constants to the module */
  d = PyModule_GetDict(module);
  
  ErrorObject = PyErr_NewException("_image.error", NULL, NULL);
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
