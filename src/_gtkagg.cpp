/* Helper functions for converting an agg image buffer to a gtk drawable
   efficiently
*/
#include <Python.h>
#include <pygobject.h>
#include <pygtk/pygtk.h>
#include <cstring>
#include <cerrno>
#include <cstdio>

#include "_backend_agg.h"

static PyTypeObject *PyGObject_Type=NULL;    


static PyObject *
_agg_to_gtk_drawable(PyObject *self, PyObject *args) {

  PyGObject *py_drawable = NULL;                         
  GdkDrawable *drawable = NULL;   
  GdkGC* gc = NULL;


  RendererAggObject* aggRenderer;

  if (!PyArg_ParseTuple(args, "O!O", PyGObject_Type, 
			&py_drawable, &aggRenderer))
      return NULL;

  drawable = GDK_DRAWABLE(py_drawable->obj);
  gc = gdk_gc_new(drawable);

  gdk_draw_rgb_32_image(drawable, gc, 0, 0, 
			aggRenderer->rbase->width(), 
			aggRenderer->rbase->height(), 
			GDK_RGB_DITHER_NORMAL,
			aggRenderer->buffer,
			aggRenderer->rbase->width()*4);

  Py_INCREF(Py_None);
  return Py_None;
}
  




static struct PyMethodDef _gtkagg_methods[] = {
  {"agg_to_gtk_drawable", (PyCFunction)_agg_to_gtk_drawable, METH_VARARGS, 
   "Draw to a gtk drawable from a agg buffer."},
  {NULL,		NULL}		/* sentinel */
};


extern "C"
DL_EXPORT(void) init_gtkagg(void)
{
  PyObject *module, *d;
  


  init_pygobject();
  init_pygtk();
  /* Create the module and add the functions */
  Py_InitModule("_gtkagg", _gtkagg_methods);  
  module = PyImport_ImportModule("gobject");
  if (module) {
    PyGObject_Type =
      (PyTypeObject*)PyObject_GetAttrString(module, "GObject");
    Py_DECREF(module);
  }
  /* Add some symbolic constants to the module */
  d = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_gtkagg.error");
  PyDict_SetItemString(d, "error", ErrorObject);
  
  /* Check for errors */
  if (PyErr_Occurred())
    Py_FatalError("can't initialize module _gtkagg");
}
