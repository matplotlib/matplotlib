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

static PyObject *ErrorObject;
static PyTypeObject *PyGObject_Type=NULL;    


static PyObject *
_agg_to_gtk_drawable(PyObject *self, PyObject *args) {

  PyGObject *py_drawable = NULL;                         
  GdkDrawable *drawable = NULL;   
  GdkGC* gc = NULL;
  
  PyObject* aggo;


  if (!PyArg_ParseTuple(args, "O!O", PyGObject_Type, 
			&py_drawable, &aggo))
      return NULL;


  RendererAgg* aggRenderer = (RendererAgg*)aggo;

  drawable = GDK_DRAWABLE(py_drawable->obj);
  gc = gdk_gc_new(drawable);

  unsigned int width = aggRenderer->get_width();
  unsigned int height = aggRenderer->get_height();

  gdk_draw_rgb_32_image(drawable, gc, 0, 0, 
			width, 
			height, 
			GDK_RGB_DITHER_NORMAL,
			aggRenderer->pixBuffer,
			width*4);

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
