/* Helper functions for converting a gd image to a gtk drawable
   efficiently
*/
#include <Python.h>
#include <gd.h>
#include <pygobject.h>
#include <pygtk/pygtk.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>


static PyObject *ErrorObject;
static PyTypeObject *PyGObject_Type=NULL;    

typedef struct i_o {
    PyObject_HEAD
    gdImagePtr imagedata;
    int multiplier_x,origin_x;
    int multiplier_y,origin_y;
    struct i_o *current_brush;
    struct i_o *current_tile;
} imageobject;


static PyObject *
_gd_to_gtk_drawable(PyObject *self, PyObject *args) {
  int x1, y1, x2, y2, w, h, i, j, c;
  PyObject *im = NULL;
  imageobject *imo = NULL;
  gdImagePtr imData;
  PyGObject *py_drawable = NULL;                         
  GdkDrawable *drawable = NULL;   
  GdkGC* gc = NULL;
  GdkColor color;
  GdkColormap* cmap = NULL;

  if (!PyArg_ParseTuple(args, "O!O", PyGObject_Type, &py_drawable, &im))
      return NULL;

  drawable = GDK_DRAWABLE(py_drawable->obj);
  gc = gdk_gc_new(drawable);
  cmap = gdk_window_get_colormap(drawable);
  imo = PyObject_GetAttrString(im, "_image");

  imData = imo->imagedata;
  w = gdImageSX(imData);
  h = gdImageSY(imData);

  //printf("cmap size: %u\n", cmap->size);
  for (i=0; i<w; ++i)
    for (j=0; j<h; ++j) {
      c = gdImageGetPixel(imData, i, j);

      color.red = 256*imData->red[c];
      color.green = 256*imData->green[c];
      color.blue = 256*imData->blue[c];
      gdk_colormap_alloc_color(cmap, &color, 1, 1);

      gdk_gc_set_foreground(gc, &color);
      //what should I compare color.pixel against here for failure 
      gdk_draw_point(drawable, gc, i, j);
      
    }


  gdImageGetClip(imo->imagedata, &x1, &y1, &x2, &y2);
  return Py_BuildValue("(ii)(ii)", x1, y1, x2, y2);

}
  




static struct PyMethodDef _gtkgd_methods[] = {
  {"gd_to_gtk_drawable", (PyCFunction)_gd_to_gtk_drawable, METH_VARARGS, 
   "Draw to a gtk drawable from a gd image."},
  {NULL,		NULL}		/* sentinel */
};



DL_EXPORT(void) init_gtkgd(void)
{
  PyObject *module, *d;
  


  init_pygobject();
  init_pygtk();
  /* Create the module and add the functions */
  Py_InitModule("_gtkgd", _gtkgd_methods);  
  module = PyImport_ImportModule("gobject");
  if (module) {
    PyGObject_Type =
      (PyTypeObject*)PyObject_GetAttrString(module, "GObject");
    Py_DECREF(module);
  }
  /* Add some symbolic constants to the module */
  d = PyModule_GetDict(module);
  ErrorObject = PyString_FromString("_gtkgd.error");
  PyDict_SetItemString(d, "error", ErrorObject);
  
  /* Check for errors */
  if (PyErr_Occurred())
    Py_FatalError("can't initialize module _gtkgd");
}
