/* -*- mode: C; c-basic-offset: 4 -*-
 * C extensions for backend_gdk
 */

#include "Python.h"
#include "numpy/arrayobject.h"

#include <pygtk/pygtk.h>


static PyTypeObject *_PyGdkPixbuf_Type;
#define PyGdkPixbuf_Type (*_PyGdkPixbuf_Type)

static PyObject *
pixbuf_get_pixels_array(PyObject *self, PyObject *args)
{
    /* 1) read in Python pixbuf, get the underlying gdk_pixbuf */
    PyGObject *py_pixbuf;
    GdkPixbuf *gdk_pixbuf;
    PyArrayObject *array;
    npy_intp dims[3] = { 0, 0, 3 };

    if (!PyArg_ParseTuple(args, "O!:pixbuf_get_pixels_array",
			  &PyGdkPixbuf_Type, &py_pixbuf))
	return NULL;

    gdk_pixbuf = GDK_PIXBUF(py_pixbuf->obj);

    /* 2) same as pygtk/gtk/gdk.c _wrap_gdk_pixbuf_get_pixels_array()
     * with 'self' changed to py_pixbuf
     */

    dims[0] = gdk_pixbuf_get_height(gdk_pixbuf);
    dims[1] = gdk_pixbuf_get_width(gdk_pixbuf);
    if (gdk_pixbuf_get_has_alpha(gdk_pixbuf))
        dims[2] = 4;

    array = (PyArrayObject *)PyArray_SimpleNewFromData(3, dims, PyArray_UBYTE,
			     (char *)gdk_pixbuf_get_pixels(gdk_pixbuf));
    if (array == NULL)
        return NULL;

    array->strides[0] = gdk_pixbuf_get_rowstride(gdk_pixbuf);
    /* the array holds a ref to the pixbuf pixels through this wrapper*/
    Py_INCREF(py_pixbuf);
    array->base = (PyObject *)py_pixbuf;
    return PyArray_Return(array);
}

static PyMethodDef _backend_gdk_functions[] = {
    { "pixbuf_get_pixels_array", (PyCFunction)pixbuf_get_pixels_array, METH_VARARGS },
    { NULL, NULL, 0 }
};

PyMODINIT_FUNC
init_backend_gdk(void)
{
    PyObject *mod;
    mod = Py_InitModule("matplotlib.backends._backend_gdk",
                                        _backend_gdk_functions);
    import_array();
    init_pygtk();

    mod = PyImport_ImportModule("gtk.gdk");
    _PyGdkPixbuf_Type = (PyTypeObject *)PyObject_GetAttrString(mod, "Pixbuf");
}
