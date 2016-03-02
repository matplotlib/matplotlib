/* -*- mode: C; c-basic-offset: 4 -*-
 * C extensions for backend_gdk
 */

#include "Python.h"
#include "numpy/arrayobject.h"

#include <pygtk/pygtk.h>

// support numpy 1.6 - this macro was renamed and deprecated at once in 1.7
#ifndef NPY_ARRAY_WRITEABLE
#define NPY_ARRAY_WRITEABLE NPY_WRITEABLE
#endif

static PyTypeObject *_PyGdkPixbuf_Type;
#define PyGdkPixbuf_Type (*_PyGdkPixbuf_Type)

static PyObject *pixbuf_get_pixels_array(PyObject *self, PyObject *args)
{
    /* 1) read in Python pixbuf, get the underlying gdk_pixbuf */
    PyGObject *py_pixbuf;
    GdkPixbuf *gdk_pixbuf;
    PyArrayObject *array;
    npy_intp dims[3] = { 0, 0, 3 };
    npy_intp strides[3];

    if (!PyArg_ParseTuple(args, "O!:pixbuf_get_pixels_array", &PyGdkPixbuf_Type, &py_pixbuf))
        return NULL;

    gdk_pixbuf = GDK_PIXBUF(py_pixbuf->obj);

    /* 2) same as pygtk/gtk/gdk.c _wrap_gdk_pixbuf_get_pixels_array()
     * with 'self' changed to py_pixbuf
     */

    dims[0] = gdk_pixbuf_get_height(gdk_pixbuf);
    dims[1] = gdk_pixbuf_get_width(gdk_pixbuf);
    if (gdk_pixbuf_get_has_alpha(gdk_pixbuf))
        dims[2] = 4;

    strides[0] = gdk_pixbuf_get_rowstride(gdk_pixbuf);
    strides[1] = dims[2];
    strides[2] = 1;

    array = (PyArrayObject*)
        PyArray_New(&PyArray_Type, 3, dims, NPY_UBYTE, strides,
                    (void*)gdk_pixbuf_get_pixels(gdk_pixbuf), 1,
                    NPY_ARRAY_WRITEABLE, NULL);

    if (array == NULL)
        return NULL;

    /* the array holds a ref to the pixbuf pixels through this wrapper*/
    Py_INCREF(py_pixbuf);
#if NPY_API_VERSION >= 0x00000007
    if (PyArray_SetBaseObject(array, (PyObject *)py_pixbuf) == -1) {
        Py_DECREF(py_pixbuf);
        Py_DECREF(array);
        return NULL;
    }
#else
    PyArray_BASE(array) = (PyObject *) py_pixbuf;
#endif
    return PyArray_Return(array);
}

static PyMethodDef _backend_gdk_functions[] = {
    { "pixbuf_get_pixels_array", (PyCFunction)pixbuf_get_pixels_array, METH_VARARGS },
    { NULL, NULL, 0 }
};

PyMODINIT_FUNC init_backend_gdk(void)
{
    PyObject *mod;
    mod = Py_InitModule("matplotlib.backends._backend_gdk", _backend_gdk_functions);
    import_array();
    init_pygtk();

    mod = PyImport_ImportModule("gtk.gdk");
    _PyGdkPixbuf_Type = (PyTypeObject *)PyObject_GetAttrString(mod, "Pixbuf");
}
