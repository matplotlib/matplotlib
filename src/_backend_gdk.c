/* -*- mode: C; c-basic-offset: 4 -*- 
 * C extensions for backend_gdk
 */

#include "Python.h"
#ifdef NUMARRAY
#include "numarray/arrayobject.h"
#else
#include "Numeric/arrayobject.h"
#endif

#include <pygtk/pygtk.h>
#include <gdk/gdkx.h>


static PyTypeObject *_PyGdkPixbuf_Type;
#define PyGdkPixbuf_Type (*_PyGdkPixbuf_Type)

/* Implement an equivalent to the pygtk method pixbuf.get_pixels_array()
 * Fedora 1,2,3 (for example) has PyGTK but does not have Numeric
 * and so does not have pixbuf.get_pixels_array().
 * Also provide numarray as well as Numeric support
 */

static PyObject *
pixbuf_get_pixels_array(PyObject *self, PyObject *args)
{
    /* 1) read in Python pixbuf, get the underlying C pixbuf */
    PyGObject *py_pixbuf;
    GdkPixbuf *pixbuf;
    PyArrayObject *array;
    int dims[3] = { 0, 0, 3 };

    if (!PyArg_ParseTuple(args, "O!:pixbuf_get_pixels_array",
			  &PyGdkPixbuf_Type, &py_pixbuf))
	return NULL;

    pixbuf = GDK_PIXBUF(py_pixbuf->obj);

    /* 2) same as pygtk/gtk/gdk.c _wrap_gdk_pixbuf_get_pixels_array()
     * with 'self' changed to py_pixbuf
     */

    dims[0] = gdk_pixbuf_get_height(pixbuf);
    dims[1] = gdk_pixbuf_get_width(pixbuf);
    if (gdk_pixbuf_get_has_alpha(pixbuf))
        dims[2] = 4;

    array = (PyArrayObject *)PyArray_FromDimsAndData(3, dims, PyArray_UBYTE,
                                        (char *)gdk_pixbuf_get_pixels(pixbuf));

    if (array == NULL)
        return NULL;

    array->strides[0] = gdk_pixbuf_get_rowstride(pixbuf);
    /* the array holds a ref to the pixbuf pixels through this wrapper*/
    Py_INCREF(py_pixbuf);
    array->base = (PyObject *)py_pixbuf;
    return PyArray_Return(array);
}

static PyMethodDef _backend_gdk_functions[] = {
    { "pixbuf_get_pixels_array", (PyCFunction)pixbuf_get_pixels_array, METH_VARARGS },
    { NULL, NULL, 0 }
};

DL_EXPORT(void)
#ifdef NUMARRAY
init_na_backend_gdk(void)
{
    PyObject *mod;
    mod = Py_InitModule("_na_backend_gdk", _backend_gdk_functions);
#else
init_nc_backend_gdk(void)
{
    PyObject *mod;
    mod = Py_InitModule("_nc_backend_gdk", _backend_gdk_functions);
#endif

    import_array();
    init_pygtk();

    mod = PyImport_ImportModule("gtk.gdk");
    _PyGdkPixbuf_Type = (PyTypeObject *)PyObject_GetAttrString(mod, "Pixbuf");
}
