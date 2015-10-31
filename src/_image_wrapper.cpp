#include "mplutils.h"
#include "_image.h"
#include "py_converters.h"

/**********************************************************************
 * Image
 * */

typedef struct
{
    PyObject_HEAD;
    Image *x;
    Py_ssize_t shape[3];
    Py_ssize_t strides[3];
    Py_ssize_t suboffsets[3];
    PyObject *dict;
} PyImage;

static PyTypeObject PyImageType;

static PyObject *PyImage_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyImage *self;
    self = (PyImage *)type->tp_alloc(type, 0);
    memset(self, 0, sizeof(PyImage));
    self->x = NULL;
    self->dict = PyDict_New();
    return (PyObject *)self;
}

static PyObject *PyImage_cnew(Image *im)
{
    PyImage *self;
    self = (PyImage *)PyImageType.tp_alloc(&PyImageType, 0);
    self->x = im;
    self->dict = PyDict_New();
    return (PyObject *)self;
}

static int PyImage_init(PyImage *self, PyObject *args, PyObject *kwds)
{
    if (!PyArg_ParseTuple(args, "")) {
        return -1;
    }

    CALL_CPP_INIT("Image", (self->x = new Image()));

    return 0;
}

static void PyImage_dealloc(PyImage *self)
{
    delete self->x;
    Py_DECREF(self->dict);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

const char *PyImage_apply_rotation__doc__ =
    "apply_rotation(angle)\n"
    "\n"
    "Apply the rotation (degrees) to image";

static PyObject *PyImage_apply_rotation(PyImage *self, PyObject *args, PyObject *kwds)
{
    double r;

    if (!PyArg_ParseTuple(args, "d:apply_rotation", &r)) {
        return NULL;
    }

    CALL_CPP("apply_rotation", (self->x->apply_rotation(r)));

    Py_RETURN_NONE;
}

const char *PyImage_set_bg__doc__ =
    "set_bg(r,g,b,a)\n"
    "\n"
    "Set the background color";

static PyObject *PyImage_set_bg(PyImage *self, PyObject *args, PyObject *kwds)
{
    double r, g, b, a;

    if (!PyArg_ParseTuple(args, "dddd:set_bg", &r, &g, &b, &a)) {
        return NULL;
    }

    CALL_CPP("set_bg", (self->x->set_bg(r, g, b, a)));

    Py_RETURN_NONE;
}

const char *PyImage_apply_scaling__doc__ =
    "apply_scaling(sx, sy)\n"
    "\n"
    "Apply the scale factors sx, sy to the transform matrix";

static PyObject *PyImage_apply_scaling(PyImage *self, PyObject *args, PyObject *kwds)
{
    double sx, sy;

    if (!PyArg_ParseTuple(args, "dd:apply_scaling", &sx, &sy)) {
        return NULL;
    }

    CALL_CPP("apply_scaling", (self->x->apply_scaling(sx, sy)));

    Py_RETURN_NONE;
}

const char *PyImage_apply_translation__doc__ =
    "apply_translation(tx, ty)\n"
    "\n"
    "Apply the translation tx, ty to the transform matrix";

static PyObject *PyImage_apply_translation(PyImage *self, PyObject *args, PyObject *kwds)
{
    double tx, ty;
    if (!PyArg_ParseTuple(args, "dd:apply_translation", &tx, &ty)) {
        return NULL;
    }

    CALL_CPP("apply_translation", self->x->apply_translation(tx, ty));

    Py_RETURN_NONE;
}

const char *PyImage_as_rgba_str__doc__ =
    "numrows, numcols, s = as_rgba_str()"
    "\n"
    "Call this function after resize to get the data as string\n"
    "The string is a numrows by numcols x 4 (RGBA) unsigned char buffer\n";

static PyObject *PyImage_as_rgba_str(PyImage *self, PyObject *args, PyObject *kwds)
{
    // TODO: This performs a copy.  Use buffer interface when possible

    PyObject *result = PyBytes_FromStringAndSize(NULL, self->x->rowsOut * self->x->colsOut * 4);
    if (result == NULL) {
        return NULL;
    }

    CALL_CPP_CLEANUP("as_rgba_str",
                     (self->x->as_rgba_str((agg::int8u *)PyBytes_AsString(result))),
                     Py_DECREF(result));

    return Py_BuildValue("nnN", self->x->rowsOut, self->x->colsOut, result);
}

const char *PyImage_color_conv__doc__ =
    "numrows, numcols, buffer = color_conv(format)"
    "\n"
    "format 0(BGRA) or 1(ARGB)\n"
    "Convert image to format and return in a writable buffer\n";

// TODO: This function is a terrible interface.  Change/remove?  Only
// used by Cairo backend.

static PyObject *PyImage_color_conv(PyImage *self, PyObject *args, PyObject *kwds)
{
    int format;

    if (!PyArg_ParseTuple(args, "i:color_conv", &format)) {
        return NULL;
    }

    Py_ssize_t size = self->x->rowsOut * self->x->colsOut * 4;
    agg::int8u *buff = (agg::int8u *)malloc(size);
    if (buff == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Out of memory");
        return NULL;
    }

    CALL_CPP_CLEANUP("color_conv",
                     (self->x->color_conv(format, buff)),
                     free(buff));

    PyObject *result = PyByteArray_FromStringAndSize((const char *)buff, size);
    free(buff);
    if (result == NULL) {
        return NULL;
    }

    return Py_BuildValue("nnN", self->x->rowsOut, self->x->colsOut, result);
}

const char *PyImage_buffer_rgba__doc__ =
    "buffer = buffer_rgba()"
    "\n"
    "Return the image buffer as rgba32\n";

static PyObject *PyImage_buffer_rgba(PyImage *self, PyObject *args, PyObject *kwds)
{
#if PY3K
    return Py_BuildValue("nny#",
                         self->x->rowsOut,
                         self->x->colsOut,
                         self->x->rbufOut,
                         self->x->rowsOut * self->x->colsOut * 4);
#else
    PyObject *buffer =
        PyBuffer_FromReadWriteMemory(self->x->rbufOut, self->x->rowsOut * self->x->colsOut * 4);
    if (buffer == NULL) {
        return NULL;
    }

    return Py_BuildValue("nnN", self->x->rowsOut, self->x->colsOut, buffer);
#endif
}

const char *PyImage_reset_matrix__doc__ =
    "reset_matrix()"
    "\n"
    "Reset the transformation matrix";

static PyObject *PyImage_reset_matrix(PyImage *self, PyObject *args, PyObject *kwds)
{
    CALL_CPP("reset_matrix", self->x->reset_matrix());

    Py_RETURN_NONE;
}

const char *PyImage_get_matrix__doc__ =
    "(m11,m21,m12,m22,m13,m23) = get_matrix()\n"
    "\n"
    "Get the affine transformation matrix\n"
    "  /m11,m12,m13\\\n"
    "  /m21,m22,m23|\n"
    "  \\ 0 , 0 , 1 /";

static PyObject *PyImage_get_matrix(PyImage *self, PyObject *args, PyObject *kwds)
{
    double m[6];
    self->x->srcMatrix.store_to(m);

    return Py_BuildValue("dddddd", m[0], m[1], m[2], m[3], m[4], m[5]);
}

const char *PyImage_resize__doc__ =
    "resize(width, height, norm=1, radius=4.0)\n"
    "\n"
    "Resize the image to width, height using interpolation\n"
    "norm and radius are optional args for some of the filters\n";

static PyObject *PyImage_resize(PyImage *self, PyObject *args, PyObject *kwds)
{
    double width;
    double height;
    double norm;
    double radius;
    const char *names[] = { "width", "height", "norm", "radius", NULL };

    if (!PyArg_ParseTupleAndKeywords(
             args, kwds, "dd|dd:resize", (char **)names, &width, &height, &norm, &radius)) {
        return NULL;
    }

    CALL_CPP("resize", (self->x->resize(width, height, norm, radius)));

    Py_RETURN_NONE;
}

const char *PyImage_get_interpolation__doc__ =
    "get_interpolation()\n"
    "\n"
    "Get the interpolation scheme to one of the module constants, "
    "one of image.NEAREST, image.BILINEAR, etc...";

static PyObject *PyImage_get_interpolation(PyImage *self, PyObject *args, PyObject *kwds)
{
    return PyLong_FromLong(self->x->interpolation);
}

const char *PyImage_set_interpolation__doc__ =
    "set_interpolation(scheme)\n"
    "\n"
    "Set the interpolation scheme to one of the module constants, "
    "eg, image.NEAREST, image.BILINEAR, etc...";

static PyObject *PyImage_set_interpolation(PyImage *self, PyObject *args, PyObject *kwds)
{
    int method;

    if (!PyArg_ParseTuple(args, "i:set_interpolation", &method)) {
        return NULL;
    }

    self->x->interpolation = method;

    Py_RETURN_NONE;
}

const char *PyImage_get_aspect__doc__ =
    "get_aspect()\n"
    "\n"
    "Get the aspect constraint constants";

static PyObject *PyImage_get_aspect(PyImage *self, PyObject *args, PyObject *kwds)
{
    return PyLong_FromLong(self->x->aspect);
}

const char *PyImage_set_aspect__doc__ =
    "set_aspect(scheme)\n"
    "\n"
    "Set the aspect ratio to one of the image module constant."
    "eg, one of image.ASPECT_PRESERVE, image.ASPECT_FREE";

static PyObject *PyImage_set_aspect(PyImage *self, PyObject *args, PyObject *kwds)
{
    int scheme;
    if (!PyArg_ParseTuple(args, "i:set_aspect", &scheme)) {
        return NULL;
    }

    self->x->aspect = scheme;

    Py_RETURN_NONE;
}

const char *PyImage_get_size__doc__ =
    "numrows, numcols = get_size()\n"
    "\n"
    "Get the number or rows and columns of the input image";

static PyObject *PyImage_get_size(PyImage *self, PyObject *args, PyObject *kwds)
{
    return Py_BuildValue("ii", self->x->rowsIn, self->x->colsIn);
}

const char *PyImage_get_resample__doc__ =
    "get_resample()\n"
    "\n"
    "Get the resample flag.";

static PyObject *PyImage_get_resample(PyImage *self, PyObject *args, PyObject *kwds)
{
    if (self->x->resample) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

const char *PyImage_set_resample__doc__ =
    "set_resample(boolean)\n"
    "\n"
    "Set the resample flag.";

static PyObject *PyImage_set_resample(PyImage *self, PyObject *args, PyObject *kwds)
{
    int resample;

    if (!PyArg_ParseTuple(args, "i:set_resample", &resample)) {
        return NULL;
    }

    self->x->resample = resample;

    Py_RETURN_NONE;
}

const char *PyImage_get_size_out__doc__ =
    "numrows, numcols = get_size_out()\n"
    "\n"
    "Get the number or rows and columns of the output image";

static PyObject *PyImage_get_size_out(PyImage *self, PyObject *args, PyObject *kwds)
{
    return Py_BuildValue("ii", self->x->rowsOut, self->x->colsOut);
}

static int PyImage_get_buffer(PyImage *self, Py_buffer *buf, int flags)
{
    Image *im = self->x;

    Py_INCREF(self);
    buf->obj = (PyObject *)self;
    buf->buf = im->bufferOut;
    buf->len = im->colsOut * im->rowsOut * 4;
    buf->readonly = 0;
    buf->format = (char *)"B";
    buf->ndim = 3;
    self->shape[0] = im->rowsOut;
    self->shape[1] = im->colsOut;
    self->shape[2] = 4;
    buf->shape = self->shape;
    self->strides[0] = im->colsOut * 4;
    self->strides[1] = 4;
    self->strides[2] = 1;
    buf->strides = self->strides;
    buf->suboffsets = NULL;
    buf->itemsize = 1;
    buf->internal = NULL;

    return 1;
}

static PyTypeObject *PyImage_init_type(PyObject *m, PyTypeObject *type)
{
    static PyMethodDef methods[] = {
        {"apply_rotation", (PyCFunction)PyImage_apply_rotation, METH_VARARGS, PyImage_apply_rotation__doc__},
        {"set_bg", (PyCFunction)PyImage_set_bg, METH_VARARGS, PyImage_set_bg__doc__},
        {"apply_scaling", (PyCFunction)PyImage_apply_scaling, METH_VARARGS, PyImage_apply_scaling__doc__},
        {"apply_translation", (PyCFunction)PyImage_apply_translation, METH_VARARGS, PyImage_apply_translation__doc__},
        {"as_rgba_str", (PyCFunction)PyImage_as_rgba_str, METH_NOARGS, PyImage_as_rgba_str__doc__},
        {"color_conv", (PyCFunction)PyImage_color_conv, METH_VARARGS, PyImage_color_conv__doc__},
        {"buffer_rgba", (PyCFunction)PyImage_buffer_rgba, METH_NOARGS, PyImage_buffer_rgba__doc__},
        {"reset_matrix", (PyCFunction)PyImage_reset_matrix, METH_NOARGS, PyImage_reset_matrix__doc__},
        {"get_matrix", (PyCFunction)PyImage_get_matrix, METH_NOARGS, PyImage_get_matrix__doc__},
        {"resize", (PyCFunction)PyImage_resize, METH_VARARGS|METH_KEYWORDS, PyImage_resize__doc__},
        {"get_interpolation", (PyCFunction)PyImage_get_interpolation, METH_NOARGS, PyImage_get_interpolation__doc__},
        {"set_interpolation", (PyCFunction)PyImage_set_interpolation, METH_VARARGS, PyImage_set_interpolation__doc__},
        {"get_aspect", (PyCFunction)PyImage_get_aspect, METH_NOARGS, PyImage_get_aspect__doc__},
        {"set_aspect", (PyCFunction)PyImage_set_aspect, METH_VARARGS, PyImage_set_aspect__doc__},
        {"get_size", (PyCFunction)PyImage_get_size, METH_NOARGS, PyImage_get_size__doc__},
        {"get_resample", (PyCFunction)PyImage_get_resample, METH_VARARGS, PyImage_get_resample__doc__},
        {"set_resample", (PyCFunction)PyImage_set_resample, METH_VARARGS, PyImage_set_resample__doc__},
        {"get_size_out", (PyCFunction)PyImage_get_size_out, METH_VARARGS, PyImage_get_size_out__doc__},
        {NULL}
    };

    static PyBufferProcs buffer_procs;
    memset(&buffer_procs, 0, sizeof(PyBufferProcs));
    buffer_procs.bf_getbuffer = (getbufferproc)PyImage_get_buffer;

    memset(type, 0, sizeof(PyTypeObject));
    type->tp_name = "matplotlib._image.Image";
    type->tp_basicsize = sizeof(PyImage);
    type->tp_dealloc = (destructor)PyImage_dealloc;
    type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_NEWBUFFER;
    type->tp_methods = methods;
    type->tp_new = PyImage_new;
    type->tp_init = (initproc)PyImage_init;
    type->tp_as_buffer = &buffer_procs;
    type->tp_dictoffset = offsetof(PyImage, dict);

    if (PyType_Ready(type) < 0) {
        return NULL;
    }

    if (PyModule_AddObject(m, "Image", (PyObject *)type)) {
        return NULL;
    }

    return type;
}

/**********************************************************************
 * Free functions
 * */

const char *image_from_images__doc__ =
    "from_images(numrows, numcols, seq)\n"
    "\n"
    "return an image instance with numrows, numcols from a seq of image\n"
    "instances using alpha blending.  seq is a list of (Image, ox, oy)";

static PyObject *image_from_images(PyObject *self, PyObject *args, PyObject *kwds)
{
    unsigned int numrows;
    unsigned int numcols;
    PyObject *images;
    size_t numimages;

    if (!PyArg_ParseTuple(args, "IIO:from_images", &numrows, &numcols, &images)) {
        return NULL;
    }

    if (!PySequence_Check(images)) {
        return NULL;
    }

    Image *im = new Image(numrows, numcols, true);
    im->clear();

    numimages = PySequence_Size(images);

    for (size_t i = 0; i < numimages; ++i) {
        PyObject *entry = PySequence_GetItem(images, i);
        if (entry == NULL) {
            delete im;
            return NULL;
        }

        PyObject *subimage;
        unsigned int x;
        unsigned int y;
        PyObject *alphaobj = NULL;
        double alpha = 0.0;

        if (!PyArg_ParseTuple(entry, "O!II|O", &PyImageType, &subimage, &x, &y, &alphaobj)) {
            Py_DECREF(entry);
            delete im;
            return NULL;
        }

        bool has_alpha = false;
        if (alphaobj != NULL && alphaobj != Py_None) {
            has_alpha = true;
            alpha = PyFloat_AsDouble(alphaobj);
            if (PyErr_Occurred()) {
                Py_DECREF(entry);
                delete im;
                return NULL;
            }
        }

        CALL_CPP("from_images",
                 (im->blend_image(*((PyImage *)subimage)->x, x, y, has_alpha, alpha)));

        Py_DECREF(entry);
    }

    return PyImage_cnew(im);
}

const char *image_fromarray__doc__ =
    "fromarray(A, isoutput)\n"
    "\n"
    "Load the image from a numpy array\n"
    "By default this function fills the input buffer, which can subsequently\n"
    "be resampled using resize.  If isoutput=1, fill the output buffer.\n"
    "This is used to support raw pixel images w/o resampling\n";

static PyObject *image_fromarray(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *array;
    int isoutput;
    const char *names[] = { "array", "isoutput", NULL };

    if (!PyArg_ParseTupleAndKeywords(
             args, kwds, "O|i:fromarray", (char **)names, &array, &isoutput)) {
        return NULL;
    }

    numpy::array_view<const double, 3> color_array;
    numpy::array_view<const double, 2> grey_array;
    Image *result = NULL;

    if (color_array.converter(array, &color_array)) {
        CALL_CPP("fromarray", result = from_color_array(color_array, isoutput));
    } else if (grey_array.converter(array, &grey_array)) {
        CALL_CPP("fromarray", result = from_grey_array(grey_array, isoutput));
    } else {
        PyErr_SetString(PyExc_ValueError, "invalid array");
        return NULL;
    }

    return PyImage_cnew(result);
}

const char *image_frombyte__doc__ =
    "frombyte(A, isoutput)\n"
    "\n"
    "Load the image from a byte array.\n"
    "By default this function fills the input buffer, which can subsequently\n"
    "be resampled using resize.  If isoutput=1, fill the output buffer.\n"
    "This is used to support raw pixel images w/o resampling.";

static PyObject *image_frombyte(PyObject *self, PyObject *args, PyObject *kwds)
{
    numpy::array_view<const uint8_t, 3> array;
    int isoutput;
    const char *names[] = { "array", "isoutput", NULL };
    Image *result;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwds,
                                     "O&|i:frombyte",
                                     (char **)names,
                                     &array.converter,
                                     &array,
                                     &isoutput)) {
        return NULL;
    }

    CALL_CPP("frombyte", (result = frombyte(array, isoutput)));

    return PyImage_cnew(result);
}

const char *image_frombuffer__doc__ =
    "frombuffer(buffer, width, height, isoutput)\n"
    "\n"
    "Load the image from a character buffer\n"
    "By default this function fills the input buffer, which can subsequently\n"
    "be resampled using resize.  If isoutput=1, fill the output buffer.\n"
    "This is used to support raw pixel images w/o resampling.";

static PyObject *image_frombuffer(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *buffer;
    unsigned x;
    unsigned y;
    int isoutput;
    const char *names[] = { "buffer", "x", "y", "isoutput", NULL };

    if (!PyArg_ParseTupleAndKeywords(
             args, kwds, "OII|i:frombuffer", (char **)names, &buffer, &x, &y, &isoutput)) {
        return NULL;
    }

    const void *rawbuf;
    Py_ssize_t buflen;
    if (PyObject_AsReadBuffer(buffer, &rawbuf, &buflen) != 0) {
        return NULL;
    }

    if (buflen != (Py_ssize_t)(y * x * 4)) {
        PyErr_SetString(PyExc_ValueError, "Buffer is incorrect length");
        return NULL;
    }

    Image *im;
    CALL_CPP("frombuffer", (im = new Image(y, x, isoutput)));

    agg::int8u *inbuf = (agg::int8u *)rawbuf;
    agg::int8u *outbuf;
    if (isoutput) {
        outbuf = im->bufferOut;
    } else {
        outbuf = im->bufferIn;
    }

    for (int i = (x * 4) * (y - 1); i >= 0; i -= (x * 4)) {
        memmove(outbuf, &inbuf[i], (x * 4));
        outbuf += x * 4;
    }

    return PyImage_cnew(im);
}

const char *image_pcolor__doc__ =
    "pcolor(x, y, data, rows, cols, bounds)\n"
    "\n"
    "Generate a pseudo-color image from data on a non-uniform grid using\n"
    "nearest neighbour or linear interpolation.\n"
    "bounds = (x_min, x_max, y_min, y_max)\n"
    "interpolation = NEAREST or BILINEAR \n";

static PyObject *image_pcolor(PyObject *self, PyObject *args, PyObject *kwds)
{
    numpy::array_view<const float, 1> x;
    numpy::array_view<const float, 1> y;
    numpy::array_view<const agg::int8u, 3> d;
    unsigned int rows;
    unsigned int cols;
    float bounds[4];
    int interpolation;
    Image *result;

    if (!PyArg_ParseTuple(args,
                          "O&O&O&II(ffff)i:pcolor",
                          &x.converter,
                          &x,
                          &y.converter,
                          &y,
                          &d.converter_contiguous,
                          &d,
                          &rows,
                          &cols,
                          &bounds[0],
                          &bounds[1],
                          &bounds[2],
                          &bounds[3],
                          &interpolation)) {
        return NULL;
    }

    CALL_CPP("pcolor", (result = pcolor(x, y, d, rows, cols, bounds, interpolation)));

    return PyImage_cnew(result);
}

const char *image_pcolor2__doc__ =
    "pcolor2(x, y, data, rows, cols, bounds, bg)\n"
    "\n"
    "Generate a pseudo-color image from data on a non-uniform grid\n"
    "specified by its cell boundaries.\n"
    "bounds = (x_left, x_right, y_bot, y_top)\n"
    "bg = ndarray of 4 uint8 representing background rgba\n";

static PyObject *image_pcolor2(PyObject *self, PyObject *args, PyObject *kwds)
{
    numpy::array_view<const double, 1> x;
    numpy::array_view<const double, 1> y;
    numpy::array_view<const agg::int8u, 3> d;
    unsigned int rows;
    unsigned int cols;
    float bounds[4];
    numpy::array_view<const agg::int8u, 1> bg;
    Image *result;

    if (!PyArg_ParseTuple(args,
                          "O&O&O&II(ffff)O&:pcolor2",
                          &x.converter,
                          &x,
                          &y.converter,
                          &y,
                          &d.converter_contiguous,
                          &d,
                          &rows,
                          &cols,
                          &bounds[0],
                          &bounds[1],
                          &bounds[2],
                          &bounds[3],
                          &bg.converter,
                          &bg)) {
        return NULL;
    }

    CALL_CPP("pcolor2", (result = pcolor2(x, y, d, rows, cols, bounds, bg)));

    return PyImage_cnew(result);
}

static PyMethodDef module_functions[] = {
    {"from_images", (PyCFunction)image_from_images, METH_VARARGS, image_from_images__doc__},
    {"fromarray", (PyCFunction)image_fromarray, METH_VARARGS|METH_KEYWORDS, image_fromarray__doc__},
    {"frombyte", (PyCFunction)image_frombyte, METH_VARARGS|METH_KEYWORDS, image_frombyte__doc__},
    {"frombuffer", (PyCFunction)image_frombuffer, METH_VARARGS|METH_KEYWORDS, image_frombuffer__doc__},
    {"pcolor", (PyCFunction)image_pcolor, METH_VARARGS, image_pcolor__doc__},
    {"pcolor2", (PyCFunction)image_pcolor2, METH_VARARGS, image_pcolor2__doc__},
    {NULL}
};

extern "C" {

#if PY3K
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_image",
    NULL,
    0,
    module_functions,
    NULL,
    NULL,
    NULL,
    NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC PyInit__image(void)

#else
#define INITERROR return

PyMODINIT_FUNC init_image(void)
#endif

{
    PyObject *m;

#if PY3K
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3("_image", module_functions, NULL);
#endif

    if (m == NULL) {
        INITERROR;
    }

    if (!PyImage_init_type(m, &PyImageType)) {
        INITERROR;
    }

    PyObject *d = PyModule_GetDict(m);

    if (add_dict_int(d, "NEAREST", Image::NEAREST) ||
        add_dict_int(d, "BILINEAR", Image::BILINEAR) ||
        add_dict_int(d, "BICUBIC", Image::BICUBIC) ||
        add_dict_int(d, "SPLINE16", Image::SPLINE16) ||
        add_dict_int(d, "SPLINE36", Image::SPLINE36) ||
        add_dict_int(d, "HANNING", Image::HANNING) ||
        add_dict_int(d, "HAMMING", Image::HAMMING) ||
        add_dict_int(d, "HERMITE", Image::HERMITE) ||
        add_dict_int(d, "KAISER", Image::KAISER) ||
        add_dict_int(d, "QUADRIC", Image::QUADRIC) ||
        add_dict_int(d, "CATROM", Image::CATROM) ||
        add_dict_int(d, "GAUSSIAN", Image::GAUSSIAN) ||
        add_dict_int(d, "BESSEL", Image::BESSEL) ||
        add_dict_int(d, "MITCHELL", Image::MITCHELL) ||
        add_dict_int(d, "SINC", Image::SINC) ||
        add_dict_int(d, "LANCZOS", Image::LANCZOS) ||
        add_dict_int(d, "BLACKMAN", Image::BLACKMAN) ||

        add_dict_int(d, "ASPECT_FREE", Image::ASPECT_FREE) ||
        add_dict_int(d, "ASPECT_PRESERVE", Image::ASPECT_PRESERVE)) {
        INITERROR;
    }

    import_array();

#if PY3K
    return m;
#endif
}

} // extern "C"
