#include "mplutils.h"
#include "_image_resample.h"
#include "_image.h"
#include "numpy_cpp.h"
#include "py_converters.h"


/**********************************************************************
 * Free functions
 * */

const char* image_resample__doc__ =
"resample(input_array, output_array, matrix, interpolation=NEAREST, alpha=1.0, norm=False, radius=1)\n\n"

"Resample input_array, blending it in-place into output_array, using an\n"
"affine transformation.\n\n"

"Parameters\n"
"----------\n"
"input_array : 2-d or 3-d Numpy array of float, double or uint8\n"
"    If 2-d, the image is grayscale.  If 3-d, the image must be of size\n"
"    4 in the last dimension and represents RGBA data.\n\n"

"output_array : 2-d or 3-d Numpy array of float, double or uint8\n"
"    The dtype and number of dimensions must match `input_array`.\n\n"

"transform : matplotlib.transforms.Transform instance\n"
"    The transformation from the input array to the output\n"
"    array.\n\n"

"interpolation : int, optional\n"
"    The interpolation method.  Must be one of the following constants\n"
"    defined in this module:\n\n"

"      NEAREST (default), BILINEAR, BICUBIC, SPLINE16, SPLINE36,\n"
"      HANNING, HAMMING, HERMITE, KAISER, QUADRIC, CATROM, GAUSSIAN,\n"
"      BESSEL, MITCHELL, SINC, LANCZOS, BLACKMAN\n\n"

"resample : bool, optional\n"
"    When `True`, use a full resampling method.  When `False`, only\n"
"    resample when the output image is larger than the input image.\n\n"

"alpha : float, optional\n"
"    The level of transparency to apply.  1.0 is completely opaque.\n"
"    0.0 is completely transparent.\n\n"

"norm : bool, optional\n"
"    Whether to norm the interpolation function.  Default is `False`.\n\n"

"radius: float, optional\n"
"    The radius of the kernel, if method is SINC, LANCZOS or BLACKMAN.\n"
"    Default is 1.\n";


static PyArrayObject *
_get_transform_mesh(PyObject *py_affine, npy_intp *dims)
{
    /* TODO: Could we get away with float, rather than double, arrays here? */

    /* Given a non-affine transform object, create a mesh that maps
    every pixel in the output image to the input image.  This is used
    as a lookup table during the actual resampling. */

    PyObject *py_inverse = NULL;
    npy_intp out_dims[3];

    out_dims[0] = dims[0] * dims[1];
    out_dims[1] = 2;

    py_inverse = PyObject_CallMethod(
        py_affine, (char *)"inverted", (char *)"", NULL);
    if (py_inverse == NULL) {
        return NULL;
    }

    numpy::array_view<double, 2> input_mesh(out_dims);
    double *p = (double *)input_mesh.data();

    for (npy_intp y = 0; y < dims[0]; ++y) {
        for (npy_intp x = 0; x < dims[1]; ++x) {
            *p++ = (double)x;
            *p++ = (double)y;
        }
    }

    PyObject *output_mesh =
        PyObject_CallMethod(
            py_inverse, (char *)"transform", (char *)"O",
            (char *)input_mesh.pyobj(), NULL);

    Py_DECREF(py_inverse);

    if (output_mesh == NULL) {
        return NULL;
    }

    PyArrayObject *output_mesh_array =
        (PyArrayObject *)PyArray_ContiguousFromAny(
            output_mesh, NPY_DOUBLE, 2, 2);

    Py_DECREF(output_mesh);

    if (output_mesh_array == NULL) {
        return NULL;
    }

    return output_mesh_array;
}


static PyObject *
image_resample(PyObject *self, PyObject* args, PyObject *kwargs)
{
    PyObject *py_input_array = NULL;
    PyObject *py_output_array = NULL;
    PyObject *py_transform = NULL;
    resample_params_t params;

    PyArrayObject *input_array = NULL;
    PyArrayObject *output_array = NULL;
    PyArrayObject *transform_mesh_array = NULL;

    params.transform_mesh = NULL;

    const char *kwlist[] = {
        "input_array", "output_array", "transform", "interpolation",
        "resample", "alpha", "norm", "radius", NULL };

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOO|iO&dO&d:resample", (char **)kwlist,
            &py_input_array, &py_output_array, &py_transform,
            &params.interpolation, &convert_bool, &params.resample,
            &params.alpha, &convert_bool, &params.norm, &params.radius)) {
        return NULL;
    }

    if (params.interpolation < 0 || params.interpolation >= _n_interpolation) {
        PyErr_Format(PyExc_ValueError, "invalid interpolation value %d",
                     params.interpolation);
        goto error;
    }

    input_array = (PyArrayObject *)PyArray_FromAny(
        py_input_array, NULL, 2, 3, NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (input_array == NULL) {
        goto error;
    }

    output_array = (PyArrayObject *)PyArray_FromAny(
        py_output_array, NULL, 2, 3, NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (output_array == NULL) {
        goto error;
    }

    if (py_transform == NULL || py_transform == Py_None) {
        params.is_affine = true;
    } else {
        PyObject *py_is_affine;
        int py_is_affine2;
        py_is_affine = PyObject_GetAttrString(py_transform, "is_affine");
        if (py_is_affine == NULL) {
            goto error;
        }

        py_is_affine2 = PyObject_IsTrue(py_is_affine);
        Py_DECREF(py_is_affine);

        if (py_is_affine2 == -1) {
            goto error;
        } else if (py_is_affine2) {
            if (!convert_trans_affine(py_transform, &params.affine)) {
                goto error;
            }
            params.is_affine = true;
        } else {
            transform_mesh_array = _get_transform_mesh(
                py_transform, PyArray_DIMS(output_array));
            if (transform_mesh_array == NULL) {
                goto error;
            }
            params.transform_mesh = (double *)PyArray_DATA(transform_mesh_array);
            params.is_affine = false;
        }
    }

    if (PyArray_NDIM(input_array) != PyArray_NDIM(output_array)) {
        PyErr_Format(
            PyExc_ValueError,
            "Mismatched number of dimensions. Got %d and %d.",
            PyArray_NDIM(input_array), PyArray_NDIM(output_array));
        goto error;
    }

    if (PyArray_TYPE(input_array) != PyArray_TYPE(output_array)) {
        PyErr_SetString(PyExc_ValueError, "Mismatched types");
        goto error;
    }

    if (PyArray_NDIM(input_array) == 3) {
        if (PyArray_DIM(output_array, 2) != 4) {
            PyErr_SetString(
                PyExc_ValueError,
                "Output array must be RGBA");
            goto error;
        }

        if (PyArray_DIM(input_array, 2) == 4) {
            switch(PyArray_TYPE(input_array)) {
            case NPY_BYTE:
            case NPY_UINT8:
                Py_BEGIN_ALLOW_THREADS
                resample(
                    (agg::rgba8 *)PyArray_DATA(input_array),
                    PyArray_DIM(input_array, 1),
                    PyArray_DIM(input_array, 0),
                    (agg::rgba8 *)PyArray_DATA(output_array),
                    PyArray_DIM(output_array, 1),
                    PyArray_DIM(output_array, 0),
                    params);
                Py_END_ALLOW_THREADS
                break;
            case NPY_UINT16:
            case NPY_INT16:
                Py_BEGIN_ALLOW_THREADS
                resample(
                    (agg::rgba16 *)PyArray_DATA(input_array),
                    PyArray_DIM(input_array, 1),
                    PyArray_DIM(input_array, 0),
                    (agg::rgba16 *)PyArray_DATA(output_array),
                    PyArray_DIM(output_array, 1),
                    PyArray_DIM(output_array, 0),
                    params);
                Py_END_ALLOW_THREADS
                break;
            case NPY_FLOAT32:
                Py_BEGIN_ALLOW_THREADS
                resample(
                    (agg::rgba32 *)PyArray_DATA(input_array),
                    PyArray_DIM(input_array, 1),
                    PyArray_DIM(input_array, 0),
                    (agg::rgba32 *)PyArray_DATA(output_array),
                    PyArray_DIM(output_array, 1),
                    PyArray_DIM(output_array, 0),
                    params);
                Py_END_ALLOW_THREADS
                break;
            case NPY_FLOAT64:
                Py_BEGIN_ALLOW_THREADS
                resample(
                    (agg::rgba64 *)PyArray_DATA(input_array),
                    PyArray_DIM(input_array, 1),
                    PyArray_DIM(input_array, 0),
                    (agg::rgba64 *)PyArray_DATA(output_array),
                    PyArray_DIM(output_array, 1),
                    PyArray_DIM(output_array, 0),
                    params);
                Py_END_ALLOW_THREADS
                break;
            default:
                PyErr_SetString(
                    PyExc_ValueError,
                    "3-dimensional arrays must be of dtype unsigned byte, "
                    "unsigned short, float32 or float64");
                goto error;
            }
        } else {
            PyErr_Format(
                PyExc_ValueError,
                "If 3-dimensional, array must be RGBA.  Got %" NPY_INTP_FMT " planes.",
                PyArray_DIM(input_array, 2));
            goto error;
        }
    } else { // NDIM == 2
        switch (PyArray_TYPE(input_array)) {
        case NPY_DOUBLE:
            Py_BEGIN_ALLOW_THREADS
            resample(
                (double *)PyArray_DATA(input_array),
                PyArray_DIM(input_array, 1),
                PyArray_DIM(input_array, 0),
                (double *)PyArray_DATA(output_array),
                PyArray_DIM(output_array, 1),
                PyArray_DIM(output_array, 0),
                params);
            Py_END_ALLOW_THREADS
            break;
        case NPY_FLOAT:
            Py_BEGIN_ALLOW_THREADS
            resample(
                (float *)PyArray_DATA(input_array),
                PyArray_DIM(input_array, 1),
                PyArray_DIM(input_array, 0),
                (float *)PyArray_DATA(output_array),
                PyArray_DIM(output_array, 1),
                PyArray_DIM(output_array, 0),
                params);
            Py_END_ALLOW_THREADS
            break;
        case NPY_UINT8:
        case NPY_BYTE:
            Py_BEGIN_ALLOW_THREADS
            resample(
                (unsigned char *)PyArray_DATA(input_array),
                PyArray_DIM(input_array, 1),
                PyArray_DIM(input_array, 0),
                (unsigned char *)PyArray_DATA(output_array),
                PyArray_DIM(output_array, 1),
                PyArray_DIM(output_array, 0),
                params);
            Py_END_ALLOW_THREADS
            break;
        case NPY_UINT16:
        case NPY_INT16:
            Py_BEGIN_ALLOW_THREADS
            resample(
                (unsigned short *)PyArray_DATA(input_array),
                PyArray_DIM(input_array, 1),
                PyArray_DIM(input_array, 0),
                (unsigned short *)PyArray_DATA(output_array),
                PyArray_DIM(output_array, 1),
                PyArray_DIM(output_array, 0),
                params);
            Py_END_ALLOW_THREADS
            break;
        default:
            PyErr_SetString(PyExc_ValueError, "Unsupported dtype");
            goto error;
        }
    }

    Py_DECREF(input_array);
    Py_XDECREF(transform_mesh_array);
    return (PyObject *)output_array;

 error:
    Py_XDECREF(input_array);
    Py_XDECREF(output_array);
    Py_XDECREF(transform_mesh_array);
    return NULL;
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
    npy_intp rows, cols;
    float bounds[4];
    int interpolation;

    if (!PyArg_ParseTuple(args,
                          "O&O&O&nn(ffff)i:pcolor",
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

    npy_intp dim[3] = {rows, cols, 4};
    numpy::array_view<const agg::int8u, 3> output(dim);

    CALL_CPP("pcolor", (pcolor(x, y, d, rows, cols, bounds, interpolation, output)));

    return output.pyobj();
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
    npy_intp rows, cols;
    float bounds[4];
    numpy::array_view<const agg::int8u, 1> bg;

    if (!PyArg_ParseTuple(args,
                          "O&O&O&nn(ffff)O&:pcolor2",
                          &x.converter_contiguous,
                          &x,
                          &y.converter_contiguous,
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

    npy_intp dim[3] = {rows, cols, 4};
    numpy::array_view<const agg::int8u, 3> output(dim);

    CALL_CPP("pcolor2", (pcolor2(x, y, d, rows, cols, bounds, bg, output)));

    return output.pyobj();
}

static PyMethodDef module_functions[] = {
    {"resample", (PyCFunction)image_resample, METH_VARARGS|METH_KEYWORDS, image_resample__doc__},
    {"pcolor", (PyCFunction)image_pcolor, METH_VARARGS, image_pcolor__doc__},
    {"pcolor2", (PyCFunction)image_pcolor2, METH_VARARGS, image_pcolor2__doc__},
    {NULL}
};

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

PyMODINIT_FUNC PyInit__image(void)
{
    PyObject *m;

    m = PyModule_Create(&moduledef);

    if (m == NULL) {
        return NULL;
    }

    if (PyModule_AddIntConstant(m, "NEAREST", NEAREST) ||
        PyModule_AddIntConstant(m, "BILINEAR", BILINEAR) ||
        PyModule_AddIntConstant(m, "BICUBIC", BICUBIC) ||
        PyModule_AddIntConstant(m, "SPLINE16", SPLINE16) ||
        PyModule_AddIntConstant(m, "SPLINE36", SPLINE36) ||
        PyModule_AddIntConstant(m, "HANNING", HANNING) ||
        PyModule_AddIntConstant(m, "HAMMING", HAMMING) ||
        PyModule_AddIntConstant(m, "HERMITE", HERMITE) ||
        PyModule_AddIntConstant(m, "KAISER", KAISER) ||
        PyModule_AddIntConstant(m, "QUADRIC", QUADRIC) ||
        PyModule_AddIntConstant(m, "CATROM", CATROM) ||
        PyModule_AddIntConstant(m, "GAUSSIAN", GAUSSIAN) ||
        PyModule_AddIntConstant(m, "BESSEL", BESSEL) ||
        PyModule_AddIntConstant(m, "MITCHELL", MITCHELL) ||
        PyModule_AddIntConstant(m, "SINC", SINC) ||
        PyModule_AddIntConstant(m, "LANCZOS", LANCZOS) ||
        PyModule_AddIntConstant(m, "BLACKMAN", BLACKMAN) ||
        PyModule_AddIntConstant(m, "_n_interpolation", _n_interpolation)) {
        return NULL;
    }

    import_array();

    return m;
}
