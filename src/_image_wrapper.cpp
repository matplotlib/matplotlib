#include "mplutils.h"
#include "_image_resample.h"
#include "numpy_cpp.h"
#include "py_converters.h"


/**********************************************************************
 * Free functions
 * */

const char* image_resample__doc__ =
"resample(input_array, output_array, transform, interpolation=NEAREST, resample=False, alpha=1.0, norm=False, radius=1.0)\n"
"--\n\n"

"Resample input_array, blending it in-place into output_array, using an\n"
"affine transformation.\n\n"

"Parameters\n"
"----------\n"
"input_array : 2-d or 3-d NumPy array of float, double or `numpy.uint8`\n"
"    If 2-d, the image is grayscale.  If 3-d, the image must be of size\n"
"    4 in the last dimension and represents RGBA data.\n\n"

"output_array : 2-d or 3-d NumPy array of float, double or `numpy.uint8`\n"
"    The dtype and number of dimensions must match `input_array`.\n\n"

"transform : matplotlib.transforms.Transform instance\n"
"    The transformation from the input array to the output array.\n\n"

"interpolation : int, default: NEAREST\n"
"    The interpolation method.  Must be one of the following constants\n"
"    defined in this module:\n\n"

"      NEAREST, BILINEAR, BICUBIC, SPLINE16, SPLINE36,\n"
"      HANNING, HAMMING, HERMITE, KAISER, QUADRIC, CATROM, GAUSSIAN,\n"
"      BESSEL, MITCHELL, SINC, LANCZOS, BLACKMAN\n\n"

"resample : bool, optional\n"
"    When `True`, use a full resampling method.  When `False`, only\n"
"    resample when the output image is larger than the input image.\n\n"

"alpha : float, default: 1\n"
"    The transparency level, from 0 (transparent) to 1 (opaque).\n\n"

"norm : bool, default: False\n"
"    Whether to norm the interpolation function.\n\n"

"radius: float, default: 1\n"
"    The radius of the kernel, if method is SINC, LANCZOS or BLACKMAN.\n";


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

    py_inverse = PyObject_CallMethod(py_affine, "inverted", NULL);
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

    PyObject *output_mesh = PyObject_CallMethod(
        py_inverse, "transform", "O", input_mesh.pyobj_steal());

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
    PyObject *py_input = NULL;
    PyObject *py_output = NULL;
    PyObject *py_transform = NULL;
    resample_params_t params;

    PyArrayObject *input = NULL;
    PyArrayObject *output = NULL;
    PyArrayObject *transform_mesh = NULL;
    int ndim;
    int type;

    params.interpolation = NEAREST;
    params.transform_mesh = NULL;
    params.resample = false;
    params.norm = false;
    params.radius = 1.0;
    params.alpha = 1.0;

    const char *kwlist[] = {
        "input_array", "output_array", "transform", "interpolation",
        "resample", "alpha", "norm", "radius", NULL };

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOO|iO&dO&d:resample", (char **)kwlist,
            &py_input, &py_output, &py_transform,
            &params.interpolation, &convert_bool, &params.resample,
            &params.alpha, &convert_bool, &params.norm, &params.radius)) {
        return NULL;
    }

    if (params.interpolation < 0 || params.interpolation >= _n_interpolation) {
        PyErr_Format(PyExc_ValueError, "Invalid interpolation value %d",
                     params.interpolation);
        goto error;
    }

    input = (PyArrayObject *)PyArray_FromAny(
        py_input, NULL, 2, 3, NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (!input) {
        goto error;
    }
    ndim = PyArray_NDIM(input);
    type = PyArray_TYPE(input);

    if (!PyArray_Check(py_output)) {
        PyErr_SetString(PyExc_ValueError, "Output array must be a NumPy array");
        goto error;
    }
    output = (PyArrayObject *)py_output;
    if (PyArray_NDIM(output) != ndim) {
        PyErr_Format(
            PyExc_ValueError,
            "Input (%dD) and output (%dD) have different dimensionalities.",
            ndim, PyArray_NDIM(output));
        goto error;
    }
    // PyArray_FromAny above checks that input is 2D or 3D.
    if (ndim == 3 && (PyArray_DIM(input, 2) != 4 || PyArray_DIM(output, 2) != 4)) {
        PyErr_Format(
            PyExc_ValueError,
            "If 3D, input and output arrays must be RGBA with shape (M, N, 4); "
            "got trailing dimensions of %" NPY_INTP_FMT " and %" NPY_INTP_FMT
            " respectively", PyArray_DIM(input, 2), PyArray_DIM(output, 2));
        goto error;
    }
    if (PyArray_TYPE(output) != type) {
        PyErr_SetString(PyExc_ValueError, "Mismatched types");
        goto error;
    }
    if (!PyArray_IS_C_CONTIGUOUS(output)) {
        PyErr_SetString(PyExc_ValueError, "Output array must be C-contiguous");
        goto error;
    }

    if (py_transform == NULL || py_transform == Py_None) {
        params.is_affine = true;
    } else {
        PyObject *py_is_affine;
        int py_is_affine2;
        py_is_affine = PyObject_GetAttrString(py_transform, "is_affine");
        if (!py_is_affine) {
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
            transform_mesh = _get_transform_mesh(
                py_transform, PyArray_DIMS(output));
            if (!transform_mesh) {
                goto error;
            }
            params.transform_mesh = (double *)PyArray_DATA(transform_mesh);
            params.is_affine = false;
        }
    }

    if (auto resampler =
            (ndim == 2) ? (
                (type == NPY_UINT8) ? resample<agg::gray8> :
                (type == NPY_INT8) ? resample<agg::gray8> :
                (type == NPY_UINT16) ? resample<agg::gray16> :
                (type == NPY_INT16) ? resample<agg::gray16> :
                (type == NPY_FLOAT32) ? resample<agg::gray32> :
                (type == NPY_FLOAT64) ? resample<agg::gray64> :
                nullptr) : (
            // ndim == 3
                (type == NPY_UINT8) ? resample<agg::rgba8> :
                (type == NPY_INT8) ? resample<agg::rgba8> :
                (type == NPY_UINT16) ? resample<agg::rgba16> :
                (type == NPY_INT16) ? resample<agg::rgba16> :
                (type == NPY_FLOAT32) ? resample<agg::rgba32> :
                (type == NPY_FLOAT64) ? resample<agg::rgba64> :
                nullptr)) {
        Py_BEGIN_ALLOW_THREADS
        resampler(
            PyArray_DATA(input), PyArray_DIM(input, 1), PyArray_DIM(input, 0),
            PyArray_DATA(output), PyArray_DIM(output, 1), PyArray_DIM(output, 0),
            params);
        Py_END_ALLOW_THREADS
    } else {
        PyErr_SetString(
            PyExc_ValueError,
            "arrays must be of dtype byte, short, float32 or float64");
        goto error;
    }

    Py_DECREF(input);
    Py_XDECREF(transform_mesh);
    Py_RETURN_NONE;

 error:
    Py_XDECREF(input);
    Py_XDECREF(transform_mesh);
    return NULL;
}

static PyMethodDef module_functions[] = {
    {"resample", (PyCFunction)image_resample, METH_VARARGS|METH_KEYWORDS, image_resample__doc__},
    {NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_image", NULL, 0, module_functions,
};

PyMODINIT_FUNC PyInit__image(void)
{
    PyObject *m;

    import_array();

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
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
