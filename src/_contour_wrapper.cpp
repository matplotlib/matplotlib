#include "_contour.h"
#include "mplutils.h"
#include "py_converters.h"
#include "py_exceptions.h"

/* QuadContourGenerator */

typedef struct
{
    PyObject_HEAD
    QuadContourGenerator* ptr;
} PyQuadContourGenerator;

static PyTypeObject PyQuadContourGeneratorType;

static PyObject* PyQuadContourGenerator_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    PyQuadContourGenerator* self;
    self = (PyQuadContourGenerator*)type->tp_alloc(type, 0);
    self->ptr = NULL;
    return (PyObject*)self;
}

const char* PyQuadContourGenerator_init__doc__ =
    "QuadContourGenerator(x, y, z, mask, corner_mask, chunk_size)\n"
    "--\n\n"
    "Create a new C++ QuadContourGenerator object\n";

static int PyQuadContourGenerator_init(PyQuadContourGenerator* self, PyObject* args, PyObject* kwds)
{
    QuadContourGenerator::CoordinateArray x, y, z;
    QuadContourGenerator::MaskArray mask;
    bool corner_mask;
    long chunk_size;

    if (!PyArg_ParseTuple(args, "O&O&O&O&O&l",
                          &x.converter_contiguous, &x,
                          &y.converter_contiguous, &y,
                          &z.converter_contiguous, &z,
                          &mask.converter_contiguous, &mask,
                          &convert_bool, &corner_mask,
                          &chunk_size)) {
        return -1;
    }

    if (x.empty() || y.empty() || z.empty() ||
        y.dim(0) != x.dim(0) || z.dim(0) != x.dim(0) ||
        y.dim(1) != x.dim(1) || z.dim(1) != x.dim(1)) {
        PyErr_SetString(PyExc_ValueError,
            "x, y and z must all be 2D arrays with the same dimensions");
        return -1;
    }

    if (z.dim(0) < 2 || z.dim(1) < 2) {
        PyErr_SetString(PyExc_ValueError,
            "x, y and z must all be at least 2x2 arrays");
        return -1;
    }

    // Mask array is optional, if set must be same size as other arrays.
    if (!mask.empty() && (mask.dim(0) != x.dim(0) || mask.dim(1) != x.dim(1))) {
        PyErr_SetString(PyExc_ValueError,
            "If mask is set it must be a 2D array with the same dimensions as x.");
        return -1;
    }

    CALL_CPP_INIT("QuadContourGenerator",
                  (self->ptr = new QuadContourGenerator(
                       x, y, z, mask, corner_mask, chunk_size)));
    return 0;
}

static void PyQuadContourGenerator_dealloc(PyQuadContourGenerator* self)
{
    delete self->ptr;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

const char* PyQuadContourGenerator_create_contour__doc__ =
    "create_contour(level)\n"
    "--\n\n"
    "Create and return a non-filled contour.";

static PyObject* PyQuadContourGenerator_create_contour(PyQuadContourGenerator* self, PyObject* args, PyObject* kwds)
{
    double level;
    if (!PyArg_ParseTuple(args, "d:create_contour", &level)) {
        return NULL;
    }

    PyObject* result;
    CALL_CPP("create_contour", (result = self->ptr->create_contour(level)));
    return result;
}

const char* PyQuadContourGenerator_create_filled_contour__doc__ =
    "create_filled_contour(lower_level, upper_level)\n"
    "--\n\n"
    "Create and return a filled contour";

static PyObject* PyQuadContourGenerator_create_filled_contour(PyQuadContourGenerator* self, PyObject* args, PyObject* kwds)
{
    double lower_level, upper_level;
    if (!PyArg_ParseTuple(args, "dd:create_filled_contour",
                          &lower_level, &upper_level)) {
        return NULL;
    }

    if (lower_level >= upper_level)
    {
        PyErr_SetString(PyExc_ValueError,
            "filled contour levels must be increasing");
        return NULL;
    }

    PyObject* result;
    CALL_CPP("create_filled_contour",
             (result = self->ptr->create_filled_contour(lower_level,
                                                        upper_level)));
    return result;
}

static PyTypeObject* PyQuadContourGenerator_init_type(PyObject* m, PyTypeObject* type)
{
    static PyMethodDef methods[] = {
        {"create_contour", (PyCFunction)PyQuadContourGenerator_create_contour, METH_VARARGS, PyQuadContourGenerator_create_contour__doc__},
        {"create_filled_contour", (PyCFunction)PyQuadContourGenerator_create_filled_contour, METH_VARARGS, PyQuadContourGenerator_create_filled_contour__doc__},
        {NULL}
    };

    memset(type, 0, sizeof(PyTypeObject));
    type->tp_name = "matplotlib.QuadContourGenerator";
    type->tp_doc = PyQuadContourGenerator_init__doc__;
    type->tp_basicsize = sizeof(PyQuadContourGenerator);
    type->tp_dealloc = (destructor)PyQuadContourGenerator_dealloc;
    type->tp_flags = Py_TPFLAGS_DEFAULT;
    type->tp_methods = methods;
    type->tp_new = PyQuadContourGenerator_new;
    type->tp_init = (initproc)PyQuadContourGenerator_init;

    if (PyType_Ready(type) < 0) {
        return NULL;
    }

    if (PyModule_AddObject(m, "QuadContourGenerator", (PyObject*)type)) {
        return NULL;
    }

    return type;
}


/* Module */

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_contour",
    NULL,
    0,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__contour(void)
{
    PyObject *m;

    m = PyModule_Create(&moduledef);

    if (m == NULL) {
        return NULL;
    }

    if (!PyQuadContourGenerator_init_type(m, &PyQuadContourGeneratorType)) {
        return NULL;
    }

    import_array();

    return m;
}
