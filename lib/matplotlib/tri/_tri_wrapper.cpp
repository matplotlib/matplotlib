#include "_tri_wrapper.h"
#include "src/mplutils.h"
#include "src/py_exceptions.h"


/* Triangulation */

static PyTypeObject PyTriangulationType;

static PyObject* PyTriangulation_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    PyTriangulation* self;
    self = (PyTriangulation*)type->tp_alloc(type, 0);
    self->ptr = NULL;
    return (PyObject*)self;
}

const char* PyTriangulation_init__doc__ =
    "Triangulation(x, y, triangles, mask, edges, neighbors)\n"
    "\n"
    "Create a new C++ Triangulation object\n"
    "This should not be called directly, instead use the python class\n"
    "matplotlib.tri.Triangulation instead.\n";

static int PyTriangulation_init(PyTriangulation* self, PyObject* args, PyObject* kwds)
{
    PyObject* x_arg;
    PyObject* y_arg ;
    PyObject* triangles_arg;
    PyObject* mask_arg;
    PyObject* edges_arg;
    PyObject* neighbors_arg;
    if (!PyArg_ParseTuple(args, "OOOOOO", &x_arg, &y_arg, &triangles_arg,
                          &mask_arg, &edges_arg, &neighbors_arg)) {
        return -1;
    }

    // x and y.
    PyArrayObject* x = (PyArrayObject*)PyArray_ContiguousFromObject(
                           x_arg, NPY_DOUBLE, 1, 1);
    PyArrayObject* y = (PyArrayObject*)PyArray_ContiguousFromObject(
                           y_arg, NPY_DOUBLE, 1, 1);
    if (x == 0 || y == 0 || PyArray_DIM(x,0) != PyArray_DIM(y,0)) {
        Py_XDECREF(x);
        Py_XDECREF(y);
        PyErr_SetString(PyExc_ValueError,
            "x and y must be 1D arrays of the same length");
    }

    // triangles.
    PyArrayObject* triangles = (PyArrayObject*)PyArray_ContiguousFromObject(
                                   triangles_arg, NPY_INT, 2, 2);
    if (triangles == 0 || PyArray_DIM(triangles,1) != 3) {
        Py_XDECREF(x);
        Py_XDECREF(y);
        Py_XDECREF(triangles);
        PyErr_SetString(PyExc_ValueError,
            "triangles must be a 2D array of shape (?,3)");
    }

    // Optional mask.
    PyArrayObject* mask = 0;
    if (mask_arg != 0 && mask_arg != Py_None)
    {
        mask = (PyArrayObject*)PyArray_ContiguousFromObject(
                   mask_arg, NPY_BOOL, 1, 1);
        if (mask == 0 || PyArray_DIM(mask,0) != PyArray_DIM(triangles,0)) {
            Py_XDECREF(x);
            Py_XDECREF(y);
            Py_XDECREF(triangles);
            Py_XDECREF(mask);
            PyErr_SetString(PyExc_ValueError,
                "mask must be a 1D array with the same length as the triangles array");
        }
    }

    // Optional edges.
    PyArrayObject* edges = 0;
    if (edges_arg != 0 && edges_arg != Py_None)
    {
        edges = (PyArrayObject*)PyArray_ContiguousFromObject(
                    edges_arg, NPY_INT, 2, 2);
        if (edges == 0 || PyArray_DIM(edges,1) != 2) {
            Py_XDECREF(x);
            Py_XDECREF(y);
            Py_XDECREF(triangles);
            Py_XDECREF(mask);
            Py_XDECREF(edges);
            PyErr_SetString(PyExc_ValueError,
                "edges must be a 2D array with shape (?,2)");
        }
    }

    // Optional neighbors.
    PyArrayObject* neighbors = 0;
    if (neighbors_arg != 0 && neighbors_arg != Py_None)
    {
        neighbors = (PyArrayObject*)PyArray_ContiguousFromObject(
                        neighbors_arg, NPY_INT, 2, 2);
        if (neighbors == 0 ||
            PyArray_DIM(neighbors,0) != PyArray_DIM(triangles,0) ||
            PyArray_DIM(neighbors,1) != PyArray_DIM(triangles,1)) {
            Py_XDECREF(x);
            Py_XDECREF(y);
            Py_XDECREF(triangles);
            Py_XDECREF(mask);
            Py_XDECREF(edges);
            Py_XDECREF(neighbors);
            PyErr_SetString(PyExc_ValueError,
                "neighbors must be a 2D array with the same shape as the triangles array");
        }
    }

    CALL_CPP_INIT("Triangulation",
                  (self->ptr = new Triangulation(x, y, triangles, mask, edges,
                                                 neighbors)));
    return 0;
}

static void PyTriangulation_dealloc(PyTriangulation* self)
{
    delete self->ptr;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

const char* PyTriangulation_calculate_plane_coefficients__doc__ =
    "calculate_plane_coefficients(z, plane_coefficients)\n"
    "\n"
    "Calculate plane equation coefficients for all unmasked triangles";

static PyObject* PyTriangulation_calculate_plane_coefficients(PyTriangulation* self, PyObject* args, PyObject* kwds)
{
    PyObject* z_arg;
    if (!PyArg_ParseTuple(args, "O:calculate_plane_coefficients", &z_arg)) {
        return NULL;
    }

    PyArrayObject* z = (PyArrayObject*)PyArray_ContiguousFromObject(
                           z_arg, NPY_DOUBLE, 1, 1);
    if (z == 0 || PyArray_DIM(z,0) != self->ptr->get_npoints()) {
        Py_XDECREF(z);
        PyErr_SetString(PyExc_ValueError,
            "z array must have same length as triangulation x and y arrays");
    }

    npy_intp dims[2] = {self->ptr->get_ntri(), 3};
    PyArrayObject* result = (PyArrayObject*)PyArray_SimpleNew(
                                2, dims, NPY_DOUBLE);

    CALL_CPP_CLEANUP("calculate_plane_coefficients",
                     (self->ptr->calculate_plane_coefficients(z, result)),
                      Py_XDECREF(z); Py_XDECREF(result));

    Py_XDECREF(z);
    return (PyObject*)result;
}

const char* PyTriangulation_get_edges__doc__ =
    "get_edges()\n"
    "\n"
    "Return edges array";

static PyObject* PyTriangulation_get_edges(PyTriangulation* self, PyObject* args, PyObject* kwds)
{
    PyArrayObject* result;
    CALL_CPP("get_edges", (result = self->ptr->get_edges()));
    Py_XINCREF(result);
    return (PyObject*)result;
}

const char* PyTriangulation_get_neighbors__doc__ =
    "get_neighbors()\n"
    "\n"
    "Return neighbors array";

static PyObject* PyTriangulation_get_neighbors(PyTriangulation* self, PyObject* args, PyObject* kwds)
{
    PyArrayObject* result;
    CALL_CPP("get_neighbors", (result = self->ptr->get_neighbors()));
    Py_XINCREF(result);
    return (PyObject*)result;
}

const char* PyTriangulation_set_mask__doc__ =
    "set_mask(mask)\n"
    "\n"
    "Set or clear the mask array.";

static PyObject* PyTriangulation_set_mask(PyTriangulation* self, PyObject* args, PyObject* kwds)
{
    PyObject* mask_arg;
    if (!PyArg_ParseTuple(args, "O:set_mask", &mask_arg)) {
        return NULL;
    }

    // Optional mask.
    PyArrayObject* mask = 0;
    if (mask_arg != 0 && mask_arg != Py_None)
    {
        mask = (PyArrayObject*)PyArray_ContiguousFromObject(
                   mask_arg, NPY_BOOL, 1, 1);
        if (mask == 0 || PyArray_DIM(mask,0) != self->ptr->get_ntri()) {
            Py_XDECREF(mask);
            PyErr_SetString(PyExc_ValueError,
                "mask must be a 1D array with the same length as the triangles array");
        }
    }

    CALL_CPP("set_mask", (self->ptr->set_mask(mask)));

    Py_RETURN_NONE;
}

static PyTypeObject* PyTriangulation_init_type(PyObject* m, PyTypeObject* type)
{
    static PyMethodDef methods[] = {
        {"calculate_plane_coefficients", (PyCFunction)PyTriangulation_calculate_plane_coefficients, METH_VARARGS, PyTriangulation_calculate_plane_coefficients__doc__},
        {"get_edges", (PyCFunction)PyTriangulation_get_edges, METH_NOARGS, PyTriangulation_get_edges__doc__},
        {"get_neighbors", (PyCFunction)PyTriangulation_get_neighbors, METH_NOARGS, PyTriangulation_get_neighbors__doc__},
        {"set_mask", (PyCFunction)PyTriangulation_set_mask, METH_VARARGS, PyTriangulation_set_mask__doc__},
        {NULL}
    };

    memset(type, 0, sizeof(PyTypeObject));
    type->tp_name = "matplotlib._tri.Triangulation";
    type->tp_doc = PyTriangulation_init__doc__;
    type->tp_basicsize = sizeof(PyTriangulation);
    type->tp_dealloc = (destructor)PyTriangulation_dealloc;
    type->tp_flags = Py_TPFLAGS_DEFAULT;
    type->tp_methods = methods;
    type->tp_new = PyTriangulation_new;
    type->tp_init = (initproc)PyTriangulation_init;

    if (PyType_Ready(type) < 0) {
        return NULL;
    }

    if (PyModule_AddObject(m, "Triangulation", (PyObject*)type)) {
        return NULL;
    }

    return type;
}


/* TriContourGenerator */

static PyTypeObject PyTriContourGeneratorType;

static PyObject* PyTriContourGenerator_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    PyTriContourGenerator* self;
    self = (PyTriContourGenerator*)type->tp_alloc(type, 0);
    self->ptr = NULL;
    return (PyObject*)self;
}

const char* PyTriContourGenerator_init__doc__ =
    "TriContourGenerator(triangulation, z)\n"
    "\n"
    "Create a new C++ TriContourGenerator object\n"
    "This should not be called directly, instead use the functions\n"
    "matplotlib.axes.tricontour and tricontourf instead.\n";

static int PyTriContourGenerator_init(PyTriContourGenerator* self, PyObject* args, PyObject* kwds)
{
    PyObject* triangulation;
    PyObject* z_arg;
    if (!PyArg_ParseTuple(args, "O!O", &PyTriangulationType, &triangulation,
                          &z_arg)) {
        return -1;
    }
    Py_INCREF(triangulation);

    int npoints = ((PyTriangulation*)triangulation)->ptr->get_npoints();

    PyArrayObject* z = (PyArrayObject*)PyArray_ContiguousFromObject(
                           z_arg, NPY_DOUBLE, 1, 1);
    if (z == 0 && PyArray_DIM(z,0) != npoints) {
        Py_DECREF(triangulation);
        Py_XDECREF(z);
        PyErr_SetString(PyExc_ValueError,
            "z must be a 1D array with the same length as the x and y arrays");
    }

    CALL_CPP_INIT("TriContourGenerator",
                  (self->ptr = new TriContourGenerator(triangulation, z)));

    return 0;
}

static void PyTriContourGenerator_dealloc(PyTriContourGenerator* self)
{
    delete self->ptr;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

const char* PyTriContourGenerator_create_contour__doc__ =
    "create_contour(level)\n"
    "\n"
    "Create and return a non-filled contour.";

static PyObject* PyTriContourGenerator_create_contour(PyTriContourGenerator* self, PyObject* args, PyObject* kwds)
{
    double level;
    if (!PyArg_ParseTuple(args, "d:create_contour", &level)) {
        return NULL;
    }

    PyObject* result;
    CALL_CPP("get_edges", (result = self->ptr->create_contour(level)));
    return result;
}

const char* PyTriContourGenerator_create_filled_contour__doc__ =
    "create_filled_contour(lower_level, upper_level)\n"
    "\n"
    "Create and return a filled contour";

static PyObject* PyTriContourGenerator_create_filled_contour(PyTriContourGenerator* self, PyObject* args, PyObject* kwds)
{
    double lower_level, upper_level;
    if (!PyArg_ParseTuple(args, "dd:create_contour", &lower_level, &upper_level)) {
        return NULL;
    }

    PyObject* result;
    CALL_CPP("get_edges",
             (result = self->ptr->create_filled_contour(lower_level,
                                                        upper_level)));
    return result;
}

static PyTypeObject* PyTriContourGenerator_init_type(PyObject* m, PyTypeObject* type)
{
    static PyMethodDef methods[] = {
        {"create_contour", (PyCFunction)PyTriContourGenerator_create_contour, METH_VARARGS, PyTriContourGenerator_create_contour__doc__},
        {"create_filled_contour", (PyCFunction)PyTriContourGenerator_create_filled_contour, METH_VARARGS, PyTriContourGenerator_create_filled_contour__doc__},
        {NULL}
    };

    memset(type, 0, sizeof(PyTypeObject));
    type->tp_name = "matplotlib._tri.TriContourGenerator";
    type->tp_doc = PyTriContourGenerator_init__doc__;
    type->tp_basicsize = sizeof(PyTriContourGenerator);
    type->tp_dealloc = (destructor)PyTriContourGenerator_dealloc;
    type->tp_flags = Py_TPFLAGS_DEFAULT;
    type->tp_methods = methods;
    type->tp_new = PyTriContourGenerator_new;
    type->tp_init = (initproc)PyTriContourGenerator_init;

    if (PyType_Ready(type) < 0) {
        return NULL;
    }

    if (PyModule_AddObject(m, "TriContourGenerator", (PyObject*)type)) {
        return NULL;
    }

    return type;
}


/* TrapezoidMapTriFinder */

static PyTypeObject PyTrapezoidMapTriFinderType;

static PyObject* PyTrapezoidMapTriFinder_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    PyTrapezoidMapTriFinder* self;
    self = (PyTrapezoidMapTriFinder*)type->tp_alloc(type, 0);
    self->ptr = NULL;
    return (PyObject*)self;
}

const char* PyTrapezoidMapTriFinder_init__doc__ =
    "TrapezoidMapTriFinder(triangulation)\n"
    "\n"
    "Create a new C++ TrapezoidMapTriFinder object\n"
    "This should not be called directly, instead use the python class\n"
    "matplotlib.tri.TrapezoidMapTriFinder instead.\n";

static int PyTrapezoidMapTriFinder_init(PyTrapezoidMapTriFinder* self, PyObject* args, PyObject* kwds)
{
    PyObject* triangulation;
    if (!PyArg_ParseTuple(args, "O!", &PyTriangulationType, &triangulation)) {
        return -1;
    }
    Py_INCREF(triangulation);

    CALL_CPP_INIT("TrapezoidMapTriFinder",
                  (self->ptr = new TrapezoidMapTriFinder(triangulation)));
    return 0;
}

static void PyTrapezoidMapTriFinder_dealloc(PyTrapezoidMapTriFinder* self)
{
    delete self->ptr;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

const char* PyTrapezoidMapTriFinder_find_many__doc__ =
    "find_many(x, y)\n"
    "\n"
    "Find indices of triangles containing the point coordinates (x, y)";

static PyObject* PyTrapezoidMapTriFinder_find_many(PyTrapezoidMapTriFinder* self, PyObject* args, PyObject* kwds)
{
    PyObject* x_arg;
    PyObject* y_arg;
    if (!PyArg_ParseTuple(args, "OO:find_many", &x_arg, &y_arg)) {
        return NULL;
    }

    PyArrayObject* x = (PyArrayObject*)PyArray_ContiguousFromObject(
                           x_arg, NPY_DOUBLE, 0, 0);
    PyArrayObject* y = (PyArrayObject*)PyArray_ContiguousFromObject(
                           y_arg, NPY_DOUBLE, 0, 0);

    bool ok = (x != 0 && y != 0 && PyArray_NDIM(x) == PyArray_NDIM(y));
    int ndim = (x == 0 ? 0 : PyArray_NDIM(x));
    for (int i = 0; ok && i < ndim; ++i)
        ok = (PyArray_DIM(x,i) == PyArray_DIM(y,i));

    if (!ok) {
        Py_XDECREF(x);
        Py_XDECREF(y);
        PyErr_SetString(PyExc_ValueError,
            "x and y must be array_like with same shape");
    }

    PyArrayObject* result;
    CALL_CPP_CLEANUP("find_many",
                     (result = self->ptr->find_many(x, y)),
                     Py_XDECREF(x); Py_XDECREF(y));

    Py_XDECREF(x);
    Py_XDECREF(y);
    return (PyObject*)result;
}

const char* PyTrapezoidMapTriFinder_get_tree_stats__doc__ =
    "get_tree_stats()\n"
    "\n"
    "Return statistics about the tree used by the trapezoid map";

static PyObject* PyTrapezoidMapTriFinder_get_tree_stats(PyTrapezoidMapTriFinder* self, PyObject* args, PyObject* kwds)
{
    PyObject* result;
    CALL_CPP("get_tree_stats", (result = self->ptr->get_tree_stats()));
    return result;
}

const char* PyTrapezoidMapTriFinder_initialize__doc__ =
    "initialize()\n"
    "\n"
    "Initialize this object, creating the trapezoid map from the triangulation";

static PyObject* PyTrapezoidMapTriFinder_initialize(PyTrapezoidMapTriFinder* self, PyObject* args, PyObject* kwds)
{
    CALL_CPP("initialize", (self->ptr->initialize()));
    Py_RETURN_NONE;
}

const char* PyTrapezoidMapTriFinder_print_tree__doc__ =
    "print_tree()\n"
    "\n"
    "Print the search tree as text to stdout; useful for debug purposes";

static PyObject* PyTrapezoidMapTriFinder_print_tree(PyTrapezoidMapTriFinder* self, PyObject* args, PyObject* kwds)
{
    CALL_CPP("print_tree", (self->ptr->print_tree()));
    Py_RETURN_NONE;
}

static PyTypeObject* PyTrapezoidMapTriFinder_init_type(PyObject* m, PyTypeObject* type)
{
    static PyMethodDef methods[] = {
        {"find_many", (PyCFunction)PyTrapezoidMapTriFinder_find_many, METH_VARARGS, PyTrapezoidMapTriFinder_find_many__doc__},
        {"get_tree_stats", (PyCFunction)PyTrapezoidMapTriFinder_get_tree_stats, METH_NOARGS, PyTrapezoidMapTriFinder_get_tree_stats__doc__},
        {"initialize", (PyCFunction)PyTrapezoidMapTriFinder_initialize, METH_NOARGS, PyTrapezoidMapTriFinder_initialize__doc__},
        {"print_tree", (PyCFunction)PyTrapezoidMapTriFinder_print_tree, METH_NOARGS, PyTrapezoidMapTriFinder_print_tree__doc__},
        {NULL}
    };

    memset(type, 0, sizeof(PyTypeObject));
    type->tp_name = "matplotlib._tri.TrapezoidMapTriFinder";
    type->tp_doc = PyTrapezoidMapTriFinder_init__doc__;
    type->tp_basicsize = sizeof(PyTrapezoidMapTriFinder);
    type->tp_dealloc = (destructor)PyTrapezoidMapTriFinder_dealloc;
    type->tp_flags = Py_TPFLAGS_DEFAULT;
    type->tp_methods = methods;
    type->tp_new = PyTrapezoidMapTriFinder_new;
    type->tp_init = (initproc)PyTrapezoidMapTriFinder_init;

    if (PyType_Ready(type) < 0) {
        return NULL;
    }

    if (PyModule_AddObject(m, "TrapezoidMapTriFinder", (PyObject*)type)) {
        return NULL;
    }

    return type;
}


/* Module */

extern "C" {

struct module_state
{
/* The Sun compiler can't handle empty structs */
#if defined(__SUNPRO_C) || defined(_MSC_VER)
    int _dummy;
#endif
};

#if PY3K
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_tri",
    NULL,
    sizeof(struct module_state),
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC PyInit__tri(void)

#else
#define INITERROR return

PyMODINIT_FUNC init_tri(void)
#endif

{
    PyObject *m;

#if PY3K
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3("_tri", NULL, NULL);
#endif

    if (m == NULL) {
        INITERROR;
    }

    if (!PyTriangulation_init_type(m, &PyTriangulationType)) {
        INITERROR;
    }
    if (!PyTriContourGenerator_init_type(m, &PyTriContourGeneratorType)) {
        INITERROR;
    }
    if (!PyTrapezoidMapTriFinder_init_type(m, &PyTrapezoidMapTriFinderType)) {
        INITERROR;
    }

    import_array();

#if PY3K
    return m;
#endif
}

} // extern "C"
