#include "mplutils.h"
#include "py_converters.h"
#include "_backend_agg.h"

typedef struct
{
    PyObject_HEAD;
    RendererAgg *x;
    Py_ssize_t shape[3];
    Py_ssize_t strides[3];
    Py_ssize_t suboffsets[3];
} PyRendererAgg;

typedef struct
{
    PyObject_HEAD;
    BufferRegion *x;
    Py_ssize_t shape[3];
    Py_ssize_t strides[3];
    Py_ssize_t suboffsets[3];
} PyBufferRegion;


/**********************************************************************
 * BufferRegion
 * */

static PyObject *PyBufferRegion_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyBufferRegion *self;
    self = (PyBufferRegion *)type->tp_alloc(type, 0);
    self->x = NULL;
    return (PyObject *)self;
}

static void PyBufferRegion_dealloc(PyBufferRegion *self)
{
    delete self->x;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *PyBufferRegion_to_string(PyBufferRegion *self, PyObject *args, PyObject *kwds)
{
    return PyBytes_FromStringAndSize((const char *)self->x->get_data(),
                                     self->x->get_height() * self->x->get_stride());
}

/* TODO: This doesn't seem to be used internally.  Remove? */

static PyObject *PyBufferRegion_set_x(PyBufferRegion *self, PyObject *args, PyObject *kwds)
{
    int x;
    if (!PyArg_ParseTuple(args, "i:set_x", &x)) {
        return NULL;
    }
    self->x->get_rect().x1 = x;

    Py_RETURN_NONE;
}

static PyObject *PyBufferRegion_set_y(PyBufferRegion *self, PyObject *args, PyObject *kwds)
{
    int y;
    if (!PyArg_ParseTuple(args, "i:set_y", &y)) {
        return NULL;
    }
    self->x->get_rect().y1 = y;

    Py_RETURN_NONE;
}

static PyObject *PyBufferRegion_get_extents(PyBufferRegion *self, PyObject *args, PyObject *kwds)
{
    agg::rect_i rect = self->x->get_rect();

    return Py_BuildValue("IIII", rect.x1, rect.y1, rect.x2, rect.y2);
}

static PyObject *PyBufferRegion_to_string_argb(PyBufferRegion *self, PyObject *args, PyObject *kwds)
{
    PyObject *bufobj;
    uint8_t *buf;

    bufobj = PyBytes_FromStringAndSize(NULL, self->x->get_height() * self->x->get_stride());
    buf = (uint8_t *)PyBytes_AS_STRING(bufobj);

    CALL_CPP_CLEANUP("to_string_argb", (self->x->to_string_argb(buf)), Py_DECREF(bufobj));

    return bufobj;
}

int PyBufferRegion_get_buffer(PyBufferRegion *self, Py_buffer *buf, int flags)
{
    Py_INCREF(self);
    buf->obj = (PyObject *)self;
    buf->buf = self->x->get_data();
    buf->len = self->x->get_width() * self->x->get_height() * 4;
    buf->readonly = 0;
    buf->format = (char *)"B";
    buf->ndim = 3;
    self->shape[0] = self->x->get_height();
    self->shape[1] = self->x->get_width();
    self->shape[2] = 4;
    buf->shape = self->shape;
    self->strides[0] = self->x->get_width() * 4;
    self->strides[1] = 4;
    self->strides[2] = 1;
    buf->strides = self->strides;
    buf->suboffsets = NULL;
    buf->itemsize = 1;
    buf->internal = NULL;

    return 1;
}

static PyTypeObject PyBufferRegionType;

static PyTypeObject *PyBufferRegion_init_type(PyObject *m, PyTypeObject *type)
{
    static PyMethodDef methods[] = {
        { "to_string", (PyCFunction)PyBufferRegion_to_string, METH_NOARGS, NULL },
        { "to_string_argb", (PyCFunction)PyBufferRegion_to_string_argb, METH_NOARGS, NULL },
        { "set_x", (PyCFunction)PyBufferRegion_set_x, METH_VARARGS, NULL },
        { "set_y", (PyCFunction)PyBufferRegion_set_y, METH_VARARGS, NULL },
        { "get_extents", (PyCFunction)PyBufferRegion_get_extents, METH_NOARGS, NULL },
        { NULL }
    };

    static PyBufferProcs buffer_procs;
    memset(&buffer_procs, 0, sizeof(PyBufferProcs));
    buffer_procs.bf_getbuffer = (getbufferproc)PyBufferRegion_get_buffer;

    memset(type, 0, sizeof(PyTypeObject));
    type->tp_name = "matplotlib.backends._backend_agg.BufferRegion";
    type->tp_basicsize = sizeof(PyBufferRegion);
    type->tp_dealloc = (destructor)PyBufferRegion_dealloc;
    type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_NEWBUFFER;
    type->tp_methods = methods;
    type->tp_new = PyBufferRegion_new;
    type->tp_as_buffer = &buffer_procs;

    if (PyType_Ready(type) < 0) {
        return NULL;
    }

    /* Don't need to add to module, since you can't create buffer
       regions directly from Python */

    return type;
}

/**********************************************************************
 * RendererAgg
 * */

static PyObject *PyRendererAgg_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyRendererAgg *self;
    self = (PyRendererAgg *)type->tp_alloc(type, 0);
    self->x = NULL;
    return (PyObject *)self;
}

static int PyRendererAgg_init(PyRendererAgg *self, PyObject *args, PyObject *kwds)
{
    unsigned int width;
    unsigned int height;
    double dpi;
    int debug = 0;

    if (!PyArg_ParseTuple(args, "IId|i:RendererAgg", &width, &height, &dpi, &debug)) {
        return -1;
    }

    if (dpi <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "dpi must be positive");
        return -1;
    }

    CALL_CPP_INIT("RendererAgg", self->x = new RendererAgg(width, height, dpi))

    return 0;
}

static void PyRendererAgg_dealloc(PyRendererAgg *self)
{
    delete self->x;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *PyRendererAgg_draw_path(PyRendererAgg *self, PyObject *args, PyObject *kwds)
{
    GCAgg gc;
    py::PathIterator path;
    agg::trans_affine trans;
    PyObject *faceobj = NULL;
    agg::rgba face;

    if (!PyArg_ParseTuple(args,
                          "O&O&O&|O:draw_path",
                          &convert_gcagg,
                          &gc,
                          &convert_path,
                          &path,
                          &convert_trans_affine,
                          &trans,
                          &faceobj)) {
        return NULL;
    }

    if (!convert_face(faceobj, gc, &face)) {
        return NULL;
    }

    CALL_CPP("draw_path", (self->x->draw_path(gc, path, trans, face)));

    Py_RETURN_NONE;
}

static PyObject *PyRendererAgg_draw_text_image(PyRendererAgg *self, PyObject *args, PyObject *kwds)
{
    numpy::array_view<agg::int8u, 2> image;
    double x;
    double y;
    double angle;
    GCAgg gc;

    if (!PyArg_ParseTuple(args,
                          "O&dddO&:draw_text_image",
                          &image.converter_contiguous,
                          &image,
                          &x,
                          &y,
                          &angle,
                          &convert_gcagg,
                          &gc)) {
        return NULL;
    }

    CALL_CPP("draw_text_image", (self->x->draw_text_image(gc, image, x, y, angle)));

    Py_RETURN_NONE;
}

PyObject *PyRendererAgg_draw_markers(PyRendererAgg *self, PyObject *args, PyObject *kwds)
{
    GCAgg gc;
    py::PathIterator marker_path;
    agg::trans_affine marker_path_trans;
    py::PathIterator path;
    agg::trans_affine trans;
    PyObject *faceobj = NULL;
    agg::rgba face;

    if (!PyArg_ParseTuple(args,
                          "O&O&O&O&O&|O:draw_markers",
                          &convert_gcagg,
                          &gc,
                          &convert_path,
                          &marker_path,
                          &convert_trans_affine,
                          &marker_path_trans,
                          &convert_path,
                          &path,
                          &convert_trans_affine,
                          &trans,
                          &faceobj)) {
        return NULL;
    }

    if (!convert_face(faceobj, gc, &face)) {
        return NULL;
    }

    CALL_CPP("draw_markers",
             (self->x->draw_markers(gc, marker_path, marker_path_trans, path, trans, face)));

    Py_RETURN_NONE;
}

static PyObject *PyRendererAgg_draw_image(PyRendererAgg *self, PyObject *args, PyObject *kwds)
{
    GCAgg gc;
    double x;
    double y;
    numpy::array_view<agg::int8u, 3> image;
    double w = 0;
    double h = 0;
    agg::trans_affine trans;
    bool resize = false;

    if (!PyArg_ParseTuple(args,
                          "O&ddO&|ddO&:draw_image",
                          &convert_gcagg,
                          &gc,
                          &x,
                          &y,
                          &image.converter_contiguous,
                          &image,
                          &w,
                          &h,
                          &convert_trans_affine,
                          &trans)) {
        return NULL;
    }

    if (PyTuple_Size(args) == 4) {
        x = mpl_round(x);
        y = mpl_round(y);
    } else {
        resize = true;
    }

    CALL_CPP("draw_image", (self->x->draw_image(gc, x, y, image, w, h, trans, resize)));

    Py_RETURN_NONE;
}

static PyObject *
PyRendererAgg_draw_path_collection(PyRendererAgg *self, PyObject *args, PyObject *kwds)
{
    GCAgg gc;
    agg::trans_affine master_transform;
    PyObject *pathobj;
    numpy::array_view<const double, 3> transforms;
    numpy::array_view<const double, 2> offsets;
    agg::trans_affine offset_trans;
    numpy::array_view<const double, 2> facecolors;
    numpy::array_view<const double, 2> edgecolors;
    numpy::array_view<const double, 1> linewidths;
    DashesVector dashes;
    numpy::array_view<const uint8_t, 1> antialiaseds;
    PyObject *ignored;
    e_offset_position offset_position;

    if (!PyArg_ParseTuple(args,
                          "O&O&OO&O&O&O&O&O&O&O&OO&:draw_path_collection",
                          &convert_gcagg,
                          &gc,
                          &convert_trans_affine,
                          &master_transform,
                          &pathobj,
                          &convert_transforms,
                          &transforms,
                          &convert_points,
                          &offsets,
                          &convert_trans_affine,
                          &offset_trans,
                          &convert_colors,
                          &facecolors,
                          &convert_colors,
                          &edgecolors,
                          &linewidths.converter,
                          &linewidths,
                          &convert_dashes_vector,
                          &dashes,
                          &antialiaseds.converter,
                          &antialiaseds,
                          &ignored,
                          &convert_offset_position,
                          &offset_position)) {
        return NULL;
    }

    try
    {
        py::PathGenerator path(pathobj);

        CALL_CPP("draw_path_collection",
                 (self->x->draw_path_collection(gc,
                                                master_transform,
                                                path,
                                                transforms,
                                                offsets,
                                                offset_trans,
                                                facecolors,
                                                edgecolors,
                                                linewidths,
                                                dashes,
                                                antialiaseds,
                                                offset_position)));
    }
    catch (py::exception &e)
    {
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject *PyRendererAgg_draw_quad_mesh(PyRendererAgg *self, PyObject *args, PyObject *kwds)
{
    GCAgg gc;
    agg::trans_affine master_transform;
    size_t mesh_width;
    size_t mesh_height;
    numpy::array_view<const double, 3> coordinates;
    numpy::array_view<const double, 2> offsets;
    agg::trans_affine offset_trans;
    numpy::array_view<const double, 2> facecolors;
    int antialiased;
    numpy::array_view<const double, 2> edgecolors;

    if (!PyArg_ParseTuple(args,
                          "O&O&IIO&O&O&O&iO&:draw_quad_mesh",
                          &convert_gcagg,
                          &gc,
                          &convert_trans_affine,
                          &master_transform,
                          &mesh_width,
                          &mesh_height,
                          &coordinates.converter,
                          &coordinates,
                          &convert_points,
                          &offsets,
                          &convert_trans_affine,
                          &offset_trans,
                          &convert_colors,
                          &facecolors,
                          &antialiased,
                          &convert_colors,
                          &edgecolors)) {
        return NULL;
    }

    CALL_CPP("draw_quad_mesh",
             (self->x->draw_quad_mesh(gc,
                                      master_transform,
                                      mesh_width,
                                      mesh_height,
                                      coordinates,
                                      offsets,
                                      offset_trans,
                                      facecolors,
                                      antialiased,
                                      edgecolors)));

    Py_RETURN_NONE;
}

static PyObject *
PyRendererAgg_draw_gouraud_triangle(PyRendererAgg *self, PyObject *args, PyObject *kwds)
{
    GCAgg gc;
    numpy::array_view<const double, 2> points;
    numpy::array_view<const double, 2> colors;
    agg::trans_affine trans;

    if (!PyArg_ParseTuple(args,
                          "O&O&O&O&|O:draw_gouraud_triangle",
                          &convert_gcagg,
                          &gc,
                          &points.converter,
                          &points,
                          &colors.converter,
                          &colors,
                          &convert_trans_affine,
                          &trans)) {
        return NULL;
    }

    if (points.dim(0) != 3 || points.dim(1) != 2) {
        PyErr_Format(PyExc_ValueError,
                     "points must be a 3x2 array, got %dx%d",
                     points.dim(0), points.dim(1));
        return NULL;
    }

    if (colors.dim(0) != 3 || colors.dim(1) != 4) {
        PyErr_Format(PyExc_ValueError,
                     "colors must be a 3x4 array, got %dx%d",
                     colors.dim(0), colors.dim(1));
        return NULL;
    }


    CALL_CPP("draw_gouraud_triangle", (self->x->draw_gouraud_triangle(gc, points, colors, trans)));

    Py_RETURN_NONE;
}

static PyObject *
PyRendererAgg_draw_gouraud_triangles(PyRendererAgg *self, PyObject *args, PyObject *kwds)
{
    GCAgg gc;
    numpy::array_view<const double, 3> points;
    numpy::array_view<const double, 3> colors;
    agg::trans_affine trans;

    if (!PyArg_ParseTuple(args,
                          "O&O&O&O&|O:draw_gouraud_triangles",
                          &convert_gcagg,
                          &gc,
                          &points.converter,
                          &points,
                          &colors.converter,
                          &colors,
                          &convert_trans_affine,
                          &trans)) {
        return NULL;
    }

    if (points.size() != 0 && (points.dim(1) != 3 || points.dim(2) != 2)) {
        PyErr_Format(PyExc_ValueError,
                     "points must be a Nx3x2 array, got %dx%dx%d",
                     points.dim(0), points.dim(1), points.dim(2));
        return NULL;
    }

    if (colors.size() != 0 && (colors.dim(1) != 3 || colors.dim(2) != 4)) {
        PyErr_Format(PyExc_ValueError,
                     "colors must be a Nx3x4 array, got %dx%dx%d",
                     colors.dim(0), colors.dim(1), colors.dim(2));
        return NULL;
    }

    if (points.size() != colors.size()) {
        PyErr_Format(PyExc_ValueError,
                     "points and colors arrays must be the same length, got %d and %d",
                     points.dim(0), colors.dim(0));
        return NULL;
    }

    CALL_CPP("draw_gouraud_triangles", self->x->draw_gouraud_triangles(gc, points, colors, trans));

    Py_RETURN_NONE;
}

static PyObject *PyRendererAgg_tostring_rgb(PyRendererAgg *self, PyObject *args, PyObject *kwds)
{
    PyObject *buffobj = NULL;

    buffobj = PyBytes_FromStringAndSize(NULL, self->x->get_width() * self->x->get_height() * 3);
    if (buffobj == NULL) {
        return NULL;
    }

    CALL_CPP_CLEANUP("tostring_rgb",
                     (self->x->tostring_rgb((uint8_t *)PyBytes_AS_STRING(buffobj))),
                     Py_DECREF(buffobj));

    return buffobj;
}

static PyObject *PyRendererAgg_tostring_argb(PyRendererAgg *self, PyObject *args, PyObject *kwds)
{
    PyObject *buffobj = NULL;

    buffobj = PyBytes_FromStringAndSize(NULL, self->x->get_width() * self->x->get_height() * 4);
    if (buffobj == NULL) {
        return NULL;
    }

    CALL_CPP_CLEANUP("tostring_argb",
                     (self->x->tostring_argb((uint8_t *)PyBytes_AS_STRING(buffobj))),
                     Py_DECREF(buffobj));

    return buffobj;
}

static PyObject *PyRendererAgg_tostring_bgra(PyRendererAgg *self, PyObject *args, PyObject *kwds)
{
    PyObject *buffobj = NULL;

    buffobj = PyBytes_FromStringAndSize(NULL, self->x->get_width() * self->x->get_height() * 4);
    if (buffobj == NULL) {
        return NULL;
    }

    CALL_CPP_CLEANUP("to_string_bgra",
                     (self->x->tostring_bgra((uint8_t *)PyBytes_AS_STRING(buffobj))),
                     Py_DECREF(buffobj));

    return buffobj;
}

static PyObject *
PyRendererAgg_get_content_extents(PyRendererAgg *self, PyObject *args, PyObject *kwds)
{
    agg::rect_i extents;

    CALL_CPP("get_content_extents", (extents = self->x->get_content_extents()));

    return Py_BuildValue(
        "iiii", extents.x1, extents.y1, extents.x2 - extents.x1, extents.y2 - extents.y1);
}

static PyObject *PyRendererAgg_buffer_rgba(PyRendererAgg *self, PyObject *args, PyObject *kwds)
{
#if PY3K
    return PyBytes_FromStringAndSize((const char *)self->x->pixBuffer,
                                     self->x->get_width() * self->x->get_height() * 4);
#else
    return PyBuffer_FromReadWriteMemory(self->x->pixBuffer,
                                        self->x->get_width() * self->x->get_height() * 4);
#endif
}

int PyRendererAgg_get_buffer(PyRendererAgg *self, Py_buffer *buf, int flags)
{
    Py_INCREF(self);
    buf->obj = (PyObject *)self;
    buf->buf = self->x->pixBuffer;
    buf->len = self->x->get_width() * self->x->get_height() * 4;
    buf->readonly = 0;
    buf->format = (char *)"B";
    buf->ndim = 3;
    self->shape[0] = self->x->get_height();
    self->shape[1] = self->x->get_width();
    self->shape[2] = 4;
    buf->shape = self->shape;
    self->strides[0] = self->x->get_width() * 4;
    self->strides[1] = 4;
    self->strides[2] = 1;
    buf->strides = self->strides;
    buf->suboffsets = NULL;
    buf->itemsize = 1;
    buf->internal = NULL;

    return 1;
}

static PyObject *PyRendererAgg_clear(PyRendererAgg *self, PyObject *args, PyObject *kwds)
{
    CALL_CPP("clear", self->x->clear());

    Py_RETURN_NONE;
}

static PyObject *PyRendererAgg_copy_from_bbox(PyRendererAgg *self, PyObject *args, PyObject *kwds)
{
    agg::rect_d bbox;
    BufferRegion *reg;
    PyObject *regobj;

    if (!PyArg_ParseTuple(args, "O&:copy_from_bbox", &convert_rect, &bbox)) {
        return 0;
    }

    CALL_CPP("copy_from_bbox", (reg = self->x->copy_from_bbox(bbox)));

    regobj = PyBufferRegion_new(&PyBufferRegionType, NULL, NULL);
    ((PyBufferRegion *)regobj)->x = reg;

    return regobj;
}

static PyObject *PyRendererAgg_restore_region(PyRendererAgg *self, PyObject *args, PyObject *kwds)
{
    PyBufferRegion *regobj;
    int xx1 = 0, yy1 = 0, xx2 = 0, yy2 = 0, x = 0, y = 0;

    if (!PyArg_ParseTuple(args,
                          "O!|iiiiii:restore_region",
                          &PyBufferRegionType,
                          &regobj,
                          &xx1,
                          &yy1,
                          &xx2,
                          &yy2,
                          &x,
                          &y)) {
        return 0;
    }

    if (PySequence_Size(args) == 1) {
        CALL_CPP("restore_region", (self->x->restore_region(*(regobj->x))));
    } else {
        CALL_CPP("restore_region", self->x->restore_region(*(regobj->x), xx1, yy1, xx2, yy2, x, y));
    }

    Py_RETURN_NONE;
}

PyTypeObject PyRendererAggType;

static PyTypeObject *PyRendererAgg_init_type(PyObject *m, PyTypeObject *type)
{
    static PyMethodDef methods[] = {
        {"draw_path", (PyCFunction)PyRendererAgg_draw_path, METH_VARARGS, NULL},
        {"draw_markers", (PyCFunction)PyRendererAgg_draw_markers, METH_VARARGS, NULL},
        {"draw_text_image", (PyCFunction)PyRendererAgg_draw_text_image, METH_VARARGS, NULL},
        {"draw_image", (PyCFunction)PyRendererAgg_draw_image, METH_VARARGS, NULL},
        {"draw_path_collection", (PyCFunction)PyRendererAgg_draw_path_collection, METH_VARARGS, NULL},
        {"draw_quad_mesh", (PyCFunction)PyRendererAgg_draw_quad_mesh, METH_VARARGS, NULL},
        {"draw_gouraud_triangle", (PyCFunction)PyRendererAgg_draw_gouraud_triangle, METH_VARARGS, NULL},
        {"draw_gouraud_triangles", (PyCFunction)PyRendererAgg_draw_gouraud_triangles, METH_VARARGS, NULL},

        {"tostring_rgb", (PyCFunction)PyRendererAgg_tostring_rgb, METH_NOARGS, NULL},
        {"tostring_argb", (PyCFunction)PyRendererAgg_tostring_argb, METH_NOARGS, NULL},
        {"tostring_bgra", (PyCFunction)PyRendererAgg_tostring_bgra, METH_NOARGS, NULL},
        {"get_content_extents", (PyCFunction)PyRendererAgg_get_content_extents, METH_NOARGS, NULL},
        {"buffer_rgba", (PyCFunction)PyRendererAgg_buffer_rgba, METH_NOARGS, NULL},
        {"clear", (PyCFunction)PyRendererAgg_clear, METH_NOARGS, NULL},

        {"copy_from_bbox", (PyCFunction)PyRendererAgg_copy_from_bbox, METH_VARARGS, NULL},
        {"restore_region", (PyCFunction)PyRendererAgg_restore_region, METH_VARARGS, NULL},
        {NULL}
    };

    static PyBufferProcs buffer_procs;
    memset(&buffer_procs, 0, sizeof(PyBufferProcs));
    buffer_procs.bf_getbuffer = (getbufferproc)PyRendererAgg_get_buffer;

    memset(type, 0, sizeof(PyTypeObject));
    type->tp_name = "matplotlib.backends._backend_agg.RendererAgg";
    type->tp_basicsize = sizeof(PyRendererAgg);
    type->tp_dealloc = (destructor)PyRendererAgg_dealloc;
    type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_NEWBUFFER;
    type->tp_methods = methods;
    type->tp_init = (initproc)PyRendererAgg_init;
    type->tp_new = PyRendererAgg_new;
    type->tp_as_buffer = &buffer_procs;

    if (PyType_Ready(type) < 0) {
        return NULL;
    }

    if (PyModule_AddObject(m, "RendererAgg", (PyObject *)type)) {
        return NULL;
    }

    return type;
}

extern "C" {

#if PY3K
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_backend_agg",
    NULL,
    0,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC PyInit__backend_agg(void)

#else
#define INITERROR return

PyMODINIT_FUNC init_backend_agg(void)
#endif

{
    PyObject *m;

#if PY3K
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3("_backend_agg", NULL, NULL);
#endif

    if (m == NULL) {
        INITERROR;
    }

    import_array();

    if (!PyRendererAgg_init_type(m, &PyRendererAggType)) {
        INITERROR;
    }

    if (!PyBufferRegion_init_type(m, &PyBufferRegionType)) {
        INITERROR;
    }

#if PY3K
    return m;
#endif
}

} // extern "C"
