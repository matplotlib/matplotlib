#define NO_IMPORT_ARRAY
#define PY_SSIZE_T_CLEAN
#include "py_converters.h"
#include "numpy_cpp.h"

#include "agg_basics.h"
#include "agg_color_rgba.h"
#include "agg_math_stroke.h"

extern "C" {

static int convert_string_enum(PyObject *obj, const char *name, const char **names, int *values, int *result)
{
    PyObject *bytesobj;
    char *str;

    if (obj == NULL || obj == Py_None) {
        return 1;
    }

    if (PyUnicode_Check(obj)) {
        bytesobj = PyUnicode_AsASCIIString(obj);
        if (bytesobj == NULL) {
            return 0;
        }
    } else if (PyBytes_Check(obj)) {
        Py_INCREF(obj);
        bytesobj = obj;
    } else {
        PyErr_Format(PyExc_TypeError, "%s must be str or bytes", name);
        return 0;
    }

    str = PyBytes_AsString(bytesobj);
    if (str == NULL) {
        Py_DECREF(bytesobj);
        return 0;
    }

    for ( ; *names != NULL; names++, values++) {
        if (strncmp(str, *names, 64) == 0) {
            *result = *values;
            Py_DECREF(bytesobj);
            return 1;
        }
    }

    PyErr_Format(PyExc_ValueError, "invalid %s value", name);
    Py_DECREF(bytesobj);
    return 0;
}

int convert_from_method(PyObject *obj, const char *name, converter func, void *p)
{
    PyObject *value;

    value = PyObject_CallMethod(obj, name, NULL);
    if (value == NULL) {
        if (!PyObject_HasAttrString(obj, name)) {
            PyErr_Clear();
            return 1;
        }
        return 0;
    }

    if (!func(value, p)) {
        Py_DECREF(value);
        return 0;
    }

    Py_DECREF(value);
    return 1;
}

int convert_from_attr(PyObject *obj, const char *name, converter func, void *p)
{
    PyObject *value;

    value = PyObject_GetAttrString(obj, name);
    if (value == NULL) {
        if (!PyObject_HasAttrString(obj, name)) {
            PyErr_Clear();
            return 1;
        }
        return 0;
    }

    if (!func(value, p)) {
        Py_DECREF(value);
        return 0;
    }

    Py_DECREF(value);
    return 1;
}

int convert_double(PyObject *obj, void *p)
{
    double *val = (double *)p;

    *val = PyFloat_AsDouble(obj);
    if (PyErr_Occurred()) {
        return 0;
    }

    return 1;
}

int convert_bool(PyObject *obj, void *p)
{
    bool *val = (bool *)p;
    switch (PyObject_IsTrue(obj)) {
        case 0: *val = false; break;
        case 1: *val = true; break;
        default: return 0;  // errored.
    }
    return 1;
}

int convert_cap(PyObject *capobj, void *capp)
{
    const char *names[] = {"butt", "round", "projecting", NULL};
    int values[] = {agg::butt_cap, agg::round_cap, agg::square_cap};
    int result = agg::butt_cap;

    if (!convert_string_enum(capobj, "capstyle", names, values, &result)) {
        return 0;
    }

    *(agg::line_cap_e *)capp = (agg::line_cap_e)result;
    return 1;
}

int convert_join(PyObject *joinobj, void *joinp)
{
    const char *names[] = {"miter", "round", "bevel", NULL};
    int values[] = {agg::miter_join_revert, agg::round_join, agg::bevel_join};
    int result = agg::miter_join_revert;

    if (!convert_string_enum(joinobj, "joinstyle", names, values, &result)) {
        return 0;
    }

    *(agg::line_join_e *)joinp = (agg::line_join_e)result;
    return 1;
}

int convert_rect(PyObject *rectobj, void *rectp)
{
    agg::rect_d *rect = (agg::rect_d *)rectp;

    if (rectobj == NULL || rectobj == Py_None) {
        rect->x1 = 0.0;
        rect->y1 = 0.0;
        rect->x2 = 0.0;
        rect->y2 = 0.0;
    } else {
        PyArrayObject *rect_arr = (PyArrayObject *)PyArray_ContiguousFromAny(
                rectobj, NPY_DOUBLE, 1, 2);
        if (rect_arr == NULL) {
            return 0;
        }

        if (PyArray_NDIM(rect_arr) == 2) {
            if (PyArray_DIM(rect_arr, 0) != 2 ||
                PyArray_DIM(rect_arr, 1) != 2) {
                PyErr_SetString(PyExc_ValueError, "Invalid bounding box");
                Py_DECREF(rect_arr);
                return 0;
            }

        } else {  // PyArray_NDIM(rect_arr) == 1
            if (PyArray_DIM(rect_arr, 0) != 4) {
                PyErr_SetString(PyExc_ValueError, "Invalid bounding box");
                Py_DECREF(rect_arr);
                return 0;
            }
        }

        double *buff = (double *)PyArray_DATA(rect_arr);
        rect->x1 = buff[0];
        rect->y1 = buff[1];
        rect->x2 = buff[2];
        rect->y2 = buff[3];

        Py_DECREF(rect_arr);
    }
    return 1;
}

int convert_rgba(PyObject *rgbaobj, void *rgbap)
{
    agg::rgba *rgba = (agg::rgba *)rgbap;

    if (rgbaobj == NULL || rgbaobj == Py_None) {
        rgba->r = 0.0;
        rgba->g = 0.0;
        rgba->b = 0.0;
        rgba->a = 0.0;
    } else {
        rgba->a = 1.0;
        if (!PyArg_ParseTuple(
                 rgbaobj, "ddd|d:rgba", &(rgba->r), &(rgba->g), &(rgba->b), &(rgba->a))) {
            return 0;
        }
    }

    return 1;
}

int convert_dashes(PyObject *dashobj, void *dashesp)
{
    Dashes *dashes = (Dashes *)dashesp;

    double dash_offset = 0.0;
    PyObject *dashes_seq = NULL;

    if (!PyArg_ParseTuple(dashobj, "dO:dashes", &dash_offset, &dashes_seq)) {
        return 0;
    }

    if (dashes_seq == Py_None) {
        return 1;
    }

    if (!PySequence_Check(dashes_seq)) {
        PyErr_SetString(PyExc_TypeError, "Invalid dashes sequence");
        return 0;
    }

    Py_ssize_t nentries = PySequence_Size(dashes_seq);
    // If the dashpattern has odd length, iterate through it twice (in
    // accordance with the pdf/ps/svg specs).
    Py_ssize_t dash_pattern_length = (nentries % 2) ? 2 * nentries : nentries;

    for (Py_ssize_t i = 0; i < dash_pattern_length; ++i) {
        PyObject *item;
        double length;
        double skip;

        item = PySequence_GetItem(dashes_seq, i % nentries);
        if (item == NULL) {
            return 0;
        }
        length = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            Py_DECREF(item);
            return 0;
        }
        Py_DECREF(item);

        ++i;

        item = PySequence_GetItem(dashes_seq, i % nentries);
        if (item == NULL) {
            return 0;
        }
        skip = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            Py_DECREF(item);
            return 0;
        }
        Py_DECREF(item);

        dashes->add_dash_pair(length, skip);
    }

    dashes->set_dash_offset(dash_offset);

    return 1;
}

int convert_dashes_vector(PyObject *obj, void *dashesp)
{
    DashesVector *dashes = (DashesVector *)dashesp;

    if (!PySequence_Check(obj)) {
        return 0;
    }

    Py_ssize_t n = PySequence_Size(obj);

    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject *item;
        Dashes subdashes;

        item = PySequence_GetItem(obj, i);
        if (item == NULL) {
            return 0;
        }

        if (!convert_dashes(item, &subdashes)) {
            Py_DECREF(item);
            return 0;
        }
        Py_DECREF(item);

        dashes->push_back(subdashes);
    }

    return 1;
}

int convert_trans_affine(PyObject *obj, void *transp)
{
    agg::trans_affine *trans = (agg::trans_affine *)transp;

    /** If None assume identity transform. */
    if (obj == NULL || obj == Py_None) {
        return 1;
    }

    PyArrayObject *array = (PyArrayObject *)PyArray_ContiguousFromAny(obj, NPY_DOUBLE, 2, 2);
    if (array == NULL) {
        return 0;
    }

    if (PyArray_DIM(array, 0) == 3 && PyArray_DIM(array, 1) == 3) {
        double *buffer = (double *)PyArray_DATA(array);
        trans->sx = buffer[0];
        trans->shx = buffer[1];
        trans->tx = buffer[2];

        trans->shy = buffer[3];
        trans->sy = buffer[4];
        trans->ty = buffer[5];

        Py_DECREF(array);
        return 1;
    }

    Py_DECREF(array);
    PyErr_SetString(PyExc_ValueError, "Invalid affine transformation matrix");
    return 0;
}

int convert_path(PyObject *obj, void *pathp)
{
    py::PathIterator *path = (py::PathIterator *)pathp;

    PyObject *vertices_obj = NULL;
    PyObject *codes_obj = NULL;
    PyObject *should_simplify_obj = NULL;
    PyObject *simplify_threshold_obj = NULL;
    bool should_simplify;
    double simplify_threshold;

    int status = 0;

    if (obj == NULL || obj == Py_None) {
        return 1;
    }

    vertices_obj = PyObject_GetAttrString(obj, "vertices");
    if (vertices_obj == NULL) {
        goto exit;
    }

    codes_obj = PyObject_GetAttrString(obj, "codes");
    if (codes_obj == NULL) {
        goto exit;
    }

    should_simplify_obj = PyObject_GetAttrString(obj, "should_simplify");
    if (should_simplify_obj == NULL) {
        goto exit;
    }
    switch (PyObject_IsTrue(should_simplify_obj)) {
        case 0: should_simplify = 0; break;
        case 1: should_simplify = 1; break;
        default: goto exit;  // errored.
    }

    simplify_threshold_obj = PyObject_GetAttrString(obj, "simplify_threshold");
    if (simplify_threshold_obj == NULL) {
        goto exit;
    }
    simplify_threshold = PyFloat_AsDouble(simplify_threshold_obj);
    if (PyErr_Occurred()) {
        goto exit;
    }

    if (!path->set(vertices_obj, codes_obj, should_simplify, simplify_threshold)) {
        goto exit;
    }

    status = 1;

exit:
    Py_XDECREF(vertices_obj);
    Py_XDECREF(codes_obj);
    Py_XDECREF(should_simplify_obj);
    Py_XDECREF(simplify_threshold_obj);

    return status;
}

int convert_pathgen(PyObject *obj, void *pathgenp)
{
    py::PathGenerator *paths = (py::PathGenerator *)pathgenp;
    if (!paths->set(obj)) {
        PyErr_SetString(PyExc_TypeError, "Not an iterable of paths");
        return 0;
    }
    return 1;
}

int convert_clippath(PyObject *clippath_tuple, void *clippathp)
{
    ClipPath *clippath = (ClipPath *)clippathp;
    py::PathIterator path;
    agg::trans_affine trans;

    if (clippath_tuple != NULL && clippath_tuple != Py_None) {
        if (!PyArg_ParseTuple(clippath_tuple,
                              "O&O&:clippath",
                              &convert_path,
                              &clippath->path,
                              &convert_trans_affine,
                              &clippath->trans)) {
            return 0;
        }
    }

    return 1;
}

int convert_snap(PyObject *obj, void *snapp)
{
    e_snap_mode *snap = (e_snap_mode *)snapp;
    if (obj == NULL || obj == Py_None) {
        *snap = SNAP_AUTO;
    } else {
        switch (PyObject_IsTrue(obj)) {
            case 0: *snap = SNAP_FALSE; break;
            case 1: *snap = SNAP_TRUE; break;
            default: return 0;  // errored.
        }
    }
    return 1;
}

int convert_sketch_params(PyObject *obj, void *sketchp)
{
    SketchParams *sketch = (SketchParams *)sketchp;

    if (obj == NULL || obj == Py_None) {
        sketch->scale = 0.0;
    } else if (!PyArg_ParseTuple(obj,
                                 "ddd:sketch_params",
                                 &sketch->scale,
                                 &sketch->length,
                                 &sketch->randomness)) {
        return 0;
    }

    return 1;
}

int convert_gcagg(PyObject *pygc, void *gcp)
{
    GCAgg *gc = (GCAgg *)gcp;

    if (!(convert_from_attr(pygc, "_linewidth", &convert_double, &gc->linewidth) &&
          convert_from_attr(pygc, "_alpha", &convert_double, &gc->alpha) &&
          convert_from_attr(pygc, "_forced_alpha", &convert_bool, &gc->forced_alpha) &&
          convert_from_attr(pygc, "_rgb", &convert_rgba, &gc->color) &&
          convert_from_attr(pygc, "_antialiased", &convert_bool, &gc->isaa) &&
          convert_from_attr(pygc, "_capstyle", &convert_cap, &gc->cap) &&
          convert_from_attr(pygc, "_joinstyle", &convert_join, &gc->join) &&
          convert_from_method(pygc, "get_dashes", &convert_dashes, &gc->dashes) &&
          convert_from_attr(pygc, "_cliprect", &convert_rect, &gc->cliprect) &&
          convert_from_method(pygc, "get_clip_path", &convert_clippath, &gc->clippath) &&
          convert_from_method(pygc, "get_snap", &convert_snap, &gc->snap_mode) &&
          convert_from_method(pygc, "get_hatch_path", &convert_path, &gc->hatchpath) &&
          convert_from_method(pygc, "get_hatch_color", &convert_rgba, &gc->hatch_color) &&
          convert_from_method(pygc, "get_hatch_linewidth", &convert_double, &gc->hatch_linewidth) &&
          convert_from_method(pygc, "get_sketch_params", &convert_sketch_params, &gc->sketch))) {
        return 0;
    }

    return 1;
}

int convert_face(PyObject *color, GCAgg &gc, agg::rgba *rgba)
{
    if (!convert_rgba(color, rgba)) {
        return 0;
    }

    if (color != NULL && color != Py_None) {
        if (gc.forced_alpha || PySequence_Size(color) == 3) {
            rgba->a = gc.alpha;
        }
    }

    return 1;
}

int convert_points(PyObject *obj, void *pointsp)
{
    numpy::array_view<double, 2> *points = (numpy::array_view<double, 2> *)pointsp;

    if (obj == NULL || obj == Py_None) {
        return 1;
    }

    points->set(obj);

    if (points->size() == 0) {
        return 1;
    }

    if (points->dim(1) != 2) {
        PyErr_Format(PyExc_ValueError,
                     "Points must be Nx2 array, got %" NPY_INTP_FMT "x%" NPY_INTP_FMT,
                     points->dim(0), points->dim(1));
        return 0;
    }

    return 1;
}

int convert_transforms(PyObject *obj, void *transp)
{
    numpy::array_view<double, 3> *trans = (numpy::array_view<double, 3> *)transp;

    if (obj == NULL || obj == Py_None) {
        return 1;
    }

    trans->set(obj);

    if (trans->size() == 0) {
        return 1;
    }

    if (trans->dim(1) != 3 || trans->dim(2) != 3) {
        PyErr_Format(PyExc_ValueError,
                     "Transforms must be Nx3x3 array, got %" NPY_INTP_FMT "x%" NPY_INTP_FMT "x%" NPY_INTP_FMT,
                     trans->dim(0), trans->dim(1), trans->dim(2));
        return 0;
    }

    return 1;
}

int convert_bboxes(PyObject *obj, void *bboxp)
{
    numpy::array_view<double, 3> *bbox = (numpy::array_view<double, 3> *)bboxp;

    if (obj == NULL || obj == Py_None) {
        return 1;
    }

    bbox->set(obj);

    if (bbox->size() == 0) {
        return 1;
    }

    if (bbox->dim(1) != 2 || bbox->dim(2) != 2) {
        PyErr_Format(PyExc_ValueError,
                     "Bbox array must be Nx2x2 array, got %" NPY_INTP_FMT "x%" NPY_INTP_FMT "x%" NPY_INTP_FMT,
                     bbox->dim(0), bbox->dim(1), bbox->dim(2));
        return 0;
    }

    return 1;
}

int convert_colors(PyObject *obj, void *colorsp)
{
    numpy::array_view<double, 2> *colors = (numpy::array_view<double, 2> *)colorsp;

    if (obj == NULL || obj == Py_None) {
        return 1;
    }

    colors->set(obj);

    if (colors->size() == 0) {
        return 1;
    }

    if (colors->dim(1) != 4) {
        PyErr_Format(PyExc_ValueError,
                     "Colors array must be Nx4 array, got %" NPY_INTP_FMT "x%" NPY_INTP_FMT,
                     colors->dim(0), colors->dim(1));
        return 0;
    }

    return 1;
}
}
