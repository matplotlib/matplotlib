#include "numpy_cpp.h"

#include "_path.h"

#include "py_converters.h"
#include "py_adaptors.h"

PyObject *convert_polygon_vector(std::vector<Polygon> &polygons)
{
    PyObject *pyresult = PyList_New(polygons.size());
    bool fix_endpoints;

    for (size_t i = 0; i < polygons.size(); ++i) {
        Polygon poly = polygons[i];
        npy_intp dims[2];
        dims[1] = 2;

        if (poly.front() != poly.back()) {
            /* Make last point same as first, if not already */
            dims[0] = (npy_intp)poly.size() + 1;
            fix_endpoints = true;
        } else {
            dims[0] = (npy_intp)poly.size();
            fix_endpoints = false;
        }

        numpy::array_view<double, 2> subresult(dims);
        memcpy(subresult.data(), &poly[0], sizeof(double) * poly.size() * 2);

        if (fix_endpoints) {
            subresult(poly.size(), 0) = poly.front().x;
            subresult(poly.size(), 1) = poly.front().y;
        }

        if (PyList_SetItem(pyresult, i, subresult.pyobj())) {
            Py_DECREF(pyresult);
            return NULL;
        }
    }

    return pyresult;
}

const char *Py_point_in_path__doc__ = "point_in_path(x, y, radius, path, trans)";

static PyObject *Py_point_in_path(PyObject *self, PyObject *args, PyObject *kwds)
{
    double x, y, r;
    py::PathIterator path;
    agg::trans_affine trans;
    bool result;

    if (!PyArg_ParseTuple(args,
                          "dddO&O&:point_in_path",
                          &x,
                          &y,
                          &r,
                          &convert_path,
                          &path,
                          &convert_trans_affine,
                          &trans)) {
        return NULL;
    }

    CALL_CPP("point_in_path", (result = point_in_path(x, y, r, path, trans)));

    if (result) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

const char *Py_points_in_path__doc__ = "points_in_path(points, radius, path, trans)";

static PyObject *Py_points_in_path(PyObject *self, PyObject *args, PyObject *kwds)
{
    numpy::array_view<const double, 2> points;
    double r;
    py::PathIterator path;
    agg::trans_affine trans;

    if (!PyArg_ParseTuple(args,
                          "O&dO&O&:points_in_path",
                          &convert_points,
                          &points,
                          &r,
                          &convert_path,
                          &path,
                          &convert_trans_affine,
                          &trans)) {
        return NULL;
    }

    npy_intp dims[] = { (npy_intp)points.size() };
    numpy::array_view<bool, 1> results(dims);

    CALL_CPP("points_in_path", (points_in_path(points, r, path, trans, results)));

    return results.pyobj();
}

const char *Py_point_on_path__doc__ = "point_on_path(x, y, radius, path, trans)";

static PyObject *Py_point_on_path(PyObject *self, PyObject *args, PyObject *kwds)
{
    double x, y, r;
    py::PathIterator path;
    agg::trans_affine trans;
    bool result;

    if (!PyArg_ParseTuple(args,
                          "dddO&O&:point_on_path",
                          &x,
                          &y,
                          &r,
                          &convert_path,
                          &path,
                          &convert_trans_affine,
                          &trans)) {
        return NULL;
    }

    CALL_CPP("point_on_path", (result = point_on_path(x, y, r, path, trans)));

    if (result) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

const char *Py_points_on_path__doc__ = "points_on_path(points, radius, path, trans)";

static PyObject *Py_points_on_path(PyObject *self, PyObject *args, PyObject *kwds)
{
    numpy::array_view<const double, 2> points;
    double r;
    py::PathIterator path;
    agg::trans_affine trans;

    if (!PyArg_ParseTuple(args,
                          "O&dO&O&:points_on_path",
                          &convert_points,
                          &points,
                          &r,
                          &convert_path,
                          &path,
                          &convert_trans_affine,
                          &trans)) {
        return NULL;
    }

    npy_intp dims[] = { (npy_intp)points.size() };
    numpy::array_view<bool, 1> results(dims);

    CALL_CPP("points_on_path", (points_on_path(points, r, path, trans, results)));

    return results.pyobj();
}

const char *Py_get_path_extents__doc__ = "get_path_extents(path, trans)";

static PyObject *Py_get_path_extents(PyObject *self, PyObject *args, PyObject *kwds)
{
    py::PathIterator path;
    agg::trans_affine trans;

    if (!PyArg_ParseTuple(
             args, "O&O&:get_path_extents", &convert_path, &path, &convert_trans_affine, &trans)) {
        return NULL;
    }

    extent_limits e;

    CALL_CPP("get_path_extents", (reset_limits(e)));
    CALL_CPP("get_path_extents", (update_path_extents(path, trans, e)));

    npy_intp dims[] = { 2, 2 };
    numpy::array_view<double, 2> extents(dims);
    extents(0, 0) = e.x0;
    extents(0, 1) = e.y0;
    extents(1, 0) = e.x1;
    extents(1, 1) = e.y1;

    return extents.pyobj();
}

const char *Py_update_path_extents__doc__ =
    "update_path_extents(path, trans, rect, minpos, ignore)";

static PyObject *Py_update_path_extents(PyObject *self, PyObject *args, PyObject *kwds)
{
    py::PathIterator path;
    agg::trans_affine trans;
    agg::rect_d rect;
    numpy::array_view<double, 1> minpos;
    int ignore;
    int changed;

    if (!PyArg_ParseTuple(args,
                          "O&O&O&O&i:update_path_extents",
                          &convert_path,
                          &path,
                          &convert_trans_affine,
                          &trans,
                          &convert_rect,
                          &rect,
                          &minpos.converter,
                          &minpos,
                          &ignore)) {
        return NULL;
    }

    if (minpos.dim(0) != 2) {
        PyErr_Format(PyExc_ValueError,
                     "minpos must be of length 2, got %d",
                     minpos.dim(0));
        return NULL;
    }

    extent_limits e;

    if (ignore) {
        CALL_CPP("update_path_extents", reset_limits(e));
    } else {
        if (rect.x1 > rect.x2) {
            e.x0 = std::numeric_limits<double>::infinity();
            e.x1 = -std::numeric_limits<double>::infinity();
        } else {
            e.x0 = rect.x1;
            e.x1 = rect.x2;
        }
        if (rect.y1 > rect.y2) {
            e.y0 = std::numeric_limits<double>::infinity();
            e.y1 = -std::numeric_limits<double>::infinity();
        } else {
            e.y0 = rect.y1;
            e.y1 = rect.y2;
        }
        e.xm = minpos(0);
        e.ym = minpos(1);
    }

    CALL_CPP("update_path_extents", (update_path_extents(path, trans, e)));

    changed = (e.x0 != rect.x1 || e.y0 != rect.y1 || e.x1 != rect.x2 || e.y1 != rect.y2 ||
               e.xm != minpos(0) || e.ym != minpos(1));

    npy_intp extentsdims[] = { 2, 2 };
    numpy::array_view<double, 2> outextents(extentsdims);
    outextents(0, 0) = e.x0;
    outextents(0, 1) = e.y0;
    outextents(1, 0) = e.x1;
    outextents(1, 1) = e.y1;

    npy_intp minposdims[] = { 2 };
    numpy::array_view<double, 1> outminpos(minposdims);
    outminpos(0) = e.xm;
    outminpos(1) = e.ym;

    return Py_BuildValue(
        "NNi", outextents.pyobj(), outminpos.pyobj(), changed);
}

const char *Py_get_path_collection_extents__doc__ = "get_path_collection_extents(";

static PyObject *Py_get_path_collection_extents(PyObject *self, PyObject *args, PyObject *kwds)
{
    agg::trans_affine master_transform;
    PyObject *pathsobj;
    numpy::array_view<const double, 3> transforms;
    numpy::array_view<const double, 2> offsets;
    agg::trans_affine offset_trans;
    extent_limits e;

    if (!PyArg_ParseTuple(args,
                          "O&OO&O&O&:get_path_collection_extents",
                          &convert_trans_affine,
                          &master_transform,
                          &pathsobj,
                          &convert_transforms,
                          &transforms,
                          &convert_points,
                          &offsets,
                          &convert_trans_affine,
                          &offset_trans)) {
        return NULL;
    }

    try
    {
        py::PathGenerator paths(pathsobj);

        CALL_CPP("get_path_collection_extents",
                 (get_path_collection_extents(
                     master_transform, paths, transforms, offsets, offset_trans, e)));
    }
    catch (py::exception &e)
    {
        return NULL;
    }

    npy_intp dims[] = { 2, 2 };
    numpy::array_view<double, 2> extents(dims);
    extents(0, 0) = e.x0;
    extents(0, 1) = e.y0;
    extents(1, 0) = e.x1;
    extents(1, 1) = e.y1;

    return extents.pyobj();
}

const char *Py_point_in_path_collection__doc__ =
    "point_in_path_collection(x, y, radius, master_transform, paths, transforms, offsets, "
    "offset_trans, filled, offset_position)";

static PyObject *Py_point_in_path_collection(PyObject *self, PyObject *args, PyObject *kwds)
{
    double x, y, radius;
    agg::trans_affine master_transform;
    PyObject *pathsobj;
    numpy::array_view<const double, 3> transforms;
    numpy::array_view<const double, 2> offsets;
    agg::trans_affine offset_trans;
    int filled;
    e_offset_position offset_position;
    std::vector<int> result;

    if (!PyArg_ParseTuple(args,
                          "dddO&OO&O&O&iO&:point_in_path_collection",
                          &x,
                          &y,
                          &radius,
                          &convert_trans_affine,
                          &master_transform,
                          &pathsobj,
                          &convert_transforms,
                          &transforms,
                          &convert_points,
                          &offsets,
                          &convert_trans_affine,
                          &offset_trans,
                          &filled,
                          &convert_offset_position,
                          &offset_position)) {
        return NULL;
    }

    try
    {
        py::PathGenerator paths(pathsobj);

        CALL_CPP("point_in_path_collection",
                 (point_in_path_collection(x,
                                           y,
                                           radius,
                                           master_transform,
                                           paths,
                                           transforms,
                                           offsets,
                                           offset_trans,
                                           filled,
                                           offset_position,
                                           result)));
    }
    catch (py::exception &e)
    {
        return NULL;
    }

    npy_intp dims[] = {(npy_intp)result.size() };
    numpy::array_view<int, 1> pyresult(dims);
    if (result.size() > 0) {
        memcpy(pyresult.data(), &result[0], result.size() * sizeof(int));
    }
    return pyresult.pyobj();
}

const char *Py_path_in_path__doc__ = "path_in_path(path_a, trans_a, path_b, trans_b)";

static PyObject *Py_path_in_path(PyObject *self, PyObject *args, PyObject *kwds)
{
    py::PathIterator a;
    agg::trans_affine atrans;
    py::PathIterator b;
    agg::trans_affine btrans;
    bool result;

    if (!PyArg_ParseTuple(args,
                          "O&O&O&O&:path_in_path",
                          &convert_path,
                          &a,
                          &convert_trans_affine,
                          &atrans,
                          &convert_path,
                          &b,
                          &convert_trans_affine,
                          &btrans)) {
        return NULL;
    }

    CALL_CPP("path_in_path", (result = path_in_path(a, atrans, b, btrans)));

    if (result) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

const char *Py_clip_path_to_rect__doc__ = "clip_path_to_rect(path, rect, inside)";

static PyObject *Py_clip_path_to_rect(PyObject *self, PyObject *args, PyObject *kwds)
{
    py::PathIterator path;
    agg::rect_d rect;
    int inside;
    std::vector<Polygon> result;

    if (!PyArg_ParseTuple(args,
                          "O&O&i:clip_path_to_rect",
                          &convert_path,
                          &path,
                          &convert_rect,
                          &rect,
                          &inside)) {
        return NULL;
    }

    CALL_CPP("clip_path_to_rect", (clip_path_to_rect(path, rect, inside, result)));

    return convert_polygon_vector(result);
}

const char *Py_affine_transform__doc__ = "affine_transform(points, trans)";

static PyObject *Py_affine_transform(PyObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *vertices_obj;
    agg::trans_affine trans;

    if (!PyArg_ParseTuple(args,
                          "OO&:affine_transform",
                          &vertices_obj,
                          &convert_trans_affine,
                          &trans)) {
        return NULL;
    }

    try {
        numpy::array_view<double, 2> vertices(vertices_obj);
        npy_intp dims[] = { (npy_intp)vertices.size(), 2 };
        numpy::array_view<double, 2> result(dims);
        CALL_CPP("affine_transform", (affine_transform_2d(vertices, trans, result)));
        return result.pyobj();
    } catch (py::exception) {
        PyErr_Clear();
        try {
            numpy::array_view<double, 1> vertices(vertices_obj);
            npy_intp dims[] = { (npy_intp)vertices.size() };
            numpy::array_view<double, 1> result(dims);
            CALL_CPP("affine_transform", (affine_transform_1d(vertices, trans, result)));
            return result.pyobj();
        } catch (py::exception) {
            return NULL;
        }
    }
}

const char *Py_count_bboxes_overlapping_bbox__doc__ = "count_bboxes_overlapping_bbox(bbox, bboxes)";

static PyObject *Py_count_bboxes_overlapping_bbox(PyObject *self, PyObject *args, PyObject *kwds)
{
    agg::rect_d bbox;
    numpy::array_view<const double, 3> bboxes;
    int result;

    if (!PyArg_ParseTuple(args,
                          "O&O&:count_bboxes_overlapping_bbox",
                          &convert_rect,
                          &bbox,
                          &convert_bboxes,
                          &bboxes)) {
        return NULL;
    }

    CALL_CPP("count_bboxes_overlapping_bbox",
             (result = count_bboxes_overlapping_bbox(bbox, bboxes)));

    return PyLong_FromLong(result);
}

const char *Py_path_intersects_path__doc__ = "path_intersects_path(path1, path2, filled=False)";

static PyObject *Py_path_intersects_path(PyObject *self, PyObject *args, PyObject *kwds)
{
    py::PathIterator p1;
    py::PathIterator p2;
    agg::trans_affine t1;
    agg::trans_affine t2;
    int filled = 0;
    const char *names[] = { "p1", "p2", "filled", NULL };
    bool result;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwds,
                                     "O&O&i:path_intersects_path",
                                     (char **)names,
                                     &convert_path,
                                     &p1,
                                     &convert_path,
                                     &p2,
                                     &filled)) {
        return NULL;
    }

    CALL_CPP("path_intersects_path", (result = path_intersects_path(p1, p2)));
    if (filled) {
        if (!result) {
            CALL_CPP("path_intersects_path",
                     (result = path_in_path(p1, t1, p2, t2)));
        }
        if (!result) {
            CALL_CPP("path_intersects_path",
                     (result = path_in_path(p2, t1, p1, t2)));
        }
    }

    if (result) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

const char *Py_convert_path_to_polygons__doc__ =
    "convert_path_to_polygons(path, trans, width=0, height=0)";

static PyObject *Py_convert_path_to_polygons(PyObject *self, PyObject *args, PyObject *kwds)
{
    py::PathIterator path;
    agg::trans_affine trans;
    double width = 0.0, height = 0.0;
    std::vector<Polygon> result;

    if (!PyArg_ParseTuple(args,
                          "O&O&|dd:convert_path_to_polygons",
                          &convert_path,
                          &path,
                          &convert_trans_affine,
                          &trans,
                          &width,
                          &height)) {
        return NULL;
    }

    CALL_CPP("convert_path_to_polygons",
             (convert_path_to_polygons(path, trans, width, height, result)));

    return convert_polygon_vector(result);
}

const char *Py_cleanup_path__doc__ =
    "cleanup_path(path, trans, remove_nans, clip_rect, snap_mode, stroke_width, simplify, "
    "return_curves, sketch)";

static PyObject *Py_cleanup_path(PyObject *self, PyObject *args, PyObject *kwds)
{
    py::PathIterator path;
    agg::trans_affine trans;
    int remove_nans;
    agg::rect_d clip_rect;
    e_snap_mode snap_mode;
    double stroke_width;
    PyObject *simplifyobj;
    bool simplify = false;
    int return_curves;
    SketchParams sketch;

    if (!PyArg_ParseTuple(args,
                          "O&O&iO&O&dOiO&:cleanup_path",
                          &convert_path,
                          &path,
                          &convert_trans_affine,
                          &trans,
                          &remove_nans,
                          &convert_rect,
                          &clip_rect,
                          &convert_snap,
                          &snap_mode,
                          &stroke_width,
                          &simplifyobj,
                          &return_curves,
                          &convert_sketch_params,
                          &sketch)) {
        return NULL;
    }

    if (simplifyobj == Py_None) {
        simplify = path.should_simplify();
    } else if (PyObject_IsTrue(simplifyobj)) {
        simplify = true;
    }

    bool do_clip = (clip_rect.x1 < clip_rect.x2 && clip_rect.y1 < clip_rect.y2);

    std::vector<double> vertices;
    std::vector<npy_uint8> codes;

    CALL_CPP("cleanup_path",
             (cleanup_path(path,
                           trans,
                           remove_nans,
                           do_clip,
                           clip_rect,
                           snap_mode,
                           stroke_width,
                           simplify,
                           return_curves,
                           sketch,
                           vertices,
                           codes)));

    size_t length = codes.size();

    npy_intp vertices_dims[] = {(npy_intp)length, 2 };
    numpy::array_view<double, 2> pyvertices(vertices_dims);

    npy_intp codes_dims[] = {(npy_intp)length };
    numpy::array_view<unsigned char, 1> pycodes(codes_dims);

    memcpy(pyvertices.data(), &vertices[0], sizeof(double) * 2 * length);
    memcpy(pycodes.data(), &codes[0], sizeof(unsigned char) * length);

    return Py_BuildValue("NN", pyvertices.pyobj(), pycodes.pyobj());
}

const char *Py_convert_to_string__doc__ = "convert_to_string(path, trans, "
    "clip_rect, simplify, sketch, precision, codes, postfix)";

static PyObject *Py_convert_to_string(PyObject *self, PyObject *args, PyObject *kwds)
{
    py::PathIterator path;
    agg::trans_affine trans;
    agg::rect_d cliprect;
    PyObject *simplifyobj;
    bool simplify = false;
    SketchParams sketch;
    int precision;
    PyObject *codesobj;
    char *codes[5];
    int postfix;
    char *buffer = NULL;
    size_t buffersize;
    PyObject *result;
    int status;

    if (!PyArg_ParseTuple(args,
                          "O&O&O&OO&iOi:convert_to_string",
                          &convert_path,
                          &path,
                          &convert_trans_affine,
                          &trans,
                          &convert_rect,
                          &cliprect,
                          &simplifyobj,
                          &convert_sketch_params,
                          &sketch,
                          &precision,
                          &codesobj,
                          &postfix)) {
        return NULL;
    }

    if (simplifyobj == Py_None) {
        simplify = path.should_simplify();
    } else if (PyObject_IsTrue(simplifyobj)) {
        simplify = true;
    }

    if (!PySequence_Check(codesobj)) {
        return NULL;
    }
    if (PySequence_Size(codesobj) != 5) {
        PyErr_SetString(
            PyExc_ValueError,
            "codes must be a 5-length sequence of byte strings");
        return NULL;
    }
    for (int i = 0; i < 5; ++i) {
        PyObject *item = PySequence_GetItem(codesobj, i);
        if (item == NULL) {
            return NULL;
        }
        codes[i] = PyBytes_AsString(item);
        if (codes[i] == NULL) {
            return NULL;
        }
    }

    CALL_CPP("convert_to_string",
             (status = convert_to_string(
                 path, trans, cliprect, simplify, sketch,
                 precision, codes, (bool)postfix, &buffer,
                 &buffersize)));

    if (status) {
        free(buffer);
        if (status == 1) {
            PyErr_SetString(PyExc_MemoryError, "Memory error");
        } else if (status == 2) {
            PyErr_SetString(PyExc_ValueError, "Malformed path codes");
        }
        return NULL;
    }

    if (buffersize == 0) {
        result = PyBytes_FromString("");
    } else {
        result = PyBytes_FromStringAndSize(buffer, buffersize);
    }

    free(buffer);

    return result;
}


const char *Py_is_sorted__doc__ = "is_sorted(array)\n\n"
    "Returns True if 1-D array is monotonically increasing, ignoring NaNs\n";

static PyObject *Py_is_sorted(PyObject *self, PyObject *obj)
{
    npy_intp size;
    bool result;

    PyArrayObject *array = (PyArrayObject *)PyArray_FromAny(
        obj, NULL, 1, 1, 0, NULL);

    if (array == NULL) {
        return NULL;
    }

    size = PyArray_DIM(array, 0);

    if (size < 2) {
        Py_DECREF(array);
        Py_RETURN_TRUE;
    }

    /* Handle just the most common types here, otherwise coerce to
    double */
    switch(PyArray_TYPE(array)) {
    case NPY_INT:
        {
            _is_sorted_int<npy_int> is_sorted;
            result = is_sorted(array);
        }
        break;

    case NPY_LONG:
        {
            _is_sorted_int<npy_long> is_sorted;
            result = is_sorted(array);
        }
        break;

    case NPY_LONGLONG:
        {
            _is_sorted_int<npy_longlong> is_sorted;
            result = is_sorted(array);
        }
        break;

    case NPY_FLOAT:
        {
            _is_sorted<npy_float> is_sorted;
            result = is_sorted(array);
        }
        break;

    case NPY_DOUBLE:
        {
            _is_sorted<npy_double> is_sorted;
            result = is_sorted(array);
        }
        break;

    default:
        {
            Py_DECREF(array);
            array = (PyArrayObject *)PyArray_FromObject(obj, NPY_DOUBLE, 1, 1);

            if (array == NULL) {
                return NULL;
            }

            _is_sorted<npy_double> is_sorted;
            result = is_sorted(array);
        }
    }

    Py_DECREF(array);

    if (result) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}


extern "C" {

    static PyMethodDef module_functions[] = {
        {"point_in_path", (PyCFunction)Py_point_in_path, METH_VARARGS, Py_point_in_path__doc__},
        {"points_in_path", (PyCFunction)Py_points_in_path, METH_VARARGS, Py_points_in_path__doc__},
        {"point_on_path", (PyCFunction)Py_point_on_path, METH_VARARGS, Py_point_on_path__doc__},
        {"points_on_path", (PyCFunction)Py_points_on_path, METH_VARARGS, Py_points_on_path__doc__},
        {"get_path_extents", (PyCFunction)Py_get_path_extents, METH_VARARGS, Py_get_path_extents__doc__},
        {"update_path_extents", (PyCFunction)Py_update_path_extents, METH_VARARGS, Py_update_path_extents__doc__},
        {"get_path_collection_extents", (PyCFunction)Py_get_path_collection_extents, METH_VARARGS, Py_get_path_collection_extents__doc__},
        {"point_in_path_collection", (PyCFunction)Py_point_in_path_collection, METH_VARARGS, Py_point_in_path_collection__doc__},
        {"path_in_path", (PyCFunction)Py_path_in_path, METH_VARARGS, Py_path_in_path__doc__},
        {"clip_path_to_rect", (PyCFunction)Py_clip_path_to_rect, METH_VARARGS, Py_clip_path_to_rect__doc__},
        {"affine_transform", (PyCFunction)Py_affine_transform, METH_VARARGS, Py_affine_transform__doc__},
        {"count_bboxes_overlapping_bbox", (PyCFunction)Py_count_bboxes_overlapping_bbox, METH_VARARGS, Py_count_bboxes_overlapping_bbox__doc__},
        {"path_intersects_path", (PyCFunction)Py_path_intersects_path, METH_VARARGS|METH_KEYWORDS, Py_path_intersects_path__doc__},
        {"convert_path_to_polygons", (PyCFunction)Py_convert_path_to_polygons, METH_VARARGS, Py_convert_path_to_polygons__doc__},
        {"cleanup_path", (PyCFunction)Py_cleanup_path, METH_VARARGS, Py_cleanup_path__doc__},
        {"convert_to_string", (PyCFunction)Py_convert_to_string, METH_VARARGS, Py_convert_to_string__doc__},
        {"is_sorted", (PyCFunction)Py_is_sorted, METH_O, Py_is_sorted__doc__},
        {NULL}
    };

#if PY3K
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_path",
        NULL,
        0,
        module_functions,
        NULL,
        NULL,
        NULL,
        NULL
    };

#define INITERROR return NULL
    PyMODINIT_FUNC PyInit__path(void)
#else
#define INITERROR return
    PyMODINIT_FUNC init_path(void)
#endif
    {
        PyObject *m;
#if PY3K
        m = PyModule_Create(&moduledef);
#else
        m = Py_InitModule3("_path", module_functions, NULL);
#endif

        if (m == NULL) {
            INITERROR;
        }

        import_array();

#if PY3K
        return m;
#endif
    }
}
