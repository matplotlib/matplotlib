#include "numpy_cpp.h"

#include "_path.h"

#include "py_converters.h"
#include "py_adaptors.h"

PyObject *convert_polygon_vector(std::vector<Polygon> &polygons)
{
    PyObject *pyresult = PyList_New(polygons.size());

    for (size_t i = 0; i < polygons.size(); ++i) {
        Polygon poly = polygons[i];
        npy_intp dims[2];
        dims[1] = 2;

        dims[0] = (npy_intp)poly.size();

        numpy::array_view<double, 2> subresult(dims);
        memcpy(subresult.data(), &poly[0], sizeof(double) * poly.size() * 2);

        if (PyList_SetItem(pyresult, i, subresult.pyobj())) {
            Py_DECREF(pyresult);
            return NULL;
        }
    }

    return pyresult;
}

const char *Py_point_in_path__doc__ =
    "point_in_path(x, y, radius, path, trans)\n"
    "--\n\n";

static PyObject *Py_point_in_path(PyObject *self, PyObject *args)
{
    double x, y, r;
    mpl::PathIterator path;
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

const char *Py_points_in_path__doc__ =
    "points_in_path(points, radius, path, trans)\n"
    "--\n\n";

static PyObject *Py_points_in_path(PyObject *self, PyObject *args)
{
    numpy::array_view<const double, 2> points;
    double r;
    mpl::PathIterator path;
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

    if (!check_trailing_shape(points, "points", 2)) {
        return NULL;
    }

    npy_intp dims[] = { (npy_intp)points.shape(0) };
    numpy::array_view<uint8_t, 1> results(dims);

    CALL_CPP("points_in_path", (points_in_path(points, r, path, trans, results)));

    return results.pyobj();
}

const char *Py_update_path_extents__doc__ =
    "update_path_extents(path, trans, rect, minpos, ignore)\n"
    "--\n\n";

static PyObject *Py_update_path_extents(PyObject *self, PyObject *args)
{
    mpl::PathIterator path;
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

    if (minpos.shape(0) != 2) {
        PyErr_Format(PyExc_ValueError,
                     "minpos must be of length 2, got %" NPY_INTP_FMT,
                     minpos.shape(0));
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

const char *Py_get_path_collection_extents__doc__ =
    "get_path_collection_extents("
    "master_transform, paths, transforms, offsets, offset_transform)\n"
    "--\n\n";

static PyObject *Py_get_path_collection_extents(PyObject *self, PyObject *args)
{
    agg::trans_affine master_transform;
    mpl::PathGenerator paths;
    numpy::array_view<const double, 3> transforms;
    numpy::array_view<const double, 2> offsets;
    agg::trans_affine offset_trans;
    extent_limits e;

    if (!PyArg_ParseTuple(args,
                          "O&O&O&O&O&:get_path_collection_extents",
                          &convert_trans_affine,
                          &master_transform,
                          &convert_pathgen,
                          &paths,
                          &convert_transforms,
                          &transforms,
                          &convert_points,
                          &offsets,
                          &convert_trans_affine,
                          &offset_trans)) {
        return NULL;
    }

    CALL_CPP("get_path_collection_extents",
             (get_path_collection_extents(
                 master_transform, paths, transforms, offsets, offset_trans, e)));

    npy_intp dims[] = { 2, 2 };
    numpy::array_view<double, 2> extents(dims);
    extents(0, 0) = e.x0;
    extents(0, 1) = e.y0;
    extents(1, 0) = e.x1;
    extents(1, 1) = e.y1;

    npy_intp minposdims[] = { 2 };
    numpy::array_view<double, 1> minpos(minposdims);
    minpos(0) = e.xm;
    minpos(1) = e.ym;

    return Py_BuildValue("NN", extents.pyobj(), minpos.pyobj());
}

const char *Py_point_in_path_collection__doc__ =
    "point_in_path_collection("
    "x, y, radius, master_transform, paths, transforms, offsets, "
    "offset_trans, filled)\n"
    "--\n\n";

static PyObject *Py_point_in_path_collection(PyObject *self, PyObject *args)
{
    double x, y, radius;
    agg::trans_affine master_transform;
    mpl::PathGenerator paths;
    numpy::array_view<const double, 3> transforms;
    numpy::array_view<const double, 2> offsets;
    agg::trans_affine offset_trans;
    bool filled;
    std::vector<int> result;

    if (!PyArg_ParseTuple(args,
                          "dddO&O&O&O&O&O&:point_in_path_collection",
                          &x,
                          &y,
                          &radius,
                          &convert_trans_affine,
                          &master_transform,
                          &convert_pathgen,
                          &paths,
                          &convert_transforms,
                          &transforms,
                          &convert_points,
                          &offsets,
                          &convert_trans_affine,
                          &offset_trans,
                          &convert_bool,
                          &filled)) {
        return NULL;
    }

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
                                       result)));

    npy_intp dims[] = {(npy_intp)result.size() };
    numpy::array_view<int, 1> pyresult(dims);
    if (result.size() > 0) {
        memcpy(pyresult.data(), &result[0], result.size() * sizeof(int));
    }
    return pyresult.pyobj();
}

const char *Py_path_in_path__doc__ =
    "path_in_path(path_a, trans_a, path_b, trans_b)\n"
    "--\n\n";

static PyObject *Py_path_in_path(PyObject *self, PyObject *args)
{
    mpl::PathIterator a;
    agg::trans_affine atrans;
    mpl::PathIterator b;
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

const char *Py_clip_path_to_rect__doc__ =
    "clip_path_to_rect(path, rect, inside)\n"
    "--\n\n";

static PyObject *Py_clip_path_to_rect(PyObject *self, PyObject *args)
{
    mpl::PathIterator path;
    agg::rect_d rect;
    bool inside;
    std::vector<Polygon> result;

    if (!PyArg_ParseTuple(args,
                          "O&O&O&:clip_path_to_rect",
                          &convert_path,
                          &path,
                          &convert_rect,
                          &rect,
                          &convert_bool,
                          &inside)) {
        return NULL;
    }

    CALL_CPP("clip_path_to_rect", (clip_path_to_rect(path, rect, inside, result)));

    return convert_polygon_vector(result);
}

const char *Py_affine_transform__doc__ =
    "affine_transform(points, trans)\n"
    "--\n\n";

static PyObject *Py_affine_transform(PyObject *self, PyObject *args)
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

    PyArrayObject* vertices_arr = (PyArrayObject *)PyArray_ContiguousFromAny(vertices_obj, NPY_DOUBLE, 1, 2);
    if (vertices_arr == NULL) {
        return NULL;
    }

    if (PyArray_NDIM(vertices_arr) == 2) {
        numpy::array_view<double, 2> vertices(vertices_arr);
        Py_DECREF(vertices_arr);

        if(!check_trailing_shape(vertices, "vertices", 2)) {
            return NULL;
        }

        npy_intp dims[] = { (npy_intp)vertices.shape(0), 2 };
        numpy::array_view<double, 2> result(dims);
        CALL_CPP("affine_transform", (affine_transform_2d(vertices, trans, result)));
        return result.pyobj();
    } else { // PyArray_NDIM(vertices_arr) == 1
        numpy::array_view<double, 1> vertices(vertices_arr);
        Py_DECREF(vertices_arr);

        npy_intp dims[] = { (npy_intp)vertices.shape(0) };
        numpy::array_view<double, 1> result(dims);
        CALL_CPP("affine_transform", (affine_transform_1d(vertices, trans, result)));
        return result.pyobj();
    }
}

const char *Py_count_bboxes_overlapping_bbox__doc__ =
    "count_bboxes_overlapping_bbox(bbox, bboxes)\n"
    "--\n\n";

static PyObject *Py_count_bboxes_overlapping_bbox(PyObject *self, PyObject *args)
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

const char *Py_path_intersects_path__doc__ =
    "path_intersects_path(path1, path2, filled=False)\n"
    "--\n\n";

static PyObject *Py_path_intersects_path(PyObject *self, PyObject *args, PyObject *kwds)
{
    mpl::PathIterator p1;
    mpl::PathIterator p2;
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

const char *Py_path_intersects_rectangle__doc__ =
    "path_intersects_rectangle("
    "path, rect_x1, rect_y1, rect_x2, rect_y2, filled=False)\n"
    "--\n\n";

static PyObject *Py_path_intersects_rectangle(PyObject *self, PyObject *args, PyObject *kwds)
{
    mpl::PathIterator path;
    double rect_x1, rect_y1, rect_x2, rect_y2;
    bool filled = false;
    const char *names[] = { "path", "rect_x1", "rect_y1", "rect_x2", "rect_y2", "filled", NULL };
    bool result;

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwds,
                                     "O&dddd|O&:path_intersects_rectangle",
                                     (char **)names,
                                     &convert_path,
                                     &path,
                                     &rect_x1,
                                     &rect_y1,
                                     &rect_x2,
                                     &rect_y2,
                                     &convert_bool,
                                     &filled)) {
        return NULL;
    }

    CALL_CPP("path_intersects_rectangle", (result = path_intersects_rectangle(path, rect_x1, rect_y1, rect_x2, rect_y2, filled)));

    if (result) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

const char *Py_convert_path_to_polygons__doc__ =
    "convert_path_to_polygons(path, trans, width=0, height=0)\n"
    "--\n\n";

static PyObject *Py_convert_path_to_polygons(PyObject *self, PyObject *args, PyObject *kwds)
{
    mpl::PathIterator path;
    agg::trans_affine trans;
    double width = 0.0, height = 0.0;
    int closed_only = 1;
    std::vector<Polygon> result;
    const char *names[] = { "path", "transform", "width", "height", "closed_only", NULL };

    if (!PyArg_ParseTupleAndKeywords(args,
                                     kwds,
                                     "O&O&|ddi:convert_path_to_polygons",
                                     (char **)names,
                                     &convert_path,
                                     &path,
                                     &convert_trans_affine,
                                     &trans,
                                     &width,
                                     &height,
                                     &closed_only)) {
        return NULL;
    }

    CALL_CPP("convert_path_to_polygons",
             (convert_path_to_polygons(path, trans, width, height, closed_only, result)));

    return convert_polygon_vector(result);
}

const char *Py_cleanup_path__doc__ =
    "cleanup_path("
    "path, trans, remove_nans, clip_rect, snap_mode, stroke_width, simplify, "
    "return_curves, sketch)\n"
    "--\n\n";

static PyObject *Py_cleanup_path(PyObject *self, PyObject *args)
{
    mpl::PathIterator path;
    agg::trans_affine trans;
    bool remove_nans;
    agg::rect_d clip_rect;
    e_snap_mode snap_mode;
    double stroke_width;
    PyObject *simplifyobj;
    bool simplify = false;
    bool return_curves;
    SketchParams sketch;

    if (!PyArg_ParseTuple(args,
                          "O&O&O&O&O&dOO&O&:cleanup_path",
                          &convert_path,
                          &path,
                          &convert_trans_affine,
                          &trans,
                          &convert_bool,
                          &remove_nans,
                          &convert_rect,
                          &clip_rect,
                          &convert_snap,
                          &snap_mode,
                          &stroke_width,
                          &simplifyobj,
                          &convert_bool,
                          &return_curves,
                          &convert_sketch_params,
                          &sketch)) {
        return NULL;
    }

    if (simplifyobj == Py_None) {
        simplify = path.should_simplify();
    } else {
        switch (PyObject_IsTrue(simplifyobj)) {
            case 0: simplify = false; break;
            case 1: simplify = true; break;
            default: return NULL;  // errored.
        }
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

const char *Py_convert_to_string__doc__ =
    "convert_to_string("
    "path, trans, clip_rect, simplify, sketch, precision, codes, postfix)\n"
    "--\n\n"
    "Convert *path* to a bytestring.\n"
    "\n"
    "The first five parameters (up to *sketch*) are interpreted as in\n"
    "`.cleanup_path`.  The following ones are detailed below.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "path : Path\n"
    "trans : Transform or None\n"
    "clip_rect : sequence of 4 floats, or None\n"
    "simplify : bool\n"
    "sketch : tuple of 3 floats, or None\n"
    "precision : int\n"
    "    The precision used to \"%.*f\"-format the values.  Trailing zeros\n"
    "    and decimal points are always removed.  (precision=-1 is a special\n"
    "    case used to implement ttconv-back-compatible conversion.)\n"
    "codes : sequence of 5 bytestrings\n"
    "    The bytes representation of each opcode (MOVETO, LINETO, CURVE3,\n"
    "    CURVE4, CLOSEPOLY), in that order.  If the bytes for CURVE3 is\n"
    "    empty, quad segments are automatically converted to cubic ones\n"
    "    (this is used by backends such as pdf and ps, which do not support\n"
    "    quads).\n"
    "postfix : bool\n"
    "    Whether the opcode comes after the values (True) or before (False).\n"
    ;

static PyObject *Py_convert_to_string(PyObject *self, PyObject *args)
{
    mpl::PathIterator path;
    agg::trans_affine trans;
    agg::rect_d cliprect;
    PyObject *simplifyobj;
    bool simplify = false;
    SketchParams sketch;
    int precision;
    char *codes[5];
    bool postfix;
    std::string buffer;
    bool status;

    if (!PyArg_ParseTuple(args,
                          "O&O&O&OO&i(yyyyy)O&:convert_to_string",
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
                          &codes[0],
                          &codes[1],
                          &codes[2],
                          &codes[3],
                          &codes[4],
                          &convert_bool,
                          &postfix)) {
        return NULL;
    }

    if (simplifyobj == Py_None) {
        simplify = path.should_simplify();
    } else {
        switch (PyObject_IsTrue(simplifyobj)) {
            case 0: simplify = false; break;
            case 1: simplify = true; break;
            default: return NULL;  // errored.
        }
    }

    CALL_CPP("convert_to_string",
             (status = convert_to_string(
                 path, trans, cliprect, simplify, sketch,
                 precision, codes, postfix, buffer)));

    if (!status) {
        PyErr_SetString(PyExc_ValueError, "Malformed path codes");
        return NULL;
    }

    return PyBytes_FromStringAndSize(buffer.c_str(), buffer.size());
}


const char *Py_is_sorted_and_has_non_nan__doc__ =
    "is_sorted_and_has_non_nan(array, /)\n"
    "--\n\n"
    "Return whether the 1D *array* is monotonically increasing, ignoring NaNs,\n"
    "and has at least one non-nan value.";

static PyObject *Py_is_sorted_and_has_non_nan(PyObject *self, PyObject *obj)
{
    bool result;

    PyArrayObject *array = (PyArrayObject *)PyArray_FromAny(
        obj, NULL, 1, 1, 0, NULL);

    if (array == NULL) {
        return NULL;
    }

    /* Handle just the most common types here, otherwise coerce to double */
    switch (PyArray_TYPE(array)) {
    case NPY_INT:
        result = is_sorted_and_has_non_nan<npy_int>(array);
        break;
    case NPY_LONG:
        result = is_sorted_and_has_non_nan<npy_long>(array);
        break;
    case NPY_LONGLONG:
        result = is_sorted_and_has_non_nan<npy_longlong>(array);
        break;
    case NPY_FLOAT:
        result = is_sorted_and_has_non_nan<npy_float>(array);
        break;
    case NPY_DOUBLE:
        result = is_sorted_and_has_non_nan<npy_double>(array);
        break;
    default:
        Py_DECREF(array);
        array = (PyArrayObject *)PyArray_FromObject(obj, NPY_DOUBLE, 1, 1);
        if (array == NULL) {
            return NULL;
        }
        result = is_sorted_and_has_non_nan<npy_double>(array);
    }

    Py_DECREF(array);

    if (result) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}


static PyMethodDef module_functions[] = {
    {"point_in_path", (PyCFunction)Py_point_in_path, METH_VARARGS, Py_point_in_path__doc__},
    {"points_in_path", (PyCFunction)Py_points_in_path, METH_VARARGS, Py_points_in_path__doc__},
    {"update_path_extents", (PyCFunction)Py_update_path_extents, METH_VARARGS, Py_update_path_extents__doc__},
    {"get_path_collection_extents", (PyCFunction)Py_get_path_collection_extents, METH_VARARGS, Py_get_path_collection_extents__doc__},
    {"point_in_path_collection", (PyCFunction)Py_point_in_path_collection, METH_VARARGS, Py_point_in_path_collection__doc__},
    {"path_in_path", (PyCFunction)Py_path_in_path, METH_VARARGS, Py_path_in_path__doc__},
    {"clip_path_to_rect", (PyCFunction)Py_clip_path_to_rect, METH_VARARGS, Py_clip_path_to_rect__doc__},
    {"affine_transform", (PyCFunction)Py_affine_transform, METH_VARARGS, Py_affine_transform__doc__},
    {"count_bboxes_overlapping_bbox", (PyCFunction)Py_count_bboxes_overlapping_bbox, METH_VARARGS, Py_count_bboxes_overlapping_bbox__doc__},
    {"path_intersects_path", (PyCFunction)Py_path_intersects_path, METH_VARARGS|METH_KEYWORDS, Py_path_intersects_path__doc__},
    {"path_intersects_rectangle", (PyCFunction)Py_path_intersects_rectangle, METH_VARARGS|METH_KEYWORDS, Py_path_intersects_rectangle__doc__},
    {"convert_path_to_polygons", (PyCFunction)Py_convert_path_to_polygons, METH_VARARGS|METH_KEYWORDS, Py_convert_path_to_polygons__doc__},
    {"cleanup_path", (PyCFunction)Py_cleanup_path, METH_VARARGS, Py_cleanup_path__doc__},
    {"convert_to_string", (PyCFunction)Py_convert_to_string, METH_VARARGS, Py_convert_to_string__doc__},
    {"is_sorted_and_has_non_nan", (PyCFunction)Py_is_sorted_and_has_non_nan, METH_O, Py_is_sorted_and_has_non_nan__doc__},
    {NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_path", NULL, 0, module_functions
};

PyMODINIT_FUNC PyInit__path(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}
