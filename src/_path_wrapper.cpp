#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "_path.h"

#include "_backend_agg_basic_types.h"
#include "py_adaptors.h"
#include "py_converters.h"

namespace py = pybind11;
using namespace pybind11::literals;

py::list
convert_polygon_vector(std::vector<Polygon> &polygons)
{
    auto result = py::list(polygons.size());

    for (size_t i = 0; i < polygons.size(); ++i) {
        const auto& poly = polygons[i];
        py::ssize_t dims[] = { static_cast<py::ssize_t>(poly.size()), 2 };
        result[i] = py::array(dims, reinterpret_cast<const double *>(poly.data()));
    }

    return result;
}

static bool
Py_point_in_path(double x, double y, double r, mpl::PathIterator path,
                 agg::trans_affine trans)
{
    return point_in_path(x, y, r, path, trans);
}

static py::array_t<double>
Py_points_in_path(py::array_t<double> points_obj, double r, mpl::PathIterator path,
                  agg::trans_affine trans)
{
    auto points = convert_points(points_obj);

    py::ssize_t dims[] = { points.shape(0) };
    py::array_t<uint8_t> results(dims);
    auto results_mutable = results.mutable_unchecked<1>();

    points_in_path(points, r, path, trans, results_mutable);

    return results;
}

static py::tuple
Py_get_path_collection_extents(agg::trans_affine master_transform,
                               mpl::PathGenerator paths,
                               py::array_t<double> transforms_obj,
                               py::array_t<double> offsets_obj,
                               agg::trans_affine offset_trans)
{
    auto transforms = convert_transforms(transforms_obj);
    auto offsets = convert_points(offsets_obj);
    extent_limits e;

    get_path_collection_extents(
        master_transform, paths, transforms, offsets, offset_trans, e);

    py::ssize_t dims[] = { 2, 2 };
    py::array_t<double> extents(dims);
    *extents.mutable_data(0, 0) = e.x0;
    *extents.mutable_data(0, 1) = e.y0;
    *extents.mutable_data(1, 0) = e.x1;
    *extents.mutable_data(1, 1) = e.y1;

    py::ssize_t minposdims[] = { 2 };
    py::array_t<double> minpos(minposdims);
    *minpos.mutable_data(0) = e.xm;
    *minpos.mutable_data(1) = e.ym;

    return py::make_tuple(extents, minpos);
}

static py::object
Py_point_in_path_collection(double x, double y, double radius,
                            agg::trans_affine master_transform, mpl::PathGenerator paths,
                            py::array_t<double> transforms_obj,
                            py::array_t<double> offsets_obj,
                            agg::trans_affine offset_trans, bool filled)
{
    auto transforms = convert_transforms(transforms_obj);
    auto offsets = convert_points(offsets_obj);
    std::vector<int> result;

    point_in_path_collection(x, y, radius, master_transform, paths, transforms, offsets,
                             offset_trans, filled, result);

    py::ssize_t dims[] = { static_cast<py::ssize_t>(result.size()) };
    return py::array(dims, result.data());
}

static bool
Py_path_in_path(mpl::PathIterator a, agg::trans_affine atrans,
                mpl::PathIterator b, agg::trans_affine btrans)
{
    return path_in_path(a, atrans, b, btrans);
}

static py::list
Py_clip_path_to_rect(mpl::PathIterator path, agg::rect_d rect, bool inside)
{
    std::vector<Polygon> result;

    clip_path_to_rect(path, rect, inside, result);

    return convert_polygon_vector(result);
}

static py::object
Py_affine_transform(py::array_t<double, py::array::c_style | py::array::forcecast> vertices_arr,
                    agg::trans_affine trans)
{
    if (vertices_arr.ndim() == 2) {
        auto vertices = vertices_arr.unchecked<2>();

        check_trailing_shape(vertices, "vertices", 2);

        py::ssize_t dims[] = { vertices.shape(0), 2 };
        py::array_t<double> result(dims);
        auto result_mutable = result.mutable_unchecked<2>();

        affine_transform_2d(vertices, trans, result_mutable);
        return result;
    } else if (vertices_arr.ndim() == 1) {
        auto vertices = vertices_arr.unchecked<1>();

        py::ssize_t dims[] = { vertices.shape(0) };
        py::array_t<double> result(dims);
        auto result_mutable = result.mutable_unchecked<1>();

        affine_transform_1d(vertices, trans, result_mutable);
        return result;
    } else {
        throw py::value_error(
            "vertices must be 1D or 2D, not" + std::to_string(vertices_arr.ndim()) + "D");
    }
}

static int
Py_count_bboxes_overlapping_bbox(agg::rect_d bbox, py::array_t<double> bboxes_obj)
{
    auto bboxes = convert_bboxes(bboxes_obj);

    return count_bboxes_overlapping_bbox(bbox, bboxes);
}

static bool
Py_path_intersects_path(mpl::PathIterator p1, mpl::PathIterator p2, bool filled)
{
    agg::trans_affine t1;
    agg::trans_affine t2;
    bool result;

    result = path_intersects_path(p1, p2);
    if (filled) {
        if (!result) {
            result = path_in_path(p1, t1, p2, t2);
        }
        if (!result) {
            result = path_in_path(p2, t1, p1, t2);
        }
    }

    return result;
}

static bool
Py_path_intersects_rectangle(mpl::PathIterator path, double rect_x1, double rect_y1,
                             double rect_x2, double rect_y2, bool filled)
{
    return path_intersects_rectangle(path, rect_x1, rect_y1, rect_x2, rect_y2, filled);
}

static py::list
Py_convert_path_to_polygons(mpl::PathIterator path, agg::trans_affine trans,
                            double width, double height, bool closed_only)
{
    std::vector<Polygon> result;

    convert_path_to_polygons(path, trans, width, height, closed_only, result);

    return convert_polygon_vector(result);
}

static py::tuple
Py_cleanup_path(mpl::PathIterator path, agg::trans_affine trans, bool remove_nans,
                agg::rect_d clip_rect, e_snap_mode snap_mode, double stroke_width,
                std::optional<bool> simplify, bool return_curves, SketchParams sketch)
{
    if (!simplify.has_value()) {
        simplify = path.should_simplify();
    }

    bool do_clip = (clip_rect.x1 < clip_rect.x2 && clip_rect.y1 < clip_rect.y2);

    std::vector<double> vertices;
    std::vector<uint8_t> codes;

    cleanup_path(path, trans, remove_nans, do_clip, clip_rect, snap_mode, stroke_width,
                 *simplify, return_curves, sketch, vertices, codes);

    auto length = static_cast<py::ssize_t>(codes.size());

    py::ssize_t vertices_dims[] = { length, 2 };
    py::array pyvertices(vertices_dims, vertices.data());

    py::ssize_t codes_dims[] = { length };
    py::array pycodes(codes_dims, codes.data());

    return py::make_tuple(pyvertices, pycodes);
}

const char *Py_convert_to_string__doc__ =
R"""(--

Convert *path* to a bytestring.

The first five parameters (up to *sketch*) are interpreted as in `.cleanup_path`. The
following ones are detailed below.

Parameters
----------
path : Path
trans : Transform or None
clip_rect : sequence of 4 floats, or None
simplify : bool
sketch : tuple of 3 floats, or None
precision : int
    The precision used to "%.*f"-format the values. Trailing zeros and decimal points
    are always removed. (precision=-1 is a special case used to implement
    ttconv-back-compatible conversion.)
codes : sequence of 5 bytestrings
    The bytes representation of each opcode (MOVETO, LINETO, CURVE3, CURVE4, CLOSEPOLY),
    in that order. If the bytes for CURVE3 is empty, quad segments are automatically
    converted to cubic ones (this is used by backends such as pdf and ps, which do not
    support quads).
postfix : bool
    Whether the opcode comes after the values (True) or before (False).
)""";

static py::object
Py_convert_to_string(mpl::PathIterator path, agg::trans_affine trans,
                     agg::rect_d cliprect, std::optional<bool> simplify,
                     SketchParams sketch, int precision,
                     std::array<std::string, 5> codes_obj, bool postfix)
{
    char *codes[5];
    std::string buffer;
    bool status;

    for (auto i = 0; i < 5; ++i) {
        codes[i] = const_cast<char *>(codes_obj[i].c_str());
    }

    if (!simplify.has_value()) {
        simplify = path.should_simplify();
    }

    status = convert_to_string(path, trans, cliprect, *simplify, sketch, precision,
                               codes, postfix, buffer);

    if (!status) {
        throw py::value_error("Malformed path codes");
    }

    return py::bytes(buffer);
}

const char *Py_is_sorted_and_has_non_nan__doc__ =
R"""(--

Return whether the 1D *array* is monotonically increasing, ignoring NaNs, and has at
least one non-nan value.)""";

static bool
Py_is_sorted_and_has_non_nan(py::object obj)
{
    bool result;

    py::array array = py::array::ensure(obj);
    if (array.ndim() != 1) {
        throw std::invalid_argument("array must be 1D");
    }

    auto dtype = array.dtype();
    /* Handle just the most common types here, otherwise coerce to double */
    if (dtype.equal(py::dtype::of<std::int32_t>())) {
        result = is_sorted_and_has_non_nan<int32_t>(array);
    } else if (dtype.equal(py::dtype::of<std::int64_t>())) {
        result = is_sorted_and_has_non_nan<int64_t>(array);
    } else if (dtype.equal(py::dtype::of<float>())) {
        result = is_sorted_and_has_non_nan<float>(array);
    } else if (dtype.equal(py::dtype::of<double>())) {
        result = is_sorted_and_has_non_nan<double>(array);
    } else {
        array = py::array_t<double>::ensure(obj);
        result = is_sorted_and_has_non_nan<double>(array);
    }

    return result;
}

PYBIND11_MODULE(_path, m, py::mod_gil_not_used())
{
    m.def("point_in_path", &Py_point_in_path,
          "x"_a, "y"_a, "radius"_a, "path"_a, "trans"_a);
    m.def("points_in_path", &Py_points_in_path,
          "points"_a, "radius"_a, "path"_a, "trans"_a);
    m.def("get_path_collection_extents", &Py_get_path_collection_extents,
          "master_transform"_a, "paths"_a, "transforms"_a, "offsets"_a,
          "offset_transform"_a);
    m.def("point_in_path_collection", &Py_point_in_path_collection,
          "x"_a, "y"_a, "radius"_a, "master_transform"_a, "paths"_a, "transforms"_a,
          "offsets"_a, "offset_trans"_a, "filled"_a);
    m.def("path_in_path", &Py_path_in_path,
          "path_a"_a, "trans_a"_a, "path_b"_a, "trans_b"_a);
    m.def("clip_path_to_rect", &Py_clip_path_to_rect,
          "path"_a, "rect"_a, "inside"_a);
    m.def("affine_transform", &Py_affine_transform,
          "points"_a, "trans"_a);
    m.def("count_bboxes_overlapping_bbox", &Py_count_bboxes_overlapping_bbox,
          "bbox"_a, "bboxes"_a);
    m.def("path_intersects_path", &Py_path_intersects_path,
          "path1"_a, "path2"_a, "filled"_a = false);
    m.def("path_intersects_rectangle", &Py_path_intersects_rectangle,
          "path"_a, "rect_x1"_a, "rect_y1"_a, "rect_x2"_a, "rect_y2"_a,
          "filled"_a = false);
    m.def("convert_path_to_polygons", &Py_convert_path_to_polygons,
          "path"_a, "trans"_a, "width"_a = 0.0, "height"_a = 0.0,
          "closed_only"_a = false);
    m.def("cleanup_path", &Py_cleanup_path,
          "path"_a, "trans"_a, "remove_nans"_a, "clip_rect"_a, "snap_mode"_a,
          "stroke_width"_a, "simplify"_a, "return_curves"_a, "sketch"_a);
    m.def("convert_to_string", &Py_convert_to_string,
          "path"_a, "trans"_a, "clip_rect"_a, "simplify"_a, "sketch"_a, "precision"_a,
          "codes"_a, "postfix"_a,
          Py_convert_to_string__doc__);
    m.def("is_sorted_and_has_non_nan", &Py_is_sorted_and_has_non_nan,
          "array"_a,
          Py_is_sorted_and_has_non_nan__doc__);
}
