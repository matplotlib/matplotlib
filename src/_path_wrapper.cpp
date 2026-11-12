#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <array>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "_path.h"

#include "agg_basic_types.h"
#include "agg_path_adaptors.h"
#include "agg_convert_overloads.h"

#include "mpl_utils.h"
#include "mpl_array.h"

namespace nb = nanobind;
using namespace nanobind::literals;

nb::list
convert_polygon_vector(std::vector<Polygon> &polygons)
{
    auto result = nb::list_builder(polygons.size());

    for (size_t i = 0; i < polygons.size(); ++i) {
        const auto& poly = polygons[i];
        auto arr = mpl::array<double, mpl::shape<-1, 2>>(poly.size());
        if (arr.shape(0) > 0) {
            memcpy(arr.data(), poly.data(), arr.nbytes());
        }
        result.put(arr);
    }

    return result.commit();
}

static bool
Py_point_in_path(double x, double y, double r,
                 const mpl::defer<mpl::PathIterator> &path_obj,
                 const mpl::defer<agg::trans_affine> &trans_obj)
{
    auto path = path_obj.value("path");
    auto trans = trans_obj.value("trans");

    return point_in_path(x, y, r, path, trans);
}

static BoolArray
Py_points_in_path(const mpl::defer<InputPointArray> &points_obj, double r,
                  const mpl::defer<mpl::PathIterator> &path_obj,
                  const mpl::defer<agg::trans_affine> &trans_obj)
{
    auto points = points_obj.value("points");
    auto path = path_obj.value("path");
    auto trans = trans_obj.value("trans");
    auto results = BoolArray(points.shape(0));

    points_in_path(points, r, path, trans, results);

    return results;
}

static nb::tuple
Py_get_path_collection_extents(const mpl::defer<agg::trans_affine> &master_transform_obj,
                               const mpl::defer<mpl::PathGenerator> &paths_obj,
                               const mpl::defer<AffineTransformArray> &transforms_obj,
                               const mpl::defer<InputPointArray> &offsets_obj,
                               const mpl::defer<agg::trans_affine> &offset_trans_obj)
{
    auto master_transform = master_transform_obj.value("master_transform");
    auto paths = paths_obj.value("paths");
    auto transforms = transforms_obj.value("transforms");
    auto offsets = offsets_obj.value("offsets");
    auto offset_trans = offset_trans_obj.value("offset_trans");
    extent_limits e;

    get_path_collection_extents(
        master_transform, paths, transforms, offsets, offset_trans, e);

    auto extents = mpl::array<double, mpl::shape<2, 2>>();

    extents(0, 0) = e.start.x;
    extents(0, 1) = e.start.y;
    extents(1, 0) = e.end.x;
    extents(1, 1) = e.end.y;

    auto minpos = mpl::array<double, mpl::shape<2>>();

    minpos(0) = e.minpos.x;
    minpos(1) = e.minpos.y;

    return nb::make_tuple(extents, minpos);
}

static IndexArray
Py_point_in_path_collection(double x, double y, double radius,
                            const mpl::defer<agg::trans_affine> &master_transform_obj,
                            const mpl::defer<mpl::PathGenerator> &paths_obj,
                            const mpl::defer<AffineTransformArray> &transforms_obj,
                            const mpl::defer<InputPointArray> &offsets_obj,
                            const mpl::defer<agg::trans_affine> &offset_trans_obj,
                            mpl::bool_ filled)
{
    auto master_transform = master_transform_obj.value("master_transform");
    auto paths = paths_obj.value("paths");
    auto transforms = transforms_obj.value("transforms");
    auto offsets = offsets_obj.value("offsets");
    auto offset_trans = offset_trans_obj.value("offset_trans");
    auto indices = std::vector<size_t>();

    point_in_path_collection(x, y, radius, master_transform, paths, transforms,
                             offsets, offset_trans, filled, indices);

    auto result = IndexArray(indices.size());
    result.assign(indices);

    return result;
}

static bool
Py_path_in_path(const mpl::defer<mpl::PathIterator> &path_a_obj,
                const mpl::defer<agg::trans_affine> &trans_a_obj,
                const mpl::defer<mpl::PathIterator> &path_b_obj,
                const mpl::defer<agg::trans_affine> &trans_b_obj)
{
    auto path_a = path_a_obj.value("path_a");
    auto trans_a = trans_a_obj.value("trans_a");
    auto path_b = path_b_obj.value("path_b");
    auto trans_b = trans_b_obj.value("trans_b");

    return path_in_path(path_a, trans_a, path_b, trans_b);
}

static nb::list
Py_clip_path_to_rect(const mpl::defer<mpl::PathIterator> &path_obj,
                     const mpl::defer<agg::rect_d> &rect_obj,
                     mpl::bool_ inside)
{
    auto path = path_obj.value("path");
    auto rect = rect_obj.value("rect");
    auto result = clip_path_to_rect(path, rect, inside);

    return convert_polygon_vector(result);
}

static OutputPointArray
Py_transform_points(const mpl::defer<InputPointArray> &points_obj,
                    const mpl::defer<agg::trans_affine> &trans_obj)
{
    auto points = points_obj.value("points");
    auto trans = trans_obj.value("trans");

    auto result = OutputPointArray(points.shape(0));
    transform_points(points, trans, result);

    return result;
}

static int
Py_count_bboxes_overlapping_bbox(const mpl::defer<agg::rect_d> &bbox_obj,
                                 const mpl::defer<BBoxArray> &bboxes_obj)
{
    auto bbox = bbox_obj.value("bbox");
    auto bboxes = bboxes_obj.value("bboxes");

    return count_bboxes_overlapping_bbox(bbox, bboxes);
}

static bool
Py_path_intersects_path(const mpl::defer<mpl::PathIterator> &path1_obj,
                        const mpl::defer<mpl::PathIterator> &path2_obj,
                        mpl::bool_ filled)
{
    auto path1 = path1_obj.value("path1");
    auto path2 = path2_obj.value("path2");

    agg::trans_affine identity;
    bool result;

    result = path_intersects_path(path1, path2);
    if (filled) {
        if (!result) {
            result = path_in_path(path1, identity, path2, identity);
        }
        if (!result) {
            result = path_in_path(path2, identity, path1, identity);
        }
    }

    return result;
}

static bool
Py_path_intersects_rectangle(const mpl::defer<mpl::PathIterator> &path_obj,
                             double rect_x1, double rect_y1,
                             double rect_x2, double rect_y2, mpl::bool_ filled)
{
    auto path = path_obj.value("path");

    return path_intersects_rectangle(path, rect_x1, rect_y1, rect_x2, rect_y2, filled);
}

static nb::list
Py_convert_path_to_polygons(const mpl::defer<mpl::PathIterator> &path_obj,
                            const mpl::defer<agg::trans_affine> &trans_obj,
                            double width, double height, mpl::bool_ closed_only)
{
    auto path = path_obj.value("path");
    auto trans = trans_obj.value("trans");
    auto result = std::vector<Polygon>();

    convert_path_to_polygons(path, trans, width, height, closed_only, result);

    return convert_polygon_vector(result);
}

static nb::tuple
Py_cleanup_path(const mpl::defer<mpl::PathIterator> &path_obj,
                const mpl::defer<agg::trans_affine> &trans_obj, mpl::bool_ remove_nans,
                const mpl::defer<agg::rect_d> &clip_rect_obj,
                const mpl::defer<e_snap_mode> snap_mode_obj, double stroke_width,
                std::optional<mpl::bool_> simplify_obj, mpl::bool_ return_curves,
                const mpl::defer<SketchParams> &sketch_obj)
{
    auto path = path_obj.value("path");
    auto trans = trans_obj.value("trans");
    auto clip_rect = clip_rect_obj.value("clip_rect");
    auto snap_mode = snap_mode_obj.value("snap_mode");
    auto sketch = sketch_obj.value("sketch");

    bool simplify = simplify_obj.has_value() ?
        simplify_obj.value() : path.should_simplify();

    bool do_clip = (clip_rect.x1 < clip_rect.x2 && clip_rect.y1 < clip_rect.y2);

    std::vector<double> vertices;
    std::vector<uint8_t> codes;

    cleanup_path(path, trans, remove_nans, do_clip, clip_rect, snap_mode, stroke_width,
                 simplify, return_curves, sketch, vertices, codes);

    auto vertices_arr = mpl::array<double, mpl::shape<-1, 2>>(vertices.size() / 2);
    vertices_arr.assign(vertices);

    auto codes_arr = mpl::array<uint8_t, mpl::ndim<1>>(codes.size());
    codes_arr.assign(codes);

    return nb::make_tuple(vertices_arr, codes_arr);
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
simplify : bool, or None
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

static nb::object
Py_convert_to_string(const mpl::defer<mpl::PathIterator> &path_obj,
                     const mpl::defer<agg::trans_affine> &trans_obj,
                     const mpl::defer<agg::rect_d> &cliprect_obj,
                     std::optional<mpl::bool_> simplify_obj,
                     const mpl::defer<SketchParams> &sketch_obj, int precision,
                     const std::array<nb::bytes, 5> &codes, mpl::bool_ postfix)
{
    auto path = path_obj.value("path");
    auto trans = trans_obj.value("trans");
    auto cliprect = cliprect_obj.value("cliprect");
    auto sketch = sketch_obj.value("sketch");

    std::string buffer;
    bool status;
    bool simplify;

    if (simplify_obj.has_value()) {
        simplify = simplify_obj.value();
    } else {
        simplify = path.should_simplify();
    }


    std::array<std::string, 5> codes_str;
    for (size_t i = 0; i < codes.size(); ++i) {
        codes_str[i].assign(codes[i].c_str(), codes[i].size());
    }

    status = convert_to_string(path, trans, cliprect, simplify, sketch, precision,
                               codes_str, postfix, buffer);

    if (!status) {
        throw nb::value_error("Malformed path codes");
    }

    return nb::bytes(buffer.data(), buffer.size());
}

const char *Py_is_sorted_and_has_non_nan__doc__ =
R"""(--

Return whether the 1D *array* is monotonically increasing, ignoring NaNs, and has at
least one non-nan value.)""";

static bool
Py_is_sorted_and_has_non_nan(nb::object in_obj)
{
    bool result;

    nb::object obj = in_obj;
    nb::ndarray<nb::ro> array;

    if (!try_cast(obj, array)) {
        obj = mpl::asarray(obj);
        try_cast(obj, array);
    }

    if (!array.is_valid()) {
        throw std::invalid_argument("array is not valid");
    }

    if (array.ndim() != 1) {
        throw std::invalid_argument("array must be 1D");
    }

    auto dtype = array.dtype();

    /* Handle just the most common types here, otherwise coerce to double */
    if (dtype == nb::dtype<std::int32_t>()) {
        result = is_sorted_and_has_non_nan<int32_t>(array.data(), array.size());
    } else if (dtype == nb::dtype<std::int64_t>()) {
        result = is_sorted_and_has_non_nan<int64_t>(array.data(), array.size());
    } else if (dtype == nb::dtype<float>()) {
        result = is_sorted_and_has_non_nan<float>(array.data(), array.size());
    } else if (dtype == nb::dtype<double>()) {
        result = is_sorted_and_has_non_nan<double>(array.data(), array.size());
    } else {
        nb::ndarray<const double> double_array;

        if (!try_cast(obj, double_array)) {
            throw std::invalid_argument("Could not coerce array to double");
        }
        result = is_sorted_and_has_non_nan<double>(
            double_array.data(), double_array.size());
    }

    return result;
}

NB_MODULE(_path, m)
{
    m.def("point_in_path", &Py_point_in_path,
          "x"_a, "y"_a, "radius"_a, "path"_a, "trans"_a.none());
    m.def("points_in_path", &Py_points_in_path,
          "points"_a, "radius"_a, "path"_a, "trans"_a.none());
    m.def("get_path_collection_extents", &Py_get_path_collection_extents,
          "master_transform"_a, "paths"_a, "transforms"_a, "offsets"_a,
          "offset_transform"_a);
    m.def("point_in_path_collection", &Py_point_in_path_collection,
          "x"_a, "y"_a, "radius"_a, "master_transform"_a, "paths"_a, "transforms"_a,
          "offsets"_a, "offset_trans"_a, "filled"_a);
    m.def("path_in_path", &Py_path_in_path,
          "path_a"_a, "trans_a"_a.none(), "path_b"_a, "trans_b"_a.none());
    m.def("clip_path_to_rect", &Py_clip_path_to_rect,
          "path"_a, "rect"_a, "inside"_a);
    m.def("transform_points", &Py_transform_points,
          "points"_a, "trans"_a);
    m.def("count_bboxes_overlapping_bbox", &Py_count_bboxes_overlapping_bbox,
          "bbox"_a, "bboxes"_a);
    m.def("path_intersects_path", &Py_path_intersects_path,
          "path1"_a, "path2"_a, "filled"_a = false);
    m.def("path_intersects_rectangle", &Py_path_intersects_rectangle,
          "path"_a, "rect_x1"_a, "rect_y1"_a, "rect_x2"_a, "rect_y2"_a,
          "filled"_a = false);
    m.def("convert_path_to_polygons", &Py_convert_path_to_polygons,
          "path"_a, "trans"_a.none(), "width"_a = 0.0, "height"_a = 0.0,
          "closed_only"_a = false);
    m.def("cleanup_path", &Py_cleanup_path,
          "path"_a, "trans"_a.none(), "remove_nans"_a, "clip_rect"_a.none(),
          "snap_mode"_a.none(), "stroke_width"_a, "simplify"_a.none(),
          "return_curves"_a, "sketch"_a.none());
    m.def("convert_to_string", &Py_convert_to_string,
          "path"_a, "trans"_a.none(), "clip_rect"_a.none(), "simplify"_a.none(),
          "sketch"_a.none(), "precision"_a, "codes"_a, "postfix"_a,
          Py_convert_to_string__doc__);
    m.def("is_sorted_and_has_non_nan", &Py_is_sorted_and_has_non_nan,
          "array"_a,
          Py_is_sorted_and_has_non_nan__doc__);
}
