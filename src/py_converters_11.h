#ifndef MPL_PY_CONVERTERS_11_H
#define MPL_PY_CONVERTERS_11_H

// pybind11 equivalent of py_converters.h

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "agg_basics.h"
#include "agg_color_rgba.h"
#include "agg_trans_affine.h"
#include "mplutils.h"

void convert_trans_affine(const py::object& transform, agg::trans_affine& affine);

inline auto convert_points(py::array_t<double> obj)
{
    if (!check_trailing_shape(obj, "points", 2)) {
        throw py::error_already_set();
    }
    return obj.unchecked<2>();
}

inline auto convert_transforms(py::array_t<double> obj)
{
    if (!check_trailing_shape(obj, "transforms", 3, 3)) {
        throw py::error_already_set();
    }
    return obj.unchecked<3>();
}

inline auto convert_bboxes(py::array_t<double> obj)
{
    if (!check_trailing_shape(obj, "bbox array", 2, 2)) {
        throw py::error_already_set();
    }
    return obj.unchecked<3>();
}

inline auto convert_colors(py::array_t<double> obj)
{
    if (!check_trailing_shape(obj, "colors", 4)) {
        throw py::error_already_set();
    }
    return obj.unchecked<2>();
}

namespace PYBIND11_NAMESPACE { namespace detail {
    template <> struct type_caster<agg::rect_d> {
    public:
        PYBIND11_TYPE_CASTER(agg::rect_d, const_name("rect_d"));

        bool load(handle src, bool) {
            if (src.is_none()) {
                value.x1 = 0.0;
                value.y1 = 0.0;
                value.x2 = 0.0;
                value.y2 = 0.0;
                return true;
            }

            auto rect_arr = py::array_t<double>::ensure(src);

            if (rect_arr.ndim() == 2) {
                if (rect_arr.shape(0) != 2 || rect_arr.shape(1) != 2) {
                    throw py::value_error("Invalid bounding box");
                }

                value.x1 = *rect_arr.data(0, 0);
                value.y1 = *rect_arr.data(0, 1);
                value.x2 = *rect_arr.data(1, 0);
                value.y2 = *rect_arr.data(1, 1);

            } else if (rect_arr.ndim() == 1) {
                if (rect_arr.shape(0) != 4) {
                    throw py::value_error("Invalid bounding box");
                }

                value.x1 = *rect_arr.data(0);
                value.y1 = *rect_arr.data(1);
                value.x2 = *rect_arr.data(2);
                value.y2 = *rect_arr.data(3);

            } else {
                throw py::value_error("Invalid bounding box");
            }

            return true;
        }
    };

    template <> struct type_caster<agg::rgba> {
    public:
        PYBIND11_TYPE_CASTER(agg::rgba, const_name("rgba"));

        bool load(handle src, bool) {
            if (src.is_none()) {
                value.r = 0.0;
                value.g = 0.0;
                value.b = 0.0;
                value.a = 0.0;
            } else {
                auto rgbatuple = src.cast<py::tuple>();
                value.r = rgbatuple[0].cast<double>();
                value.g = rgbatuple[1].cast<double>();
                value.b = rgbatuple[2].cast<double>();
                switch (rgbatuple.size()) {
                case 4:
                    value.a = rgbatuple[3].cast<double>();
                    break;
                case 3:
                    value.a = 1.0;
                    break;
                default:
                    throw py::value_error("RGBA value must be 3- or 4-tuple");
                }
            }
            return true;
        }
    };

    template <> struct type_caster<agg::trans_affine> {
    public:
        PYBIND11_TYPE_CASTER(agg::trans_affine, const_name("trans_affine"));

        bool load(handle src, bool) {
            // If None assume identity transform so leave affine unchanged
            if (src.is_none()) {
                return true;
            }

            auto array = py::array_t<double, py::array::c_style>::ensure(src);
            if (!array || array.ndim() != 2 ||
                    array.shape(0) != 3 || array.shape(1) != 3) {
                throw std::invalid_argument("Invalid affine transformation matrix");
            }

            auto buffer = array.data();
            value.sx = buffer[0];
            value.shx = buffer[1];
            value.tx = buffer[2];
            value.shy = buffer[3];
            value.sy = buffer[4];
            value.ty = buffer[5];

            return true;
        }
    };
}} // namespace PYBIND11_NAMESPACE::detail

#endif /* MPL_PY_CONVERTERS_11_H */
