/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_PY_CONVERTERS_H
#define MPL_PY_CONVERTERS_H

/***************************************************************************************
 * This module contains a number of conversion functions from Python types to C++ types.
 * Most of them meet the pybind11 type casters, and thus will automatically be applied
 * when a C++ function parameter uses their type.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "agg_basics.h"
#include "agg_color_rgba.h"
#include "agg_trans_affine.h"
#include "mplutils.h"

void convert_trans_affine(const py::object& transform, agg::trans_affine& affine);


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
            auto ndim = rect_arr ? rect_arr.ndim() : 0;

            if (ndim == 2) {
                if (rect_arr.shape(0) != 2 || rect_arr.shape(1) != 2) {
                    throw py::value_error("Invalid bounding box");
                }

                value.x1 = *rect_arr.data(0, 0);
                value.y1 = *rect_arr.data(0, 1);
                value.x2 = *rect_arr.data(1, 0);
                value.y2 = *rect_arr.data(1, 1);

            } else if (ndim == 1) {
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

#endif /* MPL_PY_CONVERTERS_H */
