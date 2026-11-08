/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_PY_CONVERTERS_H
#define MPL_PY_CONVERTERS_H

/***************************************************************************************
 * This module contains a number of conversion functions from Python types to C++ types.
 * Most of them meet the pybind11 type casters, and thus will automatically be applied
 * when a C++ function parameter uses their type.
 */

#include <pybind11/pybind11.h>


#include "agg_basics.h"
#include "agg_color_rgba.h"
#include "agg_trans_affine.h"
#include "mplutils.h"
#include "py_array.h"

namespace mpl {

namespace py = pybind11;

inline void convert(agg::rect_d &dst, const py::object &src, const char *arg_name)
{
    if (src.is_none()) {
        dst.x1 = 0.0;
        dst.y1 = 0.0;
        dst.x2 = 0.0;
        dst.y2 = 0.0;
        return;
    }

    auto arr = mpl::asarray<const double>(src);
    auto ndim = arr ? arr.ndim() : 0;

    if (ndim != 1 && ndim != 2) {
        throw py::value_error(mpl::format(
            "{} must have 1 or 2 dimensions, got {}", arg_name, ndim));
    }

    if (ndim == 2 && (arr.shape(0) != 2 || arr.shape(1) != 2)) {
        throw py::value_error(mpl::format(
            "{} must have shape (2, 2) or (4), got ({}, {})",
            arg_name, arr.shape(0), arr.shape(1)));
    }

    if (ndim == 1 && arr.shape(0) != 4) {
        throw py::value_error(mpl::format(
            "{} must have shape (2, 2) or (4), got ({})",
            arg_name, arr.shape(0)));
    }

    auto data = arr.data();
    dst.x1 = data[0];
    dst.y1 = data[1];
    dst.x2 = data[2];
    dst.y2 = data[3];
}

inline void convert(agg::rgba &dst, const py::object &src, const char *arg_name)
{
    if (src.is_none()) {
        dst.r = 0.0;
        dst.g = 0.0;
        dst.b = 0.0;
        dst.a = 0.0;
        return;
    }

    auto rgbatuple = src.cast<py::tuple>();
    dst.r = rgbatuple[0].cast<double>();
    dst.g = rgbatuple[1].cast<double>();
    dst.b = rgbatuple[2].cast<double>();
    dst.a = 1.0;

    if (rgbatuple.size() == 4) {
        dst.a = rgbatuple[3].cast<double>();
    } else if (rgbatuple.size() != 3) {
        throw py::value_error(mpl::format(
            "{} must be a tuple of length 3 or 4"));
    }
}

inline void convert(agg::trans_affine &dst, const py::object &src,
                    const char *arg_name)
{
    // If None assume identity transform
    if (src.is_none()) {
        return;
    }

    auto arr = mpl::array<double, mpl::shape<3, 3>>(src, arg_name);
    auto data = arr.data();

    dst.sx = data[0];
    dst.shx = data[1];
    dst.tx = data[2];
    dst.shy = data[3];
    dst.sy = data[4];
    dst.ty = data[5];
}

}

#endif /* MPL_PY_CONVERTERS_H */
