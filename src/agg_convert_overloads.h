/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_AGG_CONVERT_OVERLOADS_H
#define MPL_AGG_CONVERT_OVERLOADS_H

/***************************************************************************************
 * This module contains a number of conversion functions from Python types to C++ types.
 * Most of them meet the pybind11 type casters, and thus will automatically be applied
 * when a C++ function parameter uses their type.
 */

#include <nanobind/nanobind.h>


#include "agg_basics.h"
#include "agg_color_rgba.h"
#include "agg_trans_affine.h"
#include "mpl_utils.h"
#include "mpl_array.h"

namespace mpl {

namespace nb = nanobind;

inline void convert(agg::rect_d &dst, const nb::object &src, std::string arg_name)
{
    if (src.is_none()) {
        dst.x1 = 0.0;
        dst.y1 = 0.0;
        dst.x2 = 0.0;
        dst.y2 = 0.0;
        return;
    }

    nb::ndarray<const double> arr;
    if (!nb::try_cast(src, arr)) {
        nb::try_cast(mpl::asarray(src, "f8"), arr);
    }

    if (!arr.is_valid()) {
        throw nb::value_error(mpl::format(
            "{} could not be converted to a nanobind ndarray", arg_name));
    }

    auto ndim = arr.ndim();

    if (ndim != 1 && ndim != 2) {
        throw nb::value_error(mpl::format(
            "{} must have 1 or 2 dimensions, got {}", arg_name, ndim));
    }

    if (ndim == 2 && (arr.shape(0) != 2 || arr.shape(1) != 2)) {
        throw nb::value_error(mpl::format(
            "{} must have shape (2, 2) or (4), got ({}, {})",
            arg_name, arr.shape(0), arr.shape(1)));
    }

    if (ndim == 1 && arr.shape(0) != 4) {
        throw nb::value_error(mpl::format(
            "{} must have shape (2, 2) or (4), got ({})",
            arg_name, arr.shape(0)));
    }

    auto data = arr.data();
    dst.x1 = data[0];
    dst.y1 = data[1];
    dst.x2 = data[2];
    dst.y2 = data[3];
}

inline void convert(agg::rgba &dst, const nb::object &src, std::string arg_name)
{
    if (src.is_none()) {
        dst.r = 0.0;
        dst.g = 0.0;
        dst.b = 0.0;
        dst.a = 0.0;
        return;
    }

    nb::sequence seq;
    cast(seq, src, "rgba");

    auto seq_len = nb::len(seq);
    if (seq_len == 3 || seq_len == 4) {
        cast(dst.r, seq[0], "rgba.r");
        cast(dst.g, seq[1], "rgba.g");
        cast(dst.b, seq[2], "rgba.b");
        dst.a = 1.0;

        if (seq_len == 4) {
            cast(dst.a, seq[3], "rgba.a");
        }
    } else {
        throw nb::value_error(mpl::format(
            "{} must be a sequence of length 3 or 4"));
    }
}

inline void convert(agg::trans_affine &dst, const nb::object &src,
                    std::string arg_name)
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

MPL_TYPE_NAME(agg::rect_d, "numpy.typing.ArrayLike")
MPL_TYPE_NAME(agg::rgba, "Sequence")
MPL_TYPE_NAME(agg::trans_affine, "numpy.ndarray")

}

#endif /* MPL_AGG_CONVERT_OVERLOADS_H */
