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
#include "py_array.h"

void convert_trans_affine(const py::object& transform, agg::trans_affine& affine);

inline auto convert_points(py::array_t<double> obj)
{
    check_trailing_shape(obj, "points", 2);
    return obj.unchecked<2>();
}


inline auto convert_agg_rect_d(const py::object &src, const char *arg_name)
{
    if (src.is_none()) {
        return agg::rect_d(0.0, 0.0, 0.0, 0.0);
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
    return agg::rect_d(data[0], data[1], data[2], data[3]);
}

inline auto convert_agg_rgba(const py::object &src, const char *arg_name)
{
    if (src.is_none()) {
        return agg::rgba(0.0, 0.0, 0.0, 0.0);
    }

    auto rgbatuple = src.cast<py::tuple>();
    auto r = rgbatuple[0].cast<double>();
    auto g = rgbatuple[1].cast<double>();
    auto b = rgbatuple[2].cast<double>();
    auto a = 1.0;

    if (rgbatuple.size() == 4) {
        a = rgbatuple[3].cast<double>();
    } else if (rgbatuple.size() != 3) {
        throw py::value_error(mpl::format(
            "{} must be a tuple of length 3 or 4"));
    }

    return agg::rgba(r, g, b, a);
}


inline auto convert_agg_trans_affine(const py::object &src, const char *arg_name)
{
    // If None assume identity transform
    if (src.is_none()) {
        return agg::trans_affine();
    }

    auto arr = mpl::array<double, mpl::shape<3, 3>>(src, arg_name);
    auto data = arr.data();

    // trans_affine's constructor is (sx, shy, shx, sy, tx, ty)
    return agg::trans_affine(data[0], data[3], data[1], data[4], data[2], data[5]);
}


template <>
class mpl::defer<agg::rect_d> : public py::object
{
    PYBIND11_OBJECT_DEFAULT(defer, py::object, check_any)

    agg::rect_d value(const char *arg_name) const
    {
        return convert_agg_rect_d(*this, arg_name);
    }
};


template <>
class mpl::defer<agg::rgba> : public py::object
{
    PYBIND11_OBJECT_DEFAULT(defer, py::object, check_any)

    agg::rgba value(const char *arg_name) const
    {
        return convert_agg_rgba(*this, arg_name);
    }
};


template <>
class mpl::defer<agg::trans_affine> : public py::object
{
    PYBIND11_OBJECT_DEFAULT(defer, py::object, check_any)

    agg::trans_affine value(const char *arg_name) const
    {
        return convert_agg_trans_affine(*this, arg_name);
    }
};





#if 0


namespace PYBIND11_NAMESPACE { namespace detail {
    template <> struct type_caster<agg::rect_d> {
    public:
        PYBIND11_TYPE_CASTER(agg::rect_d, const_name("rect_d"));

        template <typename T = agg::rect_d>
        bool load(handle, bool) {
            static_assert(mpl::internal::always_false_v<T>,
                "rect_d caster is deprecated");
            return false;
        }
    };


    template <> struct type_caster<agg::rgba> {
    public:
        PYBIND11_TYPE_CASTER(agg::rgba, const_name("rgba"));

        template <typename T = agg::rgba>
        bool load(handle, bool) {
            static_assert(mpl::internal::always_false_v<T>,
                "rgba caster is deprecated");
            return false;
        }
    };

    template <> struct type_caster<agg::trans_affine> {
    public:
        PYBIND11_TYPE_CASTER(agg::trans_affine, const_name("trans_affine"));

        template <typename T = agg::trans_affine>
        bool load(handle, bool) {
            static_assert(mpl::internal::always_false_v<T>,
                "trans_affine caster is deprecated");
            return false;
        }
    };
}} // namespace PYBIND11_NAMESPACE::detail


#endif





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
