/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_PY_CONVERTERS_H
#define MPL_PY_CONVERTERS_H

/***************************************************************************************
 * This module contains a number of conversion functions from Python types to C++ types.
 * Most of them meet the nanobind type casters, and thus will automatically be applied
 * when a C++ function parameter uses their type.
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

#include "agg_basics.h"
#include "agg_color_rgba.h"
#include "agg_trans_affine.h"
#include "mplutils.h"

inline auto convert_points(nb::ndarray<double> obj)
{
    check_trailing_shape(obj, "points", 2);
    return obj.view<double, nb::ndim<2>>();
}

inline auto convert_transforms(nb::ndarray<double> obj)
{
    check_trailing_shape(obj, "transforms", 3, 3);
    return obj.view<double, nb::ndim<3>>();
}

inline auto convert_bboxes(nb::ndarray<double> obj)
{
    check_trailing_shape(obj, "bbox array", 2, 2);
    return obj.view<double, nb::ndim<3>>();
}

inline auto convert_colors(nb::ndarray<double> obj)
{
    check_trailing_shape(obj, "colors", 4);
    return obj.view<double, nb::ndim<2>>();
}

namespace NB_NAMESPACE { namespace detail {
    template <> struct type_caster<agg::rect_d> {
    public:
        NB_TYPE_CASTER(agg::rect_d, const_name("rect_d"));

        bool from_python(handle src, uint8_t, cleanup_list *) {
            if (src.is_none()) {
                value.x1 = 0.0;
                value.y1 = 0.0;
                value.x2 = 0.0;
                value.y2 = 0.0;
                return true;
            }

            nb::ndarray<double, nb::c_contig> rect_arr;
            auto ndim = nb::try_cast(src, rect_arr) ? rect_arr.ndim() : 0;

            if (ndim == 2) {
                if (rect_arr.shape(0) != 2 || rect_arr.shape(1) != 2) {
                    throw nb::value_error("Invalid bounding box");
                }

                value.x1 = rect_arr.data()[0];
                value.y1 = rect_arr.data()[1];
                value.x2 = rect_arr.data()[2];
                value.y2 = rect_arr.data()[3];

            } else if (ndim == 1) {
                if (rect_arr.shape(0) != 4) {
                    throw nb::value_error("Invalid bounding box");
                }

                value.x1 = rect_arr.data()[0];
                value.y1 = rect_arr.data()[1];
                value.x2 = rect_arr.data()[2];
                value.y2 = rect_arr.data()[3];

            } else {
                throw nb::value_error("Invalid bounding box");
            }

            return true;
        }
    };

    template <> struct type_caster<agg::rgba> {
    public:
        NB_TYPE_CASTER(agg::rgba, const_name("rgba"));

        bool from_python(handle src, uint8_t, cleanup_list *) {
            if (src.is_none()) {
                value.r = 0.0;
                value.g = 0.0;
                value.b = 0.0;
                value.a = 0.0;
            } else {
                auto rgbatuple = nb::cast<nb::tuple>(src);
                value.r = nb::cast<double>(rgbatuple[0]);
                value.g = nb::cast<double>(rgbatuple[1]);
                value.b = nb::cast<double>(rgbatuple[2]);
                switch (rgbatuple.size()) {
                case 4:
                    value.a = nb::cast<double>(rgbatuple[3]);
                    break;
                case 3:
                    value.a = 1.0;
                    break;
                default:
                    throw nb::value_error("RGBA value must be 3- or 4-tuple");
                }
            }
            return true;
        }
    };

    template <> struct type_caster<agg::trans_affine> {
    public:
        NB_TYPE_CASTER(agg::trans_affine, const_name("trans_affine"));

        bool from_python(handle src, uint8_t, cleanup_list *) {
            // If None assume identity transform so leave affine unchanged
            if (src.is_none()) {
                return true;
            }

            nb::ndarray<double, nb::c_contig> array;
            if (!nb::try_cast(src, array) || array.ndim() != 2 ||
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
}} // namespace NB_NAMESPACE::detail

#endif /* MPL_PY_CONVERTERS_H */
