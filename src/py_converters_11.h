#ifndef MPL_PY_CONVERTERS_11_H
#define MPL_PY_CONVERTERS_11_H

// pybind11 equivalent of py_converters.h

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "agg_basics.h"
#include "agg_trans_affine.h"
#include "path_converters.h"

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

    template <> struct type_caster<e_snap_mode> {
    public:
        PYBIND11_TYPE_CASTER(e_snap_mode, const_name("e_snap_mode"));

        bool load(handle src, bool) {
            if (src.is_none()) {
                value = SNAP_AUTO;
                return true;
            }

            value = src.cast<bool>() ? SNAP_TRUE : SNAP_FALSE;

            return true;
        }
    };

/* Remove all this macro magic after dropping NumPy usage and just include `py_adaptors.h`. */
#ifdef MPL_PY_ADAPTORS_H
    template <> struct type_caster<mpl::PathIterator> {
    public:
        PYBIND11_TYPE_CASTER(mpl::PathIterator, const_name("PathIterator"));

        bool load(handle src, bool) {
            if (src.is_none()) {
                return true;
            }

            auto vertices = src.attr("vertices");
            auto codes = src.attr("codes");
            auto should_simplify = src.attr("should_simplify").cast<bool>();
            auto simplify_threshold = src.attr("simplify_threshold").cast<double>();

            if (!value.set(vertices.ptr(), codes.ptr(),
                           should_simplify, simplify_threshold)) {
                return false;
            }

            return true;
        }
    };
#endif

/* Remove all this macro magic after dropping NumPy usage and just include `_backend_agg_basic_types.h`. */
#ifdef MPL_BACKEND_AGG_BASIC_TYPES_H
    template <> struct type_caster<SketchParams> {
    public:
        PYBIND11_TYPE_CASTER(SketchParams, const_name("SketchParams"));

        bool load(handle src, bool) {
            if (src.is_none()) {
                value.scale = 0.0;
                return true;
            }

            auto params = src.cast<std::tuple<double, double, double>>();
            std::tie(value.scale, value.length, value.randomness) = params;

            return true;
        }
    };
#endif
}} // namespace PYBIND11_NAMESPACE::detail

#endif /* MPL_PY_CONVERTERS_11_H */
