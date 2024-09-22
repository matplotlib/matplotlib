#ifndef MPL_PY_CONVERTERS_11_H
#define MPL_PY_CONVERTERS_11_H

// pybind11 equivalent of py_converters.h

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include <unordered_map>

#include "agg_basics.h"
#include "agg_color_rgba.h"
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
    template <> struct type_caster<agg::line_cap_e> {
    public:
        PYBIND11_TYPE_CASTER(agg::line_cap_e, const_name("line_cap_e"));

        bool load(handle src, bool) {
            const std::unordered_map<std::string, agg::line_cap_e> enum_values = {
                {"butt", agg::butt_cap},
                {"round", agg::round_cap},
                {"projecting", agg::square_cap},
            };
            value = enum_values.at(src.cast<std::string>());
            return true;
        }
    };

    template <> struct type_caster<agg::line_join_e> {
    public:
        PYBIND11_TYPE_CASTER(agg::line_join_e, const_name("line_join_e"));

        bool load(handle src, bool) {
            const std::unordered_map<std::string, agg::line_join_e> enum_values = {
                {"miter", agg::miter_join_revert},
                {"round", agg::round_join},
                {"bevel", agg::bevel_join},
            };
            value = agg::miter_join_revert;
            value = enum_values.at(src.cast<std::string>());
            return true;
        }
    };

    template <> struct type_caster<mpl::PathIterator> {
    public:
        PYBIND11_TYPE_CASTER(mpl::PathIterator, const_name("PathIterator"));

        bool load(handle src, bool) {
            if (src.is_none()) {
                return true;
            }

            py::object vertices = src.attr("vertices");
            py::object codes = src.attr("codes");
            auto should_simplify = src.attr("should_simplify").cast<bool>();
            auto simplify_threshold = src.attr("simplify_threshold").cast<double>();

            if (!value.set(vertices.inc_ref().ptr(), codes.inc_ref().ptr(),
                           should_simplify, simplify_threshold)) {
                throw py::error_already_set();
            }

            return true;
        }
    };
#endif

/* Remove all this macro magic after dropping NumPy usage and just include `_backend_agg_basic_types.h`. */
#ifdef MPL_BACKEND_AGG_BASIC_TYPES_H
#  ifndef MPL_PY_ADAPTORS_H
#    error "py_adaptors.h must be included to get Agg type casters"
#  endif

    template <> struct type_caster<ClipPath> {
    public:
        PYBIND11_TYPE_CASTER(ClipPath, const_name("ClipPath"));

        bool load(handle src, bool) {
            if (src.is_none()) {
                return true;
            }

            auto clippath_tuple = src.cast<py::tuple>();

            auto path = clippath_tuple[0];
            if (!path.is_none()) {
                value.path = path.cast<mpl::PathIterator>();
            }
            value.trans = clippath_tuple[1].cast<agg::trans_affine>();

            return true;
        }
    };

    template <> struct type_caster<Dashes> {
    public:
        PYBIND11_TYPE_CASTER(Dashes, const_name("Dashes"));

        bool load(handle src, bool) {
            auto dash_tuple = src.cast<py::tuple>();
            auto dash_offset = dash_tuple[0].cast<double>();
            auto dashes_seq_or_none = dash_tuple[1];

            if (dashes_seq_or_none.is_none()) {
                return true;
            }

            auto dashes_seq = dashes_seq_or_none.cast<py::sequence>();

            auto nentries = dashes_seq.size();
            // If the dashpattern has odd length, iterate through it twice (in
            // accordance with the pdf/ps/svg specs).
            auto dash_pattern_length = (nentries % 2) ? 2 * nentries : nentries;

            for (py::size_t i = 0; i < dash_pattern_length; i += 2) {
                auto length = dashes_seq[i % nentries].cast<double>();
                auto skip = dashes_seq[(i + 1) % nentries].cast<double>();

                value.add_dash_pair(length, skip);
            }

            value.set_dash_offset(dash_offset);

            return true;
        }
    };

    template <> struct type_caster<SketchParams> {
    public:
        PYBIND11_TYPE_CASTER(SketchParams, const_name("SketchParams"));

        bool load(handle src, bool) {
            if (src.is_none()) {
                value.scale = 0.0;
                value.length = 0.0;
                value.randomness = 0.0;
                return true;
            }

            auto params = src.cast<std::tuple<double, double, double>>();
            std::tie(value.scale, value.length, value.randomness) = params;

            return true;
        }
    };

    template <> struct type_caster<GCAgg> {
    public:
        PYBIND11_TYPE_CASTER(GCAgg, const_name("GCAgg"));

        bool load(handle src, bool) {
            value.linewidth = src.attr("_linewidth").cast<double>();
            value.alpha = src.attr("_alpha").cast<double>();
            value.forced_alpha = src.attr("_forced_alpha").cast<bool>();
            value.color = src.attr("_rgb").cast<agg::rgba>();
            value.isaa = src.attr("_antialiased").cast<bool>();
            value.cap = src.attr("_capstyle").cast<agg::line_cap_e>();
            value.join = src.attr("_joinstyle").cast<agg::line_join_e>();
            value.dashes = src.attr("get_dashes")().cast<Dashes>();
            value.cliprect = src.attr("_cliprect").cast<agg::rect_d>();
            /* value.clippath = src.attr("get_clip_path")().cast<ClipPath>(); */
            convert_clippath(src.attr("get_clip_path")().ptr(), &value.clippath);
            value.snap_mode = src.attr("get_snap")().cast<e_snap_mode>();
            value.hatchpath = src.attr("get_hatch_path")().cast<mpl::PathIterator>();
            value.hatch_color = src.attr("get_hatch_color")().cast<agg::rgba>();
            value.hatch_linewidth = src.attr("get_hatch_linewidth")().cast<double>();
            value.sketch = src.attr("get_sketch_params")().cast<SketchParams>();

            return true;
        }
    };
#endif
}} // namespace PYBIND11_NAMESPACE::detail

#endif /* MPL_PY_CONVERTERS_11_H */
