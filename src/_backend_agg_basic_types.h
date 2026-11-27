#ifndef MPL_BACKEND_AGG_BASIC_TYPES_H
#define MPL_BACKEND_AGG_BASIC_TYPES_H

/* Contains some simple types from the Agg backend that are also used
   by other modules */

#include <pybind11/pybind11.h>

#include <unordered_map>
#include <vector>

#include "agg_color_rgba.h"
#include "agg_math_stroke.h"
#include "agg_pixfmt_rgba.h"
#include "agg_trans_affine.h"
#include "path_converters.h"

#include "py_adaptors.h"
#include "py_converters.h"
#include "py_array.h"

namespace py = pybind11;

struct ClipPath
{
    PathIterator path;
    agg::trans_affine trans;
};

struct SketchParams
{
    double scale;
    double length;
    double randomness;
};

class Dashes
{
    typedef std::vector<std::pair<double, double> > dash_t;
    double dash_offset;
    dash_t dashes;

  public:
    double get_dash_offset() const
    {
        return dash_offset;
    }
    void set_dash_offset(double x)
    {
        dash_offset = x;
    }
    void add_dash_pair(double length, double skip)
    {
        dashes.emplace_back(length, skip);
    }
    size_t size() const
    {
        return dashes.size();
    }

    template <class T>
    void dash_to_stroke(T &stroke, double dpi, bool isaa)
    {
        double scaleddpi = dpi / 72.0;
        for (auto [val0, val1] : dashes) {
            val0 = val0 * scaleddpi;
            val1 = val1 * scaleddpi;
            if (!isaa) {
                val0 = (int)val0 + 0.5;
                val1 = (int)val1 + 0.5;
            }
            stroke.add_dash(val0, val1);
        }
        stroke.dash_start(get_dash_offset() * scaleddpi);
    }
};

typedef std::vector<Dashes> LinestyleVector;

class GCAgg
{
  public:
    GCAgg()
        : linewidth(1.0),
          alpha(1.0),
          comp_op(agg::comp_op_src_over),
          cap(agg::butt_cap),
          join(agg::round_join),
          snap_mode(SNAP_FALSE)
    {
    }

    ~GCAgg()
    {
    }

    double linewidth;
    double alpha;
    bool forced_alpha;
    agg::comp_op_e comp_op;
    agg::filling_rule_e filling_rule;
    agg::rgba color;
    bool isaa;

    agg::line_cap_e cap;
    agg::line_join_e join;

    agg::rect_d cliprect;

    ClipPath clippath;

    Dashes dashes;

    e_snap_mode snap_mode;

    PathIterator hatchpath;
    agg::rgba hatch_color;
    double hatch_linewidth;

    SketchParams sketch;

    bool has_hatchpath()
    {
        return hatchpath.total_vertices() != 0;
    }

  private:
    // prevent copying
    GCAgg(const GCAgg &);
    GCAgg &operator=(const GCAgg &);
};


namespace mpl {

inline void convert(agg::comp_op_e &dst, const py::object &src, std::string arg_name)
{
    const std::unordered_map<std::string, agg::comp_op_e> enum_values = {
        {"clear", agg::comp_op_clear},
        {"knockout", agg::comp_op_src},
        //{"dest", agg::comp_op_dst},
        {"normal", agg::comp_op_src_over},
        //{"dst_over", agg::comp_op_dst_over},
        //{"in", agg::comp_op_src_in},
        //{"dst_in", agg::comp_op_dst_in},
        //{"out", agg::comp_op_src_out},
        {"erase", agg::comp_op_dst_out},
        {"atop", agg::comp_op_src_atop},
        //{"dst_atop", agg::comp_op_dst_atop},
        {"xor", agg::comp_op_xor},
        {"plus", agg::comp_op_plus},
        {"multiply", agg::comp_op_multiply},
        {"screen", agg::comp_op_screen},
        {"overlay", agg::comp_op_overlay},
        {"darken", agg::comp_op_darken},
        {"lighten", agg::comp_op_lighten},
        {"color dodge", agg::comp_op_color_dodge},
        {"color burn", agg::comp_op_color_burn},
        {"hard light", agg::comp_op_hard_light},
        {"soft light", agg::comp_op_soft_light},
        {"difference", agg::comp_op_difference},
        {"exclusion", agg::comp_op_exclusion},
        {"hue", agg::comp_op_hsl_hue},
        {"saturation", agg::comp_op_hsl_saturation},
        {"color", agg::comp_op_hsl_color},
        {"luminosity", agg::comp_op_hsl_luminosity},
    };

    dst = enum_values.at(src.cast<std::string>());
}

inline void convert(agg::filling_rule_e &dst, const py::object &src,
                    const char *arg_name)
{
    const std::unordered_map<std::string, agg::filling_rule_e> enum_values = {
        {"nonzero", agg::fill_non_zero},
        {"evenodd", agg::fill_even_odd},
    };

    dst = enum_values.at(src.cast<std::string>());
}

inline void convert(agg::line_cap_e &dst, const py::object &src, std::string arg_name)
{
    const std::unordered_map<std::string, agg::line_cap_e> enum_values = {
        {"butt", agg::butt_cap},
        {"round", agg::round_cap},
        {"projecting", agg::square_cap},
    };

    dst = enum_values.at(src.cast<std::string>());
}

inline void convert(agg::line_join_e &dst, const py::object &src, std::string arg_name)
{
    const std::unordered_map<std::string, agg::line_join_e> enum_values = {
        {"miter", agg::miter_join_revert},
        {"round", agg::round_join},
        {"bevel", agg::bevel_join},
    };

    dst = enum_values.at(src.cast<std::string>());
}

inline void convert(ClipPath &dst, const py::object &src, std::string arg_name)
{
    if (src.is_none()) {
        return;
    }

    auto [path_obj, trans_obj] = src.cast<std::pair<py::object, py::object>>();
    if (!path_obj.is_none()) {
        convert(dst.path, path_obj, "clippath.path");
    }
    convert(dst.trans, trans_obj, "clippath.trans");
}

inline void convert(Dashes &dst, const py::object &src, std::string arg_name)
{
    auto [dash_offset, dashes_seq_or_none] =
        src.cast<std::pair<double, std::optional<py::sequence>>>();

    if (!dashes_seq_or_none) {
        return;
    }

    auto dashes_seq = *dashes_seq_or_none;

    auto nentries = dashes_seq.size();
    // If the dashpattern has odd length, iterate through it twice (in
    // accordance with the pdf/ps/svg specs).
    auto dash_pattern_length = (nentries % 2) ? 2 * nentries : nentries;

    for (py::size_t i = 0; i < dash_pattern_length; i += 2) {
        auto length = dashes_seq[i % nentries].cast<double>();
        auto skip = dashes_seq[(i + 1) % nentries].cast<double>();

        dst.add_dash_pair(length, skip);
    }

    dst.set_dash_offset(dash_offset);
}

inline void convert(LinestyleVector &dst, const py::object &src, std::string arg_name)
{
    auto src_seq = src.cast<py::sequence>();

    for (py::size_t i = 0; i < src_seq.size(); ++i) {
        Dashes dashes;
        convert(dashes, src_seq[i], "linestyle element");
        dst.push_back(dashes);
    }
}

inline void convert(SketchParams &dst, const py::object &src, std::string arg_name)
{
    if (src.is_none()) {
        dst.scale = 0.0;
        dst.length = 0.0;
        dst.randomness = 0.0;
        return;
    }

    auto params = src.cast<std::tuple<double, double, double>>();
    std::tie(dst.scale, dst.length, dst.randomness) = params;
}

inline void convert(GCAgg &gc, const py::object &src, std::string arg_name)
{
    cast(gc.linewidth, src.attr("_linewidth"), "gc.linewidth");
    cast(gc.alpha, src.attr("_alpha"), "gc.alpha");
    cast(gc.forced_alpha, src.attr("_forced_alpha"), "gc.forced_alpha");
    convert(gc.comp_op, src.attr("_blend_mode"), "gc.comp_op");
    convert(gc.filling_rule, src.attr("_fill_rule"), "gc.filling_rule");
    convert(gc.color, src.attr("_rgb"), "gc.color");
    cast(gc.isaa, src.attr("_antialiased"), "gc.isaa");
    convert(gc.cap, src.attr("_capstyle"), "gc.cap");
    convert(gc.join, src.attr("_joinstyle"), "gc.join");
    convert(gc.dashes, src.attr("get_dashes")(), "gc.dashes");
    convert(gc.cliprect, src.attr("_cliprect"), "gc.cliprect");
    convert(gc.clippath, src.attr("get_clip_path")(), "gc.clippath");
    convert(gc.snap_mode, src.attr("get_snap")(), "gc.snap_mode");
    convert(gc.hatchpath, src.attr("get_hatch_path")(), "gc.hatchpath");
    convert(gc.hatch_color, src.attr("get_hatch_color")(), "gc.hatch_color");
    cast(gc.hatch_linewidth, src.attr("get_hatch_linewidth")(), "gc.hatch_linewidth");
    convert(gc.sketch, src.attr("get_sketch_params")(), "gc.sketch");
}

} // namespace mpl

#endif
