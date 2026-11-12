#ifndef MPL_AGG_BASIC_TYPES_H
#define MPL_AGG_BASIC_TYPES_H

/* Contains some simple types from the Agg backend that are also used
   by other modules */

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <unordered_map>
#include <vector>

#include "agg_color_rgba.h"
#include "agg_math_stroke.h"
#include "agg_pixfmt_rgba.h"
#include "agg_trans_affine.h"

#include "mpl_utils.h"
#include "agg_convert_overloads.h"
#include "agg_path_adaptors.h"
#include "agg_path_converters.h"


struct ClipPath
{
    mpl::PathIterator path;
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

    mpl::PathIterator hatchpath;
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

namespace internal {

template <typename Enum>
inline void convert_enum(Enum &dst, const nb::object &src, const std::string &arg_name,
    std::initializer_list<std::pair<std::string, Enum>> list)
{
    std::string src_str;
    if (!nb::try_cast(src, src_str)) {
        throw std::invalid_argument(mpl::format("{} must be a string", arg_name));
    }

    auto predicate = [&](const auto &entry) { return entry.first == src_str; };
    const auto it = std::find_if(list.begin(), list.end(), predicate);

    if (it == list.end()) {
        throw std::invalid_argument(mpl::format(
            "{} does not have a value for '{}'", arg_name, src_str));
    }

    dst = it->second;
}

} // internal


inline void convert(agg::comp_op_e &dst, const nb::object &src, std::string arg_name)
{
    internal::convert_enum(dst, src, arg_name, {
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
    });
}

inline void convert(agg::filling_rule_e &dst, const nb::object &src,
                    std::string arg_name)
{
    internal::convert_enum(dst, src, arg_name, {
        {"nonzero", agg::fill_non_zero},
        {"evenodd", agg::fill_even_odd},
    });
}

inline void convert(agg::line_cap_e &dst, const nb::object &src, std::string arg_name)
{
    internal::convert_enum(dst, src, arg_name, {
        {"butt", agg::butt_cap},
        {"round", agg::round_cap},
        {"projecting", agg::square_cap},
    });
}

inline void convert(agg::line_join_e &dst, const nb::object &src, std::string arg_name)
{
    internal::convert_enum(dst, src, arg_name, {
        {"miter", agg::miter_join_revert},
        {"round", agg::round_join},
        {"bevel", agg::bevel_join},
    });
}

inline void convert(ClipPath &dst, const nb::object &src, std::string arg_name)
{
    if (src.is_none()) {
        return;
    }

    // Convert to a sequence as pybind11's typecaster was lenient
    nb::sequence src_seq;
    if (!nb::try_cast(src, src_seq) || nb::len(src_seq) != 2) {
        throw std::invalid_argument(mpl::format("{} must be a 2-tuple", arg_name));
    }

    nb::object path_obj = src_seq[0];
    nb::object trans_obj = src_seq[1];

    if (!path_obj.is_none()) {
        convert(dst.path, path_obj, arg_name + "[0]");
    }
    convert(dst.trans, trans_obj, arg_name + "[1]");
}

inline void convert(Dashes &dst, const nb::object &src, std::string arg_name)
{
    // Convert to a sequence as pybind11's typecaster was lenient
    nb::sequence src_seq;
    if (!nb::try_cast(src, src_seq) || nb::len(src_seq) != 2) {
        throw std::invalid_argument(mpl::format("{} must be a 2-tuple", arg_name));
    }

    double dash_offset;
    cast(dash_offset, src_seq[0], arg_name + "[0]");

    std::optional<nb::sequence> dashes_seq_or_none;
    if (!try_cast(src_seq[1], dashes_seq_or_none)) {
        throw std::invalid_argument(mpl::format(
            "{}[1] must be a sequence of floats or None", arg_name));
    }

    if (!dashes_seq_or_none) {
        return;
    }

    try {
        auto dashes_seq = *dashes_seq_or_none;

        auto nentries = nb::len(dashes_seq);
        // If the dashpattern has odd length, iterate through it twice (in
        // accordance with the pdf/ps/svg specs).
        auto dash_pattern_length = (nentries % 2) ? 2 * nentries : nentries;

        for (size_t i = 0; i < dash_pattern_length; i += 2) {
            auto length = nb::cast<double>(dashes_seq[i % nentries]);
            auto skip =  nb::cast<double>(dashes_seq[(i + 1) % nentries]);

            dst.add_dash_pair(length, skip);
        }

        dst.set_dash_offset(dash_offset);

    } catch (...) {
        throw std::invalid_argument(mpl::format(
            "{}[1] must be a sequence of floats", arg_name));
    }
}

inline void convert(LinestyleVector &dst, const nb::object &src, std::string arg_name)
{
    nb::sequence src_seq;
    if (!nb::try_cast(src, src_seq)) {
        throw std::invalid_argument(mpl::format("{} must be a sequence", arg_name));
    }

    auto length = nb::len(src_seq);

    size_t i = 0;
    try {
        for ( ; i < length; ++i) {
            Dashes dashes;
            convert(dashes, src_seq[i], "");
            dst.push_back(dashes);
        }
    } catch (...) {
        throw std::invalid_argument(mpl::format(
            "{} contained an invalid object at index {}", i));
    }
}

inline void convert(SketchParams &dst, const nb::object &src, std::string arg_name)
{
    if (src.is_none()) {
        dst.scale = 0.0;
        dst.length = 0.0;
        dst.randomness = 0.0;
        return;
    }

    // Convert to a sequence as pybind11's typecaster was lenient
    nb::sequence src_seq;
    if (!nb::try_cast(src, src_seq) || nb::len(src_seq) != 3) {
        throw std::invalid_argument(mpl::format("{} must be a 3-tuple", arg_name));
    }

    cast(dst.scale, src_seq[0], arg_name + "[0]");
    cast(dst.length, src_seq[1], arg_name + "[1]");
    cast(dst.randomness, src_seq[2], arg_name + "[2]");
}

inline void convert(GCAgg &gc, const nb::object &src, std::string arg_name)
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

MPL_TYPE_NAME(agg::comp_op_e, "BlendModeType")
MPL_TYPE_NAME(agg::filling_rule_e, "FillRuleType")
MPL_TYPE_NAME(agg::line_cap_e, "CapStyleType")
MPL_TYPE_NAME(agg::line_join_e, "JoinStyleType")
MPL_TYPE_NAME(ClipPath, "tuple[TransformedPath, Transform] | tuple[None, None]")
MPL_TYPE_NAME(Dashes, "tuple[float, ArrayLike | None]")
MPL_TYPE_NAME(LinestyleVector, "Sequence[LineStyleType]")
MPL_TYPE_NAME(SketchParams, "tuple[float, float, float]")
MPL_TYPE_NAME(GCAgg, "GraphicsContextBase")

} // namespace mpl

#endif
