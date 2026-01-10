#ifndef MPL_BACKEND_AGG_BASIC_TYPES_H
#define MPL_BACKEND_AGG_BASIC_TYPES_H

/* Contains some simple types from the Agg backend that are also used
   by other modules */

#include <pybind11/pybind11.h>

#include <unordered_map>
#include <vector>

#include "agg_color_rgba.h"
#include "agg_math_stroke.h"
#include "agg_trans_affine.h"
#include "path_converters.h"

#include "py_adaptors.h"

namespace py = pybind11;

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

typedef std::vector<Dashes> DashesVector;

class GCAgg
{
  public:
    GCAgg()
        : linewidth(1.0),
          alpha(1.0),
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

class VGCAgg
{
  public:
    VGCAgg()
    {
    }
    ~VGCAgg()
    {
    }

    py::array_t<double> alphas;
    py::array_t<uint8_t> forced_alphas;
    py::array_t<uint8_t> antialiaseds;
    py::array_t<double> linewidths;
    py::array_t<double> edgecolors;
    py::array_t<double> facecolors;

    std::vector<agg::line_cap_e> capstyles;
    std::vector<agg::line_join_e> joinstyles;

    agg::rect_d cliprect;

    ClipPath clippath;

    std::vector<Dashes> dashes;

    std::vector<mpl::PathIterator> hatchpaths;
    py::array_t<double> hatch_colors;
    py::array_t<double> hatch_linewidths;

    std::vector<e_snap_mode> snap_modes;

    std::vector<SketchParams> sketches;

  private:
    // prevent copying
    VGCAgg(const VGCAgg &);
    VGCAgg &operator=(const VGCAgg &);
};

namespace PYBIND11_NAMESPACE { namespace detail {
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
            value = enum_values.at(src.cast<std::string>());
            return true;
        }
    };

    template <> struct type_caster<ClipPath> {
    public:
        PYBIND11_TYPE_CASTER(ClipPath, const_name("ClipPath"));

        bool load(handle src, bool) {
            if (src.is_none()) {
                return true;
            }

            auto [path, trans] =
                src.cast<std::pair<std::optional<mpl::PathIterator>, agg::trans_affine>>();
            if (path) {
                value.path = *path;
            }
            value.trans = trans;

            return true;
        }
    };

    template <> struct type_caster<Dashes> {
    public:
        PYBIND11_TYPE_CASTER(Dashes, const_name("Dashes"));

        bool load(handle src, bool) {
            auto [dash_offset, dashes_seq_or_none] =
                src.cast<std::pair<double, std::optional<py::sequence>>>();

            if (!dashes_seq_or_none) {
                return true;
            }

            auto dashes_seq = *dashes_seq_or_none;

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
            value.clippath = src.attr("get_clip_path")().cast<ClipPath>();
            value.snap_mode = src.attr("get_snap")().cast<e_snap_mode>();
            value.hatchpath = src.attr("get_hatch_path")().cast<mpl::PathIterator>();
            value.hatch_color = src.attr("get_hatch_color")().cast<agg::rgba>();
            value.hatch_linewidth = src.attr("get_hatch_linewidth")().cast<double>();
            value.sketch = src.attr("get_sketch_params")().cast<SketchParams>();

            return true;
        }
    };

    template <> struct type_caster<VGCAgg> {
    public:
        PYBIND11_TYPE_CASTER(VGCAgg, const_name("VGCAgg"));

        bool load(handle src, bool) {
            value.alphas = src.attr("get_alphas")().cast<py::array_t<double>>();
            value.forced_alphas = src.attr("get_forced_alphas")().cast<py::array_t<uint8_t>>();
            value.antialiaseds = src.attr("get_antialiaseds")().cast<py::array_t<uint8_t>>();
            value.capstyles = src.attr("get_capstyles")().cast<std::vector<agg::line_cap_e>>();
            value.dashes = src.attr("get_dashes")().cast<std::vector<Dashes>>();
            value.joinstyles = src.attr("get_joinstyles")().cast<std::vector<agg::line_join_e>>();
            value.linewidths = src.attr("get_linewidths")().cast<py::array_t<double>>();
            value.edgecolors = src.attr("get_edgecolors")().cast<py::array_t<double>>().reshape({-1, 4});
            value.facecolors = src.attr("get_facecolors")().cast<py::array_t<double>>().reshape({-1, 4});
            value.hatchpaths = src.attr("get_hatch_paths")().cast<std::vector<mpl::PathIterator>>();
            value.hatch_colors = src.attr("get_hatch_colors")().cast<py::array_t<double>>().reshape({-1, 4});
            value.hatch_linewidths = src.attr("get_hatch_linewidths")().cast<py::array_t<double>>();
            value.snap_modes = src.attr("get_snaps")().cast<std::vector<e_snap_mode>>();
            value.sketches = src.attr("get_sketches_params")().cast<std::vector<SketchParams>>();
            value.cliprect = src.attr("get_clip_rectangle")().cast<agg::rect_d>();
            value.clippath = src.attr("get_clip_path")().cast<ClipPath>();

            return true;
        }
    };
}} // namespace PYBIND11_NAMESPACE::detail

#endif
