#ifndef __BACKEND_AGG_BASIC_TYPES_H__
#define __BACKEND_AGG_BASIC_TYPES_H__

/* Contains some simple types from the Agg backend that are also used
   by other modules */

#include <vector>

#include "agg_color_rgba.h"
#include "agg_math_stroke.h"
#include "path_converters.h"

#include "py_adaptors.h"

struct ClipPath
{
    py::PathIterator path;
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
        dashes.push_back(std::make_pair(length, skip));
    }
    size_t size() const
    {
        return dashes.size();
    }

    template <class T>
    void dash_to_stroke(T &stroke, double dpi, bool isaa)
    {
        for (dash_t::const_iterator i = dashes.begin(); i != dashes.end(); ++i) {
            double val0 = i->first;
            double val1 = i->second;
            val0 = val0 * dpi / 72.0;
            val1 = val1 * dpi / 72.0;
            if (!isaa) {
                val0 = (int)val0 + 0.5;
                val1 = (int)val1 + 0.5;
            }
            stroke.add_dash(val0, val1);
        }
        stroke.dash_start(get_dash_offset() * dpi / 72.0);
    }
};

typedef std::vector<Dashes> DashesVector;

enum e_offset_position {
    OFFSET_POSITION_FIGURE,
    OFFSET_POSITION_DATA
};

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

    py::PathIterator hatchpath;
    agg::rgba hatch_color;
    double hatch_linewidth;

    SketchParams sketch;

    bool has_hatchpath()
    {
        return hatchpath.total_vertices();
    }

  private:
    // prevent copying
    GCAgg(const GCAgg &);
    GCAgg &operator=(const GCAgg &);
};

#endif
