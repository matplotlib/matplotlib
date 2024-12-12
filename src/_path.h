/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_PATH_H
#define MPL_PATH_H

#include <limits>
#include <math.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

#include "agg_conv_contour.h"
#include "agg_conv_curve.h"
#include "agg_conv_stroke.h"
#include "agg_conv_transform.h"
#include "agg_trans_affine.h"

#include "path_converters.h"
#include "_backend_agg_basic_types.h"

const size_t NUM_VERTICES[] = { 1, 1, 1, 2, 3 };

struct XY
{
    double x;
    double y;

    XY(double x_, double y_) : x(x_), y(y_)
    {
    }

    bool operator==(const XY& o)
    {
        return (x == o.x && y == o.y);
    }

    bool operator!=(const XY& o)
    {
        return (x != o.x || y != o.y);
    }
};

typedef std::vector<XY> Polygon;

void _finalize_polygon(std::vector<Polygon> &result, int closed_only)
{
    if (result.size() == 0) {
        return;
    }

    Polygon &polygon = result.back();

    /* Clean up the last polygon in the result.  */
    if (polygon.size() == 0) {
        result.pop_back();
    } else if (closed_only) {
        if (polygon.size() < 3) {
            result.pop_back();
        } else if (polygon.front() != polygon.back()) {
            polygon.push_back(polygon.front());
        }
    }
}

//
// The following function was found in the Agg 2.3 examples (interactive_polygon.cpp).
// It has been generalized to work on (possibly curved) polylines, rather than
// just polygons.  The original comments have been kept intact.
//  -- Michael Droettboom 2007-10-02
//
//======= Crossings Multiply algorithm of InsideTest ========================
//
// By Eric Haines, 3D/Eye Inc, erich@eye.com
//
// This version is usually somewhat faster than the original published in
// Graphics Gems IV; by turning the division for testing the X axis crossing
// into a tricky multiplication test this part of the test became faster,
// which had the additional effect of making the test for "both to left or
// both to right" a bit slower for triangles than simply computing the
// intersection each time.  The main increase is in triangle testing speed,
// which was about 15% faster; all other polygon complexities were pretty much
// the same as before.  On machines where division is very expensive (not the
// case on the HP 9000 series on which I tested) this test should be much
// faster overall than the old code.  Your mileage may (in fact, will) vary,
// depending on the machine and the test data, but in general I believe this
// code is both shorter and faster.  This test was inspired by unpublished
// Graphics Gems submitted by Joseph Samosky and Mark Haigh-Hutchinson.
// Related work by Samosky is in:
//
// Samosky, Joseph, "SectionView: A system for interactively specifying and
// visualizing sections through three-dimensional medical image data",
// M.S. Thesis, Department of Electrical Engineering and Computer Science,
// Massachusetts Institute of Technology, 1993.
//
// Shoot a test ray along +X axis.  The strategy is to compare vertex Y values
// to the testing point's Y and quickly discard edges which are entirely to one
// side of the test ray.  Note that CONVEX and WINDING code can be added as
// for the CrossingsTest() code; it is left out here for clarity.
//
// Input 2D polygon _pgon_ with _numverts_ number of vertices and test point
// _point_, returns 1 if inside, 0 if outside.
template <class PathIterator, class PointArray, class ResultArray>
void point_in_path_impl(PointArray &points, PathIterator &path, ResultArray &inside_flag)
{
    uint8_t yflag1;
    double vtx0, vty0, vtx1, vty1;
    double tx, ty;
    double sx, sy;
    double x, y;
    size_t i;
    bool all_done;

    size_t n = safe_first_shape(points);

    std::vector<uint8_t> yflag0(n);
    std::vector<uint8_t> subpath_flag(n);

    path.rewind(0);

    for (i = 0; i < n; ++i) {
        inside_flag[i] = 0;
    }

    unsigned code = 0;
    do {
        if (code != agg::path_cmd_move_to) {
            code = path.vertex(&x, &y);
            if (code == agg::path_cmd_stop ||
                (code & agg::path_cmd_end_poly) == agg::path_cmd_end_poly) {
                continue;
            }
        }

        sx = vtx0 = vtx1 = x;
        sy = vty0 = vty1 = y;

        for (i = 0; i < n; ++i) {
            ty = points(i, 1);

            if (std::isfinite(ty)) {
                // get test bit for above/below X axis
                yflag0[i] = (vty0 >= ty);

                subpath_flag[i] = 0;
            }
        }

        do {
            code = path.vertex(&x, &y);

            // The following cases denote the beginning on a new subpath
            if (code == agg::path_cmd_stop ||
                (code & agg::path_cmd_end_poly) == agg::path_cmd_end_poly) {
                x = sx;
                y = sy;
            } else if (code == agg::path_cmd_move_to) {
                break;
            }

            for (i = 0; i < n; ++i) {
                tx = points(i, 0);
                ty = points(i, 1);

                if (!(std::isfinite(tx) && std::isfinite(ty))) {
                    continue;
                }

                yflag1 = (vty1 >= ty);
                // Check if endpoints straddle (are on opposite sides) of
                // X axis (i.e. the Y's differ); if so, +X ray could
                // intersect this edge.  The old test also checked whether
                // the endpoints are both to the right or to the left of
                // the test point.  However, given the faster intersection
                // point computation used below, this test was found to be
                // a break-even proposition for most polygons and a loser
                // for triangles (where 50% or more of the edges which
                // survive this test will cross quadrants and so have to
                // have the X intersection computed anyway).  I credit
                // Joseph Samosky with inspiring me to try dropping the
                // "both left or both right" part of my code.
                if (yflag0[i] != yflag1) {
                    // Check intersection of pgon segment with +X ray.
                    // Note if >= point's X; if so, the ray hits it.  The
                    // division operation is avoided for the ">=" test by
                    // checking the sign of the first vertex wrto the test
                    // point; idea inspired by Joseph Samosky's and Mark
                    // Haigh-Hutchinson's different polygon inclusion
                    // tests.
                    if (((vty1 - ty) * (vtx0 - vtx1) >= (vtx1 - tx) * (vty0 - vty1)) == yflag1) {
                        subpath_flag[i] ^= 1;
                    }
                }

                // Move to the next pair of vertices, retaining info as
                // possible.
                yflag0[i] = yflag1;
            }

            vtx0 = vtx1;
            vty0 = vty1;

            vtx1 = x;
            vty1 = y;
        } while (code != agg::path_cmd_stop &&
                 (code & agg::path_cmd_end_poly) != agg::path_cmd_end_poly);

        all_done = true;
        for (i = 0; i < n; ++i) {
            tx = points(i, 0);
            ty = points(i, 1);

            if (!(std::isfinite(tx) && std::isfinite(ty))) {
                continue;
            }

            yflag1 = (vty1 >= ty);
            if (yflag0[i] != yflag1) {
                if (((vty1 - ty) * (vtx0 - vtx1) >= (vtx1 - tx) * (vty0 - vty1)) == yflag1) {
                    subpath_flag[i] = subpath_flag[i] ^ true;
                }
            }
            inside_flag[i] |= subpath_flag[i];
            if (inside_flag[i] == 0) {
                all_done = false;
            }
        }

        if (all_done) {
            break;
        }
    } while (code != agg::path_cmd_stop);
}

template <class PathIterator, class PointArray, class ResultArray>
inline void points_in_path(PointArray &points,
                           const double r,
                           PathIterator &path,
                           agg::trans_affine &trans,
                           ResultArray &result)
{
    typedef agg::conv_transform<PathIterator> transformed_path_t;
    typedef PathNanRemover<transformed_path_t> no_nans_t;
    typedef agg::conv_curve<no_nans_t> curve_t;
    typedef agg::conv_contour<curve_t> contour_t;

    for (auto i = 0; i < safe_first_shape(points); ++i) {
        result[i] = false;
    }

    if (path.total_vertices() < 3) {
        return;
    }

    transformed_path_t trans_path(path, trans);
    no_nans_t no_nans_path(trans_path, true, path.has_codes());
    curve_t curved_path(no_nans_path);
    if (r != 0.0) {
        contour_t contoured_path(curved_path);
        contoured_path.width(r);
        point_in_path_impl(points, contoured_path, result);
    } else {
        point_in_path_impl(points, curved_path, result);
    }
}

template <class PathIterator>
inline bool point_in_path(
    double x, double y, const double r, PathIterator &path, agg::trans_affine &trans)
{
    py::ssize_t shape[] = {1, 2};
    py::array_t<double> points_arr(shape);
    *points_arr.mutable_data(0, 0) = x;
    *points_arr.mutable_data(0, 1) = y;
    auto points = points_arr.mutable_unchecked<2>();

    int result[1];
    result[0] = 0;

    points_in_path(points, r, path, trans, result);

    return result[0] != 0;
}

template <class PathIterator>
inline bool point_on_path(
    double x, double y, const double r, PathIterator &path, agg::trans_affine &trans)
{
    typedef agg::conv_transform<PathIterator> transformed_path_t;
    typedef PathNanRemover<transformed_path_t> no_nans_t;
    typedef agg::conv_curve<no_nans_t> curve_t;
    typedef agg::conv_stroke<curve_t> stroke_t;

    py::ssize_t shape[] = {1, 2};
    py::array_t<double> points_arr(shape);
    *points_arr.mutable_data(0, 0) = x;
    *points_arr.mutable_data(0, 1) = y;
    auto points = points_arr.mutable_unchecked<2>();

    int result[1];
    result[0] = 0;

    transformed_path_t trans_path(path, trans);
    no_nans_t nan_removed_path(trans_path, true, path.has_codes());
    curve_t curved_path(nan_removed_path);
    stroke_t stroked_path(curved_path);
    stroked_path.width(r * 2.0);
    point_in_path_impl(points, stroked_path, result);
    return result[0] != 0;
}

struct extent_limits
{
    double x0;
    double y0;
    double x1;
    double y1;
    double xm;
    double ym;
};

void reset_limits(extent_limits &e)
{
    e.x0 = std::numeric_limits<double>::infinity();
    e.y0 = std::numeric_limits<double>::infinity();
    e.x1 = -std::numeric_limits<double>::infinity();
    e.y1 = -std::numeric_limits<double>::infinity();
    /* xm and ym are the minimum positive values in the data, used
       by log scaling */
    e.xm = std::numeric_limits<double>::infinity();
    e.ym = std::numeric_limits<double>::infinity();
}

inline void update_limits(double x, double y, extent_limits &e)
{
    if (x < e.x0)
        e.x0 = x;
    if (y < e.y0)
        e.y0 = y;
    if (x > e.x1)
        e.x1 = x;
    if (y > e.y1)
        e.y1 = y;
    /* xm and ym are the minimum positive values in the data, used
       by log scaling */
    if (x > 0.0 && x < e.xm)
        e.xm = x;
    if (y > 0.0 && y < e.ym)
        e.ym = y;
}

template <class PathIterator>
void update_path_extents(PathIterator &path, agg::trans_affine &trans, extent_limits &extents)
{
    typedef agg::conv_transform<PathIterator> transformed_path_t;
    typedef PathNanRemover<transformed_path_t> nan_removed_t;
    double x, y;
    unsigned code;

    transformed_path_t tpath(path, trans);
    nan_removed_t nan_removed(tpath, true, path.has_codes());

    nan_removed.rewind(0);

    while ((code = nan_removed.vertex(&x, &y)) != agg::path_cmd_stop) {
        if ((code & agg::path_cmd_end_poly) == agg::path_cmd_end_poly) {
            continue;
        }
        update_limits(x, y, extents);
    }
}

template <class PathGenerator, class TransformArray, class OffsetArray>
void get_path_collection_extents(agg::trans_affine &master_transform,
                                 PathGenerator &paths,
                                 TransformArray &transforms,
                                 OffsetArray &offsets,
                                 agg::trans_affine &offset_trans,
                                 extent_limits &extent)
{
    if (offsets.size() != 0 && offsets.shape(1) != 2) {
        throw std::runtime_error("Offsets array must have shape (N, 2)");
    }

    auto Npaths = paths.size();
    auto Noffsets = safe_first_shape(offsets);
    auto N = std::max(Npaths, Noffsets);
    auto Ntransforms = std::min(safe_first_shape(transforms), N);

    agg::trans_affine trans;

    reset_limits(extent);

    for (auto i = 0; i < N; ++i) {
        typename PathGenerator::path_iterator path(paths(i % Npaths));
        if (Ntransforms) {
            py::ssize_t ti = i % Ntransforms;
            trans = agg::trans_affine(transforms(ti, 0, 0),
                                      transforms(ti, 1, 0),
                                      transforms(ti, 0, 1),
                                      transforms(ti, 1, 1),
                                      transforms(ti, 0, 2),
                                      transforms(ti, 1, 2));
        } else {
            trans = master_transform;
        }

        if (Noffsets) {
            double xo = offsets(i % Noffsets, 0);
            double yo = offsets(i % Noffsets, 1);
            offset_trans.transform(&xo, &yo);
            trans *= agg::trans_affine_translation(xo, yo);
        }

        update_path_extents(path, trans, extent);
    }
}

template <class PathGenerator, class TransformArray, class OffsetArray>
void point_in_path_collection(double x,
                              double y,
                              double radius,
                              agg::trans_affine &master_transform,
                              PathGenerator &paths,
                              TransformArray &transforms,
                              OffsetArray &offsets,
                              agg::trans_affine &offset_trans,
                              bool filled,
                              std::vector<int> &result)
{
    auto Npaths = paths.size();

    if (Npaths == 0) {
        return;
    }

    auto Noffsets = safe_first_shape(offsets);
    auto N = std::max(Npaths, Noffsets);
    auto Ntransforms = std::min(safe_first_shape(transforms), N);

    agg::trans_affine trans;

    for (auto i = 0; i < N; ++i) {
        typename PathGenerator::path_iterator path = paths(i % Npaths);

        if (Ntransforms) {
            auto ti = i % Ntransforms;
            trans = agg::trans_affine(transforms(ti, 0, 0),
                                      transforms(ti, 1, 0),
                                      transforms(ti, 0, 1),
                                      transforms(ti, 1, 1),
                                      transforms(ti, 0, 2),
                                      transforms(ti, 1, 2));
            trans *= master_transform;
        } else {
            trans = master_transform;
        }

        if (Noffsets) {
            double xo = offsets(i % Noffsets, 0);
            double yo = offsets(i % Noffsets, 1);
            offset_trans.transform(&xo, &yo);
            trans *= agg::trans_affine_translation(xo, yo);
        }

        if (filled) {
            if (point_in_path(x, y, radius, path, trans)) {
                result.push_back(i);
            }
        } else {
            if (point_on_path(x, y, radius, path, trans)) {
                result.push_back(i);
            }
        }
    }
}

template <class PathIterator1, class PathIterator2>
bool path_in_path(PathIterator1 &a,
                  agg::trans_affine &atrans,
                  PathIterator2 &b,
                  agg::trans_affine &btrans)
{
    typedef agg::conv_transform<PathIterator2> transformed_path_t;
    typedef PathNanRemover<transformed_path_t> no_nans_t;
    typedef agg::conv_curve<no_nans_t> curve_t;

    if (a.total_vertices() < 3) {
        return false;
    }

    transformed_path_t b_path_trans(b, btrans);
    no_nans_t b_no_nans(b_path_trans, true, b.has_codes());
    curve_t b_curved(b_no_nans);

    double x, y;
    b_curved.rewind(0);
    while (b_curved.vertex(&x, &y) != agg::path_cmd_stop) {
        if (!point_in_path(x, y, 0.0, a, atrans)) {
            return false;
        }
    }

    return true;
}

/** The clip_path_to_rect code here is a clean-room implementation of
    the Sutherland-Hodgman clipping algorithm described here:

  https://en.wikipedia.org/wiki/Sutherland-Hodgman_clipping_algorithm
*/

namespace clip_to_rect_filters
{
/* There are four different passes needed to create/remove
   vertices (one for each side of the rectangle).  The differences
   between those passes are encapsulated in these functor classes.
*/
struct bisectx
{
    double m_x;

    bisectx(double x) : m_x(x)
    {
    }

    inline void bisect(double sx, double sy, double px, double py, double *bx, double *by) const
    {
        *bx = m_x;
        double dx = px - sx;
        double dy = py - sy;
        *by = sy + dy * ((m_x - sx) / dx);
    }
};

struct xlt : public bisectx
{
    xlt(double x) : bisectx(x)
    {
    }

    inline bool is_inside(double x, double y) const
    {
        return x <= m_x;
    }
};

struct xgt : public bisectx
{
    xgt(double x) : bisectx(x)
    {
    }

    inline bool is_inside(double x, double y) const
    {
        return x >= m_x;
    }
};

struct bisecty
{
    double m_y;

    bisecty(double y) : m_y(y)
    {
    }

    inline void bisect(double sx, double sy, double px, double py, double *bx, double *by) const
    {
        *by = m_y;
        double dx = px - sx;
        double dy = py - sy;
        *bx = sx + dx * ((m_y - sy) / dy);
    }
};

struct ylt : public bisecty
{
    ylt(double y) : bisecty(y)
    {
    }

    inline bool is_inside(double x, double y) const
    {
        return y <= m_y;
    }
};

struct ygt : public bisecty
{
    ygt(double y) : bisecty(y)
    {
    }

    inline bool is_inside(double x, double y) const
    {
        return y >= m_y;
    }
};
}

template <class Filter>
inline void clip_to_rect_one_step(const Polygon &polygon, Polygon &result, const Filter &filter)
{
    double sx, sy, px, py, bx, by;
    bool sinside, pinside;
    result.clear();

    if (polygon.size() == 0) {
        return;
    }

    sx = polygon.back().x;
    sy = polygon.back().y;
    for (Polygon::const_iterator i = polygon.begin(); i != polygon.end(); ++i) {
        px = i->x;
        py = i->y;

        sinside = filter.is_inside(sx, sy);
        pinside = filter.is_inside(px, py);

        if (sinside ^ pinside) {
            filter.bisect(sx, sy, px, py, &bx, &by);
            result.push_back(XY(bx, by));
        }

        if (pinside) {
            result.push_back(XY(px, py));
        }

        sx = px;
        sy = py;
    }
}

template <class PathIterator>
void
clip_path_to_rect(PathIterator &path, agg::rect_d &rect, bool inside, std::vector<Polygon> &results)
{
    double xmin, ymin, xmax, ymax;
    if (rect.x1 < rect.x2) {
        xmin = rect.x1;
        xmax = rect.x2;
    } else {
        xmin = rect.x2;
        xmax = rect.x1;
    }

    if (rect.y1 < rect.y2) {
        ymin = rect.y1;
        ymax = rect.y2;
    } else {
        ymin = rect.y2;
        ymax = rect.y1;
    }

    if (!inside) {
        std::swap(xmin, xmax);
        std::swap(ymin, ymax);
    }

    typedef agg::conv_curve<PathIterator> curve_t;
    curve_t curve(path);

    Polygon polygon1, polygon2;
    double x = 0, y = 0;
    unsigned code = 0;
    curve.rewind(0);

    do {
        // Grab the next subpath and store it in polygon1
        polygon1.clear();
        do {
            if (code == agg::path_cmd_move_to) {
                polygon1.push_back(XY(x, y));
            }

            code = curve.vertex(&x, &y);

            if (code == agg::path_cmd_stop) {
                break;
            }

            if (code != agg::path_cmd_move_to) {
                polygon1.push_back(XY(x, y));
            }
        } while ((code & agg::path_cmd_end_poly) != agg::path_cmd_end_poly);

        // The result of each step is fed into the next (note the
        // swapping of polygon1 and polygon2 at each step).
        clip_to_rect_one_step(polygon1, polygon2, clip_to_rect_filters::xlt(xmax));
        clip_to_rect_one_step(polygon2, polygon1, clip_to_rect_filters::xgt(xmin));
        clip_to_rect_one_step(polygon1, polygon2, clip_to_rect_filters::ylt(ymax));
        clip_to_rect_one_step(polygon2, polygon1, clip_to_rect_filters::ygt(ymin));

        // Empty polygons aren't very useful, so skip them
        if (polygon1.size()) {
            _finalize_polygon(results, 1);
            results.push_back(polygon1);
        }
    } while (code != agg::path_cmd_stop);

    _finalize_polygon(results, 1);
}

template <class VerticesArray, class ResultArray>
void affine_transform_2d(VerticesArray &vertices, agg::trans_affine &trans, ResultArray &result)
{
    if (vertices.size() != 0 && vertices.shape(1) != 2) {
        throw std::runtime_error("Invalid vertices array.");
    }

    size_t n = vertices.shape(0);
    double x;
    double y;
    double t0;
    double t1;
    double t;

    for (size_t i = 0; i < n; ++i) {
        x = vertices(i, 0);
        y = vertices(i, 1);

        t0 = trans.sx * x;
        t1 = trans.shx * y;
        t = t0 + t1 + trans.tx;
        result(i, 0) = t;

        t0 = trans.shy * x;
        t1 = trans.sy * y;
        t = t0 + t1 + trans.ty;
        result(i, 1) = t;
    }
}

template <class VerticesArray, class ResultArray>
void affine_transform_1d(VerticesArray &vertices, agg::trans_affine &trans, ResultArray &result)
{
    if (vertices.shape(0) != 2) {
        throw std::runtime_error("Invalid vertices array.");
    }

    double x;
    double y;
    double t0;
    double t1;
    double t;

    x = vertices(0);
    y = vertices(1);

    t0 = trans.sx * x;
    t1 = trans.shx * y;
    t = t0 + t1 + trans.tx;
    result(0) = t;

    t0 = trans.shy * x;
    t1 = trans.sy * y;
    t = t0 + t1 + trans.ty;
    result(1) = t;
}

template <class BBoxArray>
int count_bboxes_overlapping_bbox(agg::rect_d &a, BBoxArray &bboxes)
{
    agg::rect_d b;
    int count = 0;

    if (a.x2 < a.x1) {
        std::swap(a.x1, a.x2);
    }
    if (a.y2 < a.y1) {
        std::swap(a.y1, a.y2);
    }

    size_t num_bboxes = safe_first_shape(bboxes);
    for (size_t i = 0; i < num_bboxes; ++i) {
        b = agg::rect_d(bboxes(i, 0, 0), bboxes(i, 0, 1), bboxes(i, 1, 0), bboxes(i, 1, 1));

        if (b.x2 < b.x1) {
            std::swap(b.x1, b.x2);
        }
        if (b.y2 < b.y1) {
            std::swap(b.y1, b.y2);
        }
        if (!((b.x2 <= a.x1) || (b.y2 <= a.y1) || (b.x1 >= a.x2) || (b.y1 >= a.y2))) {
            ++count;
        }
    }

    return count;
}


inline bool isclose(double a, double b)
{
    // relative and absolute tolerance values are chosen empirically
    // it looks the atol value matters here because of round-off errors
    const double rtol = 1e-10;
    const double atol = 1e-13;

    // as per python's math.isclose
    return fabs(a-b) <= fmax(rtol * fmax(fabs(a), fabs(b)), atol);
}


inline bool segments_intersect(const double &x1,
                               const double &y1,
                               const double &x2,
                               const double &y2,
                               const double &x3,
                               const double &y3,
                               const double &x4,
                               const double &y4)
{
    // determinant
    double den = ((y4 - y3) * (x2 - x1)) - ((x4 - x3) * (y2 - y1));

    // If den == 0 we have two possibilities:
    if (isclose(den, 0.0)) {
        double t_area = (x2*y3 - x3*y2) - x1*(y3 - y2) + y1*(x3 - x2);
        // 1 - If the area of the triangle made by the 3 first points (2 from the first segment
        // plus one from the second) is zero, they are collinear
        if (isclose(t_area, 0.0)) {
            if (x1 == x2 && x2 == x3) { // segments have infinite slope (vertical lines)
                                        // and lie on the same line
                return (fmin(y1, y2) <= fmin(y3, y4) && fmin(y3, y4) <= fmax(y1, y2)) ||
                    (fmin(y3, y4) <= fmin(y1, y2) && fmin(y1, y2) <= fmax(y3, y4));
            }
            else {
                return (fmin(x1, x2) <= fmin(x3, x4) && fmin(x3, x4) <= fmax(x1, x2)) ||
                        (fmin(x3, x4) <= fmin(x1, x2) && fmin(x1, x2) <= fmax(x3, x4));
            }
        }
        // 2 - If t_area is not zero, the segments are parallel, but not collinear
        else {
            return false;
        }
    }

    const double n1 = ((x4 - x3) * (y1 - y3)) - ((y4 - y3) * (x1 - x3));
    const double n2 = ((x2 - x1) * (y1 - y3)) - ((y2 - y1) * (x1 - x3));

    const double u1 = n1 / den;
    const double u2 = n2 / den;

    return ((u1 > 0.0 || isclose(u1, 0.0)) &&
            (u1 < 1.0 || isclose(u1, 1.0)) &&
            (u2 > 0.0 || isclose(u2, 0.0)) &&
            (u2 < 1.0 || isclose(u2, 1.0)));
}

template <class PathIterator1, class PathIterator2>
bool path_intersects_path(PathIterator1 &p1, PathIterator2 &p2)
{
    typedef PathNanRemover<mpl::PathIterator> no_nans_t;
    typedef agg::conv_curve<no_nans_t> curve_t;

    if (p1.total_vertices() < 2 || p2.total_vertices() < 2) {
        return false;
    }

    no_nans_t n1(p1, true, p1.has_codes());
    no_nans_t n2(p2, true, p2.has_codes());

    curve_t c1(n1);
    curve_t c2(n2);

    double x11, y11, x12, y12;
    double x21, y21, x22, y22;

    c1.vertex(&x11, &y11);
    while (c1.vertex(&x12, &y12) != agg::path_cmd_stop) {
        // if the segment in path 1 is (almost) 0 length, skip to next vertex
        if ((isclose((x11 - x12) * (x11 - x12) + (y11 - y12) * (y11 - y12), 0))){
            continue;
        }
        c2.rewind(0);
        c2.vertex(&x21, &y21);

        while (c2.vertex(&x22, &y22) != agg::path_cmd_stop) {
            // if the segment in path 2 is (almost) 0 length, skip to next vertex
            if ((isclose((x21 - x22) * (x21 - x22) + (y21 - y22) * (y21 - y22), 0))){
                continue;
            }

            if (segments_intersect(x11, y11, x12, y12, x21, y21, x22, y22)) {
                return true;
            }
            x21 = x22;
            y21 = y22;
        }
        x11 = x12;
        y11 = y12;
    }

    return false;
}

// returns whether the segment from (x1,y1) to (x2,y2)
// intersects the rectangle centered at (cx,cy) with size (w,h)
// see doc/segment_intersects_rectangle.svg for a more detailed explanation
inline bool segment_intersects_rectangle(double x1, double y1,
                                         double x2, double y2,
                                         double cx, double cy,
                                         double w, double h)
{
    return fabs(x1 + x2 - 2.0 * cx) < fabs(x1 - x2) + w &&
           fabs(y1 + y2 - 2.0 * cy) < fabs(y1 - y2) + h &&
           2.0 * fabs((x1 - cx) * (y1 - y2) - (y1 - cy) * (x1 - x2)) <
               w * fabs(y1 - y2) + h * fabs(x1 - x2);
}

template <class PathIterator>
bool path_intersects_rectangle(PathIterator &path,
                               double rect_x1, double rect_y1,
                               double rect_x2, double rect_y2,
                               bool filled)
{
    typedef PathNanRemover<mpl::PathIterator> no_nans_t;
    typedef agg::conv_curve<no_nans_t> curve_t;

    if (path.total_vertices() == 0) {
        return false;
    }

    no_nans_t no_nans(path, true, path.has_codes());
    curve_t curve(no_nans);

    double cx = (rect_x1 + rect_x2) * 0.5, cy = (rect_y1 + rect_y2) * 0.5;
    double w = fabs(rect_x1 - rect_x2), h = fabs(rect_y1 - rect_y2);

    double x1, y1, x2, y2;

    curve.vertex(&x1, &y1);
    if (2.0 * fabs(x1 - cx) <= w && 2.0 * fabs(y1 - cy) <= h) {
        return true;
    }

    while (curve.vertex(&x2, &y2) != agg::path_cmd_stop) {
        if (segment_intersects_rectangle(x1, y1, x2, y2, cx, cy, w, h)) {
            return true;
        }
        x1 = x2;
        y1 = y2;
    }

    if (filled) {
        agg::trans_affine trans;
        if (point_in_path(cx, cy, 0.0, path, trans)) {
            return true;
        }
    }

    return false;
}

template <class PathIterator>
void convert_path_to_polygons(PathIterator &path,
                              agg::trans_affine &trans,
                              double width,
                              double height,
                              int closed_only,
                              std::vector<Polygon> &result)
{
    typedef agg::conv_transform<mpl::PathIterator> transformed_path_t;
    typedef PathNanRemover<transformed_path_t> nan_removal_t;
    typedef PathClipper<nan_removal_t> clipped_t;
    typedef PathSimplifier<clipped_t> simplify_t;
    typedef agg::conv_curve<simplify_t> curve_t;

    bool do_clip = width != 0.0 && height != 0.0;
    bool simplify = path.should_simplify();

    transformed_path_t tpath(path, trans);
    nan_removal_t nan_removed(tpath, true, path.has_codes());
    clipped_t clipped(nan_removed, do_clip, width, height);
    simplify_t simplified(clipped, simplify, path.simplify_threshold());
    curve_t curve(simplified);

    result.push_back(Polygon());
    Polygon *polygon = &result.back();
    double x, y;
    unsigned code;

    while ((code = curve.vertex(&x, &y)) != agg::path_cmd_stop) {
        if ((code & agg::path_cmd_end_poly) == agg::path_cmd_end_poly) {
            _finalize_polygon(result, 1);
            result.push_back(Polygon());
            polygon = &result.back();
        } else {
            if (code == agg::path_cmd_move_to) {
                _finalize_polygon(result, closed_only);
                result.push_back(Polygon());
                polygon = &result.back();
            }
            polygon->push_back(XY(x, y));
        }
    }

    _finalize_polygon(result, closed_only);
}

template <class VertexSource>
void
__cleanup_path(VertexSource &source, std::vector<double> &vertices, std::vector<uint8_t> &codes)
{
    unsigned code;
    double x, y;
    do {
        code = source.vertex(&x, &y);
        vertices.push_back(x);
        vertices.push_back(y);
        codes.push_back(static_cast<uint8_t>(code));
    } while (code != agg::path_cmd_stop);
}

template <class PathIterator>
void cleanup_path(PathIterator &path,
                  agg::trans_affine &trans,
                  bool remove_nans,
                  bool do_clip,
                  const agg::rect_base<double> &rect,
                  e_snap_mode snap_mode,
                  double stroke_width,
                  bool do_simplify,
                  bool return_curves,
                  SketchParams sketch_params,
                  std::vector<double> &vertices,
                  std::vector<unsigned char> &codes)
{
    typedef agg::conv_transform<mpl::PathIterator> transformed_path_t;
    typedef PathNanRemover<transformed_path_t> nan_removal_t;
    typedef PathClipper<nan_removal_t> clipped_t;
    typedef PathSnapper<clipped_t> snapped_t;
    typedef PathSimplifier<snapped_t> simplify_t;
    typedef agg::conv_curve<simplify_t> curve_t;
    typedef Sketch<curve_t> sketch_t;

    transformed_path_t tpath(path, trans);
    nan_removal_t nan_removed(tpath, remove_nans, path.has_codes());
    clipped_t clipped(nan_removed, do_clip, rect);
    snapped_t snapped(clipped, snap_mode, path.total_vertices(), stroke_width);
    simplify_t simplified(snapped, do_simplify, path.simplify_threshold());

    vertices.reserve(path.total_vertices() * 2);
    codes.reserve(path.total_vertices());

    if (return_curves && sketch_params.scale == 0.0) {
        __cleanup_path(simplified, vertices, codes);
    } else {
        curve_t curve(simplified);
        sketch_t sketch(curve, sketch_params.scale, sketch_params.length, sketch_params.randomness);
        __cleanup_path(sketch, vertices, codes);
    }
}

void quad2cubic(double x0, double y0,
                double x1, double y1,
                double x2, double y2,
                double *outx, double *outy)
{

    outx[0] = x0 + 2./3. * (x1 - x0);
    outy[0] = y0 + 2./3. * (y1 - y0);
    outx[1] = outx[0] + 1./3. * (x2 - x0);
    outy[1] = outy[0] + 1./3. * (y2 - y0);
    outx[2] = x2;
    outy[2] = y2;
}


void __add_number(double val, char format_code, int precision,
                  std::string& buffer)
{
    if (precision == -1) {
        // Special-case for compat with old ttconv code, which *truncated*
        // values with a cast to int instead of rounding them as printf
        // would do.  The only point where non-integer values arise is from
        // quad2cubic conversion (as we already perform a first truncation
        // on Python's side), which can introduce additional floating point
        // error (by adding 2/3 delta-x and then 1/3 delta-x), so compensate by
        // first rounding to the closest 1/3 and then truncating.
        char str[255];
        PyOS_snprintf(str, 255, "%d", (int)(round(val * 3)) / 3);
        buffer += str;
    } else {
        char *str = PyOS_double_to_string(
          val, format_code, precision, Py_DTSF_ADD_DOT_0, NULL);
        // Delete trailing zeros and decimal point
        char *c = str + strlen(str) - 1;  // Start at last character.
        // Rewind through all the zeros and, if present, the trailing decimal
        // point.  Py_DTSF_ADD_DOT_0 ensures we won't go past the start of str.
        while (*c == '0') {
            --c;
        }
        if (*c == '.') {
            --c;
        }
        try {
            buffer.append(str, c + 1);
        } catch (std::bad_alloc& e) {
            PyMem_Free(str);
            throw e;
        }
        PyMem_Free(str);
    }
}


template <class PathIterator>
bool __convert_to_string(PathIterator &path,
                         int precision,
                         char **codes,
                         bool postfix,
                         std::string& buffer)
{
    const char format_code = 'f';

    double x[3];
    double y[3];
    double last_x = 0.0;
    double last_y = 0.0;

    unsigned code;

    while ((code = path.vertex(&x[0], &y[0])) != agg::path_cmd_stop) {
        if (code == CLOSEPOLY) {
            buffer += codes[4];
        } else if (code < 5) {
            size_t size = NUM_VERTICES[code];

            for (size_t i = 1; i < size; ++i) {
                unsigned subcode = path.vertex(&x[i], &y[i]);
                if (subcode != code) {
                    return false;
                }
            }

            /* For formats that don't support quad curves, convert to
               cubic curves */
            if (code == CURVE3 && codes[code - 1][0] == '\0') {
                quad2cubic(last_x, last_y, x[0], y[0], x[1], y[1], x, y);
                code++;
                size = 3;
            }

            if (!postfix) {
                buffer += codes[code - 1];
                buffer += ' ';
            }

            for (size_t i = 0; i < size; ++i) {
                __add_number(x[i], format_code, precision, buffer);
                buffer += ' ';
                __add_number(y[i], format_code, precision, buffer);
                buffer += ' ';
            }

            if (postfix) {
                buffer += codes[code - 1];
            }

            last_x = x[size - 1];
            last_y = y[size - 1];
        } else {
            // Unknown code value
            return false;
        }

        buffer += '\n';
    }

    return true;
}

template <class PathIterator>
bool convert_to_string(PathIterator &path,
                       agg::trans_affine &trans,
                       agg::rect_d &clip_rect,
                       bool simplify,
                       SketchParams sketch_params,
                       int precision,
                       char **codes,
                       bool postfix,
                       std::string& buffer)
{
    size_t buffersize;
    typedef agg::conv_transform<mpl::PathIterator> transformed_path_t;
    typedef PathNanRemover<transformed_path_t> nan_removal_t;
    typedef PathClipper<nan_removal_t> clipped_t;
    typedef PathSimplifier<clipped_t> simplify_t;
    typedef agg::conv_curve<simplify_t> curve_t;
    typedef Sketch<curve_t> sketch_t;

    bool do_clip = (clip_rect.x1 < clip_rect.x2 && clip_rect.y1 < clip_rect.y2);

    transformed_path_t tpath(path, trans);
    nan_removal_t nan_removed(tpath, true, path.has_codes());
    clipped_t clipped(nan_removed, do_clip, clip_rect);
    simplify_t simplified(clipped, simplify, path.simplify_threshold());

    buffersize = (size_t) path.total_vertices() * (precision + 5) * 4;
    if (buffersize == 0) {
        return true;
    }

    if (sketch_params.scale != 0.0) {
        buffersize *= 10;
    }

    buffer.reserve(buffersize);

    if (sketch_params.scale == 0.0) {
        return __convert_to_string(simplified, precision, codes, postfix, buffer);
    } else {
        curve_t curve(simplified);
        sketch_t sketch(curve, sketch_params.scale, sketch_params.length, sketch_params.randomness);
        return __convert_to_string(sketch, precision, codes, postfix, buffer);
    }

}

template<class T>
bool is_sorted_and_has_non_nan(py::array_t<T> array)
{
    auto size = array.shape(0);
    using limits = std::numeric_limits<T>;
    T last = limits::has_infinity ? -limits::infinity() : limits::min();
    bool found_non_nan = false;

    for (auto i = 0; i < size; ++i) {
        T current = *array.data(i);
        // The following tests !isnan(current), but also works for integral
        // types.  (The isnan(IntegralType) overload is absent on MSVC.)
        if (current == current) {
            found_non_nan = true;
            if (current < last) {
                return false;
            }
            last = current;
        }
    }
    return found_non_nan;
};


#endif
