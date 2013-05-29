/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef __PATH_CONVERTERS_H__
#define __PATH_CONVERTERS_H__

#include <stdlib.h>
#include "CXX/Objects.hxx"
#include "numpy/arrayobject.h"
#include "agg_path_storage.h"
#include "agg_clip_liang_barsky.h"
#include "MPL_isnan.h"
#include "mplutils.h"
#include "agg_conv_segmentator.h"

/*
 This file contains a number of vertex converters that modify
 paths. They all work as iterators, where the output is generated
 on-the-fly, and don't require a copy of the full data.

 Each class represents a discrete step in a "path-cleansing" pipeline.
 They are currently applied in the following order in the Agg backend:

   1. Affine transformation (implemented in Agg, not here)

   2. PathNanRemover: skips over segments containing non-finite numbers
      by inserting MOVETO commands

   3. PathClipper: Clips line segments to a given rectangle.  This is
      helpful for data reduction, and also to avoid a limitation in
      Agg where coordinates can not be larger than 24-bit signed
      integers.

   4. PathSnapper: Rounds the path to the nearest center-pixels.
      This makes rectilinear curves look much better.

   5. PathSimplifier: Removes line segments from highly dense paths
      that would not have an impact on their appearance.  Speeds up
      rendering and reduces file sizes.

   6. curve-to-line-segment conversion (implemented in Agg, not here)

   7. stroking (implemented in Agg, not here)
 */

/************************************************************
 This is a base class for vertex converters that need to queue their
 output.  It is designed to be as fast as possible vs. the STL's queue
 which is more flexible.
 */
template<int QueueSize>
class EmbeddedQueue
{
protected:
    EmbeddedQueue() :
        m_queue_read(0), m_queue_write(0)
    {
        // empty
    }

    struct item
    {
        item()
        {

        }

        inline void
        set(const unsigned cmd_, const double& x_, const double& y_)
        {
            cmd = cmd_;
            x = x_;
            y = y_;
        }
        unsigned cmd;
        double x;
        double y;
    };
    int  m_queue_read;
    int  m_queue_write;
    item m_queue[QueueSize];

    inline void
    queue_push(const unsigned cmd, const double& x, const double& y)
    {
        m_queue[m_queue_write++].set(cmd, x, y);
    }

    inline bool
    queue_nonempty()
    {
        return m_queue_read < m_queue_write;
    }

    inline bool
    queue_pop(unsigned *cmd, double *x, double *y)
    {
        if (queue_nonempty())
        {
            const item& front = m_queue[m_queue_read++];
            *cmd = front.cmd;
            *x = front.x;
            *y = front.y;

            return true;
        }

        m_queue_read = 0;
        m_queue_write = 0;

        return false;
    }

    inline void
    queue_clear()
    {
        m_queue_read = 0;
        m_queue_write = 0;
    }
};

/*
  PathNanRemover is a vertex converter that removes non-finite values
  from the vertices list, and inserts MOVETO commands as necessary to
  skip over them.  If a curve segment contains at least one non-finite
  value, the entire curve segment will be skipped.
 */
template<class VertexSource>
class PathNanRemover : protected EmbeddedQueue<4>
{
    VertexSource* m_source;
    bool m_remove_nans;
    bool m_has_curves;
    static const unsigned char num_extra_points_map[16];

public:
    /* has_curves should be true if the path contains bezier curve
       segments, as this requires a slower algorithm to remove the
       NaNs.  When in doubt, set to true.
     */
    PathNanRemover(VertexSource& source, bool remove_nans, bool has_curves) :
        m_source(&source), m_remove_nans(remove_nans), m_has_curves(has_curves)
    {
        // empty
    }

    inline void
    rewind(unsigned path_id)
    {
        queue_clear();
        m_source->rewind(path_id);
    }

    inline unsigned
    vertex(double* x, double *y)
    {
        unsigned code;

        if (!m_remove_nans)
        {
            return m_source->vertex(x, y);
        }

        if (m_has_curves)
        {
            /* This is the slow method for when there might be curves. */
            if (queue_pop(&code, x, y))
            {
                return code;
            }

            bool needs_move_to = false;
            while (true)
            {
                /* The approach here is to push each full curve
                   segment into the queue.  If any non-finite values
                   are found along the way, the queue is emptied, and
                   the next curve segment is handled. */
                code = m_source->vertex(x, y);
                if (code == agg::path_cmd_stop ||
                        code == (agg::path_cmd_end_poly | agg::path_flags_close))
                {
                    return code;
                }

                if (needs_move_to)
                {
                    queue_push(agg::path_cmd_move_to, *x, *y);
                }

                size_t num_extra_points = num_extra_points_map[code & 0xF];
                bool has_nan = (MPL_notisfinite64(*x) || MPL_notisfinite64(*y));
                queue_push(code, *x, *y);
                /* Note: this test can not be short-circuited, since we need to
                   advance through the entire curve no matter what */
                for (size_t i = 0; i < num_extra_points; ++i)
                {
                    m_source->vertex(x, y);
                    has_nan |= (MPL_notisfinite64(*x) || MPL_notisfinite64(*y));
                    queue_push(code, *x, *y);
                }

                if (!has_nan)
                {
                    break;
                }

                queue_clear();

                /* If the last point is finite, we use that for the
                   moveto, otherwise, we'll use the first vertex of
                   the next curve. */
                if (!(MPL_notisfinite64(*x) || MPL_notisfinite64(*y)))
                {
                    queue_push(agg::path_cmd_move_to, *x, *y);
                    needs_move_to = false;
                }
                else
                {
                    needs_move_to = true;
                }
            }

            if (queue_pop(&code, x, y))
            {
                return code;
            }
            else
            {
                return agg::path_cmd_stop;
            }
        }
        else // !m_has_curves
        {
            /* This is the fast path for when we know we have no curves */
            code = m_source->vertex(x, y);

            if (code == agg::path_cmd_stop ||
                code == (agg::path_cmd_end_poly | agg::path_flags_close))
            {
                return code;
            }

            if (MPL_notisfinite64(*x) || MPL_notisfinite64(*y))
            {
                do
                {
                    code = m_source->vertex(x, y);
                    if (code == agg::path_cmd_stop ||
                        code == (agg::path_cmd_end_poly | agg::path_flags_close))
                    {
                        return code;
                    }
                }
                while (MPL_notisfinite64(*x) || MPL_notisfinite64(*y));
                return agg::path_cmd_move_to;
            }

            return code;
        }
    }
};

/* Defines when path segment types have more than one vertex */
template<class VertexSource>
const unsigned char PathNanRemover<VertexSource>::num_extra_points_map[] =
    {0, 0, 0, 1,
     2, 0, 0, 0,
     0, 0, 0, 0,
     0, 0, 0, 0
    };

/************************************************************
 PathClipper uses the Liang-Barsky line clipping algorithm (as
 implemented in Agg) to clip the path to a given rectangle.  Lines
 will never extend outside of the rectangle.  Curve segments are not
 clipped, but are always included in their entirety.
 */
template<class VertexSource>
class PathClipper
{
    VertexSource*          m_source;
    bool                   m_do_clipping;
    agg::rect_base<double> m_cliprect;
    double                 m_lastX;
    double                 m_lastY;
    bool                   m_moveto;
    double                 m_nextX;
    double                 m_nextY;
    bool                   m_has_next;
    double                 m_initX;
    double                 m_initY;
    bool                   m_has_init;
    bool                   m_broke_path;

public:
    PathClipper(VertexSource& source, bool do_clipping,
                double width, double height) :
        m_source(&source), m_do_clipping(do_clipping),
        m_cliprect(-1.0, -1.0, width + 1.0, height + 1.0), m_moveto(true),
        m_has_next(false), m_has_init(false), m_broke_path(false)
    {
        // empty
    }

    PathClipper(VertexSource& source, bool do_clipping,
                const agg::rect_base<double>& rect) :
        m_source(&source), m_do_clipping(do_clipping),
        m_cliprect(rect), m_moveto(true), m_has_next(false),
        m_has_init(false), m_broke_path(false)
    {
        m_cliprect.x1 -= 1.0;
        m_cliprect.y1 -= 1.0;
        m_cliprect.x2 += 1.0;
        m_cliprect.y2 += 1.0;
    }

    inline void
    rewind(unsigned path_id)
    {
        m_has_next = false;
        m_moveto = true;
        m_source->rewind(path_id);
    }

    unsigned
    vertex(double* x, double* y)
    {
        unsigned code;

        if (m_do_clipping)
        {
            /* This is the slow path where we actually do clipping */

            if (m_has_next)
            {
                m_has_next = false;
                *x = m_nextX;
                *y = m_nextY;
                return agg::path_cmd_line_to;
            }

            while ((code = m_source->vertex(x, y)) != agg::path_cmd_stop)
            {
                if (code == agg::path_cmd_move_to)
                {
                    m_initX = *x;
                    m_initY = *y;
                    m_has_init = true;
                    m_moveto = true;
                }
                if (m_moveto)
                {
                    m_moveto = false;
                    code = agg::path_cmd_move_to;
                    break;
                }
                else if (code == agg::path_cmd_line_to)
                {
                    double x0, y0, x1, y1;
                    x0 = m_lastX;
                    y0 = m_lastY;
                    x1 = *x;
                    y1 = *y;
                    m_lastX = *x;
                    m_lastY = *y;
                    unsigned moved = agg::clip_line_segment(&x0, &y0, &x1, &y1, m_cliprect);
                    // moved >= 4 - Fully clipped
                    // moved & 1 != 0 - First point has been moved
                    // moved & 2 != 0 - Second point has been moved
                    if (moved < 4)
                    {
                        if (moved & 1)
                        {
                            *x = x0;
                            *y = y0;
                            m_nextX = x1;
                            m_nextY = y1;
                            m_has_next = true;
                            m_broke_path = true;
                            return agg::path_cmd_move_to;
                        }
                        *x = x1;
                        *y = y1;
                        return code;
                    }
                }
                else if (code == (agg::path_cmd_end_poly | agg::path_flags_close)
                         && m_broke_path && m_has_init)
                {
                    *x = m_initX;
                    *y = m_initY;
                    return agg::path_cmd_line_to;
                }
                else
                {
                    break;
                }
            }

            m_lastX = *x;
            m_lastY = *y;
            return code;
        }
        else
        {
            // If not doing any clipping, just pass along the vertices
            // verbatim
            return m_source->vertex(x, y);
        }
    }
};

/************************************************************
 PathSnapper rounds vertices to their nearest center-pixels.  This
 makes rectilinear paths (rectangles, horizontal and vertical lines
 etc.) look much cleaner.
*/
enum e_snap_mode
{
    SNAP_AUTO,
    SNAP_FALSE,
    SNAP_TRUE
};

template<class VertexSource>
class PathSnapper
{
private:
    VertexSource* m_source;
    bool          m_snap;
    double        m_snap_value;

    static bool
    should_snap(VertexSource& path,
                e_snap_mode snap_mode,
                unsigned total_vertices)
    {
        // If this contains only straight horizontal or vertical lines, it should be
        // snapped to the nearest pixels
        double x0, y0, x1, y1;
        unsigned code;

        switch (snap_mode)
        {
        case SNAP_AUTO:
            if (total_vertices > 1024)
            {
                return false;
            }

            code = path.vertex(&x0, &y0);
            if (code == agg::path_cmd_stop)
            {
                return false;
            }

            while ((code = path.vertex(&x1, &y1)) != agg::path_cmd_stop)
            {
                switch (code)
                {
                case agg::path_cmd_curve3:
                case agg::path_cmd_curve4:
                    return false;
                case agg::path_cmd_line_to:
                    if (!(fabs(x0 - x1) < 1e-4 || fabs(y0 - y1) < 1e-4))
                    {
                        return false;
                    }
                }
                x0 = x1;
                y0 = y1;
            }

            return true;
        case SNAP_FALSE:
            return false;
        case SNAP_TRUE:
            return true;
        }

        return false;
    }

public:
    /*
      snap_mode should be one of:
        - SNAP_AUTO: Examine the path to determine if it should be snapped
        - SNAP_TRUE: Force snapping
        - SNAP_FALSE: No snapping
    */
    PathSnapper(VertexSource& source, e_snap_mode snap_mode,
                unsigned total_vertices = 15, double stroke_width = 0.0) :
        m_source(&source)
    {
        m_snap = should_snap(source, snap_mode, total_vertices);

        if (m_snap)
        {
            int is_odd = (int)mpl_round(stroke_width) % 2;
            m_snap_value = (is_odd) ? 0.5 : 0.0;
        }

        source.rewind(0);
    }

    inline void
    rewind(unsigned path_id)
    {
        m_source->rewind(path_id);
    }

    inline unsigned
    vertex(double* x, double* y)
    {
        unsigned code;
        code = m_source->vertex(x, y);
        if (m_snap && agg::is_vertex(code))
        {
            *x = floor(*x + 0.5) + m_snap_value;
            *y = floor(*y + 0.5) + m_snap_value;
        }
        return code;
    }

    inline bool
    is_snapping()
    {
        return m_snap;
    }
};

/************************************************************
 PathSimplifier reduces the number of vertices in a dense path without
 changing its appearance.
*/
template<class VertexSource>
class PathSimplifier : protected EmbeddedQueue<9>
{
public:
    /* Set simplify to true to perform simplification */
    PathSimplifier(VertexSource& source, bool do_simplify, double simplify_threshold) :
        m_source(&source), m_simplify(do_simplify),
        m_simplify_threshold(simplify_threshold*simplify_threshold),
        m_moveto(true), m_after_moveto(false),
        m_lastx(0.0), m_lasty(0.0), m_clipped(false),
        m_origdx(0.0), m_origdy(0.0),
        m_origdNorm2(0.0), m_dnorm2Max(0.0),
        m_lastMax(false), m_nextX(0.0), m_nextY(0.0),
        m_lastWrittenX(0.0), m_lastWrittenY(0.0)
    {
        // empty
    }

    inline void
    rewind(unsigned path_id)
    {
        queue_clear();
        m_moveto = true;
        m_source->rewind(path_id);
    }

    unsigned
    vertex(double* x, double* y)
    {
        unsigned cmd;

        /* The simplification algorithm doesn't support curves or compound paths
           so we just don't do it at all in that case... */
        if (!m_simplify)
        {
            return m_source->vertex(x, y);
        }

        /* idea: we can skip drawing many lines: lines < 1 pixel in
           length, and we can combine sequential parallel lines into a
           single line instead of redrawing lines over the same
           points.  The loop below works a bit like a state machine,
           where what it does depends on what it did in the last
           looping. To test whether sequential lines are close to
           parallel, I calculate the distance moved perpendicular to
           the last line. Once it gets too big, the lines cannot be
           combined. */

        /* This code was originally written by Allan Haldane and I
           have modified to work in-place -- meaning not creating an
           entirely new path list each time.  In order to do that
           without too much additional code complexity, it keeps a
           small queue around so that multiple points can be emitted
           in a single call, and those points will be popped from the
           queue in subsequent calls.  The following block will empty
           the queue before proceeding to the main loop below.
           -- Michael Droettboom */

        if (queue_pop(&cmd, x, y))
        {
            return cmd;
        }

        /* The main simplification loop.  The point is to consume only
           as many points as necessary until something has been added
           to the outbound queue, not to run through the entire path
           in one go.  This eliminates the need to allocate and fill
           an entire additional path array on each draw. */
        while ((cmd = m_source->vertex(x, y)) != agg::path_cmd_stop)
        {
            /* if we are starting a new path segment, move to the first point
               + init */

            if (m_moveto || cmd == agg::path_cmd_move_to)
            {
                /* m_moveto check is not generally needed because
                   m_source generates an initial moveto; but it is
                   retained for safety in case circumstances arise
                   where this is not true. */
                if (m_origdNorm2 != 0.0 && !m_after_moveto)
                {
                    /* m_origdNorm2 is nonzero only if we have a
                       vector; the m_after_moveto check ensures we
                       push this vector to the queue only once. */
                    _push(x, y);
                }
                m_after_moveto = true;
                m_lastx = *x;
                m_lasty = *y;
                m_moveto = false;
                m_origdNorm2 = 0.0;
                m_clipped = true;
                if (queue_nonempty())
                {
                    /* If we did a push, empty the queue now. */
                    break;
                }
                continue;
            }
            m_after_moveto = false;

            /* NOTE: We used to skip this very short segments, but if
               you have a lot of them cumulatively, you can miss
               maxima or minima in the data. */

            /* Don't render line segments less than one pixel long */
            /* if (fabs(*x - m_lastx) < 1.0 && fabs(*y - m_lasty) < 1.0) */
            /* { */
            /*     continue; */
            /* } */

            /* if we have no orig vector, set it to this vector and
               continue.  this orig vector is the reference vector we
               will build up the line to */
            if (m_origdNorm2 == 0.0)
            {
                if (m_clipped)
                {
                    queue_push(agg::path_cmd_move_to, m_lastx, m_lasty);
                    m_clipped = false;
                }

                m_origdx = *x - m_lastx;
                m_origdy = *y - m_lasty;
                m_origdNorm2 = m_origdx * m_origdx + m_origdy * m_origdy;

                //set all the variables to reflect this new orig vector
                m_dnorm2Max = m_origdNorm2;
                m_lastMax = true;

                m_nextX = m_lastWrittenX = m_lastx = *x;
                m_nextY = m_lastWrittenY = m_lasty = *y;
                continue;
            }

            /* If got to here, then we have an orig vector and we just got
               a vector in the sequence. */

            /* Check that the perpendicular distance we have moved
               from the last written point compared to the line we are
               building is not too much. If o is the orig vector (we
               are building on), and v is the vector from the last
               written point to the current point, then the
               perpendicular vector is p = v - (o.v)o, and we
               normalize o (by dividing the second term by o.o). */

            /* get the v vector */
            double totdx = *x - m_lastWrittenX;
            double totdy = *y - m_lastWrittenY;
            double totdot = m_origdx * totdx + m_origdy * totdy;

            /* get the para vector ( = (o.v)o/(o.o)) */
            double paradx = totdot * m_origdx / m_origdNorm2;
            double parady = totdot * m_origdy / m_origdNorm2;

            /* get the perp vector ( = v - para) */
            double perpdx = totdx - paradx;
            double perpdy = totdy - parady;
            double perpdNorm2 = perpdx * perpdx + perpdy * perpdy;

            /* If the perp vector is less than some number of (squared)
               pixels in size, then merge the current vector */
            if (perpdNorm2 < m_simplify_threshold)
            {
                /* check if the current vector is parallel or
                   anti-parallel to the orig vector. If it is
                   parallel, test if it is the longest of the vectors
                   we are merging in that direction. */
                double paradNorm2 = paradx * paradx + parady * parady;

                m_lastMax = false;
                if (totdot > 0.0)
                {
                    if (paradNorm2 > m_dnorm2Max)
                    {
                        m_lastMax = true;
                        m_dnorm2Max = paradNorm2;
                        m_nextX = *x;
                        m_nextY = *y;
                    }
                }
                else
                {
                    _push(&m_lastx, &m_lasty);
                    _push(x, y);
                    break;
                }

                m_lastx = *x;
                m_lasty = *y;
                continue;
            }

            /* If we get here, then this vector was not similar enough to the
               line we are building, so we need to draw that line and start the
               next one. */

            /* If the line needs to extend in the opposite direction from the
               direction we are drawing in, move back to we start drawing from
               back there. */
            _push(x, y);

            break;
        }

        /* Fill the queue with the remaining vertices if we've finished the
           path in the above loop. */
        if (cmd == agg::path_cmd_stop)
        {
            if (m_origdNorm2 != 0.0)
            {
                queue_push((m_moveto || m_after_moveto) ?
                           agg::path_cmd_move_to : agg::path_cmd_line_to,
                           m_nextX, m_nextY);
                m_moveto = false;
            }
            queue_push((m_moveto || m_after_moveto) ?
                       agg::path_cmd_move_to : agg::path_cmd_line_to,
                       m_lastx, m_lasty);
            m_moveto = false;
            queue_push(agg::path_cmd_stop, 0.0, 0.0);
        }

        /* Return the first item in the queue, if any, otherwise
           indicate that we're done. */
        if (queue_pop(&cmd, x, y))
        {
            return cmd;
        }
        else
        {
            return agg::path_cmd_stop;
        }
    }

private:
    VertexSource* m_source;
    bool          m_simplify;
    double        m_simplify_threshold;

    bool   m_moveto;
    bool   m_after_moveto;
    double m_lastx, m_lasty;
    bool   m_clipped;

    double m_origdx;
    double m_origdy;
    double m_origdNorm2;
    double m_dnorm2Max;
    bool   m_lastMax;
    double m_nextX;
    double m_nextY;
    double m_lastWrittenX;
    double m_lastWrittenY;

    inline void
    _push(double* x, double* y)
    {
        queue_push(agg::path_cmd_line_to, m_nextX, m_nextY);

        /* If we clipped some segments between this line and the next line
           we are starting, we also need to move to the last point. */
        if (m_clipped)
        {
            queue_push(agg::path_cmd_move_to, m_lastx, m_lasty);
        }
        else if (!m_lastMax)
        {
            /* If the last line was not the longest line, then move
               back to the end point of the last line in the
               sequence. Only do this if not clipped, since in that
               case lastx,lasty is not part of the line just drawn. */

            /* Would be move_to if not for the artifacts */
            queue_push(agg::path_cmd_line_to, m_lastx, m_lasty);
        }

        /* Now reset all the variables to get ready for the next line */
        m_origdx = *x - m_lastx;
        m_origdy = *y - m_lasty;
        m_origdNorm2 = m_origdx * m_origdx + m_origdy * m_origdy;

        m_dnorm2Max = m_origdNorm2;
        m_lastMax = true;
        m_lastWrittenX = m_queue[m_queue_write-1].x;
        m_lastWrittenY = m_queue[m_queue_write-1].y;
        m_lastx = m_nextX = *x;
        m_lasty = m_nextY = *y;

        m_clipped = false;
    }
};

template<class VertexSource>
class Sketch
{
public:
    /*
       scale: the scale of the wiggle perpendicular to the original
       line (in pixels)

       length: the base wavelength of the wiggle along the
       original line (in pixels)

       randomness: the factor that the sketch length will randomly
       shrink and expand.
    */
    Sketch(VertexSource& source, double scale, double length, double randomness) :
        m_source(&source), m_scale(scale), m_length(length),
        m_randomness(randomness), m_segmented(source), m_last_x(0.0),
        m_last_y(0.0), m_has_last(false), m_p(0.0)
    {
        rewind(0);
    }

    unsigned
    vertex(double* x, double* y)
    {
        if (m_scale == 0.0)
        {
            return m_source->vertex(x, y);
        }

        unsigned code = m_segmented.vertex(x, y);

        if (code == agg::path_cmd_move_to) {
            m_has_last = false;
            m_p = 0.0;
        }

        if (m_has_last) {
            // We want the "cursor" along the sine wave to move at a
            // random rate.
            double d_rand = rand() / double(RAND_MAX);
            double d_M_PI = 3.14159265358979323846;
            m_p += pow(m_randomness, d_rand * 2.0 - 1.0);
            double r = sin(m_p / (m_length / (d_M_PI * 2.0))) * m_scale;
            double den = m_last_x - *x;
            double num = m_last_y - *y;
            double len = num*num + den*den;
            m_last_x = *x;
            m_last_y = *y;
            if (len != 0) {
                len = sqrt(len);
                *x += r * num / len;
                *y += r * -den / len;
            }
        } else {
            m_last_x = *x;
            m_last_y = *y;
        }

        m_has_last = true;

        return code;
    }

    inline void
    rewind(unsigned path_id)
    {
        srand(0);
        m_has_last = false;
        m_p = 0.0;
        if (m_scale != 0.0) {
            m_segmented.rewind(path_id);
        } else {
            m_source->rewind(path_id);
        }
    }

private:
    VertexSource*                       m_source;
    double                              m_scale;
    double                              m_length;
    double                              m_randomness;
    agg::conv_segmentator<VertexSource> m_segmented;
    double                              m_last_x;
    double                              m_last_y;
    bool                                m_has_last;
    double                              m_p;
};

#endif // __PATH_CONVERTERS_H__
