/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef __PATH_CONVERTERS_H__
#define __PATH_CONVERTERS_H__

#include <cmath>
#include <stdint.h>
#include "agg_path_storage.h"
#include "agg_clip_liang_barsky.h"
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
template <int QueueSize>
class EmbeddedQueue
{
  protected:
    EmbeddedQueue() : m_queue_read(0), m_queue_write(0)
    {
        // empty
    }

    struct item
    {
        item()
        {
        }

        inline void set(const unsigned cmd_, const double x_, const double y_)
        {
            cmd = cmd_;
            x = x_;
            y = y_;
        }
        unsigned cmd;
        double x;
        double y;
    };
    int m_queue_read;
    int m_queue_write;
    item m_queue[QueueSize];

    inline void queue_push(const unsigned cmd, const double x, const double y)
    {
        m_queue[m_queue_write++].set(cmd, x, y);
    }

    inline bool queue_nonempty()
    {
        return m_queue_read < m_queue_write;
    }

    inline bool queue_pop(unsigned *cmd, double *x, double *y)
    {
        if (queue_nonempty()) {
            const item &front = m_queue[m_queue_read++];
            *cmd = front.cmd;
            *x = front.x;
            *y = front.y;

            return true;
        }

        m_queue_read = 0;
        m_queue_write = 0;

        return false;
    }

    inline void queue_clear()
    {
        m_queue_read = 0;
        m_queue_write = 0;
    }
};

/* Defines when path segment types have more than one vertex */
static const size_t num_extra_points_map[] =
    {0, 0, 0, 1,
     2, 0, 0, 0,
     0, 0, 0, 0,
     0, 0, 0, 0
    };

/* An implementation of a simple linear congruential random number
   generator.  This is a "classic" and fast RNG which works fine for
   our purposes of sketching lines, but should not be used for things
   that matter, like crypto.  We are implementing this ourselves
   rather than using the C stdlib so that the seed state is not shared
   with other third-party code. There are recent C++ options, but we
   still require nothing later than C++98 for compatibility
   reasons. */
class RandomNumberGenerator
{
private:
    /* These are the same constants from MS Visual C++, which
       has the nice property that the modulus is 2^32, thus
       saving an explicit modulo operation
    */
    static const uint32_t a = 214013;
    static const uint32_t c = 2531011;
    uint32_t m_seed;

public:
    RandomNumberGenerator() : m_seed(0) {}
    RandomNumberGenerator(int seed) : m_seed(seed) {}

    void seed(int seed)
    {
        m_seed = seed;
    }

    double get_double()
    {
        m_seed = (a * m_seed + c);
        return (double)m_seed / (double)(1LL << 32);
    }
};

/*
  PathNanRemover is a vertex converter that removes non-finite values
  from the vertices list, and inserts MOVETO commands as necessary to
  skip over them.  If a curve segment contains at least one non-finite
  value, the entire curve segment will be skipped.
 */
template <class VertexSource>
class PathNanRemover : protected EmbeddedQueue<4>
{
    VertexSource *m_source;
    bool m_remove_nans;
    bool m_has_curves;

  public:
    /* has_curves should be true if the path contains bezier curve
       segments, as this requires a slower algorithm to remove the
       NaNs.  When in doubt, set to true.
     */
    PathNanRemover(VertexSource &source, bool remove_nans, bool has_curves)
        : m_source(&source), m_remove_nans(remove_nans), m_has_curves(has_curves)
    {
        // empty
    }

    inline void rewind(unsigned path_id)
    {
        queue_clear();
        m_source->rewind(path_id);
    }

    inline unsigned vertex(double *x, double *y)
    {
        unsigned code;

        if (!m_remove_nans) {
            return m_source->vertex(x, y);
        }

        if (m_has_curves) {
            /* This is the slow method for when there might be curves. */
            if (queue_pop(&code, x, y)) {
                return code;
            }

            bool needs_move_to = false;
            while (true) {
                /* The approach here is to push each full curve
                   segment into the queue.  If any non-finite values
                   are found along the way, the queue is emptied, and
                   the next curve segment is handled. */
                code = m_source->vertex(x, y);
                if (code == agg::path_cmd_stop ||
                    code == (agg::path_cmd_end_poly | agg::path_flags_close)) {
                    return code;
                }

                if (needs_move_to) {
                    queue_push(agg::path_cmd_move_to, *x, *y);
                }

                size_t num_extra_points = num_extra_points_map[code & 0xF];
                bool has_nan = (!(std::isfinite(*x) && std::isfinite(*y)));
                queue_push(code, *x, *y);

                /* Note: this test can not be short-circuited, since we need to
                   advance through the entire curve no matter what */
                for (size_t i = 0; i < num_extra_points; ++i) {
                    m_source->vertex(x, y);
                    has_nan = has_nan || !(std::isfinite(*x) && std::isfinite(*y));
                    queue_push(code, *x, *y);
                }

                if (!has_nan) {
                    break;
                }

                queue_clear();

                /* If the last point is finite, we use that for the
                   moveto, otherwise, we'll use the first vertex of
                   the next curve. */
                if (std::isfinite(*x) && std::isfinite(*y)) {
                    queue_push(agg::path_cmd_move_to, *x, *y);
                    needs_move_to = false;
                } else {
                    needs_move_to = true;
                }
            }

            if (queue_pop(&code, x, y)) {
                return code;
            } else {
                return agg::path_cmd_stop;
            }
        } else // !m_has_curves
        {
            /* This is the fast path for when we know we have no curves */
            code = m_source->vertex(x, y);

            if (code == agg::path_cmd_stop ||
                code == (agg::path_cmd_end_poly | agg::path_flags_close)) {
                return code;
            }

            if (!(std::isfinite(*x) && std::isfinite(*y))) {
                do {
                    code = m_source->vertex(x, y);
                    if (code == agg::path_cmd_stop ||
                        code == (agg::path_cmd_end_poly | agg::path_flags_close)) {
                        return code;
                    }
                } while (!(std::isfinite(*x) && std::isfinite(*y)));
                return agg::path_cmd_move_to;
            }

            return code;
        }
    }
};

/************************************************************
 PathClipper uses the Liang-Barsky line clipping algorithm (as
 implemented in Agg) to clip the path to a given rectangle.  Lines
 will never extend outside of the rectangle.  Curve segments are not
 clipped, but are always included in their entirety.
 */
template <class VertexSource>
class PathClipper : public EmbeddedQueue<3>
{
    VertexSource *m_source;
    bool m_do_clipping;
    agg::rect_base<double> m_cliprect;
    double m_lastX;
    double m_lastY;
    bool m_moveto;
    double m_initX;
    double m_initY;
    bool m_has_init;

  public:
    PathClipper(VertexSource &source, bool do_clipping, double width, double height)
        : m_source(&source),
          m_do_clipping(do_clipping),
          m_cliprect(-1.0, -1.0, width + 1.0, height + 1.0),
          m_moveto(true),
          m_has_init(false)
    {
        // empty
    }

    PathClipper(VertexSource &source, bool do_clipping, const agg::rect_base<double> &rect)
        : m_source(&source),
          m_do_clipping(do_clipping),
          m_cliprect(rect),
          m_moveto(true),
          m_has_init(false)
    {
        m_cliprect.x1 -= 1.0;
        m_cliprect.y1 -= 1.0;
        m_cliprect.x2 += 1.0;
        m_cliprect.y2 += 1.0;
    }

    inline void rewind(unsigned path_id)
    {
        m_has_init = false;
        m_moveto = true;
        m_source->rewind(path_id);
    }

    int draw_clipped_line(double x0, double y0, double x1, double y1)
    {
        unsigned moved = agg::clip_line_segment(&x0, &y0, &x1, &y1, m_cliprect);
        // moved >= 4 - Fully clipped
        // moved & 1 != 0 - First point has been moved
        // moved & 2 != 0 - Second point has been moved
        if (moved < 4) {
            if (moved & 1 || m_moveto) {
                queue_push(agg::path_cmd_move_to, x0, y0);
            }
            queue_push(agg::path_cmd_line_to, x1, y1);

            m_moveto = false;
            return 1;
        }

        return 0;
    }

    unsigned vertex(double *x, double *y)
    {
        unsigned code;
        bool emit_moveto = false;

        if (m_do_clipping) {
            /* This is the slow path where we actually do clipping */

            if (queue_pop(&code, x, y)) {
                return code;
            }

            while ((code = m_source->vertex(x, y)) != agg::path_cmd_stop) {
                emit_moveto = false;

                switch (code) {
                case (agg::path_cmd_end_poly | agg::path_flags_close):
                    if (m_has_init) {
                        draw_clipped_line(m_lastX, m_lastY, m_initX, m_initY);
                    }
                    queue_push(
                        agg::path_cmd_end_poly | agg::path_flags_close,
                        m_lastX, m_lastY);
                    goto exit_loop;

                case agg::path_cmd_move_to:

                    // was the last command a moveto (and we have
                    // seen at least one command ?
                    // if so, shove it in the queue if in clip box
                    if (m_moveto && m_has_init &&
                        m_lastX >= m_cliprect.x1 &&
                        m_lastX <= m_cliprect.x2 &&
                        m_lastY >= m_cliprect.y1 &&
                        m_lastY <= m_cliprect.y2) {
                        // push the last moveto onto the queue
                        queue_push(agg::path_cmd_move_to, m_lastX, m_lastY);
                        // flag that we need to emit it
                        emit_moveto = true;
                    }
                    // update the internal state for this moveto
                    m_initX = m_lastX = *x;
                    m_initY = m_lastY = *y;
                    m_has_init = true;
                    m_moveto = true;
                    // if the last command was moveto exit the loop to emit the code
                    if (emit_moveto) {
                        goto exit_loop;
                    }
                    // else, break and get the next point
                    break;

                case agg::path_cmd_line_to:
                    if (draw_clipped_line(m_lastX, m_lastY, *x, *y)) {
                        m_lastX = *x;
                        m_lastY = *y;
                        goto exit_loop;
                    }
                    m_lastX = *x;
                    m_lastY = *y;
                    break;

                default:
                    if (m_moveto) {
                        queue_push(agg::path_cmd_move_to, m_lastX, m_lastY);
                        m_moveto = false;
                    }

                    queue_push(code, *x, *y);
                    m_lastX = *x;
                    m_lastY = *y;
                    goto exit_loop;
                }
            }

        exit_loop:

            if (queue_pop(&code, x, y)) {
                return code;
            }

            if (m_moveto &&
                m_lastX >= m_cliprect.x1 &&
                m_lastX <= m_cliprect.x2 &&
                m_lastY >= m_cliprect.y1 &&
                m_lastY <= m_cliprect.y2) {
                *x = m_lastX;
                *y = m_lastY;
                m_moveto = false;
                return agg::path_cmd_move_to;
            }

            return agg::path_cmd_stop;
        } else {
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
enum e_snap_mode {
    SNAP_AUTO,
    SNAP_FALSE,
    SNAP_TRUE
};

template <class VertexSource>
class PathSnapper
{
  private:
    VertexSource *m_source;
    bool m_snap;
    double m_snap_value;

    static bool should_snap(VertexSource &path, e_snap_mode snap_mode, unsigned total_vertices)
    {
        // If this contains only straight horizontal or vertical lines, it should be
        // snapped to the nearest pixels
        double x0 = 0, y0 = 0, x1 = 0, y1 = 0;
        unsigned code;

        switch (snap_mode) {
        case SNAP_AUTO:
            if (total_vertices > 1024) {
                return false;
            }

            code = path.vertex(&x0, &y0);
            if (code == agg::path_cmd_stop) {
                return false;
            }

            while ((code = path.vertex(&x1, &y1)) != agg::path_cmd_stop) {
                switch (code) {
                case agg::path_cmd_curve3:
                case agg::path_cmd_curve4:
                    return false;
                case agg::path_cmd_line_to:
                    if (!(fabs(x0 - x1) < 1e-4 || fabs(y0 - y1) < 1e-4)) {
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
    PathSnapper(VertexSource &source,
                e_snap_mode snap_mode,
                unsigned total_vertices = 15,
                double stroke_width = 0.0)
        : m_source(&source)
    {
        m_snap = should_snap(source, snap_mode, total_vertices);

        if (m_snap) {
            int is_odd = (int)mpl_round(stroke_width) % 2;
            m_snap_value = (is_odd) ? 0.5 : 0.0;
        }

        source.rewind(0);
    }

    inline void rewind(unsigned path_id)
    {
        m_source->rewind(path_id);
    }

    inline unsigned vertex(double *x, double *y)
    {
        unsigned code;
        code = m_source->vertex(x, y);
        if (m_snap && agg::is_vertex(code)) {
            *x = floor(*x + 0.5) + m_snap_value;
            *y = floor(*y + 0.5) + m_snap_value;
        }
        return code;
    }

    inline bool is_snapping()
    {
        return m_snap;
    }
};

/************************************************************
 PathSimplifier reduces the number of vertices in a dense path without
 changing its appearance.
*/
template <class VertexSource>
class PathSimplifier : protected EmbeddedQueue<9>
{
  public:
    /* Set simplify to true to perform simplification */
    PathSimplifier(VertexSource &source, bool do_simplify, double simplify_threshold)
        : m_source(&source),
          m_simplify(do_simplify),
          /* we square simplify_threshold so that we can compute
             norms without doing the square root every step. */
          m_simplify_threshold(simplify_threshold * simplify_threshold),

          m_moveto(true),
          m_after_moveto(false),
          m_clipped(false),

          // the x, y values from last iteration
          m_lastx(0.0),
          m_lasty(0.0),

          // the dx, dy comprising the original vector, used in conjunction
          // with m_currVecStart* to define the original vector.
          m_origdx(0.0),
          m_origdy(0.0),

          // the squared norm of the original vector
          m_origdNorm2(0.0),

          // maximum squared norm of vector in forward (parallel) direction
          m_dnorm2ForwardMax(0.0),
          // maximum squared norm of vector in backward (anti-parallel) direction
          m_dnorm2BackwardMax(0.0),

          // was the last point the furthest from lastWritten in the
          // forward (parallel) direction?
          m_lastForwardMax(false),
          // was the last point the furthest from lastWritten in the
          // backward (anti-parallel) direction?
          m_lastBackwardMax(false),

          // added to queue when _push is called
          m_nextX(0.0),
          m_nextY(0.0),

          // added to queue when _push is called if any backwards
          // (anti-parallel) vectors were observed
          m_nextBackwardX(0.0),
          m_nextBackwardY(0.0),

          // start of the current vector that is being simplified
          m_currVecStartX(0.0),
          m_currVecStartY(0.0)
    {
        // empty
    }

    inline void rewind(unsigned path_id)
    {
        queue_clear();
        m_moveto = true;
        m_source->rewind(path_id);
    }

    unsigned vertex(double *x, double *y)
    {
        unsigned cmd;

        /* The simplification algorithm doesn't support curves or compound paths
           so we just don't do it at all in that case... */
        if (!m_simplify) {
            return m_source->vertex(x, y);
        }

        /* idea: we can skip drawing many lines: we can combine
           sequential parallel lines into a
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

        /* This code was originally written by Allan Haldane and
           updated by Michael Droettboom. I have modified it to
           handle anti-parallel vectors. This is done essentially
           the same way as parallel vectors, but requires a little
           additional book-keeping to track whether or not we have
           observed an anti-parallel vector during the current run.
           -- Kevin Rose */

        if (queue_pop(&cmd, x, y)) {
            return cmd;
        }

        /* The main simplification loop.  The point is to consume only
           as many points as necessary until something has been added
           to the outbound queue, not to run through the entire path
           in one go.  This eliminates the need to allocate and fill
           an entire additional path array on each draw. */
        while ((cmd = m_source->vertex(x, y)) != agg::path_cmd_stop) {
            /* if we are starting a new path segment, move to the first point
               + init */

            if (m_moveto || cmd == agg::path_cmd_move_to) {
                /* m_moveto check is not generally needed because
                   m_source generates an initial moveto; but it is
                   retained for safety in case circumstances arise
                   where this is not true. */
                if (m_origdNorm2 != 0.0 && !m_after_moveto) {
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
                m_dnorm2BackwardMax = 0.0;
                m_clipped = true;
                if (queue_nonempty()) {
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
            if (m_origdNorm2 == 0.0) {
                if (m_clipped) {
                    queue_push(agg::path_cmd_move_to, m_lastx, m_lasty);
                    m_clipped = false;
                }

                m_origdx = *x - m_lastx;
                m_origdy = *y - m_lasty;
                m_origdNorm2 = m_origdx * m_origdx + m_origdy * m_origdy;

                // set all the variables to reflect this new orig vector
                m_dnorm2ForwardMax = m_origdNorm2;
                m_dnorm2BackwardMax = 0.0;
                m_lastForwardMax = true;
                m_lastBackwardMax = false;

                m_currVecStartX = m_lastx;
                m_currVecStartY = m_lasty;
                m_nextX = m_lastx = *x;
                m_nextY = m_lasty = *y;
                continue;
            }

            /* If got to here, then we have an orig vector and we just got
               a vector in the sequence. */

            /* Check that the perpendicular distance we have moved
               from the last written point compared to the line we are
               building is not too much. If o is the orig vector (we
               are building on), and v is the vector from the last
               written point to the current point, then the
               perpendicular vector is p = v - (o.v)o/(o.o)
               (here, a.b indicates the dot product of a and b). */

            /* get the v vector */
            double totdx = *x - m_currVecStartX;
            double totdy = *y - m_currVecStartY;

            /* get the dot product o.v */
            double totdot = m_origdx * totdx + m_origdy * totdy;

            /* get the para vector ( = (o.v)o/(o.o)) */
            double paradx = totdot * m_origdx / m_origdNorm2;
            double parady = totdot * m_origdy / m_origdNorm2;

            /* get the perp vector ( = v - para) */
            double perpdx = totdx - paradx;
            double perpdy = totdy - parady;

            /* get the squared norm of perp vector ( = p.p) */
            double perpdNorm2 = perpdx * perpdx + perpdy * perpdy;

            /* If the perpendicular vector is less than
               m_simplify_threshold pixels in size, then merge
               current x,y with the current vector */
            if (perpdNorm2 < m_simplify_threshold) {
                /* check if the current vector is parallel or
                   anti-parallel to the orig vector. In either case,
                   test if it is the longest of the vectors
                   we are merging in that direction. If it is, then
                   update the current vector in that direction. */
                double paradNorm2 = paradx * paradx + parady * parady;

                m_lastForwardMax = false;
                m_lastBackwardMax = false;
                if (totdot > 0.0) {
                    if (paradNorm2 > m_dnorm2ForwardMax) {
                        m_lastForwardMax = true;
                        m_dnorm2ForwardMax = paradNorm2;
                        m_nextX = *x;
                        m_nextY = *y;
                    }
                } else {
                    if (paradNorm2 > m_dnorm2BackwardMax) {
                        m_lastBackwardMax = true;
                        m_dnorm2BackwardMax = paradNorm2;
                        m_nextBackwardX = *x;
                        m_nextBackwardY = *y;
                    }
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
        if (cmd == agg::path_cmd_stop) {
            if (m_origdNorm2 != 0.0) {
                queue_push((m_moveto || m_after_moveto) ? agg::path_cmd_move_to
                                                        : agg::path_cmd_line_to,
                           m_nextX,
                           m_nextY);
                if (m_dnorm2BackwardMax > 0.0) {
                    queue_push((m_moveto || m_after_moveto) ? agg::path_cmd_move_to
                                                            : agg::path_cmd_line_to,
                               m_nextBackwardX,
                               m_nextBackwardY);
                }
                m_moveto = false;
            }
            queue_push((m_moveto || m_after_moveto) ? agg::path_cmd_move_to : agg::path_cmd_line_to,
                       m_lastx,
                       m_lasty);
            m_moveto = false;
            queue_push(agg::path_cmd_stop, 0.0, 0.0);
        }

        /* Return the first item in the queue, if any, otherwise
           indicate that we're done. */
        if (queue_pop(&cmd, x, y)) {
            return cmd;
        } else {
            return agg::path_cmd_stop;
        }
    }

  private:
    VertexSource *m_source;
    bool m_simplify;
    double m_simplify_threshold;

    bool m_moveto;
    bool m_after_moveto;
    bool m_clipped;
    double m_lastx, m_lasty;

    double m_origdx;
    double m_origdy;
    double m_origdNorm2;
    double m_dnorm2ForwardMax;
    double m_dnorm2BackwardMax;
    bool m_lastForwardMax;
    bool m_lastBackwardMax;
    double m_nextX;
    double m_nextY;
    double m_nextBackwardX;
    double m_nextBackwardY;
    double m_currVecStartX;
    double m_currVecStartY;

    inline void _push(double *x, double *y)
    {
        bool needToPushBack = (m_dnorm2BackwardMax > 0.0);

        /* If we observed any backward (anti-parallel) vectors, then
           we need to push both forward and backward vectors. */
        if (needToPushBack) {
            /* If the last vector seen was the maximum in the forward direction,
               then we need to push the forward after the backward. Otherwise,
               the last vector seen was the maximum in the backward direction,
               or somewhere in between, either way we are safe pushing forward
               before backward. */
            if (m_lastForwardMax) {
                queue_push(agg::path_cmd_line_to, m_nextBackwardX, m_nextBackwardY);
                queue_push(agg::path_cmd_line_to, m_nextX, m_nextY);
            } else {
                queue_push(agg::path_cmd_line_to, m_nextX, m_nextY);
                queue_push(agg::path_cmd_line_to, m_nextBackwardX, m_nextBackwardY);
            }
        } else {
            /* If we did not observe any backwards vectors, just push forward. */
            queue_push(agg::path_cmd_line_to, m_nextX, m_nextY);
        }

        /* If we clipped some segments between this line and the next line
           we are starting, we also need to move to the last point. */
        if (m_clipped) {
            queue_push(agg::path_cmd_move_to, m_lastx, m_lasty);
        } else if ((!m_lastForwardMax) && (!m_lastBackwardMax)) {
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

        m_dnorm2ForwardMax = m_origdNorm2;
        m_lastForwardMax = true;
        m_currVecStartX = m_queue[m_queue_write - 1].x;
        m_currVecStartY = m_queue[m_queue_write - 1].y;
        m_lastx = m_nextX = *x;
        m_lasty = m_nextY = *y;
        m_dnorm2BackwardMax = 0.0;
        m_lastBackwardMax = false;

        m_clipped = false;
    }
};

template <class VertexSource>
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
    Sketch(VertexSource &source, double scale, double length, double randomness)
        : m_source(&source),
          m_scale(scale),
          m_length(length),
          m_randomness(randomness),
          m_segmented(source),
          m_last_x(0.0),
          m_last_y(0.0),
          m_has_last(false),
          m_p(0.0),
          m_rand(0)
    {
        rewind(0);
    }

    unsigned vertex(double *x, double *y)
    {
        if (m_scale == 0.0) {
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
            double d_rand = m_rand.get_double();
            double d_M_PI = 3.14159265358979323846;
            m_p += pow(m_randomness, d_rand * 2.0 - 1.0);
            double r = sin(m_p / (m_length / (d_M_PI * 2.0))) * m_scale;
            double den = m_last_x - *x;
            double num = m_last_y - *y;
            double len = num * num + den * den;
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

    inline void rewind(unsigned path_id)
    {
        m_has_last = false;
        m_p = 0.0;
        if (m_scale != 0.0) {
            m_rand.seed(0);
            m_segmented.rewind(path_id);
        } else {
            m_source->rewind(path_id);
        }
    }

  private:
    VertexSource *m_source;
    double m_scale;
    double m_length;
    double m_randomness;
    agg::conv_segmentator<VertexSource> m_segmented;
    double m_last_x;
    double m_last_y;
    bool m_has_last;
    double m_p;
    RandomNumberGenerator m_rand;
};

#endif // __PATH_CONVERTERS_H__
