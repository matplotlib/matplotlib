#ifndef __AGG_PY_PATH_ITERATOR_H__
#define __AGG_PY_PATH_ITERATOR_H__

#include "CXX/Objects.hxx"
#define PY_ARRAY_TYPES_PREFIX NumPy
#include "numpy/arrayobject.h"
#include "agg_path_storage.h"
#include "MPL_isnan.h"
#include "mplutils.h"
#include <queue>

class PathIterator
{
    PyArrayObject* m_vertices;
    PyArrayObject* m_codes;
    size_t m_iterator;
    size_t m_total_vertices;
    bool m_should_simplify;

public:
    PathIterator(const Py::Object& path_obj) :
    m_vertices(NULL), m_codes(NULL), m_iterator(0), m_should_simplify(false)
    {
        Py::Object vertices_obj = path_obj.getAttr("vertices");
        Py::Object codes_obj = path_obj.getAttr("codes");
        Py::Object should_simplify_obj = path_obj.getAttr("should_simplify");

        m_vertices = (PyArrayObject*)PyArray_FromObject
                     (vertices_obj.ptr(), PyArray_DOUBLE, 2, 2);
        if (!m_vertices ||
            PyArray_DIM(m_vertices, 1) != 2)
        {
            throw Py::ValueError("Invalid vertices array.");
        }

        if (codes_obj.ptr() != Py_None)
        {
            m_codes = (PyArrayObject*)PyArray_FromObject
                      (codes_obj.ptr(), PyArray_UINT8, 1, 1);
            if (!m_codes)
                throw Py::ValueError("Invalid codes array.");
            if (PyArray_DIM(m_codes, 0) != PyArray_DIM(m_vertices, 0))
                throw Py::ValueError("Codes array is wrong length");
        }

        m_should_simplify = should_simplify_obj.isTrue();
        m_total_vertices = m_vertices->dimensions[0];
    }

    ~PathIterator()
    {
        Py_XDECREF(m_vertices);
        Py_XDECREF(m_codes);
    }

    static const unsigned code_map[];

private:
    inline void vertex(const unsigned idx, double* x, double* y)
    {
        char* pair = (char*)PyArray_GETPTR2(m_vertices, idx, 0);
        *x = *(double*)pair;
        *y = *(double*)(pair + PyArray_STRIDE(m_vertices, 1));
    }

    inline unsigned vertex_with_code(const unsigned idx, double* x, double* y)
    {
        vertex(idx, x, y);
        if (m_codes)
        {
            return code_map[(int)*(char *)PyArray_GETPTR1(m_codes, idx)];
        }
        else
        {
            return idx == 0 ? agg::path_cmd_move_to : agg::path_cmd_line_to;
        }
    }

public:
    inline unsigned vertex(double* x, double* y)
    {
        if (m_iterator >= m_total_vertices) return agg::path_cmd_stop;
        unsigned code = vertex_with_code(m_iterator++, x, y);

        if (MPL_notisfinite64(*x) || MPL_notisfinite64(*y))
        {
            do
            {
                if (m_iterator < m_total_vertices)
                {
                    vertex(m_iterator++, x, y);
                }
                else
                {
                    return agg::path_cmd_stop;
                }
            } while (MPL_notisfinite64(*x) || MPL_notisfinite64(*y));
            return agg::path_cmd_move_to;
        }

        return code;
    }

    inline void rewind(unsigned path_id)
    {
        m_iterator = path_id;
    }

    inline unsigned total_vertices()
    {
        return m_total_vertices;
    }

    inline bool should_simplify()
    {
        return m_should_simplify;
    }
};

// Maps path codes on the Python side to agg path commands
const unsigned PathIterator::code_map[] =
    {0,
     agg::path_cmd_move_to,
     agg::path_cmd_line_to,
     agg::path_cmd_curve3,
     agg::path_cmd_curve4,
     agg::path_cmd_end_poly | agg::path_flags_close
    };

#define DEBUG_SIMPLIFY 0

template<class VertexSource>
class SimplifyPath
{
public:
    SimplifyPath(VertexSource& source, bool quantize, bool simplify,
                 double width = 0.0, double height = 0.0) :
            m_source(&source), m_quantize(quantize), m_simplify(simplify),
            m_width(width + 1.0), m_height(height + 1.0), m_queue_read(0), m_queue_write(0),
            m_moveto(true), m_lastx(0.0), m_lasty(0.0), m_clipped(false),
            m_do_clipping(width > 0.0 && height > 0.0),
            m_origdx(0.0), m_origdy(0.0),
            m_origdNorm2(0.0), m_dnorm2Max(0.0), m_dnorm2Min(0.0),
            m_haveMin(false), m_lastMax(false), m_maxX(0.0), m_maxY(0.0),
            m_minX(0.0), m_minY(0.0), m_lastWrittenX(0.0), m_lastWrittenY(0.0),
            m_done(false)
#if DEBUG_SIMPLIFY
            , m_pushed(0), m_skipped(0)
#endif
    {
        // empty
    }

#if DEBUG_SIMPLIFY
    ~SimplifyPath()
    {
        if (m_simplify)
            printf("%d %d\n", m_pushed, m_skipped);
    }
#endif

    void rewind(unsigned path_id)
    {
        m_source->rewind(path_id);
    }

    unsigned vertex(double* x, double* y)
    {
        unsigned cmd;

        // The simplification algorithm doesn't support curves or compound paths
        // so we just don't do it at all in that case...
        if (!m_simplify)
        {
            cmd = m_source->vertex(x, y);
            if (m_quantize && agg::is_vertex(cmd))
            {
                *x = mpl_round(*x) + 0.5;
                *y = mpl_round(*y) + 0.5;
            }
            return cmd;
        }

        //idea: we can skip drawing many lines: lines < 1 pixel in length, lines
        //outside of the drawing area, and we can combine sequential parallel lines
        //into a single line instead of redrawing lines over the same points.
        //The loop below works a bit like a state machine, where what it does depends
        //on what it did in the last looping. To test whether sequential lines
        //are close to parallel, I calculate the distance moved perpendicular to the
        //last line. Once it gets too big, the lines cannot be combined.

        // This code was originally written by someone else (John Hunter?) and I
        // have modified to work in-place -- meaning not creating an entirely
        // new path list each time.  In order to do that without too much
        // additional code complexity, it keeps a small queue around so that
        // multiple points can be emitted in a single call, and those points
        // will be popped from the queue in subsequent calls.  The following
        // block will empty the queue before proceeding to the main loop below.
        //  -- Michael Droettboom
        if (m_queue_read < m_queue_write)
        {
            const item& front = m_queue[m_queue_read++];
            unsigned cmd = front.cmd;
            *x = front.x;
            *y = front.y;
#if DEBUG_SIMPLIFY
            printf((cmd == agg::path_cmd_move_to) ? "|" : "-");
#endif
            return cmd;
        }

        m_queue_read = 0;
        m_queue_write = 0;

        // If the queue is now empty, and the path was fully consumed
        // in the last call to the main loop, return agg::path_cmd_stop to
        // signal that there are no more points to emit.
        if (m_done)
        {
#if DEBUG_SIMPLIFY
            printf(".\n");
#endif
            return agg::path_cmd_stop;
        }

        // The main simplification loop.  The point is to consume only as many
        // points as necessary until something has been added to the outbound
        // queue, not to run through the entire path in one go.  This
        // eliminates the need to allocate and fill an entire additional path
        // array on each draw.
        while ((cmd = m_source->vertex(x, y)) != agg::path_cmd_stop)
        {
            // Do any quantization if requested
            if (m_quantize && agg::is_vertex(cmd))
            {
                *x = mpl_round(*x) + 0.5;
                *y = mpl_round(*y) + 0.5;
            }

            //if we are starting a new path segment, move to the first point
            // + init
            if (m_moveto)
            {
                m_lastx = *x;
                m_lasty = *y;
                m_moveto = false;
                m_origdNorm2 = 0.0;
#if DEBUG_SIMPLIFY
                m_pushed++;
                printf("|");
#endif
                return agg::path_cmd_move_to;
            }

            // Don't render line segments less than one pixel long
            if (fabs(*x - m_lastx) < 1.0 && fabs(*y - m_lasty) < 1.0)
            {
#if DEBUG_SIMPLIFY
                m_skipped++;
#endif
                continue;
            }

            //skip any lines that are outside the drawing area. Note: More lines
            //could be clipped, but a more involved calculation would be needed
            if (m_do_clipping &&
                ((*x < -1.0 && m_lastx < -1.0) ||
                 (*x > m_width && m_lastx > m_width) ||
                 (*y < -1.0 && m_lasty < -1.0) ||
                 (*y > m_height && m_lasty > m_height)))
            {
                m_lastx = *x;
                m_lasty = *y;
                m_clipped = true;
#if DEBUG_SIMPLIFY
                m_skipped++;
#endif
                continue;
            }

            // if we have no orig vector, set it to this vector and
            // continue.
            // this orig vector is the reference vector we will build
            // up the line to

            if (m_origdNorm2 == 0)
            {
                if (m_clipped)
                {
                    m_queue[m_queue_write++].set(agg::path_cmd_move_to, m_lastx, m_lasty);
                    m_clipped = false;
                }

                m_origdx = *x - m_lastx;
                m_origdy = *y - m_lasty;
                m_origdNorm2 = m_origdx*m_origdx + m_origdy*m_origdy;

                //set all the variables to reflect this new orig vecor
                m_dnorm2Max = m_origdNorm2;
                m_dnorm2Min = 0.0;
                m_haveMin = false;
                m_lastMax = true;

                m_lastx = m_maxX = *x;
                m_lasty = m_maxY = *y;
                m_lastWrittenX = m_minX = m_lastx;
                m_lastWrittenY = m_minY = m_lasty;
#if DEBUG_SIMPLIFY
                m_skipped++;
#endif
                continue;
            }

            //if got to here, then we have an orig vector and we just got
            //a vector in the sequence.

            //check that the perpendicular distance we have moved from the
            //last written point compared to the line we are building is not too
            //much. If o is the orig vector (we are building on), and v is the
            //vector from the last written point to the current point, then the
            //perpendicular vector is  p = v - (o.v)o,  and we normalize o  (by
            //dividing the second term by o.o).

            // get the v vector
            double totdx = *x - m_lastWrittenX;
            double totdy = *y - m_lastWrittenY;
            double totdot = m_origdx*totdx + m_origdy*totdy;

            // get the para vector ( = (o.v)o/(o.o))
            double paradx = totdot*m_origdx/m_origdNorm2;
            double parady = totdot*m_origdy/m_origdNorm2;

            // get the perp vector ( = v - para)
            double perpdx = totdx - paradx;
            double perpdy = totdy - parady;
            double perpdNorm2 = perpdx*perpdx + perpdy*perpdy;

            //if the perp vector is less than some number of (squared)
            //pixels in size, then merge the current vector
            if (perpdNorm2 < 0.25)
            {
                //check if the current vector is parallel or
                //anti-parallel to the orig vector. If it is parallel, test
                //if it is the longest of the vectors we are merging in that
                //direction. If anti-p, test if it is the longest in the
                //opposite direction (the min of our final line)

                double paradNorm2 = paradx*paradx + parady*parady;

                m_lastMax = false;
                if (totdot >= 0)
                {
                    if (paradNorm2 > m_dnorm2Max)
                    {
                        m_lastMax = true;
                        m_dnorm2Max = paradNorm2;
                        m_maxX = m_lastWrittenX + paradx;
                        m_maxY = m_lastWrittenY + parady;
                    }
                }
                else
                {
                    m_haveMin = true;
                    if (paradNorm2 > m_dnorm2Min)
                    {
                        m_dnorm2Min = paradNorm2;
                        m_minX = m_lastWrittenX + paradx;
                        m_minY = m_lastWrittenY + parady;
                    }
                }

                m_lastx = *x;
                m_lasty = *y;
#if DEBUG_SIMPLIFY
                m_skipped++;
#endif
                continue;
            }

            //if we get here, then this vector was not similar enough to the
            //line we are building, so we need to draw that line and start the
            //next one.

            //if the line needs to extend in the opposite direction from the
            //direction we are drawing in, move back to we start drawing from
            //back there.
            if (m_haveMin)
            {
                m_queue[m_queue_write++].set(agg::path_cmd_line_to, m_minX, m_minY);
            }
            m_queue[m_queue_write++].set(agg::path_cmd_line_to, m_maxX, m_maxY);

            //if we clipped some segments between this line and the next line
            //we are starting, we also need to move to the last point.
            if (m_clipped) {
                m_queue[m_queue_write++].set(agg::path_cmd_move_to, m_lastx, m_lasty);
            }
            else if (!m_lastMax)
            {
                //if the last line was not the longest line, then move back to
                //the end point of the last line in the sequence. Only do this
                //if not clipped, since in that case lastx,lasty is not part of
                //the line just drawn.

                //Would be move_to if not for the artifacts
                m_queue[m_queue_write++].set(agg::path_cmd_line_to, m_lastx, m_lasty);
            }

            //now reset all the variables to get ready for the next line
            m_origdx = *x - m_lastx;
            m_origdy = *y - m_lasty;
            m_origdNorm2 = m_origdx*m_origdx + m_origdy*m_origdy;

            m_dnorm2Max = m_origdNorm2;
            m_dnorm2Min = 0.0;
            m_haveMin = false;
            m_lastMax = true;
            m_lastx = m_maxX = *x;
            m_lasty = m_maxY = *y;
            m_lastWrittenX = m_minX = m_lastx;
            m_lastWrittenY = m_minY = m_lasty;

            m_clipped = false;
#if DEBUG_SIMPLIFY
            m_pushed += m_queue_write - m_queue_read;
#endif
            break;
        }

        // Fill the queue with the remaining vertices if we've finished the
        // path in the above loop.  Mark the path as done, so we don't call
        // m_source->vertex again and segfault.
        if (cmd == agg::path_cmd_stop)
        {
            if (m_origdNorm2 != 0)
            {
                if (m_haveMin)
                {
                    m_queue[m_queue_write++].set(agg::path_cmd_line_to, m_minX, m_minY);
                }
                m_queue[m_queue_write++].set(agg::path_cmd_line_to, m_maxX, m_maxY);
            }
            m_done = true;
        }

        // Return the first item in the queue, if any, otherwise
        // indicate that we're done.
        if (m_queue_read < m_queue_write)
        {
            const item& front = m_queue[m_queue_read++];
            unsigned cmd = front.cmd;
            *x = front.x;
            *y = front.y;
#if DEBUG_SIMPLIFY
            printf((cmd == agg::path_cmd_move_to) ? "|" : "-");
#endif
            return cmd;
        }
        else
        {
#if DEBUG_SIMPLIFY
            printf(".\n");
#endif
            return agg::path_cmd_stop;
        }
    }

private:
    VertexSource* m_source;
    bool m_quantize;
    bool m_simplify;
    double m_width, m_height;

    struct item
    {
        item() {}
        inline void set(const unsigned cmd_, const double& x_, const double& y_)
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
    item m_queue[6];

    bool m_moveto;
    double m_lastx, m_lasty;
    bool m_clipped;
    bool m_do_clipping;

    double m_origdx;
    double m_origdy;
    double m_origdNorm2;
    double m_dnorm2Max;
    double m_dnorm2Min;
    bool m_haveMin;
    bool m_lastMax;
    double m_maxX;
    double m_maxY;
    double m_minX;
    double m_minY;
    double m_lastWrittenX;
    double m_lastWrittenY;
    bool m_done;

#if DEBUG_SIMPLIFY
    unsigned m_pushed;
    unsigned m_skipped;
#endif
};

#endif // __AGG_PY_PATH_ITERATOR_H__
