/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_PY_ADAPTORS_H
#define MPL_PY_ADAPTORS_H
#define PY_SSIZE_T_CLEAN
/***************************************************************************
 * This module contains a number of C++ classes that adapt Python data
 * structures to C++ and Agg-friendly interfaces.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "agg_basics.h"

namespace py = pybind11;

namespace mpl {

/************************************************************
 * mpl::PathIterator acts as a bridge between NumPy and Agg.  Given a
 * pair of NumPy arrays, vertices and codes, it iterates over
 * those vertices and codes, using the standard Agg vertex source
 * interface:
 *
 *     unsigned vertex(double* x, double* y)
 */
class PathIterator
{
    /* We hold references to the Python objects, not just the
       underlying data arrays, so that Python reference counting
       can work.
    */
    py::array_t<double> m_vertices;
    py::array_t<uint8_t> m_codes;

    unsigned m_iterator;
    unsigned m_total_vertices;

    /* This class doesn't actually do any simplification, but we
       store the value here, since it is obtained from the Python
       object.
    */
    bool m_should_simplify;
    double m_simplify_threshold;

  public:
    inline PathIterator()
        : m_iterator(0),
          m_total_vertices(0),
          m_should_simplify(false),
          m_simplify_threshold(1.0 / 9.0)
    {
    }

    inline PathIterator(py::object vertices, py::object codes, bool should_simplify,
                        double simplify_threshold)
        : m_iterator(0)
    {
        set(vertices, codes, should_simplify, simplify_threshold);
    }

    inline PathIterator(py::object vertices, py::object codes)
        : m_iterator(0)
    {
        set(vertices, codes);
    }

    inline PathIterator(const PathIterator &other)
    {
        m_vertices = other.m_vertices;
        m_codes = other.m_codes;

        m_iterator = 0;
        m_total_vertices = other.m_total_vertices;

        m_should_simplify = other.m_should_simplify;
        m_simplify_threshold = other.m_simplify_threshold;
    }

    inline void
    set(py::object vertices, py::object codes, bool should_simplify, double simplify_threshold)
    {
        m_should_simplify = should_simplify;
        m_simplify_threshold = simplify_threshold;

        m_vertices = vertices.cast<py::array_t<double>>();
        if (m_vertices.ndim() != 2 || m_vertices.shape(1) != 2) {
            throw py::value_error("Invalid vertices array");
        }
        m_total_vertices = m_vertices.shape(0);

        m_codes.release().dec_ref();
        if (!codes.is_none()) {
            m_codes = codes.cast<py::array_t<uint8_t>>();
            if (m_codes.ndim() != 1 || m_codes.shape(0) != m_total_vertices) {
                throw py::value_error("Invalid codes array");
            }
        }

        m_iterator = 0;
    }

    inline void set(py::object vertices, py::object codes)
    {
        set(vertices, codes, false, 0.0);
    }

    inline unsigned vertex(double *x, double *y)
    {
        if (m_iterator >= m_total_vertices) {
            *x = 0.0;
            *y = 0.0;
            return agg::path_cmd_stop;
        }

        const size_t idx = m_iterator++;

        *x = *m_vertices.data(idx, 0);
        *y = *m_vertices.data(idx, 1);

        if (m_codes) {
            return *m_codes.data(idx);
        } else {
            return idx == 0 ? agg::path_cmd_move_to : agg::path_cmd_line_to;
        }
    }

    inline void rewind(unsigned path_id)
    {
        m_iterator = path_id;
    }

    inline unsigned total_vertices() const
    {
        return m_total_vertices;
    }

    inline bool should_simplify() const
    {
        return m_should_simplify;
    }

    inline double simplify_threshold() const
    {
        return m_simplify_threshold;
    }

    inline bool has_codes() const
    {
        return bool(m_codes);
    }

    inline void *get_id()
    {
        return (void *)m_vertices.ptr();
    }
};

class PathGenerator
{
    py::sequence m_paths;
    Py_ssize_t m_npaths;

  public:
    typedef PathIterator path_iterator;

    PathGenerator() : m_npaths(0) {}

    void set(py::object obj)
    {
        m_paths = obj.cast<py::sequence>();
        m_npaths = m_paths.size();
    }

    Py_ssize_t num_paths() const
    {
        return m_npaths;
    }

    Py_ssize_t size() const
    {
        return m_npaths;
    }

    path_iterator operator()(size_t i)
    {
        path_iterator path;

        auto item = m_paths[i % m_npaths];
        path = item.cast<path_iterator>();
        return path;
    }
};
}

namespace PYBIND11_NAMESPACE { namespace detail {
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

            value.set(vertices, codes, should_simplify, simplify_threshold);

            return true;
        }
    };

    template <> struct type_caster<mpl::PathGenerator> {
    public:
        PYBIND11_TYPE_CASTER(mpl::PathGenerator, const_name("PathGenerator"));

        bool load(handle src, bool) {
            value.set(py::reinterpret_borrow<py::object>(src));
            return true;
        }
    };
}} // namespace PYBIND11_NAMESPACE::detail

#endif
