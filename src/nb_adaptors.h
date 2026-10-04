/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_PY_ADAPTORS_H
#define MPL_PY_ADAPTORS_H
#define PY_SSIZE_T_CLEAN
/***************************************************************************
 * This module contains a number of C++ classes that adapt Python data
 * structures to C++ and Agg-friendly interfaces.
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cstdint>
#include "agg_basics.h"

namespace nb = nanobind;

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
    nb::ndarray<double, nb::numpy, nb::c_contig> m_vertices;
    nb::ndarray<uint8_t, nb::numpy, nb::c_contig> m_codes;

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

    inline PathIterator(nb::object vertices, nb::object codes, bool should_simplify,
                        double simplify_threshold)
        : m_iterator(0)
    {
        set(vertices, codes, should_simplify, simplify_threshold);
    }

    inline PathIterator(nb::object vertices, nb::object codes)
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
    set(nb::object vertices, nb::object codes, bool should_simplify, double simplify_threshold)
    {
        m_should_simplify = should_simplify;
        m_simplify_threshold = simplify_threshold;

        m_vertices = nb::cast<nb::ndarray<double, nb::c_contig>>(vertices);
        if (m_vertices.ndim() != 2 || m_vertices.shape(1) != 2) {
            throw nb::value_error("Invalid vertices array");
        }
        m_total_vertices = m_vertices.shape(0);

        m_codes = nb::ndarray<uint8_t, nb::c_contig>();
        if (!codes.is_none()) {
            m_codes = nb::cast<nb::ndarray<uint8_t, nb::c_contig>>(codes);
            if (m_codes.ndim() != 1 || m_codes.shape(0) != m_total_vertices) {
                throw nb::value_error("Invalid codes array");
            }
        }

        m_iterator = 0;
    }

    inline void set(nb::object vertices, nb::object codes)
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

        *x = m_vertices.data()[idx * 2];
        *y = m_vertices.data()[idx * 2 + 1];

        if (m_codes.is_valid()) {
            return m_codes.data()[idx];
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
        return m_codes.is_valid();
    }

    inline void *get_id()
    {
        return (void *)m_vertices.data();
    }
};

class PathGenerator
{
    nb::object m_paths;
    Py_ssize_t m_npaths;

  public:
    typedef PathIterator path_iterator;

    PathGenerator() : m_npaths(0) {}

    void set(nb::object obj)
    {
        m_paths = obj;
        m_npaths = nb::len(m_paths);
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

        nb::object item = m_paths[nb::int_(i % m_npaths)];
        path = nb::cast<path_iterator>(item);
        return path;
    }
};
}

namespace nanobind { namespace detail {
    template <> struct type_caster<mpl::PathIterator> {
    public:
        NB_TYPE_CASTER(mpl::PathIterator, const_name("PathIterator"))

        bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
            if (src.is_none()) {
                return true;
            }

            try {
                object vertices = src.attr("vertices");
                object codes = src.attr("codes");
                bool should_simplify = cast<bool>(src.attr("should_simplify"));
                double simplify_threshold = cast<double>(src.attr("simplify_threshold"));

                value.set(vertices, codes, should_simplify, simplify_threshold);
            } catch (...) {
                return false;
            }

            return true;
        }
    };

    template <> struct type_caster<mpl::PathGenerator> {
    public:
        NB_TYPE_CASTER(mpl::PathGenerator, const_name("PathGenerator"))

        bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
            try {
                value.set(borrow<object>(src));
            } catch (...) {
                return false;
            }
            return true;
        }
    };
}} // namespace nanobind::detail

#endif
