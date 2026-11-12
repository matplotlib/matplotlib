/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_AGG_PATH_ADAPTORS_H
#define MPL_AGG_PATH_ADAPTORS_H
#define PY_SSIZE_T_CLEAN
/***************************************************************************
 * This module contains a number of C++ classes that adapt Python data
 * structures to C++ and Agg-friendly interfaces.
 */

#include <nanobind/nanobind.h>

#include "agg_basics.h"
#include "mpl_array.h"

namespace mpl {

class PathIterator;
inline void convert(PathIterator &dst, const nb::object &src, std::string arg_name);

namespace nb = nanobind;

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
public:
    using VerticesArray = mpl::array<const double, mpl::shape<-1, 2>>;
    using CodesArray = mpl::array<const uint8_t, mpl::ndim<1>>;

    inline PathIterator()
        : m_iterator(0),
          m_total_vertices(0),
          m_should_simplify(false),
          m_simplify_threshold(1.0 / 9.0)
    {
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
    set(VerticesArray &vertices, std::optional<CodesArray> &codes,
        bool should_simplify, double simplify_threshold)
    {
        m_vertices = vertices;
        m_codes = codes;

        m_iterator = 0;
        m_total_vertices = m_vertices.shape(0);

        m_should_simplify = should_simplify;
        m_simplify_threshold = simplify_threshold;
    }

    inline unsigned vertex(double *x, double *y)
    {
        if (m_iterator >= m_total_vertices) {
            *x = 0.0;
            *y = 0.0;
            return agg::path_cmd_stop;
        }

        const size_t idx = m_iterator++;

        auto data = m_vertices.data();
        *x = data[idx*2];
        *y = data[idx*2 + 1];

        if (m_codes) {
            return m_codes.value().data()[idx];
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
        return (void *)m_vertices.data();
    }

  private:
    /* We hold references to the Python objects, not just the
       underlying data arrays, so that Python reference counting
       can work.
    */
    VerticesArray m_vertices = VerticesArray::unmanaged_empty();
    std::optional<CodesArray> m_codes;

    unsigned m_iterator;
    unsigned m_total_vertices;

    /* This class doesn't actually do any simplification, but we
       store the value here, since it is obtained from the Python
       object.
    */
    bool m_should_simplify;
    double m_simplify_threshold;
};

class PathGenerator
{
    nb::sequence m_paths;
    size_t m_npaths;

  public:
    typedef PathIterator path_iterator;

    PathGenerator() : m_npaths(0) {}

    void set(nb::sequence &seq)
    {
        m_paths = seq;
        m_npaths = nb::len(m_paths);
    }

    size_t num_paths() const
    {
        return m_npaths;
    }

    size_t size() const
    {
        return m_npaths;
    }

    path_iterator operator()(size_t i)
    {
        PathIterator path;
        try {
            convert(path, m_paths[i % m_npaths], "");
        } catch (...) {
            throw nb::value_error(mpl::format("Invalid path at index {}", i));
        }
        return path;
    }
};


inline void convert(PathIterator &dst, const nb::object &src, std::string arg_name)
{
    if (src.is_none()) {
        return;
    }

    nb::object vertices_obj;
    nb::object codes_obj;
    nb::object should_simplify_obj;
    nb::object simplify_threshold_obj;

    try {
        vertices_obj = src.attr("vertices");
        codes_obj = src.attr("codes");
        should_simplify_obj = src.attr("should_simplify");
        simplify_threshold_obj = src.attr("simplify_threshold");
    } catch (...) {
        throw std::invalid_argument(mpl::format("{} is not a valid path", arg_name));
    }

    auto vertices = PathIterator::VerticesArray(vertices_obj, arg_name + ".vertices");

    std::optional<PathIterator::CodesArray> codes;
    if (!codes_obj.is_none()) {
        codes = PathIterator::CodesArray(codes_obj, arg_name + ".codes");
    }

    double simplify_threshold;
    cast(simplify_threshold, simplify_threshold_obj, arg_name + ".simplify_threshold");

    dst.set(vertices, codes, mpl::is_true(should_simplify_obj), simplify_threshold);
}


inline void convert(PathGenerator &dst, const nb::object &src, std::string arg_name)
{
    nb::sequence seq;
    if (nb::try_cast(src, seq)) {
        dst.set(seq);
    } else {
        throw nb::type_error(mpl::format("{} is not a sequence of paths", arg_name));
    }
}

} // namespace mpl

#endif // MPL_AGG_PATH_ADAPTORS_H
