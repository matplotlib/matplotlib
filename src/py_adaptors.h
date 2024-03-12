/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_PY_ADAPTORS_H
#define MPL_PY_ADAPTORS_H
#define PY_SSIZE_T_CLEAN
/***************************************************************************
 * This module contains a number of C++ classes that adapt Python data
 * structures to C++ and Agg-friendly interfaces.
 */

#include <Python.h>

#include "numpy/arrayobject.h"

#include "agg_basics.h"
#include "py_exceptions.h"

extern "C" {
int convert_path(PyObject *obj, void *pathp);
}

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
    PyArrayObject *m_vertices;
    PyArrayObject *m_codes;

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
        : m_vertices(NULL),
          m_codes(NULL),
          m_iterator(0),
          m_total_vertices(0),
          m_should_simplify(false),
          m_simplify_threshold(1.0 / 9.0)
    {
    }

    inline PathIterator(PyObject *vertices,
                        PyObject *codes,
                        bool should_simplify,
                        double simplify_threshold)
        : m_vertices(NULL), m_codes(NULL), m_iterator(0)
    {
        if (!set(vertices, codes, should_simplify, simplify_threshold))
            throw mpl::exception();
    }

    inline PathIterator(PyObject *vertices, PyObject *codes)
        : m_vertices(NULL), m_codes(NULL), m_iterator(0)
    {
        if (!set(vertices, codes))
            throw mpl::exception();
    }

    inline PathIterator(const PathIterator &other)
    {
        Py_XINCREF(other.m_vertices);
        m_vertices = other.m_vertices;

        Py_XINCREF(other.m_codes);
        m_codes = other.m_codes;

        m_iterator = 0;
        m_total_vertices = other.m_total_vertices;

        m_should_simplify = other.m_should_simplify;
        m_simplify_threshold = other.m_simplify_threshold;
    }

    ~PathIterator()
    {
        Py_XDECREF(m_vertices);
        Py_XDECREF(m_codes);
    }

    inline int
    set(PyObject *vertices, PyObject *codes, bool should_simplify, double simplify_threshold)
    {
        m_should_simplify = should_simplify;
        m_simplify_threshold = simplify_threshold;

        Py_XDECREF(m_vertices);
        m_vertices = (PyArrayObject *)PyArray_FromObject(vertices, NPY_DOUBLE, 2, 2);

        if (!m_vertices || PyArray_DIM(m_vertices, 1) != 2) {
            PyErr_SetString(PyExc_ValueError, "Invalid vertices array");
            return 0;
        }

        Py_XDECREF(m_codes);
        m_codes = NULL;

        if (codes != NULL && codes != Py_None) {
            m_codes = (PyArrayObject *)PyArray_FromObject(codes, NPY_UINT8, 1, 1);

            if (!m_codes || PyArray_DIM(m_codes, 0) != PyArray_DIM(m_vertices, 0)) {
                PyErr_SetString(PyExc_ValueError, "Invalid codes array");
                return 0;
            }
        }

        m_total_vertices = (unsigned)PyArray_DIM(m_vertices, 0);
        m_iterator = 0;

        return 1;
    }

    inline int set(PyObject *vertices, PyObject *codes)
    {
        return set(vertices, codes, false, 0.0);
    }

    inline unsigned vertex(double *x, double *y)
    {
        if (m_iterator >= m_total_vertices) {
            *x = 0.0;
            *y = 0.0;
            return agg::path_cmd_stop;
        }

        const size_t idx = m_iterator++;

        char *pair = (char *)PyArray_GETPTR2(m_vertices, idx, 0);
        *x = *(double *)pair;
        *y = *(double *)(pair + PyArray_STRIDE(m_vertices, 1));

        if (m_codes != NULL) {
            return (unsigned)(*(char *)PyArray_GETPTR1(m_codes, idx));
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
        return m_codes != NULL;
    }

    inline void *get_id()
    {
        return (void *)m_vertices;
    }
};

class PathGenerator
{
    PyObject *m_paths;
    Py_ssize_t m_npaths;

  public:
    typedef PathIterator path_iterator;

    PathGenerator() : m_paths(NULL), m_npaths(0) {}

    ~PathGenerator()
    {
        Py_XDECREF(m_paths);
    }

    int set(PyObject *obj)
    {
        if (!PySequence_Check(obj)) {
            return 0;
        }

        Py_XDECREF(m_paths);
        m_paths = obj;
        Py_INCREF(m_paths);

        m_npaths = PySequence_Size(m_paths);

        return 1;
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
        PyObject *item;

        item = PySequence_GetItem(m_paths, i % m_npaths);
        if (item == NULL) {
            throw mpl::exception();
        }
        if (!convert_path(item, &path)) {
            Py_DECREF(item);
            throw mpl::exception();
        }
        Py_DECREF(item);
        return path;
    }
};
}

#endif
