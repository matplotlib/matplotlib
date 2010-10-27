/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef __AGG_PY_PATH_ITERATOR_H__
#define __AGG_PY_PATH_ITERATOR_H__

#include "CXX/Objects.hxx"
#include "numpy/arrayobject.h"
#include "agg_path_storage.h"

/*
 This file contains a vertex source to adapt Python Numpy arrays to
 Agg paths.  It works as an iterator, and converts on-the-fly without
 the need for a full copy of the data.
 */

/************************************************************
 PathIterator acts as a bridge between Numpy and Agg.  Given a pair of
 Numpy arrays, vertices and codes, it iterates over those vertices and
 codes, using the standard Agg vertex source interface:

    unsigned vertex(double* x, double* y)
 */
class PathIterator
{
    /* We hold references to the Python objects, not just the
       underlying data arrays, so that Python reference counting can
       work.
    */
    Py::Object m_vertices;
    Py::Object m_codes;

    size_t m_iterator;
    size_t m_total_vertices;

    /* This class doesn't actually do any simplification, but we
       store the value here, since it is obtained from the Python object.
    */
    bool m_should_simplify;
    double m_simplify_threshold;

public:
    /* path_obj is an instance of the class Path as defined in path.py */
    inline PathIterator(const Py::Object& path_obj) :
            m_vertices(), m_codes(), m_iterator(0), m_should_simplify(false),
            m_simplify_threshold(1.0 / 9.0)
    {
        Py::Object vertices_obj           = path_obj.getAttr("vertices");
        Py::Object codes_obj              = path_obj.getAttr("codes");
        Py::Object should_simplify_obj    = path_obj.getAttr("should_simplify");
        Py::Object simplify_threshold_obj = path_obj.getAttr("simplify_threshold");

        PyObject* vertices_arr = PyArray_FromObject(vertices_obj.ptr(), PyArray_DOUBLE, 2, 2);
        if (!vertices_arr)
        {
            throw Py::ValueError("Invalid vertices array.");
        }

        m_vertices = Py::Object(vertices_arr, true);
        if (PyArray_DIM(m_vertices.ptr(), 1) != 2)
        {
            throw Py::ValueError("Invalid vertices array.");
        }

        if (codes_obj.ptr() != Py_None)
        {
            PyObject* codes_arr = PyArray_FromObject(codes_obj.ptr(), PyArray_UINT8, 1, 1);

            if (!codes_arr)
            {
                throw Py::ValueError("Invalid codes array.");
            }

            m_codes = Py::Object(codes_arr, true);
            if (PyArray_DIM(m_codes.ptr(), 0) != PyArray_DIM(m_vertices.ptr(), 0))
            {
                throw Py::ValueError("Codes array is wrong length");
            }
        }

        m_should_simplify    = should_simplify_obj.isTrue();
        m_total_vertices     = PyArray_DIM(m_vertices.ptr(), 0);
        m_simplify_threshold = Py::Float(simplify_threshold_obj);
    }

    ~PathIterator()
    {

    }

    inline unsigned vertex(double* x, double* y)
    {
        if (m_iterator >= m_total_vertices) return agg::path_cmd_stop;

        const size_t idx = m_iterator++;

        char* pair = (char*)PyArray_GETPTR2(m_vertices.ptr(), idx, 0);
        *x = *(double*)pair;
        *y = *(double*)(pair + PyArray_STRIDE(m_vertices.ptr(), 1));

        if (!m_codes.isNone())
        {
            return (unsigned)(*(char *)PyArray_GETPTR1(m_codes.ptr(), idx));
        }
        else
        {
            return idx == 0 ? agg::path_cmd_move_to : agg::path_cmd_line_to;
        }
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

    inline double simplify_threshold()
    {
        return m_simplify_threshold;
    }

    inline bool has_curves()
    {
        return !m_codes.isNone();
    }
};

#endif // __AGG_PY_PATH_ITERATOR_H__
