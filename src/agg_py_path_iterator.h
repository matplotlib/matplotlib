#ifndef __AGG_PY_PATH_ITERATOR_H__
#define __AGG_PY_PATH_ITERATOR_H__

#include "CXX/Objects.hxx"
#define PY_ARRAY_TYPES_PREFIX NumPy
#include "numpy/arrayobject.h"
#include "agg_path_storage.h"
#include "MPL_isnan.h"

class PathIterator
{
    PyArrayObject* m_vertices;
    PyArrayObject* m_codes;
    size_t m_iterator;
    size_t m_total_vertices;

public:
    PathIterator(const Py::Object& path_obj) :
            m_vertices(NULL), m_codes(NULL), m_iterator(0)
    {
        Py::Object vertices_obj = path_obj.getAttr("vertices");
        Py::Object codes_obj = path_obj.getAttr("codes");

        m_vertices = (PyArrayObject*)PyArray_FromObject
                     (vertices_obj.ptr(), PyArray_DOUBLE, 2, 2);
        if (!m_vertices || PyArray_NDIM(m_vertices) != 2 || PyArray_DIM(m_vertices, 1) != 2)
            throw Py::ValueError("Invalid vertices array.");

        if (codes_obj.ptr() != Py_None)
        {
            m_codes = (PyArrayObject*)PyArray_FromObject
                      (codes_obj.ptr(), PyArray_UINT8, 1, 1);
            if (!m_codes)
                throw Py::ValueError("Invalid codes array.");
        }

        m_total_vertices = m_vertices->dimensions[0];
    }

    ~PathIterator()
    {
        Py_XDECREF(m_vertices);
        Py_XDECREF(m_codes);
    }

    static const unsigned code_map[];

private:
    inline unsigned vertex(unsigned idx, double* x, double* y)
    {
        char* pair = (char*)PyArray_GETPTR2(m_vertices, idx, 0);
        *x = *(double*)pair;
        *y = *(double*)(pair + PyArray_STRIDE(m_vertices, 1));
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
        unsigned code = vertex(m_iterator++, x, y);
        while ((MPL_isnan64(*x) || MPL_isnan64(*y)) &&
               m_iterator < m_total_vertices) {
          vertex(m_iterator++, x, y);
          code = agg::path_cmd_move_to;
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

    inline bool has_curves()
    {
        return m_codes;
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

#endif // __AGG_PY_PATH_ITERATOR_H__
