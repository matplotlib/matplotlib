/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_PY_ADAPTORS_H
#define MPL_PY_ADAPTORS_H
#define PY_SSIZE_T_CLEAN
/***************************************************************************
 * This module contains a number of C++ classes that adapt Python data
 * structures to C++ and Agg-friendly interfaces.
 */

#include <Python.h>
#include <cstdint>

#include "numpy/arrayobject.h"

#include "py_exceptions.h"

extern "C" {
int convert_path(PyObject *obj, void *pathp);
}

namespace py
{

template<typename T, size_t Dim>
class StridedMemoryBase {
protected:
    char *m_data;
    npy_intp *m_strides;

    explicit StridedMemoryBase(char *data, npy_intp *strides)
        : m_data(data), m_strides(strides)
    {
    }

public:
    StridedMemoryBase()
        : m_data(nullptr), m_strides(nullptr)
    {
    }

    explicit StridedMemoryBase(PyArrayObject *array)
    {
        if (PyArray_NDIM(array) != Dim) {
            PyErr_SetString(PyExc_ValueError, "Invalid array dimensionality");
            throw py::exception();
        }
        m_data = PyArray_BYTES(array);
        m_strides = PyArray_STRIDES(array);
    }

    operator bool() const {
        return m_data != nullptr;
    }

    void reset() {
        m_data = nullptr;
        m_strides = nullptr;
    }

    T* data() {
        return reinterpret_cast<T*>(m_data);
    }
};

#define INHERIT_CONSTRUCTORS(CLS, DIM)                                         \
  CLS() {}                                                                     \
  explicit CLS(char *data, npy_intp *strides): StridedMemoryBase<T, DIM>(data, strides) {} \
  explicit CLS(PyArrayObject *array): StridedMemoryBase<T, DIM>(array) {}


template<typename T>
class StridedMemory1D : public StridedMemoryBase<T, 1> {
public:
    INHERIT_CONSTRUCTORS(StridedMemory1D, 1)
    T operator[](size_t idx) const {
        return *reinterpret_cast<T*>(this->m_data + *this->m_strides * idx);
    }
};

template<typename T>
class StridedMemory2D : public StridedMemoryBase<T, 2> {
public:
    INHERIT_CONSTRUCTORS(StridedMemory2D, 2)
    StridedMemory1D<T> operator[](size_t idx) const {
        return StridedMemory1D<T>(this->m_data + *this->m_strides * idx,
                                  this->m_strides + 1);
    }
};

template<typename T>
class StridedMemory3D : public StridedMemoryBase<T, 3> {
public:
    INHERIT_CONSTRUCTORS(StridedMemory3D, 3)
    StridedMemory2D<T> operator[](size_t idx) const {
        return StridedMemory2D<T>(this->m_data + *this->m_strides * idx,
                                  this->m_strides + 1);
    }
};

#undef INHERIT_CONSTRUCTORS

/************************************************************
 * py::PathIterator acts as a bridge between Numpy and Agg.  Given a
 * pair of Numpy arrays, vertices and codes, it iterates over
 * those vertices and codes, using the standard Agg vertex source
 * interface:
 *
 *     unsigned vertex(double* x, double* y)
 */
class PathIterator
{
    /* XXX: This class does not own the data! It should really be used as an
       iterator, where it is the container that manages the lifetime of the
       paths.
    */
    StridedMemory2D<double> m_vertices;
    StridedMemory1D<uint8_t> m_codes;

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
        : m_vertices(),
          m_codes(),
          m_iterator(0),
          m_total_vertices(0),
          m_should_simplify(false),
          m_simplify_threshold(1.0 / 9.0)
    {
    }

    inline PathIterator(const StridedMemory2D<double>& vertices,
                        const StridedMemory1D<uint8_t>& codes,
                        unsigned total_vertices,
                        bool should_simplify,
                        double simplify_threshold)
        : m_vertices(vertices),
          m_codes(codes),
          m_iterator(0),
          m_total_vertices(total_vertices),
          m_should_simplify(should_simplify),
          m_simplify_threshold(simplify_threshold)
    {
    }

    inline PathIterator(PyObject *vertices,
                        PyObject *codes,
                        bool should_simplify,
                        double simplify_threshold)
        : m_vertices(),
          m_codes(),
          m_iterator(0)
    {
        if (!set(vertices, codes, should_simplify, simplify_threshold))
            throw py::exception();
    }

    inline PathIterator(PyObject *vertices, PyObject *codes)
        : m_vertices(),
          m_codes(),
          m_iterator(0)
    {
        if (!set(vertices, codes))
            throw py::exception();
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

    inline int
    set(PyObject *vertices, PyObject *codes, bool should_simplify, double simplify_threshold)
    {
        m_should_simplify = should_simplify;
        m_simplify_threshold = simplify_threshold;

        PyArrayObject *vertices_arr =
            (PyArrayObject *)PyArray_FromObject(vertices, NPY_DOUBLE, 2, 2);
        if (!vertices_arr || PyArray_DIM(vertices_arr, 1) != 2) {
            PyErr_SetString(PyExc_ValueError, "Invalid vertices array");
            return 0;
        }
        m_vertices = StridedMemory2D<double>(vertices_arr);

        if (codes != NULL && codes != Py_None) {
            PyArrayObject *codes_arr = (PyArrayObject *)PyArray_FromObject(codes, NPY_UINT8, 1, 1);
            if (!codes_arr || PyArray_DIM(codes_arr, 0) != PyArray_DIM(vertices_arr, 0)) {
                PyErr_SetString(PyExc_ValueError, "Invalid codes array");
                return 0;
            }
            m_codes = StridedMemory1D<uint8_t>(codes_arr);
        } else {
            m_codes.reset();
        }

        m_total_vertices = (unsigned)PyArray_DIM(vertices_arr, 0);
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

        StridedMemory1D<double> vertex = m_vertices[idx];
        *x = vertex[0];
        *y = vertex[1];

        if (m_codes) {
            return static_cast<unsigned>(m_codes[idx]);
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

    inline bool has_curves() const
    {
        return m_codes;
    }

    inline void *get_id()
    {
        return (void *)m_vertices.data();
    }
};

class PathGenerator
{
    Py_ssize_t m_npaths;
    bool m_is_optimized;

    // Used for optimized path collections
    StridedMemory3D<double> m_vertices;
    StridedMemory2D<uint8_t> m_codes;
    unsigned m_path_length;
    bool m_should_simplify;
    double m_simplify_threshold;

    // Used for general sequences
    PyObject *m_paths;

  public:
    typedef PathIterator path_iterator;

    PathGenerator(PyObject *obj)
        : m_npaths(0),
          m_is_optimized(false),
          m_vertices(),
          m_codes(),
          m_path_length(0),
          m_should_simplify(false),
          m_simplify_threshold(0),
          m_paths(NULL)
    {
        if (!PySequence_Check(obj)) {
            throw py::exception();
        }

        m_paths = obj;
        Py_INCREF(m_paths);

        m_npaths = PySequence_Size(m_paths);

        PyObject *is_uniform_obj = PyObject_GetAttrString(obj, "_is_uniform_path_collection");
        PyErr_Clear(); // The attribute might not be there.
        m_is_optimized = is_uniform_obj == Py_True;
        Py_XDECREF(is_uniform_obj);
        if (m_is_optimized) {
            PyArrayObject *vertices_obj = (PyArrayObject*)PyObject_GetAttrString(obj, "_vertices");
            PyArrayObject *codes_obj = (PyArrayObject*)PyObject_GetAttrString(obj, "_codes");
            PyObject *should_simplify_obj = PyObject_GetAttrString(obj, "should_simplify");
            PyObject *simplify_threshold_obj = PyObject_GetAttrString(obj, "simplify_threshold");
            if (vertices_obj == NULL || codes_obj == NULL ||
                should_simplify_obj == NULL || simplify_threshold_obj == NULL) {
                PyErr_SetString(PyExc_ValueError, "Expected a uniform path collection");
                goto end;
            }
            if (!PyArray_Check(vertices_obj) || !PyArray_Check(codes_obj)) {
                PyErr_SetString(PyExc_ValueError, "Vertices and codes should be NumPy arrays");
                goto end;
            }
            m_vertices = StridedMemory3D<double>(vertices_obj);
            m_codes = StridedMemory2D<uint8_t>(codes_obj);
            m_path_length = PyArray_DIM(vertices_obj, 1);
            m_should_simplify = should_simplify_obj == Py_True;
            m_simplify_threshold = PyFloat_AsDouble(simplify_threshold_obj);
end:
            Py_XDECREF(vertices_obj);
            Py_XDECREF(codes_obj);
            Py_XDECREF(should_simplify_obj);
            Py_XDECREF(simplify_threshold_obj);
            // Check that PyFloat_AsDouble succeeded
            if (PyErr_Occurred()) {
                throw py::exception();
            }
        }
    }

    ~PathGenerator()
    {
        Py_XDECREF(m_paths);
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
        if (m_is_optimized) {
            return path_iterator(m_vertices[i], m_codes[i], m_path_length,
                                 m_should_simplify, m_simplify_threshold);
        }

        path_iterator path;
        PyObject *item = PySequence_GetItem(m_paths, i % m_npaths);
        if (item == NULL) {
            throw py::exception();
        }
        if (!convert_path(item, &path)) {
            Py_DECREF(item);
            throw py::exception();
        }
        Py_DECREF(item);
        return path;
    }
};

}

#endif
