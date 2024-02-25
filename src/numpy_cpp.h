/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_NUMPY_CPP_H
#define MPL_NUMPY_CPP_H
#define PY_SSIZE_T_CLEAN
/***************************************************************************
 * This file is based on original work by Mark Wiebe, available at:
 *
 *    http://github.com/mwiebe/numpy-cpp
 *
 * However, the needs of matplotlib wrappers, such as treating an
 * empty array as having the correct dimensions, have made this rather
 * matplotlib-specific, so it's no longer compatible with the
 * original.
 */

#ifdef _POSIX_C_SOURCE
#    undef _POSIX_C_SOURCE
#endif
#ifndef _AIX
#ifdef _XOPEN_SOURCE
#    undef _XOPEN_SOURCE
#endif
#endif

// Prevent multiple conflicting definitions of swab from stdlib.h and unistd.h
#if defined(__sun) || defined(sun)
#if defined(_XPG4)
#undef _XPG4
#endif
#if defined(_XPG3)
#undef _XPG3
#endif
#endif

#include <Python.h>
#include <numpy/ndarrayobject.h>

#include "py_exceptions.h"

#include <complex>

namespace numpy
{

// Type traits for the NumPy types
template <typename T>
struct type_num_of;

/* Be careful with bool arrays as python has sizeof(npy_bool) == 1, but it is
 * not always the case that sizeof(bool) == 1.  Using the array_view_accessors
 * is always fine regardless of sizeof(bool), so do this rather than using
 * array.data() and pointer arithmetic which will not work correctly if
 * sizeof(bool) != 1. */
template <> struct type_num_of<bool>
{
    enum {
        value = NPY_BOOL
    };
};
template <>
struct type_num_of<npy_byte>
{
    enum {
        value = NPY_BYTE
    };
};
template <>
struct type_num_of<npy_ubyte>
{
    enum {
        value = NPY_UBYTE
    };
};
template <>
struct type_num_of<npy_short>
{
    enum {
        value = NPY_SHORT
    };
};
template <>
struct type_num_of<npy_ushort>
{
    enum {
        value = NPY_USHORT
    };
};
template <>
struct type_num_of<npy_int>
{
    enum {
        value = NPY_INT
    };
};
template <>
struct type_num_of<npy_uint>
{
    enum {
        value = NPY_UINT
    };
};
template <>
struct type_num_of<npy_long>
{
    enum {
        value = NPY_LONG
    };
};
template <>
struct type_num_of<npy_ulong>
{
    enum {
        value = NPY_ULONG
    };
};
template <>
struct type_num_of<npy_longlong>
{
    enum {
        value = NPY_LONGLONG
    };
};
template <>
struct type_num_of<npy_ulonglong>
{
    enum {
        value = NPY_ULONGLONG
    };
};
template <>
struct type_num_of<npy_float>
{
    enum {
        value = NPY_FLOAT
    };
};
template <>
struct type_num_of<npy_double>
{
    enum {
        value = NPY_DOUBLE
    };
};
#if NPY_LONGDOUBLE != NPY_DOUBLE
template <>
struct type_num_of<npy_longdouble>
{
    enum {
        value = NPY_LONGDOUBLE
    };
};
#endif
template <>
struct type_num_of<npy_cfloat>
{
    enum {
        value = NPY_CFLOAT
    };
};
template <>
struct type_num_of<std::complex<npy_float> >
{
    enum {
        value = NPY_CFLOAT
    };
};
template <>
struct type_num_of<npy_cdouble>
{
    enum {
        value = NPY_CDOUBLE
    };
};
template <>
struct type_num_of<std::complex<npy_double> >
{
    enum {
        value = NPY_CDOUBLE
    };
};
#if NPY_CLONGDOUBLE != NPY_CDOUBLE
template <>
struct type_num_of<npy_clongdouble>
{
    enum {
        value = NPY_CLONGDOUBLE
    };
};
template <>
struct type_num_of<std::complex<npy_longdouble> >
{
    enum {
        value = NPY_CLONGDOUBLE
    };
};
#endif
template <>
struct type_num_of<PyObject *>
{
    enum {
        value = NPY_OBJECT
    };
};
template <typename T>
struct type_num_of<T &>
{
    enum {
        value = type_num_of<T>::value
    };
};
template <typename T>
struct type_num_of<const T>
{
    enum {
        value = type_num_of<T>::value
    };
};

template <typename T>
struct is_const
{
    enum {
        value = false
    };
};
template <typename T>
struct is_const<const T>
{
    enum {
        value = true
    };
};

namespace detail
{
template <template <typename, int> class AV, typename T, int ND>
class array_view_accessors;

template <template <typename, int> class AV, typename T>
class array_view_accessors<AV, T, 1>
{
  public:
    typedef AV<T, 1> AVC;
    typedef T sub_t;

    T &operator()(npy_intp i)
    {
        AVC *self = static_cast<AVC *>(this);

        return *reinterpret_cast<T *>(self->m_data + self->m_strides[0] * i);
    }

    const T &operator()(npy_intp i) const
    {
        const AVC *self = static_cast<const AVC *>(this);

        return *reinterpret_cast<const T *>(self->m_data + self->m_strides[0] * i);
    }

    T &operator[](npy_intp i)
    {
        AVC *self = static_cast<AVC *>(this);

        return *reinterpret_cast<T *>(self->m_data + self->m_strides[0] * i);
    }

    const T &operator[](npy_intp i) const
    {
        const AVC *self = static_cast<const AVC *>(this);

        return *reinterpret_cast<const T *>(self->m_data + self->m_strides[0] * i);
    }
};

template <template <typename, int> class AV, typename T>
class array_view_accessors<AV, T, 2>
{
  public:
    typedef AV<T, 2> AVC;
    typedef AV<T, 1> sub_t;

    T &operator()(npy_intp i, npy_intp j)
    {
        AVC *self = static_cast<AVC *>(this);

        return *reinterpret_cast<T *>(self->m_data + self->m_strides[0] * i +
                                      self->m_strides[1] * j);
    }

    const T &operator()(npy_intp i, npy_intp j) const
    {
        const AVC *self = static_cast<const AVC *>(this);

        return *reinterpret_cast<const T *>(self->m_data + self->m_strides[0] * i +
                                            self->m_strides[1] * j);
    }

    sub_t subarray(npy_intp i) const
    {
        const AVC *self = static_cast<const AVC *>(this);

        return sub_t(self->m_arr,
                     self->m_data + self->m_strides[0] * i,
                     self->m_shape + 1,
                     self->m_strides + 1);
    }
};

template <template <typename, int> class AV, typename T>
class array_view_accessors<AV, T, 3>
{
  public:
    typedef AV<T, 3> AVC;
    typedef AV<T, 2> sub_t;

    T &operator()(npy_intp i, npy_intp j, npy_intp k)
    {
        AVC *self = static_cast<AVC *>(this);

        return *reinterpret_cast<T *>(self->m_data + self->m_strides[0] * i +
                                      self->m_strides[1] * j + self->m_strides[2] * k);
    }

    const T &operator()(npy_intp i, npy_intp j, npy_intp k) const
    {
        const AVC *self = static_cast<const AVC *>(this);

        return *reinterpret_cast<const T *>(self->m_data + self->m_strides[0] * i +
                                            self->m_strides[1] * j + self->m_strides[2] * k);
    }

    sub_t subarray(npy_intp i) const
    {
        const AVC *self = static_cast<const AVC *>(this);

        return sub_t(self->m_arr,
                     self->m_data + self->m_strides[0] * i,
                     self->m_shape + 1,
                     self->m_strides + 1);
    }


};

// When adding instantiations of array_view_accessors, remember to add entries
// to zeros[] below.

}

static npy_intp zeros[] = { 0, 0, 0 };

template <typename T, int ND>
class array_view : public detail::array_view_accessors<array_view, T, ND>
{
    friend class detail::array_view_accessors<numpy::array_view, T, ND>;

  private:
    // Copies of the array data
    PyArrayObject *m_arr;
    npy_intp *m_shape;
    npy_intp *m_strides;
    char *m_data;

  public:
    typedef T value_type;

    enum {
        ndim = ND
    };

    array_view() : m_arr(NULL), m_data(NULL)
    {
        m_shape = zeros;
        m_strides = zeros;
    }

    array_view(PyObject *arr, bool contiguous = false) : m_arr(NULL), m_data(NULL)
    {
        if (!set(arr, contiguous)) {
            throw mpl::exception();
        }
    }

    array_view(const array_view &other) : m_arr(NULL), m_data(NULL)
    {
        m_arr = other.m_arr;
        Py_XINCREF(m_arr);
        m_data = other.m_data;
        m_shape = other.m_shape;
        m_strides = other.m_strides;
    }

    array_view(PyArrayObject *arr, char *data, npy_intp *shape, npy_intp *strides)
    {
        m_arr = arr;
        Py_XINCREF(arr);
        m_data = data;
        m_shape = shape;
        m_strides = strides;
    }

    array_view(PyArrayObject *arr)
    {
        m_arr = arr;
        Py_XINCREF(arr);
        m_shape = PyArray_DIMS(m_arr);
        m_strides = PyArray_STRIDES(m_arr);
        m_data = PyArray_BYTES(m_arr);
    }

    array_view(npy_intp shape[ND]) : m_arr(NULL), m_shape(NULL), m_strides(NULL), m_data(NULL)
    {
        PyObject *arr = PyArray_SimpleNew(ND, shape, type_num_of<T>::value);
        if (arr == NULL) {
            throw mpl::exception();
        }
        if (!set(arr, true)) {
            Py_DECREF(arr);
            throw mpl::exception();
        }
        Py_DECREF(arr);
    }

    ~array_view()
    {
        Py_XDECREF(m_arr);
    }

    array_view& operator=(const array_view &other)
    {
        if (this != &other)
        {
            Py_XDECREF(m_arr);
            m_arr = other.m_arr;
            Py_XINCREF(m_arr);
            m_data = other.m_data;
            m_shape = other.m_shape;
            m_strides = other.m_strides;
        }
        return *this;
    }

    bool set(PyObject *arr, bool contiguous = false)
    {
        PyArrayObject *tmp;

        if (arr == NULL || arr == Py_None) {
            Py_XDECREF(m_arr);
            m_arr = NULL;
            m_data = NULL;
            m_shape = zeros;
            m_strides = zeros;
        } else {
            if (contiguous) {
                tmp = (PyArrayObject *)PyArray_ContiguousFromAny(arr, type_num_of<T>::value, 0, ND);
            } else {
                tmp = (PyArrayObject *)PyArray_FromObject(arr, type_num_of<T>::value, 0, ND);
            }
            if (tmp == NULL) {
                return false;
            }

            if (PyArray_NDIM(tmp) == 0 || PyArray_DIM(tmp, 0) == 0) {
                Py_XDECREF(m_arr);
                m_arr = NULL;
                m_data = NULL;
                m_shape = zeros;
                m_strides = zeros;
                if (PyArray_NDIM(tmp) == 0 && ND == 0) {
                    m_arr = tmp;
                    return true;
                }
            }
            if (PyArray_NDIM(tmp) != ND) {
                PyErr_Format(PyExc_ValueError,
                             "Expected %d-dimensional array, got %d",
                             ND,
                             PyArray_NDIM(tmp));
                Py_DECREF(tmp);
                return false;
            }

            /* Copy some of the data to the view object for faster access */
            Py_XDECREF(m_arr);
            m_arr = tmp;
            m_shape = PyArray_DIMS(m_arr);
            m_strides = PyArray_STRIDES(m_arr);
            m_data = PyArray_BYTES(tmp);
        }

        return true;
    }

    npy_intp shape(size_t i) const
    {
        if (i >= ND) {
            return 0;
        }
        return m_shape[i];
    }

    size_t size() const;

    // Do not use this for array_view<bool, ND>.  See comment near top of file.
    const T *data() const
    {
        return (const T *)m_data;
    }

    // Do not use this for array_view<bool, ND>.  See comment near top of file.
    T *data()
    {
        return (T *)m_data;
    }

    // Return a new reference.
    PyObject *pyobj()
    {
        Py_XINCREF(m_arr);
        return (PyObject *)m_arr;
    }

    // Steal a reference.
    PyObject *pyobj_steal()
    {
        return (PyObject *)m_arr;
    }

    static int converter(PyObject *obj, void *arrp)
    {
        array_view<T, ND> *arr = (array_view<T, ND> *)arrp;

        if (!arr->set(obj)) {
            return 0;
        }

        return 1;
    }

    static int converter_contiguous(PyObject *obj, void *arrp)
    {
        array_view<T, ND> *arr = (array_view<T, ND> *)arrp;

        if (!arr->set(obj, true)) {
            return 0;
        }

        return 1;
    }
};

/* In most cases, code should use safe_first_shape(obj) instead of obj.shape(0), since
   safe_first_shape(obj) == 0 when any dimension is 0. */
template <typename T, int ND>
size_t
safe_first_shape(const array_view<T, ND> &a)
{
    bool empty = (ND == 0);
    for (size_t i = 0; i < ND; i++) {
        if (a.shape(i) == 0) {
            empty = true;
        }
    }
    if (empty) {
        return 0;
    } else {
        return (size_t)a.shape(0);
    }
}

template <typename T, int ND>
size_t
array_view<T, ND>::size() const
{
    return safe_first_shape<T, ND>(*this);
}

} // namespace numpy


#endif
