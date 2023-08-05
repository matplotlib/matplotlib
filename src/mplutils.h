/* -*- mode: c++; c-basic-offset: 4 -*- */

/* Small utilities that are shared by most extension modules. */

#ifndef MPLUTILS_H
#define MPLUTILS_H
#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <stdint.h>

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


inline int mpl_round_to_int(double v)
{
    return (int)(v + ((v >= 0.0) ? 0.5 : -0.5));
}

inline double mpl_round(double v)
{
    return (double)mpl_round_to_int(v);
}

// 'kind' codes for paths.
enum {
    STOP = 0,
    MOVETO = 1,
    LINETO = 2,
    CURVE3 = 3,
    CURVE4 = 4,
    CLOSEPOLY = 0x4f
};

const size_t NUM_VERTICES[] = { 1, 1, 1, 2, 3, 1 };

inline int prepare_and_add_type(PyTypeObject *type, PyObject *module)
{
    if (PyType_Ready(type)) {
        return -1;
    }
    char const* ptr = strrchr(type->tp_name, '.');
    if (!ptr) {
        PyErr_SetString(PyExc_ValueError, "tp_name should be a qualified name");
        return -1;
    }
    if (PyModule_AddObject(module, ptr + 1, (PyObject *)type)) {
        return -1;
    }
    return 0;
}

#ifdef __cplusplus  // not for macosx.m
// Check that array has shape (N, d1) or (N, d1, d2).  We cast d1, d2 to longs
// so that we don't need to access the NPY_INTP_FMT macro here.

template<typename T>
inline bool check_trailing_shape(T array, char const* name, long d1)
{
    if (array.dim(1) != d1) {
        PyErr_Format(PyExc_ValueError,
                     "%s must have shape (N, %ld), got (%ld, %ld)",
                     name, d1, array.dim(0), array.dim(1));
        return false;
    }
    return true;
}

template<typename T>
inline bool check_trailing_shape(T array, char const* name, long d1, long d2)
{
    if (array.dim(1) != d1 || array.dim(2) != d2) {
        PyErr_Format(PyExc_ValueError,
                     "%s must have shape (N, %ld, %ld), got (%ld, %ld, %ld)",
                     name, d1, d2, array.dim(0), array.dim(1), array.dim(2));
        return false;
    }
    return true;
}
#endif

#endif
