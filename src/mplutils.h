/* -*- mode: c++; c-basic-offset: 4 -*- */

/* Small utilities that are shared by most extension modules. */

#ifndef MPLUTILS_H
#define MPLUTILS_H
#define PY_SSIZE_T_CLEAN

#include <Python.h>

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

#ifdef __cplusplus  // not for macosx.m
// Check that array has shape (N, d1) or (N, d1, d2).  We cast d1, d2 to longs
// so that we don't need to access the NPY_INTP_FMT macro here.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

template<typename T>
inline void check_trailing_shape(T array, char const* name, long d1)
{
    if (array.ndim() != 2) {
        throw py::value_error(
            "Expected 2-dimensional array, got %d"_s.format(array.ndim()));
    }
    if (array.size() == 0) {
        // Sometimes things come through as atleast_2d, etc., but they're empty, so
        // don't bother enforcing the trailing shape.
        return;
    }
    if (array.shape(1) != d1) {
        throw py::value_error(
            "%s must have shape (N, %d), got (%d, %d)"_s.format(
                name, d1, array.shape(0), array.shape(1)));
    }
}

template<typename T>
inline void check_trailing_shape(T array, char const* name, long d1, long d2)
{
    if (array.ndim() != 3) {
        throw py::value_error(
            "Expected 3-dimensional array, got %d"_s.format(array.ndim()));
    }
    if (array.size() == 0) {
        // Sometimes things come through as atleast_3d, etc., but they're empty, so
        // don't bother enforcing the trailing shape.
        return;
    }
    if (array.shape(1) != d1 || array.shape(2) != d2) {
        throw py::value_error(
            "%s must have shape (N, %d, %d), got (%d, %d, %d)"_s.format(
                name, d1, d2, array.shape(0), array.shape(1), array.shape(2)));
    }
}

/* In most cases, code should use safe_first_shape(obj) instead of obj.shape(0), since
   safe_first_shape(obj) == 0 when any dimension is 0. */
template <typename T, py::ssize_t ND>
py::ssize_t
safe_first_shape(const py::detail::unchecked_reference<T, ND> &a)
{
    bool empty = (ND == 0);
    for (py::ssize_t i = 0; i < ND; i++) {
        if (a.shape(i) == 0) {
            empty = true;
        }
    }
    if (empty) {
        return 0;
    } else {
        return a.shape(0);
    }
}
#endif

#endif
