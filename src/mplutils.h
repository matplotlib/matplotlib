/* -*- mode: c++; c-basic-offset: 4 -*- */

/* Small utilities that are shared by most extension modules. */

#ifndef MPLUTILS_H
#define MPLUTILS_H
#define PY_SSIZE_T_CLEAN

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

#include <Python.h>

#undef CLAMP
#define CLAMP(x, low, high) (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))

#undef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

inline double mpl_round(double v)
{
    return (double)(int)(v + ((v >= 0.0) ? 0.5 : -0.5));
}

enum {
    STOP = 0,
    MOVETO = 1,
    LINETO = 2,
    CURVE3 = 3,
    CURVE4 = 4,
    ENDPOLY = 0x4f
};

const size_t NUM_VERTICES[] = { 1, 1, 1, 2, 3, 1 };

extern "C" int add_dict_int(PyObject *dict, const char *key, long val);

#endif
