/* -*- mode: c++; c-basic-offset: 4 -*- */

/* mplutils.h   --
 *
 * $Header$
 * $Log$
 * Revision 1.2  2004/11/24 15:26:12  jdh2358
 * added Printf
 *
 * Revision 1.1  2004/06/24 20:11:17  jdh2358
 * added mpl src
 *
 */

#ifndef _MPLUTILS_H
#define _MPLUTILS_H

#include <string>
#include <iostream>
#include <sstream>

#include <Python.h>

#if PY_MAJOR_VERSION >= 3
#define PY3K 1
#define Py_TPFLAGS_HAVE_NEWBUFFER 0
#else
#define PY3K 0
#endif

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

#if defined(_MSC_VER) && (_MSC_VER == 1400)

/* Required by libpng and zlib */
#pragma comment(lib, "bufferoverflowU")

/* std::max and std::min are missing in Windows Server 2003 R2
   Platform SDK compiler.  See matplotlib bug #3067191 */
namespace std
{

template <class T>
inline T max(const T &a, const T &b)
{
    return (a > b) ? a : b;
}

template <class T>
inline T min(const T &a, const T &b)
{
    return (a < b) ? a : b;
}
}

#endif

#endif
