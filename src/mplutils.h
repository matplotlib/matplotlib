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

#include <Python.h>

#include <string>
#include <iostream>
#include <sstream>

#if PY_MAJOR_VERSION >= 3
#define PY3K 1
#else
#define PY3K 0
#endif

void _VERBOSE(const std::string&);


#undef  CLAMP
#define CLAMP(x, low, high)  (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))

#undef  MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

inline double mpl_round(double v)
{
    return (double)(int)(v + ((v >= 0.0) ? 0.5 : -0.5));
}

class Printf
{
private :
    char *buffer;
public :
    Printf(const char *, ...);
    ~Printf();
    std::string str()
    {
        return buffer;
    }
    friend std::ostream &operator <<(std::ostream &, const Printf &);
};

#if defined(_MSC_VER) && (_MSC_VER == 1400)

/* Required by libpng and zlib */
#pragma comment(lib, "bufferoverflowU")

/* std::max and std::min are missing in Windows Server 2003 R2
   Platform SDK compiler.  See matplotlib bug #3067191 */
namespace std {

    template <class T> inline T max(const T& a, const T& b)
    {
        return (a > b) ? a : b;
    }

    template <class T> inline T min(const T& a, const T& b)
    {
        return (a < b) ? a : b;
    }

}

#endif

#endif
