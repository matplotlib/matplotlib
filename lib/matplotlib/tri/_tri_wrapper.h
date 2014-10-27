#ifndef _TRI_WRAPPER_H
#define _TRI_WRAPPER_H

#include "_tri.h"

typedef struct
{
    PyObject_HEAD;
    Triangulation* ptr;
} PyTriangulation;

typedef struct
{
    PyObject_HEAD;
    TriContourGenerator* ptr;
} PyTriContourGenerator;

typedef struct
{
    PyObject_HEAD;
    TrapezoidMapTriFinder* ptr;
} PyTrapezoidMapTriFinder;

#endif // _TRI_WRAPPER_H
