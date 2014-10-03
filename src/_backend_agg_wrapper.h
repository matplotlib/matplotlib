#ifndef __BACKEND_AGG_WRAPPER_H__
#define __BACKEND_AGG_WRAPPER_H__

#include "mplutils.h"
#include "py_converters.h"
#include "_backend_agg.h"

extern "C" {

typedef struct
{
    PyObject_HEAD;
    RendererAgg *x;
    Py_ssize_t shape[3];
    Py_ssize_t strides[3];
    Py_ssize_t suboffsets[3];
} PyRendererAgg;

typedef struct
{
    PyObject_HEAD;
    BufferRegion *x;
    Py_ssize_t shape[3];
    Py_ssize_t strides[3];
    Py_ssize_t suboffsets[3];
} PyBufferRegion;
}

#endif
