#ifndef PATH_CLEANUP_H
#define PATH_CLEANUP_H

#include <Python.h>

enum e_quantize_mode
{
    QUANTIZE_AUTO,
    QUANTIZE_FALSE,
    QUANTIZE_TRUE
};

void*
get_path_iterator(
    PyObject* path, PyObject* trans, int remove_nans, int do_clip,
    double rect[4], enum e_quantize_mode quantize_mode, int do_simplify);

unsigned
get_vertex(void* pipeline, double* x, double* y);

void
free_path_iterator(void* pipeline);

#endif /* PATH_CLEANUP_H */
