/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef PATH_CLEANUP_H
#define PATH_CLEANUP_H

#include <Python.h>

enum e_snap_mode {
    SNAP_AUTO,
    SNAP_FALSE,
    SNAP_TRUE
};

void *get_path_iterator(PyObject *path,
                        PyObject *trans,
                        int remove_nans,
                        int do_clip,
                        double rect[4],
                        enum e_snap_mode snap_mode,
                        double stroke_width,
                        int do_simplify);

unsigned get_vertex(void *pipeline, double *x, double *y);

void free_path_iterator(void *pipeline);

#endif /* PATH_CLEANUP_H */
