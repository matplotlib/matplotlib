/* -*- mode: c++; c-basic-offset: 4 -*- */

#include <Python.h>
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"

#include "py_converters.h"

#include "py_adaptors.h"
#include "agg_conv_transform.h"
#include "path_converters.h"

class PathCleanupIterator
{
    typedef agg::conv_transform<py::PathIterator> transformed_path_t;
    typedef PathNanRemover<transformed_path_t> nan_removal_t;
    typedef PathClipper<nan_removal_t> clipped_t;
    typedef PathSnapper<clipped_t> snapped_t;
    typedef PathSimplifier<snapped_t> simplify_t;
    typedef Sketch<simplify_t> sketch_t;

    py::PathIterator m_path_iter;
    agg::trans_affine m_transform;
    transformed_path_t m_transformed;
    nan_removal_t m_nan_removed;
    clipped_t m_clipped;
    snapped_t m_snapped;
    simplify_t m_simplify;
    sketch_t m_sketch;

  public:
    PathCleanupIterator(PyObject *path,
                        agg::trans_affine trans,
                        bool remove_nans,
                        bool do_clip,
                        const agg::rect_base<double> &rect,
                        e_snap_mode snap_mode,
                        double stroke_width,
                        bool do_simplify,
                        double sketch_scale,
                        double sketch_length,
                        double sketch_randomness)
        : m_transform(trans),
          m_transformed(m_path_iter, m_transform),
          m_nan_removed(m_transformed, remove_nans, m_path_iter.has_curves()),
          m_clipped(m_nan_removed, do_clip, rect),
          m_snapped(m_clipped, snap_mode, m_path_iter.total_vertices(), stroke_width),
          m_simplify(m_snapped,
                     do_simplify && m_path_iter.should_simplify(),
                     m_path_iter.simplify_threshold()),
          m_sketch(m_simplify, sketch_scale, sketch_length, sketch_randomness)
    {
        convert_path(path, &m_path_iter);

        Py_INCREF(path);
        m_path_iter.rewind(0);
    }

    unsigned vertex(double *x, double *y)
    {
        return m_simplify.vertex(x, y);
    }
};

extern "C" {
void *get_path_iterator(PyObject *path,
                        PyObject *trans,
                        int remove_nans,
                        int do_clip,
                        double rect[4],
                        e_snap_mode snap_mode,
                        double stroke_width,
                        int do_simplify,
                        double sketch_scale,
                        double sketch_length,
                        double sketch_randomness)
{
    agg::trans_affine agg_trans;
    if (!convert_trans_affine(trans, &agg_trans)) {
        return NULL;
    }
    agg::rect_base<double> clip_rect(rect[0], rect[1], rect[2], rect[3]);

    PathCleanupIterator *pipeline = new PathCleanupIterator(path,
                                                            agg_trans,
                                                            remove_nans != 0,
                                                            do_clip != 0,
                                                            clip_rect,
                                                            snap_mode,
                                                            stroke_width,
                                                            do_simplify != 0,
                                                            sketch_scale,
                                                            sketch_length,
                                                            sketch_randomness);

    return (void *)pipeline;
}

unsigned get_vertex(void *pipeline, double *x, double *y)
{
    PathCleanupIterator *pipeline_iter = (PathCleanupIterator *)pipeline;

    unsigned code = pipeline_iter->vertex(x, y);
    return code;
}

void free_path_iterator(void *pipeline)
{
    PathCleanupIterator *pipeline_iter = (PathCleanupIterator *)pipeline;

    delete pipeline_iter;
}
}
