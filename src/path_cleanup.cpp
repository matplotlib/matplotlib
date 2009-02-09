#include <Python.h>
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"

#include "agg_py_path_iterator.h"
#include "agg_conv_transform.h"
#include "agg_py_transforms.h"
#include "path_converters.h"

class PathCleanupIterator
{
    typedef agg::conv_transform<PathIterator>  transformed_path_t;
    typedef PathNanRemover<transformed_path_t> nan_removal_t;
    typedef PathClipper<nan_removal_t>         clipped_t;
    typedef PathQuantizer<clipped_t>           quantized_t;
    typedef PathSimplifier<quantized_t>        simplify_t;

    Py::Object         m_path_obj;
    PathIterator       m_path_iter;
    agg::trans_affine  m_transform;
    transformed_path_t m_transformed;
    nan_removal_t      m_nan_removed;
    clipped_t          m_clipped;
    quantized_t        m_quantized;
    simplify_t         m_simplify;

public:
    PathCleanupIterator(PyObject* path, agg::trans_affine trans,
                        bool remove_nans, bool do_clip,
                        const agg::rect_base<double>& rect,
                        e_quantize_mode quantize_mode, bool do_simplify) :
        m_path_obj(path, true),
        m_path_iter(m_path_obj),
        m_transform(trans),
        m_transformed(m_path_iter, m_transform),
        m_nan_removed(m_transformed, remove_nans, m_path_iter.has_curves()),
        m_clipped(m_nan_removed, do_clip, rect),
        m_quantized(m_clipped, quantize_mode, m_path_iter.total_vertices()),
        m_simplify(m_quantized, do_simplify && m_path_iter.should_simplify(),
                   m_path_iter.simplify_threshold())
    {
        Py_INCREF(path);
        m_path_iter.rewind(0);
    }

    unsigned vertex(double* x, double* y)
    {
        return m_simplify.vertex(x, y);
    }
};

extern "C" {
    void*
    get_path_iterator(
        PyObject* path, PyObject* trans, int remove_nans, int do_clip,
        double rect[4], e_quantize_mode quantize_mode, int do_simplify)
    {
        agg::trans_affine agg_trans = py_to_agg_transformation_matrix(trans, false);
        agg::rect_base<double> clip_rect(rect[0], rect[1], rect[2], rect[3]);

        PathCleanupIterator* pipeline = new PathCleanupIterator(
            path, agg_trans, remove_nans != 0, do_clip != 0,
            clip_rect, quantize_mode, do_simplify != 0);

        return (void*)pipeline;
    }

    unsigned
    get_vertex(void* pipeline, double* x, double* y)
    {
        PathCleanupIterator* pipeline_iter = (PathCleanupIterator*)pipeline;

        unsigned code = pipeline_iter->vertex(x, y);
        return code;
    }

    void
    free_path_iterator(void* pipeline)
    {
        PathCleanupIterator* pipeline_iter = (PathCleanupIterator*)pipeline;

        delete pipeline_iter;
    }
}

