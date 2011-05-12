/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef __AGG_PY_TRANSFORMS_H__
#define __AGG_PY_TRANSFORMS_H__

#include "agg_trans_affine.h"

/** A helper function to convert from a Numpy affine transformation matrix
 *  to an agg::trans_affine.
 */
agg::trans_affine
py_to_agg_transformation_matrix(PyObject* obj, bool errors = true);

bool
py_convert_bbox(PyObject* bbox_obj, double& l, double& b, double& r, double& t);

#endif // __AGG_PY_TRANSFORMS_H__
