#ifndef __AGG_PY_TRANSFORMS_H__
#define __AGG_PY_TRANSFORMS_H__

#define PY_ARRAY_TYPES_PREFIX NumPy
#include "numpy/arrayobject.h"

#include "CXX/Objects.hxx"
#include "agg_trans_affine.h"


/** A helper function to convert from a Numpy affine transformation matrix
 *  to an agg::trans_affine.
 */
agg::trans_affine py_to_agg_transformation_matrix(const Py::Object& obj, bool errors = true) {
  PyArrayObject* matrix = NULL;
  
  try {
    if (obj.ptr() == Py_None)
      throw std::exception();
    matrix = (PyArrayObject*) PyArray_FromObject(obj.ptr(), PyArray_DOUBLE, 2, 2);
    if (!matrix)
      throw std::exception();
    if (matrix->nd == 2 || matrix->dimensions[0] == 3 || matrix->dimensions[1] == 3) {
      size_t stride0 = matrix->strides[0];
      size_t stride1 = matrix->strides[1];
      char* row0 = matrix->data;
      char* row1 = row0 + stride0;
      
      double a = *(double*)(row0);
      row0 += stride1;
      double c = *(double*)(row0);
      row0 += stride1;
      double e = *(double*)(row0);
      
      double b = *(double*)(row1);
      row1 += stride1;
      double d = *(double*)(row1);
      row1 += stride1;
      double f = *(double*)(row1);
      
      Py_XDECREF(matrix);
      
      return agg::trans_affine(a, b, c, d, e, f);
    }

    throw std::exception();
  } catch (...) {
    if (errors) {
      Py_XDECREF(matrix);
      throw Py::TypeError("Invalid affine transformation matrix");
    }
  }

  Py_XDECREF(matrix);
  return agg::trans_affine();
}

#endif // __AGG_PY_TRANSFORMS_H__
