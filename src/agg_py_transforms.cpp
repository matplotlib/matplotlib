/* -*- mode: c++; c-basic-offset: 4 -*- */

#include <Python.h>

#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"

#include "CXX/Objects.hxx"
#include "agg_trans_affine.h"

/** A helper function to convert from a Numpy affine transformation matrix
 *  to an agg::trans_affine.
 */
agg::trans_affine
py_to_agg_transformation_matrix(PyObject* obj, bool errors = true)
{
    PyArrayObject* matrix = NULL;

    try
    {
        if (obj == Py_None)
            throw std::exception();
        matrix = (PyArrayObject*) PyArray_FromObject(obj, PyArray_DOUBLE, 2, 2);
        if (!matrix)
            throw std::exception();
        if (PyArray_NDIM(matrix) == 2 || PyArray_DIM(matrix, 0) == 3 || PyArray_DIM(matrix, 1) == 3)
        {
            size_t stride0 = PyArray_STRIDE(matrix, 0);
            size_t stride1 = PyArray_STRIDE(matrix, 1);
            char* row0 = PyArray_BYTES(matrix);
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
    }
    catch (...)
    {
        if (errors)
        {
            Py_XDECREF(matrix);
            throw Py::TypeError("Invalid affine transformation matrix");
        }
    }

    Py_XDECREF(matrix);
    return agg::trans_affine();
}

bool
py_convert_bbox(PyObject* bbox_obj, double& l, double& b, double& r, double& t)
{
    PyArrayObject* bbox = NULL;

    if (bbox_obj == Py_None)
        return false;

    try
    {
        bbox = (PyArrayObject*) PyArray_FromObject(bbox_obj, PyArray_DOUBLE, 2, 2);

        if (!bbox || PyArray_NDIM(bbox) != 2 || PyArray_DIM(bbox, 0) != 2 || PyArray_DIM(bbox, 1) != 2)
        {
            throw Py::TypeError
            ("Expected a bbox array");
        }

        l = *(double*)PyArray_GETPTR2(bbox, 0, 0);
        b = *(double*)PyArray_GETPTR2(bbox, 0, 1);
        r = *(double*)PyArray_GETPTR2(bbox, 1, 0);
        t = *(double*)PyArray_GETPTR2(bbox, 1, 1);

        Py_XDECREF(bbox);
        bbox = NULL;
        return true;
    }
    catch (...)
    {
        Py_XDECREF(bbox);
        bbox = NULL;
        throw;
    }

    return false;
}
