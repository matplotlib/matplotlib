#ifndef MPL_PY_CONVERTERS_11_H
#define MPL_PY_CONVERTERS_11_H

// pybind11 equivalent of py_converters.h

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "agg_trans_affine.h"

void convert_trans_affine(const py::object& transform, agg::trans_affine& affine);

#endif
