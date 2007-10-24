#ifndef __AGG_PY_PATH_ITERATOR_H__
#define __AGG_PY_PATH_ITERATOR_H__

#include "CXX/Objects.hxx"
#define PY_ARRAY_TYPES_PREFIX NumPy
#include "numpy/arrayobject.h"
#include "agg_path_storage.h"

class PathIterator {
  PyArrayObject* vertices;
  PyArrayObject* codes;
  size_t m_iterator;
  size_t m_total_vertices;

public:
  PathIterator(const Py::Object& path_obj) :
    vertices(NULL), codes(NULL), m_iterator(0) {
    Py::Object vertices_obj = path_obj.getAttr("vertices");
    Py::Object codes_obj = path_obj.getAttr("codes");
    
    vertices = (PyArrayObject*)PyArray_FromObject
      (vertices_obj.ptr(), PyArray_DOUBLE, 2, 2);
    if (!vertices || vertices->nd != 2 || vertices->dimensions[1] != 2)
      throw Py::ValueError("Invalid vertices array.");

    codes = (PyArrayObject*)PyArray_FromObject
      (codes_obj.ptr(), PyArray_UINT8, 1, 1);
    if (!codes) 
      throw Py::ValueError("Invalid codes array.");
    
    if (codes->dimensions[0] != vertices->dimensions[0])
      throw Py::ValueError("vertices and codes arrays are not the same length.");

    m_total_vertices = codes->dimensions[0];
  }

  ~PathIterator() {
    Py_XDECREF(vertices);
    Py_XDECREF(codes);
  }

  static const char code_map[];

  inline unsigned vertex(unsigned idx, double* x, double* y) {
    if (idx > m_total_vertices)
      throw Py::RuntimeError("Requested vertex past end");
    *x = *(double*)PyArray_GETPTR2(vertices, idx, 0);
    *y = *(double*)PyArray_GETPTR2(vertices, idx, 1);
    return code_map[(int)*(char *)PyArray_GETPTR1(codes, idx)];
  }

  inline unsigned vertex(double* x, double* y) {
    if (m_iterator >= m_total_vertices) return agg::path_cmd_stop;
    return vertex(m_iterator++, x, y);
  }

  inline void rewind(unsigned path_id) {
    m_iterator = path_id;
  }

  inline unsigned total_vertices() {
    return m_total_vertices;
  }
};

// Maps path codes on the Python side to agg path commands
const char PathIterator::code_map[] = 
  {0, 
   agg::path_cmd_move_to, 
   agg::path_cmd_line_to, 
   agg::path_cmd_curve3,
   agg::path_cmd_curve4,
   agg::path_cmd_end_poly | agg::path_flags_close};

#endif // __AGG_PY_PATH_ITERATOR_H__
