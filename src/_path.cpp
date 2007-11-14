#include "agg_py_path_iterator.h"
#include "agg_py_transforms.h"

#include "CXX/Extensions.hxx"

#include "agg_conv_curve.h"
#include "agg_conv_stroke.h"
#include "agg_conv_transform.h"
#include "agg_path_storage.h"
#include "agg_trans_affine.h"

// MGDTODO: Un-CXX-ify this module

// the extension module
class _path_module : public Py::ExtensionModule<_path_module>
{
public:
  _path_module()
    : Py::ExtensionModule<_path_module>( "_path" )
  {
    add_varargs_method("point_in_path", &_path_module::point_in_path,
		       "point_in_path(x, y, path, trans)");
    add_varargs_method("point_on_path", &_path_module::point_on_path,
		       "point_on_path(x, y, r, path, trans)");
    add_varargs_method("get_path_extents", &_path_module::get_path_extents,
		       "get_path_extents(path, trans)");
    add_varargs_method("get_path_collection_extents", &_path_module::get_path_collection_extents,
		       "get_path_collection_extents(trans, paths, transforms, offsets, offsetTrans)");
    add_varargs_method("point_in_path_collection", &_path_module::point_in_path_collection,
		       "point_in_path_collection(x, y, r, trans, paths, transforms, offsets, offsetTrans, filled)");
    add_varargs_method("path_in_path", &_path_module::path_in_path,
		       "point_in_path_collection(a, atrans, b, btrans)");
    add_varargs_method("clip_path_to_rect", &_path_module::clip_path_to_rect,
		       "clip_path_to_rect(path, bbox, inside)");

    initialize("Helper functions for paths");
  }

  virtual ~_path_module() {}

private:

  Py::Object point_in_path(const Py::Tuple& args);
  Py::Object point_on_path(const Py::Tuple& args);
  Py::Object get_path_extents(const Py::Tuple& args);
  Py::Object get_path_collection_extents(const Py::Tuple& args);
  Py::Object point_in_path_collection(const Py::Tuple& args);
  Py::Object path_in_path(const Py::Tuple& args);
  Py::Object clip_path_to_rect(const Py::Tuple& args);
};

//
// The following function was found in the Agg 2.3 examples (interactive_polygon.cpp).
// It has been generalized to work on (possibly curved) polylines, rather than
// just polygons.  The original comments have been kept intact.
//  -- Michael Droettboom 2007-10-02
//
//======= Crossings Multiply algorithm of InsideTest ======================== 
//
// By Eric Haines, 3D/Eye Inc, erich@eye.com
//
// This version is usually somewhat faster than the original published in
// Graphics Gems IV; by turning the division for testing the X axis crossing
// into a tricky multiplication test this part of the test became faster,
// which had the additional effect of making the test for "both to left or
// both to right" a bit slower for triangles than simply computing the
// intersection each time.  The main increase is in triangle testing speed,
// which was about 15% faster; all other polygon complexities were pretty much
// the same as before.  On machines where division is very expensive (not the
// case on the HP 9000 series on which I tested) this test should be much
// faster overall than the old code.  Your mileage may (in fact, will) vary,
// depending on the machine and the test data, but in general I believe this
// code is both shorter and faster.  This test was inspired by unpublished
// Graphics Gems submitted by Joseph Samosky and Mark Haigh-Hutchinson.
// Related work by Samosky is in:
//
// Samosky, Joseph, "SectionView: A system for interactively specifying and
// visualizing sections through three-dimensional medical image data",
// M.S. Thesis, Department of Electrical Engineering and Computer Science,
// Massachusetts Institute of Technology, 1993.
//
// Shoot a test ray along +X axis.  The strategy is to compare vertex Y values
// to the testing point's Y and quickly discard edges which are entirely to one
// side of the test ray.  Note that CONVEX and WINDING code can be added as
// for the CrossingsTest() code; it is left out here for clarity.
//
// Input 2D polygon _pgon_ with _numverts_ number of vertices and test point
// _point_, returns 1 if inside, 0 if outside.
template<class T>
bool point_in_path_impl(double tx, double ty, T& path) {
  int yflag0, yflag1, inside_flag;
  double vtx0, vty0, vtx1, vty1, sx, sy;
  double x, y;

  path.rewind(0);

  inside_flag = 0;

  unsigned code = 0;
  while (true) {
    if (code != agg::path_cmd_move_to)
      code = path.vertex(&x, &y);

    sx = vtx0 = x;
    sy = vty0 = y;

    // get test bit for above/below X axis
    yflag0 = (vty0 >= ty);

    vtx1 = x;
    vty1 = x;

    inside_flag = 0;
    while (true) {
      code = path.vertex(&x, &y);

      // The following cases denote the beginning on a new subpath
      if (code == agg::path_cmd_stop || (code & agg::path_cmd_end_poly) == agg::path_cmd_end_poly) {
	x = sx; y = sy;
      } else if (code == agg::path_cmd_move_to)
	break;

      yflag1 = (vty1 >= ty);
      // Check if endpoints straddle (are on opposite sides) of X axis
      // (i.e. the Y's differ); if so, +X ray could intersect this edge.
      // The old test also checked whether the endpoints are both to the
      // right or to the left of the test point.  However, given the faster
      // intersection point computation used below, this test was found to
      // be a break-even proposition for most polygons and a loser for
      // triangles (where 50% or more of the edges which survive this test
      // will cross quadrants and so have to have the X intersection computed
      // anyway).  I credit Joseph Samosky with inspiring me to try dropping
      // the "both left or both right" part of my code.
      if (yflag0 != yflag1) {
	// Check intersection of pgon segment with +X ray.
	// Note if >= point's X; if so, the ray hits it.
	// The division operation is avoided for the ">=" test by checking
	// the sign of the first vertex wrto the test point; idea inspired
	// by Joseph Samosky's and Mark Haigh-Hutchinson's different
	// polygon inclusion tests.
	if ( ((vty1-ty) * (vtx0-vtx1) >=
	      (vtx1-tx) * (vty0-vty1)) == yflag1 ) {
	  inside_flag ^= 1;
	}
      }

      // Move to the next pair of vertices, retaining info as possible.
      yflag0 = yflag1;
      vtx0 = vtx1;
      vty0 = vty1;
	
      vtx1 = x;
      vty1 = y;

      if (code == agg::path_cmd_stop || 
	  (code & agg::path_cmd_end_poly) == agg::path_cmd_end_poly)
	break;
    }

    yflag1 = (vty1 >= ty);
    if (yflag0 != yflag1) {
      if ( ((vty1-ty) * (vtx0-vtx1) >=
	    (vtx1-tx) * (vty0-vty1)) == yflag1 ) {
	inside_flag ^= 1;
      }
    }

    if (inside_flag != 0)
      return true;

    if (code == agg::path_cmd_stop)
      break;
  }

  return (inside_flag != 0);
}

inline bool point_in_path(double x, double y, PathIterator& path, const agg::trans_affine& trans) {
  typedef agg::conv_transform<PathIterator> transformed_path_t;
  typedef agg::conv_curve<transformed_path_t> curve_t;
  
  if (path.total_vertices() < 3)
    return false;

  transformed_path_t trans_path(path, trans);
  curve_t curved_path(trans_path);
  return point_in_path_impl(x, y, curved_path);
}

inline bool point_on_path(double x, double y, double r, PathIterator& path, const agg::trans_affine& trans) {
  typedef agg::conv_transform<PathIterator> transformed_path_t;
  typedef agg::conv_curve<transformed_path_t> curve_t;
  typedef agg::conv_stroke<curve_t> stroke_t;

  transformed_path_t trans_path(path, trans);
  curve_t curved_path(trans_path);
  stroke_t stroked_path(curved_path);
  stroked_path.width(r * 2.0);
  return point_in_path_impl(x, y, stroked_path);
}

Py::Object _path_module::point_in_path(const Py::Tuple& args) {
  args.verify_length(4);
  
  double x = Py::Float(args[0]);
  double y = Py::Float(args[1]);
  PathIterator path(args[2]);
  agg::trans_affine trans = py_to_agg_transformation_matrix(args[3]);

  if (::point_in_path(x, y, path, trans))
    return Py::Int(1);
  return Py::Int(0);
}

Py::Object _path_module::point_on_path(const Py::Tuple& args) {
  args.verify_length(5);
  
  double x = Py::Float(args[0]);
  double y = Py::Float(args[1]);
  double r = Py::Float(args[2]);
  PathIterator path(args[3]);
  agg::trans_affine trans = py_to_agg_transformation_matrix(args[4]);

  if (::point_on_path(x, y, r, path, trans))
    return Py::Int(1);
  return Py::Int(0);
}

void get_path_extents(PathIterator& path, const agg::trans_affine& trans, 
		      double* x0, double* y0, double* x1, double* y1) {
  typedef agg::conv_transform<PathIterator> transformed_path_t;
  typedef agg::conv_curve<transformed_path_t> curve_t;
  double x, y;
  unsigned code;

  transformed_path_t tpath(path, trans);
  curve_t curved_path(tpath);

  curved_path.rewind(0);

  while ((code = curved_path.vertex(&x, &y)) != agg::path_cmd_stop) {
    if ((code & agg::path_cmd_end_poly) == agg::path_cmd_end_poly)
      continue;
    if (x < *x0) *x0 = x;
    if (y < *y0) *y0 = y;
    if (x > *x1) *x1 = x;
    if (y > *y1) *y1 = y;
  }
}

Py::Object _path_module::get_path_extents(const Py::Tuple& args) {
  args.verify_length(2);
  
  PathIterator path(args[0]);
  agg::trans_affine trans = py_to_agg_transformation_matrix(args[1]);

  double x0 =  std::numeric_limits<double>::infinity();
  double y0 =  std::numeric_limits<double>::infinity();
  double x1 = -std::numeric_limits<double>::infinity();
  double y1 = -std::numeric_limits<double>::infinity();

  ::get_path_extents(path, trans, &x0, &y0, &x1, &y1);

  Py::Tuple result(4);
  result[0] = Py::Float(x0);
  result[1] = Py::Float(y0);
  result[2] = Py::Float(x1);
  result[3] = Py::Float(y1);
  return result;
}

Py::Object _path_module::get_path_collection_extents(const Py::Tuple& args) {
  args.verify_length(5);

  //segments, trans, clipbox, colors, linewidths, antialiaseds
  agg::trans_affine	  master_transform = py_to_agg_transformation_matrix(args[0]);
  Py::SeqBase<Py::Object> paths		   = args[1];
  Py::SeqBase<Py::Object> transforms_obj   = args[2];
  Py::Object              offsets_obj      = args[3];
  agg::trans_affine       offset_trans     = py_to_agg_transformation_matrix(args[4], false);

  PyArrayObject* offsets = NULL;
  double x0, y0, x1, y1;

  try {
    offsets = (PyArrayObject*)PyArray_FromObject(offsets_obj.ptr(), PyArray_DOUBLE, 0, 2);
    if (!offsets || 
	(PyArray_NDIM(offsets) == 2 && PyArray_DIM(offsets, 1) != 2) || 
	(PyArray_NDIM(offsets) == 1 && PyArray_DIM(offsets, 0) != 0)) {
      throw Py::ValueError("Offsets array must be Nx2");
    }

    size_t Npaths      = paths.length();
    size_t Noffsets    = offsets->dimensions[0];
    size_t N	       = std::max(Npaths, Noffsets);
    size_t Ntransforms = std::min(transforms_obj.length(), N);
    size_t i;

    // Convert all of the transforms up front
    typedef std::vector<agg::trans_affine> transforms_t;
    transforms_t transforms;
    transforms.reserve(Ntransforms);
    for (i = 0; i < Ntransforms; ++i) {
      agg::trans_affine trans = py_to_agg_transformation_matrix
	(transforms_obj[i], false);
      trans *= master_transform;
      transforms.push_back(trans);
    }
    
    // The offset each of those and collect the mins/maxs
    x0 = std::numeric_limits<double>::infinity();
    y0 = std::numeric_limits<double>::infinity();
    x1 = -std::numeric_limits<double>::infinity();
    y1 = -std::numeric_limits<double>::infinity();
    agg::trans_affine trans;

    for (i = 0; i < N; ++i) {
      PathIterator path(paths[i % Npaths]);
      if (Ntransforms) {
	trans = transforms[i % Ntransforms];
      } else {
	trans = master_transform;
      }

      if (Noffsets) {
	double xo                = *(double*)PyArray_GETPTR2(offsets, i % Noffsets, 0);
	double yo                = *(double*)PyArray_GETPTR2(offsets, i % Noffsets, 1);
	offset_trans.transform(&xo, &yo);
	trans *= agg::trans_affine_translation(xo, yo);
      }

      ::get_path_extents(path, trans, &x0, &y0, &x1, &y1);
    }
  } catch (...) {
    Py_XDECREF(offsets);
    throw;
  }

  Py_XDECREF(offsets);

  Py::Tuple result(4);
  result[0] = Py::Float(x0);
  result[1] = Py::Float(y0);
  result[2] = Py::Float(x1);
  result[3] = Py::Float(y1);
  return result;
}

Py::Object _path_module::point_in_path_collection(const Py::Tuple& args) {
  args.verify_length(9);

  //segments, trans, clipbox, colors, linewidths, antialiaseds
  double		  x		   = Py::Float(args[0]);
  double		  y		   = Py::Float(args[1]);
  double                  radius           = Py::Float(args[2]);
  agg::trans_affine	  master_transform = py_to_agg_transformation_matrix(args[3]);
  Py::SeqBase<Py::Object> paths		   = args[4];
  Py::SeqBase<Py::Object> transforms_obj   = args[5];
  Py::SeqBase<Py::Object> offsets_obj      = args[6];
  agg::trans_affine       offset_trans     = py_to_agg_transformation_matrix(args[7]);
  bool                    filled           = Py::Int(args[8]);
  
  PyArrayObject* offsets = (PyArrayObject*)PyArray_FromObject(offsets_obj.ptr(), PyArray_DOUBLE, 0, 2);
  if (!offsets || 
      (PyArray_NDIM(offsets) == 2 && PyArray_DIM(offsets, 1) != 2) || 
      (PyArray_NDIM(offsets) == 1 && PyArray_DIM(offsets, 0) != 0)) {
    throw Py::ValueError("Offsets array must be Nx2");
  }

  size_t Npaths	     = paths.length();
  size_t Noffsets    = offsets->dimensions[0];
  size_t N	     = std::max(Npaths, Noffsets);
  size_t Ntransforms = std::min(transforms_obj.length(), N);
  size_t i;

  // Convert all of the transforms up front
  typedef std::vector<agg::trans_affine> transforms_t;
  transforms_t transforms;
  transforms.reserve(Ntransforms);
  for (i = 0; i < Ntransforms; ++i) {
    agg::trans_affine trans = py_to_agg_transformation_matrix
      (transforms_obj[i], false);
    trans *= master_transform;
    transforms.push_back(trans);
  }

  Py::List result;
  agg::trans_affine trans;

  for (i = 0; i < N; ++i) {
    PathIterator path(paths[i % Npaths]);

    if (Ntransforms) {
      trans = transforms[i % Ntransforms];
    } else {
      trans = master_transform;
    }

    if (Noffsets) {
      double xo = *(double*)PyArray_GETPTR2(offsets, i % Noffsets, 0);
      double yo = *(double*)PyArray_GETPTR2(offsets, i % Noffsets, 1);
      offset_trans.transform(&xo, &yo);
      trans *= agg::trans_affine_translation(xo, yo);
    }

    if (filled) {
      if (::point_in_path(x, y, path, trans))
	result.append(Py::Int((int)i));
    } else {
      if (::point_on_path(x, y, radius, path, trans))
	result.append(Py::Int((int)i));
    }
  }

  return result;
}

bool path_in_path(PathIterator& a, const agg::trans_affine& atrans,
		  PathIterator& b, const agg::trans_affine& btrans) {
  typedef agg::conv_transform<PathIterator> transformed_path_t;
  typedef agg::conv_curve<transformed_path_t> curve_t;

  if (a.total_vertices() < 3)
    return false;
  
  transformed_path_t b_path_trans(b, btrans);
  curve_t b_curved(b_path_trans);

  double x, y;
  b_curved.rewind(0);
  while (b_curved.vertex(&x, &y) != agg::path_cmd_stop) {
    if (!::point_in_path(x, y, a, atrans))
      return false;
  }
  
  return true;
}

Py::Object _path_module::path_in_path(const Py::Tuple& args) {
  args.verify_length(4);

  PathIterator a(args[0]);
  agg::trans_affine atrans = py_to_agg_transformation_matrix(args[1]);
  PathIterator b(args[2]);
  agg::trans_affine btrans = py_to_agg_transformation_matrix(args[3]);

  return Py::Int(::path_in_path(a, atrans, b, btrans));
}

/** The clip_path_to_rect code here is a clean-room implementation of the
    Sutherland-Hodgman clipping algorithm described here:
  
  http://en.wikipedia.org/wiki/Sutherland-Hodgman_clipping_algorithm
*/

typedef std::vector<std::pair<double, double> > Polygon;

namespace clip_to_rect_filters {
  /* There are four different passes needed to create/remove vertices
     (one for each side of the rectangle).  The differences between those
     passes are encapsulated in these functor classes.
  */
  struct bisectx {
    double m_x;

    bisectx(double x) : m_x(x) {}

    void bisect(double sx, double sy, double px, double py, double* bx, double* by) const {
      *bx = m_x;
      double dx = px - sx;
      double dy = py - sy;
      *by = sy + dy * ((m_x - sx) / dx);
    }
  };

  struct xlt : public bisectx {
    xlt(double x) : bisectx(x) {}

    bool operator()(double x, double y) const {
      return x <= m_x;
    }
  };

  struct xgt : public bisectx {
    xgt(double x) : bisectx(x) {}
    
    bool operator()(double x, double y) const {
      return x >= m_x;
    }
  };

  struct bisecty {
    double m_y;

    bisecty(double y) : m_y(y) {}

    void bisect(double sx, double sy, double px, double py, double* bx, double* by) const {
      *by = m_y;
      double dx = px - sx;
      double dy = py - sy;
      *bx = sx + dx * ((m_y - sy) / dy);
    }
  };

  struct ylt : public bisecty {
    ylt(double y) : bisecty(y) {}

    bool operator()(double x, double y) const {
      return y <= m_y;
    }
  };

  struct ygt : public bisecty {
    ygt(double y) : bisecty(y) {}

    bool operator()(double x, double y) const {
      return y >= m_y;
    }
  };
}

template<class Filter>
void clip_to_rect_one_step(const Polygon& polygon, Polygon& result, const Filter& filter) {
  double sx, sy, px, py, bx, by;
  bool sinside, pinside;
  result.clear();

  if (polygon.size() == 0)
    return;

  sx = polygon.back().first;
  sy = polygon.back().second;
  for (Polygon::const_iterator i = polygon.begin(); i != polygon.end(); ++i) {
    px = i->first;
    py = i->second;
    
    sinside = filter(sx, sy);
    pinside = filter(px, py);
    
    if (sinside) {
      if (pinside) {
	result.push_back(std::make_pair(px, py));
      } else {
	filter.bisect(sx, sy, px, py, &bx, &by);
	result.push_back(std::make_pair(bx, by));
      }
    } else {
      if (pinside) {
	filter.bisect(sx, sy, px, py, &bx, &by);
	result.push_back(std::make_pair(bx, by));
	result.push_back(std::make_pair(px, py));
      }
    }
    
    sx = px; sy = py;
  }
}   

void clip_to_rect(PathIterator& path, 
		  double x0, double y0, double x1, double y1, 
		  bool inside, std::vector<Polygon>& results) {
  double xmin, ymin, xmax, ymax;
  if (x0 < x1) {
    xmin = x0; xmax = x1;
  } else {
    xmin = x1; xmax = x0;
  }

  if (y0 < y1) {
    ymin = y0; ymax = y1;
  } else {
    ymin = y1; ymax = y0;
  }

  if (!inside) {
    std::swap(xmin, xmax);
    std::swap(ymin, ymax);
  }

  Polygon polygon1, polygon2;
  double x, y;
  unsigned code = 0;
  path.rewind(0);

  while (true) {
    polygon1.clear();
    while (true) {
      if (code == agg::path_cmd_move_to)
	polygon1.push_back(std::make_pair(x, y));

      code = path.vertex(&x, &y);

      if (code == agg::path_cmd_stop)
	break;

      if (code != agg::path_cmd_move_to)
	polygon1.push_back(std::make_pair(x, y));

      if ((code & agg::path_cmd_end_poly) == agg::path_cmd_end_poly) {
	break;
      } else if (code == agg::path_cmd_move_to) {
	break;
      }
    }

    // The result of each step is fed into the next (note the
    // swapping of polygon1 and polygon2 at each step).
    clip_to_rect_one_step(polygon1, polygon2, clip_to_rect_filters::xlt(xmax));
    clip_to_rect_one_step(polygon2, polygon1, clip_to_rect_filters::xgt(xmin));
    clip_to_rect_one_step(polygon1, polygon2, clip_to_rect_filters::ylt(ymax));
    clip_to_rect_one_step(polygon2, polygon1, clip_to_rect_filters::ygt(ymin));

    if (polygon1.size())
      results.push_back(polygon1);

    if (code == agg::path_cmd_stop)
      break;
  }
}

Py::Object _path_module::clip_path_to_rect(const Py::Tuple &args) {
  args.verify_length(3);
  
  PathIterator path(args[0]);
  Py::Object bbox_obj = args[1];
  bool inside = Py::Int(args[2]);

  double x0, y0, x1, y1;
  if (!py_convert_bbox(bbox_obj.ptr(), x0, y0, x1, y1))
    throw Py::TypeError("Argument 2 to clip_to_rect must be a Bbox object.");

  std::vector<Polygon> results;

  ::clip_to_rect(path, x0, y0, x1, y1, inside, results);

  // MGDTODO: Not exception safe
  int dims[2];
  dims[1] = 2;
  PyObject* py_results = PyList_New(results.size());
  for (std::vector<Polygon>::const_iterator p = results.begin(); p != results.end(); ++p) {
    size_t size = p->size();
    dims[0] = p->size();
    PyArrayObject* pyarray = (PyArrayObject*)PyArray_FromDims(2, dims, PyArray_DOUBLE);
    for (size_t i = 0; i < size; ++i) {
      ((double *)pyarray->data)[2*i] = (*p)[i].first;
      ((double *)pyarray->data)[2*i+1] = (*p)[i].second;
    }
    // MGDTODO: Error check
    PyList_SetItem(py_results, p - results.begin(), (PyObject *)pyarray);
  }

  return Py::Object(py_results, true);
}

struct XY {
  double x;
  double y;
};

extern "C"
DL_EXPORT(void)
  init_path(void)
{
  import_array();
  
  static _path_module* _path = NULL;
  _path = new _path_module;
};
