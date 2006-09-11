#include <functional>
#include <limits>


#include "_transforms.h"
#include "mplutils.h"
#include "MPL_isnan.h"

#ifdef NUMARRAY
#   include "numarray/arrayobject.h"
#else
#   ifdef NUMERIC
#       include "Numeric/arrayobject.h"
#   else
#       define PY_ARRAY_TYPES_PREFIX NumPy
#       include "numpy/arrayobject.h"
#   endif
#endif

Value::~Value() {
  _VERBOSE("Value::~Value");

}

Py::Object
Value::set(const Py::Tuple & args) {
  _VERBOSE("Value::set");
  args.verify_length(1);

  _val = Py::Float( args[0] );
  return Py::Object();
}

Py::Object
Value::get(const Py::Tuple & args) {
  _VERBOSE("Value::get");
  args.verify_length(0);

  return Py::Float( _val );
}

int
LazyValue::compare(const Py::Object &other) {
  if (!check(other))
    throw Py::TypeError("Can on compare LazyValues with LazyValues");
  LazyValue* pother = static_cast<LazyValue*>(other.ptr());
  double valself = val();
  double valother = pother->val();

  int ret;
  if (valself<valother) ret=-1;
  else if (valself==valother) ret=0;
  else ret=1;
  return ret;
}

Py::Object
LazyValue::number_add( const Py::Object &o ) {
  _VERBOSE("LazyValue::number");


  if (!LazyValue::check(o))
    throw Py::TypeError("Can only add LazyValues with other LazyValues");

  LazyValue* rhs = static_cast<LazyValue*>(o.ptr());

  return Py::asObject(new BinOp(this, rhs, BinOp::ADD));
}

Py::Object
LazyValue::number_divide( const Py::Object &o ) {
  _VERBOSE("LazyValue::number");

  //std::cout << "initing divide" << std::endl;
  if (!LazyValue::check(o))
    throw Py::TypeError("Can only divide LazyValues with other LazyValues");

  LazyValue* rhs = static_cast<LazyValue*>(o.ptr());
  BinOp* op = new BinOp(this, rhs, BinOp::DIVIDE);
  //std::cout << "initing divide done" << std::endl;
  return Py::asObject(op);
}

Py::Object
LazyValue::number_multiply( const Py::Object &o ) {
  _VERBOSE("LazyValue::number");


  if (!LazyValue::check(o))
    throw Py::TypeError("Can only multiply LazyValues with other LazyValues");

  LazyValue* rhs = static_cast<LazyValue*>(o.ptr());
  return Py::asObject(new BinOp(this, rhs, BinOp::MULTIPLY));
}

Py::Object
LazyValue::number_subtract( const Py::Object &o ) {
  _VERBOSE("LazyValue::number");


  if (!LazyValue::check(o))
    throw Py::TypeError("Can only subtract LazyValues with other LazyValues");

  LazyValue* rhs = static_cast<LazyValue*>(o.ptr());
  return Py::asObject(new BinOp(this, rhs, BinOp::SUBTRACT));
}

BinOp::BinOp(LazyValue* lhs, LazyValue* rhs, int opcode) :
  _lhs(lhs), _rhs(rhs), _opcode(opcode) {
  _VERBOSE("BinOp::BinOp");
  Py_INCREF(lhs);
  Py_INCREF(rhs);
}

BinOp::~BinOp() {
  _VERBOSE("BinOp::~BinOp");
  Py_DECREF(_lhs);
  Py_DECREF(_rhs);
}

Py::Object
BinOp::get(const Py::Tuple & args) {
  _VERBOSE("BinOp::get");
  args.verify_length(0);
  double x = val();
  return Py::Float( x );
}

Point::Point(LazyValue* x, LazyValue*  y) : _x(x), _y(y) {
  _VERBOSE("Point::Point");
  Py_INCREF(x);
  Py_INCREF(y);
}

Point::~Point()
{
  _VERBOSE("Point::~Point");
  Py_DECREF(_x);
  Py_DECREF(_y);

}

Interval::Interval(LazyValue* val1, LazyValue* val2) :
  _val1(val1), _val2(val2), _minpos(NULL) {
  _VERBOSE("Interval::Interval");
  Py_INCREF(val1);
  Py_INCREF(val2);
};

Interval::~Interval() {
  _VERBOSE("Interval::~Interval");
  Py_DECREF(_val1);
  Py_DECREF(_val2);

}

Py::Object
Interval::update(const Py::Tuple &args) {
  _VERBOSE("Interval::update");
  args.verify_length(2);

  Py::SeqBase<Py::Object> vals = args[0];

  //don't use current bounds when updating box if ignore==1
  int ignore = Py::Int(args[1]);
  size_t Nval = vals.length();
  if (Nval==0) return Py::Object();

  double minx = _val1->val();
  double maxx = _val2->val();

  double thisval;
  if (ignore) {
    thisval = Py::Float(vals[0]);
    minx = thisval;
    maxx = thisval;
  }


  for (size_t i=0; i<Nval; ++i) {
    thisval = Py::Float(vals[i]);
    if (thisval<minx) minx = thisval;
    if (thisval>maxx) maxx = thisval;
    _minpos->update(thisval);
  }


  _val1->set_api(minx);
  _val2->set_api(maxx);
  return Py::Object();
}

Bbox::Bbox(Point* ll, Point* ur) : _ll(ll), _ur(ur), _ignore(1) {
  _VERBOSE("Bbox::Bbox");

  Py_INCREF(ll);
  Py_INCREF(ur);
};


Bbox::~Bbox() {
  _VERBOSE("Bbox::~Bbox");
  Py_DECREF(_ll);
  Py_DECREF(_ur);
}

Py::Object
Bbox::_deepcopy() {

  double minx = _ll->xval();
  double miny = _ll->yval();

  double maxx = _ur->xval();
  double maxy = _ur->yval();

  return Py::asObject( new Bbox( new Point(new Value(minx), new Value(miny) ),
				 new Point(new Value(maxx), new Value(maxy) )));
}
Py::Object
Bbox::deepcopy(const Py::Tuple &args) {
  _VERBOSE("Bbox::deepcopy");
  args.verify_length(0);
  return _deepcopy();
}

Py::Object
Bbox::scale(const Py::Tuple &args) {
  _VERBOSE("Bbox::scale");
  args.verify_length(2);
  double sx = Py::Float(args[0]);
  double sy = Py::Float(args[1]);

  double minx = _ll->xval();
  double miny = _ll->yval();
  double maxx = _ur->xval();
  double maxy = _ur->yval();

  double w = maxx-minx;
  double h = maxy-miny;


  double deltaw = (sx*w-w)/2.0;
  double deltah = (sy*h-h)/2.0;

  _ll->x_api()->set_api(minx-deltaw);
  _ur->x_api()->set_api(maxx+deltaw);

  _ll->y_api()->set_api(miny-deltah);
  _ur->y_api()->set_api(maxy+deltah);

  return Py::Object();
}

Py::Object
Bbox::get_bounds(const Py::Tuple & args) {
  _VERBOSE("Bbox::get_bounds");
  args.verify_length(0);



  double minx = _ll->xval();
  double miny = _ll->yval();
  double maxx = _ur->xval();
  double maxy = _ur->yval();

  double width  = maxx - minx;
  double height = maxy - miny;

  Py::Tuple ret(4);
  ret[0] = Py::Float(minx);
  ret[1] = Py::Float(miny);
  ret[2] = Py::Float(width);
  ret[3] = Py::Float(height);
  return ret;
}

Py::Object
Bbox::count_contains(const Py::Tuple &args) {
  _VERBOSE("Bbox::count_contains");
  args.verify_length(1);

  Py::SeqBase<Py::Object> xys = args[0];
  size_t Nxys = xys.length();
  long count = 0;

  double minx = _ll->xval();
  double miny = _ll->yval();
  double maxx = _ur->xval();
  double maxy = _ur->yval();

  for(size_t i=0; i < Nxys; i++) {
    Py::SeqBase<Py::Object> xy(xys[i]);
    xy.verify_length(2);
    double x = Py::Float(xy[0]);
    double y = Py::Float(xy[1]);
    int inx = ( (x>=minx) && (x<=maxx) || (x>=maxx) && (x<=minx) );
    if (!inx) continue;
    int iny = ( (y>=miny) && (y<=maxy) || (y>=maxy) && (y<=miny) );
    if (!iny) continue;
    count += 1;
  }
  return Py::Int(count);
}

Py::Object
Bbox::contains(const Py::Tuple &args) {
  _VERBOSE("Bbox::contains");
  args.verify_length(2);

  double x = Py::Float(args[0]);
  double y = Py::Float(args[1]);

  double minx = _ll->xval();
  double miny = _ll->yval();
  double maxx = _ur->xval();
  double maxy = _ur->yval();

  int inx = ( (x>=minx) && (x<=maxx) || (x>=maxx) && (x<=minx) );
  if (!inx) return Py::Int(0);
  int iny = ( (y>=miny) && (y<=maxy) || (y>=maxy) && (y<=miny) );
  return Py::Int(iny);
}

Py::Object
Bbox::overlaps(const Py::Tuple &args) {
  _VERBOSE("Bbox::overlaps");
  args.verify_length(1);

  if (! check(args[0]))
    throw Py::TypeError("Expected a bbox");

  int x = Py::Int( overlapsx(args) );
  int y = Py::Int( overlapsy(args) );
  return Py::Int(x&&y);
}

Py::Object
Bbox::ignore(const Py::Tuple &args) {
  _VERBOSE("Bbox::ignore");
  args.verify_length(1);
  _ignore = Py::Int(args[0]);
  return Py::Object();
}

Py::Object
Bbox::overlapsx(const Py::Tuple &args) {
  _VERBOSE("Bbox::overlapsx");
  args.verify_length(1);

  if (! check(args[0]))
    throw Py::TypeError("Expected a bbox");

  Bbox* other = static_cast<Bbox*>(args[0].ptr());

  double minx = _ll->xval();
  double maxx = _ur->xval();

  double ominx = other->_ll->xval();
  double omaxx = other->_ur->xval();

  int b =  ( ( (ominx>=minx) && (ominx<=maxx)) ||
	     ( (minx>=ominx) && (minx<=omaxx)) );
  return Py::Int(b);

}

Py::Object
Bbox::overlapsy(const Py::Tuple &args) {
  _VERBOSE("Bbox::overlapsy");
  args.verify_length(1);

  if (! check(args[0]))
    throw Py::TypeError("Expected a bbox");

  Bbox* other = static_cast<Bbox*>(args[0].ptr());

  double miny = _ll->yval();
  double maxy = _ur->yval();

  double ominy = other->_ll->yval();
  double omaxy = other->_ur->yval();



  int b =  ( ( (ominy>=miny) && (ominy<=maxy)) ||
	     ( (miny>=ominy) && (miny<=omaxy)) );
  return Py::Int(b);

}


/*
As for how the datalim handling works, the syntax is

  self.dataLim.update(xys, ignore)

Note this is different than the ax.update_datalim method, which calls
it.  datalim is a bbox which has an ignore state variable (boolean).

The ignore argument to update datalim can take on three values

  0: do not ignore the current limits and update them with the xys
  1: ignore the current datalim limits and override with xys
 -1: use the datalim ignore state to determine the ignore settings

This seems a bit complex but arose from experience.  Basically a lot
of different objects want to add their data to the datalim.  In most
use cases, you want the first object to add data to ignore the current
limits (which are just default values) and subsequent objects to add
to the datalim taking into account the previous limits.  The default
behavior of datalim is to set ignore to 1, and after the first call
with -1 set ignore to 0.  Thus everyone can call with -1 and have the
desired default behavior .  I hope you are all confused now.

One can manually set the ignore state var with

  datalim.ignore(1)
*/
Py::Object
Bbox::update(const Py::Tuple &args) {
  _VERBOSE("Bbox::update");
  args.verify_length(2);

  Py::Object test = args[0];
  if (test.hasAttr("shape")) return Bbox::update_numerix_xy(args);

  Py::SeqBase<Py::Object> xys = args[0];

  //don't use current bounds on first update
  int ignore = Py::Int(args[1]);
  if (ignore==-1) {
    ignore = _ignore;
    _ignore = 0; // don't ignore future updates
  }

  size_t Nx = xys.length();
  if (Nx==0) return Py::Object();


  double minx = _ll->xval();
  double maxx = _ur->xval();
  double miny = _ll->yval();
  double maxy = _ur->yval();

  Py::Tuple tup;
  if (ignore) {
    tup = xys[0];
    double x = Py::Float(tup[0]);
    double y = Py::Float(tup[1]);

    minx=x;
    maxx=x;
    miny=y;
    maxy=y;
  }


  for (size_t i=0; i<Nx; ++i) {
    tup = xys[i];
    double x = Py::Float(tup[0]);
    double y = Py::Float(tup[1]);
    _posx.update(x);
    _posy.update(y);
    if (x<minx) minx=x;
    if (x>maxx) maxx=x;
    if (y<miny) miny=y;
    if (y>maxy) maxy=y;

  }


  _ll->x_api()->set_api(minx);
  _ll->y_api()->set_api(miny);
  _ur->x_api()->set_api(maxx);
  _ur->y_api()->set_api(maxy);
  return Py::Object();
}

// Replace update with the following?
Py::Object
Bbox::update_numerix_xy(const Py::Tuple &args) {
  //update the box from the numerix array xy
  _VERBOSE("Bbox::update_numerix_xy");

  args.verify_length(2);

  Py::Object xyo = args[0];

  PyArrayObject *xyin = (PyArrayObject *) PyArray_FromObject(xyo.ptr(),
                                                      PyArray_DOUBLE, 2, 2);

  if (xyin==NULL)
    throw Py::TypeError("Bbox::update_numerix_xy expected numerix array");

  size_t Nxy = xyin->dimensions[0];
  size_t N2 = xyin->dimensions[1];

  if (N2 != 2)
    throw Py::ValueError("xy array must have shape (N, 2)");

  //don't use current bounds when updating box if ignore==1


  if (Nxy==0) return Py::Object();

  double minx = _ll->xval();
  double maxx = _ur->xval();
  double miny = _ll->yval();
  double maxy = _ur->yval();

  double thisx, thisy;
  //don't use current bounds on first update
  int ignore = Py::Int(args[1]);
  if (ignore==-1) {
    ignore = _ignore;
    _ignore = 0; // don't ignore future updates
  }
  if (ignore) {
    minx = miny = std::numeric_limits<double>::max();
    maxx = maxy = -std::numeric_limits<double>::max();
  }

  int ngood = 0;
  for (size_t i=0; i< Nxy; ++i) {
    thisx = *(double *)(xyin->data + i*xyin->strides[0]);
    thisy = *(double *)(xyin->data + i*xyin->strides[0] + xyin->strides[1]);
    if (MPL_isnan64(thisx) || MPL_isnan64(thisy)) continue;
    _posx.update(thisx);
    _posy.update(thisy);
    if (thisx<minx) minx=thisx;
    if (thisx>maxx) maxx=thisx;
    if (thisy<miny) miny=thisy;
    if (thisy>maxy) maxy=thisy;
    ngood++;
  }

  Py_XDECREF(xyin);
  if (ngood) {
    _ll->x_api()->set_api(minx);
    _ll->y_api()->set_api(miny);
    _ur->x_api()->set_api(maxx);
    _ur->y_api()->set_api(maxy);
  }
  return Py::Object();
}


Py::Object
Bbox::update_numerix(const Py::Tuple &args) {
  //update the box from the numerix arrays x and y
  _VERBOSE("Bbox::update_numerix");

  args.verify_length(3);

  Py::Object xo = args[0];
  Py::Object yo = args[1];

  PyArrayObject *x = (PyArrayObject *) PyArray_ContiguousFromObject(xo.ptr(), PyArray_DOUBLE, 1, 1);

  if (x==NULL)
    throw Py::TypeError("Bbox::update_numerix expected numerix array");


  PyArrayObject *y = (PyArrayObject *) PyArray_ContiguousFromObject(yo.ptr(), PyArray_DOUBLE, 1, 1);

  if (y==NULL)
    throw Py::TypeError("Bbox::update_numerix expected numerix array");


  size_t Nx = x->dimensions[0];
  size_t Ny = y->dimensions[0];

  if (Nx!=Ny)
    throw Py::ValueError("x and y must be equal length sequences");

  //don't use current bounds when updating box if ignore==1


  if (Nx==0) return Py::Object();

  double minx = _ll->xval();
  double maxx = _ur->xval();
  double miny = _ll->yval();
  double maxy = _ur->yval();

  double thisx, thisy;
  //don't use current bounds on first update
  int ignore = Py::Int(args[2]);
  if (ignore==-1) {
    ignore = _ignore;
    _ignore = 0; // don't ignore future updates
  }
  if (ignore) {
    int xok=0;
    int yok=0;
    // loop through values until we find some nans...
    for (size_t i=0; i< Nx; ++i) {
      thisx = *(double *)(x->data + i*x->strides[0]);
      thisy = *(double *)(y->data + i*y->strides[0]);

      if (!xok) {
	if (!MPL_isnan64(thisx)) {
	  minx=thisx;
	  maxx=thisx;
	  xok=1;
	}
      }

      if (!yok) {
	if (!MPL_isnan64(thisy)) {
	  miny=thisy;
	  maxy=thisy;
	  yok=1;
	}
      }

      if (xok && yok) break;
    }
  }

  for (size_t i=0; i< Nx; ++i) {
    thisx = *(double *)(x->data + i*x->strides[0]);
    thisy = *(double *)(y->data + i*y->strides[0]);

    _posx.update(thisx);
    _posy.update(thisy);
    if (thisx<minx) minx=thisx;
    if (thisx>maxx) maxx=thisx;
    if (thisy<miny) miny=thisy;
    if (thisy>maxy) maxy=thisy;


  }

  Py_XDECREF(x);
  Py_XDECREF(y);


  _ll->x_api()->set_api(minx);
  _ll->y_api()->set_api(miny);
  _ur->x_api()->set_api(maxx);
  _ur->y_api()->set_api(maxy);
  return Py::Object();
}

Func::~Func() {
  _VERBOSE("Func::~Func");
}


Py::Object
Func::map(const Py::Tuple &args) {
  _VERBOSE("Func::map");
  args.verify_length(1);
  double xin = Py::Float(args[0]);

  double xout;
  try {
    xout = this->operator()(xin);
  }
  catch(...) {
    throw Py::ValueError("Domain error on Func::map");
  }

  return Py::Float(xout);

};

Py::Object
Func::inverse(const Py::Tuple &args) {
  _VERBOSE("Func::inverse");

  args.verify_length(1);
  double xin = Py::Float(args[0]);

  double xout = this->inverse_api(xin);
  return Py::Float(xout);
};

/*
Py::Object
FuncUserDef::map(const Py::Tuple &args) {
} // FuncUserDef::map()

Py::Object
FuncUserDef::inverse(const Py::Tuple &args) {
  _VERBOSE("FuncUserDef::inverse");

  return Func::inverse(args);
} // inverse()
*/

Py::Object
FuncXY::map(const Py::Tuple &args) {
  _VERBOSE("FuncXY::map");

  args.verify_length(2);
  double xin = Py::Float(args[0]);
  double yin = Py::Float(args[1]);

  std::pair<double, double> xy;
  try {
    xy = this->operator()(xin, yin);
  }
  catch(...) {
    throw Py::ValueError("Domain error on FuncXY nonlinear transform");
  }

  Py::Tuple ret(2);
  double xout = xy.first;
  double yout = xy.second;
  ret[0] = Py::Float(xout);
  ret[1] = Py::Float(yout);
  return ret;

  //return Py::Object();
};

Py::Object
FuncXY::inverse(const Py::Tuple &args) {
  _VERBOSE("FuncXY::inverse");

  args.verify_length(2);
  double xin = Py::Float(args[0]);
  double yin = Py::Float(args[1]);

  std::pair<double, double> xy = this->inverse_api(xin, yin);

  Py::Tuple ret(2);
  double xout = xy.first;
  double yout = xy.second;
  ret[0] = Py::Float(xout);
  ret[1] = Py::Float(yout);
  return ret;

};


Transformation::~Transformation() {
  _VERBOSE("Transformation::~Transformation");
  if (_transOffset!=NULL) {
    Py_DECREF(_transOffset);
  }

}

Py::Object
Transformation::as_vec6(const Py::Tuple & args) {
  _VERBOSE("Transformation::as_vec6");
  throw Py::RuntimeError("This transformation does not support as_vec6");
  return Py::Object();
}



Py::Object
Transformation::get_funcx(const Py::Tuple & args) {
  _VERBOSE("Transformation::get_funcx");
  throw Py::RuntimeError("This transformation does not support get_funcx");
  return Py::Object();
}

Py::Object
Transformation::get_funcy(const Py::Tuple & args) {
  _VERBOSE("Transformation::get_funcy");
  throw Py::RuntimeError("This transformation does not support get_funcy");
  return Py::Object();
}


Py::Object
Transformation::set_funcx(const Py::Tuple & args) {
  _VERBOSE("Transformation::set_funcx");
  throw Py::RuntimeError("This transformation does not support set_funcx");
  return Py::Object();
}

Py::Object
Transformation::set_funcy(const Py::Tuple & args) {
  _VERBOSE("Transformation::set_funcy");
  throw Py::RuntimeError("This transformation does not support set_funcy");
  return Py::Object();
}


Py::Object
Transformation::get_funcxy(const Py::Tuple & args) {
  _VERBOSE("Transformation::get_funcxy");
  throw Py::RuntimeError("This transformation does not support get_funcxy");
  return Py::Object();
}


Py::Object
Transformation::set_funcxy(const Py::Tuple & args) {
  _VERBOSE("Transformation::set_funcxy");
  throw Py::RuntimeError("This transformation does not support set_funcxy");
  return Py::Object();
}

Py::Object
Transformation::get_bbox1(const Py::Tuple & args) {
  _VERBOSE("Transformation::get_bbox1");
  throw Py::RuntimeError("This transformation does not support get_bbox1");
  return Py::Object();
}

Py::Object
Transformation::get_bbox2(const Py::Tuple & args) {
  _VERBOSE("Transformation::get_bbox2");
  throw Py::RuntimeError("This transformation does not support get_bbox2");
  return Py::Object();
}


Py::Object
Transformation::set_bbox1(const Py::Tuple & args) {
  _VERBOSE("Transformation::set_bbox1");
  throw Py::RuntimeError("This transformation does not support set_bbox1");
  return Py::Object();
}

Py::Object
Transformation::set_bbox2(const Py::Tuple & args) {
  _VERBOSE("Transformation::set_bbox2");
  throw Py::RuntimeError("This transformation does not support set_bbox1");
  return Py::Object();
}

Py::Object
Transformation::set_offset(const Py::Tuple & args) {
  _VERBOSE("Transformation::set_offset");
  args.verify_length(2);

  Py::SeqBase<Py::Object> xy = args[0];
  //std::cout << "checking args" << std::endl;
  if (!check(args[1]))
    throw Py::TypeError("Transformation::set_offset(xy,trans) requires trans to be a Transformation instance");

  //std::cout << "getting x,y" << std::endl;

  _usingOffset = 1;
  _xo = Py::Float(xy[0]);
  _yo = Py::Float(xy[1]);
  //std::cout << "casting" << std::endl;
  _transOffset = static_cast<Transformation*>(args[1].ptr());
  //std::cout << "increffing" << std::endl;
  Py_INCREF(_transOffset);
  //std::cout << "returning" << std::endl;
  return Py::Object();
}



Py::Object
Transformation::inverse_xy_tup(const Py::Tuple & args) {
  _VERBOSE("Transformation::inverse_xy_tup");
  args.verify_length(1);

  Py::Tuple tup = args[0];
  double xin = Py::Float(tup[0]);
  double yin = Py::Float(tup[1]);

  try {
    if (!_frozen) eval_scalars();
  }
  catch(...) {
    throw Py::ValueError("Domain error on inverse_xy_tup");
  }


  inverse_api(xin, yin);
  Py::Tuple ret(2);
  ret[0] = Py::Float(xy.first);
  ret[1] = Py::Float(xy.second);
  return ret;

}

Py::Object
Transformation::inverse_numerix_xy(const Py::Tuple & args) {
  _VERBOSE("Transformation::inverse_numerix_xy");
  args.verify_length(1);

  Py::Object xyo = args[0];

  PyArrayObject *xyin = (PyArrayObject *) PyArray_FromObject(xyo.ptr(),
                                                   PyArray_DOUBLE, 2, 2);

  if (xyin==NULL)
    throw Py::TypeError("Transformation::inverse_numerix_xy expected numerix array");

  size_t Nxy = xyin->dimensions[0];
  size_t N2 = xyin->dimensions[1];

  if (N2!=2) {
    Py_XDECREF(xyin);
    throw Py::ValueError("xy must have shape (N,2)");
  }

  // evaluate the lazy objects
  try {
    if (!_frozen) eval_scalars();
  }
  catch(...) {
    Py_XDECREF(xyin);
    throw Py::ValueError("Domain error on Transformation::inverse_numerix_xy");
  }

  int dimensions[2];
  dimensions[0] = Nxy;
  dimensions[1] = 2;

  PyArrayObject *retxy = (PyArrayObject *)PyArray_FromDims(2,dimensions,
                                                            PyArray_DOUBLE);
  if (retxy==NULL) {
    Py_XDECREF(xyin);
    throw Py::RuntimeError("Could not create return xy array");
  }

  double nan = std::numeric_limits<float>::quiet_NaN();
  for (size_t i=0; i< Nxy; ++i) {
    double thisx = *(double *)(xyin->data + i*xyin->strides[0]);
    double thisy = *(double *)(xyin->data + i*xyin->strides[0] +
                                                xyin->strides[1]);
    try {
      inverse_api(thisx, thisy);
    }
    catch(...) {
      xy.first = nan;
      xy.second = nan;
      //throw Py::ValueError("Domain error on Transformation::inverse_numerix_xy");
    }
    *(double *)(retxy->data + i*retxy->strides[0]) = xy.first;
    *(double *)(retxy->data + i*retxy->strides[0] +
                                    retxy->strides[1]) = xy.second;
  }

  Py_XDECREF(xyin);
  return Py::asObject((PyObject *)retxy);
}




Py::Object
Transformation::xy_tup(const Py::Tuple & args) {
  _VERBOSE("Transformation::xy_tup");
  args.verify_length(1);

  try {
    if (!_frozen) eval_scalars();
  }
  catch(...) {
    throw Py::ValueError("Domain error on nonlinear transform");
  }


  Py::SeqBase<Py::Object> xytup = args[0];
  double x = Py::Float(xytup[0]);
  double y = Py::Float(xytup[1]);


  Py::Tuple out(2);
  try {
    this->operator()(x, y);
  }
  catch(...) {
    throw Py::ValueError("Domain error on nTransformation::xy_tup operator()(x,y)");
  }

  out[0] = Py::Float( xy.first );
  out[1] = Py::Float( xy.second );
  return out;
}

Py::Object
Transformation::seq_x_y(const Py::Tuple & args) {
  _VERBOSE("Transformation::seq_x_y");
  args.verify_length(2);


  Py::SeqBase<Py::Object> x = args[0];
  Py::SeqBase<Py::Object> y = args[1];

  size_t Nx = x.length();
  size_t Ny = y.length();

  if (Nx!=Ny)
    throw Py::ValueError("x and y must be equal length sequences");

  // evaluate the lazy objects
  try {
    if (!_frozen) eval_scalars();
  }
  catch(...) {
    throw Py::ValueError("Domain error on Transformation::seq_x_y");
  }


  Py::Tuple xo(Nx);
  Py::Tuple yo(Nx);


  for (size_t i=0; i< Nx; ++i) {
    double thisx = Py::Float(x[i]);
    double thisy = Py::Float(y[i]);
    try {
      this->operator()(thisx, thisy);
    }
    catch(...) {
      throw Py::ValueError("Domain error on Transformation::seq_x_y operator()(thisx, thisy)");
    }

    xo[i] = Py::Float( xy.first );
    yo[i] = Py::Float( xy.second );
  }

  Py::Tuple ret(2);
  ret[0] = xo;
  ret[1] = yo;
  return ret;
}

Py::Object
Transformation::numerix_xy(const Py::Tuple & args) {
  _VERBOSE("Transformation::numerix_xy");
  args.verify_length(1);

  Py::Object xyo = args[0];

  PyArrayObject *xyin = (PyArrayObject *) PyArray_FromObject(xyo.ptr(),
                                                   PyArray_DOUBLE, 2, 2);

  if (xyin==NULL)
    throw Py::TypeError("Transformation::numerix_xy expected numerix array");

  size_t Nxy = xyin->dimensions[0];
  size_t N2 = xyin->dimensions[1];

  if (N2!=2) {
    Py_XDECREF(xyin);
    throw Py::ValueError("xy must have shape (N,2)");
  }

  // evaluate the lazy objects
  try {
    if (!_frozen) eval_scalars();
  }
  catch(...) {
    Py_XDECREF(xyin);
    throw Py::ValueError("Domain error on Transformation::numerix_xy");
  }

  int dimensions[2];
  dimensions[0] = Nxy;
  dimensions[1] = 2;

  PyArrayObject *retxy = (PyArrayObject *)PyArray_FromDims(2,dimensions,
                                                            PyArray_DOUBLE);
  if (retxy==NULL) {
    Py_XDECREF(xyin);
    throw Py::RuntimeError("Could not create return xy array");
  }

  double nan = std::numeric_limits<float>::quiet_NaN();
  for (size_t i=0; i< Nxy; ++i) {
    double thisx = *(double *)(xyin->data + i*xyin->strides[0]);
    double thisy = *(double *)(xyin->data + i*xyin->strides[0] +
                                                xyin->strides[1]);
    try {
      this->operator()(thisx, thisy);
    }
    catch(...) {
      xy.first = nan;
      xy.second = nan;
      //throw Py::ValueError("Domain error on Transformation::numerix_xy");
    }
    *(double *)(retxy->data + i*retxy->strides[0]) = xy.first;
    *(double *)(retxy->data + i*retxy->strides[0] +
                                    retxy->strides[1]) = xy.second;
  }

  Py_XDECREF(xyin);
  return Py::asObject((PyObject *)retxy);
}


Py::Object
Transformation::numerix_x_y(const Py::Tuple & args) {
  _VERBOSE("Transformation::numerix_x_y");
  args.verify_length(2);


  Py::Object xo = args[0];
  Py::Object yo = args[1];

  PyArrayObject *x = (PyArrayObject *) PyArray_ContiguousFromObject(xo.ptr(), PyArray_DOUBLE, 1, 1);

  if (x==NULL)
    throw Py::TypeError("Transformation::numerix_x_y expected numerix array");


  PyArrayObject *y = (PyArrayObject *) PyArray_ContiguousFromObject(yo.ptr(), PyArray_DOUBLE, 1, 1);

  if (y==NULL)
    throw Py::TypeError("Transformation::numerix_x_y expected numerix array");


  size_t Nx = x->dimensions[0];
  size_t Ny = y->dimensions[0];

  if (Nx!=Ny)
    throw Py::ValueError("x and y must be equal length sequences");

  // evaluate the lazy objects
  try {
    if (!_frozen) eval_scalars();
  }
  catch(...) {
    throw Py::ValueError("Domain error on Transformation::numerix_x_y");
  }


  int dimensions[1];
  dimensions[0] = Nx;


  PyArrayObject *retx = (PyArrayObject *)PyArray_FromDims(1,dimensions,PyArray_DOUBLE);
  if (retx==NULL) {
    Py_XDECREF(x);
    Py_XDECREF(y);
    throw Py::RuntimeError("Could not create return x array");
  }

  PyArrayObject *rety = (PyArrayObject *)PyArray_FromDims(1,dimensions,PyArray_DOUBLE);
  if (rety==NULL) {
    Py_XDECREF(x);
    Py_XDECREF(y);
    throw Py::RuntimeError("Could not create return x array");
  }

  double nan = std::numeric_limits<float>::quiet_NaN();

  for (size_t i=0; i< Nx; ++i) {

    double thisx = *(double *)(x->data + i*x->strides[0]);
    double thisy = *(double *)(y->data + i*y->strides[0]);
    //std::cout << "calling operator " << thisx << " " << thisy << " " << std::endl;
    try {
      this->operator()(thisx, thisy);
    }
    catch(...) {
      xy.first = nan;
      xy.second = nan;
      //throw Py::ValueError("Domain error on Transformation::numerix_x_y");
    }

    *(double *)(retx->data + i*retx->strides[0]) = xy.first;
    *(double *)(rety->data + i*rety->strides[0]) = xy.second;
  }

  Py_XDECREF(x);
  Py_XDECREF(y);

  Py::Tuple ret(2);
  ret[0] = Py::Object((PyObject*)retx);
  ret[1] = Py::Object((PyObject*)rety);
  Py_XDECREF(retx);
  Py_XDECREF(rety);
  return ret;
}

Py::Object
Transformation::nonlinear_only_numerix(const Py::Tuple & args, const Py::Dict &kwargs) {
  _VERBOSE("Transformation::nonlinear_only_numerix");
  args.verify_length(2);

  int returnMask = false;
  if (kwargs.hasKey("returnMask")) {
    returnMask = Py::Int(kwargs["returnMask"]);
  }

  Py::Object xo = args[0];
  Py::Object yo = args[1];

  PyArrayObject *x = (PyArrayObject *) PyArray_ContiguousFromObject(xo.ptr(), PyArray_DOUBLE, 1, 1);

  if (x==NULL)
    throw Py::TypeError("Transformation::nonlinear_only_numerix expected numerix array");


  PyArrayObject *y = (PyArrayObject *) PyArray_ContiguousFromObject(yo.ptr(), PyArray_DOUBLE, 1, 1);

  if (y==NULL)
    throw Py::TypeError("Transformation::nonlinear_only_numerix expected numerix array");


  size_t Nx = x->dimensions[0];
  size_t Ny = y->dimensions[0];

  if (Nx!=Ny)
    throw Py::ValueError("x and y must be equal length sequences");

  int dimensions[1];
  dimensions[0] = Nx;


  PyArrayObject *retx = (PyArrayObject *)PyArray_FromDims(1,dimensions,PyArray_DOUBLE);
  if (retx==NULL) {
    Py_XDECREF(x);
    Py_XDECREF(y);
    throw Py::RuntimeError("Could not create return x array");
  }

  PyArrayObject *rety = (PyArrayObject *)PyArray_FromDims(1,dimensions,PyArray_DOUBLE);
  if (rety==NULL) {
    Py_XDECREF(x);
    Py_XDECREF(y);
    Py_XDECREF(retx);
    throw Py::RuntimeError("Could not create return x array");
  }

  PyArrayObject *retmask = NULL;

  if (returnMask) {
    retmask = (PyArrayObject *)PyArray_FromDims(1,dimensions,PyArray_UBYTE);
    if (retmask==NULL) {
      Py_XDECREF(x);
      Py_XDECREF(y);
      Py_XDECREF(retx);
      Py_XDECREF(rety);
      throw Py::RuntimeError("Could not create return mask array");
    }

  }


  for (size_t i=0; i< Nx; ++i) {

    double thisx = *(double *)(x->data + i*x->strides[0]);
    double thisy = *(double *)(y->data + i*y->strides[0]);
    if (MPL_isnan64(thisx) || MPL_isnan64(thisy)) {
      if (returnMask) {
	*(unsigned char *)(retmask->data + i*retmask->strides[0]) = 0;
      }
      double MPLnan; // don't require C99 math features - find our own nan
      if (MPL_isnan64(thisx)) {
	MPLnan=thisx;
      } else {
	MPLnan=thisy;
      }
      *(double *)(retx->data + i*retx->strides[0]) = MPLnan;
      *(double *)(rety->data + i*rety->strides[0]) = MPLnan;
    } else {
      try {
	this->nonlinear_only_api(&thisx, &thisy);
      }
      catch(...) {

	if (returnMask) {
	  *(unsigned char *)(retmask->data + i*retmask->strides[0]) = 0;
	  *(double *)(retx->data + i*retx->strides[0]) = 0.0;
	  *(double *)(rety->data + i*rety->strides[0]) = 0.0;
	  continue;
	}
	else {
	  throw Py::ValueError("Domain error on this->nonlinear_only_api(&thisx, &thisy) in Transformation::nonlinear_only_numerix");
	}
      }

      *(double *)(retx->data + i*retx->strides[0]) = thisx;
      *(double *)(rety->data + i*rety->strides[0]) = thisy;
      if (returnMask) {
	*(unsigned char *)(retmask->data + i*retmask->strides[0]) = 1;
      }
    }

  }

  Py_XDECREF(x);
  Py_XDECREF(y);

  if (returnMask) {
    Py::Tuple ret(3);
    ret[0] = Py::Object((PyObject*)retx);
    ret[1] = Py::Object((PyObject*)rety);
    ret[2] = Py::Object((PyObject*)retmask);
    Py_XDECREF(retx);
    Py_XDECREF(rety);
    Py_XDECREF(retmask);
    return ret;
  }
  else {
    Py::Tuple ret(2);
    ret[0] = Py::Object((PyObject*)retx);
    ret[1] = Py::Object((PyObject*)rety);
    Py_XDECREF(retx);
    Py_XDECREF(rety);
    return ret;

  }


}

Py::Object
Transformation::seq_xy_tups(const Py::Tuple & args) {
  _VERBOSE("Transformation::seq_xy_tups");
  args.verify_length(1);

  Py::Object test = args[0];
  if (test.hasAttr("shape")) return Transformation::numerix_xy(args);

  Py::SeqBase<Py::Object> xytups = args[0];


  size_t Nx = xytups.length();

  try {
    if (!_frozen) eval_scalars();
  }
  catch(...) {
    throw Py::ValueError("Domain error on Transformation::seq_xy_tups");
  }

  Py::Tuple ret(Nx);
  Py::SeqBase<Py::Object> xytup;



  for (size_t i=0; i< Nx; ++i) {
    xytup = Py::SeqBase<Py::Object>( xytups[i] );

    double thisx = Py::Float(xytup[0]);
    double thisy = Py::Float(xytup[1]);

    try {
      this->operator()(thisx, thisy);
    }
    catch(...) {
      throw Py::ValueError("Domain error on nonlinear Transformation::seq_xy_tups operator()(thisx, thisy)");
    }


    Py::Tuple out(2);
    out[0] = Py::Float( xy.first );
    out[1] = Py::Float( xy.second );
    ret[i] = out;
  }

  return ret;
}


BBoxTransformation::BBoxTransformation(Bbox *b1, Bbox *b2) :
    Transformation(),
    _b1(b1), _b2(b2)  {
  _VERBOSE("BBoxTransformation::BBoxTransformation");
  Py_INCREF(b1);
  Py_INCREF(b2);

}

BBoxTransformation::~BBoxTransformation() {
  _VERBOSE("BBoxTransformation::~BBoxTransformation");
  Py_DECREF(_b1);
  Py_DECREF(_b2);
}

Py::Object
BBoxTransformation::get_bbox1(const Py::Tuple & args) {
  _VERBOSE("BBoxTransformation::get_bbox1");
  args.verify_length(0);
  return Py::Object(_b1);
}

Py::Object
BBoxTransformation::get_bbox2(const Py::Tuple & args) {
  _VERBOSE("BBoxTransformation::get_bbox2");
  args.verify_length(0);
  return Py::Object(_b2);
}


Py::Object
BBoxTransformation::set_bbox1(const Py::Tuple & args) {
  _VERBOSE("BBoxTransformation::set_bbox1");
  args.verify_length(1);
  if (!Bbox::check(args[0]))
    throw Py::TypeError("set_bbox1(func) expected a func instance");
  _b1 = static_cast<Bbox*>(args[0].ptr());
  Py_INCREF(_b1);
  return Py::Object();
}

Py::Object
BBoxTransformation::set_bbox2(const Py::Tuple & args) {
  _VERBOSE("BBoxTransformation::set_bbox2");
  args.verify_length(1);
  if (!Bbox::check(args[0]))
    throw Py::TypeError("set_bbox2(func) expected a func instance");
  _b2 = static_cast<Bbox*>(args[0].ptr());
  Py_INCREF(_b2);
  return Py::Object();
}

void
BBoxTransformation::affine_params_api(double* a, double* b, double* c, double* d, double* tx, double* ty) {
  //get the scale and translation factors of the separable transform
  //sx, sy, tx, ty
    if (!_frozen) eval_scalars();
    *a = _sx;
    *b = 0.0;
    *c = 0.0;
    *d = _sy;

    *tx = _tx;
    *ty = _ty;

    if (_usingOffset) {
      *tx  += _xot;
      *ty  += _yot;
    }

}

SeparableTransformation::SeparableTransformation(Bbox *b1, Bbox *b2, Func *funcx, Func *funcy) :
    BBoxTransformation(b1, b2),
    _funcx(funcx), _funcy(funcy)  {
  _VERBOSE("SeparableTransformation::SeparableTransformation");
  Py_INCREF(funcx);
  Py_INCREF(funcy);

}


SeparableTransformation::~SeparableTransformation() {
  _VERBOSE("SeparableTransformation::~SeparableTransformation");
  Py_DECREF(_funcx);
  Py_DECREF(_funcy);
}

Py::Object
SeparableTransformation::get_funcx(const Py::Tuple & args) {
  _VERBOSE("SeparableTransformation::get_funcx");
  args.verify_length(0);
  return Py::Object(_funcx);
}

Py::Object
SeparableTransformation::get_funcy(const Py::Tuple & args) {
  _VERBOSE("SeparableTransformation::get_funcy");
  args.verify_length(0);
  return Py::Object(_funcy);
}


Py::Object
SeparableTransformation::set_funcx(const Py::Tuple & args) {
  _VERBOSE("SeparableTransformation::set_funcx");
  args.verify_length(1);
  if (!Func::check(args[0]))
    throw Py::TypeError("set_funcx(func) expected a func instance");
  _funcx = static_cast<Func*>(args[0].ptr());
  Py_INCREF(_funcx);
  return Py::Object();
}

Py::Object
SeparableTransformation::set_funcy(const Py::Tuple & args) {
  _VERBOSE("SeparableTransformation::set_funcy");
  args.verify_length(1);
  if (!Func::check(args[0]))
    throw Py::TypeError("set_funcy(func) expected a func instance");
  _funcy = static_cast<Func*>(args[0].ptr());
  Py_INCREF(_funcy);
  return Py::Object();
}




void
SeparableTransformation::nonlinear_only_api(double *x, double *y) {

  double thisx = _funcx->operator()(*x);
  double thisy = _funcy->operator()(*y);
  *x = thisx;
  *y = thisy;
}

std::pair<double, double>&
SeparableTransformation::operator()(const double& x, const double& y) {
  _VERBOSE("SeparableTransformation::operator");

  // calling function must first call eval_scalars

  xy.first  = _sx * _funcx->operator()(x) + _tx ;
  xy.second = _sy * _funcy->operator()(y) + _ty ;

  if (_usingOffset) {
    xy.first  += _xot;
    xy.second += _yot;
  }


  return xy;
}

void
SeparableTransformation::arrayOperator(const int length, const double x[], const double y[], double newx[], double newy[]) {
  _VERBOSE("SeparableTransformation::arrayOperator");

  _funcx->arrayOperator(length, x, newx);
  _funcy->arrayOperator(length, y, newy);

  // calling function must first call eval_scalars
  if (_usingOffset) {
	for(int i=0; i < length; i++)
	{
		newx[i] = (_sx * newx[i]) + _tx + _xot ;
		newy[i] = (_sy * newy[i]) + _ty + _yot ;
	}
  }
  else {
	for(int i=0; i < length; i++)
	{
		newx[i] = (_sx * newx[i]) + _tx ;
		newy[i] = (_sy * newy[i]) + _ty ;
	}
  }

}


std::pair<double, double> &
SeparableTransformation::inverse_api(const double &x, const double &y) {
  _VERBOSE("SeparableTransformation::inverse_api");

  // calling function must first call eval_scalars_inverse and
  // _transOffset->eval_scalars_inverse()


  if (!_invertible)
    throw Py::RuntimeError("Transformation is not invertible");

  double xin = x;
  double yin = y;

  if (_usingOffset) {
    xin  -= _xot;
    yin  -= _yot;
  }

  xy.first  = _funcx->inverse_api( _isx * xin  +  _itx );
  xy.second = _funcy->inverse_api( _isy * yin  +  _ity );

  return xy;
}



void
SeparableTransformation::eval_scalars(void) {
  _VERBOSE("SeparableTransformation::eval_scalars");
  double xminIn;
  double xmaxIn;
  double yminIn;
  double ymaxIn;

  xminIn  = _funcx->operator()( _b1->ll_api()->xval() );
  xmaxIn  = _funcx->operator()( _b1->ur_api()->xval() );
  yminIn  = _funcy->operator()( _b1->ll_api()->yval() );
  ymaxIn  = _funcy->operator()( _b1->ur_api()->yval() );

  double xminOut  = _b2->ll_api()->xval();
  double xmaxOut  = _b2->ur_api()->xval();
  double yminOut  = _b2->ll_api()->yval();
  double ymaxOut  = _b2->ur_api()->yval();

  double widthIn  = xmaxIn  - xminIn;
  double widthOut = xmaxOut - xminOut;

  double heightIn  = ymaxIn  - yminIn;
  double heightOut = ymaxOut - yminOut;
  //std::cout <<"heightout, heightin = "  << heightOut << " " << heightIn << std::endl;
  if (widthIn==0)
    throw Py::ZeroDivisionError("SeparableTransformation::eval_scalars xin interval is zero; cannot transform");

  if (heightIn==0)
    throw Py::ZeroDivisionError("SeparableTransformation::eval_scalars yin interval is zero; cannot transform");



  _sx = widthOut/widthIn;
  _sy = heightOut/heightIn;

  _tx = -xminIn*_sx + xminOut;
  _ty = -yminIn*_sy + yminOut;


  //now do the inverse mapping
  if ( (widthOut==0) || (widthOut==0) ) {
    _invertible = false;
  }
  else {
    _isx = widthIn/widthOut;
    _isy = heightIn/heightOut;

    _itx = -xminOut*_isx + xminIn;
    _ity = -yminOut*_isy + yminIn;
  }

  if (_usingOffset) {

      _transOffset->eval_scalars();
      _transOffset->operator()(_xo, _yo);
      _xot = _transOffset->xy.first;
      _yot = _transOffset->xy.second;
  }
}


Py::Object
SeparableTransformation::deepcopy(const Py::Tuple &args) {
  _VERBOSE("SeparableTransformation::deepcopy");
  args.verify_length(0);
  return Py::asObject( new SeparableTransformation(
        static_cast<Bbox*>((_b1->_deepcopy()).ptr()),
        static_cast<Bbox*>((_b2->_deepcopy()).ptr()), _funcx,_funcy ));
}

Py::Object
SeparableTransformation::shallowcopy(const Py::Tuple &args) {
  _VERBOSE("SeparableTransformation::shallowcopy");
  args.verify_length(0);
  return Py::asObject( new SeparableTransformation(_b1, _b2, _funcx,_funcy ));
}

NonseparableTransformation::NonseparableTransformation(Bbox *b1, Bbox *b2, FuncXY *funcxy) :
    BBoxTransformation(b1, b2),
    _funcxy(funcxy)  {
  _VERBOSE("NonseparableTransformation::NonseparableTransformation");
  Py_INCREF(funcxy);
}


NonseparableTransformation::~NonseparableTransformation() {
  _VERBOSE("NonseparableTransformation::~NonseparableTransformation");
  Py_DECREF(_funcxy);
}

Py::Object
NonseparableTransformation::get_funcxy(const Py::Tuple & args) {
  _VERBOSE("NonseparableTransformation::get_funcxy");
  args.verify_length(0);
  return Py::Object(_funcxy);
}


Py::Object
NonseparableTransformation::set_funcxy(const Py::Tuple & args) {
  _VERBOSE("NonseparableTransformation::set_funcx");
  args.verify_length(1);
  if (!FuncXY::check(args[0]))
    throw Py::TypeError("set_funcxy(func) expected a func instance");
  _funcxy = static_cast<FuncXY*>(args[0].ptr());
  Py_INCREF(_funcxy);
  return Py::Object();
}

void
NonseparableTransformation::nonlinear_only_api(double *x, double *y) {

  xy = _funcxy->operator()(*x,*y);
  *x = xy.first;
  *y = xy.second;
}


std::pair<double, double>&
NonseparableTransformation::operator()(const double& x, const double& y) {
  _VERBOSE("NonseparableTransformation::operator");

  // calling function must first call eval_scalars
  xy = _funcxy->operator()(x,y);

  //std::cout << "operator(x,y) In: " << x << " " << y << " " << xy.first << " " << xy.second << std::endl;
  xy.first  = _sx * xy.first  +  _tx ;
  xy.second = _sy * xy.second +  _ty;
  //std::cout << "operator(x,y) out: " << xy.first << " " << xy.second << std::endl;
  if (_usingOffset) {
    xy.first  += _xot;
    xy.second += _yot;
  }


  return xy;
}

void
NonseparableTransformation::arrayOperator(const int length, const double x[], const double y[], double newx[], double newy[]) {
  _VERBOSE("NonseparableTransformation::operator");

  _funcxy->arrayOperator(length, x, y, newx, newy);
  if (_usingOffset) {
	for(int i=0; i < length; i++)
	{
		newx[i] = (_sx * newx[i]) + _tx + _xot ;
		newy[i] = (_sy * newy[i]) + _ty + _yot ;
	}
  }
  else {
	for(int i=0; i < length; i++)
	{
		xy = _funcxy->operator()(x[i], y[i]);
		newx[i] = (_sx * newx[i]) + _tx ;
		newy[i] = (_sy * newy[i]) + _ty ;
	}
  }
  return;
}

std::pair<double, double> &
NonseparableTransformation::inverse_api(const double &x, const double &y) {
  _VERBOSE("NonseparableTransformation::inverse_api");

  // calling function must first call eval_scalars_inverse and
  // _transOffset->eval_scalars_inverse()


  if (!_invertible)
    throw Py::RuntimeError("Transformation is not invertible");

  double xin = x;
  double yin = y;

  if (_usingOffset) {
    xin  -= _xot;
    yin  -= _yot;
  }

  xy  = _funcxy->inverse_api( _isx * xin  +  _itx,
			      _isy * yin  +  _ity );

  return xy;
}


void
NonseparableTransformation::eval_scalars(void) {
  _VERBOSE("NonseparableTransformation::eval_scalars");
  std::pair<double, double> xyminIn = _funcxy->
    operator()( _b1->ll_api()->xval(), _b1->ll_api()->yval());

  std::pair<double, double> xymaxIn = _funcxy->
    operator()( _b1->ur_api()->xval(), _b1->ur_api()->yval());

  std::pair<double, double> xyminOut( _b2->ll_api()->xval(), _b2->ll_api()->yval() );

  std::pair<double, double> xymaxOut( _b2->ur_api()->xval(), _b2->ur_api()->yval() );



  double widthIn  = xymaxIn.first  - xyminIn.first;
  double widthOut = xymaxOut.first - xyminOut.first;

  double heightIn  = xymaxIn.second  - xyminIn.second;
  double heightOut = xymaxOut.second - xyminOut.second;

  if (widthIn==0)
    throw Py::ZeroDivisionError("NonseparableTransformation::eval_scalars xin interval is zero; cannot transform");

  if (heightIn==0)
    throw Py::ZeroDivisionError("NonseparableTransformation::eval_scalars yin interval is zero; cannot transform");



  _sx = widthOut/widthIn;
  _sy = heightOut/heightIn;

  _tx = -xyminIn.first*_sx + xyminOut.first;
  _ty = -xyminIn.second*_sy + xyminOut.second;

  /*
 std::cout <<"corners in "
	   << xyminIn.first << " " << xyminIn.second <<  " "
	   << xymaxIn.first << " " << xymaxIn.second <<  std::endl;
 std::cout <<"w,h in "  << widthIn << " " << heightIn <<  std::endl;
 std::cout <<"heightout, heightin = "  << heightOut << " " << heightIn << std::endl;
 std::cout <<"sx,sy,tx,ty = "  << _sx << " " << _sy <<  " " << _tx << " " << _ty << std::endl;
  */
  //now do the inverse mapping
  if ( (widthOut==0) || (widthOut==0) ) {
    _invertible = false;
  }
  else {
    _isx = widthIn/widthOut;
    _isy = heightIn/heightOut;

    _itx = -xyminOut.first*_isx + xyminIn.first;
    _ity = -xyminOut.second*_isy + xyminIn.second;
  }

  if (_usingOffset) {
    _transOffset->eval_scalars();
    _transOffset->operator()(_xo, _yo);
    _xot = _transOffset->xy.first;
    _yot = _transOffset->xy.second;
  }
}


Py::Object
NonseparableTransformation::deepcopy(const Py::Tuple &args) {
  _VERBOSE("NonseparableTransformation::deepcopy");
  args.verify_length(0);
  return Py::asObject( new NonseparableTransformation( static_cast<Bbox*>((_b1->_deepcopy()).ptr()),static_cast<Bbox*>((_b2->_deepcopy()).ptr()), _funcxy ));
}

Py::Object
NonseparableTransformation::shallowcopy(const Py::Tuple &args) {
  _VERBOSE("NonseparableTransformation::shallowcopy");
  args.verify_length(0);
  return Py::asObject( new NonseparableTransformation(_b1,_b2, _funcxy ));
}


Affine::Affine(LazyValue *a, LazyValue *b,  LazyValue *c,
	       LazyValue *d, LazyValue *tx, LazyValue *ty) :
  _a(a), _b(b), _c(c), _d(d), _tx(tx), _ty(ty) {
  _VERBOSE("Affine::Affine");
  Py_INCREF(a);
  Py_INCREF(b);
  Py_INCREF(c);
  Py_INCREF(d);
  Py_INCREF(tx);
  Py_INCREF(ty);

}

Affine::~Affine() {
  _VERBOSE("Affine::~Affine");
  Py_DECREF(_a);
  Py_DECREF(_b);
  Py_DECREF(_c);
  Py_DECREF(_d);
  Py_DECREF(_tx);
  Py_DECREF(_ty);
}


void
Affine::affine_params_api(double* a, double* b, double* c, double*d, double* tx, double* ty) {

  *a = _a->val();
  *b = _b->val();
  *c = _c->val();
  *d = _d->val();
  *tx = _tx->val();
  *ty = _ty->val();

}
Py::Object
Affine::as_vec6(const Py::Tuple &args) {
  _VERBOSE("Affine::as_vec6");
  //return the affine as length 6 list
  args.verify_length(0);
  Py::List ret(6);
  ret[0] = Py::Object(_a);
  ret[1] = Py::Object(_b);
  ret[2] = Py::Object(_c);
  ret[3] = Py::Object(_d);
  ret[4] = Py::Object(_tx);
  ret[5] = Py::Object(_ty);
  return ret;
}




std::pair<double, double> &
Affine::operator()(const double &x, const double &y) {
  _VERBOSE("Affine::operator");
  xy.first  = _aval*x + _cval*y + _txval;
  xy.second = _bval*x + _dval*y + _tyval;

  if (_usingOffset) {
    xy.first  += _xot;
    xy.second += _yot;
  }

  return xy;
}


std::pair<double, double> &
Affine::inverse_api(const double &x, const double &y) {
  _VERBOSE("Affine::inverse_api");

  if (!_invertible)
    throw Py::RuntimeError("Transformation is not invertible");

  double xin = x;
  double yin = y;

  if (_usingOffset) {
    xin  -= _xot;
    yin  -= _yot;
  }

  xin -= _txval;
  yin -= _tyval;

  xy.first  = _iaval*xin + _icval*yin;
  xy.second = _ibval*xin + _idval*yin;
  return xy;

}

void
Affine::eval_scalars(void) {
  _VERBOSE("Affine::eval_scalars");
  _aval  = _a->val();
  _bval  = _b->val();
  _cval  = _c->val();
  _dval  = _d->val();
  _txval = _tx->val();
  _tyval = _ty->val();


  double det = _aval*_dval - _bval*_cval;
  if (det==0) {
    _invertible = false;
  }
  else {
    double scale = 1.0/det;
    _iaval = scale*_dval;
    _ibval = -scale*_bval;
    _icval = -scale*_cval;
    _idval = scale*_aval;
  }

  if (_usingOffset) {
    _transOffset->eval_scalars();
    _transOffset->operator()(_xo, _yo);
    _xot = _transOffset->xy.first;
    _yot = _transOffset->xy.second;
  }

  _VERBOSE("Affine::eval_scalars DONE");
}

Py::Object
Affine::deepcopy(const Py::Tuple &args) {
  _VERBOSE("Affine::deepcopy");
  args.verify_length(0);
  try {
    eval_scalars();
  }
  catch(...) {
    throw Py::ValueError("Domain error on Affine deepcopy");
  }

  return Py::asObject( new Affine( new Value(_aval),new Value(_bval), new Value(_cval),
                                   new Value(_dval),new Value(_txval),new Value(_tyval) ));
}

Py::Object
Affine::shallowcopy(const Py::Tuple &args) {
  _VERBOSE("Affine::shallowcopy");
  args.verify_length(0);

  return Py::asObject( new Affine( _a, _b, _c, _d, _tx, _ty ));
}



/* ------------ module methods ------------- */
Py::Object
_transforms_module::new_value (const Py::Tuple &args)
{
  _VERBOSE("_transforms_module::new_value ");
  args.verify_length(1);
  double val = Py::Float(args[0]);
  return Py::asObject( new Value(val) );
}


Py::Object
_transforms_module::new_point (const Py::Tuple &args)
{
  _VERBOSE("_transforms_module::new_point ");
  args.verify_length(2);

  LazyValue *x, *y;

  if (BinOp::check(args[0]))
    x = static_cast<BinOp*>(args[0].ptr());
  else if (Value::check(args[0]))
    x = static_cast<Value*>(args[0].ptr());
  else
    throw Py::TypeError("Can only create points from LazyValues");

  if (BinOp::check(args[1]))
    y = static_cast<BinOp*>(args[1].ptr());
  else if (Value::check(args[1]))
    y = static_cast<Value*>(args[1].ptr());
  else
    throw Py::TypeError("Can only create points from LazyValues");

  return Py::asObject(new Point(x, y));

}


Py::Object
_transforms_module::new_interval (const Py::Tuple &args)
{
  _VERBOSE("_transforms_module::new_interval ");

  args.verify_length(2);

  if (!LazyValue::check(args[0]))
    throw Py::TypeError("Interval(val1, val2) expected a LazyValue for val1");
  if (!LazyValue::check(args[1]))
    throw Py::TypeError("Interval(val1, val2) expected a LazyValue for val2");


  LazyValue* v1 = static_cast<LazyValue*>(args[0].ptr());
  LazyValue* v2 = static_cast<LazyValue*>(args[1].ptr());
  return Py::asObject(new Interval(v1, v2) );
}

Py::Object
_transforms_module::new_bbox (const Py::Tuple &args)
{
  _VERBOSE("_transforms_module::new_bbox ");

  args.verify_length(2);

  if (!Point::check(args[0]))
    throw Py::TypeError("Point(p1,p2) expected a Point for p1");
  if (!Point::check(args[1]))
    throw Py::TypeError("Point(p1,p2) expected a Point for p2");

  Point* ll = static_cast<Point*>(args[0].ptr());
  Point* ur = static_cast<Point*>(args[1].ptr());
  return Py::asObject(new Bbox(ll, ur) );
}

Py::Object
_transforms_module::new_affine (const Py::Tuple &args) {
  _VERBOSE("_transforms_module::new_affine ");

  args.verify_length(6);

  LazyValue::check(args[0]);
  LazyValue::check(args[1]);
  LazyValue::check(args[2]);
  LazyValue::check(args[3]);
  LazyValue::check(args[4]);
  LazyValue::check(args[5]);

  LazyValue* a  = static_cast<LazyValue*>(args[0].ptr());
  LazyValue* b  = static_cast<LazyValue*>(args[1].ptr());
  LazyValue* c  = static_cast<LazyValue*>(args[2].ptr());
  LazyValue* d  = static_cast<LazyValue*>(args[3].ptr());
  LazyValue* tx = static_cast<LazyValue*>(args[4].ptr());
  LazyValue* ty = static_cast<LazyValue*>(args[5].ptr());
  return Py::asObject(new Affine(a, b, c, d, tx, ty));

}



Py::Object
_transforms_module::new_func (const Py::Tuple &args)
{
  _VERBOSE("_transforms_module::new_func ");
  args.verify_length(1);
  int typecode = Py::Int(args[0]);
  return Py::asObject(new Func(typecode));
}

Py::Object
_transforms_module::new_funcxy (const Py::Tuple &args)
{
  _VERBOSE("_transforms_module::new_funcxy ");
  args.verify_length(1);
  int typecode = Py::Int(args[0]);
  return Py::asObject(new FuncXY(typecode));
}

Py::Object
_transforms_module::new_func_userdef (const Py::Tuple &args)
{
  _VERBOSE("_transforms_module::new_func_userdef ");
  args.verify_length(2);
  Py::Object py_forward_fn = args[0];
  Py::Object py_inverse_fn = args[1];
  FuncUserDef *pfunc = new FuncUserDef();
  pfunc->set_function(py_forward_fn, py_inverse_fn);

  return Py::asObject(pfunc);
}


Py::Object
_transforms_module::new_separable_transformation (const Py::Tuple &args)
{
  _VERBOSE("_transforms_module::new_separable_transformation ");
  args.verify_length(4);
  if (!Bbox::check(args[0]))
    throw Py::TypeError("SeparableTransform(box1, box2, funcx, funcy) expected a Bbox for box1");
  if (!Bbox::check(args[1]))
    throw Py::TypeError("SeparableTransform(box1, box2, funcx, funcy) expected a Bbox for box2");
  if (!Func::check(args[2]))
    throw Py::TypeError("SeparableTransform(box1, box2, funcx, funcy) expected a Func for funcx");
  if (!Func::check(args[3]))
    throw Py::TypeError("SeparableTransform(box1, box2, funcx, funcy) expected a Func for funcy");


  Bbox* box1  = static_cast<Bbox*>(args[0].ptr());
  Bbox* box2  = static_cast<Bbox*>(args[1].ptr());
  Func* funcx  = static_cast<Func*>(args[2].ptr());
  Func* funcy  = static_cast<Func*>(args[3].ptr());

  return Py::asObject( new SeparableTransformation(box1, box2, funcx, funcy) );
}

Py::Object
_transforms_module::new_nonseparable_transformation (const Py::Tuple &args)
{
  _VERBOSE("_transforms_module::new_nonseparable_transformation ");
  args.verify_length(3);
  if (!Bbox::check(args[0]))
    throw Py::TypeError("NonseparableTransform(box1, box2, funcxy) expected a Bbox for box1");
  if (!Bbox::check(args[1]))
    throw Py::TypeError("NonseparableTransform(box1, box2, funcxy) expected a Bbox for box2");
  if (!FuncXY::check(args[2]))
    throw Py::TypeError("NonseparableTransform(box1, box2, funcxy, funcy) expected a FuncXY for funcxy");


  Bbox* box1  = static_cast<Bbox*>(args[0].ptr());
  Bbox* box2  = static_cast<Bbox*>(args[1].ptr());
  FuncXY* funcxy  = static_cast<FuncXY*>(args[2].ptr());

  return Py::asObject( new NonseparableTransformation(box1, box2, funcxy) );
}

void
LazyValue::init_type()
{
  _VERBOSE("LazyValue::init_type");
  behaviors().name("LazyValue");
  behaviors().doc("A lazy evaluation float, with arithmetic");
  behaviors().supportNumberType();
  behaviors().supportCompare();
  add_varargs_method("get",    &LazyValue::get,     "get()\n");
  add_varargs_method("set",    &LazyValue::set,     "set(val)\n");
}

void
Value::init_type()
{
  _VERBOSE("Value::init_type");
  behaviors().name("Value");
  behaviors().doc("A mutable float");
  behaviors().supportNumberType();}


void
BinOp::init_type()
{
  _VERBOSE("BinOp::init_type");
  behaviors().name("BinOp");
  behaviors().doc("A binary operation on lazy values");
  behaviors().supportNumberType();
}

void
Point::init_type()
{
  _VERBOSE("Point::init_type");
  behaviors().name("Point");
  behaviors().doc("A point x, y");

  add_varargs_method("x",    &Point::x,     "x()\n");
  add_varargs_method("y",    &Point::y,     "y()\n");
  add_varargs_method("reference_count", &Point::reference_count);
}

void
Interval::init_type()
{
  _VERBOSE("Interval::init_type");
  behaviors().name("Interval");
  behaviors().doc("A 1D interval");

  add_varargs_method("contains", &Interval::contains, "contains(x)\n");
  add_varargs_method("update", &Interval::update, "update(vals)\n");
  add_varargs_method("contains_open", &Interval::contains_open, "contains_open(x)\n");
  add_varargs_method("get_bounds", &Interval::get_bounds, "get_bounds()\n");
  add_varargs_method("set_bounds", &Interval::set_bounds, "set_bounds()\n");
  add_varargs_method("shift", &Interval::shift, "shift()\n");
  add_varargs_method("span", &Interval::span, "span()\n");
  add_varargs_method("val1", &Interval::val1, "val1()\n");
  add_varargs_method("val2", &Interval::val2, "val2()\n");
  add_varargs_method("minpos", &Interval::minpos, "minpos()\n");

}

void
Bbox::init_type()
{
  _VERBOSE("Bbox::init_type");
  behaviors().name("Bbox");
  behaviors().doc("A 2D bounding box");
  //behaviors().supportRepr();
  //behaviors().supportGetattr();
  //behaviors().supportStr();

  add_varargs_method("ll", 	&Bbox::ll, "ll()\n");
  add_varargs_method("ur", 	&Bbox::ur, "ur()\n");
  add_varargs_method("contains" , &Bbox::contains, "contains(x,y)\n");
  add_varargs_method("count_contains", &Bbox::count_contains, "count_contains(xys)\n");
  add_varargs_method("overlaps" , &Bbox::overlaps, "overlaps(bbox)\n");
  add_varargs_method("overlapsx" , &Bbox::overlapsx, "overlapsx(bbox)\n");
  add_varargs_method("overlapsy" , &Bbox::overlapsy, "overlapsy(bbox)\n");
  add_varargs_method("intervalx" , &Bbox::intervalx, "intervalx()\n");
  add_varargs_method("intervaly" , &Bbox::intervaly, "intervaly()\n");

  add_varargs_method("get_bounds", &Bbox::get_bounds, "get_bounds()\n");
  add_varargs_method("update" , &Bbox::update, "update(xys, ignore)\n");
  add_varargs_method("update_numerix" , &Bbox::update_numerix, "update_numerix(x, u, ignore)\n");
  add_varargs_method("update_numerix_xy" , &Bbox::update_numerix_xy, "update_numerix_xy(xy, ignore)\n");
  add_varargs_method("width", 	&Bbox::width, "width()\n");
  add_varargs_method("height", 	&Bbox::height, "height()\n");
  add_varargs_method("xmax", 	&Bbox::xmax, "xmax()\n");
  add_varargs_method("ymax", 	&Bbox::ymax, "ymax()\n");
  add_varargs_method("xmin", 	&Bbox::xmin, "xmin()\n");
  add_varargs_method("ymin", 	&Bbox::ymin, "ymin()\n");

  add_varargs_method("ignore", 	 &Bbox::ignore, "ignore(int)");
  add_varargs_method("scale", 	 &Bbox::scale, "scale(sx,sy)");
  add_varargs_method("deepcopy", &Bbox::deepcopy, "deepcopy()\n");
}



void
Func::init_type()
{
  _VERBOSE("Func::init_type");
  behaviors().name("Func");
  behaviors().doc("Map double -> double");
  behaviors().supportRepr();
  behaviors().supportGetattr();
  add_varargs_method("map",     &Func::map, "map(x)\n");
  add_varargs_method("inverse", &Func::inverse, "inverse(y)\n");
  add_varargs_method("set_type", &Func::set_type, "set_type(TYPE)\n");
  add_varargs_method("get_type", &Func::get_type, "get_type()\n");
}

void
FuncUserDef::init_type()
{
  _VERBOSE("FuncUserDef::init_type");
  behaviors().name("FuncUserDef");
  behaviors().doc("Map double -> double");
  behaviors().supportRepr();
  behaviors().supportGetattr();
  add_varargs_method("map",     &FuncUserDef::map, "map(x)\n");
  add_varargs_method("inverse", &FuncUserDef::inverse, "inverse(y)\n");
  add_varargs_method("set_type", &FuncUserDef::set_type, "set_type(TYPE)\n");
  add_varargs_method("get_type", &FuncUserDef::get_type, "get_type()\n");
} // FuncUserDef::init_type()

void
FuncXY::init_type()
{
  _VERBOSE("FuncXY::init_type");
  behaviors().name("FuncXY");
  behaviors().doc("Map double,double -> funcx(double), funcy(double)");
  add_varargs_method("map", &FuncXY::map, "map(x,y)\n");
  add_varargs_method("inverse", &FuncXY::inverse, "inverse(x,y)\n");
  add_varargs_method("set_type", &FuncXY::set_type, "set_type(TYPE)\n");
  add_varargs_method("get_type", &FuncXY::get_type, "get_type()\n");
}

void
Transformation::init_type()
{
  _VERBOSE("Transformation::init_type");
  behaviors().name("Transformation");
  behaviors().doc("Transformation base class");


  add_varargs_method("freeze",   &Transformation::freeze,  "freeze(); eval and freeze the lazy objects\n");
  add_varargs_method("thaw",   &Transformation::thaw,  "thaw(); release the laszy objects\n");

  add_varargs_method("get_bbox1",   &Transformation::get_bbox1,  "get_bbox1(); return the input bbox\n");
  add_varargs_method("get_bbox2",   &Transformation::get_bbox2,  "get_bbox2(); return the output bbox\n");

  add_varargs_method("set_bbox1",   &Transformation::set_bbox1,  "set_bbox1(); set the input bbox\n");
  add_varargs_method("set_bbox2",   &Transformation::set_bbox2,  "set_bbox2(); set the output bbox\n");

  add_varargs_method("get_funcx",   &Transformation::get_funcx,  "get_funcx(); return the Func instance on x\n");
  add_varargs_method("get_funcy",   &Transformation::get_funcy,  "get_funcy(); return the Func instance on y\n");

  add_varargs_method("set_funcx",   &Transformation::set_funcx,  "set_funcx(); set the Func instance on x\n");
  add_varargs_method("set_funcy",   &Transformation::set_funcy,  "set_funcy(); set the Func instance on y\n");


  add_varargs_method("get_funcxy",   &Transformation::get_funcxy,  "get_funcxy(); return the FuncXY instance\n");
  add_varargs_method("set_funcxy",   &Transformation::set_funcxy,  "set_funcxy(); set the FuncXY instance\n");

  add_varargs_method("xy_tup",   &Transformation::xy_tup,  "xy_tup(xy)\n");
  add_varargs_method("seq_x_y",  &Transformation::seq_x_y, "seq_x_y(x, y)\n");
  add_varargs_method("numerix_x_y",  &Transformation::numerix_x_y, "numerix_x_y(x, y)\n");
  add_keyword_method("nonlinear_only_numerix",  &Transformation::nonlinear_only_numerix, "nonlinear_only_numerix\n");
  add_varargs_method("need_nonlinear",  &Transformation::need_nonlinear, "need_nonlinear\n");
  add_varargs_method("seq_xy_tups", &Transformation::seq_xy_tups, "seq_xy_tups(seq)\n");
  add_varargs_method("numerix_xy", &Transformation::numerix_xy, "numerix_xy(XY)\n");
  add_varargs_method("inverse_numerix_xy", &Transformation::inverse_numerix_xy, "inverse_numerix_xy(XY)\n");
  add_varargs_method("inverse_xy_tup",   &Transformation::inverse_xy_tup,  "inverse_xy_tup(xy)\n");

  add_varargs_method("set_offset",   &Transformation::set_offset,  "set_offset(xy, trans)\n");

  add_varargs_method("as_vec6", &Transformation::as_vec6, "as_vec6(): return the affine as length 6 list of Values\n");
  add_varargs_method("as_vec6_val", &Transformation::as_vec6_val, "as_vec6_val(): return the affine as length 6 list of float\n");
  add_varargs_method("deepcopy", &Transformation::deepcopy, "deepcopy()\n");
  add_varargs_method("shallowcopy", &Transformation::shallowcopy, "shallowcopy()\n");

}

void
Affine::init_type()
{
  _VERBOSE("Affine::init_type");
  behaviors().name("Affine");
  behaviors().doc("A mutable float");
}


void
SeparableTransformation::init_type()
{
  _VERBOSE("SeparableTransformation::init_type");
  behaviors().name("SeparableTransformation");
  behaviors().doc("SeparableTransformation(box1, box2, funcx, funcy); x and y transformations are independet");

}

void
NonseparableTransformation::init_type()
{
  _VERBOSE("NonseparableTransformation::init_type");
  behaviors().name("NonseparableTransformation");
  behaviors().doc("NonseparableTransformation(box1, box2, funcxy); x and y transformations are not independent");

}





extern "C"
DL_EXPORT(void)
#ifdef NUMARRAY
  init_na_transforms(void)
#else
#   ifdef NUMERIC
  init_nc_transforms(void)
#   else
  init_ns_transforms(void)
#   endif
#endif
{
  static _transforms_module* _transforms = new _transforms_module;

#ifdef NUMARRAY
  _VERBOSE("init_na_transforms");
#else
#   ifdef NUMERIC
  _VERBOSE("init_nc_transforms");
#   else
  _VERBOSE("init_ns_transforms");
#   endif
#endif

  import_array();

  Py::Dict d = _transforms->moduleDictionary();
  d["LOG10"] = Py::Int((int)Func::LOG10);
  d["IDENTITY"] = Py::Int((int)Func::IDENTITY);
  d["POLAR"] = Py::Int((int)FuncXY::POLAR);;
};

