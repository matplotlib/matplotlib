#include <functional>
#include <numeric>
#include "_transforms.h"
 
#ifdef NUMARRAY
#include "numarray/arrayobject.h" 
#else
#include "Numeric/arrayobject.h" 
#endif   

#define DEBUG_MEM 0


Value::~Value() {
  //std::cout << "bye bye Value" << std::endl;
}

Py::Object
Value::set(const Py::Tuple & args) {
  args.verify_length(1);

  _val = Py::Float( args[0] ); 
  return Py::Object();
}

Py::Object
Value::get(const Py::Tuple & args) {
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
  
  
  if (!LazyValue::check(o)) 
    throw Py::TypeError("Can only add LazyValues with other LazyValues");
  
  LazyValue* rhs = static_cast<LazyValue*>(o.ptr());
 
  return Py::asObject(new BinOp(this, rhs, BinOp::ADD));
} 

Py::Object 
LazyValue::number_divide( const Py::Object &o ) {
  
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
  
  
  if (!LazyValue::check(o)) 
    throw Py::TypeError("Can only multiply LazyValues with other LazyValues");
  
  LazyValue* rhs = static_cast<LazyValue*>(o.ptr());
  return Py::asObject(new BinOp(this, rhs, BinOp::MULTIPLY));
} 

Py::Object 
LazyValue::number_subtract( const Py::Object &o ) {
  
  
  if (!LazyValue::check(o)) 
    throw Py::TypeError("Can only subtract LazyValues with other LazyValues");
  
  LazyValue* rhs = static_cast<LazyValue*>(o.ptr());
  return Py::asObject(new BinOp(this, rhs, BinOp::SUBTRACT));
} 

BinOp::BinOp(LazyValue* lhs, LazyValue* rhs, int opcode) : 
  _lhs(lhs), _rhs(rhs), _opcode(opcode) {
  Py_INCREF(lhs);
  Py_INCREF(rhs);
}

BinOp::~BinOp() {
  Py_INCREF(_lhs);
  Py_INCREF(_rhs);
  if (DEBUG_MEM) std::cout << "bye bye BinOp" << std::endl;
}

Py::Object
BinOp::get(const Py::Tuple & args) {
  args.verify_length(0);
  double x = val();
  return Py::Float( x ); 
}

Point::Point(LazyValue* x, LazyValue*  y) : _x(x), _y(y) { 
  Py_INCREF(x);
  Py_INCREF(y);
}

Point::~Point()
{
  Py_DECREF(_x);
  Py_DECREF(_y);

  if (DEBUG_MEM) std::cout << "bye bye Point" << std::endl;
}

Interval::Interval(LazyValue* val1, LazyValue* val2) : 
  _val1(val1), _val2(val2) {
  Py_INCREF(val1);
  Py_INCREF(val2);
};

Interval::~Interval() {
  Py_DECREF(_val1);
  Py_DECREF(_val2);

  if (DEBUG_MEM) std::cout << "bye bye Interval" << std::endl;
}

Py::Object 
Interval::update(const Py::Tuple &args) {
  args.verify_length(2);

  Py::SeqBase<Py::Object> vals = args[0];

  //don't use current bounds when updating box if ignore==1
  int ignore = Py::Int(args[1]);  
  size_t Nval = vals.length();
  if (Nval==0) return Py::Object();

  double minx = _val1->val();
  double maxx = _val2->val();

  if (ignore) {
    minx = std::numeric_limits<double>::max();
    maxx = std::numeric_limits<double>::min();
  }

  double thisval;
  for (size_t i=0; i<Nval; ++i) {
    thisval = Py::Float(vals[i]);

    if (thisval<minx) minx=thisval;
    if (thisval>maxx) maxx=thisval;
  } 


  _val1->set_api(minx);
  _val2->set_api(maxx);
  return Py::Object();
}

Bbox::Bbox(Point* ll, Point* ur) : _ll(ll), _ur(ur) {
  Py_INCREF(ll);
  Py_INCREF(ur);
};
  

Bbox::~Bbox() {
  Py_DECREF(_ll);
  Py_DECREF(_ur);
  if (DEBUG_MEM) std::cout << "bye bye Bbox" << std::endl;
}

Py::Object 
Bbox::deepcopy(const Py::Tuple &args) {
  args.verify_length(0);
  
  double minx = _ll->xval();
  double miny = _ll->yval();
  
  double maxx = _ur->xval();
  double maxy = _ur->yval();
  
  return Py::asObject( new Bbox( new Point(new Value(minx), new Value(miny) ),
				 new Point(new Value(maxx), new Value(maxy) )));
}

Py::Object 
Bbox::scale(const Py::Tuple &args) {
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
Bbox::contains(const Py::Tuple &args) {
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
  args.verify_length(1);

  if (! check(args[0]))
    throw Py::TypeError("Expected a bbox");

  int x = Py::Int( overlapsx(args) );
  int y = Py::Int( overlapsy(args) );
  return Py::Int(x&&y);
}

Py::Object 
Bbox::overlapsx(const Py::Tuple &args) {
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
 

Py::Object 
Bbox::update(const Py::Tuple &args) {
  args.verify_length(2);

  Py::SeqBase<Py::Object> xys = args[0];

  //don't use current bounds when updating box if ignore==1
  int ignore = Py::Int(args[1]);  
  size_t Nx = xys.length();
  if (Nx==0) return Py::Object();

  double minx = _ll->xval();
  double maxx = _ur->xval();
  double miny = _ll->yval();
  double maxy = _ur->yval();

  if (ignore) {
    minx = std::numeric_limits<double>::max();
    maxx = std::numeric_limits<double>::min();
    miny = std::numeric_limits<double>::max();
    maxy = std::numeric_limits<double>::min();
  }

  Py::Tuple tup;
  for (size_t i=0; i<Nx; ++i) {
    tup = xys[i];
    double x = Py::Float(tup[0]);
    double y = Py::Float(tup[1]);

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

Func::~Func() {
  if (DEBUG_MEM) std::cout << "bye bye Func" << std::endl;
}


Py::Object 
Func::map(const Py::Tuple &args) {
  
  args.verify_length(1);
  double xin = Py::Float(args[0]);
  
  double xout = this->operator()(xin);
  return Py::Float(xout);

};

Py::Object 
Func::inverse(const Py::Tuple &args) {
  
  args.verify_length(1);
  double xin = Py::Float(args[0]);
  
  double xout = this->inverse_api(xin);
  return Py::Float(xout);
};

FuncXY::FuncXY(Func* funcx, Func* funcy) : 
  _funcx(funcx), _funcy(funcy) {
    Py_INCREF(funcx);
    Py_INCREF(funcy);
}

FuncXY::~FuncXY() {
  Py_DECREF(_funcx);
  Py_DECREF(_funcy);

  if (DEBUG_MEM) std::cout << "bye bye FuncXY" << std::endl;
}

Py::Object 
FuncXY::map(const Py::Tuple &args) {
  
  args.verify_length(2);
  double xin = Py::Float(args[0]);
  double yin = Py::Float(args[1]);
  
  std::pair<double, double> xy = this->operator()(xin, yin);

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



Py::Object
FuncXY::set_funcx(const Py::Tuple & args) {
  args.verify_length(1);
  
  if (!Func::check(args[0])) 
    throw Py::TypeError("set_funcx(func) expected a Func instance");
  
  _funcx = static_cast<Func*>(args[0].ptr());
  Py_INCREF(_funcx);
  return Py::Object();
}

Py::Object
FuncXY::set_funcy(const Py::Tuple & args) {
  args.verify_length(1);
  
  if (!Func::check(args[0])) 
    throw Py::TypeError("set_funcy(func) expected a Func instance");
  
  _funcy = static_cast<Func*>(args[0].ptr());
  Py_INCREF(_funcy);
  return Py::Object();
}

Py::Object
FuncXY::get_funcx(const Py::Tuple & args) {
  args.verify_length(0);
  return Py::Object(_funcx);
}

Py::Object
FuncXY::get_funcy(const Py::Tuple & args) {
  args.verify_length(0);
  return Py::Object(_funcy);
}

PolarXY::~PolarXY() {
  if (DEBUG_MEM) std::cout << "bye bye PolarXY" << std::endl;
}

Transformation::~Transformation() {
  if (DEBUG_MEM) std::cout << "bye bye Transformation" << std::endl;
}

Py::Object
Transformation::as_vec6(const Py::Tuple & args) {
  throw Py::RuntimeError("This transformation does not support as_vec6");
  return Py::Object();
}


Py::Object
Transformation::get_funcx(const Py::Tuple & args) {
  throw Py::RuntimeError("This transformation does not support get_funcx");
  return Py::Object();
}

Py::Object
Transformation::get_funcy(const Py::Tuple & args) {
  throw Py::RuntimeError("This transformation does not support get_funcy");
  return Py::Object();
}


Py::Object
Transformation::set_funcx(const Py::Tuple & args) {
  throw Py::RuntimeError("This transformation does not support set_funcx");
  return Py::Object();
}

Py::Object
Transformation::set_funcy(const Py::Tuple & args) {
  throw Py::RuntimeError("This transformation does not support set_funcy");
  return Py::Object();
}



Py::Object
Transformation::get_bbox1(const Py::Tuple & args) {
  throw Py::RuntimeError("This transformation does not support get_bbox1"); 
  return Py::Object();
}

Py::Object
Transformation::get_bbox2(const Py::Tuple & args) {
  throw Py::RuntimeError("This transformation does not support get_bbox2"); 
  return Py::Object();
}


Py::Object
Transformation::set_bbox1(const Py::Tuple & args) {
  throw Py::RuntimeError("This transformation does not support set_bbox1"); 
  return Py::Object();
}

Py::Object
Transformation::set_bbox2(const Py::Tuple & args) {
  throw Py::RuntimeError("This transformation does not support set_bbox1"); 
  return Py::Object();
}

Py::Object
Transformation::set_offset(const Py::Tuple & args) {
  args.verify_length(2);

  Py::SeqBase<Py::Object> xy = args[0];

  if (!check(args[1]))
    throw Py::TypeError("Transformation::set_offset(xy,trans) requires trans to be a Transformation instance");
  
  _usingOffset = 1;
  _xo = Py::Float(xy[0]);
  _yo = Py::Float(xy[1]);
  _transOffset = static_cast<Transformation*>(args[1].ptr());
  Py_INCREF(_transOffset);
  return Py::Object();
}



Py::Object
Transformation::inverse_xy_tup(const Py::Tuple & args) {
  args.verify_length(1);

  Py::Tuple tup = args[0];
  double xin = Py::Float(tup[0]);
  double yin = Py::Float(tup[1]);

  if (!_frozen) eval_scalars();
  
  inverse_api(xin, yin);
  Py::Tuple ret(2);
  ret[0] = Py::Float(xy.first);
  ret[1] = Py::Float(xy.second);
  return ret;
  
}

Py::Object
Transformation::xy_tup(const Py::Tuple & args) {
  args.verify_length(1);

  if (!_frozen) eval_scalars();

  Py::SeqBase<Py::Object> xytup = args[0];
  double x = Py::Float(xytup[0]);
  double y = Py::Float(xytup[1]);
  

  Py::Tuple out(2);
  this->operator()(x, y);
  out[0] = Py::Float( xy.first ); 
  out[1] = Py::Float( xy.second ); 
  return out;
}

Py::Object
Transformation::seq_x_y(const Py::Tuple & args) {
  args.verify_length(2);
  
  Py::SeqBase<Py::Object> x = args[0];
  Py::SeqBase<Py::Object> y = args[1];
  
  size_t Nx = x.length();
  size_t Ny = y.length();
  
  if (Nx!=Ny) 
    throw Py::ValueError("x and y must be equal length sequences");

  // evaluate the lazy objects  
  if (!_frozen) eval_scalars();

  Py::Tuple xo(Nx);
  Py::Tuple yo(Nx);
  
 
  for (size_t i=0; i< Nx; ++i) {
    double thisx = Py::Float(x[i]);
    double thisy = Py::Float(y[i]);
    this->operator()(thisx, thisy);
    xo[i] = Py::Float( xy.first );
    yo[i] = Py::Float( xy.second );
  }
  
  Py::Tuple ret(2);
  ret[0] = xo;
  ret[1] = yo;
  return ret;
}

Py::Object
Transformation::numerix_x_y(const Py::Tuple & args) {
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
  if (!_frozen) eval_scalars();

  int dimensions[1];
  dimensions[0] = Nx;
  PyArrayObject *retx = (PyArrayObject *)PyArray_FromDims(1,dimensions,PyArray_DOUBLE);
  if (retx==NULL)
    throw Py::RuntimeError("Could not create return x array");

  PyArrayObject *rety = (PyArrayObject *)PyArray_FromDims(1,dimensions,PyArray_DOUBLE);
  if (retx==NULL)
    throw Py::RuntimeError("Could not create return x array");

  for (size_t i=0; i< Nx; ++i) {

    double thisx = *(double *)(x->data + i*x->strides[0]);
    double thisy = *(double *)(y->data + i*y->strides[0]);
    //std::cout << "calling operator " << thisx << " " << thisy << " " << std::endl;
    this->operator()(thisx, thisy);
    *(double *)(retx->data + i*retx->strides[0]) = xy.first;
    *(double *)(rety->data + i*rety->strides[0]) = xy.second;
  }

  Py::Tuple ret(2);
  ret[0] = Py::Object((PyObject*)retx);
  ret[1] = Py::Object((PyObject*)rety);
  return ret;
}

Py::Object
Transformation::seq_xy_tups(const Py::Tuple & args) {
  args.verify_length(1);
  
  Py::SeqBase<Py::Object> xytups = args[0];
  
  size_t Nx = xytups.length();

  if (!_frozen) eval_scalars();

  Py::Tuple ret(Nx);
  Py::SeqBase<Py::Object> xytup;
  
  
  
  for (size_t i=0; i< Nx; ++i) {
    xytup = Py::SeqBase<Py::Object>( xytups[i] );

    double thisx = Py::Float(xytup[0]);
    double thisy = Py::Float(xytup[1]);

    this->operator()(thisx, thisy);

    Py::Tuple out(2);
    out[0] = Py::Float( xy.first );
    out[1] = Py::Float( xy.second );
    ret[i] = out;
  }
  
  return ret;
}


SeparableTransformation::SeparableTransformation(Bbox *b1, Bbox *b2, Func *funcx, Func *funcy) : 
    Transformation(), 
    _b1(b1), _b2(b2), _funcx(funcx), _funcy(funcy)  {
  Py_INCREF(b1);
  Py_INCREF(b2);
  Py_INCREF(funcx);
  Py_INCREF(funcy);
  
}


SeparableTransformation::~SeparableTransformation() {
  Py_DECREF(_b1);
  Py_DECREF(_b2);
  Py_DECREF(_funcx);
  Py_DECREF(_funcy);
  if (DEBUG_MEM) std::cout << "bye bye SeparableTransformation" << std::endl;
}

Py::Object
SeparableTransformation::get_funcx(const Py::Tuple & args) {
  args.verify_length(0);
  return Py::Object(_funcx);
}

Py::Object
SeparableTransformation::get_funcy(const Py::Tuple & args) {
  args.verify_length(0);
  return Py::Object(_funcy);
}


Py::Object
SeparableTransformation::set_funcx(const Py::Tuple & args) {
  args.verify_length(1);
  if (!Func::check(args[0])) 
    throw Py::TypeError("set_funcx(func) expected a func instance");
  _funcx = static_cast<Func*>(args[0].ptr());
  Py_INCREF(_funcx);
  return Py::Object();
}

Py::Object
SeparableTransformation::set_funcy(const Py::Tuple & args) {
  args.verify_length(1);
  if (!Func::check(args[0])) 
    throw Py::TypeError("set_funcy(func) expected a func instance");
  _funcy = static_cast<Func*>(args[0].ptr());
  Py_INCREF(_funcy);
  return Py::Object();
}



Py::Object
SeparableTransformation::get_bbox1(const Py::Tuple & args) {
  args.verify_length(0);
  return Py::Object(_b1);
}

Py::Object
SeparableTransformation::get_bbox2(const Py::Tuple & args) {
  args.verify_length(0);
  return Py::Object(_b2);
}


Py::Object
SeparableTransformation::set_bbox1(const Py::Tuple & args) {
  args.verify_length(1);
  if (!Bbox::check(args[0])) 
    throw Py::TypeError("set_bbox1(func) expected a func instance");
  _b1 = static_cast<Bbox*>(args[0].ptr());
  Py_INCREF(_b1);
  return Py::Object();
}

Py::Object
SeparableTransformation::set_bbox2(const Py::Tuple & args) {
  args.verify_length(1);
  if (!Bbox::check(args[0])) 
    throw Py::TypeError("set_bbox2(func) expected a func instance");
  _b2 = static_cast<Bbox*>(args[0].ptr());
  Py_INCREF(_b2);
  return Py::Object();
}


std::pair<double, double>&
SeparableTransformation::operator()(const double& x, const double& y) {

  // calling function must first call eval_scalars
  double fx = _funcx->operator()(x);
  double fy = _funcy->operator()(y);
  

  xy.first  = _sx * fx  +  _tx ;
  xy.second = _sy * fy  +  _ty;

  if (_usingOffset) {
    xy.first  += _xot;
    xy.second += _yot;
  }
  

  return xy;
}


std::pair<double, double> &
SeparableTransformation::inverse_api(const double &x, const double &y) {

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
  double xminIn  = _funcx->operator()( _b1->ll_api()->xval() );
  double xmaxIn  = _funcx->operator()( _b1->ur_api()->xval() );
  double yminIn  = _funcy->operator()( _b1->ll_api()->yval() );
  double ymaxIn  = _funcy->operator()( _b1->ur_api()->yval() );

  double xminOut  = _b2->ll_api()->xval();
  double xmaxOut  = _b2->ur_api()->xval();
  double yminOut  = _b2->ll_api()->yval();
  double ymaxOut  = _b2->ur_api()->yval();

  double widthIn  = xmaxIn  - xminIn;
  double widthOut = xmaxOut - xminOut;

  double heightIn  = ymaxIn  - yminIn;
  double heightOut = ymaxOut - yminOut;

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

Affine::Affine(LazyValue *a, LazyValue *b,  LazyValue *c, 
	       LazyValue *d, LazyValue *tx, LazyValue *ty) : 
  _a(a), _b(b), _c(c), _d(d), _tx(tx), _ty(ty) {
  Py_INCREF(a);
  Py_INCREF(b);
  Py_INCREF(c);
  Py_INCREF(d);
  Py_INCREF(tx);
  Py_INCREF(ty);

}

Affine::~Affine() {
  Py_DECREF(_a);
  Py_DECREF(_b);
  Py_DECREF(_c);
  Py_DECREF(_d);
  Py_DECREF(_tx);
  Py_DECREF(_ty);
  if (DEBUG_MEM) std::cout << "bye bye Affine" << std::endl;
}


Py::Object 
Affine::as_vec6(const Py::Tuple &args) {
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
    _ibval = scale*_cval;
    _icval = -scale*_bval;
    _idval = scale*_aval;
  }
  
  if (_usingOffset) {
    _transOffset->eval_scalars();
    _transOffset->operator()(_xo, _yo);
    _xot = _transOffset->xy.first;
    _yot = _transOffset->xy.second;
  }
}

 

/* ------------ module methods ------------- */
Py::Object _transforms_module::new_value (const Py::Tuple &args)
{
  args.verify_length(1);
  double val = Py::Float(args[0]);
  return Py::asObject( new Value(val) );
}   


Py::Object _transforms_module::new_point (const Py::Tuple &args)
{
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


Py::Object _transforms_module::new_interval (const Py::Tuple &args)
{
  
  args.verify_length(2);
  
  if (!LazyValue::check(args[0])) 
    throw Py::TypeError("Interval(val1, val2) expected a LazyValue for val1");
  if (!LazyValue::check(args[1])) 
    throw Py::TypeError("Interval(val1, val2) expected a LazyValue for val2");
  
  
  LazyValue* v1 = static_cast<LazyValue*>(args[0].ptr());
  LazyValue* v2 = static_cast<LazyValue*>(args[1].ptr());
  return Py::asObject(new Interval(v1, v2) );  
}

Py::Object _transforms_module::new_bbox (const Py::Tuple &args)
{
  
  args.verify_length(2);
  
  if (!Point::check(args[0])) 
    throw Py::TypeError("Point(p1,p2) expected a Point for p1");
  if (!Point::check(args[1])) 
    throw Py::TypeError("Point(p1,p2) expected a Point for p2");
  
  Point* ll = static_cast<Point*>(args[0].ptr());
  Point* ur = static_cast<Point*>(args[1].ptr());
  return Py::asObject(new Bbox(ll, ur) );  
}

Py::Object _transforms_module::new_affine (const Py::Tuple &args) {

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

 

Py::Object _transforms_module::new_func (const Py::Tuple &args)
{
  args.verify_length(1);
  int typecode = Py::Int(args[0]);
  return Py::asObject(new Func(typecode));
}   

Py::Object _transforms_module::new_funcxy (const Py::Tuple &args)
{
  args.verify_length(2);
  if (!Func::check(args[0]))
    throw Py::TypeError("FuncXY(funcx, funcy) expected a Func instance for funcx)");
  if (!Func::check(args[1]))
    throw Py::TypeError("FuncXY(funcx, funcy) expected a Func instance for funcy)");
  Func* funcx  = static_cast<Func*>(args[0].ptr());
  Func* funcy  = static_cast<Func*>(args[1].ptr());

  return Py::asObject(new FuncXY(funcx, funcy));
}   


Py::Object _transforms_module::new_polarxy (const Py::Tuple &args)
{
  args.verify_length(0);
  return Py::asObject( new PolarXY() );
}   

Py::Object _transforms_module::new_separable_transformation (const Py::Tuple &args)
{
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

void LazyValue::init_type()
{
  behaviors().name("LazyValue");
  behaviors().doc("A lazy evaluation float, with arithmetic");
  behaviors().supportNumberType();
  behaviors().supportCompare();
  add_varargs_method("get",    &LazyValue::get,     "get()\n");
  add_varargs_method("set",    &LazyValue::set,     "set(val)\n");
}

void Value::init_type()
{
  behaviors().name("Value");
  behaviors().doc("A mutable float");
  behaviors().supportNumberType();}


void BinOp::init_type()
{
  behaviors().name("BinOp");
  behaviors().doc("A binary operation on lazy values");
  behaviors().supportNumberType();
} 

void Point::init_type()
{
  behaviors().name("Point");
  behaviors().doc("A point x, y");
    
  add_varargs_method("x",    &Point::x,     "x()\n");
  add_varargs_method("y",    &Point::y,     "y()\n");
  add_varargs_method("reference_count", &Point::reference_count);
}

void Interval::init_type()
{
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
}

void Bbox::init_type()
{
  behaviors().name("Bbox");
  behaviors().doc("A 2D bounding box");
  
  add_varargs_method("ll", 	&Bbox::ll, "ll()\n");
  add_varargs_method("ur", 	&Bbox::ur, "ur()\n");
  add_varargs_method("contains" , &Bbox::contains, "contains(x,y)\n");
  add_varargs_method("overlaps" , &Bbox::overlaps, "overlaps(bbox)\n");
  add_varargs_method("overlapsx" , &Bbox::overlapsx, "overlapsx(bbox)\n");
  add_varargs_method("overlapsy" , &Bbox::overlapsy, "overlapsy(bbox)\n");
  add_varargs_method("intervalx" , &Bbox::intervalx, "intervalx()\n");
  add_varargs_method("intervaly" , &Bbox::intervaly, "intervaly()\n");

  add_varargs_method("get_bounds", &Bbox::get_bounds, "get_bounds()\n");
  add_varargs_method("update" , &Bbox::update, "update(xys, ignore)\n");
  add_varargs_method("width", 	&Bbox::width, "width()\n");
  add_varargs_method("height", 	&Bbox::height, "height()\n");
  add_varargs_method("xmax", 	&Bbox::xmax, "xmax()\n");
  add_varargs_method("ymax", 	&Bbox::ymax, "ymax()\n");
  add_varargs_method("xmin", 	&Bbox::xmin, "xmin()\n");
  add_varargs_method("ymin", 	&Bbox::ymin, "ymin()\n");

  add_varargs_method("scale", 	 &Bbox::scale, "scale(sx,sy)");
  add_varargs_method("deepcopy", &Bbox::deepcopy, "deepcopy()\n");
}  



void Func::init_type()
{
  behaviors().name("Func");
  behaviors().doc("Map double -> double");
  behaviors().supportRepr();  
  add_varargs_method("map",     &Func::map, "map(x)\n");
  add_varargs_method("inverse", &Func::inverse, "inverse(y)\n");
  add_varargs_method("set_type", &Func::set_type, "set_type(TYPE)\n");
  add_varargs_method("get_type", &Func::get_type, "get_type()\n");
} 

void FuncXY::init_type()
{
  behaviors().name("FuncXY");
  behaviors().doc("Map double,double -> funcx(double), funcy(double)");
  add_varargs_method("map", &FuncXY::map, "map(x,y)\n");
  add_varargs_method("inverse", &FuncXY::inverse, "inverse(x,y)\n");
  add_varargs_method("set_funcx", &FuncXY::set_funcx, "set_funcx(func)\n");
  add_varargs_method("set_funcy", &FuncXY::set_funcy, "set_funcy(func)\n");
  add_varargs_method("get_funcx", &FuncXY::get_funcx, "get_funcx(func)\n");
  add_varargs_method("get_funcy", &FuncXY::get_funcy, "get_funcy(func)\n");
} 
 
void PolarXY::init_type()
{
  behaviors().name("PolarXY");
  behaviors().doc("map r, theta -> r*cos(theta), r*sin(theta)");
} 


void Transformation::init_type()
{
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
 

  add_varargs_method("xy_tup",   &Transformation::xy_tup,  "xy_tup(xy)\n");
  add_varargs_method("seq_x_y",  &Transformation::seq_x_y, "seq_x_y(x, y)\n");
  add_varargs_method("numerix_x_y",  &Transformation::numerix_x_y, "numerix_x_y(x, y)\n");
  add_varargs_method("seq_xy_tups", &Transformation::seq_xy_tups, "seq_xy_tups(seq)\n");  
  add_varargs_method("inverse_xy_tup",   &Transformation::inverse_xy_tup,  "inverse_xy_tup(xy)\n");

  add_varargs_method("set_offset",   &Transformation::set_offset,  "set_offset(xy, trans)\n");

  add_varargs_method("as_vec6", &Transformation::as_vec6, "as_vec6(): return the affine as length 6 list of Values\n");

}

void Affine::init_type()
{
  behaviors().name("Affine");
  behaviors().doc("A mutable float");
}


void SeparableTransformation::init_type()
{
  behaviors().name("SeparableTransformation");
  behaviors().doc("SeparableTransformation(box1, box2, funcx, funcy); x and y transformations are independet");

}

 



extern "C"
DL_EXPORT(void)
  init_transforms(void)
{
  static _transforms_module* _transforms = new _transforms_module;

  import_array();  


  Py::Dict d = _transforms->moduleDictionary();
  d["LOG10"] = Py::Int((int)Func::LOG10);
  d["IDENTITY"] = Py::Int((int)Func::IDENTITY);
};






