/* transforms.h	-- 
   The transformation classes and functions
 */

#ifndef _TRANSFORMS_H
#define _TRANSFORMS_H
#include <iostream>
#include <cmath>
#include <utility>
#include "CXX/Extensions.hxx"

template <class T> inline T&
max (T& a, T& b)
{
  if (a > b) return a; else return b;
}

template <class T> inline T&
min (T& a, T& b)
{
  if (a < b) return a; else return b;
}


class LazyValue : public Py::PythonExtension<LazyValue> {
public:

  virtual Py::Object get(const Py::Tuple &args) {
    double x = val();
    return Py::Float(x);
  }

  virtual Py::Object set(const Py::Tuple &args) {
    throw Py::RuntimeError("set not supported on this lazy value");
  }

  int compare(const Py::Object &other);
  virtual void set_api(const double&) {
    throw Py::RuntimeError("set not supported on this lazy value");
  }
  
  static void init_type(void);
  Py::Object number_add( const Py::Object & );  
  Py::Object number_divide( const Py::Object & );  
  Py::Object number_multiply( const Py::Object & );  
  Py::Object number_subtract( const Py::Object & );  
  virtual double val()=0;
};



// a mutable float
class Value: public LazyValue {
public:
  Value(double val) : _val(val) {}


  static void init_type(void);
  Py::Object set(const Py::Tuple &args);
  Py::Object get(const Py::Tuple &args);
  double val() {return _val;}

  void set_api(const double& x) {
    _val = x;
  }

private:
  double _val;
};



// binary operations on lazy values
class BinOp: public LazyValue {
public:
  BinOp(LazyValue* lhs, LazyValue* rhs, int opcode) : 
    _lhs(lhs), _rhs(rhs), _opcode(opcode) {
    //std::cout << "called binop init " << lhs->val() << " " << rhs->val() << std::endl;
  }

  static void init_type(void);
  Py::Object get(const Py::Tuple &args);
  double val() {
    //std::cout << "called binop val " << std::endl;
    double lhs = _lhs->val();
    double rhs = _rhs->val();
    double ret;
    if (_opcode==ADD) ret = lhs + rhs;
    else if (_opcode==MULTIPLY) ret =  lhs * rhs;
    else if (_opcode==DIVIDE) {
      //std::cout << "divide: " << lhs << " " << rhs << std::endl;
      if (rhs==0.0) 
	throw Py::ZeroDivisionError("Attempted divide by zero in BinOp::val()");
      ret = lhs / rhs;
    }
    else if (_opcode==SUBTRACT) ret = lhs-rhs;
    else 
      throw Py::ValueError("Unrecognized op code");


    //std::cout << "called binop val done " << std::endl;
    return ret;
  }

  enum {ADD, MULTIPLY, SUBTRACT, DIVIDE};  
private:
  LazyValue* _lhs;
  LazyValue* _rhs;
  const int _opcode;
};

class Point: public Py::PythonExtension<Point> {
public:
  Point(LazyValue* x, LazyValue*  y) : _x(x), _y(y) {}
  ~Point() { }
  static void init_type(void);

  // get the x Value instance
  Py::Object x(const Py::Tuple &args) { return Py::Object(_x); }

  // get the y Value instance 
  Py::Object y(const Py::Tuple &args) { return Py::Object(_y); }

  LazyValue* x_api() { return _x;}
  LazyValue* y_api() { return _y;}

  // for the extension API
  double  xval() {return _x->val();}
  double  yval() {return _y->val();}

private:
  LazyValue *_x;
  LazyValue *_y;
};



class Interval: public Py::PythonExtension<Interval> {
public:
  Interval(LazyValue* val1, LazyValue* val2) : _val1(val1), _val2(val2) {};

  static void init_type(void);
  
  Py::Object contains( const Py::Tuple &args) {
    args.verify_length(1);
    double x =  Py::Float(args[0]);
    int b = contains_api( x);
    return Py::Int(b);
  }

  int contains_api( const double& val) {

    double val1 = _val1->val();
    double val2 = _val2->val();

    return  ( (val>=val1) && (val<=val2) || (val>=val2) && (val<=val1) );

  }

  //update the interval to contain all points in seq of floats
  Py::Object update( const Py::Tuple &args);

  // x is in the open interval
  Py::Object contains_open( const Py::Tuple &args) {
    args.verify_length(1);
    double x =  Py::Float(args[0]);
    double val1 = _val1->val();
    double val2 = _val2->val();

    int b = ( (x>val1) && (x<val2) || (x>val2) && (x<val1) );
    return Py::Int(b);
  }

  Py::Object span( const Py::Tuple &args) {
    args.verify_length(0);
    double l =  _val2->val() - _val1->val();
    return Py::Float(l);
  }

  // return bounds as val1, val2 where val1 and 2 are floats
  Py::Object get_bounds( const Py::Tuple &args) {
    args.verify_length(0);
    Py::Tuple tup(2);
    double v1 = _val1->val();
    double v2 = _val2->val();
    
    tup[0] = Py::Float(v1);
    tup[1] = Py::Float(v2);
    return tup;
  }

  // set bounds on val1, val2 where args are floats
  Py::Object set_bounds( const Py::Tuple &args) {
    args.verify_length(2);

    double v1 = Py::Float(args[0]);
    double v2 = Py::Float(args[1]);
    _val1->set_api(v1);
    _val2->set_api(v2);
    return Py::Object();
  }


  // shift interval by step amount
  Py::Object shift( const Py::Tuple &args) {
    args.verify_length(1);

    double x = Py::Float(args[0]);
    double v1 = _val1->val();
    double v2 = _val2->val();

    _val1->set_api(v1+x);
    _val2->set_api(v2+x);
    return Py::Object();
  }

  Py::Object val1( const Py::Tuple &args) {return Py::Object(_val1);}
  Py::Object val2( const Py::Tuple &args) {return Py::Object(_val2);}

  
  
private:
  LazyValue* _val1;
  LazyValue* _val2;

};

class Bbox: public Py::PythonExtension<Bbox> {
public:
  Bbox(Point* ll, Point* ur) : _ll(ll), _ur(ur) {};
  static void init_type(void);

  // return lower left point
  Py::Object ll(const Py::Tuple &args) { return Py::Object(_ll); }

  // return upper right point
  Py::Object ur(const Py::Tuple &args) { return Py::Object(_ur); }

  Py::Object deepcopy(const Py::Tuple &args);
  Py::Object scale(const Py::Tuple &args);
  // get the l,b,w,h bounds
  Py::Object get_bounds(const Py::Tuple &args);

  Py::Object intervalx(const Py::Tuple &args) {
    return Py::Object( new Interval( _ll->x_api(), _ur->x_api()));
  }

  Py::Object intervaly(const Py::Tuple &args) {
    return Py::Object( new Interval( _ll->y_api(), _ur->y_api()));
  }

  Interval* intervalx_api() {
    return new Interval( _ll->x_api(), _ur->x_api());
  }

  Interval* intervaly_api() {
    return new Interval( _ll->y_api(), _ur->y_api());
  }

  // update the current bbox with data from xy tuples
  Py::Object update(const Py::Tuple &args);
  Py::Object contains(const Py::Tuple &args);

  Py::Object width(const Py::Tuple &args) {
    double w = _ur->xval() - _ll->xval();
    return Py::Float(w);
  }

  Py::Object height(const Py::Tuple &args) { 
    double h = _ur->yval() - _ll->yval();
    return Py::Float(h);
  }

  Py::Object xmax(const Py::Tuple &args) { 
    double x = _ur->xval();
    return Py::Float(x);
  }

  Py::Object ymax(const Py::Tuple &args) { 
    double y = _ur->yval();
    return Py::Float(y);
  }

  Py::Object xmin(const Py::Tuple &args) { 
    double x =  _ll->xval();
    return Py::Float(x);
  }

  Py::Object ymin(const Py::Tuple &args) { 
    double y =  _ll->yval();
    return Py::Float(y);
  }

  //return true if bboxes overlap
  Py::Object overlaps(const Py::Tuple &args); 
  //return true if the x extent overlaps
  Py::Object overlapsx(const Py::Tuple &args); 
  //return true if the x extent overlaps
  Py::Object overlapsy(const Py::Tuple &args); 

  
  ~Bbox() {}  

  Point* ll_api() {return _ll;}
  Point* ur_api() {return _ur;}
private:
  Point *_ll;
  Point *_ur;
};




//abstract base class for a function that takes maps a double to a
//double.  Also can serve as a lazy value evaluator
class Func : public Py::PythonExtension<Func> { 
public:

  static void init_type(void);
  

  // the python forward and inverse functions
  Py::Object map(const Py::Tuple &args);
  Py::Object inverse(const Py::Tuple &args);

  // the api forward and inverse functions
  virtual double  operator()(const double& )=0;
  virtual double  inverse_api(const double& )=0;

};

class Identity : public Func { 
public:

  static void init_type(void);

  // the api forward and inverse functions
  double  operator()(const double& x) {return x;}
  double  inverse_api(const double& x) {return x;}
};

class Log : public Func { 
public:

  static void init_type(void);

  double operator()(const double& x) { 
    if (x<=0) 
      throw Py::ValueError("Cannot take log of nonpositive value");
    return log10(x);
  };

  //the inverse mapping
  double inverse_api(const double& x) { 
    return pow(x,10.0);
  };
};

class FuncXY : public Py::PythonExtension<FuncXY> { 
public:
  FuncXY(Func* funcx=NULL, Func* funcy=NULL) : _funcx(funcx), _funcy(funcy){}
  static void init_type(void);

  Py::Object map(const Py::Tuple &args);
  Py::Object inverse(const Py::Tuple &args);

  virtual Py::Object set_funcx(const Py::Tuple &args);
  virtual Py::Object set_funcy(const Py::Tuple &args);
  virtual Py::Object get_funcx(const Py::Tuple &args);
  virtual Py::Object get_funcy(const Py::Tuple &args);


  // the api forward and inverse functions
  virtual std::pair<double, double>  operator()(const double& x, const double& y ) {
    return std::pair<double, double>( _funcx->operator()(x),
				      _funcy->operator()(y) );
  }
  virtual std::pair<double, double>  inverse_api(const double& x, const double& y ) {
    return std::pair<double, double>( _funcx->inverse_api(x),
				      _funcy->inverse_api(y) );
  }
private:
  Func* _funcx;
  Func* _funcy;

};

// the x and y mappings are independent of one another, eg linear,
// log, semilogx or semilogy
class PolarXY : public FuncXY { 
public:

  static void init_type(void);

  // the api forward and inverse functions; theta in radians
  std::pair<double, double>  operator()(const double& r, const double& theta ) {
    return std::pair<double, double>( r*cos(theta), r*sin(theta) );
  }
  std::pair<double, double>  inverse_api(const double& x, const double& y ) {
    double r = sqrt( x*x + y*y);
    if (r==0)
      throw Py::ValueError("Cannot invert zero radius polar");
    double theta = acos(x/r);
    return std::pair<double, double>(r, theta);
  }

  Py::Object set_funcx(const Py::Tuple &args) {
    throw Py::RuntimeError("set_funcx meaningless for polar transforms");
  }
  Py::Object set_funcy(const Py::Tuple &args) {
    throw Py::RuntimeError("set_funcy meaningless for polar transforms");
  }
  Py::Object get_funcx(const Py::Tuple &args) {
    throw Py::RuntimeError("get_funcx meaningless for polar transforms");
  }
  Py::Object get_funcy(const Py::Tuple &args) {
    throw Py::RuntimeError("get_funcy meaningless for polar transforms");
  }

};

class Transformation: public Py::PythonExtension<Transformation> {
public:
  Transformation() : _usingOffset(0), _transOffset(NULL), 
		     _xo(0), _yo(0), 
		     _invertible(true), _frozen(false) {}
  static void init_type(void);

  // for separable transforms 
  virtual Py::Object get_bbox1(const Py::Tuple &args);
  virtual Py::Object get_bbox2(const Py::Tuple &args);
  virtual Py::Object set_bbox1(const Py::Tuple &args);
  virtual Py::Object set_bbox2(const Py::Tuple &args);

  virtual Py::Object get_funcx(const Py::Tuple &args);
  virtual Py::Object get_funcy(const Py::Tuple &args);
  virtual Py::Object set_funcx(const Py::Tuple &args);
  virtual Py::Object set_funcy(const Py::Tuple &args);

  // for affine transforms 
  virtual Py::Object as_vec6(const Py::Tuple &args);

  // for all children
  Py::Object xy_tup(const Py::Tuple &args);
  Py::Object seq_xy_tups(const Py::Tuple &args);
  Py::Object seq_x_y(const Py::Tuple &args);
  Py::Object numerix_x_y(const Py::Tuple &args);
  Py::Object inverse_xy_tup(const Py::Tuple &args);

  //freeze the lazy values and don't relax until thawed
  Py::Object freeze(const Py::Tuple &args) {
      // evaluate the lazy objects  
    if (!_frozen) {
      eval_scalars();
      if (_usingOffset) _transOffset->eval_scalars();
      _frozen = true;
    }
    return Py::Object();
  }

  Py::Object thaw(const Py::Tuple &args) {
    _frozen = false;
    return Py::Object();
  }

  //the the (optional) offset as xy, transOffset.  After
  //transformation, if the offset is set, the transformed points will
  //be translated by transOffset(xy)
  Py::Object set_offset(const Py::Tuple &args);
  
  virtual std::pair<double, double> & operator()(const double &x, const double &y)=0;
  virtual std::pair<double, double> & inverse_api(const double &x, const double &y)=0;


  virtual void eval_scalars(void)=0;
  std::pair<double, double> xy;

protected:
  // the post transformation offsets, if any
  bool _usingOffset;
  Transformation *_transOffset;
  double _xo, _yo, _xot, _yot;
  bool _invertible, _frozen;
};

class SeparableTransformation: public Transformation {
public:
  SeparableTransformation(Bbox *b1, Bbox *b2, Func *funcx, Func *funcy) : 
    Transformation(), 
    _b1(b1), _b2(b2), _funcx(funcx), _funcy(funcy)  {}

  static void init_type(void);

  Py::Object get_bbox1(const Py::Tuple &args);
  Py::Object get_bbox2(const Py::Tuple &args);
  Py::Object set_bbox1(const Py::Tuple &args);
  Py::Object set_bbox2(const Py::Tuple &args);

  Py::Object get_funcx(const Py::Tuple &args);
  Py::Object get_funcy(const Py::Tuple &args);
  Py::Object set_funcx(const Py::Tuple &args);
  Py::Object set_funcy(const Py::Tuple &args);
  Py::Object set_offset(const Py::Tuple &args);  
  std::pair<double, double> & operator()(const double &x, const double &y);
  std::pair<double, double> & inverse_api(const double &x, const double &y);


protected:
  Bbox *_b1, *_b2;
  Func *_funcx, *_funcy;

  double _sx, _sy, _tx, _ty;
  double _isx, _isy, _itx, _ity;  // the inverse params

  void eval_scalars(void);

};



class Affine: public Transformation {
public:
  Affine(LazyValue *a, LazyValue *b,  LazyValue *c, 
	 LazyValue *d, LazyValue *tx, LazyValue *ty) : 
    _a(a), _b(b), _c(c), _d(d), _tx(tx), _ty(ty) {}

  static void init_type(void);

  Py::Object as_vec6(const Py::Tuple &args);

  std::pair<double, double> & operator()(const double &x, const double &y);
  std::pair<double, double> & inverse_api(const double &x, const double &y);
  void eval_scalars(void);
  
private:
  LazyValue *_a;
  LazyValue *_b;
  LazyValue *_c;
  LazyValue *_d;
  LazyValue *_tx;
  LazyValue *_ty;

  double _aval;
  double _bval;
  double _cval;
  double _dval;
  double _txval;
  double _tyval;

  double _iaval;
  double _ibval;
  double _icval;
  double _idval;
  double _itxval;
  double _ityval;

};

// the extension module
class _transforms_module : public Py::ExtensionModule<_transforms_module>

{
public:
  _transforms_module()
    : Py::ExtensionModule<_transforms_module>( "_transforms" )
  {
    LazyValue::init_type();
    Value::init_type();
    BinOp::init_type();

    Point::init_type();
    Interval::init_type();
    Bbox::init_type();


    
    Func::init_type();
    Identity::init_type();
    Log::init_type();
    
    FuncXY::init_type();
    PolarXY::init_type();

    Transformation::init_type();     
    SeparableTransformation::init_type();     
    Affine::init_type();

    add_varargs_method("Value", &_transforms_module::new_value, 
		       "Value(x)");
    add_varargs_method("Point", &_transforms_module::new_point, 
		       "Point(x, y)");

    add_varargs_method("Bbox", &_transforms_module::new_bbox, 
		       "Bbox(ll, ur)");
    add_varargs_method("Interval", &_transforms_module::new_interval, 
		       "Interval(val1, val2)");

    add_varargs_method("Identity", &_transforms_module::new_identity, 
		       "Identity())");
    add_varargs_method("Log", &_transforms_module::new_log, 
		       "Log())");

    add_varargs_method("FuncXY", &_transforms_module::new_funcxy, 
		       "FuncXY(funcx, funcy)");
    add_varargs_method("PolarXY", &_transforms_module::new_polarxy, 
		       "PolarXY");

    add_varargs_method("SeparableTransformation", 
		       &_transforms_module::new_separable_transformation, 
		       "SeparableTransformation(box1, box2, funcx, funcy))");
    add_varargs_method("Affine", &_transforms_module::new_affine, 
		       "Affine(a,b,c,d,tx,ty)");
    initialize( "The _transforms module" );
  }
  
  virtual ~_transforms_module() {}
  
private:
  
  Py::Object new_value (const Py::Tuple &args);
  Py::Object new_point (const Py::Tuple &args);

  Py::Object new_bbox (const Py::Tuple &args);
  Py::Object new_interval (const Py::Tuple &args);
  Py::Object new_affine (const Py::Tuple &args);

  Py::Object new_identity (const Py::Tuple &args);
  Py::Object new_log (const Py::Tuple &args);

  Py::Object new_funcxy (const Py::Tuple &args);
  Py::Object new_polarxy (const Py::Tuple &args);

  Py::Object new_separable_transformation (const Py::Tuple &args);
};




#endif
