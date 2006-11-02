/* transforms.h	--
   The transformation classes and functions
 */

#ifndef _TRANSFORMS_H
#define _TRANSFORMS_H
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <limits>
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
  ~Value();
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
  BinOp(LazyValue* lhs, LazyValue* rhs, int opcode);
  ~BinOp();

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
  Point(LazyValue* x, LazyValue*  y);
  ~Point();
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

  Py::Object reference_count (const Py::Tuple& args)
  {
    return Py::Int(this->ob_refcnt);
  }

private:
  LazyValue *_x;
  LazyValue *_y;
};


class MinPositive {
public:
  MinPositive() : val(std::numeric_limits<double>::max()) {};
  ~MinPositive() {};


  //update the minimum positive value with float
  void update( double x) {
    if (x>0 && x<val) val = x;
  }

  double val;
};


class Interval: public Py::PythonExtension<Interval> {
public:
  Interval(LazyValue* val1, LazyValue* val2);
  ~Interval();

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
  Py::Object minpos( const Py::Tuple &args) {

    double valpos = std::numeric_limits<double>::max();
    if (_minpos!=NULL)
      valpos = _minpos->val;


    double val1 = _val1->val();
    double val2 = _val2->val();

    if (val1<0 && val2<0) {
      valpos = -1.0;
    }
    else {
      if (val1>0 && val1<valpos) valpos = val1;
      if (val2>0 && val2<valpos) valpos = val2;
    }
    return Py::Float(valpos);

  }

  void set_minpos(MinPositive* p) {_minpos = p;}

private:
  LazyValue* _val1;
  LazyValue* _val2;
  MinPositive* _minpos;
};




class Bbox: public Py::PythonExtension<Bbox> {
public:
  Bbox(Point* ll, Point* ur);
  ~Bbox();
  static void init_type(void);

  /*
  Py::Object getattr( const char *name )
	{
	  std::cout <<"called getattr"<<std::endl;
	return getattr_methods( name );
	}

  Py::Object repr () const {
    std::cout <<"called repr"<<std::endl;
    return Py::String("repr: hi mom");}
  //Py::Object str () const {return Py::String("str:  hi mom");}
  */
  // return lower left point
  Py::Object ll(const Py::Tuple &args) { return Py::Object(_ll); }

  // return upper right point
  Py::Object ur(const Py::Tuple &args) { return Py::Object(_ur); }

  Py::Object deepcopy(const Py::Tuple &args);
  Py::Object _deepcopy(void);
  Py::Object scale(const Py::Tuple &args);
  // get the l,b,w,h bounds
  Py::Object get_bounds(const Py::Tuple &args);

  Py::Object intervalx(const Py::Tuple &args) {
    Interval* intv = new Interval( _ll->x_api(), _ur->x_api());
    intv->set_minpos(&_posx);
    return Py::asObject( intv);
  }

  Py::Object intervaly(const Py::Tuple &args) {
    Interval* intv = new Interval( _ll->y_api(), _ur->y_api());
    intv->set_minpos(&_posy);
    return Py::asObject( intv);
  }


  // update the current bbox with data from xy tuples
  Py::Object update(const Py::Tuple &args);
  Py::Object update_numerix( const Py::Tuple &args);
  Py::Object update_numerix_xy( const Py::Tuple &args);
  Py::Object contains(const Py::Tuple &args);
  Py::Object count_contains(const Py::Tuple &args);

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
  Py::Object overlaps(const Py::Tuple &args, const Py::Dict &kwargs);
  //return true if the x extent overlaps
  Py::Object overlapsx(const Py::Tuple &args, const Py::Dict &kwargs);
  //return true if the x extent overlaps
  Py::Object overlapsy(const Py::Tuple &args, const Py::Dict &kwargs);

  //set the ignore attr
  Py::Object ignore(const Py::Tuple &args);



  Point* ll_api() {return _ll;}
  Point* ur_api() {return _ur;}
private:

  Point *_ll;
  Point *_ur;
  MinPositive _posx, _posy;
  int _ignore; // ignore the past when updating datalim
};




//abstract base class for a function that takes maps a double to a
//double.  Also can serve as a lazy value evaluator
class Func : public Py::PythonExtension<Func> {
public:
  Func( unsigned int type=IDENTITY ) : _type(type) {};
  ~Func();

  static void init_type(void);

  Py::Object str() { return Py::String(as_string());}
  Py::Object repr() { return Py::String(as_string());}
  std::string as_string() {
    if   (_type==IDENTITY) return "Identity";
    else if (_type==LOG10) return "Log10";
    else throw Py::ValueError("Unrecognized function type");
  }

  Py::Object set_type(const Py::Tuple &args) {
    args.verify_length(1);
    _type  = Py::Int(args[0]);
    return Py::Object();
  }

  int get_type_api() const {
    return (int)_type;
  }

  Py::Object get_type(const Py::Tuple &args) {
    return Py::Int((int)_type);
  }

  // the python forward and inverse functions
  Py::Object map(const Py::Tuple &args);
  Py::Object inverse(const Py::Tuple &args);

  // the api forward and inverse functions
  double  operator()(const double& x) {
    if (_type==IDENTITY) return x;
    else if (_type==LOG10) {
      if (x<=0) {
	//throw Py::ValueError("test throw");
	throw std::domain_error("Cannot take log of nonpositive value");

      }
      return log10(x);
    }
    else
      throw Py::ValueError("Unrecognized function type");
  }
  double  inverse_api(const double& x) {
    if   (_type==IDENTITY) return x;
    else if (_type==LOG10) return pow(10.0, x);
    else throw Py::ValueError("Unrecognized function type");
	}
  // the api array operator functions

  void arrayOperator(const int length, const double x[], double newx[]) {
    if (_type==IDENTITY){
	for(int i=0; i < length; i++)
		newx[i] = x[i];
	}
    else if (_type==LOG10) {
	for(int i=0; i < length; i++)
	{
		if (x<=0) { throw std::domain_error("Cannot take log of nonpositive value"); }
		else newx[i] = log10(x[i]);
	}
      }
    else
      throw Py::ValueError("Unrecognized function type");
  }
  void arrayInverse(const int length, const double x[], double newx[]) {
    if (_type==IDENTITY){
	for(int i=0; i < length; i++)
		newx[i] = x[i];
	}
    else if (_type==LOG10) {
	for(int i=0; i < length; i++)
		newx[i] = pow(10.0, x[i]);
      }
    else
      throw Py::ValueError("Unrecognized function type");
  }

  enum {IDENTITY, LOG10};
private:
  unsigned int _type;
};

class FuncXY : public Py::PythonExtension<FuncXY> {
public:
  FuncXY( unsigned int type=POLAR ) : _type(type) {};
  ~FuncXY() {};

  static void init_type(void);
  Py::Object set_type(const Py::Tuple &args) {
    args.verify_length(1);
    _type  = Py::Int(args[0]);
    return Py::Object();
  }

  Py::Object get_type(const Py::Tuple &args) {
    return Py::Int((int)_type);
  }

  Py::Object map(const Py::Tuple &args);
  Py::Object inverse(const Py::Tuple &args);

  std::pair<double, double>  operator()(const double& x, const double& y ) {
    if   (_type==POLAR) {
      //x,y are thetaa, r
      return std::pair<double, double>( y*cos(x), y*sin(x) );
    }
    else throw Py::ValueError("Unrecognized function type");
  }
  std::pair<double, double>  inverse_api(const double& x, const double& y ) {
    if   (_type==POLAR) {
      double r = sqrt( x*x + y*y);
      if (r==0)
	throw Py::ValueError("Cannot invert zero radius polar");
      double theta = acos(x/r);
      if (y<0) theta = 2*3.1415926535897931-theta;
      return std::pair<double, double>(theta, r);
    }
    else throw Py::ValueError("Unrecognized function type");
  }

  void arrayOperator(const int length, const double x[], const double y[], double newx[], double newy[]) {
    if   (_type==POLAR) {
      //x,y are theta, r
	for(int i=0; i < length; i++)
	{
		newx[i] = y[i] * cos(x[i]);
		newy[i] = y[i] * sin(x[i]);
	}
    }
    else throw Py::ValueError("Unrecognized function type");
  }
  void arrayInverse(const int length, const double x[], const double y[], double newx[], double newy[]) {
    if   (_type==POLAR) {
	for(int i=0; i < length; i++)
	{
      		double r = sqrt( x[i]*x[i] + y[i]*y[i]);
      		if (r==0)
			throw Py::ValueError("Cannot invert zero radius polar");
      		double theta = acos(x[i]/r);
      		if (y<0) theta = 2*3.1415926535897931-theta;
      		newx[i] = theta;
		newy[i] = r;
	}
    }
    else throw Py::ValueError("Unrecognized function type");
  }

enum {POLAR};
private:
  unsigned int _type;
};

class Transformation: public Py::PythonExtension<Transformation> {
public:
  Transformation() : _usingOffset(0), _transOffset(NULL),
		     _xo(0), _yo(0),
		     _invertible(true), _frozen(false) {}
  ~Transformation();

  static void init_type(void);

  //return whether a nonlinear transform is needed
  virtual bool need_nonlinear_api() {return true;};
  Py::Object need_nonlinear(const Py::Tuple& args) {
    return Py::Int(need_nonlinear_api());
  }

  //all derived classes must implement these
  // do the nonlinear transformation compenent in place
  virtual void nonlinear_only_api(double *x, double *y) {}; // do nothing

  // the affine for the linear part
  virtual void affine_params_api(double* a, double* b, double* c, double*d, double* tx, double* ty)=0;

  Py::Object as_vec6_val(const Py::Tuple &args) {
    double a,b,c,d,tx,ty;

    try {
      affine_params_api(&a, &b, &c, &d, &tx, &ty);
    }
  catch(...) {
    throw Py::ValueError("Domain error on as_vec6_val in Transformation::as_vec6_val");
  }

    Py::Tuple ret(6);
    ret[0] = Py::Float(a);
    ret[1] = Py::Float(b);
    ret[2] = Py::Float(c);
    ret[3] = Py::Float(d);
    ret[4] = Py::Float(tx);
    ret[5] = Py::Float(ty);
    return ret;

  }



  // for custom methods of derived classes to work in pycxx, the base
  // class must define them all.  I define them, and raise by default.
  // Ugly, yes.  for bbox transforms
  virtual Py::Object get_bbox1(const Py::Tuple &args);
  virtual Py::Object get_bbox2(const Py::Tuple &args);
  virtual Py::Object set_bbox1(const Py::Tuple &args);
  virtual Py::Object set_bbox2(const Py::Tuple &args);

  // for separable transforms
  virtual Py::Object get_funcx(const Py::Tuple &args);
  virtual Py::Object get_funcy(const Py::Tuple &args);
  virtual Py::Object set_funcx(const Py::Tuple &args);
  virtual Py::Object set_funcy(const Py::Tuple &args);

  // for nonseparable transforms
  virtual Py::Object get_funcxy(const Py::Tuple &args);
  virtual Py::Object set_funcxy(const Py::Tuple &args);


  // for affine transforms
  virtual Py::Object as_vec6(const Py::Tuple &args);


  // for all children
  Py::Object xy_tup(const Py::Tuple &args);
  Py::Object seq_xy_tups(const Py::Tuple &args);
  Py::Object seq_x_y(const Py::Tuple &args);
  Py::Object numerix_x_y(const Py::Tuple &args);
  Py::Object numerix_xy(const Py::Tuple &args);
  Py::Object inverse_numerix_xy(const Py::Tuple &args);
  Py::Object nonlinear_only_numerix(const Py::Tuple &args, const Py::Dict &kwargs);
  Py::Object inverse_xy_tup(const Py::Tuple &args);
  virtual Py::Object deepcopy(const Py::Tuple &args) =0;
  virtual Py::Object shallowcopy(const Py::Tuple &args) =0;

  //freeze the lazy values and don't relax until thawed
  Py::Object freeze(const Py::Tuple &args) {
      // evaluate the lazy objects
    if (!_frozen) {
      try {
	eval_scalars();
      }
      catch(...) {
	throw Py::ValueError("Domain error on eval_scalars in Transformation::freeze");
      }

      if (_usingOffset) {
	try {
	  _transOffset->eval_scalars();
	}
	catch(...) {
	  throw Py::ValueError("Domain error on eval_scalars in transoffset Transformation::eval_scalars");
	}

      }

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
  virtual void arrayOperator(const int length, const double x[], const double y[], double newx[], double newy[])
	{throw Py::ValueError("Function arrayOperator not implemented for this class");}

  virtual void eval_scalars(void)=0;
  std::pair<double, double> xy;

protected:
  // the post transformation offsets, if any

  bool _usingOffset;
  Transformation *_transOffset;
  double _xo, _yo, _xot, _yot;
  bool _invertible, _frozen;

};


class BBoxTransformation: public Transformation {
public:
  BBoxTransformation(Bbox *b1, Bbox *b2);
  ~BBoxTransformation();
  Py::Object get_bbox1(const Py::Tuple &args);
  Py::Object get_bbox2(const Py::Tuple &args);
  Py::Object set_bbox1(const Py::Tuple &args);
  Py::Object set_bbox2(const Py::Tuple &args);

  void affine_params_api(double* a, double* b, double* c, double*d, double* tx, double* ty);
protected:
  Bbox *_b1, *_b2;
  double _sx, _sy, _tx, _ty;      // the bbox transform params
  double _isx, _isy, _itx, _ity;  // the bbox inverse params

  virtual void eval_scalars(void)=0;

};


class SeparableTransformation: public BBoxTransformation {
public:
  SeparableTransformation(Bbox *b1, Bbox *b2, Func *funcx, Func *funcy);
  ~SeparableTransformation();

  static void init_type(void);



  bool need_nonlinear_api() {
    return !(_funcx->get_type_api()==Func::IDENTITY &&
	     _funcy->get_type_api()==Func::IDENTITY);
  }

  Py::Object as_vec6_val(const Py::Tuple &args);
  Py::Object get_funcx(const Py::Tuple &args);
  Py::Object get_funcy(const Py::Tuple &args);
  Py::Object set_funcx(const Py::Tuple &args);
  Py::Object set_funcy(const Py::Tuple &args);
  Py::Object set_offset(const Py::Tuple &args);

  void nonlinear_only_api(double *x, double *y);
  std::pair<double, double> & operator()(const double &x, const double &y);
  std::pair<double, double> & inverse_api(const double &x, const double &y);
  void arrayOperator(const int length, const double x[], const double y[], double newx[], double newy[]);

  Py::Object deepcopy(const Py::Tuple &args) ;
  Py::Object shallowcopy(const Py::Tuple &args) ;

protected:
  Func *_funcx, *_funcy;
  void eval_scalars(void);
};

class NonseparableTransformation: public BBoxTransformation {
public:
  NonseparableTransformation(Bbox *b1, Bbox *b2, FuncXY *funcxy);
  ~NonseparableTransformation();

  static void init_type(void);

  Py::Object get_funcxy(const Py::Tuple &args);
  Py::Object set_funcxy(const Py::Tuple &args);
  Py::Object set_offset(const Py::Tuple &args);
  std::pair<double, double> & operator()(const double &x, const double &y);
  std::pair<double, double> & inverse_api(const double &x, const double &y);
  void arrayOperator(const int length, const double x[], const double y[], double newx[], double newy[]);
  Py::Object deepcopy(const Py::Tuple &args) ;
  Py::Object shallowcopy(const Py::Tuple &args) ;

  void nonlinear_only_api(double *x, double *y);

protected:
  FuncXY *_funcxy;
  void eval_scalars(void);
};


class Affine: public Transformation {
public:
  Affine(LazyValue *a, LazyValue *b,  LazyValue *c,
	 LazyValue *d, LazyValue *tx, LazyValue *ty);

  ~Affine();

  static void init_type(void);

  bool need_nonlinear_api() {
    return false;
  }

  Py::Object as_vec6(const Py::Tuple &args);

  std::pair<double, double> & operator()(const double &x, const double &y);
  std::pair<double, double> & inverse_api(const double &x, const double &y);
  void eval_scalars(void);
  Py::Object deepcopy(const Py::Tuple &args);
  Py::Object shallowcopy(const Py::Tuple &args);

  void affine_params_api(double* a, double* b, double* c, double*d, double* tx, double* ty);
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
    FuncXY::init_type();
    Transformation::init_type();
    SeparableTransformation::init_type();
    NonseparableTransformation::init_type();
    Affine::init_type();

    add_varargs_method("Value", &_transforms_module::new_value,
		       "Value(x)");
    add_varargs_method("Point", &_transforms_module::new_point,
		       "Point(x, y)");

    add_varargs_method("Bbox", &_transforms_module::new_bbox,
		       "Bbox(ll, ur)");
    add_varargs_method("Interval", &_transforms_module::new_interval,
		       "Interval(val1, val2)");

    add_varargs_method("Func", &_transforms_module::new_func,
		       "Func(typecode)");
    add_varargs_method("FuncXY", &_transforms_module::new_funcxy,
		       "FuncXY(funcx, funcy)");

    add_varargs_method("SeparableTransformation",
		       &_transforms_module::new_separable_transformation,
		       "SeparableTransformation(box1, box2, funcx, funcy))");
    add_varargs_method("NonseparableTransformation",
		       &_transforms_module::new_nonseparable_transformation,
		       "NonseparableTransformation(box1, box2, funcxy))");
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
  Py::Object new_func (const Py::Tuple &args);
  Py::Object new_funcxy (const Py::Tuple &args);
  Py::Object new_separable_transformation (const Py::Tuple &args);
  Py::Object new_nonseparable_transformation (const Py::Tuple &args);
};




#endif
