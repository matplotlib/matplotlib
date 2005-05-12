%include "typemaps.i"

// Some of these typemaps were borrowed or adpated from kiva's agg
// typemaps.

// Map a Python sequence into any sized C double array
// This handles arrays and sequences with non-float values correctly.
%typemap(python,in) double[ANY](double temp[$1_dim0]) {
  int i;
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_TypeError,"Expecting a sequence");
    return NULL;
  }
  if (PyObject_Length($input) != $1_dim0) {
    PyErr_SetString(PyExc_ValueError,"Expecting a sequence with $1_dim0 elements");
    return NULL;
  }
  for (i =0; i < $1_dim0; i++) {
    PyObject *o = PySequence_GetItem($input,i);
    if (PyFloat_Check(o)) {
      temp[i] = PyFloat_AsDouble(o);
    }  
    else {
      PyObject* converted = PyNumber_Float(o);
      if (!converted) {
	PyErr_SetString(PyExc_TypeError,"Expecting a sequence of floats");
	return NULL;
      }
      temp[i] = PyFloat_AsDouble(converted);  
      Py_DECREF(converted);
    }
  }
  $1 = &temp[0];
}

%typemap(python,in) double *parl(double temp[6]) {
  int i;
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_TypeError,"Expecting a sequence");
    return NULL;
  }
  if (PyObject_Length($input) != 6) {
    PyErr_SetString(PyExc_ValueError,"Expecting a sequence with 6 elements");
    return NULL;
  }
  for (i=0; i < 6; i++) {
    PyObject *o = PySequence_GetItem($input,i);
    if (PyFloat_Check(o)) {
      temp[i] = PyFloat_AsDouble(o);
    }  
    else {
      PyObject* converted = PyNumber_Float(o);
      if (!converted) {
	PyErr_SetString(PyExc_TypeError,"Expecting a sequence of floats");
	return NULL;
      }
      temp[i] = PyFloat_AsDouble(converted);  
      Py_DECREF(converted);
    }
  }
  $1 = &temp[0];
}

%typemap(typecheck,precedence=SWIG_TYPECHECK_DOUBLE_ARRAY ) double *parl 
{
  $1 = ($input != 0);
}

// map an 6 element double* output into a length 6 tuple
%typemap(in, numinputs=0) double *array6 (double temp[6]) {
  $1 = temp;
}

%typemap(argout) double *array6 {
  // Append output value $1 to $result
  PyObject *ret = PyTuple_New(6);
  for (unsigned i=0; i<6; i++)
    PyTuple_SetItem(ret,i,PyFloat_FromDouble($1[i]));
  $result = ret;
}


// --------------------------------------------------------------------------
//
// vertex() returns ( cmd, x, y)
//
// This tells SWIG to treat an double * argument with name 'x' as
// an output value.  We'll append the value to the current result which 
// is guaranteed to be a List object by SWIG.
// --------------------------------------------------------------------------
%typemap(in,numinputs=0) (double *vertex_x, double* vertex_y)(double temp1, 
                                                              double temp2)
{
    temp1 = 0; $1 = &temp1;
    temp2 = 0; $2 = &temp2;
}

%typemap(argout) (double *vertex_x, double* vertex_y)
{
    PyObject *px = PyFloat_FromDouble(*$1);
    PyObject *py = PyFloat_FromDouble(*$2);
    PyObject *return_val = PyTuple_New(3);
    PyTuple_SetItem(return_val,0,$result);
    // result is what was returned from vertex
    PyTuple_SetItem(return_val,1,px);
    PyTuple_SetItem(return_val,2,py);
    //Py_DECREF($result);
    $result = return_val;
}



