//SWIG interface to agg_basics
%module agg
%{
#include "agg_basics.h"
#include "agg_trans_affine.h"
#include "agg_path_storage.h"  
  
using namespace agg;
  
  %}	

%include "agg_basics.i"


%typemap(argout) double *array6 {

  // Append output value $1 to $result
  $1 = PyString_AsString($input);   /* char *str */
  $2 = PyString_Size($input);       /* int len   */
  PyObject *ret = PyTuple_New(6);
  for (unsigned i=0; i<6; i++)
    PyTuple_SetItem(ret,i,PyFloat_FromDouble($1[i]));
  $result = ret;
}

// typemap for an incoming buffer
%typemap(in) (char *rbuffer, unsigned len) {
   if (!PyInt_Check($input)) {
       PyErr_SetString(PyExc_ValueError, "Expecting an integer");
       return NULL;
   }
   $2 = PyInt_AsLong($input);
   if ($2 < 0) {
       PyErr_SetString(PyExc_ValueError, "Positive integer expected");
       return NULL;
   }
   $1 = (char *) malloc($2);
}

// Return the buffer.  Discarding any previous return result
%typemap(argout) (char *rbuffer, unsigned len) {
   Py_XDECREF($result);   /* Blow away any previous result */
   if ($result < 0) {      /* Check for I/O error */
       free($1);
       PyErr_SetFromErrno(PyExc_IOError);
       return NULL;
   }
   $result = PyString_FromStringAndSize($1,$result);
   free($1);
}

%include "agg_trans_affine.i"
%include "agg_path_storage.i"
