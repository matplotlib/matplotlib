#include "Python.h"
#include "MPL_isnan.h"

static PyObject *
isnan64(PyObject *self, PyObject *args)
{
  double input;
  if (!PyArg_ParseTuple(args, "d",
			&input))
    return NULL;

  if (MPL_isnan64(input)) {
    Py_INCREF(Py_True);
    return Py_True;
  }
  
  Py_INCREF(Py_False);
  return Py_False;
}

static PyMethodDef _isnan_functions[] = {
    { "isnan64", (PyCFunction)isnan64, METH_VARARGS },
    { NULL, NULL, 0 }
};

DL_EXPORT(void)
init_isnan(void)
{
    PyObject *mod;
    mod = Py_InitModule("matplotlib._isnan",
			_isnan_functions);
}
