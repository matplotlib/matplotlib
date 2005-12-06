#include "Python.h"
#include <windows.h>

static PyObject *
_GetForegroundWindow(PyObject *module, PyObject *args)
{
	HWND handle = GetForegroundWindow();
	if (!PyArg_ParseTuple(args, ":GetForegroundWindow"))
		return NULL;
	return PyInt_FromLong((long) handle);
}

static PyObject *
_SetForegroundWindow(PyObject *module, PyObject *args)
{
	HWND handle;
	if (!PyArg_ParseTuple(args, "l:SetForegroundWindow", &handle))
		return NULL;
	if (!SetForegroundWindow(handle))
		return PyErr_Format(PyExc_RuntimeError,
				    "Error setting window");
	Py_INCREF(Py_None);
	return Py_None;
}

static PyMethodDef _windowing_methods[] = {
{"GetForegroundWindow", _GetForegroundWindow, METH_VARARGS},
{"SetForegroundWindow", _SetForegroundWindow, METH_VARARGS},
{NULL, NULL}
};

extern "C" DL_EXPORT(void) init_windowing()
{
	Py_InitModule("_windowing", _windowing_methods);
}
