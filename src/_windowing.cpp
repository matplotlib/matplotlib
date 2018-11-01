#include "Python.h"
#include <windows.h>

static PyObject *
_GetForegroundWindow(PyObject *module, PyObject *args)
{
    HWND handle = GetForegroundWindow();
    if (!PyArg_ParseTuple(args, ":GetForegroundWindow"))
    {
        return NULL;
    }
    return PyLong_FromSize_t((size_t)handle);
}

static PyObject *
_SetForegroundWindow(PyObject *module, PyObject *args)
{
    HWND handle;
    if (!PyArg_ParseTuple(args, "n:SetForegroundWindow", &handle))
    {
        return NULL;
    }
    if (!SetForegroundWindow(handle))
    {
        return PyErr_Format(PyExc_RuntimeError,
                            "Error setting window");
    }
    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef _windowing_methods[] =
{
    {"GetForegroundWindow", _GetForegroundWindow, METH_VARARGS},
    {"SetForegroundWindow", _SetForegroundWindow, METH_VARARGS},
    {NULL, NULL}
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_windowing",
        "",
        -1,
        _windowing_methods,
        NULL,
        NULL,
        NULL,
        NULL
};

PyMODINIT_FUNC PyInit__windowing(void)
{
    PyObject *module = PyModule_Create(&moduledef);
    return module;
}
