#define PY_SSIZE_T_CLEAN
#include <Python.h>
#ifdef _WIN32
#include <Objbase.h>
#include <Shobjidl.h>
#include <Windows.h>
#endif

static PyObject* mpl_GetCurrentProcessExplicitAppUserModelID(PyObject* module)
{
#ifdef _WIN32
    wchar_t* appid = NULL;
    HRESULT hr = GetCurrentProcessExplicitAppUserModelID(&appid);
    if (FAILED(hr)) {
        return PyErr_SetFromWindowsErr(hr);
    }
    PyObject* py_appid = PyUnicode_FromWideChar(appid, -1);
    CoTaskMemFree(appid);
    return py_appid;
#else
    Py_RETURN_NONE;
#endif
}

static PyObject* mpl_SetCurrentProcessExplicitAppUserModelID(PyObject* module, PyObject* arg)
{
#ifdef _WIN32
    wchar_t* appid = PyUnicode_AsWideCharString(arg, NULL);
    if (!appid) {
        return NULL;
    }
    HRESULT hr = SetCurrentProcessExplicitAppUserModelID(appid);
    PyMem_Free(appid);
    if (FAILED(hr)) {
        return PyErr_SetFromWindowsErr(hr);
    }
    Py_RETURN_NONE;
#else
    Py_RETURN_NONE;
#endif
}

static PyObject* mpl_GetForegroundWindow(PyObject* module)
{
#ifdef _WIN32
  return PyLong_FromVoidPtr(GetForegroundWindow());
#else
  Py_RETURN_NONE;
#endif
}

static PyObject* mpl_SetForegroundWindow(PyObject* module, PyObject *arg)
{
#ifdef _WIN32
  HWND handle = PyLong_AsVoidPtr(arg);
  if (PyErr_Occurred()) {
    return NULL;
  }
  if (!SetForegroundWindow(handle)) {
    return PyErr_Format(PyExc_RuntimeError, "Error setting window");
  }
  Py_RETURN_NONE;
#else
  Py_RETURN_NONE;
#endif
}

static PyMethodDef functions[] = {
    {"Win32_GetCurrentProcessExplicitAppUserModelID",
     (PyCFunction)mpl_GetCurrentProcessExplicitAppUserModelID, METH_NOARGS,
     "Win32_GetCurrentProcessExplicitAppUserModelID()\n--\n\n"
     "Wrapper for Windows's GetCurrentProcessExplicitAppUserModelID.  On \n"
     "non-Windows platforms, always returns None."},
    {"Win32_SetCurrentProcessExplicitAppUserModelID",
     (PyCFunction)mpl_SetCurrentProcessExplicitAppUserModelID, METH_O,
     "Win32_SetCurrentProcessExplicitAppUserModelID(appid, /)\n--\n\n"
     "Wrapper for Windows's SetCurrentProcessExplicitAppUserModelID.  On \n"
     "non-Windows platforms, a no-op."},
    {"Win32_GetForegroundWindow",
     (PyCFunction)mpl_GetForegroundWindow, METH_NOARGS,
     "Win32_GetForegroundWindow()\n--\n\n"
     "Wrapper for Windows' GetForegroundWindow.  On non-Windows platforms, \n"
     "always returns None."},
    {"Win32_SetForegroundWindow",
     (PyCFunction)mpl_SetForegroundWindow, METH_O,
     "Win32_SetForegroundWindow(hwnd)\n--\n\n"
     "Wrapper for Windows' SetForegroundWindow.  On non-Windows platforms, \n"
     "a no-op."},
    {NULL, NULL}};  // sentinel.
static PyModuleDef util_module = {
    PyModuleDef_HEAD_INIT, "_c_internal_utils", "", 0, functions, NULL, NULL, NULL, NULL};

#pragma GCC visibility push(default)
PyMODINIT_FUNC PyInit__c_internal_utils(void)
{
    return PyModule_Create(&util_module);
}
