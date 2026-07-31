#import "MPLUtils.h"

// Acquire the GIL, call a method with no args, discarding the result and
// printing any exception.
void gil_call_method(PyObject* obj, const char* name)
{
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* result = PyObject_CallMethod(obj, name, NULL);
    if (result) {
        Py_DECREF(result);
    } else {
        PyErr_Print();
    }
    PyGILState_Release(gstate);
}

void process_event(char const* cls_name, char const* fmt, ...)
{
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* module = NULL, * cls = NULL,
            * args = NULL, * kwargs = NULL,
            * event = NULL, * result = NULL;
    va_list argp;
    va_start(argp, fmt);
    if (!(module = PyImport_ImportModule("matplotlib.backend_bases"))
        || !(cls = PyObject_GetAttrString(module, cls_name))
        || !(args = PyTuple_New(0))
        || !(kwargs = Py_VaBuildValue(fmt, argp))
        || !(event = PyObject_Call(cls, args, kwargs))
        || !(result = PyObject_CallMethod(event, "_process", ""))) {
        PyErr_Print();
    }
    va_end(argp);
    Py_XDECREF(module);
    Py_XDECREF(cls);
    Py_XDECREF(args);
    Py_XDECREF(kwargs);
    Py_XDECREF(event);
    Py_XDECREF(result);
    PyGILState_Release(gstate);
}
