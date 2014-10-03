/* -*- mode: c++; c-basic-offset: 4 -*- */

#include "mplutils.h"

int add_dict_int(PyObject *dict, const char *key, long val)
{
    PyObject *valobj;
    valobj = PyLong_FromLong(val);
    if (valobj == NULL) {
        return 1;
    }

    if (PyDict_SetItemString(dict, (char *)key, valobj)) {
        Py_DECREF(valobj);
        return 1;
    }

    Py_DECREF(valobj);

    return 0;
}
