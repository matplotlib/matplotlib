#ifndef __FILE_COMPAT_H__
#define __FILE_COMPAT_H__

#include "numpy/npy_3kcompat.h"

#if NPY_API_VERSION < 0x4 /* corresponds to Numpy 1.5 */
/*
 * PyFile_* compatibility
 */
#if defined(NPY_PY3K)

/*
 * Get a FILE* handle to the file represented by the Python object
 */
static NPY_INLINE FILE*
npy_PyFile_Dup(PyObject *file, char *mode)
{
    int fd, fd2;
    PyObject *ret, *os;
    Py_ssize_t pos;
    FILE *handle;
    /* Flush first to ensure things end up in the file in the correct order */
    ret = PyObject_CallMethod(file, "flush", "");
    if (ret == NULL) {
        return NULL;
    }
    Py_DECREF(ret);
    fd = PyObject_AsFileDescriptor(file);
    if (fd == -1) {
        return NULL;
    }
    os = PyImport_ImportModule("os");
    if (os == NULL) {
        return NULL;
    }
    ret = PyObject_CallMethod(os, "dup", "i", fd);
    Py_DECREF(os);
    if (ret == NULL) {
        return NULL;
    }
    fd2 = PyNumber_AsSsize_t(ret, NULL);
    Py_DECREF(ret);
#ifdef _WIN32
    handle = _fdopen(fd2, mode);
#else
    handle = fdopen(fd2, mode);
#endif
    if (handle == NULL) {
        PyErr_SetString(PyExc_IOError,
                        "Getting a FILE* from a Python file object failed");
    }
    ret = PyObject_CallMethod(file, "tell", "");
    if (ret == NULL) {
        fclose(handle);
        return NULL;
    }
    pos = PyNumber_AsSsize_t(ret, PyExc_OverflowError);
    Py_DECREF(ret);
    if (PyErr_Occurred()) {
        fclose(handle);
        return NULL;
    }
    npy_fseek(handle, pos, SEEK_SET);
    return handle;
}

/*
 * Close the dup-ed file handle, and seek the Python one to the current position
 */
static NPY_INLINE int
npy_PyFile_DupClose(PyObject *file, FILE* handle)
{
    PyObject *ret;
    Py_ssize_t position;
    position = npy_ftell(handle);
    fclose(handle);

    ret = PyObject_CallMethod(file, "seek", NPY_SSIZE_T_PYFMT "i", position, 0);
    if (ret == NULL) {
        return -1;
    }
    Py_DECREF(ret);
    return 0;
}

static NPY_INLINE int
npy_PyFile_Check(PyObject *file)
{
    int fd;
    fd = PyObject_AsFileDescriptor(file);
    if (fd == -1) {
        PyErr_Clear();
        return 0;
    }
    return 1;
}

#else

#define npy_PyFile_Dup(file, mode) PyFile_AsFile(file)
#define npy_PyFile_DupClose(file, handle) (0)

#endif

static NPY_INLINE PyObject*
npy_PyFile_OpenFile(PyObject *filename, const char *mode)
{
    PyObject *open;
    open = PyDict_GetItemString(PyEval_GetBuiltins(), "open");
    if (open == NULL) {
        return NULL;
    }
    return PyObject_CallFunction(open, "Os", filename, mode);
}

#endif /* NPY_API_VERSION < 0x4 */

#if NPY_API_VERSION < 0x7 /* corresponds to Numpy 1.7 */

static NPY_INLINE int
npy_PyFile_CloseFile(PyObject *file)
{
    PyObject *ret;

    ret = PyObject_CallMethod(file, "close", NULL);
    if (ret == NULL) {
        return -1;
    }
    Py_DECREF(ret);
    return 0;
}

#endif /* NPY_API_VERSION < 0x7 */

#endif /* ifndef __FILE_COMPAT_H__ */
