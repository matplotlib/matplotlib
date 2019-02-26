#ifndef MPL_FILE_COMPAT_H
#define MPL_FILE_COMPAT_H

#include <Python.h>
#include <stdio.h>
#include "numpy/npy_common.h"
#include "numpy/ndarrayobject.h"
#include "mplutils.h"

#ifdef __cplusplus
extern "C" {
#endif
#if defined(_MSC_VER) && defined(_WIN64) && (_MSC_VER > 1400)
    #include <io.h>
    #define mpl_fseek _fseeki64
    #define mpl_ftell _ftelli64
    #define mpl_lseek _lseeki64
    #define mpl_off_t npy_int64

    #if NPY_SIZEOF_INT == 8
        #define MPL_OFF_T_PYFMT "i"
    #elif NPY_SIZEOF_LONG == 8
        #define MPL_OFF_T_PYFMT "l"
    #elif NPY_SIZEOF_LONGLONG == 8
        #define MPL_OFF_T_PYFMT "L"
    #else
        #error Unsupported size for type off_t
    #endif
#else
    #define mpl_fseek fseek
    #define mpl_ftell ftell
    #define mpl_lseek lseek
    #define mpl_off_t off_t

    #if NPY_SIZEOF_INT == NPY_SIZEOF_SHORT
        #define MPL_OFF_T_PYFMT "h"
    #elif NPY_SIZEOF_INT == NPY_SIZEOF_INT
        #define MPL_OFF_T_PYFMT "i"
    #elif NPY_SIZEOF_INT == NPY_SIZEOF_LONG
        #define MPL_OFF_T_PYFMT "l"
    #elif NPY_SIZEOF_INT == NPY_SIZEOF_LONGLONG
        #define MPL_OFF_T_PYFMT "L"
    #else
        #error Unsupported size for type off_t
    #endif
#endif

/*
 * PyFile_* compatibility
 */

/*
 * Get a FILE* handle to the file represented by the Python object
 */
static NPY_INLINE FILE *mpl_PyFile_Dup(PyObject *file, char *mode, mpl_off_t *orig_pos)
{
    int fd, fd2;
    PyObject *ret, *os;
    mpl_off_t pos;
    FILE *handle;

    if (mode[0] != 'r') {
        /* Flush first to ensure things end up in the file in the correct order */
        ret = PyObject_CallMethod(file, (char *)"flush", (char *)"");
        if (ret == NULL) {
            return NULL;
        }
        Py_DECREF(ret);
    }

    fd = PyObject_AsFileDescriptor(file);
    if (fd == -1) {
        return NULL;
    }

    /* The handle needs to be dup'd because we have to call fclose
       at the end */
    os = PyImport_ImportModule("os");
    if (os == NULL) {
        return NULL;
    }
    ret = PyObject_CallMethod(os, (char *)"dup", (char *)"i", fd);
    Py_DECREF(os);
    if (ret == NULL) {
        return NULL;
    }
    fd2 = (int)PyNumber_AsSsize_t(ret, NULL);
    Py_DECREF(ret);

/* Convert to FILE* handle */
#ifdef _WIN32
    handle = _fdopen(fd2, mode);
#else
    handle = fdopen(fd2, mode);
#endif
    if (handle == NULL) {
        PyErr_SetString(PyExc_IOError, "Getting a FILE* from a Python file object failed");
        return NULL;
    }

    /* Record the original raw file handle position */
    *orig_pos = mpl_ftell(handle);
    if (*orig_pos == -1) {
        // handle is a stream, so we don't have to worry about this
        return handle;
    }

    /* Seek raw handle to the Python-side position */
    ret = PyObject_CallMethod(file, (char *)"tell", (char *)"");
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
    if (mpl_fseek(handle, pos, SEEK_SET) == -1) {
        PyErr_SetString(PyExc_IOError, "seeking file failed");
        return NULL;
    }
    return handle;
}

/*
 * Close the dup-ed file handle, and seek the Python one to the current position
 */
static NPY_INLINE int mpl_PyFile_DupClose(PyObject *file, FILE *handle, mpl_off_t orig_pos)
{
    PyObject *exc_type = NULL, *exc_value = NULL, *exc_tb = NULL;
    PyErr_Fetch(&exc_type, &exc_value, &exc_tb);

    int fd;
    PyObject *ret;
    mpl_off_t position;

    position = mpl_ftell(handle);

    /* Close the FILE* handle */
    fclose(handle);

    /* Restore original file handle position, in order to not confuse
       Python-side data structures.  Note that this would fail if an exception
       is currently set, which can happen as this function is called in cleanup
       code, so we need to carefully fetch and restore the exception state. */
    fd = PyObject_AsFileDescriptor(file);
    if (fd == -1) {
        goto fail;
    }
    if (mpl_lseek(fd, orig_pos, SEEK_SET) != -1) {
        if (position == -1) {
            PyErr_SetString(PyExc_IOError, "obtaining file position failed");
            goto fail;
        }

        /* Seek Python-side handle to the FILE* handle position */
        ret = PyObject_CallMethod(file, (char *)"seek", (char *)(MPL_OFF_T_PYFMT "i"), position, 0);
        if (ret == NULL) {
            goto fail;
        }
        Py_DECREF(ret);
    }
    PyErr_Restore(exc_type, exc_value, exc_tb);
    return 0;
fail:
    Py_XDECREF(exc_type);
    Py_XDECREF(exc_value);
    Py_XDECREF(exc_tb);
    return -1;
}

static NPY_INLINE int mpl_PyFile_Check(PyObject *file)
{
    int fd;
    fd = PyObject_AsFileDescriptor(file);
    if (fd == -1) {
        PyErr_Clear();
        return 0;
    }
    return 1;
}

static NPY_INLINE PyObject *mpl_PyFile_OpenFile(PyObject *filename, const char *mode)
{
    PyObject *open;
    open = PyDict_GetItemString(PyEval_GetBuiltins(), "open");
    if (open == NULL) {
        return NULL;
    }
    return PyObject_CallFunction(open, (char *)"Os", filename, mode);
}

static NPY_INLINE int mpl_PyFile_CloseFile(PyObject *file)
{
    PyObject *type, *value, *tb;
    PyErr_Fetch(&type, &value, &tb);

    PyObject *ret;

    ret = PyObject_CallMethod(file, (char *)"close", NULL);
    if (ret == NULL) {
        goto fail;
    }
    Py_DECREF(ret);
    PyErr_Restore(type, value, tb);
    return 0;
fail:
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(tb);
    return -1;
}

#ifdef __cplusplus
}
#endif

#endif /* ifndef MPL_FILE_COMPAT_H */
