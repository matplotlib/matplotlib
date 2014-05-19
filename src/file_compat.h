#ifndef __FILE_COMPAT_H__
#define __FILE_COMPAT_H__

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
    #define npy_fseek _fseeki64
    #define npy_ftell _ftelli64
    #define npy_lseek _lseeki64
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
    #define npy_fseek fseek
    #define npy_ftell ftell
    #define npy_lseek lseek
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
#if PY3K

/*
 * Get a FILE* handle to the file represented by the Python object
 */
static NPY_INLINE FILE*
mpl_PyFile_Dup(PyObject *file, char *mode, mpl_off_t *orig_pos)
{
    int fd, fd2;
    PyObject *ret, *os;
    mpl_off_t pos;
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

    /* The handle needs to be dup'd because we have to call fclose
       at the end */
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

    /* Convert to FILE* handle */
#ifdef _WIN32
    handle = _fdopen(fd2, mode);
#else
    handle = fdopen(fd2, mode);
#endif
    if (handle == NULL) {
        PyErr_SetString(PyExc_IOError,
                        "Getting a FILE* from a Python file object failed");
    }

    /* Record the original raw file handle position */
    *orig_pos = npy_ftell(handle);
    if (*orig_pos == -1) {
        // handle is a stream, so we don't have to worry about this
        return handle;
    }

    /* Seek raw handle to the Python-side position */
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
    if (npy_fseek(handle, pos, SEEK_SET) == -1) {
        PyErr_SetString(PyExc_IOError, "seeking file failed");
        return NULL;
    }
    return handle;
}

/*
 * Close the dup-ed file handle, and seek the Python one to the current position
 */
static NPY_INLINE int
mpl_PyFile_DupClose(PyObject *file, FILE* handle, mpl_off_t orig_pos)
{
    int fd;
    PyObject *ret;
    mpl_off_t position;

    position = npy_ftell(handle);

    /* Close the FILE* handle */
    fclose(handle);

    /* Restore original file handle position, in order to not confuse
       Python-side data structures */
    fd = PyObject_AsFileDescriptor(file);
    if (fd == -1) {
        return -1;
    }
    if (npy_lseek(fd, orig_pos, SEEK_SET) != -1) {
        if (position == -1) {
            PyErr_SetString(PyExc_IOError, "obtaining file position failed");
            return -1;
        }

        /* Seek Python-side handle to the FILE* handle position */
        ret = PyObject_CallMethod(file, "seek", MPL_OFF_T_PYFMT "i", position, 0);
        if (ret == NULL) {
            return -1;
        }
        Py_DECREF(ret);
    }
    return 0;
}

static NPY_INLINE int
mpl_PyFile_Check(PyObject *file)
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

#define mpl_PyFile_Dup(file, mode, orig_pos_p) PyFile_AsFile(file)
#define mpl_PyFile_DupClose(file, handle, orig_pos) (0)
#define mpl_PyFile_Check PyFile_Check

#endif

static NPY_INLINE PyObject*
mpl_PyFile_OpenFile(PyObject *filename, const char *mode)
{
    PyObject *open;
    open = PyDict_GetItemString(PyEval_GetBuiltins(), "open");
    if (open == NULL) {
        return NULL;
    }
    return PyObject_CallFunction(open, (char*)"Os", filename, mode);
}

static NPY_INLINE int
mpl_PyFile_CloseFile(PyObject *file)
{
    PyObject *ret;

    ret = PyObject_CallMethod(file, (char*)"close", NULL);
    if (ret == NULL) {
        return -1;
    }
    Py_DECREF(ret);
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* ifndef __FILE_COMPAT_H__ */
