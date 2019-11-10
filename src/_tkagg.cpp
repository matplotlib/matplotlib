/* -*- mode: c++; c-basic-offset: 4 -*- */

/*
 * This code is derived from The Python Imaging Library and is covered
 * by the PIL license.
 *
 * See LICENSE/LICENSE.PIL for details.
 *
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifdef _WIN32
#include <windows.h>
#define PSAPI_VERSION 1
#include <psapi.h>  // Must be linked with 'psapi' library
#define dlsym GetProcAddress
#else
#include <dlfcn.h>
#endif

// Include our own excerpts from the Tcl / Tk headers
#include "_tkmini.h"
#include "py_converters.h"

// Global vars for Tk functions.  We load these symbols from the tkinter
// extension module or loaded Tk libraries at run-time.
static Tk_FindPhoto_t TK_FIND_PHOTO;
static Tk_PhotoPutBlock_NoComposite_t TK_PHOTO_PUT_BLOCK_NO_COMPOSITE;

static PyObject *mpl_tk_blit(PyObject *self, PyObject *args)
{
    Tcl_Interp *interp;
    char const *photo_name;
    int height, width;
    unsigned char *data_ptr;
    int o0, o1, o2, o3;
    int x1, x2, y1, y2;
    Tk_PhotoHandle photo;
    Tk_PhotoImageBlock block;
    if (!PyArg_ParseTuple(args, "O&s(iiO&)(iiii)(iiii):blit",
                          convert_voidptr, &interp, &photo_name,
                          &height, &width, convert_voidptr, &data_ptr,
                          &o0, &o1, &o2, &o3,
                          &x1, &x2, &y1, &y2)) {
        goto exit;
    }
    if (!(photo = TK_FIND_PHOTO(interp, photo_name))) {
        PyErr_SetString(PyExc_ValueError, "Failed to extract Tk_PhotoHandle");
        goto exit;
    }
    if (0 > y1 || y1 > y2 || y2 > height || 0 > x1 || x1 > x2 || x2 > width) {
        PyErr_SetString(PyExc_ValueError, "Attempting to draw out of bounds");
        goto exit;
    }

    block.pixelPtr = data_ptr + 4 * ((height - y2) * width + x1);
    block.width = x2 - x1;
    block.height = y2 - y1;
    block.pitch = 4 * width;
    block.pixelSize = 4;
    block.offset[0] = o0;
    block.offset[1] = o1;
    block.offset[2] = o2;
    block.offset[3] = o3;
    TK_PHOTO_PUT_BLOCK_NO_COMPOSITE(
        photo, &block, x1, height - y2, x2 - x1, y2 - y1);
exit:
    if (PyErr_Occurred()) {
        return NULL;
    } else {
        Py_RETURN_NONE;
    }
}

#ifdef _WIN32
static PyObject *
Win32_GetForegroundWindow(PyObject *module, PyObject *args)
{
    HWND handle = GetForegroundWindow();
    if (!PyArg_ParseTuple(args, ":GetForegroundWindow")) {
        return NULL;
    }
    return PyLong_FromSize_t((size_t)handle);
}

static PyObject *
Win32_SetForegroundWindow(PyObject *module, PyObject *args)
{
    HWND handle;
    if (!PyArg_ParseTuple(args, "n:SetForegroundWindow", &handle)) {
        return NULL;
    }
    if (!SetForegroundWindow(handle)) {
        return PyErr_Format(PyExc_RuntimeError, "Error setting window");
    }
    Py_INCREF(Py_None);
    return Py_None;
}
#endif

static PyMethodDef functions[] = {
    { "blit", (PyCFunction)mpl_tk_blit, METH_VARARGS },
#ifdef _WIN32
    { "Win32_GetForegroundWindow", (PyCFunction)Win32_GetForegroundWindow, METH_VARARGS },
    { "Win32_SetForegroundWindow", (PyCFunction)Win32_SetForegroundWindow, METH_VARARGS },
#endif
    { NULL, NULL } /* sentinel */
};

// Functions to fill global Tk function pointers by dynamic loading

template <class T>
int load_tk(T lib)
{
    // Try to fill Tk global vars with function pointers.  Return the number of
    // functions found.
    return
        !!(TK_FIND_PHOTO =
            (Tk_FindPhoto_t)dlsym(lib, "Tk_FindPhoto")) +
        !!(TK_PHOTO_PUT_BLOCK_NO_COMPOSITE =
            (Tk_PhotoPutBlock_NoComposite_t)dlsym(lib, "Tk_PhotoPutBlock_NoComposite"));
}

#ifdef _WIN32

/*
 * On Windows, we can't load the tkinter module to get the Tk symbols, because
 * Windows does not load symbols into the library name-space of importing
 * modules. So, knowing that tkinter has already been imported by Python, we
 * scan all modules in the running process for the Tk function names.
 */

void load_tkinter_funcs(void)
{
    // Load Tk functions by searching all modules in current process.
    HMODULE hMods[1024];
    HANDLE hProcess;
    DWORD cbNeeded;
    unsigned int i;
    // Returns pseudo-handle that does not need to be closed
    hProcess = GetCurrentProcess();
    // Iterate through modules in this process looking for Tk names.
    if (EnumProcessModules(hProcess, hMods, sizeof(hMods), &cbNeeded)) {
        for (i = 0; i < (cbNeeded / sizeof(HMODULE)); i++) {
            if (load_tk(hMods[i])) {
                return;
            }
        }
    }
}

#else  // not Windows

/*
 * On Unix, we can get the Tk symbols from the tkinter module, because tkinter
 * uses these symbols, and the symbols are therefore visible in the tkinter
 * dynamic library (module).
 */

void load_tkinter_funcs(void)
{
    // Load tkinter global funcs from tkinter compiled module.
    void *main_program, *tkinter_lib;
    PyObject *module = NULL, *py_path = NULL, *py_path_b = NULL;
    char *path;

    // Try loading from the main program namespace first.
    main_program = dlopen(NULL, RTLD_LAZY);
    if (load_tk(main_program)) {
        goto exit;
    }
    // Clear exception triggered when we didn't find symbols above.
    PyErr_Clear();

    // Handle PyPy first, as that import will correctly fail on CPython.
    module = PyImport_ImportModule("_tkinter.tklib_cffi");   // PyPy
    if (!module) {
        PyErr_Clear();
        module = PyImport_ImportModule("_tkinter");  // CPython
    }
    if (!(module &&
          (py_path = PyObject_GetAttrString(module, "__file__")) &&
          (py_path_b = PyUnicode_EncodeFSDefault(py_path)) &&
          (path = PyBytes_AsString(py_path_b)))) {
        goto exit;
    }
    tkinter_lib = dlopen(path, RTLD_LAZY);
    if (!tkinter_lib) {
        PyErr_SetString(PyExc_RuntimeError, dlerror());
        goto exit;
    }
    load_tk(tkinter_lib);
    // dlclose is safe because tkinter has been imported.
    dlclose(tkinter_lib);
    goto exit;
exit:
    Py_XDECREF(module);
    Py_XDECREF(py_path);
    Py_XDECREF(py_path_b);
}
#endif // end not Windows

static PyModuleDef _tkagg_module = {
    PyModuleDef_HEAD_INIT, "_tkagg", "", -1, functions, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__tkagg(void)
{
    load_tkinter_funcs();
    if (PyErr_Occurred()) {
        return NULL;
    } else if (!TK_FIND_PHOTO) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to load Tk_FindPhoto");
        return NULL;
    } else if (!TK_PHOTO_PUT_BLOCK_NO_COMPOSITE) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to load Tk_PhotoPutBlock_NoComposite");
        return NULL;
    }
    return PyModule_Create(&_tkagg_module);
}
