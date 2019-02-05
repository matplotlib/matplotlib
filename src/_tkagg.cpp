/* -*- mode: c++; c-basic-offset: 4 -*- */

/*
 * This code is derived from The Python Imaging Library and is covered
 * by the PIL license.
 *
 * See LICENSE/LICENSE.PIL for details.
 *
 */

#include <Python.h>
#include <cstdlib>
#include <cstdio>
#include <sstream>

#include <agg_basics.h> // agg:int8u

// Include our own excerpts from the Tcl / Tk headers
#include "_tkmini.h"

#include "py_converters.h"

#if defined(_MSC_VER)
#  define IMG_FORMAT "%d %d %Iu"
#else
#  define IMG_FORMAT "%d %d %zu"
#endif
#define BBOX_FORMAT "%f %f %f %f"

typedef struct
{
    PyObject_HEAD
    Tcl_Interp *interp;
} TkappObject;

// Global vars for Tcl / Tk functions.  We load these symbols from the tkinter
// extension module or loaded Tcl / Tk libraries at run-time.
static Tcl_CreateCommand_t TCL_CREATE_COMMAND;
static Tcl_AppendResult_t TCL_APPEND_RESULT;
static Tk_MainWindow_t TK_MAIN_WINDOW;
static Tk_FindPhoto_t TK_FIND_PHOTO;
static Tk_PhotoPutBlock_NoComposite_t TK_PHOTO_PUT_BLOCK_NO_COMPOSITE;
static Tk_PhotoBlank_t TK_PHOTO_BLANK;

static int PyAggImagePhoto(ClientData clientdata, Tcl_Interp *interp, int
        argc, char **argv)
{
    Tk_PhotoHandle photo;
    Tk_PhotoImageBlock block;

    // vars for blitting

    size_t pdata;
    int wdata, hdata, bbox_parse;
    float x1, x2, y1, y2;
    bool has_bbox;
    agg::int8u *destbuffer, *buffer;
    int destx, desty, destwidth, destheight, deststride;

    long mode;
    long nval;
    if (TK_MAIN_WINDOW(interp) == NULL) {
        // Will throw a _tkinter.TclError with "this isn't a Tk application"
        return TCL_ERROR;
    }

    if (argc != 5) {
        TCL_APPEND_RESULT(interp, "usage: ", argv[0], " destPhoto srcImage", (char *)NULL);
        return TCL_ERROR;
    }

    /* get Tcl PhotoImage handle */
    photo = TK_FIND_PHOTO(interp, argv[1]);
    if (photo == NULL) {
        TCL_APPEND_RESULT(interp, "destination photo must exist", (char *)NULL);
        return TCL_ERROR;
    }
    /* get buffer from str which is "height width ptr" */
    if (sscanf(argv[2], IMG_FORMAT, &hdata, &wdata, &pdata) != 3) {
        TCL_APPEND_RESULT(interp,
                          "error reading data, expected height width ptr",
                          (char *)NULL);
        return TCL_ERROR;
    }
    buffer = (agg::int8u*)pdata;

    /* get array mode (0=mono, 1=rgb, 2=rgba) */
    mode = atol(argv[3]);
    if ((mode != 0) && (mode != 1) && (mode != 2)) {
        TCL_APPEND_RESULT(interp, "illegal image mode", (char *)NULL);
        return TCL_ERROR;
    }

    /* check for bbox/blitting */
    bbox_parse = sscanf(argv[4], BBOX_FORMAT, &x1, &x2, &y1, &y2);
    if (bbox_parse == 4) {
        has_bbox = true;
    }
    else if ((bbox_parse == 1) && (x1 == 0)){
        has_bbox = false;
    } else {
        TCL_APPEND_RESULT(interp, "illegal bbox", (char *)NULL);
        return TCL_ERROR;
    }

    if (has_bbox) {
        int srcstride = wdata * 4;
        destx = (int)x1;
        desty = (int)(hdata - y2);
        destwidth = (int)(x2 - x1);
        destheight = (int)(y2 - y1);
        deststride = 4 * destwidth;

        try {
            destbuffer = new agg::int8u[deststride * destheight];
        }
        catch (const std::bad_alloc &) {
            TCL_APPEND_RESULT(interp, "could not allocate memory", (char *)NULL);
            return TCL_ERROR;
        }

        for (int i = 0; i < destheight; ++i) {
            memcpy(destbuffer + (deststride * i),
                   &buffer[(i + desty) * srcstride + (destx * 4)],
                   deststride);
        }
    } else {
        destbuffer = NULL;
        destx = desty = destwidth = destheight = deststride = 0;
    }

    /* setup tkblock */
    block.pixelSize = 1;
    if (mode == 0) {
        block.offset[0] = block.offset[1] = block.offset[2] = 0;
        nval = 1;
    } else {
        block.offset[0] = 0;
        block.offset[1] = 1;
        block.offset[2] = 2;
        if (mode == 1) {
            block.offset[3] = 0;
            block.pixelSize = 3;
            nval = 3;
        } else {
            block.offset[3] = 3;
            block.pixelSize = 4;
            nval = 4;
        }
    }

    if (has_bbox) {
        block.width = destwidth;
        block.height = destheight;
        block.pitch = deststride;
        block.pixelPtr = destbuffer;

        TK_PHOTO_PUT_BLOCK_NO_COMPOSITE(photo, &block, destx, desty,
                destwidth, destheight);
        delete[] destbuffer;

    } else {
        block.width = wdata;
        block.height = hdata;
        block.pitch = (int)block.width * nval;
        block.pixelPtr = buffer;

        /* Clear current contents */
        TK_PHOTO_BLANK(photo);
        /* Copy opaque block to photo image, and leave the rest to TK */
        TK_PHOTO_PUT_BLOCK_NO_COMPOSITE(photo, &block, 0, 0, block.width,
                block.height);
    }

    return TCL_OK;
}

static PyObject *_tkinit(PyObject *self, PyObject *args)
{
    Tcl_Interp *interp;
    TkappObject *app;

    PyObject *arg;
    int is_interp;
    if (!PyArg_ParseTuple(args, "Oi", &arg, &is_interp)) {
        return NULL;
    }

    if (is_interp) {
        interp = (Tcl_Interp *)PyLong_AsVoidPtr(arg);
    } else {
        /* Do it the hard way.  This will break if the TkappObject
           layout changes */
        app = (TkappObject *)arg;
        interp = app->interp;
    }

    /* This will bomb if interp is invalid... */

    TCL_CREATE_COMMAND(interp,
                       "PyAggImagePhoto",
                       (Tcl_CmdProc *)PyAggImagePhoto,
                       (ClientData)0,
                       (Tcl_CmdDeleteProc *)NULL);

    Py_INCREF(Py_None);
    return Py_None;
}

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

static PyMethodDef functions[] = {
    /* Tkinter interface stuff */
    { "tkinit", (PyCFunction)_tkinit, 1 },
    { "blit", (PyCFunction)mpl_tk_blit, 1 },
    { NULL, NULL } /* sentinel */
};

// Functions to fill global TCL / Tk function pointers by dynamic loading
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)

/*
 * On Windows, we can't load the tkinter module to get the TCL or Tk symbols,
 * because Windows does not load symbols into the library name-space of
 * importing modules. So, knowing that tkinter has already been imported by
 * Python, we scan all modules in the running process for the TCL and Tk
 * function names.
 */
#include <windows.h>
#define PSAPI_VERSION 1
#include <psapi.h>
// Must be linked with 'psapi' library

FARPROC _dfunc(HMODULE lib_handle, const char *func_name)
{
    // Load function `func_name` from `lib_handle`.
    // Set Python exception if we can't find `func_name` in `lib_handle`.
    // Returns function pointer or NULL if not present.

    char message[100];

    FARPROC func = GetProcAddress(lib_handle, func_name);
    if (func == NULL) {
        sprintf(message, "Cannot load function %s", func_name);
        PyErr_SetString(PyExc_RuntimeError, message);
    }
    return func;
}

int get_tcl(HMODULE hMod)
{
    // Try to fill TCL global vars with function pointers. Return 0 for no
    // functions found, 1 for all functions found, -1 for some but not all
    // functions found.
    TCL_CREATE_COMMAND = (Tcl_CreateCommand_t)
        GetProcAddress(hMod, "Tcl_CreateCommand");
    if (TCL_CREATE_COMMAND == NULL) {  // Maybe not TCL module
        return 0;
    }
    TCL_APPEND_RESULT = (Tcl_AppendResult_t) _dfunc(hMod, "Tcl_AppendResult");
    return (TCL_APPEND_RESULT == NULL) ? -1 : 1;
}

int get_tk(HMODULE hMod)
{
    // Try to fill Tk global vars with function pointers. Return 0 for no
    // functions found, 1 for all functions found, -1 for some but not all
    // functions found.
    TK_MAIN_WINDOW = (Tk_MainWindow_t)
        GetProcAddress(hMod, "Tk_MainWindow");
    if (TK_MAIN_WINDOW == NULL) {  // Maybe not Tk module
        return 0;
    }
    return  // -1 if any remaining symbols are NULL
        ((TK_FIND_PHOTO = (Tk_FindPhoto_t)
          _dfunc(hMod, "Tk_FindPhoto")) == NULL) ||
        ((TK_PHOTO_PUT_BLOCK_NO_COMPOSITE = (Tk_PhotoPutBlock_NoComposite_t)
          _dfunc(hMod, "Tk_PhotoPutBlock_NoComposite")) == NULL) ||
        ((TK_PHOTO_BLANK = (Tk_PhotoBlank_t)
          _dfunc(hMod, "Tk_PhotoBlank")) == NULL)
        ? -1 : 1;
}

void load_tkinter_funcs(void)
{
    // Load TCL and Tk functions by searching all modules in current process.
    // Sets an error on failure.

    HMODULE hMods[1024];
    HANDLE hProcess;
    DWORD cbNeeded;
    unsigned int i;
    int found_tcl = 0;
    int found_tk = 0;

    // Returns pseudo-handle that does not need to be closed
    hProcess = GetCurrentProcess();

    // Iterate through modules in this process looking for TCL / Tk names
    if (EnumProcessModules(hProcess, hMods, sizeof(hMods), &cbNeeded)) {
        for (i = 0; i < (cbNeeded / sizeof(HMODULE)); i++) {
            if (!found_tcl) {
                found_tcl = get_tcl(hMods[i]);
                if (found_tcl == -1) {
                    return;
                }
            }
            if (!found_tk) {
                found_tk = get_tk(hMods[i]);
                if (found_tk == -1) {
                    return;
                }
            }
            if (found_tcl && found_tk) {
                return;
            }
        }
    }

    if (found_tcl == 0) {
        PyErr_SetString(PyExc_RuntimeError, "Could not find TCL routines");
    } else {
        PyErr_SetString(PyExc_RuntimeError, "Could not find Tk routines");
    }
    return;
}

#else  // not Windows

/*
 * On Unix, we can get the TCL and Tk symbols from the tkinter module, because
 * tkinter uses these symbols, and the symbols are therefore visible in the
 * tkinter dynamic library (module).
 */

#include <dlfcn.h>

void *_dfunc(void *lib_handle, const char *func_name)
{
    // Load function `func_name` from `lib_handle`.
    // Set Python exception if we can't find `func_name` in `lib_handle`.
    // Returns function pointer or NULL if not present.

    void* func;
    // Reset errors.
    dlerror();
    func = dlsym(lib_handle, func_name);
    if (func == NULL) {
        PyErr_SetString(PyExc_RuntimeError, dlerror());
    }
    return func;
}

int _func_loader(void *lib)
{
    // Fill global function pointers from dynamic lib.
    // Return 1 if any pointer is NULL, 0 otherwise.
    return
         ((TCL_CREATE_COMMAND = (Tcl_CreateCommand_t)
           _dfunc(lib, "Tcl_CreateCommand")) == NULL) ||
         ((TCL_APPEND_RESULT = (Tcl_AppendResult_t)
           _dfunc(lib, "Tcl_AppendResult")) == NULL) ||
         ((TK_MAIN_WINDOW = (Tk_MainWindow_t)
           _dfunc(lib, "Tk_MainWindow")) == NULL) ||
         ((TK_FIND_PHOTO = (Tk_FindPhoto_t)
           _dfunc(lib, "Tk_FindPhoto")) == NULL) ||
         ((TK_PHOTO_PUT_BLOCK_NO_COMPOSITE = (Tk_PhotoPutBlock_NoComposite_t)
           _dfunc(lib, "Tk_PhotoPutBlock_NoComposite")) == NULL) ||
         ((TK_PHOTO_BLANK = (Tk_PhotoBlank_t)
           _dfunc(lib, "Tk_PhotoBlank")) == NULL);
}

void load_tkinter_funcs(void)
{
    // Load tkinter global funcs from tkinter compiled module.
    // Sets an error on failure.
    void *main_program, *tkinter_lib;
    PyObject *module = NULL, *py_path = NULL, *py_path_b = NULL;
    char *path;

    // Try loading from the main program namespace first.
    main_program = dlopen(NULL, RTLD_LAZY);
    if (_func_loader(main_program) == 0) {
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
    _func_loader(tkinter_lib);
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
    return PyErr_Occurred() ? NULL : PyModule_Create(&_tkagg_module);
}
