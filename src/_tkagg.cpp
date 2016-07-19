/* -*- mode: c++; c-basic-offset: 4 -*- */

/*
 * The Python Imaging Library.
 * $Id$
 *
 */

#include <Python.h>
#include <cstdlib>
#include <cstdio>
#include <sstream>

#include "py_converters.h"

// Include our own excerpts from the Tcl / Tk headers
#include "_tkmini.h"

#if defined(_MSC_VER)
#  define SIZE_T_FORMAT "%Iu"
#else
#  define SIZE_T_FORMAT "%zu"
#endif

typedef struct
{
    PyObject_HEAD;
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
    PyObject *bufferobj;

    // vars for blitting
    PyObject *bboxo;

    size_t aggl, bboxl;
    bool has_bbox;
    uint8_t *destbuffer;
    int destx, desty, destwidth, destheight, deststride;
    //unsigned long tmp_ptr;

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
    /* get array (or object that can be converted to array) pointer */
    if (sscanf(argv[2], SIZE_T_FORMAT, &aggl) != 1) {
        TCL_APPEND_RESULT(interp, "error casting pointer", (char *)NULL);
        return TCL_ERROR;
    }
    bufferobj = (PyObject *)aggl;

    numpy::array_view<uint8_t, 3> buffer;
    try {
        buffer = numpy::array_view<uint8_t, 3>(bufferobj);
    } catch (...) {
        TCL_APPEND_RESULT(interp, "buffer is of wrong type", (char *)NULL);
        PyErr_Clear();
        return TCL_ERROR;
    }
    int srcheight = buffer.dim(0);

    /* XXX insert aggRenderer type check */

    /* get array mode (0=mono, 1=rgb, 2=rgba) */
    mode = atol(argv[3]);
    if ((mode != 0) && (mode != 1) && (mode != 2)) {
        TCL_APPEND_RESULT(interp, "illegal image mode", (char *)NULL);
        return TCL_ERROR;
    }

    /* check for bbox/blitting */
    if (sscanf(argv[4], SIZE_T_FORMAT, &bboxl) != 1) {
        TCL_APPEND_RESULT(interp, "error casting pointer", (char *)NULL);
        return TCL_ERROR;
    }
    bboxo = (PyObject *)bboxl;

    if (bboxo != NULL && bboxo != Py_None) {
        agg::rect_d rect;
        if (!convert_rect(bboxo, &rect)) {
            return TCL_ERROR;
        }

        has_bbox = true;

        destx = (int)rect.x1;
        desty = srcheight - (int)rect.y2;
        destwidth = (int)(rect.x2 - rect.x1);
        destheight = (int)(rect.y2 - rect.y1);
        deststride = 4 * destwidth;

        destbuffer = new agg::int8u[deststride * destheight];
        if (destbuffer == NULL) {
            TCL_APPEND_RESULT(interp, "could not allocate memory", (char *)NULL);
            return TCL_ERROR;
        }

        for (int i = 0; i < destheight; ++i) {
            memcpy(destbuffer + (deststride * i),
                   &buffer(i + desty, destx, 0),
                   deststride);
        }
    } else {
        has_bbox = false;
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
        block.width = buffer.dim(1);
        block.height = buffer.dim(0);
        block.pitch = (int)block.width * nval;
        block.pixelPtr = buffer.data();

        /* Clear current contents */
        TK_PHOTO_BLANK(photo);
        /* Copy opaque block to photo image, and leave the rest to TK */
        TK_PHOTO_PUT_BLOCK_NO_COMPOSITE(photo, &block, 0, 0, block.width,
                block.height);
    }

    return TCL_OK;
}

static PyObject *_pyobj_addr(PyObject *self, PyObject *args)
{
    PyObject *pyobj;
    if (!PyArg_ParseTuple(args, "O", &pyobj)) {
        return NULL;
    }
    return Py_BuildValue("n", (Py_ssize_t)pyobj);
}

static PyObject *_tkinit(PyObject *self, PyObject *args)
{
    Tcl_Interp *interp;
    TkappObject *app;

    Py_ssize_t arg;
    int is_interp;
    if (!PyArg_ParseTuple(args, "ni", &arg, &is_interp)) {
        return NULL;
    }

    if (is_interp) {
        interp = (Tcl_Interp *)arg;
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

static PyMethodDef functions[] = {
    /* Tkinter interface stuff */
    { "_pyobj_addr", (PyCFunction)_pyobj_addr, 1 }, { "tkinit", (PyCFunction)_tkinit, 1 },
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
    TCL_APPEND_RESULT = (Tcl_AppendResult_t) _dfunc(hMod,
            "Tcl_AppendResult");
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
    return ( // -1 if any remaining symbols are NULL
        ((TK_FIND_PHOTO = (Tk_FindPhoto_t)
          _dfunc(hMod, "Tk_FindPhoto")) == NULL) ||
        ((TK_PHOTO_PUT_BLOCK_NO_COMPOSITE = (Tk_PhotoPutBlock_NoComposite_t)
          _dfunc(hMod, "Tk_PhotoPutBlock_NoComposite")) == NULL) ||
        ((TK_PHOTO_BLANK = (Tk_PhotoBlank_t)
          _dfunc(hMod, "Tk_PhotoBlank")) == NULL))
        ? -1 : 1;
}

int load_tkinter_funcs(void)
{
    // Load TCL and Tk functions by searching all modules in current process.
    // Return 0 for success, non-zero for failure.

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
                    return 1;
                }
            }
            if (!found_tk) {
                found_tk = get_tk(hMods[i]);
                if (found_tk == -1) {
                    return 1;
                }
            }
            if (found_tcl && found_tk) {
                return 0;
            }
        }
    }

    if (found_tcl == 0) {
        PyErr_SetString(PyExc_RuntimeError, "Could not find TCL routines");
    } else {
        PyErr_SetString(PyExc_RuntimeError, "Could not find Tk routines");
    }
    return 1;
}

#else  // not Windows

/*
 * On Unix, we can get the TCL and Tk synbols from the tkinter module, because
 * tkinter uses these symbols, and the symbols are therefore visible in the
 * tkinter dynamic library (module).
 */
#if PY3K
#define TKINTER_PKG "tkinter"
#define TKINTER_MOD "_tkinter"
// From module __file__ attribute to char *string for dlopen.
char *fname2char(PyObject *fname)
{
    PyObject *bytes = PyUnicode_EncodeFSDefault(fname);
    if (bytes == NULL) {
        return NULL;
    }
    return PyBytes_AsString(bytes);
}
#else
#define TKINTER_PKG "Tkinter"
#define TKINTER_MOD "tkinter"
// From module __file__ attribute to char *string for dlopen
#define fname2char(s) (PyString_AsString(s))
#endif

#include <dlfcn.h>

void *_dfunc(void *lib_handle, const char *func_name)
{
    // Load function `func_name` from `lib_handle`.
    // Set Python exception if we can't find `func_name` in `lib_handle`.
    // Returns function pointer or NULL if not present.

    // Reset errors.
    dlerror();
    void *func = dlsym(lib_handle, func_name);
    if (func == NULL) {
        const char *error = dlerror();
        PyErr_SetString(PyExc_RuntimeError, error);
    }
    return func;
}

int _func_loader(void *lib)
{
    // Fill global function pointers from dynamic lib.
    // Return 1 if any pointer is NULL, 0 otherwise.
    return (
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
           _dfunc(lib, "Tk_PhotoBlank")) == NULL));
}

int load_tkinter_funcs(void)
{
    // Load tkinter global funcs from tkinter compiled module.
    // Return 0 for success, non-zero for failure.
    int ret = -1;
    void *tkinter_lib;
    char *tkinter_libname;
    PyObject *pModule = NULL, *pSubmodule = NULL, *pString = NULL;

    pModule = PyImport_ImportModule(TKINTER_PKG);
    if (pModule == NULL) {
        goto exit;
    }
    pSubmodule = PyObject_GetAttrString(pModule, TKINTER_MOD);
    if (pSubmodule == NULL) {
        goto exit;
    }
    pString = PyObject_GetAttrString(pSubmodule, "__file__");
    if (pString == NULL) {
        goto exit;
    }
    tkinter_libname = fname2char(pString);
    if (tkinter_libname == NULL) {
        goto exit;
    }
    tkinter_lib = dlopen(tkinter_libname, RTLD_LAZY);
    if (tkinter_lib == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "Cannot dlopen tkinter module file");
        goto exit;
    }
    ret = _func_loader(tkinter_lib);
    // dlclose probably safe because tkinter has been imported.
    dlclose(tkinter_lib);
exit:
    Py_XDECREF(pModule);
    Py_XDECREF(pSubmodule);
    Py_XDECREF(pString);
    return ret;
}
#endif // end not Windows

#if PY3K
static PyModuleDef _tkagg_module = { PyModuleDef_HEAD_INIT, "_tkagg", "",   -1,  functions,
                                     NULL,                  NULL,     NULL, NULL };

PyMODINIT_FUNC PyInit__tkagg(void)
{
    PyObject *m;

    m = PyModule_Create(&_tkagg_module);

    import_array();

    return (load_tkinter_funcs() == 0) ? m : NULL;
}
#else
PyMODINIT_FUNC init_tkagg(void)
{
    import_array();

    Py_InitModule("_tkagg", functions);

    load_tkinter_funcs();
}
#endif
