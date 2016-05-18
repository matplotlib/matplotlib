/* -*- mode: c++; c-basic-offset: 4 -*- */

/*
 * The Python Imaging Library.
 * $Id$
 *
 */

/* This is needed for (at least) Tk 8.4.1, otherwise the signature of
** Tk_PhotoPutBlock changes.
*/
#define USE_COMPOSITELESS_PHOTO_PUT_BLOCK

#include <Python.h>
#include <cstdlib>
#include <cstdio>
#include <sstream>

#include "py_converters.h"

extern "C"
{
#ifdef __APPLE__
#  ifdef TK_FRAMEWORK
#     include <Tcl/tcl.h>
#     include <Tk/tk.h>
#  else
#     include <tk.h>
#  endif
#else
#  include <tk.h>
#endif
}

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

// Load TCL / Tk symbols from tkinter extension module at run-time.
// Typedefs, global vars for TCL / Tk library functions.
typedef Tcl_Command (*tcl_cc)(Tcl_Interp *interp,
        const char *cmdName, Tcl_CmdProc *proc,
        ClientData clientData,
        Tcl_CmdDeleteProc *deleteProc);
static tcl_cc TCL_CREATE_COMMAND;
typedef void (*tcl_app_res) (Tcl_Interp *interp, ...);
static tcl_app_res TCL_APPEND_RESULT;
typedef Tk_Window (*tk_mw) (Tcl_Interp *interp);
static tk_mw TK_MAIN_WINDOW;
typedef Tk_PhotoHandle (*tk_fp) (Tcl_Interp *interp, const char *imageName);
static tk_fp TK_FIND_PHOTO;
typedef void (*tk_ppb_nc) (Tk_PhotoHandle handle,
        Tk_PhotoImageBlock *blockPtr, int x, int y,
        int width, int height);
static tk_ppb_nc TK_PHOTO_PUTBLOCK;
typedef void (*tk_pb) (Tk_PhotoHandle handle);
static tk_pb TK_PHOTO_BLANK;

static int PyAggImagePhoto(ClientData clientdata, Tcl_Interp *interp, int argc, char **argv)
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

        TK_PHOTO_PUTBLOCK(photo, &block, destx, desty, destwidth, destheight);
        delete[] destbuffer;

    } else {
        block.width = buffer.dim(1);
        block.height = buffer.dim(0);
        block.pitch = (int)block.width * nval;
        block.pixelPtr = buffer.data();

        /* Clear current contents */
        TK_PHOTO_BLANK(photo);
        /* Copy opaque block to photo image, and leave the rest to TK */
        TK_PHOTO_PUTBLOCK(photo, &block, 0, 0, block.width, block.height);
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

// Functions to fill global TCL / Tk function pointers from tkinter module.

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

#if defined(_MSC_VER)
#include <windows.h>
#define LIB_PTR_TYPE HMODULE
#define LOAD_LIB(name) LoadLibrary(name)
#define CLOSE_LIB(name) FreeLibrary(name)
FARPROC _dfunc(LIB_PTR_TYPE lib_handle, const char *func_name)
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
#else
#include <dlfcn.h>
#define LIB_PTR_TYPE void*
#define LOAD_LIB(name) dlopen(name, RTLD_LAZY)
#define CLOSE_LIB(name) dlclose(name)
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
#endif

int _func_loader(LIB_PTR_TYPE lib)
{
    // Fill global function pointers from dynamic lib.
    // Return 1 if any pointer is NULL, 0 otherwise.
    return (
         ((TCL_CREATE_COMMAND = (tcl_cc)
           _dfunc(lib, "Tcl_CreateCommand")) == NULL) ||
         ((TCL_APPEND_RESULT = (tcl_app_res)
           _dfunc(lib, "Tcl_AppendResult")) == NULL) ||
         ((TK_MAIN_WINDOW = (tk_mw)
           _dfunc(lib, "Tk_MainWindow")) == NULL) ||
         ((TK_FIND_PHOTO = (tk_fp)
           _dfunc(lib, "Tk_FindPhoto")) == NULL) ||
         ((TK_PHOTO_PUTBLOCK = (tk_ppb_nc)
           _dfunc(lib, "Tk_PhotoPutBlock_NoComposite")) == NULL) ||
         ((TK_PHOTO_BLANK = (tk_pb)
           _dfunc(lib, "Tk_PhotoBlank")) == NULL));
}

int load_tkinter_funcs(void)
{
    // Load tkinter global funcs from tkinter compiled module.
    // Return 0 for success, non-zero for failure.
    int ret = -1;
    LIB_PTR_TYPE tkinter_lib;
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
    tkinter_lib = LOAD_LIB(tkinter_libname);
    if (tkinter_lib == NULL) {
        PyErr_SetString(PyExc_RuntimeError,
                "Cannot dlopen tkinter module file");
        goto exit;
    }
    ret = _func_loader(tkinter_lib);
    // dlclose probably safe because tkinter has been imported.
    CLOSE_LIB(tkinter_lib);
exit:
    Py_XDECREF(pModule);
    Py_XDECREF(pSubmodule);
    Py_XDECREF(pString);
    return ret;
}

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
