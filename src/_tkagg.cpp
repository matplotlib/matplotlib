/* -*- mode: c++; c-basic-offset: 4 -*- */

// Where is PIL?
//
// Many years ago, Matplotlib used to include code from PIL (the Python Imaging
// Library).  Since then, the code has changed a lot - the organizing principle
// and methods of operation are now quite different.  Because our review of
// the codebase showed that all the code that came from PIL was removed or
// rewritten, we have removed the PIL licensing information.  If you want PIL,
// you can get it at https://python-pillow.org/

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
// Windows 8.1
#define WINVER 0x0603
#define _WIN32_WINNT 0x0603
#endif

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifdef _WIN32
#define WIN32_DLL
#endif
#ifdef __CYGWIN__
/*
 * Unfortunately cygwin's libdl inherits restrictions from the underlying
 * Windows OS, at least currently. Therefore, a symbol may be loaded from a
 * module by dlsym() only if it is really located in the given module,
 * dependencies are not included. So we have to use native WinAPI on Cygwin
 * also.
 */
#define WIN32_DLL
static inline PyObject *PyErr_SetFromWindowsErr(int ierr) {
    PyErr_SetString(PyExc_OSError, "Call to EnumProcessModules failed");
    return NULL;
}
#endif

#ifdef WIN32_DLL
#include <string>
#include <windows.h>
#include <commctrl.h>
#define PSAPI_VERSION 1
#include <psapi.h>  // Must be linked with 'psapi' library
#define dlsym GetProcAddress
#else
#include <dlfcn.h>
#endif

// Include our own excerpts from the Tcl / Tk headers
#include "_tkmini.h"

static int convert_voidptr(PyObject *obj, void *p)
{
    void **val = (void **)p;
    *val = PyLong_AsVoidPtr(obj);
    return *val != NULL ? 1 : !PyErr_Occurred();
}

// Global vars for Tk functions.  We load these symbols from the tkinter
// extension module or loaded Tk libraries at run-time.
static Tk_FindPhoto_t TK_FIND_PHOTO;
static Tk_PhotoPutBlock_t TK_PHOTO_PUT_BLOCK;
// Global vars for Tcl functions.  We load these symbols from the tkinter
// extension module or loaded Tcl libraries at run-time.
static Tcl_SetVar_t TCL_SETVAR;

static PyObject *mpl_tk_blit(PyObject *self, PyObject *args)
{
    Tcl_Interp *interp;
    char const *photo_name;
    int height, width;
    unsigned char *data_ptr;
    int comp_rule;
    int put_retval;
    int o0, o1, o2, o3;
    int x1, x2, y1, y2;
    Tk_PhotoHandle photo;
    Tk_PhotoImageBlock block;
    if (!PyArg_ParseTuple(args, "O&s(iiO&)i(iiii)(iiii):blit",
                          convert_voidptr, &interp, &photo_name,
                          &height, &width, convert_voidptr, &data_ptr,
                          &comp_rule,
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
    if (comp_rule != TK_PHOTO_COMPOSITE_OVERLAY && comp_rule != TK_PHOTO_COMPOSITE_SET) {
        PyErr_SetString(PyExc_ValueError, "Invalid comp_rule argument");
        goto exit;
    }

    Py_BEGIN_ALLOW_THREADS
    block.pixelPtr = data_ptr + 4 * ((height - y2) * width + x1);
    block.width = x2 - x1;
    block.height = y2 - y1;
    block.pitch = 4 * width;
    block.pixelSize = 4;
    block.offset[0] = o0;
    block.offset[1] = o1;
    block.offset[2] = o2;
    block.offset[3] = o3;
    put_retval = TK_PHOTO_PUT_BLOCK(
        interp, photo, &block, x1, height - y2, x2 - x1, y2 - y1, comp_rule);
    Py_END_ALLOW_THREADS
    if (put_retval == TCL_ERROR) {
        return PyErr_NoMemory();
    }

exit:
    if (PyErr_Occurred()) {
        return NULL;
    } else {
        Py_RETURN_NONE;
    }
}

#ifdef WIN32_DLL
LRESULT CALLBACK
DpiSubclassProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam,
                UINT_PTR uIdSubclass, DWORD_PTR dwRefData)
{
    switch (uMsg) {
    case WM_DPICHANGED:
        // This function is a subclassed window procedure, and so is run during
        // the Tcl/Tk event loop. Unfortunately, Tkinter has a *second* lock on
        // Tcl threading that is not exposed publicly, but is currently taken
        // while we're in the window procedure. So while we can take the GIL to
        // call Python code, we must not also call *any* Tk code from Python.
        // So stay with Tcl calls in C only.
        {
            // This variable naming must match the name used in
            // lib/matplotlib/backends/_backend_tk.py:FigureManagerTk.
            std::string var_name("window_dpi");
            var_name += std::to_string((unsigned long long)hwnd);

            // X is high word, Y is low word, but they are always equal.
            std::string dpi = std::to_string(LOWORD(wParam));

            Tcl_Interp* interp = (Tcl_Interp*)dwRefData;
            TCL_SETVAR(interp, var_name.c_str(), dpi.c_str(), 0);
        }
        return 0;
    case WM_NCDESTROY:
        RemoveWindowSubclass(hwnd, DpiSubclassProc, uIdSubclass);
        break;
    }

    return DefSubclassProc(hwnd, uMsg, wParam, lParam);
}
#endif

static PyObject*
mpl_tk_enable_dpi_awareness(PyObject* self, PyObject*const* args,
                            Py_ssize_t nargs)
{
    if (nargs != 2) {
        return PyErr_Format(PyExc_TypeError,
                            "enable_dpi_awareness() takes 2 positional "
                            "arguments but %zd were given",
                            nargs);
    }

#ifdef WIN32_DLL
    HWND frame_handle = NULL;
    Tcl_Interp *interp = NULL;

    if (!convert_voidptr(args[0], &frame_handle)) {
        return NULL;
    }
    if (!convert_voidptr(args[1], &interp)) {
        return NULL;
    }

#ifdef _DPI_AWARENESS_CONTEXTS_
    HMODULE user32 = LoadLibrary("user32.dll");

    typedef DPI_AWARENESS_CONTEXT (WINAPI *GetWindowDpiAwarenessContext_t)(HWND);
    GetWindowDpiAwarenessContext_t GetWindowDpiAwarenessContextPtr =
        (GetWindowDpiAwarenessContext_t)GetProcAddress(
            user32, "GetWindowDpiAwarenessContext");
    if (GetWindowDpiAwarenessContextPtr == NULL) {
        FreeLibrary(user32);
        Py_RETURN_FALSE;
    }

    typedef BOOL (WINAPI *AreDpiAwarenessContextsEqual_t)(DPI_AWARENESS_CONTEXT,
                                                          DPI_AWARENESS_CONTEXT);
    AreDpiAwarenessContextsEqual_t AreDpiAwarenessContextsEqualPtr =
        (AreDpiAwarenessContextsEqual_t)GetProcAddress(
            user32, "AreDpiAwarenessContextsEqual");
    if (AreDpiAwarenessContextsEqualPtr == NULL) {
        FreeLibrary(user32);
        Py_RETURN_FALSE;
    }

    DPI_AWARENESS_CONTEXT ctx = GetWindowDpiAwarenessContextPtr(frame_handle);
    bool per_monitor = (
        AreDpiAwarenessContextsEqualPtr(
            ctx, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) ||
        AreDpiAwarenessContextsEqualPtr(
            ctx, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE));

    if (per_monitor) {
        // Per monitor aware means we need to handle WM_DPICHANGED by wrapping
        // the Window Procedure, and the Python side needs to trace the Tk
        // window_dpi variable stored on interp.
        SetWindowSubclass(frame_handle, DpiSubclassProc, 0, (DWORD_PTR)interp);
    }
    FreeLibrary(user32);
    return PyBool_FromLong(per_monitor);
#endif
#endif

    Py_RETURN_NONE;
}

static PyMethodDef functions[] = {
    { "blit", (PyCFunction)mpl_tk_blit, METH_VARARGS },
    { "enable_dpi_awareness", (PyCFunction)mpl_tk_enable_dpi_awareness,
      METH_FASTCALL },
    { NULL, NULL } /* sentinel */
};

// Functions to fill global Tcl/Tk function pointers by dynamic loading.

template <class T>
bool load_tcl_tk(T lib)
{
    // Try to fill Tcl/Tk global vars with function pointers.  Return whether
    // all of them have been filled.
    if (auto ptr = dlsym(lib, "Tcl_SetVar")) {
        TCL_SETVAR = (Tcl_SetVar_t)ptr;
    }
    if (auto ptr = dlsym(lib, "Tk_FindPhoto")) {
        TK_FIND_PHOTO = (Tk_FindPhoto_t)ptr;
    }
    if (auto ptr = dlsym(lib, "Tk_PhotoPutBlock")) {
        TK_PHOTO_PUT_BLOCK = (Tk_PhotoPutBlock_t)ptr;
    }
    return TCL_SETVAR && TK_FIND_PHOTO && TK_PHOTO_PUT_BLOCK;
}

#ifdef WIN32_DLL

/* On Windows, we can't load the tkinter module to get the Tcl/Tk symbols,
 * because Windows does not load symbols into the library name-space of
 * importing modules. So, knowing that tkinter has already been imported by
 * Python, we scan all modules in the running process for the Tcl/Tk function
 * names.
 */

void load_tkinter_funcs(void)
{
    HANDLE process = GetCurrentProcess();  // Pseudo-handle, doesn't need closing.
    HMODULE* modules = NULL;
    DWORD size;
    if (!EnumProcessModules(process, NULL, 0, &size)) {
        PyErr_SetFromWindowsErr(0);
        goto exit;
    }
    if (!(modules = static_cast<HMODULE*>(malloc(size)))) {
        PyErr_NoMemory();
        goto exit;
    }
    if (!EnumProcessModules(process, modules, size, &size)) {
        PyErr_SetFromWindowsErr(0);
        goto exit;
    }
    for (unsigned i = 0; i < size / sizeof(HMODULE); ++i) {
        if (load_tcl_tk(modules[i])) {
            return;
        }
    }
exit:
    free(modules);
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
    void *main_program = NULL, *tkinter_lib = NULL;
    PyObject *module = NULL, *py_path = NULL, *py_path_b = NULL;
    char *path;

    // Try loading from the main program namespace first.
    main_program = dlopen(NULL, RTLD_LAZY);
    if (load_tcl_tk(main_program)) {
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
    if (load_tcl_tk(tkinter_lib)) {
        goto exit;
    }

exit:
    // We don't need to keep a reference open as the main program & tkinter
    // have been imported.  Try to close each library separately (otherwise the
    // second dlclose could clear a dlerror from the first dlclose).
    bool raised_dlerror = false;
    if (main_program && dlclose(main_program) && !raised_dlerror) {
        PyErr_SetString(PyExc_RuntimeError, dlerror());
        raised_dlerror = true;
    }
    if (tkinter_lib && dlclose(tkinter_lib) && !raised_dlerror) {
        PyErr_SetString(PyExc_RuntimeError, dlerror());
        raised_dlerror = true;
    }
    Py_XDECREF(module);
    Py_XDECREF(py_path);
    Py_XDECREF(py_path_b);
}
#endif // end not Windows

static PyModuleDef _tkagg_module = {
    PyModuleDef_HEAD_INIT, "_tkagg", NULL, -1, functions
};

PyMODINIT_FUNC PyInit__tkagg(void)
{
    load_tkinter_funcs();
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    // Always raise ImportError (normalizing a previously set exception if
    // needed) to interact properly with backend auto-fallback.
    if (value) {
        PyErr_NormalizeException(&type, &value, &traceback);
        PyErr_SetObject(PyExc_ImportError, value);
        return NULL;
    } else if (!TCL_SETVAR) {
        PyErr_SetString(PyExc_ImportError, "Failed to load Tcl_SetVar");
        return NULL;
    } else if (!TK_FIND_PHOTO) {
        PyErr_SetString(PyExc_ImportError, "Failed to load Tk_FindPhoto");
        return NULL;
    } else if (!TK_PHOTO_PUT_BLOCK) {
        PyErr_SetString(PyExc_ImportError, "Failed to load Tk_PhotoPutBlock");
        return NULL;
    }
    return PyModule_Create(&_tkagg_module);
}
