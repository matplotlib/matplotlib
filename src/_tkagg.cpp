/* -*- mode: c++; c-basic-offset: 4 -*- */

// Where is PIL?
//
// Many years ago, Matplotlib used to include code from PIL (the Python Imaging
// Library).  Since then, the code has changed a lot - the organizing principle
// and methods of operation are now quite different.  Because our review of
// the codebase showed that all the code that came from PIL was removed or
// rewritten, we have removed the PIL licensing information.  If you want PIL,
// you can get it at https://python-pillow.org/

#include <Python.h>
#include <new>
#include <stdexcept>
#include <string>
#include <tuple>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
// Windows 8.1
#define WINVER 0x0603
#define _WIN32_WINNT 0x0603
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace pybind11::literals;

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
#include <vector>

#include <windows.h>
#include <commctrl.h>
#define PSAPI_VERSION 1
#include <psapi.h>  // Must be linked with 'psapi' library
#define dlsym GetProcAddress
#define UNUSED_ON_NON_WINDOWS(x) x
// Check for old headers that do not defined HiDPI functions and constants.
#if defined(__MINGW64_VERSION_MAJOR)
static_assert(__MINGW64_VERSION_MAJOR >= 6,
              "mingw-w64-x86_64-headers >= 6 are required when compiling with MinGW");
#endif
#else
#include <dlfcn.h>
#define UNUSED_ON_NON_WINDOWS Py_UNUSED
#endif

// Include our own excerpts from the Tcl / Tk headers
#include "_tkmini.h"

template <class T>
static T
convert_voidptr(const py::object &obj)
{
    auto result = static_cast<T>(PyLong_AsVoidPtr(obj.ptr()));
    if (PyErr_Occurred()) {
        throw py::error_already_set();
    }
    return result;
}

// Global vars for Tk functions.  We load these symbols from the tkinter
// extension module or loaded Tk libraries at run-time.
static Tk_FindPhoto_t TK_FIND_PHOTO;
static Tk_PhotoPutBlock_t TK_PHOTO_PUT_BLOCK;
// Global vars for Tcl functions.  We load these symbols from the tkinter
// extension module or loaded Tcl libraries at run-time.
static Tcl_SetVar_t TCL_SETVAR;

static void
mpl_tk_blit(py::object interp_obj, const char *photo_name,
            py::array_t<unsigned char> data, int comp_rule,
            std::tuple<int, int, int, int> offset, std::tuple<int, int, int, int> bbox)
{
    auto interp = convert_voidptr<Tcl_Interp *>(interp_obj);

    Tk_PhotoHandle photo;
    if (!(photo = TK_FIND_PHOTO(interp, photo_name))) {
        throw py::value_error("Failed to extract Tk_PhotoHandle");
    }

    auto data_ptr = data.mutable_unchecked<3>();  // Checks ndim and writeable flag.
    if (data.shape(2) != 4) {
        throw py::value_error(
            "Data pointer must be RGBA; last dimension is {}, not 4"_s.format(
                data.shape(2)));
    }
    if (data.shape(0) > INT_MAX) {  // Limited by Tk_PhotoPutBlock argument type.
        throw std::range_error(
            "Height ({}) exceeds maximum allowable size ({})"_s.format(
                data.shape(0), INT_MAX));
    }
    if (data.shape(1) > INT_MAX / 4) {  // Limited by Tk_PhotoImageBlock.pitch field.
        throw std::range_error(
            "Width ({}) exceeds maximum allowable size ({})"_s.format(
                data.shape(1), INT_MAX / 4));
    }
    const auto height = static_cast<int>(data.shape(0));
    const auto width = static_cast<int>(data.shape(1));
    int x1, x2, y1, y2;
    std::tie(x1, x2, y1, y2) = bbox;
    if (0 > y1 || y1 > y2 || y2 > height || 0 > x1 || x1 > x2 || x2 > width) {
        throw py::value_error("Attempting to draw out of bounds");
    }
    if (comp_rule != TK_PHOTO_COMPOSITE_OVERLAY && comp_rule != TK_PHOTO_COMPOSITE_SET) {
        throw py::value_error("Invalid comp_rule argument");
    }

    int put_retval;
    Tk_PhotoImageBlock block;
    block.pixelPtr = data_ptr.mutable_data(height - y2, x1, 0);
    block.width = x2 - x1;
    block.height = y2 - y1;
    block.pitch = 4 * width;
    block.pixelSize = 4;
    std::tie(block.offset[0], block.offset[1], block.offset[2], block.offset[3]) = offset;
    {
        py::gil_scoped_release release;
        put_retval = TK_PHOTO_PUT_BLOCK(
            interp, photo, &block, x1, height - y2, x2 - x1, y2 - y1, comp_rule);
    }
    if (put_retval == TCL_ERROR) {
        throw std::bad_alloc();
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

static py::object
mpl_tk_enable_dpi_awareness(py::object UNUSED_ON_NON_WINDOWS(frame_handle_obj),
                            py::object UNUSED_ON_NON_WINDOWS(interp_obj))
{
#ifdef WIN32_DLL
    auto frame_handle = convert_voidptr<HWND>(frame_handle_obj);
    auto interp = convert_voidptr<Tcl_Interp *>(interp_obj);

#ifdef _DPI_AWARENESS_CONTEXTS_
    HMODULE user32 = LoadLibrary("user32.dll");

    typedef DPI_AWARENESS_CONTEXT (WINAPI *GetWindowDpiAwarenessContext_t)(HWND);
    GetWindowDpiAwarenessContext_t GetWindowDpiAwarenessContextPtr =
        (GetWindowDpiAwarenessContext_t)GetProcAddress(
            user32, "GetWindowDpiAwarenessContext");
    if (GetWindowDpiAwarenessContextPtr == NULL) {
        FreeLibrary(user32);
        return py::cast(false);
    }

    typedef BOOL (WINAPI *AreDpiAwarenessContextsEqual_t)(DPI_AWARENESS_CONTEXT,
                                                          DPI_AWARENESS_CONTEXT);
    AreDpiAwarenessContextsEqual_t AreDpiAwarenessContextsEqualPtr =
        (AreDpiAwarenessContextsEqual_t)GetProcAddress(
            user32, "AreDpiAwarenessContextsEqual");
    if (AreDpiAwarenessContextsEqualPtr == NULL) {
        FreeLibrary(user32);
        return py::cast(false);
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
    return py::cast(per_monitor);
#endif
#endif

    return py::none();
}

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

static void
load_tkinter_funcs()
{
    HANDLE process = GetCurrentProcess();  // Pseudo-handle, doesn't need closing.
    DWORD size;
    if (!EnumProcessModules(process, NULL, 0, &size)) {
        PyErr_SetFromWindowsErr(0);
        throw py::error_already_set();
    }
    auto count = size / sizeof(HMODULE);
    auto modules = std::vector<HMODULE>(count);
    if (!EnumProcessModules(process, modules.data(), size, &size)) {
        PyErr_SetFromWindowsErr(0);
        throw py::error_already_set();
    }
    for (auto mod: modules) {
        if (load_tcl_tk(mod)) {
            return;
        }
    }
}

#else  // not Windows

/*
 * On Unix, we can get the Tk symbols from the tkinter module, because tkinter
 * uses these symbols, and the symbols are therefore visible in the tkinter
 * dynamic library (module).
 */

static void
load_tkinter_funcs()
{
    // Load tkinter global funcs from tkinter compiled module.

    // Try loading from the main program namespace first.
    auto main_program = dlopen(NULL, RTLD_LAZY);
    auto success = load_tcl_tk(main_program);
    // We don't need to keep a reference open as the main program always exists.
    if (dlclose(main_program)) {
        throw std::runtime_error(dlerror());
    }
    if (success) {
        return;
    }

    py::object module;
    // Handle PyPy first, as that import will correctly fail on CPython.
    try {
        module = py::module_::import("_tkinter.tklib_cffi");  // PyPy
    } catch (py::error_already_set &e) {
        module = py::module_::import("_tkinter");  // CPython
    }
    auto py_path = module.attr("__file__");
    auto py_path_b = py::reinterpret_steal<py::bytes>(
        PyUnicode_EncodeFSDefault(py_path.ptr()));
    std::string path = py_path_b;
    auto tkinter_lib = dlopen(path.c_str(), RTLD_LAZY);
    if (!tkinter_lib) {
        throw std::runtime_error(dlerror());
    }
    load_tcl_tk(tkinter_lib);
    // We don't need to keep a reference open as tkinter has been imported.
    if (dlclose(tkinter_lib)) {
        throw std::runtime_error(dlerror());
    }
}
#endif // end not Windows

PYBIND11_MODULE(_tkagg, m)
{
    try {
        load_tkinter_funcs();
    } catch (py::error_already_set& e) {
        // Always raise ImportError to interact properly with backend auto-fallback.
        py::raise_from(e, PyExc_ImportError, "failed to load tkinter functions");
        throw py::error_already_set();
    }

    if (!TCL_SETVAR) {
        throw py::import_error("Failed to load Tcl_SetVar");
    } else if (!TK_FIND_PHOTO) {
        throw py::import_error("Failed to load Tk_FindPhoto");
    } else if (!TK_PHOTO_PUT_BLOCK) {
        throw py::import_error("Failed to load Tk_PhotoPutBlock");
    }

    m.def("blit", &mpl_tk_blit,
          "interp"_a, "photo_name"_a, "data"_a, "comp_rule"_a, "offset"_a, "bbox"_a);
    m.def("enable_dpi_awareness", &mpl_tk_enable_dpi_awareness,
          "frame_handle"_a, "interp"_a);

    m.attr("TK_PHOTO_COMPOSITE_OVERLAY") = TK_PHOTO_COMPOSITE_OVERLAY;
    m.attr("TK_PHOTO_COMPOSITE_SET") = TK_PHOTO_COMPOSITE_SET;
}
