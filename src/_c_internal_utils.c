#define PY_SSIZE_T_CLEAN
#include <Python.h>
#ifdef __linux__
#include <dlfcn.h>
#endif
#ifdef _WIN32
#include <Objbase.h>
#include <Shobjidl.h>
#include <Windows.h>
#endif

static PyObject*
mpl_display_is_valid(PyObject* module)
{
#ifdef __linux__
    void* libX11;
    // The getenv check is redundant but helps performance as it is much faster
    // than dlopen().
    if (getenv("DISPLAY")
        && (libX11 = dlopen("libX11.so.6", RTLD_LAZY))) {
        struct Display* display = NULL;
        struct Display* (* XOpenDisplay)(char const*) =
            dlsym(libX11, "XOpenDisplay");
        int (* XCloseDisplay)(struct Display*) =
            dlsym(libX11, "XCloseDisplay");
        if (XOpenDisplay && XCloseDisplay
                && (display = XOpenDisplay(NULL))) {
            XCloseDisplay(display);
        }
        if (dlclose(libX11)) {
            PyErr_SetString(PyExc_RuntimeError, dlerror());
            return NULL;
        }
        if (display) {
            Py_RETURN_TRUE;
        }
    }
    void* libwayland_client;
    if (getenv("WAYLAND_DISPLAY")
        && (libwayland_client = dlopen("libwayland-client.so.0", RTLD_LAZY))) {
        struct wl_display* display = NULL;
        struct wl_display* (* wl_display_connect)(char const*) =
            dlsym(libwayland_client, "wl_display_connect");
        void (* wl_display_disconnect)(struct wl_display*) =
            dlsym(libwayland_client, "wl_display_disconnect");
        if (wl_display_connect && wl_display_disconnect
                && (display = wl_display_connect(NULL))) {
            wl_display_disconnect(display);
        }
        if (dlclose(libwayland_client)) {
            PyErr_SetString(PyExc_RuntimeError, dlerror());
            return NULL;
        }
        if (display) {
            Py_RETURN_TRUE;
        }
    }
    Py_RETURN_FALSE;
#else
    Py_RETURN_TRUE;
#endif
}

static PyObject*
mpl_GetCurrentProcessExplicitAppUserModelID(PyObject* module)
{
#ifdef _WIN32
    wchar_t* appid = NULL;
    HRESULT hr = GetCurrentProcessExplicitAppUserModelID(&appid);
    if (FAILED(hr)) {
#if defined(PYPY_VERSION_NUM) && PYPY_VERSION_NUM < 0x07030600
        /* Remove when we require PyPy 7.3.6 */
        PyErr_SetFromWindowsErr(hr);
        return NULL;
#else
        return PyErr_SetFromWindowsErr(hr);
#endif
    }
    PyObject* py_appid = PyUnicode_FromWideChar(appid, -1);
    CoTaskMemFree(appid);
    return py_appid;
#else
    Py_RETURN_NONE;
#endif
}

static PyObject*
mpl_SetCurrentProcessExplicitAppUserModelID(PyObject* module, PyObject* arg)
{
#ifdef _WIN32
    wchar_t* appid = PyUnicode_AsWideCharString(arg, NULL);
    if (!appid) {
        return NULL;
    }
    HRESULT hr = SetCurrentProcessExplicitAppUserModelID(appid);
    PyMem_Free(appid);
    if (FAILED(hr)) {
#if defined(PYPY_VERSION_NUM) && PYPY_VERSION_NUM < 0x07030600
        /* Remove when we require PyPy 7.3.6 */
        PyErr_SetFromWindowsErr(hr);
        return NULL;
#else
        return PyErr_SetFromWindowsErr(hr);
#endif
    }
    Py_RETURN_NONE;
#else
    Py_RETURN_NONE;
#endif
}

static PyObject*
mpl_GetForegroundWindow(PyObject* module)
{
#ifdef _WIN32
  return PyLong_FromVoidPtr(GetForegroundWindow());
#else
  Py_RETURN_NONE;
#endif
}

static PyObject*
mpl_SetForegroundWindow(PyObject* module, PyObject *arg)
{
#ifdef _WIN32
  HWND handle = PyLong_AsVoidPtr(arg);
  if (PyErr_Occurred()) {
    return NULL;
  }
  if (!SetForegroundWindow(handle)) {
    return PyErr_Format(PyExc_RuntimeError, "Error setting window");
  }
  Py_RETURN_NONE;
#else
  Py_RETURN_NONE;
#endif
}

static PyObject*
mpl_SetProcessDpiAwareness_max(PyObject* module)
{
#ifdef _WIN32
#ifdef _DPI_AWARENESS_CONTEXTS_
    // These functions and options were added in later Windows 10 updates, so
    // must be loaded dynamically.
    typedef BOOL (WINAPI *IsValidDpiAwarenessContext_t)(DPI_AWARENESS_CONTEXT);
    typedef BOOL (WINAPI *SetProcessDpiAwarenessContext_t)(DPI_AWARENESS_CONTEXT);

    HMODULE user32 = LoadLibrary("user32.dll");
    IsValidDpiAwarenessContext_t IsValidDpiAwarenessContextPtr =
        (IsValidDpiAwarenessContext_t)GetProcAddress(
            user32, "IsValidDpiAwarenessContext");
    SetProcessDpiAwarenessContext_t SetProcessDpiAwarenessContextPtr =
        (SetProcessDpiAwarenessContext_t)GetProcAddress(
            user32, "SetProcessDpiAwarenessContext");
    DPI_AWARENESS_CONTEXT ctxs[3] = {
        DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,  // Win10 Creators Update
        DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE,     // Win10
        DPI_AWARENESS_CONTEXT_SYSTEM_AWARE};         // Win10
    if (IsValidDpiAwarenessContextPtr != NULL
            && SetProcessDpiAwarenessContextPtr != NULL) {
        for (int i = 0; i < sizeof(ctxs) / sizeof(DPI_AWARENESS_CONTEXT); ++i) {
            if (IsValidDpiAwarenessContextPtr(ctxs[i])) {
                SetProcessDpiAwarenessContextPtr(ctxs[i]);
                break;
            }
        }
    } else {
        // Added in Windows Vista.
        SetProcessDPIAware();
    }
    FreeLibrary(user32);
#else
    // Added in Windows Vista.
    SetProcessDPIAware();
#endif
#endif
    Py_RETURN_NONE;
}

static PyMethodDef functions[] = {
    {"display_is_valid", (PyCFunction)mpl_display_is_valid, METH_NOARGS,
     "display_is_valid()\n--\n\n"
     "Check whether the current X11 or Wayland display is valid.\n\n"
     "On Linux, returns True if either $DISPLAY is set and XOpenDisplay(NULL)\n"
     "succeeds, or $WAYLAND_DISPLAY is set and wl_display_connect(NULL)\n"
     "succeeds.\n\n"
     "On other platforms, always returns True."},
    {"Win32_GetCurrentProcessExplicitAppUserModelID",
     (PyCFunction)mpl_GetCurrentProcessExplicitAppUserModelID, METH_NOARGS,
     "Win32_GetCurrentProcessExplicitAppUserModelID()\n--\n\n"
     "Wrapper for Windows's GetCurrentProcessExplicitAppUserModelID.\n\n"
     "On non-Windows platforms, always returns None."},
    {"Win32_SetCurrentProcessExplicitAppUserModelID",
     (PyCFunction)mpl_SetCurrentProcessExplicitAppUserModelID, METH_O,
     "Win32_SetCurrentProcessExplicitAppUserModelID(appid, /)\n--\n\n"
     "Wrapper for Windows's SetCurrentProcessExplicitAppUserModelID.\n\n"
     "On non-Windows platforms, does nothing."},
    {"Win32_GetForegroundWindow",
     (PyCFunction)mpl_GetForegroundWindow, METH_NOARGS,
     "Win32_GetForegroundWindow()\n--\n\n"
     "Wrapper for Windows' GetForegroundWindow.\n\n"
     "On non-Windows platforms, always returns None."},
    {"Win32_SetForegroundWindow",
     (PyCFunction)mpl_SetForegroundWindow, METH_O,
     "Win32_SetForegroundWindow(hwnd, /)\n--\n\n"
     "Wrapper for Windows' SetForegroundWindow.\n\n"
     "On non-Windows platforms, does nothing."},
    {"Win32_SetProcessDpiAwareness_max",
     (PyCFunction)mpl_SetProcessDpiAwareness_max, METH_NOARGS,
     "Win32_SetProcessDpiAwareness_max()\n--\n\n"
     "Set Windows' process DPI awareness to best option available.\n\n"
     "On non-Windows platforms, does nothing."},
    {NULL, NULL}};  // sentinel.
static PyModuleDef util_module = {
    PyModuleDef_HEAD_INIT, "_c_internal_utils", NULL, 0, functions
};

#pragma GCC visibility push(default)
PyMODINIT_FUNC PyInit__c_internal_utils(void)
{
    return PyModule_Create(&util_module);
}
