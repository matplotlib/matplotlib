#include <Python.h>
/* Python.h must be included before any system headers,
    to ensure visibility macros are properly set. */
#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
// Windows 10, for latest HiDPI API support.
#define WINVER 0x0A00
#define _WIN32_WINNT 0x0A00
#endif
#include <pybind11/pybind11.h>
#ifdef __linux__
#include <dlfcn.h>
#endif
#ifdef _WIN32
#include <Objbase.h>
#include <Shobjidl.h>
#include <Windows.h>
#define UNUSED_ON_NON_WINDOWS(x) x
#else
#define UNUSED_ON_NON_WINDOWS Py_UNUSED
#endif

namespace py = pybind11;
using namespace pybind11::literals;

static bool
mpl_display_is_valid(void)
{
#ifdef __linux__
    void* libX11;
    // The getenv check is redundant but helps performance as it is much faster
    // than dlopen().
    if (getenv("DISPLAY")
        && (libX11 = dlopen("libX11.so.6", RTLD_LAZY))) {
        typedef struct Display* (*XOpenDisplay_t)(char const*);
        typedef int (*XCloseDisplay_t)(struct Display*);
        struct Display* display = NULL;
        XOpenDisplay_t XOpenDisplay = (XOpenDisplay_t)dlsym(libX11, "XOpenDisplay");
        XCloseDisplay_t XCloseDisplay = (XCloseDisplay_t)dlsym(libX11, "XCloseDisplay");
        if (XOpenDisplay && XCloseDisplay
                && (display = XOpenDisplay(NULL))) {
            XCloseDisplay(display);
        }
        if (dlclose(libX11)) {
            throw std::runtime_error(dlerror());
        }
        if (display) {
            return true;
        }
    }
    void* libwayland_client;
    if (getenv("WAYLAND_DISPLAY")
        && (libwayland_client = dlopen("libwayland-client.so.0", RTLD_LAZY))) {
        typedef struct wl_display* (*wl_display_connect_t)(char const*);
        typedef void (*wl_display_disconnect_t)(struct wl_display*);
        struct wl_display* display = NULL;
        wl_display_connect_t wl_display_connect =
            (wl_display_connect_t)dlsym(libwayland_client, "wl_display_connect");
        wl_display_disconnect_t wl_display_disconnect =
            (wl_display_disconnect_t)dlsym(libwayland_client, "wl_display_disconnect");
        if (wl_display_connect && wl_display_disconnect
                && (display = wl_display_connect(NULL))) {
            wl_display_disconnect(display);
        }
        if (dlclose(libwayland_client)) {
            throw std::runtime_error(dlerror());
        }
        if (display) {
            return true;
        }
    }
    return false;
#else
    return true;
#endif
}

static py::object
mpl_GetCurrentProcessExplicitAppUserModelID(void)
{
#ifdef _WIN32
    wchar_t* appid = NULL;
    HRESULT hr = GetCurrentProcessExplicitAppUserModelID(&appid);
    if (FAILED(hr)) {
        PyErr_SetFromWindowsErr(hr);
        throw py::error_already_set();
    }
    auto py_appid = py::cast(appid);
    CoTaskMemFree(appid);
    return py_appid;
#else
    return py::none();
#endif
}

static void
mpl_SetCurrentProcessExplicitAppUserModelID(const wchar_t* UNUSED_ON_NON_WINDOWS(appid))
{
#ifdef _WIN32
    HRESULT hr = SetCurrentProcessExplicitAppUserModelID(appid);
    if (FAILED(hr)) {
        PyErr_SetFromWindowsErr(hr);
        throw py::error_already_set();
    }
#endif
}

static py::object
mpl_GetForegroundWindow(void)
{
#ifdef _WIN32
  if (HWND hwnd = GetForegroundWindow()) {
    return py::capsule(hwnd, "HWND");
  } else {
    return py::none();
  }
#else
  return py::none();
#endif
}

static void
mpl_SetForegroundWindow(py::capsule UNUSED_ON_NON_WINDOWS(handle_p))
{
#ifdef _WIN32
    if (handle_p.name() != "HWND") {
        throw std::runtime_error("Handle must be a value returned from Win32_GetForegroundWindow");
    }
    HWND handle = static_cast<HWND>(handle_p.get_pointer());
    if (!SetForegroundWindow(handle)) {
        throw std::runtime_error("Error setting window");
    }
#endif
}

static void
mpl_SetProcessDpiAwareness_max(void)
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
}

PYBIND11_MODULE(_c_internal_utils, m)
{
    m.def(
        "display_is_valid", &mpl_display_is_valid,
        R"""(        --
        Check whether the current X11 or Wayland display is valid.

        On Linux, returns True if either $DISPLAY is set and XOpenDisplay(NULL)
        succeeds, or $WAYLAND_DISPLAY is set and wl_display_connect(NULL)
        succeeds.

        On other platforms, always returns True.)""");
    m.def(
        "Win32_GetCurrentProcessExplicitAppUserModelID",
        &mpl_GetCurrentProcessExplicitAppUserModelID,
        R"""(        --
        Wrapper for Windows's GetCurrentProcessExplicitAppUserModelID.

        On non-Windows platforms, always returns None.)""");
    m.def(
        "Win32_SetCurrentProcessExplicitAppUserModelID",
        &mpl_SetCurrentProcessExplicitAppUserModelID,
        "appid"_a, py::pos_only(),
        R"""(        --
        Wrapper for Windows's SetCurrentProcessExplicitAppUserModelID.

        On non-Windows platforms, does nothing.)""");
    m.def(
        "Win32_GetForegroundWindow", &mpl_GetForegroundWindow,
        R"""(        --
        Wrapper for Windows' GetForegroundWindow.

        On non-Windows platforms, always returns None.)""");
    m.def(
        "Win32_SetForegroundWindow", &mpl_SetForegroundWindow,
        "hwnd"_a,
        R"""(        --
        Wrapper for Windows' SetForegroundWindow.

        On non-Windows platforms, does nothing.)""");
    m.def(
        "Win32_SetProcessDpiAwareness_max", &mpl_SetProcessDpiAwareness_max,
        R"""(        --
        Set Windows' process DPI awareness to best option available.

        On non-Windows platforms, does nothing.)""");
}
