"""
A wrapper around a native macOS NSWindow, for use by the GUI backends.

On macOS a backgrounded application cannot lift a window above the foreground
application, or take keyboard focus, using a toolkit's normal raise call. The
native NSWindow methods do work. The Qt, Tk and wx backends obtain their
NSWindow in different ways (see the constructors below) and then share the
operations on `MacOSWindow`. Keeping it a class makes it straightforward to add
further NSWindow operations later (e.g. window level or frame queries).
"""
import ctypes
import ctypes.util
import functools


@functools.cache
def _objc():
    objc = ctypes.CDLL(ctypes.util.find_library("objc"))
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    return objc


def _msg(receiver, selector, *args, argtypes=()):
    # Send an Objective-C message. objc_msgSend is variadic, so a prototype is
    # built per call signature; restype/argtypes MUST be pointer-width or 64-bit
    # object pointers are truncated and crash on arm64.
    objc = _objc()
    send = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, *argtypes)(
            ("objc_msgSend", objc))
    return send(ctypes.c_void_p(receiver), objc.sel_registerName(selector), *args)


class MacOSWindow:
    """A native macOS NSWindow, wrapped for use by the GUI backends."""

    def __init__(self, nswindow):
        # *nswindow* is the NSWindow pointer (int), or None/0 if unavailable.
        self._nswindow = nswindow or None

    @classmethod
    def from_nsview(cls, nsview):
        """Build from an NSView pointer (Qt's ``winId()``, wx's ``GetHandle()``)."""
        nswindow = _msg(int(nsview), b"window") if nsview else None
        return cls(nswindow)

    @classmethod
    def from_tk_drawable(cls, drawable):
        """Build from a Tk drawable (a window's ``winfo_id()``)."""
        try:
            get = ctypes.CDLL(None).Tk_MacOSXGetNSWindowForDrawable
        except (OSError, AttributeError):
            return cls(None)
        get.restype = ctypes.c_void_p
        get.argtypes = [ctypes.c_void_p]
        return cls(get(ctypes.c_void_p(int(drawable))))

    def raise_window(self, *, with_focus):
        """
        Raise the window to the front.

        If *with_focus*, also bring the application forward and give the window
        keyboard focus; otherwise raise it without taking focus.
        """
        if not self._nswindow:
            return
        if with_focus:
            # activateIgnoringOtherApps: is deprecated but used deliberately:
            # the modern -activate is cooperative and will not foreground a
            # background app when called programmatically (see backend_macosx).
            nsapp = _msg(_objc().objc_getClass(b"NSApplication"),
                         b"sharedApplication")
            _msg(nsapp, b"activateIgnoringOtherApps:", True,
                 argtypes=(ctypes.c_bool,))
            _msg(self._nswindow, b"makeKeyAndOrderFront:", None,
                 argtypes=(ctypes.c_void_p,))
        else:
            _msg(self._nswindow, b"orderFrontRegardless")
