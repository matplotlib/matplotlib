#define PY_SSIZE_T_CLEAN
#include <Cocoa/Cocoa.h>
#include <ApplicationServices/ApplicationServices.h>
#include <Python.h>
#include "mplutils.h"

/* Proper way to check for the OS X version we are compiling for, from
 * https://developer.apple.com/library/archive/documentation/DeveloperTools/Conceptual/cross_development/Using/using.html

 * Renamed symbols cause deprecation warnings, so define macros for the new
 * names if we are compiling on an older SDK */
#if __MAC_OS_X_VERSION_MIN_REQUIRED < 101400
#define NSButtonTypeMomentaryLight           NSMomentaryLightButton
#define NSButtonTypePushOnPushOff            NSPushOnPushOffButton
#define NSBezelStyleShadowlessSquare         NSShadowlessSquareBezelStyle
#define CGContext                            graphicsPort
#endif


/* Various NSApplicationDefined event subtypes */
#define STOP_EVENT_LOOP 2
#define WINDOW_CLOSING 3


/* Keep track of number of windows present
   Needed to know when to stop the NSApp */
static long FigureWindowCount = 0;

/* Keep track of modifier key states for flagsChanged
   to keep track of press vs release */
static bool lastCommand = false;
static bool lastControl = false;
static bool lastShift = false;
static bool lastOption = false;
static bool lastCapsLock = false;
/* Keep track of whether this specific key modifier was pressed or not */
static bool keyChangeCommand = false;
static bool keyChangeControl = false;
static bool keyChangeShift = false;
static bool keyChangeOption = false;
static bool keyChangeCapsLock = false;
/* Keep track of the current mouse up/down state for open/closed cursor hand */
static bool leftMouseGrabbing = false;
/* Keep track of whether stdin has been received */
static bool stdin_received = false;
static bool stdin_sigint = false;
// Global variable to store the original SIGINT handler
static struct sigaction originalSigintAction = {0};

// Signal handler for SIGINT, only sets a flag to exit the run loop
static void handleSigint(int signal) {
    stdin_sigint = true;
}

static int wait_for_stdin() {
    @autoreleasepool {
        stdin_received = false;
        stdin_sigint = false;

        // Set up a SIGINT handler to interrupt the event loop if ctrl+c comes in too
        struct sigaction customAction = {0};
        customAction.sa_handler = handleSigint;
        // Set the new handler and store the old one
        sigaction(SIGINT, &customAction, &originalSigintAction);

        // Create an NSFileHandle for standard input
        NSFileHandle *stdinHandle = [NSFileHandle fileHandleWithStandardInput];

        // Register for data available notifications on standard input
        [[NSNotificationCenter defaultCenter] addObserverForName: NSFileHandleDataAvailableNotification
                                                          object: stdinHandle
                                                           queue: [NSOperationQueue mainQueue] // Use the main queue
                                                      usingBlock: ^(NSNotification *notification) {
                                                                    // Mark that input has been received
                                                                    stdin_received = true;
                                                                    }
        ];

        // Wait in the background for anything that happens to stdin
        [stdinHandle waitForDataInBackgroundAndNotify];

        // continuously run an event loop until the stdin_received flag is set to exit
        while (!stdin_received && !stdin_sigint) {
            while (true) {
                NSEvent *event = [NSApp nextEventMatchingMask: NSEventMaskAny
                                                    untilDate: [NSDate distantPast]
                                                       inMode: NSDefaultRunLoopMode
                                                      dequeue: YES];
                if (!event) { break; }
                [NSApp sendEvent: event];
            }
            // We need to run the run loop for a short time to allow the
            // events to be processed and keep flushing them while we wait for stdin
            [[NSRunLoop currentRunLoop] runUntilDate: [NSDate dateWithTimeIntervalSinceNow: 0.01]];
        }
        // Remove the input handler as an observer
        [[NSNotificationCenter defaultCenter] removeObserver: stdinHandle];

        // Restore the original SIGINT handler upon exiting the function
        sigaction(SIGINT, &originalSigintAction, NULL);
        return 1;
    }
}

/* ---------------------------- Cocoa classes ---------------------------- */

@interface Window : NSWindow
{   PyObject* manager;
}
- (Window*)initWithContentRect:(NSRect)rect styleMask:(unsigned int)mask backing:(NSBackingStoreType)bufferingType defer:(BOOL)deferCreation withManager: (PyObject*)theManager;
- (NSRect)constrainFrameRect:(NSRect)rect toScreen:(NSScreen*)screen;
- (BOOL)closeButtonPressed;
@end

@interface View : NSView <NSWindowDelegate>
{   PyObject* canvas;
    NSRect rubberband;
    @public double device_scale;
}
- (void)dealloc;
- (void)drawRect:(NSRect)rect;
- (void)updateDevicePixelRatio:(double)scale;
- (void)windowDidChangeBackingProperties:(NSNotification*)notification;
- (void)windowDidResize:(NSNotification*)notification;
- (View*)initWithFrame:(NSRect)rect;
- (void)setCanvas: (PyObject*)newCanvas;
- (void)windowWillClose:(NSNotification*)notification;
- (BOOL)windowShouldClose:(NSNotification*)notification;
- (void)mouseEntered:(NSEvent*)event;
- (void)mouseExited:(NSEvent*)event;
- (void)mouseDown:(NSEvent*)event;
- (void)mouseUp:(NSEvent*)event;
- (void)mouseDragged:(NSEvent*)event;
- (void)mouseMoved:(NSEvent*)event;
- (void)rightMouseDown:(NSEvent*)event;
- (void)rightMouseUp:(NSEvent*)event;
- (void)rightMouseDragged:(NSEvent*)event;
- (void)otherMouseDown:(NSEvent*)event;
- (void)otherMouseUp:(NSEvent*)event;
- (void)otherMouseDragged:(NSEvent*)event;
- (void)setRubberband:(NSRect)rect;
- (void)removeRubberband;
- (const char*)convertKeyEvent:(NSEvent*)event;
- (void)keyDown:(NSEvent*)event;
- (void)keyUp:(NSEvent*)event;
- (void)scrollWheel:(NSEvent *)event;
- (BOOL)acceptsFirstResponder;
- (void)flagsChanged:(NSEvent*)event;
@end

/* ---------------------------- Python classes ---------------------------- */

// Acquire the GIL, call a method with no args, discarding the result and
// printing any exception.
static void gil_call_method(PyObject* obj, const char* name)
{
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* result = PyObject_CallMethod(obj, name, NULL);
    if (result) {
        Py_DECREF(result);
    } else {
        PyErr_Print();
    }
    PyGILState_Release(gstate);
}

void process_event(char const* cls_name, char const* fmt, ...)
{
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* module = NULL, * cls = NULL,
            * args = NULL, * kwargs = NULL,
            * event = NULL, * result = NULL;
    va_list argp;
    va_start(argp, fmt);
    if (!(module = PyImport_ImportModule("matplotlib.backend_bases"))
        || !(cls = PyObject_GetAttrString(module, cls_name))
        || !(args = PyTuple_New(0))
        || !(kwargs = Py_VaBuildValue(fmt, argp))
        || !(event = PyObject_Call(cls, args, kwargs))
        || !(result = PyObject_CallMethod(event, "_process", ""))) {
        PyErr_Print();
    }
    va_end(argp);
    Py_XDECREF(module);
    Py_XDECREF(cls);
    Py_XDECREF(args);
    Py_XDECREF(kwargs);
    Py_XDECREF(event);
    Py_XDECREF(result);
    PyGILState_Release(gstate);
}

static bool backend_inited = false;

static void lazy_init(void) {
    if (backend_inited) { return; }
    backend_inited = true;

    NSApp = [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

    // Run our own event loop while waiting for stdin on the Python side
    // this is needed to keep the application responsive while waiting for input
    PyOS_InputHook = wait_for_stdin;
}

static PyObject*
event_loop_is_running(PyObject* self)
{
    if (backend_inited) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

static PyObject*
wake_on_fd_write(PyObject* unused, PyObject* args)
{
    int fd;
    if (!PyArg_ParseTuple(args, "i", &fd)) { return NULL; }
    NSFileHandle* fh = [[NSFileHandle alloc] initWithFileDescriptor: fd];
    [fh waitForDataInBackgroundAndNotify];
    [[NSNotificationCenter defaultCenter]
        addObserverForName: NSFileHandleDataAvailableNotification
                    object: fh
                     queue: nil
                usingBlock: ^(NSNotification* note) {
                    PyGILState_STATE gstate = PyGILState_Ensure();
                    PyErr_CheckSignals();
                    PyGILState_Release(gstate);
                }];
    Py_RETURN_NONE;
}

static PyObject*
stop(PyObject* self)
{
    [NSApp stop: nil];
    // Post an event to trigger the actual stopping.
    [NSApp postEvent: [NSEvent otherEventWithType: NSEventTypeApplicationDefined
                                         location: NSZeroPoint
                                    modifierFlags: 0
                                        timestamp: 0
                                     windowNumber: 0
                                          context: nil
                                          subtype: 0
                                            data1: 0
                                            data2: 0]
             atStart: YES];
    Py_RETURN_NONE;
}

static CGFloat _get_device_scale(CGContextRef cr)
{
    CGSize pixelSize = CGContextConvertSizeToDeviceSpace(cr, CGSizeMake(1, 1));
    return pixelSize.width;
}

int mpl_check_modifier(
        NSUInteger modifiers, NSEventModifierFlags flag,
        PyObject* list, char const* name)
{
    int status = 0;
    if (modifiers & flag) {
        PyObject* py_name = NULL;
        if (!(py_name = PyUnicode_FromString(name))
            || PyList_Append(list, py_name)) {
            status = -1;  // failure
        }
        Py_XDECREF(py_name);
    }
    return status;
}

PyObject* mpl_modifiers(NSEvent* event)
{
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* list = NULL;
    if (!(list = PyList_New(0))) {
        goto exit;
    }
    NSUInteger modifiers = [event modifierFlags];
    if (mpl_check_modifier(modifiers, NSEventModifierFlagControl, list, "ctrl")
        || mpl_check_modifier(modifiers, NSEventModifierFlagOption, list, "alt")
        || mpl_check_modifier(modifiers, NSEventModifierFlagShift, list, "shift")
        || mpl_check_modifier(modifiers, NSEventModifierFlagCommand, list, "cmd")) {
        Py_CLEAR(list);  // On failure, return NULL with an exception set.
    }
exit:
    PyGILState_Release(gstate);
    return list;
}

typedef struct {
    PyObject_HEAD
    View* view;
} FigureCanvas;

static PyTypeObject FigureCanvasType;

static PyObject*
FigureCanvas_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    lazy_init();
    FigureCanvas *self = (FigureCanvas*)type->tp_alloc(type, 0);
    if (!self) { return NULL; }
    self->view = [View alloc];
    return (PyObject*)self;
}

static int
FigureCanvas_init(FigureCanvas *self, PyObject *args, PyObject *kwds)
{
    if (!self->view) {
        PyErr_SetString(PyExc_RuntimeError, "NSView* is NULL");
        return -1;
    }
    PyObject *builtins = NULL,
             *super_obj = NULL,
             *super_init = NULL,
             *init_res = NULL,
             *wh = NULL;
    // super(FigureCanvasMac, self).__init__(*args, **kwargs)
    if (!(builtins = PyImport_AddModule("builtins"))  // borrowed.
            || !(super_obj = PyObject_CallMethod(builtins, "super", "OO", &FigureCanvasType, self))
            || !(super_init = PyObject_GetAttrString(super_obj, "__init__"))
            || !(init_res = PyObject_Call(super_init, args, kwds))) {
        goto exit;
    }
    int width, height;
    if (!(wh = PyObject_CallMethod((PyObject*)self, "get_width_height", ""))
            || !PyArg_ParseTuple(wh, "ii", &width, &height)) {
        goto exit;
    }
    NSRect rect = NSMakeRect(0.0, 0.0, width, height);
    self->view = [self->view initWithFrame: rect];
    self->view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    int opts = (NSTrackingMouseEnteredAndExited | NSTrackingMouseMoved |
                NSTrackingActiveInKeyWindow | NSTrackingInVisibleRect);
    [self->view addTrackingArea: [
        [NSTrackingArea alloc] initWithRect: rect
                                    options: opts
                                      owner: self->view
                                   userInfo: nil]];
    [self->view setCanvas: (PyObject*)self];

exit:
    Py_XDECREF(super_obj);
    Py_XDECREF(super_init);
    Py_XDECREF(init_res);
    Py_XDECREF(wh);
    return PyErr_Occurred() ? -1 : 0;
}

static void
FigureCanvas_dealloc(FigureCanvas* self)
{
    [self->view setCanvas: NULL];
    [self->view release];
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject*
FigureCanvas_repr(FigureCanvas* self)
{
    return PyUnicode_FromFormat("FigureCanvas object %p wrapping NSView %p",
                               (void*)self, (void*)(self->view));
}

static PyObject*
FigureCanvas_update(FigureCanvas* self)
{
    [self->view setNeedsDisplay: YES];
    Py_RETURN_NONE;
}

static PyObject*
FigureCanvas_flush_events(FigureCanvas* self)
{
    // We run the app, matching any events that are waiting in the queue
    // to process, breaking out of the loop when no events remain and
    // displaying the canvas if needed.
    NSEvent *event;
    while (true) {
        event = [NSApp nextEventMatchingMask: NSEventMaskAny
                                   untilDate: [NSDate distantPast]
                                      inMode: NSDefaultRunLoopMode
                                     dequeue: YES];
        if (!event) {
            break;
        }
        [NSApp sendEvent:event];
    }
    [self->view displayIfNeeded];
    Py_RETURN_NONE;
}

static PyObject*
FigureCanvas_set_cursor(PyObject* unused, PyObject* args)
{
    int i;
    if (!PyArg_ParseTuple(args, "i", &i)) { return NULL; }
    switch (i) {
      case 1: [[NSCursor arrowCursor] set]; break;
      case 2: [[NSCursor pointingHandCursor] set]; break;
      case 3: [[NSCursor crosshairCursor] set]; break;
      case 4:
        if (leftMouseGrabbing) {
            [[NSCursor closedHandCursor] set];
        } else {
            [[NSCursor openHandCursor] set];
        }
        break;
      /* OSX handles busy state itself so no need to set a cursor here */
      case 5: break;
      case 6: [[NSCursor resizeLeftRightCursor] set]; break;
      case 7: [[NSCursor resizeUpDownCursor] set]; break;
      default: return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
FigureCanvas_set_rubberband(FigureCanvas* self, PyObject *args)
{
    View* view = self->view;
    if (!view) {
        PyErr_SetString(PyExc_RuntimeError, "NSView* is NULL");
        return NULL;
    }
    int x0, y0, x1, y1;
    if (!PyArg_ParseTuple(args, "iiii", &x0, &y0, &x1, &y1)) {
        return NULL;
    }
    x0 /= view->device_scale;
    x1 /= view->device_scale;
    y0 /= view->device_scale;
    y1 /= view->device_scale;
    NSRect rubberband = NSMakeRect(x0 < x1 ? x0 : x1, y0 < y1 ? y0 : y1,
                                   abs(x1 - x0), abs(y1 - y0));
    [view setRubberband: rubberband];
    Py_RETURN_NONE;
}

static PyObject*
FigureCanvas_remove_rubberband(FigureCanvas* self)
{
    [self->view removeRubberband];
    Py_RETURN_NONE;
}

static PyObject*
FigureCanvas__start_event_loop(FigureCanvas* self, PyObject* args, PyObject* keywords)
{
    float timeout = 0.0;

    static char* kwlist[] = {"timeout", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "f", kwlist, &timeout)) {
        return NULL;
    }

    NSDate* date =
        (timeout > 0.0) ? [NSDate dateWithTimeIntervalSinceNow: timeout]
                        : [NSDate distantFuture];
    while (true)
    {   NSEvent* event = [NSApp nextEventMatchingMask: NSEventMaskAny
                                            untilDate: date
                                               inMode: NSDefaultRunLoopMode
                                              dequeue: YES];
       if (!event || [event type]==NSEventTypeApplicationDefined) { break; }
       [NSApp sendEvent: event];
    }

    Py_RETURN_NONE;
}

static PyObject*
FigureCanvas_stop_event_loop(FigureCanvas* self)
{
    NSEvent* event = [NSEvent otherEventWithType: NSEventTypeApplicationDefined
                                        location: NSZeroPoint
                                   modifierFlags: 0
                                       timestamp: 0.0
                                    windowNumber: 0
                                         context: nil
                                         subtype: STOP_EVENT_LOOP
                                           data1: 0
                                           data2: 0];
    [NSApp postEvent: event atStart: true];
    Py_RETURN_NONE;
}

static PyTypeObject FigureCanvasType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_macosx.FigureCanvas",
    .tp_basicsize = sizeof(FigureCanvas),
    .tp_dealloc = (destructor)FigureCanvas_dealloc,
    .tp_repr = (reprfunc)FigureCanvas_repr,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_init = (initproc)FigureCanvas_init,
    .tp_new = (newfunc)FigureCanvas_new,
    .tp_doc = "A FigureCanvas object wraps a Cocoa NSView object.",
    .tp_methods = (PyMethodDef[]){
        {"update",
         (PyCFunction)FigureCanvas_update,
         METH_NOARGS,
         NULL},  // docstring inherited
        {"flush_events",
         (PyCFunction)FigureCanvas_flush_events,
         METH_NOARGS,
         NULL},  // docstring inherited
        {"set_cursor",
         (PyCFunction)FigureCanvas_set_cursor,
         METH_VARARGS,
         "Set the active cursor."},
        {"set_rubberband",
         (PyCFunction)FigureCanvas_set_rubberband,
         METH_VARARGS,
         "Specify a new rubberband rectangle and invalidate it."},
        {"remove_rubberband",
         (PyCFunction)FigureCanvas_remove_rubberband,
         METH_NOARGS,
         "Remove the current rubberband rectangle."},
        {"_start_event_loop",
         (PyCFunction)FigureCanvas__start_event_loop,
         METH_KEYWORDS | METH_VARARGS,
         NULL},  // docstring inherited
        {"stop_event_loop",
         (PyCFunction)FigureCanvas_stop_event_loop,
         METH_NOARGS,
         NULL},  // docstring inherited
        {}  // sentinel
    },
};

typedef struct {
    PyObject_HEAD
    Window* window;
} FigureManager;

static PyObject*
FigureManager_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    lazy_init();
    Window* window = [Window alloc];
    if (!window) { return NULL; }
    FigureManager *self = (FigureManager*)type->tp_alloc(type, 0);
    if (!self) {
        [window release];
        return NULL;
    }
    self->window = window;
    ++FigureWindowCount;
    return (PyObject*)self;
}

static int
FigureManager_init(FigureManager *self, PyObject *args, PyObject *kwds)
{
    PyObject* canvas;
    if (!PyArg_ParseTuple(args, "O", &canvas)) {
        return -1;
    }

    View* view = ((FigureCanvas*)canvas)->view;
    if (!view) {  /* Something really weird going on */
        PyErr_SetString(PyExc_RuntimeError, "NSView* is NULL");
        return -1;
    }

    PyObject* size = PyObject_CallMethod(canvas, "get_width_height", "");
    int width, height;
    if (!size || !PyArg_ParseTuple(size, "ii", &width, &height)) {
        Py_XDECREF(size);
        return -1;
    }
    Py_DECREF(size);

    NSRect rect = NSMakeRect( /* x */ 100, /* y */ 350, width, height);

    self->window = [self->window initWithContentRect: rect
                                         styleMask: NSWindowStyleMaskTitled
                                                  | NSWindowStyleMaskClosable
                                                  | NSWindowStyleMaskResizable
                                                  | NSWindowStyleMaskMiniaturizable
                                           backing: NSBackingStoreBuffered
                                             defer: YES
                                       withManager: (PyObject*)self];
    Window* window = self->window;
    [window setDelegate: view];
    [window makeFirstResponder: view];
    [[window contentView] addSubview: view];
    [view updateDevicePixelRatio: [window backingScaleFactor]];

    return 0;
}

static PyObject*
FigureManager__set_window_mode(FigureManager* self, PyObject* args)
{
    const char* window_mode;
    if (!PyArg_ParseTuple(args, "s", &window_mode) || !self->window) {
        return NULL;
    }

    NSString* window_mode_str = [NSString stringWithUTF8String: window_mode];
    if ([window_mode_str isEqualToString: @"tab"]) {
        [self->window setTabbingMode: NSWindowTabbingModePreferred];
    } else if ([window_mode_str isEqualToString: @"window"]) {
        [self->window setTabbingMode: NSWindowTabbingModeDisallowed];
    } else { // system settings
        [self->window setTabbingMode: NSWindowTabbingModeAutomatic];
    }
    Py_RETURN_NONE;
}

static PyObject*
FigureManager_repr(FigureManager* self)
{
    return PyUnicode_FromFormat("FigureManager object %p wrapping NSWindow %p",
                               (void*) self, (void*)(self->window));
}

static void
FigureManager_dealloc(FigureManager* self)
{
    [self->window close];
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject*
FigureManager__show(FigureManager* self)
{
    [self->window makeKeyAndOrderFront: nil];
    Py_RETURN_NONE;
}

static PyObject*
FigureManager__raise(FigureManager* self)
{
    [self->window orderFrontRegardless];
    Py_RETURN_NONE;
}

static PyObject*
FigureManager_destroy(FigureManager* self)
{
    [self->window close];
    self->window = NULL;
    Py_RETURN_NONE;
}

static PyObject*
FigureManager_set_icon(PyObject* null, PyObject* args) {
    PyObject* icon_path;
    if (!PyArg_ParseTuple(args, "O&", &PyUnicode_FSDecoder, &icon_path)) {
        return NULL;
    }
    const char* icon_path_ptr = PyUnicode_AsUTF8(icon_path);
    if (!icon_path_ptr) {
        Py_DECREF(icon_path);
        return NULL;
    }
    @autoreleasepool {
        NSString* ns_icon_path = [NSString stringWithUTF8String: icon_path_ptr];
        Py_DECREF(icon_path);
        if (!ns_icon_path) {
            PyErr_SetString(PyExc_RuntimeError, "Could not convert to NSString*");
            return NULL;
        }
        NSImage* image = [[[NSImage alloc] initByReferencingFile: ns_icon_path] autorelease];
        if (!image) {
            PyErr_SetString(PyExc_RuntimeError, "Could not create NSImage*");
            return NULL;
        }
        if (!image.valid) {
            PyErr_SetString(PyExc_RuntimeError, "Image is not valid");
            return NULL;
        }
        @try {
          NSApplication* app = [NSApplication sharedApplication];
          app.applicationIconImage = image;
        }
        @catch (NSException* exception) {
            PyErr_SetString(PyExc_RuntimeError, exception.reason.UTF8String);
            return NULL;
        }
    }
    Py_RETURN_NONE;
}

static PyObject*
FigureManager_set_window_title(FigureManager* self,
                               PyObject *args, PyObject *kwds)
{
    const char* title;
    if (!PyArg_ParseTuple(args, "s", &title)) {
        return NULL;
    }
    [self->window setTitle: [NSString stringWithUTF8String: title]];
    Py_RETURN_NONE;
}

static PyObject*
FigureManager_get_window_title(FigureManager* self)
{
    NSString* title = [self->window title];
    if (title) {
        return PyUnicode_FromString([title UTF8String]);
    } else {
        Py_RETURN_NONE;
    }
}

static PyObject*
FigureManager_resize(FigureManager* self, PyObject *args, PyObject *kwds)
{
    int width, height;
    if (!PyArg_ParseTuple(args, "ii", &width, &height)) {
        return NULL;
    }
    Window* window = self->window;
    if (window) {
        CGFloat device_pixel_ratio = [window backingScaleFactor];
        width /= device_pixel_ratio;
        height /= device_pixel_ratio;
        // 36 comes from hard-coded size of toolbar later in code
        [window setContentSize: NSMakeSize(width, height + 36.)];
    }
    Py_RETURN_NONE;
}

static PyObject*
FigureManager_full_screen_toggle(FigureManager* self)
{
    [self->window toggleFullScreen: nil];
    Py_RETURN_NONE;
}

static PyTypeObject FigureManagerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_macosx.FigureManager",
    .tp_basicsize = sizeof(FigureManager),
    .tp_dealloc = (destructor)FigureManager_dealloc,
    .tp_repr = (reprfunc)FigureManager_repr,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_init = (initproc)FigureManager_init,
    .tp_new = (newfunc)FigureManager_new,
    .tp_doc = "A FigureManager object wraps a Cocoa NSWindow object.",
    .tp_methods = (PyMethodDef[]){  // All docstrings are inherited.
        {"_show",
         (PyCFunction)FigureManager__show,
         METH_NOARGS},
        {"_raise",
         (PyCFunction)FigureManager__raise,
         METH_NOARGS},
        {"destroy",
         (PyCFunction)FigureManager_destroy,
         METH_NOARGS},
        {"_set_window_mode",
         (PyCFunction)FigureManager__set_window_mode,
         METH_VARARGS,
         "Set the window open mode (system, tab, window)"},
        {"set_icon",
         (PyCFunction)FigureManager_set_icon,
         METH_STATIC | METH_VARARGS,
         "Set application icon"},
        {"set_window_title",
         (PyCFunction)FigureManager_set_window_title,
         METH_VARARGS},
        {"get_window_title",
         (PyCFunction)FigureManager_get_window_title,
         METH_NOARGS},
        {"resize",
         (PyCFunction)FigureManager_resize,
         METH_VARARGS},
        {"full_screen_toggle",
         (PyCFunction)FigureManager_full_screen_toggle,
         METH_NOARGS},
        {}  // sentinel
    },
};

@interface NavigationToolbar2Handler : NSObject
{   PyObject* toolbar;
    NSButton* panbutton;
    NSButton* zoombutton;
}
- (NavigationToolbar2Handler*)initWithToolbar:(PyObject*)toolbar;
- (void)installCallbacks:(SEL[7])actions forButtons:(NSButton*[7])buttons;
- (void)home:(id)sender;
- (void)back:(id)sender;
- (void)forward:(id)sender;
- (void)pan:(id)sender;
- (void)zoom:(id)sender;
- (void)configure_subplots:(id)sender;
- (void)save_figure:(id)sender;
@end

typedef struct {
    PyObject_HEAD
    NSTextView* messagebox;
    NavigationToolbar2Handler* handler;
    int height;
} NavigationToolbar2;

@implementation NavigationToolbar2Handler
- (NavigationToolbar2Handler*)initWithToolbar:(PyObject*)theToolbar
{
    [self init];
    toolbar = theToolbar;
    return self;
}

- (void)installCallbacks:(SEL[7])actions forButtons:(NSButton*[7])buttons
{
    int i;
    for (i = 0; i < 7; i++) {
        SEL action = actions[i];
        NSButton* button = buttons[i];
        [button setTarget: self];
        [button setAction: action];
        if (action == @selector(pan:)) { panbutton = button; }
        if (action == @selector(zoom:)) { zoombutton = button; }
    }
}

-(void)home:(id)sender { gil_call_method(toolbar, "home"); }
-(void)back:(id)sender { gil_call_method(toolbar, "back"); }
-(void)forward:(id)sender { gil_call_method(toolbar, "forward"); }

-(void)pan:(id)sender
{
    if ([sender state]) { [zoombutton setState:NO]; }
    gil_call_method(toolbar, "pan");
}

-(void)zoom:(id)sender
{
    if ([sender state]) { [panbutton setState:NO]; }
    gil_call_method(toolbar, "zoom");
}

-(void)configure_subplots:(id)sender { gil_call_method(toolbar, "configure_subplots"); }
-(void)save_figure:(id)sender { gil_call_method(toolbar, "save_figure"); }
@end

static PyObject*
NavigationToolbar2_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    lazy_init();
    NavigationToolbar2Handler* handler = [NavigationToolbar2Handler alloc];
    if (!handler) { return NULL; }
    NavigationToolbar2 *self = (NavigationToolbar2*)type->tp_alloc(type, 0);
    if (!self) {
        [handler release];
        return NULL;
    }
    self->handler = handler;
    return (PyObject*)self;
}

static int
NavigationToolbar2_init(NavigationToolbar2 *self, PyObject *args, PyObject *kwds)
{
    FigureCanvas* canvas;
    const char* images[7];
    const char* tooltips[7];

    const float gap = 2;
    const int height = 36;
    const int imagesize = 24;

    if (!PyArg_ParseTuple(args, "O!(sssssss)(sssssss)",
                &FigureCanvasType, &canvas,
                &images[0], &images[1], &images[2], &images[3],
                &images[4], &images[5], &images[6],
                &tooltips[0], &tooltips[1], &tooltips[2], &tooltips[3],
                &tooltips[4], &tooltips[5], &tooltips[6])) {
        return -1;
    }

    View* view = canvas->view;
    if (!view) {
        PyErr_SetString(PyExc_RuntimeError, "NSView* is NULL");
        return -1;
    }

    self->height = height;

    NSRect bounds = [view bounds];
    NSWindow* window = [view window];

    bounds.origin.y += height;
    [view setFrame: bounds];

    bounds.size.height += height;
    [window setContentSize: bounds.size];

    NSButton* buttons[7];
    SEL actions[7] = {@selector(home:),
                      @selector(back:),
                      @selector(forward:),
                      @selector(pan:),
                      @selector(zoom:),
                      @selector(configure_subplots:),
                      @selector(save_figure:)};
    NSButtonType buttontypes[7] = {NSButtonTypeMomentaryLight,
                                   NSButtonTypeMomentaryLight,
                                   NSButtonTypeMomentaryLight,
                                   NSButtonTypePushOnPushOff,
                                   NSButtonTypePushOnPushOff,
                                   NSButtonTypeMomentaryLight,
                                   NSButtonTypeMomentaryLight};

    NSRect rect;
    NSSize size;
    NSSize scale;

    rect = NSMakeRect(0, 0, imagesize, imagesize);
    rect = [window convertRectToBacking: rect];
    size = rect.size;
    scale = NSMakeSize(imagesize / size.width, imagesize / size.height);

    rect.size.width = 32;
    rect.size.height = 32;
    rect.origin.x = gap;
    rect.origin.y = 0.5*(height - rect.size.height);

    for (int i = 0; i < 7; i++) {
        NSString* filename = [NSString stringWithUTF8String: images[i]];
        NSString* tooltip = [NSString stringWithUTF8String: tooltips[i]];
        NSImage* image = [[NSImage alloc] initWithContentsOfFile: filename];
        buttons[i] = [[NSButton alloc] initWithFrame: rect];
        [image setSize: size];
        // Specify that it is a template image so the content tint
        // color gets updated with the system theme (dark/light)
        [image setTemplate: YES];
        [buttons[i] setBezelStyle: NSBezelStyleShadowlessSquare];
        [buttons[i] setButtonType: buttontypes[i]];
        [buttons[i] setImage: image];
        [buttons[i] scaleUnitSquareToSize: scale];
        [buttons[i] setImagePosition: NSImageOnly];
        [buttons[i] setToolTip: tooltip];
        [[window contentView] addSubview: buttons[i]];
        [buttons[i] release];
        [image release];
        rect.origin.x += rect.size.width + gap;
    }

    self->handler = [self->handler initWithToolbar: (PyObject*)self];
    [self->handler installCallbacks: actions forButtons: buttons];

    NSFont* font = [NSFont systemFontOfSize: 0.0];
    // rect.origin.x is now at the far right edge of the buttons
    // we want the messagebox to take up the rest of the toolbar area
    // Make it a zero-width box if we don't have enough room
    rect.size.width = fmax(bounds.size.width - rect.origin.x, 0);
    rect.origin.x = bounds.size.width - rect.size.width;
    NSTextView* messagebox = [[[NSTextView alloc] initWithFrame: rect] autorelease];
    messagebox.textContainer.maximumNumberOfLines = 2;
    messagebox.textContainer.lineBreakMode = NSLineBreakByTruncatingTail;
    messagebox.alignment = NSTextAlignmentRight;
    [messagebox setFont: font];
    [messagebox setDrawsBackground: NO];
    [messagebox setSelectable: NO];
    /* if selectable, the messagebox can become first responder,
     * which is not supposed to happen */
    [[window contentView] addSubview: messagebox];
    [messagebox release];
    [[window contentView] display];

    self->messagebox = messagebox;
    return 0;
}

static void
NavigationToolbar2_dealloc(NavigationToolbar2 *self)
{
    [self->handler release];
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject*
NavigationToolbar2_repr(NavigationToolbar2* self)
{
    return PyUnicode_FromFormat("NavigationToolbar2 object %p", (void*)self);
}

static PyObject*
NavigationToolbar2_set_message(NavigationToolbar2 *self, PyObject* args)
{
    const char* message;

    if (!PyArg_ParseTuple(args, "s", &message)) { return NULL; }

    NSTextView* messagebox = self->messagebox;

    if (messagebox) {
        NSString* text = [NSString stringWithUTF8String: message];
        [messagebox setString: text];

        // Adjust width and height with the window size and content
        NSRect rectWindow = [messagebox.superview frame];
        NSRect rect = [messagebox frame];
        // Entire region to the right of the buttons
        rect.size.width = rectWindow.size.width - rect.origin.x;
        [messagebox setFrame: rect];
        // We want to control the vertical position of
        // the rect by the content size to center it vertically
        [messagebox.layoutManager ensureLayoutForTextContainer: messagebox.textContainer];
        NSRect contentRect = [messagebox.layoutManager usedRectForTextContainer: messagebox.textContainer];
        rect.origin.y = 0.5 * (self->height - contentRect.size.height);
        rect.size.height = contentRect.size.height;
        [messagebox setFrame: rect];
        // Disable cursorRects so that the cursor doesn't get updated by events
        // in NSApp (like resizing TextViews), we want to handle the cursor
        // changes from within MPL with set_cursor() ourselves
        [[messagebox.superview window] disableCursorRects];
    }

    Py_RETURN_NONE;
}

static PyTypeObject NavigationToolbar2Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_macosx.NavigationToolbar2",
    .tp_basicsize = sizeof(NavigationToolbar2),
    .tp_dealloc = (destructor)NavigationToolbar2_dealloc,
    .tp_repr = (reprfunc)NavigationToolbar2_repr,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_init = (initproc)NavigationToolbar2_init,
    .tp_new = (newfunc)NavigationToolbar2_new,
    .tp_doc = "NavigationToolbar2",
    .tp_methods = (PyMethodDef[]){  // All docstrings are inherited.
        {"set_message",
         (PyCFunction)NavigationToolbar2_set_message,
         METH_VARARGS},
        {}  // sentinel
    },
};

static PyObject*
choose_save_file(PyObject* unused, PyObject* args)
{
    int result;
    const char* title;
    const char* directory;
    const char* default_filename;
    if (!PyArg_ParseTuple(args, "sss", &title, &directory, &default_filename)) {
        return NULL;
    }
    NSSavePanel* panel = [NSSavePanel savePanel];
    [panel setTitle: [NSString stringWithUTF8String: title]];
    [panel setDirectoryURL: [NSURL fileURLWithPath: [NSString stringWithUTF8String: directory]
                                       isDirectory: YES]];
    [panel setNameFieldStringValue: [NSString stringWithUTF8String: default_filename]];
    result = [panel runModal];
    if (result == NSModalResponseOK) {
        NSString *filename = [[panel URL] path];
        if (!filename) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to obtain filename");
            return 0;
        }
        return PyUnicode_FromString([filename UTF8String]);
    }
    Py_RETURN_NONE;
}

@implementation Window
- (Window*)initWithContentRect:(NSRect)rect styleMask:(unsigned int)mask backing:(NSBackingStoreType)bufferingType defer:(BOOL)deferCreation withManager: (PyObject*)theManager
{
    self = [super initWithContentRect: rect
                            styleMask: mask
                              backing: bufferingType
                                defer: deferCreation];
    manager = theManager;
    Py_INCREF(manager);
    return self;
}

- (NSRect)constrainFrameRect:(NSRect)rect toScreen:(NSScreen*)screen
{
    /* Allow window sizes larger than the screen */
    NSRect suggested = [super constrainFrameRect: rect toScreen: screen];
    const CGFloat difference = rect.size.height - suggested.size.height;
    suggested.origin.y -= difference;
    suggested.size.height += difference;
    return suggested;
}

- (BOOL)closeButtonPressed
{
    gil_call_method(manager, "_close_button_pressed");
    return YES;
}

- (void)close
{
    [super close];
    --FigureWindowCount;
    if (!FigureWindowCount) [NSApp stop: self];
    /* This is needed for show(), which should exit from [NSApp run]
     * after all windows are closed.
     */
    // For each new window, we have incremented the manager reference, so
    // we need to bring that down during close and not just dealloc.
    Py_DECREF(manager);
}
@end

@implementation View
- (View*)initWithFrame:(NSRect)rect
{
    self = [super initWithFrame: rect];
    rubberband = NSZeroRect;
    device_scale = 1;
    return self;
}

- (void)dealloc
{
    FigureCanvas* fc = (FigureCanvas*)canvas;
    if (fc) { fc->view = NULL; }
    [super dealloc];
}

- (void)setCanvas: (PyObject*)newCanvas
{
    canvas = newCanvas;
}

static void _buffer_release(void* info, const void* data, size_t size) {
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyBuffer_Release((Py_buffer *)info);
    free(info);
    PyGILState_Release(gstate);
}

static int _copy_agg_buffer(CGContextRef cr, PyObject *renderer)
{
    Py_buffer *buffer = malloc(sizeof(Py_buffer));

    if (PyObject_GetBuffer(renderer, buffer, PyBUF_CONTIG_RO) == -1) {
        PyErr_Print();
        return 1;
    }

    if (buffer->ndim != 3 || buffer->shape[2] != 4) {
        _buffer_release(buffer, NULL, 0);
        return 1;
    }

    const Py_ssize_t nrows = buffer->shape[0];
    const Py_ssize_t ncols = buffer->shape[1];
    const size_t bytesPerComponent = 1;
    const size_t bitsPerComponent = 8 * bytesPerComponent;
    const size_t nComponents = 4; /* red, green, blue, alpha */
    const size_t bitsPerPixel = bitsPerComponent * nComponents;
    const size_t bytesPerRow = nComponents * bytesPerComponent * ncols;

    CGColorSpaceRef colorspace = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    if (!colorspace) {
        _buffer_release(buffer, NULL, 0);
        return 1;
    }

    CGDataProviderRef provider = CGDataProviderCreateWithData(buffer,
                                                              buffer->buf,
                                                              buffer->len,
                                                              _buffer_release);
    if (!provider) {
        _buffer_release(buffer, NULL, 0);
        CGColorSpaceRelease(colorspace);
        return 1;
    }

    CGBitmapInfo bitmapInfo = kCGBitmapByteOrderDefault | kCGImageAlphaLast;
    CGImageRef bitmap = CGImageCreate(ncols,
                                      nrows,
                                      bitsPerComponent,
                                      bitsPerPixel,
                                      bytesPerRow,
                                      colorspace,
                                      bitmapInfo,
                                      provider,
                                      NULL,
                                      false,
                                      kCGRenderingIntentDefault);
    CGColorSpaceRelease(colorspace);
    CGDataProviderRelease(provider);

    if (!bitmap) {
        return 1;
    }

    CGFloat deviceScale = _get_device_scale(cr);
    CGContextSaveGState(cr);
    CGContextDrawImage(cr, CGRectMake(0, 0, ncols/deviceScale, nrows/deviceScale), bitmap);
    CGImageRelease(bitmap);
    CGContextRestoreGState(cr);

    return 0;
}

-(void)drawRect:(NSRect)rect
{
    PyObject* renderer = NULL;
    PyObject* renderer_buffer = NULL;

    PyGILState_STATE gstate = PyGILState_Ensure();

    CGContextRef cr = [[NSGraphicsContext currentContext] CGContext];

    if (!(renderer = PyObject_CallMethod(canvas, "get_renderer", ""))
        || !(renderer_buffer = PyObject_GetAttrString(renderer, "_renderer"))) {
        PyErr_Print();
        goto exit;
    }
    if (_copy_agg_buffer(cr, renderer_buffer)) {
        printf("copy_agg_buffer failed\n");
        goto exit;
    }
    if (!NSIsEmptyRect(rubberband)) {
        // We use bezier paths so we can stroke the outside with a dash
        // pattern alternating white/black with two separate paths offset
        // in phase.
        NSBezierPath *white_path = [NSBezierPath bezierPathWithRect: rubberband];
        NSBezierPath *black_path = [NSBezierPath bezierPathWithRect: rubberband];
        CGFloat dash_pattern[2] = {3, 3};
        [white_path setLineDash: dash_pattern count: 2 phase: 0];
        [black_path setLineDash: dash_pattern count: 2 phase: 3];
        [[NSColor whiteColor] setStroke];
        [white_path stroke];
        [[NSColor blackColor] setStroke];
        [black_path stroke];
    }

  exit:
    Py_XDECREF(renderer_buffer);
    Py_XDECREF(renderer);

    PyGILState_Release(gstate);
}

- (void)updateDevicePixelRatio:(double)scale
{
    PyObject* change = NULL;
    PyGILState_STATE gstate = PyGILState_Ensure();

    device_scale = scale;
    if (!(change = PyObject_CallMethod(canvas, "_set_device_pixel_ratio", "d", device_scale))) {
        PyErr_Print();
        goto exit;
    }
    if (PyObject_IsTrue(change)) {
        // Notify that there was a resize_event that took place
        process_event(
            "ResizeEvent", "{s:s, s:O}",
            "name", "resize_event", "canvas", canvas);
        gil_call_method(canvas, "draw_idle");
        [self setNeedsDisplay: YES];
    }

  exit:
    Py_XDECREF(change);

    PyGILState_Release(gstate);
}

- (void)windowDidChangeBackingProperties:(NSNotification *)notification
{
    Window* window = [notification object];

    [self updateDevicePixelRatio: [window backingScaleFactor]];
}

- (void)windowDidResize: (NSNotification*)notification
{
    int width, height;
    Window* window = [notification object];
    NSSize size = [[window contentView] frame].size;
    NSRect rect = [self frame];

    size.height -= rect.origin.y;
    width = size.width;
    height = size.height;

    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* result = PyObject_CallMethod(
            canvas, "resize", "ii", width, height);
    if (result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);
    [self setNeedsDisplay: YES];
}

- (void)windowWillClose:(NSNotification*)notification
{
    process_event(
        "CloseEvent", "{s:s, s:O}",
        "name", "close_event", "canvas", canvas);
}

- (BOOL)windowShouldClose:(NSNotification*)notification
{
    NSWindow* window = [self window];
    NSEvent* event = [NSEvent otherEventWithType: NSEventTypeApplicationDefined
                                        location: NSZeroPoint
                                   modifierFlags: 0
                                       timestamp: 0.0
                                    windowNumber: 0
                                         context: nil
                                         subtype: WINDOW_CLOSING
                                           data1: 0
                                           data2: 0];
    [NSApp postEvent: event atStart: true];
    if ([window respondsToSelector: @selector(closeButtonPressed)]) {
        BOOL closed = [((Window*) window) closeButtonPressed];
        /* If closed, the window has already been closed via the manager. */
        if (closed) { return NO; }
    }
    return YES;
}

- (void)mouseEntered:(NSEvent *)event
{
    int x, y;
    NSPoint location = [event locationInWindow];
    location = [self convertPoint: location fromView: nil];
    x = location.x * device_scale;
    y = location.y * device_scale;
    process_event(
        "LocationEvent", "{s:s, s:O, s:i, s:i, s:N}",
        "name", "figure_enter_event", "canvas", canvas, "x", x, "y", y,
        "modifiers", mpl_modifiers(event));
}

- (void)mouseExited:(NSEvent *)event
{
    int x, y;
    NSPoint location = [event locationInWindow];
    location = [self convertPoint: location fromView: nil];
    x = location.x * device_scale;
    y = location.y * device_scale;
    process_event(
        "LocationEvent", "{s:s, s:O, s:i, s:i, s:N}",
        "name", "figure_leave_event", "canvas", canvas, "x", x, "y", y,
        "modifiers", mpl_modifiers(event));
}

- (void)mouseDown:(NSEvent *)event
{
    int x, y;
    int button;
    int dblclick = 0;
    NSPoint location = [event locationInWindow];
    location = [self convertPoint: location fromView: nil];
    x = location.x * device_scale;
    y = location.y * device_scale;
    switch ([event type])
    {    case NSEventTypeLeftMouseDown:
         {   unsigned int modifier = [event modifierFlags];
             if (modifier & NSEventModifierFlagControl)
                 /* emulate a right-button click */
                 button = 3;
             else if (modifier & NSEventModifierFlagOption)
                 /* emulate a middle-button click */
                 button = 2;
             else
             {
                 button = 1;
                 if ([NSCursor currentCursor]==[NSCursor openHandCursor]) {
                     leftMouseGrabbing = true;
                     [[NSCursor closedHandCursor] set];
                 }
             }
             break;
         }
         case NSEventTypeOtherMouseDown: button = 2; break;
         case NSEventTypeRightMouseDown: button = 3; break;
         default: return; /* Unknown mouse event */
    }
    if ([event clickCount] == 2) {
      dblclick = 1;
    }
    process_event(
        "MouseEvent", "{s:s, s:O, s:i, s:i, s:i, s:i, s:N}",
        "name", "button_press_event", "canvas", canvas, "x", x, "y", y,
        "button", button, "dblclick", dblclick, "modifiers", mpl_modifiers(event));
}

- (void)mouseUp:(NSEvent *)event
{
    int button;
    int x, y;
    NSPoint location = [event locationInWindow];
    location = [self convertPoint: location fromView: nil];
    x = location.x * device_scale;
    y = location.y * device_scale;
    switch ([event type])
    {    case NSEventTypeLeftMouseUp:
             leftMouseGrabbing = false;
             button = 1;
             if ([NSCursor currentCursor]==[NSCursor closedHandCursor])
                 [[NSCursor openHandCursor] set];
             break;
         case NSEventTypeOtherMouseUp: button = 2; break;
         case NSEventTypeRightMouseUp: button = 3; break;
         default: return; /* Unknown mouse event */
    }
    process_event(
        "MouseEvent", "{s:s, s:O, s:i, s:i, s:i, s:N}",
        "name", "button_release_event", "canvas", canvas, "x", x, "y", y,
        "button", button, "modifiers", mpl_modifiers(event));
}

- (void)mouseMoved:(NSEvent *)event
{
    int x, y;
    NSPoint location = [event locationInWindow];
    location = [self convertPoint: location fromView: nil];
    x = location.x * device_scale;
    y = location.y * device_scale;
    process_event(
        "MouseEvent", "{s:s, s:O, s:i, s:i, s:N}",
        "name", "motion_notify_event", "canvas", canvas, "x", x, "y", y,
        "modifiers", mpl_modifiers(event));
}

- (void)mouseDragged:(NSEvent *)event
{
    int x, y;
    NSPoint location = [event locationInWindow];
    location = [self convertPoint: location fromView: nil];
    x = location.x * device_scale;
    y = location.y * device_scale;
    process_event(
        "MouseEvent", "{s:s, s:O, s:i, s:i, s:N}",
        "name", "motion_notify_event", "canvas", canvas, "x", x, "y", y,
        "modifiers", mpl_modifiers(event));
}

- (void)rightMouseDown:(NSEvent *)event { [self mouseDown: event]; }
- (void)rightMouseUp:(NSEvent *)event { [self mouseUp: event]; }
- (void)rightMouseDragged:(NSEvent *)event { [self mouseDragged: event]; }
- (void)otherMouseDown:(NSEvent *)event { [self mouseDown: event]; }
- (void)otherMouseUp:(NSEvent *)event { [self mouseUp: event]; }
- (void)otherMouseDragged:(NSEvent *)event { [self mouseDragged: event]; }

- (void)setRubberband:(NSRect)rect
{
    // The space we want to redraw is a union of the previous rubberband
    // with the new rubberband and then expanded (negative inset) by one
    // in each direction to account for the stroke linewidth.
    [self setNeedsDisplayInRect: NSInsetRect(NSUnionRect(rect, rubberband), -1, -1)];
    rubberband = rect;
}

- (void)removeRubberband
{
    if (NSIsEmptyRect(rubberband)) { return; }
    [self setNeedsDisplayInRect: rubberband];
    rubberband = NSZeroRect;
}

- (const char*)convertKeyEvent:(NSEvent*)event
{
    NSMutableString* returnkey = [NSMutableString string];
    if (keyChangeControl) {
        // When control is the key that was pressed, return the full word
        [returnkey appendString:@"control+"];
    } else if (([event modifierFlags] & NSEventModifierFlagControl)) {
        // If control is already pressed, return the shortened version
        [returnkey appendString:@"ctrl+"];
    }
    if (([event modifierFlags] & NSEventModifierFlagOption) || keyChangeOption) {
        [returnkey appendString:@"alt+" ];
    }
    if (([event modifierFlags] & NSEventModifierFlagCommand) || keyChangeCommand) {
        [returnkey appendString:@"cmd+" ];
    }
    // Don't print caps_lock unless it was the key that got pressed
    if (keyChangeCapsLock) {
        [returnkey appendString:@"caps_lock+" ];
    }

    // flagsChanged event can't handle charactersIgnoringModifiers
    // because it was a modifier key that was pressed/released
    if (event.type != NSEventTypeFlagsChanged) {
        NSString* specialchar;
        switch ([[event charactersIgnoringModifiers] characterAtIndex:0]) {
            case NSLeftArrowFunctionKey: specialchar = @"left"; break;
            case NSRightArrowFunctionKey: specialchar = @"right"; break;
            case NSUpArrowFunctionKey: specialchar = @"up"; break;
            case NSDownArrowFunctionKey: specialchar = @"down"; break;
            case NSF1FunctionKey: specialchar = @"f1"; break;
            case NSF2FunctionKey: specialchar = @"f2"; break;
            case NSF3FunctionKey: specialchar = @"f3"; break;
            case NSF4FunctionKey: specialchar = @"f4"; break;
            case NSF5FunctionKey: specialchar = @"f5"; break;
            case NSF6FunctionKey: specialchar = @"f6"; break;
            case NSF7FunctionKey: specialchar = @"f7"; break;
            case NSF8FunctionKey: specialchar = @"f8"; break;
            case NSF9FunctionKey: specialchar = @"f9"; break;
            case NSF10FunctionKey: specialchar = @"f10"; break;
            case NSF11FunctionKey: specialchar = @"f11"; break;
            case NSF12FunctionKey: specialchar = @"f12"; break;
            case NSF13FunctionKey: specialchar = @"f13"; break;
            case NSF14FunctionKey: specialchar = @"f14"; break;
            case NSF15FunctionKey: specialchar = @"f15"; break;
            case NSF16FunctionKey: specialchar = @"f16"; break;
            case NSF17FunctionKey: specialchar = @"f17"; break;
            case NSF18FunctionKey: specialchar = @"f18"; break;
            case NSF19FunctionKey: specialchar = @"f19"; break;
            case NSScrollLockFunctionKey: specialchar = @"scroll_lock"; break;
            case NSBreakFunctionKey: specialchar = @"break"; break;
            case NSInsertFunctionKey: specialchar = @"insert"; break;
            case NSDeleteFunctionKey: specialchar = @"delete"; break;
            case NSHomeFunctionKey: specialchar = @"home"; break;
            case NSEndFunctionKey: specialchar = @"end"; break;
            case NSPageDownFunctionKey: specialchar = @"pagedown"; break;
            case NSPageUpFunctionKey: specialchar = @"pageup"; break;
            case NSDeleteCharacter: specialchar = @"backspace"; break;
            case NSEnterCharacter: specialchar = @"enter"; break;
            case NSTabCharacter: specialchar = @"tab"; break;
            case NSCarriageReturnCharacter: specialchar = @"enter"; break;
            case NSBackTabCharacter: specialchar = @"backtab"; break;
            case 27: specialchar = @"escape"; break;
            default: specialchar = nil;
        }
        if (specialchar) {
            if (([event modifierFlags] & NSEventModifierFlagShift) || keyChangeShift) {
                [returnkey appendString:@"shift+"];
            }
            [returnkey appendString:specialchar];
        } else {
            [returnkey appendString:[event charactersIgnoringModifiers]];
        }
    } else {
        if (([event modifierFlags] & NSEventModifierFlagShift) || keyChangeShift) {
            [returnkey appendString:@"shift+"];
        }
        // Since it was a modifier event trim the final character of the string
        // because we added in "+" earlier
        [returnkey setString: [returnkey substringToIndex:[returnkey length] - 1]];
    }

    return [returnkey UTF8String];
}

- (void)keyDown:(NSEvent*)event
{
    const char* s = [self convertKeyEvent: event];
    NSPoint location = [[self window] mouseLocationOutsideOfEventStream];
    location = [self convertPoint: location fromView: nil];
    int x = location.x * device_scale,
        y = location.y * device_scale;
    if (s) {
        process_event(
            "KeyEvent", "{s:s, s:O, s:s, s:i, s:i}",
            "name", "key_press_event", "canvas", canvas, "key", s, "x", x, "y", y);
    } else {
        process_event(
            "KeyEvent", "{s:s, s:O, s:O, s:i, s:i}",
            "name", "key_press_event", "canvas", canvas, "key", Py_None, "x", x, "y", y);
    }
}

- (void)keyUp:(NSEvent*)event
{
    const char* s = [self convertKeyEvent: event];
    NSPoint location = [[self window] mouseLocationOutsideOfEventStream];
    location = [self convertPoint: location fromView: nil];
    int x = location.x * device_scale,
        y = location.y * device_scale;
    if (s) {
        process_event(
            "KeyEvent", "{s:s, s:O, s:s, s:i, s:i}",
            "name", "key_release_event", "canvas", canvas, "key", s, "x", x, "y", y);
    } else {
        process_event(
            "KeyEvent", "{s:s, s:O, s:O, s:i, s:i}",
            "name", "key_release_event", "canvas", canvas, "key", Py_None, "x", x, "y", y);
    }
}

- (void)scrollWheel:(NSEvent*)event
{
    int step;
    float d = [event deltaY];
    if (d > 0) { step = 1; }
    else if (d < 0) { step = -1; }
    else return;
    NSPoint location = [event locationInWindow];
    NSPoint point = [self convertPoint: location fromView: nil];
    int x = (int)round(point.x * device_scale);
    int y = (int)round(point.y * device_scale - 1);
    process_event(
        "MouseEvent", "{s:s, s:O, s:i, s:i, s:i, s:N}",
        "name", "scroll_event", "canvas", canvas,
        "x", x, "y", y, "step", step, "modifiers", mpl_modifiers(event));
}

- (BOOL)acceptsFirstResponder
{
    return YES;
}

// flagsChanged gets called whenever a  modifier key is pressed OR released
// so we need to handle both cases here
- (void)flagsChanged:(NSEvent *)event
{
    bool isPress = false; // true if key is pressed, false if key was released

    // Each if clause tests the two cases for each of the keys we can handle
    // 1. If the modifier flag "command key" is pressed and it was not previously
    // 2. If the modifier flag "command key" is not pressed and it was previously
    // !! converts the result of the bitwise & operator to a logical boolean,
    // which allows us to then bitwise xor (^) the result with a boolean (lastCommand).
    if (!!([event modifierFlags] & NSEventModifierFlagCommand) ^ lastCommand) {
        // Command pressed/released
        lastCommand = !lastCommand;
        keyChangeCommand = true;
        isPress = lastCommand;
    } else if (!!([event modifierFlags] & NSEventModifierFlagControl) ^ lastControl) {
        // Control pressed/released
        lastControl = !lastControl;
        keyChangeControl = true;
        isPress = lastControl;
    } else if (!!([event modifierFlags] & NSEventModifierFlagShift) ^ lastShift) {
        // Shift pressed/released
        lastShift = !lastShift;
        keyChangeShift = true;
        isPress = lastShift;
    } else if (!!([event modifierFlags] & NSEventModifierFlagOption) ^ lastOption) {
        // Option pressed/released
        lastOption = !lastOption;
        keyChangeOption = true;
        isPress = lastOption;
    } else if (!!([event modifierFlags] & NSEventModifierFlagCapsLock) ^ lastCapsLock) {
        // Capslock pressed/released
        lastCapsLock = !lastCapsLock;
        keyChangeCapsLock = true;
        isPress = lastCapsLock;
    } else {
        // flag we don't handle
        return;
    }

    if (isPress) {
        [self keyDown:event];
    } else {
        [self keyUp:event];
    }

    // Reset the state for the key changes after handling the event
    keyChangeCommand = false;
    keyChangeControl = false;
    keyChangeShift = false;
    keyChangeOption = false;
    keyChangeCapsLock = false;
}
@end

static PyObject*
show(PyObject* self)
{
    [NSApp activateIgnoringOtherApps: YES];
    NSArray *windowsArray = [NSApp windows];
    NSEnumerator *enumerator = [windowsArray objectEnumerator];
    NSWindow *window;
    while ((window = [enumerator nextObject])) {
        [window orderFront:nil];
    }
    Py_BEGIN_ALLOW_THREADS
    [NSApp run];
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

typedef struct {
    PyObject_HEAD
    NSTimer* timer;

} Timer;

static PyObject*
Timer_new(PyTypeObject* type, PyObject *args, PyObject *kwds)
{
    lazy_init();
    Timer* self = (Timer*)type->tp_alloc(type, 0);
    if (!self) {
        return NULL;
    }
    self->timer = NULL;
    return (PyObject*) self;
}

static PyObject*
Timer_repr(Timer* self)
{
    return PyUnicode_FromFormat("Timer object %p wrapping NSTimer %p",
                               (void*) self, (void*)(self->timer));
}

static PyObject*
Timer__timer_start(Timer* self, PyObject* args)
{
    NSTimeInterval interval;
    PyObject* py_interval = NULL, * py_single = NULL, * py_on_timer = NULL;
    int single;
    if (!(py_interval = PyObject_GetAttrString((PyObject*)self, "_interval"))
        || ((interval = PyFloat_AsDouble(py_interval) / 1000.), PyErr_Occurred())
        || !(py_single = PyObject_GetAttrString((PyObject*)self, "_single"))
        || ((single = PyObject_IsTrue(py_single)) == -1)
        || !(py_on_timer = PyObject_GetAttrString((PyObject*)self, "_on_timer"))) {
        goto exit;
    }
    if (!PyMethod_Check(py_on_timer)) {
        PyErr_SetString(PyExc_RuntimeError, "_on_timer should be a Python method");
        goto exit;
    }

    // hold a reference to the timer so we can invalidate/stop it later
    self->timer = [NSTimer scheduledTimerWithTimeInterval: interval
                                            repeats: !single
                                              block: ^(NSTimer *timer) {
        gil_call_method((PyObject*)self, "_on_timer");
    }];
exit:
    Py_XDECREF(py_interval);
    Py_XDECREF(py_single);
    Py_XDECREF(py_on_timer);
    if (PyErr_Occurred()) {
        return NULL;
    } else {
        Py_RETURN_NONE;
    }
}

static void
Timer__timer_stop_impl(Timer* self)
{
    if (self->timer) {
        [self->timer invalidate];
        self->timer = NULL;
    }
}

static PyObject*
Timer__timer_stop(Timer* self)
{
    Timer__timer_stop_impl(self);
    Py_RETURN_NONE;
}

static void
Timer_dealloc(Timer* self)
{
    Timer__timer_stop_impl(self);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyTypeObject TimerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_macosx.Timer",
    .tp_basicsize = sizeof(Timer),
    .tp_dealloc = (destructor)Timer_dealloc,
    .tp_repr = (reprfunc)Timer_repr,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = (newfunc)Timer_new,
    .tp_doc = "A Timer object that contains an NSTimer that gets added to the event loop when started.",
    .tp_methods = (PyMethodDef[]){  // All docstrings are inherited.
        {"_timer_start",
         (PyCFunction)Timer__timer_start,
         METH_VARARGS},
        {"_timer_stop",
         (PyCFunction)Timer__timer_stop,
         METH_NOARGS},
        {}  // sentinel
    },
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_macosx", "Mac OS X native backend", -1,
    (PyMethodDef[]){
        {"event_loop_is_running",
         (PyCFunction)event_loop_is_running,
         METH_NOARGS,
         "Return whether the OSX backend has set up the NSApp main event loop."},
        {"wake_on_fd_write",
         (PyCFunction)wake_on_fd_write,
         METH_VARARGS,
         "Arrange for Python to invoke its signal handlers when (any) data is\n"
         "written on the file descriptor given as argument."},
        {"stop",
         (PyCFunction)stop,
         METH_NOARGS,
         "Stop the NSApp."},
        {"show",
         (PyCFunction)show,
         METH_NOARGS,
         "Show all the figures and enter the main loop.\n"
         "\n"
         "This function does not return until all Matplotlib windows are closed,\n"
         "and is normally not needed in interactive sessions."},
        {"choose_save_file",
         (PyCFunction)choose_save_file,
         METH_VARARGS,
         "Query the user for a location where to save a file."},
        {}  /* Sentinel */
    },
};

#pragma GCC visibility push(default)

PyObject* PyInit__macosx(void)
{
    PyObject *m;
    if (!(m = PyModule_Create(&moduledef))
        || prepare_and_add_type(&FigureCanvasType, m)
        || prepare_and_add_type(&FigureManagerType, m)
        || prepare_and_add_type(&NavigationToolbar2Type, m)
        || prepare_and_add_type(&TimerType, m)) {
        Py_XDECREF(m);
        return NULL;
    }
    return m;
}

#pragma GCC visibility pop
