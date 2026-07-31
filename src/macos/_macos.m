#define PY_SSIZE_T_CLEAN
#import <Cocoa/Cocoa.h>
#import <ApplicationServices/ApplicationServices.h>
#import <Python.h>
#import "MPLUtils.h"
#import "MPLAppDelegate.h"
#import "MPLFigureCanvas.h"
#import "MPLFigureManager.h"
#import "MPLNavigationToolbar2.h"

#if !__has_feature(objc_arc_fields)
#error "The macOS backend requires ARC C struct fields support (objc_arc_fields)."
#endif

/* Various NSApplicationDefined event subtypes */
#define STOP_EVENT_LOOP 2


/* When calling into Objective-C from Python, wrap the calls with
   BEGIN_OBJC_ENTRY and END_OBJC_ENTRY. This will set up an autorelease
   pool as well as catch any Obj-C exceptions thrown. These macros
   should be used for any call exposed to Python via the external module
   interface.

   To avoid undefined behavior, each END_OBJC_ENTRY should be followed
   by a return statement which handles the rare case when an Objective-C
   exception was thrown.

   As a convenience, the RETURN_NULL_OR_NONE macro can be used for functions
   that return a PyObject */
#define BEGIN_OBJC_ENTRY \
    @autoreleasepool { @try {

#define END_OBJC_ENTRY \
    } @catch (NSException *e) { errSetException(e); } }

#define RETURN_NULL_OR_NONE \
    if (PyErr_Occurred()) { \
        return NULL; \
    } else { \
        Py_RETURN_NONE; \
    }


/* Variable for our delegate since it needs a +1 reference count. */
static id<NSApplicationDelegate> appDelegate = nil;

/* Variables to keep track of state and window count for show() */
static BOOL IsRunningFromShow = NO;
static NSHashTable<NSWindow *> *FigureWindowHashTable = nil;

// Global variable to store the original SIGINT handler
static PyOS_sighandler_t originalSigintAction = NULL;

// Convert an Objective-C exception into a Python RuntimeError
static void errSetException(NSException *exception) {
    PyErr_SetString(PyExc_RuntimeError, [[exception reason] UTF8String]);
}

// Stop the current app's run loop, sending an event to ensure it actually stops
static void stopWithEvent(void) {
    [NSApp stop: nil];
    // Post an event to trigger the actual stopping.
    // +[NSEvent otherEventWithType:...] is declared nullable but will not return
    // nil for these constant, valid arguments; guard defensively anyway.
    NSEvent* event = [NSEvent otherEventWithType: NSEventTypeApplicationDefined
                                        location: NSZeroPoint
                                   modifierFlags: 0
                                       timestamp: 0
                                    windowNumber: 0
                                         context: nil
                                         subtype: 0
                                           data1: 0
                                           data2: 0];
    if (event) {
        [NSApp postEvent: event atStart: YES];
    }
}

// Signal handler for SIGINT, only argument matching for stopWithEvent
static void handleSigint(int signal) {
    stopWithEvent();
}

// Helper function to flush all events.
// This is needed in some instances to ensure e.g. that windows are properly closed.
// It is used in the input hook as well as wrapped in a version callable from Python.
static void flushEvents(void) {
    while (true) {
        @autoreleasepool {
            NSEvent* event = [NSApp nextEventMatchingMask: NSEventMaskAny
                                                untilDate: [NSDate distantPast]
                                                   inMode: NSDefaultRunLoopMode
                                                  dequeue: YES];
            if (!event) {
                break;
            }
            [NSApp sendEvent:event];
        }
    }
}

static int wait_for_stdin(void) {
    BEGIN_OBJC_ENTRY

    // Short circuit if no windows are active
    // Rely on Python's input handling to manage CPU usage
    // This queries the NSApp, rather than using our FigureWindowHashTable because that is modified when events still
    // need to be processed to properly close the windows.
    @autoreleasepool {
        if (![[NSApp windows] count]) {
            flushEvents();
            return 1;
        }
    }

    // Set up a SIGINT handler to interrupt the event loop if ctrl+c comes in too
    originalSigintAction = PyOS_setsig(SIGINT, handleSigint);

    // Create an NSFileHandle for standard input
    NSFileHandle *stdinHandle = [NSFileHandle fileHandleWithStandardInput];


    // Register for data available notifications on standard input
    id notificationID = [[NSNotificationCenter defaultCenter] addObserverForName: NSFileHandleDataAvailableNotification
                                                                          object: stdinHandle
                                                                           queue: [NSOperationQueue mainQueue] // Use the main queue
                                                                      usingBlock: ^(NSNotification *notification) {stopWithEvent();}
    ];

    // Wait in the background for anything that happens to stdin
    [stdinHandle waitForDataInBackgroundAndNotify];

    // Run the application's event loop, which will be interrupted on stdin or SIGINT
    [NSApp run];

    // Remove the input handler as an observer
    [[NSNotificationCenter defaultCenter] removeObserver: notificationID];


    // Restore the original SIGINT handler upon exiting the function
    PyOS_setsig(SIGINT, originalSigintAction);

    return 1;

    END_OBJC_ENTRY
    return 0;
}


/* ---------------------------- Python classes ---------------------------- */


static bool backend_inited = false;

static void lazy_init(void) {
    if (backend_inited) { return; }
    backend_inited = true;

    NSApp = [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    appDelegate = [[MPLAppDelegate alloc] init];
    [NSApp setDelegate:appDelegate];

    // Run our own event loop while waiting for stdin on the Python side
    // this is needed to keep the application responsive while waiting for input
    PyOS_InputHook = wait_for_stdin;
}

static PyObject *
event_loop_is_running(PyObject *self)
{
    BEGIN_OBJC_ENTRY

    if (backend_inited) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }

    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
wake_on_fd_write(PyObject *unused, PyObject *args)
{
    BEGIN_OBJC_ENTRY
    int fd;
    if (!PyArg_ParseTuple(args, "i", &fd)) { return NULL; }
    NSFileHandle* fh = [[NSFileHandle alloc] initWithFileDescriptor: fd];
    __block id notificationID = [[NSNotificationCenter defaultCenter]
        addObserverForName: NSFileHandleDataAvailableNotification
                    object: fh
                     queue: nil
                usingBlock: ^(NSNotification* note) {
                    NSFileHandle *strongFileHandle __attribute__((unused)) = fh;
                    PyGILState_STATE gstate = PyGILState_Ensure();
                    PyErr_CheckSignals();
                    PyGILState_Release(gstate);
                    [[NSNotificationCenter defaultCenter] removeObserver:notificationID];
                }];
    [fh waitForDataInBackgroundAndNotify];
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
stop(PyObject *self, PyObject *unused)
{
    BEGIN_OBJC_ENTRY
    stopWithEvent();
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}


#pragma mark - FigureCanvas Type

typedef struct {
    PyObject_HEAD
    __strong MPLFigureCanvas *object;
} FigureCanvas;

static PyTypeObject FigureCanvasType;

static PyObject *
FigureCanvas_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    BEGIN_OBJC_ENTRY

    lazy_init();
    return (PyObject *)((FigureCanvas *)type->tp_alloc(type, 0));

    END_OBJC_ENTRY
    return NULL;
}

static int
FigureCanvas_init(FigureCanvas *self, PyObject *args, PyObject *kwds)
{
    BEGIN_OBJC_ENTRY
    MPLFigureCanvas *wrappedObject;
    NSTrackingArea *trackingArea;
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
    if (!(wh = PyObject_CallMethod((PyObject *)self, "get_width_height", ""))
            || !PyArg_ParseTuple(wh, "ii", &width, &height)) {
        goto exit;
    }
    NSRect rect = NSMakeRect(0.0, 0.0, width, height);
    wrappedObject = [[MPLFigureCanvas alloc] initWithFrame: rect];
    wrappedObject.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    int opts = (NSTrackingMouseEnteredAndExited | NSTrackingMouseMoved |
                NSTrackingActiveInKeyWindow | NSTrackingInVisibleRect);
    trackingArea = [[NSTrackingArea alloc] initWithRect: rect
                                                options: opts
                                                  owner: wrappedObject
                                               userInfo: nil];
    [wrappedObject addTrackingArea:trackingArea];
    self->object = wrappedObject;
    [self->object setPyObject:(PyObject *)self];

exit:
    Py_XDECREF(super_obj);
    Py_XDECREF(super_init);
    Py_XDECREF(init_res);
    Py_XDECREF(wh);

    END_OBJC_ENTRY
    return PyErr_Occurred() ? -1 : 0;
}

static void
FigureCanvas_dealloc(FigureCanvas *self)
{
    BEGIN_OBJC_ENTRY
    [self->object setPyObject:NULL];
    self->object = nil;
    END_OBJC_ENTRY
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
FigureCanvas_repr(FigureCanvas *self)
{
    return PyUnicode_FromFormat("FigureCanvas<%p> wrapping MPLFigureCanvas<%p>",
                                (void *)self, (__bridge void *)self->object);
}

static PyObject *
FigureCanvas_update(FigureCanvas *self)
{
    BEGIN_OBJC_ENTRY
    [self->object setNeedsDisplay: YES];
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE;
}

static PyObject *
FigureCanvas_flush_events(FigureCanvas *self)
{
    BEGIN_OBJC_ENTRY
    // We run the app, matching any events that are waiting in the queue
    // to process, breaking out of the loop when no events remain and
    // displaying the canvas if needed.
    Py_BEGIN_ALLOW_THREADS

    flushEvents();

    Py_END_ALLOW_THREADS

    [self->object displayIfNeeded];
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject * __attribute__((unused))
FigureCanvas_request_idle_draw(FigureCanvas *self)
{
    BEGIN_OBJC_ENTRY
    // Will be implemented
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
FigureCanvas_set_cursor(FigureCanvas *self, PyObject *args)
{
    BEGIN_OBJC_ENTRY
    int i;
    if (!PyArg_ParseTuple(args, "i", &i)) { return NULL; }
    switch (i) {
      case 1: [[NSCursor arrowCursor] set]; break;
      case 2: [[NSCursor pointingHandCursor] set]; break;
      case 3: [[NSCursor crosshairCursor] set]; break;
      case 4:
        if (mpl_leftMouseGrabbing) {
            [[NSCursor closedHandCursor] set];
        } else {
            [[NSCursor openHandCursor] set];
        }
        break;
      /* macOS handles busy state itself so no need to set a cursor here */
      case 5: break;
      case 6: [[NSCursor resizeLeftRightCursor] set]; break;
      case 7: [[NSCursor resizeUpDownCursor] set]; break;
      default: return NULL;
    }
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
FigureCanvas_set_rubberband(FigureCanvas *self, PyObject *args)
{
    BEGIN_OBJC_ENTRY
    MPLFigureCanvas *figureCanvas = self->object;
    if (!figureCanvas) {
        PyErr_SetString(PyExc_RuntimeError, "MPLFigureCanvas* is NULL");
        return NULL;
    }
    int x0, y0, x1, y1;
    if (!PyArg_ParseTuple(args, "iiii", &x0, &y0, &x1, &y1)) {
        return NULL;
    }
    x0 /= figureCanvas->device_scale;
    x1 /= figureCanvas->device_scale;
    y0 /= figureCanvas->device_scale;
    y1 /= figureCanvas->device_scale;
    NSRect rubberband = NSMakeRect(x0 < x1 ? x0 : x1, y0 < y1 ? y0 : y1,
                                   abs(x1 - x0), abs(y1 - y0));
    [figureCanvas setRubberband: rubberband];
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
FigureCanvas_remove_rubberband(FigureCanvas *self)
{
    BEGIN_OBJC_ENTRY
    [self->object removeRubberband];
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
FigureCanvas__start_event_loop(FigureCanvas *self, PyObject *args, PyObject *keywords)
{
    BEGIN_OBJC_ENTRY
    float timeout = 0.0;

    static char *kwlist[] = {"timeout", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "f", kwlist, &timeout)) {
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS

    NSDate *date =
        (timeout > 0.0) ? [NSDate dateWithTimeIntervalSinceNow: timeout]
                        : [NSDate distantFuture];
    while (true) {
        @autoreleasepool {
            NSEvent *event = [NSApp nextEventMatchingMask: NSEventMaskAny
                                                untilDate: date
                                                   inMode: NSDefaultRunLoopMode
                                                  dequeue: YES];
            if (!event || [event type]==NSEventTypeApplicationDefined) { break; }
            [NSApp sendEvent: event];
        }
    }

    Py_END_ALLOW_THREADS

    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
FigureCanvas_stop_event_loop(FigureCanvas *self)
{
    BEGIN_OBJC_ENTRY
    // +[NSEvent otherEventWithType:...] is declared nullable but will not return
    // nil for these constant, valid arguments; guard defensively anyway.
    NSEvent* event = [NSEvent otherEventWithType: NSEventTypeApplicationDefined
                                        location: NSZeroPoint
                                   modifierFlags: 0
                                       timestamp: 0.0
                                    windowNumber: 0
                                         context: nil
                                         subtype: STOP_EVENT_LOOP
                                           data1: 0
                                           data2: 0];
    if (event) {
        [NSApp postEvent: event atStart: true];
    }
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyTypeObject FigureCanvasType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "matplotlib.backends._macosx.FigureCanvas",
    .tp_doc = PyDoc_STR("A FigureCanvas object wraps a Cocoa NSView object."),
    .tp_basicsize = sizeof(FigureCanvas),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,

    .tp_new = (newfunc)FigureCanvas_new,
    .tp_init = (initproc)FigureCanvas_init,
    .tp_dealloc = (destructor)FigureCanvas_dealloc,
    .tp_repr = (reprfunc)FigureCanvas_repr,

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
         PyDoc_STR("Set the active cursor.")},
        {"set_rubberband",
         (PyCFunction)FigureCanvas_set_rubberband,
         METH_VARARGS,
         PyDoc_STR("Specify a new rubberband rectangle and invalidate it.")},
        {"remove_rubberband",
         (PyCFunction)FigureCanvas_remove_rubberband,
         METH_NOARGS,
         PyDoc_STR("Remove the current rubberband rectangle.")},
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


#pragma mark - FigureManager Type

static PyTypeObject FigureManagerType;  // forward declaration, needed in destroy()

typedef struct {
    PyObject_HEAD
    __strong Window *object;
} FigureManager;

static PyObject *
FigureManager_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    BEGIN_OBJC_ENTRY
    if (![NSThread isMainThread]) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "Cannot create a GUI FigureManager outside the main thread "
            "using the MacOS backend. Use a non-interactive "
            "backend like 'agg' to make plots on worker threads."
        );
        return NULL;
    }

    lazy_init();
    return (PyObject *)((FigureManager *)type->tp_alloc(type, 0));

    END_OBJC_ENTRY
    return NULL;
}

static int
FigureManager_init(FigureManager *self, PyObject *args, PyObject *kwds)
{
    BEGIN_OBJC_ENTRY
    PyObject *canvas;
    if (!PyArg_ParseTuple(args, "O", &canvas)) {
        return -1;
    }

    MPLFigureCanvas *figureCanvas = ((FigureCanvas*)canvas)->object;
    if (!figureCanvas) {  /* Something really weird going on */
        PyErr_SetString(PyExc_RuntimeError, "MPLFigureCanvas* is NULL");
        return -1;
    }

    PyObject *size = PyObject_CallMethod(canvas, "get_width_height", "");
    int width, height;
    if (!size || !PyArg_ParseTuple(size, "ii", &width, &height)) {
        Py_XDECREF(size);
        return -1;
    }
    Py_DECREF(size);

    NSRect rect = NSMakeRect( /* x */ 100, /* y */ 350, width, height);

    Window* window = [[Window alloc] initWithContentRect: rect
                                               styleMask: NSWindowStyleMaskTitled
                                                        | NSWindowStyleMaskClosable
                                                        | NSWindowStyleMaskResizable
                                                        | NSWindowStyleMaskMiniaturizable
                                                 backing: NSBackingStoreBuffered
                                                   defer: YES];
    [window setDelegate: figureCanvas];
    [window makeFirstResponder: figureCanvas];
    [window setReleasedWhenClosed:NO];
    [[window contentView] addSubview: figureCanvas];
    [figureCanvas updateDevicePixelRatio: [window backingScaleFactor]];

    self->object = window;
    [self->object setPyObject:(PyObject *)self];

    if (!FigureWindowHashTable) {
        FigureWindowHashTable = [NSHashTable weakObjectsHashTable];
    }
    [FigureWindowHashTable addObject:window];

    END_OBJC_ENTRY
    return 0;
}

static PyObject * __attribute__((unused))
FigureManager__set_window_appearance(FigureManager *self, PyObject *args)
{
    BEGIN_OBJC_ENTRY
    // Will be implemented
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
FigureManager__set_window_mode(FigureManager *self, PyObject *args)
{
    BEGIN_OBJC_ENTRY
    const char *window_mode;
    if (!PyArg_ParseTuple(args, "s", &window_mode) || !self->object) {
        return NULL;
    }

    NSString* window_mode_str = [NSString stringWithUTF8String: window_mode];
    if ([window_mode_str isEqualToString: @"tab"]) {
        [self->object setTabbingMode: NSWindowTabbingModePreferred];
    } else if ([window_mode_str isEqualToString: @"window"]) {
        [self->object setTabbingMode: NSWindowTabbingModeDisallowed];
    } else { // system settings
        [self->object setTabbingMode: NSWindowTabbingModeAutomatic];
    }
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
FigureManager_repr(FigureManager *self)
{
    return PyUnicode_FromFormat("FigureManager<%p> wrapping Window<%p>",
                                (void *)self, (__bridge void *)self->object);
}

static void
FigureManager__closeAndClearWindow(FigureManager *self)
{
    if (self->object) {
        [self->object close];
        [self->object setDelegate:nil];
        [self->object setPyObject:NULL];
        [FigureWindowHashTable removeObject:self->object];
        self->object = nil;

        if ([FigureWindowHashTable count] == 0 && IsRunningFromShow) {
            [NSApp stop:nil];
        }
    }
}

static void
FigureManager_dealloc(FigureManager *self)
{
    BEGIN_OBJC_ENTRY
    FigureManager__closeAndClearWindow(self);
    END_OBJC_ENTRY
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
FigureManager__show(FigureManager *self)
{
    BEGIN_OBJC_ENTRY
    [self->object makeKeyAndOrderFront: nil];
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
FigureManager__raise(FigureManager *self)
{
    BEGIN_OBJC_ENTRY
    [self->object orderFrontRegardless];
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
FigureManager_destroy(FigureManager *self)
{
    BEGIN_OBJC_ENTRY
    FigureManager__closeAndClearWindow(self);

    // call super(self, FigureManager).destroy() - it seems we need the
    // explicit arguments, and just super() doesn't work in the C API.
    PyObject *super_obj = PyObject_CallFunctionObjArgs(
        (PyObject *)&PySuper_Type,
        (PyObject *)&FigureManagerType,
        self,
        NULL
    );
    if (super_obj == NULL) {
        return NULL; // error
    }
    PyObject *result = PyObject_CallMethod(super_obj, "destroy", NULL);
    Py_DECREF(super_obj);
    if (result == NULL) {
        return NULL; // error
    }
    Py_DECREF(result);

    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
FigureManager_set_icon(PyObject *null, PyObject *args) {
    BEGIN_OBJC_ENTRY
    PyObject* icon_path;
    if (!PyArg_ParseTuple(args, "O&", &PyUnicode_FSDecoder, &icon_path)) {
        return NULL;
    }
    const char* icon_path_ptr = PyUnicode_AsUTF8(icon_path);
    if (!icon_path_ptr) {
        Py_DECREF(icon_path);
        return NULL;
    }

    NSString* ns_icon_path = [NSString stringWithUTF8String: icon_path_ptr];
    Py_DECREF(icon_path);
    if (!ns_icon_path) {
        PyErr_SetString(PyExc_RuntimeError, "Could not convert to NSString*");
        return NULL;
    }
    NSImage* image = [[NSImage alloc] initByReferencingFile: ns_icon_path];
    if (!image) {
        PyErr_SetString(PyExc_RuntimeError, "Could not create NSImage*");
        return NULL;
    }
    if (!image.valid) {
        PyErr_SetString(PyExc_RuntimeError, "Image is not valid");
        return NULL;
    }

    NSApplication* app = [NSApplication sharedApplication];
    app.applicationIconImage = image;

    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
FigureManager_set_window_title(FigureManager* self,
                               PyObject *args, PyObject *kwds)
{
    BEGIN_OBJC_ENTRY
    const char* title;
    if (!PyArg_ParseTuple(args, "s", &title)) {
        return NULL;
    }
    // PyArg_ParseTuple "s" guarantees valid UTF-8, so stringWithUTF8String: will
    // not return nil here; the nullable annotation is a false positive.
    // NOLINTNEXTLINE(clang-analyzer-nullability.NullablePassedToNonnull)
    [self->object setTitle: [NSString stringWithUTF8String: title]];
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
FigureManager_get_window_title(FigureManager *self)
{
    BEGIN_OBJC_ENTRY
    NSString *title = [self->object title];
    if (title) {
        return PyUnicode_FromString([title UTF8String]);
    }
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
FigureManager_resize(FigureManager *self, PyObject *args, PyObject *kwds)
{
    BEGIN_OBJC_ENTRY
    int width, height;
    if (!PyArg_ParseTuple(args, "ii", &width, &height)) {
        return NULL;
    }
    Window* window = self->object;
    if (window) {
        CGFloat device_pixel_ratio = [window backingScaleFactor];
        width /= device_pixel_ratio;
        height /= device_pixel_ratio;
        // 36 comes from hard-coded size of toolbar later in code
        [window setContentSize: NSMakeSize(width, height + 36.)];
    }
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
FigureManager_full_screen_toggle(FigureManager *self)
{
    BEGIN_OBJC_ENTRY
    [self->object toggleFullScreen: nil];
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyTypeObject FigureManagerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "matplotlib.backends._macosx.FigureManager",
    .tp_doc = PyDoc_STR("A FigureManager object wraps a Cocoa NSWindow object."),
    .tp_basicsize = sizeof(FigureManager),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,

    .tp_new = (newfunc)FigureManager_new,
    .tp_init = (initproc)FigureManager_init,
    .tp_dealloc = (destructor)FigureManager_dealloc,
    .tp_repr = (reprfunc)FigureManager_repr,

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
         PyDoc_STR("Set the window open mode (system, tab, window)")},
        {"set_icon",
         (PyCFunction)FigureManager_set_icon,
         METH_STATIC | METH_VARARGS,
         PyDoc_STR("Set application icon")},
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


#pragma mark - NavigationToolbar2 Type

typedef struct {
    PyObject_HEAD
    __strong NSTextView *messagebox;
    __strong MPLNavigationToolbar2 *object;
    int height;
} NavigationToolbar2;

static PyObject *
NavigationToolbar2_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    BEGIN_OBJC_ENTRY
    lazy_init();
    NavigationToolbar2 *self = (NavigationToolbar2 *)type->tp_alloc(type, 0);
    return (PyObject *)self;
    END_OBJC_ENTRY
    return NULL;
}

static int
NavigationToolbar2_init(NavigationToolbar2 *self, PyObject *args, PyObject *kwds)
{
    BEGIN_OBJC_ENTRY
    FigureCanvas *canvas;
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

    MPLFigureCanvas *figureCanvas = canvas->object;
    if (!figureCanvas) {
        PyErr_SetString(PyExc_RuntimeError, "MPLFigureCanvas* is NULL");
        return -1;
    }

    self->height = height;

    NSRect bounds = [figureCanvas bounds];
    NSWindow* window = [figureCanvas window];

    bounds.origin.y += height;
    [figureCanvas setFrame: bounds];

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
        // PyArg_ParseTuple "s" guarantees valid UTF-8; stringWithUTF8String: will not return nil.
        NSString* filename = [NSString stringWithUTF8String: images[i]];
        NSString* tooltip = [NSString stringWithUTF8String: tooltips[i]];
        // NOLINTNEXTLINE(clang-analyzer-nullability.NullablePassedToNonnull)
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
        rect.origin.x += rect.size.width + gap;
    }

    MPLNavigationToolbar2 *wrappedObject;
    wrappedObject = [[MPLNavigationToolbar2 alloc] init];
    [wrappedObject setPyObject:(PyObject*)self];
    [wrappedObject installCallbacks: actions forButtons: buttons];

    NSFont* font = [NSFont systemFontOfSize: 0.0];
    // rect.origin.x is now at the far right edge of the buttons
    // we want the messagebox to take up the rest of the toolbar area
    // Make it a zero-width box if we don't have enough room
    rect.size.width = fmax(bounds.size.width - rect.origin.x, 0);
    rect.origin.x = bounds.size.width - rect.size.width;
    NSTextView* messagebox = [[NSTextView alloc] initWithFrame: rect];
    messagebox.textContainer.maximumNumberOfLines = 2;
    messagebox.textContainer.lineBreakMode = NSLineBreakByTruncatingTail;
    messagebox.alignment = NSTextAlignmentRight;
    [messagebox setFont: font];
    [messagebox setDrawsBackground: NO];
    [messagebox setSelectable: NO];
    /* if selectable, the messagebox can become first responder,
     * which is not supposed to happen */
    [[window contentView] addSubview: messagebox];
    [[window contentView] display];

    self->object = wrappedObject;
    self->messagebox = messagebox;
    END_OBJC_ENTRY
    return 0;
}

static void
NavigationToolbar2_dealloc(NavigationToolbar2 *self)
{
    BEGIN_OBJC_ENTRY
    [self->object setPyObject:NULL];
    self->object = nil;
    self->messagebox = nil;
    END_OBJC_ENTRY
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
NavigationToolbar2_repr(NavigationToolbar2* self)
{
    return PyUnicode_FromFormat("NavigationToolbar2<%p> wrapping MPLNavigationToolbar2<%p>",
                                (void *)self, (__bridge void *)self->object);
}

static PyObject *
NavigationToolbar2_set_message(NavigationToolbar2 *self, PyObject *args)
{
    BEGIN_OBJC_ENTRY
    const char *message;

    if (!PyArg_ParseTuple(args, "s", &message)) { return NULL; }

    NSTextView* messagebox = self->messagebox;

    if (messagebox) {
        // PyArg_ParseTuple "s" guarantees valid UTF-8; stringWithUTF8String: will not return nil.
        NSString* text = [NSString stringWithUTF8String: message];
        // NOLINTNEXTLINE(clang-analyzer-nullability.NullablePassedToNonnull)
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

    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyTypeObject NavigationToolbar2Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "matplotlib.backends._macosx.NavigationToolbar2",
    .tp_doc = PyDoc_STR("NavigationToolbar2"),
    .tp_basicsize = sizeof(NavigationToolbar2),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,

    .tp_new = (newfunc)NavigationToolbar2_new,
    .tp_init = (initproc)NavigationToolbar2_init,
    .tp_dealloc = (destructor)NavigationToolbar2_dealloc,
    .tp_repr = (reprfunc)NavigationToolbar2_repr,

    .tp_methods = (PyMethodDef[]){  // All docstrings are inherited.
        {"set_message",
         (PyCFunction)NavigationToolbar2_set_message,
         METH_VARARGS},
        {}  // sentinel
    },
};

static PyObject *
choose_save_file(PyObject *unused, PyObject *args)
{
    BEGIN_OBJC_ENTRY

    int result;
    const char* title;
    const char* directory;
    const char* default_filename;
    if (!PyArg_ParseTuple(args, "sss", &title, &directory, &default_filename)) {
        return NULL;
    }
    NSSavePanel* panel = [NSSavePanel savePanel];
    [panel setTitle: [NSString stringWithUTF8String: title]];
    // PyArg_ParseTuple "s" guarantees valid UTF-8; stringWithUTF8String: will not return nil.
    // NOLINTNEXTLINE(clang-analyzer-nullability.NullablePassedToNonnull)
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

    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
show(PyObject *self)
{
    BEGIN_OBJC_ENTRY

    // Iterating over -[NSApp windows] will add the windows to the topmost
    // autorelease pool, wrap in @autoreleasepool as -[NSApp run] is long-running.
    @autoreleasepool {
        [NSApp activateIgnoringOtherApps: YES];

        for (NSWindow *window in [FigureWindowHashTable allObjects]) {
            [window orderFront:nil];
        }
    }

    Py_BEGIN_ALLOW_THREADS
    IsRunningFromShow = YES;
    [NSApp run];
    IsRunningFromShow = NO;
    Py_END_ALLOW_THREADS

    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}


#pragma mark - Timer Type

typedef struct {
    PyObject_HEAD
    __strong NSTimer *timer;
    BOOL shouldInvalidate;
} Timer;

static PyObject *
Timer_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    BEGIN_OBJC_ENTRY
    lazy_init();
    Timer *self = (Timer *)type->tp_alloc(type, 0);
    return (PyObject *)self;
    END_OBJC_ENTRY
    return NULL;
}

static PyObject *
Timer_repr(Timer *self)
{
    return PyUnicode_FromFormat("Timer<%p> wrapping NSTimer<%p>",
                                (void *)self, (__bridge void *)self->timer);
}

static void
Timer__timer_stop_impl(Timer *self)
{
    if (self->shouldInvalidate) {
        [self->timer invalidate];
        self->shouldInvalidate = NO;
    }
    self->timer = nil;
}

static PyObject *
Timer__timer_start(Timer *self, PyObject *args)
{
    BEGIN_OBJC_ENTRY
    NSTimer *timer;
    NSTimeInterval interval;
    PyObject *py_interval = NULL, *py_single = NULL, *py_on_timer = NULL;
    int single;
    if (!(py_interval = PyObject_GetAttrString((PyObject *)self, "_interval"))
        || ((void)((interval = PyFloat_AsDouble(py_interval) / 1000.)), PyErr_Occurred())
        || !(py_single = PyObject_GetAttrString((PyObject *)self, "_single"))
        || ((single = PyObject_IsTrue(py_single)) == -1)
        || !(py_on_timer = PyObject_GetAttrString((PyObject *)self, "_on_timer"))) {
        goto exit;
    }
    if (!PyMethod_Check(py_on_timer)) {
        PyErr_SetString(PyExc_RuntimeError, "_on_timer should be a Python method");
        goto exit;
    }

    // Stop any previous timers if start() was called multiple times
    Timer__timer_stop_impl(self);

    // hold a reference to the timer so we can invalidate/stop it later
    timer = [NSTimer timerWithTimeInterval: interval
                                   repeats: !single
                                     block: ^(NSTimer *timer) {
        gil_call_method((PyObject *)self, "_on_timer");
        if (single) {
            // A single-shot timer will be automatically invalidated when it fires, so
            // we shouldn't do it ourselves when the object is deleted.
            self->shouldInvalidate = NO;
        }
    }];

    // Schedule the timer on the main run loop which is needed
    // when updating the UI from a background thread
    [[NSRunLoop mainRunLoop] addTimer: timer forMode: NSRunLoopCommonModes];

    self->timer = timer;
    self->shouldInvalidate = YES;

exit:
    Py_XDECREF(py_interval);
    Py_XDECREF(py_single);
    Py_XDECREF(py_on_timer);
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
Timer__timer_stop(Timer *self)
{
    BEGIN_OBJC_ENTRY
    Timer__timer_stop_impl(self);
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static void
Timer_dealloc(Timer *self)
{
    BEGIN_OBJC_ENTRY
    Timer__timer_stop_impl(self);
    END_OBJC_ENTRY
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyTypeObject TimerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "matplotlib.backends._macosx.Timer",
    .tp_doc = PyDoc_STR("A Timer object that contains an NSTimer that gets added to "
                        "the event loop when started."),
    .tp_basicsize = sizeof(Timer),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,

    .tp_new = (newfunc)Timer_new,
    .tp_dealloc = (destructor)Timer_dealloc,
    .tp_repr = (reprfunc)Timer_repr,

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
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_macosx",
    .m_doc = PyDoc_STR("Mac OS X native backend"),
    .m_size = -1,
    .m_methods = (PyMethodDef[]){
        {"event_loop_is_running",
         (PyCFunction)event_loop_is_running,
         METH_NOARGS,
         PyDoc_STR(
            "Return whether the macosx backend has set up the NSApp main event loop.")},
        {"wake_on_fd_write",
         (PyCFunction)wake_on_fd_write,
         METH_VARARGS,
         PyDoc_STR(
            "Arrange for Python to invoke its signal handlers when (any) data is\n"
            "written on the file descriptor given as argument.")},
        {"stop",
         (PyCFunction)stop,
         METH_VARARGS,
         PyDoc_STR("Stop the NSApp.")},
        {"show",
         (PyCFunction)show,
         METH_NOARGS,
         PyDoc_STR(
            "Show all the figures and enter the main loop.\n"
            "\n"
            "This function does not return until all Matplotlib windows are closed,\n"
            "and is normally not needed in interactive sessions.")},
        {"choose_save_file",
         (PyCFunction)choose_save_file,
         METH_VARARGS,
         PyDoc_STR("Query the user for a location where to save a file.")},
        {}  /* Sentinel */
    },
};

#pragma GCC visibility push(default)

PyMODINIT_FUNC
PyInit__macos(void)
{
    PyObject *m;
    if (!(m = PyModule_Create(&moduledef))
        || PyModule_AddType(m, &FigureCanvasType)
        || PyModule_AddType(m, &FigureManagerType)
        || PyModule_AddType(m, &NavigationToolbar2Type)
        || PyModule_AddType(m, &TimerType)) {
        Py_XDECREF(m);
        return NULL;
    }
#ifdef Py_GIL_DISABLED
    PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif
    return m;
}

#pragma GCC visibility pop
