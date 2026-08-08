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
    } @catch (NSException *e) { sErrSetException(e); } }

#define RETURN_NULL_OR_NONE \
    if (PyErr_Occurred()) { \
        return NULL; \
    } else { \
        Py_RETURN_NONE; \
    }


/* Variable for our delegate since it needs a +1 reference count. */
static id<NSApplicationDelegate> sAppDelegate = nil;

/* Variables to keep track of state and window count for show() */
static BOOL sIsRunningFromShow = NO;
static NSHashTable<MPLFigureManager *> *sFigureManagerHashTable = nil;

// Set to YES in _init() if initialization was successful
static BOOL sIsInitialized = NO;

// Convert an Objective-C exception into a Python RuntimeError
static void sErrSetException(NSException *exception)
{
    const char *cString = [[exception reason] UTF8String];
    if (!cString) cString = [[exception name] UTF8String];
    if (!cString) cString = "Objective-C Exception";

    PyErr_SetString(PyExc_RuntimeError, cString);
}


// Old implementation, goes away with MPLEventLoop PR
static void sStopWithEvent(void)
{
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


// Old implementation, goes away with MPLEventLoop PR
static void sHandleSigint(int signal)
{
    sStopWithEvent();
}

// Old implementation, goes away with MPLEventLoop PR
static void flushEvents(void)
{
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

// Old implementation, goes away with MPLEventLoop PR
static int sInputHook(void)
{
    BEGIN_OBJC_ENTRY

    // Short circuit if no windows are active
    // Rely on Python's input handling to manage CPU usage
    // This queries the NSApp, rather than using our FigureWindowCount because that is decremented when events still
    // need to be processed to properly close the windows.
    @autoreleasepool {
        if (![[NSApp windows] count]) {
            flushEvents();
            return 1;
        }
    }

    // Set up a SIGINT handler to interrupt the event loop if ctrl+c comes in too
    PyOS_sighandler_t originalSigintHandler = PyOS_setsig(SIGINT, sHandleSigint);

    // Create an NSFileHandle for standard input
    NSFileHandle *stdinHandle = [NSFileHandle fileHandleWithStandardInput];


    // Register for data available notifications on standard input
    id notificationID = [[NSNotificationCenter defaultCenter] addObserverForName: NSFileHandleDataAvailableNotification
                                                                          object: stdinHandle
                                                                           queue: [NSOperationQueue mainQueue] // Use the main queue
                                                                      usingBlock: ^(NSNotification *notification) {sStopWithEvent();}
    ];

    // Wait in the background for anything that happens to stdin
    [stdinHandle waitForDataInBackgroundAndNotify];

    // Run the application's event loop, which will be interrupted on stdin or SIGINT
    [NSApp run];

    // Remove the input handler as an observer
    [[NSNotificationCenter defaultCenter] removeObserver: notificationID];


    // Restore the original SIGINT handler upon exiting the function
    PyOS_setsig(SIGINT, originalSigintHandler);

    return 1;

    END_OBJC_ENTRY
    return 0;
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
    return (PyObject *)((FigureCanvas *)type->tp_alloc(type, 0));
}

static int
FigureCanvas_init(FigureCanvas *self, PyObject *args, PyObject *kwds)
{
    BEGIN_OBJC_ENTRY

    int width, height;
    if (!PyArg_ParseTuple(args, "ii", &width, &height)) {
        return -1;
    }

    NSRect rect = NSMakeRect(0.0, 0.0, width, height);
    self->object = [[MPLFigureCanvas alloc] initWithFrame: rect];
    [self->object setPyObject:(PyObject *)self];

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
FigureCanvas_update_layer_contents(FigureCanvas *self, PyObject *args)
{
    BEGIN_OBJC_ENTRY
    PyObject *bufferPyObject;
    if (!PyArg_ParseTuple(args, "O", &bufferPyObject)) { return NULL; }

    ssize_t shape[3];
    NSData *buffer = MPLGetBufferWithPyObject(bufferPyObject, 3, shape);
    if (!buffer) { return NULL; }

    if (shape[0] <= 0 || shape[1] <= 0 || shape[2] != 4) {
        PyErr_SetString(PyExc_RuntimeError, "Unexpected buffer shape");
        return NULL;
    }

    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)buffer);

    if (!provider) {
        PyErr_SetString(PyExc_RuntimeError, "CGDataProviderCreateWithCFData() failed");
        return NULL;
    }

    [self->object updateLayerContentsWithDataProvider: provider
                                          deviceWidth: (size_t)shape[1]
                                         deviceHeight: (size_t)shape[0]];

    CGDataProviderRelease(provider);

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

    [self->object requestDisplayLayerWithNeedsDraw:NO];
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
FigureCanvas_request_display_layer(FigureCanvas *self, PyObject *args)
{
    BEGIN_OBJC_ENTRY

    int needsDraw;
    if (!PyArg_ParseTuple(args, "p", &needsDraw)) { return NULL; }

    [self->object requestDisplayLayerWithNeedsDraw:(needsDraw > 0)];
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
FigureCanvas_set_cursor(FigureCanvas *self, PyObject *args)
{
    BEGIN_OBJC_ENTRY

    int cursorType;
    if (!PyArg_ParseTuple(args, "i", &cursorType)) {
        return NULL;
    }

    [self->object updateCursorType:cursorType];

    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
FigureCanvas_set_rubberband(FigureCanvas *self, PyObject *args)
{
    BEGIN_OBJC_ENTRY

    int x0, y0, x1, y1;
    if (!PyArg_ParseTuple(args, "iiii", &x0, &y0, &x1, &y1)) {
        return NULL;
    }

    [self->object updateRubberbandWithDeviceX0:x0 y0:y0 x1:x1 y1:y1];

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
    .tp_name = "matplotlib.backends._macos.FigureCanvas",
    .tp_doc = PyDoc_STR("A FigureCanvas object wraps a Cocoa NSView object."),
    .tp_basicsize = sizeof(FigureCanvas),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,

    .tp_new = (newfunc)FigureCanvas_new,
    .tp_init = (initproc)FigureCanvas_init,
    .tp_dealloc = (destructor)FigureCanvas_dealloc,
    .tp_repr = (reprfunc)FigureCanvas_repr,

    .tp_methods = (PyMethodDef[]){
        {"_update_layer_contents",
         (PyCFunction)FigureCanvas_update_layer_contents,
         METH_VARARGS,
         NULL},  // docstring inherited
        {"flush_events",
         (PyCFunction)FigureCanvas_flush_events,
         METH_NOARGS,
         NULL},  // docstring inherited
        {"_request_display_layer",
         (PyCFunction)FigureCanvas_request_display_layer,
         METH_VARARGS,
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
    __strong MPLFigureManager *object;
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

    return (PyObject *)((FigureManager *)type->tp_alloc(type, 0));

    END_OBJC_ENTRY
    return NULL;
}

static int
FigureManager_init(FigureManager *self, PyObject *args, PyObject *kwds)
{
    BEGIN_OBJC_ENTRY
    PyObject *figureCanvasPyObject;
    if (!PyArg_ParseTuple(args, "O!", &FigureCanvasType, &figureCanvasPyObject)) {
        return -1;
    }

    MPLFigureCanvas *figureCanvas = ((FigureCanvas *)figureCanvasPyObject)->object;

    self->object = [[MPLFigureManager alloc] initWithFigureCanvas:figureCanvas];
    [self->object setPyObject:(PyObject *)self];

    if (!sFigureManagerHashTable) {
        sFigureManagerHashTable = [NSHashTable weakObjectsHashTable];
    }
    [sFigureManagerHashTable addObject:self->object];

    END_OBJC_ENTRY
    return PyErr_Occurred() ? -1 : 0;
}

static PyObject *
FigureManager__set_window_appearance(FigureManager *self, PyObject *args)
{
    BEGIN_OBJC_ENTRY
    [self->object updateWindowAppearance:MPLGetStringWithPySequence(args)];
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
FigureManager__set_window_mode(FigureManager *self, PyObject *args)
{
    BEGIN_OBJC_ENTRY
    [self->object updateWindowMode:MPLGetStringWithPySequence(args)];
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
FigureManager_repr(FigureManager *self)
{
    return PyUnicode_FromFormat("FigureManager<%p> wrapping MPLFigureManager<%p>",
                                (void *)self, (__bridge void *)self->object);
}

static void
FigureManager__close_and_clear_window_impl(FigureManager *self)
{
    if (self->object) {
        [sFigureManagerHashTable removeObject:self->object];

        [self->object setPyObject:NULL];
        [self->object close];
        self->object = nil;

        if ([sFigureManagerHashTable count] == 0 && sIsRunningFromShow) {
            [NSApp stop:nil];
        }
    }
}

static void
FigureManager_dealloc(FigureManager *self)
{
    BEGIN_OBJC_ENTRY
    FigureManager__close_and_clear_window_impl(self);
    END_OBJC_ENTRY
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
FigureManager__show(FigureManager *self)
{
    BEGIN_OBJC_ENTRY
    [self->object show];
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
FigureManager__raise(FigureManager *self)
{
    BEGIN_OBJC_ENTRY
    [self->object raise];
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
FigureManager__close_and_clear_window(FigureManager *self)
{
    BEGIN_OBJC_ENTRY
    FigureManager__close_and_clear_window_impl(self);
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
FigureManager_set_window_title(FigureManager *self,
                               PyObject *args, PyObject *kwds)
{
    BEGIN_OBJC_ENTRY
    NSString *title = MPLGetStringWithPySequence(args);
    if (title) [self->object setWindowTitle:title];
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
FigureManager_get_window_title(FigureManager *self)
{
    BEGIN_OBJC_ENTRY
    NSString *title = [self->object windowTitle];
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
    [self->object resizeToDeviceWidth:width height:height];
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
FigureManager_full_screen_toggle(FigureManager *self)
{
    BEGIN_OBJC_ENTRY
    [self->object toggleFullScreen];
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyTypeObject FigureManagerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "matplotlib.backends._macos.FigureManager",
    .tp_doc = PyDoc_STR("A FigureManager object wraps a "
                        "MPLFigureManager Objective-C object."),
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
        {"_close_and_clear_window",
         (PyCFunction)FigureManager__close_and_clear_window,
         METH_NOARGS},
        {"_set_window_appearance",
         (PyCFunction)FigureManager__set_window_appearance,
         METH_VARARGS,
         PyDoc_STR("Set the window appearance (system, light, dark)")},
        {"_set_window_mode",
         (PyCFunction)FigureManager__set_window_mode,
         METH_VARARGS,
         PyDoc_STR("Set the window open mode (system, tab, window)")},
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
    __strong MPLNavigationToolbar2 *object;
} NavigationToolbar2;

static PyObject *
NavigationToolbar2_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    return (PyObject *)((NavigationToolbar2*)type->tp_alloc(type, 0));
}

static int
NavigationToolbar2_init(NavigationToolbar2 *self, PyObject *args, PyObject *kwds)
{
    BEGIN_OBJC_ENTRY

    PyObject *figureCanvasPyObject;
    if (!PyArg_ParseTuple(args, "O!", &FigureCanvasType, &figureCanvasPyObject)) {
        return -1;
    }

    MPLFigureCanvas *figureCanvas = ((FigureCanvas *)figureCanvasPyObject)->object;

    if (!figureCanvas) {
        PyErr_SetString(PyExc_RuntimeError, "MPLFigureCanvas is NULL");
        return -1;
    }

    if ([[figureCanvas manager] toolbar]) {
        PyErr_SetString(PyExc_RuntimeError, "MPLFigureManager already has a toolbar");
        return -1;
    }

    MPLNavigationToolbar2 *toolbar = [[MPLNavigationToolbar2 alloc] init];
    [toolbar setPyObject:(PyObject *)self];
    self->object = toolbar;

    [[figureCanvas manager] installToolbar:toolbar];

    END_OBJC_ENTRY
    return PyErr_Occurred() ? -1 : 0;
}

static void
NavigationToolbar2_dealloc(NavigationToolbar2 *self)
{
    BEGIN_OBJC_ENTRY
    [self->object setPyObject:NULL];
    self->object = nil;
    END_OBJC_ENTRY
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
NavigationToolbar2_repr(NavigationToolbar2 *self)
{
    return PyUnicode_FromFormat("NavigationToolbar2<%p> wrapping MPLNavigationToolbar2<%p>",
                                (void *)self, (__bridge void *)self->object);
}

static PyObject *
NavigationToolbar2_add_item(NavigationToolbar2 *self, PyObject *args)
{
    BEGIN_OBJC_ENTRY

    MPLStringArray *strings = MPLGetStringArrayWithPySequence(args);
    if (!strings) return NULL;

    if ([strings count] != 4) {
        PyErr_SetString(PyExc_RuntimeError, "Invalid arguments to add_item");
        return NULL;
    }

    [self->object addItemWithTitle: [strings objectAtIndex:0]
                           tooltip: [strings objectAtIndex:1]
                         imagePath: [strings objectAtIndex:2]
                      callbackName: [strings objectAtIndex:3]];

    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
NavigationToolbar2_add_separator(NavigationToolbar2 *self, PyObject *args)
{
    BEGIN_OBJC_ENTRY
    [self->object addSeparator];
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
NavigationToolbar2_update_selected_item(NavigationToolbar2 *self, PyObject *args)
{
    BEGIN_OBJC_ENTRY
    NSString *callbackName = MPLGetStringWithPySequence(args);
    if (callbackName) [self->object updateSelectedItem:callbackName];
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
NavigationToolbar2_update_history_items(NavigationToolbar2 *self, PyObject *args)
{
    BEGIN_OBJC_ENTRY
    int backEnabled, forwardEnabled;
    if (PyArg_ParseTuple(args, "ii", &backEnabled, &forwardEnabled)) {
        [self->object updateHistoryItemsWithBackEnabled: (backEnabled > 0)
                                         forwardEnabled: (forwardEnabled > 0)];
    }
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
NavigationToolbar2_set_message(NavigationToolbar2 *self, PyObject *args)
{
    BEGIN_OBJC_ENTRY
    NSString *message = MPLGetStringWithPySequence(args);
    if (message) [self->object updateMessage:message];
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyTypeObject NavigationToolbar2Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "matplotlib.backends._macos.NavigationToolbar2",
    .tp_doc = PyDoc_STR("NavigationToolbar2"),
    .tp_basicsize = sizeof(NavigationToolbar2),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,

    .tp_new = (newfunc)NavigationToolbar2_new,
    .tp_init = (initproc)NavigationToolbar2_init,
    .tp_dealloc = (destructor)NavigationToolbar2_dealloc,
    .tp_repr = (reprfunc)NavigationToolbar2_repr,

    .tp_methods = (PyMethodDef[]){
        {"add_item",
         (PyCFunction)NavigationToolbar2_add_item,
         METH_VARARGS,
         PyDoc_STR("Adds an item to the toolbar.")},
        {"add_separator",
         (PyCFunction)NavigationToolbar2_add_separator,
         METH_NOARGS,
         PyDoc_STR("Adds a separator to the toolbar.")},
        {"update_selected_item",
         (PyCFunction)NavigationToolbar2_update_selected_item,
         METH_VARARGS,
         PyDoc_STR("Selects the item with the specified callback name.")},
        {"update_history_items",
         (PyCFunction)NavigationToolbar2_update_history_items,
         METH_VARARGS,
         PyDoc_STR("Sets the enabled status of the back/forward items")},
        {"set_message",
         (PyCFunction)NavigationToolbar2_set_message,
         METH_VARARGS},
        {}  // sentinel
    },
};


#pragma mark - Timer Type

typedef struct {
    PyObject_HEAD
    __strong NSTimer *timer;
    BOOL shouldInvalidate;
} Timer;

static PyObject *
Timer_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    return (PyObject *)((Timer*)type->tp_alloc(type, 0));
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
        MPLCallMethod((PyObject *)self, "_on_timer", "");
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
    .tp_name = "matplotlib.backends._macos.Timer",
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


#pragma mark - Module

static PyObject *
_macos__init(PyObject *unused, PyObject *args)
{
    BEGIN_OBJC_ENTRY

    PyObject *imagesPyDict;
    if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &imagesPyDict)) { return NULL; }

    NSDictionary *imagesDictionary = MPLGetStringDictionaryWithPyDict(imagesPyDict);
    if (!imagesDictionary) { return NULL; }

    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        if (!NSApp) {
            NSApp = [NSApplication sharedApplication];
        }

        if (![NSApp delegate]) {
            sAppDelegate = [[MPLAppDelegate alloc] initWithImageDictionary:imagesDictionary];
            [NSApp setDelegate:sAppDelegate];
        }

        sIsInitialized = YES;

        // Run our own event loop while waiting for stdin on the Python side
        // this is needed to keep the application responsive while waiting for input
        PyOS_InputHook = sInputHook;
    });

    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
_macos_is_initialized(PyObject *self)
{
    if (sIsInitialized) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}


static PyObject*
_macos_wake_on_fd_write(PyObject* unused, PyObject* args)
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
                    NSFileHandle* strongFileHandle __attribute__((unused)) = fh;
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
_macos_stop(PyObject *self, PyObject *unused)
{
    BEGIN_OBJC_ENTRY
    sStopWithEvent();
    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
_macos_show(PyObject *self)
{
    BEGIN_OBJC_ENTRY

    // Iterating over sFigureManagerHashTable will add the managers to the topmost
    // autorelease pool, wrap in @autoreleasepool as -[NSApp run] is long-running.
    @autoreleasepool {
        [NSApp activateIgnoringOtherApps: YES];
        for (MPLFigureManager *manager in [sFigureManagerHashTable allObjects]) {
            [manager raise];
        }
    }

    if ([NSApp isRunning]) {
        PyErr_SetString(PyExc_RuntimeError, "An event loop is already running");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    sIsRunningFromShow = YES;
    [NSApp run];
    sIsRunningFromShow = NO;
    Py_END_ALLOW_THREADS

    END_OBJC_ENTRY
    RETURN_NULL_OR_NONE
}

static PyObject *
_macos_choose_save_file(PyObject *unused, PyObject *args)
{
    BEGIN_OBJC_ENTRY

    MPLStringArray *strings = MPLGetStringArrayWithPySequence(args);
    if ([strings count] != 3) {
        PyErr_SetString(PyExc_RuntimeError, "Invalid arguments to choose_save_file");
        return NULL;
    }

    NSString *title = [strings objectAtIndex:0];
    NSString *directory = [strings objectAtIndex:1];
    NSString *defaultFilename = [strings objectAtIndex:2];

    NSSavePanel *panel = [NSSavePanel savePanel];
    [panel setTitle:title];
    [panel setDirectoryURL:[NSURL fileURLWithPath:directory isDirectory:YES]];
    [panel setNameFieldStringValue:defaultFilename];

    __block NSModalResponse modalResponse;
    modalResponse = [panel runModal];

    if (modalResponse == NSModalResponseOK) {
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


static int
_macos_mod_exec(PyObject *m)
{
    static BOOL sLoaded = NO;

    // Use an os_unfair_lock as PyMutex requires Python >= 3.13
    static os_unfair_lock sLoadedLock = OS_UNFAIR_LOCK_INIT;

    BOOL wasModuleAlreadyLoaded = NO;

    os_unfair_lock_lock(&sLoadedLock);
    wasModuleAlreadyLoaded = sLoaded;
    sLoaded = YES;
    os_unfair_lock_unlock(&sLoadedLock);

    if (wasModuleAlreadyLoaded) {
        PyErr_SetString(PyExc_ImportError,
                        "cannot load module more than once per process");
        return -1;
    }

    if (PyModule_AddType(m, &FigureCanvasType)
        || PyModule_AddType(m, &FigureManagerType)
        || PyModule_AddType(m, &NavigationToolbar2Type)
        || PyModule_AddType(m, &TimerType)) {
        return -1;
    }
    return 0;
}

static struct PyModuleDef _macos_moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_macos",
    .m_doc = PyDoc_STR("macOS native backend"),
    .m_size = 0,
    .m_slots = (PyModuleDef_Slot[]){
        {Py_mod_exec, _macos_mod_exec},
        {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
#ifdef Py_GIL_DISABLED
        {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
        {0, NULL}
    },
    .m_methods = (PyMethodDef[]){
        {"_init",
         (PyCFunction)_macos__init,
         METH_VARARGS,
         PyDoc_STR(
            "Perform a one-time initialization of the backend. Sets up the NSApp delegate"
            "if one is not already present.")},
        {"is_initialized",
         (PyCFunction)_macos_is_initialized,
         METH_NOARGS,
         PyDoc_STR(
            "Return whether _init() has been called .")},
        {"wake_on_fd_write",
         (PyCFunction)_macos_wake_on_fd_write,
         METH_VARARGS,
         PyDoc_STR(
            "Arrange for Python to invoke its signal handlers when (any) data is\n"
            "written on the file descriptor given as argument.")},
        {"stop",
         (PyCFunction)_macos_stop,
         METH_VARARGS,
         PyDoc_STR("Stop the NSApp.")},
        {"show",
         (PyCFunction)_macos_show,
         METH_NOARGS,
         PyDoc_STR(
            "Show all the figures and enter the main loop.\n"
            "\n"
            "This function does not return until all Matplotlib windows are closed,\n"
            "and is normally not needed in interactive sessions.")},
        {"choose_save_file",
         (PyCFunction)_macos_choose_save_file,
         METH_VARARGS,
         PyDoc_STR("Query the user for a location where to save a file.")},
        {}  /* Sentinel */
    },
};

#pragma GCC visibility push(default)

PyMODINIT_FUNC
PyInit__macos(void)
{
    return PyModuleDef_Init(&_macos_moduledef);
}

#pragma GCC visibility pop
