#define PY_SSIZE_T_CLEAN
#include <Cocoa/Cocoa.h>
#include <ApplicationServices/ApplicationServices.h>
#include <sys/socket.h>
#include <Python.h>

#ifndef PYPY
/* Remove this once Python is fixed: https://bugs.python.org/issue23237 */
#define PYOSINPUTHOOK_REPETITIVE 1
#endif

/* Proper way to check for the OS X version we are compiling for, from
 * https://developer.apple.com/library/archive/documentation/DeveloperTools/Conceptual/cross_development/Using/using.html
 */
#if __MAC_OS_X_VERSION_MIN_REQUIRED >= 1070
#define COMPILING_FOR_10_7
#endif

/* Renamed symbols cause deprecation warnings, so define macros for the new
 * names if we are compiling on an older SDK */
#if __MAC_OS_X_VERSION_MIN_REQUIRED < 101000
#define NSModalResponseOK                    NSOKButton
#endif
#if __MAC_OS_X_VERSION_MIN_REQUIRED < 101200
#define NSEventMaskAny                       NSAnyEventMask
#define NSEventTypeApplicationDefined        NSApplicationDefined
#define NSEventModifierFlagCommand           NSCommandKeyMask
#define NSEventModifierFlagControl           NSControlKeyMask
#define NSEventModifierFlagOption            NSAlternateKeyMask
#define NSEventModifierFlagShift             NSShiftKeyMask
#define NSEventTypeKeyUp                     NSKeyUp
#define NSEventTypeKeyDown                   NSKeyDown
#define NSEventTypeMouseMoved                NSMouseMoved
#define NSEventTypeLeftMouseDown             NSLeftMouseDown
#define NSEventTypeRightMouseDown            NSRightMouseDown
#define NSEventTypeOtherMouseDown            NSOtherMouseDown
#define NSEventTypeLeftMouseDragged          NSLeftMouseDragged
#define NSEventTypeRightMouseDragged         NSRightMouseDragged
#define NSEventTypeOtherMouseDragged         NSOtherMouseDragged
#define NSEventTypeLeftMouseUp               NSLeftMouseUp
#define NSEventTypeRightMouseUp              NSRightMouseUp
#define NSEventTypeOtherMouseUp              NSOtherMouseUp
#define NSWindowStyleMaskClosable            NSClosableWindowMask
#define NSWindowStyleMaskMiniaturizable      NSMiniaturizableWindowMask
#define NSWindowStyleMaskResizable           NSResizableWindowMask
#define NSWindowStyleMaskTitled              NSTitledWindowMask
#endif
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

/* -------------------------- Helper function ---------------------------- */

static void
_stdin_callback(CFReadStreamRef stream, CFStreamEventType eventType, void* info)
{
    CFRunLoopRef runloop = info;
    CFRunLoopStop(runloop);
}

static int sigint_fd = -1;

static void _sigint_handler(int sig)
{
    const char c = 'i';
    write(sigint_fd, &c, 1);
}

static void _sigint_callback(CFSocketRef s,
                             CFSocketCallBackType type,
                             CFDataRef address,
                             const void * data,
                             void *info)
{
    char c;
    int* interrupted = info;
    CFSocketNativeHandle handle = CFSocketGetNative(s);
    CFRunLoopRef runloop = CFRunLoopGetCurrent();
    read(handle, &c, 1);
    *interrupted = 1;
    CFRunLoopStop(runloop);
}

static CGEventRef _eventtap_callback(
    CGEventTapProxy proxy, CGEventType type, CGEventRef event, void *refcon)
{
    CFRunLoopRef runloop = refcon;
    CFRunLoopStop(runloop);
    return event;
}

static int wait_for_stdin(void)
{
    int interrupted = 0;
    const UInt8 buffer[] = "/dev/fd/0";
    const CFIndex n = (CFIndex)strlen((char*)buffer);
    CFRunLoopRef runloop = CFRunLoopGetCurrent();
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(kCFAllocatorDefault,
                                                           buffer,
                                                           n,
                                                           false);
    CFReadStreamRef stream = CFReadStreamCreateWithFile(kCFAllocatorDefault,
                                                        url);
    CFRelease(url);

    CFReadStreamOpen(stream);
#ifdef PYOSINPUTHOOK_REPETITIVE
    if (!CFReadStreamHasBytesAvailable(stream))
    /* This is possible because of how PyOS_InputHook is called from Python */
#endif
    {
        int error;
        int channel[2];
        CFSocketRef sigint_socket = NULL;
        PyOS_sighandler_t py_sigint_handler = NULL;
        CFStreamClientContext clientContext = {0, NULL, NULL, NULL, NULL};
        clientContext.info = runloop;
        CFReadStreamSetClient(stream,
                              kCFStreamEventHasBytesAvailable,
                              _stdin_callback,
                              &clientContext);
        CFReadStreamScheduleWithRunLoop(stream, runloop, kCFRunLoopDefaultMode);
        error = socketpair(AF_UNIX, SOCK_STREAM, 0, channel);
        if (!error) {
            CFSocketContext context;
            context.version = 0;
            context.info = &interrupted;
            context.retain = NULL;
            context.release = NULL;
            context.copyDescription = NULL;
            fcntl(channel[0], F_SETFL, O_WRONLY | O_NONBLOCK);
            sigint_socket = CFSocketCreateWithNative(
                kCFAllocatorDefault,
                channel[1],
                kCFSocketReadCallBack,
                _sigint_callback,
                &context);
            if (sigint_socket) {
                CFRunLoopSourceRef source = CFSocketCreateRunLoopSource(
                    kCFAllocatorDefault, sigint_socket, 0);
                CFRelease(sigint_socket);
                if (source) {
                    CFRunLoopAddSource(runloop, source, kCFRunLoopDefaultMode);
                    CFRelease(source);
                    sigint_fd = channel[0];
                    py_sigint_handler = PyOS_setsig(SIGINT, _sigint_handler);
                }
            }
        }

        NSEvent* event;
        while (true) {
            while (true) {
                event = [NSApp nextEventMatchingMask: NSEventMaskAny
                                           untilDate: [NSDate distantPast]
                                              inMode: NSDefaultRunLoopMode
                                             dequeue: YES];
                if (!event) { break; }
                [NSApp sendEvent: event];
            }
            CFRunLoopRun();
            if (interrupted || CFReadStreamHasBytesAvailable(stream)) { break; }
        }

        if (py_sigint_handler) { PyOS_setsig(SIGINT, py_sigint_handler); }
        CFReadStreamUnscheduleFromRunLoop(
            stream, runloop, kCFRunLoopCommonModes);
        if (sigint_socket) { CFSocketInvalidate(sigint_socket); }
        if (!error) {
            close(channel[0]);
            close(channel[1]);
        }
    }
    CFReadStreamClose(stream);
    CFRelease(stream);
    if (interrupted) {
        errno = EINTR;
        raise(SIGINT);
        return -1;
    }
    return 1;
}

/* ---------------------------- Cocoa classes ---------------------------- */

@interface WindowServerConnectionManager : NSObject
{
}
+ (WindowServerConnectionManager*)sharedManager;
- (void)launch:(NSNotification*)notification;
@end

@interface Window : NSWindow
{   PyObject* manager;
}
- (Window*)initWithContentRect:(NSRect)rect styleMask:(unsigned int)mask backing:(NSBackingStoreType)bufferingType defer:(BOOL)deferCreation withManager: (PyObject*)theManager;
- (NSRect)constrainFrameRect:(NSRect)rect toScreen:(NSScreen*)screen;
- (BOOL)closeButtonPressed;
- (void)dealloc;
@end

@interface View : NSView <NSWindowDelegate>
{   PyObject* canvas;
    NSRect rubberband;
    NSTrackingRectTag tracking;
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
- (BOOL)isFlipped;
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
//- (void)flagsChanged:(NSEvent*)event;
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

static bool backend_inited = false;

static void lazy_init(void) {
    if (backend_inited) { return; }
    backend_inited = true;

    NSApp = [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

#ifndef PYPY
    /* TODO: remove ifndef after the new PyPy with the PyOS_InputHook implementation
    get released: https://bitbucket.org/pypy/pypy/commits/caaf91a */
    PyOS_InputHook = wait_for_stdin;
#endif

    WindowServerConnectionManager* connectionManager = [WindowServerConnectionManager sharedManager];
    NSWorkspace* workspace = [NSWorkspace sharedWorkspace];
    NSNotificationCenter* notificationCenter = [workspace notificationCenter];
    [notificationCenter addObserver: connectionManager
                           selector: @selector(launch:)
                               name: NSWorkspaceDidLaunchApplicationNotification
                             object: nil];
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

static CGFloat _get_device_scale(CGContextRef cr)
{
    CGSize pixelSize = CGContextConvertSizeToDeviceSpace(cr, CGSizeMake(1, 1));
    return pixelSize.width;
}

typedef struct {
    PyObject_HEAD
    View* view;
} FigureCanvas;

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
    int width;
    int height;
    if (!self->view) {
        PyErr_SetString(PyExc_RuntimeError, "NSView* is NULL");
        return -1;
    }
    if (!PyArg_ParseTuple(args, "ii", &width, &height)) { return -1; }

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
    return 0;
}

static void
FigureCanvas_dealloc(FigureCanvas* self)
{
    if (self->view) {
        [self->view setCanvas: NULL];
        [self->view release];
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject*
FigureCanvas_repr(FigureCanvas* self)
{
    return PyUnicode_FromFormat("FigureCanvas object %p wrapping NSView %p",
                               (void*)self, (void*)(self->view));
}

static PyObject*
FigureCanvas_draw(FigureCanvas* self)
{
    View* view = self->view;
    if (view) {  /* The figure may have been closed already */
        [view display];
    }
    Py_RETURN_NONE;
}

static PyObject*
FigureCanvas_draw_idle(FigureCanvas* self)
{
    View* view = self->view;
    if (!view) {
        PyErr_SetString(PyExc_RuntimeError, "NSView* is NULL");
        return NULL;
    }
    [view setNeedsDisplay: YES];
    Py_RETURN_NONE;
}

static PyObject*
FigureCanvas_flush_events(FigureCanvas* self)
{
    View* view = self->view;
    if (!view) {
        PyErr_SetString(PyExc_RuntimeError, "NSView* is NULL");
        return NULL;
    }
    [view displayIfNeeded];
    Py_RETURN_NONE;
}

static PyObject*
FigureCanvas_set_rubberband(FigureCanvas* self, PyObject *args)
{
    View* view = self->view;
    int x0, y0, x1, y1;
    NSRect rubberband;
    if (!view) {
        PyErr_SetString(PyExc_RuntimeError, "NSView* is NULL");
        return NULL;
    }
    if (!PyArg_ParseTuple(args, "iiii", &x0, &y0, &x1, &y1)) { return NULL; }

    x0 /= view->device_scale;
    x1 /= view->device_scale;
    y0 /= view->device_scale;
    y1 /= view->device_scale;

    if (x1 > x0) {
        rubberband.origin.x = x0;
        rubberband.size.width = x1 - x0;
    } else {
        rubberband.origin.x = x1;
        rubberband.size.width = x0 - x1;
    }
    if (y1 > y0) {
        rubberband.origin.y = y0;
        rubberband.size.height = y1 - y0;
    } else {
        rubberband.origin.y = y1;
        rubberband.size.height = y0 - y1;
    }

    [view setRubberband: rubberband];
    Py_RETURN_NONE;
}

static PyObject*
FigureCanvas_remove_rubberband(FigureCanvas* self)
{
    View* view = self->view;
    if (!view) {
        PyErr_SetString(PyExc_RuntimeError, "NSView* is NULL");
        return NULL;
    }
    [view removeRubberband];
    Py_RETURN_NONE;
}

static PyObject*
FigureCanvas_start_event_loop(FigureCanvas* self, PyObject* args, PyObject* keywords)
{
    float timeout = 0.0;

    static char* kwlist[] = {"timeout", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "f", kwlist, &timeout)) {
        return NULL;
    }

    int error;
    int interrupted = 0;
    int channel[2];
    CFSocketRef sigint_socket = NULL;
    PyOS_sighandler_t py_sigint_handler = NULL;

    CFRunLoopRef runloop = CFRunLoopGetCurrent();

    error = pipe(channel);
    if (!error) {
        CFSocketContext context = {0, NULL, NULL, NULL, NULL};
        fcntl(channel[1], F_SETFL, O_WRONLY | O_NONBLOCK);

        context.info = &interrupted;
        sigint_socket = CFSocketCreateWithNative(kCFAllocatorDefault,
                                                 channel[0],
                                                 kCFSocketReadCallBack,
                                                 _sigint_callback,
                                                 &context);
        if (sigint_socket) {
            CFRunLoopSourceRef source = CFSocketCreateRunLoopSource(
                kCFAllocatorDefault, sigint_socket, 0);
            CFRelease(sigint_socket);
            if (source) {
                CFRunLoopAddSource(runloop, source, kCFRunLoopDefaultMode);
                CFRelease(source);
                sigint_fd = channel[1];
                py_sigint_handler = PyOS_setsig(SIGINT, _sigint_handler);
            }
        }
        else
            close(channel[0]);
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

    if (py_sigint_handler) { PyOS_setsig(SIGINT, py_sigint_handler); }
    if (sigint_socket) { CFSocketInvalidate(sigint_socket); }
    if (!error) { close(channel[1]); }
    if (interrupted) { raise(SIGINT); }

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
        {"draw",
         (PyCFunction)FigureCanvas_draw,
         METH_NOARGS,
         NULL},  // docstring inherited
        {"draw_idle",
         (PyCFunction)FigureCanvas_draw_idle,
         METH_NOARGS,
         NULL},  // docstring inherited
        {"flush_events",
         (PyCFunction)FigureCanvas_flush_events,
         METH_NOARGS,
         NULL},  // docstring inherited
        {"set_rubberband",
         (PyCFunction)FigureCanvas_set_rubberband,
         METH_VARARGS,
         "Specifies a new rubberband rectangle and invalidates it."},
        {"remove_rubberband",
         (PyCFunction)FigureCanvas_remove_rubberband,
         METH_NOARGS,
         "Removes the current rubberband rectangle."},
        {"start_event_loop",
         (PyCFunction)FigureCanvas_start_event_loop,
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
    NSRect rect;
    Window* window;
    View* view;
    PyObject* size;
    int width, height;
    PyObject* obj;
    FigureCanvas* canvas;

    if (!self->window) {
        PyErr_SetString(PyExc_RuntimeError, "NSWindow* is NULL");
        return -1;
    }

    if (!PyArg_ParseTuple(args, "O", &obj)) { return -1; }

    canvas = (FigureCanvas*)obj;
    view = canvas->view;
    if (!view) { /* Something really weird going on */
        PyErr_SetString(PyExc_RuntimeError, "NSView* is NULL");
        return -1;
    }

    size = PyObject_CallMethod(obj, "get_width_height", "");
    if (!size) { return -1; }
    if (!PyArg_ParseTuple(size, "ii", &width, &height)) {
        Py_DECREF(size);
        return -1;
    }
    Py_DECREF(size);

    rect.origin.x = 100;
    rect.origin.y = 350;
    rect.size.height = height;
    rect.size.width = width;

    self->window = [self->window initWithContentRect: rect
                                         styleMask: NSWindowStyleMaskTitled
                                                  | NSWindowStyleMaskClosable
                                                  | NSWindowStyleMaskResizable
                                                  | NSWindowStyleMaskMiniaturizable
                                           backing: NSBackingStoreBuffered
                                             defer: YES
                                       withManager: (PyObject*)self];
    window = self->window;
    [window setDelegate: view];
    [window makeFirstResponder: view];
    [[window contentView] addSubview: view];
    [view updateDevicePixelRatio: [window backingScaleFactor]];

    return 0;
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
    Window* window = self->window;
    if (window) {
        [window close];
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject*
FigureManager_show(FigureManager* self)
{
    Window* window = self->window;
    if (window) {
        [window makeKeyAndOrderFront: nil];
        [window orderFrontRegardless];
    }
    Py_RETURN_NONE;
}

static PyObject*
FigureManager_destroy(FigureManager* self)
{
    Window* window = self->window;
    if (window) {
        [window close];
        self->window = NULL;
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
    Window* window = self->window;
    if (window) {
        NSString* ns_title = [[[NSString alloc]
                               initWithCString: title
                               encoding: NSUTF8StringEncoding] autorelease];
        [window setTitle: ns_title];
    }
    Py_RETURN_NONE;
}

static PyObject*
FigureManager_get_window_title(FigureManager* self)
{
    Window* window = self->window;
    PyObject* result = NULL;
    if (window) {
        NSString* title = [window title];
        if (title) {
            const char* cTitle = [title UTF8String];
            result = PyUnicode_FromString(cTitle);
        }
    }
    if (result) {
        return result;
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
        {"show",
         (PyCFunction)FigureManager_show,
         METH_NOARGS},
        {"destroy",
         (PyCFunction)FigureManager_destroy,
         METH_NOARGS},
        {"set_window_title",
         (PyCFunction)FigureManager_set_window_title,
         METH_VARARGS},
        {"get_window_title",
         (PyCFunction)FigureManager_get_window_title,
         METH_NOARGS},
        {"resize",
         (PyCFunction)FigureManager_resize,
         METH_VARARGS},
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
    NSPopUpButton* menu;
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
#ifdef COMPILING_FOR_10_7
    rect = [window convertRectToBacking: rect];
#endif
    size = rect.size;
    scale = NSMakeSize(imagesize / size.width, imagesize / size.height);

    rect.size.width = 32;
    rect.size.height = 32;
    rect.origin.x = gap;
    rect.origin.y = 0.5*(height - rect.size.height);

    for (int i = 0; i < 7; i++) {
        NSString* filename = [NSString stringWithCString: images[i]
                                                encoding: NSUTF8StringEncoding];
        NSString* tooltip = [NSString stringWithCString: tooltips[i]
                                               encoding: NSUTF8StringEncoding];
        NSImage* image = [[NSImage alloc] initWithContentsOfFile: filename];
        buttons[i] = [[NSButton alloc] initWithFrame: rect];
        [image setSize: size];
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
    rect.size.width = 300;
    rect.size.height = 0;
    rect.origin.x += height;
    NSTextView* messagebox = [[NSTextView alloc] initWithFrame: rect];
    if (@available(macOS 10.11, *)) {
        messagebox.textContainer.maximumNumberOfLines = 2;
        messagebox.textContainer.lineBreakMode = NSLineBreakByTruncatingTail;
    }
    [messagebox setFont: font];
    [messagebox setDrawsBackground: NO];
    [messagebox setSelectable: NO];
    /* if selectable, the messagebox can become first responder,
     * which is not supposed to happen */
    rect = [messagebox frame];
    rect.origin.y = 0.5 * (height - rect.size.height);
    [messagebox setFrameOrigin: rect.origin];
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

    if (!PyArg_ParseTuple(args, "y", &message)) { return NULL; }

    NSTextView* messagebox = self->messagebox;

    if (messagebox) {
        NSString* text = [NSString stringWithUTF8String: message];
        [messagebox setString: text];

        // Adjust width with the window size
        NSRect rectWindow = [messagebox.superview frame];
        NSRect rect = [messagebox frame];
        rect.size.width = rectWindow.size.width - rect.origin.x;
        [messagebox setFrame: rect];

        // Adjust height with the content size
        [messagebox.layoutManager ensureLayoutForTextContainer: messagebox.textContainer];
        NSRect contentSize = [messagebox.layoutManager usedRectForTextContainer: messagebox.textContainer];
        rect = [messagebox frame];
        rect.origin.y = 0.5 * (self->height - contentSize.size.height);
        [messagebox setFrame: rect];
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
    const char* default_filename;
    if (!PyArg_ParseTuple(args, "ss", &title, &default_filename)) {
        return NULL;
    }
    NSSavePanel* panel = [NSSavePanel savePanel];
    [panel setTitle: [NSString stringWithCString: title
                                        encoding: NSASCIIStringEncoding]];
    NSString* ns_default_filename =
        [[NSString alloc]
         initWithCString: default_filename
         encoding: NSUTF8StringEncoding];
    [panel setNameFieldStringValue: ns_default_filename];
    result = [panel runModal];
    [ns_default_filename release];
    if (result == NSModalResponseOK) {
        NSURL* url = [panel URL];
        NSString* filename = [url path];
        if (!filename) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to obtain filename");
            return 0;
        }
        unsigned int n = [filename length];
        unichar* buffer = malloc(n*sizeof(unichar));
        [filename getCharacters: buffer];
        PyObject* string = PyUnicode_FromKindAndData(PyUnicode_2BYTE_KIND, buffer, n);
        free(buffer);
        return string;
    }
    Py_RETURN_NONE;
}

static PyObject*
set_cursor(PyObject* unused, PyObject* args)
{
    int i;
    if (!PyArg_ParseTuple(args, "i", &i)) { return NULL; }
    switch (i) {
      case 1: [[NSCursor arrowCursor] set]; break;
      case 2: [[NSCursor pointingHandCursor] set]; break;
      case 3: [[NSCursor crosshairCursor] set]; break;
      case 4: [[NSCursor openHandCursor] set]; break;
      /* OSX handles busy state itself so no need to set a cursor here */
      case 5: break;
      case 6: [[NSCursor resizeLeftRightCursor] set]; break;
      case 7: [[NSCursor resizeUpDownCursor] set]; break;
      default: return NULL;
    }
    Py_RETURN_NONE;
}

@implementation WindowServerConnectionManager
static WindowServerConnectionManager *sharedWindowServerConnectionManager = nil;

+ (WindowServerConnectionManager *)sharedManager
{
    if (sharedWindowServerConnectionManager == nil) {
        sharedWindowServerConnectionManager = [[super allocWithZone:NULL] init];
    }
    return sharedWindowServerConnectionManager;
}

+ (id)allocWithZone:(NSZone *)zone
{
    return [[self sharedManager] retain];
}

+ (id)copyWithZone:(NSZone *)zone
{
    return self;
}

+ (id)retain
{
    return self;
}

- (NSUInteger)retainCount
{
    return NSUIntegerMax;  //denotes an object that cannot be released
}

- (oneway void)release
{
    // Don't release a singleton object
}

- (id)autorelease
{
    return self;
}

- (void)launch:(NSNotification*)notification
{
    CFRunLoopRef runloop;
    CFMachPortRef port;
    CFRunLoopSourceRef source;
    NSDictionary* dictionary = [notification userInfo];
    if (![[dictionary valueForKey:@"NSApplicationName"]
          isEqualToString:@"Python"])
        return;
    NSNumber* psnLow = [dictionary valueForKey: @"NSApplicationProcessSerialNumberLow"];
    NSNumber* psnHigh = [dictionary valueForKey: @"NSApplicationProcessSerialNumberHigh"];
    ProcessSerialNumber psn;
    psn.highLongOfPSN = [psnHigh intValue];
    psn.lowLongOfPSN = [psnLow intValue];
    runloop = CFRunLoopGetCurrent();
    port = CGEventTapCreateForPSN(&psn,
                                  kCGHeadInsertEventTap,
                                  kCGEventTapOptionListenOnly,
                                  kCGEventMaskForAllEvents,
                                  &_eventtap_callback,
                                  runloop);
    source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault,
                                           port,
                                           0);
    CFRunLoopAddSource(runloop, source, kCFRunLoopDefaultMode);
    CFRelease(port);
}
@end

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
    gil_call_method(manager, "close");
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
}

- (void)dealloc
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    Py_DECREF(manager);
    PyGILState_Release(gstate);
    /* The reference count of the view that was added as a subview to the
     * content view of this window was increased during the call to addSubview,
     * and is decreased during the call to [super dealloc].
     */
    [super dealloc];
}
@end

@implementation View
- (BOOL)isFlipped
{
    return NO;
}

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
    PyBuffer_Release((Py_buffer *)info);
    free(info);
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

    CGColorSpaceRef colorspace = CGColorSpaceCreateWithName(kCGColorSpaceGenericRGB);
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

    if (!(renderer = PyObject_CallMethod(canvas, "_draw", "", NULL))
        || !(renderer_buffer = PyObject_GetAttrString(renderer, "_renderer"))) {
        PyErr_Print();
        goto exit;
    }
    if (_copy_agg_buffer(cr, renderer_buffer)) {
        printf("copy_agg_buffer failed\n");
        goto exit;
    }
    if (!NSIsEmptyRect(rubberband)) {
        NSFrameRect(rubberband);
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
    if (!(change = PyObject_CallMethod(canvas, "_set_device_pixel_ratio", "d", device_scale, NULL))) {
        PyErr_Print();
        goto exit;
    }
    if (PyObject_IsTrue(change)) {
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
    PyGILState_STATE gstate;
    PyObject* result;

    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(canvas, "close_event", "");
    if (result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);
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
    PyGILState_STATE gstate;
    PyObject* result;

    int x, y;
    NSPoint location = [event locationInWindow];
    location = [self convertPoint: location fromView: nil];
    x = location.x * device_scale;
    y = location.y * device_scale;

    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(canvas, "enter_notify_event", "O(ii)",
            Py_None, x, y);

    if (result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);
}

- (void)mouseExited:(NSEvent *)event
{
    PyGILState_STATE gstate;
    PyObject* result;

    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(canvas, "leave_notify_event", "");
    if (result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);
}

- (void)mouseDown:(NSEvent *)event
{
    int x, y;
    int num;
    int dblclick = 0;
    PyObject* result;
    PyGILState_STATE gstate;
    NSPoint location = [event locationInWindow];
    location = [self convertPoint: location fromView: nil];
    x = location.x * device_scale;
    y = location.y * device_scale;
    switch ([event type])
    {    case NSEventTypeLeftMouseDown:
         {   unsigned int modifier = [event modifierFlags];
             if (modifier & NSEventModifierFlagControl)
                 /* emulate a right-button click */
                 num = 3;
             else if (modifier & NSEventModifierFlagOption)
                 /* emulate a middle-button click */
                 num = 2;
             else
             {
                 num = 1;
                 if ([NSCursor currentCursor]==[NSCursor openHandCursor])
                     [[NSCursor closedHandCursor] set];
             }
             break;
         }
         case NSEventTypeOtherMouseDown: num = 2; break;
         case NSEventTypeRightMouseDown: num = 3; break;
         default: return; /* Unknown mouse event */
    }
    if ([event clickCount] == 2) {
      dblclick = 1;
    }
    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(canvas, "button_press_event", "iiii", x, y, num, dblclick);
    if (result)
        Py_DECREF(result);
    else
        PyErr_Print();

    PyGILState_Release(gstate);
}

- (void)mouseUp:(NSEvent *)event
{
    int num;
    int x, y;
    PyObject* result;
    PyGILState_STATE gstate;
    NSPoint location = [event locationInWindow];
    location = [self convertPoint: location fromView: nil];
    x = location.x * device_scale;
    y = location.y * device_scale;
    switch ([event type])
    {    case NSEventTypeLeftMouseUp:
             num = 1;
             if ([NSCursor currentCursor]==[NSCursor closedHandCursor])
                 [[NSCursor openHandCursor] set];
             break;
         case NSEventTypeOtherMouseUp: num = 2; break;
         case NSEventTypeRightMouseUp: num = 3; break;
         default: return; /* Unknown mouse event */
    }
    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(canvas, "button_release_event", "iii", x, y, num);
    if (result)
        Py_DECREF(result);
    else
        PyErr_Print();

    PyGILState_Release(gstate);
}

- (void)mouseMoved:(NSEvent *)event
{
    int x, y;
    NSPoint location = [event locationInWindow];
    location = [self convertPoint: location fromView: nil];
    x = location.x * device_scale;
    y = location.y * device_scale;
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* result = PyObject_CallMethod(canvas, "motion_notify_event", "ii", x, y);
    if (result)
        Py_DECREF(result);
    else
        PyErr_Print();

    PyGILState_Release(gstate);
}

- (void)mouseDragged:(NSEvent *)event
{
    int x, y;
    NSPoint location = [event locationInWindow];
    location = [self convertPoint: location fromView: nil];
    x = location.x * device_scale;
    y = location.y * device_scale;
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* result = PyObject_CallMethod(canvas, "motion_notify_event", "ii", x, y);
    if (result)
        Py_DECREF(result);
    else
        PyErr_Print();

    PyGILState_Release(gstate);
}

- (void)rightMouseDown:(NSEvent *)event { [self mouseDown: event]; }
- (void)rightMouseUp:(NSEvent *)event { [self mouseUp: event]; }
- (void)rightMouseDragged:(NSEvent *)event { [self mouseDragged: event]; }
- (void)otherMouseDown:(NSEvent *)event { [self mouseDown: event]; }
- (void)otherMouseUp:(NSEvent *)event { [self mouseUp: event]; }
- (void)otherMouseDragged:(NSEvent *)event { [self mouseDragged: event]; }

- (void)setRubberband:(NSRect)rect
{
    if (!NSIsEmptyRect(rubberband)) { [self setNeedsDisplayInRect: rubberband]; }
    rubberband = rect;
    [self setNeedsDisplayInRect: rubberband];
}

- (void)removeRubberband
{
    if (NSIsEmptyRect(rubberband)) { return; }
    [self setNeedsDisplayInRect: rubberband];
    rubberband = NSZeroRect;
}

- (const char*)convertKeyEvent:(NSEvent*)event
{
    NSDictionary* specialkeymappings = [NSDictionary dictionaryWithObjectsAndKeys:
                                        @"left", [NSNumber numberWithUnsignedLong:NSLeftArrowFunctionKey],
                                        @"right", [NSNumber numberWithUnsignedLong:NSRightArrowFunctionKey],
                                        @"up", [NSNumber numberWithUnsignedLong:NSUpArrowFunctionKey],
                                        @"down", [NSNumber numberWithUnsignedLong:NSDownArrowFunctionKey],
                                        @"f1", [NSNumber numberWithUnsignedLong:NSF1FunctionKey],
                                        @"f2", [NSNumber numberWithUnsignedLong:NSF2FunctionKey],
                                        @"f3", [NSNumber numberWithUnsignedLong:NSF3FunctionKey],
                                        @"f4", [NSNumber numberWithUnsignedLong:NSF4FunctionKey],
                                        @"f5", [NSNumber numberWithUnsignedLong:NSF5FunctionKey],
                                        @"f6", [NSNumber numberWithUnsignedLong:NSF6FunctionKey],
                                        @"f7", [NSNumber numberWithUnsignedLong:NSF7FunctionKey],
                                        @"f8", [NSNumber numberWithUnsignedLong:NSF8FunctionKey],
                                        @"f9", [NSNumber numberWithUnsignedLong:NSF9FunctionKey],
                                        @"f10", [NSNumber numberWithUnsignedLong:NSF10FunctionKey],
                                        @"f11", [NSNumber numberWithUnsignedLong:NSF11FunctionKey],
                                        @"f12", [NSNumber numberWithUnsignedLong:NSF12FunctionKey],
                                        @"f13", [NSNumber numberWithUnsignedLong:NSF13FunctionKey],
                                        @"f14", [NSNumber numberWithUnsignedLong:NSF14FunctionKey],
                                        @"f15", [NSNumber numberWithUnsignedLong:NSF15FunctionKey],
                                        @"f16", [NSNumber numberWithUnsignedLong:NSF16FunctionKey],
                                        @"f17", [NSNumber numberWithUnsignedLong:NSF17FunctionKey],
                                        @"f18", [NSNumber numberWithUnsignedLong:NSF18FunctionKey],
                                        @"f19", [NSNumber numberWithUnsignedLong:NSF19FunctionKey],
                                        @"scroll_lock", [NSNumber numberWithUnsignedLong:NSScrollLockFunctionKey],
                                        @"break", [NSNumber numberWithUnsignedLong:NSBreakFunctionKey],
                                        @"insert", [NSNumber numberWithUnsignedLong:NSInsertFunctionKey],
                                        @"delete", [NSNumber numberWithUnsignedLong:NSDeleteFunctionKey],
                                        @"home", [NSNumber numberWithUnsignedLong:NSHomeFunctionKey],
                                        @"end", [NSNumber numberWithUnsignedLong:NSEndFunctionKey],
                                        @"pagedown", [NSNumber numberWithUnsignedLong:NSPageDownFunctionKey],
                                        @"pageup", [NSNumber numberWithUnsignedLong:NSPageUpFunctionKey],
                                        @"backspace", [NSNumber numberWithUnsignedLong:NSDeleteCharacter],
                                        @"enter", [NSNumber numberWithUnsignedLong:NSEnterCharacter],
                                        @"tab", [NSNumber numberWithUnsignedLong:NSTabCharacter],
                                        @"enter", [NSNumber numberWithUnsignedLong:NSCarriageReturnCharacter],
                                        @"backtab", [NSNumber numberWithUnsignedLong:NSBackTabCharacter],
                                        @"escape", [NSNumber numberWithUnsignedLong:27],
                                        nil
                                        ];

    NSMutableString* returnkey = [NSMutableString string];
    if ([event modifierFlags] & NSEventModifierFlagControl) {
        [returnkey appendString:@"ctrl+" ];
    }
    if ([event modifierFlags] & NSEventModifierFlagOption) {
        [returnkey appendString:@"alt+" ];
    }
    if ([event modifierFlags] & NSEventModifierFlagCommand) {
        [returnkey appendString:@"cmd+" ];
    }

    unichar uc = [[event charactersIgnoringModifiers] characterAtIndex:0];
    NSString* specialchar = [specialkeymappings objectForKey:[NSNumber numberWithUnsignedLong:uc]];
    if (specialchar) {
        if ([event modifierFlags] & NSEventModifierFlagShift) {
            [returnkey appendString:@"shift+" ];
        }
        [returnkey appendString:specialchar];
    }
    else
        [returnkey appendString:[event charactersIgnoringModifiers]];

    return [returnkey UTF8String];
}

- (void)keyDown:(NSEvent*)event
{
    PyObject* result;
    const char* s = [self convertKeyEvent: event];
    PyGILState_STATE gstate = PyGILState_Ensure();
    if (!s) {
        result = PyObject_CallMethod(canvas, "key_press_event", "O", Py_None);
    } else {
        result = PyObject_CallMethod(canvas, "key_press_event", "s", s);
    }
    if (result)
        Py_DECREF(result);
    else
        PyErr_Print();

    PyGILState_Release(gstate);
}

- (void)keyUp:(NSEvent*)event
{
    PyObject* result;
    const char* s = [self convertKeyEvent: event];
    PyGILState_STATE gstate = PyGILState_Ensure();
    if (!s) {
        result = PyObject_CallMethod(canvas, "key_release_event", "O", Py_None);
    } else {
        result = PyObject_CallMethod(canvas, "key_release_event", "s", s);
    }
    if (result)
        Py_DECREF(result);
    else
        PyErr_Print();

    PyGILState_Release(gstate);
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

    PyObject* result;
    PyGILState_STATE gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(canvas, "scroll_event", "iii", x, y, step);
    if (result)
        Py_DECREF(result);
    else
        PyErr_Print();

    PyGILState_Release(gstate);
}

- (BOOL)acceptsFirstResponder
{
    return YES;
}

/* This is all wrong. Address of pointer is being passed instead of pointer, keynames don't
   match up with what the front-end and does the front-end even handle modifier keys by themselves?

- (void)flagsChanged:(NSEvent*)event
{
    const char *s = NULL;
    if (([event modifierFlags] & NSControlKeyMask) == NSControlKeyMask)
        s = "control";
    else if (([event modifierFlags] & NSShiftKeyMask) == NSShiftKeyMask)
        s = "shift";
    else if (([event modifierFlags] & NSAlternateKeyMask) == NSAlternateKeyMask)
        s = "alt";
    else return;
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* result = PyObject_CallMethod(canvas, "key_press_event", "s", &s);
    if (result)
        Py_DECREF(result);
    else
        PyErr_Print();

    PyGILState_Release(gstate);
}
 */
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
    CFRunLoopTimerRef timer;
} Timer;

static PyObject*
Timer_new(PyTypeObject* type, PyObject *args, PyObject *kwds)
{
    lazy_init();
    Timer* self = (Timer*)type->tp_alloc(type, 0);
    if (!self) { return NULL; }
    self->timer = NULL;
    return (PyObject*) self;
}

static PyObject*
Timer_repr(Timer* self)
{
    return PyUnicode_FromFormat("Timer object %p wrapping CFRunLoopTimerRef %p",
                               (void*) self, (void*)(self->timer));
}

static void timer_callback(CFRunLoopTimerRef timer, void* info)
{
    gil_call_method(info, "_on_timer");
}

static void context_cleanup(const void* info)
{
    Py_DECREF((PyObject*)info);
}

static PyObject*
Timer__timer_start(Timer* self, PyObject* args)
{
    CFRunLoopRef runloop;
    CFRunLoopTimerRef timer;
    CFRunLoopTimerContext context;
    CFAbsoluteTime firstFire;
    CFTimeInterval interval;
    PyObject* py_interval = NULL, * py_single = NULL, * py_on_timer = NULL;
    int single;
    runloop = CFRunLoopGetCurrent();
    if (!runloop) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to obtain run loop");
        return NULL;
    }
    if (!(py_interval = PyObject_GetAttrString((PyObject*)self, "_interval"))
        || ((interval = PyFloat_AsDouble(py_interval) / 1000.), PyErr_Occurred())
        || !(py_single = PyObject_GetAttrString((PyObject*)self, "_single"))
        || ((single = PyObject_IsTrue(py_single)) == -1)
        || !(py_on_timer = PyObject_GetAttrString((PyObject*)self, "_on_timer"))) {
        goto exit;
    }
    // (current time + interval) is time of first fire.
    firstFire = CFAbsoluteTimeGetCurrent() + interval;
    if (single) {
        interval = 0;
    }
    if (!PyMethod_Check(py_on_timer)) {
        PyErr_SetString(PyExc_RuntimeError, "_on_timer should be a Python method");
        goto exit;
    }
    Py_INCREF(self);
    context.version = 0;
    context.retain = NULL;
    context.release = context_cleanup;
    context.copyDescription = NULL;
    context.info = self;
    timer = CFRunLoopTimerCreate(kCFAllocatorDefault,
                                 firstFire,
                                 interval,
                                 0,
                                 0,
                                 timer_callback,
                                 &context);
    if (!timer) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create timer");
        goto exit;
    }
    if (self->timer) {
        CFRunLoopTimerInvalidate(self->timer);
        CFRelease(self->timer);
    }
    CFRunLoopAddTimer(runloop, timer, kCFRunLoopCommonModes);
    /* Don't release the timer here, since the run loop may be destroyed and
     * the timer lost before we have a chance to decrease the reference count
     * of the attribute */
    self->timer = timer;
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

static PyObject*
Timer__timer_stop(Timer* self)
{
    if (self->timer) {
        CFRunLoopTimerInvalidate(self->timer);
        CFRelease(self->timer);
        self->timer = NULL;
    }
    Py_RETURN_NONE;
}

static void
Timer_dealloc(Timer* self)
{
    Timer__timer_stop(self);
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
    .tp_doc = "A Timer object wraps a CFRunLoopTimerRef and can add it to the event loop.",
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
        {"set_cursor",
         (PyCFunction)set_cursor,
         METH_VARARGS,
         "Set the active cursor."},
        {}  /* Sentinel */
    },
};

#pragma GCC visibility push(default)

PyObject* PyInit__macosx(void)
{
    if (PyType_Ready(&FigureCanvasType) < 0
     || PyType_Ready(&FigureManagerType) < 0
     || PyType_Ready(&NavigationToolbar2Type) < 0
     || PyType_Ready(&TimerType) < 0)
        return NULL;

    PyObject *module = PyModule_Create(&moduledef);
    if (!module) {
        return NULL;
    }

    PyModule_AddObject(module, "FigureCanvas", (PyObject*) &FigureCanvasType);
    PyModule_AddObject(module, "FigureManager", (PyObject*) &FigureManagerType);
    PyModule_AddObject(module, "NavigationToolbar2", (PyObject*) &NavigationToolbar2Type);
    PyModule_AddObject(module, "Timer", (PyObject*) &TimerType);

    return module;
}

#pragma GCC visibility pop
