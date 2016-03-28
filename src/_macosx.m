#include <Cocoa/Cocoa.h>
#include <ApplicationServices/ApplicationServices.h>
#include <sys/socket.h>
#include <Python.h>

#define PYOSINPUTHOOK_REPETITIVE 1 /* Remove this once Python is fixed */

#if PY_MAJOR_VERSION >= 3
#define PY3K 1
#else
#define PY3K 0
#endif

#if (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION >=3)
#define PY33 1
#else
#define PY33 0
#endif

/* Must define Py_TYPE for Python 2.5 or older */
#ifndef Py_TYPE
# define Py_TYPE(o) ((o)->ob_type)
#endif

/* Proper way to check for the OS X version we are compiling for, from
   http://developer.apple.com/documentation/DeveloperTools/Conceptual/cross_development */
#if __MAC_OS_X_VERSION_MIN_REQUIRED >= 1050
#define COMPILING_FOR_10_5
#endif
#if __MAC_OS_X_VERSION_MIN_REQUIRED >= 1060
#define COMPILING_FOR_10_6
#endif
#if __MAC_OS_X_VERSION_MIN_REQUIRED >= 1070
#define COMPILING_FOR_10_7
#endif
#if __MAC_OS_X_VERSION_MIN_REQUIRED >= 10100
#define COMPILING_FOR_10_10
#endif

/* Use Atsui for Mac OS X 10.4, CoreText for Mac OS X 10.5 */
#ifndef COMPILING_FOR_10_5
static int ngc = 0;    /* The number of graphics contexts in use */

#include <Carbon/Carbon.h>

#endif

/* CGFloat was defined in Mac OS X 10.5 */
#ifndef CGFLOAT_DEFINED
#define CGFloat float
#endif


/* Various NSApplicationDefined event subtypes */
#define STOP_EVENT_LOOP 2
#define WINDOW_CLOSING 3

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

static CGEventRef _eventtap_callback(CGEventTapProxy proxy, CGEventType type, CGEventRef event, void *refcon)
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
    {
#endif
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
        if (error==0)
        {
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
            if (sigint_socket)
            {
                CFRunLoopSourceRef source;
                source = CFSocketCreateRunLoopSource(kCFAllocatorDefault,
                                                     sigint_socket,
                                                     0);
                CFRelease(sigint_socket);
                if (source)
                {
                    CFRunLoopAddSource(runloop, source, kCFRunLoopDefaultMode);
                    CFRelease(source);
                    sigint_fd = channel[0];
                    py_sigint_handler = PyOS_setsig(SIGINT, _sigint_handler);
                }
            }
        }

        NSEvent* event;
        NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
        while (true) {
            while (true) {
                event = [NSApp nextEventMatchingMask: NSAnyEventMask
                                           untilDate: [NSDate distantPast]
                                              inMode: NSDefaultRunLoopMode
                                             dequeue: YES];
                if (!event) break;
                [NSApp sendEvent: event];
            }
            CFRunLoopRun();
            if (interrupted || CFReadStreamHasBytesAvailable(stream)) break;
        }
        [pool release];

        if (py_sigint_handler) PyOS_setsig(SIGINT, py_sigint_handler);
        CFReadStreamUnscheduleFromRunLoop(stream,
                                          runloop,
                                          kCFRunLoopCommonModes);
        if (sigint_socket) CFSocketInvalidate(sigint_socket);
        if (error==0) {
            close(channel[0]);
            close(channel[1]);
        }
#ifdef PYOSINPUTHOOK_REPETITIVE
    }
#endif
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

@interface ToolWindow : NSWindow
{
}
- (ToolWindow*)initWithContentRect:(NSRect)rect master:(NSWindow*)window;
- (void)masterCloses:(NSNotification*)notification;
- (void)close;
@end

#ifdef COMPILING_FOR_10_6
@interface View : NSView <NSWindowDelegate>
#else
@interface View : NSView
#endif
{   PyObject* canvas;
    NSRect rubberband;
    BOOL inside;
    NSTrackingRectTag tracking;
    @public double device_scale;
}
- (void)dealloc;
- (void)drawRect:(NSRect)rect;
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

@interface ScrollableButton : NSButton
{
    SEL scrollWheelUpAction;
    SEL scrollWheelDownAction;
}
- (void)setScrollWheelUpAction:(SEL)action;
- (void)setScrollWheelDownAction:(SEL)action;
- (void)scrollWheel:(NSEvent *)event;
@end

@interface MenuItem: NSMenuItem
{   int index;
}
+ (MenuItem*)menuItemWithTitle:(NSString*)title;
+ (MenuItem*)menuItemSelectAll;
+ (MenuItem*)menuItemInvertAll;
+ (MenuItem*)menuItemForAxis:(int)i;
- (void)toggle:(id)sender;
- (void)selectAll:(id)sender;
- (void)invertAll:(id)sender;
- (int)index;
@end

/* ---------------------------- Python classes ---------------------------- */

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
    FigureCanvas *self = (FigureCanvas*)type->tp_alloc(type, 0);
    if (!self) return NULL;
    self->view = [View alloc];
    return (PyObject*)self;
}

static int
FigureCanvas_init(FigureCanvas *self, PyObject *args, PyObject *kwds)
{
    int width;
    int height;
    if(!self->view)
    {
        PyErr_SetString(PyExc_RuntimeError, "NSView* is NULL");
        return -1;
    }

    if(!PyArg_ParseTuple(args, "ii", &width, &height)) return -1;

    NSRect rect = NSMakeRect(0.0, 0.0, width, height);
    self->view = [self->view initWithFrame: rect];
    [self->view setCanvas: (PyObject*)self];
    return 0;
}

static void
FigureCanvas_dealloc(FigureCanvas* self)
{
    if (self->view)
    {
        [self->view setCanvas: NULL];
        [self->view release];
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject*
FigureCanvas_repr(FigureCanvas* self)
{
#if PY3K
    return PyUnicode_FromFormat("FigureCanvas object %p wrapping NSView %p",
                               (void*)self, (void*)(self->view));
#else
    return PyString_FromFormat("FigureCanvas object %p wrapping NSView %p",
                               (void*)self, (void*)(self->view));
#endif
}

static PyObject*
FigureCanvas_draw(FigureCanvas* self)
{
    View* view = self->view;

    if(view) /* The figure may have been closed already */
    {
        /* Whereas drawRect creates its own autorelease pool, apparently
         * [view display] also needs one. Create and release it here. */
        NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
        [view display];
        [pool release];
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
FigureCanvas_invalidate(FigureCanvas* self)
{
    View* view = self->view;
    if(!view)
    {
        PyErr_SetString(PyExc_RuntimeError, "NSView* is NULL");
        return NULL;
    }
    [view setNeedsDisplay: YES];
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
FigureCanvas_flush_events(FigureCanvas* self)
{
    View* view = self->view;
    if(!view)
    {
        PyErr_SetString(PyExc_RuntimeError, "NSView* is NULL");
        return NULL;
    }
    [view displayIfNeeded];
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
FigureCanvas_set_rubberband(FigureCanvas* self, PyObject *args)
{
    View* view = self->view;
    int x0, y0, x1, y1;
    NSRect rubberband;
    if(!view)
    {
        PyErr_SetString(PyExc_RuntimeError, "NSView* is NULL");
        return NULL;
    }
    if(!PyArg_ParseTuple(args, "iiii", &x0, &y0, &x1, &y1)) return NULL;

    x0 /= view->device_scale;
    x1 /= view->device_scale;
    y0 /= view->device_scale;
    y1 /= view->device_scale;

    if (x1 > x0)
    {
        rubberband.origin.x = x0;
        rubberband.size.width = x1 - x0;
    }
    else
    {
        rubberband.origin.x = x1;
        rubberband.size.width = x0 - x1;
    }
    if (y1 > y0)
    {
        rubberband.origin.y = y0;
        rubberband.size.height = y1 - y0;
    }
    else
    {
        rubberband.origin.y = y1;
        rubberband.size.height = y0 - y1;
    }

    [view setRubberband: rubberband];
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
FigureCanvas_remove_rubberband(FigureCanvas* self)
{
    View* view = self->view;
    if(!view)
    {
        PyErr_SetString(PyExc_RuntimeError, "NSView* is NULL");
        return NULL;
    }
    [view removeRubberband];
    Py_INCREF(Py_None);
    return Py_None;
}

static NSImage* _read_ppm_image(PyObject* obj)
{
    int width;
    int height;
    const char* data;
    int n;
    int i;
    NSBitmapImageRep* bitmap;
    unsigned char* bitmapdata;

    if (!obj) return NULL;
    if (!PyTuple_Check(obj)) return NULL;
    if (!PyArg_ParseTuple(obj, "iit#", &width, &height, &data, &n)) return NULL;
    if (width*height*3 != n) return NULL; /* RGB image uses 3 colors / pixel */

    bitmap = [[NSBitmapImageRep alloc]
                  initWithBitmapDataPlanes: NULL
                                pixelsWide: width
                                pixelsHigh: height
                             bitsPerSample: 8
                           samplesPerPixel: 3
                                  hasAlpha: NO
                                  isPlanar: NO
                            colorSpaceName: NSDeviceRGBColorSpace
                              bitmapFormat: 0
                               bytesPerRow: width*3
                               bitsPerPixel: 24];
    if (!bitmap) return NULL;
    bitmapdata = [bitmap bitmapData];
    for (i = 0; i < n; i++) bitmapdata[i] = data[i];

    NSSize size = NSMakeSize(width, height);
    NSImage* image = [[NSImage alloc] initWithSize: size];
    if (image) [image addRepresentation: bitmap];

    [bitmap release];

    return image;
}

static PyObject*
FigureCanvas_start_event_loop(FigureCanvas* self, PyObject* args, PyObject* keywords)
{
    float timeout = 0.0;

    static char* kwlist[] = {"timeout", NULL};
    if(!PyArg_ParseTupleAndKeywords(args, keywords, "f", kwlist, &timeout))
        return NULL;

    int error;
    int interrupted = 0;
    int channel[2];
    CFSocketRef sigint_socket = NULL;
    PyOS_sighandler_t py_sigint_handler = NULL;

    CFRunLoopRef runloop = CFRunLoopGetCurrent();

    error = pipe(channel);
    if (error==0)
    {
        CFSocketContext context = {0, NULL, NULL, NULL, NULL};
        fcntl(channel[1], F_SETFL, O_WRONLY | O_NONBLOCK);

        context.info = &interrupted;
        sigint_socket = CFSocketCreateWithNative(kCFAllocatorDefault,
                                                 channel[0],
                                                 kCFSocketReadCallBack,
                                                 _sigint_callback,
                                                 &context);
        if (sigint_socket)
        {
            CFRunLoopSourceRef source;
            source = CFSocketCreateRunLoopSource(kCFAllocatorDefault,
                                                 sigint_socket,
                                                 0);
            CFRelease(sigint_socket);
            if (source)
            {
                CFRunLoopAddSource(runloop, source, kCFRunLoopDefaultMode);
                CFRelease(source);
                sigint_fd = channel[1];
                py_sigint_handler = PyOS_setsig(SIGINT, _sigint_handler);
            }
        }
        else
            close(channel[0]);
    }

    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    NSDate* date =
        (timeout > 0.0) ? [NSDate dateWithTimeIntervalSinceNow: timeout]
                        : [NSDate distantFuture];
    while (true)
    {   NSEvent* event = [NSApp nextEventMatchingMask: NSAnyEventMask
                                            untilDate: date
                                               inMode: NSDefaultRunLoopMode
                                              dequeue: YES];
       if (!event || [event type]==NSApplicationDefined) break;
       [NSApp sendEvent: event];
    }
    [pool release];

    if (py_sigint_handler) PyOS_setsig(SIGINT, py_sigint_handler);

    if (sigint_socket) CFSocketInvalidate(sigint_socket);
    if (error==0) close(channel[1]);
    if (interrupted) raise(SIGINT);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
FigureCanvas_stop_event_loop(FigureCanvas* self)
{
    NSEvent* event = [NSEvent otherEventWithType: NSApplicationDefined
                                        location: NSZeroPoint
                                   modifierFlags: 0
                                       timestamp: 0.0
                                    windowNumber: 0
                                         context: nil
                                         subtype: STOP_EVENT_LOOP
                                           data1: 0
                                           data2: 0];
    [NSApp postEvent: event atStart: true];
    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef FigureCanvas_methods[] = {
    {"draw",
     (PyCFunction)FigureCanvas_draw,
     METH_NOARGS,
     "Draws the canvas."
    },
    {"invalidate",
     (PyCFunction)FigureCanvas_invalidate,
     METH_NOARGS,
     "Invalidates the canvas."
    },
    {"flush_events",
     (PyCFunction)FigureCanvas_flush_events,
     METH_NOARGS,
     "Flush the GUI events for the figure."
    },
    {"set_rubberband",
     (PyCFunction)FigureCanvas_set_rubberband,
     METH_VARARGS,
     "Specifies a new rubberband rectangle and invalidates it."
    },
    {"remove_rubberband",
     (PyCFunction)FigureCanvas_remove_rubberband,
     METH_NOARGS,
     "Removes the current rubberband rectangle."
    },
    {"start_event_loop",
     (PyCFunction)FigureCanvas_start_event_loop,
     METH_KEYWORDS | METH_VARARGS,
     "Runs the event loop until the timeout or until stop_event_loop is called.\n",
    },
    {"stop_event_loop",
     (PyCFunction)FigureCanvas_stop_event_loop,
     METH_NOARGS,
     "Stops the event loop that was started by start_event_loop.\n",
    },
    {NULL}  /* Sentinel */
};

static char FigureCanvas_doc[] =
"A FigureCanvas object wraps a Cocoa NSView object.\n";

static PyTypeObject FigureCanvasType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_macosx.FigureCanvas",    /*tp_name*/
    sizeof(FigureCanvas),      /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)FigureCanvas_dealloc,     /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    (reprfunc)FigureCanvas_repr,     /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,        /*tp_flags*/
    FigureCanvas_doc,          /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    FigureCanvas_methods,      /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)FigureCanvas_init,      /* tp_init */
    0,                         /* tp_alloc */
    FigureCanvas_new,          /* tp_new */
};

typedef struct {
    PyObject_HEAD
    Window* window;
} FigureManager;

static PyObject*
FigureManager_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    Window* window = [Window alloc];
    if (!window) return NULL;
    FigureManager *self = (FigureManager*)type->tp_alloc(type, 0);
    if (!self)
    {
        [window release];
        return NULL;
    }
    self->window = window;
    return (PyObject*)self;
}

static int
FigureManager_init(FigureManager *self, PyObject *args, PyObject *kwds)
{
    NSRect rect;
    Window* window;
    View* view;
    const char* title;
    PyObject* size;
    int width, height;
    PyObject* obj;
    FigureCanvas* canvas;

    if(!self->window)
    {
        PyErr_SetString(PyExc_RuntimeError, "NSWindow* is NULL");
        return -1;
    }

    if(!PyArg_ParseTuple(args, "Os", &obj, &title)) return -1;

    canvas = (FigureCanvas*)obj;
    view = canvas->view;
    if (!view) /* Something really weird going on */
    {
        PyErr_SetString(PyExc_RuntimeError, "NSView* is NULL");
        return -1;
    }

    size = PyObject_CallMethod(obj, "get_width_height", "");
    if(!size) return -1;
    if(!PyArg_ParseTuple(size, "ii", &width, &height))
    {    Py_DECREF(size);
         return -1;
    }
    Py_DECREF(size);

    rect.origin.x = 100;
    rect.origin.y = 350;
    rect.size.height = height;
    rect.size.width = width;

    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    self->window = [self->window initWithContentRect: rect
                                         styleMask: NSTitledWindowMask
                                                  | NSClosableWindowMask
                                                  | NSResizableWindowMask
                                                  | NSMiniaturizableWindowMask
                                           backing: NSBackingStoreBuffered
                                             defer: YES
                                       withManager: (PyObject*)self];
    window = self->window;
    [window setTitle: [NSString stringWithCString: title
                                         encoding: NSASCIIStringEncoding]];

    [window setAcceptsMouseMovedEvents: YES];
    [window setDelegate: view];
    [window makeFirstResponder: view];
    [[window contentView] addSubview: view];

    [pool release];
    return 0;
}

static PyObject*
FigureManager_repr(FigureManager* self)
{
#if PY3K
    return PyUnicode_FromFormat("FigureManager object %p wrapping NSWindow %p",
                               (void*) self, (void*)(self->window));
#else
    return PyString_FromFormat("FigureManager object %p wrapping NSWindow %p",
                               (void*) self, (void*)(self->window));
#endif
}

static void
FigureManager_dealloc(FigureManager* self)
{
    Window* window = self->window;
    if(window)
    {
        NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
        [window close];
        [pool release];
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject*
FigureManager_show(FigureManager* self)
{
    Window* window = self->window;
    if(window)
    {
        NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
        [window makeKeyAndOrderFront: nil];
        [window orderFrontRegardless];
        [pool release];
    }
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
FigureManager_destroy(FigureManager* self)
{
    Window* window = self->window;
    if(window)
    {
        NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
        [window close];
        [pool release];
        self->window = NULL;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
FigureManager_set_window_title(FigureManager* self,
                               PyObject *args, PyObject *kwds)
{
    char* title;
    if(!PyArg_ParseTuple(args, "es", "UTF-8", &title))
        return NULL;

    Window* window = self->window;
    if(window)
    {
        NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
        NSString* ns_title = [[[NSString alloc]
                               initWithCString: title
                               encoding: NSUTF8StringEncoding] autorelease];
        [window setTitle: ns_title];
        [pool release];
    }
    PyMem_Free(title);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
FigureManager_get_window_title(FigureManager* self)
{
    Window* window = self->window;
    PyObject* result = NULL;
    if(window)
    {
        NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
        NSString* title = [window title];
        if (title) {
            const char* cTitle = [title UTF8String];
            result = PyUnicode_FromString(cTitle);
        }
        [pool release];
    }
    if (result) {
        return result;
    } else {
        Py_INCREF(Py_None);
        return Py_None;
    }
}

static PyMethodDef FigureManager_methods[] = {
    {"show",
     (PyCFunction)FigureManager_show,
     METH_NOARGS,
     "Shows the window associated with the figure manager."
    },
    {"destroy",
     (PyCFunction)FigureManager_destroy,
     METH_NOARGS,
     "Closes the window associated with the figure manager."
    },
    {"set_window_title",
     (PyCFunction)FigureManager_set_window_title,
     METH_VARARGS,
     "Sets the title of the window associated with the figure manager."
    },
    {"get_window_title",
     (PyCFunction)FigureManager_get_window_title,
     METH_NOARGS,
     "Returns the title of the window associated with the figure manager."
    },
    {NULL}  /* Sentinel */
};

static char FigureManager_doc[] =
"A FigureManager object wraps a Cocoa NSWindow object.\n";

static PyTypeObject FigureManagerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_macosx.FigureManager",   /*tp_name*/
    sizeof(FigureManager),     /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)FigureManager_dealloc,     /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    (reprfunc)FigureManager_repr,     /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,        /*tp_flags*/
    FigureManager_doc,         /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    FigureManager_methods,     /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)FigureManager_init,      /* tp_init */
    0,                         /* tp_alloc */
    FigureManager_new,          /* tp_new */
};

@interface NavigationToolbarHandler : NSObject
{   PyObject* toolbar;
}
- (NavigationToolbarHandler*)initWithToolbar:(PyObject*)toolbar;
-(void)left:(id)sender;
-(void)right:(id)sender;
-(void)up:(id)sender;
-(void)down:(id)sender;
-(void)zoominx:(id)sender;
-(void)zoominy:(id)sender;
-(void)zoomoutx:(id)sender;
-(void)zoomouty:(id)sender;
@end

typedef struct {
    PyObject_HEAD
    NSPopUpButton* menu;
    NavigationToolbarHandler* handler;
} NavigationToolbar;

@implementation NavigationToolbarHandler
- (NavigationToolbarHandler*)initWithToolbar:(PyObject*)theToolbar
{   [self init];
    toolbar = theToolbar;
    return self;
}

-(void)left:(id)sender
{   PyObject* result;
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(toolbar, "panx", "i", -1);
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);
}

-(void)right:(id)sender
{   PyObject* result;
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(toolbar, "panx", "i", 1);
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);
}

-(void)up:(id)sender
{   PyObject* result;
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(toolbar, "pany", "i", 1);
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);
}

-(void)down:(id)sender
{   PyObject* result;
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(toolbar, "pany", "i", -1);
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);
}

-(void)zoominx:(id)sender
{   PyObject* result;
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(toolbar, "zoomx", "i", 1);
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);
}

-(void)zoomoutx:(id)sender
{   PyObject* result;
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(toolbar, "zoomx", "i", -1);
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);
}

-(void)zoominy:(id)sender
{   PyObject* result;
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(toolbar, "zoomy", "i", 1);
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);
}

-(void)zoomouty:(id)sender
{   PyObject* result;
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(toolbar, "zoomy", "i", -1);
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);
}

-(void)save_figure:(id)sender
{   PyObject* result;
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(toolbar, "save_figure", "");
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);
}
@end

static PyObject*
NavigationToolbar_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    NavigationToolbarHandler* handler = [NavigationToolbarHandler alloc];
    if (!handler) return NULL;
    NavigationToolbar *self = (NavigationToolbar*)type->tp_alloc(type, 0);
    if (!self)
    {   [handler release];
        return NULL;
    }
    self->handler = handler;
    return (PyObject*)self;
}

static int
NavigationToolbar_init(NavigationToolbar *self, PyObject *args, PyObject *kwds)
{
    int i;
    NSRect rect;

    const float smallgap = 2;
    const float biggap = 10;
    const int height = 32;

    PyObject* images;
    PyObject* obj;

    FigureCanvas* canvas;
    View* view;

    obj = PyObject_GetAttrString((PyObject*)self, "canvas");
    if (obj==NULL)
    {
        PyErr_SetString(PyExc_AttributeError, "Attempt to install toolbar for NULL canvas");
        return -1;
    }
    Py_DECREF(obj); /* Don't increase the reference count */
    if (!PyObject_IsInstance(obj, (PyObject*) &FigureCanvasType))
    {
        PyErr_SetString(PyExc_TypeError, "Attempt to install toolbar for object that is not a FigureCanvas");
        return -1;
    }
    canvas = (FigureCanvas*)obj;
    view = canvas->view;
    if(!view)
    {
        PyErr_SetString(PyExc_RuntimeError, "NSView* is NULL");
        return -1;
    }

    if(!PyArg_ParseTuple(args, "O", &images)) return -1;
    if(!PyDict_Check(images)) return -1;

    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    NSRect bounds = [view bounds];
    NSWindow* window = [view window];

    bounds.origin.y += height;
    [view setFrame: bounds];

    bounds.size.height += height;
    [window setContentSize: bounds.size];

    char* imagenames[9] = {"stock_left",
                           "stock_right",
                           "stock_zoom-in",
                           "stock_zoom-out",
                           "stock_up",
                           "stock_down",
                           "stock_zoom-in",
                           "stock_zoom-out",
                           "stock_save_as"};

    NSString* tooltips[9] = {
        @"Pan left with click or wheel mouse (bidirectional)",
        @"Pan right with click or wheel mouse (bidirectional)",
        @"Zoom In X (shrink the x axis limits) with click or wheel mouse (bidirectional)",
        @"Zoom Out X (expand the x axis limits) with click or wheel mouse (bidirectional)",
        @"Pan up with click or wheel mouse (bidirectional)",
        @"Pan down with click or wheel mouse (bidirectional)",
        @"Zoom in Y (shrink the y axis limits) with click or wheel mouse (bidirectional)",
        @"Zoom Out Y (expand the y axis limits) with click or wheel mouse (bidirectional)",
        @"Save the figure"};

    SEL actions[9] = {@selector(left:),
                      @selector(right:),
                      @selector(zoominx:),
                      @selector(zoomoutx:),
                      @selector(up:),
                      @selector(down:),
                      @selector(zoominy:),
                      @selector(zoomouty:),
                      @selector(save_figure:)};

    SEL scroll_actions[9][2] = {{@selector(left:),    @selector(right:)},
                                {@selector(left:),    @selector(right:)},
                                {@selector(zoominx:), @selector(zoomoutx:)},
                                {@selector(zoominx:), @selector(zoomoutx:)},
                                {@selector(up:),      @selector(down:)},
                                {@selector(up:),      @selector(down:)},
                                {@selector(zoominy:), @selector(zoomouty:)},
                                {@selector(zoominy:), @selector(zoomouty:)},
                                {nil,nil},
                               };


    rect.size.width = 120;
    rect.size.height = 24;
    rect.origin.x = biggap;
    rect.origin.y = 0.5*(height - rect.size.height);
    self->menu = [[NSPopUpButton alloc] initWithFrame: rect
                                            pullsDown: YES];
    [self->menu setAutoenablesItems: NO];
    [[window contentView] addSubview: self->menu];
    [self->menu release];
    rect.origin.x += rect.size.width + biggap;
    rect.size.width = 24;

    self->handler = [self->handler initWithToolbar: (PyObject*)self];
    for (i = 0; i < 9; i++)
    {
        NSButton* button;
        SEL scrollWheelUpAction = scroll_actions[i][0];
        SEL scrollWheelDownAction = scroll_actions[i][1];
        if (scrollWheelUpAction && scrollWheelDownAction)
        {
            ScrollableButton* scrollable_button = [ScrollableButton alloc];
            [scrollable_button initWithFrame: rect];
            [scrollable_button setScrollWheelUpAction: scrollWheelUpAction];
            [scrollable_button setScrollWheelDownAction: scrollWheelDownAction];
            button = (NSButton*)scrollable_button;
        }
        else
        {
            button = [NSButton alloc];
            [button initWithFrame: rect];
        }
        PyObject* imagedata = PyDict_GetItemString(images, imagenames[i]);
        NSImage* image = _read_ppm_image(imagedata);
        [button setBezelStyle: NSShadowlessSquareBezelStyle];
        [button setButtonType: NSMomentaryLightButton];
        if(image)
        {
            [button setImage: image];
            [image release];
        }
        [button setToolTip: tooltips[i]];
        [button setTarget: self->handler];
        [button setAction: actions[i]];
        [[window contentView] addSubview: button];
        [button release];
        rect.origin.x += rect.size.width + smallgap;
    }
    [[window contentView] display];
    [pool release];

    return 0;
}

static void
NavigationToolbar_dealloc(NavigationToolbar *self)
{
    [self->handler release];
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject*
NavigationToolbar_repr(NavigationToolbar* self)
{
#if PY3K
    return PyUnicode_FromFormat("NavigationToolbar object %p", (void*)self);
#else
    return PyString_FromFormat("NavigationToolbar object %p", (void*)self);
#endif
}

static char NavigationToolbar_doc[] =
"NavigationToolbar\n";

static PyObject*
NavigationToolbar_update (NavigationToolbar* self)
{
    int n;
    NSPopUpButton* button = self->menu;
    if (!button)
    {
        PyErr_SetString(PyExc_RuntimeError, "Menu button is NULL");
        return NULL;
    }

    PyObject* canvas = PyObject_GetAttrString((PyObject*)self, "canvas");
    if (canvas==NULL)
    {
        PyErr_SetString(PyExc_AttributeError, "Failed to find canvas");
        return NULL;
    }
    Py_DECREF(canvas); /* Don't keep a reference here */
    PyObject* figure = PyObject_GetAttrString(canvas, "figure");
    if (figure==NULL)
    {
        PyErr_SetString(PyExc_AttributeError, "Failed to find figure");
        return NULL;
    }
    Py_DECREF(figure); /* Don't keep a reference here */
    PyObject* axes = PyObject_GetAttrString(figure, "axes");
    if (axes==NULL)
    {
        PyErr_SetString(PyExc_AttributeError, "Failed to find figure axes");
        return NULL;
    }
    Py_DECREF(axes); /* Don't keep a reference here */
    if (!PyList_Check(axes))
    {
        PyErr_SetString(PyExc_TypeError, "Figure axes is not a list");
        return NULL;
    }
    n = PyList_GET_SIZE(axes);

    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    [button removeAllItems];

    NSMenu* menu = [button menu];
    [menu addItem: [MenuItem menuItemWithTitle: @"Axes"]];

    if (n==0)
    {
        [button setEnabled: NO];
    }
    else
    {
        int i;
        [menu addItem: [MenuItem menuItemSelectAll]];
        [menu addItem: [MenuItem menuItemInvertAll]];
        [menu addItem: [NSMenuItem separatorItem]];
        for (i = 0; i < n; i++)
        {
            [menu addItem: [MenuItem menuItemForAxis: i]];
        }
        [button setEnabled: YES];
    }
    [pool release];
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
NavigationToolbar_get_active (NavigationToolbar* self)
{
    NSPopUpButton* button = self->menu;
    if (!button)
    {
        PyErr_SetString(PyExc_RuntimeError, "Menu button is NULL");
        return NULL;
    }
    NSMenu* menu = [button menu];
    NSArray* items = [menu itemArray];
    unsigned int n = [items count];
    int* states = calloc(n, sizeof(int));
    if (!states)
    {
        PyErr_SetString(PyExc_RuntimeError, "calloc failed");
        return NULL;
    }
    int i;
    unsigned int m = 0;
    NSEnumerator* enumerator = [items objectEnumerator];
    MenuItem* item;
    while ((item = [enumerator nextObject]))
    {
        if ([item isSeparatorItem]) continue;
        i = [item index];
        if (i < 0) continue;
        if ([item state]==NSOnState)
        {
            states[i] = 1;
            m++;
        }
    }
    int j = 0;
    PyObject* list = PyList_New(m);
    for (i = 0; i < n; i++)
    {
        if(states[i]==1)
        {
            PyList_SET_ITEM(list, j, PyLong_FromLong(i));
            j++;
        }
    }
    free(states);
    return list;
}

static PyMethodDef NavigationToolbar_methods[] = {
    {"update",
     (PyCFunction)NavigationToolbar_update,
     METH_NOARGS,
     "Updates the toolbar menu."
    },
    {"get_active",
     (PyCFunction)NavigationToolbar_get_active,
     METH_NOARGS,
     "Returns a list of integers identifying which items in the menu are selected."
    },
    {NULL}  /* Sentinel */
};

static PyTypeObject NavigationToolbarType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_macosx.NavigationToolbar", /*tp_name*/
    sizeof(NavigationToolbar), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)NavigationToolbar_dealloc,     /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    (reprfunc)NavigationToolbar_repr,     /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,        /*tp_flags*/
    NavigationToolbar_doc,     /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    NavigationToolbar_methods, /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)NavigationToolbar_init,      /* tp_init */
    0,                         /* tp_alloc */
    NavigationToolbar_new,     /* tp_new */
};

@interface NavigationToolbar2Handler : NSObject
{   PyObject* toolbar;
    NSButton* panbutton;
    NSButton* zoombutton;
}
- (NavigationToolbar2Handler*)initWithToolbar:(PyObject*)toolbar;
- (void)installCallbacks:(SEL[7])actions forButtons: (NSButton*[7])buttons;
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
    NSText* messagebox;
    NavigationToolbar2Handler* handler;
} NavigationToolbar2;

@implementation NavigationToolbar2Handler
- (NavigationToolbar2Handler*)initWithToolbar:(PyObject*)theToolbar
{   [self init];
    toolbar = theToolbar;
    return self;
}

- (void)installCallbacks:(SEL[7])actions forButtons: (NSButton*[7])buttons
{
    int i;
    for (i = 0; i < 7; i++)
    {
        SEL action = actions[i];
        NSButton* button = buttons[i];
        [button setTarget: self];
        [button setAction: action];
        if (action==@selector(pan:)) panbutton = button;
        if (action==@selector(zoom:)) zoombutton = button;
    }
}

-(void)home:(id)sender
{   PyObject* result;
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(toolbar, "home", "");
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);
}

-(void)back:(id)sender
{   PyObject* result;
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(toolbar, "back", "");
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);
}

-(void)forward:(id)sender
{   PyObject* result;
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(toolbar, "forward", "");
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);
}

-(void)pan:(id)sender
{   PyObject* result;
    PyGILState_STATE gstate;
    if ([sender state])
    {
        if (zoombutton) [zoombutton setState: NO];
    }
    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(toolbar, "pan", "");
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);
}

-(void)zoom:(id)sender
{   PyObject* result;
    PyGILState_STATE gstate;
    if ([sender state])
    {
        if (panbutton) [panbutton setState: NO];
    }
    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(toolbar, "zoom", "");
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);
}

-(void)configure_subplots:(id)sender
{   PyObject* canvas;
    View* view;
    PyObject* size;
    NSRect rect;
    int width, height;

    rect.origin.x = 100;
    rect.origin.y = 350;
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* master = PyObject_GetAttrString(toolbar, "canvas");
    if (master==nil)
    {
        PyErr_Print();
        PyGILState_Release(gstate);
        return;
    }
    canvas = PyObject_CallMethod(toolbar, "prepare_configure_subplots", "");
    if(!canvas)
    {
        PyErr_Print();
        Py_DECREF(master);
        PyGILState_Release(gstate);
        return;
    }

    view = ((FigureCanvas*)canvas)->view;
    if (!view) /* Something really weird going on */
    {
        PyErr_SetString(PyExc_RuntimeError, "NSView* is NULL");
        PyErr_Print();
        Py_DECREF(canvas);
        Py_DECREF(master);
        PyGILState_Release(gstate);
        return;
    }

    size = PyObject_CallMethod(canvas, "get_width_height", "");
    Py_DECREF(canvas);
    if(!size)
    {
        PyErr_Print();
        Py_DECREF(master);
        PyGILState_Release(gstate);
        return;
    }

    int ok = PyArg_ParseTuple(size, "ii", &width, &height);
    Py_DECREF(size);
    if (!ok)
    {
        PyErr_Print();
        Py_DECREF(master);
        PyGILState_Release(gstate);
        return;
    }

    NSWindow* mw = [((FigureCanvas*)master)->view window];
    Py_DECREF(master);
    PyGILState_Release(gstate);

    rect.size.width = width;
    rect.size.height = height;

    ToolWindow* window = [ [ToolWindow alloc] initWithContentRect: rect
                                                           master: mw];
    [window setContentView: view];
    [view release];
    [window makeKeyAndOrderFront: nil];
}

-(void)save_figure:(id)sender
{   PyObject* result;
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(toolbar, "save_figure", "");
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);
}
@end

static PyObject*
NavigationToolbar2_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    NavigationToolbar2Handler* handler = [NavigationToolbar2Handler alloc];
    if (!handler) return NULL;
    NavigationToolbar2 *self = (NavigationToolbar2*)type->tp_alloc(type, 0);
    if (!self)
    {
        [handler release];
        return NULL;
    }
    self->handler = handler;
    return (PyObject*)self;
}

static int
NavigationToolbar2_init(NavigationToolbar2 *self, PyObject *args, PyObject *kwds)
{
    PyObject* obj;
    FigureCanvas* canvas;
    View* view;

    int i;
    NSRect rect;
    NSSize size;
    NSSize scale;

    const float gap = 2;
    const int height = 36;
    const int imagesize = 24;

    const char* basedir;

    obj = PyObject_GetAttrString((PyObject*)self, "canvas");
    if (obj==NULL)
    {
        PyErr_SetString(PyExc_AttributeError, "Attempt to install toolbar for NULL canvas");
        return -1;
    }
    Py_DECREF(obj); /* Don't increase the reference count */
    if (!PyObject_IsInstance(obj, (PyObject*) &FigureCanvasType))
    {
        PyErr_SetString(PyExc_TypeError, "Attempt to install toolbar for object that is not a FigureCanvas");
        return -1;
    }
    canvas = (FigureCanvas*)obj;
    view = canvas->view;
    if(!view)
    {
        PyErr_SetString(PyExc_RuntimeError, "NSView* is NULL");
        return -1;
    }

    if(!PyArg_ParseTuple(args, "s", &basedir)) return -1;

    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    NSRect bounds = [view bounds];
    NSWindow* window = [view window];

    bounds.origin.y += height;
    [view setFrame: bounds];

    bounds.size.height += height;
    [window setContentSize: bounds.size];

    NSString* dir = [NSString stringWithCString: basedir
                                       encoding: NSASCIIStringEncoding];

    NSButton* buttons[7];

    NSString* images[7] = {@"home.pdf",
                           @"back.pdf",
                           @"forward.pdf",
                           @"move.pdf",
                           @"zoom_to_rect.pdf",
                           @"subplots.pdf",
                           @"filesave.pdf"};

    NSString* tooltips[7] = {@"Reset original view",
                             @"Back to  previous view",
                             @"Forward to next view",
                             @"Pan axes with left mouse, zoom with right",
                             @"Zoom to rectangle",
                             @"Configure subplots",
                             @"Save the figure"};

    SEL actions[7] = {@selector(home:),
                      @selector(back:),
                      @selector(forward:),
                      @selector(pan:),
                      @selector(zoom:),
                      @selector(configure_subplots:),
                      @selector(save_figure:)};

    NSButtonType buttontypes[7] = {NSMomentaryLightButton,
                                   NSMomentaryLightButton,
                                   NSMomentaryLightButton,
                                   NSPushOnPushOffButton,
                                   NSPushOnPushOffButton,
                                   NSMomentaryLightButton,
                                   NSMomentaryLightButton};

    rect.origin.x = 0;
    rect.origin.y = 0;
    rect.size.width = imagesize;
    rect.size.height = imagesize;
#ifdef COMPILING_FOR_10_7
    rect = [window convertRectToBacking: rect];
#endif
    size = rect.size;
    scale.width = imagesize / size.width;
    scale.height = imagesize / size.height;

    rect.size.width = 32;
    rect.size.height = 32;
    rect.origin.x = gap;
    rect.origin.y = 0.5*(height - rect.size.height);

    for (i = 0; i < 7; i++)
    {
        NSString* filename = [dir stringByAppendingPathComponent: images[i]];
        NSImage* image = [[NSImage alloc] initWithContentsOfFile: filename];
        buttons[i] = [[NSButton alloc] initWithFrame: rect];
        [image setSize: size];
        [buttons[i] setBezelStyle: NSShadowlessSquareBezelStyle];
        [buttons[i] setButtonType: buttontypes[i]];
        [buttons[i] setImage: image];
        [buttons[i] scaleUnitSquareToSize: scale];
        [buttons[i] setImagePosition: NSImageOnly];
        [buttons[i] setToolTip: tooltips[i]];
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
    NSText* messagebox = [[NSText alloc] initWithFrame: rect];
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

    [pool release];

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
#if PY3K
    return PyUnicode_FromFormat("NavigationToolbar2 object %p", (void*)self);
#else
    return PyString_FromFormat("NavigationToolbar2 object %p", (void*)self);
#endif
}

static char NavigationToolbar2_doc[] =
"NavigationToolbar2\n";

static PyObject*
NavigationToolbar2_set_message(NavigationToolbar2 *self, PyObject* args)
{
    const char* message;

#if PY3K
    if(!PyArg_ParseTuple(args, "y", &message)) return NULL;
#else
    if(!PyArg_ParseTuple(args, "s", &message)) return NULL;
#endif

    NSText* messagebox = self->messagebox;

    if (messagebox)
    {   NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
        NSString* text = [NSString stringWithUTF8String: message];
        [messagebox setString: text];
        [pool release];
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef NavigationToolbar2_methods[] = {
    {"set_message",
     (PyCFunction)NavigationToolbar2_set_message,
     METH_VARARGS,
     "Set the message to be displayed on the toolbar."
    },
    {NULL}  /* Sentinel */
};

static PyTypeObject NavigationToolbar2Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_macosx.NavigationToolbar2", /*tp_name*/
    sizeof(NavigationToolbar2), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)NavigationToolbar2_dealloc,     /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    (reprfunc)NavigationToolbar2_repr,     /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,        /*tp_flags*/
    NavigationToolbar2_doc,    /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    NavigationToolbar2_methods, /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)NavigationToolbar2_init,      /* tp_init */
    0,                         /* tp_alloc */
    NavigationToolbar2_new,    /* tp_new */
};

static PyObject*
choose_save_file(PyObject* unused, PyObject* args)
{
    int result;
    const char* title;
    char* default_filename;
    if(!PyArg_ParseTuple(args, "ses", &title, "UTF-8", &default_filename))
        return NULL;

    NSSavePanel* panel = [NSSavePanel savePanel];
    [panel setTitle: [NSString stringWithCString: title
                                        encoding: NSASCIIStringEncoding]];
    NSString* ns_default_filename =
        [[NSString alloc]
         initWithCString: default_filename
         encoding: NSUTF8StringEncoding];
    PyMem_Free(default_filename);
#ifdef COMPILING_FOR_10_6
    [panel setNameFieldStringValue: ns_default_filename];
    result = [panel runModal];
#else
    result = [panel runModalForDirectory: nil file: ns_default_filename];
#endif
    [ns_default_filename release];
#ifdef COMPILING_FOR_10_10
    if (result == NSModalResponseOK)
#else
    if (result == NSOKButton)
#endif
    {
#ifdef COMPILING_FOR_10_6
        NSURL* url = [panel URL];
        NSString* filename = [url path];
        if (!filename) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to obtain filename");
            return 0;
        }
#else
        NSString* filename = [panel filename];
#endif
        unsigned int n = [filename length];
        unichar* buffer = malloc(n*sizeof(unichar));
        [filename getCharacters: buffer];
#if PY33
        PyObject* string =  PyUnicode_FromKindAndData(PyUnicode_2BYTE_KIND, buffer, n);
#else
        PyObject* string =  PyUnicode_FromUnicode(buffer, n);
#endif
        free(buffer);
        return string;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
set_cursor(PyObject* unused, PyObject* args)
{
    int i;
    if(!PyArg_ParseTuple(args, "i", &i)) return NULL;
    switch (i)
    { case 0: [[NSCursor pointingHandCursor] set]; break;
      case 1: [[NSCursor arrowCursor] set]; break;
      case 2: [[NSCursor crosshairCursor] set]; break;
      case 3: [[NSCursor openHandCursor] set]; break;
      default: return NULL;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

@implementation WindowServerConnectionManager
static WindowServerConnectionManager *sharedWindowServerConnectionManager = nil;

+ (WindowServerConnectionManager *)sharedManager
{
    if (sharedWindowServerConnectionManager == nil)
    {
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
    PyObject* result;
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(manager, "close", "");
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);
    return YES;
}

- (void)close
{
    [super close];
    NSArray *windowsArray = [NSApp windows];
    if([windowsArray count]==0) [NSApp stop: self];
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

@implementation ToolWindow
- (ToolWindow*)initWithContentRect:(NSRect)rect master:(NSWindow*)window
{
    [self initWithContentRect: rect
                    styleMask: NSTitledWindowMask
                             | NSClosableWindowMask
                             | NSResizableWindowMask
                             | NSMiniaturizableWindowMask
                      backing: NSBackingStoreBuffered
                        defer: YES];
    [self setTitle: @"Subplot Configuration Tool"];
    [[NSNotificationCenter defaultCenter] addObserver: self
                                             selector: @selector(masterCloses:)
                                                 name: NSWindowWillCloseNotification
                                               object: window];
    return self;
}

- (void)masterCloses:(NSNotification*)notification
{
    [self close];
}

- (void)close
{
    [[NSNotificationCenter defaultCenter] removeObserver: self];
    [super close];
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
    inside = false;
    tracking = 0;
    device_scale = 1;
    return self;
}

- (void)dealloc
{
    FigureCanvas* fc = (FigureCanvas*)canvas;
    if (fc) fc->view = NULL;
    [self removeTrackingRect: tracking];
    [super dealloc];
}

- (void)setCanvas: (PyObject*)newCanvas
{
    canvas = newCanvas;
}

static void _buffer_release(void* info, const void* data, size_t size) {
    PyBuffer_Release((Py_buffer *)info);
}

static int _copy_agg_buffer(CGContextRef cr, PyObject *renderer)
{
    Py_buffer buffer;

    if (PyObject_GetBuffer(renderer, &buffer, PyBUF_CONTIG_RO) == -1) {
        PyErr_Print();
        return 1;
    }

    if (buffer.ndim != 3 || buffer.shape[2] != 4) {
        PyBuffer_Release(&buffer);
        return 1;
    }

    const Py_ssize_t nrows = buffer.shape[0];
    const Py_ssize_t ncols = buffer.shape[1];
    const size_t bytesPerComponent = 1;
    const size_t bitsPerComponent = 8 * bytesPerComponent;
    const size_t nComponents = 4; /* red, green, blue, alpha */
    const size_t bitsPerPixel = bitsPerComponent * nComponents;
    const size_t bytesPerRow = nComponents * bytesPerComponent * ncols;

    CGColorSpaceRef colorspace = CGColorSpaceCreateWithName(kCGColorSpaceGenericRGB);
    if (!colorspace) {
        PyBuffer_Release(&buffer);
        return 1;
    }

    CGDataProviderRef provider = CGDataProviderCreateWithData(&buffer,
                                                              buffer.buf,
                                                              buffer.len,
                                                              _buffer_release);
    if (!provider) {
        PyBuffer_Release(&buffer);
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
        PyBuffer_Release(&buffer);
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

    CGContextRef cr = [[NSGraphicsContext currentContext] graphicsPort];

    double new_device_scale = _get_device_scale(cr);

    if (device_scale != new_device_scale) {
        device_scale = new_device_scale;
        if (!PyObject_CallMethod(canvas, "_set_device_scale", "d", device_scale, NULL)) {
            PyErr_Print();
            goto exit;
        }
    }

    renderer = PyObject_CallMethod(canvas, "_draw", "", NULL);
    if (!renderer)
    {
        PyErr_Print();
        goto exit;
    }

    renderer_buffer = PyObject_GetAttrString(renderer, "_renderer");
    if (!renderer_buffer) {
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

- (void)windowDidResize: (NSNotification*)notification
{
    int width, height;
    Window* window = [notification object];
    NSSize size = [[window contentView] frame].size;
    NSRect rect = [self frame];

    size.height -= rect.origin.y;
    width = size.width;
    height = size.height;

    [self setFrameSize: size];

    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* result = PyObject_CallMethod(
            canvas, "resize", "ii", width, height);
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);
    if (tracking) [self removeTrackingRect: tracking];
    tracking = [self addTrackingRect: [self bounds]
                               owner: self
                            userData: nil
                        assumeInside: NO];
    [self setNeedsDisplay: YES];
}

- (void)windowWillClose:(NSNotification*)notification
{
    PyGILState_STATE gstate;
    PyObject* result;

    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(canvas, "close_event", "");
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);
}

- (BOOL)windowShouldClose:(NSNotification*)notification
{
    NSWindow* window = [self window];
    NSEvent* event = [NSEvent otherEventWithType: NSApplicationDefined
                                        location: NSZeroPoint
                                   modifierFlags: 0
                                       timestamp: 0.0
                                    windowNumber: 0
                                         context: nil
                                         subtype: WINDOW_CLOSING
                                           data1: 0
                                           data2: 0];
    [NSApp postEvent: event atStart: true];
    if ([window respondsToSelector: @selector(closeButtonPressed)])
    { BOOL closed = [((Window*) window) closeButtonPressed];
      /* If closed, the window has already been closed via the manager. */
      if (closed) return NO;
    }
    return YES;
}

- (void)mouseEntered:(NSEvent *)event
{
    PyGILState_STATE gstate;
    PyObject* result;
    NSWindow* window = [self window];
    if ([window isKeyWindow]==false) return;

    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(canvas, "enter_notify_event", "");
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);

    [window setAcceptsMouseMovedEvents: YES];
    inside = true;
}

- (void)mouseExited:(NSEvent *)event
{
    PyGILState_STATE gstate;
    PyObject* result;
    NSWindow* window = [self window];
    if ([window isKeyWindow]==false) return;

    if (inside==false) return;
    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(canvas, "leave_notify_event", "");
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);

    [[self window] setAcceptsMouseMovedEvents: NO];
    inside = false;
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
    {    case NSLeftMouseDown:
         {   unsigned int modifier = [event modifierFlags];
             if (modifier & NSControlKeyMask)
                 /* emulate a right-button click */
                 num = 3;
             else if (modifier & NSAlternateKeyMask)
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
         case NSOtherMouseDown: num = 2; break;
         case NSRightMouseDown: num = 3; break;
         default: return; /* Unknown mouse event */
    }
    if ([event clickCount] == 2) {
      dblclick = 1;
    }
    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(canvas, "button_press_event", "iiii", x, y, num, dblclick);
    if(result)
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
    {    case NSLeftMouseUp:
             num = 1;
             if ([NSCursor currentCursor]==[NSCursor closedHandCursor])
                 [[NSCursor openHandCursor] set];
             break;
         case NSOtherMouseUp: num = 2; break;
         case NSRightMouseUp: num = 3; break;
         default: return; /* Unknown mouse event */
    }
    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(canvas, "button_release_event", "iii", x, y, num);
    if(result)
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
    if(result)
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
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();

    PyGILState_Release(gstate);
}

- (void)rightMouseDown:(NSEvent *)event
{
    int x, y;
    int num = 3;
    int dblclick = 0;
    PyObject* result;
    PyGILState_STATE gstate;
    NSPoint location = [event locationInWindow];
    location = [self convertPoint: location fromView: nil];
    x = location.x * device_scale;
    y = location.y * device_scale;
    gstate = PyGILState_Ensure();
    if ([event clickCount] == 2) {
      dblclick = 1;
    }
    result = PyObject_CallMethod(canvas, "button_press_event", "iiii", x, y, num, dblclick);
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();

    PyGILState_Release(gstate);
}

- (void)rightMouseUp:(NSEvent *)event
{
    int x, y;
    int num = 3;
    PyObject* result;
    PyGILState_STATE gstate;
    NSPoint location = [event locationInWindow];
    location = [self convertPoint: location fromView: nil];
    x = location.x * device_scale;
    y = location.y * device_scale;
    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(canvas, "button_release_event", "iii", x, y, num);
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();

    PyGILState_Release(gstate);
}

- (void)rightMouseDragged:(NSEvent *)event
{
    int x, y;
    NSPoint location = [event locationInWindow];
    location = [self convertPoint: location fromView: nil];
    x = location.x * device_scale;
    y = location.y * device_scale;
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* result = PyObject_CallMethod(canvas, "motion_notify_event", "ii", x, y);
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();

    PyGILState_Release(gstate);
}

- (void)otherMouseDown:(NSEvent *)event
{
    int x, y;
    int num = 2;
    int dblclick = 0;
    PyObject* result;
    PyGILState_STATE gstate;
    NSPoint location = [event locationInWindow];
    location = [self convertPoint: location fromView: nil];
    x = location.x * device_scale;
    y = location.y * device_scale;
    gstate = PyGILState_Ensure();
    if ([event clickCount] == 2) {
      dblclick = 1;
    }
    result = PyObject_CallMethod(canvas, "button_press_event", "iiii", x, y, num, dblclick);
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();

    PyGILState_Release(gstate);
}

- (void)otherMouseUp:(NSEvent *)event
{
    int x, y;
    int num = 2;
    PyObject* result;
    PyGILState_STATE gstate;
    NSPoint location = [event locationInWindow];
    location = [self convertPoint: location fromView: nil];
    x = location.x * device_scale;
    y = location.y * device_scale;
    gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(canvas, "button_release_event", "iii", x, y, num);
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();

    PyGILState_Release(gstate);
}

- (void)otherMouseDragged:(NSEvent *)event
{
    int x, y;
    NSPoint location = [event locationInWindow];
    location = [self convertPoint: location fromView: nil];
    x = location.x * device_scale;
    y = location.y * device_scale;
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* result = PyObject_CallMethod(canvas, "motion_notify_event", "ii", x, y);
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();

    PyGILState_Release(gstate);
}

- (void)setRubberband:(NSRect)rect
{
    if (!NSIsEmptyRect(rubberband)) [self setNeedsDisplayInRect: rubberband];
    rubberband = rect;
    [self setNeedsDisplayInRect: rubberband];
}

- (void)removeRubberband
{
    if (NSIsEmptyRect(rubberband)) return;
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
    if ([event modifierFlags] & NSControlKeyMask)
        [returnkey appendString:@"ctrl+" ];
    if ([event modifierFlags] & NSAlternateKeyMask)
        [returnkey appendString:@"alt+" ];
    if ([event modifierFlags] & NSCommandKeyMask)
        [returnkey appendString:@"cmd+" ];

    unichar uc = [[event charactersIgnoringModifiers] characterAtIndex:0];
    NSString* specialchar = [specialkeymappings objectForKey:[NSNumber numberWithUnsignedLong:uc]];
    if (specialchar)
        [returnkey appendString:specialchar];
    else
        [returnkey appendString:[event charactersIgnoringModifiers]];

    return [returnkey UTF8String];
}

- (void)keyDown:(NSEvent*)event
{
    PyObject* result;
    const char* s = [self convertKeyEvent: event];
    PyGILState_STATE gstate = PyGILState_Ensure();
    if (s==NULL)
    {
        result = PyObject_CallMethod(canvas, "key_press_event", "O", Py_None);
    }
    else
    {
        result = PyObject_CallMethod(canvas, "key_press_event", "s", s);
    }
    if(result)
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
    if (s==NULL)
    {
        result = PyObject_CallMethod(canvas, "key_release_event", "O", Py_None);
    }
    else
    {
        result = PyObject_CallMethod(canvas, "key_release_event", "s", s);
    }
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();

    PyGILState_Release(gstate);
}

- (void)scrollWheel:(NSEvent*)event
{
    int step;
    float d = [event deltaY];
    if (d > 0) step = 1;
    else if (d < 0) step = -1;
    else return;
    NSPoint location = [event locationInWindow];
    NSPoint point = [self convertPoint: location fromView: nil];
    int x = (int)round(point.x * device_scale);
    int y = (int)round(point.y * device_scale - 1);

    PyObject* result;
    PyGILState_STATE gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(canvas, "scroll_event", "iii", x, y, step);
    if(result)
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
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();

    PyGILState_Release(gstate);
}
 */
@end

@implementation ScrollableButton
- (void)setScrollWheelUpAction:(SEL)action
{
    scrollWheelUpAction = action;
}

- (void)setScrollWheelDownAction:(SEL)action
{
    scrollWheelDownAction = action;
}

- (void)scrollWheel:(NSEvent*)event
{
    float d = [event deltaY];
    Window* target = [self target];
    if (d > 0)
        [NSApp sendAction: scrollWheelUpAction to: target from: self];
    else if (d < 0)
        [NSApp sendAction: scrollWheelDownAction to: target from: self];
}
@end

@implementation MenuItem
+ (MenuItem*)menuItemWithTitle: (NSString*)title
{
    MenuItem* item = [[MenuItem alloc] initWithTitle: title
                                              action: nil
                                       keyEquivalent: @""];
    item->index = -1;
    return [item autorelease];
}

+ (MenuItem*)menuItemForAxis: (int)i
{
    NSString* title = [NSString stringWithFormat: @"Axis %d", i+1];
    MenuItem* item = [[MenuItem alloc] initWithTitle: title
                                              action: @selector(toggle:)
                                       keyEquivalent: @""];
    [item setTarget: item];
    [item setState: NSOnState];
    item->index = i;
    return [item autorelease];
}

+ (MenuItem*)menuItemSelectAll
{
    MenuItem* item = [[MenuItem alloc] initWithTitle: @"Select All"
                                              action: @selector(selectAll:)
                                       keyEquivalent: @""];
    [item setTarget: item];
    item->index = -1;
    return [item autorelease];
}

+ (MenuItem*)menuItemInvertAll
{
    MenuItem* item = [[MenuItem alloc] initWithTitle: @"Invert All"
                                              action: @selector(invertAll:)
                                       keyEquivalent: @""];
    [item setTarget: item];
    item->index = -1;
    return [item autorelease];
}

- (void)toggle:(id)sender
{
    if ([self state]) [self setState: NSOffState];
    else [self setState: NSOnState];
}

- (void)selectAll:(id)sender
{
    NSMenu* menu = [sender menu];
    if(!menu) return; /* Weird */
    NSArray* items = [menu itemArray];
    NSEnumerator* enumerator = [items objectEnumerator];
    MenuItem* item;
    while ((item = [enumerator nextObject]))
    {
        if (item->index >= 0) [item setState: NSOnState];
    }
}

- (void)invertAll:(id)sender
{
    NSMenu* menu = [sender menu];
    if(!menu) return; /* Weird */
    NSArray* items = [menu itemArray];
    NSEnumerator* enumerator = [items objectEnumerator];
    MenuItem* item;
    while ((item = [enumerator nextObject]))
    {
        if (item->index < 0) continue;
        if ([item state]==NSOffState) [item setState: NSOnState];
        else [item setState: NSOffState];
    }
}

- (int)index
{
    return self->index;
}
@end

static PyObject*
show(PyObject* self)
{
    [NSApp activateIgnoringOtherApps: YES];
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    NSArray *windowsArray = [NSApp windows];
    NSEnumerator *enumerator = [windowsArray objectEnumerator];
    NSWindow *window;
    while ((window = [enumerator nextObject])) {
        [window orderFront:nil];
    }
    [pool release];
    Py_BEGIN_ALLOW_THREADS
    [NSApp run];
    Py_END_ALLOW_THREADS
    Py_INCREF(Py_None);
    return Py_None;
}

typedef struct {
    PyObject_HEAD
    CFRunLoopTimerRef timer;
} Timer;

static PyObject*
Timer_new(PyTypeObject* type, PyObject *args, PyObject *kwds)
{
    Timer* self = (Timer*)type->tp_alloc(type, 0);
    if (!self) return NULL;
    self->timer = NULL;
    return (PyObject*) self;
}

static PyObject*
Timer_repr(Timer* self)
{
#if PY3K
    return PyUnicode_FromFormat("Timer object %p wrapping CFRunLoopTimerRef %p",
                               (void*) self, (void*)(self->timer));
#else
    return PyString_FromFormat("Timer object %p wrapping CFRunLoopTimerRef %p",
                               (void*) self, (void*)(self->timer));
#endif
}

static char Timer_doc[] =
"A Timer object wraps a CFRunLoopTimerRef and can add it to the event loop.\n";

static void timer_callback(CFRunLoopTimerRef timer, void* info)
{
    PyObject* method = info;
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* result = PyObject_CallFunction(method, NULL);
    if (result==NULL) PyErr_Print();
    PyGILState_Release(gstate);
}

static PyObject*
Timer__timer_start(Timer* self, PyObject* args)
{
    CFRunLoopRef runloop;
    CFRunLoopTimerRef timer;
    CFRunLoopTimerContext context;
    double milliseconds;
    CFTimeInterval interval;
    PyObject* attribute;
    PyObject* failure;
    runloop = CFRunLoopGetCurrent();
    if (!runloop) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to obtain run loop");
        return NULL;
    }
    context.version = 0;
    context.retain = 0;
    context.release = 0;
    context.copyDescription = 0;
    attribute = PyObject_GetAttrString((PyObject*)self, "_interval");
    if (attribute==NULL)
    {
        PyErr_SetString(PyExc_AttributeError, "Timer has no attribute '_interval'");
        return NULL;
    }
    milliseconds = PyFloat_AsDouble(attribute);
    failure = PyErr_Occurred();
    Py_DECREF(attribute);
    if (failure) return NULL;
    attribute = PyObject_GetAttrString((PyObject*)self, "_single");
    if (attribute==NULL)
    {
        PyErr_SetString(PyExc_AttributeError, "Timer has no attribute '_single'");
        return NULL;
    }
    switch (PyObject_IsTrue(attribute)) {
        case 1:
            interval = 0;
            break;
        case 0:
            interval = milliseconds / 1000.0;
            break;
        case -1:
        default:
            PyErr_SetString(PyExc_ValueError, "Cannot interpret _single attribute as True of False");
            return NULL;
    }
    attribute = PyObject_GetAttrString((PyObject*)self, "_on_timer");
    if (attribute==NULL)
    {
        PyErr_SetString(PyExc_AttributeError, "Timer has no attribute '_on_timer'");
        return NULL;
    }
    if (!PyMethod_Check(attribute)) {
        PyErr_SetString(PyExc_RuntimeError, "_on_timer should be a Python method");
        return NULL;
    }
    context.info = attribute;
    timer = CFRunLoopTimerCreate(kCFAllocatorDefault,
                                 0,
                                 interval,
                                 0,
                                 0,
                                 timer_callback,
                                 &context);
    if (!timer) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create timer");
        return NULL;
    }
    Py_INCREF(attribute);
    if (self->timer) {
        CFRunLoopTimerGetContext(self->timer, &context);
        attribute = context.info;
        Py_DECREF(attribute);
        CFRunLoopRemoveTimer(runloop, self->timer, kCFRunLoopCommonModes);
        CFRelease(self->timer);
    }
    CFRunLoopAddTimer(runloop, timer, kCFRunLoopCommonModes);
    /* Don't release the timer here, since the run loop may be destroyed and
     * the timer lost before we have a chance to decrease the reference count
     * of the attribute */
    self->timer = timer;
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
Timer__timer_stop(Timer* self)
{
    if (self->timer) {
        PyObject* attribute;
        CFRunLoopTimerContext context;
        CFRunLoopTimerGetContext(self->timer, &context);
        attribute = context.info;
        Py_DECREF(attribute);
        CFRunLoopRef runloop = CFRunLoopGetCurrent();
        if (runloop) {
            CFRunLoopRemoveTimer(runloop, self->timer, kCFRunLoopCommonModes);
        }
        CFRelease(self->timer);
        self->timer = NULL;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

static void
Timer_dealloc(Timer* self)
{
    Timer__timer_stop(self);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyMethodDef Timer_methods[] = {
    {"_timer_start",
     (PyCFunction)Timer__timer_start,
     METH_VARARGS,
     "Initialize and start the timer."
    },
    {"_timer_stop",
     (PyCFunction)Timer__timer_stop,
     METH_NOARGS,
     "Stop the timer."
    },
    {NULL}  /* Sentinel */
};

static PyTypeObject TimerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_macosx.Timer",           /*tp_name*/
    sizeof(Timer),             /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)Timer_dealloc,     /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    (reprfunc)Timer_repr,      /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,        /*tp_flags*/
    Timer_doc,                 /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    Timer_methods,             /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    Timer_new,                 /* tp_new */
};

static bool verify_framework(void)
{
#ifdef COMPILING_FOR_10_6
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    NSRunningApplication* app = [NSRunningApplication currentApplication];
    NSApplicationActivationPolicy activationPolicy = [app activationPolicy];
    [pool release];
    switch (activationPolicy) {
        case NSApplicationActivationPolicyRegular:
        case NSApplicationActivationPolicyAccessory:
            return true;
        case NSApplicationActivationPolicyProhibited:
            break;
    }
#else
    ProcessSerialNumber psn;
    if (CGMainDisplayID()!=0
     && GetCurrentProcess(&psn)==noErr
     && SetFrontProcess(&psn)==noErr) return true;
#endif
    PyErr_SetString(PyExc_RuntimeError,
        "Python is not installed as a framework. The Mac OS X backend will "
        "not be able to function correctly if Python is not installed as a "
        "framework. See the Python documentation for more information on "
        "installing Python as a framework on Mac OS X. Please either reinstall "
        "Python as a framework, or try one of the other backends. If you are "
        "Working with Matplotlib in a virtual enviroment see 'Working with "
        "Matplotlib in Virtual environments' in the Matplotlib FAQ");
    return false;
}

static struct PyMethodDef methods[] = {
   {"show",
    (PyCFunction)show,
    METH_NOARGS,
    "Show all the figures and enter the main loop.\nThis function does not return until all Matplotlib windows are closed,\nand is normally not needed in interactive sessions."
   },
   {"choose_save_file",
    (PyCFunction)choose_save_file,
    METH_VARARGS,
    "Closes the window."
   },
   {"set_cursor",
    (PyCFunction)set_cursor,
    METH_VARARGS,
    "Sets the active cursor."
   },
   {NULL,          NULL, 0, NULL}/* sentinel */
};

#if PY3K

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_macosx",
    "Mac OS X native backend",
    -1,
    methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyObject* PyInit__macosx(void)

#else

void init_macosx(void)
#endif
{
    PyObject *module;

    if (PyType_Ready(&FigureCanvasType) < 0
     || PyType_Ready(&FigureManagerType) < 0
     || PyType_Ready(&NavigationToolbarType) < 0
     || PyType_Ready(&NavigationToolbar2Type) < 0
     || PyType_Ready(&TimerType) < 0)
#if PY3K
        return NULL;
#else
        return;
#endif

    NSApp = [NSApplication sharedApplication];

    if (!verify_framework())
#if PY3K
        return NULL;
#else
        return;
#endif

#if PY3K
    module = PyModule_Create(&moduledef);
    if (module==NULL) return NULL;
#else
    module = Py_InitModule4("_macosx",
                            methods,
                            "Mac OS X native backend",
                            NULL,
                            PYTHON_API_VERSION);
#endif

    Py_INCREF(&FigureCanvasType);
    Py_INCREF(&FigureManagerType);
    Py_INCREF(&NavigationToolbarType);
    Py_INCREF(&NavigationToolbar2Type);
    Py_INCREF(&TimerType);
    PyModule_AddObject(module, "FigureCanvas", (PyObject*) &FigureCanvasType);
    PyModule_AddObject(module, "FigureManager", (PyObject*) &FigureManagerType);
    PyModule_AddObject(module, "NavigationToolbar", (PyObject*) &NavigationToolbarType);
    PyModule_AddObject(module, "NavigationToolbar2", (PyObject*) &NavigationToolbar2Type);
    PyModule_AddObject(module, "Timer", (PyObject*) &TimerType);

    PyOS_InputHook = wait_for_stdin;

    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    WindowServerConnectionManager* connectionManager = [WindowServerConnectionManager sharedManager];
    NSWorkspace* workspace = [NSWorkspace sharedWorkspace];
    NSNotificationCenter* notificationCenter = [workspace notificationCenter];
    [notificationCenter addObserver: connectionManager
                           selector: @selector(launch:)
                               name: NSWorkspaceDidLaunchApplicationNotification
                             object: nil];
    [pool release];
#if PY3K
    return module;
#endif
}
