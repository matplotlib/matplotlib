#include <Cocoa/Cocoa.h>
#include <ApplicationServices/ApplicationServices.h>
#include <sys/socket.h>
#include <Python.h>
#include "numpy/arrayobject.h"
#include "path_cleanup.h"

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

/* Must define PyVarObject_HEAD_INIT for Python 2.5 or older */
#ifndef PyVarObject_HEAD_INIT
#define PyVarObject_HEAD_INIT(type, size)       \
        PyObject_HEAD_INIT(type) size,
#endif

/* Proper way to check for the OS X version we are compiling for, from
   http://developer.apple.com/documentation/DeveloperTools/Conceptual/cross_development */
#if __MAC_OS_X_VERSION_MIN_REQUIRED >= 1050
#define COMPILING_FOR_10_5
#endif
#if __MAC_OS_X_VERSION_MIN_REQUIRED >= 1060
#define COMPILING_FOR_10_6
#endif

static int nwin = 0;   /* The number of open windows */

/* Use Atsui for Mac OS X 10.4, CoreText for Mac OS X 10.5 */
#ifndef COMPILING_FOR_10_5
static int ngc = 0;    /* The number of graphics contexts in use */


/* For drawing Unicode strings with ATSUI */
static ATSUStyle style = NULL;
static ATSUTextLayout layout = NULL;
#endif

/* CGFloat was defined in Mac OS X 10.5 */
#ifndef CGFLOAT_DEFINED
#define CGFloat float
#endif


/* Various NSApplicationDefined event subtypes */
#define STDIN_READY 0
#define SIGINT_CALLED 1
#define STOP_EVENT_LOOP 2
#define WINDOW_CLOSING 3

/* Path definitions */
#define STOP      0
#define MOVETO    1
#define LINETO    2
#define CURVE3    3
#define CURVE4    4
#define CLOSEPOLY 0x4f

/* Hatching */
#define HATCH_SIZE 72

/* -------------------------- Helper function ---------------------------- */

static void stdin_ready(CFReadStreamRef readStream, CFStreamEventType eventType, void* context)
{
    NSEvent* event = [NSEvent otherEventWithType: NSApplicationDefined
                                        location: NSZeroPoint
                                   modifierFlags: 0
                                       timestamp: 0.0
                                    windowNumber: 0
                                         context: nil
                                         subtype: STDIN_READY
                                           data1: 0
                                           data2: 0];
    [NSApp postEvent: event atStart: true];
}

static int sigint_fd = -1;

static void _sigint_handler(int sig)
{
    const char c = 'i';
    write(sigint_fd, &c, 1);
}

static void _callback(CFSocketRef s,
                      CFSocketCallBackType type,
                      CFDataRef address,
                      const void * data,
                      void *info)
{
    char c;
    CFSocketNativeHandle handle = CFSocketGetNative(s);
    read(handle, &c, 1);
    NSEvent* event = [NSEvent otherEventWithType: NSApplicationDefined
                                        location: NSZeroPoint
                                   modifierFlags: 0
                                       timestamp: 0.0
                                    windowNumber: 0
                                         context: nil
                                         subtype: SIGINT_CALLED
                                           data1: 0
                                           data2: 0];
    [NSApp postEvent: event atStart: true];
}

static int wait_for_stdin(void)
{
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
    if (!CFReadStreamHasBytesAvailable(stream))
    /* This is possible because of how PyOS_InputHook is called from Python */
    {
        int error;
        int interrupted = 0;
        int channel[2];
        CFSocketRef sigint_socket = NULL;
        PyOS_sighandler_t py_sigint_handler = NULL;
        CFStreamClientContext clientContext = {0, NULL, NULL, NULL, NULL};
        CFReadStreamSetClient(stream,
                              kCFStreamEventHasBytesAvailable,
                              stdin_ready,
                              &clientContext);
        CFReadStreamScheduleWithRunLoop(stream, runloop, kCFRunLoopCommonModes);
        error = pipe(channel);
        if (error==0)
        {
            fcntl(channel[1], F_SETFL, O_WRONLY | O_NONBLOCK);

            sigint_socket = CFSocketCreateWithNative(kCFAllocatorDefault,
                                                     channel[0],
                                                     kCFSocketReadCallBack,
                                                     _callback,
                                                     NULL);
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
        NSDate* date = [NSDate distantFuture];
        while (true)
        {   NSEvent* event = [NSApp nextEventMatchingMask: NSAnyEventMask
                                                untilDate: date
                                                   inMode: NSDefaultRunLoopMode
                                                  dequeue: YES];
           if (!event) break; /* No windows open */
           if ([event type]==NSApplicationDefined)
           {   short subtype = [event subtype];
               if (subtype==STDIN_READY) break;
               if (subtype==SIGINT_CALLED)
               {   interrupted = true;
                   break;
               }
           }
           [NSApp sendEvent: event];
        }
        [pool release];

        if (py_sigint_handler) PyOS_setsig(SIGINT, py_sigint_handler);
        CFReadStreamUnscheduleFromRunLoop(stream,
                                          runloop,
                                          kCFRunLoopCommonModes);
        if (sigint_socket) CFSocketInvalidate(sigint_socket);
        if (error==0) close(channel[1]);
        if (interrupted) raise(SIGINT);
    }
    CFReadStreamClose(stream);
    return 1;
}

#ifndef COMPILING_FOR_10_5
static int _init_atsui(void)
{
    OSStatus status;

    status = ATSUCreateStyle(&style);
    if (status!=noErr)
    {
        PyErr_SetString(PyExc_RuntimeError, "ATSUCreateStyle failed");
        return 0;
    }

    status = ATSUCreateTextLayout(&layout);
    if (status!=noErr)
    {
        status = ATSUDisposeStyle(style);
        if (status!=noErr)
            PyErr_WarnEx(PyExc_RuntimeWarning, "ATSUDisposeStyle failed", 1);
        PyErr_SetString(PyExc_RuntimeError, "ATSUCreateTextLayout failed");
        return 0;
    }


    return 1;
}

static void _dealloc_atsui(void)
{
    OSStatus status;

    status = ATSUDisposeStyle(style);
    if (status!=noErr)
        PyErr_WarnEx(PyExc_RuntimeWarning, "ATSUDisposeStyle failed", 1);

    status = ATSUDisposeTextLayout(layout);
    if (status!=noErr)
        PyErr_WarnEx(PyExc_RuntimeWarning, "ATSUDisposeTextLayout failed", 1);
}
#endif

static int _draw_path(CGContextRef cr, void* iterator, int nmax)
{
    double x1, y1, x2, y2, x3, y3;
    static unsigned code = STOP;
    static double xs, ys;
    CGPoint current;
    int n = 0;

    if (code == MOVETO) CGContextMoveToPoint(cr, xs, ys);

    while (true)
    {
        code = get_vertex(iterator, &x1, &y1);
        if (code == CLOSEPOLY)
        {
            CGContextClosePath(cr);
            n++;
        }
        else if (code == STOP)
        {
            break;
        }
        else if (code == MOVETO)
        {
            CGContextMoveToPoint(cr, x1, y1);
        }
        else if (code==LINETO)
        {
            CGContextAddLineToPoint(cr, x1, y1);
            n++;
        }
        else if (code==CURVE3)
        {
            get_vertex(iterator, &xs, &ys);
            CGContextAddQuadCurveToPoint(cr, x1, y1, xs, ys);
            n+=2;
        }
        else if (code==CURVE4)
        {
            get_vertex(iterator, &x2, &y2);
            get_vertex(iterator, &xs, &ys);
            CGContextAddCurveToPoint(cr, x1, y1, x2, y2, xs, ys);
            n+=3;
        }
        if (n >= nmax)
        {
            switch (code)
            {
                case MOVETO:
                case LINETO:
                    xs = x1;
                    ys = y1;
                    break;
                case CLOSEPOLY:
                    current = CGContextGetPathCurrentPoint(cr);
                    xs = current.x;
                    ys = current.y;
                    break;
                /* nothing needed for CURVE3, CURVE4 */
            }
            code = MOVETO;
            return -n;
        }
    }
    return n;
}

static void _draw_hatch(void *info, CGContextRef cr)
{
    int n;
    PyObject* hatchpath = (PyObject*)info;
    PyObject* transform;
    int nd = 2;
    npy_intp dims[2] = {3, 3};
    int typenum = NPY_DOUBLE;
    double data[9] = {HATCH_SIZE, 0, 0, 0, HATCH_SIZE, 0, 0, 0, 1};
    double rect[4] = { 0.0, 0.0, HATCH_SIZE, HATCH_SIZE};
    transform = PyArray_SimpleNewFromData(nd, dims, typenum, data);
    if (!transform)
    {
        PyGILState_STATE gstate = PyGILState_Ensure();
        PyErr_Print();
        PyGILState_Release(gstate);
        return;
    }
    void* iterator  = get_path_iterator(hatchpath,
                                        transform,
                                        0,
                                        0,
                                        rect,
                                        SNAP_FALSE,
                                        1.0,
                                        0);
    Py_DECREF(transform);
    if (!iterator)
    {
        PyGILState_STATE gstate = PyGILState_Ensure();
        PyErr_SetString(PyExc_RuntimeError, "failed to obtain path iterator for hatching");
        PyErr_Print();
        PyGILState_Release(gstate);
        return;
    }
    n = _draw_path(cr, iterator, INT_MAX);
    free_path_iterator(iterator);
    if (n==0) return;
    CGContextSetLineWidth(cr, 1.0);
    CGContextSetLineCap(cr, kCGLineCapSquare);
    CGContextDrawPath(cr, kCGPathFillStroke);
}

static void _release_hatch(void* info)
{
    PyObject* hatchpath = (PyObject*)info;
    Py_DECREF(hatchpath);
}

/* ---------------------------- Cocoa classes ---------------------------- */


@interface Window : NSWindow
{   PyObject* manager;
}
- (Window*)initWithContentRect:(NSRect)rect styleMask:(unsigned int)mask backing:(NSBackingStoreType)bufferingType defer:(BOOL)deferCreation withManager: (PyObject*)theManager;
- (NSRect)constrainFrameRect:(NSRect)rect toScreen:(NSScreen*)screen;
- (BOOL)closeButtonPressed;
- (void)close;
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

typedef struct {
    PyObject_HEAD
    CGContextRef cr;
    NSSize size;
    int level;
    BOOL forced_alpha;
    CGFloat color[4];
    float dpi;
} GraphicsContext;

static CGMutablePathRef _create_path(void* iterator)
{
    unsigned code;
    CGMutablePathRef p;
    double x1, y1, x2, y2, x3, y3;

    p = CGPathCreateMutable();
    if (!p) return NULL;

    while (true)
    {
        code = get_vertex(iterator, &x1, &y1);
        if (code == CLOSEPOLY)
        {
            CGPathCloseSubpath(p);
        }
        else if (code == STOP)
        {
            break;
        }
        else if (code == MOVETO)
        {
            CGPathMoveToPoint(p, NULL, x1, y1);
        }
        else if (code==LINETO)
        {
            CGPathAddLineToPoint(p, NULL, x1, y1);
        }
        else if (code==CURVE3)
        {
            get_vertex(iterator, &x2, &y2);
            CGPathAddQuadCurveToPoint(p, NULL, x1, y1, x2, y2);
        }
        else if (code==CURVE4)
        {
            get_vertex(iterator, &x2, &y2);
            get_vertex(iterator, &x3, &y3);
            CGPathAddCurveToPoint(p, NULL, x1, y1, x2, y2, x3, y3);
        }
    }

    return p;
}

static int _get_snap(GraphicsContext* self, enum e_snap_mode* mode)
{
    PyObject* snap = PyObject_CallMethod((PyObject*)self, "get_snap", "");
    if(!snap) return 0;
    if(snap==Py_None) *mode = SNAP_AUTO;
    else if (PyBool_Check(snap)) *mode = SNAP_TRUE;
    else *mode = SNAP_FALSE;
    Py_DECREF(snap);
    return 1;
}

static PyObject*
GraphicsContext_new(PyTypeObject* type, PyObject *args, PyObject *kwds)
{
    GraphicsContext* self = (GraphicsContext*)type->tp_alloc(type, 0);
    if (!self) return NULL;
    self->cr = NULL;
    self->level = 0;
    self->forced_alpha = FALSE;

#ifndef COMPILING_FOR_10_5
    if (ngc==0)
    {
        int ok = _init_atsui();
        if (!ok)
        {
            return NULL;
        }
    }
    ngc++;
#endif

    return (PyObject*) self;
}

#ifndef COMPILING_FOR_10_5
static void
GraphicsContext_dealloc(GraphicsContext *self)
{
    ngc--;
    if (ngc==0) _dealloc_atsui();

    Py_TYPE(self)->tp_free((PyObject*)self);
}
#endif

static PyObject*
GraphicsContext_repr(GraphicsContext* self)
{
#if PY3K
    return PyUnicode_FromFormat("GraphicsContext object %p wrapping the Quartz 2D graphics context %p", (void*)self, (void*)(self->cr));
#else
    return PyString_FromFormat("GraphicsContext object %p wrapping the Quartz 2D graphics context %p", (void*)self, (void*)(self->cr));
#endif
}

static PyObject*
GraphicsContext_save (GraphicsContext* self)
{
    CGContextRef cr = self->cr;
    if (!cr)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGContextRef is NULL");
        return NULL;
    }
    CGContextSaveGState(cr);
    self->level++;
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
GraphicsContext_restore (GraphicsContext* self)
{
    CGContextRef cr = self->cr;
    if (!cr)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGContextRef is NULL");
        return NULL;
    }
    if (self->level==0)
    {
        PyErr_SetString(PyExc_RuntimeError,
            "Attempting to execute CGContextRestoreGState on an empty stack");
        return NULL;
    }
    CGContextRestoreGState(cr);
    self->level--;
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
GraphicsContext_set_alpha (GraphicsContext* self, PyObject* args)
{
    float alpha;
    int forced = 0;
    if (!PyArg_ParseTuple(args, "f|i", &alpha, &forced)) return NULL;
    CGContextRef cr = self->cr;
    if (!cr)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGContextRef is NULL");
        return NULL;
    }
    CGContextSetAlpha(cr, alpha);
    self->forced_alpha = (BOOL)(forced || (alpha != 1.0));

    Py_INCREF(Py_None);
    return Py_None;
}

static BOOL
_set_antialiased(CGContextRef cr, PyObject* antialiased)
{
    const int shouldAntialias = PyObject_IsTrue(antialiased);
    if (shouldAntialias < 0)
    {
        PyErr_SetString(PyExc_ValueError,
                        "Failed to read antialiaseds variable");
        return false;
    }
    CGContextSetShouldAntialias(cr, shouldAntialias);
    return true;
}

static PyObject*
GraphicsContext_set_antialiased (GraphicsContext* self, PyObject* args)
{
    CGContextRef cr = self->cr;
    if (!cr)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGContextRef is NULL");
        return NULL;
    }
    if (!_set_antialiased(cr, args)) return NULL;
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
GraphicsContext_set_capstyle (GraphicsContext* self, PyObject* args)
{
    char* string;
    CGLineCap cap;

    if (!PyArg_ParseTuple(args, "s", &string)) return NULL;

    if (!strcmp(string, "butt")) cap = kCGLineCapButt;
    else if (!strcmp(string, "round")) cap = kCGLineCapRound;
    else if (!strcmp(string, "projecting")) cap = kCGLineCapSquare;
    else
    {
        PyErr_SetString(PyExc_ValueError,
                        "capstyle should be 'butt', 'round', or 'projecting'");
        return NULL;
    }
    CGContextRef cr = self->cr;
    if (!cr)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGContextRef is NULL");
        return NULL;
    }
    CGContextSetLineCap(cr, cap);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
GraphicsContext_set_clip_rectangle (GraphicsContext* self, PyObject* args)
{
    CGRect rect;
    float x, y, width, height;
    if (!PyArg_ParseTuple(args, "(ffff)", &x, &y, &width, &height)) return NULL;

    rect.origin.x = x;
    rect.origin.y = y;
    rect.size.width = width;
    rect.size.height = height;

    CGContextRef cr = self->cr;
    if (!cr)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGContextRef is NULL");
        return NULL;
    }

    CGContextClipToRect(cr, rect);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
GraphicsContext_set_clip_path (GraphicsContext* self, PyObject* args)
{
    int n;
    CGContextRef cr = self->cr;

    PyObject* path;
    int nd = 2;
    npy_intp dims[2] = {3, 3};
    int typenum = NPY_DOUBLE;
    double data[] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

    if (!cr)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGContextRef is NULL");
        return NULL;
    }

    if(!PyArg_ParseTuple(args, "O", &path)) return NULL;

    PyObject* transform = PyArray_SimpleNewFromData(nd, dims, typenum, data);
    if (!transform) return NULL;

    double rect[4] = {0.0, 0.0, self->size.width, self->size.height};
    void* iterator  = get_path_iterator(path,
                                        transform,
                                        0,
                                        0,
                                        rect,
                                        SNAP_AUTO,
                                        1.0,
                                        0);
    Py_DECREF(transform);
    if (!iterator)
    {
        PyErr_SetString(PyExc_RuntimeError,
            "set_clip_path: failed to obtain path iterator for clipping");
        return NULL;
    }
    n = _draw_path(cr, iterator, INT_MAX);
    free_path_iterator(iterator);

    if (n > 0) CGContextClip(cr);

    Py_INCREF(Py_None);
    return Py_None;
}

static BOOL
_set_dashes(CGContextRef cr, PyObject* linestyle)
{
    CGFloat phase = 0.0;
    PyObject* offset;
    PyObject* dashes;

    if (!PyArg_ParseTuple(linestyle, "OO", &offset, &dashes))
    {
        PyErr_SetString(PyExc_TypeError,
            "failed to obtain the offset and dashes from the linestyle");
        return false;
    }

    if (offset!=Py_None)
    {
        if (PyFloat_Check(offset)) phase = PyFloat_AS_DOUBLE(offset);
#if PY3K
        else if (PyLong_Check(offset)) phase = PyLong_AS_LONG(offset);
#else
        else if (PyInt_Check(offset)) phase = PyInt_AS_LONG(offset);
#endif
        else
        {
            PyErr_SetString(PyExc_TypeError,
                            "offset should be a floating point value");
            return false;
        }
    }

    if (dashes!=Py_None)
    {
        if (PyList_Check(dashes)) dashes = PyList_AsTuple(dashes);
        else if (PyTuple_Check(dashes)) Py_INCREF(dashes);
        else
        {
            PyErr_SetString(PyExc_TypeError,
                            "dashes should be a tuple or a list");
            return false;
        }
        int n = PyTuple_GET_SIZE(dashes);
        int i;
        CGFloat* lengths = malloc(n*sizeof(CGFloat));
        if(!lengths)
        {
            PyErr_SetString(PyExc_MemoryError, "Failed to store dashes");
            Py_DECREF(dashes);
            return false;
        }
        for (i = 0; i < n; i++)
        {
            PyObject* value = PyTuple_GET_ITEM(dashes, i);
            if (PyFloat_Check(value))
                lengths[i] = (CGFloat) PyFloat_AS_DOUBLE(value);
#if PY3K
            else if (PyLong_Check(value))
                lengths[i] = (CGFloat) PyLong_AS_LONG(value);
#else
            else if (PyInt_Check(value))
                lengths[i] = (CGFloat) PyInt_AS_LONG(value);
#endif
            else break;
        }
        Py_DECREF(dashes);
        if (i < n) /* break encountered */
        {
            free(lengths);
            PyErr_SetString(PyExc_TypeError, "Failed to read dashes");
            return false;
        }
        CGContextSetLineDash(cr, phase, lengths, n);
        free(lengths);
    }
    else
        CGContextSetLineDash(cr, phase, NULL, 0);

    return true;
}

static PyObject*
GraphicsContext_set_dashes (GraphicsContext* self, PyObject* args)
{
    CGContextRef cr = self->cr;
    if (!cr)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGContextRef is NULL");
        return NULL;
    }

    if (!_set_dashes(cr, args))
        return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
GraphicsContext_set_foreground(GraphicsContext* self, PyObject* args)
{
    float r, g, b, a;
    if(!PyArg_ParseTuple(args, "(ffff)", &r, &g, &b, &a)) return NULL;

    CGContextRef cr = self->cr;
    if (!cr)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGContextRef is NULL");
        return NULL;
    }

    if (self->forced_alpha)
    {
        // Transparency is applied to layer
        // Let it override (rather than multiply with) the alpha of the
        // stroke/fill colors
        a = 1.0;
    }

    CGContextSetRGBStrokeColor(cr, r, g, b, a);
    CGContextSetRGBFillColor(cr, r, g, b, a);

    self->color[0] = r;
    self->color[1] = g;
    self->color[2] = b;
    self->color[3] = a;

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
GraphicsContext_set_graylevel(GraphicsContext* self, PyObject* args)
{   float gray;
    if(!PyArg_ParseTuple(args, "f", &gray)) return NULL;

    CGContextRef cr = self->cr;
    if (!cr)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGContextRef is NULL");
        return NULL;
    }

    CGContextSetGrayStrokeColor(cr, gray, 1.0);
    CGContextSetGrayFillColor(cr, gray, 1.0);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
GraphicsContext_set_dpi (GraphicsContext* self, PyObject* args)
{
  if (!PyArg_ParseTuple(args, "f", &(self->dpi))) return NULL;

  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject*
GraphicsContext_set_linewidth (GraphicsContext* self, PyObject* args)
{
    float width;
    if (!PyArg_ParseTuple(args, "f", &width)) return NULL;

    CGContextRef cr = self->cr;
    if (!cr)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGContextRef is NULL");
        return NULL;
    }

    // Convert points to pixels
    width *= self->dpi / 72.0;
    CGContextSetLineWidth(cr, width);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
GraphicsContext_set_joinstyle(GraphicsContext* self, PyObject* args)
{   char* string;
    CGLineJoin join;

    if (!PyArg_ParseTuple(args, "s", &string)) return NULL;

    if (!strcmp(string, "miter")) join = kCGLineJoinMiter;
    else if (!strcmp(string, "round")) join = kCGLineJoinRound;
    else if (!strcmp(string, "bevel")) join = kCGLineJoinBevel;
    else
    {
        PyErr_SetString(PyExc_ValueError,
                        "joinstyle should be 'miter', 'round', or 'bevel'");
        return NULL;
    }

    CGContextRef cr = self->cr;
    if (!cr)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGContextRef is NULL");
        return NULL;
    }
    CGContextSetLineJoin(cr, join);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
GraphicsContext_draw_path (GraphicsContext* self, PyObject* args)
{
    PyObject* path;
    PyObject* transform;
    PyObject* rgbFace;
    float linewidth;

    int n;

    void* iterator;

    CGContextRef cr = self->cr;
    double rect[4] = { 0.0, 0.0, self->size.width, self->size.height};

    if (!cr)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGContextRef is NULL");
        return NULL;
    }

    if(!PyArg_ParseTuple(args, "OOf|O",
                               &path,
                               &transform,
                               &linewidth,
                               &rgbFace)) return NULL;

    if(rgbFace==Py_None) rgbFace = NULL;

    iterator  = get_path_iterator(path,
                                  transform,
                                  1,
                                  0,
                                  rect,
                                  SNAP_AUTO,
                                  linewidth,
                                  rgbFace == NULL);
    if (!iterator)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "draw_path: failed to obtain path iterator");
        return NULL;
    }

    if(rgbFace)
    {
        float r, g, b, a;
        a = 1.0;
        if (!PyArg_ParseTuple(rgbFace, "fff|f", &r, &g, &b, &a))
            return NULL;
        if (self->forced_alpha)
            a = 1.0;

        n = _draw_path(cr, iterator, INT_MAX);
        if (n > 0)
        {
            CGContextSaveGState(cr);
            CGContextSetRGBFillColor(cr, r, g, b, a);
            CGContextDrawPath(cr, kCGPathFillStroke);
            CGContextRestoreGState(cr);
        }
    }
    else
    {
        const int nmax = 100;
        while (true)
        {
            n = _draw_path(cr, iterator, nmax);
            if (n != 0) CGContextStrokePath(cr);
            if (n >= 0) break;
        }
    }
    free_path_iterator(iterator);

    PyObject* hatchpath;
    hatchpath = PyObject_CallMethod((PyObject*)self, "get_hatch_path", "");
    if (!hatchpath)
    {
        return NULL;
    }
    else if (hatchpath==Py_None)
    {
        Py_DECREF(hatchpath);
    }
    else
    {
        CGPatternRef pattern;
        CGColorSpaceRef baseSpace;
        CGColorSpaceRef patternSpace;
        static const CGPatternCallbacks callbacks = {0,
                                                     &_draw_hatch,
                                                     &_release_hatch};
        baseSpace = CGColorSpaceCreateWithName(kCGColorSpaceGenericRGB);
        if (!baseSpace)
        {
            Py_DECREF(hatchpath);
            PyErr_SetString(PyExc_RuntimeError,
                "draw_path: CGColorSpaceCreateWithName failed");
            return NULL;
        }
        patternSpace = CGColorSpaceCreatePattern(baseSpace);
        CGColorSpaceRelease(baseSpace);
        if (!patternSpace)
        {
            Py_DECREF(hatchpath);
            PyErr_SetString(PyExc_RuntimeError,
                "draw_path: CGColorSpaceCreatePattern failed");
            return NULL;
        }
        CGContextSetFillColorSpace(cr, patternSpace);
        CGColorSpaceRelease(patternSpace);

        pattern = CGPatternCreate((void*)hatchpath,
                                  CGRectMake(0, 0, HATCH_SIZE, HATCH_SIZE),
                                  CGAffineTransformIdentity,
                                  HATCH_SIZE, HATCH_SIZE,
                                  kCGPatternTilingNoDistortion,
                                  false,
                                  &callbacks);
        CGContextSetFillPattern(cr, pattern, self->color);
        CGPatternRelease(pattern);
        iterator  = get_path_iterator(path,
                                      transform,
                                      1,
                                      0,
                                      rect,
                                      SNAP_AUTO,
                                      linewidth,
                                      0);
        if (!iterator)
        {
            Py_DECREF(hatchpath);
            PyErr_SetString(PyExc_RuntimeError,
                "draw_path: failed to obtain path iterator for hatching");
            return NULL;
        }
        n = _draw_path(cr, iterator, INT_MAX);
        free_path_iterator(iterator);
        if (n > 0) CGContextFillPath(cr);
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
GraphicsContext_draw_markers (GraphicsContext* self, PyObject* args)
{
    PyObject* marker_path;
    PyObject* marker_transform;
    PyObject* path;
    PyObject* transform;
    float linewidth;
    PyObject* rgbFace;

    int ok;
    float r, g, b, a;

    CGMutablePathRef marker;
    void* iterator;
    double rect[4] = {0.0, 0.0, self->size.width, self->size.height};
    enum e_snap_mode mode;
    double xc, yc;
    unsigned code;

    CGContextRef cr = self->cr;

    if (!cr)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGContextRef is NULL");
        return NULL;
    }

    if(!PyArg_ParseTuple(args, "OOOOf|O",
                               &marker_path,
                               &marker_transform,
                               &path,
                               &transform,
                               &linewidth,
                               &rgbFace)) return NULL;

    if(rgbFace==Py_None) rgbFace = NULL;

    if (rgbFace)
    {
        a = 1.0;
        ok = PyArg_ParseTuple(rgbFace, "fff|f", &r, &g, &b, &a);
        if (!ok)
        {
            return NULL;
        }
        if (self->forced_alpha)
            a = 1.0;
        CGContextSetRGBFillColor(cr, r, g, b, a);
    }

    ok = _get_snap(self, &mode);
    if (!ok)
    {
        return NULL;
    }

    iterator = get_path_iterator(marker_path,
                                 marker_transform,
                                 0,
                                 0,
                                 rect,
                                 mode,
                                 linewidth,
                                 0);
    if (!iterator)
    {
        PyErr_SetString(PyExc_RuntimeError,
            "draw_markers: failed to obtain path iterator for marker");
        return NULL;
    }
    marker = _create_path(iterator);
    free_path_iterator(iterator);
    if (!marker)
    {
        PyErr_SetString(PyExc_RuntimeError,
            "draw_markers: failed to draw marker path");
        return NULL;
    }
    iterator = get_path_iterator(path,
                                 transform,
                                 1,
                                 1,
                                 rect,
                                 SNAP_TRUE,
                                 1.0,
                                 0);
    if (!iterator)
    {
        CGPathRelease(marker);
        PyErr_SetString(PyExc_RuntimeError,
            "draw_markers: failed to obtain path iterator");
        return NULL;
    }

    while (true)
    {
        code = get_vertex(iterator, &xc, &yc);
        if (code == STOP)
        {
            break;
        }
        else if (code == MOVETO || code == LINETO || code == CURVE3 || code ==CURVE4)
        {
            CGContextSaveGState(cr);
            CGContextTranslateCTM(cr, xc, yc);
            CGContextAddPath(cr, marker);
            CGContextRestoreGState(cr);
        }
        if(rgbFace) CGContextDrawPath(cr, kCGPathFillStroke);
        else CGContextStrokePath(cr);
    }
    free_path_iterator(iterator);
    CGPathRelease(marker);

    Py_INCREF(Py_None);
    return Py_None;
}

static int _transformation_converter(PyObject* object, void* pointer)
{
    CGAffineTransform* matrix = (CGAffineTransform*)pointer;
    if (!PyArray_Check(object) || PyArray_NDIM(object)!=2
        || PyArray_DIM(object, 0)!=3 || PyArray_DIM(object, 1)!=3)
        {
            PyErr_SetString(PyExc_ValueError,
                "transformation matrix is not a 3x3 NumPy array");
            return 0;
        }
    const double a =  *(double*)PyArray_GETPTR2(object, 0, 0);
    const double b =  *(double*)PyArray_GETPTR2(object, 1, 0);
    const double c =  *(double*)PyArray_GETPTR2(object, 0, 1);
    const double d =  *(double*)PyArray_GETPTR2(object, 1, 1);
    const double tx =  *(double*)PyArray_GETPTR2(object, 0, 2);
    const double ty =  *(double*)PyArray_GETPTR2(object, 1, 2);
    *matrix = CGAffineTransformMake(a, b, c, d, tx, ty);
    return 1;
}

static PyObject*
GraphicsContext_draw_path_collection (GraphicsContext* self, PyObject* args)
{
    CGAffineTransform master;
    PyObject* path_ids;
    PyObject* all_transforms;
    PyObject* offsets;
    CGAffineTransform offset_transform;
    PyObject* facecolors;
    PyObject* edgecolors;
    PyObject* linewidths;
    PyObject* linestyles;
    PyObject* antialiaseds;

    int offset_position = 0;

    CGContextRef cr = self->cr;

    if (!cr)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGContextRef is NULL");
        return NULL;
    }

    if(!PyArg_ParseTuple(args, "O&OOOO&OOOOOi",
                               _transformation_converter, &master,
                               &path_ids,
                               &all_transforms,
                               &offsets,
                               _transformation_converter, &offset_transform,
                               &facecolors,
                               &edgecolors,
                               &linewidths,
                               &linestyles,
                               &antialiaseds,
                               &offset_position))
        return NULL;

    int ok = 1;
    Py_ssize_t i;

    CGMutablePathRef* p = NULL;
    CGAffineTransform* transforms = NULL;
    CGPoint *toffsets = NULL;
    CGPatternRef pattern = NULL;
    CGColorSpaceRef patternSpace = NULL;

    PyObject* hatchpath;
    hatchpath = PyObject_CallMethod((PyObject*)self, "get_hatch_path", "");
    if (!hatchpath)
    {
        return NULL;
    }
    else if (hatchpath==Py_None)
    {
        Py_DECREF(hatchpath);
    }
    else
    {
        CGColorSpaceRef baseSpace;
        static const CGPatternCallbacks callbacks = {0,
                                                     &_draw_hatch,
                                                     &_release_hatch};
        baseSpace = CGColorSpaceCreateWithName(kCGColorSpaceGenericRGB);
        if (!baseSpace)
        {
            Py_DECREF(hatchpath);
            PyErr_SetString(PyExc_RuntimeError,
                "draw_path: CGColorSpaceCreateWithName failed");
            return NULL;
        }
        patternSpace = CGColorSpaceCreatePattern(baseSpace);
        CGColorSpaceRelease(baseSpace);
        if (!patternSpace)
        {
            Py_DECREF(hatchpath);
            PyErr_SetString(PyExc_RuntimeError,
                "draw_path: CGColorSpaceCreatePattern failed");
            return NULL;
        }
        pattern = CGPatternCreate((void*)hatchpath,
                                  CGRectMake(0, 0, HATCH_SIZE, HATCH_SIZE),
                                  CGAffineTransformIdentity,
                                  HATCH_SIZE, HATCH_SIZE,
                                  kCGPatternTilingNoDistortion,
                                  false,
                                  &callbacks);
    }

    /* --------- Prepare some variables for the path iterator ------------- */
    void* iterator;
    double rect[4] = {0.0, 0.0, self->size.width, self->size.height};
    enum e_snap_mode mode;
    ok = _get_snap(self, &mode);
    if (!ok)
    {
        return NULL;
    }

    /* ------------------- Check paths ------------------------------------ */

    if (!PySequence_Check(path_ids))
    {
        PyErr_SetString(PyExc_ValueError, "paths must be a sequence object");
        return NULL;
    }
    const Py_ssize_t Npaths = PySequence_Size(path_ids);
    
    /* -------------------------------------------------------------------- */

    CGContextSaveGState(cr);
    
    /* ------------------- Check facecolors array ------------------------- */

    facecolors = PyArray_FromObject(facecolors, NPY_DOUBLE, 1, 2);
    if (!facecolors ||
        (PyArray_NDIM(facecolors)==1 && PyArray_DIM(facecolors, 0)!=0) ||
        (PyArray_NDIM(facecolors)==2 && PyArray_DIM(facecolors, 1)!=4))
    {
        PyErr_SetString(PyExc_ValueError, "Facecolors must by a Nx4 numpy array or empty");
        ok = 0;
        goto exit;
    }
    Py_ssize_t Nfacecolors = PyArray_DIM(facecolors, 0);

    /* ------------------- Check edgecolors array ------------------------- */

    edgecolors = PyArray_FromObject(edgecolors, NPY_DOUBLE, 1, 2);
    if (!edgecolors ||
        (PyArray_NDIM(edgecolors)==1 && PyArray_DIM(edgecolors, 0)!=0) ||
        (PyArray_NDIM(edgecolors)==2 && PyArray_DIM(edgecolors, 1)!=4))
    {
        PyErr_SetString(PyExc_ValueError, "Edgecolors must by a Nx4 numpy array or empty");
        ok = 0;
        goto exit;
    }
    Py_ssize_t Nedgecolors = PyArray_DIM(edgecolors, 0);

    /* -------------------------------------------------------------------- */

    if ((Nfacecolors==0 && Nedgecolors==0) || Npaths==0) /* Nothing to do */
        goto exit;

    /* ------------------- Check offsets array ---------------------------- */

    offsets = PyArray_FromObject(offsets, NPY_DOUBLE, 0, 2);

    if (!offsets ||
        (PyArray_NDIM(offsets)==2 && PyArray_DIM(offsets, 1)!=2) ||
        (PyArray_NDIM(offsets)==1 && PyArray_DIM(offsets, 0)!=0))
    {
        Py_XDECREF(offsets);
        PyErr_SetString(PyExc_ValueError, "Offsets array must be Nx2");
        ok = 0;
        goto exit;
    }
    const Py_ssize_t Noffsets = PyArray_DIM(offsets, 0);
    if (Noffsets > 0) {
        toffsets = malloc(Noffsets*sizeof(CGPoint));
        if (!toffsets)
        {
            Py_DECREF(offsets);
            ok = 0;
            goto exit;
        }
        CGPoint point;
        for (i = 0; i < Noffsets; i++)
        {
            point.x = (CGFloat) (*(double*)PyArray_GETPTR2(offsets, i, 0));
            point.y = (CGFloat) (*(double*)PyArray_GETPTR2(offsets, i, 1));
            toffsets[i] = CGPointApplyAffineTransform(point, offset_transform);
        }
    }
    Py_DECREF(offsets);

    /* ------------------- Check transforms ------------------------------- */

    if (!PySequence_Check(all_transforms))
    {
        PyErr_SetString(PyExc_ValueError, "transforms must be a sequence object");
        return NULL;
    }
    const Py_ssize_t Ntransforms = PySequence_Size(all_transforms);
    if (Ntransforms > 0)
    {
        transforms = malloc(Ntransforms*sizeof(CGAffineTransform));
        if (!transforms)
            goto exit;
        for (i = 0; i < Ntransforms; i++)
        {
            PyObject* transform = PySequence_ITEM(all_transforms, i);
            if (!transform) goto exit;
            ok = _transformation_converter(transform, &transforms[i]);
            Py_DECREF(transform);
            if (!ok) goto exit;
        }
    }

    /* -------------------------------------------------------------------- */

    p = malloc(Npaths*sizeof(CGMutablePathRef));
    if (!p)
    {
        ok = 0;
        goto exit;
    }
    for (i = 0; i < Npaths; i++)
    {
        PyObject* path;
        PyObject* transform;
        p[i] = NULL;
        PyObject* path_id = PySequence_ITEM(path_ids, i);
        if (!path_id)
        {
            ok = 0;
            goto exit;
        }
        if (!PyTuple_Check(path_id) || PyTuple_Size(path_id)!=2)
        {
            ok = 0;
            PyErr_SetString(PyExc_RuntimeError,
                            "path_id should be a tuple of two items");
            goto exit;
        }
        path = PyTuple_GET_ITEM(path_id, 0);
        transform = PyTuple_GET_ITEM(path_id, 1);
        iterator = get_path_iterator(path,
                                     transform,
                                     1,
                                     0,
                                     rect,
                                     mode,
                                     1.0,
                                     /* Hardcoding stroke width to 1.0
                                        here, but for true
                                        correctness, the paths would
                                        need to be set up for each
                                        different linewidth that may
                                        be applied below.  This
                                        difference is very minute in
                                        practice, so this hardcoding
                                        is probably ok for now.  --
                                        MGD */
                                     0);
        if (!iterator)
        {
            PyErr_SetString(PyExc_RuntimeError,
                            "failed to obtain path iterator");
            ok = 0;
            goto exit;
        }
        p[i] = _create_path(iterator);
        free_path_iterator(iterator);
        Py_DECREF(path_id);
        if (!p[i])
        {
            PyErr_SetString(PyExc_RuntimeError, "failed to create path");
            ok = 0;
            goto exit;
        }
    }

    /* ------------------- Check the other arguments ---------------------- */

    if (!PySequence_Check(linewidths))
    {
        PyErr_SetString(PyExc_ValueError, "linewidths must be a sequence object");
        ok = 0;
        goto exit;
    }
    if (!PySequence_Check(linestyles))
    {
        PyErr_SetString(PyExc_ValueError, "linestyles must be a sequence object");
        ok = 0;
        goto exit;
    }
    if (!PySequence_Check(antialiaseds))
    {
        PyErr_SetString(PyExc_ValueError, "antialiaseds must be a sequence object");
        ok = 0;
        goto exit;
    }

    Py_ssize_t Nlinewidths = PySequence_Size(linewidths);
    Py_ssize_t Nlinestyles = PySequence_Size(linestyles);
    Py_ssize_t Naa         = PySequence_Size(antialiaseds);

    /* Preset graphics context properties if possible */
    if (Naa==1)
    {
        PyObject* antialiased = PySequence_ITEM(antialiaseds, 0);
        if (antialiased)
        {
            ok = _set_antialiased(cr, antialiased);
            Py_DECREF(antialiased);
        }
        else
        {
            PyErr_SetString(PyExc_SystemError,
                            "Failed to read element from antialiaseds array");
            ok = 0;
        }
        if (!ok) goto exit;
    }

    if (Nlinewidths==0 || Nedgecolors==0)
        CGContextSetLineWidth(cr, 0.0);
    else if (Nlinewidths==1)
    {
        PyObject* linewidth = PySequence_ITEM(linewidths, 0);
        if (!linewidth)
        {
            PyErr_SetString(PyExc_SystemError,
                            "Failed to read element from linewidths array");
            ok = 0;
            goto exit;
        }
        CGContextSetLineWidth(cr, (CGFloat)PyFloat_AsDouble(linewidth));
        Py_DECREF(linewidth);
    }

    if (Nlinestyles==1)
    {
        PyObject* linestyle = PySequence_ITEM(linestyles, 0);
        if (!linestyle)
        {
            PyErr_SetString(PyExc_SystemError,
                            "Failed to read element from linestyles array");
            ok = 0;
            goto exit;
        }
        ok = _set_dashes(cr, linestyle);
        Py_DECREF(linestyle);
        if (!ok) goto exit;
    }

    if (Nedgecolors==1)
    {
        const double r = *(double*)PyArray_GETPTR2(edgecolors, 0, 0);
        const double g = *(double*)PyArray_GETPTR2(edgecolors, 0, 1);
        const double b = *(double*)PyArray_GETPTR2(edgecolors, 0, 2);
        const double a = *(double*)PyArray_GETPTR2(edgecolors, 0, 3);
        CGContextSetRGBStrokeColor(cr, r, g, b, a);
        self->color[0] = r;
        self->color[1] = g;
        self->color[2] = b;
        self->color[3] = a;
    }
    else /* We may need these for hatching */
    {
        self->color[0] = 0;
        self->color[1] = 0;
        self->color[2] = 0;
        self->color[3] = 1;
    }

    if (Nfacecolors==1)
    {
        const double r = *(double*)PyArray_GETPTR2(facecolors, 0, 0);
        const double g = *(double*)PyArray_GETPTR2(facecolors, 0, 1);
        const double b = *(double*)PyArray_GETPTR2(facecolors, 0, 2);
        const double a = *(double*)PyArray_GETPTR2(facecolors, 0, 3);
        CGContextSetRGBFillColor(cr, r, g, b, a);
    }

    CGPoint translation = CGPointZero;

    const Py_ssize_t N = Npaths > Noffsets ? Npaths : Noffsets;
    for (i = 0; i < N; i++)
    {
        if (CGPathIsEmpty(p[i % Npaths])) continue;

        if (Noffsets)
        {
            CGAffineTransform t;
            CGPoint origin;
            translation = toffsets[i % Noffsets];
            if (offset_position)
            {
                t = master;
                if (Ntransforms)
                    t = CGAffineTransformConcat(transforms[i % Ntransforms], t);
                translation = CGPointApplyAffineTransform(translation, t);
                origin = CGPointApplyAffineTransform(CGPointZero, t);
                translation.x = - (origin.x - translation.x);
                translation.y = - (origin.y - translation.y);
            }
            CGContextTranslateCTM(cr, translation.x, translation.y);
        }

        if (Naa > 1)
        {
            PyObject* antialiased = PySequence_ITEM(antialiaseds, i % Naa);
            if (antialiased)
            {
                ok = _set_antialiased(cr, antialiased);
                Py_DECREF(antialiased);
            }
            else
            {
                PyErr_SetString(PyExc_SystemError,
                    "Failed to read element from antialiaseds array");
                ok = 0;
            }
            if (!ok) goto exit;
        }

        if (Nlinewidths > 1)
        {
            PyObject* linewidth = PySequence_ITEM(linewidths, i % Nlinewidths);
            if (!linewidth)
            {
                PyErr_SetString(PyExc_SystemError,
                                "Failed to read element from linewidths array");
                ok = 0;
                goto exit;
            }
            CGContextSetLineWidth(cr, (CGFloat)PyFloat_AsDouble(linewidth));
            Py_DECREF(linewidth);
        }

        if (Nlinestyles > 1)
        {
            PyObject* linestyle = PySequence_ITEM(linestyles, i % Nlinestyles);
            if (!linestyle)
            {
                PyErr_SetString(PyExc_SystemError,
                                "Failed to read element from linestyles array");
                ok = 0;
                goto exit;
            }
            ok = _set_dashes(cr, linestyle);
            Py_DECREF(linestyle);
            if (!ok) goto exit;
        }

        if (Nedgecolors > 1)
        {
            npy_intp fi = i % Nedgecolors;
            const double r = *(double*)PyArray_GETPTR2(edgecolors, fi, 0);
            const double g = *(double*)PyArray_GETPTR2(edgecolors, fi, 1);
            const double b = *(double*)PyArray_GETPTR2(edgecolors, fi, 2);
            const double a = *(double*)PyArray_GETPTR2(edgecolors, fi, 3);
            CGContextSetRGBStrokeColor(cr, r, g, b, a);
        }

        CGContextAddPath(cr, p[i % Npaths]);

        if (Nfacecolors > 1)
        {
            npy_intp fi = i % Nfacecolors;
            const double r = *(double*)PyArray_GETPTR2(facecolors, fi, 0);
            const double g = *(double*)PyArray_GETPTR2(facecolors, fi, 1);
            const double b = *(double*)PyArray_GETPTR2(facecolors, fi, 2);
            const double a = *(double*)PyArray_GETPTR2(facecolors, fi, 3);
            CGContextSetRGBFillColor(cr, r, g, b, a);
            if (Nedgecolors > 0) CGContextDrawPath(cr, kCGPathFillStroke);
            else CGContextFillPath(cr);
        }
        else if (Nfacecolors==1)
        {
            if (Nedgecolors > 0) CGContextDrawPath(cr, kCGPathFillStroke);
            else CGContextFillPath(cr);
        }
        else /* We checked Nedgecolors != 0 above */
            CGContextStrokePath(cr);

        if (pattern)
        {
            CGContextSaveGState(cr);
            CGContextSetFillColorSpace(cr, patternSpace);
            CGContextSetFillPattern(cr, pattern, self->color);
            CGContextAddPath(cr, p[i % Npaths]);
            CGContextFillPath(cr);
            CGContextRestoreGState(cr);
        }

        if (Noffsets)
            CGContextTranslateCTM(cr, -translation.x, -translation.y);
    }

exit:
    CGContextRestoreGState(cr);
    Py_XDECREF(facecolors);
    Py_XDECREF(edgecolors);
    if (pattern) CGPatternRelease(pattern);
    if (patternSpace) CGColorSpaceRelease(patternSpace);
    if (transforms) free(transforms);
    if (toffsets) free(toffsets);
    if (p)
    {
        for (i = 0; i < Npaths; i++)
        {
            if (!p[i]) break;
            CGPathRelease(p[i]);
        }
        free(p);
    }
    if (!ok) return NULL;
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
GraphicsContext_draw_quad_mesh (GraphicsContext* self, PyObject* args)
{
    CGAffineTransform master;
    int meshWidth;
    int meshHeight;
    PyObject* coordinates;
    PyObject* offsets;
    CGAffineTransform offset_transform;
    PyObject* facecolors;
    int antialiased;
    PyObject* edgecolors;

    CGPoint *toffsets = NULL;

    CGContextRef cr = self->cr;

    if (!cr)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGContextRef is NULL");
        return NULL;
    }

    if(!PyArg_ParseTuple(args, "O&iiOOO&OiO",
                               _transformation_converter, &master,
                               &meshWidth,
                               &meshHeight,
                               &coordinates,
                               &offsets,
                               _transformation_converter, &offset_transform,
                               &facecolors,
                               &antialiased,
                               &edgecolors)) return NULL;

    int ok = 1;
    CGContextSaveGState(cr);

    /* ------------------- Check coordinates array ------------------------ */

    coordinates = PyArray_FromObject(coordinates, NPY_DOUBLE, 3, 3);
    if (!coordinates ||
        PyArray_NDIM(coordinates) != 3 || PyArray_DIM(coordinates, 2) != 2)
    {
        PyErr_SetString(PyExc_ValueError, "Invalid coordinates array");
        ok = 0;
        goto exit;
    }

    /* ------------------- Check offsets array ---------------------------- */

    offsets = PyArray_FromObject(offsets, NPY_DOUBLE, 0, 2);

    if (!offsets ||
        (PyArray_NDIM(offsets)==2 && PyArray_DIM(offsets, 1)!=2) ||
        (PyArray_NDIM(offsets)==1 && PyArray_DIM(offsets, 0)!=0))
    {
        Py_XDECREF(offsets);
        PyErr_SetString(PyExc_ValueError, "Offsets array must be Nx2");
        ok = 0;
        goto exit;
    }
    const Py_ssize_t Noffsets = PyArray_DIM(offsets, 0);
    if (Noffsets > 0) {
        int i;
        toffsets = malloc(Noffsets*sizeof(CGPoint));
        if (!toffsets)
        {
            Py_DECREF(offsets);
            ok = 0;
            goto exit;
        }
        CGPoint point;
        for (i = 0; i < Noffsets; i++)
        {
            point.x = (CGFloat) (*(double*)PyArray_GETPTR2(offsets, i, 0));
            point.y = (CGFloat) (*(double*)PyArray_GETPTR2(offsets, i, 1));
            toffsets[i] = CGPointApplyAffineTransform(point, offset_transform);
        }
    }
    Py_DECREF(offsets);

    /* ------------------- Check facecolors array ------------------------- */

    facecolors = PyArray_FromObject(facecolors, NPY_DOUBLE, 1, 2);
    if (!facecolors ||
        (PyArray_NDIM(facecolors)==1 && PyArray_DIM(facecolors, 0)!=0) ||
        (PyArray_NDIM(facecolors)==2 && PyArray_DIM(facecolors, 1)!=4))
    {
        PyErr_SetString(PyExc_ValueError, "facecolors must by a Nx4 numpy array or empty");
        ok = 0;
        goto exit;
    }

    /* ------------------- Check edgecolors array ------------------------- */

    edgecolors = PyArray_FromObject(edgecolors, NPY_DOUBLE, 1, 2);
    if (!edgecolors ||
        (PyArray_NDIM(edgecolors)==1 && PyArray_DIM(edgecolors, 0)!=0) ||
        (PyArray_NDIM(edgecolors)==2 && PyArray_DIM(edgecolors, 1)!=4))
    {
        PyErr_SetString(PyExc_ValueError, "edgecolors must by a Nx4 numpy array or empty");
        ok = 0;
        goto exit;
    }

    /* ------------------- Check the other arguments ---------------------- */

    size_t Npaths      = meshWidth * meshHeight;
    size_t Nfacecolors = (size_t) PyArray_DIM(facecolors, 0);
    size_t Nedgecolors = (size_t) PyArray_DIM(edgecolors, 0);
    if ((Nfacecolors == 0 && Nedgecolors == 0) || Npaths == 0)
    {
        /* Nothing to do here */
        goto exit;
    }

    size_t i = 0;
    size_t iw = 0;
    size_t ih = 0;

    /* Preset graphics context properties if possible */
    CGContextSetShouldAntialias(cr, antialiased);

    if (Nfacecolors==1)
    {
        const double r = *(double*)PyArray_GETPTR2(facecolors, 0, 0);
        const double g = *(double*)PyArray_GETPTR2(facecolors, 0, 1);
        const double b = *(double*)PyArray_GETPTR2(facecolors, 0, 2);
        const double a = *(double*)PyArray_GETPTR2(facecolors, 0, 3);
        CGContextSetRGBFillColor(cr, r, g, b, a);
        if (antialiased && Nedgecolors==0)
        {
            CGContextSetRGBStrokeColor(cr, r, g, b, a);
        }
    }
    if (Nedgecolors==1)
    {
        const double r = *(double*)PyArray_GETPTR2(edgecolors, 0, 0);
        const double g = *(double*)PyArray_GETPTR2(edgecolors, 0, 1);
        const double b = *(double*)PyArray_GETPTR2(edgecolors, 0, 2);
        const double a = *(double*)PyArray_GETPTR2(edgecolors, 0, 3);
        CGContextSetRGBStrokeColor(cr, r, g, b, a);
    }

    CGPoint translation = CGPointZero;
    double x, y;
    for (ih = 0; ih < meshHeight; ih++)
    {
        for (iw = 0; iw < meshWidth; iw++, i++)
        {
            CGPoint points[4];

            x = *(double*)PyArray_GETPTR3(coordinates, ih, iw, 0);
            y = *(double*)PyArray_GETPTR3(coordinates, ih, iw, 1);
            if (isnan(x) || isnan(y)) continue;
            points[0].x = (CGFloat)x;
            points[0].y = (CGFloat)y;

            x = *(double*)PyArray_GETPTR3(coordinates, ih, iw+1, 0);
            y = *(double*)PyArray_GETPTR3(coordinates, ih, iw+1, 1);
            if (isnan(x) || isnan(y)) continue;
            points[1].x = (CGFloat)x;
            points[1].y = (CGFloat)y;

            x = *(double*)PyArray_GETPTR3(coordinates, ih+1, iw+1, 0);
            y = *(double*)PyArray_GETPTR3(coordinates, ih+1, iw+1, 1);
            if (isnan(x) || isnan(y)) continue;
            points[2].x = (CGFloat)x;
            points[2].y = (CGFloat)y;

            x = *(double*)PyArray_GETPTR3(coordinates, ih+1, iw, 0);
            y = *(double*)PyArray_GETPTR3(coordinates, ih+1, iw, 1);
            if (isnan(x) || isnan(y)) continue;
            points[3].x = (CGFloat)x;
            points[3].y = (CGFloat)y;

            points[0] = CGPointApplyAffineTransform(points[0], master);
            points[1] = CGPointApplyAffineTransform(points[1], master);
            points[2] = CGPointApplyAffineTransform(points[2], master);
            points[3] = CGPointApplyAffineTransform(points[3], master);

            if (Noffsets)
            {
                translation = toffsets[i % Noffsets];
                CGContextTranslateCTM(cr, translation.x, translation.y);
            }

            CGContextMoveToPoint(cr, points[3].x, points[3].y);
            CGContextAddLines(cr, points, 4);
            CGContextClosePath(cr);

            if (Nfacecolors > 1)
            {
                npy_intp fi = i % Nfacecolors;
                const double r = *(double*)PyArray_GETPTR2(facecolors, fi, 0);
                const double g = *(double*)PyArray_GETPTR2(facecolors, fi, 1);
                const double b = *(double*)PyArray_GETPTR2(facecolors, fi, 2);
                const double a = *(double*)PyArray_GETPTR2(facecolors, fi, 3);
                CGContextSetRGBFillColor(cr, r, g, b, a);
                if (antialiased && Nedgecolors==0)
                {
                    CGContextSetRGBStrokeColor(cr, r, g, b, a);
                }
            }
            if (Nedgecolors > 1)
            {
                npy_intp fi = i % Nedgecolors;
                const double r = *(double*)PyArray_GETPTR2(edgecolors, fi, 0);
                const double g = *(double*)PyArray_GETPTR2(edgecolors, fi, 1);
                const double b = *(double*)PyArray_GETPTR2(edgecolors, fi, 2);
                const double a = *(double*)PyArray_GETPTR2(edgecolors, fi, 3);
                CGContextSetRGBStrokeColor(cr, r, g, b, a);
            }
	
            if (Nfacecolors > 0)
            {
                if (Nedgecolors > 0 || antialiased)
                {
                    CGContextDrawPath(cr, kCGPathFillStroke);
                }
                else
                {
                    CGContextFillPath(cr);
                }
            }
            else if (Nedgecolors > 0)
            {
                CGContextStrokePath(cr);
            }
            if (Noffsets)
            {
                CGContextTranslateCTM(cr, -translation.x, -translation.y);
            }
        }
    }

exit:
    CGContextRestoreGState(cr);
    if (toffsets) free(toffsets);
    Py_XDECREF(facecolors);
    Py_XDECREF(coordinates);

    if (!ok) return NULL;

    Py_INCREF(Py_None);
    return Py_None;
}

static int _find_minimum(CGFloat values[3])
{
    int i = 0;
    CGFloat minimum = values[0];
    if (values[1] < minimum)
    {
        minimum = values[1];
        i = 1;
    }
    if (values[2] < minimum)
        i = 2;
    return i;
}

static int _find_maximum(CGFloat values[3])
{
    int i = 0;
    CGFloat maximum = values[0];
    if (values[1] > maximum)
    {
        maximum = values[1];
        i = 1;
    }
    if (values[2] > maximum)
        i = 2;
    return i;
}

static void
_rgba_color_evaluator(void* info, const CGFloat input[], CGFloat outputs[])
{
    const CGFloat c1 = input[0];
    const CGFloat c0 = 1.0 - c1;
    CGFloat(* color)[4] = info;
    outputs[0] = c0 * color[0][0] + c1 * color[1][0];
    outputs[1] = c0 * color[0][1] + c1 * color[1][1];
    outputs[2] = c0 * color[0][2] + c1 * color[1][2];
    outputs[3] = c0 * color[0][3] + c1 * color[1][3];
}

static void
_gray_color_evaluator(void* info, const CGFloat input[], CGFloat outputs[])
{
    const CGFloat c1 = input[0];
    const CGFloat c0 = 1.0 - c1;
    CGFloat(* color)[2] = info;
    outputs[0] = c0 * color[0][0] + c1 * color[1][0];
    outputs[1] = c0 * color[0][1] + c1 * color[1][1];
}

static int
_shade_one_color(CGContextRef cr, CGFloat colors[3], CGPoint points[3], int icolor)
{
    const int imin = _find_minimum(colors);
    const int imax = _find_maximum(colors);

    float numerator;
    float denominator;
    float ac;
    float as;
    float phi;
    float distance;
    CGPoint start;
    CGPoint end;
    static CGFunctionCallbacks callbacks = {0, &_rgba_color_evaluator, free};
    CGFloat domain[2] = {0.0, 1.0};
    CGFloat range[8] = {0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    CGFunctionRef function;

    CGFloat(* rgba)[4] = malloc(2*sizeof(CGFloat[4]));
    if (!rgba) return -1;
    else {
        rgba[0][0] = 0.0;
        rgba[0][1] = 0.0;
        rgba[0][2] = 0.0;
        rgba[0][3] = 1.0;
        rgba[1][0] = 0.0;
        rgba[1][1] = 0.0;
        rgba[1][2] = 0.0;
        rgba[1][3] = 1.0;
    }

    denominator = (points[1].x-points[0].x)*(points[2].y-points[0].y)
                - (points[2].x-points[0].x)*(points[1].y-points[0].y);
    numerator = (colors[1]-colors[0])*(points[2].y-points[0].y)
              - (colors[2]-colors[0])*(points[1].y-points[0].y);
    ac = numerator / denominator;
    numerator = (colors[2]-colors[0])*(points[1].x-points[0].x)
              - (colors[1]-colors[0])*(points[2].x-points[0].x);
    as = numerator / denominator;
    phi = atan2(as, ac);

    start.x = points[imin].x;
    start.y = points[imin].y;

    rgba[0][icolor] = colors[imin];
    rgba[1][icolor] = colors[imax];

    distance = (points[imax].x-points[imin].x) * cos(phi) + (points[imax].y-points[imin].y) * sin(phi);

    end.x = start.x + distance * cos(phi);
    end.y = start.y + distance * sin(phi);

    function = CGFunctionCreate(rgba,
                                1, /* one input (position) */
                                domain,
                                4, /* rgba output */
                                range,
                                &callbacks);
    if (function)
    {
        CGColorSpaceRef colorspace = CGColorSpaceCreateWithName(kCGColorSpaceGenericRGB);
        CGShadingRef shading = CGShadingCreateAxial(colorspace,
                                                    start,
                                                    end,
                                                    function,
                                                    true,
                                                    true);
        CGFunctionRelease(function);
        if (shading)
        {
            CGContextDrawShading(cr, shading);
            CGShadingRelease(shading);
            return 1;
        }
    }
    free(rgba);
    return -1;
}

static CGRect _find_enclosing_rect(CGPoint points[3])
{
    CGFloat left = points[0].x;
    CGFloat right = points[0].x;
    CGFloat bottom = points[0].y;
    CGFloat top = points[0].y;
    if (points[1].x < left) left = points[1].x;
    if (points[1].x > right) right = points[1].x;
    if (points[2].x < left) left = points[2].x;
    if (points[2].x > right) right = points[2].x;
    if (points[1].y < bottom) bottom = points[1].y;
    if (points[1].y > top) top = points[1].y;
    if (points[2].y < bottom) bottom = points[2].y;
    if (points[2].y > top) top = points[2].y;
    return CGRectMake(left,bottom,right-left,top-bottom);
}

static int
_shade_alpha(CGContextRef cr, CGFloat alphas[3], CGPoint points[3])
{
    const int imin = _find_minimum(alphas);
    const int imax = _find_maximum(alphas);

    if (alphas[imin]==1.0) return 0;

    CGRect rect = _find_enclosing_rect(points);
    const size_t width = (size_t)rect.size.width;
    const size_t height = (size_t)rect.size.height;
    if (width==0 || height==0) return 0;

    void* data = malloc(width*height);

    CGColorSpaceRef colorspace = CGColorSpaceCreateDeviceGray();
    CGContextRef bitmap = CGBitmapContextCreate(data,
                                                width,
                                                height,
                                                8,
                                                width,
                                                colorspace,
                                                0);
    CGColorSpaceRelease(colorspace);

    if (imin==imax)
    {
        CGRect bitmap_rect = rect;
        bitmap_rect.origin = CGPointZero;
        CGContextSetGrayFillColor(bitmap, alphas[0], 1.0);
        CGContextFillRect(bitmap, bitmap_rect);
    }
    else
    {
        float numerator;
        float denominator;
        float ac;
        float as;
        float phi;
        float distance;
        CGPoint start;
        CGPoint end;
        CGFloat(*gray)[2] = malloc(2*sizeof(CGFloat[2]));

        static CGFunctionCallbacks callbacks = {0, &_gray_color_evaluator, free};
        CGFloat domain[2] = {0.0, 1.0};
        CGFloat range[2] = {0.0, 1.0};
        CGShadingRef shading = NULL;
        CGFunctionRef function;

        gray[0][1] = 1.0;
        gray[1][1] = 1.0;

        denominator = (points[1].x-points[0].x)*(points[2].y-points[0].y)
                    - (points[2].x-points[0].x)*(points[1].y-points[0].y);
        numerator = (alphas[1]-alphas[0])*(points[2].y-points[0].y)
                  - (alphas[2]-alphas[0])*(points[1].y-points[0].y);
        ac = numerator / denominator;
        numerator = (alphas[2]-alphas[0])*(points[1].x-points[0].x)
                  - (alphas[1]-alphas[0])*(points[2].x-points[0].x);
        as = numerator / denominator;
        phi = atan2(as, ac);

        start.x = points[imin].x - rect.origin.x;
        start.y = points[imin].y - rect.origin.y;

        gray[0][0] = alphas[imin];
        gray[1][0] = alphas[imax];

        distance = (points[imax].x-points[imin].x) * cos(phi) + (points[imax].y-points[imin].y) * sin(phi);

        end.x = start.x + distance * cos(phi);
        end.y = start.y + distance * sin(phi);

        function = CGFunctionCreate(gray,
                                    1, /* one input (position) */
                                    domain,
                                    1, /* one output (gray level) */
                                    range,
                                    &callbacks);
        if (function)
        {
            shading = CGShadingCreateAxial(colorspace,
                                           start,
                                           end,
                                           function,
                                           true,
                                           true);
            CGFunctionRelease(function);
        }
        if (shading)
        {
            CGContextDrawShading(bitmap, shading);
            CGShadingRelease(shading);
        }
        else
        {
            free(gray);
        }
    }

    CGImageRef mask = CGBitmapContextCreateImage(bitmap);
    CGContextClipToMask(cr, rect, mask);
    CGImageRelease(mask);
    free(data);
    return 0;
}

static PyObject*
GraphicsContext_draw_gouraud_triangle (GraphicsContext* self, PyObject* args)

{
    PyObject* coordinates;
    PyObject* colors;

    CGPoint points[3];
    CGFloat intensity[3];

    int i = 0;

    CGContextRef cr = self->cr;
    if (!cr)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGContextRef is NULL");
        return NULL;
    }

    if(!PyArg_ParseTuple(args, "OO", &coordinates, &colors)) return NULL;

    /* ------------------- Check coordinates array ------------------------ */

    coordinates = PyArray_FromObject(coordinates, NPY_DOUBLE, 2, 2);
    if (!coordinates ||
        PyArray_DIM(coordinates, 0) != 3 || PyArray_DIM(coordinates, 1) != 2)
    {
        PyErr_SetString(PyExc_ValueError, "Invalid coordinates array");
        Py_XDECREF(coordinates);
        return NULL;
    }
    points[0].x = *((double*)(PyArray_GETPTR2(coordinates, 0, 0)));
    points[0].y = *((double*)(PyArray_GETPTR2(coordinates, 0, 1)));
    points[1].x = *((double*)(PyArray_GETPTR2(coordinates, 1, 0)));
    points[1].y = *((double*)(PyArray_GETPTR2(coordinates, 1, 1)));
    points[2].x = *((double*)(PyArray_GETPTR2(coordinates, 2, 0)));
    points[2].y = *((double*)(PyArray_GETPTR2(coordinates, 2, 1)));

    /* ------------------- Check colors array ----------------------------- */

    colors = PyArray_FromObject(colors, NPY_DOUBLE, 2, 2);
    if (!colors ||
        PyArray_DIM(colors, 0) != 3 || PyArray_DIM(colors, 1) != 4)
    {
        PyErr_SetString(PyExc_ValueError, "colors must by a 3x4 array");
        Py_DECREF(coordinates);
        Py_XDECREF(colors);
        return NULL;
    }

    /* ----- Draw the gradients separately for each color component ------- */
    CGContextSaveGState(cr);
    CGContextMoveToPoint(cr, points[0].x, points[0].y);
    CGContextAddLineToPoint(cr, points[1].x, points[1].y);
    CGContextAddLineToPoint(cr, points[2].x, points[2].y);
    CGContextClip(cr);
    intensity[0] = *((double*)(PyArray_GETPTR2(colors, 0, 3)));
    intensity[1] = *((double*)(PyArray_GETPTR2(colors, 1, 3)));
    intensity[2] = *((double*)(PyArray_GETPTR2(colors, 2, 3)));
    if (_shade_alpha(cr, intensity, points)!=-1) {
        CGContextBeginTransparencyLayer(cr, NULL);
        CGContextSetBlendMode(cr, kCGBlendModeScreen);
        for (i = 0; i < 3; i++)
        {
            intensity[0] = *((double*)(PyArray_GETPTR2(colors, 0, i)));
            intensity[1] = *((double*)(PyArray_GETPTR2(colors, 1, i)));
            intensity[2] = *((double*)(PyArray_GETPTR2(colors, 2, i)));
            if (!_shade_one_color(cr, intensity, points, i)) break;
        }
        CGContextEndTransparencyLayer(cr);
    }
    CGContextRestoreGState(cr);

    Py_DECREF(coordinates);
    Py_DECREF(colors);

    if (i < 3) /* break encountered */
    {
        PyErr_SetString(PyExc_MemoryError, "insufficient memory in draw_gouraud_triangle");
        return NULL;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

#ifdef COMPILING_FOR_10_5
static CTFontRef
#else
static ATSFontRef
#endif
setfont(CGContextRef cr, PyObject* family, float size, const char weight[],
        const char italic[])
{
#define NMAP 40
#define NFONT 31
    int i, j, n;
    const char* temp;
    const char* name = "Times-Roman";
    CFStringRef string;
#ifdef COMPILING_FOR_10_5
    CTFontRef font = 0;
#else
    ATSFontRef font = 0;
#endif
#if PY3K
    PyObject* ascii = NULL;
#endif

    const int k = (strcmp(italic, "italic") ? 0 : 2)
                + (strcmp(weight, "bold") ? 0 : 1);

    struct {char* name; int index;} map[NMAP] = {
        {"New Century Schoolbook", 0},
        {"Century Schoolbook L", 0},
        {"Utopia", 1},
        {"ITC Bookman", 2},
        {"Bookman", 2},
        {"Bitstream Vera Serif", 3},
        {"Nimbus Roman No9 L", 4},
        {"Times New Roman", 5},
        {"Times", 6},
        {"Palatino", 7},
        {"Charter", 8},
        {"serif", 0},
        {"Lucida Grande", 9},
        {"Verdana", 10},
        {"Geneva", 11},
        {"Lucida", 12},
        {"Bitstream Vera Sans", 13},
        {"Arial", 14},
        {"Helvetica", 15},
        {"Avant Garde", 16},
        {"sans-serif", 15},
        {"Apple Chancery", 17},
        {"Textile", 18},
        {"Zapf Chancery", 19},
        {"Sand", 20},
        {"cursive", 17},
        {"Comic Sans MS", 21},
        {"Chicago", 22},
        {"Charcoal", 23},
        {"Impact", 24},
        {"Western", 25},
        {"fantasy", 21},
        {"Andale Mono", 26},
        {"Bitstream Vera Sans Mono", 27},
        {"Nimbus Mono L", 28},
        {"Courier", 29},
        {"Courier New", 30},
        {"Fixed", 30},
        {"Terminal", 30},
        {"monospace", 30},
    };

    const char* psnames[NFONT][4] = {
      {"CenturySchoolbook",                   /* 0 */
       "CenturySchoolbook-Bold",
       "CenturySchoolbook-Italic",
       "CenturySchoolbook-BoldItalic"},
      {"Utopia",                              /* 1 */
       "Utopia-Bold",
       "Utopia-Italic",
       "Utopia-BoldItalic"},
      {"Bookman-Light",                       /* 2 */
       "Bookman-Bold",
       "Bookman-LightItalic",
       "Bookman-BoldItalic"},
      {"BitstreamVeraSerif-Roman",            /* 3 */
       "BitstreamVeraSerif-Bold",
       "",
       ""},
      {"NimbusRomNo9L-Reg",                   /* 4 */
       "NimbusRomNo9T-Bol",
       "NimbusRomNo9L-RegIta",
       "NimbusRomNo9T-BolIta"},
      {"TimesNewRomanPSMT",                   /* 5 */
       "TimesNewRomanPS-BoldMT",
       "TimesNewRomanPS-ItalicMT",
       "TimesNewRomanPS-BoldItalicMT"},
      {"Times-Roman",                         /* 6 */
       "Times-Bold",
       "Times-Italic",
       "Times-BoldItalic"},
      {"Palatino-Roman",                      /* 7 */
       "Palatino-Bold",
       "Palatino-Italic",
       "Palatino-BoldItalic"},
      {"CharterBT-Roman",                     /* 8 */
       "CharterBT-Bold",
       "CharterBT-Italic",
       "CharterBT-BoldItalic"},
      {"LucidaGrande",                        /* 9 */
       "LucidaGrande-Bold",
       "",
       ""},
      {"Verdana",                            /* 10 */
       "Verdana-Bold",
       "Verdana-Italic",
       "Verdana-BoldItalic"},
      {"Geneva",                             /* 11 */
       "",
       "",
       ""},
      {"LucidaSans",                         /* 12 */
       "LucidaSans-Demi",
       "LucidaSans-Italic",
       "LucidaSans-DemiItalic"},
      {"BitstreamVeraSans-Roman",            /* 13 */
       "BitstreamVeraSans-Bold",
       "BitstreamVeraSans-Oblique",
       "BitstreamVeraSans-BoldOblique"},
      {"ArialMT",                            /* 14 */
       "Arial-BoldMT",
       "Arial-ItalicMT",
       "Arial-BoldItalicMT"},
      {"Helvetica",                          /* 15 */
       "Helvetica-Bold",
       "Arial-ItalicMT",
       "Arial-BoldItalicMT"},
      {"AvantGardeITC-Book",                 /* 16 */
       "AvantGardeITC-Demi",
       "AvantGardeITC-BookOblique",
       "AvantGardeITC-DemiOblique"},
      {"Apple-Chancery",                     /* 17 */
       "",
       "",
       ""},
      {"TextileRegular",                     /* 18 */
       "",
       "",
       ""},
      {"ZapfChancery-Roman",                 /* 19 */
       "ZapfChancery-Bold",
       "ZapfChancery-Italic",
       "ZapfChancery-MediumItalic"},
      {"SandRegular",                        /* 20 */
       "",
       "",
       ""},
      {"ComicSansMS",                        /* 21 */
       "ComicSansMS-Bold",
       "",
       ""},
      {"Chicago",                            /* 22 */
       "",
       "",
       ""},
      {"Charcoal",                           /* 23 */
       "",
       "",
       ""},
      {"Impact",                             /* 24 */
       "",
       "",
       ""},
      {"Playbill",                           /* 25 */
       "",
       "",
       ""},
      {"AndaleMono",                         /* 26 */
       "",
       "",
       ""},
      {"BitstreamVeraSansMono-Roman",        /* 27 */
       "BitstreamVeraSansMono-Bold",
       "BitstreamVeraSansMono-Oblique",
       "BitstreamVeraSansMono-BoldOb"},
      {"NimbusMonL-Reg",                     /* 28 */
       "NimbusMonL-Bol",
       "NimbusMonL-RegObl",
       "NimbusMonL-BolObl"},
      {"Courier",                            /* 29 */
       "Courier-Bold",
       "",
       ""},
      {"CourierNewPS",                       /* 30 */
       "CourierNewPS-BoldMT",
       "CourierNewPS-ItalicMT",
       "CourierNewPS-Bold-ItalicMT"},
    };

    if(!PyList_Check(family)) return 0;
    n = PyList_GET_SIZE(family);

    for (i = 0; i < n; i++)
    {
        PyObject* item = PyList_GET_ITEM(family, i);
#if PY3K
        ascii = PyUnicode_AsASCIIString(item);
        if(!ascii) return 0;
        temp = PyBytes_AS_STRING(ascii);
#else
        if(!PyString_Check(item)) return 0;
        temp = PyString_AS_STRING(item);
#endif
        for (j = 0; j < NMAP; j++)
        {    if (!strcmp(map[j].name, temp))
             {    temp = psnames[map[j].index][k];
                  break;
             }
        }
        /* If the font name is not found in mapping, we assume */
        /* that the user specified the Postscript name directly */

        /* Check if this font can be found on the system */
        string = CFStringCreateWithCString(kCFAllocatorDefault,
                                           temp,
                                           kCFStringEncodingMacRoman);
#ifdef COMPILING_FOR_10_5
        font = CTFontCreateWithName(string, size, NULL);
#else
        font = ATSFontFindFromPostScriptName(string, kATSOptionFlagsDefault);
#endif

        CFRelease(string);

        if(font)
        {
            name = temp;
            break;
        }
#if PY3K
        Py_DECREF(ascii);
        ascii = NULL;
#endif
    }
    if(!font)
    {   string = CFStringCreateWithCString(kCFAllocatorDefault,
                                           name,
                                           kCFStringEncodingMacRoman);
#ifdef COMPILING_FOR_10_5
        font = CTFontCreateWithName(string, size, NULL);
#else
        font = ATSFontFindFromPostScriptName(string, kATSOptionFlagsDefault);
#endif
        CFRelease(string);
    }
#ifndef COMPILING_FOR_10_5
    CGContextSelectFont(cr, name, size, kCGEncodingMacRoman);
#endif
#if PY3K
    Py_XDECREF(ascii);
#endif
    return font;
}

#ifdef COMPILING_FOR_10_5
static PyObject*
GraphicsContext_draw_text (GraphicsContext* self, PyObject* args)
{
    float x;
    float y;
    int n;
    PyObject* family;
    float size;
    const char* weight;
    const char* italic;
    float angle;
    CTFontRef font;
    CGColorRef color;
    CGFloat descent;
#if PY33
    const char* text;
#else
    const UniChar* text;
#endif

    CFStringRef keys[2];
    CFTypeRef values[2];

    CGContextRef cr = self->cr;
    if (!cr)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGContextRef is NULL");
        return NULL;
    }
#if PY33
    if(!PyArg_ParseTuple(args, "ffs#Ofssf",
                                &x,
                                &y,
                                &text,
                                &n,
                                &family,
                                &size,
                                &weight,
                                &italic,
                                &angle)) return NULL;
    CFStringRef s = CFStringCreateWithCString(kCFAllocatorDefault, text, kCFStringEncodingUTF8);
#else
    if(!PyArg_ParseTuple(args, "ffu#Ofssf",
                                &x,
                                &y,
                                &text,
                                &n,
                                &family,
                                &size,
                                &weight,
                                &italic,
                                &angle)) return NULL;
    CFStringRef s = CFStringCreateWithCharacters(kCFAllocatorDefault, text, n);
#endif

    font = setfont(cr, family, size, weight, italic);

    color = CGColorCreateGenericRGB(self->color[0],
                                    self->color[1],
                                    self->color[2],
                                    self->color[3]);

    keys[0] = kCTFontAttributeName;
    keys[1] = kCTForegroundColorAttributeName;
    values[0] = font;
    values[1] = color;
    CFDictionaryRef attributes = CFDictionaryCreate(kCFAllocatorDefault,
                                        (const void**)&keys,
                                        (const void**)&values,
                                        2,
                                        &kCFTypeDictionaryKeyCallBacks,
                                        &kCFTypeDictionaryValueCallBacks);
    CGColorRelease(color);
    CFRelease(font);

    CFAttributedStringRef string = CFAttributedStringCreate(kCFAllocatorDefault,
                                                            s,
                                                            attributes);
    CFRelease(s);
    CFRelease(attributes);

    CTLineRef line = CTLineCreateWithAttributedString(string);
    CFRelease(string);

    CTLineGetTypographicBounds(line, NULL, &descent, NULL);

    if (!line)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "CTLineCreateWithAttributedString failed");
        return NULL;
    }

    CGContextSetTextMatrix(cr, CGAffineTransformIdentity);
    if (angle)
    {
        CGContextSaveGState(cr);
        CGContextTranslateCTM(cr, x, y);
        CGContextRotateCTM(cr, angle*M_PI/180);
        CTLineDraw(line, cr);
        CGContextRestoreGState(cr);
    }
    else
    {
        CGContextSetTextPosition(cr, x, y);
        CTLineDraw(line, cr);
    }
    CFRelease(line);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
GraphicsContext_get_text_width_height_descent(GraphicsContext* self, PyObject* args)
{
    int n;
    PyObject* family;
    float size;
    const char* weight;
    const char* italic;

#if PY33
    const char* text;
#else
    const UniChar* text;
#endif

    CGFloat ascent;
    CGFloat descent;
    double width;
    CGRect rect;

    CTFontRef font;

    CGContextRef cr = self->cr;
    if (!cr)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGContextRef is NULL");
        return NULL;
    }

#if PY33
    if(!PyArg_ParseTuple(args, "s#Ofss",
                                &text,
                                &n,
                                &family,
                                &size,
                                &weight,
                                &italic)) return NULL;
    CFStringRef s = CFStringCreateWithCString(kCFAllocatorDefault, text, kCFStringEncodingUTF8);
#else
    if(!PyArg_ParseTuple(args, "u#Ofss",
                                &text,
                                &n,
                                &family,
                                &size,
                                &weight,
                                &italic)) return NULL;
    CFStringRef s = CFStringCreateWithCharacters(kCFAllocatorDefault, text, n);
#endif

    font = setfont(cr, family, size, weight, italic);

    CFStringRef keys[1];
    CFTypeRef values[1];

    keys[0] = kCTFontAttributeName;
    values[0] = font;
    CFDictionaryRef attributes = CFDictionaryCreate(kCFAllocatorDefault,
                                        (const void**)&keys,
                                        (const void**)&values,
                                        1,
                                        &kCFTypeDictionaryKeyCallBacks,
                                        &kCFTypeDictionaryValueCallBacks);
    CFRelease(font);

    CFAttributedStringRef string = CFAttributedStringCreate(kCFAllocatorDefault,
                                                            s,
                                                            attributes);
    CFRelease(s);
    CFRelease(attributes);

    CTLineRef line = CTLineCreateWithAttributedString(string);
    CFRelease(string);

    if (!line)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "CTLineCreateWithAttributedString failed");
        return NULL;
    }

    width = CTLineGetTypographicBounds(line, &ascent, &descent, NULL);
    rect = CTLineGetImageBounds(line, cr);

    CFRelease(line);

    return Py_BuildValue("fff", width, rect.size.height, descent);
}

#else // Text drawing for OSX versions <10.5

static PyObject*
GraphicsContext_draw_text (GraphicsContext* self, PyObject* args)
{
    float x;
    float y;
    const UniChar* text;
    int n;
    PyObject* family;
    float size;
    const char* weight;
    const char* italic;
    float angle;
    ATSFontRef atsfont;
    CGContextRef cr = self->cr;
    if (!cr)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGContextRef is NULL");
        return NULL;
    }
    if(!PyArg_ParseTuple(args, "ffu#Ofssf",
                                &x,
                                &y,
                                &text,
                                &n,
                                &family,
                                &size,
                                &weight,
                                &italic,
                                &angle)) return NULL;

    atsfont = setfont(cr, family, size, weight, italic);

    OSStatus status;

    ATSUAttributeTag tags[] =  {kATSUFontTag, kATSUSizeTag, kATSUQDBoldfaceTag};
    ByteCount sizes[] = {sizeof(ATSUFontID), sizeof(Fixed), sizeof(Boolean)};
    Fixed atsuSize = Long2Fix(size);
    Boolean isBold = FALSE; /* setfont takes care of this */

    ATSUAttributeValuePtr values[] = {&atsfont, &atsuSize, &isBold};
    status = ATSUSetAttributes(style, 3, tags, sizes, values);
    if (status!=noErr)
    {
        PyErr_SetString(PyExc_RuntimeError, "ATSUSetAttributes failed");
        return NULL;
    }

    status = ATSUSetTextPointerLocation(layout,
                    text,
                    kATSUFromTextBeginning,  /* offset from beginning */
                    kATSUToTextEnd,          /* length of text range */
                    n);                      /* length of text buffer */
    if (status!=noErr)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "ATSUCreateTextLayoutWithTextPtr failed");
        return NULL;
    }

    status = ATSUSetRunStyle(layout,
                             style,
                             kATSUFromTextBeginning,
                             kATSUToTextEnd);
    if (status!=noErr)
    {
        PyErr_SetString(PyExc_RuntimeError, "ATSUSetRunStyle failed");
        return NULL;
    }

    Fixed atsuAngle = X2Fix(angle);
    ATSUAttributeTag tags2[] = {kATSUCGContextTag, kATSULineRotationTag};
    ByteCount sizes2[] = {sizeof (CGContextRef), sizeof(Fixed)};
    ATSUAttributeValuePtr values2[] = {&cr, &atsuAngle};
    status = ATSUSetLayoutControls(layout, 2, tags2, sizes2, values2);
    if (status!=noErr)
    {
        PyErr_SetString(PyExc_RuntimeError, "ATSUSetLayoutControls failed");
        return NULL;
    }

    status = ATSUDrawText(layout,
                          kATSUFromTextBeginning,
                          kATSUToTextEnd,
                          X2Fix(x),
                          X2Fix(y));
    if (status!=noErr)
    {
        PyErr_SetString(PyExc_RuntimeError, "ATSUDrawText failed");
        return NULL;
    }
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
GraphicsContext_get_text_width_height_descent(GraphicsContext* self, PyObject* args)
{
    const UniChar* text;
    int n;
    PyObject* family;
    float size;
    const char* weight;
    const char* italic;

    ATSFontRef atsfont;

    CGContextRef cr = self->cr;
    if (!cr)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGContextRef is NULL");
        return NULL;
    }

    if(!PyArg_ParseTuple(args, "u#Ofss", &text, &n, &family, &size, &weight, &italic)) return NULL;

    atsfont = setfont(cr, family, size, weight, italic);

    OSStatus status = noErr;
    ATSUAttributeTag tags[] = {kATSUFontTag,
                               kATSUSizeTag,
                               kATSUQDBoldfaceTag,
                               kATSUQDItalicTag};
    ByteCount sizes[] = {sizeof(ATSUFontID),
                         sizeof(Fixed),
                         sizeof(Boolean),
                         sizeof(Boolean)};
    Fixed atsuSize = Long2Fix(size);
    Boolean isBold = FALSE; /* setfont takes care of this */
    Boolean isItalic = FALSE; /* setfont takes care of this */
    ATSUAttributeValuePtr values[] = {&atsfont, &atsuSize, &isBold, &isItalic};

    status = ATSUSetAttributes(style, 4, tags, sizes, values);
    if (status!=noErr)
    {
        PyErr_SetString(PyExc_RuntimeError, "ATSUSetAttributes failed");
        return NULL;
    }

    status = ATSUSetTextPointerLocation(layout,
                    text,
                    kATSUFromTextBeginning,  /* offset from beginning */
                    kATSUToTextEnd,          /* length of text range */
                    n);                      /* length of text buffer */
    if (status!=noErr)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "ATSUCreateTextLayoutWithTextPtr failed");
        return NULL;
    }

    status = ATSUSetRunStyle(layout,
                             style,
                             kATSUFromTextBeginning,
                             kATSUToTextEnd);
    if (status!=noErr)
    {
        PyErr_SetString(PyExc_RuntimeError, "ATSUSetRunStyle failed");
        return NULL;
    }

    ATSUAttributeTag tag = kATSUCGContextTag;
    ByteCount bc = sizeof (CGContextRef);
    ATSUAttributeValuePtr value = &cr;
    status = ATSUSetLayoutControls(layout, 1, &tag, &bc, &value);
    if (status!=noErr)
    {
        PyErr_SetString(PyExc_RuntimeError, "ATSUSetLayoutControls failed");
        return NULL;
    }

    ATSUTextMeasurement before;
    ATSUTextMeasurement after;
    ATSUTextMeasurement ascent;
    ATSUTextMeasurement descent;
    status = ATSUGetUnjustifiedBounds(layout,
                                      kATSUFromTextBeginning, kATSUToTextEnd,
                                      &before, &after, &ascent, &descent);
    if (status!=noErr)
    {
        PyErr_SetString(PyExc_RuntimeError, "ATSUGetUnjustifiedBounds failed");
        return NULL;
    }

    const float width = FixedToFloat(after-before);
    const float height = FixedToFloat(ascent-descent);

    return Py_BuildValue("fff", width, height, FixedToFloat(descent));
}
#endif

static void _data_provider_release(void* info, const void* data, size_t size)
{
    PyObject* image = (PyObject*)info;
    Py_DECREF(image);
}

static PyObject*
GraphicsContext_draw_mathtext(GraphicsContext* self, PyObject* args)
{
    float x, y, angle;
    npy_intp nrows, ncols;
    int n;

    PyObject* object;
    PyArrayObject* image;

    CGContextRef cr = self->cr;
    if (!cr)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGContextRef is NULL");
        return NULL;
    }

    if(!PyArg_ParseTuple(args, "fffO", &x, &y, &angle, &object)) return NULL;

    /* ------------- Check the image ---------------------------- */
    if(!PyArray_Check (object))
    {
        PyErr_SetString(PyExc_TypeError, "image should be a NumPy array.");
        return NULL;
    }
    image = (PyArrayObject*) object;
    if(PyArray_NDIM(image) != 2)
    {
        PyErr_Format(PyExc_TypeError,
                         "image has incorrect rank (%d expected 2)",
                         PyArray_NDIM(image));
        return NULL;
    }
    if (PyArray_TYPE(image) != NPY_UBYTE)
    {
        PyErr_SetString(PyExc_TypeError,
                        "image has incorrect type (should be uint8)");
        return NULL;
    }
    if (!PyArray_ISCONTIGUOUS(image))
    {
        PyErr_SetString(PyExc_TypeError, "image array is not contiguous");
        return NULL;
    }

    nrows = PyArray_DIM(image, 0);
    ncols = PyArray_DIM(image, 1);
    if (nrows != (int) nrows || ncols != (int) ncols)
    {
        PyErr_SetString(PyExc_RuntimeError, "bitmap image too large");
        return NULL;
    }
    n = nrows * ncols;
    Py_INCREF(object);

    const size_t bytesPerComponent = 1;
    const size_t bitsPerComponent = 8 * bytesPerComponent;
    const size_t nComponents = 1; /* gray */
    const size_t bitsPerPixel = bitsPerComponent * nComponents;
    const size_t bytesPerRow = nComponents * bytesPerComponent * ncols;
    CGDataProviderRef provider = CGDataProviderCreateWithData(object,
                                                              PyArray_DATA(image),
                                                              n,
                                                              _data_provider_release);
    CGImageRef bitmap = CGImageMaskCreate ((int) ncols,
                                           (int) nrows,
                                           bitsPerComponent,
                                           bitsPerPixel,
                                           bytesPerRow,
                                           provider,
                                           NULL,
                                           false);
    CGDataProviderRelease(provider);

    if(!bitmap)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGImageMaskCreate failed");
        return NULL;
    }

    if (angle==0.0)
    {
        CGContextDrawImage(cr, CGRectMake(x,y,ncols,nrows), bitmap);
    }
    else
    {
        CGContextSaveGState(cr);
        CGContextTranslateCTM(cr, x, y);
        CGContextRotateCTM(cr, angle*M_PI/180);
        CGContextDrawImage(cr, CGRectMake(0,0,ncols,nrows), bitmap);
        CGContextRestoreGState(cr);
    }
    CGImageRelease(bitmap);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject*
GraphicsContext_draw_image(GraphicsContext* self, PyObject* args)
{
    float x, y;
    int nrows, ncols;
    const char* data;
    int n;
    PyObject* image;

    CGContextRef cr = self->cr;
    if (!cr)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGContextRef is NULL");
        return NULL;
    }

    if(!PyArg_ParseTuple(args, "ffiiO", &x,
                                        &y,
                                        &nrows,
                                        &ncols,
                                        &image)) return NULL;

    CGColorSpaceRef colorspace;
    CGDataProviderRef provider;

    if (!PyBytes_Check(image))
    {
#if PY3K
        PyErr_SetString(PyExc_RuntimeError, "image is not a bytes object");
#else
        PyErr_SetString(PyExc_RuntimeError, "image is not a str object");
#endif
        return NULL;
    }

    const size_t bytesPerComponent = 1;
    const size_t bitsPerComponent = 8 * bytesPerComponent;
    const size_t nComponents = 4; /* red, green, blue, alpha */
    const size_t bitsPerPixel = bitsPerComponent * nComponents;
    const size_t bytesPerRow = nComponents * bytesPerComponent * ncols;

    colorspace = CGColorSpaceCreateWithName(kCGColorSpaceGenericRGB);
    if (!colorspace)
    {
        PyErr_SetString(PyExc_RuntimeError, "failed to create a color space");
        return NULL;
    }

    Py_INCREF(image);
#ifdef PY3K
    n = PyBytes_GET_SIZE(image);
    data = PyBytes_AS_STRING(image);
#else
    n = PyString_GET_SIZE(image);
    data = PyString_AS_STRING(image);
#endif

    provider = CGDataProviderCreateWithData(image,
                                            data,
                                            n,
                                            _data_provider_release);
    CGImageRef bitmap = CGImageCreate (ncols,
                                       nrows,
                                       bitsPerComponent,
                                       bitsPerPixel,
                                       bytesPerRow,
                                       colorspace,
                                       kCGImageAlphaLast,
                                       provider,
                                       NULL,
                                       false,
				       kCGRenderingIntentDefault);
    CGColorSpaceRelease(colorspace);
    CGDataProviderRelease(provider);

    if(!bitmap)
    {
        PyErr_SetString(PyExc_RuntimeError, "CGImageMaskCreate failed");
        return NULL;
    }

    CGContextDrawImage(cr, CGRectMake(x,y,ncols,nrows), bitmap);
    CGImageRelease(bitmap);

    Py_INCREF(Py_None);
    return Py_None;
}


static PyMethodDef GraphicsContext_methods[] = {
    {"save",
     (PyCFunction)GraphicsContext_save,
     METH_NOARGS,
     "Saves the current graphics context onto the stack."
    },
    {"restore",
     (PyCFunction)GraphicsContext_restore,
     METH_NOARGS,
     "Restores the current graphics context from the stack."
    },
    {"get_text_width_height_descent",
     (PyCFunction)GraphicsContext_get_text_width_height_descent,
     METH_VARARGS,
     "Returns the width, height, and descent of the text."
    },
    {"set_alpha",
     (PyCFunction)GraphicsContext_set_alpha,
      METH_VARARGS,
     "Sets the opacitiy level for objects drawn in a graphics context"
    },
    {"set_antialiased",
     (PyCFunction)GraphicsContext_set_antialiased,
     METH_VARARGS,
     "Sets anti-aliasing on or off for a graphics context."
    },
    {"set_capstyle",
     (PyCFunction)GraphicsContext_set_capstyle,
     METH_VARARGS,
     "Sets the style for the endpoints of lines in a graphics context."
    },
    {"set_clip_rectangle",
     (PyCFunction)GraphicsContext_set_clip_rectangle,
     METH_VARARGS,
     "Sets the clipping path to the area defined by the specified rectangle."
    },
    {"set_clip_path",
     (PyCFunction)GraphicsContext_set_clip_path,
     METH_VARARGS,
     "Sets the clipping path."
    },
    {"set_dashes",
     (PyCFunction)GraphicsContext_set_dashes,
     METH_VARARGS,
     "Sets the pattern for dashed lines in a graphics context."
    },
    {"set_foreground",
     (PyCFunction)GraphicsContext_set_foreground,
     METH_VARARGS,
     "Sets the current stroke and fill color to a value in the DeviceRGB color space."
    },
    {"set_graylevel",
     (PyCFunction)GraphicsContext_set_graylevel,
     METH_VARARGS,
     "Sets the current stroke and fill color to a value in the DeviceGray color space."
    },
    {"set_dpi",
     (PyCFunction)GraphicsContext_set_dpi,
     METH_VARARGS,
     "Sets the dpi for a graphics context."
    },
    {"set_linewidth",
     (PyCFunction)GraphicsContext_set_linewidth,
     METH_VARARGS,
     "Sets the line width for a graphics context."
    },
    {"set_joinstyle",
     (PyCFunction)GraphicsContext_set_joinstyle,
     METH_VARARGS,
     "Sets the style for the joins of connected lines in a graphics context."
    },
    {"draw_path",
     (PyCFunction)GraphicsContext_draw_path,
     METH_VARARGS,
     "Draw a path in the graphics context and strokes and (if rgbFace is not None) fills it."
    },
    {"draw_markers",
     (PyCFunction)GraphicsContext_draw_markers,
     METH_VARARGS,
     "Draws a marker in the graphics context at each of the vertices in path."
    },
    {"draw_path_collection",
     (PyCFunction)GraphicsContext_draw_path_collection,
     METH_VARARGS,
     "Draw a collection of paths in the graphics context."
    },
    {"draw_quad_mesh",
     (PyCFunction)GraphicsContext_draw_quad_mesh,
     METH_VARARGS,
     "Draws a mesh in the graphics context."
    },
    {"draw_gouraud_triangle",
     (PyCFunction)GraphicsContext_draw_gouraud_triangle,
     METH_VARARGS,
     "Draws a Gouraud-shaded triangle in the graphics context."
    },
    {"draw_text",
     (PyCFunction)GraphicsContext_draw_text,
     METH_VARARGS,
     "Draw a string at (x,y) with the given properties in the graphics context."
    },
    {"draw_mathtext",
     (PyCFunction)GraphicsContext_draw_mathtext,
     METH_VARARGS,
     "Draw a TeX string at (x,y) as a bitmap in the graphics context."
    },
    {"draw_image",
     (PyCFunction)GraphicsContext_draw_image,
     METH_VARARGS,
     "Draw an image at (x,y) in the graphics context."
    },
    {NULL}  /* Sentinel */
};

static char GraphicsContext_doc[] =
"A GraphicsContext object wraps a Quartz 2D graphics context\n"
"(CGContextRef). Most methods either draw into the graphics context\n"
"(moveto, lineto, etc.) or set the drawing properties (set_linewidth,\n"
"set_joinstyle, etc.).\n";

static PyTypeObject GraphicsContextType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_macosx.GraphicsContext", /*tp_name*/
    sizeof(GraphicsContext),   /*tp_basicsize*/
    0,                         /*tp_itemsize*/
#ifdef COMPILING_FOR_10_5
    0,                         /*tp_dealloc*/
#else
    (destructor)GraphicsContext_dealloc,     /*tp_dealloc*/
#endif
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    (reprfunc)GraphicsContext_repr,     /*tp_repr*/
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
    GraphicsContext_doc,       /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    GraphicsContext_methods,   /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    GraphicsContext_new,       /* tp_new */
};

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
FigureCanvas_write_bitmap(FigureCanvas* self, PyObject* args)
{
    View* view = self->view;
    int n;
    const unichar* characters;
    NSSize size;
    double width, height, dpi;

    if(!view)
    {
        PyErr_SetString(PyExc_RuntimeError, "NSView* is NULL");
        return NULL;
    }
    /* NSSize contains CGFloat; cannot use size directly */
    if(!PyArg_ParseTuple(args, "u#ddd",
                                &characters, &n, &width, &height, &dpi)) return NULL;
    size.width = width;
    size.height = height;

    /* This function may be called from inside the event loop, when an
     * autorelease pool is available, or from Python, when no autorelease
     * pool is available. To be able to handle the latter case, we need to
     * create an autorelease pool here. */

    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];

    NSRect rect = [view bounds];

    NSString* filename = [NSString stringWithCharacters: characters
                                                 length: (unsigned)n];
    NSString* extension = [filename pathExtension];

    /* Calling dataWithPDFInsideRect on the view causes its update status
     * to be cleared. Save the status here, and invalidate the view if not
     * up to date after calling dataWithPDFInsideRect. */
    const BOOL invalid = [view needsDisplay];
    NSData* data = [view dataWithPDFInsideRect: rect];
    if (invalid) [view setNeedsDisplay: YES];

    NSImage* image = [[NSImage alloc] initWithData: data];
    NSImage *resizedImage = [[NSImage alloc] initWithSize:size];

    [resizedImage lockFocus];
    [image drawInRect:NSMakeRect(0, 0, width, height) fromRect:NSZeroRect operation:NSCompositeSourceOver fraction:1.0];
    [resizedImage unlockFocus];
    data = [resizedImage TIFFRepresentation];
    [image release];
    [resizedImage release];

    NSBitmapImageRep* rep = [NSBitmapImageRep imageRepWithData:data];

    NSSize pxlSize = NSMakeSize([rep pixelsWide], [rep pixelsHigh]);
    NSSize newSize = NSMakeSize(72.0 * pxlSize.width / dpi, 72.0 * pxlSize.height / dpi);

    [rep setSize:newSize];

    NSBitmapImageFileType filetype;
    if ([extension isEqualToString: @"bmp"])
        filetype = NSBMPFileType;
    else if ([extension isEqualToString: @"gif"])
        filetype = NSGIFFileType;
    else if ([extension isEqualToString: @"jpg"] ||
             [extension isEqualToString: @"jpeg"])
        filetype = NSJPEGFileType;
    else if ([extension isEqualToString: @"png"])
        filetype = NSPNGFileType;
    else if ([extension isEqualToString: @"tiff"] ||
             [extension isEqualToString: @"tif"])
        filetype = NSTIFFFileType;
    else
    {   PyErr_SetString(PyExc_ValueError, "Unknown file type");
        return NULL;
    }

    data = [rep representationUsingType:filetype properties:nil];

    [data writeToFile: filename atomically: YES];
    [pool release];

    Py_INCREF(Py_None);
    return Py_None;
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
                                                 _callback,
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
    {"write_bitmap",
     (PyCFunction)FigureCanvas_write_bitmap,
     METH_VARARGS,
     "Saves the figure to the specified file as a bitmap\n"
     "(bmp, gif, jpeg, or png).\n"
    },
    {"start_event_loop",
     (PyCFunction)FigureCanvas_start_event_loop,
     METH_KEYWORDS,
     "Runs the event loop until the timeout or until stop_event_loop is called.\n",
    },
    {"stop_event_loop",
     (PyCFunction)FigureCanvas_stop_event_loop,
     METH_KEYWORDS,
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

    NSApp = [NSApplication sharedApplication];
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

    nwin++;

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
        NSString* ns_title = [[NSString alloc]
                              initWithCString: title
                              encoding: NSUTF8StringEncoding];
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
#if PY3K || (PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION >= 6)
            result = PyUnicode_FromString(cTitle);
#else
            result = PyString_FromString(cTitle);
#endif
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
    int* states = malloc(n*sizeof(int));
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
        else states[i] = 0;
    }
    int j = 0;
    PyObject* list = PyList_New(m);
    for (i = 0; i < n; i++)
    {
        if(states[i]==1)
        {
#if PY3K
            PyList_SET_ITEM(list, j, PyLong_FromLong(i));
#else
            PyList_SET_ITEM(list, j, PyInt_FromLong(i));
#endif
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

    const float gap = 2;
    const int height = 36;

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

    NSString* images[7] = {@"home.png",
                           @"back.png",
                           @"forward.png",
                           @"move.png",
                           @"zoom_to_rect.png",
                           @"subplots.png",
                           @"filesave.png"};

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

    rect.size.width = 32;
    rect.size.height = 32;
    rect.origin.x = gap;
    rect.origin.y = 0.5*(height - rect.size.height);
    for (i = 0; i < 7; i++)
    {
        const NSSize size = {24, 24};
        NSString* filename = [dir stringByAppendingPathComponent: images[i]];
        NSImage* image = [[NSImage alloc] initWithContentsOfFile: filename];
        buttons[i] = [[NSButton alloc] initWithFrame: rect];
        [image setSize: size];
        [buttons[i] setBezelStyle: NSShadowlessSquareBezelStyle];
        [buttons[i] setButtonType: buttontypes[i]];
        [buttons[i] setImage: image];
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
    if (result == NSOKButton)
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
    nwin--;
    if(nwin==0) [NSApp stop: self];
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

-(void)drawRect:(NSRect)rect
{
    PyObject* result;
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject* figure = PyObject_GetAttrString(canvas, "figure");
    if (!figure)
    {
        PyErr_Print();
        PyGILState_Release(gstate);
        return;
    }
    PyObject* renderer = PyObject_GetAttrString(canvas, "renderer");
    if (!renderer)
    {
        PyErr_Print();
        Py_DECREF(figure);
        PyGILState_Release(gstate);
        return;
    }
    GraphicsContext* gc = (GraphicsContext*) PyObject_GetAttrString(renderer, "gc");
    if (!gc)
    {
        PyErr_Print();
        Py_DECREF(figure);
        Py_DECREF(renderer);
        PyGILState_Release(gstate);
        return;
    }

    gc->size = [self frame].size;

    CGContextRef cr = (CGContextRef) [[NSGraphicsContext currentContext] graphicsPort];
    gc->cr = cr;
    gc->level = 0;

    result = PyObject_CallMethod(figure, "draw", "O", renderer);
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();

    gc->cr = nil;

    if (!NSIsEmptyRect(rubberband)) NSFrameRect(rubberband);

    Py_DECREF(gc);
    Py_DECREF(figure);
    Py_DECREF(renderer);

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
    PyObject* result = PyObject_CallMethod(canvas, "resize", "ii", width, height);
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
    x = location.x;
    y = location.y;
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
    x = location.x;
    y = location.y;
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
    x = location.x;
    y = location.y;
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
    x = location.x;
    y = location.y;
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
    x = location.x;
    y = location.y;
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
    x = location.x;
    y = location.y;
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
    x = location.x;
    y = location.y;
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
    x = location.x;
    y = location.y;
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
    x = location.x;
    y = location.y;
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
    x = location.x;
    y = location.y;
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
    int x = (int)round(point.x);
    int y = (int)round(point.y - 1);

    PyObject* result;
    PyGILState_STATE gstate = PyGILState_Ensure();
    result = PyObject_CallMethod(canvas, "scroll_event", "iii", x, y, step);
    if(result)
        Py_DECREF(result);
    else
        PyErr_Print();

    PyGILState_Release(gstate);
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
    if(nwin > 0)
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
        [NSApp run];
    }
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

static void
Timer_dealloc(Timer* self)
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
    Py_TYPE(self)->tp_free((PyObject*)self);
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

static PyMethodDef Timer_methods[] = {
    {"_timer_start",
     (PyCFunction)Timer__timer_start,
     METH_VARARGS,
     "Initialize and start the timer."
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

#ifdef WITH_NEXT_FRAMEWORK
    PyObject *module;
    import_array();

    if (PyType_Ready(&GraphicsContextType) < 0
     || PyType_Ready(&FigureCanvasType) < 0
     || PyType_Ready(&FigureManagerType) < 0
     || PyType_Ready(&NavigationToolbarType) < 0
     || PyType_Ready(&NavigationToolbar2Type) < 0
     || PyType_Ready(&TimerType) < 0)
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

    Py_INCREF(&GraphicsContextType);
    Py_INCREF(&FigureCanvasType);
    Py_INCREF(&FigureManagerType);
    Py_INCREF(&NavigationToolbarType);
    Py_INCREF(&NavigationToolbar2Type);
    Py_INCREF(&TimerType);
    PyModule_AddObject(module, "GraphicsContext", (PyObject*) &GraphicsContextType);
    PyModule_AddObject(module, "FigureCanvas", (PyObject*) &FigureCanvasType);
    PyModule_AddObject(module, "FigureManager", (PyObject*) &FigureManagerType);
    PyModule_AddObject(module, "NavigationToolbar", (PyObject*) &NavigationToolbarType);
    PyModule_AddObject(module, "NavigationToolbar2", (PyObject*) &NavigationToolbar2Type);
    PyModule_AddObject(module, "Timer", (PyObject*) &TimerType);

    PyOS_InputHook = wait_for_stdin;

#if PY3K
    return module;
#endif
#else
    /* WITH_NEXT_FRAMEWORK is not defined. This means that Python is not
     * installed as a framework, and therefore the Mac OS X backend will
     * not interact properly with the window manager.
     */
    PyErr_SetString(PyExc_RuntimeError,
        "Python is not installed as a framework. The Mac OS X backend will "
        "not be able to function correctly if Python is not installed as a "
        "framework. See the Python documentation for more information on "
        "installing Python as a framework on Mac OS X. Please either reinstall "
        "Python as a framework, or try one of the other backends.");
#if PY3K
    return NULL;
#else
    return;
#endif
#endif
}
