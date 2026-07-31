#import "MPLFigureCanvas.h"
#import "MPLUtils.h"
#import "MPLFigureManager.h"


static void _buffer_release(void* info, const void* data, size_t size) {
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyBuffer_Release((Py_buffer *)info);
    free(info);
    PyGILState_Release(gstate);
}


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

bool mpl_leftMouseGrabbing = false;

static CGFloat _get_device_scale(CGContextRef cr)
{
    CGSize pixelSize = CGContextConvertSizeToDeviceSpace(cr, CGSizeMake(1, 1));
    return pixelSize.width;
}

bool mpl_check_button(bool present, PyObject* set, char const* name) {
    PyObject* module = NULL, * cls = NULL, * button = NULL;
    bool failed = (
        present
        && (!(module = PyImport_ImportModule("matplotlib.backend_bases"))
            || !(cls = PyObject_GetAttrString(module, "MouseButton"))
            || !(button = PyObject_GetAttrString(cls, name))
            || PySet_Add(set, button)));
    Py_XDECREF(module);
    Py_XDECREF(cls);
    Py_XDECREF(button);
    return failed;
}

PyObject* mpl_buttons()
{
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* set = NULL;
    NSUInteger buttons = [NSEvent pressedMouseButtons];

    if (!(set = PySet_New(NULL))
        || mpl_check_button(buttons & (1 << 0), set, "LEFT")
        || mpl_check_button(buttons & (1 << 1), set, "RIGHT")
        || mpl_check_button(buttons & (1 << 2), set, "MIDDLE")
        || mpl_check_button(buttons & (1 << 3), set, "BACK")
        || mpl_check_button(buttons & (1 << 4), set, "FORWARD")) {
        Py_CLEAR(set);  // On failure, return NULL with an exception set.
    }
    PyGILState_Release(gstate);
    return set;
}

bool mpl_check_modifier(bool present, PyObject* list, char const* name)
{
    PyObject* py_name = NULL;
    bool failed = (
        present
        && (!(py_name = PyUnicode_FromString(name))
            || (PyList_Append(list, py_name))));
    Py_XDECREF(py_name);
    return failed;
}

PyObject* mpl_modifiers(NSEvent* event)
{
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* list = NULL;
    NSUInteger modifiers = [event modifierFlags];
    if (!(list = PyList_New(0))
        || mpl_check_modifier(modifiers & NSEventModifierFlagControl, list, "ctrl")
        || mpl_check_modifier(modifiers & NSEventModifierFlagOption, list, "alt")
        || mpl_check_modifier(modifiers & NSEventModifierFlagShift, list, "shift")
        || mpl_check_modifier(modifiers & NSEventModifierFlagCommand, list, "cmd")) {
        Py_CLEAR(list);  // On failure, return NULL with an exception set.
    }
    PyGILState_Release(gstate);
    return list;
}


@implementation MPLFigureCanvas {
    // Private ivars will live here
}


#pragma mark - Lifecycle

- (instancetype) initWithFrame:(NSRect)rect
{
    if (self = [super initWithFrame: rect]) {
        rubberband = NSZeroRect;
        device_scale = 1;
    }
    return self;
}


#pragma mark - Superclass Overrides

// This will become a -viewDidChangeBackingProperties override
- (void) updateDevicePixelRatio:(double)scale
{
    PyObject *change = NULL;
    PyGILState_STATE gstate = PyGILState_Ensure();

    device_scale = scale;

    if (!(change = PyObject_CallMethod(_pyObject, "_set_device_pixel_ratio", "d", device_scale))) {
        PyErr_Print();
        goto exit;
    }

    if (PyObject_IsTrue(change)) {
        // Notify that there was a resize_event that took place
        process_event(
            "ResizeEvent", "{s:s, s:O}",
            "name", "resize_event", "canvas", _pyObject);
        gil_call_method(_pyObject, "draw_idle");
        [self setNeedsDisplay: YES];
    }

exit:
    Py_XDECREF(change);

    PyGILState_Release(gstate);
}

-(void) drawRect:(NSRect)rect
{
    PyObject* renderer = NULL;
    PyObject* renderer_buffer = NULL;

    PyGILState_STATE gstate = PyGILState_Ensure();

    CGContextRef cr = [[NSGraphicsContext currentContext] CGContext];

    if (!(renderer = PyObject_CallMethod(_pyObject, "get_renderer", ""))
        || !(renderer_buffer = PyObject_CallMethod(renderer, "buffer_rgba", ""))) {
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

// This becomes a -setFrameSize: override
- (void) windowDidResize:(NSNotification*)notification
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
            _pyObject, "resize", "ii", width, height);
    if (result)
        Py_DECREF(result);
    else
        PyErr_Print();
    PyGILState_Release(gstate);
    [self setNeedsDisplay: YES];
}

- (BOOL) acceptsFirstResponder
{
    return YES;
}


#pragma mark - NSWindowDelegate

// This goes away and we will use a -viewDidChangeBackingProperties override
- (void) windowDidChangeBackingProperties:(NSNotification *)notification
{
    Window *window = [notification object];

    [self updateDevicePixelRatio: [window backingScaleFactor]];
}

// This gets moved to MPLFigureManager, which will be a NSWindowController subclass
- (void) windowWillClose:(NSNotification *)notification
{
    // A view should not be the delegate of a window, this check
    // will go away with next refactor
    Window *window = (Window *)[self window];
    if ([window isKindOfClass:[Window class]]) {
        gil_call_method([window pyObject], "_handle_window_will_close");
    }
}

// This gets moved to MPLFigureManager, which will be a NSWindowController subclass
- (BOOL) windowShouldClose:(NSNotification *)notification
{
    // A view should not be the delegate of a window, this check
    // will go away with next refactor
    Window *window = (Window *)[self window];
    if ([window isKindOfClass:[Window class]]) {
        gil_call_method([window pyObject], "_handle_window_should_close");
    }
    return YES;
}


#pragma mark - Keyboard Events

- (NSString *) convertKeyEvent:(NSEvent *)event
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
            // charactersIgnoringModifiers is nullable; guard defensively in case
            // an unexpected event type reaches this path.
            NSString* chars = [event charactersIgnoringModifiers];
            if (chars) {
                [returnkey appendString:chars];
            }
        }
    } else {
        if (([event modifierFlags] & NSEventModifierFlagShift) || keyChangeShift) {
            [returnkey appendString:@"shift+"];
        }
        // Since it was a modifier event trim the final character of the string
        // because we added in "+" earlier
        [returnkey setString: [returnkey substringToIndex:[returnkey length] - 1]];
    }

    return returnkey;
}

- (void) keyDown:(NSEvent *)event
{
    const char* s = [[self convertKeyEvent: event] UTF8String];
    NSPoint location = [[self window] mouseLocationOutsideOfEventStream];
    location = [self convertPoint: location fromView: nil];
    int x = location.x * device_scale,
        y = location.y * device_scale;
    if (s) {
        process_event(
            "KeyEvent", "{s:s, s:O, s:s, s:i, s:i}",
            "name", "key_press_event", "canvas", _pyObject, "key", s, "x", x, "y", y);
    } else {
        process_event(
            "KeyEvent", "{s:s, s:O, s:O, s:i, s:i}",
            "name", "key_press_event", "canvas", _pyObject, "key", Py_None, "x", x, "y", y);
    }
}

- (void) keyUp:(NSEvent *)event
{
    const char* s = [[self convertKeyEvent: event] UTF8String];
    NSPoint location = [[self window] mouseLocationOutsideOfEventStream];
    location = [self convertPoint: location fromView: nil];
    int x = location.x * device_scale,
        y = location.y * device_scale;
    if (s) {
        process_event(
            "KeyEvent", "{s:s, s:O, s:s, s:i, s:i}",
            "name", "key_release_event", "canvas", _pyObject, "key", s, "x", x, "y", y);
    } else {
        process_event(
            "KeyEvent", "{s:s, s:O, s:O, s:i, s:i}",
            "name", "key_release_event", "canvas", _pyObject, "key", Py_None, "x", x, "y", y);
    }
}

// flagsChanged gets called whenever a  modifier key is pressed OR released
// so we need to handle both cases here
- (void) flagsChanged:(NSEvent *)event
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


#pragma mark - Mouse Events

- (void) mouseEntered:(NSEvent *)event
{
    int x, y;
    NSPoint location = [event locationInWindow];
    location = [self convertPoint: location fromView: nil];
    x = location.x * device_scale;
    y = location.y * device_scale;
    process_event(
        "LocationEvent", "{s:s, s:O, s:i, s:i, s:N}",
        "name", "figure_enter_event", "canvas", _pyObject, "x", x, "y", y,
        "modifiers", mpl_modifiers(event));
}

- (void) mouseExited:(NSEvent *)event
{
    int x, y;
    NSPoint location = [event locationInWindow];
    location = [self convertPoint: location fromView: nil];
    x = location.x * device_scale;
    y = location.y * device_scale;
    process_event(
        "LocationEvent", "{s:s, s:O, s:i, s:i, s:N}",
        "name", "figure_leave_event", "canvas", _pyObject, "x", x, "y", y,
        "modifiers", mpl_modifiers(event));
}


- (void) mouseMoved:(NSEvent *)event
{
    int x, y;
    NSPoint location = [event locationInWindow];
    location = [self convertPoint: location fromView: nil];
    x = location.x * device_scale;
    y = location.y * device_scale;
    process_event(
        "MouseEvent", "{s:s, s:O, s:i, s:i, s:N, s:N}",
        "name", "motion_notify_event", "canvas", _pyObject, "x", x, "y", y,
        "buttons", mpl_buttons(), "modifiers", mpl_modifiers(event));
}

- (void) scrollWheel:(NSEvent *)event
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
        "name", "scroll_event", "canvas", _pyObject,
        "x", x, "y", y, "step", step, "modifiers", mpl_modifiers(event));
}

- (void) mouseDown:(NSEvent *)event
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
                     mpl_leftMouseGrabbing = true;
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
        "name", "button_press_event", "canvas", _pyObject, "x", x, "y", y,
        "button", button, "dblclick", dblclick, "modifiers", mpl_modifiers(event));
}

- (void) mouseUp:(NSEvent *)event
{
    int button;
    int x, y;
    NSPoint location = [event locationInWindow];
    location = [self convertPoint: location fromView: nil];
    x = location.x * device_scale;
    y = location.y * device_scale;
    switch ([event type])
    {    case NSEventTypeLeftMouseUp:
             mpl_leftMouseGrabbing = false;
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
        "name", "button_release_event", "canvas", _pyObject, "x", x, "y", y,
        "button", button, "modifiers", mpl_modifiers(event));
}

// Funnel other down/up events to -mouseDown: or -mouseUp:
- (void) rightMouseDown:(NSEvent *)event { [self mouseDown:event]; }
- (void) otherMouseDown:(NSEvent *)event { [self mouseDown:event]; }
- (void) rightMouseUp:  (NSEvent *)event { [self mouseUp:event]; }
- (void) otherMouseUp:  (NSEvent *)event { [self mouseUp:event]; }

// Funnel dragged events to -mouseMoved:
- (void) mouseDragged:     (NSEvent *)event { [self mouseMoved:event]; }
- (void) rightMouseDragged:(NSEvent *)event { [self mouseMoved:event]; }
- (void) otherMouseDragged:(NSEvent *)event { [self mouseMoved:event]; }


#pragma mark - Public Methods

// This will become -updateLayerWithBuffer:
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

// Becomes -updateRubberbandWithDeviceX0:y0:x1:y1:
- (void) setRubberband:(NSRect)rect
{
    // The space we want to redraw is a union of the previous rubberband
    // with the new rubberband and then expanded (negative inset) by one
    // in each direction to account for the stroke linewidth.
    [self setNeedsDisplayInRect: NSInsetRect(NSUnionRect(rect, rubberband), -1, -1)];
    rubberband = rect;
}

- (void) removeRubberband
{
    if (NSIsEmptyRect(rubberband)) { return; }
    [self setNeedsDisplayInRect: rubberband];
    rubberband = NSZeroRect;
}


@end
