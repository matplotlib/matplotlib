#import "MPLUtils.h"

#define PY_SSIZE_T_CLEAN
#import "Python.h"


void _MPLUnavailable(const char *s)
{
    [NSException raise: NSInvalidArgumentException
                format: @"'%s' called but marked with __attribute__((unavailable))", s];

    __builtin_unreachable();
}


os_log_t MPLGetLogger(void)
{
    static os_log_t sLogger = nil;

    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sLogger = os_log_create("org.matplotlib", "org.matplotlib");
    });

    return sLogger;
}


#pragma mark - Python Utility Functions

void MPLCallMethod(MPLPyObjectRef pyObject, const char *name, char const *format, ...)
{
    // It is possible for Obj-C objects to momentarily outlive their paired Python
    // counterparts, especially when dealing with AppKit objects. Hence, allow
    // messaging a NULL pyObject to be a no-op.
    if (!pyObject) return;

    PyGILState_STATE gilState = PyGILState_Ensure();

    PyObject *result = NULL;

    // Null or empty string, simply use PyObject_CallMethod()
    if (!format || !format[0]) {
        result = PyObject_CallMethod(pyObject, name, NULL);

    } else {
        va_list va;
        va_start(va, format);

        PyObject *args = Py_VaBuildValue(format, va);
        PyObject *method = PyObject_GetAttrString(pyObject, name);

        // "Py_BuildValue() does not always build a tuple."
        if (args && !PyTuple_Check(args)) {
            PyObject *tuple = PyTuple_Pack(1, args);
            Py_DECREF(args);
            args = tuple;
        }

        if (method && args) {
            result = PyObject_Call(method, args, NULL);
        }

        Py_XDECREF(method);
        Py_XDECREF(args);

        va_end(va);
    }

    if (result) {
        Py_DECREF(result);
    } else {
        PyErr_Print();
    }

    PyGILState_Release(gilState);
}


NSString *MPLGetStringWithPyString(MPLPyObjectRef pyString)
{
    if (!pyString) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "Input is NULL");
        return nil;
    }

    if (!PyUnicode_Check(pyString)) {
        PyErr_SetString(PyExc_RuntimeError, "Input is not a string");
        return nil;
    }

    Py_ssize_t size = 0;
    const char *cString = PyUnicode_AsUTF8AndSize(pyString, &size);

    if (!cString) {
        // PyUnicode_AsUTF8AndSize() should set error in this case
        return nil;
    }

    // This should never happen, but we are about to cast from signed to unsigned
    if (size < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Size is less than 0");
        return nil;
    }

    NSUInteger length = (NSUInteger)size;
    NSString *result = [[NSString alloc] initWithBytes:cString length:length encoding:NSUTF8StringEncoding];

    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Could not create NSString");
    }

    return result;
}


MPLStringArray *MPLGetStringArrayWithPySequence(MPLPyObjectRef pySequence)
{
    if (!pySequence) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "Input is NULL");
        return nil;
    }

    if (!PySequence_Check(pySequence)) {
        PyErr_SetString(PyExc_RuntimeError, "Input is not a sequence");
        return nil;
    }

    Py_ssize_t size = PySequence_Size(pySequence);
    if (size < 0) {
        // PySequence_Size() should set error in this case
        return nil;
    }

    NSMutableArray *result = [NSMutableArray arrayWithCapacity:(NSUInteger)size];

    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject *pyItem = PySequence_GetItem(pySequence, i);  // New reference
        NSString *string = MPLGetStringWithPyString(pyItem);
        Py_DECREF(pyItem);

        if (string) {
            [result addObject:string];
        } else {
            return nil;
        }
    }

    return [result copy];
}


NSString *MPLGetStringWithPySequence(MPLPyObjectRef _Nullable pySequence)
{
    MPLStringArray *array = MPLGetStringArrayWithPySequence(pySequence);

    if (array && ([array count] != 1)) {
        PyErr_SetString(PyExc_RuntimeError, "Input is not a sequence of exactly one string");
        return nil;
    }

    return [array lastObject];
}


MPLStringDictionary *MPLGetStringDictionaryWithPyDict(MPLPyObjectRef dict)
{
    if (!dict) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "Input is NULL");
        return nil;
    }

    if (!PyDict_Check(dict)) {
        PyErr_SetString(PyExc_RuntimeError, "Input is not a dict");
        return nil;
    }

    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    PyObject *pyKey = NULL;
    PyObject *pyValue = NULL;
    Py_ssize_t position = 0;

    while (PyDict_Next(dict, &position, &pyKey, &pyValue)) {
        NSString *key = MPLGetStringWithPyString(pyKey);
        NSString *value = MPLGetStringWithPyString(pyValue);
        if (!key || !value) return nil;

        [result setObject:value forKey:key];
    }

    return [result copy];
}


NSData * _Nullable MPLGetBufferWithPyObject(
    MPLPyObjectRef _Nullable pyObject,
    size_t expectedDimensions,
    ssize_t * _Nullable outShape
) {
    if (!pyObject) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "Input is NULL");
        return nil;
    }

    Py_buffer *buffer = malloc(sizeof(Py_buffer));

    if (PyObject_GetBuffer(pyObject, buffer, PyBUF_CONTIG_RO) == -1) {
        free(buffer);
        return nil;
    }

    void (^deallocator)(void *, NSUInteger) = ^(void *unused1, NSUInteger unused2) {
        PyGILState_STATE gstate = PyGILState_Ensure();
        PyBuffer_Release((Py_buffer *)buffer);
        free(buffer);
        PyGILState_Release(gstate);
    };

    if (!buffer->buf || buffer->len <= 0) {
        PyErr_SetString(PyExc_RuntimeError, "Buffer is invalid");
        deallocator(NULL, 0);
        return nil;
    }

    if (expectedDimensions && (expectedDimensions != buffer->ndim)) {
        PyErr_SetString(PyExc_RuntimeError, "Unexpected buffer dimensions");
        deallocator(NULL, 0);
        return nil;
    }

    if (expectedDimensions && outShape) {
        memcpy(outShape, buffer->shape, sizeof(ssize_t) * expectedDimensions);
    }

    return [[NSData alloc] initWithBytesNoCopy: buffer->buf
                                        length: buffer->len
                                   deallocator: deallocator];
}


#pragma mark - Graphics Utility Functions

CGRect MPLGetCenteredRect(CGRect bounds, CGSize size)
{
    return CGRectMake(
        bounds.origin.x + ((bounds.size.width  - size.width)  / 2.0),
        bounds.origin.y + ((bounds.size.height - size.height) / 2.0),
        size.width,
        size.height
    );
}


NSColor *MPLGetRGBColor(int rgb, CGFloat alpha)
{
    float r = (((rgb & 0xFF0000) >> 16) / 255.0);
    float g = (((rgb & 0x00FF00) >>  8) / 255.0);
    float b = (((rgb & 0x0000FF) >>  0) / 255.0);

    return [NSColor colorWithSRGBRed:r green:g blue:b alpha:alpha];
}


MPLViewAppearance MPLGetViewAppearance(NSView *view)
{
    NSString *bestMatch = [[view effectiveAppearance] bestMatchFromAppearancesWithNames:@[
        NSAppearanceNameAqua,
        NSAppearanceNameDarkAqua,
        NSAppearanceNameAccessibilityHighContrastAqua,
        NSAppearanceNameAccessibilityHighContrastDarkAqua
    ]];

    if ([bestMatch isEqualToString:NSAppearanceNameDarkAqua]) {
        return MPLViewAppearanceDark;

    } else if ([bestMatch isEqualToString:NSAppearanceNameAccessibilityHighContrastAqua]) {
        return MPLViewAppearanceHighContrastLight;

    } else if ([bestMatch isEqualToString:NSAppearanceNameAccessibilityHighContrastDarkAqua]) {
        return MPLViewAppearanceHighContrastDark;
    }

    return MPLViewAppearanceLight;
}


void MPLAddContinuousRoundedRect(
    CGContextRef context,
    CGRect rect,
    CGFloat tl, // topLeftRadius
    CGFloat tr, // topRightRadius
    CGFloat bl, // bottomLeftRadius
    CGFloat br  // bottomRightRadius
) {
    /*
        These constants are widely-circulated in the macOS/iOS developer community.
        They were reversed-engineered from Apple's own squircle implementation when
        iOS 7 originally shipped back in 2013.
    */
    const CGFloat mA = 1.528665;
    const CGFloat mB = 1.08849;   // Actual AppKit value. Typically '1.088493' in blog posts.
    const CGFloat mC = 0.868407;
    const CGFloat mD = 0.631494;
    const CGFloat mE = 0.0749114; // Actual AppKit value. Typically '0.074911' in blog posts.
    const CGFloat mF = 0.372824;
    const CGFloat mG = 0.169060;

    __auto_type corner = ^(
        CGFloat cx,  CGFloat cy,
        CGFloat dx0, CGFloat dy0, CGFloat dx1, CGFloat dy1
    ) {
        CGContextAddLineToPoint(context, cx + dx0 * mA, cy + dy0 * mA);

        CGContextAddCurveToPoint(context,
            cx + dx0 * mB,              cy + dy0 * mB,
            cx + dx0 * mC,              cy + dy0 * mC,
            cx + (dx0 * mD + dx1 * mE), cy + (dy0 * mD + dy1 * mE)
        );

        CGContextAddCurveToPoint(context,
            cx + (dx0 * mF + dx1 * mG),  cy + (dy0 * mF + dy1 * mG),
            cx + (dx0 * mG + dx1 * mF),  cy + (dy0 * mG + dy1 * mF),
            cx + (dx0 * mE + dx1 * mD),  cy + (dy0 * mE + dy1 * mD)
        );

        CGContextAddCurveToPoint(context,
            cx + dx1 * mC,  cy + dy1 * mC,
            cx + dx1 * mB,  cy + dy1 * mB,
            cx + dx1 * mA,  cy + dy1 * mA
        );
    };

    CGContextMoveToPoint(context, CGRectGetMinX(rect) + mA * tl, CGRectGetMinY(rect));

    corner( CGRectGetMaxX(rect), CGRectGetMinY(rect), -tr,  0,   0,   tr );
    corner( CGRectGetMaxX(rect), CGRectGetMaxY(rect),  0,  -br, -br,  0  );
    corner( CGRectGetMinX(rect), CGRectGetMaxY(rect),  bl,  0,   0,  -bl );
    corner( CGRectGetMinX(rect), CGRectGetMinY(rect),  0,   tl,  tl,  0  );

    CGContextClosePath(context);
}


static CGImageRef _Nullable sCreateImage(
    CGSize size, CGFloat scale, BOOL flipped,
    CFStringRef colorSpaceName, size_t componentCount, CGBitmapInfo bitmapInfo,
    void (^callback)(CGContextRef)
) {
    if (size.width <= 0 || size.height <= 0) return NULL;

    CGColorSpaceRef colorSpace = CGColorSpaceCreateWithName(colorSpaceName);
    if (!colorSpace) return NULL;

    size_t pixelsWide = size.width  * scale;
    size_t pixelsHigh = size.height * scale;

    CGContextRef context = CGBitmapContextCreate(
        NULL, pixelsWide, pixelsHigh,
        8, pixelsWide * componentCount, colorSpace,
        bitmapInfo
    );

    CGImageRef result = NULL;

    if (context) {
        if (flipped) {
            CGContextTranslateCTM(context, 0, pixelsHigh);
            CGContextScaleCTM(context, scale, -scale);
        }

        NSGraphicsContext *savedContext = [NSGraphicsContext currentContext];
        [NSGraphicsContext setCurrentContext:[NSGraphicsContext graphicsContextWithCGContext:context flipped:flipped]];

        callback(context);

        [NSGraphicsContext setCurrentContext:savedContext];

        result = CGBitmapContextCreateImage(context);
        CFRelease(context);
    }

    CGColorSpaceRelease(colorSpace);

    return result;
}


CGImageRef MPLCreateImage(CGSize size, CGFloat scale, void (^callback)(CGContextRef))
{
    CGBitmapInfo bitmapInfo = 0 | kCGImageAlphaPremultipliedFirst | kCGImageByteOrder32Little;
    return sCreateImage(size, scale, YES, kCGColorSpaceSRGB, 4, bitmapInfo, callback);
}
