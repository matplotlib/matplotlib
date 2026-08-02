#import "MPLUtils.h"


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

void MPLCallMethod(PyObject *pyObject, const char *name, char const *format, ...)
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


NSString *MPLGetStringWithPyString(PyObject *pyString)
{
    if (!pyString) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "Input is NULL");
        return nil;
    }

    if (!PyUnicode_Check(pyString)) {
        PyErr_SetString(PyExc_RuntimeError, "Input is not a string");
        return nil;
    }

    const char *cString = PyUnicode_AsUTF8(pyString);
    if (!cString) {
        // PyUnicode_AsUTF8() should set error in this case
        return nil;
    }

    NSString *result = [NSString stringWithUTF8String:cString];
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Could not create NSString");
    }

    return result;
}


MPLStringArray *MPLGetStringArrayWithPySequence(PyObject *pySequence)
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


NSString *MPLGetStringWithPySequence(PyObject *pySequence)
{
    MPLStringArray *array = MPLGetStringArrayWithPySequence(pySequence);

    if (array && ([array count] != 1)) {
        PyErr_SetString(PyExc_RuntimeError, "Input is not a sequence of exactly one string");
        return nil;
    }

    return [array lastObject];
}


MPLStringDictionary *MPLGetStringDictionaryWithPyDict(PyObject *dict)
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
    PyObject * _Nullable pyObject,
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
