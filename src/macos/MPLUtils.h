#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>
#import <CoreGraphics/CoreGraphics.h>
#import <Python.h>
#import <OSLog/OSLog.h>

NS_ASSUME_NONNULL_BEGIN

typedef NSArray<NSString *> MPLStringArray;
typedef NSDictionary<NSString *, NSString *> MPLStringDictionary;


/*
    When a method or function is NS_UNAVAILABLE, call MPLUnavailable()
    in the implementation to throw a runtime error.
*/
extern void _MPLUnavailable(const char *prettyFunction) __attribute__((__noreturn__));

#define MPLUnavailable() _MPLUnavailable(__PRETTY_FUNCTION__)


/*
    Use the macOS unified logging system for debug logs. Logs are recorded
    with almost no overhead unless a viewer is attached.
*/
extern os_log_t MPLGetLogger(void);

#define MPLLog(format, ...) os_log_debug(MPLGetLogger(), format, ##__VA_ARGS__)


/*
    Acquire the GIL, call a method with the specified arguments,
    discard the result, print any exception.
*/
extern void MPLCallMethod(
    PyObject * _Nullable pyObject,
    const char *name,
    char const * _Nullable format, ...
);


/*
    Converts a Python str into an NSString.
    Returns nil and raises a Python exception if the str could not be converted.
*/
extern NSString * _Nullable MPLGetStringWithPyString(PyObject * _Nullable string);


/*
    Converts a Python sequence of exactly one str object into an NSString.
    Returns nil and raises a Python exception if the sequence is not exactly one
    string or if the string could not be converted into an NSString.
*/
extern NSString * _Nullable MPLGetStringWithPySequence(PyObject * _Nullable pySequence);


/*
    Converts a Python sequence of str objects into an NSArray of NSString objects.
    Returns nil and raises a Python exception if 'sequence' is not a sequence,
    any item is not a string, or any item could not be converted into an NSString.
*/
extern MPLStringArray * _Nullable MPLGetStringArrayWithPySequence(
    PyObject * _Nullable pySequence
);

/*
    Converts a Python dict to an NSDictionary, keys/values must be strings.
    Returns nil and raises a Python exception if 'dict' is not a dict, any
    key/value was not a str, or any str could not be converted into an NSString.
*/
extern MPLStringDictionary * _Nullable MPLGetStringDictionaryWithPyDict(
    PyObject * _Nullable dict
);


/*
    Calls getbuffer() on a Python object and returns the buffer as an NSData.
    If expectedDimensions is non-0, verifies against ndim and fills outShape
    Returns nil and raises a Python exception if any of the following occur:
    1) getbuffer() call fails
    2) buffer->buf is NULL
    3) buffer->len is <= 0
    4) expectedDimensions is non-0 and not equal to buffer->ndim
*/
extern NSData * _Nullable MPLGetBufferWithPyObject(
    PyObject * _Nullable pyObject,
    size_t expectedDimensions,
    ssize_t * _Nullable outShape
);


/*
    Create a sRGB+alpha image of the specified width, height, and scale factor.
    (0, 0) corresponds to the upper-left corner.
*/
extern CGImageRef _Nullable MPLCreateImage(
    CGSize size,
    CGFloat scale,
    void (^callback)(CGContextRef)
);


NS_ASSUME_NONNULL_END
