#import <Foundation/Foundation.h>
#import <Python.h>

void gil_call_method(PyObject* obj, const char* name);

void process_event(char const* cls_name, char const* fmt, ...);
