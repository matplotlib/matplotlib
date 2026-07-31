#import <AppKit/AppKit.h>
#import <Python.h>

@interface Window : NSWindow
- (NSRect)constrainFrameRect:(NSRect)rect toScreen:(NSScreen*)screen;
@property (nonatomic, assign) PyObject *pyObject;
@end
