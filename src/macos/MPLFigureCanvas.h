#import <AppKit/AppKit.h>
#import <Python.h>

NS_ASSUME_NONNULL_BEGIN

@class MPLFigureManager;

@interface MPLFigureCanvas : NSView

- (instancetype) initWithFrame:(NSRect)rect;

- (void) updateLayerContentsWithDataProvider: (CGDataProviderRef) provider
                                 deviceWidth: (size_t) deviceWidth
                                deviceHeight: (size_t) deviceHeight;

- (void) updateCursorType:(int)cursorType;
- (void) updateRubberbandWithDeviceX0:(int)x0 y0:(int)y0 x1:(int)x1 y1:(int)y1;
- (void) removeRubberband;
- (void) requestDisplayLayerWithNeedsDraw:(BOOL)needsDraw; // Thread-safe

@property (nonatomic, assign, nullable) PyObject *pyObject;

@property (nonatomic, weak, nullable) MPLFigureManager *manager;

@end

NS_ASSUME_NONNULL_END
