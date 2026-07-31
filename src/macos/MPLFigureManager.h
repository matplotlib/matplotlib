#import <AppKit/AppKit.h>
#import <Python.h>
#import "MPLUtils.h"

NS_ASSUME_NONNULL_BEGIN

@class MPLFigureCanvas, MPLNavigationToolbar2;

@interface MPLFigureManager : NSWindowController

- (instancetype) initWithFigureCanvas:(MPLFigureCanvas *)figureCanvas NS_DESIGNATED_INITIALIZER;

- (instancetype) initWithWindow:(nullable NSWindow *)window NS_UNAVAILABLE;
- (nullable instancetype) initWithCoder:(NSCoder *)coder NS_UNAVAILABLE;

- (void) show;
- (void) raise;
- (void) toggleFullScreen;
- (void) resizeToDeviceWidth:(int)width height:(int)height;
- (void) updateWindowAppearance:(nullable NSString *)windowAppearance;
- (void) updateWindowMode:(nullable NSString *)windowMode;
- (void) installToolbar:(MPLNavigationToolbar2 *)toolbar;

@property (nonatomic, assign, nullable) PyObject *pyObject;

@property (nonatomic) NSString *windowTitle;

@property (nonatomic, readonly) MPLFigureCanvas *figureCanvas;
@property (nonatomic, nullable, readonly) MPLNavigationToolbar2 *toolbar;

@end

NS_ASSUME_NONNULL_END
