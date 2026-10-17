#import "MPLFigureManager.h"
#import "MPLFigureCanvas.h"
#import "MPLNavigationToolbar2.h"
#import "MPLUtils.h"

@interface MPLFigureManager () <NSWindowDelegate>
@end


@interface MPLUnconstrainedWindow : NSWindow
@end


@implementation MPLUnconstrainedWindow

- (NSRect) constrainFrameRect:(NSRect)rect toScreen:(NSScreen *)screen
{
    // Allow the window height to be larger than the screen height
    CGRect suggestedRect = [super constrainFrameRect:rect toScreen:screen];

    suggestedRect.origin.y -= (rect.size.height - suggestedRect.size.height);
    suggestedRect.size.height = rect.size.height;

    return suggestedRect;
}

@end


@implementation MPLFigureManager


#pragma mark - Lifecycle

- (instancetype) initWithFigureCanvas:(MPLFigureCanvas *)figureCanvas
{
    CGRect figureCanvasFrame = [figureCanvas frame];
    CGRect contentRect = figureCanvasFrame;
    contentRect.origin = CGPointMake(100, 350);

    NSWindowStyleMask styleMask = NSWindowStyleMaskTitled
                                | NSWindowStyleMaskClosable
                                | NSWindowStyleMaskResizable
                                | NSWindowStyleMaskMiniaturizable;

    NSWindow *window = [[MPLUnconstrainedWindow alloc] initWithContentRect: contentRect
                                                                 styleMask: styleMask
                                                                   backing: NSBackingStoreBuffered
                                                                     defer: YES];

    if ((self = [super initWithWindow:window])) {
        [window setDelegate:self];
        [window makeFirstResponder:figureCanvas];
        [window setReleasedWhenClosed:NO];

        // Match the window's color space to our Agg buffer.
        // This prevents an in-process color space conversion when compositing and
        // may allow for a GPU-accelerated conversion at the WindowServer level.
        [window setColorSpace:[NSColorSpace sRGBColorSpace]];

        // We want to handle the cursor changes from within MPL with set_cursor() ourselves
        [window disableCursorRects];

        [figureCanvas setAutoresizingMask:NSViewWidthSizable|NSViewHeightSizable];
        [[window contentView] addSubview:figureCanvas];

        [figureCanvas setManager:self];
        _figureCanvas = figureCanvas;

        MPLLog("[Lifecycle] MPLFigureManager<%p> init", self);
    }

    return self;
}


- (instancetype) initWithWindow:(nullable NSWindow *)window
{
    MPLUnavailable();
}


- (nullable instancetype) initWithCoder:(NSCoder *)coder
{
    MPLUnavailable();
}


- (void) dealloc
{
    MPLLog("[Lifecycle] MPLFigureManager<%p> dealloc", self);
}


#pragma mark - NSWindow Delegate Methods

- (void) windowWillClose:(NSNotification *)notification
{
    MPLCallMethod(_pyObject, "_handle_window_will_close", "");
}


- (BOOL) windowShouldClose:(NSWindow *)sender
{
    MPLCallMethod(_pyObject, "_handle_window_should_close", "");
    return YES;
}


#pragma mark - Public Methods

- (void) show
{
    [[self window] makeKeyAndOrderFront:nil];
}


- (void) raise
{
    [[self window] orderFrontRegardless];
}


- (void) toggleFullScreen
{
    [[self window] toggleFullScreen:nil];
}


- (void) resizeToDeviceWidth:(int)width height:(int)height
{
    NSWindow *window = [self window];
    CGRect rect = CGRectMake(0, 0, width, height);
    rect = [window convertRectFromBacking:rect];

    if (_toolbar) {
        rect.size.height += [_toolbar frame].size.height;
    }

    [window setContentSize:rect.size];
}


- (void) updateWindowAppearance:(nullable NSString *)windowAppearance
{
    NSAppearanceName name = windowAppearance ? [@{
        @"light": NSAppearanceNameAqua,
        @"dark":  NSAppearanceNameDarkAqua
    } objectForKey:windowAppearance] : nil;

    [[self window] setAppearance:(name ? [NSAppearance appearanceNamed:name] : nil)];
}


- (void) updateWindowMode:(nullable NSString *)windowMode
{
    NSNumber *tabbingModeValue = windowMode ? [@{
        @"system": @( NSWindowTabbingModeAutomatic  ),
        @"tab":    @( NSWindowTabbingModePreferred  ),
        @"window": @( NSWindowTabbingModeDisallowed )
    } objectForKey:windowMode] : nil;

    [[self window] setTabbingMode:[tabbingModeValue integerValue]];
}


- (void) installToolbar:(MPLNavigationToolbar2 *)toolbar
{
    if (_toolbar) return;
    _toolbar = toolbar;

    NSWindow *window = [self window];
    NSView   *canvas = [self figureCanvas];

    CGRect  windowFrame = [window frame];
    NSView *contentView = [window contentView];

    CGRect bounds = [contentView bounds];
    CGRect canvasFrame  = [canvas frame];
    CGRect toolbarFrame = [toolbar frame];

    // Expand window downwards
    windowFrame.origin.y -= toolbarFrame.size.height;
    windowFrame.size.height += toolbarFrame.size.height;
    [window setFrame:windowFrame display:NO];

    // Move canvas upwards
    canvasFrame.origin.y = toolbarFrame.size.height;
    [canvas setFrame:canvasFrame];

    // Adjust toolbar width and place at origin of window
    toolbarFrame.origin = CGPointZero;
    toolbarFrame.size.width = bounds.size.width;
    [toolbar setAutoresizingMask:NSViewMaxYMargin|NSViewWidthSizable];
    [toolbar setFrame:toolbarFrame];

    [contentView addSubview:toolbar];
}


#pragma mark - Accessors

- (void) setWindowTitle:(NSString *)title
{
    [[self window] setTitle:title];
}


- (NSString *) windowTitle
{
    return [[self window] title];
}


@end
