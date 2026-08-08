#import "MPLFigureCanvas.h"
#import "MPLUtils.h"
#import <QuartzCore/CAShapeLayer.h>

@interface MPLFigureCanvas () <CALayerDelegate>
@end


@implementation MPLFigureCanvas {
    BOOL _isLeftMouseDown;
    BOOL _isHandCursorActive;
    NSEventModifierFlags _previousModifierFlags;
    CALayer *_canvasLayer;
    CALayer *_rubberbandLayer;
    BOOL _needsDrawOnNextDisplayLayer;
}

- (instancetype) initWithFrame:(NSRect)rect
{
    if (self = [super initWithFrame: rect]) {
        NSTrackingAreaOptions options = (
            NSTrackingMouseEnteredAndExited | NSTrackingMouseMoved |
            NSTrackingActiveInKeyWindow | NSTrackingInVisibleRect
        );

        CALayer *canvasLayer = [CALayer layer];
        [canvasLayer setDelegate:self];
        [canvasLayer setContentsGravity:kCAGravityResize];
        [canvasLayer setBackgroundColor:[[NSColor whiteColor] CGColor]];
        [canvasLayer setOpaque:YES];

        [self setLayer:canvasLayer];
        [self setWantsLayer:YES];
        
        CALayer *rubberbandLayer = [CALayer layer];
        [rubberbandLayer setDelegate:self];
        [rubberbandLayer setNeedsDisplayOnBoundsChange:YES];
        
        _canvasLayer = canvasLayer;
        _rubberbandLayer = rubberbandLayer;

        [self addTrackingArea:[[NSTrackingArea alloc] initWithRect: CGRectZero
                                                           options: options
                                                             owner: self
                                                          userInfo: nil]];

        MPLLog("[Lifecycle] MPLFigureCanvas<%p> init", self);
    }

    return self;
}


- (void) dealloc
{
    MPLLog("[Lifecycle] MPLFigureCanvas<%p> dealloc", self);
}


#pragma mark - Superclass Overrides

- (void) viewDidChangeBackingProperties
{
    CGFloat scaleFactor = [[self window] backingScaleFactor];
    if (!scaleFactor) scaleFactor = 1;

    [_canvasLayer setContentsScale:scaleFactor];
    [_rubberbandLayer setContentsScale:scaleFactor];

    int width, height;
    [self _getDeviceSizeWithSize:[self frame].size width:&width height:&height];

    MPLCallMethod(_pyObject, "_handle_view_did_change_backing_properties", "dii",
        scaleFactor, width, height
    );
}


- (void) setFrameSize:(NSSize)newSize
{
    CGSize oldSize = [self frame].size;
    [super setFrameSize:newSize];

    if (!CGSizeEqualToSize(oldSize, newSize)) {
        [self _callHandleResize];
    }
}


- (BOOL) acceptsFirstResponder
{
    return YES;
}


#pragma mark - CALayerDelegate

- (void) displayLayer:(CALayer *)layer
{
    if (layer == _canvasLayer) {
        int needsDraw = _needsDrawOnNextDisplayLayer ? 1 : 0;
        MPLCallMethod(_pyObject, "_handle_display_layer", "i", needsDraw);

    } else if (layer == _rubberbandLayer) {
        [self _displayRubberbandLayer];
    }
}


- (nullable id<CAAction>) actionForLayer:(CALayer *)layer forKey:(NSString *)event
{
    return [NSNull null];
}


#pragma mark - Private Methods

- (void) _getDevicePointWithWindowLocation:(CGPoint)windowLocation x:(int *)outX y:(int *)outY
{
    CGPoint viewLocation = [self convertPoint:windowLocation fromView:nil];
    viewLocation = [self convertPointToBacking:viewLocation];

    *outX = viewLocation.x;
    *outY = viewLocation.y;
}


- (void) _getDevicePointWithEvent:(NSEvent *)event x:(int *)outX y:(int *)outY
{
    NSPoint windowLocation = [event locationInWindow];

    [self _getDevicePointWithWindowLocation:windowLocation x:outX y:outY];
}


- (void) _getDeviceSizeWithSize:(CGSize)size width:(int *)outWidth height:(int *)outHeight
{
    CGSize deviceSize = [self convertSizeToBacking:size];

    *outWidth  = (int)deviceSize.width;
    *outHeight = (int)deviceSize.height;
}


- (void) _callHandleResize
{
    int width, height;
    [self _getDeviceSizeWithSize:[self frame].size width:&width height:&height];

    MPLCallMethod(_pyObject, "_handle_resize", "ii", width, height);
}


- (void) _updateHandCursor
{
    if (_isHandCursorActive) {
        [(_isLeftMouseDown ? [NSCursor closedHandCursor] : [NSCursor openHandCursor]) set];
    }
}


- (void) _updateRubberbandLayerWithFrame:(CGRect)frame
{
    if (CGRectIsEmpty(frame)) {
        [_rubberbandLayer removeFromSuperlayer];
        [_rubberbandLayer setContents:nil];

    } else {
        if (![_rubberbandLayer superlayer]) {
            [_canvasLayer addSublayer:_rubberbandLayer];
        }

        [_rubberbandLayer setFrame:frame];
    }
}


- (void) _displayRubberbandLayer
{
    CGRect bounds = [_rubberbandLayer bounds];
    CGFloat contentsScale = [_rubberbandLayer contentsScale];

    CGImageRef contents = MPLCreateImage(bounds.size, contentsScale, ^(CGContextRef context) {
        CGRect strokeRect = CGRectInset(bounds, 0.5, 0.5);
        CGFloat dashPattern[2] = { 3.0, 3.0 };
        
        CGContextSetGrayStrokeColor(context, 1.0, 1.0);
        CGContextSetLineDash(context, 0.0, dashPattern, 2);
        CGContextStrokeRect(context, strokeRect);

        CGContextSetGrayStrokeColor(context, 0.0, 1.0);
        CGContextSetLineDash(context, 3.0, dashPattern, 2);
        CGContextStrokeRect(context, strokeRect);
    });

    [_rubberbandLayer setContents:(__bridge id)contents];

    CGImageRelease(contents);
}


#pragma mark - Keyboard Events

- (NSString *) _mappedStringWithCharacters:(NSString *)characters
{
    static NSDictionary *sKeyMap = nil;
    
    if (!sKeyMap) sKeyMap = @{
        @( NSLeftArrowFunctionKey  ): @"left",         @( NSRightArrowFunctionKey ): @"right",
        @( NSUpArrowFunctionKey    ): @"up",           @( NSDownArrowFunctionKey  ): @"down",
        @( NSF1FunctionKey         ): @"f1",           @( NSF2FunctionKey         ): @"f2",
        @( NSF3FunctionKey         ): @"f3",           @( NSF4FunctionKey         ): @"f4",
        @( NSF5FunctionKey         ): @"f5",           @( NSF6FunctionKey         ): @"f6",
        @( NSF7FunctionKey         ): @"f7",           @( NSF8FunctionKey         ): @"f8",
        @( NSF9FunctionKey         ): @"f9",           @( NSF10FunctionKey        ): @"f10",
        @( NSF11FunctionKey        ): @"f11",          @( NSF12FunctionKey        ): @"f12",
        @( NSF13FunctionKey        ): @"f13",          @( NSF14FunctionKey        ): @"f14",
        @( NSF15FunctionKey        ): @"f15",          @( NSF16FunctionKey        ): @"f16",
        @( NSF17FunctionKey        ): @"f17",          @( NSF18FunctionKey        ): @"f18",
        @( NSF19FunctionKey        ): @"f19",          @( NSF20FunctionKey        ): @"f20",
        @( NSScrollLockFunctionKey ): @"scroll_lock",  @( NSBreakFunctionKey      ): @"break",
        @( NSInsertFunctionKey     ): @"insert",       @( NSDeleteFunctionKey     ): @"delete",
        @( NSHomeFunctionKey       ): @"home",         @( NSEndFunctionKey        ): @"end",
        @( NSPageDownFunctionKey   ): @"pagedown",     @( NSPageUpFunctionKey     ): @"pageup",
        @( NSDeleteCharacter       ): @"backspace",    @( NSBackTabCharacter      ): @"backtab",
        @( NSEnterCharacter        ): @"enter",        @( NSTabCharacter          ): @"tab",
        @( NSCarriageReturnCharacter ): @"enter",
        @( 27 ): @"escape" // No AppKit constant for Escape
    };

    return ([characters length] > 0) ?
        [sKeyMap objectForKey:@( [characters characterAtIndex:0] )] :
        nil;
}


- (NSString *) _keyStringWithString: (nullable NSString *) characters
                      modifierFlags: (NSEventModifierFlags) flags
                      controlString: (NSString *) controlString
{
    NSMutableArray *array = [NSMutableArray array];

    if (flags & NSEventModifierFlagControl  ) [array addObject:controlString];
    if (flags & NSEventModifierFlagOption   ) [array addObject:@"alt"];
    if (flags & NSEventModifierFlagCommand  ) [array addObject:@"cmd"];
    if (flags & NSEventModifierFlagCapsLock ) [array addObject:@"caps_lock"];
    if (flags & NSEventModifierFlagShift    ) [array addObject:@"shift"];

    if (characters) [array addObject:characters];

    return [array componentsJoinedByString:@"+"];
}


- (void) _callHandleKeyWithKeyString:(NSString *)keyString isPress:(BOOL)isPress
{
    NSPoint windowLocation = [[self window] mouseLocationOutsideOfEventStream];

    int x, y;
    [self _getDevicePointWithWindowLocation:windowLocation x:&x y:&y];

    const char *keyCString = [keyString UTF8String];
    if (!keyCString) return;

    MPLCallMethod(_pyObject, "_handle_key", "isii", (int)isPress, keyCString, x, y);
}


- (void) _handleKeyDownOrUp:(NSEvent *)event isPress:(BOOL)isPress
{
    NSEventModifierFlags flags = [event modifierFlags];
    NSString *characters = [event charactersIgnoringModifiers];
    NSString *mappedString = [self _mappedStringWithCharacters:characters];
    NSString *stringToUse;

    // -charactersIgnoringModifiers doesn't "ignore" the shift modifier so
    // strip it from flags before calling -_keyString...
    if (!mappedString) {
        stringToUse = characters;
        flags = flags & ~NSEventModifierFlagShift;
    } else {
        stringToUse = mappedString;
    }

    NSString *keyString = [self _keyStringWithString: stringToUse
                                       modifierFlags: flags
                                       controlString: @"ctrl"];

    [self _callHandleKeyWithKeyString:keyString isPress:isPress];
}


- (void) keyDown:(NSEvent *)event
{
    [self _handleKeyDownOrUp:event isPress:YES];
}


- (void) keyUp:(NSEvent *)event
{
    [self _handleKeyDownOrUp:event isPress:NO];
}


- (void) flagsChanged:(NSEvent *)event
{
    NSEventModifierFlags currentFlags = [event modifierFlags] & (
        NSEventModifierFlagControl  | NSEventModifierFlagOption | NSEventModifierFlagCommand |
        NSEventModifierFlagCapsLock | NSEventModifierFlagShift
    );

    if (currentFlags == _previousModifierFlags) return;

    NSString *keyString = [self _keyStringWithString: nil
                                       modifierFlags: (currentFlags | _previousModifierFlags)
                                       controlString: @"control"];

    BOOL isPress = currentFlags > _previousModifierFlags;
    [self _callHandleKeyWithKeyString:keyString isPress:isPress];

    _previousModifierFlags = currentFlags;
}


#pragma mark - Mouse Events

- (void) _handleMouseDownOrUp:(NSEvent *)event isPress:(BOOL)isPress
{
    NSInteger buttonNumber = [event buttonNumber];
    NSEventModifierFlags modifierFlags = [event modifierFlags];

    int x, y;
    [self _getDevicePointWithEvent:event x:&x y:&y];

    if ([event type] == NSEventTypeLeftMouseDown) {
        if (modifierFlags & NSEventModifierFlagControl) {
             // emulate a right-button click
             buttonNumber = 1;

        } else if (modifierFlags & NSEventModifierFlagOption) {
             // emulate a middle-button click
             buttonNumber = 2;
        }

        _isLeftMouseDown = YES;

        [self _updateHandCursor];

    } else if ([event type] == NSEventTypeLeftMouseUp) {
        _isLeftMouseDown = NO;

        [self _updateHandCursor];
    }

    if (isPress) {
        MPLCallMethod(_pyObject, "_handle_mouse_down",
            "iilki", x, y, buttonNumber, modifierFlags,
            (int)([event clickCount] == 2 ? 1 : 0)
        );
    } else {
        MPLCallMethod(_pyObject, "_handle_mouse_up",
            "iilk", x, y, buttonNumber, modifierFlags
        );
    }
}


- (void) mouseEntered:(NSEvent *)event
{
    int x, y;
    [self _getDevicePointWithEvent:event x:&x y:&y];

    MPLCallMethod(_pyObject, "_handle_mouse_entered", "iik", x, y, [event modifierFlags]);
}


- (void) mouseExited:(NSEvent *)event
{
    int x, y;
    [self _getDevicePointWithEvent:event x:&x y:&y];

    MPLCallMethod(_pyObject, "_handle_mouse_exited", "iik", x, y, [event modifierFlags]);
}


- (void) mouseMoved:(NSEvent *)event
{
    int x, y;
    [self _getDevicePointWithEvent:event x:&x y:&y];

    MPLCallMethod(_pyObject, "_handle_mouse_moved",
        "iikk", x, y, [NSEvent pressedMouseButtons], [event modifierFlags]
    );
}


- (void) scrollWheel:(NSEvent *)event
{
    float deltaY = [event deltaY];
    int step = (deltaY > 0) - (deltaY < 0); // step = -1, 0, or 1
    if (step == 0) return;

    int x, y;
    [self _getDevicePointWithEvent:event x:&x y:&y];

    MPLCallMethod(_pyObject, "_handle_scroll_wheel",
        "iiik", x, y, step, [event modifierFlags]
    );
}


// Funnel all button events to -_handleMouseDownOrUp:isPress:
- (void) mouseDown:     (NSEvent *)event { [self _handleMouseDownOrUp:event isPress:YES]; }
- (void) rightMouseDown:(NSEvent *)event { [self _handleMouseDownOrUp:event isPress:YES]; }
- (void) otherMouseDown:(NSEvent *)event { [self _handleMouseDownOrUp:event isPress:YES]; }
- (void) mouseUp:       (NSEvent *)event { [self _handleMouseDownOrUp:event isPress:NO]; }
- (void) rightMouseUp:  (NSEvent *)event { [self _handleMouseDownOrUp:event isPress:NO]; }
- (void) otherMouseUp:  (NSEvent *)event { [self _handleMouseDownOrUp:event isPress:NO]; }


// Funnel dragged events to mouseMoved:
- (void) mouseDragged:     (NSEvent *)event { [self mouseMoved:event]; }
- (void) rightMouseDragged:(NSEvent *)event { [self mouseMoved:event]; }
- (void) otherMouseDragged:(NSEvent *)event { [self mouseMoved:event]; }


#pragma mark - Public Methods

- (void) updateLayerContentsWithDataProvider: (CGDataProviderRef) provider
                                 deviceWidth: (size_t) deviceWidth
                                deviceHeight: (size_t) deviceHeight
{
    CGColorSpaceRef colorSpace = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGBitmapInfo bitmapInfo = 0 | kCGBitmapByteOrderDefault | kCGImageAlphaLast;

    CGImageRef contents = colorSpace ? CGImageCreate(
        deviceWidth, deviceHeight, 8, 32, deviceWidth * 4,
        colorSpace, bitmapInfo, provider, NULL, false,
        kCGRenderingIntentDefault
    ) : NULL;

    if (contents) {
        [_canvasLayer setContents:(__bridge id)contents];
    }

    CGColorSpaceRelease(colorSpace);
    CGImageRelease(contents);
}


- (void) updateCursorType:(int)cursorType
{
    _isHandCursorActive = (cursorType == 4);

    if (_isHandCursorActive) {
        [self _updateHandCursor];

    } else {
        NSCursor *cursor;

        if      (cursorType == 1) cursor = [NSCursor arrowCursor];
        else if (cursorType == 2) cursor = [NSCursor pointingHandCursor];
        else if (cursorType == 3) cursor = [NSCursor crosshairCursor];
        // 4 is Cursors.MOVE - handled above
        // 5 is Cursors.WAIT - macOS automatically shows the busy cursor as needed
        else if (cursorType == 6) cursor = [NSCursor resizeLeftRightCursor];
        else if (cursorType == 7) cursor = [NSCursor resizeUpDownCursor];

        [cursor set];
    }
}


- (void) updateRubberbandWithDeviceX0:(int)x0 y0:(int)y0 x1:(int)x1 y1:(int)y1
{
    CGRect rect = CGRectStandardize(CGRectMake(x0, y0, x1 - x0, y1 - y0));
    CGRect rubberbandFrame = [self convertRectFromBacking:rect];
    [self _updateRubberbandLayerWithFrame:rubberbandFrame];
}


- (void) removeRubberband
{
    [self _updateRubberbandLayerWithFrame:CGRectZero];
}


- (void) requestDisplayLayerWithNeedsDraw:(BOOL)needsDraw
{
    if ([NSThread isMainThread]) {
        _needsDrawOnNextDisplayLayer = needsDraw;
        [_canvasLayer setNeedsDisplay];

    } else {
        __weak id weakSelf = self;

        dispatch_async(dispatch_get_main_queue(), ^{
            [weakSelf requestDisplayLayerWithNeedsDraw:needsDraw];
        });
    }
}

@end
