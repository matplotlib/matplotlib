#import "MPLNavigationToolbar2.h"
#import "MPLUtils.h"

static const CGFloat sToolbarHeight = 36;
static const CGFloat sButtonHeight = 32;
static const CGFloat sButtonGap = 2;

@interface MPLNavigationToolbar2 ()
@end


@implementation MPLNavigationToolbar2 {
    NSMutableArray *_buttons;
    NSMutableArray *_callbackNames; // Same count and indices as _buttons

    NSView *_buttonContainer;
    NSTextField *_messageField;
    CGFloat _nextButtonX;
}


#pragma mark - Lifecycle

- (instancetype) init
{
    if ((self = [super initWithFrame:CGRectMake(0, 0, 400, sToolbarHeight)])) {
        _buttons = [NSMutableArray array];
        _callbackNames = [NSMutableArray array];

        CGFloat buttonYOrigin  = floor((sToolbarHeight - sButtonHeight) / 2.0);
        CGRect  containerFrame = CGRectMake(sButtonGap, buttonYOrigin, 200, sButtonHeight);

        _buttonContainer = [[NSView alloc] initWithFrame:containerFrame];
        [self addSubview:_buttonContainer];

        [self _addMessageField];

        MPLLog("[Lifecycle] MPLNavigationToolbar2<%p> init", self);
    }

    return self;
}


- (instancetype) initWithFrame:(NSRect)frame
{
    MPLUnavailable();
}


- (nullable instancetype) initWithCoder:(NSCoder *)coder
{
    MPLUnavailable();
}


- (void) dealloc
{
    MPLLog("[Lifecycle] MPLNavigationToolbar2<%p> dealloc", self);
}


#pragma mark - Private Methods

- (void) _addMessageField
{
    NSFont *font = [NSFont monospacedDigitSystemFontOfSize:13.0 weight:NSFontWeightRegular];

    NSTextField *messageField = [[NSTextField alloc] initWithFrame:CGRectZero];

    [messageField setTranslatesAutoresizingMaskIntoConstraints:NO];
    [messageField setAlignment:NSTextAlignmentRight];
    [messageField setFont:font];
    [messageField setDrawsBackground:NO];
    [messageField setBordered:NO];
    [messageField setBezeled:NO];
    [messageField setSelectable:NO];
    [messageField setEditable:NO];
    [messageField setLineBreakMode:NSLineBreakByTruncatingHead];

    [messageField setContentCompressionResistancePriority: NSLayoutPriorityDefaultLow
                                           forOrientation: NSLayoutConstraintOrientationHorizontal];

    [self addSubview:messageField];

    [NSLayoutConstraint activateConstraints:@[
        [[messageField leftAnchor]    constraintEqualToAnchor:[_buttonContainer rightAnchor]],
        [[messageField rightAnchor]   constraintEqualToAnchor:[self rightAnchor] constant:-sButtonGap],
        [[messageField centerYAnchor] constraintEqualToAnchor:[self centerYAnchor]]
    ]];

    _messageField = messageField;
}


- (nullable NSButton *) _buttonWithCallbackName:(NSString *)callbackName
{
    NSUInteger index = [_callbackNames indexOfObject:callbackName];

    return (index != NSNotFound) ? [_buttons objectAtIndex:index] : nil;
}


- (void) _callMethodForButton:(id)sender
{
    NSUInteger index = [_buttons indexOfObject:sender];

    if (index != NSNotFound) {
        const char *callbackName = [[_callbackNames objectAtIndex:index] UTF8String];
        if (callbackName) MPLCallMethod(_pyObject, callbackName, "");
    }
}


#pragma mark - Public Methods

- (void) addItemWithTitle: (NSString *) title
                  tooltip: (NSString *) tooltip
                imagePath: (NSString *) imagePath
             callbackName: (NSString *) callbackName
{
    NSImage *image = [[NSImage alloc] initWithContentsOfFile:imagePath];
    [image setSize:CGSizeMake(24, 24)];
    [image setTemplate:YES];

    CGRect frame = CGRectMake(_nextButtonX, 0, sButtonHeight, sButtonHeight);

    NSButton *button = [[NSButton alloc] initWithFrame:frame];
    [button setBezelStyle:NSBezelStyleSmallSquare];
    [button setButtonType:NSButtonTypeMomentaryLight];
    [button setImage:image];
    [button setImagePosition:NSImageOnly];
    [button setImageScaling:NSImageScaleProportionallyDown];
    [button setAutoresizingMask:NSViewMaxXMargin | NSViewMinYMargin| NSViewMaxYMargin];
    [button setAccessibilityLabel:title];
    [button setToolTip:tooltip];
    [button setTarget:self];
    [button setAction:@selector(_callMethodForButton:)];

    [_buttons addObject:button];
    [_callbackNames addObject:callbackName];

    _nextButtonX = CGRectGetMaxX(frame) + sButtonGap;

    CGRect containerFrame = [_buttonContainer frame];
    containerFrame.size.width = _nextButtonX;
    [_buttonContainer addSubview:button];
    [_buttonContainer setFrame:containerFrame];
}


- (void) addSeparator
{
    // For now, do nothing to match existing implementation
}


- (void) updateSelectedItem:(NSString *)callbackName
{
    for (NSButton *button in _buttons) {
        [button setState:NSControlStateValueOff];
    }

    NSButton *button = [self _buttonWithCallbackName:callbackName];
    [button setButtonType:NSButtonTypePushOnPushOff];
    [button setState:NSControlStateValueOn];
}


- (void) updateMessage:(NSString *)message
{
    [_messageField setStringValue:message];
}


- (void) updateHistoryItemsWithBackEnabled: (BOOL) backEnabled
                            forwardEnabled: (BOOL) forwardEnabled
{
    [[self _buttonWithCallbackName:@"back"]    setEnabled:backEnabled];
    [[self _buttonWithCallbackName:@"forward"] setEnabled:forwardEnabled];
}

@end
