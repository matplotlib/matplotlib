#import "MPLNavigationToolbar2.h"
#import "MPLUtils.h"

static const CGFloat sToolbarHeight = 44;

static const CGFloat sLeftMargin  = 4;          // Left edge to first button
static const CGFloat sRightMargin = 6;          // Right edge to message field

static const CGFloat sButtonWidth = 36;         // Normal button width
static const CGFloat sButtonHeight = 36;        // Normal button height
static const CGFloat sButtonPadding = 0;        // Padding between normal buttons
static const CGFloat sButtonGroupPadding = 8;   // Padding between button groups

static const CGFloat sCompactThreshold = 320;   // Threshold for entering compact width mode
static const CGFloat sCompactButtonWidth = 44;  // Width of "..." menu button for compact width mode


static unsigned int sBackgroundHexValues[][3][2] = {
    {
        // Light     |   Active      Inactive
        /* Top    */ {   0xf8f8f8,   0xf0f0f0   },
        /* Bottom */ {   0xe8e8e8,   0xe0e0e0   },
        /* Shadow */ {   0x404040,   0x383838   },
    }, {
        // Dark      |   Active      Inactive
        /* Top    */ {   0x2c2c2c,   0x181818   },
        /* Bottom */ {   0x202020,   0x101010   },
        /* Shadow */ {   0x808080,   0x808080   },
    }, {
        // AX Light  |   Active      Inactive
        /* Top    */ {   0xfcfcfc,   0xf0f0f0   },
        /* Bottom */ {   0xfcfcfc,   0xf0f0f0   },
        /* Shadow */ {   0x000000,   0x000000   },
    }, {
        // AX Dark   |   Active      Inactive
        /* Top    */ {   0x383838,   0x282828   },
        /* Bottom */ {   0x383838,   0x282828   },
        /* Shadow */ {   0x989898,   0x989898   },
    }
};


static unsigned int sButtonHexValues[][3][4] = {
    {
        // Light     |   Normal      Pressed     Selected    Selected+Pressed
        /* Top    */ {   0xffffff,   0xf0f0f0,   0xe0e0e0,   0xd0d0d0   },
        /* Bottom */ {   0xf0f0f0,   0xe0e0e0,   0xd0d0d0,   0xc0c0c0   },
        /* Icon   */ {   0x404040,   0x383838,   0x101010,   0x000000   },
    }, {
        // Dark      |   Normal      Pressed     Selected    Selected+Pressed
        /* Top    */ {   0x484848,   0x585858,   0x787878,   0x808080   },
        /* Bottom */ {   0x282828,   0x383838,   0x585858,   0x606060   },
        /* Icon   */ {   0xd8d8d8,   0xe0e0e0,   0xf0f0f0,   0xf8f8f8   },
    }, {
        // AX Light  |   Normal      Pressed     Selected    Selected+Pressed
        /* Top    */ {   0xffffff,   0xd9d9d9,   0x585858,   0x404040   },
        /* Bottom */ {   0xffffff,   0xd9d9d9,   0x585858,   0x404040   },
        /* Icon   */ {   0x000000,   0x000000,   0xffffff,   0xffffff   },
    }, {
        // AX Dark   |   Normal      Pressed     Selected    Selected+Pressed
        /* Top    */ {   0x383838,   0xc8c8c8,   0xafafaf,   0xc8c8c8   },
        /* Bottom */ {   0x383838,   0xc8c8c8,   0xafafaf,   0xc8c8c8   },
        /* Icon   */ {   0xc3c3c3,   0x000000,   0x000000,   0x000000   },
    }
};


#pragma mark - Callback Maps

static NSDictionary<NSNumber *, NSString *> *sTagToCallbackNameMap = nil;
static NSDictionary<NSString *, NSNumber *> *sCallbackNameToTagMap = nil;

static void sGenerateCallbackMaps(void)
{
    if (!sTagToCallbackNameMap) {
        sTagToCallbackNameMap = @{
            @( MPLToolbarTagHome              ): @"home",
            @( MPLToolbarTagBack              ): @"back",
            @( MPLToolbarTagForward           ): @"forward",
            @( MPLToolbarTagPan               ): @"pan",
            @( MPLToolbarTagZoom              ): @"zoom",
            @( MPLToolbarTagConfigureSubplots ): @"configure_subplots",
            @( MPLToolbarTagNameSaveFigure    ): @"save_figure",
        };

        sCallbackNameToTagMap = [NSMutableDictionary dictionary];

        for (NSNumber *tagNumber in sTagToCallbackNameMap) {
            NSString *callbackName = [sTagToCallbackNameMap objectForKey:tagNumber];
            [sCallbackNameToTagMap setValue:tagNumber forKey:callbackName];
        }
    }
}


#pragma mark - MPLToolbarBackgroundView

@interface MPLToolbarBackgroundView : NSView
@end


@implementation MPLToolbarBackgroundView

- (void) drawRect:(NSRect)dirtyRect
{
    CGContextRef context = [[NSGraphicsContext currentContext] CGContext];
    CGRect bounds = [self bounds];

    MPLViewAppearance viewAppearance = MPLGetViewAppearance(self);

    BOOL isHighContrast = (
        viewAppearance == MPLViewAppearanceHighContrastLight ||
        viewAppearance == MPLViewAppearanceHighContrastDark
    );

    size_t appearanceIndex = MPLGetViewAppearance(self);
    if (appearanceIndex >= (sizeof(sBackgroundHexValues) / sizeof(sBackgroundHexValues[0]))) {
        appearanceIndex = 0;
    }

    size_t stateIndex = [[self window] isKeyWindow] ? 0 : 1;

    NSColor *topColor    = MPLGetRGBColor( sBackgroundHexValues[appearanceIndex][0][stateIndex], 1.0 );
    NSColor *bottomColor = MPLGetRGBColor( sBackgroundHexValues[appearanceIndex][1][stateIndex], 1.0 );
    NSColor *shadowColor = MPLGetRGBColor( sBackgroundHexValues[appearanceIndex][2][stateIndex], 1.0 );

    CGContextClipToRect(context, bounds);

    NSGradient *gradient = [[NSGradient alloc] initWithColors:@[ topColor, bottomColor ]];
    [gradient drawInRect:bounds angle:-90];

    if (isHighContrast) {
        [shadowColor set];

        CGRect lineRect = bounds;
        lineRect.origin.y = CGRectGetMaxY(bounds) - 1.0;
        lineRect.size.height = 1.0;
        CGContextFillRect(context, lineRect);

    } else {
        CGContextSetShadowWithColor(context, CGSizeMake(0, 0), 1, [shadowColor CGColor]);
        bounds.origin.y += bounds.size.height;
        CGContextFillRect(context, bounds);
    }
}

@end


#pragma mark - MPLToolbarButton

@interface MPLToolbarButton : NSButton

@property (nonatomic, readonly, getter=isLeftmost) BOOL leftmost;
@property (nonatomic, readonly, getter=isSelectable) BOOL selectable;
@property (nonatomic, getter=isSelected) BOOL selected;
@property (nonatomic) BOOL drawsDots;

@end


@implementation MPLToolbarButton

- (instancetype) initWithFrame:(CGRect)frame leftmost:(BOOL)leftmost selectable:(BOOL)selectable
{
    if ((self = [super initWithFrame:frame])) {
        _leftmost = leftmost;
        _selectable = selectable;

        [self setFocusRingType:NSFocusRingTypeExterior];
        [self setBezelStyle:NSBezelStyleSmallSquare];
        [self setButtonType:(selectable ? NSButtonTypePushOnPushOff : NSButtonTypeMomentaryLight)];
    }

    return self;
}


- (BOOL) isFlipped
{
    return YES;
}


- (CGRect) _insetBounds
{
    return CGRectInset([self bounds], 2, 2);
}


- (void) _addRoundedPathWithRect:(CGRect)rect
{
    CGContextRef context = [[NSGraphicsContext currentContext] CGContext];

    CGFloat cornerRadius = 8;
    CGFloat bottomLeftCornerRadius = 8;

    // Increase bottom left corner radius on macOS 26+ to match window
    if (_leftmost) {
        if (@available(macOS 26.0, *)) {
            bottomLeftCornerRadius = 10;
        }
    }

    MPLAddContinuousRoundedRect(
        context,
        rect,
        cornerRadius,
        cornerRadius,
        bottomLeftCornerRadius,
        cornerRadius
    );
}


- (void) drawRect:(NSRect)dirtyRect
{
    CGContextRef context = [[NSGraphicsContext currentContext] CGContext];

    CGRect insetBounds = [self _insetBounds];

    BOOL isSelected = _selectable && ([self state] == NSControlStateValueOn);
    BOOL isPressed  = [self isHighlighted];
    BOOL isKeyWindow = [[self window] isKeyWindow];

    MPLViewAppearance viewAppearance = MPLGetViewAppearance(self);

    BOOL isDark = (
        viewAppearance == MPLViewAppearanceDark ||
        viewAppearance == MPLViewAppearanceHighContrastDark
    );

    BOOL isHighContrast = (
        viewAppearance == MPLViewAppearanceHighContrastLight ||
        viewAppearance == MPLViewAppearanceHighContrastDark
    );

    size_t appearanceIndex = viewAppearance;
    if (appearanceIndex >= (sizeof(sButtonHexValues) / sizeof(sButtonHexValues[0]))) {
        appearanceIndex = 0;
    }

    size_t stateIndex = (isSelected ? 2 : 0) | (isPressed ? 1 : 0);

    NSColor *topColor    = MPLGetRGBColor( sButtonHexValues[appearanceIndex][0][stateIndex], 1.0 );
    NSColor *bottomColor = MPLGetRGBColor( sButtonHexValues[appearanceIndex][1][stateIndex], 1.0 );
    NSColor *iconColor   = MPLGetRGBColor( sButtonHexValues[appearanceIndex][2][stateIndex], 1.0 );

    if (!isKeyWindow) {
        CGFloat alpha = isHighContrast ? (isDark ? 0.75 : 0.85) : (isDark ? 0.8 : 0.9);

        topColor    = [topColor    colorWithAlphaComponent:alpha];
        bottomColor = [bottomColor colorWithAlphaComponent:alpha];
        iconColor   = [iconColor   colorWithAlphaComponent:alpha];
    }

    if (![self isEnabled]) {
        iconColor = [iconColor colorWithAlphaComponent:0.5];
    }

    if (topColor && bottomColor) {
        [self _addRoundedPathWithRect:insetBounds];

        CGContextSaveGState(context);
        if (!isHighContrast) CGContextSetShadow(context, CGSizeMake(0, -0.5), 1.5);
        CGContextBeginTransparencyLayer(context, NULL);

        CGContextClip(context);

        NSGradient *gradient = [[NSGradient alloc] initWithColors:@[ topColor, bottomColor ]];
        [gradient drawInRect:insetBounds angle:90];

        if (isDark && !isHighContrast) {
            [self _addRoundedPathWithRect:insetBounds];

            CGContextAddRect(context, CGRectInset(insetBounds, -10, -10));
            [[NSColor blackColor] set];

            NSColor *shadowColor = [NSColor colorWithWhite:1.0 alpha:0.25];
            CGContextSetShadowWithColor(context, CGSizeMake(0, -0.5), 1, [shadowColor CGColor]);
            CGContextSetBlendMode(context, kCGBlendModeLighten);
            CGContextEOFillPath(context);
        }

        CGContextEndTransparencyLayer(context);
        CGContextRestoreGState(context);
    }

    // Draw selection or high contrast outline
    if (isSelected || isHighContrast) {
        [self _addRoundedPathWithRect:insetBounds];

        NSColor *outlineColor;

        if (isHighContrast) {
            // Clip to rounded path to reduce stroke width
            CGContextClip(context);
            [self _addRoundedPathWithRect:insetBounds];

            outlineColor = MPLGetRGBColor(isDark ? 0xbababa : 0x606060, 1.0);

            if (!isKeyWindow) {
                outlineColor = [outlineColor colorWithAlphaComponent:0.75];
            }

        } else {
            outlineColor = MPLGetRGBColor(isDark ? 0xb0b0b0 : 0xa8a8a8, 1.0);
        }

        [outlineColor set];
        CGContextSetLineWidth(context, 2.0);
        CGContextStrokePath(context);
    }

    // Draw icon
    if (iconColor) {
        CGContextSaveGState(context);

        if (isDark && !isHighContrast) {
            NSColor *shadowColor = [NSColor colorWithWhite:0.0 alpha:0.4];
            CGContextSetShadowWithColor(context, CGSizeMake(0, -0.5), 4, [shadowColor CGColor]);
        }

        CGContextBeginTransparencyLayer(context, NULL);

        NSImage *image = [self image];
        if (image) {
            CGRect iconRect = MPLGetCenteredRect(insetBounds, CGSizeMake(24, 24));
            [image drawInRect:iconRect];

            [iconColor set];
            CGContextSetBlendMode(context, kCGBlendModeSourceIn);
            CGContextFillRect(context, iconRect);

        // It's easier to draw the three dots in code rather than sending up a PDF
        } else if (_drawsDots) {
            [iconColor set];

            CGRect dotContainer = MPLGetCenteredRect(insetBounds, CGSizeMake(22, 6));
            CGRect dotRect = dotContainer;

            dotRect.size.width = 6;
            CGContextFillEllipseInRect(context, dotRect);

            dotRect.origin.x = CGRectGetMaxX(dotRect) + 2;
            CGContextFillEllipseInRect(context, dotRect);

            dotRect.origin.x = CGRectGetMaxX(dotRect) + 2;
            CGContextFillEllipseInRect(context, dotRect);
        }

        CGContextEndTransparencyLayer(context);
        CGContextRestoreGState(context);
    }
}


- (void) drawFocusRingMask
{
    CGContextRef context = [[NSGraphicsContext currentContext] CGContext];
    [self _addRoundedPathWithRect:[self _insetBounds]];
    CGContextFillPath(context);
}


- (NSEdgeInsets) alignmentRectInsets
{
    return NSEdgeInsetsZero;
}


@end


#pragma mark - MPLNavigationToolbar2

@interface MPLNavigationToolbar2 ()  <NSMenuItemValidation>
@end


@implementation MPLNavigationToolbar2 {
    NSMutableArray<MPLToolbarButton *> *_buttons;
    NSMutableDictionary<NSNumber *, MPLToolbarButton *> *_tagToButtonMap;

    MPLToolbarButton *_compactModeButton;
    NSMenu *_compactModeMenu;

    MPLToolbarButton *_backButton;
    MPLToolbarButton *_forwardButton;
    MPLToolbarButton *_panButton;
    MPLToolbarButton *_zoomButton;

    MPLToolbarBackgroundView *_backgroundView;

    NSView *_buttonContainer;
    NSView *_messageContainer;
    NSTextField *_messageField;
    CGFloat _nextButtonX;
}


#pragma mark - Lifecycle

- (instancetype) init
{
    if ((self = [super initWithFrame:CGRectMake(0, 0, 400, sToolbarHeight)])) {
        sGenerateCallbackMaps();

        _buttons = [NSMutableArray array];
        _tagToButtonMap = [NSMutableDictionary dictionary];

        _compactModeMenu = [[NSMenu alloc] initWithTitle:@""];

        [self _setupViews];

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


#pragma mark - Superclass Overrides

- (void) layout
{
    [super layout];

    CGRect bounds = [self bounds];

    NSView *leftView;

    if (bounds.size.width < sCompactThreshold) {
        [_buttonContainer setHidden:YES];
        [_compactModeButton setHidden:NO];
        leftView = _compactModeButton;

    } else {
        [_buttonContainer setHidden:NO];
        [_compactModeButton setHidden:YES];
        leftView = _buttonContainer;
    }

    CGFloat messageX = CGRectGetMaxX([leftView frame]) + sButtonPadding;
    CGFloat messageWidth = (bounds.size.width - sRightMargin) - messageX;

    [_messageContainer setFrame:CGRectMake(messageX, 0, messageWidth, sToolbarHeight)];
}


- (void) viewDidMoveToWindow
{
    [super viewDidMoveToWindow];

    NSNotificationCenter *defaultCenter = [NSNotificationCenter defaultCenter];
    NSWindow *window = [self window];

    [defaultCenter removeObserver:self name:NSWindowDidBecomeKeyNotification object:nil];
    [defaultCenter removeObserver:self name:NSWindowDidResignKeyNotification object:nil];

    if (window) {
        [defaultCenter addObserver: self
                          selector: @selector(_handleKeyWindowDidChange:)
                              name: NSWindowDidBecomeKeyNotification
                            object: window];

        [defaultCenter addObserver: self
                          selector: @selector(_handleKeyWindowDidChange:)
                              name: NSWindowDidResignKeyNotification
                            object: window];
    }
}


#pragma mark - Private Methods

- (void) _setupViews
{
    MPLToolbarBackgroundView *backgroundView;
    MPLToolbarButton *compactModeButton;
    NSView *buttonContainer;
    NSView *messageContainer;
    NSTextField *messageField;

    // Setup background view
    {
        backgroundView = [[MPLToolbarBackgroundView alloc] initWithFrame:[self bounds]];
        [backgroundView setAutoresizingMask:NSViewWidthSizable|NSViewHeightSizable];
    }

    // Container for "normal" buttons and compact mode button
    {
        CGFloat buttonYOrigin  = floor((sToolbarHeight - sButtonHeight) / 2.0);
        CGRect  buttonContainerFrame = CGRectMake(sLeftMargin, buttonYOrigin, 200, sButtonHeight);

        buttonContainer = [[NSView alloc] initWithFrame:buttonContainerFrame];

        CGRect compactFrame = CGRectMake(sLeftMargin, buttonYOrigin, sCompactButtonWidth, sButtonHeight);

        compactModeButton = [[MPLToolbarButton alloc] initWithFrame:compactFrame leftmost:YES selectable:NO];
        [compactModeButton setDrawsDots:YES];

        [compactModeButton setTarget:self];
        [compactModeButton setAction:@selector(_showCompactModeMenu:)];
    }

    // Message field and container
    {
        NSFont *font = [NSFont monospacedDigitSystemFontOfSize:13.0 weight:NSFontWeightRegular];

        messageField = [[NSTextField alloc] initWithFrame:CGRectZero];

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

        messageContainer = [[NSView alloc] initWithFrame:CGRectZero];

        [messageContainer addSubview:messageField];
        [self addSubview:messageContainer];

        [NSLayoutConstraint activateConstraints:@[
            [[messageField leftAnchor]    constraintEqualToAnchor:[messageContainer leftAnchor]],
            [[messageField rightAnchor]   constraintEqualToAnchor:[messageContainer rightAnchor]],
            [[messageField centerYAnchor] constraintEqualToAnchor:[messageContainer centerYAnchor]],
        ]];
    };

    [backgroundView addSubview:compactModeButton];
    [backgroundView addSubview:buttonContainer];
    [backgroundView addSubview:messageContainer];

    [self addSubview:backgroundView];

    _backgroundView    = backgroundView;
    _compactModeButton = compactModeButton;
    _buttonContainer   = buttonContainer;
    _messageContainer  = messageContainer;
    _messageField      = messageField;
}


- (void) _handleKeyWindowDidChange:(NSNotification *)note
{
    for (MPLToolbarButton *button in _buttons) {
        [button setNeedsDisplay:YES];
    }

    [_backgroundView setNeedsDisplay:YES];
}


- (void) _showCompactModeMenu:(id)sender
{
    [_compactModeMenu popUpMenuPositioningItem: nil
                                    atLocation: NSMakePoint(0, CGRectGetMaxY([sender bounds]))
                                        inView: sender];
}


#pragma mark - NSMenuItemValidation Delegate Methods

- (BOOL) validateMenuItem:(NSMenuItem *)menuItem
{
    SEL action = [menuItem action];

    if (action == @selector(performToolbarCallback:)) {
        MPLToolbarTag tag = [menuItem tag];
        MPLToolbarButton *button = [_tagToButtonMap objectForKey:@(tag)];

        if (tag == MPLToolbarTagPan || tag == MPLToolbarTagZoom) {
            [menuItem setState:[button state]];
        }

        return [button isEnabled];
    }

    return YES;
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

    MPLToolbarTag tag = [[sCallbackNameToTagMap objectForKey:callbackName] integerValue];

    BOOL leftmost = [_buttons count] == 0;
    BOOL selectable = (tag == MPLToolbarTagPan || tag == MPLToolbarTagZoom);

    CGRect frame = CGRectMake(_nextButtonX, 0, sButtonWidth, sButtonHeight);

    MPLToolbarButton *button = [[MPLToolbarButton alloc] initWithFrame: frame
                                                              leftmost: leftmost
                                                            selectable: selectable];

    [button setImage:image];
    [button setAutoresizingMask:NSViewMaxXMargin | NSViewMinYMargin| NSViewMaxYMargin];
    [button setAccessibilityLabel:title];
    [button setToolTip:tooltip];

    if (tag != MPLToolbarTagNone) {
        SEL action = @selector(performToolbarCallback:);

        [button setTarget:self];
        [button setAction:action];
        [button setTag:tag];

        NSMenuItem *menuItem = [_compactModeMenu addItemWithTitle:title action:action keyEquivalent:@""];
        [menuItem setTag:tag];

        if (tag == MPLToolbarTagZoom) {
            [_compactModeMenu addItem:[NSMenuItem separatorItem]];
        }
    }

    [_buttons addObject:button];
    [_tagToButtonMap setObject:button forKey:@(tag)];

    _nextButtonX = CGRectGetMaxX(frame) + sButtonPadding;

    CGRect containerFrame = [_buttonContainer frame];
    containerFrame.size.width = _nextButtonX;
    [_buttonContainer addSubview:button];
    [_buttonContainer setFrame:containerFrame];
}


- (void) addSeparator
{
    _nextButtonX += (sButtonGroupPadding - sButtonPadding);

    [_compactModeMenu addItem:[NSMenuItem separatorItem]];
}


- (void) updateSelectedItem:(NSString *)callbackName
{
    for (MPLToolbarButton *button in _buttons) {
        [button setState:NSControlStateValueOff];
    }

    MPLToolbarTag tag = [[sCallbackNameToTagMap objectForKey:callbackName] integerValue];
    MPLToolbarButton *button = [_tagToButtonMap objectForKey:@(tag)];
    [button setState:NSControlStateValueOn];
}


- (void) updateMessage:(NSString *)message
{
    [_messageField setStringValue:message];
}


- (void) updateHistoryItemsWithBackEnabled: (BOOL) backEnabled
                            forwardEnabled: (BOOL) forwardEnabled
{
    [[_tagToButtonMap objectForKey:@(MPLToolbarTagBack)]    setEnabled:backEnabled];
    [[_tagToButtonMap objectForKey:@(MPLToolbarTagForward)] setEnabled:forwardEnabled];
}


- (IBAction) performToolbarCallback:(id)sender
{
    MPLToolbarTag tag = [sender tag];
    NSString *callbackName = [sTagToCallbackNameMap objectForKey:@(tag)];

    const char *cString = [callbackName cStringUsingEncoding:NSUTF8StringEncoding];
    if (cString) MPLCallMethod(_pyObject, cString, "");
}


@end
