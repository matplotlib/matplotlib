#import "MPLSubplotTool.h"
#import "MPLFigureManager.h"
#import "MPLUtils.h"

@interface MPLSubplotTool () <NSWindowDelegate>

@property (nonatomic) double left;
@property (nonatomic) double bottom;
@property (nonatomic) double right;
@property (nonatomic) double top;
@property (nonatomic) double wspace;
@property (nonatomic) double hspace;

- (void) copyValues:(id)sender;

@end


@interface MPLSubplotToolContentView : NSView
@property (nonatomic, weak) MPLSubplotTool *subplotTool;
@end


@implementation MPLSubplotToolContentView

- (BOOL) acceptsFirstResponder
{
    return YES;
}


- (void) copy:(id)sender
{
    [_subplotTool copyValues:self];
}

@end


@implementation MPLSubplotTool {
    NSStackView *_stackView;
    BOOL _inUpdate;
}

// The display order in the UI, not the order sent/received across the bridge
+ (NSArray<NSString *> *) _subplotParameterOrder
{
    return @[ @"top", @"bottom", @"left", @"right", @"hspace", @"wspace" ];
}


#pragma mark - Lifecycle

- (instancetype) initWithFigureManager:(MPLFigureManager *)manager
{
    CGRect contentRect = CGRectMake(0, 0, 400, 260);

    NSWindowStyleMask styleMask = NSWindowStyleMaskTitled
                                | NSWindowStyleMaskClosable
                                | NSWindowStyleMaskResizable
                                | NSWindowStyleMaskMiniaturizable;

    NSWindow *window = [[NSWindow alloc] initWithContentRect: contentRect
                                                   styleMask: styleMask
                                                     backing: NSBackingStoreBuffered
                                                       defer: YES
                                                      screen: [[manager window] screen]];

    CGRect contentFrame = [[window contentView] frame];
    MPLSubplotToolContentView *contentView = [[MPLSubplotToolContentView alloc] initWithFrame:contentFrame];
    [contentView setSubplotTool:self];
    [window setContentView:contentView];

    [window setAppearance:[[manager window] appearance]];
    [window makeFirstResponder:contentView];
    [window setTitle:[NSString stringWithFormat:@"%@ Subplots", [manager windowTitle]]];
    [window setDelegate:self];
    [window setReleasedWhenClosed:NO];
    [window center];

    if (self = [super initWithWindow:window]) {
        [self _buildUI];
        MPLLog("[Lifecycle] MPLSubplotTool<%p> init", self);
    }

    return self;
}


- (void) dealloc
{
    MPLLog("[Lifecycle] MPLSubplotTool<%p> dealloc", self);
}


#pragma mark - Superclass Overrides

- (void) observeValueForKeyPath: (NSString *) keyPath
                       ofObject: (id) object
                         change: (NSDictionary<NSKeyValueChangeKey, id> *) change
                        context: (void *) context
{
    if ((object == self) && keyPath && [[MPLSubplotTool _subplotParameterOrder] containsObject:keyPath]) {
        if (!_inUpdate) {
            [self _sendParamsChanged];
        }
    } else {
        [super observeValueForKeyPath:keyPath ofObject:object change:change context:context];
    }
}


- (BOOL) validateValue: (inout id _Nullable * _Nonnull) ioValue
                forKey: (NSString *) inKey
                 error: (out NSError **) outError
{
    NSString *pairedKey;
    BOOL checkHigher = NO;

    if      ([inKey isEqualToString:@"left"])   { pairedKey = @"right";   checkHigher = YES; }
    else if ([inKey isEqualToString:@"right"])  { pairedKey = @"left";    checkHigher = NO;  }
    else if ([inKey isEqualToString:@"top"])    { pairedKey = @"bottom";  checkHigher = NO;  }
    else if ([inKey isEqualToString:@"bottom"]) { pairedKey = @"top";     checkHigher = YES; }

    if (pairedKey) {
        double myValue = [*ioValue doubleValue];
        double pairedValue = [[self valueForKey:pairedKey] doubleValue];

        if (checkHigher) {
            if (myValue >= pairedValue) *ioValue = @( pairedValue - 0.001 );
        } else {
            if (myValue <= pairedValue) *ioValue = @( pairedValue + 0.001 );
        }

        return YES;

    } else {
        return [super validateValue:ioValue forKey:inKey error:outError];
    }
}


#pragma mark - Private Methods

- (void) _buildUI
{
    NSStackView *stackView = [NSStackView stackViewWithViews:@[ ]];
    NSMutableArray *valueRowViews = [NSMutableArray array];

    __block NSTextField *firstLabelField;
    __block NSSlider    *firstSlider;
    __block NSTextField *firstValueField;

    __auto_type addParameterRow = ^(NSString *keyName) {
        NSView *rowView = [[NSView alloc] init];

        double minValue = 0.0;
        double maxValue = 1.0;

        NSDictionary *bindingOptions = @{ NSValidatesImmediatelyBindingOption: @YES };

        NSNumberFormatter *formatter = [[NSNumberFormatter alloc] init];
        [formatter setNumberStyle:NSNumberFormatterDecimalStyle];
        [formatter setMinimumFractionDigits:1];
        [formatter setMaximumFractionDigits:3];
        [formatter setMinimum:@(minValue)];
        [formatter setMaximum:@(maxValue)];

        NSTextField *labelField = [NSTextField labelWithString:keyName];
        [labelField setAlignment:NSTextAlignmentRight];
        [labelField setTranslatesAutoresizingMaskIntoConstraints:NO];
        [[labelField cell] setControlSize:NSControlSizeRegular];

        NSSlider *slider = [[NSSlider alloc] init];
        [slider setMinValue:minValue];
        [slider setMaxValue:maxValue];
        [slider setContinuous:YES];
        [slider setTranslatesAutoresizingMaskIntoConstraints:NO];
        [slider bind:NSValueBinding toObject:self withKeyPath:keyName options:bindingOptions];

        NSTextField *valueField = [[NSTextField alloc] init];
        [valueField setFormatter:formatter];
        [[valueField cell] setControlSize: NSControlSizeRegular];
        [valueField setTranslatesAutoresizingMaskIntoConstraints:NO];
        [valueField bind:NSValueBinding toObject:self withKeyPath:keyName options:bindingOptions];

        [slider setAccessibilityTitleUIElement:labelField];
        [valueField setAccessibilityTitleUIElement:labelField];

        [rowView addSubview:labelField];
        [rowView addSubview:slider];
        [rowView addSubview:valueField];

        [labelField setContentCompressionResistancePriority:NSLayoutPriorityRequired forOrientation:NSLayoutConstraintOrientationHorizontal];

        [stackView addArrangedSubview:rowView];
        [valueRowViews addObject:rowView];

        [self addObserver:self forKeyPath:keyName options:0 context:NULL];

        [NSLayoutConstraint activateConstraints:@[
            [[labelField leftAnchor]  constraintEqualToAnchor:[rowView    leftAnchor]  constant:0],
            [[slider     leftAnchor]  constraintEqualToAnchor:[labelField rightAnchor] constant:12],
            [[valueField leftAnchor]  constraintEqualToAnchor:[slider     rightAnchor] constant:12],
            [[valueField rightAnchor] constraintEqualToAnchor:[rowView    rightAnchor] constant:0],
            [[valueField widthAnchor] constraintEqualToConstant:64],

            [[labelField firstBaselineAnchor] constraintEqualToAnchor:[slider firstBaselineAnchor]],
            [[valueField firstBaselineAnchor] constraintEqualToAnchor:[slider firstBaselineAnchor]],

            [[slider centerYAnchor] constraintEqualToAnchor:[rowView centerYAnchor]],

            [[rowView heightAnchor] constraintGreaterThanOrEqualToConstant:24]
        ]];

        // All sliders should be equal widths, etc.
        if (firstLabelField && firstSlider && firstValueField) {
            [NSLayoutConstraint activateConstraints:@[
                [[firstLabelField widthAnchor] constraintEqualToAnchor:[labelField widthAnchor]],
                [[firstSlider     widthAnchor] constraintEqualToAnchor:[slider     widthAnchor]],
                [[firstValueField widthAnchor] constraintEqualToAnchor:[valueField widthAnchor]],
            ]];

        } else {
            firstLabelField = labelField;
            firstSlider     = slider;
            firstValueField = valueField;
        }
    };

    __auto_type makeButton = ^(NSString *title, SEL action) {
        NSButton *button = [NSButton buttonWithTitle:title target:self action:action];

        [button setContentCompressionResistancePriority:NSLayoutPriorityDefaultHigh forOrientation:NSLayoutConstraintOrientationHorizontal];
        [button setContentHuggingPriority:1 forOrientation:NSLayoutConstraintOrientationHorizontal];

        return button;
    };

    __auto_type addButtonsRow = ^() {
        NSButton *copyButton  = makeButton(@"Copy Values",  @selector(copyValues:));
        NSButton *tightButton = makeButton(@"Tight Layout", @selector(selectTightLayout:));
        NSButton *resetButton = makeButton(@"Reset",        @selector(resetLayout:));

        NSStackView *rowView = [NSStackView stackViewWithViews:@[ copyButton, tightButton, resetButton ]];

        [rowView setOrientation:NSUserInterfaceLayoutOrientationHorizontal];
        [rowView setAlignment:NSLayoutAttributeCenterY];
        [rowView setDistribution:NSStackViewDistributionFillEqually];
        [rowView setSpacing:12.0];
        [rowView setEdgeInsets:NSEdgeInsetsMake(0, 20, 0, 20)];

        [stackView addArrangedSubview:rowView];

        [stackView setCustomSpacing:12 afterView:[valueRowViews lastObject]];

        [NSLayoutConstraint activateConstraints:@[
            [[stackView leftAnchor]   constraintEqualToAnchor:[rowView leftAnchor]],
            [[stackView rightAnchor]  constraintEqualToAnchor:[rowView rightAnchor]]
        ]];
    };

    for (NSString *key in [MPLSubplotTool _subplotParameterOrder]) {
        addParameterRow(key);
    }

    addButtonsRow();

    [stackView setOrientation:NSUserInterfaceLayoutOrientationVertical];
    [stackView setDistribution:NSStackViewDistributionFillEqually];
    [stackView setSpacing:4];
    [stackView setEdgeInsets:NSEdgeInsetsMake(20, 20, 20, 20)];

    NSView *contentView = [[self window] contentView];
    [contentView addSubview:stackView];

    [NSLayoutConstraint activateConstraints:@[
        [[stackView leftAnchor]    constraintEqualToAnchor:[contentView leftAnchor]],
        [[stackView rightAnchor]   constraintEqualToAnchor:[contentView rightAnchor]],
        [[stackView topAnchor]     constraintEqualToAnchor:[contentView topAnchor]],
        [[stackView bottomAnchor]  constraintEqualToAnchor:[contentView bottomAnchor]],
    ]];

    _stackView = stackView;
}


- (void) _sendParamsChanged
{
    if (!_inUpdate) {
        MPLCallMethod(_pyObject, "_send_params_to_figure", "dddddd",
            _left, _bottom, _right, _top, _wspace, _hspace
        );
    }
}


#pragma mark - Public Methods

- (void) updateWithLeft: (double) left
                 bottom: (double) bottom
                  right: (double) right
                    top: (double) top
                 wspace: (double) wspace
                 hspace: (double) hspace
{
    _inUpdate = YES;

    [self setLeft:left];
    [self setBottom:bottom];
    [self setRight:right];
    [self setTop:top];
    [self setWspace:wspace];
    [self setHspace:hspace];

    _inUpdate = NO;
}


#pragma mark - Actions

- (void) copyValues:(id)sender
{
    NSMutableArray *keyPairs = [NSMutableArray array];

    for (NSString *keyName in [MPLSubplotTool _subplotParameterOrder]) {
        NSNumber *number = [self valueForKey:keyName];

        if ([number isKindOfClass:[NSNumber class]]) {
            [keyPairs addObject:[NSString stringWithFormat:@"%@=%.3g", keyName, [number doubleValue]]];
        }
    }

    NSString *string = [keyPairs componentsJoinedByString:@",\n"];

    NSPasteboard *pasteboard = [NSPasteboard generalPasteboard];
    [pasteboard clearContents];
    [pasteboard setString:string forType:NSPasteboardTypeString];
}


- (void) selectTightLayout:(id)sender
{
    MPLCallMethod(_pyObject, "_handle_tight_button", NULL);
}


- (void) resetLayout:(id)sender
{
    MPLCallMethod(_pyObject, "_handle_reset_button", NULL);
}

@end
