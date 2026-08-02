#import "MPLAppDelegate.h"
#import "MPLUtils.h"


// These are standard selectors which AppKit never exposes in any header.
// They are typically shown only in Interface Builder as a potential action.
@interface NSObject ()
- (void) closeAll:(id)sender;
- (void) undo:(id)sender;
- (void) redo:(id)sender;
@end


@implementation MPLAppDelegate

#pragma mark - Lifecycle

- (instancetype) initWithImageDictionary:(MPLStringDictionary *)imageDictionary
{
    if ((self = [super init])) {
        _imageDictionary = imageDictionary;
        MPLLog("[Lifecycle] MPLAppDelegate<%p> init", self);
    }

    return self;
}


- (void) dealloc
{
    MPLLog("[Lifecycle] MPLAppDelegate<%p> dealloc", self);
}


#pragma mark - Superclass Overrides

- (BOOL) applicationSupportsSecureRestorableState:(NSApplication *)app
{
    return YES;
}


- (void) applicationWillFinishLaunching:(NSNotification *)notification
{
    [self _buildMainMenu];
    [self _buildAppIcon];

    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
}


#pragma mark - Private Methods

- (void) _buildMainMenu
{
    NSEventModifierFlags command       = NSEventModifierFlagCommand;
    NSEventModifierFlags optionCommand = NSEventModifierFlagOption | command;
    NSEventModifierFlags shiftCommand  = NSEventModifierFlagShift  | command;

    __block NSMenu *currentMenu;
    __block NSMenuItem *currentItem;

    NSMenu *mainMenu = [[NSMenu alloc] init];

    __auto_type menu = ^(NSString *title) {
        NSMenu *menu = [[NSMenu alloc] init];

        NSMenuItem *menuItem = [[NSMenuItem alloc] init];
        [menuItem setTitle:title];
        [menuItem setSubmenu:menu];
        [menuItem setTarget:menu];
        [menuItem setAction:@selector(submenuAction:)];
        [mainMenu addItem:menuItem];

        currentMenu = menu;
    };

    __auto_type item = ^(NSString *title, NSEventModifierFlags flags, NSString *keyEquivalent, SEL action) {
        NSMenuItem *item = [[NSMenuItem alloc] init];

        [item setTitle:title];
        [item setKeyEquivalent:keyEquivalent];
        [item setKeyEquivalentModifierMask:flags];
        [item setAction:action];

        [currentMenu addItem:item];

        currentItem = item;
    };

    __auto_type separator = ^() {
        [currentMenu addItem:[NSMenuItem separatorItem]];
    };

    menu(@"Matplotlib");
    item(@"Hide Matplotlib", command,       @"h", @selector(hide:));
    item(@"Hide Others",     optionCommand, @"h", @selector(hideOtherApplications:));
    item(@"Show All",        0,             @"",  @selector(unhideAllApplications:));
    separator();
    item(@"Quit Matplotlib", command,       @"q", @selector(terminate:));

    menu(@"File");
    item(@"Close",     command,       @"w", @selector(performClose:));
    item(@"Close All", optionCommand, @"w", @selector(closeAll:));
    [currentItem setTarget:NSApp];
    [currentItem setAlternate:YES];

    menu(@"Edit");
    item(@"Undo",       command,      @"z", @selector(undo:));
    item(@"Redo",       shiftCommand, @"z", @selector(redo:));
    separator();
    item(@"Cut",        command,      @"x", @selector(cut:));
    item(@"Copy",       command,      @"c", @selector(copy:));
    item(@"Paste",      command,      @"v", @selector(paste:));
    item(@"Delete",     0,            @"",  @selector(delete:));
    item(@"Select All", command,      @"a", @selector(selectAll:));

    menu(@"Window");
    item(@"Minimize",           command, @"m", @selector(performMiniaturize:));
    item(@"Zoom",               0,       @"",  @selector(performZoom:));
    separator();
    item(@"Bring All to Front", 0,       @"",  @selector(arrangeInFront:));
    [NSApp setWindowsMenu:currentMenu];

    menu(@"Help");
    [NSApp setHelpMenu:currentMenu];

    [NSApp setMainMenu:mainMenu];

    for (NSWindow *window in [NSApp windows]) {
        [NSApp addWindowsItem:window title:[window title] filename:NO];
    }
}


- (void) _buildAppIcon
{
    NSString *imagePath = [_imageDictionary objectForKey:@"matplotlib"];
    NSURL *imageURL = imagePath ? [NSURL fileURLWithPath:imagePath] : nil;

    if (imageURL) {
        NSImage *image = [[NSImage alloc] initWithContentsOfURL:imageURL];
        [NSApp setApplicationIconImage:image];
    }
}


@end
