#import "MPLAppDelegate.h"
#import "MPLUtils.h"


// These are standard selectors which AppKit never exposes in any header.
// They are typically shown only in Interface Builder as a potential action.
@interface NSObject (MissingPublicMethods)
- (void) closeAll:(id)sender;
- (void) undo:(id)sender;
- (void) redo:(id)sender;
@end


@implementation MPLAppDelegate

#pragma mark - Lifecycle

- (instancetype) initWithImageDictionary: (MPLStringDictionary *) imageDictionary
                             useDarkIcon: (BOOL) useDarkIcon
{
    if ((self = [super init])) {
        _imageDictionary = imageDictionary;
        _useDarkIcon = useDarkIcon;

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
    MPLStringDictionary *imageDictionary = _imageDictionary;
    NSImage *shadowImage;
    NSImage *maskImage;
    NSImage *darkBackgroundImage;

    BOOL useDarkIcon = _useDarkIcon;

    __auto_type getImage = ^(NSString *key) {
        NSString *path = [imageDictionary objectForKey:key];
        return path ? [[NSImage alloc] initWithContentsOfFile:path] : nil;
    };

    if (@available(macOS 27.0, *)) {
        darkBackgroundImage = getImage(@"macos_appicon_bgdark27");
    } else if (@available(macOS 26.0, *)) {
        darkBackgroundImage = getImage(@"macos_appicon_bgdark26");
    }

    if (@available(macOS 26.0, *)) {
        shadowImage = getImage(@"macos_appicon_shadow26");
        maskImage   = getImage(@"macos_appicon_mask26");
    } else {
        shadowImage = getImage(@"macos_appicon_shadow11");
        maskImage   = getImage(@"macos_appicon_mask11");
    }

    NSImage *contentImage = useDarkIcon ?
        getImage(@"macos_appicon_dark") :
        getImage(@"macos_appicon_light");

    if (!maskImage || !shadowImage || !contentImage) return;

    // Use standard size of 256 physical pixels (128x128 @2x)
    CGSize iconSize = CGSizeMake(256, 256);
    CGImageRef iconCGImage = MPLCreateImage(iconSize, 1, ^(CGContextRef context) {
        CGRect iconBounds = CGRectMake(0, 0, iconSize.width, iconSize.height);

        __auto_type getCGImage = ^(NSImage *nsImage) {
            CGRect proposedRect = iconBounds;
            return [nsImage CGImageForProposedRect: &proposedRect
                                           context: [NSGraphicsContext currentContext]
                                             hints: nil];
        };

        __auto_type getFrame = ^(CGImageRef image) {
            CGSize size = CGSizeMake(CGImageGetWidth(image), CGImageGetHeight(image));
            return MPLGetCenteredRect(iconBounds, size);
        };

        CGImageRef maskCGImage    = MPLCopyGrayscaleNonAlphaImage(getCGImage(maskImage));
        CGImageRef shadowCGImage  = getCGImage(shadowImage);
        CGImageRef contentCGImage = getCGImage(contentImage);
        CGImageRef darkBGCGImage  = getCGImage(darkBackgroundImage);

        // MPLCreateImage() uses an upper-left-origin to match other graphics libraries.
        // Reset to lower-left-origin coordinate system since we are drawing images,
        // as CGContextDrawImage() translates the CTM by the image height and flips
        // prior to drawing.
        CGContextTranslateCTM(context, 0, iconSize.height);
        CGContextScaleCTM(context, 1, -1);

        CGContextDrawImage(context, getFrame(shadowCGImage), shadowCGImage);
        CGContextClipToMask(context, getFrame(maskCGImage), maskCGImage);

        if (useDarkIcon) {
            if (darkBackgroundImage) {
                CGContextDrawImage(context, getFrame(darkBGCGImage), darkBGCGImage);
            } else {
                NSGradient *gradient = [[NSGradient alloc] initWithColors:@[
                    MPLGetRGBColor(0x303030, 1.0),
                    MPLGetRGBColor(0x181818, 1.0)
                ]];

                [gradient drawInRect:CGRectMake(0, 0, 256, 256) angle:-90];
            }
        }

        CGContextDrawImage(context, getFrame(contentCGImage), contentCGImage);

        CGImageRelease(maskCGImage);
    });

    if (iconCGImage) {
        NSImage *iconImage = [[NSImage alloc] initWithCGImage:iconCGImage size:iconSize];
        [NSApp setApplicationIconImage:iconImage];
        CGImageRelease(iconCGImage);
    }
}


@end
