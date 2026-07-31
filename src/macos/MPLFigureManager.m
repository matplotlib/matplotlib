#import "MPLFigureManager.h"
#import "MPLUtils.h"


@implementation Window

- (NSRect) constrainFrameRect:(NSRect)rect toScreen:(NSScreen *)screen
{
    // Allow the window height to be larger than the screen height
    CGRect suggestedRect = [super constrainFrameRect:rect toScreen:screen];

    const CGFloat difference = rect.size.height - suggestedRect.size.height;
    suggestedRect.origin.y -= difference;
    suggestedRect.size.height += difference;

    return suggestedRect;
}

@end
