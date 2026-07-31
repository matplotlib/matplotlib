#import "MPLNavigationToolbar2.h"
#import "MPLUtils.h"


@implementation MPLNavigationToolbar2

- (void)installCallbacks:(SEL[7])actions forButtons:(__strong NSButton*[7])buttons
{
    for (int i = 0; i < 7; i++) {
        SEL action = actions[i];
        NSButton* button = buttons[i];
        [button setTarget: self];
        [button setAction: action];
        if (action == @selector(pan:)) { _panButton = button; }
        if (action == @selector(zoom:)) { _zoomButton = button; }
    }
}

-(void)home:(id)sender { gil_call_method(_pyObject, "home"); }
-(void)back:(id)sender { gil_call_method(_pyObject, "back"); }
-(void)forward:(id)sender { gil_call_method(_pyObject, "forward"); }

-(void)pan:(id)sender
{
    if ([sender state]) { [_zoomButton setState:NO]; }
    gil_call_method(_pyObject, "pan");
}

-(void)zoom:(id)sender
{
    if ([sender state]) { [_panButton setState:NO]; }
    gil_call_method(_pyObject, "zoom");
}

-(void)configure_subplots:(id)sender { gil_call_method(_pyObject, "configure_subplots"); }
-(void)save_figure:(id)sender { gil_call_method(_pyObject, "save_figure"); }
@end
