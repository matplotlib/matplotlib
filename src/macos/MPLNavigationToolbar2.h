#import <AppKit/AppKit.h>
#import <Python.h>

@interface MPLNavigationToolbar2 : NSObject
- (void)installCallbacks:(SEL[7])actions forButtons:(__strong NSButton*[7])buttons;
- (void)home:(id)sender;
- (void)back:(id)sender;
- (void)forward:(id)sender;
- (void)pan:(id)sender;
- (void)zoom:(id)sender;
- (void)configure_subplots:(id)sender;
- (void)save_figure:(id)sender;
@property (nonatomic, assign) PyObject *pyObject;
@property (nonatomic, readonly) NSButton *panButton;
@property (nonatomic, readonly) NSButton *zoomButton;
@end
