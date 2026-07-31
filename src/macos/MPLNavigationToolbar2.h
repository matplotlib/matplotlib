#import <AppKit/AppKit.h>
#import <Python.h>

NS_ASSUME_NONNULL_BEGIN

@interface MPLNavigationToolbar2 : NSView

- (instancetype) init NS_DESIGNATED_INITIALIZER;

- (instancetype) initWithFrame:(NSRect)frame NS_UNAVAILABLE;
- (nullable instancetype) initWithCoder:(NSCoder *)coder NS_UNAVAILABLE;

@property (nonatomic, assign, nullable) PyObject *pyObject;

- (void) addItemWithTitle: (NSString *) title
                  tooltip: (NSString *) tooltip
                imagePath: (NSString *) imagePath
             callbackName: (NSString *) callbackName;

- (void) addSeparator;

- (void) updateSelectedItem:(NSString *)callback;
- (void) updateMessage:(NSString *)message;

- (void) updateHistoryItemsWithBackEnabled: (BOOL) backEnabled
                            forwardEnabled: (BOOL) forwardEnabled;

@end

NS_ASSUME_NONNULL_END
