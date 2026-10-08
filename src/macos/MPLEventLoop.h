#import <AppKit/AppKit.h>
#import "MPLUtils.h"

NS_ASSUME_NONNULL_BEGIN

@interface MPLEventLoop : NSObject

+ (instancetype) sharedInstance;

// Calls to -spin... may be nested.
- (void) spinUntilStandardInput;
- (void) spinUntilNoEvents;

// Calls to -run... may not be nested. Will call -[NSApp run]
- (void) runUntilTimeout:(double)timeout; // 0 = forever

- (void) runUntilStopCondition:(BOOL (^)(void))stopCondition;
- (void) checkStopCondition;

// Stops all active loops (both -spin... and -run...)
- (void) stop;

- (void) wrapModalLoopWithLabel:(NSString *)label callback:(void (^)(void))callback;

- (void) updateCheckSignalsFileDescriptor:(int)fd;

@end

NS_ASSUME_NONNULL_END
