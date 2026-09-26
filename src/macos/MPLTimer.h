#import <AppKit/AppKit.h>
#import "MPLUtils.h"

NS_ASSUME_NONNULL_BEGIN

@interface MPLTimer : NSObject

- (void) start;
- (void) stop;

- (void) updateIntervalInMsecs:(int)intervalInMsecs;

@property (nonatomic, getter=isSingleShot) BOOL singleShot;
@property (nonatomic, assign, nullable) MPLPyObjectRef pyObject;

@end

NS_ASSUME_NONNULL_END
