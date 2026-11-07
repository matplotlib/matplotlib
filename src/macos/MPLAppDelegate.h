#import <AppKit/AppKit.h>
#import "MPLUtils.h"

NS_ASSUME_NONNULL_BEGIN

@interface MPLAppDelegate : NSObject <NSApplicationDelegate>

- (instancetype) initWithImageDictionary:(MPLStringDictionary *)imageDictionary;

@property (nonatomic, readonly) MPLStringDictionary *imageDictionary;

@end

NS_ASSUME_NONNULL_END
