#import <AppKit/AppKit.h>
#import "MPLUtils.h"

NS_ASSUME_NONNULL_BEGIN

@interface MPLAppDelegate : NSObject <NSApplicationDelegate>

- (instancetype) initWithImageDictionary: (MPLStringDictionary *) imageDictionary
                             useDarkIcon: (BOOL) useDarkIcon;

@property (nonatomic, readonly) MPLStringDictionary *imageDictionary;
@property (nonatomic, readonly) BOOL useDarkIcon;

@end

NS_ASSUME_NONNULL_END
