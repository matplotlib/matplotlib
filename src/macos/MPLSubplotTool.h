#import <AppKit/AppKit.h>
#import "MPLUtils.h"

NS_ASSUME_NONNULL_BEGIN

@class MPLFigureManager;


@interface MPLSubplotTool : NSWindowController

- (instancetype) initWithFigureManager:(MPLFigureManager *)manager NS_DESIGNATED_INITIALIZER;

- (instancetype) initWithWindow:(nullable NSWindow *)window NS_UNAVAILABLE;
- (nullable instancetype) initWithCoder:(NSCoder *)coder NS_UNAVAILABLE;

- (void) updateWithLeft: (double) left
                 bottom: (double) bottom
                  right: (double) right
                    top: (double) top
                 wspace: (double) wspace
                 hspace: (double) hspace;

@property (nonatomic, assign, nullable) MPLPyObjectRef pyObject;

@end

NS_ASSUME_NONNULL_END
