#import "MPLTimer.h"
#import "MPLUtils.h"


@implementation MPLTimer {
    dispatch_source_t _source;
    uint64_t _intervalInNsecs;
}


#pragma mark - Lifecycle

- (instancetype) init
{
    if ((self = [super init])) {
        MPLLog("[Lifecycle] MPLTimer<%p> init", self);
    }

    return self;
}


- (void) dealloc
{
    MPLLog("[Lifecycle] MPLTimer<%p> dealloc", self);
}


#pragma mark - Private Methods

- (void) _clearSource
{
    _source = nil;
}


- (void) _cancelAndClearSource
{
    dispatch_source_t source = _source;
    _source = nil;
    if (source) dispatch_source_cancel(source);
}


- (void) _handleTimerTick
{
    MPLCallMethod(_pyObject, "_on_timer", "");

    if ([self isSingleShot]) {
        [self _cancelAndClearSource];
    }
}


- (void) _restartTimer
{
    dispatch_source_t source = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_TIMER, 0, 0, dispatch_get_main_queue()
    );

    dispatch_time_t start = dispatch_time(DISPATCH_TIME_NOW, _intervalInNsecs);
    dispatch_source_set_timer(source, start, _intervalInNsecs, 0);

    __weak MPLTimer *weakSelf = self;
    dispatch_source_set_event_handler( source, ^{ [weakSelf _handleTimerTick]; });
    dispatch_source_set_cancel_handler(source, ^{ [weakSelf _clearSource]; });

    [self _cancelAndClearSource];
    _source = source;
    dispatch_activate(source);
}


#pragma mark - Public Methods

- (void) start
{
    [self _restartTimer];
}


- (void) stop
{
    [self _cancelAndClearSource];
}


- (void) updateIntervalInMsecs:(int)intervalInMsecs
{
    _intervalInNsecs = intervalInMsecs * NSEC_PER_MSEC;
    if (_source) [self _restartTimer];
}


@end
