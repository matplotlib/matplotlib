#import "MPLEventLoop.h"
#import "MPLUtils.h"

typedef NS_ENUM(NSInteger, LoopType) {
    LoopTypeNone,
    LoopTypeRun,
    LoopTypeSpin,
    LoopTypeModal
};


@implementation MPLEventLoop {
    LoopType _currentLoopType;

    // Keeps track of all active loops (both -spin... and -run...)
    // Needed so we can post enough events to cancel all loops
    NSInteger _loopCount;

    // Each -spin... loop checks this to see if it should stop.
    // We can't use -[NSApplication isRunning] as spin loops can be used
    // while not in a running state (flush_events())
    BOOL _shouldStop;

    // Makes sure that [NSApp run] was called at least once for proper initialization
    BOOL _wasRunCalledAtLeastOnce;

    // Block for use with -runUntilStopCondition: / -checkStopCondition
    BOOL (^_stopCondition)(void);

    dispatch_source_t _checkSignalsDispatchSource;
}


#pragma mark - Lifecycle

+ (instancetype) sharedInstance
{
    static MPLEventLoop *sharedInstance;

    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedInstance = [[MPLEventLoop alloc] init];
    });

    return sharedInstance;
}


- (instancetype) init
{
    if ((self = [super init])) {
        _wasRunCalledAtLeastOnce = [NSApp isRunning];
    }

    return self;
}


#pragma mark - Private Methods

- (void) _postInternalEvent
{
    /*
        There exists a very small possibility that we are running as a framework
        inside of another app which is using NSEventTypeApplicationDefined events.
        Hence, set subtype to a value unlikely to conflict.
    */
    NSEvent *event = [NSEvent otherEventWithType: NSEventTypeApplicationDefined
                                        location: NSZeroPoint
                                   modifierFlags: 0
                                       timestamp: 0
                                    windowNumber: 0
                                         context: nil
                                         subtype: INT16_MAX
                                           data1: 0
                                           data2: 0];

    /*
        +[NSEvent otherEventWithType:...] is declared nullable but will not return
        nil for these constant, valid arguments; guard defensively anyway.
    */
    if (event) {
        [NSApp postEvent:event atStart:YES];
    }
}


- (void) _runForOneCycle
{
    __weak id weakSelf = self;

    dispatch_async(dispatch_get_main_queue(), ^{
        [NSApp stop:self];
        [weakSelf _postInternalEvent];
    });

    [NSApp run];
}


- (void) _wrapLoopWithType:(LoopType)loopType label:(NSString *)label callback:(void (^)(void))callback
{
    if (loopType == LoopTypeSpin && !_wasRunCalledAtLeastOnce) {
        MPLLog("[EventLoop] Calling -[NSApplication run] for one cycle to perform initialization");
        [self _runForOneCycle];
    }
    _wasRunCalledAtLeastOnce = YES;

    LoopType previousLoopType = _currentLoopType;
    _currentLoopType = loopType;
    _loopCount++;

    MPLLog("[EventLoop] +++ loop #%ld entry +++ %@", (long)_loopCount, label);
    callback();
    MPLLog("[EventLoop] --- loop #%ld exit  --- %@", (long)_loopCount, label);

    _loopCount--;
    _currentLoopType = previousLoopType;

    if (_shouldStop) {
        if (_loopCount > 0) {
            [self stop];
        } else {
            _shouldStop = NO;
        }
    }
}


#pragma mark - Public Methods

- (void) spinUntilStandardInput
{
    __block int inputDidOccur = 0;
    __weak id weakSelf = self;

    dispatch_source_t source = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_READ, STDIN_FILENO, 0,
        dispatch_get_main_queue()
    );

    dispatch_source_set_event_handler(source, ^{
        inputDidOccur = 1;
        dispatch_source_cancel(source);
        [weakSelf _postInternalEvent];
    });

    dispatch_resume(source);

    [self _wrapLoopWithType:LoopTypeSpin label:@"spinUntilStandardInput" callback:^{
        // We purposely do not check _shouldStop here. In the event of a Control-C,
        // we can't return to Python until inputDidOccur is YES, else we will hang
        // while Python waits in my_fgets().
        while (!inputDidOccur) {
            @autoreleasepool {
                NSEvent *event = [NSApp nextEventMatchingMask: NSEventMaskAny
                                                    untilDate: [NSDate distantFuture]
                                                       inMode: NSDefaultRunLoopMode
                                                      dequeue: YES];

                if (event) {
                    [NSApp sendEvent:event];
                }
            }
        }
    }];
}


- (void) spinUntilNoEvents
{
    [self _wrapLoopWithType:LoopTypeSpin label:@"spinUntilNoEvents" callback:^{
        while (!self->_shouldStop) {
            @autoreleasepool {
                NSEvent *event = [NSApp nextEventMatchingMask: NSEventMaskAny
                                                    untilDate: [NSDate distantPast]
                                                       inMode: NSDefaultRunLoopMode
                                                      dequeue: YES];

                if (!event) break;

                [NSApp sendEvent:event];
            }
        }
    }];
}


- (void) runUntilTimeout:(double)timeout
{
    if (timeout > 0.0) {
        [self performSelector:@selector(stop) withObject:nil afterDelay:timeout];
    }

    [self _wrapLoopWithType:LoopTypeRun label:@"runUntilTimeout" callback:^{
        [NSApp run];
    }];

    [NSObject cancelPreviousPerformRequestsWithTarget:self selector:@selector(stop) object:nil];
}


- (void) runUntilStopCondition:(BOOL (^)(void))stopCondition
{
    _stopCondition = stopCondition;

    [self _wrapLoopWithType:LoopTypeRun label:@"runUntilStopCondition" callback:^{
        [NSApp run];
    }];

    _stopCondition = nil;
}


- (void) checkStopCondition
{
    if (_stopCondition && _stopCondition()) {
        [self stop];
    }
}


- (void) stop
{
    MPLLog("[EventLoop] stop requested, _loopCount = %ld", (long)_loopCount);

    _shouldStop = YES;

    if (_currentLoopType == LoopTypeRun) {
        [NSApp stop:self];
    } else if (_currentLoopType == LoopTypeModal) {
        [NSApp stopModal];
    }

    [self _postInternalEvent];
}


- (void) wrapModalLoopWithLabel:(NSString *)label callback:(void (^)(void))callback
{
    [self _wrapLoopWithType:LoopTypeModal label:label callback:callback];
}


- (void) updateCheckSignalsFileDescriptor:(int)fd
{
    if (_checkSignalsDispatchSource) {
        dispatch_cancel(_checkSignalsDispatchSource);
        _checkSignalsDispatchSource = nil;
    }

    if (fd < 0) return;

    int duplicatedFD = dup(fd);
    if (duplicatedFD < 0) return;

    dispatch_source_t source = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_READ, duplicatedFD, 0,
        dispatch_get_main_queue()
    );

    dispatch_source_set_cancel_handler(source, ^{
        close(duplicatedFD);
    });

    __weak dispatch_source_t weakSource = source;
    dispatch_source_set_event_handler(source, ^{
        MPLCheckSignals();
        dispatch_source_cancel(weakSource);
    });

    _checkSignalsDispatchSource = source;
    dispatch_resume(_checkSignalsDispatchSource);
}


@end
