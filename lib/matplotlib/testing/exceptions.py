class KnownFailureTest(Exception):
    '''Raise this exception to mark a test as a known failing test.'''
    pass

class KnownFailureDidNotFailTest(Exception):
    '''Raise this exception to mark a test should have failed but did not.'''
    pass

class ImageComparisonFailure(AssertionError):
    '''Raise this exception to mark a test as a comparison between two images.'''
