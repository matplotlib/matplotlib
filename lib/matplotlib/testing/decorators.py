from matplotlib.testing.noseclasses import KnownFailureTest, \
     KnownFailureDidNotFailTest
import sys

def knownfailureif(fail_condition, msg=None):
    # based on numpy.testing.dec.knownfailureif
    if msg is None:
        msg = 'Test known to fail'
    def known_fail_decorator(f):
        # Local import to avoid a hard nose dependency and only incur the
        # import time overhead at actual test-time.
        import nose
        def failer(*args, **kwargs):
            try:
                # Always run the test (to generate images).
                result = f(*args, **kwargs)
            except:
                if fail_condition:
                    raise KnownFailureTest(msg)
                else:
                    raise
            if fail_condition:
                raise KnownFailureDidNotFailTest(msg)
            return result
        return nose.tools.make_decorator(f)(failer)
    return known_fail_decorator
