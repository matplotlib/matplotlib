from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import six
import sys
from .. import copy_metadata, skip
from . import knownfail
from .exceptions import KnownFailureDidNotFailTest


def skipif(skip_condition, *args, **kwargs):
    if isinstance(skip_condition, bool) and 'reason' not in kwargs:
        raise ValueError("you need to specify reason=STRING "
                         "when using booleans as conditions.")

    def skip_decorator(func):
        import inspect

        def skipper(*_args, **_kwargs):
            condition, msg = skip_condition, kwargs.get('reason')  # local copy
            if isinstance(condition, six.string_types):
                globs = {'os': os, 'sys': sys}
                try:
                    globs.update(func.__globals__)
                except AttributeError:
                    globs.update(func.func_globals)
                if msg is None:
                    msg = condition
                condition = eval(condition, globs)
            else:
                condition = bool(condition)

            if condition:
                skip(msg)
            else:
                return func(*_args, **_kwargs)

        if inspect.isclass(func):
            setup = getattr(func, 'setup_class', classmethod(lambda _: None))
            setup = skip_decorator(setup.__func__)
            setup = setup.__get__(func)
            setattr(func, 'setup_class', setup)
            return func

        return copy_metadata(func, skipper)

    return skip_decorator


def knownfailureif(fail_condition, msg=None, known_exception_class=None):
    # based on numpy.testing.dec.knownfailureif
    if msg is None:
        msg = 'Test known to fail'

    def known_fail_decorator(f):
        def failer(*args, **kwargs):
            try:
                # Always run the test (to generate images).
                result = f(*args, **kwargs)
            except Exception as err:
                if fail_condition:
                    if known_exception_class is not None:
                        if not isinstance(err, known_exception_class):
                            # This is not the expected exception
                            raise
                    knownfail(msg)
                else:
                    raise
            if fail_condition and fail_condition != 'indeterminate':
                raise KnownFailureDidNotFailTest(msg)
            return result
        return copy_metadata(f, failer)
    return known_fail_decorator
