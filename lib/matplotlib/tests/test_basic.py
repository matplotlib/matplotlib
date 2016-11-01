from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import sys

import warnings

from nose.tools import assert_equal

from ..testing.decorators import knownfailureif, skipif


SKIPIF_CONDITION = []


def setup_module():
    SKIPIF_CONDITION.append(None)


def test_simple():
    assert_equal(1 + 1, 2)


@knownfailureif(True)
def test_simple_knownfail():
    # Test the known fail mechanism.
    assert_equal(1 + 1, 3)


@skipif(True, reason="skipif decorator test with bool condition passed")
def test_skipif_bool():
    assert False, "skipif decorator does not work with bool condition"


@skipif('SKIPIF_CONDITION',
        reason="skipif decorator test with string condition passed")
def test_skipif_string():
    assert False, "skipif decorator does not work with string condition"


@skipif(True, reason="skipif decorator on class test passed")
class Test_skipif_on_class(object):
    def test(self):
        assert False, "skipif decorator does not work on classes"


class Test_skipif_on_method(object):
    @skipif(True, reason="skipif decorator on method test passed")
    def test(self):
        assert False, "skipif decorator does not work on methods"


@skipif(True, reason="skipif decorator on classmethod test passed")
class Test_skipif_on_classmethod(object):
    @classmethod
    def setup_class(cls):
        pass

    def test(self):
        assert False, "skipif decorator does not work on classmethods"


def test_override_builtins():
    import pylab

    ok_to_override = {
        '__name__',
        '__doc__',
        '__package__',
        '__loader__',
        '__spec__',
        'any',
        'all',
        'sum'
    }

    # We could use six.moves.builtins here, but that seems
    # to do a little more than just this.
    if six.PY3:
        builtins = sys.modules['builtins']
    else:
        builtins = sys.modules['__builtin__']

    overridden = False
    for key in dir(pylab):
        if key in dir(builtins):
            if (getattr(pylab, key) != getattr(builtins, key) and
                    key not in ok_to_override):
                print("'%s' was overridden in globals()." % key)
                overridden = True

    assert not overridden


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
