from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

from nose.tools import assert_equal

from matplotlib.testing.decorators import knownfailureif
from pylab import *


def test_simple():
    assert_equal(1 + 1, 2)


@knownfailureif(True)
def test_simple_knownfail():
    # Test the known fail mechanism.
    assert_equal(1 + 1, 3)


def test_override_builtins():
    ok_to_override = set([
        '__name__',
        '__doc__',
        '__package__',
        '__loader__',
        '__spec__',
        'any',
        'all',
        'sum'
    ])

    # We could use six.moves.builtins here, but that seems
    # to do a little more than just this.
    if six.PY3:
        builtins = sys.modules['builtins']
    else:
        builtins = sys.modules['__builtin__']

    overridden = False
    for key in globals().keys():
        if key in dir(builtins):
            if (globals()[key] != getattr(builtins, key) and
                    key not in ok_to_override):
                print("'%s' was overridden in globals()." % key)
                overridden = True

    assert not overridden


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
