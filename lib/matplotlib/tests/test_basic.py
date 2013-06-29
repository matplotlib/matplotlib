from __future__ import print_function
from nose.tools import assert_equal
from matplotlib.testing.decorators import knownfailureif
import sys

def test_simple():
    assert_equal(1+1,2)

@knownfailureif(True)
def test_simple_knownfail():
    assert_equal(1+1,3)

from pylab import *
def test_override_builtins():
    ok_to_override = set([
        '__name__',
        '__doc__',
        '__package__',
        '__loader__',
        'any',
        'all',
        'sum'
    ])

    if sys.version_info[0] >= 3:
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
