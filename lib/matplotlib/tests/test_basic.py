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
        'any',
        'all',
        'sum'
    ])

    overridden = False
    for key in globals().keys():
        if key in dir(sys.modules["__builtin__"]):
            if (globals()[key] != getattr(sys.modules["__builtin__"], key) and
                key not in ok_to_override):
                print "'%s' was overridden in globals()." % key
                overridden = True

    assert not overridden
