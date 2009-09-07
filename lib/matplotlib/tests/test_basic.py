from nose.tools import assert_equal
from matplotlib.testing.decorators import knownfailureif

def test_simple():
    assert_equal(1+1,2)

@knownfailureif(True)
def test_simple_knownfail():
    assert_equal(1+1,3)
