from nose.tools import assert_equal
from matplotlib.testing.decorators import knownfailureif

def test_simple():
    '''very simple example test'''
    assert_equal(1+1,2)

@knownfailureif(True)
def test_simple_fail():
    '''very simple example test that should fail'''
    assert_equal(1+1,3)
