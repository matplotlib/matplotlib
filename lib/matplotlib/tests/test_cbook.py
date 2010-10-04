import numpy as np
import matplotlib.cbook as cbook
from nose.tools import assert_equal

def test_is_string_like():
    y = np.arange( 10 )
    assert_equal( cbook.is_string_like( y ), False )
    y.shape = 10, 1
    assert_equal( cbook.is_string_like( y ), False )
    y.shape = 1, 10
    assert_equal( cbook.is_string_like( y ), False )

    assert cbook.is_string_like( "hello world" )
    assert_equal( cbook.is_string_like(10), False )

def test_restrict_dict():
    d = {'foo': 'bar', 1: 2}
    d1 = cbook.restrict_dict(d, ['foo', 1])
    assert_equal(d1, d)
    d2 = cbook.restrict_dict(d, ['bar', 2])
    assert_equal(d2, {})
    d3 = cbook.restrict_dict(d, {'foo': 1})
    assert_equal(d3, {'foo': 'bar'})
    d4 = cbook.restrict_dict(d, {})
    assert_equal(d4, {})
    d5 = cbook.restrict_dict(d, set(['foo',2]))
    assert_equal(d5, {'foo': 'bar'})
    # check that d was not modified
    assert_equal(d, {'foo': 'bar', 1: 2})
