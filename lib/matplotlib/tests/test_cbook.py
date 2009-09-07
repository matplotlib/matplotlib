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
