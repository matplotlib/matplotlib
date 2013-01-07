from __future__ import print_function
import numpy as np
from numpy.testing.utils import assert_array_equal
import matplotlib.cbook as cbook
import matplotlib.colors as mcolors
from nose.tools import assert_equal, raises
from datetime import datetime

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

from matplotlib.cbook import delete_masked_points as dmp

class Test_delete_masked_points:
    def setUp(self):
        self.mask1 = [False, False, True, True, False, False]
        self.arr0 = np.arange(1.0,7.0)
        self.arr1 = [1,2,3,np.nan,np.nan,6]
        self.arr2 = np.array(self.arr1)
        self.arr3 = np.ma.array(self.arr2, mask=self.mask1)
        self.arr_s = ['a', 'b', 'c', 'd', 'e', 'f']
        self.arr_s2 = np.array(self.arr_s)
        self.arr_dt = [datetime(2008, 1, 1), datetime(2008, 1, 2),
                       datetime(2008, 1, 3), datetime(2008, 1, 4),
                       datetime(2008, 1, 5), datetime(2008, 1, 6)]
        self.arr_dt2 = np.array(self.arr_dt)
        self.arr_colors = ['r', 'g', 'b', 'c', 'm', 'y']
        self.arr_rgba = mcolors.colorConverter.to_rgba_array(self.arr_colors)

    @raises(ValueError)
    def test_bad_first_arg(self):
        dmp('a string', self.arr0)

    def test_string_seq(self):
        actual = dmp(self.arr_s, self.arr1)
        ind = [0, 1, 2, 5]
        expected = (self.arr_s2.take(ind), self.arr2.take(ind))
        assert_array_equal(actual[0], expected[0])
        assert_array_equal(actual[1], expected[1])

    def test_datetime(self):
        actual = dmp(self.arr_dt, self.arr3)
        ind = [0, 1,  5]
        expected = (self.arr_dt2.take(ind),
                    self.arr3.take(ind).compressed())
        assert_array_equal(actual[0], expected[0])
        assert_array_equal(actual[1], expected[1])

    def test_rgba(self):
        actual = dmp(self.arr3, self.arr_rgba)
        ind = [0, 1, 5]
        expected = (self.arr3.take(ind).compressed(),
                    self.arr_rgba.take(ind, axis=0))
        assert_array_equal(actual[0], expected[0])
        assert_array_equal(actual[1], expected[1])


def test_allequal():
    assert(cbook.allequal([1, 1, 1]))
    assert(not cbook.allequal([1, 1, 0]))
    assert(cbook.allequal([]))
    assert(cbook.allequal(('a', 'a')))
    assert(not cbook.allequal(('a', 'b')))
