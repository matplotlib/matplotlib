import unittest
from datetime import datetime

import numpy as np
import matplotlib.cbook as cbook
import matplotlib.colors as mcolors

from matplotlib.cbook import delete_masked_points as dmp

class Test_delete_masked_points(unittest.TestCase):
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

    def test_bad_first_arg(self):
        self.assertRaises(ValueError, dmp, 'a string', self.arr0)

    def test_string_seq(self):
        actual = dmp(self.arr_s, self.arr1)
        ind = [0, 1, 2, 5]
        expected = (self.arr_s2.take(ind), self.arr2.take(ind))

    def test_datetime(self):
        actual = dmp(self.arr_dt, self.arr3)
        ind = [0, 1,  5]
        expected = (self.arr_dt2.take(ind),
                    self.arr3.take(ind).compressed())
        self.assert_(np.all(actual[0] == expected[0]) and
                     np.all(actual[1] == expected[1]))

    def test_rgba(self):
        actual = dmp(self.arr3, self.arr_rgba)
        ind = [0, 1, 5]
        expected = (self.arr3.take(ind).compressed(),
                    self.arr_rgba.take(ind, axis=0))
        self.assert_(np.all(actual[0] == expected[0]) and
                     np.all(actual[1] == expected[1]))


if __name__=='__main__':
    unittest.main()
