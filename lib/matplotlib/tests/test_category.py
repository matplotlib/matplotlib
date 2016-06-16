"""Catch all for categorical functions
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import unittest

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import cleanup, knownfailureif
import matplotlib.category as cat


class FakeAxis(object):
    def __init__(self):
        self.unit_data = []


class TestStrCategoryConverter(unittest.TestCase):
    """
    Based on the pandas conversion and factorization tests:

    ref: /pandas/tseries/tests/test_converter.py
         /pandas/tests/test_algos.py:TestFactorize
    """

    def setUp(self):
        self.cc = cat.StrCategoryConverter()
        self.axis = FakeAxis()

    def test_convert_accepts_unicode(self):
        # possibly not needed in PY3
        self.axis.unit_data = [('a', 0), ('b', 1)]
        c1 = self.cc.convert("a", None, self.axis)
        c2 = self.cc.convert(u"a", None, self.axis)
        self.assertEqual(c1, c2)
        # single values always set at 0
        c1 = self.cc.convert(["a", "b"], None, self.axis)
        c2 = self.cc.convert([u"a", u"b"], None, self.axis)
        np.testing.assert_array_equal(c1, c2)

    def test_convert_single(self):
        self.axis.unit_data = [('a', 0)]
        act = self.cc.convert("a", None, self.axis)
        exp = 0
        self.assertEqual(act, exp)

    def test_convert_basic(self):
        data = ['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c']
        exp = [0, 1, 1, 0, 0, 2, 2, 2]
        self.axis.unit_data = [('a', 0), ('b', 1), ('c', 2)]
        act = self.cc.convert(data, None, self.axis)
        np.testing.assert_array_equal(act, exp)

    def test_convert_mixed(self):
        data = ['A', 'A', np.nan, 'B', -np.inf, 3.14, np.inf]
        exp = [1, 1, -1, 2, 3, 0, 4]
        self.axis.unit_data = [('nan', -1), ('3.14', 0),
                               ('A', 1), ('B', 2),
                               ('-inf', 3), ('inf', 4)]
        act = self.cc.convert(data, None, self.axis)
        np.testing.assert_array_equal(act, exp)

    def test_axisinfo(self):
        self.axis.unit_data = [('a', 0)]
        ax = self.cc.axisinfo(None, self.axis)
        self.assertTrue(isinstance(ax.majloc, cat.StrCategoryLocator))
        self.assertTrue(isinstance(ax.majfmt, cat.StrCategoryFormatter))

    def test_default_units(self):
        self.assertEqual(self.cc.default_units(["a"], self.axis), None)


class TestMapCategories(unittest.TestCase):
    def test_map_data(self):
        act = cat.map_categories("a")
        exp = [('a', 0)]
        self.assertListEqual(act, exp)

    def test_map_data_basic(self):
        data = ['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c']
        exp = [('a', 0), ('b', 1), ('c', 2)]
        act = cat.map_categories(data)
        self.assertListEqual(act, exp)

    def test_map_data_mixed(self):
        data = ['A', 'A', np.nan, 'B', -np.inf, 3.14, np.inf]
        exp = [('nan', -1), ('3.14', 0),
               ('A', 1), ('B', 2), ('-inf', -3), ('inf', -2)]

        act = cat.map_categories(data)
        self.assertListEqual(sorted(act), sorted(exp))

    def test_update_map(self):
        data = ['b', 'd', 'e', np.inf]
        old_map = [('a', 0), ('d', 1)]
        exp = [('inf', -2), ('a', 0), ('d', 1),
               ('b', 2), ('e', 3)]
        act = cat.map_categories(data, old_map)
        self.assertListEqual(sorted(act), sorted(exp))


class TestStrCategoryLocator(unittest.TestCase):
    def setUp(self):
        self.locs = list(range(10))

    def test_CategoricalLocator(self):
        ticks = cat.StrCategoryLocator(self.locs)
        np.testing.assert_equal(ticks.tick_values(None, None),
                                self.locs)


class TestStrCategoryFormatter(unittest.TestCase):
    def setUp(self):
        self.seq = ["hello", "world", "hi"]

    def test_CategoricalFormatter(self):
        labels = cat.StrCategoryFormatter(self.seq)
        self.assertEqual(labels('a', 1), "world")


class TestPlot(unittest.TestCase):
    @classmethod
    def setupClass(cls):
        cls.lt = lambda tl: [l.get_text() for l in tl]

    def setUp(self):
        self.d = ['a', 'b', 'c', 'a']
        self.dticks = [0, 1, 2]
        self.dlabels = ['a', 'b', 'c']
        self.dunit_data = [('a', 0), ('b', 1), ('c', 2)]

        self.dm = ['here', np.nan, 'here', 'there']
        self.dmticks = [-1, 0, 1]
        self.dmlabels = ['nan', 'here', 'there']
        self.dmunit_data = [('nan', -1), ('here', 0), ('there', 1)]

    @cleanup
    def test_plot_1d(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.d)
        fig.canvas.draw()

        np.testing.assert_array_equal(ax.get_yticks(), self.dticks)
        self.assertListEqual(TestPlot.lt(ax.get_yticklabels()),
                             self.dlabels)
        self.assertListEqual(ax.yaxis.unit_data, self.dunit_data)

    @cleanup
    def test_plot_1d_missing(self):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.dm)
        fig.canvas.draw()

        np.testing.assert_array_equal(ax.get_yticks(), self.dmticks)
        self.assertListEqual(TestPlot.lt(ax.get_yticklabels()),
                             self.dmlabels)
        self.assertListEqual(ax.yaxis.unit_data, self.dmunit_data)

    @cleanup
    def test_plot_2d(self):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.dm, self.d)
        fig.canvas.draw()

        np.testing.assert_array_equal(ax.get_xticks(), self.dmticks)
        self.assertListEqual(TestPlot.lt(ax.get_xticklabels()),
                             self.dmlabels)
        self.assertListEqual(ax.xaxis.unit_data, self.dmunit_data)

        np.testing.assert_array_equal(ax.get_yticks(), self.dticks)
        self.assertListEqual(TestPlot.lt(ax.get_yticklabels()),
                             self.dlabels)
        self.assertListEqual(ax.yaxis.unit_data, self.dunit_data)

    @cleanup
    def test_scatter_2d(self):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(self.dm, self.d)
        fig.canvas.draw()

        np.testing.assert_array_equal(ax.get_xticks(), self.dmticks)
        self.assertListEqual(TestPlot.lt(ax.get_xticklabels()),
                             self.dmlabels)
        self.assertListEqual(ax.xaxis.unit_data, self.dmunit_data)

        np.testing.assert_array_equal(ax.get_yticks(), self.dticks)
        self.assertListEqual(TestPlot.lt(ax.get_yticklabels()),
                             self.dlabels)
        self.assertListEqual(ax.yaxis.unit_data, self.dunit_data)

    @unittest.expectedFailure
    @cleanup
    def test_plot_update(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(['a', 'b'])
        ax.plot(['a', 'b', 'd'])
        ax.plot(['b', 'c', 'd'])
        fig.canvas.draw()

        labels_new = ['a', 'b', 'd', 'c']
        ticks_new = [0, 1, 2, 3]
        self.assertListEqual(ax.yaxis.unit_data,
                             list(zip(labels_new, ticks_new)))
        np.testing.assert_array_equal(ax.get_yticks(), ticks_new)
        self.assertListEqual(TestPlot.lt(ax.get_yticklabels()), labels_new)
