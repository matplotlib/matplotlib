# -*- coding: utf-8 -*-
"""Catch all for categorical functions"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import unittest

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import cleanup
import matplotlib.category as cat


class TestConvertToString(unittest.TestCase):
    def setUp(self):
        pass

    def test_string(self):
        self.assertEqual("abc", cat.convert_to_string("abc"))

    def test_unicode(self):
        self.assertEqual("Здравствуйте мир",
                         cat.convert_to_string("Здравствуйте мир"))

    def test_decimal(self):
        self.assertEqual("3.14", cat.convert_to_string(3.14))

    def test_nan(self):
        self.assertEqual("nan", cat.convert_to_string(np.nan))

    def test_posinf(self):
        self.assertEqual("inf", cat.convert_to_string(np.inf))

    def test_neginf(self):
        self.assertEqual("-inf", cat.convert_to_string(-np.inf))


class TestMapCategories(unittest.TestCase):
    def test_map_unicode(self):
        act = cat.map_categories("Здравствуйте мир")
        exp = [("Здравствуйте мир", 0)]
        self.assertListEqual(act, exp)

    def test_map_data(self):
        act = cat.map_categories("hello world")
        exp = [("hello world", 0)]
        self.assertListEqual(act, exp)

    def test_map_data_basic(self):
        data = ['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c']
        exp = [('a', 0), ('b', 1), ('c', 2)]
        act = cat.map_categories(data)
        self.assertListEqual(sorted(act), sorted(exp))

    def test_map_data_mixed(self):
        data = ['A', 'A', np.nan, 'B', -np.inf, 3.14, np.inf]
        exp = [('nan', -1), ('3.14', 0),
               ('A', 1), ('B', 2), ('-inf', -3), ('inf', -2)]

        act = cat.map_categories(data)
        self.assertListEqual(sorted(act), sorted(exp))

    @unittest.SkipTest
    def test_update_map(self):
        data = ['b', 'd', 'e', np.inf]
        old_map = [('a', 0), ('d', 1)]
        exp = [('inf', -2), ('a', 0), ('d', 1),
               ('b', 2), ('e', 3)]
        act = cat.map_categories(data, old_map)
        self.assertListEqual(sorted(act), sorted(exp))


class FakeAxis(object):
    def __init__(self):
        self.unit_data = []


class TestStrCategoryConverter(unittest.TestCase):
    """Based on the pandas conversion and factorization tests:

    ref: /pandas/tseries/tests/test_converter.py
         /pandas/tests/test_algos.py:TestFactorize
    """

    def setUp(self):
        self.cc = cat.StrCategoryConverter()
        self.axis = FakeAxis()

    def test_convert_unicode(self):
        self.axis.unit_data = [("Здравствуйте мир", 42)]
        act = self.cc.convert("Здравствуйте мир", None, self.axis)
        exp = 42
        self.assertEqual(act, exp)

    def test_convert_single(self):
        self.axis.unit_data = [("hello world", 42)]
        act = self.cc.convert("hello world", None, self.axis)
        exp = 42
        self.assertEqual(act, exp)

    def test_convert_basic(self):
        data = ['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c']
        exp = [0, 1, 1, 0, 0, 2, 2, 2]
        self.axis.unit_data = [('a', 0), ('b', 1), ('c', 2)]
        act = self.cc.convert(data, None, self.axis)
        np.testing.assert_array_equal(act, exp)

    def test_convert_mixed(self):
        data = ['A', 'A', np.nan, 'B', -np.inf, 3.14, np.inf]
        exp = [1, 1, -1, 2, 100, 0, 200]
        self.axis.unit_data = [('nan', -1), ('3.14', 0),
                               ('A', 1), ('B', 2),
                               ('-inf', 100), ('inf', 200)]
        act = self.cc.convert(data, None, self.axis)
        np.testing.assert_array_equal(act, exp)

    def test_axisinfo(self):
        self.axis.unit_data = [('a', 0)]
        ax = self.cc.axisinfo(None, self.axis)
        self.assertTrue(isinstance(ax.majloc, cat.StrCategoryLocator))
        self.assertTrue(isinstance(ax.majfmt, cat.StrCategoryFormatter))

    def test_default_units(self):
        self.assertEqual(self.cc.default_units(["a"], self.axis), None)


class TestStrCategoryLocator(unittest.TestCase):
    def setUp(self):
        self.locs = list(range(10))

    def test_StrCategoryLocator(self):
        ticks = cat.StrCategoryLocator(self.locs)
        np.testing.assert_equal(ticks.tick_values(None, None),
                                self.locs)


class TestStrCategoryFormatter(unittest.TestCase):
    def setUp(self):
        self.seq = ["hello", "world", "hi"]

    def test_StrCategoryFormatter(self):
        labels = cat.StrCategoryFormatter(self.seq)
        self.assertEqual(labels('a', 1), "world")


def lt(tl):
    return [l.get_text() for l in tl]


class TestPlot(unittest.TestCase):

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
    def test_plot_unicode(self):
        # needs image test -  works but
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        words = ['Здравствуйте', 'привет']
        locs = [0.0, 1.0]
        ax.plot(words)
        fig.canvas.draw()

        self.assertListEqual(ax.yaxis.unit_data,
                             list(zip(words, locs)))
        np.testing.assert_array_equal(ax.get_yticks(), locs)
        self.assertListEqual(lt(ax.get_yticklabels()), words)

    @cleanup
    def test_plot_1d(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.d)
        fig.canvas.draw()

        np.testing.assert_array_equal(ax.get_yticks(), self.dticks)
        self.assertListEqual(lt(ax.get_yticklabels()),
                             self.dlabels)
        self.assertListEqual(ax.yaxis.unit_data, self.dunit_data)

    @cleanup
    def test_plot_1d_missing(self):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.dm)
        fig.canvas.draw()

        np.testing.assert_array_equal(ax.get_yticks(), self.dmticks)
        self.assertListEqual(lt(ax.get_yticklabels()),
                             self.dmlabels)
        self.assertListEqual(ax.yaxis.unit_data, self.dmunit_data)

    @cleanup
    def test_plot_2d(self):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.dm, self.d)
        fig.canvas.draw()

        np.testing.assert_array_equal(ax.get_xticks(), self.dmticks)
        self.assertListEqual(lt(ax.get_xticklabels()),
                             self.dmlabels)
        self.assertListEqual(ax.xaxis.unit_data, self.dmunit_data)

        np.testing.assert_array_equal(ax.get_yticks(), self.dticks)
        self.assertListEqual(lt(ax.get_yticklabels()),
                             self.dlabels)
        self.assertListEqual(ax.yaxis.unit_data, self.dunit_data)

    @cleanup
    def test_scatter_2d(self):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(self.dm, self.d)
        fig.canvas.draw()

        np.testing.assert_array_equal(ax.get_xticks(), self.dmticks)
        self.assertListEqual(lt(ax.get_xticklabels()),
                             self.dmlabels)
        self.assertListEqual(ax.xaxis.unit_data, self.dmunit_data)

        np.testing.assert_array_equal(ax.get_yticks(), self.dticks)
        self.assertListEqual(lt(ax.get_yticklabels()),
                             self.dlabels)
        self.assertListEqual(ax.yaxis.unit_data, self.dunit_data)

    @unittest.SkipTest
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
        self.assertListEqual(lt(ax.get_yticklabels()), labels_new)
