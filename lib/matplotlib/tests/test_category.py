# -*- coding: utf-8 -*-
"""Catch all for categorical functions"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import pytest
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.category as cat



class TestUnitData(object):
    test_cases = {'single': ("hello world", ["hello world"], [0]),
                  'unicode': ("Здравствуйте мир", ["Здравствуйте мир"], [0]),
                  'mixed': (['A', 'A', np.nan, 'B', -np.inf, 3.14, np.inf],
                       ['-inf', '3.14', 'A', 'B', 'inf', 'nan'],
                       [-3.0, 0, 1, 2, -2.0, -1.0])}
 
    ids, data = zip(*six.iteritems(test_cases))

    @pytest.mark.parametrize("data, seq, locs", data, ids=ids)
    def test_unit(self, data, seq, locs):
        act = cat.UnitData(data)
        assert act.seq == seq
        assert act.locs == locs

    def test_update_map(self):
        data = ['a', 'd']
        oseq = ['a', 'd']
        olocs = [0, 1]

        data_update = ['b', 'd', 'e', np.inf]
        useq = ['a', 'd', 'b', 'e', 'inf']
        ulocs = [0, 1, 2, 3, -2]

        unitdata = cat.UnitData(data)
        assert unitdata.seq == oseq
        assert unitdata.locs == olocs

        unitdata.update(data_update)
        assert unitdata.seq == useq
        assert unitdata.locs == ulocs


class FakeAxis(object):
    def __init__(self, unit_data):
        self.unit_data = unit_data


class MockUnitData(object):
    def __init__(self, data):
        seq, locs = zip(*data)
        self.seq = list(seq)
        self.locs = list(locs)


class TestStrCategoryConverter(object):
    """Based on the pandas conversion and factorization tests:

    ref: /pandas/tseries/tests/test_converter.py
         /pandas/tests/test_algos.py:TestFactorize
    """

    test_cases = {"unicode": ("Здравствуйте мир", [("Здравствуйте мир", 42)], 42),
                  "ascii" : ("hello world", [("hello world", 42)], 42),
                  "single" : (['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c'],
                          [('a', 0), ('b', 1), ('c', 2)],
                            [0, 1, 1, 0, 0, 2, 2, 2]),
                  "mixed": (['A', 'A', np.nan, 'B', -np.inf, 3.14, np.inf],
                      [('nan', -1), ('3.14', 0), ('A', 1), ('B', 2),
                        ('-inf', 100), ('inf', 200)], 
                        [1, 1, -1, 2, 100, 0, 200]),
                 "integer string": (["!", "0"], [("!", 0), ("0", 1)], [0, 1]),
                 "number": (0.0, [(0.0, 0.0)], 0.0)}

    ids, data = zip(*six.iteritems(test_cases))

    @pytest.fixture(autouse=True)
    def mock_axis(self, request):
        self.cc = cat.StrCategoryConverter()

    @pytest.mark.parametrize("data, unitmap, exp", data, ids=ids)
    def test_convert(self, data, unitmap, exp):
        MUD = MockUnitData(unitmap)
        axis = FakeAxis(MUD)
        act = self.cc.convert(data, None, axis)
        np.testing.assert_array_equal(act, exp)

    def test_axisinfo(self):
        MUD = MockUnitData([(None, None)])
        axis = FakeAxis(MUD)
        ax = self.cc.axisinfo(None, axis)
        assert isinstance(ax.majloc, cat.StrCategoryLocator)
        assert isinstance(ax.majfmt, cat.StrCategoryFormatter)

    def test_default_units(self):
        axis = FakeAxis(None)
        assert self.cc.default_units(["a"], axis) is None


class TestStrCategoryLocator(object):
    def test_StrCategoryLocator(self):
        locs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ticks = cat.StrCategoryLocator(locs)
        np.testing.assert_array_equal(ticks.tick_values(None, None), locs)


class TestStrCategoryFormatter(object):
    def test_StrCategoryFormatter(self):
        seq = ["hello", "world", "hi"]
        labels = cat.StrCategoryFormatter(seq)
        assert labels('a', 1) == "world"

    def test_StrCategoryFormatterUnicode(self):
        seq = ["Здравствуйте", "привет"]
        labels = cat.StrCategoryFormatter(seq)
        assert labels('a', 1) == "привет"


def lt(tl):
    return [l.get_text() for l in tl]


class TestPlot(object):
    bytes_cases = {'string list': ['a', 'b', 'c'],
                    'bytes list': [b'a', b'b', b'c'],
                    'bytes ndarray' : np.array([b'a', b'b', b'c'])}

    bytes_ids, bytes_data = zip(*six.iteritems(bytes_cases))

    numlike_cases = {'string list': ['1', '11', '3'],
                     'string ndarray' : np.array(['1', '11', '3']),
                     'bytes list' : [b'1', b'11', b'3'],
                     'bytes ndarray' : np.array([b'1', b'11', b'3'])}

    numlike_ids, numlike_data = zip(*six.iteritems(numlike_cases))

    @pytest.fixture
    def data(self):
        self.d = ['a', 'b', 'c', 'a']
        self.dticks = [0, 1, 2]
        self.dlabels = ['a', 'b', 'c']
        unitmap = [('a', 0), ('b', 1), ('c', 2)]
        self.dunit_data = MockUnitData(unitmap)

    @pytest.fixture
    def missing_data(self):
        self.dm = ['here', np.nan, 'here', 'there']
        self.dmticks = [0, -1, 1]
        self.dmlabels = ['here', 'nan', 'there']
        unitmap = [('here', 0), ('nan', -1), ('there', 1)]
        self.dmunit_data = MockUnitData(unitmap)

    def axis_test(self, axis, ticks, labels, unit_data):
        np.testing.assert_array_equal(axis.get_majorticklocs(), ticks)
        assert lt(axis.get_majorticklabels()) == labels
        np.testing.assert_array_equal(axis.unit_data.locs, unit_data.locs)
        assert axis.unit_data.seq == unit_data.seq

    def test_plot_unicode(self):
        words = ['Здравствуйте', 'привет']
        locs = [0.0, 1.0]
        unit_data = MockUnitData(zip(words, locs))

        fig, ax = plt.subplots()
        ax.plot(words)
        fig.canvas.draw()

        self.axis_test(ax.yaxis, locs, words, unit_data)

    @pytest.mark.usefixtures("data")
    def test_plot_1d(self):
        fig, ax = plt.subplots()
        ax.plot(self.d)
        fig.canvas.draw()

        self.axis_test(ax.yaxis, self.dticks, self.dlabels, self.dunit_data)

    @pytest.mark.usefixtures("missing_data")
    def test_plot_1d_missing(self):
        fig, ax = plt.subplots()
        ax.plot(self.dm)
        fig.canvas.draw()

        self.axis_test(ax.yaxis, self.dmticks, self.dmlabels, self.dmunit_data)

    @pytest.mark.usefixtures("data")
    @pytest.mark.parametrize("bars", bytes_data, ids=bytes_ids)
    def test_plot_bytes(self, bars):
        counts = np.array([4, 6, 5])

        fig, ax = plt.subplots()
        ax.bar(bars, counts)
        fig.canvas.draw()

        self.axis_test(ax.xaxis, self.dticks, self.dlabels, self.dunit_data)

    @pytest.mark.parametrize("bars", numlike_data, ids=numlike_ids)
    def test_plot_numlike(self, bars):
        counts = np.array([4, 6, 5])

        fig, ax = plt.subplots()
        ax.bar(bars, counts)
        fig.canvas.draw()

        unitmap = MockUnitData([('1', 0), ('11', 1), ('3', 2)])
        self.axis_test(ax.xaxis, [0, 1, 2], ['1', '11', '3'], unitmap)

    @pytest.mark.usefixtures("data", "missing_data")
    def test_plot_2d(self):
        fig, ax = plt.subplots()
        ax.plot(self.dm, self.d)
        fig.canvas.draw()

        self.axis_test(ax.xaxis, self.dmticks, self.dmlabels, self.dmunit_data)
        self.axis_test(ax.yaxis, self.dticks, self.dlabels, self.dunit_data)

    @pytest.mark.usefixtures("data", "missing_data")
    def test_scatter_2d(self):

        fig, ax = plt.subplots()
        ax.scatter(self.dm, self.d)
        fig.canvas.draw()

        self.axis_test(ax.xaxis, self.dmticks, self.dmlabels, self.dmunit_data)
        self.axis_test(ax.yaxis, self.dticks, self.dlabels, self.dunit_data)

    def test_plot_update(self):
        fig, ax = plt.subplots()

        ax.plot(['a', 'b'])
        ax.plot(['a', 'b', 'd'])
        ax.plot(['b', 'c', 'd'])
        fig.canvas.draw()

        labels = ['a', 'b', 'd', 'c']
        ticks = [0, 1, 2, 3]
        unit_data = MockUnitData(list(zip(labels, ticks)))

        self.axis_test(ax.yaxis, ticks, labels, unit_data)

    def test_scatter_update(self):
        fig, ax = plt.subplots()

        ax.scatter(['a', 'b'], [0., 3.])
        ax.scatter(['a', 'b', 'd'], [1., 2., 3.])
        ax.scatter(['b', 'c', 'd'], [4., 1., 2.])
        fig.canvas.draw()

        labels = ['a', 'b', 'd', 'c']
        ticks = [0, 1, 2, 3]
        unit_data = MockUnitData(list(zip(labels, ticks)))
        self.axis_test(ax.xaxis, ticks, labels, unit_data)
