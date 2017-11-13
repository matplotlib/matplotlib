# -*- coding: utf-8 -*-
"""Catch all for categorical functions"""
from __future__ import absolute_import, division, print_function

from collections import OrderedDict
import six
import pytest
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.category as cat


class TestUnitData(object):
    test_cases = [('single', ("hello world", ["hello world"], [0])),
                  ('unicode', (u"Здравствуйте мир", [u"Здравствуйте мир"], [0])),
                  ('mixed', (['A', 'A', np.nan, 'B', -np.inf, 3.14, np.inf],
                             ['A', 'nan', 'B', '-inf', '3.14', 'inf'],
                             [0, np.nan, 1, 2, 3, 4]))]

    ids, data = zip(*test_cases)

    @pytest.mark.parametrize("data, seq, locs", data, ids=ids)
    def test_unit(self, data, seq, locs):
        act = cat.UnitData(data)
        assert act._seq == seq
        assert act._locs == locs

    def test_update_map(self):
        data = ['a', 'd']
        oseq = ['a', 'd']
        olocs = [0, 1]

        data_update = ['b', 'd', 'e', np.inf]
        useq = ['a', 'd', 'b', 'e', 'inf']
        ulocs = [0, 1, 2, 3, 4]

        unitdata = cat.UnitData(data)
        assert unitdata._seq == oseq
        assert unitdata._locs == olocs

        unitdata.update(data_update)
        assert unitdata._seq == useq
        assert unitdata._locs == ulocs


class FakeAxis(object):
    def __init__(self, unit_data):
        self.unit_data = unit_data


class MockUnitData(object):
    def __init__(self, data, labels=None):
        self._mapping = OrderedDict(data)
        if labels:
            self._seq = labels
        else:
            self._seq = list(self._mapping.keys())
        self._locs = list(self._mapping.values())


class TestStrCategoryConverter(object):
    """Based on the pandas conversion and factorization tests:

    ref: /pandas/tseries/tests/test_converter.py
         /pandas/tests/test_algos.py:TestFactorize
    """

    test_cases = [("unicode", {u"Здравствуйте мир": 42}),
                  ("ascii", {"hello world": 42}),
                  ("single", {'a': 0, 'b': 1, 'c': 2}),
                  ("mixed", {3.14: 0, 'A': 1, 'B': 2,
                             -np.inf: 3, np.inf: 4, np.nan: 5}),
                  ("integer string", {"!": 0, "0": 1, 0: 1}),
                  ("number", {0.0: 0.0}),
                  ("number string", {'42': 0, 42: 1})]

    ids, unitmaps = zip(*test_cases)

    @pytest.fixture(autouse=True)
    def mock_axis(self, request):
        self.cc = cat.StrCategoryConverter()

    @pytest.mark.parametrize("unitmap", unitmaps, ids=ids)
    def test_convert(self, unitmap):
        data, exp = zip(*six.iteritems(unitmap))
        MUD = MockUnitData(unitmap)
        axis = FakeAxis(MUD)
        act = self.cc.convert(data, None, axis)
        np.testing.assert_allclose(act, exp)

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


def axis_test(axis, ticks, labels, unit_data):
    np.testing.assert_array_equal(axis.get_majorticklocs(), ticks)
    assert lt(axis.get_majorticklabels()) == labels
    np.testing.assert_array_equal(axis.unit_data._locs, unit_data._locs)
    assert axis.unit_data._seq == unit_data._seq


class TestBarsBytes(object):
    bytes_cases = [('string list', ['a', 'b', 'c']),
                   ('bytes list', [b'a', b'b', b'c']),
                   ('bytes ndarray', np.array([b'a', b'b', b'c']))]

    bytes_ids, bytes_data = zip(*bytes_cases)

    @pytest.mark.parametrize("bars", bytes_data, ids=bytes_ids)
    def test_plot_bytes(self, bars):

        unitmap = MockUnitData([('a', 0), ('b', 1), ('c', 2)])

        counts = np.array([4, 6, 5])
        fig, ax = plt.subplots()
        ax.bar(bars, counts)
        fig.canvas.draw()
        axis_test(ax.xaxis, [0, 1, 2], ['a', 'b', 'c'], unitmap)


class TestBarsNumlike(object):
    numlike_cases = [('string list', ['1', '11', '3']),
                     ('string ndarray', np.array(['1', '11', '3'])),
                     ('bytes list', [b'1', b'11', b'3']),
                     ('bytes ndarray', np.array([b'1', b'11', b'3']))]

    numlike_ids, numlike_data = zip(*numlike_cases)

    @pytest.mark.parametrize("bars", numlike_data, ids=numlike_ids)
    def test_plot_numlike(self, bars):
        counts = np.array([4, 6, 5])

        fig, ax = plt.subplots()
        ax.bar(bars, counts)
        fig.canvas.draw()

        unitmap = MockUnitData([('1', 0), ('11', 1), ('3', 2)])
        axis_test(ax.xaxis, [0, 1, 2], ['1', '11', '3'], unitmap)


class TestPlotTypes(object):
    @pytest.fixture
    def complete_data(self):
        self.complete = ['a', 'b', 'c', 'a']
        self.complete_ticks = [0, 1, 2]
        self.complete_labels = ['a', 'b', 'c']
        unitmap = [('a', 0), ('b', 1), ('c', 2)]
        self.complete_unit_data = MockUnitData(unitmap)

    @pytest.fixture
    def missing_data(self):
        self.missing = ['here', np.nan, 'here', 'there']
        self.missing_ticks = [0, np.nan, 1]
        self.missing_labels = ['here', 'nan', 'there']
        unitmap = [('here', 0), (np.nan, np.nan), ('there', 1)]
        self.missing_unit_data = MockUnitData(unitmap,
                                              labels=self.missing_labels)

    def test_plot_unicode(self):
        words = [u'Здравствуйте', u'привет']
        locs = [0.0, 1.0]
        unit_data = MockUnitData(zip(words, locs))

        fig, ax = plt.subplots()
        ax.plot(words)
        fig.canvas.draw()

        axis_test(ax.yaxis, locs, words, unit_data)

    @pytest.mark.usefixtures("complete_data")
    def test_plot_yaxis(self):
        fig, ax = plt.subplots()
        ax.plot(self.complete)
        fig.canvas.draw()
        axis_test(ax.yaxis, self.complete_ticks, self.complete_labels,
                  self.complete_unit_data)

    @pytest.mark.xfail(reason="scatter/plot inconsistencies")
    @pytest.mark.usefixtures("missing_data")
    def test_plot_yaxis_missing_data(self):
        fig, ax = plt.subplots()
        ax.plot(self.missing)
        fig.canvas.draw()
        axis_test(ax.yaxis, self.missing_ticks, self.missing_labels,
                  self.missing_unit_data)

    @pytest.mark.xfail(reason="scatter/plot inconsistencies")
    @pytest.mark.usefixtures("complete_data", "missing_data")
    def test_plot_missing_xaxis_yaxis(self):
        fig, ax = plt.subplots()
        ax.plot(self.missing, self.complete)
        fig.canvas.draw()

        axis_test(ax.xaxis, self.missing_ticks, self.missing_labels,
                  self.missing_unit_data)
        axis_test(ax.yaxis, self.complete_ticks, self.complete_labels,
                  self.complete_unit_data)

    @pytest.mark.usefixtures("complete_data", "missing_data")
    def test_scatter_missing_xaxis_yaxis(self):
        fig, ax = plt.subplots()
        ax.scatter(self.missing, self.complete)
        fig.canvas.draw()
        axis_test(ax.xaxis, self.missing_ticks, self.missing_labels,
                  self.missing_unit_data)
        axis_test(ax.yaxis, self.complete_ticks, self.complete_labels,
                  self.complete_unit_data)


class TestUpdatePlot(object):

    def test_update_plot(self):
        fig, ax = plt.subplots()
        ax.plot(['a', 'b'])
        ax.plot(['a', 'b', 'd'])
        ax.plot(['b', 'c', 'd'])
        fig.canvas.draw()

        labels = ['a', 'b', 'd', 'c']
        ticks = [0, 1, 2, 3]
        unitmap = MockUnitData(list(zip(labels, ticks)))

        axis_test(ax.yaxis, ticks, labels, unitmap)

    def test_update_scatter(self):
        fig, ax = plt.subplots()
        ax.scatter(['a', 'b'], [0., 3.])
        ax.scatter(['a', 'b', 'd'], [1., 2., 3.])
        ax.scatter(['b', 'c', 'd'], [4., 1., 2.])
        fig.canvas.draw()

        labels = ['a', 'b', 'd', 'c']
        ticks = [0, 1, 2, 3]
        unitmap = MockUnitData(list(zip(labels, ticks)))

        axis_test(ax.xaxis, ticks, labels, unitmap)
