# -*- coding: utf-8 -*-
"""Catch all for categorical functions"""
from __future__ import absolute_import, division, print_function

import six
import pytest
import numpy as np
from numpy.testing import assert_array_equal

import matplotlib.pyplot as plt
import matplotlib.category as cat
from matplotlib.axes import Axes


class TestUnitData(object):
    test_cases = [('single', ("hello world", ["hello world"], [0])),
                  ('unicode', (u"Здравствуйте мир", [u"Здравствуйте мир"], [0])),
                  ('mixed', (['A', 'A', 'B'],
                             ['A', 'B', ],
                             [0, 1]))]

    ids, data = zip(*test_cases)

    @pytest.mark.parametrize("data, seq, locs", data, ids=ids)
    def test_unit(self, data, seq, locs):
        act = cat.UnitData()
        for v in data:
            act.update(data)
        assert list(act._mapping.keys()) == seq
        assert list(act._mapping.values()) == locs

    def test_update_map(self):
        oseq = ['a', 'd']
        olocs = [0, 1]

        data_update = ['b', 'd', 'e']
        useq = ['a', 'd', 'b', 'e']
        ulocs = [0, 1, 2, 3]

        unitdata = cat.UnitData(zip(oseq, olocs))
        assert list(unitdata._mapping.keys()) == oseq
        assert list(unitdata._mapping.values()) == olocs

        unitdata.update(data_update)
        assert list(unitdata._mapping.keys()) == useq
        assert list(unitdata._mapping.values()) == ulocs


class FakeAxis(object):
    def __init__(self, unit_data):
        self.units = unit_data


class TestStrCategoryConverter(object):
    """Based on the pandas conversion and factorization tests:

    ref: /pandas/tseries/tests/test_converter.py
         /pandas/tests/test_algos.py:TestFactorize
    """

    test_cases = [("unicode", {u"Здравствуйте мир": 42}),
                  ("ascii", {"hello world": 42}),
                  ("single", {'a': 0, 'b': 1, 'c': 2})]

    ids, unitmaps = zip(*test_cases)

    @pytest.fixture(autouse=True)
    def mock_axis(self, request):
        self.cc = cat.StrCategoryConverter()

    @pytest.mark.parametrize("unitmap", unitmaps, ids=ids)
    def test_convert(self, unitmap):
        data, exp = zip(*six.iteritems(unitmap))
        MUD = cat.UnitData(unitmap)
        axis = FakeAxis(MUD)
        act = self.cc.convert(data, None, axis)
        np.testing.assert_allclose(act, exp)

    def test_axisinfo(self):
        MUD = cat.UnitData()
        axis = FakeAxis(MUD)
        ax = self.cc.axisinfo(None, axis)
        assert isinstance(ax.majloc, cat.StrCategoryLocator)
        assert isinstance(ax.majfmt, cat.StrCategoryFormatter)

    def test_default_units(self):
        axis = FakeAxis(None)
        assert isinstance(self.cc.default_units(["a"], axis), cat.UnitData)


class TestStrCategoryLocator(object):
    def test_StrCategoryLocator(self):
        locs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        u = cat.UnitData()
        for j in range(11):
            u.update(str(j))
        ticks = cat.StrCategoryLocator(u)
        np.testing.assert_array_equal(ticks.tick_values(None, None), locs)


class TestStrCategoryFormatter(object):
    def test_StrCategoryFormatter(self):
        seq = ["hello", "world", "hi"]
        u = cat.UnitData()
        u.update(seq)
        labels = cat.StrCategoryFormatter(u)
        assert labels(1, 1) == "world"

    def test_StrCategoryFormatterUnicode(self):
        seq = ["Здравствуйте", "привет"]
        u = cat.UnitData()
        u.update(seq)
        labels = cat.StrCategoryFormatter(u)
        assert labels(1, 1) == "привет"


def lt(tl):
    return [l.get_text() for l in tl]


def axis_test(axis, ticks, labels, unit_data):
    assert axis.get_majorticklocs() == ticks
    assert lt(axis.get_majorticklabels()) == labels
    assert axis.units._mapping == unit_data._mapping


class TestBarsBytes(object):
    bytes_cases = [('string list', ['a', 'b', 'c']),
                   ]

    bytes_ids, bytes_data = zip(*bytes_cases)

    @pytest.mark.parametrize("bars", bytes_data, ids=bytes_ids)
    def test_plot_bytes(self, bars):

        unitmap = cat.UnitData([('a', 0), ('b', 1), ('c', 2)])

        counts = np.array([4, 6, 5])
        fig, ax = plt.subplots()
        ax.bar(bars, counts)
        fig.canvas.draw()
        axis_test(ax.xaxis, [0, 1, 2], ['a', 'b', 'c'], unitmap)


class TestBarsNumlike(object):
    numlike_cases = [('string list', ['1', '11', '3']),
                     ('string ndarray', np.array(['1', '11', '3']))]

    numlike_ids, numlike_data = zip(*numlike_cases)

    @pytest.mark.parametrize("bars", numlike_data, ids=numlike_ids)
    def test_plot_numlike(self, bars):
        counts = np.array([4, 6, 5])

        fig, ax = plt.subplots()
        ax.bar(bars, counts)
        fig.canvas.draw()

        unitmap = cat.UnitData([('1', 0), ('11', 1), ('3', 2)])
        axis_test(ax.xaxis, [0, 1, 2], ['1', '11', '3'], unitmap)


class TestPlotTypes(object):
    @pytest.fixture
    def complete_data(self):
        self.complete = ['a', 'b', 'c', 'a']
        self.complete_ticks = [0, 1, 2]
        self.complete_labels = ['a', 'b', 'c']
        unitmap = [('a', 0), ('b', 1), ('c', 2)]
        self.complete_unit_data = cat.UnitData(unitmap)

    def test_plot_unicode(self):
        words = [u'Здравствуйте', u'привет']
        locs = [0.0, 1.0]
        unit_data = cat.UnitData(zip(words, locs))

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


class TestUpdatePlot(object):

    def test_update_plot(self):
        fig, ax = plt.subplots()
        ax.plot(['a', 'b'])
        ax.plot(['a', 'b', 'd'])
        ax.plot(['b', 'c', 'd'])
        fig.canvas.draw()

        labels = ['a', 'b', 'd', 'c']
        ticks = [0, 1, 2, 3]
        unitmap = cat.UnitData(list(zip(labels, ticks)))

        axis_test(ax.yaxis, ticks, labels, unitmap)

    def test_update_scatter(self):
        fig, ax = plt.subplots()
        ax.scatter(['a', 'b'], [0., 3.])
        ax.scatter(['a', 'b', 'd'], [1., 2., 3.])
        ax.scatter(['b', 'c', 'd'], [4., 1., 2.])
        fig.canvas.draw()

        labels = ['a', 'b', 'd', 'c']
        ticks = [0, 1, 2, 3]
        unitmap = cat.UnitData(list(zip(labels, ticks)))

        axis_test(ax.xaxis, ticks, labels, unitmap)


@pytest.fixture
def ax():
    return plt.figure().subplots()


@pytest.mark.parametrize(
    "data, expected_indices, expected_labels",
    [(["Здравствуйте мир"], [0], ["Здравствуйте мир"]),
     (["a", "b", "b", "a", "c", "c"], [0, 1, 1, 0, 2, 2], ["a", "b", "c"]),
     (["foo", "bar"], range(2), ["foo", "bar"]),
     (np.array(["1", "11", "3"]), range(3), ["1", "11", "3"])])
def test_simple(ax, data, expected_indices, expected_labels):
    l, = ax.plot(data)
    assert_array_equal(l.get_ydata(orig=False), expected_indices)
    assert isinstance(ax.yaxis.major.locator, cat.StrCategoryLocator)
    assert isinstance(ax.yaxis.major.formatter, cat.StrCategoryFormatter)
    ax.figure.canvas.draw()
    labels = [label.get_text() for label in ax.yaxis.get_majorticklabels()]
    assert labels == expected_labels


def test_default_units(ax):
    ax.plot(["a"])
    du = ax.yaxis.converter.default_units(["a"], ax.yaxis)
    assert isinstance(du, cat.UnitData)


def test_update(ax):
    l1, = ax.plot(["a", "d"])
    l2, = ax.plot(["b", "d", "e"])
    assert_array_equal(l1.get_ydata(orig=False), [0, 1])
    assert_array_equal(l2.get_ydata(orig=False), [2, 1, 3])
    assert ax.yaxis.units._vals == ["a", "d", "b", "e"]
    assert ax.yaxis.units._mapping == {"a": 0, "d": 1, "b": 2, "e": 3}


@pytest.mark.parametrize("plotter", [Axes.plot, Axes.scatter, Axes.bar])
def test_StrCategoryLocator(ax, plotter):
    ax.plot(["a", "b", "c"])
    assert_array_equal(ax.yaxis.major.locator(), range(3))


@pytest.mark.parametrize("plotter", [Axes.plot, Axes.scatter, Axes.bar])
def test_StrCategoryFormatter(ax, plotter):
    plotter(ax, range(2), ["hello", "мир"])
    assert ax.yaxis.major.formatter(0, 0) == "hello"
    assert ax.yaxis.major.formatter(1, 1) == "мир"
    assert ax.yaxis.major.formatter(2, 2) == ""
    assert ax.yaxis.major.formatter(0, None) == ""
