# -*- coding: utf-8 -*-
"""Catch all for categorical functions"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pytest
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.category as cat

import unittest


class TestUnitData(object):
    testdata = [("hello world", {"hello world": 0}),
                ("Здравствуйте мир", {"Здравствуйте мир": 0})]
    ids = ["single", "unicode"]

    @pytest.mark.parametrize("data, mapping", testdata, ids=ids)
    def test_unit(self, data, mapping):
        assert cat.UnitData(data)._mapping == mapping

    def test_update_map(self):
        unitdata = cat.UnitData(['a', 'd'])
        assert unitdata._mapping == {'a': 0, 'd': 1}
        unitdata.update(['b', 'd', 'e'])
        assert unitdata._mapping == {'a': 0, 'd': 1, 'b': 2, 'e': 3}


def _mock_unit_data(mapping):
    ud = cat.UnitData([])
    ud._mapping.update(mapping)
    return ud


class FakeAxis(object):
    def __init__(self, unit_data):
        self.unit_data = unit_data


class TestStrCategoryConverter(object):
    """Based on the pandas conversion and factorization tests:

    ref: /pandas/tseries/tests/test_converter.py
         /pandas/tests/test_algos.py:TestFactorize
    """
    testdata = [("Здравствуйте мир", {"Здравствуйте мир": 42}, 42),
                ("hello world", {"hello world": 42}, 42),
                (['a', 'b', 'b', 'a', 'a', 'c', 'c', 'c'],
                 {'a': 0, 'b': 1, 'c': 2},
                 [0, 1, 1, 0, 0, 2, 2, 2])]
    ids = ["unicode", "single", "basic"]

    @pytest.fixture(autouse=True)
    def mock_axis(self, request):
        self.cc = cat.StrCategoryConverter()

    @pytest.mark.parametrize("data, unitmap, exp", testdata, ids=ids)
    def test_convert(self, data, unitmap, exp):
        axis = FakeAxis(_mock_unit_data(unitmap))
        act = self.cc.convert(data, None, axis)
        np.testing.assert_array_equal(act, exp)

    def test_axisinfo(self):
        axis = FakeAxis(_mock_unit_data({None: None}))
        ax = self.cc.axisinfo(None, axis)
        assert isinstance(ax.majloc, cat.StrCategoryLocator)
        assert isinstance(ax.majfmt, cat.StrCategoryFormatter)

    def test_default_units(self):
        axis = FakeAxis(None)
        assert self.cc.default_units(["a"], axis) is None


class TestStrCategoryLocator(object):
    def test_StrCategoryLocator(self):
        locs = list(range(10))
        ticks = cat.StrCategoryLocator({str(x): x for x in locs})
        np.testing.assert_array_equal(ticks.tick_values(None, None), locs)


class TestStrCategoryFormatter(unittest.TestCase):
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
    bytes_data = [
        ['a', 'b', 'c'],
        [b'a', b'b', b'c'],
        np.array([b'a', b'b', b'c'])
    ]

    bytes_ids = ['string list', 'bytes list', 'bytes ndarray']

    numlike_data = [
        ['1', '11', '3'],
        np.array(['1', '11', '3']),
        [b'1', b'11', b'3'],
        np.array([b'1', b'11', b'3']),
    ]

    numlike_ids = [
        'string list', 'string ndarray', 'bytes list', 'bytes ndarray'
    ]

    @pytest.fixture
    def data(self):
        self.d = ['a', 'b', 'c', 'a']
        self.dticks = [0, 1, 2]
        self.dlabels = ['a', 'b', 'c']

    def axis_test(self, axis, ticks, labels, unit_data):
        np.testing.assert_array_equal(axis.get_majorticklocs(), ticks)
        assert lt(axis.get_majorticklabels()) == labels
        assert axis.unit_data._mapping == unit_data._mapping

    def test_plot_unicode(self):
        words = ['Здравствуйте', 'привет']
        locs = [0.0, 1.0]
        unit_data = _mock_unit_data(dict(zip(words, locs)))

        fig, ax = plt.subplots()
        ax.plot(words)
        fig.canvas.draw()

        self.axis_test(ax.yaxis, locs, words, unit_data)

    @pytest.mark.usefixtures("data")
    def test_plot_1d(self):
        fig, ax = plt.subplots()
        ax.plot(self.d)
        fig.canvas.draw()

        unit_data = _mock_unit_data({'a': 0, 'b': 1, 'c': 2})
        self.axis_test(ax.yaxis, self.dticks, self.dlabels, unit_data)

    @pytest.mark.usefixtures("data")
    @pytest.mark.parametrize("bars", bytes_data, ids=bytes_ids)
    def test_plot_bytes(self, bars):
        counts = np.array([4, 6, 5])

        fig, ax = plt.subplots()
        ax.bar(bars, counts)
        fig.canvas.draw()

        unit_data = _mock_unit_data(dict(zip(bars, range(3))))
        self.axis_test(ax.xaxis, self.dticks, self.dlabels, unit_data)

    @pytest.mark.parametrize("bars", numlike_data, ids=numlike_ids)
    def test_plot_numlike(self, bars):
        counts = np.array([4, 6, 5])

        fig, ax = plt.subplots()
        ax.bar(bars, counts)
        fig.canvas.draw()

        unit_data = _mock_unit_data(dict(zip(bars, range(3))))
        self.axis_test(ax.xaxis, [0, 1, 2], ['1', '11', '3'], unit_data)

    def test_plot_update(self):
        fig, ax = plt.subplots()

        ax.plot(['a', 'b'])
        ax.plot(['a', 'b', 'd'])
        ax.plot(['b', 'c', 'd'])
        fig.canvas.draw()

        labels = ['a', 'b', 'd', 'c']
        ticks = [0, 1, 2, 3]
        unit_data = _mock_unit_data(dict(zip(labels, ticks)))
        self.axis_test(ax.yaxis, ticks, labels, unit_data)

    def test_scatter_update(self):
        fig, ax = plt.subplots()

        ax.scatter(['a', 'b'], [0., 3.])
        ax.scatter(['a', 'b', 'd'], [1., 2., 3.])
        ax.scatter(['b', 'c', 'd'], [4., 1., 2.])
        fig.canvas.draw()

        labels = ['a', 'b', 'd', 'c']
        ticks = [0, 1, 2, 3]
        unit_data = _mock_unit_data(dict(zip(labels, ticks)))
        self.axis_test(ax.xaxis, ticks, labels, unit_data)
