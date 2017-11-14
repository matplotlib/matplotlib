# -*- coding: utf-8 -*-
"""Catch all for categorical functions"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from matplotlib import category as cat, pyplot as plt
from matplotlib.axes import Axes


@pytest.fixture
def ax():
    return plt.figure().subplots()


@pytest.mark.parametrize(
    "data, expected_indices, expected_labels",
    [("hello world", [0], ["hello world"]),
     (["Здравствуйте мир"], [0], ["Здравствуйте мир"]),
     (["a", "b", "b", "a", "c", "c"], [0, 1, 1, 0, 2, 2], ["a", "b", "c"]),
     ([b"foo", b"bar"], range(2), ["foo", "bar"]),
     (np.array(["1", "11", "3"]), range(3), ["1", "11", "3"]),
     (np.array([b"1", b"11", b"3"]), range(3), ["1", "11", "3"])])
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
    assert isinstance(du, cat._CategoricalUnit)


def test_update(ax):
    l1, = ax.plot(["a", "d"])
    l2, = ax.plot(["b", "d", "e"])
    assert_array_equal(l1.get_ydata(orig=False), [0, 1])
    assert_array_equal(l2.get_ydata(orig=False), [2, 1, 3])
    assert ax.yaxis.units._vals == ["a", "d", "b", "e"]
    assert ax.yaxis.units._val_to_idx == {"a": 0, "d": 1, "b": 2, "e": 3}


@pytest.mark.parametrize("plotter", [Axes.plot, Axes.scatter, Axes.bar])
def test_StrCategoryLocator(ax, plotter):
    ax.plot(["a", "b", "c"])
    assert_array_equal(ax.yaxis.major.locator(), range(3))


@pytest.mark.parametrize("plotter", [Axes.plot, Axes.scatter, Axes.bar])
def test_StrCategoryFormatter(ax, plotter):
    plotter(ax, range(2), ["hello", "мир"])
    assert ax.yaxis.major.formatter(object(), 0) == "hello"
    assert ax.yaxis.major.formatter(object(), 1) == "мир"
    assert ax.yaxis.major.formatter(object(), 2) == ""
    assert ax.yaxis.major.formatter(object(), None) == ""
